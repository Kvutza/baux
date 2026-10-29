use std::mem::size_of;

use cuda_core::{DeviceBuffer, LaunchConfig1D, simt::LaunchConfig};

use super::*;

fn dense_tiles(leaves: &[DenseLeaf]) -> CudaResult<Vec<DenseTile>> {
    let mut tiles = Vec::new();
    for (leaf_index, leaf) in leaves.iter().enumerate() {
        let leaf_index = to_u32(leaf_index, "dense leaf count")?;
        let length = usize::try_from(leaf.length)
            .map_err(|_| "CUDA dense leaf length exceeds usize".to_string())?;
        let mut start = 0;
        while start < length {
            let tile_length = (length - start).min(DENSE_ELEMENTS);
            tiles.push(DenseTile {
                leaf: leaf_index,
                start: to_u32(start, "dense leaf length")?,
                length: to_u32(tile_length, "dense tile length")?,
                pad: 0,
            });
            start += tile_length;
        }
    }
    if tiles.is_empty() {
        return Err("CUDA dense apply requires non-empty leaves".to_string());
    }
    Ok(tiles)
}

pub fn dense_apply(
    base: &[f32],
    leaves: &[DenseLeaf],
    terms: &[DenseTerm],
) -> CudaResult<Vec<f32>> {
    if base.is_empty() || leaves.is_empty() || terms.is_empty() {
        return Err("CUDA dense apply requires base values, leaves, and terms".to_string());
    }
    let tiles = dense_tiles(leaves)?;

    let runtime = Runtime::new()?;
    let base_buffer = DeviceBuffer::from_host(&runtime.stream, base).map_err(cuda_error)?;
    let leaf_buffer = DeviceBuffer::from_host(&runtime.stream, leaves).map_err(cuda_error)?;
    let term_buffer = DeviceBuffer::from_host(&runtime.stream, terms).map_err(cuda_error)?;
    let tile_buffer = DeviceBuffer::from_host(&runtime.stream, &tiles).map_err(cuda_error)?;
    let mut output = DeviceBuffer::zeroed(&runtime.stream, base.len()).map_err(cuda_error)?;
    let config = LaunchConfig {
        grid_dim: (to_u32(tiles.len(), "dense tile count")?, 1, 1),
        block_dim: (THREADS, 1, 1),
        shared_mem_bytes: 0,
    };
    // SAFETY: validation in Baux guarantees complete, disjoint leaves. Each
    // CUDA block owns one tile and each tile thread owns distinct outputs.
    unsafe {
        runtime
            .module
            .apply_dense(
                &runtime.stream,
                config,
                &base_buffer,
                &leaf_buffer,
                &term_buffer,
                &tile_buffer,
                &mut output,
                to_u32(terms.len(), "dense term count")?,
            )
            .map_err(cuda_error)?;
    }
    runtime.context.check_err().map_err(cuda_error)?;
    read_prefix(&output, &runtime.stream, base.len())
}

pub struct Bf16Engine {
    runtime: Runtime,
    base: DeviceBuffer<u16>,
    candidate: DeviceBuffer<u16>,
    leaves: DeviceBuffer<DenseLeaf>,
    tiles: DeviceBuffer<DenseTile>,
    terms: DeviceBuffer<DenseTerm>,
    status: DeviceBuffer<u32>,
    term_capacity: usize,
    tile_count: usize,
    len: usize,
}

impl Bf16Engine {
    pub fn new(base: &[u16], leaves: &[DenseLeaf]) -> CudaResult<Self> {
        if base.is_empty() || leaves.is_empty() {
            return Err("CUDA BF16 tree requires base values and leaves".to_string());
        }
        let tiles = dense_tiles(leaves)?;
        let runtime = Runtime::new()?;
        let mut engine = Self {
            base: DeviceBuffer::from_host(&runtime.stream, base).map_err(cuda_error)?,
            candidate: DeviceBuffer::from_host(&runtime.stream, base).map_err(cuda_error)?,
            leaves: DeviceBuffer::from_host(&runtime.stream, leaves).map_err(cuda_error)?,
            tiles: DeviceBuffer::from_host(&runtime.stream, &tiles).map_err(cuda_error)?,
            terms: DeviceBuffer::zeroed(&runtime.stream, 1).map_err(cuda_error)?,
            status: DeviceBuffer::zeroed(&runtime.stream, tiles.len()).map_err(cuda_error)?,
            term_capacity: 1,
            tile_count: tiles.len(),
            len: base.len(),
            runtime,
        };
        engine.validate()?;
        Ok(engine)
    }

    /// Copy a contiguous BF16 CUDA allocation into persistent Baux storage.
    ///
    /// # Safety
    /// `pointer` must address at least `len * 2` readable bytes on CUDA device
    /// 0.
    pub unsafe fn from_device(pointer: u64, len: usize, leaves: &[DenseLeaf]) -> CudaResult<Self> {
        if pointer == 0 || len == 0 || leaves.is_empty() {
            return Err("CUDA BF16 tree requires a device base and leaves".to_string());
        }
        let tiles = dense_tiles(leaves)?;
        let runtime = Runtime::new()?;
        let base = DeviceBuffer::zeroed(&runtime.stream, len).map_err(cuda_error)?;
        unsafe {
            cuda_core::simt::memory::memcpy_dtod_async(
                base.cu_deviceptr(),
                pointer,
                len.checked_mul(size_of::<u16>())
                    .ok_or("CUDA BF16 byte count overflow")?,
                runtime.stream.cu_stream(),
            )
            .map_err(cuda_error)?;
        }
        runtime.stream.synchronize().map_err(cuda_error)?;
        let candidate = DeviceBuffer::zeroed(&runtime.stream, len).map_err(cuda_error)?;
        unsafe {
            cuda_core::simt::memory::memcpy_dtod_async(
                candidate.cu_deviceptr(),
                base.cu_deviceptr(),
                len * size_of::<u16>(),
                runtime.stream.cu_stream(),
            )
            .map_err(cuda_error)?;
        }
        runtime.stream.synchronize().map_err(cuda_error)?;
        let mut engine = Self {
            base,
            candidate,
            leaves: DeviceBuffer::from_host(&runtime.stream, leaves).map_err(cuda_error)?,
            tiles: DeviceBuffer::from_host(&runtime.stream, &tiles).map_err(cuda_error)?,
            terms: DeviceBuffer::zeroed(&runtime.stream, 1).map_err(cuda_error)?,
            status: DeviceBuffer::zeroed(&runtime.stream, tiles.len()).map_err(cuda_error)?,
            term_capacity: 1,
            tile_count: tiles.len(),
            len,
            runtime,
        };
        engine.validate()?;
        Ok(engine)
    }

    pub fn materialize(&mut self, terms: &[DenseTerm]) -> CudaResult<()> {
        if terms.is_empty() {
            return Err("CUDA BF16 tree requires dense terms".to_string());
        }
        if terms.len() > self.term_capacity {
            self.term_capacity = terms
                .len()
                .checked_next_power_of_two()
                .ok_or("CUDA BF16 term capacity overflow")?;
            self.terms = DeviceBuffer::zeroed(&self.runtime.stream, self.term_capacity)
                .map_err(cuda_error)?;
        }
        copy_prefix(&self.terms, terms, &self.runtime.stream)?;
        self.clear_status()?;
        let config = LaunchConfig {
            grid_dim: (to_u32(self.tile_count, "dense tile count")?, 1, 1),
            block_dim: (THREADS, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            self.runtime
                .module
                .apply_bf16(
                    &self.runtime.stream,
                    config,
                    &self.base,
                    &self.leaves,
                    &self.terms,
                    &self.tiles,
                    &mut self.candidate,
                    &mut self.status,
                    to_u32(terms.len(), "dense term count")?,
                )
                .map_err(cuda_error)?;
        }
        self.runtime.context.check_err().map_err(cuda_error)?;
        if let Err(error) = self.check_status("CUDA BF16 perturbation overflowed FP32") {
            self.reset_candidate()?;
            return Err(error);
        }
        Ok(())
    }

    pub fn candidate(&self) -> CudaResult<Vec<u16>> {
        read_prefix(&self.candidate, &self.runtime.stream, self.len)
    }

    pub fn device_ptr(&self, stream: Option<i64>) -> CudaResult<(u64, usize, usize)> {
        sync_stream(&self.runtime, stream)?;
        Ok((
            self.candidate.cu_deviceptr(),
            self.len * size_of::<u16>(),
            0,
        ))
    }

    fn validate(&mut self) -> CudaResult<()> {
        self.clear_status()?;
        let config = LaunchConfig {
            grid_dim: (to_u32(self.tile_count, "dense tile count")?, 1, 1),
            block_dim: (THREADS, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            self.runtime
                .module
                .check_bf16(
                    &self.runtime.stream,
                    config,
                    &self.base,
                    &self.leaves,
                    &self.tiles,
                    &mut self.status,
                )
                .map_err(cuda_error)?;
        }
        self.runtime.context.check_err().map_err(cuda_error)?;
        self.check_status("CUDA BF16 base values must be finite")
    }

    fn clear_status(&self) -> CudaResult<()> {
        unsafe {
            cuda_core::simt::memory::memset_d8_async(
                self.status.cu_deviceptr(),
                0,
                self.status.len() * size_of::<u32>(),
                self.runtime.stream.cu_stream(),
            )
            .map_err(cuda_error)
        }
    }

    fn check_status(&self, message: &str) -> CudaResult<()> {
        let status = read_prefix(&self.status, &self.runtime.stream, self.tile_count)?;
        if status.contains(&1) {
            Err(message.to_string())
        } else {
            Ok(())
        }
    }

    fn reset_candidate(&self) -> CudaResult<()> {
        unsafe {
            cuda_core::simt::memory::memcpy_dtod_async(
                self.candidate.cu_deviceptr(),
                self.base.cu_deviceptr(),
                self.len
                    .checked_mul(size_of::<u16>())
                    .ok_or("CUDA BF16 byte count overflow")?,
                self.runtime.stream.cu_stream(),
            )
            .map_err(cuda_error)
        }
    }
}

pub struct DenseLinearEngine {
    runtime: Runtime,
    input: DeviceBuffer<f32>,
    weight: DeviceBuffer<f32>,
    bias: DeviceBuffer<f32>,
    terms: DeviceBuffer<DenseTerm>,
    output: DeviceBuffer<f32>,
    term_capacity: usize,
    params: DenseLinearParams,
}

impl DenseLinearEngine {
    pub fn new(
        weight: &[f32],
        bias: Option<&[f32]>,
        mut params: DenseLinearParams,
    ) -> CudaResult<Self> {
        if params.rows == 0 || params.columns == 0 {
            return Err("CUDA dense linear requires non-zero rows and columns".to_string());
        }
        let expected = params.rows as usize * params.columns as usize;
        if weight.len() != expected {
            return Err(format!(
                "CUDA dense linear has {} weights, expected {expected}",
                weight.len()
            ));
        }
        if bias.is_some_and(|values| values.len() != params.rows as usize) {
            return Err("CUDA dense linear bias length must equal rows".to_string());
        }
        params.has_bias = u32::from(bias.is_some());
        params.term_count = 0;
        let runtime = Runtime::new()?;
        let no_bias = [0.0_f32];
        let input =
            DeviceBuffer::zeroed(&runtime.stream, params.columns as usize).map_err(cuda_error)?;
        let weight = DeviceBuffer::from_host(&runtime.stream, weight).map_err(cuda_error)?;
        let bias = DeviceBuffer::from_host(&runtime.stream, bias.unwrap_or(&no_bias))
            .map_err(cuda_error)?;
        let terms = DeviceBuffer::zeroed(&runtime.stream, 1).map_err(cuda_error)?;
        let output =
            DeviceBuffer::zeroed(&runtime.stream, params.rows as usize).map_err(cuda_error)?;
        Ok(Self {
            runtime,
            input,
            weight,
            bias,
            terms,
            output,
            term_capacity: 1,
            params,
        })
    }

    pub fn eval(&mut self, input: &[f32], terms: &[DenseTerm]) -> CudaResult<Vec<f32>> {
        if input.len() != self.params.columns as usize {
            return Err(format!(
                "CUDA dense linear input has {} values, expected {}",
                input.len(),
                self.params.columns
            ));
        }
        if terms.is_empty() {
            return Err("CUDA dense linear requires perturbation terms".to_string());
        }
        if terms.len() > self.term_capacity {
            self.term_capacity = terms
                .len()
                .checked_next_power_of_two()
                .ok_or("CUDA dense linear term capacity overflow")?;
            self.terms = DeviceBuffer::zeroed(&self.runtime.stream, self.term_capacity)
                .map_err(cuda_error)?;
        }
        copy_prefix(&self.input, input, &self.runtime.stream)?;
        copy_prefix(&self.terms, terms, &self.runtime.stream)?;
        self.params.term_count = to_u32(terms.len(), "dense linear term count")?;
        let launch = self
            .runtime
            .module
            .prepare_dense_linear(LaunchConfig1D::new(self.params.rows, THREADS, 0))
            .map_err(cuda_error)?;
        // SAFETY: every row owns one block, each block reduces exactly one
        // output, and all resident buffers match the validated dimensions.
        self.runtime
            .module
            .dense_linear(
                &self.runtime.stream,
                &launch,
                &self.input,
                &self.weight,
                &self.bias,
                &self.terms,
                &mut self.output,
                self.params,
            )
            .map_err(cuda_error)?;
        self.runtime.context.check_err().map_err(cuda_error)?;
        read_prefix(
            &self.output,
            &self.runtime.stream,
            self.params.rows as usize,
        )
    }
}
