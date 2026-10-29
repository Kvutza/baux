//! Independent device snapshots using the existing KNN and posterior kernels.
use super::*;

pub struct DevicePosterior {
    weights: DeviceBuffer<f32>,
    l2: DeviceBuffer<f32>,
    means: DeviceBuffer<f32>,
    errors: DeviceBuffer<f32>,
    indices: DeviceBuffer<u32>,
    queries: usize,
    spec: WeightedSpec,
    runtime: Arc<Runtime>,
}

pub struct DeviceDraw {
    values: DeviceBuffer<f64>,
    _runtime: Arc<Runtime>,
}

fn snapshot<T: DeviceCopy>(
    source: &DeviceBuffer<T>,
    len: usize,
    stream: &CudaStream,
) -> CudaResult<DeviceBuffer<T>> {
    let result = DeviceBuffer::zeroed(stream, len).map_err(cuda_error)?;
    // SAFETY: both typed allocations cover len elements on the same context.
    unsafe {
        cuda_core::simt::memory::memcpy_dtod_async(
            result.cu_deviceptr(),
            source.cu_deviceptr(),
            len.checked_mul(size_of::<T>()).ok_or("snapshot overflow")?,
            stream.cu_stream(),
        )
        .map_err(cuda_error)?;
    }
    Ok(result)
}

impl CudaIndex {
    /// # Safety
    /// pointer addresses query_count*dims contiguous FP32 values on device 0.
    /// The producer must have synchronized to the legacy default CUDA stream.
    pub unsafe fn device_posterior(
        &mut self,
        pointer: u64,
        query_count: usize,
        outcomes: &[f32],
        variances: Option<&[f32]>,
        scales: &[f32],
        spec: WeightedSpec,
    ) -> CudaResult<DevicePosterior> {
        if pointer == 0 || query_count == 0 {
            return Err("device queries must be nonempty".into());
        }
        self.check_weighted(outcomes, variances, scales, spec)?;
        let problem = ScanProblem {
            rows: self.row_count,
            dims: self.dims,
            queries: query_count,
            neighbors: spec.input_k,
        };
        let values = query_count
            .checked_mul(spec.metrics)
            .ok_or("posterior size overflow")?;
        let indices = query_count
            .checked_mul(spec.used_k)
            .ok_or("neighbor size overflow")?;
        let weights = indices
            .checked_mul(spec.metrics)
            .ok_or("weight size overflow")?;
        self.posterior.ensure(
            &self.runtime.stream,
            outcomes.len(),
            variances.map_or(1, <[f32]>::len),
            scales.len(),
            values,
            weights,
            indices,
        )?;
        self.posterior
            .sync(&self.runtime.stream, outcomes, scales)?;
        self.posterior
            .sync_variances(&self.runtime.stream, variances)?;
        let stage = self.stage_input(problem, None, pointer)?;
        self.launch_weighted(&stage, spec, variances.is_some())?;
        let output = DevicePosterior {
            runtime: Arc::clone(&self.runtime),
            weights: snapshot(&self.posterior.weights, weights, &self.runtime.stream)?,
            l2: snapshot(&self.posterior.l2, values, &self.runtime.stream)?,
            means: snapshot(&self.posterior.means, values, &self.runtime.stream)?,
            errors: snapshot(&self.posterior.errors, values, &self.runtime.stream)?,
            indices: snapshot(&self.posterior.indices, indices, &self.runtime.stream)?,
            queries: query_count,
            spec,
        };
        self.runtime.stream.synchronize().map_err(cuda_error)?;
        self.runtime.context.check_err().map_err(cuda_error)?;
        Ok(output)
    }
}

impl DevicePosterior {
    pub fn pointer(&self, error: bool) -> u64 {
        if error {
            self.errors.cu_deviceptr()
        } else {
            self.means.cu_deviceptr()
        }
    }

    pub fn draw(&self, seeds: &[u64]) -> CudaResult<DeviceDraw> {
        if seeds.is_empty() {
            return Err("draw seeds must be nonempty".into());
        }
        let count = seeds
            .len()
            .checked_mul(self.queries)
            .and_then(|v| v.checked_mul(self.spec.metrics))
            .ok_or("draw size overflow")?;
        let mut values = DeviceBuffer::zeroed(&self.runtime.stream, count).map_err(cuda_error)?;
        let seed_buffer =
            DeviceBuffer::from_host(&self.runtime.stream, seeds).map_err(cuda_error)?;
        let params = DrawParams {
            queries: to_u32(self.queries, "queries")?,
            neighbors: to_u32(self.spec.used_k, "neighbors")?,
            metrics: to_u32(self.spec.metrics, "metrics")?,
            seed_count: to_u32(seeds.len(), "seeds")?,
        };
        let blocks = to_u32(count, "draw count")?.div_ceil(THREADS);
        let launch = self
            .runtime
            .module
            .prepare_posterior_draw(LaunchConfig1D::new(blocks, THREADS, 0))
            .map_err(cuda_error)?;
        self.runtime
            .module
            .posterior_draw(
                &self.runtime.stream,
                &launch,
                &self.weights,
                &self.l2,
                &self.means,
                &self.errors,
                &self.indices,
                &seed_buffer,
                &mut values,
                params,
            )
            .map_err(cuda_error)?;
        self.runtime.stream.synchronize().map_err(cuda_error)?;
        Ok(DeviceDraw {
            _runtime: Arc::clone(&self.runtime),
            values,
        })
    }
}

impl DeviceDraw {
    pub fn pointer(&self) -> u64 {
        self.values.cu_deviceptr()
    }
}
