use crate::{CudaResult, copy_prefix, cuda_error, read_prefix, timing_event};
use baux::semantics::random::splitmix64 as splitmix;
use baux_cuda_device::{NeighborPoint, fbt_model};
use cuda_core::{CudaStream, DeviceBuffer, LaunchConfig1D};

pub(super) struct ProposalSpec {
    pub elements: usize,
    pub block_len: u32,
    pub scales: DeviceBuffer<f32>,
    pub host_scales: Vec<f32>,
    pub tensor_id: u64,
    pub logical_start: u64,
    pub transpose_columns: u32,
}

pub(super) struct NeighborRange {
    pub buffer: usize,
    pub start: u32,
    pub count: u32,
}

pub(super) struct NeighborSampler {
    pub points: DeviceBuffer<NeighborPoint>,
    pub seeds: DeviceBuffer<u64>,
    pub pool: DeviceBuffer<u16>,
    pub history: DeviceBuffer<u16>,
    pub distances: DeviceBuffer<f32>,
    pub ranges: Vec<NeighborRange>,
    pub total: u32,
}

pub(super) struct GaussianRandom {
    pub thresholds: DeviceBuffer<u32>,
    pub widths: DeviceBuffer<f32>,
    pub densities: DeviceBuffer<f32>,
    pub status: Option<ProposalBuffers>,
}

pub(super) struct ProposalBuffers {
    pub partials: DeviceBuffer<u32>,
    pub summary: DeviceBuffer<u32>,
}

pub(super) struct WeightBackup {
    pub buffers: Vec<DeviceBuffer<u16>>,
}

pub(super) fn block_key(tensor_id: u64, start: u64, length: u64) -> u64 {
    splitmix(tensor_id ^ splitmix(start) ^ splitmix(length ^ 0xa076_1d64_78bd_642f))
}

impl NeighborSampler {
    pub(super) fn new(specs: &[ProposalSpec], stream: &CudaStream) -> CudaResult<Self> {
        // The packed CUDA embedding is a second physical view of Metal's one
        // logical embedding/readout tensor.  It is materialized, but only the
        // row-major readout participates in neighbour geometry.
        let dimensions = specs
            .iter()
            .filter(|spec| spec.transpose_columns == 0)
            .try_fold(0_usize, |sum, spec| sum.checked_add(spec.elements))
            .ok_or("CUDA neighbour dimensions overflow")?;
        let mut points = Vec::new();
        let mut ranges = Vec::new();
        for (buffer, spec) in specs.iter().enumerate() {
            if spec.transpose_columns != 0 {
                continue;
            }
            let range_start = u32::try_from(points.len()).map_err(cuda_error)?;
            let block_len = spec.block_len as usize;
            for block in 0..spec.host_scales.len() {
                let count = (8192usize.saturating_mul(block_len) / dimensions)
                    .max(8)
                    .min(block_len);
                let key = block_key(
                    spec.tensor_id,
                    spec.logical_start + (block * block_len) as u64,
                    block_len as u64,
                );
                let scale = spec.host_scales[block];
                let weight = 1.0 / (block_len as f32 * scale * scale);
                for stratum in 0..count {
                    let start = stratum * block_len / count;
                    let end = (stratum + 1) * block_len / count;
                    let element = start
                        + splitmix(key ^ (stratum as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15))
                            as usize
                            % (end - start);
                    points.push(NeighborPoint {
                        physical: (block * block_len + element) as u64,
                        key,
                        element: element as u64,
                        metric_weight: (end - start) as f32 * weight,
                        scale,
                    });
                }
            }
            let count = u32::try_from(points.len()).map_err(cuda_error)? - range_start;
            ranges.push(NeighborRange {
                buffer,
                start: range_start,
                count,
            });
        }
        let total = u32::try_from(points.len()).map_err(cuda_error)?;
        if total == 0 {
            return Err("CUDA neighbour quadrature is empty".into());
        }
        let total_usize = total as usize;
        Ok(Self {
            points: DeviceBuffer::from_host(stream, &points).map_err(cuda_error)?,
            seeds: DeviceBuffer::zeroed(stream, 4).map_err(cuda_error)?,
            pool: DeviceBuffer::zeroed(stream, 4 * total_usize).map_err(cuda_error)?,
            history: DeviceBuffer::zeroed(stream, 128 * total_usize).map_err(cuda_error)?,
            distances: DeviceBuffer::zeroed(stream, 4 * 128).map_err(cuda_error)?,
            ranges,
            total,
        })
    }

    pub(super) fn reset_history(
        &mut self,
        backup: &WeightBackup,
        module: &fbt_model::LoadedModule,
        stream: &CudaStream,
    ) -> CudaResult<()> {
        for range in &self.ranges {
            let base = backup
                .buffers
                .get(range.buffer)
                .ok_or("CUDA neighbour base inventory is incomplete")?;
            let launch = module
                .prepare_mhc4_capture(LaunchConfig1D::new(range.count.div_ceil(256), 256, 0))
                .map_err(cuda_error)?;
            module
                .mhc4_capture(
                    stream,
                    &launch,
                    base,
                    &self.points,
                    &mut self.history,
                    range.start,
                    range.count,
                    self.total,
                    0,
                )
                .map_err(cuda_error)?;
        }
        Ok(())
    }

    pub(super) fn pool_distances(
        &mut self,
        backup: &WeightBackup,
        random: &GaussianRandom,
        module: &fbt_model::LoadedModule,
        stream: &CudaStream,
        radius: f32,
        seeds: &[u64],
        history_count: usize,
    ) -> CudaResult<Vec<Vec<f32>>> {
        if history_count == 0
            || history_count > baux::semantics::optimizer::MAX_HISTORY
            || !matches!(seeds.len(), 1 | 4)
        {
            return Err("CUDA mHC-4 neighbour history must be in 1..=128".into());
        }
        copy_prefix(&self.seeds, seeds, stream)?;
        for range in &self.ranges {
            let base = backup
                .buffers
                .get(range.buffer)
                .ok_or("CUDA neighbour base inventory is incomplete")?;
            let launch = module
                .prepare_mhc4_pool(LaunchConfig1D::new(range.count.div_ceil(256), 256, 0))
                .map_err(cuda_error)?;
            module
                .mhc4_pool(
                    stream,
                    &launch,
                    base,
                    &self.points,
                    &self.seeds,
                    &random.thresholds,
                    &random.widths,
                    &random.densities,
                    &mut self.pool,
                    range.start,
                    range.count,
                    self.total,
                    radius,
                    seeds.len() as u32,
                )
                .map_err(cuda_error)?;
        }
        let pairs = u32::try_from(seeds.len() * history_count).map_err(cuda_error)?;
        let launch = module
            .prepare_mhc4_distances(LaunchConfig1D::new(pairs, 256, 0))
            .map_err(cuda_error)?;
        module
            .mhc4_distances(
                stream,
                &launch,
                &self.pool,
                &self.history,
                &self.points,
                &mut self.distances,
                self.total,
                history_count as u32,
            )
            .map_err(cuda_error)?;
        let values = read_prefix(&self.distances, stream, seeds.len() * history_count)?;
        Ok((0..seeds.len())
            .map(|candidate| {
                values[candidate * history_count..(candidate + 1) * history_count].to_vec()
            })
            .collect())
    }

    pub(super) fn archive(
        &mut self,
        candidate: usize,
        row: usize,
        stream: &CudaStream,
    ) -> CudaResult<()> {
        if candidate >= 4 || row >= 128 {
            return Err("CUDA neighbour archive index is out of range".into());
        }
        let bytes = self.total as usize * std::mem::size_of::<u16>();
        unsafe {
            cuda_core::simt::memory::memcpy_dtod_async(
                self.history.cu_deviceptr() + (row * bytes) as u64,
                self.pool.cu_deviceptr() + (candidate * bytes) as u64,
                bytes,
                stream.cu_stream(),
            )
            .map_err(cuda_error)
        }
    }
}

impl GaussianRandom {
    pub(super) fn new(stream: &CudaStream) -> CudaResult<Self> {
        let tables = &baux::semantics::gaussian::ZIGGURAT;
        if !baux::semantics::gaussian::normal_bounds(
            tables.thresholds[0],
            tables.thresholds[192],
            tables.thresholds[129],
            tables.thresholds[221],
        ) {
            return Err("CUDA Gaussian table does not satisfy integer retry witnesses".into());
        }
        Ok(Self {
            thresholds: DeviceBuffer::from_host(stream, &tables.thresholds).map_err(cuda_error)?,
            widths: DeviceBuffer::from_host(stream, &tables.widths).map_err(cuda_error)?,
            densities: DeviceBuffer::from_host(stream, &tables.densities).map_err(cuda_error)?,
            status: None,
        })
    }

    pub(super) fn prepare(
        &mut self,
        specs: &[ProposalSpec],
        stream: &CudaStream,
    ) -> CudaResult<()> {
        let blocks = specs
            .iter()
            .map(|spec| spec.elements.div_ceil(256))
            .max()
            .filter(|&n| n != 0)
            .ok_or("CUDA proposal inventory is empty")?;
        u32::try_from(blocks).map_err(cuda_error)?;
        u32::try_from(specs.len()).map_err(cuda_error)?;
        if self.status.as_ref().is_none_or(|status| {
            status.partials.len() < blocks || status.summary.len() < specs.len()
        }) {
            self.status = Some(ProposalBuffers {
                partials: DeviceBuffer::zeroed(stream, blocks).map_err(cuda_error)?,
                summary: DeviceBuffer::zeroed(stream, specs.len()).map_err(cuda_error)?,
            });
        }
        Ok(())
    }

    pub(super) fn step_gaussian(
        &mut self,
        specs: &[ProposalSpec],
        backup: &WeightBackup,
        candidates: &mut [&mut DeviceBuffer<u16>],
        module: &fbt_model::LoadedModule,
        stream: &CudaStream,
        radius: f32,
        seed: u64,
    ) -> CudaResult<f32> {
        self.prepare(specs, stream)?;
        let status = self
            .status
            .as_mut()
            .ok_or("CUDA proposal status is missing")?;
        let summary_launch = module
            .prepare_proposal_status(LaunchConfig1D::new(1, 256, 0))
            .map_err(cuda_error)?;
        let start = timing_event(stream)?;
        if candidates.len() != specs.len() {
            return Err("CUDA proposal did not visit every model buffer".into());
        }
        for (index, candidate) in candidates.iter_mut().enumerate() {
            let base = backup
                .buffers
                .get(index)
                .ok_or("CUDA proposal base is incomplete")?;
            let spec = specs
                .get(index)
                .ok_or("CUDA proposal scale inventory is incomplete")?;
            if candidate.len() != spec.elements || base.len() != spec.elements {
                return Err("CUDA proposal buffer shape differs from its inventory".into());
            }
            let elements = u64::try_from(candidate.len()).map_err(cuda_error)?;
            let blocks = u32::try_from(candidate.len().div_ceil(256)).map_err(cuda_error)?;
            let launch = module
                .prepare_gaussian_fp16(LaunchConfig1D::new(blocks, 256, 0))
                .map_err(cuda_error)?;
            module
                .gaussian_fp16(
                    stream,
                    &launch,
                    base,
                    &spec.scales,
                    &self.thresholds,
                    &self.widths,
                    &self.densities,
                    candidate,
                    &mut status.partials,
                    elements,
                    spec.block_len,
                    radius,
                    spec.tensor_id,
                    spec.logical_start,
                    spec.transpose_columns,
                    seed,
                )
                .map_err(cuda_error)?;
            module
                .proposal_status(
                    stream,
                    &summary_launch,
                    &status.partials,
                    &mut status.summary,
                    blocks,
                    u32::try_from(index).map_err(cuda_error)?,
                )
                .map_err(cuda_error)?;
        }
        let end = timing_event(stream)?;
        let flags = read_prefix(&status.summary, stream, specs.len())?
            .into_iter()
            .fold(0, |flags, next| flags | next);
        baux::semantics::proposal::validate(flags, false)?;
        start.elapsed_ms(&end).map_err(cuda_error)
    }

    pub(super) fn materialize(
        &mut self,
        specs: &[ProposalSpec],
        backup: &WeightBackup,
        candidates: &mut [&mut DeviceBuffer<u16>],
        module: &fbt_model::LoadedModule,
        stream: &CudaStream,
        radius: f32,
        seed: u64,
    ) -> CudaResult<f32> {
        if !radius.is_finite() || radius <= 0.0 || backup.buffers.len() != specs.len() {
            return Err("CUDA Gaussian proposal is invalid".into());
        }
        let result = self.step_gaussian(specs, backup, candidates, module, stream, radius, seed);
        if let Err(error) = result {
            backup.restore(candidates, stream).map_err(|restore| {
                format!("{error}; restoring incumbent also failed: {restore}")
            })?;
            return Err(error);
        }
        result
    }
}

fn check_clean(buffers: &[DeviceBuffer<u16>; 3], stream: &CudaStream) -> CudaResult<()> {
    for buffer in buffers {
        let values = read_prefix(buffer, stream, buffer.len())?;
        if values.iter().any(|&v| v != 0x3c00) {
            return Err("rejected proposal did not restore the complete incumbent".into());
        }
    }
    Ok(())
}

pub(super) fn check() -> CudaResult<()> {
    let context = cuda_core::CudaContext::new(0).map_err(cuda_error)?;
    let stream = context.default_stream();
    let module = crate::capture::load_model(&context)?;
    let lengths = [1025, 33, 257];
    let upload = |count| DeviceBuffer::from_host(&stream, &vec![0x3c00; count]).map_err(cuda_error);
    let mut specs = Vec::new();
    for (index, &elements) in lengths.iter().enumerate() {
        let host_scales = vec![0.25; elements];
        specs.push(ProposalSpec {
            elements,
            block_len: 1,
            scales: DeviceBuffer::from_host(&stream, &host_scales).map_err(cuda_error)?,
            host_scales,
            tensor_id: index as u64 + 1,
            logical_start: 0,
            transpose_columns: 0,
        });
    }
    let mut buffers = [
        upload(lengths[0])?,
        upload(lengths[1])?,
        upload(lengths[2])?,
    ];
    let mut random = GaussianRandom::new(&stream)?;
    let backup = WeightBackup::new(
        &[&buffers[0], &buffers[1], &buffers[2]],
        specs.len(),
        &stream,
    )?;
    // copy_prefix's safe slice interface also applies to pinned memory, whose
    // DMA is genuinely asynchronous unless the helper waits for completion.
    let mut staging = cuda_core::PinnedHostBuffer::from_slice(&context, &vec![0.5_f32; lengths[0]])
        .map_err(cuda_error)?;
    copy_prefix(&specs[0].scales, staging.as_slice(), &stream)?;
    staging.as_mut_slice().fill(0.75);
    if read_prefix(&specs[0].scales, &stream, lengths[0])? != vec![0.5; lengths[0]] {
        return Err("borrowed pinned upload outlived its source borrow".into());
    }
    copy_prefix(&specs[0].scales, &vec![0.25; lengths[0]], &stream)?;
    {
        let [b0, b1, b2] = &mut buffers;
        if random
            .materialize(
                &specs,
                &backup,
                &mut [b0, b1, b2],
                &module,
                &stream,
                0.0,
                17,
            )
            .is_ok()
        {
            return Err("proposal fixture accepted zero radius".into());
        }
        random.materialize(
            &specs,
            &backup,
            &mut [b0, b1, b2],
            &module,
            &stream,
            0.01,
            17,
        )?;
    }
    let status = random
        .status
        .as_ref()
        .ok_or("proposal fixture omitted status")?;
    let actual = read_prefix(&status.summary, &stream, lengths.len())?;
    if actual != vec![baux::semantics::proposal::CHANGED; lengths.len()] {
        return Err(format!(
            "valid proposal reported incorrect flags: {actual:?}"
        ));
    }
    check_reject(
        &mut buffers,
        &mut specs,
        &mut random,
        &backup,
        &module,
        &stream,
        lengths[0],
    )?;
    context.check_err().map_err(cuda_error)?;
    println!(
        "PROPOSAL_VALIDITY ok=true tensors=1025,33,257 valid=true invalid_tail=rejected \
         unchanged=rejected rollback=exact recovery=true"
    );
    Ok(())
}

fn check_reject(
    buffers: &mut [DeviceBuffer<u16>; 3],
    specs: &mut [ProposalSpec],
    random: &mut GaussianRandom,
    backup: &WeightBackup,
    module: &fbt_model::LoadedModule,
    stream: &CudaStream,
    first_len: usize,
) -> CudaResult<()> {
    // Overflow only the final coordinate of the first tensor. The other
    // tensors have smaller status ranges; their reuse must not erase this flag.
    let mut scales = vec![0.25; first_len];
    *scales.last_mut().ok_or("empty fixture tensor")? = f32::MAX;
    copy_prefix(&specs[0].scales, &scales, stream)?;
    let result = {
        let [b0, b1, b2] = buffers;
        random.materialize(specs, backup, &mut [b0, b1, b2], module, stream, 0.01, 17)
    };
    if !matches!(result, Err(ref error) if error == "selected proposal is nonfinite") {
        return Err(format!(
            "invalid tail proposal was not rejected: {result:?}"
        ));
    }
    check_clean(buffers, stream)?;
    copy_prefix(&specs[0].scales, &vec![0.25; first_len], stream)?;
    let result = {
        let [b0, b1, b2] = buffers;
        random.materialize(specs, backup, &mut [b0, b1, b2], module, stream, 0.01, 17)?;
        // Ordinary nonzero Gaussian draws at this radius round away at the unit
        // incumbent. Materialization must not inject a one-ULP replacement
        // update.
        random.materialize(
            specs,
            backup,
            &mut [b0, b1, b2],
            module,
            stream,
            1.0e-20,
            17,
        )
    };
    if !matches!(result, Err(ref error) if error == "selected proposal did not change") {
        return Err(format!(
            "rounded-away Gaussian proposal was not rejected: {result:?}"
        ));
    }
    check_clean(buffers, stream)?;
    // Force the fixture's normal draws to zero without altering production
    // tables or kernels; all candidates then equal the finite incumbent.
    copy_prefix(&random.thresholds, &[u32::MAX; 256], stream)?;
    copy_prefix(&random.widths, &[0.0; 256], stream)?;
    let result = {
        let [b0, b1, b2] = buffers;
        random.materialize(specs, backup, &mut [b0, b1, b2], module, stream, 0.01, 17)
    };
    if !matches!(result, Err(ref error) if error == "selected proposal did not change") {
        return Err(format!("unchanged proposal was not rejected: {result:?}"));
    }
    check_clean(buffers, stream)?;
    copy_prefix(
        &random.thresholds,
        &baux::semantics::gaussian::ZIGGURAT.thresholds,
        stream,
    )?;
    copy_prefix(
        &random.widths,
        &baux::semantics::gaussian::ZIGGURAT.widths,
        stream,
    )?;
    let [b0, b1, b2] = buffers;
    random.materialize(specs, backup, &mut [b0, b1, b2], module, stream, 0.01, 17)?;
    Ok(())
}

pub(super) fn copy_buffer(
    destination: &DeviceBuffer<u16>,
    source: &DeviceBuffer<u16>,
    stream: &CudaStream,
) -> CudaResult<()> {
    if destination.len() != source.len() {
        return Err("CUDA weight backup shape mismatch".into());
    }
    unsafe {
        cuda_core::simt::memory::memcpy_dtod_async(
            destination.cu_deviceptr(),
            source.cu_deviceptr(),
            source
                .len()
                .checked_mul(std::mem::size_of::<u16>())
                .ok_or("CUDA weight backup byte overflow")?,
            stream.cu_stream(),
        )
        .map_err(cuda_error)
    }
}

impl WeightBackup {
    pub(super) fn new(
        sources: &[&DeviceBuffer<u16>],
        expected: usize,
        stream: &CudaStream,
    ) -> CudaResult<Self> {
        let mut buffers = Vec::with_capacity(expected);
        for source in sources {
            let backup = DeviceBuffer::zeroed(stream, source.len()).map_err(cuda_error)?;
            copy_buffer(&backup, source, stream)?;
            buffers.push(backup);
        }
        if buffers.len() != expected {
            return Err("CUDA proposal inventory does not cover every model buffer".into());
        }
        Ok(Self { buffers })
    }

    pub(super) fn restore(
        &self,
        destinations: &mut [&mut DeviceBuffer<u16>],
        stream: &CudaStream,
    ) -> CudaResult<f32> {
        if destinations.len() != self.buffers.len() {
            return Err("CUDA weight backup contains extra buffers".into());
        }
        let start = timing_event(stream)?;
        for (destination, source) in destinations.iter_mut().zip(&self.buffers) {
            copy_buffer(destination, source, stream)?;
        }
        let end = timing_event(stream)?;
        start.elapsed_ms(&end).map_err(cuda_error)
    }

    pub(super) fn commit(
        &mut self,
        sources: &[&DeviceBuffer<u16>],
        stream: &CudaStream,
    ) -> CudaResult<f32> {
        if sources.len() != self.buffers.len() {
            return Err("CUDA weight backup contains extra buffers".into());
        }
        let start = timing_event(stream)?;
        for (destination, source) in self.buffers.iter().zip(sources) {
            copy_buffer(destination, source, stream)?;
        }
        let end = timing_event(stream)?;
        start.elapsed_ms(&end).map_err(cuda_error)
    }

    pub(super) fn step_basis0(
        &self,
        specs: &[ProposalSpec],
        candidates: &mut [&mut DeviceBuffer<u16>],
        module: &fbt_model::LoadedModule,
        stream: &CudaStream,
        radius: f32,
    ) -> CudaResult<f32> {
        if !radius.is_finite() || radius <= 0.0 || self.buffers.len() != specs.len() {
            return Err("CUDA polynomial-threshold proposal is invalid".into());
        }
        if candidates.len() != specs.len() {
            return Err("CUDA proposal did not visit every model buffer".into());
        }
        let start = timing_event(stream)?;
        for (index, candidate) in candidates.iter_mut().enumerate() {
            let base = self
                .buffers
                .get(index)
                .ok_or("CUDA proposal base is incomplete")?;
            let spec = specs
                .get(index)
                .ok_or("CUDA proposal scale inventory is incomplete")?;
            let elements = u64::try_from(candidate.len()).map_err(cuda_error)?;
            let blocks = u32::try_from(candidate.len().div_ceil(256)).map_err(cuda_error)?;
            let launch = module
                .prepare_threshold_basis0_fp16(LaunchConfig1D::new(blocks, 256, 0))
                .map_err(cuda_error)?;
            module
                .threshold_basis0_fp16(
                    stream,
                    &launch,
                    base,
                    &spec.scales,
                    candidate,
                    elements,
                    spec.block_len,
                    radius,
                )
                .map_err(cuda_error)?;
        }
        let end = timing_event(stream)?;
        start.elapsed_ms(&end).map_err(cuda_error)
    }
}
