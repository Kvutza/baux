use std::mem::size_of;

use cuda_core::{CudaStream, DeviceBuffer, LaunchConfig1D};

use super::*;

mod ask_step;
use ask_step::{bf16_tiles, validate_tell};

const BF16_PENDING: usize = 32;
const BF16_SMALL_TILE_ELEMENTS: usize = 4_096;
const BF16_TILE_ELEMENTS: usize = 16_384;

/// Selected seed, acquisition score, nominal radius, and per-leaf (changed,
/// squared L2).
pub type ProposalDescription = (u64, f32, f32, Vec<(u64, f64)>);

struct Bf16Scratch {
    history_capacity: usize,
    candidate_capacity: usize,
    region_capacity: usize,
    partial_capacity: usize,
    status_capacity: usize,
    history_slots: DeviceBuffer<u32>,
    outcomes: DeviceBuffer<f32>,
    variances: DeviceBuffer<f32>,
    seeds: DeviceBuffer<Seed>,
    draws: DeviceBuffer<f32>,
    scores: DeviceBuffer<f32>,
    partials: DeviceBuffer<f32>,
    tile_status: DeviceBuffer<u32>,
    selection: DeviceBuffer<Selection>,
    trial_slots: DeviceBuffer<u32>,
    destinations: DeviceBuffer<u32>,
    status: DeviceBuffer<u32>,
    changes: DeviceBuffer<Bf16Change>,
}

struct AskInput<'a> {
    base_slot: usize,
    history: usize,
    trial_slots: &'a [u32],
    seeds: &'a [u64],
    seed_root: Option<u64>,
    candidates_per_region: usize,
    coefficient: f32,
    draw_seed: u64,
    config: RoundConfig,
}

#[derive(Debug, Clone)]
pub struct TellOutput {
    pub accepted: Vec<bool>,
    pub length: f64,
    pub best: f32,
    pub best_variance: f32,
    pub history: usize,
    pub restarts: usize,
    pub restarted: bool,
}

#[derive(Clone, Copy)]
struct SearchShape {
    regions: usize,
    candidates: usize,
    blocks: usize,
    partials: usize,
    status: usize,
}

#[derive(Clone, Copy)]
struct SingleRow {
    slot: usize,
    base_slot: usize,
    coefficient: f32,
    resident: bool,
}

impl Bf16Scratch {
    fn new(stream: &CudaStream) -> CudaResult<Self> {
        Ok(Self {
            history_capacity: 1,
            candidate_capacity: 1,
            region_capacity: 1,
            partial_capacity: 1,
            status_capacity: 1,
            history_slots: DeviceBuffer::zeroed(stream, 1).map_err(cuda_error)?,
            outcomes: DeviceBuffer::zeroed(stream, 1).map_err(cuda_error)?,
            variances: DeviceBuffer::zeroed(stream, 1).map_err(cuda_error)?,
            seeds: DeviceBuffer::zeroed(stream, 1).map_err(cuda_error)?,
            draws: DeviceBuffer::zeroed(stream, 1).map_err(cuda_error)?,
            scores: DeviceBuffer::zeroed(stream, 1).map_err(cuda_error)?,
            partials: DeviceBuffer::zeroed(stream, 1).map_err(cuda_error)?,
            tile_status: DeviceBuffer::zeroed(stream, 1).map_err(cuda_error)?,
            selection: DeviceBuffer::zeroed(stream, 1).map_err(cuda_error)?,
            trial_slots: DeviceBuffer::zeroed(stream, 1).map_err(cuda_error)?,
            destinations: DeviceBuffer::zeroed(stream, 1).map_err(cuda_error)?,
            status: DeviceBuffer::zeroed(stream, 1).map_err(cuda_error)?,
            changes: DeviceBuffer::zeroed(stream, 1).map_err(cuda_error)?,
        })
    }

    fn ensure(
        &mut self,
        stream: &CudaStream,
        history: usize,
        candidates: usize,
        regions: usize,
        partials: usize,
        status: usize,
    ) -> CudaResult<()> {
        let history_capacity = next_capacity(history, "BF16 history")?;
        let candidate_capacity = next_capacity(candidates, "BF16 candidates")?;
        let region_capacity = next_capacity(regions, "BF16 regions")?;
        let partial_capacity = next_capacity(partials, "BF16 distance partials")?;
        let status_capacity = next_capacity(status, "BF16 status")?;
        if history_capacity > self.history_capacity {
            self.history_slots =
                DeviceBuffer::zeroed(stream, history_capacity).map_err(cuda_error)?;
            self.outcomes = DeviceBuffer::zeroed(stream, history_capacity).map_err(cuda_error)?;
            self.variances = DeviceBuffer::zeroed(stream, history_capacity).map_err(cuda_error)?;
            self.draws = DeviceBuffer::zeroed(stream, history_capacity).map_err(cuda_error)?;
            self.history_capacity = history_capacity;
        }
        if candidate_capacity > self.candidate_capacity {
            self.seeds = DeviceBuffer::zeroed(stream, candidate_capacity).map_err(cuda_error)?;
            self.scores = DeviceBuffer::zeroed(stream, candidate_capacity).map_err(cuda_error)?;
            self.candidate_capacity = candidate_capacity;
        }
        if partial_capacity > self.partial_capacity {
            self.partials = DeviceBuffer::zeroed(stream, partial_capacity).map_err(cuda_error)?;
            self.tile_status =
                DeviceBuffer::zeroed(stream, partial_capacity).map_err(cuda_error)?;
            self.partial_capacity = partial_capacity;
        }
        if region_capacity > self.region_capacity {
            self.selection = DeviceBuffer::zeroed(stream, region_capacity).map_err(cuda_error)?;
            self.trial_slots = DeviceBuffer::zeroed(stream, region_capacity).map_err(cuda_error)?;
            self.destinations =
                DeviceBuffer::zeroed(stream, region_capacity).map_err(cuda_error)?;
            self.region_capacity = region_capacity;
        }
        if status_capacity > self.status_capacity {
            self.status = DeviceBuffer::zeroed(stream, status_capacity).map_err(cuda_error)?;
            self.changes = DeviceBuffer::zeroed(stream, status_capacity).map_err(cuda_error)?;
            self.status_capacity = status_capacity;
        }
        Ok(())
    }
}

/// CUDA-resident BF16 candidate scoring and materialization.
pub struct Bf16SearchEngine {
    runtime: Runtime,
    rows: DeviceBuffer<u16>,
    batch: DeviceBuffer<u16>,
    reference: DeviceBuffer<u16>,
    reference_scales: DeviceBuffer<f32>,
    reference_partials: DeviceBuffer<f32>,
    tile_offsets: DeviceBuffer<u32>,
    sampler: u32,
    reference_seed: Option<u64>,
    paired_relative: Option<usize>,
    search_capacity: Option<usize>,
    started: bool,
    best_source: DeviceBuffer<u32>,
    single_slot: Option<SingleRow>,
    leaves: DeviceBuffer<Bf16Leaf>,
    tiles: DeviceBuffer<DenseTile>,
    row_len: usize,
    row_stride: usize,
    slots: usize,
    tile_count: usize,
    tile_leaves: Vec<usize>,
    leaf_count: usize,
    scratch: Bf16Scratch,
    state: DeviceBuffer<SearchState>,
    summary: DeviceBuffer<TellSummary>,
    accepted: DeviceBuffer<u32>,
    tell_values: DeviceBuffer<f32>,
    tell_variances: DeviceBuffer<f32>,
    profiling: bool,
    fused_pairs: bool,
    last_profile: Option<ProposalProfile>,
}

impl Bf16SearchEngine {
    pub fn new(base: &[u16], leaves: &[Bf16Leaf], slots: usize) -> CudaResult<Self> {
        validate_bf16(base.len(), leaves)?;
        if base.iter().any(|value| !bf16_finite(*value)) {
            return Err("CUDA BF16 search base values must be finite".to_string());
        }
        let mut engine = Self::allocate(base.len(), leaves, slots)?;
        copy_prefix(&engine.rows, base, &engine.runtime.stream)?;
        engine.validate(0)?;
        Ok(engine)
    }

    /// Copy a contiguous BF16 CUDA allocation into persistent search storage.
    ///
    /// # Safety
    /// `pointer` must address at least `len * 2` readable bytes on CUDA device
    /// 0.
    pub unsafe fn from_device(
        pointer: u64,
        len: usize,
        leaves: &[Bf16Leaf],
        slots: usize,
    ) -> CudaResult<Self> {
        if pointer == 0 {
            return Err("CUDA BF16 search requires a device base".to_string());
        }
        validate_bf16(len, leaves)?;
        let mut engine = Self::allocate(len, leaves, slots)?;
        unsafe {
            cuda_core::simt::memory::memcpy_dtod_async(
                engine.rows.cu_deviceptr(),
                pointer,
                row_bytes(len)?,
                engine.runtime.stream.cu_stream(),
            )
            .map_err(cuda_error)?;
        }
        engine.validate(0)?;
        Ok(engine)
    }

    fn allocate(len: usize, leaves: &[Bf16Leaf], slots: usize) -> CudaResult<Self> {
        if slots < 2 {
            return Err("CUDA BF16 search requires at least two row slots".to_string());
        }
        let tiles = bf16_tiles(leaves)?;
        let tile_leaves = tiles.iter().map(|tile| tile.leaf as usize).collect();
        let leaf_count = leaves.len();
        let runtime = Runtime::new()?;
        let row_stride = len
            .checked_add(127)
            .ok_or("CUDA BF16 row stride overflow")?
            & !127;
        let row_count = slots
            .checked_mul(row_stride)
            .ok_or("CUDA BF16 resident row count overflow")?;
        let rows = DeviceBuffer::zeroed(&runtime.stream, row_count).map_err(cuda_error)?;
        let batch = DeviceBuffer::zeroed(&runtime.stream, 1).map_err(cuda_error)?;
        let reference = DeviceBuffer::zeroed(&runtime.stream, 1).map_err(cuda_error)?;
        let reference_scales = DeviceBuffer::zeroed(&runtime.stream, 1).map_err(cuda_error)?;
        let reference_partials = DeviceBuffer::zeroed(&runtime.stream, 1).map_err(cuda_error)?;
        let offsets = (0..=leaves.len())
            .map(|leaf| {
                to_u32(
                    tiles.partition_point(|tile| (tile.leaf as usize) < leaf),
                    "BF16 leaf tiles",
                )
            })
            .collect::<CudaResult<Vec<_>>>()?;
        let tile_offsets =
            DeviceBuffer::from_host(&runtime.stream, &offsets).map_err(cuda_error)?;
        let best_source = DeviceBuffer::zeroed(&runtime.stream, 1).map_err(cuda_error)?;
        let leaves = DeviceBuffer::from_host(&runtime.stream, leaves).map_err(cuda_error)?;
        let tile_count = tiles.len();
        let tiles = DeviceBuffer::from_host(&runtime.stream, &tiles).map_err(cuda_error)?;
        let scratch = Bf16Scratch::new(&runtime.stream)?;
        let state = DeviceBuffer::zeroed(&runtime.stream, 1).map_err(cuda_error)?;
        let summary = DeviceBuffer::zeroed(&runtime.stream, 1).map_err(cuda_error)?;
        let accepted = DeviceBuffer::zeroed(&runtime.stream, BF16_PENDING).map_err(cuda_error)?;
        let tell_values =
            DeviceBuffer::zeroed(&runtime.stream, BF16_PENDING).map_err(cuda_error)?;
        let tell_variances =
            DeviceBuffer::zeroed(&runtime.stream, BF16_PENDING).map_err(cuda_error)?;
        Ok(Self {
            runtime,
            rows,
            batch,
            reference,
            reference_scales,
            reference_partials,
            tile_offsets,
            sampler: 0,
            reference_seed: None,
            paired_relative: None,
            search_capacity: None,
            started: false,
            best_source,
            single_slot: None,
            leaves,
            tiles,
            row_len: len,
            row_stride,
            slots,
            tile_count,
            tile_leaves,
            leaf_count,
            scratch,
            state,
            summary,
            accepted,
            tell_values,
            tell_variances,
            profiling: false,
            fused_pairs: true,
            last_profile: None,
        })
    }

    pub fn set_profiling(&mut self, enabled: bool) {
        set_profile(enabled, &mut self.profiling, &mut self.last_profile);
    }

    pub fn set_fused_pairs(&mut self, enabled: bool) {
        self.fused_pairs = enabled;
    }

    pub fn enable_correlated(&mut self, reference_seed: u64) -> CudaResult<()> {
        if self.sampler != 0 || self.single_slot.is_some() {
            return Err("Correlated BF16 sampling must be enabled before the first ask".into());
        }
        // Delay the model-sized reference until ask, after the caller can
        // release its input copy.
        self.reference_seed = Some(reference_seed);
        self.sampler = 1;
        Ok(())
    }

    pub fn enable_gaussian(&mut self) -> CudaResult<()> {
        if self.sampler != 0 || self.single_slot.is_some() {
            return Err("Gaussian BF16 sampling must be enabled before the first ask".into());
        }
        self.sampler = 2;
        Ok(())
    }

    /// Initialize the pinned zero anchor before the first correlated round.
    pub fn enable_relative(&mut self, failure_tolerance: usize) -> CudaResult<()> {
        if self.sampler != 1
            || self.started
            || self.paired_relative.is_some()
            || !self.search_capacity.is_some_and(|capacity| capacity >= 2)
            || failure_tolerance == 0
            || u32::try_from(failure_tolerance).is_err()
        {
            return Err(
                "Paired-relative mode requires a fresh initialized correlated search, capacity >= \
                 2, and a positive u32 failure tolerance"
                    .into(),
            );
        }
        copy_prefix(&self.scratch.history_slots, &[1], &self.runtime.stream)?;
        copy_prefix(&self.scratch.outcomes, &[0.0], &self.runtime.stream)?;
        copy_prefix(&self.scratch.variances, &[0.0], &self.runtime.stream)?;
        self.paired_relative = Some(failure_tolerance);
        Ok(())
    }

    fn ensure_reference(&mut self) -> CudaResult<()> {
        let Some(seed) = self.reference_seed else {
            return Ok(());
        };
        self.reference =
            DeviceBuffer::zeroed(&self.runtime.stream, self.row_len).map_err(cuda_error)?;
        self.reference_scales =
            DeviceBuffer::zeroed(&self.runtime.stream, self.leaf_count).map_err(cuda_error)?;
        self.reference_partials =
            DeviceBuffer::zeroed(&self.runtime.stream, self.tile_count).map_err(cuda_error)?;
        self.launch_reference(seed, true)?;
        self.check_reference()?;
        self.reference_seed = None;
        Ok(())
    }

    fn check_reference(&self) -> CudaResult<()> {
        let scales = read_prefix(
            &self.reference_scales,
            &self.runtime.stream,
            self.leaf_count,
        )?;
        if scales
            .iter()
            .any(|scale| !scale.is_finite() || *scale <= 0.0)
        {
            return Err("BF16 Gaussian reference must have finite positive tensor RMS".into());
        }
        Ok(())
    }

    pub fn read_reference(&mut self) -> CudaResult<Vec<u16>> {
        if self.sampler != 1 {
            return Err("Only correlated sampling stores a reference".into());
        }
        self.ensure_reference()?;
        self.check_reference()?;
        read_prefix(&self.reference, &self.runtime.stream, self.row_len)
    }

    fn launch_reference(&mut self, seed: u64, initialize: bool) -> CudaResult<()> {
        let blocks = self.tile_count;
        let launch = self
            .runtime
            .module
            .prepare_reference_bf16(LaunchConfig1D::new(
                to_u32(blocks, "BF16 reference blocks")?,
                THREADS,
                0,
            ))
            .map_err(cuda_error)?;
        self.runtime
            .module
            .reference_bf16(
                &self.runtime.stream,
                &launch,
                &mut self.reference,
                &mut self.reference_partials,
                &self.reference_scales,
                &self.leaves,
                &self.tiles,
                &self.scratch.selection,
                &self.scratch.seeds,
                &self.accepted,
                &self.summary,
                seed,
                u32::from(initialize),
            )
            .map_err(cuda_error)?;
        let launch = self
            .runtime
            .module
            .prepare_reference_rmsbf16(LaunchConfig1D::new(
                to_u32(self.leaf_count, "BF16 reference tensors")?,
                THREADS,
                0,
            ))
            .map_err(cuda_error)?;
        self.runtime
            .module
            .reference_rmsbf16(
                &self.runtime.stream,
                &launch,
                &self.reference_partials,
                &self.leaves,
                &self.tile_offsets,
                &mut self.reference_scales,
                &self.accepted,
                &self.summary,
                u32::from(initialize),
            )
            .map_err(cuda_error)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn init_search(
        &mut self,
        base_value: f32,
        base_variance: f32,
        capacity: usize,
        length_init: f64,
        length_min: f64,
        length_max: f64,
    ) -> CudaResult<()> {
        if self.paired_relative.is_some() {
            return Err("Cannot reinitialize paired-relative BF16 search".into());
        }
        if capacity == 0
            || capacity > MAX_HISTORY
            || !base_value.is_finite()
            || !base_variance.is_finite()
            || base_variance < 0.0
            || !length_init.is_finite()
            || !length_min.is_finite()
            || !length_max.is_finite()
            || length_min <= 0.0
            || length_init < length_min
            || length_init > length_max
        {
            return Err("CUDA BF16 search state is invalid".to_string());
        }
        self.scratch
            .ensure(&self.runtime.stream, capacity, 1, 1, 1, self.tile_count)?;
        copy_prefix(&self.scratch.history_slots, &[1], &self.runtime.stream)?;
        copy_prefix(&self.scratch.outcomes, &[base_value], &self.runtime.stream)?;
        copy_prefix(
            &self.scratch.variances,
            &[base_variance],
            &self.runtime.stream,
        )?;
        let state = SearchState {
            length: length_init,
            length_init,
            length_min,
            length_max,
            best: base_value,
            best_variance: base_variance,
            trust_best: f64::from(base_value),
            hist_min: f64::from(base_value),
            hist_max: f64::from(base_value),
            prev_obs: 1,
            successes: 0,
            failures: 0,
            restarts: 0,
            history: 1,
            status: 0,
        };
        copy_prefix(&self.state, &[state], &self.runtime.stream)?;
        self.search_capacity = Some(capacity);
        Ok(())
    }

    pub fn tell(
        &mut self,
        trial_slots: &[u32],
        values: &[f32],
        variances: &[f32],
        capacity: usize,
        failure_tolerance: usize,
    ) -> CudaResult<TellOutput> {
        self.validate_tell(trial_slots, values.len(), variances.len(), capacity)?;
        copy_prefix(&self.tell_values, values, &self.runtime.stream)?;
        copy_prefix(&self.tell_variances, variances, &self.runtime.stream)?;
        self.launch_tell(trial_slots, values.len(), capacity, failure_tolerance)?;
        self.collect_tell(values.len())
    }

    pub fn queue_values(
        &mut self,
        trial_slots: &[u32],
        values: &[f32],
        variances: &[f32],
        capacity: usize,
        failure_tolerance: usize,
    ) -> CudaResult<()> {
        self.validate_tell(trial_slots, values.len(), variances.len(), capacity)?;
        copy_prefix(&self.tell_values, values, &self.runtime.stream)?;
        copy_prefix(&self.tell_variances, variances, &self.runtime.stream)?;
        self.launch_tell(trial_slots, values.len(), capacity, failure_tolerance)
    }

    /// Consume contiguous device-0 FP32 rewards without staging them through
    /// Python.
    ///
    /// # Safety
    /// The pointers must each address `count * 4` readable bytes on CUDA device
    /// 0.
    pub unsafe fn tell_device(
        &mut self,
        trial_slots: &[u32],
        values: u64,
        variances: Option<u64>,
        count: usize,
        capacity: usize,
        failure_tolerance: usize,
    ) -> CudaResult<TellOutput> {
        self.validate_tell(trial_slots, count, count, capacity)?;
        if values == 0 || variances == Some(0) {
            return Err("CUDA BF16 tell requires valid device rewards".to_string());
        }
        let bytes = count
            .checked_mul(size_of::<f32>())
            .ok_or("CUDA BF16 tell byte count overflow")?;
        unsafe {
            cuda_core::simt::memory::memcpy_dtod_async(
                self.tell_values.cu_deviceptr(),
                values,
                bytes,
                self.runtime.stream.cu_stream(),
            )
            .map_err(cuda_error)?;
            if let Some(pointer) = variances {
                cuda_core::simt::memory::memcpy_dtod_async(
                    self.tell_variances.cu_deviceptr(),
                    pointer,
                    bytes,
                    self.runtime.stream.cu_stream(),
                )
                .map_err(cuda_error)?;
            } else {
                cuda_core::simt::memory::memset_d8_async(
                    self.tell_variances.cu_deviceptr(),
                    0,
                    bytes,
                    self.runtime.stream.cu_stream(),
                )
                .map_err(cuda_error)?;
            }
        }
        self.launch_tell(trial_slots, count, capacity, failure_tolerance)?;
        self.collect_tell(count)
    }

    /// Queue resident rewards and trust-region adaptation without a host
    /// readback.
    ///
    /// # Safety
    /// The pointers must each address `count * 4` readable bytes on CUDA device
    /// 0.
    pub unsafe fn queue_tell(
        &mut self,
        trial_slots: &[u32],
        values: u64,
        variances: Option<u64>,
        count: usize,
        capacity: usize,
        failure_tolerance: usize,
    ) -> CudaResult<()> {
        self.validate_tell(trial_slots, count, count, capacity)?;
        if values == 0 || variances == Some(0) {
            return Err("CUDA BF16 tell requires valid device rewards".to_string());
        }
        let bytes = count
            .checked_mul(size_of::<f32>())
            .ok_or("CUDA BF16 tell byte count overflow")?;
        unsafe {
            cuda_core::simt::memory::memcpy_dtod_async(
                self.tell_values.cu_deviceptr(),
                values,
                bytes,
                self.runtime.stream.cu_stream(),
            )
            .map_err(cuda_error)?;
            if let Some(pointer) = variances {
                cuda_core::simt::memory::memcpy_dtod_async(
                    self.tell_variances.cu_deviceptr(),
                    pointer,
                    bytes,
                    self.runtime.stream.cu_stream(),
                )
                .map_err(cuda_error)?;
            } else {
                cuda_core::simt::memory::memset_d8_async(
                    self.tell_variances.cu_deviceptr(),
                    0,
                    bytes,
                    self.runtime.stream.cu_stream(),
                )
                .map_err(cuda_error)?;
            }
        }
        self.launch_tell(trial_slots, count, capacity, failure_tolerance)
    }

    fn validate_tell(
        &self,
        trial_slots: &[u32],
        values: usize,
        variances: usize,
        capacity: usize,
    ) -> CudaResult<()> {
        if self.sampler == 1 {
            self.check_reference()?;
        }
        if trial_slots.is_empty()
            || trial_slots.len() > BF16_PENDING
            || values != trial_slots.len()
            || variances != trial_slots.len()
            || capacity == 0
            || capacity > MAX_HISTORY
        {
            return Err("CUDA BF16 tell shape is invalid".to_string());
        }
        for &slot in trial_slots {
            self.check_slot(slot as usize)?;
        }
        Ok(())
    }

    fn launch_tell(
        &mut self,
        trial_slots: &[u32],
        count: usize,
        capacity: usize,
        failure_tolerance: usize,
    ) -> CudaResult<()> {
        self.launch_tellupdate(trial_slots, count, capacity, failure_tolerance, None, None)
    }

    /// Queue one correlated observation and an explicit paired promotion
    /// decision.
    #[allow(clippy::too_many_arguments)]
    pub fn queue_paired(
        &mut self,
        trial_slot: u32,
        value: f32,
        variance: f32,
        incumbent_value: f32,
        incumbent_variance: f32,
        accept: bool,
        capacity: usize,
    ) -> CudaResult<()> {
        if self.sampler != 1
            || !value.is_finite()
            || !variance.is_finite()
            || variance < 0.0
            || !incumbent_value.is_finite()
            || !incumbent_variance.is_finite()
            || incumbent_variance < 0.0
        {
            return Err(
                "Paired tell requires correlated sampling and finite rewards/nonnegative variances"
                    .into(),
            );
        }
        self.validate_tell(&[trial_slot], 1, 1, capacity)?;
        copy_prefix(&self.tell_values, &[value], &self.runtime.stream)?;
        copy_prefix(&self.tell_variances, &[variance], &self.runtime.stream)?;
        self.launch_tellupdate(
            &[trial_slot],
            1,
            capacity,
            1,
            Some((incumbent_value, incumbent_variance, accept)),
            None,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn queue_relative(
        &mut self,
        trial_slot: u32,
        value: f32,
        variance: f32,
        incumbent_value: f32,
        incumbent_variance: f32,
        improvement: f32,
        improvement_variance: f32,
        accept: bool,
        capacity: usize,
        reject_is_failure: bool,
    ) -> CudaResult<()> {
        let tolerance = self
            .paired_relative
            .ok_or("Enable paired-relative mode before telling relative observations")?;
        validate_tell(
            self.search_capacity,
            self.slots,
            capacity,
            trial_slot,
            self.single_slot,
        )?;
        if self.search_capacity != Some(capacity)
            || self.sampler != 1
            || [value, incumbent_value, improvement]
                .iter()
                .any(|value| !value.is_finite())
            || [variance, incumbent_variance, improvement_variance]
                .iter()
                .any(|value| !value.is_finite() || *value < 0.0)
        {
            return Err(
                "Paired-relative tell requires the initialized capacity, finite rewards and \
                 improvement, and nonnegative finite variances"
                    .into(),
            );
        }
        self.validate_tell(&[trial_slot], 1, 1, capacity)?;
        copy_prefix(&self.tell_values, &[value], &self.runtime.stream)?;
        copy_prefix(&self.tell_variances, &[variance], &self.runtime.stream)?;
        self.launch_tellupdate(
            &[trial_slot],
            1,
            capacity,
            tolerance,
            Some((incumbent_value, incumbent_variance, accept)),
            Some((improvement, improvement_variance, reject_is_failure)),
        )?;
        self.single_slot = None;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn launch_tellupdate(
        &mut self,
        trial_slots: &[u32],
        count: usize,
        capacity: usize,
        failure_tolerance: usize,
        paired: Option<(f32, f32, bool)>,
        relative: Option<(f32, f32, bool)>,
    ) -> CudaResult<()> {
        if self.paired_relative.is_some() != relative.is_some() {
            return Err("Cannot mix paired-relative and legacy BF16 tells".into());
        }
        self.started = true;
        // Legacy DLPack has no read-only flag. Reject consumer writes before
        // any reward or history update can accept the borrowed pending row.
        if let Some(single) = self.single_slot {
            if trial_slots == [single.slot as u32] {
                self.runtime.context.synchronize().map_err(cuda_error)?;
                let launch = self
                    .runtime
                    .module
                    .prepare_write_bf16(LaunchConfig1D::new(
                        to_u32(self.tile_count, "BF16 verification tiles")?,
                        THREADS,
                        0,
                    ))
                    .map_err(cuda_error)?;
                self.runtime
                    .module
                    .write_bf16(
                        &self.runtime.stream,
                        &launch,
                        &mut self.rows,
                        &mut self.batch,
                        &mut self.scratch.changes,
                        &self.state,
                        &self.scratch.seeds,
                        &self.reference,
                        &self.reference_scales,
                        &self.scratch.selection,
                        &self.leaves,
                        &self.tiles,
                        &self.scratch.trial_slots,
                        &mut self.scratch.status,
                        self.row_stride as u64,
                        self.row_len as u64,
                        to_u32(single.base_slot, "BF16 base slot")?,
                        to_u32(self.tile_count, "BF16 tile count")?,
                        single.coefficient,
                        u32::from(single.resident),
                        0,
                        1,
                        self.sampler,
                    )
                    .map_err(cuda_error)?;
            }
        }
        copy_prefix(&self.scratch.trial_slots, trial_slots, &self.runtime.stream)?;
        let launch = self
            .runtime
            .module
            .prepare_tell_bf16(LaunchConfig1D::new(1, THREADS, 0))
            .map_err(cuda_error)?;
        let params = TellParams {
            row_stride: self.row_stride as u64,
            row_len: self.row_len as u64,
            trials: to_u32(count, "BF16 tell count")?,
            capacity: to_u32(capacity, "BF16 history capacity")?,
            failure_tolerance: to_u32(failure_tolerance.max(1), "BF16 failure tolerance")?,
            status_count: to_u32(
                count
                    .checked_mul(self.tile_count)
                    .ok_or("BF16 proposal status count overflow")?,
                "BF16 proposal status count",
            )?,
            correlated: self.sampler,
            paired: if relative.is_some() {
                2
            } else {
                u32::from(paired.is_some())
            },
            accept: u32::from(paired.is_some_and(|(_, _, accept)| accept)),
            incumbent_value: paired.map_or(0.0, |(value, _, _)| value),
            incumbent_variance: paired.map_or(0.0, |(_, variance, _)| variance),
            improvement: relative.map_or(0.0, |(value, _, _)| value),
            improvement_variance: relative.map_or(0.0, |(_, variance, _)| variance),
            reject_is_failure: u32::from(relative.is_none_or(|(_, _, failure)| failure)),
        };
        self.runtime
            .module
            .tell_bf16(
                &self.runtime.stream,
                &launch,
                &mut self.scratch.history_slots,
                &mut self.scratch.outcomes,
                &mut self.scratch.variances,
                &self.scratch.trial_slots,
                &mut self.scratch.destinations,
                &self.tell_values,
                &self.tell_variances,
                &self.scratch.status,
                &mut self.accepted,
                &mut self.state,
                &mut self.summary,
                &self.scratch.selection,
                &mut self.best_source,
                params,
            )
            .map_err(cuda_error)?;
        let blocks = self.row_len.div_ceil(THREADS as usize).min(65_535);
        let launch = self
            .runtime
            .module
            .prepare_copy_tellbf16(LaunchConfig1D::new(
                to_u32(blocks, "BF16 copy blocks")?,
                THREADS,
                0,
            ))
            .map_err(cuda_error)?;
        self.runtime
            .module
            .copy_tellbf16(
                &self.runtime.stream,
                &launch,
                &mut self.rows,
                &self.scratch.trial_slots,
                &self.scratch.destinations,
                &self.summary,
                &self.best_source,
                params,
            )
            .map_err(cuda_error)?;
        if self.sampler == 1 {
            self.launch_reference(0, false)?;
        }
        self.runtime.context.check_err().map_err(cuda_error)
    }

    pub fn collect_tell(&self, count: usize) -> CudaResult<TellOutput> {
        if count == 0 || count > BF16_PENDING {
            return Err("CUDA BF16 summary count is invalid".to_string());
        }
        let summary = read_prefix(&self.summary, &self.runtime.stream, 1)?[0];
        if summary.status != 0 {
            return Err("CUDA BF16 proposal or rewards are invalid".to_string());
        }
        if self.sampler == 1 {
            self.check_reference()?;
        }
        let accepted = read_prefix(&self.accepted, &self.runtime.stream, count)?
            .into_iter()
            .map(|value| value != 0)
            .collect();
        Ok(TellOutput {
            accepted,
            length: summary.length,
            best: summary.best,
            best_variance: summary.best_variance,
            history: summary.history as usize,
            restarts: summary.restarts as usize,
            restarted: summary.restarted != 0,
        })
    }
}

fn validate_bf16(len: usize, leaves: &[Bf16Leaf]) -> CudaResult<()> {
    if len == 0 || leaves.is_empty() {
        return Err("CUDA BF16 search requires base values and leaves".to_string());
    }
    let mut expected = 0u64;
    for leaf in leaves {
        if leaf.offset != expected || leaf.length == 0 {
            return Err("CUDA BF16 leaves must form a contiguous non-empty layout".to_string());
        }
        if !leaf.scale.is_finite()
            || leaf.scale <= 0.0
            || !leaf.weight.is_finite()
            || leaf.weight <= 0.0
        {
            return Err("CUDA BF16 leaf scales and weights must be positive".to_string());
        }
        u32::try_from(leaf.length).map_err(|_| "CUDA BF16 leaf length exceeds u32".to_string())?;
        expected = expected
            .checked_add(leaf.length)
            .ok_or("CUDA BF16 leaf layout overflow")?;
    }
    if expected != len as u64 {
        return Err(format!(
            "CUDA BF16 leaf layout covers {expected} values, expected {len}"
        ));
    }
    Ok(())
}

fn bf16_finite(value: u16) -> bool {
    f32::from_bits(u32::from(value) << 16).is_finite()
}

fn row_bytes(len: usize) -> CudaResult<usize> {
    len.checked_mul(size_of::<u16>())
        .ok_or("CUDA BF16 row byte count overflow".to_string())
}

fn next_capacity(value: usize, name: &str) -> CudaResult<usize> {
    value
        .max(1)
        .checked_next_power_of_two()
        .ok_or_else(|| format!("CUDA {name} capacity overflow"))
}
