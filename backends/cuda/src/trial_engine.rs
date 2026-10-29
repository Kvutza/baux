use cuda_core::{DeviceBuffer, LaunchConfig1D};

use super::*;

struct TrialAsk<'a> {
    base_slot: usize,
    history_slots: &'a [u32],
    outcomes: &'a [f32],
    trial_slot: usize,
    seeds: &'a [u64],
    draws: &'a [f32],
    regions: usize,
    candidates_per_region: usize,
    centers: &'a [CenterStep],
    region_centers: &'a [u32],
    leaves: &'a [Leaf],
    config: RoundConfig,
    materialize_row: bool,
    read_scores: bool,
}

pub struct TrialEngine {
    runtime: Runtime,
    rows: DeviceBuffer<u8>,
    leaves: DeviceBuffer<Leaf>,
    tiles: DeviceBuffer<Tile>,
    row_bytes: usize,
    row_stride: usize,
    slots: usize,
    scratch: Scratch,
    profiling: bool,
    last_profile: Option<ProposalProfile>,
}

impl TrialEngine {
    pub fn new(base: &[u8], leaves: &[Leaf], tiles: &[Tile], slots: usize) -> CudaResult<Self> {
        if base.is_empty() {
            return Err("CUDA trial base row must not be empty".to_string());
        }
        if leaves.is_empty() || tiles.is_empty() {
            return Err("CUDA trial layout must contain leaves and tiles".to_string());
        }
        if slots < 2 {
            return Err("CUDA trial engine requires at least two row slots".to_string());
        }
        trial_layout(base.len(), leaves, tiles)?;
        let runtime = Runtime::new()?;
        let row_bytes = base.len();
        let row_stride = row_bytes
            .checked_add(255)
            .ok_or("CUDA resident row stride overflow")?
            & !255;
        let total_bytes = slots
            .checked_mul(row_stride)
            .ok_or("CUDA resident row byte count overflow")?;
        let rows = DeviceBuffer::zeroed(&runtime.stream, total_bytes).map_err(cuda_error)?;
        copy_prefix(&rows, base, &runtime.stream)?;
        let leaves = DeviceBuffer::from_host(&runtime.stream, leaves).map_err(cuda_error)?;
        let tiles = DeviceBuffer::from_host(&runtime.stream, tiles).map_err(cuda_error)?;
        let scratch = Scratch::new(&runtime.stream)?;
        Ok(Self {
            runtime,
            rows,
            leaves,
            tiles,
            row_bytes,
            row_stride,
            slots,
            scratch,
            profiling: false,
            last_profile: None,
        })
    }

    pub fn set_profiling(&mut self, enabled: bool) {
        set_profile(enabled, &mut self.profiling, &mut self.last_profile);
    }

    #[allow(clippy::too_many_arguments)]
    pub fn ask(
        &mut self,
        base_slot: usize,
        history_slots: &[u32],
        outcomes: &[f32],
        trial_slot: usize,
        seeds: &[u64],
        draws: &[f32],
        leaves: &[Leaf],
        config: RoundConfig,
        materialize_row: bool,
    ) -> CudaResult<(usize, f32)> {
        let (selections, _) = self.ask_impl(TrialAsk {
            base_slot,
            history_slots,
            outcomes,
            trial_slot,
            seeds,
            draws,
            regions: 1,
            candidates_per_region: seeds.len(),
            centers: &[],
            region_centers: &[],
            leaves,
            config,
            materialize_row,
            read_scores: false,
        })?;
        let selection = selections[0];
        Ok((selection.index as usize, selection.score))
    }

    #[allow(clippy::too_many_arguments)]
    pub fn ask_scores(
        &mut self,
        base_slot: usize,
        history_slots: &[u32],
        outcomes: &[f32],
        trial_slot: usize,
        seeds: &[u64],
        draws: &[f32],
        leaves: &[Leaf],
        config: RoundConfig,
        materialize_row: bool,
    ) -> CudaResult<(usize, Vec<f32>)> {
        let (selections, scores) = self.ask_impl(TrialAsk {
            base_slot,
            history_slots,
            outcomes,
            trial_slot,
            seeds,
            draws,
            regions: 1,
            candidates_per_region: seeds.len(),
            centers: &[],
            region_centers: &[],
            leaves,
            config,
            materialize_row,
            read_scores: true,
        })?;
        let selection = selections[0];
        Ok((selection.index as usize, scores))
    }

    #[allow(clippy::too_many_arguments)]
    pub fn ask_sparse(
        &mut self,
        base_slot: usize,
        history_slots: &[u32],
        outcomes: &[f32],
        trial_slot: usize,
        seeds: &[u64],
        draws: &[f32],
        edits: &[SparseEdit],
        num_pert: usize,
        leaves: &[Leaf],
        config: RoundConfig,
    ) -> CudaResult<(usize, f32)> {
        self.check_sparse(
            base_slot,
            history_slots,
            outcomes,
            trial_slot,
            seeds,
            draws,
            edits,
            num_pert,
            leaves,
            config,
        )?;
        let history = to_u32(history_slots.len(), "history rows")?;
        let candidates = to_u32(seeds.len(), "candidate count")?;
        let num_pert = to_u32(num_pert, "sparse edit count")?;
        self.scratch
            .ensure(&self.runtime.stream, history_slots.len(), seeds.len())?;
        self.scratch
            .ensure_edits(&self.runtime.stream, edits.len())?;
        copy_prefix(
            &self.scratch.history_slots,
            history_slots,
            &self.runtime.stream,
        )?;
        copy_prefix(&self.scratch.outcomes, outcomes, &self.runtime.stream)?;
        copy_prefix(
            &self.scratch.seeds,
            &pack_seeds(seeds),
            &self.runtime.stream,
        )?;
        if config.acquisition == 1 {
            copy_prefix(&self.scratch.draws, draws, &self.runtime.stream)?;
        }
        copy_prefix(&self.scratch.edits, edits, &self.runtime.stream)?;
        copy_prefix(&self.leaves, leaves, &self.runtime.stream)?;
        self.launch_sparse(base_slot, trial_slot, history, candidates, num_pert, config)?;
        let selection = read_prefix(&self.scratch.selection, &self.runtime.stream, 1)?[0];
        if selection.index >= candidates {
            return Err("CUDA sparse selection is out of bounds".to_string());
        }
        Ok((selection.index as usize, selection.score))
    }

    #[allow(clippy::too_many_arguments)]
    fn check_sparse(
        &self,
        base_slot: usize,
        history_slots: &[u32],
        outcomes: &[f32],
        trial_slot: usize,
        seeds: &[u64],
        draws: &[f32],
        edits: &[SparseEdit],
        num_pert: usize,
        leaves: &[Leaf],
        config: RoundConfig,
    ) -> CudaResult<()> {
        self.check_slot(base_slot)?;
        self.check_slot(trial_slot)?;
        if base_slot == trial_slot {
            return Err("CUDA sparse base and destination slots must differ".to_string());
        }
        if history_slots.is_empty() || history_slots.len() > MAX_HISTORY {
            return Err(format!(
                "CUDA sparse history must contain 1..={MAX_HISTORY} rows"
            ));
        }
        if history_slots.len() != outcomes.len() {
            return Err("CUDA sparse history slots and outcomes differ in length".to_string());
        }
        if seeds.is_empty() || num_pert == 0 {
            return Err("CUDA sparse candidates and num_pert are invalid".to_string());
        }
        if config.acquisition == 1 && draws.len() != history_slots.len() {
            return Err("CUDA sparse draws must match history length".to_string());
        }
        if edits.len() != seeds.len().saturating_mul(num_pert) {
            return Err("CUDA sparse edit count does not match candidates".to_string());
        }
        sparse_layout(self, history_slots, edits, leaves, config)
    }

    fn launch_sparse(
        &mut self,
        base_slot: usize,
        trial_slot: usize,
        history: u32,
        candidates: u32,
        num_pert: u32,
        config: RoundConfig,
    ) -> CudaResult<()> {
        let base_launch = self
            .runtime
            .module
            .prepare_base_distance(LaunchConfig1D::new(history, THREADS, 0))
            .map_err(cuda_error)?;
        let score_launch = self
            .runtime
            .module
            .prepare_score_sparse(LaunchConfig1D::new(candidates, THREADS, 0))
            .map_err(cuda_error)?;
        let pick_launch = self
            .runtime
            .module
            .prepare_pick_trial(LaunchConfig1D::new(1, THREADS, 0))
            .map_err(cuda_error)?;
        self.runtime
            .module
            .base_distance(
                &self.runtime.stream,
                &base_launch,
                &self.rows,
                &self.scratch.history_slots,
                &self.leaves,
                &mut self.scratch.base_distances,
                to_u32(self.row_stride, "row stride")?,
                history,
                to_u32(base_slot, "base slot")?,
            )
            .map_err(cuda_error)?;
        self.runtime
            .module
            .score_sparse(
                &self.runtime.stream,
                &score_launch,
                &self.rows,
                &self.scratch.history_slots,
                &self.scratch.outcomes,
                &self.scratch.seeds,
                &self.scratch.draws,
                &self.leaves,
                &self.scratch.edits,
                &self.scratch.base_distances,
                &mut self.scratch.scores,
                to_u32(self.row_stride, "row stride")?,
                history,
                candidates,
                num_pert,
                to_u32(base_slot, "base slot")?,
                to_u32(config.neighbors, "neighbors")?,
                config.acquisition,
                config.epistemic_scale,
                config.aleatoric_scale,
                config.y_scale,
                config.beta,
            )
            .map_err(cuda_error)?;
        self.runtime
            .module
            .pick_trial(
                &self.runtime.stream,
                &pick_launch,
                &self.scratch.scores,
                &mut self.scratch.selection,
                1,
                candidates,
            )
            .map_err(cuda_error)?;
        self.write_sparse(base_slot, trial_slot, num_pert)
    }

    fn write_sparse(
        &mut self,
        base_slot: usize,
        trial_slot: usize,
        num_pert: u32,
    ) -> CudaResult<()> {
        let source = self.rows.cu_deviceptr() + (base_slot * self.row_stride) as u64;
        let destination = self.rows.cu_deviceptr() + (trial_slot * self.row_stride) as u64;
        unsafe {
            cuda_core::simt::memory::memcpy_dtod_async(
                destination,
                source,
                self.row_bytes,
                self.runtime.stream.cu_stream(),
            )
            .map_err(cuda_error)?;
        }
        let write_launch = self
            .runtime
            .module
            .prepare_write_sparse(LaunchConfig1D::new(1, 1, 0))
            .map_err(cuda_error)?;
        self.runtime
            .module
            .write_sparse(
                &self.runtime.stream,
                &write_launch,
                &mut self.rows,
                &self.scratch.seeds,
                &self.scratch.selection,
                &self.leaves,
                &self.scratch.edits,
                to_u32(self.row_stride, "row stride")?,
                to_u32(trial_slot, "trial slot")?,
                num_pert,
            )
            .map_err(cuda_error)?;
        self.runtime.context.check_err().map_err(cuda_error)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn ask_multi(
        &mut self,
        base_slot: usize,
        history_slots: &[u32],
        outcomes: &[f32],
        regions: usize,
        candidates_per_region: usize,
        seeds: &[u64],
        draws: &[f32],
        centers: &[CenterStep],
        region_centers: &[u32],
        leaves: &[Leaf],
        config: RoundConfig,
    ) -> CudaResult<Vec<(usize, f32)>> {
        let (selections, _) = self.ask_impl(TrialAsk {
            base_slot,
            history_slots,
            outcomes,
            trial_slot: base_slot,
            seeds,
            draws,
            regions,
            candidates_per_region,
            centers,
            region_centers,
            leaves,
            config,
            materialize_row: false,
            read_scores: false,
        })?;
        Ok(selections
            .into_iter()
            .map(|selection| (selection.index as usize, selection.score))
            .collect())
    }

    fn ask_impl(&mut self, input: TrialAsk<'_>) -> CudaResult<(Vec<Selection>, Vec<f32>)> {
        self.last_profile = None;
        self.check_trial(&input)?;
        let (history, candidates) = self.upload_trial(&input)?;
        let score_launch = self
            .runtime
            .module
            .prepare_score_trials(LaunchConfig1D::new(candidates, THREADS, 0))
            .map_err(cuda_error)?;
        let pick_launch = self
            .runtime
            .module
            .prepare_pick_trial(LaunchConfig1D::new(
                to_u32(input.regions, "region count")?,
                THREADS,
                0,
            ))
            .map_err(cuda_error)?;
        let profile = self.profiling || std::env::var_os("BAUX_CUDA_PROFILE").is_some();
        let score_start = profile
            .then(|| timing_event(&self.runtime.stream))
            .transpose()?;
        self.runtime
            .module
            .score_trials(
                &self.runtime.stream,
                &score_launch,
                &self.rows,
                &self.scratch.history_slots,
                &self.scratch.outcomes,
                &self.scratch.seeds,
                &self.scratch.draws,
                &self.leaves,
                &self.scratch.centers,
                &self.scratch.candidate_centers,
                &mut self.scratch.scores,
                to_u32(self.row_stride, "row stride")?,
                history,
                candidates,
                to_u32(input.base_slot, "base slot")?,
                to_u32(input.centers.len(), "center count")?,
                to_u32(input.config.neighbors, "neighbors")?,
                input.config.acquisition,
                input.config.epistemic_scale,
                input.config.aleatoric_scale,
                input.config.y_scale,
                input.config.beta,
            )
            .map_err(cuda_error)?;
        let score_end = profile
            .then(|| timing_event(&self.runtime.stream))
            .transpose()?;
        self.runtime
            .module
            .pick_trial(
                &self.runtime.stream,
                &pick_launch,
                &self.scratch.scores,
                &mut self.scratch.selection,
                to_u32(input.regions, "region count")?,
                to_u32(input.candidates_per_region, "candidates per region")?,
            )
            .map_err(cuda_error)?;
        let pick_end = profile
            .then(|| timing_event(&self.runtime.stream))
            .transpose()?;
        if input.materialize_row {
            self.launch_write(input.base_slot, input.trial_slot)?;
        }
        let materialize_end = if profile && input.materialize_row {
            Some(timing_event(&self.runtime.stream)?)
        } else {
            None
        };
        self.runtime.context.check_err().map_err(cuda_error)?;
        self.collect_trial(&input, score_start, score_end, pick_end, materialize_end)
    }

    fn check_trial(&self, input: &TrialAsk<'_>) -> CudaResult<()> {
        self.check_slot(input.base_slot)?;
        self.check_slot(input.trial_slot)?;
        if input.materialize_row && input.base_slot == input.trial_slot {
            return Err("CUDA trial base and destination slots must differ".to_string());
        }
        if input.history_slots.is_empty() || input.history_slots.len() > MAX_HISTORY {
            return Err(format!(
                "CUDA trial history must contain 1..={MAX_HISTORY} rows"
            ));
        }
        if input.history_slots.len() != input.outcomes.len() {
            return Err("CUDA trial history slots and outcomes differ in length".to_string());
        }
        if input.seeds.is_empty() {
            return Err("CUDA trial seeds must be non-empty".to_string());
        }
        if input.config.acquisition == 1 && input.draws.len() != input.history_slots.len() {
            return Err("CUDA trial draws must match history length".to_string());
        }
        if input.regions == 0 || input.candidates_per_region == 0 {
            return Err("CUDA multi-region search requires non-zero dimensions".to_string());
        }
        trial_config(self, input)
    }

    fn upload_trial(&mut self, input: &TrialAsk<'_>) -> CudaResult<(u32, u32)> {
        let history = to_u32(input.history_slots.len(), "history rows")?;
        let candidates = to_u32(input.seeds.len(), "candidate count")?;
        self.scratch.ensure(
            &self.runtime.stream,
            input.history_slots.len(),
            input.seeds.len(),
        )?;
        self.scratch
            .ensure_centers(&self.runtime.stream, input.centers.len())?;
        copy_prefix(
            &self.scratch.history_slots,
            input.history_slots,
            &self.runtime.stream,
        )?;
        copy_prefix(&self.scratch.outcomes, input.outcomes, &self.runtime.stream)?;
        copy_prefix(
            &self.scratch.seeds,
            &pack_seeds(input.seeds),
            &self.runtime.stream,
        )?;
        if input.config.acquisition == 1 {
            copy_prefix(&self.scratch.draws, input.draws, &self.runtime.stream)?;
        }
        copy_prefix(&self.leaves, input.leaves, &self.runtime.stream)?;
        if !input.centers.is_empty() {
            let candidate_centers = input
                .region_centers
                .iter()
                .flat_map(|&center| std::iter::repeat_n(center, input.candidates_per_region))
                .collect::<Vec<_>>();
            copy_prefix(&self.scratch.centers, input.centers, &self.runtime.stream)?;
            copy_prefix(
                &self.scratch.candidate_centers,
                &candidate_centers,
                &self.runtime.stream,
            )?;
        }
        Ok((history, candidates))
    }

    fn collect_trial(
        &mut self,
        input: &TrialAsk<'_>,
        score_start: Option<cuda_core::CudaEvent>,
        score_end: Option<cuda_core::CudaEvent>,
        pick_end: Option<cuda_core::CudaEvent>,
        materialize_end: Option<cuda_core::CudaEvent>,
    ) -> CudaResult<(Vec<Selection>, Vec<f32>)> {
        let selections = read_prefix(&self.scratch.selection, &self.runtime.stream, input.regions)?;
        for (region, selection) in selections.iter().enumerate() {
            let first = region * input.candidates_per_region;
            let end = first + input.candidates_per_region;
            if !(first..end).contains(&(selection.index as usize)) {
                return Err(format!(
                    "CUDA region {region} selected invalid trial index {}",
                    selection.index
                ));
            }
        }
        let scores = if input.read_scores {
            read_prefix(
                &self.scratch.scores,
                &self.runtime.stream,
                input.seeds.len(),
            )?
        } else {
            Vec::new()
        };
        self.last_profile = match (score_start, score_end, pick_end) {
            (Some(score_start), Some(score_end), Some(pick_end)) => {
                let events = ProposalEvents {
                    score_start,
                    score_end,
                    pick_end,
                    materialize_end,
                };
                Some(events.profile()?)
            }
            _ => None,
        };
        Ok((selections, scores))
    }

    pub fn last_profile(&self) -> Option<ProposalProfile> {
        self.last_profile
    }

    pub fn materialize(
        &mut self,
        base_slot: usize,
        trial_slot: usize,
        seed: u64,
        leaves: &[Leaf],
    ) -> CudaResult<()> {
        self.check_slot(base_slot)?;
        self.check_slot(trial_slot)?;
        if base_slot == trial_slot {
            return Err("CUDA trial base and destination slots must differ".to_string());
        }
        if leaves.len() != self.leaves.len() {
            return Err("CUDA trial leaf layout changed after engine creation".to_string());
        }
        copy_prefix(
            &self.scratch.seeds,
            &pack_seeds(&[seed]),
            &self.runtime.stream,
        )?;
        copy_prefix(
            &self.scratch.selection,
            &[Selection {
                index: 0,
                score: 0.0,
            }],
            &self.runtime.stream,
        )?;
        copy_prefix(&self.leaves, leaves, &self.runtime.stream)?;
        // SAFETY: the validated base and trial slots are distinct row regions,
        // and write_trial gives each tile thread a unique output byte.
        self.launch_write(base_slot, trial_slot)?;
        self.runtime.stream.synchronize().map_err(cuda_error)
    }

    pub fn read(&self, slot: usize) -> CudaResult<Vec<u8>> {
        self.check_slot(slot)?;
        self.runtime.stream.synchronize().map_err(cuda_error)?;
        let mut row = vec![0_u8; self.row_bytes];
        let source = self.rows.cu_deviceptr() + (slot * self.row_stride) as u64;
        unsafe {
            cuda_core::simt::memory::memcpy_dtoh_async(
                row.as_mut_ptr(),
                source,
                self.row_bytes,
                self.runtime.stream.cu_stream(),
            )
            .map_err(cuda_error)?;
        }
        self.runtime.stream.synchronize().map_err(cuda_error)?;
        Ok(row)
    }

    /// Return a synchronized device pointer to a resident row.
    ///
    /// The pointer remains valid only while this engine is alive. Callers must
    /// retain the engine owner for the lifetime of every foreign GPU view.
    pub fn device_row(&self, slot: usize) -> CudaResult<(u64, usize, usize)> {
        Ok(self.device_rows(&[slot])?[0])
    }

    /// Return synchronized device pointers to several resident rows.
    pub fn device_rows(&self, slots: &[usize]) -> CudaResult<Vec<(u64, usize, usize)>> {
        for &slot in slots {
            self.check_slot(slot)?;
        }
        self.runtime.stream.synchronize().map_err(cuda_error)?;
        slots
            .iter()
            .map(|slot| {
                let offset = slot
                    .checked_mul(self.row_stride)
                    .ok_or("CUDA row pointer offset overflow")?;
                Ok((self.rows.cu_deviceptr() + offset as u64, self.row_bytes, 0))
            })
            .collect()
    }

    pub fn write(&mut self, slot: usize, row: &[u8]) -> CudaResult<()> {
        self.check_slot(slot)?;
        if row.len() != self.row_bytes {
            return Err(format!(
                "CUDA row has {} bytes, expected {}",
                row.len(),
                self.row_bytes
            ));
        }
        self.runtime.stream.synchronize().map_err(cuda_error)?;
        let destination = self.rows.cu_deviceptr() + (slot * self.row_stride) as u64;
        unsafe {
            cuda_core::simt::memory::memcpy_htod_sync(destination, row.as_ptr(), row.len())
                .map_err(cuda_error)
        }
    }

    fn launch_write(&mut self, base_slot: usize, trial_slot: usize) -> CudaResult<()> {
        let launch = self
            .runtime
            .module
            .prepare_write_trial(LaunchConfig1D::new(
                to_u32(self.tiles.len(), "tile count")?,
                THREADS,
                0,
            ))
            .map_err(cuda_error)?;
        self.runtime
            .module
            .write_trial(
                &self.runtime.stream,
                &launch,
                &mut self.rows,
                &self.scratch.seeds,
                &self.scratch.selection,
                &self.leaves,
                &self.tiles,
                to_u32(self.row_stride, "row stride")?,
                to_u32(base_slot, "base slot")?,
                to_u32(trial_slot, "trial slot")?,
            )
            .map_err(cuda_error)
    }

    fn check_slot(&self, slot: usize) -> CudaResult<()> {
        if slot >= self.slots {
            Err(format!(
                "CUDA row slot {slot} exceeds capacity {}",
                self.slots
            ))
        } else {
            Ok(())
        }
    }
}

fn pack_seeds(seeds: &[u64]) -> Vec<Seed> {
    seeds
        .iter()
        .map(|&seed| Seed {
            low: seed as u32,
            high: (seed >> 32) as u32,
        })
        .collect()
}

fn sparse_layout(
    engine: &TrialEngine,
    history_slots: &[u32],
    edits: &[SparseEdit],
    leaves: &[Leaf],
    config: RoundConfig,
) -> CudaResult<()> {
    if config.neighbors == 0 || config.neighbors > history_slots.len() {
        return Err("CUDA sparse neighbor count exceeds resident history".to_string());
    }
    if leaves.len() != engine.leaves.len() {
        return Err("CUDA sparse leaf layout changed after construction".to_string());
    }
    trial_leaves(engine.row_bytes, leaves)?;
    for &slot in history_slots {
        engine.check_slot(slot as usize)?;
    }
    for edit in edits {
        let leaf = leaves
            .get(edit.leaf as usize)
            .ok_or("CUDA sparse edit leaf is out of bounds")?;
        if edit.element >= leaf.length {
            return Err("CUDA sparse edit element is out of bounds".to_string());
        }
    }
    Ok(())
}

fn trial_config(engine: &TrialEngine, input: &TrialAsk<'_>) -> CudaResult<()> {
    let expected_candidates = input
        .regions
        .checked_mul(input.candidates_per_region)
        .ok_or("CUDA multi-region candidate count overflow")?;
    if input.seeds.len() != expected_candidates {
        return Err(format!(
            "CUDA expected {expected_candidates} candidates for {} regions, got {}",
            input.regions,
            input.seeds.len()
        ));
    }
    validate_centers(input.centers, input.region_centers, input.regions)?;
    if input.materialize_row && (input.regions != 1 || !input.centers.is_empty()) {
        return Err("CUDA can materialize only a single root-based trial".to_string());
    }
    if input.config.neighbors == 0 || input.config.neighbors > input.history_slots.len() {
        return Err("CUDA trial neighbor count exceeds resident history".to_string());
    }
    if input.leaves.len() != engine.leaves.len() {
        return Err("CUDA trial leaf layout changed after engine creation".to_string());
    }
    trial_leaves(engine.row_bytes, input.leaves)?;
    trial_values(engine, input)
}

fn trial_values(engine: &TrialEngine, input: &TrialAsk<'_>) -> CudaResult<()> {
    if input.outcomes.iter().any(|value| !value.is_finite())
        || input.draws.iter().any(|value| !value.is_finite())
    {
        return Err("CUDA trial outcomes and draws must be finite".to_string());
    }
    if input.config.acquisition > 2
        || !input.config.epistemic_scale.is_finite()
        || !input.config.aleatoric_scale.is_finite()
        || !input.config.y_scale.is_finite()
        || !input.config.beta.is_finite()
    {
        return Err("CUDA trial acquisition configuration is invalid".to_string());
    }
    for &slot in input.history_slots {
        engine.check_slot(slot as usize)?;
    }
    Ok(())
}
