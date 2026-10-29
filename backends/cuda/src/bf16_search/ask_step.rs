use std::collections::BTreeSet;
use std::mem::size_of;

use cuda_core::{DeviceBuffer, LaunchConfig1D};

use super::*;

impl Bf16SearchEngine {
    #[allow(clippy::too_many_arguments)]
    pub fn ask(
        &mut self,
        base_slot: usize,
        history: usize,
        trial_slots: &[u32],
        seeds: &[u64],
        candidates_per_region: usize,
        coefficient: f32,
        draw_seed: u64,
        config: RoundConfig,
    ) -> CudaResult<Vec<Selection>> {
        self.last_profile = None;
        let input = AskInput {
            base_slot,
            history,
            trial_slots,
            seeds,
            seed_root: None,
            candidates_per_region,
            coefficient,
            draw_seed,
            config,
        };
        let shape = self.check_ask(&input)?;
        self.upload(&input, shape)?;
        let events = self.launch(&input, shape)?;
        let selections = self.collect(&input, shape)?;
        if let Some(events) = events {
            let profile = events.profile()?;
            self.last_profile = Some(profile);
        }
        Ok(selections)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn ask_seeded(
        &mut self,
        base_slot: usize,
        history: usize,
        trial_slots: &[u32],
        candidates_per_region: usize,
        seed_root: u64,
        coefficient: f32,
        draw_seed: u64,
        config: RoundConfig,
    ) -> CudaResult<()> {
        self.last_profile = None;
        let input = AskInput {
            base_slot,
            history,
            trial_slots,
            seeds: &[],
            seed_root: Some(seed_root),
            candidates_per_region,
            coefficient,
            draw_seed,
            config,
        };
        let shape = self.check_ask(&input)?;
        self.upload(&input, shape)?;
        let events = self.launch(&input, shape)?;
        if let Some(events) = events {
            let profile = events.profile()?;
            self.last_profile = Some(profile);
        }
        Ok(())
    }

    fn check_ask(&self, input: &AskInput<'_>) -> CudaResult<SearchShape> {
        if self.paired_relative.is_some() {
            validate_layout(
                self.search_capacity,
                self.slots,
                input.base_slot,
                input.history,
                input.trial_slots,
            )?;
            if self.single_slot.is_some() {
                return Err("Tell the pending paired-relative proposal before another ask".into());
            }
        }
        if self.sampler == 1 && self.reference_seed.is_none() {
            self.check_reference()?;
        }
        if self.sampler == 1
            && (input.seed_root.is_none()
                || input.trial_slots.len() != 1
                || input.candidates_per_region != 4)
        {
            return Err(
                "Correlated BF16 search requires one seeded arm with four candidates".into(),
            );
        }
        self.check_slot(input.base_slot)?;
        let history = input.history;
        let regions = input.trial_slots.len();
        if history == 0 || history > MAX_HISTORY {
            return Err(format!(
                "CUDA BF16 history must contain 1..={MAX_HISTORY} rows"
            ));
        }
        if regions == 0 || input.candidates_per_region == 0 {
            return Err("CUDA BF16 search requires regions and candidates".to_string());
        }
        let candidates = regions
            .checked_mul(input.candidates_per_region)
            .ok_or("CUDA BF16 candidate count overflow")?;
        if input.seed_root.is_none() && input.seeds.len() != candidates {
            return Err("CUDA BF16 seeds do not match the search shape".to_string());
        }
        if input.config.neighbors == 0 || input.config.neighbors > history {
            return Err("CUDA BF16 neighbor count exceeds resident history".to_string());
        }
        if !input.coefficient.is_finite() || input.coefficient <= 0.0 {
            return Err("CUDA BF16 perturbation coefficient must be positive".to_string());
        }
        validate_ask(input)?;
        let mut destinations = BTreeSet::new();
        for &slot in input.trial_slots {
            self.check_slot(slot as usize)?;
            if slot as usize == input.base_slot || !destinations.insert(slot) {
                return Err("CUDA BF16 trial slots must be distinct from live rows".to_string());
            }
        }
        let status = regions
            .checked_mul(self.tile_count)
            .ok_or("CUDA BF16 status count overflow")?;
        let blocks = candidates
            .checked_mul(self.tile_count)
            .ok_or("CUDA BF16 distance block count overflow")?;
        let partials = blocks
            .checked_mul(history)
            .ok_or("CUDA BF16 distance partial count overflow")?;
        Ok(SearchShape {
            regions,
            candidates,
            blocks,
            partials,
            status,
        })
    }

    fn upload(&mut self, input: &AskInput<'_>, shape: SearchShape) -> CudaResult<()> {
        self.started = true;
        self.ensure_reference()?;
        self.scratch.ensure(
            &self.runtime.stream,
            input.history,
            shape.candidates,
            shape.regions,
            shape.partials,
            shape.status,
        )?;
        let batch_len = if shape.regions == 1 {
            1
        } else {
            shape
                .regions
                .checked_mul(self.row_len)
                .ok_or("CUDA BF16 batch size overflow")?
        };
        if batch_len > self.batch.len() {
            self.batch =
                DeviceBuffer::zeroed(&self.runtime.stream, batch_len).map_err(cuda_error)?;
        }
        if let Some(root) = input.seed_root {
            let launch = self
                .runtime
                .module
                .prepare_seed_bf16(LaunchConfig1D::new(
                    to_u32(
                        shape.candidates.div_ceil(THREADS as usize),
                        "BF16 seed blocks",
                    )?,
                    THREADS,
                    0,
                ))
                .map_err(cuda_error)?;
            self.runtime
                .module
                .seed_bf16(
                    &self.runtime.stream,
                    &launch,
                    &mut self.scratch.seeds,
                    root,
                    to_u32(shape.candidates, "BF16 seed count")?,
                    self.sampler,
                )
                .map_err(cuda_error)?;
        } else {
            let seeds = input
                .seeds
                .iter()
                .map(|seed| Seed {
                    low: *seed as u32,
                    high: (*seed >> 32) as u32,
                })
                .collect::<Vec<_>>();
            copy_prefix(&self.scratch.seeds, &seeds, &self.runtime.stream)?;
        }
        copy_prefix(
            &self.scratch.trial_slots,
            input.trial_slots,
            &self.runtime.stream,
        )?;
        self.clear_status(shape.status)
    }

    fn launch(
        &mut self,
        input: &AskInput<'_>,
        shape: SearchShape,
    ) -> CudaResult<Option<ProposalEvents>> {
        let candidates = to_u32(shape.candidates, "BF16 candidate count")?;
        let regions = to_u32(shape.regions, "BF16 region count")?;
        let score_launch = self
            .runtime
            .module
            .prepare_score_bf16(LaunchConfig1D::new(candidates, THREADS, 0))
            .map_err(cuda_error)?;
        let draw_launch = self
            .runtime
            .module
            .prepare_draw_bf16(LaunchConfig1D::new(
                to_u32(input.history.div_ceil(THREADS as usize), "BF16 draw blocks")?,
                THREADS,
                0,
            ))
            .map_err(cuda_error)?;
        let pick_launch = self
            .runtime
            .module
            .prepare_pick_trial(LaunchConfig1D::new(regions, THREADS, 0))
            .map_err(cuda_error)?;
        let write_launch = self
            .runtime
            .module
            .prepare_write_bf16(LaunchConfig1D::new(
                to_u32(shape.status, "BF16 write blocks")?,
                THREADS,
                0,
            ))
            .map_err(cuda_error)?;
        let profile = self.profiling || std::env::var_os("BAUX_CUDA_PROFILE").is_some();
        let score_start = profile
            .then(|| timing_event(&self.runtime.stream))
            .transpose()?;
        let params = Bf16Score {
            row_stride: self.row_stride as u64,
            coefficient: input.coefficient,
            epistemic_scale: input.config.epistemic_scale,
            aleatoric_scale: input.config.aleatoric_scale,
            y_scale: input.config.y_scale,
            beta: input.config.beta,
            history: to_u32(input.history, "BF16 history rows")?,
            candidates,
            base_slot: to_u32(input.base_slot, "BF16 base slot")?,
            neighbors: to_u32(input.config.neighbors, "BF16 neighbors")?,
            acquisition: input.config.acquisition,
            tiles: to_u32(self.tile_count, "BF16 distance tile count")?,
            resident: u32::from(input.seed_root.is_some()),
            correlated: self.sampler,
            adaptive_neighbors: u32::from(input.config.adaptive_neighbors),
        };
        launch_distance(self, shape, params)?;
        self.runtime
            .module
            .draw_bf16(
                &self.runtime.stream,
                &draw_launch,
                &mut self.scratch.draws,
                &self.scratch.history_slots,
                &self.state,
                input.draw_seed,
                params.history,
                params.resident,
            )
            .map_err(cuda_error)?;
        self.runtime
            .module
            .score_bf16(
                &self.runtime.stream,
                &score_launch,
                &self.scratch.partials,
                &self.scratch.tile_status,
                &self.scratch.outcomes,
                &self.scratch.variances,
                &self.state,
                &self.scratch.draws,
                &mut self.scratch.scores,
                params,
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
                regions,
                to_u32(input.candidates_per_region, "BF16 candidates per region")?,
            )
            .map_err(cuda_error)?;
        let pick_end = profile
            .then(|| timing_event(&self.runtime.stream))
            .transpose()?;
        self.runtime
            .module
            .write_bf16(
                &self.runtime.stream,
                &write_launch,
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
                to_u32(input.base_slot, "BF16 base slot")?,
                to_u32(self.tile_count, "BF16 tile count")?,
                input.coefficient,
                u32::from(input.seed_root.is_some()),
                u32::from(shape.regions > 1),
                0,
                self.sampler,
            )
            .map_err(cuda_error)?;
        let materialize_end = profile
            .then(|| timing_event(&self.runtime.stream))
            .transpose()?;
        self.runtime.context.check_err().map_err(cuda_error)?;
        self.single_slot = (shape.regions == 1).then_some(SingleRow {
            slot: input.trial_slots[0] as usize,
            base_slot: input.base_slot,
            coefficient: input.coefficient,
            resident: input.seed_root.is_some(),
        });
        Ok(match (score_start, score_end, pick_end, materialize_end) {
            (Some(score_start), Some(score_end), Some(pick_end), Some(materialize_end)) => {
                Some(ProposalEvents {
                    score_start,
                    score_end,
                    pick_end,
                    materialize_end: Some(materialize_end),
                })
            }
            _ => None,
        })
    }

    fn collect(&self, input: &AskInput<'_>, shape: SearchShape) -> CudaResult<Vec<Selection>> {
        let selections = read_prefix(&self.scratch.selection, &self.runtime.stream, shape.regions)?;
        for (region, selection) in selections.iter().enumerate() {
            let first = region * input.candidates_per_region;
            let end = first + input.candidates_per_region;
            if !(first..end).contains(&(selection.index as usize)) {
                self.reset_trials(input.base_slot, input.trial_slots)?;
                return Err(format!(
                    "CUDA BF16 region {region} selected invalid trial index {}",
                    selection.index
                ));
            }
        }
        let status = read_prefix(&self.scratch.status, &self.runtime.stream, shape.status)?;
        if status.contains(&1) {
            self.reset_trials(input.base_slot, input.trial_slots)?;
            return Err("CUDA BF16 search perturbation overflowed FP32".to_string());
        }
        Ok(selections)
    }

    pub fn copy_row(&self, source: usize, destination: usize) -> CudaResult<()> {
        self.check_slot(source)?;
        self.check_slot(destination)?;
        if source == destination {
            return Ok(());
        }
        unsafe {
            cuda_core::simt::memory::memcpy_dtod_async(
                self.row_pointer(destination)?,
                self.row_pointer(source)?,
                row_bytes(self.row_len)?,
                self.runtime.stream.cu_stream(),
            )
            .map_err(cuda_error)
        }
    }

    /// Reuse an unused pending row as a disposable incumbent snapshot.
    pub fn snapshot_incumbent(&self, destination: usize) -> CudaResult<()> {
        if destination == 0 {
            return Err("The incumbent snapshot must not alias canonical row zero".into());
        }
        // A released consumer may have used a stream other than our producer
        // stream.
        self.runtime.context.synchronize().map_err(cuda_error)?;
        self.copy_row(0, destination)
    }

    pub fn sync_consumers(&self) -> CudaResult<()> {
        self.runtime.context.synchronize().map_err(cuda_error)
    }

    pub fn read(&self, slot: usize) -> CudaResult<Vec<u16>> {
        self.check_slot(slot)?;
        let mut output = Vec::<u16>::with_capacity(self.row_len);
        unsafe {
            cuda_core::simt::memory::memcpy_dtoh_async(
                output.as_mut_ptr(),
                self.row_pointer(slot)?,
                row_bytes(self.row_len)?,
                self.runtime.stream.cu_stream(),
            )
            .map_err(cuda_error)?;
        }
        self.runtime.stream.synchronize().map_err(cuda_error)?;
        unsafe {
            output.set_len(self.row_len);
        }
        Ok(output)
    }

    pub fn device_row(&self, slot: usize, stream: Option<i64>) -> CudaResult<(u64, usize, usize)> {
        self.check_slot(slot)?;
        sync_stream(&self.runtime, stream)?;
        Ok((self.row_pointer(slot)?, row_bytes(self.row_len)?, 0))
    }

    /// Gather resident trial slots into one contiguous device matrix.
    pub fn device_batch(
        &mut self,
        slots: &[u32],
        stream: Option<i64>,
    ) -> CudaResult<(u64, usize, usize)> {
        if slots.is_empty() || slots.len() > BF16_PENDING {
            return Err("CUDA BF16 batch shape is invalid".to_string());
        }
        if slots.len() == 1 {
            let (pointer, _, _) = self.device_row(slots[0] as usize, stream)?;
            return Ok((pointer, 1, self.row_len));
        }
        for &slot in slots {
            self.check_slot(slot as usize)?;
        }
        let elements = slots
            .len()
            .checked_mul(self.row_len)
            .ok_or("CUDA BF16 batch size overflow")?;
        if elements > self.batch.len() {
            self.batch =
                DeviceBuffer::zeroed(&self.runtime.stream, elements).map_err(cuda_error)?;
        }
        let bytes = row_bytes(self.row_len)?;
        for (row, &slot) in slots.iter().enumerate() {
            let offset = row
                .checked_mul(bytes)
                .ok_or("CUDA BF16 batch offset overflow")?;
            let destination = self
                .batch
                .cu_deviceptr()
                .checked_add(offset as u64)
                .ok_or("CUDA BF16 batch pointer overflow")?;
            unsafe {
                cuda_core::simt::memory::memcpy_dtod_async(
                    destination,
                    self.row_pointer(slot as usize)?,
                    bytes,
                    self.runtime.stream.cu_stream(),
                )
                .map_err(cuda_error)?;
            }
        }
        sync_stream(&self.runtime, stream)?;
        Ok((self.batch.cu_deviceptr(), slots.len(), self.row_len))
    }

    pub fn device_round(
        &self,
        rows: usize,
        stream: Option<i64>,
    ) -> CudaResult<(u64, usize, usize)> {
        if rows == 1 {
            if let Some(single) = self.single_slot {
                let (pointer, _, _) = self.device_row(single.slot, stream)?;
                return Ok((pointer, 1, self.row_len));
            }
        }
        let elements = rows
            .checked_mul(self.row_len)
            .ok_or("CUDA BF16 round size overflow")?;
        if rows == 0 || elements > self.batch.len() {
            return Err("CUDA BF16 round shape is invalid".to_string());
        }
        sync_stream(&self.runtime, stream)?;
        Ok((self.batch.cu_deviceptr(), rows, self.row_len))
    }

    pub fn last_profile(&self) -> Option<ProposalProfile> {
        self.last_profile
    }

    pub fn describe(&self, arms: usize) -> CudaResult<Vec<ProposalDescription>> {
        if arms == 0 || arms > self.scratch.region_capacity {
            return Err("Invalid BF16 proposal description count".into());
        }
        let count = arms
            .checked_mul(self.tile_count)
            .ok_or("BF16 diagnostics overflow")?;
        let status = read_prefix(&self.scratch.status, &self.runtime.stream, count)?;
        if status.iter().any(|&value| value != 0) {
            return Err("Cannot describe an invalid BF16 proposal".into());
        }
        let choices = read_prefix(&self.scratch.selection, &self.runtime.stream, arms)?;
        let seeds = read_prefix(
            &self.scratch.seeds,
            &self.runtime.stream,
            self.scratch.candidate_capacity,
        )?;
        let changes = read_prefix(&self.scratch.changes, &self.runtime.stream, count)?;
        let state = read_prefix(&self.state, &self.runtime.stream, 1)?;
        let mut descriptions = Vec::with_capacity(arms);
        for (arm, choice) in choices.iter().enumerate() {
            if !choice.score.is_finite() {
                return Err("BF16 acquisition score is nonfinite".into());
            }
            let seed = seeds
                .get(choice.index as usize)
                .ok_or("Invalid BF16 selected seed")?;
            let mut blocks = vec![(0_u64, 0.0_f64); self.leaf_count];
            for (tile, &leaf) in self.tile_leaves.iter().enumerate() {
                let change = changes[arm * self.tile_count + tile];
                if !change.squared.is_finite() || change.squared < 0.0 {
                    return Err("Nonfinite BF16 perturbation diagnostics".into());
                }
                blocks[leaf].0 += u64::from(change.changed);
                blocks[leaf].1 += f64::from(change.squared);
            }
            descriptions.push((
                u64::from(seed.low) | (u64::from(seed.high) << 32),
                choice.score,
                if self.sampler == 1 {
                    let factor = if choice.index & 1 == 0 { 0.5 } else { 2.0 };
                    (state[0].length * factor)
                        .max(state[0].length_min)
                        .min(state[0].length_max) as f32
                } else {
                    state[0].length as f32
                },
                blocks,
            ));
        }
        Ok(descriptions)
    }

    pub fn geometry(&self, arms: usize) -> CudaResult<Vec<(usize, f32)>> {
        if arms == 0 || arms > self.scratch.region_capacity {
            return Err("Invalid BF16 proposal geometry count".into());
        }
        let choices = read_prefix(&self.scratch.selection, &self.runtime.stream, arms)?;
        choices
            .into_iter()
            .map(|choice| {
                if !choice.score.is_finite()
                    || choice.index as usize >= self.scratch.candidate_capacity
                {
                    return Err("Invalid BF16 proposal geometry".into());
                }
                let persistence = if self.sampler == 1 && choice.index < 2 {
                    0.75
                } else {
                    0.0
                };
                Ok((choice.index as usize, persistence))
            })
            .collect()
    }

    pub fn len(&self) -> usize {
        self.row_len
    }

    pub fn is_empty(&self) -> bool {
        self.row_len == 0
    }

    pub(super) fn validate(&mut self, slot: usize) -> CudaResult<()> {
        self.check_slot(slot)?;
        self.scratch
            .ensure(&self.runtime.stream, 1, 1, 1, 1, self.tile_count)?;
        self.clear_status(self.tile_count)?;
        let launch = self
            .runtime
            .module
            .prepare_check_search(LaunchConfig1D::new(
                to_u32(self.tile_count, "BF16 tile count")?,
                THREADS,
                0,
            ))
            .map_err(cuda_error)?;
        self.runtime
            .module
            .check_search(
                &self.runtime.stream,
                &launch,
                &self.rows,
                &self.leaves,
                &self.tiles,
                &mut self.scratch.status,
                self.row_stride as u64,
                to_u32(slot, "BF16 row slot")?,
            )
            .map_err(cuda_error)?;
        self.runtime.context.check_err().map_err(cuda_error)?;
        let status = read_prefix(&self.scratch.status, &self.runtime.stream, self.tile_count)?;
        if status.contains(&1) {
            Err("CUDA BF16 search base values must be finite".to_string())
        } else {
            Ok(())
        }
    }

    fn clear_status(&self, count: usize) -> CudaResult<()> {
        unsafe {
            cuda_core::simt::memory::memset_d8_async(
                self.scratch.status.cu_deviceptr(),
                0,
                count
                    .checked_mul(size_of::<u32>())
                    .ok_or("CUDA BF16 status byte count overflow")?,
                self.runtime.stream.cu_stream(),
            )
            .map_err(cuda_error)
        }
    }

    fn reset_trials(&self, base_slot: usize, slots: &[u32]) -> CudaResult<()> {
        for &slot in slots {
            self.copy_row(base_slot, slot as usize)?;
        }
        self.runtime.stream.synchronize().map_err(cuda_error)
    }

    fn row_pointer(&self, slot: usize) -> CudaResult<u64> {
        let offset = slot
            .checked_mul(self.row_stride)
            .and_then(|value| value.checked_mul(size_of::<u16>()))
            .ok_or("CUDA BF16 row pointer overflow")?;
        Ok(self.rows.cu_deviceptr() + offset as u64)
    }

    pub(super) fn check_slot(&self, slot: usize) -> CudaResult<()> {
        if slot >= self.slots {
            Err(format!(
                "CUDA BF16 row slot {slot} exceeds capacity {}",
                self.slots
            ))
        } else {
            Ok(())
        }
    }
}

fn launch_distance(
    engine: &mut Bf16SearchEngine,
    shape: SearchShape,
    params: Bf16Score,
) -> CudaResult<()> {
    if engine.fused_pairs && shape.candidates == 4 {
        let pair_blocks = shape
            .candidates
            .div_ceil(2)
            .checked_mul(engine.tile_count)
            .ok_or("CUDA BF16 fused distance block count overflow")?;
        let launch = engine
            .runtime
            .module
            .prepare_distance2_bf16(LaunchConfig1D::new(
                to_u32(pair_blocks, "BF16 fused distance blocks")?,
                THREADS,
                0,
            ))
            .map_err(cuda_error)?;
        engine
            .runtime
            .module
            .distance2_bf16(
                &engine.runtime.stream,
                &launch,
                &engine.rows,
                &engine.scratch.history_slots,
                &engine.reference,
                &engine.reference_scales,
                &engine.state,
                &engine.scratch.seeds,
                &engine.leaves,
                &engine.tiles,
                &mut engine.scratch.partials,
                &mut engine.scratch.tile_status,
                params,
            )
            .map_err(cuda_error)
    } else {
        let launch = engine
            .runtime
            .module
            .prepare_distance_bf16(LaunchConfig1D::new(
                to_u32(shape.blocks, "BF16 distance blocks")?,
                THREADS,
                0,
            ))
            .map_err(cuda_error)?;
        engine
            .runtime
            .module
            .distance_bf16(
                &engine.runtime.stream,
                &launch,
                &engine.rows,
                &engine.scratch.history_slots,
                &engine.reference,
                &engine.reference_scales,
                &engine.state,
                &engine.scratch.seeds,
                &engine.leaves,
                &engine.tiles,
                &mut engine.scratch.partials,
                &mut engine.scratch.tile_status,
                params,
            )
            .map_err(cuda_error)
    }
}

pub(super) fn validate_layout(
    configured_capacity: Option<usize>,
    slots: usize,
    base_slot: usize,
    capacity: usize,
    trial_slots: &[u32],
) -> CudaResult<()> {
    if configured_capacity != Some(capacity)
        || capacity < 2
        || base_slot != 0
        || trial_slots.len() != 1
        || trial_slots[0] as usize <= capacity
        || trial_slots[0] as usize >= slots
    {
        return Err(
            "Paired-relative search requires base slot 0, the initialized history capacity, and \
             one trial in pending storage beyond history"
                .into(),
        );
    }
    Ok(())
}

pub(super) fn validate_tell(
    configured_capacity: Option<usize>,
    slots: usize,
    capacity: usize,
    trial_slot: u32,
    pending: Option<SingleRow>,
) -> CudaResult<()> {
    validate_layout(configured_capacity, slots, 0, capacity, &[trial_slot])?;
    if !pending.is_some_and(|pending| {
        pending.slot == trial_slot as usize && pending.base_slot == 0 && pending.resident
    }) {
        return Err("Paired-relative tell must match the outstanding selected trial slot".into());
    }
    Ok(())
}

#[cfg(test)]
mod relative_validation_tests {
    use super::{SingleRow, validate_layout, validate_tell};

    #[test]
    fn layout_guards() {
        for capacity in [2, 3, 128] {
            let slots = capacity + 3;
            let pending = (capacity + 1) as u32;
            assert!(validate_layout(Some(capacity), slots, 0, capacity, &[pending]).is_ok());
            for slot in 0..=capacity {
                assert!(
                    validate_layout(Some(capacity), slots, 0, capacity, &[slot as u32]).is_err()
                );
            }
            for invalid in [slots as u32, u32::MAX] {
                assert!(validate_layout(Some(capacity), slots, 0, capacity, &[invalid]).is_err());
            }
            for history in [0, 1, capacity - 1, capacity + 1, 256] {
                assert!(validate_layout(Some(capacity), slots, 0, history, &[pending]).is_err());
            }
            assert!(validate_layout(None, slots, 0, capacity, &[pending]).is_err());
            assert!(validate_layout(Some(capacity), slots, 1, capacity, &[pending]).is_err());
            assert!(validate_layout(Some(capacity), slots, 0, capacity, &[]).is_err());
            assert!(
                validate_layout(Some(capacity), slots, 0, capacity, &[pending, pending + 1])
                    .is_err()
            );
        }
    }

    #[test]
    fn tell_selection() {
        let pending = SingleRow {
            slot: 3,
            base_slot: 0,
            coefficient: 1.0,
            resident: true,
        };
        assert!(validate_tell(Some(2), 5, 2, 3, Some(pending)).is_ok());
        assert!(validate_tell(Some(2), 5, 2, 3, None).is_err());
        assert!(validate_tell(Some(2), 5, 2, 4, Some(pending)).is_err());
        assert!(validate_tell(Some(2), 5, 3, 4, Some(pending)).is_err());
        for invalid in [
            SingleRow {
                base_slot: 1,
                ..pending
            },
            SingleRow {
                resident: false,
                ..pending
            },
        ] {
            assert!(validate_tell(Some(2), 5, 2, 3, Some(invalid)).is_err());
        }
        for live in 0..=2 {
            assert!(
                validate_tell(
                    Some(2),
                    5,
                    2,
                    live,
                    Some(SingleRow {
                        slot: live as usize,
                        ..pending
                    })
                )
                .is_err()
            );
        }
    }
}

fn validate_ask(input: &AskInput<'_>) -> CudaResult<()> {
    let config = input.config;
    if config.acquisition > 2
        || !config.epistemic_scale.is_finite()
        || config.epistemic_scale < 0.0
        || !config.aleatoric_scale.is_finite()
        || config.aleatoric_scale < 0.0
        || !config.y_scale.is_finite()
        || config.y_scale < 0.0
        || !config.beta.is_finite()
    {
        return Err("CUDA BF16 acquisition configuration is invalid".to_string());
    }
    Ok(())
}

pub(super) fn bf16_tiles(leaves: &[Bf16Leaf]) -> CudaResult<Vec<DenseTile>> {
    let total = leaves.iter().try_fold(0usize, |total, leaf| {
        let length = usize::try_from(leaf.length)
            .map_err(|_| "CUDA BF16 leaf length exceeds usize".to_string())?;
        total
            .checked_add(length)
            .ok_or("CUDA BF16 tile length overflow".to_string())
    })?;
    let tile_elements = if total <= DENSE_ELEMENTS {
        BF16_SMALL_TILE_ELEMENTS
    } else {
        BF16_TILE_ELEMENTS
    };
    let mut tiles = Vec::new();
    for (leaf_index, leaf) in leaves.iter().enumerate() {
        let leaf_index = to_u32(leaf_index, "BF16 leaf count")?;
        let length = usize::try_from(leaf.length)
            .map_err(|_| "CUDA BF16 leaf length exceeds usize".to_string())?;
        let mut start = 0usize;
        while start < length {
            let tile_length = (length - start).min(tile_elements);
            tiles.push(DenseTile {
                leaf: leaf_index,
                start: to_u32(start, "BF16 leaf offset")?,
                length: to_u32(tile_length, "BF16 tile length")?,
                pad: 0,
            });
            start += tile_length;
        }
    }
    Ok(tiles)
}
