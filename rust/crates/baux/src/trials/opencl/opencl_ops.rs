use opencl3::kernel::ExecuteKernel;
use opencl3::memory::{Buffer, CL_MEM_READ_ONLY, CL_MEM_READ_WRITE};
use opencl3::types::CL_BLOCKING;

use super::*;

impl Engine {
    #[allow(clippy::too_many_arguments)]
    pub(in crate::trials) fn ask_sparse(
        &mut self,
        base_slot: usize,
        history: &[(usize, f32)],
        trial_slot: usize,
        seeds: &[u64],
        edits: &[SparseEdit],
        num_pert: usize,
        leaves: &[Parameter],
        config: RoundConfig,
    ) -> Result<(usize, f32), String> {
        let history_count = if self.state.is_some() {
            self.capacity
        } else {
            history.len()
        };
        if self.state.is_none() && base_slot == trial_slot {
            return Err("OpenCL sparse base and destination slots must differ".to_string());
        }
        if history_count == 0 || history_count > super::super::MAX_HISTORY {
            return Err(format!(
                "OpenCL sparse history must contain 1..={} rows",
                super::super::MAX_HISTORY
            ));
        }
        if seeds.is_empty() {
            return Err("OpenCL sparse trials require at least one candidate".to_string());
        }
        if num_pert == 0 {
            return Err("OpenCL sparse edit count must be positive".to_string());
        }
        if self.state.is_none() && edits.len() != seeds.len().saturating_mul(num_pert) {
            return Err("OpenCL sparse edit count does not match candidates".to_string());
        }
        if config.surrogate.neighbors == 0 || config.surrogate.neighbors > history_count {
            return Err("OpenCL sparse neighbor count exceeds resident history".to_string());
        }
        self.ensure_candidates(seeds.len())?;
        self.ensure_edits(seeds.len().saturating_mul(num_pert))?;
        let history_slots: Vec<u32> = history
            .iter()
            .map(|&(slot, _)| to_u32(slot, "history slot"))
            .collect::<Result<_, _>>()?;
        let outcomes: Vec<f32> = history.iter().map(|&(_, value)| value).collect();
        let packed_seeds: Vec<Seed> = seeds
            .iter()
            .map(|&seed| Seed {
                low: seed as u32,
                high: (seed >> 32) as u32,
            })
            .collect();
        let steps = if self.region.is_some() {
            Vec::new()
        } else {
            make_steps(leaves, config.proposal.radius)
        };
        self.write_inputs(&history_slots, &outcomes, &packed_seeds, &steps)?;
        self.fill_draws(config)?;
        if self.state.is_none() {
            unsafe {
                self.queue
                    .enqueue_write_buffer(&mut self.scratch.edits, CL_BLOCKING, 0, edits, &[])
                    .map_err(|error| format!("failed to write OpenCL sparse edits: {error}"))?;
            }
        }
        let params = self.sparse_params(
            base_slot,
            trial_slot,
            history_count,
            seeds.len(),
            num_pert,
            leaves,
            config,
        )?;
        if self.state.is_some() {
            unsafe {
                ExecuteKernel::new(&self.fill_edits)
                    .set_arg(&self.scratch.edits)
                    .set_arg(&self.scratch.seeds)
                    .set_arg(&self.scratch.leaves)
                    .set_arg(params.as_ref())
                    .set_global_work_size(seeds.len())
                    .set_local_work_size(1)
                    .enqueue_nd_range(&self.queue)
                    .map_err(|error| error.to_string())?;
            }
        }
        self.run_sparse(base_slot, trial_slot, history_count, seeds.len(), &params)
    }

    #[allow(clippy::too_many_arguments)]
    pub(in crate::trials) fn sparse_stream(
        &mut self,
        base_slot: usize,
        history: &[(usize, f32)],
        trial_slot: usize,
        base_seed: u64,
        count: usize,
        num_pert: usize,
        leaves: &[Parameter],
        config: RoundConfig,
    ) -> Result<(usize, u64, f32), String> {
        let history_count = if self.state.is_some() {
            self.capacity
        } else {
            history.len()
        };
        if self.state.is_none() && base_slot == trial_slot {
            return Err("OpenCL sparse base and destination slots must differ".to_string());
        }
        if history_count == 0 || history_count > super::super::MAX_HISTORY {
            return Err(format!(
                "OpenCL sparse history must contain 1..={} rows",
                super::super::MAX_HISTORY
            ));
        }
        if config.surrogate.neighbors == 0 || config.surrogate.neighbors > history_count {
            return Err("OpenCL sparse neighbor count exceeds resident history".to_string());
        }
        self.ensure_candidates(count)?;
        self.ensure_edits(count.saturating_mul(num_pert))?;
        let history_slots: Vec<u32> = history
            .iter()
            .map(|&(slot, _)| to_u32(slot, "history slot"))
            .collect::<Result<_, _>>()?;
        let outcomes: Vec<f32> = history.iter().map(|&(_, value)| value).collect();
        let steps = if self.region.is_some() {
            Vec::new()
        } else {
            make_steps(leaves, config.proposal.radius)
        };
        self.write_static(&history_slots, &outcomes, &steps)?;
        self.fill_seeds(base_seed, count)?;
        self.fill_draws(config)?;
        let params = self.sparse_params(
            base_slot,
            trial_slot,
            history_count,
            count,
            num_pert,
            leaves,
            config,
        )?;
        unsafe {
            ExecuteKernel::new(&self.fill_edits)
                .set_arg(&self.scratch.edits)
                .set_arg(&self.scratch.seeds)
                .set_arg(&self.scratch.leaves)
                .set_arg(params.as_ref())
                .set_global_work_size(count)
                .set_local_work_size(1)
                .enqueue_nd_range(&self.queue)
                .map_err(|error| format!("failed to launch OpenCL sparse edit fill: {error}"))?;
        }
        let (index, score) =
            self.run_sparse(base_slot, trial_slot, history_count, count, &params)?;
        Ok((
            index,
            super::super::cpu::seed_at(base_seed, index as u32),
            score,
        ))
    }

    #[allow(clippy::too_many_arguments)]
    fn sparse_params(
        &mut self,
        base_slot: usize,
        trial_slot: usize,
        history_count: usize,
        count: usize,
        num_pert: usize,
        leaves: &[Parameter],
        config: RoundConfig,
    ) -> Result<std::sync::Arc<Buffer<Params>>, String> {
        let dimensions = leaves.iter().map(|leaf| leaf.length).sum::<usize>();
        let params = Params {
            row_bytes: to_u32(self.row_bytes, "row bytes")?,
            history: to_u32(history_count, "history length")?,
            candidates: to_u32(count, "candidate count")?,
            leaves: to_u32(leaves.len(), "leaf count")?,
            tiles: to_u32(self.tile_count, "tile count")?,
            neighbors: to_u32(config.surrogate.neighbors, "neighbor count")?,
            base_slot: to_u32(base_slot, "base slot")?,
            trial_slot: to_u32(trial_slot, "trial slot")?,
            center_count: to_u32(dimensions, "dimension count")?,
            acquisition: crate::weights::acquisition_code(config.acquisition.kind),
            epistemic_scale: config.surrogate.epistemic_scale,
            aleatoric_scale: config.surrogate.aleatoric_scale,
            y_scale: config.surrogate.output_scale,
            beta: config.acquisition.beta,
            num_pert: to_u32(num_pert, "sparse edit count")?,
        };
        self.parameters(params)
    }

    fn run_sparse(
        &self,
        base_slot: usize,
        trial_slot: usize,
        history_count: usize,
        count: usize,
        params: &Buffer<Params>,
    ) -> Result<(usize, f32), String> {
        unsafe {
            ExecuteKernel::new(&self.base_distance)
                .set_arg(&self.rows)
                .set_arg(&self.scratch.history_slots)
                .set_arg(&self.scratch.leaves)
                .set_arg(&self.scratch.base_distances)
                .set_arg(params)
                .set_global_work_size(history_count * THREADS)
                .set_local_work_size(THREADS)
                .enqueue_nd_range(&self.queue)
                .map_err(|error| {
                    format!("failed to launch OpenCL sparse base distances: {error}")
                })?;
            ExecuteKernel::new(&self.score_sparse)
                .set_arg(&self.rows)
                .set_arg(&self.scratch.history_slots)
                .set_arg(&self.scratch.outcomes)
                .set_arg(&self.scratch.seeds)
                .set_arg(&self.scratch.draws)
                .set_arg(&self.scratch.leaves)
                .set_arg(&self.scratch.edits)
                .set_arg(&self.scratch.base_distances)
                .set_arg(&self.scratch.scores)
                .set_arg(params)
                .set_global_work_size(count * THREADS)
                .set_local_work_size(THREADS)
                .enqueue_nd_range(&self.queue)
                .map_err(|error| format!("failed to launch OpenCL sparse scoring: {error}"))?;
            ExecuteKernel::new(&self.pick)
                .set_arg(&self.scratch.scores)
                .set_arg(&self.scratch.choice)
                .set_arg(&self.scratch.selected_scores)
                .set_arg(params)
                .set_global_work_size(THREADS)
                .set_local_work_size(THREADS)
                .enqueue_nd_range(&self.queue)
                .map_err(|error| format!("failed to launch OpenCL sparse selection: {error}"))?;
        }
        if self.state.is_none() {
            self.copy_sparse(base_slot, trial_slot)?;
        }
        unsafe {
            ExecuteKernel::new(&self.write_sparse)
                .set_arg(&self.rows)
                .set_arg(&self.scratch.seeds)
                .set_arg(&self.scratch.choice)
                .set_arg(&self.scratch.leaves)
                .set_arg(&self.scratch.edits)
                .set_arg(params)
                .set_global_work_size(1)
                .set_local_work_size(1)
                .enqueue_nd_range(&self.queue)
                .map_err(|error| format!("failed to launch OpenCL sparse write: {error}"))?;
        }
        let mut choice = [0u32];
        let mut score = [0.0f32];
        unsafe {
            self.queue
                .enqueue_read_buffer(&self.scratch.choice, CL_BLOCKING, 0, &mut choice, &[])
                .map_err(|error| format!("failed to read OpenCL sparse choice: {error}"))?;
            self.queue
                .enqueue_read_buffer(
                    &self.scratch.selected_scores,
                    CL_BLOCKING,
                    0,
                    &mut score,
                    &[],
                )
                .map_err(|error| format!("failed to read OpenCL sparse winning score: {error}"))?;
        }
        let index = choice[0] as usize;
        if index >= count {
            return Err("OpenCL sparse trial choice exceeds candidate count".into());
        }
        Ok((index, score[0]))
    }

    fn copy_sparse(&self, base_slot: usize, trial_slot: usize) -> Result<(), String> {
        let (source, mut destination) = unsafe {
            let source = self
                .rows
                .create_sub_buffer(CL_MEM_READ_ONLY, base_slot * self.row_bytes, self.row_bytes)
                .map_err(|error| format!("failed to create OpenCL sparse source row: {error}"))?;
            let destination = self
                .rows
                .create_sub_buffer(
                    CL_MEM_READ_WRITE,
                    trial_slot * self.row_bytes,
                    self.row_bytes,
                )
                .map_err(|error| {
                    format!("failed to create OpenCL sparse destination row: {error}")
                })?;
            (source, destination)
        };
        unsafe {
            self.queue
                .enqueue_copy_buffer(&source, &mut destination, 0, 0, self.row_bytes, &[])
                .map_err(|error| format!("failed to copy OpenCL sparse trial row: {error}"))?;
        }
        Ok(())
    }

    pub(in crate::trials) fn materialize(
        &mut self,
        base_slot: usize,
        trial_slot: usize,
        seed: u64,
        steps: &[LeafStep],
    ) -> Result<(), String> {
        self.ensure_candidates(1)?;
        let seeds = [Seed {
            low: seed as u32,
            high: (seed >> 32) as u32,
        }];
        let choice = [0_u32];
        unsafe {
            self.queue
                .enqueue_write_buffer(&mut self.scratch.seeds, CL_BLOCKING, 0, &seeds, &[])
                .map_err(|error| format!("failed to write OpenCL trial seed: {error}"))?;
            self.queue
                .enqueue_write_buffer(&mut self.scratch.choice, CL_BLOCKING, 0, &choice, &[])
                .map_err(|error| format!("failed to write OpenCL trial choice: {error}"))?;
        }
        self.sync_leaves(steps)?;

        let params = Params {
            row_bytes: to_u32(self.row_bytes, "row bytes")?,
            history: 0,
            candidates: 1,
            leaves: to_u32(steps.len(), "leaf count")?,
            tiles: to_u32(self.tile_count, "tile count")?,
            neighbors: 0,
            base_slot: to_u32(base_slot, "base slot")?,
            trial_slot: to_u32(trial_slot, "trial slot")?,
            center_count: 0,
            acquisition: 0,
            epistemic_scale: 0.0,
            aleatoric_scale: 0.0,
            y_scale: 0.0,
            beta: 0.0,
            num_pert: 0,
        };
        let params = self.parameters(params)?;
        unsafe {
            ExecuteKernel::new(&self.write)
                .set_arg(&self.rows)
                .set_arg(&self.scratch.seeds)
                .set_arg(&self.scratch.choice)
                .set_arg(&self.scratch.leaves)
                .set_arg(&self.scratch.tiles)
                .set_arg(params.as_ref())
                .set_global_work_size(self.tile_count * THREADS)
                .set_local_work_size(THREADS)
                .enqueue_nd_range(&self.queue)
                .map_err(|error| format!("failed to launch OpenCL trial write: {error}"))?;
        }
        self.queue
            .finish()
            .map_err(|error| format!("failed to finish OpenCL trial write: {error}"))
    }

    #[allow(clippy::too_many_arguments)]
    pub(in crate::trials) fn ask_regions(
        &mut self,
        base_slot: usize,
        history: &[(usize, f32)],
        num_regions: usize,
        seeds_per_region: usize,
        seeds: &[u64],
        leaves: &[Parameter],
        config: RoundConfig,
    ) -> Result<Vec<(usize, f32)>, String> {
        self.regions_core(
            base_slot,
            history,
            num_regions,
            seeds_per_region,
            None,
            seeds,
            None,
            leaves,
            config,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(in crate::trials) fn regions_stream(
        &mut self,
        base_slot: usize,
        history: &[(usize, f32)],
        num_regions: usize,
        seeds_per_region: usize,
        base_seed: u64,
        leaves: &[Parameter],
        config: RoundConfig,
    ) -> Result<Vec<(usize, f32)>, String> {
        self.regions_core(
            base_slot,
            history,
            num_regions,
            seeds_per_region,
            None,
            &[],
            Some(base_seed),
            leaves,
            config,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(in crate::trials) fn ask_centers(
        &mut self,
        base_slot: usize,
        history: &[(usize, f32)],
        seeds_per_region: usize,
        centers: &[Center],
        region_centers: &[usize],
        seeds: &[u64],
        leaves: &[Parameter],
        config: RoundConfig,
    ) -> Result<Vec<(usize, f32)>, String> {
        self.regions_core(
            base_slot,
            history,
            region_centers.len(),
            seeds_per_region,
            Some((centers, region_centers)),
            seeds,
            None,
            leaves,
            config,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(in crate::trials) fn centers_stream(
        &mut self,
        base_slot: usize,
        history: &[(usize, f32)],
        seeds_per_region: usize,
        centers: &[Center],
        region_centers: &[usize],
        base_seed: u64,
        leaves: &[Parameter],
        config: RoundConfig,
    ) -> Result<Vec<(usize, f32)>, String> {
        self.regions_core(
            base_slot,
            history,
            region_centers.len(),
            seeds_per_region,
            Some((centers, region_centers)),
            &[],
            Some(base_seed),
            leaves,
            config,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn regions_core(
        &mut self,
        base_slot: usize,
        history: &[(usize, f32)],
        num_regions: usize,
        seeds_per_region: usize,
        tree: Option<(&[Center], &[usize])>,
        seeds: &[u64],
        base_seed: Option<u64>,
        leaves: &[Parameter],
        config: RoundConfig,
    ) -> Result<Vec<(usize, f32)>, String> {
        let history_count = if self.state.is_some() {
            self.capacity
        } else {
            history.len()
        };
        if num_regions == 0 || seeds_per_region == 0 {
            return Err("multi-TR search requires non-zero regions and candidates".to_string());
        }
        let total_candidates = num_regions
            .checked_mul(seeds_per_region)
            .ok_or("multi-TR candidate count overflow")?;
        if base_seed.is_none() && seeds.len() != total_candidates {
            return Err(format!(
                "expected {total_candidates} seeds for {num_regions} regions, got {}",
                seeds.len()
            ));
        }

        self.ensure_candidates(total_candidates)?;
        let distance_groups = total_candidates
            .div_ceil(2)
            .checked_mul(self.tile_count)
            .ok_or("distance dispatch size overflow")?;
        let history_slots: Vec<u32> = history
            .iter()
            .map(|&(slot, _)| to_u32(slot, "history slot"))
            .collect::<Result<_, _>>()?;
        let outcomes: Vec<f32> = history.iter().map(|&(_, value)| value).collect();
        let packed_seeds: Vec<Seed> = seeds
            .iter()
            .map(|&seed| Seed {
                low: seed as u32,
                high: (seed >> 32) as u32,
            })
            .collect();
        let steps = if self.region.is_some() {
            Vec::new()
        } else {
            make_steps(leaves, config.proposal.radius)
        };
        if let Some(seed) = base_seed {
            self.write_static(&history_slots, &outcomes, &steps)?;
            self.fill_seeds(seed, total_candidates)?;
        } else {
            self.write_inputs(&history_slots, &outcomes, &packed_seeds, &steps)?;
        }
        self.fill_draws(config)?;
        let center_count = match tree {
            Some((centers, region_centers)) => {
                self.write_centers(centers, region_centers, seeds_per_region)?;
                centers.len()
            }
            None => 0,
        };
        let params = Params {
            row_bytes: to_u32(self.row_bytes, "row bytes")?,
            history: to_u32(history_count, "history length")?,
            candidates: to_u32(total_candidates, "candidate count")?,
            leaves: to_u32(leaves.len(), "leaf count")?,
            tiles: to_u32(self.tile_count, "tile count")?,
            neighbors: to_u32(config.surrogate.neighbors, "neighbor count")?,
            base_slot: to_u32(base_slot, "base slot")?,
            trial_slot: to_u32(base_slot, "trial slot")?,
            center_count: to_u32(center_count, "center count")?,
            acquisition: crate::weights::acquisition_code(config.acquisition.kind),
            epistemic_scale: config.surrogate.epistemic_scale,
            aleatoric_scale: config.surrogate.aleatoric_scale,
            y_scale: config.surrogate.output_scale,
            beta: config.acquisition.beta,
            num_pert: 0,
        };
        let params = self.parameters(params)?;
        self.run_scores(&params, distance_groups, total_candidates)?;
        self.pick_regions(num_regions, seeds_per_region)
    }

    fn ensure_edits(&mut self, count: usize) -> Result<(), String> {
        if count == 0 {
            return Err("sparse edit buffer cannot be empty".to_string());
        }
        if count <= self.scratch.edit_capacity {
            return Ok(());
        }
        let capacity = count.next_power_of_two();
        self.scratch.edits = buffer(&self.context, capacity, CL_MEM_READ_ONLY, "sparse edits")?;
        self.scratch.edit_capacity = capacity;
        Ok(())
    }

    fn pick_regions(
        &self,
        num_regions: usize,
        seeds_per_region: usize,
    ) -> Result<Vec<(usize, f32)>, String> {
        let params = MultiTrParams {
            num_regions: to_u32(num_regions, "region count")?,
            candidates_per_region: to_u32(seeds_per_region, "candidates per region")?,
        };
        unsafe {
            ExecuteKernel::new(&self.multi_tr_pick)
                .set_arg(&self.scratch.scores)
                .set_arg(&self.scratch.choice)
                .set_arg(&self.scratch.selected_scores)
                .set_arg(&params)
                .set_global_work_size(
                    num_regions
                        .checked_mul(THREADS)
                        .ok_or("multi-TR selection work size overflow")?,
                )
                .set_local_work_size(THREADS)
                .enqueue_nd_range(&self.queue)
                .map_err(|error| format!("failed to launch OpenCL multi-TR selection: {error}"))?;
        }
        let mut choices = vec![0_u32; num_regions];
        let mut scores = vec![0.0_f32; num_regions];
        unsafe {
            self.queue
                .enqueue_read_buffer(&self.scratch.choice, CL_BLOCKING, 0, &mut choices, &[])
                .map_err(|error| format!("failed to read OpenCL multi-TR choices: {error}"))?;
            self.queue
                .enqueue_read_buffer(
                    &self.scratch.selected_scores,
                    CL_BLOCKING,
                    0,
                    &mut scores,
                    &[],
                )
                .map_err(|error| {
                    format!("failed to read OpenCL multi-TR selected scores: {error}")
                })?;
        }
        Ok(choices
            .into_iter()
            .zip(scores)
            .map(|(index, score)| (index as usize, score))
            .collect())
    }

    fn write_centers(
        &mut self,
        centers: &[Center],
        region_centers: &[usize],
        candidates_per_region: usize,
    ) -> Result<(), String> {
        if centers.len() > self.scratch.center_capacity {
            let capacity = centers.len().next_power_of_two();
            self.scratch.centers = buffer(&self.context, capacity, CL_MEM_READ_ONLY, "centers")?;
            self.scratch.center_capacity = capacity;
        }
        let steps: Vec<CenterStep> = centers
            .iter()
            .map(|center| {
                Ok(CenterStep {
                    parent: center
                        .parent
                        .map(|parent| to_u32(parent, "center parent"))
                        .transpose()?
                        .unwrap_or(u32::MAX),
                    seed: Seed {
                        low: center.seed as u32,
                        high: (center.seed >> 32) as u32,
                    },
                })
            })
            .collect::<Result<_, String>>()?;
        let region_centers: Vec<u32> = region_centers
            .iter()
            .map(|&center| to_u32(center, "region center"))
            .collect::<Result<_, _>>()?;
        let candidate_centers: Vec<u32> = region_centers
            .into_iter()
            .flat_map(|center| std::iter::repeat_n(center, candidates_per_region))
            .collect();
        unsafe {
            self.queue
                .enqueue_write_buffer(&mut self.scratch.centers, CL_BLOCKING, 0, &steps, &[])
                .map_err(|error| format!("failed to write OpenCL centers: {error}"))?;
            self.queue
                .enqueue_write_buffer(
                    &mut self.scratch.candidate_centers,
                    CL_BLOCKING,
                    0,
                    &candidate_centers,
                    &[],
                )
                .map_err(|error| format!("failed to write OpenCL candidate centers: {error}"))?;
        }
        Ok(())
    }

    pub(in crate::trials) fn start_state(
        &mut self,
        capacity: usize,
        value: f32,
    ) -> Result<(), String> {
        let mut values = [u32::MAX; 36];
        let total = self.rows.size().map_err(|error| error.to_string())? / self.row_bytes;
        values[..4].copy_from_slice(&[0, 1, capacity as u32, total as u32]);
        let mut state = buffer(&self.context, 36, CL_MEM_READ_WRITE, "search state")?;
        unsafe {
            self.queue
                .enqueue_write_buffer(&mut state, CL_BLOCKING, 0, &values, &[])
                .map_err(|error| error.to_string())?;
            self.queue
                .enqueue_write_buffer(
                    &mut self.scratch.history_slots,
                    CL_BLOCKING,
                    0,
                    &[0u32],
                    &[],
                )
                .map_err(|error| error.to_string())?;
            self.queue
                .enqueue_write_buffer(&mut self.scratch.outcomes, CL_BLOCKING, 0, &[value], &[])
                .map_err(|error| error.to_string())?;
        }
        self.state = Some(state);
        self.capacity = capacity;
        Ok(())
    }

    pub(in crate::trials) fn observe(&self, handle: usize, value: f32) -> Result<(), String> {
        let state = self.state.as_ref().ok_or("search state is not resident")?;
        let region = self.region.as_ref().ok_or("region is not resident")?;
        unsafe {
            ExecuteKernel::new(&self.advance)
                .set_arg(state)
                .set_arg(&self.scratch.history_slots)
                .set_arg(&self.scratch.outcomes)
                .set_arg(region.as_ref())
                .set_arg(&self.shadow)
                .set_arg(&(handle as u32))
                .set_arg(&f64::from(value).to_bits())
                .set_global_work_size(1)
                .enqueue_nd_range(&self.queue)
                .map_err(|error| error.to_string())?
                .wait()
                .map_err(|error| error.to_string())?;
        }
        Ok(())
    }

    pub(in crate::trials) fn state_word(&self, index: usize) -> Result<u32, String> {
        let state = self.state.as_ref().ok_or("search state is not resident")?;
        let mut word = [0u32];
        unsafe {
            self.queue
                .enqueue_read_buffer(state, CL_BLOCKING, index * size_of::<u32>(), &mut word, &[])
                .map_err(|error| error.to_string())?;
        }
        Ok(word[0])
    }
}

#[cfg(test)]
mod posterior_tests {
    use super::*;

    #[test]
    fn noise_slots() {
        let leaves = [Parameter::new(0, 1, 8, 1.0, 1.0, 1.0).unwrap()];
        let mut engine = match Engine::new(&[128], &leaves, 8) {
            Ok(engine) => engine,
            Err(error)
                if error.contains("no OpenCL GPU or CPU device")
                    || error.contains("CL_PLATFORM_NOT_FOUND_KHR")
                    || error.contains("failed to enumerate OpenCL") =>
            {
                return;
            }
            Err(error) => panic!("{error}"),
        };
        for (slot, value) in [(1, 129), (4, 131), (6, 132)] {
            engine.write(slot, &[value]).unwrap();
        }
        let centers = [
            Center {
                parent: None,
                seed: 3,
            },
            Center {
                parent: Some(0),
                seed: 5,
            },
        ];
        let edits = [SparseEdit {
            leaf: 0,
            element: 0,
        }; 4];
        for aleatoric_scale in [0.05, 1.0e25] {
            let config = {
                let mut config = RoundConfig::default();
                config.proposal.radius = 0.0;
                config.surrogate.neighbors = 2;
                config.acquisition.kind = crate::weights::AcquisitionKind::Thompson;
                config.acquisition.seed = 0x1234_5678_9abc_def0;
                config.surrogate.aleatoric_scale = aleatoric_scale;
                config
            };
            // The nearest rows occupy slots 1 and 4, regardless of history
            // order. Compute the reference norm in f64 so tiny
            // squared weights survive.
            let weights = [1.0f32, 9.0].map(|distance| {
                f64::from(
                    1.0 / (1.0e-9 + config.surrogate.epistemic_scale * distance + aleatoric_scale),
                )
            });
            let noise = [1, 4].map(|slot| {
                f64::from(crate::hash::normal_metric(config.acquisition.seed, slot, 0) as f32)
            });
            let z = (weights[0] * noise[0] + weights[1] * noise[1])
                / (weights[0] * weights[0] + weights[1] * weights[1]).sqrt();
            let se = (1.0 / (weights[0] + weights[1]).max(1.0e-12)).sqrt();
            let expected = (se * z) as f32;
            let check = |score: f32| {
                assert!(score.is_finite());
                assert!(
                    (score - expected).abs() < 2.0e-5 * expected.abs().max(1.0),
                    "score={score}, expected={expected}"
                );
            };
            for history in [
                [(6, 0.0), (1, 0.0), (4, 0.0)],
                [(4, 0.0), (6, 0.0), (1, 0.0)],
            ] {
                for seeds in [&[7u64][..], &[13, 7, 13, 11][..]] {
                    let (index, score) = engine
                        .ask(0, &history, 7, seeds, &leaves, config, false)
                        .unwrap();
                    assert_eq!(index, 0);
                    check(score);
                    let (index, score) = engine
                        .ask_sparse(
                            0,
                            &history,
                            7,
                            seeds,
                            &edits[..seeds.len()],
                            1,
                            &leaves,
                            config,
                        )
                        .unwrap();
                    assert_eq!(index, 0);
                    check(score);
                }
                check(
                    engine
                        .ask_stream(0, &history, 7, 19, 2, &leaves, config, false)
                        .unwrap()
                        .2,
                );
                check(
                    engine
                        .sparse_stream(0, &history, 7, 19, 2, 1, &leaves, config)
                        .unwrap()
                        .2,
                );
                for (_, score) in engine
                    .regions_stream(0, &history, 2, 2, 19, &leaves, config)
                    .unwrap()
                {
                    check(score);
                }
                for (_, score) in engine
                    .centers_stream(0, &history, 2, &centers, &[0, 1], 19, &leaves, config)
                    .unwrap()
                {
                    check(score);
                }
            }
        }

        // A device-managed search supplies no host history to the draw upload.
        engine.start_state(3, 0.0).unwrap();
        unsafe {
            engine
                .queue
                .enqueue_write_buffer(
                    engine.state.as_mut().unwrap(),
                    CL_BLOCKING,
                    0,
                    &[0u32, 3],
                    &[],
                )
                .unwrap();
            engine
                .queue
                .enqueue_write_buffer(
                    &mut engine.scratch.history_slots,
                    CL_BLOCKING,
                    0,
                    &[4u32, 6, 1],
                    &[],
                )
                .unwrap();
            engine
                .queue
                .enqueue_write_buffer(
                    &mut engine.scratch.outcomes,
                    CL_BLOCKING,
                    0,
                    &[0.0f32; 3],
                    &[],
                )
                .unwrap();
        }
        let config = {
            let mut config = RoundConfig::default();
            config.proposal.radius = 0.0;
            config.surrogate.neighbors = 2;
            config.acquisition.kind = crate::weights::AcquisitionKind::Thompson;
            config.acquisition.seed = 42;
            config
        };
        let history = [(4, 0.0), (6, 0.0), (1, 0.0)];
        engine.fill_draws(config).unwrap();
        let expected = crate::weights::history_draws(&history, config.acquisition.seed);
        let mut actual = [0.0f32; 8];
        unsafe {
            engine
                .queue
                .enqueue_read_buffer(&engine.scratch.draws, CL_BLOCKING, 0, &mut actual, &[])
                .unwrap();
        }
        assert_eq!([actual[4], actual[6], actual[1]].as_slice(), expected);
    }
}
