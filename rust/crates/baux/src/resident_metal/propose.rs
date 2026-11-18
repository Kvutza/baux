//! Generate and rank a trial parameter row without evaluating its objective.

use super::*;
use crate::apple_gpu::{gpu_interval, thread_group};
use metal::MTLResourceOptions;
use std::sync::Arc;

fn fresh_scratch(runtime: &Runtime, buf: &mut Arc<Buffer>) {
    if Arc::strong_count(buf) > 1 {
        let fresh = runtime.device.new_buffer(
            buf.length(),
            MTLResourceOptions::StorageModeShared | MTLResourceOptions::HazardTrackingModeTracked,
        );
        *buf = Arc::new(fresh);
    }
}

pub(super) fn cow_buffer(runtime: &Runtime, buf: &mut Arc<Buffer>) -> Result<(), String> {
    if Arc::strong_count(buf) > 1 {
        let fresh = runtime.device.new_buffer(
            buf.length(),
            MTLResourceOptions::StorageModeShared | MTLResourceOptions::HazardTrackingModeTracked,
        );
        let command = runtime.queue.new_command_buffer();
        let blit = command.new_blit_command_encoder();
        blit.copy_from_buffer(&**buf, 0, &fresh, 0, buf.length());
        blit.end_encoding();
        finish(command)?;
        *buf = Arc::new(fresh);
    }
    Ok(())
}

impl SearchState {
    fn prepare_scratch(&mut self) {
        fresh_scratch(&self.runtime, &mut self.proposal);
        fresh_scratch(&self.runtime, &mut self.partials);
        fresh_scratch(&self.runtime, &mut self.pool_geometry);
        fresh_scratch(&self.runtime, &mut self.pool_aggregates);
        fresh_scratch(&self.runtime, &mut self.candidate_scores);
        fresh_scratch(&self.runtime, &mut self.selected_indices_gpu);
        fresh_scratch(&self.runtime, &mut self.axis_candidates_gpu);
        fresh_scratch(&self.runtime, &mut self.decision);
        fresh_scratch(&self.runtime, &mut self.pool_distances);
        if let Some(selector) = &mut self.objective_selector {
            fresh_scratch(&self.runtime, &mut selector.parameters);
            fresh_scratch(&self.runtime, &mut selector.report);
            fresh_scratch(&self.runtime, &mut selector.vectors);
        }
    }

    pub fn propose(
        self,
        seed: u64,
        mut config: RoundConfig,
        initial_slot: usize,
        forced: Option<usize>,
    ) -> Result<(Self, (Buffer, Proposals, bool)), String> {
        let history = self.history_len()?;
        let initializing = history < config.surrogate.neighbors;
        let pool_count = self.pool_layout.count() as usize;
        let arms = self.pool_layout.arms() as usize;
        let slots = self.pool_layout.slots() as usize;
        let initial_slot = initial_slot % pool_count;
        let (state, row) = if initializing {
            self.begin_initial(seed, initial_slot)?
        } else {
            config.surrogate.neighbors = config.surrogate.neighbors.min(history);
            match forced {
                Some(slot) => self.begin_forced(seed, config, slot % pool_count)?,
                None => self.begin_ask(arms, slots, seed, config)?,
            }
        };
        let (state, proposal) = state.finish_ask()?;
        Ok((state, (row, proposal, initializing)))
    }

    pub fn ask_round(
        self,
        arms: usize,
        candidates: usize,
        seed: u64,
        config: RoundConfig,
    ) -> Result<(Self, Proposals), String> {
        autoreleasepool(|| self.ask_inner(arms, candidates, seed, config))
    }

    pub fn begin_ask(
        self,
        arms: usize,
        candidates: usize,
        seed: u64,
        config: RoundConfig,
    ) -> Result<(Self, Buffer), String> {
        self.begin_mode(arms, candidates, seed, config, None)
    }

    /// Enqueue asynchronous pool evaluation + materialization and return a
    /// fenced write-once [`DeviceRow`] without waiting on the CPU.
    pub fn fenced_ask(
        self,
        arms: usize,
        candidates: usize,
        seed: u64,
        config: RoundConfig,
    ) -> Result<(Self, DeviceRow), String> {
        let (state, _buf) = self.begin_mode(arms, candidates, seed, config, None)?;
        let row = DeviceRow::new(
            Arc::clone(&state.proposal),
            state.async_command.clone(),
            state.dimensions,
        );
        Ok((state, row))
    }

    pub fn begin_forced(
        self,
        seed: u64,
        config: RoundConfig,
        candidate: usize,
    ) -> Result<(Self, Buffer), String> {
        if candidate >= self.pool_layout.count() as usize {
            return Err("Diagnostic candidate must be within pool".into());
        }
        let arms = self.pool_layout.arms() as usize;
        let slots = self.pool_layout.slots() as usize;
        self.begin_mode(arms, slots, seed, config, Some(candidate))
    }

    pub fn diagnostic_row(
        &self,
        round: &Proposals,
        root: u64,
        config: RoundConfig,
        candidate: usize,
    ) -> Result<Buffer, String> {
        self.check_round(round)?;
        let decision = read::<Decision>(&self.decision, 1)[0];
        if candidate >= self.pool_layout.count() as usize || root != decision.root_seed {
            return Err("Diagnostic pool candidate or root mismatch".into());
        }
        let out = self.runtime.buffer::<u16>(self.dimensions);
        let command = self.runtime.queue.new_command_buffer();
        self.encode_select(command, root, config, Some(candidate));
        let encoder =
            command.compute_command_encoder_with_dispatch_type(metal::MTLDispatchType::Concurrent);
        self.dispatch_candidate(encoder, &out, candidate as u32, 0, round.sobol_offset);
        encoder.end_encoding();
        finish(command)?;
        Ok(out)
    }

    /// Materialize all multi-arm selected candidates from
    /// `round.selected_indices` into independent write-once fenced
    /// [`DeviceRow`] buffers on a single pipelined `MTLCommandBuffer`.
    pub fn materialize_batch(&self, round: &Proposals) -> Result<Vec<DeviceRow>, String> {
        self.check_round(round)?;
        if round.selected_indices.len() <= 1 {
            return Ok(vec![round.device_row().clone()]);
        }
        let mut rows = Vec::with_capacity(round.selected_indices.len());
        let mut out_arcs = Vec::with_capacity(round.selected_indices.len());
        for (arm_pos, &_idx) in round.selected_indices.iter().enumerate() {
            if arm_pos == 0 {
                out_arcs.push(Arc::clone(round.device_row().arc()));
            } else {
                out_arcs.push(Arc::new(self.runtime.buffer::<u16>(self.dimensions)));
            }
        }
        let command = self.runtime.queue.new_command_buffer();
        let encoder =
            command.compute_command_encoder_with_dispatch_type(metal::MTLDispatchType::Concurrent);
        for (arm_pos, &cand_idx) in round.selected_indices.iter().enumerate() {
            if arm_pos == 0 {
                continue;
            }
            self.dispatch_candidate(
                encoder,
                &out_arcs[arm_pos],
                cand_idx as u32,
                0,
                round.sobol_offset,
            );
        }
        encoder.end_encoding();
        command.commit();
        let fence = Arc::new(command.to_owned());
        for (arm_pos, arc) in out_arcs.into_iter().enumerate() {
            let f = if arm_pos == 0 {
                None
            } else {
                Some(Arc::clone(&fence))
            };
            rows.push(DeviceRow::new(arc, f, self.dimensions));
        }
        Ok(rows)
    }

    pub fn begin_initial(self, seed: u64, candidate: usize) -> Result<(Self, Buffer), String> {
        if !self.implicit_history
            || self.initial_observations == 0
            || self.history >= self.initial_observations
        {
            return Err("Metal ENN initialization is complete or not configured".into());
        }
        if candidate >= self.pool_layout.count() as usize {
            return Err("Metal initialization candidate must be within pool".into());
        }
        if !self.profiling
            && self.resident_history == 1
            && self.resident_identities[0] == self.base_id
        {
            return self.initial_row(seed, candidate);
        }
        let arms = self.pool_layout.arms() as usize;
        let slots = self.pool_layout.slots() as usize;
        self.begin_mode(arms, slots, seed, RoundConfig::default(), Some(candidate))
    }

    pub fn initial_row(mut self, root: u64, candidate: usize) -> Result<(Self, Buffer), String> {
        self.check_idle()?;
        self = self.activate_morbo()?;
        self.prepare_scratch();
        unsafe {
            std::ptr::write_bytes(
                self.partials.contents().cast::<u8>(),
                0,
                self.partials.length() as usize,
            );
            std::ptr::write_bytes(
                self.pool_geometry.contents().cast::<u8>(),
                0,
                self.pool_geometry.length() as usize,
            );
        }
        let radius = self.radius(candidate);
        let params = Params {
            seed: self.procedural_seed(root, candidate),
            stream_seed: root,
            radius,
            candidate: candidate as u32,
            ..self.pool_params(root)
        };
        let command = self.runtime.queue.new_command_buffer();
        self.encode_proposal(command, params);
        self.encode_select(command, root, RoundConfig::default(), Some(candidate));
        command.commit();
        self.async_command = Some(Arc::new(command.to_owned()));
        let row = (*self.proposal).clone();
        Ok((self, row))
    }

    pub fn begin_mode(
        mut self,
        arms: usize,
        candidates: usize,
        seed: u64,
        config: RoundConfig,
        forced_candidate: Option<usize>,
    ) -> Result<(Self, Buffer), String> {
        self.check_idle()?;
        self = self.activate_morbo()?;
        if self.history == 0 {
            return Err("Measure the initial Metal BF16 incumbent before asking".into());
        }
        if !((arms == 1 && (candidates == 1 || candidates == self.pool_layout.count() as usize))
            || (arms == self.pool_layout.arms() as usize
                && candidates == self.pool_layout.slots() as usize))
        {
            return Err(
                "Metal ask shape must match its configured procedural layout or use 1xN flat view"
                    .into(),
            );
        }
        check_ask(config)?;
        self.validate_selector()?;
        self.last_profile = None;
        self.prepare_scratch();

        if self.profiling {
            self.profile_commands.clear();
            let command = self.runtime.queue.new_command_buffer().to_owned();
            self.encode_pool(&command, self.pool_params(seed));
            command.set_label("BF16 proposal pool");
            command.commit();
            self.profile_commands.push(Arc::new(command));
            let command = self.runtime.queue.new_command_buffer().to_owned();
            self.encode_select(&command, seed, config, forced_candidate);
            command.set_label("BF16 proposal selection");
            command.commit();
            self.profile_commands.push(Arc::new(command));
            if self.pool_layout.count() > 1 {
                let command = self.runtime.queue.new_command_buffer().to_owned();
                self.encode_row(&command);
                command.set_label("BF16 proposal materialization");
                command.commit();
                self.profile_commands.push(Arc::new(command));
            }
            self.async_command = self.profile_commands.last().cloned();
            let row = (*self.proposal).clone();
            return Ok((self, row));
        }
        let command = self.runtime.queue.new_command_buffer();
        self.encode_ask(command, seed, config, forced_candidate);
        command.set_label("BF16 resident proposal");
        command.commit();
        self.async_command = Some(Arc::new(command.to_owned()));
        let row = (*self.proposal).clone();
        Ok((self, row))
    }

    pub fn finish_ask(mut self) -> Result<(Self, Proposals), String> {
        let command = self
            .async_command
            .take()
            .ok_or_else(|| "No asynchronous Metal BF16 ask is pending".to_string())?;
        command.wait_until_completed();
        if command.status() != MTLCommandBufferStatus::Completed {
            self.profile_commands.clear();
            return Err(format!(
                "Metal BF16 asynchronous ask failed: {:?}",
                command.status()
            ));
        }
        self.finish_profile()?;
        self.publish_ask()
    }

    /// Drain any outstanding ask without corrupting or poisoning committed
    /// state.
    pub fn abort_ask(mut self) -> Self {
        if let Some(command) = self.async_command.take() {
            command.wait_until_completed();
        }
        self.profile_commands.clear();
        self.pending = None;
        self.evaluation = None;
        self
    }

    pub fn ask_inner(
        self,
        arms: usize,
        candidates: usize,
        seed: u64,
        config: RoundConfig,
    ) -> Result<(Self, Proposals), String> {
        let start = Instant::now();
        let (state, _row) = self.begin_ask(arms, candidates, seed, config)?;
        state.finish_askat(start)
    }

    pub fn finish_askat(mut self, start: Instant) -> Result<(Self, Proposals), String> {
        let command = self
            .async_command
            .take()
            .ok_or_else(|| "No asynchronous Metal BF16 ask is pending".to_string())?;
        command.wait_until_completed();
        if command.status() != MTLCommandBufferStatus::Completed {
            self.profile_commands.clear();
            return Err(format!(
                "Metal BF16 asynchronous ask failed: {:?}",
                command.status()
            ));
        }
        self.finish_profile()?;
        self.publish_start(start)
    }

    pub fn finish_profile(&mut self) -> Result<(), String> {
        if !self.profiling {
            self.last_profile = None;
            return Ok(());
        }
        if !matches!(self.profile_commands.len(), 2 | 3) {
            return Err("Metal BF16 controller trace is incomplete".into());
        }
        let intervals: Vec<_> = self
            .profile_commands
            .iter()
            .map(|command| {
                if command.status() != MTLCommandBufferStatus::Completed {
                    return Err(format!(
                        "Metal BF16 traced ask failed: {:?}",
                        command.status()
                    ));
                }
                gpu_interval(command)
                    .ok_or_else(|| "Metal did not expose a controller GPU interval".to_string())
            })
            .collect::<Result<_, _>>()?;
        let milliseconds =
            |index: usize| ((intervals[index].1 - intervals[index].0) * 1000.0) as f32;
        let last = intervals.len() - 1;
        let (score_ms, pick_ms, materialize_ms, total_ms) = (
            milliseconds(0),
            milliseconds(1),
            if intervals.len() == 3 {
                milliseconds(2)
            } else {
                0.0
            },
            ((intervals[last].1 - intervals[0].0) * 1000.0) as f32,
        );
        self.last_profile = Some(ProposalProfile {
            score_ms,
            pick_ms,
            materialize_ms,
            total_ms,
        });
        self.profile_commands.clear();
        Ok(())
    }

    pub fn sample_distances(mut self, root_seed: u64) -> Result<(Self, Vec<Vec<f64>>), String> {
        autoreleasepool(|| {
            self.check_idle()?;
            if self.history == 0 {
                return Err(
                    "Measure the initial Metal BF16 incumbent before sampling distances".into(),
                );
            }
            self.prepare_scratch();
            unsafe {
                self.decision.contents().cast::<Decision>().write(Decision {
                    root_seed,
                    valid: 0,
                    ..Decision::default()
                });
            }
            let pool_count = self.pool_layout.count() as usize;
            let cand_groups = pool_count.div_ceil(4).max(1) as u64;
            let pairs = self.resident_history.div_ceil(2).max(1);
            let base_index = self.identities[..self.history]
                .iter()
                .position(|&identity| identity == self.base_id)
                .unwrap_or(0);
            let command = self.runtime.queue.new_command_buffer();
            self.encode_pool(command, self.pool_params(root_seed));
            let select_params = self.select_params(root_seed, RoundConfig::default(), None);
            let reducer = command.new_compute_command_encoder();
            reducer.set_compute_pipeline_state(&self.reduction_pipeline);
            for (index, buffer) in [
                &*self.partials,
                &*self.pool_geometry,
                &*self.pool_aggregates,
            ]
            .iter()
            .enumerate()
            {
                reducer.set_buffer(index as u64, Some(buffer), 0);
            }
            reducer.set_bytes(
                3,
                size_of::<SelectionParams>() as u64,
                (&select_params as *const SelectionParams).cast(),
            );
            reducer.dispatch_thread_groups(
                metal::MTLSize {
                    width: pairs as u64,
                    height: cand_groups,
                    depth: 1,
                },
                thread_group(256),
            );
            reducer.end_encoding();
            finish(command)?;

            let aggregates = read::<Partial>(&self.pool_aggregates, pairs * pool_count);
            let matrix = self.distance_matrix(
                &select_params,
                &aggregates,
                base_index,
                pool_count,
                root_seed,
            );
            Ok((self, matrix))
        })
    }

    fn distance_matrix(
        &self,
        select_params: &SelectionParams,
        aggregates: &[Partial],
        base_index: usize,
        pool_count: usize,
        root_seed: u64,
    ) -> Vec<Vec<f64>> {
        let mut matrix_f64 = vec![vec![0.0_f64; self.history]; pool_count];
        unsafe {
            let dst = self.pool_distances.contents().cast::<f32>();
            for (candidate, row_f64) in matrix_f64.iter_mut().enumerate() {
                let norm = aggregates[candidate].squared;
                for (row, cell) in row_f64.iter_mut().enumerate() {
                    let base_d = if self.implicit_history {
                        self.pairwise_distances[base_index * MAX_HISTORY + row]
                    } else {
                        0.0
                    };
                    *cell = f64::from(base_d + norm);
                }
                for row in (0..self.resident_history).step_by(2) {
                    let aggregate = aggregates[(row / 2) * pool_count + candidate];
                    let first_index = if self.implicit_history {
                        select_params.resident_indices[row] as usize
                    } else {
                        row
                    };
                    if first_index < self.history {
                        row_f64[first_index] = f64::from(aggregate.anchor);
                    }
                    if row + 1 < self.resident_history {
                        let second_index = if self.implicit_history {
                            select_params.resident_indices[row + 1] as usize
                        } else {
                            row + 1
                        };
                        if second_index < self.history {
                            row_f64[second_index] = f64::from(aggregate.rejected);
                        }
                    }
                }
                if select_params.axis_enabled != 0 {
                    let axis_cand = self.axis_candidate(root_seed, candidate).unwrap_or(0.0);
                    for (row, cell) in row_f64.iter_mut().enumerate() {
                        let delta = axis_cand - select_params.axis_history[row];
                        let val = (*cell as f32) + select_params.axis_weight * delta * delta;
                        *cell = f64::from(val);
                    }
                }
                for (row, &val) in row_f64.iter().enumerate() {
                    dst.add(candidate * self.history + row).write(val as f32);
                }
            }
        }
        matrix_f64
    }

    #[allow(clippy::too_many_arguments)]
    pub fn materialize_selected(
        mut self,
        root_seed: u64,
        selected: usize,
        score: f32,
        predicted_mean: f32,
        predicted_standard_error: f32,
        incumbent_mean: f32,
        incumbent_standard_error: f32,
        neighbors: u32,
        incumbent_neighbors: u32,
        pool_predictions: &[(f32, f32, f32)],
    ) -> Result<(Self, (Buffer, Proposals)), String> {
        autoreleasepool(|| {
            self.check_idle()?;
            let pool_count = self.pool_layout.count() as usize;
            if selected >= pool_count {
                return Err("Selected candidate index exceeds pool count".into());
            }
            let prev = read::<Decision>(&self.decision, 1)[0];
            let already_materialized =
                pool_count == 1 && selected == 0 && prev.valid == 0 && prev.root_seed == root_seed;
            let mut candidate_means = [f32::NAN; 4];
            let mut candidate_standard_errors = [f32::NAN; 4];
            let mut candidate_scores = [f32::NAN; 4];
            for (slot, &(mean, se, cand_score)) in pool_predictions.iter().take(4).enumerate() {
                candidate_means[slot] = mean;
                candidate_standard_errors[slot] = se;
                candidate_scores[slot] = cand_score;
            }
            fresh_scratch(&self.runtime, &mut self.candidate_scores);
            fresh_scratch(&self.runtime, &mut self.selected_indices_gpu);
            unsafe {
                let cs_ptr = self.candidate_scores.contents().cast::<CandidateScore>();
                for cand in 0..pool_count {
                    let (mean, se, cand_score) = pool_predictions.get(cand).copied().unwrap_or((
                        f32::NAN,
                        f32::NAN,
                        f32::NAN,
                    ));
                    cs_ptr.add(cand).write(CandidateScore {
                        mean,
                        standard_error: se,
                        score: cand_score,
                        neighbors: if cand == selected { neighbors } else { 0 },
                        changed: 1,
                        invalid: u32::from(!cand_score.is_finite()),
                    });
                }
                self.selected_indices_gpu
                    .contents()
                    .cast::<u32>()
                    .write(selected as u32);
            }
            let decision = Decision {
                index: selected as u32,
                valid: 1,
                root_seed,
                seed: self.procedural_seed(root_seed, selected),
                radius: self.radius(selected),
                score,
                mode: self.sampler.shader(),
                num_selected: 1,
                predicted_mean,
                predicted_standard_error,
                incumbent_mean,
                incumbent_standard_error,
                neighbors,
                incumbent_neighbors,
                candidate_means,
                candidate_standard_errors,
                candidate_scores,
            };
            fresh_scratch(&self.runtime, &mut self.decision);
            unsafe {
                self.decision.contents().cast::<Decision>().write(decision);
            }
            let start = Instant::now();
            if !already_materialized {
                fresh_scratch(&self.runtime, &mut self.proposal);
                let command = self.runtime.queue.new_command_buffer();
                self.encode_row(command);
                finish(command)?;
            }
            let (state, proposal) = self.publish_start(start)?;
            let row = (*state.proposal).clone();
            Ok((state, (row, proposal)))
        })
    }

    pub fn materialize_selection(
        self,
        root_seed: u64,
        selection: &baux::semantics::optimizer::PoolSelection,
    ) -> Result<(Self, (Buffer, Proposals)), String> {
        self.materialize_selected(
            root_seed,
            selection.selected,
            selection.candidate.score,
            selection.candidate.mean,
            selection.candidate.standard_error,
            selection.incumbent.mean,
            selection.incumbent.standard_error,
            selection.candidate.neighbors as u32,
            selection.incumbent.neighbors as u32,
            &selection.pool_predictions,
        )
    }
}
