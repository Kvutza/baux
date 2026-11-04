//! Generate and rank a trial parameter row without evaluating its objective.
//!
//! `propose` chooses initialization or surrogate selection, waits for Metal,
//! and publishes one pending `Proposals` handle. `begin_*`/`finish_*` expose
//! the same queue boundary for callers that append evaluator work themselves.
//! The returned buffer is reusable proposal storage, not an immutable model.
//! A matching tell and synchronization must complete before the next ask.
//!
//! A one-element pool still runs this transaction but offers acquisition no
//! alternative to select. Neither materialization nor publication commits an
//! update; `observations` chooses acceptance and `history` applies it.

use super::*;
use crate::apple_gpu::gpu_interval;

impl SearchState {
    /// Produce one pending, fully materialized trial from the configured pool.
    ///
    /// If history has fewer than `config.surrogate.neighbors` observations,
    /// materialize `initial_slot` without acquisition choice. Otherwise
    /// rank with ENN or use `forced` for a selection ablation; both indices
    /// are reduced modulo the pool count here. Waits for the GPU and
    /// returns row, handle, and the initialization flag needed to choose
    /// the corresponding tell policy. Does not evaluate a forward pass,
    /// record a loss, or accept an update.
    pub fn propose(
        &mut self,
        seed: u64,
        mut config: RoundConfig,
        initial_slot: usize,
        forced: Option<usize>,
    ) -> Result<(Buffer, Proposals, bool), String> {
        let history = self.history_len()?;
        let initializing = history < config.surrogate.neighbors;
        let pool_count = self.pool_layout.count() as usize;
        let initial_slot = initial_slot % pool_count;
        let row = if initializing {
            self.begin_initial(seed, initial_slot)?
        } else {
            config.surrogate.neighbors = config.surrogate.neighbors.min(history);
            match forced {
                Some(slot) => self.begin_forced(seed, config, slot % pool_count)?,
                None => self.begin_ask(
                    self.pool_layout.arms() as usize,
                    self.pool_layout.slots() as usize,
                    seed,
                    config,
                )?,
            }
        };
        let proposal = self.finish_ask()?;
        Ok((row, proposal, initializing))
    }

    /// Synchronously surrogate-score the pool and publish one trial handle.
    /// `candidates` is slots per arm, not objective calls. Shape must match the
    /// configured layout or its accepted flat 1xN view. The trial uses the
    /// state's precision/program; this API is not limited to correlated BF16.
    pub fn ask_round(
        &mut self,
        arms: usize,
        candidates: usize,
        seed: u64,
        config: RoundConfig,
    ) -> Result<Proposals, String> {
        autoreleasepool(|| self.ask_inner(arms, candidates, seed, config))
    }

    /// Submit pool selection and one row materialization without waiting.
    /// Requires an idle state with an initial observation. The caller may
    /// append evaluator work on the same ordered queue, then call
    /// `finish_ask` before tell. Returns shared scratch storage; its
    /// contents are not CPU-ready on return. No candidate objective is
    /// evaluated by this method.
    pub fn begin_ask(
        &mut self,
        arms: usize,
        candidates: usize,
        seed: u64,
        config: RoundConfig,
    ) -> Result<Buffer, String> {
        self.begin_mode(arms, candidates, seed, config, None)
    }

    /// Submit a precommitted pool index instead of acquisition maximization.
    /// Rejects out-of-range indices and otherwise follows `begin_ask`'s queue
    /// lifecycle. This changes selection only, not generation or tell policy.
    pub fn begin_forced(
        &mut self,
        seed: u64,
        config: RoundConfig,
        candidate: usize,
    ) -> Result<Buffer, String> {
        if candidate >= self.pool_layout.count() as usize {
            return Err("Diagnostic candidate must be within pool".into());
        }
        self.begin_mode(
            self.pool_layout.arms() as usize,
            self.pool_layout.slots() as usize,
            seed,
            config,
            Some(candidate),
        )
    }

    /// Materialize a counterfactual from an already-scored pool. Diagnostics
    /// must restore the original selection before tell: this overwrites both
    /// the shared GPU decision and proposal row. Requires the pending handle
    /// and its original root seed, waits for completion, and records no loss.
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
        let command = self.runtime.queue.new_command_buffer();
        self.encode_select(command, root, config, Some(candidate));
        self.encode_row(command);
        finish(command)?;
        Ok(self.proposal.clone())
    }

    /// Submit a specified slot while implicit history is still warming up.
    /// Rejects a completed/unconfigured warm-up or invalid slot. Profiling uses
    /// the ordinary staged path with forced selection; otherwise `initial_row`
    /// avoids full pool ranking. The caller must finish and measure the row.
    pub fn begin_initial(&mut self, seed: u64, candidate: usize) -> Result<Buffer, String> {
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
        self.begin_mode(
            self.pool_layout.arms() as usize,
            self.pool_layout.slots() as usize,
            seed,
            RoundConfig::default(),
            Some(candidate),
        )
    }

    /// Submit one warm-up row and its forced-selection descriptor.
    /// The caller has validated initialization and slot bounds. Activates the
    /// region, clears geometry scratch, and stores the outstanding command; it
    /// does not add a history observation.
    pub fn initial_row(&mut self, root: u64, candidate: usize) -> Result<Buffer, String> {
        self.check_idle()?;
        self.activate_morbo()?;
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
        self.async_command = Some(command.to_owned());
        Ok(self.proposal.clone())
    }

    /// Shared asynchronous ask implementation for normal and forced selection.
    /// Validates state, shape, and acquisition, prepares geometry as required,
    /// then enqueues selection and materialization. Sampled/procedural history
    /// skips full-pool materialization; profiling separates stages into ordered
    /// commands. Mutates queue/scratch state but leaves the incumbent
    /// unchanged.
    pub fn begin_mode(
        &mut self,
        arms: usize,
        candidates: usize,
        seed: u64,
        config: RoundConfig,
        forced_candidate: Option<usize>,
    ) -> Result<Buffer, String> {
        self.check_idle()?;
        self.activate_morbo()?;
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
        if self.profiling {
            self.profile_commands.clear();
            let command = self.runtime.queue.new_command_buffer().to_owned();
            self.encode_pool(&command, self.pool_params(seed));
            command.set_label("BF16 proposal pool");
            command.commit();
            self.profile_commands.push(command);
            let command = self.runtime.queue.new_command_buffer().to_owned();
            self.encode_select(&command, seed, config, forced_candidate);
            command.set_label("BF16 proposal selection");
            command.commit();
            self.profile_commands.push(command);
            if self.pool_layout.count() > 1 {
                let command = self.runtime.queue.new_command_buffer().to_owned();
                self.encode_row(&command);
                command.set_label("BF16 proposal materialization");
                command.commit();
                self.profile_commands.push(command);
            }
            self.async_command = self.profile_commands.last().cloned();
            return Ok(self.proposal.clone());
        }
        let command = self.runtime.queue.new_command_buffer();
        self.encode_ask(command, seed, config, forced_candidate);
        command.set_label("BF16 resident proposal");
        command.commit();
        self.async_command = Some(command.to_owned());
        Ok(self.proposal.clone())
    }

    /// Wait for the pending command, validate profiling, and publish its
    /// handle. Consumes the outstanding command; GPU failure poisons the
    /// search state. The handle identifies this transaction, but its
    /// backing row remains reusable scratch. Publication neither evaluates
    /// nor accepts the row.
    pub fn finish_ask(&mut self) -> Result<Proposals, String> {
        let command = self
            .async_command
            .take()
            .ok_or_else(|| "No asynchronous Metal BF16 ask is pending".to_string())?;
        command.wait_until_completed();
        if command.status() != MTLCommandBufferStatus::Completed {
            self.poisoned = true;
            return Err(format!(
                "Metal BF16 asynchronous ask failed: {:?}",
                command.status()
            ));
        }
        self.finish_profile()?;
        self.publish_ask()
    }

    /// Drain any outstanding ask and permanently poison this search state.
    /// Used after evaluator failure; it does not roll back or enable retry on
    /// the same state. Poisoning also occurs when no command is outstanding.
    pub fn abort_ask(&mut self) {
        if let Some(command) = self.async_command.take() {
            command.wait_until_completed();
        }
        self.poisoned = true;
    }

    /// Start the host timer, submit an ask, and publish after GPU completion.
    /// Internal synchronous wrapper; objective evaluation is outside its timer.
    pub fn ask_inner(
        &mut self,
        arms: usize,
        candidates: usize,
        seed: u64,
        config: RoundConfig,
    ) -> Result<Proposals, String> {
        let start = Instant::now();
        self.begin_ask(arms, candidates, seed, config)?;
        self.finish_askat(start)
    }

    /// Finish an outstanding ask with the caller's host timing origin.
    /// Like `finish_ask`, consumes the command and poisons on GPU failure;
    /// `publish_start` uses `start` only for fallback profiling metadata.
    pub fn finish_askat(&mut self, start: Instant) -> Result<Proposals, String> {
        let command = self
            .async_command
            .take()
            .ok_or_else(|| "No asynchronous Metal BF16 ask is pending".to_string())?;
        command.wait_until_completed();
        if command.status() != MTLCommandBufferStatus::Completed {
            self.poisoned = true;
            return Err(format!(
                "Metal BF16 asynchronous ask failed: {:?}",
                command.status()
            ));
        }
        self.finish_profile()?;
        self.publish_start(start)
    }

    /// Collect completed ask-stage GPU timestamps in milliseconds.
    /// Expects two commands for sampled/procedural geometry or three for an
    /// explicit pool. `total_ms` spans their GPU intervals, not end-to-end host
    /// round time. Clears staged commands on success; rejects missing timings.
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

    /// Primitive 1 (GPU Engine Seam): Sample candidate-pool coordinates and
    /// compute the C x N block-weighted squared distance matrix in unified
    /// memory without invoking the Metal selection shader.
    pub fn sample_distances(&mut self, root_seed: u64) -> Result<Vec<Vec<f64>>, String> {
        autoreleasepool(|| {
            self.check_idle()?;
            if self.history == 0 {
                return Err(
                    "Measure the initial Metal BF16 incumbent before sampling distances".into(),
                );
            }
            unsafe {
                self.decision.contents().cast::<Decision>().write(Decision {
                    root_seed,
                    valid: 0,
                    ..Decision::default()
                });
            }
            let pool_count = self.pool_layout.count() as usize;
            let base_index = self.identities[..self.history]
                .iter()
                .position(|&identity| identity == self.base_id)
                .unwrap_or(0);
            let command = self.runtime.queue.new_command_buffer();
            self.encode_pool(command, self.pool_params(root_seed));
            let select_params = self.select_params(root_seed, RoundConfig::default(), None);
            let reducer = command.new_compute_command_encoder();
            reducer.set_compute_pipeline_state(&self.reduction_pipeline);
            for (index, buffer) in [&self.partials, &self.pool_geometry, &self.pool_aggregates]
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
                crate::apple_gpu::thread_group(self.resident_history.div_ceil(2).max(1) as u64),
                crate::apple_gpu::thread_group(256),
            );
            reducer.end_encoding();
            finish(command)?;

            let pairs = self.resident_history.div_ceil(2).max(1);
            let aggregates = read::<Partial>(&self.pool_aggregates, pairs * 4);
            Ok(self.distance_matrix(&select_params, &aggregates, base_index, pool_count))
        })
    }

    fn distance_matrix(
        &self,
        select_params: &SelectionParams,
        aggregates: &[Partial],
        base_index: usize,
        pool_count: usize,
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
                    let aggregate = aggregates[(row / 2) * 4 + candidate];
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
                    for (row, cell) in row_f64.iter_mut().enumerate() {
                        let delta = select_params.axis_candidates[candidate]
                            - select_params.axis_history[row];
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

    /// Primitive 2 (GPU Engine Seam): Write the host-selected candidate
    /// descriptor to unified memory and materialize the single winning
    /// parameter vector into `self.proposal`.
    #[allow(clippy::too_many_arguments)]
    pub fn materialize_selected(
        &mut self,
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
    ) -> Result<(Buffer, Proposals), String> {
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
            let decision = Decision {
                index: selected as u32,
                valid: 1,
                root_seed,
                seed: self.procedural_seed(root_seed, selected),
                radius: self.radius(selected),
                score,
                mode: self.perturbation.shader(),
                pad: 0,
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
            unsafe {
                self.decision.contents().cast::<Decision>().write(decision);
            }
            let start = Instant::now();
            if !already_materialized {
                let command = self.runtime.queue.new_command_buffer();
                self.encode_row(command);
                finish(command)?;
            }
            let proposal = self.publish_start(start)?;
            Ok((self.proposal.clone(), proposal))
        })
    }

    /// Materialize the winning candidate directly from a surrogate
    /// `PoolSelection`.
    pub fn materialize_selection(
        &mut self,
        root_seed: u64,
        selection: &baux::semantics::optimizer::PoolSelection,
    ) -> Result<(Buffer, Proposals), String> {
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
