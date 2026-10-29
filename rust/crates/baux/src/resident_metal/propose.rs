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
    /// configured layout or its accepted flat 1x4 view. The trial uses the
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
    pub(crate) fn begin_forced(
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
    pub(crate) fn diagnostic_row(
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
    pub(crate) fn begin_initial(&mut self, seed: u64, candidate: usize) -> Result<Buffer, String> {
        if !self.implicit_history
            || self.initial_observations == 0
            || self.history >= self.initial_observations
        {
            return Err("Metal ENN initialization is complete or not configured".into());
        }
        if candidate >= self.pool_layout.count() as usize {
            return Err("Metal initialization candidate must be within pool".into());
        }
        if !self.profiling {
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
    /// region, prepares program/reference state, clears geometry scratch, and
    /// stores the outstanding command; it does not add a history observation.
    pub(super) fn initial_row(&mut self, root: u64, candidate: usize) -> Result<Buffer, String> {
        self.check_idle()?;
        self.activate_morbo()?;
        self.ensure_ref()?;
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
            basis_seed: 0,
            radius,
            alternate_radius: radius,
            candidate: candidate as u32,
            tiles: self.tiles.len() as u32,
            history: self.physical_history() as u32,
            mode: self.perturbation.shader(),
            program: self.proposal_program(),
            base_slot: self.resident_identities[..self.resident_history]
                .iter()
                .position(|&identity| identity == self.base_id)
                .and_then(|slot| u32::try_from(slot).ok())
                .unwrap_or(u32::MAX),
            ..Params::default()
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
    pub(super) fn begin_mode(
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
        if !((arms == 1 && candidates == 4)
            || (arms == self.pool_layout.arms() as usize
                && candidates == self.pool_layout.slots() as usize))
        {
            return Err(
                "Metal ask shape must match its configured four-candidate procedural layout or \
                 use the flat 1x4 view"
                    .into(),
            );
        }
        check_ask(config)?;
        self.validate_selector()?;
        self.last_profile = None;
        self.ensure_ref()?;
        let analytic_pool = self.analytic_pool();
        if self.profiling {
            self.profile_commands.clear();
            if !analytic_pool {
                let command = self.runtime.queue.new_command_buffer().to_owned();
                self.encode_pool(&command, self.pool_params(seed));
                command.set_label("BF16 proposal pool");
                command.commit();
                self.profile_commands.push(command);
            }
            let command = self.runtime.queue.new_command_buffer().to_owned();
            self.encode_select(&command, seed, config, forced_candidate);
            command.set_label("BF16 proposal selection");
            command.commit();
            self.profile_commands.push(command);
            let command = self.runtime.queue.new_command_buffer().to_owned();
            self.encode_row(&command);
            command.set_label("BF16 proposal materialization");
            command.commit();
            self.profile_commands.push(command);
            self.async_command = self.profile_commands.last().cloned();
            return Ok(self.proposal.clone());
        }
        let command = self.runtime.queue.new_command_buffer();
        if !analytic_pool
            && (forced_candidate.is_some()
                || !(self.realized_history() && self.history > self.resident_history))
        {
            self.encode_pool(command, self.pool_params(seed));
        }
        self.encode_select(command, seed, config, forced_candidate);
        self.encode_row(command);
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
    pub(super) fn ask_inner(
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
    pub(super) fn finish_askat(&mut self, start: Instant) -> Result<Proposals, String> {
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
    pub(super) fn finish_profile(&mut self) -> Result<(), String> {
        if !self.profiling {
            self.last_profile = None;
            return Ok(());
        }
        let analytic_pool = self.analytic_pool();
        let expected = if analytic_pool { 2 } else { 3 };
        if self.profile_commands.len() != expected {
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
        let (score_ms, pick_ms, materialize_ms, total_ms) = if analytic_pool {
            (
                0.0,
                milliseconds(0),
                milliseconds(1),
                ((intervals[1].1 - intervals[0].0) * 1000.0) as f32,
            )
        } else {
            (
                milliseconds(0),
                milliseconds(1),
                milliseconds(2),
                ((intervals[2].1 - intervals[0].0) * 1000.0) as f32,
            )
        };
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
            self.ensure_ref()?;
            let pool_count = self.pool_layout.count() as usize;
            let matrix_f64 = if self.neighbors.is_some() {
                let command = self.runtime.queue.new_command_buffer();
                self.neighbors
                    .as_ref()
                    .unwrap()
                    .encode(self, command, root_seed);
                finish(command)?;
                let mut neighbors = self.neighbors.take().unwrap();
                let mat = neighbors.compute_distances(self);
                self.neighbors = Some(neighbors);
                mat
            } else {
                let base_index = self.identities[..self.history]
                    .iter()
                    .position(|&identity| identity == self.base_id)
                    .unwrap_or(0);
                let mut mat = vec![vec![0.0_f64; self.history]; 4];
                unsafe {
                    let dst = self.pool_distances.contents().cast::<f32>();
                    for candidate in 0..4 {
                        let norm = self.latent_norm(candidate);
                        for i in 0..self.history {
                            let base_d = self.pairwise_distances[base_index * MAX_HISTORY + i];
                            let val = f64::from(base_d) + f64::from(norm);
                            mat[candidate][i] = val;
                            dst.add(candidate * self.history + i).write(val as f32);
                        }
                    }
                }
                mat
            };
            Ok(matrix_f64.into_iter().take(pool_count).collect())
        })
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
        pool_predictions: [(f32, f32, f32); 4],
    ) -> Result<(Buffer, Proposals), String> {
        autoreleasepool(|| {
            self.check_idle()?;
            let pool_count = self.pool_layout.count() as usize;
            if selected >= pool_count {
                return Err("Selected candidate index exceeds pool count".into());
            }
            self.ensure_ref()?;
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
            let decision = Decision {
                index: selected as u32,
                valid: 1,
                root_seed,
                seed: self.procedural_seed(root_seed, selected),
                basis_seed: 0,
                radius: self.radius(selected),
                score,
                mode: self.perturbation.shader(),
                program: self.proposal_program(),
                predicted_mean,
                predicted_standard_error,
                incumbent_mean,
                incumbent_standard_error,
                neighbors,
                incumbent_neighbors,
                candidate_means: pool_predictions.map(|p| p.0),
                candidate_standard_errors: pool_predictions.map(|p| p.1),
                candidate_scores: pool_predictions.map(|p| p.2),
            };
            unsafe {
                self.decision.contents().cast::<Decision>().write(decision);
            }
            let start = Instant::now();
            let command = self.runtime.queue.new_command_buffer();
            self.encode_row(command);
            finish(command)?;
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
            selection.pool_predictions,
        )
    }
}
