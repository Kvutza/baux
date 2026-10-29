//! Commit a decided trial and retain its observation under bounded storage.
//!
//! Acceptance is supplied by `observations`, not recomputed here. The current
//! incumbent is protected when logical history evicts its oldest other
//! identity; rejected trials can still be retained as surrogate data. Logical
//! observations, resident full parameter rows, and sampled parameter rows are
//! different stores.
//!
//! This is an ordered mutation, not an atomic database transaction: GPU commit
//! precedes some bookkeeping/fitting and controller work. A later error need
//! not mean that weights were untouched. The caller must propagate the error,
//! not blindly replay the tell against partially advanced state.

use super::*;
use crate::apple_gpu::gpu_interval;

impl SearchState {
    #[allow(clippy::too_many_arguments)]
    /// Wrap scalar reward/error metadata as an objective observation and
    /// delegate the already-decided acceptance, history retention, and
    /// radius adaptation. `adapt_region` controls controller work, not
    /// whether weights may commit.
    pub(super) fn tell_absolute(
        &mut self,
        round: &Proposals,
        value: f32,
        variance: f32,
        incumbent_value: f32,
        incumbent_variance: f32,
        accept: bool,
        trust_incumbent: Option<f32>,
        trust_outcome: Option<TrustRegionOutcome>,
        reliability_evidence: Option<reliability_region::ReliabilityEvidence>,
        adapt_region: bool,
    ) -> Result<(), String> {
        self.absolute_objectives(
            round,
            crate::objective_observation::ObjectiveObservation::scalar_adapter(value, variance),
            incumbent_value,
            incumbent_variance,
            accept,
            trust_incumbent,
            trust_outcome,
            reliability_evidence,
            adapt_region,
        )
    }

    #[allow(clippy::too_many_arguments)]
    /// Apply one pending proposal's observation and caller-supplied acceptance.
    /// Validates handle ownership and absolute-observation mode, chooses an
    /// archive slot, commits accepted weights, records geometry/outcomes, fits
    /// ENN when configured, and adapts the active controller. On success clears
    /// `pending` and queues the acceptance flag for `sync`. Later failures can
    /// follow an earlier weight commit; this method does not promise rollback.
    pub(super) fn absolute_objectives(
        &mut self,
        round: &Proposals,
        observation: crate::objective_observation::ObjectiveObservation,
        incumbent_value: f32,
        incumbent_variance: f32,
        accept: bool,
        trust_incumbent: Option<f32>,
        trust_outcome: Option<TrustRegionOutcome>,
        reliability_evidence: Option<reliability_region::ReliabilityEvidence>,
        adapt_region: bool,
    ) -> Result<(), String> {
        let mut mutated = false;
        let result = (|| {
            let crate::objective_observation::ObjectiveEstimate {
                mean: value,
                variance,
            } = observation.control();
            self.tell_profile = None;
            self.check_round(round)?;
            if self.relative {
                return Err("Cannot mix absolute and paired-relative observations".into());
            }
            check_scores(&[value, incumbent_value], &[variance, incumbent_variance])?;
            self.objective_history.validate(observation)?;
            let (identity, commit) = validate_commit(self, round, accept)?;
            let target_slot = choose_slot(self)?;
            let implicit_distances = self.observation_distances(round)?;
            mutated = true;
            let slot = if self.resident_history == self.history_rows.len() {
                0
            } else {
                self.resident_history
            };
            let command = self.runtime.queue.new_command_buffer().to_owned();
            let zero_copy = self.commit_weights(round, accept, &command)?;
            self.archive_proposal(slot, zero_copy);
            self.archive_samples(round, target_slot);
            record_resident(self, &command, accept, slot, identity, commit.retained())?;
            self.record_axis(round, target_slot, accept);
            self.record_observation(
                round,
                identity,
                target_slot,
                (value, variance),
                observation,
                implicit_distances,
                accept,
            )?;
            self.best = if accept { value } else { incumbent_value };
            self.best_variance = if accept { variance } else { incumbent_variance };
            let update = self.adapt_observation(
                observation,
                value,
                trust_incumbent,
                trust_outcome,
                reliability_evidence,
                adapt_region,
            );
            if let Err(error) = update {
                self.poisoned = true;
                return Err(error);
            }
            self.pending = None;
            self.evaluation = None;
            self.queued = Some(accept);
            Ok(())
        })();
        if mutated && result.is_err() {
            self.poisoned = true;
        }
        result
    }

    /// If accepted, update reference state and promote the materialized trial.
    /// Exact-history mode copies into the stable base allocation; otherwise it
    /// swaps base/proposal ownership after GPU completion. Returns whether that
    /// zero-copy swap occurred, so archiving does not recycle a live incumbent.
    /// GPU/reference failure poisons state; rejection leaves weights unchanged.
    fn commit_weights(
        &mut self,
        round: &Proposals,
        accept: bool,
        command: &CommandBufferRef,
    ) -> Result<bool, String> {
        if accept {
            if !self.independent_fp16 {
                self.encode_ref(
                    &command,
                    Params {
                        seed: round.seed,
                        candidate: round.index as u32,
                        ..Params::default()
                    },
                );
            }
            if self.exact_history {
                let blit = command.new_blit_command_encoder();
                blit.copy_from_buffer(&self.proposal, 0, &self.base, 0, self.row_bytes());
                blit.end_encoding();
            }
            if let Err(error) = finish(&command) {
                self.poisoned = true;
                return Err(error);
            }
            if !self.exact_history {
                std::mem::swap(&mut self.base, &mut self.proposal);
            }
            if let Err(error) = self.check_reference() {
                self.poisoned = true;
                return Err(error);
            }
        }
        Ok(accept && !self.exact_history)
    }

    /// Copy the selected candidate's sampled parameters into its logical slot
    /// when sampled neighbor geometry exists. Does not archive a full model or
    /// decide acceptance; rejected observations need geometry as well.
    fn archive_samples(&mut self, round: &Proposals, target_slot: usize) {
        if let Some(neighbors) = &mut self.neighbors {
            neighbors.archive(target_slot, round.index);
        }
    }

    /// Rotate resident allocations without reusing the incumbent as scratch.
    /// Latent mode retains only an incumbent alias; other modes archive the
    /// trial row and recycle the displaced allocation. May allocate replacement
    /// scratch when an alias would otherwise corrupt retained/current weights.
    fn archive_proposal(&mut self, slot: usize, zero_copy: bool) {
        if self.latent_history {
            if zero_copy {
                self.history_rows[0] = self.base.clone();
            }
            return;
        }
        if zero_copy {
            // `base` now owns the accepted candidate. Archive that allocation
            // and recycle only the FIFO allocation being replaced as proposal
            // scratch. The previous incumbent may still be referenced by an
            // older history row, so writing through `proposal` would corrupt
            // replay history after the next ask.
            self.proposal = std::mem::replace(&mut self.history_rows[slot], self.base.clone());
            if std::ptr::eq::<metal::BufferRef>(&*self.proposal, &*self.base) {
                self.proposal = self.runtime.buffer::<u16>(self.dimensions);
            }
            return;
        }
        std::mem::swap(&mut self.proposal, &mut self.history_rows[slot]);
        if std::ptr::eq::<metal::BufferRef>(&*self.proposal, &*self.base) {
            // The overwritten FIFO slot was the current incumbent's shared
            // history view. Do not reuse that allocation as proposal scratch.
            self.proposal = self.runtime.buffer::<u16>(self.dimensions);
        }
    }

    /// Map ask-time history identities to current logical slots.
    /// Implicit history requires one distance for every retained observation;
    /// missing identities are errors. An optional scalar search axis is removed
    /// to leave tensor-only components. Explicit resident mode returns `None`.
    fn observation_distances(
        &self,
        round: &Proposals,
    ) -> Result<Option<Vec<(usize, f32)>>, String> {
        let distances = if self.implicit_history {
            if round.history_distances.len() != self.history {
                return Err("Metal implicit history distance row has the wrong length".into());
            }
            Some(
                round
                    .history_distances
                    .iter()
                    .map(|&(previous_identity, distance)| {
                        self.identities[..self.history]
                            .iter()
                            .position(|&candidate| candidate == previous_identity)
                            .map(|previous| {
                                let tensor_distance = if let (Some(axis), Some(coordinate)) =
                                    (&self.axis, round.axis_coordinate)
                                {
                                    (distance
                                        - self.axis_distance(coordinate, axis.history[previous]))
                                    .max(0.0)
                                } else {
                                    distance
                                };
                                (previous, tensor_distance)
                            })
                            .ok_or_else(|| {
                                "Metal implicit history distance has an unknown identity"
                                    .to_string()
                            })
                    })
                    .collect::<Result<Vec<_>, _>>()?,
            )
        } else {
            None
        };
        Ok(distances)
    }
    /// Write the trial's reward/error, identity, objective vector, and
    /// geometry. Implicit storage updates symmetric distance/family rows
    /// and optional exact-replay metadata, then fits ENN after warm-up if
    /// enabled. Explicit storage advances its FIFO. Assumes slot/distance
    /// and family-component invariants established by proposal publication
    /// and the tell caller.
    fn record_observation(
        &mut self,
        round: &Proposals,
        identity: i64,
        slot: usize,
        (value, variance): (f32, f32),
        observation: crate::objective_observation::ObjectiveObservation,
        distances: Option<Vec<(usize, f32)>>,
        accept: bool,
    ) -> Result<(), String> {
        if self.implicit_history {
            let next = slot;
            if self.exact_history {
                if round.block_scales.len() != self.blocks.len() {
                    return Err("Exact replay proposal has the wrong block-scale count".into());
                }
                let step = ReplayStep {
                    seed: round.seed,
                    radius: round.length,
                    accepted: u32::from(accept),
                    candidate: round.index as u32,
                    mode: self.perturbation.shader(),
                };
                // Shared Metal buffers are CPU-visible and no command is using
                // these future-history slots while tell owns the state.
                unsafe {
                    self.replay_steps
                        .contents()
                        .cast::<ReplayStep>()
                        .add(next)
                        .write(step);
                    std::ptr::copy_nonoverlapping(
                        round.block_scales.as_ptr(),
                        self.replay_scales
                            .contents()
                            .cast::<f32>()
                            .add(next * self.blocks.len()),
                        self.blocks.len(),
                    );
                }
            }
            self.outcomes[next] = value;
            self.variances[next] = variance;
            self.identities[next] = identity;
            for &(previous, distance) in distances.as_ref().unwrap() {
                self.pairwise_distances[next * MAX_HISTORY + previous] = distance;
                self.pairwise_distances[previous * MAX_HISTORY + next] = distance;
            }
            self.pairwise_distances[next * MAX_HISTORY + next] = 0.0;
            if self.history < MAX_HISTORY {
                self.history += 1;
            }
            if let Some(family) = &mut self.family {
                let components = round
                    .family_distances
                    .as_ref()
                    .ok_or("Missing family components")?;
                for (previous, &values) in components.iter().enumerate() {
                    family.components[next * MAX_HISTORY + previous] = values;
                    family.components[previous * MAX_HISTORY + next] = values;
                }
                family.components[next * MAX_HISTORY + next] = [0.0; FAMILIES];
            }
            self.objective_history
                .record_slot(next, identity, observation, accept);
            if self.fit_params && self.history >= self.initial_observations {
                self.fit_enn()?;
            }
        } else {
            self.outcomes[slot] = value;
            self.variances[slot] = variance;
            self.identities[slot] = identity;
            self.objective_history.record(identity, observation, accept);
            if self.history == self.history_rows.len() {
                self.outcomes.rotate_left(1);
                self.variances.rotate_left(1);
                self.identities.rotate_left(1);
            } else {
                self.history += 1;
            }
        }
        Ok(())
    }
    /// Dispatch controller work after observation/weight bookkeeping.
    /// Priority is multi-region control, then requested reliability adaptation,
    /// then ordinary trust-region updates. Reliability requires supplied model
    /// evidence; a trust outcome requires its incumbent baseline. With
    /// adaptation disabled, only the non-MORBO observation counter
    /// advances.
    fn adapt_observation(
        &mut self,
        observation: crate::objective_observation::ObjectiveObservation,
        value: f32,
        trust_incumbent: Option<f32>,
        trust_outcome: Option<TrustRegionOutcome>,
        reliability_evidence: Option<reliability_region::ReliabilityEvidence>,
        adapt_region: bool,
    ) -> Result<(), String> {
        let update = if self.has_morbo() {
            self.update_morbo(observation)
        } else if adapt_region && self.reliability.is_some() {
            reliability_evidence
                .ok_or_else(|| "Reliability control requires a model-aware tell".to_string())
                .and_then(|evidence| self.update_reliability(evidence))
        } else if adapt_region {
            match (trust_incumbent, trust_outcome) {
                (Some(incumbent), Some(outcome)) => self.update_outcome(value, incumbent, outcome),
                (Some(incumbent), None) => self.update_noisy(value, incumbent),
                (None, None) => self.update_region(value),
                (None, Some(_)) => Err("Trust outcome requires an incumbent".into()),
            }
        } else {
            self.trust_observations += 1;
            Ok(())
        };
        update
    }

    /// Primitive 3 (GPU Engine Seam): Archive the selected candidate's sampled
    /// coordinates into `target_slot`, commit or reject the GPU weight buffer,
    /// and update the host-controlled radius without invoking legacy tell
    /// policy.
    #[allow(clippy::too_many_arguments)]
    pub fn commit_host(
        &mut self,
        round: &Proposals,
        target_slot: usize,
        observation: crate::objective_observation::ObjectiveObservation,
        accept: bool,
        next_length: f64,
    ) -> Result<(), String> {
        let mut mutated = false;
        let result = metal::objc::rc::autoreleasepool(|| {
            let crate::objective_observation::ObjectiveEstimate {
                mean: value,
                variance,
            } = observation.control();
            self.tell_profile = None;
            self.check_round(round)?;
            check_scores(&[value], &[variance])?;
            self.objective_history.validate(observation)?;
            if target_slot >= MAX_HISTORY || target_slot > self.history {
                return Err("Host target slot is out of bounds".into());
            }
            let (identity, commit) = validate_commit(self, round, accept)?;
            let implicit_distances = self.observation_distances(round)?;
            mutated = true;
            let slot = if self.resident_history == self.history_rows.len() {
                0
            } else {
                self.resident_history
            };
            let command = self.runtime.queue.new_command_buffer().to_owned();
            let zero_copy = self.commit_weights(round, accept, &command)?;
            self.archive_proposal(slot, zero_copy);
            self.archive_samples(round, target_slot);
            self.observation = identity;
            self.base_id = commit.retained();
            self.record_axis(round, target_slot, accept);
            self.outcomes[target_slot] = value;
            self.variances[target_slot] = variance;
            self.identities[target_slot] = identity;
            if let Some(distances) = &implicit_distances {
                for &(previous, distance) in distances {
                    self.pairwise_distances[target_slot * MAX_HISTORY + previous] = distance;
                    self.pairwise_distances[previous * MAX_HISTORY + target_slot] = distance;
                }
                self.pairwise_distances[target_slot * MAX_HISTORY + target_slot] = 0.0;
            }
            if self.history < MAX_HISTORY && target_slot == self.history {
                self.history += 1;
            }
            if let Some(family) = &mut self.family {
                let components = round
                    .family_distances
                    .as_ref()
                    .ok_or("Missing family components")?;
                for (previous, &values) in components.iter().enumerate() {
                    family.components[target_slot * MAX_HISTORY + previous] = values;
                    family.components[previous * MAX_HISTORY + target_slot] = values;
                }
                family.components[target_slot * MAX_HISTORY + target_slot] = [0.0; FAMILIES];
            }
            self.objective_history
                .record_slot(target_slot, identity, observation, accept);
            if accept {
                self.best = value;
                self.best_variance = variance;
            }
            self.length =
                next_length.clamp(self.length_config.length_min, self.length_config.length_max);
            self.pending = None;
            self.evaluation = None;
            self.queued = Some(accept);
            Ok(())
        });
        if mutated && result.is_err() {
            self.poisoned = true;
        }
        result
    }
}

fn validate_commit(
    state: &SearchState,
    round: &Proposals,
    accept: bool,
) -> Result<(i64, baux::semantics::optimizer::TellDecision), String> {
    let identity = state
        .observation
        .checked_add(1)
        .ok_or("Observation ID overflow")?;
    let commit = baux::semantics::optimizer::TellDecision::new(
        baux::semantics::optimizer::RoundId {
            owner: state.owner,
            proposal: state.pending.as_ref().ok_or("No pending proposal")?.id,
            incumbent: state.base_id,
        },
        baux::semantics::optimizer::RoundId {
            owner: round.owner,
            proposal: round.id,
            incumbent: round.base_id,
        },
        identity,
        accept,
    )
    .ok_or("stale candidate commit")?;
    Ok((identity, commit))
}

fn choose_slot(state: &SearchState) -> Result<usize, String> {
    if state.implicit_history {
        if state.exact_history {
            baux::semantics::neighbor::retention(&state.identities[..state.history], state.base_id)
                .map_err(str::to_owned)
        } else {
            let slope = state.fitted_enn.map_or(1.0, |(_, epistemic, _, y_scale)| {
                epistemic.max(0.0).sqrt() * y_scale.max(0.0)
            });
            baux::semantics::neighbor::retention_cone(
                &state.identities[..state.history],
                state.base_id,
                &state.outcomes[..state.history],
                &state.variances[..state.history],
                &state.pairwise_distances,
                MAX_HISTORY,
                slope,
            )
            .map_err(str::to_owned)
        }
    } else if state.resident_history == state.history_rows.len() {
        Ok(0)
    } else {
        Ok(state.resident_history)
    }
}

fn record_resident(
    state: &mut SearchState,
    command: &CommandBufferRef,
    accept: bool,
    slot: usize,
    identity: i64,
    retained: i64,
) -> Result<(), String> {
    if state.profiling {
        let reference_ms = if accept {
            gpu_interval(command)
                .map(|(start, end)| ((end - start) * 1000.0) as f32)
                .ok_or_else(|| "Metal did not expose a tell GPU interval".to_string())?
        } else {
            0.0
        };
        state.tell_profile = Some(TellProfile {
            reference_ms,
            history_copy_ms: 0.0,
            total_ms: reference_ms,
        });
    }
    state.observation = identity;
    state.base_id = retained;
    if !state.latent_history {
        state.resident_identities[slot] = identity;
        if state.resident_history == state.history_rows.len() {
            state.history_rows.rotate_left(1);
            state.resident_identities.rotate_left(1);
        } else {
            state.resident_history += 1;
        }
    }
    Ok(())
}
