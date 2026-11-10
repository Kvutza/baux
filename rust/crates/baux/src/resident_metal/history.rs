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

impl SearchState {
    #[allow(clippy::too_many_arguments)]
    /// Wrap scalar reward/error metadata as an objective observation and
    /// delegate the already-decided acceptance, history retention, and
    /// radius adaptation. `adapt_region` controls controller work, not
    /// whether weights may commit.
    pub fn tell_absolute(
        &mut self,
        round: &Proposals,
        value: f32,
        variance: f32,
        incumbent_value: f32,
        incumbent_variance: f32,
        accept: bool,
        trust_incumbent: Option<f32>,
        trust_outcome: Option<TrustRegionOutcome>,
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
    pub fn absolute_objectives(
        &mut self,
        round: &Proposals,
        observation: crate::objective_observation::ObjectiveObservation,
        incumbent_value: f32,
        incumbent_variance: f32,
        accept: bool,
        trust_incumbent: Option<f32>,
        trust_outcome: Option<TrustRegionOutcome>,
        adapt_region: bool,
    ) -> Result<(), String> {
        let mut mutated = false;
        let result = (|| {
            let crate::objective_observation::ObjectiveEstimate {
                mean: value,
                variance,
            } = observation.control();
            self.tell_profile = None;
            self.check_unleased()?;
            self.check_round(round)?;
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
            record_resident(self, &command, accept, slot, identity, commit.retained())?;
            self.record_axis(round, target_slot, accept);
            self.record_observation(
                identity,
                target_slot,
                (value, variance),
                observation,
                implicit_distances,
                accept,
            )?;
            if accept || !self.best.is_finite() {
                self.best = if accept { value } else { incumbent_value };
                self.best_variance = if accept { variance } else { incumbent_variance };
            }
            let update = self.adapt_observation(
                observation,
                value,
                trust_incumbent,
                trust_outcome,
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

    /// If accepted, promote the materialized trial.
    /// Swaps base/proposal ownership after GPU completion. Returns whether that
    /// zero-copy swap occurred, so archiving does not recycle a live incumbent.
    /// Rejection leaves weights unchanged.
    fn commit_weights(
        &mut self,
        _round: &Proposals,
        accept: bool,
        _command: &CommandBufferRef,
    ) -> Result<bool, String> {
        if accept {
            std::mem::swap(&mut self.base, &mut self.proposal);
        }
        Ok(accept)
    }

    /// Rotate resident allocations without reusing the incumbent as scratch.
    /// Latent mode retains only an incumbent alias; other modes archive the
    /// trial row and recycle the displaced allocation. May allocate replacement
    /// scratch when an alias would otherwise corrupt retained/current weights.
    fn archive_proposal(&mut self, slot: usize, zero_copy: bool) {
        if self.history_rows.is_empty() {
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
    /// Requires one distance for every retained observation; missing identities
    /// are errors. An optional scalar search axis is removed to leave
    /// tensor-only components.
    fn observation_distances(
        &self,
        round: &Proposals,
    ) -> Result<Option<Vec<(usize, f32)>>, String> {
        if round.history_distances.len() != self.history {
            return Err("Metal implicit history distance row has the wrong length".into());
        }
        let distances = round
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
                            (distance - self.axis_distance(coordinate, axis.history[previous]))
                                .max(0.0)
                        } else {
                            distance
                        };
                        (previous, tensor_distance)
                    })
                    .ok_or_else(|| {
                        "Metal implicit history distance has an unknown identity".to_string()
                    })
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Some(distances))
    }
    /// Write the trial's reward/error, identity, objective vector, and
    /// geometry. Updates symmetric distance rows and fits ENN after warm-up if
    /// enabled. Explicit storage advances its FIFO. Assumes slot/distance
    /// invariants established by proposal publication and the tell caller.
    pub(super) fn record_observation(
        &mut self,
        identity: i64,
        slot: usize,
        (value, variance): (f32, f32),
        observation: crate::objective_observation::ObjectiveObservation,
        distances: Option<Vec<(usize, f32)>>,
        accept: bool,
    ) -> Result<(), String> {
        if self.history_rows.is_empty() && !self.implicit_history {
            if accept || !self.best.is_finite() {
                self.outcomes[0] = value;
                self.variances[0] = variance;
                self.identities[0] = identity;
            }
            self.history = 1;
            self.objective_history.record(identity, observation, accept);
            return Ok(());
        }
        if self.implicit_history {
            let next = slot;
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
            self.objective_history
                .record_slot(next, identity, observation, accept);
            if self.fit_params && self.history >= self.initial_observations {
                self.fit_enn()?;
            }
        } else {
            self.outcomes[slot] = value;
            self.variances[slot] = variance;
            self.identities[slot] = identity;
            for &(previous, distance) in distances.as_ref().unwrap() {
                self.pairwise_distances[slot * MAX_HISTORY + previous] = distance;
                self.pairwise_distances[previous * MAX_HISTORY + slot] = distance;
            }
            self.pairwise_distances[slot * MAX_HISTORY + slot] = 0.0;
            self.objective_history.record(identity, observation, accept);
            if self.history == self.history_rows.len() {
                let h = self.history;
                if h > 1 {
                    let mut rotated = vec![0.0_f32; h * h];
                    for r in 0..h {
                        for c in 0..h {
                            rotated[r * h + c] = self.pairwise_distances
                                [((r + 1) % h) * MAX_HISTORY + ((c + 1) % h)];
                        }
                    }
                    for r in 0..h {
                        for c in 0..h {
                            self.pairwise_distances[r * MAX_HISTORY + c] = rotated[r * h + c];
                        }
                    }
                }
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
    /// Priority is multi-region (MORBO) control, then ordinary TuRBO
    /// trust-region updates. A trust outcome requires its incumbent baseline.
    /// With adaptation disabled, only the non-MORBO observation counter
    /// advances.
    fn adapt_observation(
        &mut self,
        observation: crate::objective_observation::ObjectiveObservation,
        value: f32,
        trust_incumbent: Option<f32>,
        trust_outcome: Option<TrustRegionOutcome>,
        adapt_region: bool,
    ) -> Result<(), String> {
        let update = if self.has_morbo() {
            self.update_morbo(observation)
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
            self.check_unleased()?;
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
            record_resident(self, &command, accept, slot, identity, commit.retained())?;
            store_host(
                self,
                round,
                target_slot,
                identity,
                (value, variance),
                observation,
                accept,
                &implicit_distances,
            )?;
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

    pub fn pairwise_distance(&self, row: usize, column: usize) -> f64 {
        f64::from(self.total_distance(row, column))
    }
}

#[allow(clippy::too_many_arguments)]
fn store_host(
    state: &mut SearchState,
    round: &Proposals,
    target_slot: usize,
    identity: i64,
    (value, variance): (f32, f32),
    observation: crate::objective_observation::ObjectiveObservation,
    accept: bool,
    implicit_distances: &Option<Vec<(usize, f32)>>,
) -> Result<(), String> {
    state.store_axis(round, target_slot, accept);
    state.outcomes[target_slot] = value;
    state.variances[target_slot] = variance;
    state.identities[target_slot] = identity;
    if let Some(distances) = implicit_distances {
        for &(previous, distance) in distances {
            state.pairwise_distances[target_slot * MAX_HISTORY + previous] = distance;
            state.pairwise_distances[previous * MAX_HISTORY + target_slot] = distance;
        }
        state.pairwise_distances[target_slot * MAX_HISTORY + target_slot] = 0.0;
    }
    if state.history < MAX_HISTORY && target_slot == state.history {
        state.history += 1;
    }
    state
        .objective_history
        .record_slot(target_slot, identity, observation, accept);
    Ok(())
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

pub(super) fn choose_slot(state: &SearchState) -> Result<usize, String> {
    if state.implicit_history {
        baux::semantics::neighbor::retention(&state.identities[..state.history], state.base_id)
            .map_err(str::to_owned)
    } else if state.resident_history == state.history_rows.len() {
        Ok(0)
    } else {
        Ok(state.resident_history)
    }
}

fn record_resident(
    state: &mut SearchState,
    _command: &CommandBufferRef,
    _accept: bool,
    slot: usize,
    identity: i64,
    retained: i64,
) -> Result<(), String> {
    if state.profiling {
        state.tell_profile = Some(TellProfile {
            reference_ms: 0.0,
            history_copy_ms: 0.0,
            total_ms: 0.0,
        });
    }
    state.observation = identity;
    state.base_id = retained;
    if state.history_rows.is_empty() {
        state.resident_identities[0] = retained;
        return Ok(());
    }
    state.resident_identities[slot] = identity;
    if state.resident_history == state.history_rows.len() {
        state.history_rows.rotate_left(1);
        state.resident_identities.rotate_left(1);
    } else {
        state.resident_history += 1;
    }
    Ok(())
}
