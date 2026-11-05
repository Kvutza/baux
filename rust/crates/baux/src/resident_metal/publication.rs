//! Validate a completed GPU decision and publish one pending proposal handle.
//!
//! Publication records the seed, program, radius, predictions, geometry, and
//! realized changed-word counts needed by evaluation and tell. It does not
//! evaluate objective quality, commit weights, or turn a procedural geometry
//! estimate into an exact distance. Handles identify transactions; parameter
//! buffers are still owned and reused by the search state.

use super::*;

fn validate_decision(
    decision: &Decision,
    pool_count: u32,
    perturbation: ProposalKernel,
) -> Result<(), String> {
    if decision.valid == 0
        || decision.index >= pool_count
        || !decision.score.is_finite()
        || !decision.predicted_mean.is_finite()
        || !decision.predicted_standard_error.is_finite()
        || decision.predicted_standard_error < 0.0
        || !decision.incumbent_mean.is_finite()
        || !decision.incumbent_standard_error.is_finite()
        || decision.incumbent_standard_error < 0.0
    {
        return Err(format!(
            "No finite Metal BF16 acquisition candidate: valid={} index={} score={} \
             predicted=({}, {}) incumbent=({}, {})",
            decision.valid,
            decision.index,
            decision.score,
            decision.predicted_mean,
            decision.predicted_standard_error,
            decision.incumbent_mean,
            decision.incumbent_standard_error,
        ));
    }
    if decision.mode != perturbation.shader() {
        return Err("Metal BF16 selected perturbation mode changed".into());
    }
    Ok(())
}

fn pool_descriptions(
    state: &SearchState,
    distances: &[f32],
    decision: &Decision,
    pool_count: usize,
) -> Result<Vec<PoolDescription>, String> {
    let index = decision.index as usize;
    if distances[..pool_count * state.history]
        .iter()
        .any(|distance| !distance.is_finite())
    {
        return Err("Metal BF16 candidate-pool history distance is nonfinite".into());
    }
    let pool = distances
        .chunks_exact(state.history)
        .enumerate()
        .take(pool_count)
        .map(|(candidate, row)| {
            (
                candidate,
                state.procedural_seed(decision.root_seed, candidate),
                state.radius(candidate),
                0.0,
                state.identities[..state.history]
                    .iter()
                    .copied()
                    .zip(row.iter().copied())
                    .collect(),
            )
        })
        .collect::<Vec<PoolDescription>>();
    if pool[index].1 != decision.seed || pool[index].2 != decision.radius {
        return Err("Metal BF16 selected descriptor disagrees with its pool".into());
    }
    Ok(pool)
}

fn block_changes(
    state: &SearchState,
    partials: &[Partial],
    index: usize,
) -> Result<Vec<(u64, f64)>, String> {
    let selected = &partials[index * state.tiles.len()..(index + 1) * state.tiles.len()];
    let mut changes = vec![(0u64, 0.0f64); state.blocks.len()];
    let mut invalid = false;
    let mut any_changed = false;
    for (partial, tile) in selected.iter().zip(&state.tiles) {
        if partial.invalid != 0 || !partial.squared.is_finite() {
            invalid = true;
        }
        if partial.changed != 0 {
            any_changed = true;
        }
        let entry = &mut changes[tile.leaf as usize];
        entry.0 += u64::from(partial.changed);
        entry.1 += f64::from(partial.squared);
    }
    if state.implicit_history {
        let flags = u32::from(invalid) * baux::semantics::proposal::INVALID
            | u32::from(any_changed) * baux::semantics::proposal::CHANGED;
        baux::semantics::proposal::validate(flags, false)?;
    }
    Ok(changes)
}

impl SearchState {
    /// Publish after completion using a timing origin at publication itself.
    /// Staged GPU profiling, when present, supplies ask timings separately;
    /// this wrapper's fallback timer is not end-to-end proposal latency.
    pub fn publish_ask(&mut self) -> Result<Proposals, String> {
        self.publish_start(Instant::now())
    }

    /// Validate the selected descriptor and attach host-side transaction
    /// metadata. Requires completed GPU writes. Rejects invalid selection,
    /// mismatched seed/radius/program, nonfinite distances, and
    /// invalid/unchanged selected latent-mode rows (except DUST's permitted
    /// unchanged row). Aggregates realized per-block changes, then sets
    /// `pending`, advances the handle ID, and marks the search started.
    /// Expected/procedural norms remain estimates even when reported
    /// alongside realized changes; finite metadata is not evidence of a
    /// useful proposal.
    pub fn publish_start(&mut self, start: Instant) -> Result<Proposals, String> {
        let total_ms = start.elapsed().as_secs_f32() * 1000.0;
        let decision = read::<Decision>(&self.decision, 1)[0];
        let pool_count = self.pool_layout.count();
        validate_decision(&decision, pool_count, self.perturbation)?;
        let index = decision.index as usize;
        let distances = read::<f32>(&self.pool_distances, pool_count as usize * self.history);
        let pool = pool_descriptions(self, &distances, &decision, pool_count as usize)?;
        let partials = unsafe {
            std::slice::from_raw_parts(
                self.partials.contents().cast::<Partial>(),
                self.tiles.len() * pool_count as usize,
            )
        };
        let (pool_radii, pool_cosines) = self.read_geometry()?;
        let history_distances = pool[index].4.clone();
        let axis_coordinate = self.axis_candidate(decision.root_seed, index);
        let axis_value = self
            .axis
            .as_ref()
            .zip(axis_coordinate)
            .map(|(axis, coordinate)| axis.value(coordinate));
        let changes = block_changes(self, &partials, index)?;
        let round = Proposals {
            pool_layout: self.pool_layout,
            owner: self.owner,
            id: self.next_id,
            base_id: self.base_id,
            index,
            seed: decision.seed,
            score: decision.score,
            length: decision.radius,
            predicted_mean: decision.predicted_mean,
            predicted_standard_error: decision.predicted_standard_error,
            incumbent_mean: decision.incumbent_mean,
            incumbent_standard_error: decision.incumbent_standard_error,
            neighbors: decision.neighbors,
            incumbent_neighbors: decision.incumbent_neighbors,
            pool_predictions: (0..pool_count as usize)
                .map(|candidate| {
                    (
                        decision.candidate_means[candidate],
                        decision.candidate_standard_errors[candidate],
                        decision.candidate_scores[candidate],
                    )
                })
                .collect(),
            changes,
            history_distances,
            block_scales: self.blocks.iter().map(|b| b.scale).collect(),
            pool,
            pool_radii,
            pool_cosines,
            axis_coordinate,
            axis_value,
            direction_norm: pool_radii[index] * pool_radii[index],
        };
        self.pending = Some(round.clone());
        self.next_id = self.next_id.wrapping_add(1);
        self.started = true;
        if self.profiling && self.last_profile.is_none() {
            self.last_profile = Some(ProposalProfile {
                score_ms: total_ms,
                pick_ms: 0.0,
                materialize_ms: 0.0,
                total_ms,
            });
        }
        Ok(round)
    }

    /// Read pool radii and optional cosine diagnostics in the chosen geometry.
    fn read_geometry(&self) -> Result<([f32; 4], [Option<f32>; 6]), String> {
        let geometry = read::<f32>(&self.pool_geometry, POOL_METRICS * self.tiles.len());
        if geometry.iter().any(|value| !value.is_finite()) {
            return Err("Metal BF16 candidate-pool geometry is nonfinite".into());
        }
        let mut totals = [0.0f64; POOL_METRICS];
        for (metric, total) in totals.iter_mut().enumerate() {
            *total = geometry[metric * self.tiles.len()..(metric + 1) * self.tiles.len()]
                .iter()
                .map(|&value| f64::from(value))
                .sum();
        }
        if totals[..self.pool_layout.count() as usize]
            .iter()
            .any(|&norm| norm < 0.0)
        {
            return Err("Metal BF16 candidate pool has negative squared radius".into());
        }
        let pool_radii = std::array::from_fn(|candidate| totals[candidate].max(0.0).sqrt() as f32);
        #[cfg(not(test))]
        let pool_cosines = [None; 6];
        #[cfg(test)]
        let pool_cosines = {
            let mut pool_cosines = [None; 6];
            for (pair, &(left, right)) in POOL_PAIRS.iter().enumerate() {
                let denominator = f64::from(pool_radii[left]) * f64::from(pool_radii[right]);
                if denominator == 0.0 {
                    continue;
                }
                let cosine = totals[pair + 4] / denominator;
                if !cosine.is_finite() || cosine.abs() > 1.001 {
                    return Err("Metal BF16 candidate-pool cosine is invalid".into());
                }
                pool_cosines[pair] = Some(cosine.clamp(-1.0, 1.0) as f32);
            }
            pool_cosines
        };
        Ok((pool_radii, pool_cosines))
    }
}
