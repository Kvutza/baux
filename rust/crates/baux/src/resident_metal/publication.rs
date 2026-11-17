//! Validate a completed GPU decision and publish one pending proposal handle.

use super::*;
use baux::semantics::optimizer::{AskPlan, RoundId, WorldId};
use std::sync::Arc;

fn validate_decision(
    decision: &Decision,
    pool_count: u32,
    expected_mode: u32,
) -> Result<(), String> {
    if decision.valid == 0
        || decision.index >= pool_count
        || !decision.score.is_finite()
        || !decision.predicted_mean.is_finite()
        || !decision.predicted_standard_error.is_finite()
        || !decision.predicted_standard_error.is_sign_positive()
            && decision.predicted_standard_error < 0.0
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
    if decision.mode != expected_mode {
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
    pub fn publish_ask(self) -> Result<(Self, Proposals), String> {
        self.publish_start(Instant::now())
    }

    pub fn publish_start(mut self, start: Instant) -> Result<(Self, Proposals), String> {
        let total_ms = start.elapsed().as_secs_f32() * 1000.0;
        let decision = read::<Decision>(&self.decision, 1)[0];
        let pool_count = self.pool_layout.count();
        validate_decision(&decision, pool_count, self.sampler.shader())?;
        let index = decision.index as usize;
        let distances = read::<f32>(&self.pool_distances, pool_count as usize * self.history);
        let pool = pool_descriptions(&self, &distances, &decision, pool_count as usize)?;
        let partials = unsafe {
            std::slice::from_raw_parts(
                self.partials.contents().cast::<Partial>(),
                self.tiles.len() * pool_count as usize,
            )
        };
        let (candidate_radii, pool_radii, pool_cosines) = self.read_geometry()?;
        let history_distances = pool[index].4.clone();
        let axis_coordinate = self.axis_candidate(decision.root_seed, index);
        let axis_value = self
            .axis
            .as_ref()
            .zip(axis_coordinate)
            .map(|(axis, coordinate)| axis.value(coordinate));
        let changes = block_changes(&self, partials, index)?;
        let cand_scores = read::<CandidateScore>(&self.candidate_scores, pool_count as usize);
        let any_cand_changed = cand_scores
            .iter()
            .any(|cs| cs.invalid == 0 && cs.changed != 0);
        let pool_predictions = cand_scores
            .iter()
            .map(|cs| {
                if cs.invalid == 0 && (!any_cand_changed || cs.changed != 0) {
                    (cs.mean, cs.standard_error, cs.score)
                } else {
                    (f32::NAN, f32::NAN, f32::NAN)
                }
            })
            .collect();
        let num_selected = (decision.num_selected as usize).clamp(1, pool_count as usize);
        let raw_selected = read::<u32>(&self.selected_indices_gpu, num_selected);
        let mut selected_indices: Vec<usize> = raw_selected
            .into_iter()
            .filter(|&v| v < pool_count)
            .map(|v| v as usize)
            .collect();
        if selected_indices.is_empty() {
            selected_indices.push(index);
        }
        let selected_radius = candidate_radii.get(index).copied().unwrap_or(0.0);
        let device_row = DeviceRow::new(Arc::clone(&self.proposal), None, self.dimensions);
        let sobol_offset = self.sobol_offset;
        let round = Proposals {
            pool_layout: self.pool_layout,
            owner: 0,
            id: self.next_id,
            base_id: self.base_id,
            index,
            selected_indices,
            seed: decision.seed,
            score: decision.score,
            length: decision.radius,
            predicted_mean: decision.predicted_mean,
            predicted_standard_error: decision.predicted_standard_error,
            incumbent_mean: decision.incumbent_mean,
            incumbent_standard_error: decision.incumbent_standard_error,
            neighbors: decision.neighbors,
            incumbent_neighbors: decision.incumbent_neighbors,
            pool_predictions,
            changes,
            history_distances,
            block_scales: self.blocks.iter().map(|b| b.scale).collect(),
            pool,
            candidate_radii,
            pool_radii,
            pool_cosines,
            axis_coordinate,
            axis_value,
            direction_norm: selected_radius * selected_radius,
            sobol_offset,
            device_row,
        };
        if matches!(
            self.sampler,
            manifold::ResidentSampler::Sobol | manifold::ResidentSampler::SobolMasked
        ) {
            self.sobol_offset = self.sobol_offset.wrapping_add(pool_count);
        }
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
        Ok((self, round))
    }

    pub fn plan_evaluation(
        mut self,
        round: &Proposals,
        world: WorldId,
    ) -> Result<(Self, AskPlan), String> {
        self.check_round(round)?;
        if self.base_id != round.base_id || self.evaluation.is_some_and(|planned| planned != world)
        {
            return Err("evaluation plan changed incumbent or world".into());
        }
        self.evaluation = Some(world);
        let plan = AskPlan::new(
            RoundId {
                owner: round.owner,
                proposal: round.id,
                incumbent: round.base_id,
            },
            world,
        );
        Ok((self, plan))
    }

    fn read_geometry(&self) -> Result<(Vec<f32>, [f32; 4], [Option<f32>; 6]), String> {
        let pool_count = self.pool_layout.count() as usize;
        let metric_count = initialization::metric_count(pool_count)?;
        let geometry = read::<f32>(&self.pool_geometry, metric_count * self.tiles.len());
        if geometry[..pool_count * self.tiles.len()]
            .iter()
            .any(|value| !value.is_finite())
        {
            return Err("Metal BF16 candidate-pool geometry is nonfinite".into());
        }
        let mut totals = vec![0.0f64; metric_count];
        for (metric, total) in totals.iter_mut().enumerate() {
            *total = geometry[metric * self.tiles.len()..(metric + 1) * self.tiles.len()]
                .iter()
                .map(|&value| f64::from(value))
                .sum();
        }
        if totals[..pool_count].iter().any(|&norm| norm < 0.0) {
            return Err("Metal BF16 candidate pool has negative squared radius".into());
        }
        let candidate_radii: Vec<f32> = totals[..pool_count]
            .iter()
            .map(|&norm| norm.max(0.0).sqrt() as f32)
            .collect();
        let pool_radii = std::array::from_fn(|c| candidate_radii.get(c).copied().unwrap_or(0.0));
        #[cfg(not(test))]
        let pool_cosines = [None; 6];
        #[cfg(test)]
        let pool_cosines = {
            let dot_base = pool_count.max(4);
            let mut pool_cosines = [None; 6];
            for (pair, &(left, right)) in POOL_PAIRS.iter().enumerate() {
                if left >= pool_count || right >= pool_count {
                    continue;
                }
                let dot = totals[dot_base + pair];
                if !dot.is_finite() {
                    return Err("Metal BF16 candidate-pool geometry is nonfinite".into());
                }
                let denominator = f64::from(pool_radii[left]) * f64::from(pool_radii[right]);
                if denominator == 0.0 {
                    continue;
                }
                let cosine = dot / denominator;
                if !cosine.is_finite() || cosine.abs() > 1.001 {
                    return Err("Metal BF16 candidate-pool cosine is invalid".into());
                }
                pool_cosines[pair] = Some(cosine.clamp(-1.0, 1.0) as f32);
            }
            pool_cosines
        };
        Ok((candidate_radii, pool_radii, pool_cosines))
    }
}
