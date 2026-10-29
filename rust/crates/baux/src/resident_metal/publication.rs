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
    proposal_program: u32,
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
    if decision.program != proposal_program {
        return Err("Metal BF16 selected proposal program changed".into());
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
                if !state.independent_fp16 && state.pool_layout.slots() != 1 && candidate < 2 {
                    0.75
                } else {
                    0.0
                },
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
    if state.analytic_pool() {
        let selected = &partials[index * state.tiles.len()..(index + 1) * state.tiles.len()];
        let invalid = selected
            .iter()
            .any(|partial| partial.invalid != 0 || !partial.squared.is_finite());
        let changed = selected.iter().any(|partial| partial.changed != 0);
        let flags = u32::from(invalid) * baux::semantics::proposal::INVALID
            | u32::from(changed) * baux::semantics::proposal::CHANGED;
        baux::semantics::proposal::validate(flags, false)?;
    }
    let mut changes = vec![(0u64, 0.0f64); state.blocks.len()];
    for (tile_index, tile) in state.tiles.iter().enumerate() {
        let partial = partials[index * state.tiles.len() + tile_index];
        changes[tile.leaf as usize].0 += u64::from(partial.changed);
        changes[tile.leaf as usize].1 += f64::from(partial.squared);
    }
    Ok(changes)
}

fn latent_distances(
    state: &SearchState,
    family: &FamilyHistory,
    index: usize,
    base: usize,
) -> Result<Vec<[f32; FAMILIES]>, String> {
    let norms = state.latent_components(index);
    let mut rows = Vec::with_capacity(state.history);
    for row in 0..state.history {
        let base_row = family.components[base * MAX_HISTORY + row];
        let mut entry = [0.0f32; FAMILIES];
        for group in 0..FAMILIES {
            entry[group] = (base_row[group] + norms[group]).max(0.0);
        }
        rows.push(entry);
    }
    Ok(rows)
}

fn unrealized_components(
    state: &SearchState,
    family: &FamilyHistory,
    partials: &[Partial],
    index: usize,
    base: usize,
) -> Result<Vec<[f32; FAMILIES]>, String> {
    if let Some(neighbors) = &state.neighbors {
        Ok(neighbors.components(index, state.history))
    } else if state.latent_history {
        latent_distances(state, family, index, base)
    } else {
        let mut norms = [0.0; FAMILIES];
        let mut resident = [[0.0; FAMILIES]; 2];
        for (t, tile) in state.tiles.iter().enumerate() {
            let leaf = tile.leaf as usize;
            let g = family.groups[leaf];
            let p = partials[index * state.tiles.len() + t];
            norms[g] += p.squared * family.original[leaf].weight;
            resident[0][g] += p.anchor / family.weights[g];
            resident[1][g] += p.rejected / family.weights[g];
        }
        let mut rows = (0..state.history)
            .map(|i| {
                std::array::from_fn(|g| family.components[base * MAX_HISTORY + i][g] + norms[g])
            })
            .collect::<Vec<_>>();
        for (slot, values) in resident.iter().enumerate().take(state.resident_history) {
            let row = state.identities[..state.history]
                .iter()
                .position(|&id| id == state.resident_identities[slot])
                .ok_or("Family history lost resident identity")?;
            rows[row] = *values;
        }
        if rows.iter().flatten().any(|v| !v.is_finite() || *v < 0.0) {
            return Err("Invalid family distance component".into());
        }
        Ok(rows)
    }
}

fn family_distances(
    state: &SearchState,
    distances: &[f32],
    partials: &[Partial],
    decision: &Decision,
    index: usize,
) -> Result<(Option<BoundReport>, Option<Vec<[f32; FAMILIES]>>), String> {
    if state.realized_history() && state.history > state.resident_history && state.family.is_some()
    {
        let values = read::<f32>(&state.pool_family_distances, 4 * state.history * FAMILIES);
        let bounds = Some(state.inspect_bounds(distances, &values, decision.root_seed));
        let family_distances = Some(
            values[index * state.history * FAMILIES..(index + 1) * state.history * FAMILIES]
                .chunks_exact(FAMILIES)
                .map(|row| row.try_into().unwrap())
                .collect(),
        );
        Ok((bounds, family_distances))
    } else if let Some(family) = &state.family {
        let base = state.identities[..state.history]
            .iter()
            .position(|&id| id == state.base_id)
            .ok_or("Family history lost its incumbent")?;
        let rows = unrealized_components(state, family, partials, index, base)?;
        Ok((None, Some(rows)))
    } else {
        Ok((None, None))
    }
}

impl SearchState {
    /// Publish after completion using a timing origin at publication itself.
    /// Staged GPU profiling, when present, supplies ask timings separately;
    /// this wrapper's fallback timer is not end-to-end proposal latency.
    pub(super) fn publish_ask(&mut self) -> Result<Proposals, String> {
        self.publish_start(Instant::now())
    }

    /// Validate the selected descriptor and attach host-side transaction
    /// metadata. Requires completed GPU writes. Rejects invalid selection,
    /// mismatched seed/radius/program, nonfinite distances, and
    /// invalid/unchanged selected latent-mode rows (except DUST's permitted
    /// unchanged row). Aggregates realized per-block changes and records
    /// available family geometry, then sets `pending`, advances the handle
    /// ID, and marks the search started. Expected/procedural norms remain
    /// estimates even when reported alongside realized changes; finite
    /// metadata is not evidence of a useful proposal.
    pub(super) fn publish_start(&mut self, start: Instant) -> Result<Proposals, String> {
        let total_ms = start.elapsed().as_secs_f32() * 1000.0;
        let decision = read::<Decision>(&self.decision, 1)[0];
        let pool_count = self.pool_layout.count();
        validate_decision(
            &decision,
            pool_count,
            self.perturbation,
            self.proposal_program(),
        )?;
        let index = decision.index as usize;
        let distances = read::<f32>(&self.pool_distances, 4 * self.history);
        let pool = pool_descriptions(self, &distances, &decision, pool_count as usize)?;
        let partials = read::<Partial>(&self.partials, self.tiles.len() * 4);
        let (pool_radii, pool_cosines, reference_cosines) = self.read_geometry()?;
        let history_distances = pool[index].4.clone();
        let axis_coordinate = self.axis_candidate(decision.root_seed, index);
        let axis_value = self
            .axis
            .as_ref()
            .zip(axis_coordinate)
            .map(|(axis, coordinate)| axis.value(coordinate));
        let changes = block_changes(self, &partials, index)?;
        let (bounds, family_distances) =
            family_distances(self, &distances, &partials, &decision, index)?;
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
            pool_predictions: std::array::from_fn(|candidate| {
                (
                    decision.candidate_means[candidate],
                    decision.candidate_standard_errors[candidate],
                    decision.candidate_scores[candidate],
                )
            }),
            changes,
            history_distances,
            family_distances,
            block_scales: self.blocks.iter().map(|b| b.scale).collect(),
            pool,
            pool_radii,
            pool_cosines,
            reference_cosines,
            bounds,
            axis_coordinate,
            axis_value,
            basis_seed: decision.basis_seed,
            direction_norm: self.latent_norm(index),
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
    /// Latent mode returns square roots of `latent_norm` with no cosines, NOT
    /// measured full-row radii. Other modes reduce GPU partials, validate
    /// finite nonnegative norms, and compute cosine checks only in test
    /// builds.
    fn read_geometry(&self) -> Result<([f32; 4], [Option<f32>; 6], [Option<f32>; 4]), String> {
        if self.analytic_pool() {
            return Ok((
                std::array::from_fn(|candidate| self.latent_norm(candidate).sqrt()),
                [None; 6],
                [None; 4],
            ));
        }
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
        let (pool_cosines, reference_cosines) = ([None; 6], [None; 4]);
        #[cfg(test)]
        let (pool_cosines, reference_cosines) = {
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
            if totals[10] < 0.0 {
                return Err("Metal BF16 reference has negative squared radius".into());
            }
            let reference_radius = totals[10].sqrt();
            let mut reference_cosines = [None; 4];
            for candidate in 0..4 {
                let denominator = f64::from(pool_radii[candidate]) * reference_radius;
                if denominator == 0.0 {
                    continue;
                }
                let cosine = totals[candidate + 11] / denominator;
                if !cosine.is_finite() || cosine.abs() > 1.001 {
                    return Err("Metal BF16 candidate-reference cosine is invalid".into());
                }
                reference_cosines[candidate] = Some(cosine.clamp(-1.0, 1.0) as f32);
            }
            (pool_cosines, reference_cosines)
        };
        Ok((pool_radii, pool_cosines, reference_cosines))
    }
}
