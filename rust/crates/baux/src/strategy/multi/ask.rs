//! Issue regional proposals while respecting per-slot capacity.

use baux::semantics::key::Key;
use rand::RngCore;
use rand::seq::SliceRandom;

use super::super::selection::{CandidateSegment, pool_region, pool_scores};
use super::*;
use crate::candidates::{SobolStep, build_pool};
use crate::optimizer::multi_tr::MultiTrustRegionState;
use crate::posterior::{RemappedPool, materialize_selected};

fn restart_slot(
    multi_tr: &mut MultiTrustRegionState,
    region: usize,
    rng: &mut dyn RngCore,
) -> Result<(), ENNError> {
    let center = multi_tr
        .choose_center(region, rng)
        .map_err(|error| ENNError::InvalidParameter(error.to_string()))?;
    multi_tr
        .restart_region(region, &center.view())
        .map_err(|error| ENNError::InvalidParameter(error.to_string()))
}

pub(super) fn restart_inactive(
    multi_tr: &mut MultiTrustRegionState,
    rng: &mut dyn RngCore,
) -> Result<(), ENNError> {
    for region in 0..multi_tr.num_regions() {
        if !multi_tr.active_mask[region] {
            restart_slot(multi_tr, region, rng)?;
        }
    }
    Ok(())
}

struct RegionSpec {
    region: usize,
    arms: usize,
    num_candidates: usize,
    key: Key,
    sobol: Option<SobolStep>,
    center: Array1<f64>,
    lower: Array1<f64>,
    upper: Array1<f64>,
}

fn collect_specs(
    optimizer: &mut BayesOptimizer,
    batches: Vec<crate::optimizer::RegionBatch>,
    ls_ref: Option<ArrayView1<f64>>,
    rng: &mut dyn RngCore,
) -> Result<(Vec<RegionSpec>, Option<Array2<f64>>), ENNError> {
    let num_dim = optimizer.num_dim();
    let config = optimizer.proposals.config().clone();
    let pending_points = match optimizer.regions.as_ref() {
        TrustRegionState::Multi {
            pending: Some(proposal),
            ..
        } => {
            let rows = proposal
                .outcomes
                .iter()
                .enumerate()
                .filter_map(|(row, outcome)| {
                    matches!(outcome, EvaluationStatus::Pending).then_some(row)
                })
                .collect::<Vec<_>>();
            Some(proposal.x.select(Axis(0), &rows))
        }
        _ => None,
    };

    let mut specs = Vec::with_capacity(batches.len());
    for batch in batches {
        let (center, (lower, upper)) = match optimizer.regions.as_ref() {
            TrustRegionState::Multi { state, .. } => (
                state.centers.row(batch.region).to_owned(),
                state.compute_bounds(batch.region, ls_ref.as_ref()),
            ),
            _ => unreachable!("multi-region ask requires TrustRegionState::Multi"),
        };
        let num_candidates = config.num_candidates(num_dim, batch.len);
        optimizer.telemetry.num_candidates += num_candidates;
        let key = Key::new(rng.next_u64());
        let sobol = optimizer.proposals.advance_sobol(num_candidates)?;
        specs.push(RegionSpec {
            region: batch.region,
            arms: batch.len,
            num_candidates,
            key,
            sobol,
            center,
            lower,
            upper,
        });
    }
    Ok((specs, pending_points))
}

fn select_regions(
    optimizer: &BayesOptimizer,
    specs: &[RegionSpec],
    pending_points: Option<&Array2<f64>>,
    rng: &mut dyn RngCore,
) -> Result<(Array2<f64>, Vec<CandidateSegment>), ENNError> {
    let TrustRegionState::Multi { state, .. } = optimizer.regions.as_ref() else {
        unreachable!("select_regions requires TrustRegionState::Multi");
    };
    let config = optimizer.proposals.config();
    let mut blocks = Vec::with_capacity(specs.len());
    let mut segments = Vec::with_capacity(specs.len());
    let mut offset = 0usize;

    for spec in specs {
        let pool = build_pool(
            spec.key,
            spec.center.as_slice().expect("center contiguous"),
            spec.lower.as_slice().expect("lower contiguous"),
            spec.upper.as_slice().expect("upper contiguous"),
            spec.num_candidates,
            config.candidate_rv,
            config.coord_system,
            &config.tied_dims,
            false,
            config.num_pert,
            spec.sobol.as_ref(),
        )?;
        let selected = match pending_points {
            Some(points) => {
                let remapped = RemappedPool::exclude_pending(&pool, points, spec.arms)?;
                pool_region(optimizer, state, spec.region, &remapped, spec.arms, rng)?
            }
            None => pool_region(optimizer, state, spec.region, &pool, spec.arms, rng)?,
        };
        let end = offset + selected.nrows();
        segments.push(CandidateSegment {
            start: offset,
            end,
            arms: selected.nrows(),
            region: spec.region,
        });
        offset = end;
        blocks.push(selected);
    }

    let views = blocks.iter().map(|b| b.view()).collect::<Vec<_>>();
    let combined = ndarray::concatenate(Axis(0), &views)
        .map_err(|error| ENNError::InvalidParameter(error.to_string()))?;
    Ok((combined, segments))
}

fn joint_regions(
    optimizer: &BayesOptimizer,
    specs: &[RegionSpec],
    pending_points: Option<&Array2<f64>>,
    num_arms: usize,
    rng: &mut dyn RngCore,
) -> Result<(Array2<f64>, Vec<CandidateSegment>), ENNError> {
    let TrustRegionState::Multi { state, .. } = optimizer.regions.as_ref() else {
        unreachable!("joint_regions requires TrustRegionState::Multi");
    };
    let config = optimizer.proposals.config();
    let mut scored = Vec::new();

    for (spec_idx, spec) in specs.iter().enumerate() {
        let pool = build_pool(
            spec.key,
            spec.center.as_slice().expect("center contiguous"),
            spec.lower.as_slice().expect("lower contiguous"),
            spec.upper.as_slice().expect("upper contiguous"),
            spec.num_candidates,
            config.candidate_rv,
            config.coord_system,
            &config.tied_dims,
            false,
            config.num_pert,
            spec.sobol.as_ref(),
        )?;
        let scores = match pending_points {
            Some(points) => {
                let remapped = RemappedPool::exclude_pending(&pool, points, spec.arms)?;
                pool_scores(optimizer, state, spec.region, &remapped, rng)?
            }
            None => pool_scores(optimizer, state, spec.region, &pool, rng)?,
        };
        for (cand, score) in scores.into_iter().enumerate() {
            scored.push((score, spec_idx, cand));
        }
    }

    scored.shuffle(rng);
    scored.sort_by(|left, right| {
        use baux::semantics::optimizer::surrogate::BetterScore;
        BetterScore(right.0).cmp(&BetterScore(left.0))
    });
    let chosen = &scored[..num_arms.min(scored.len())];

    let mut blocks = Vec::new();
    let mut winners = Vec::new();
    let mut offset = 0usize;
    for (spec_idx, spec) in specs.iter().enumerate() {
        let group: Vec<usize> = chosen
            .iter()
            .filter(|&&(_, idx, _)| idx == spec_idx)
            .map(|&(_, _, cand)| cand)
            .collect();
        if group.is_empty() {
            continue;
        }
        let pool = build_pool(
            spec.key,
            spec.center.as_slice().expect("center contiguous"),
            spec.lower.as_slice().expect("lower contiguous"),
            spec.upper.as_slice().expect("upper contiguous"),
            spec.num_candidates,
            config.candidate_rv,
            config.coord_system,
            &config.tied_dims,
            false,
            config.num_pert,
            spec.sobol.as_ref(),
        )?;
        let block = match pending_points {
            Some(points) => {
                let remapped = RemappedPool::exclude_pending(&pool, points, spec.arms)?;
                materialize_selected(&remapped, &group)
            }
            None => materialize_selected(&pool, &group),
        };
        let end = offset + block.nrows();
        winners.push(CandidateSegment {
            start: offset,
            end,
            arms: block.nrows(),
            region: spec.region,
        });
        offset = end;
        blocks.push(block);
    }

    let views = blocks.iter().map(|b| b.view()).collect::<Vec<_>>();
    let combined = ndarray::concatenate(Axis(0), &views)
        .map_err(|error| ENNError::InvalidParameter(error.to_string()))?;
    Ok((combined, winners))
}

#[cfg(test)]
fn exclude_pending(
    candidates: Array2<f64>,
    pending: &Array2<f64>,
    arms: usize,
) -> Result<Array2<f64>, ENNError> {
    let keep = candidates
        .rows()
        .into_iter()
        .enumerate()
        .filter_map(|(row, candidate)| {
            pending
                .rows()
                .into_iter()
                .all(|point| point != candidate)
                .then_some(row)
        })
        .collect::<Vec<_>>();
    if keep.len() < arms {
        return Err(ENNError::InvalidParameter(
            "candidate pool has too few points distinct from pending evaluations".into(),
        ));
    }
    Ok(candidates.select(Axis(0), &keep))
}

fn append_pending(
    state: &MultiTrustRegionState,
    existing: &PendingProposal,
    next: PendingProposal,
) -> Result<PendingProposal, ENNError> {
    let old_arms = existing.arms.as_ref().ok_or_else(|| {
        ENNError::InvalidParameter("regional ask cannot extend warm-up work".into())
    })?;
    let new_arms = next.arms.ok_or_else(|| {
        ENNError::InvalidParameter("regional ask requires region provenance".into())
    })?;
    // Retain resolved rows only until their region has received controller
    // credit. That keeps a slow region's group intact while fast regions issue
    // subsequent groups without double counting the old results.
    let keep = old_arms
        .iter()
        .enumerate()
        .filter_map(|(row, arm)| {
            (matches!(existing.outcomes[row], EvaluationStatus::Pending)
                || state.validate(arm.ticket).is_ok())
            .then_some(row)
        })
        .collect::<Vec<_>>();
    let old_x = existing.x.select(Axis(0), &keep);
    let x = ndarray::concatenate(Axis(0), &[old_x.view(), next.x.view()])
        .map_err(|error| ENNError::InvalidParameter(error.to_string()))?;
    let mut ids = keep
        .iter()
        .map(|&row| existing.ids[row])
        .collect::<Vec<_>>();
    ids.extend(next.ids);
    let mut outcomes = keep
        .iter()
        .map(|&row| existing.outcomes[row])
        .collect::<Vec<_>>();
    outcomes.extend(next.outcomes);
    let mut arms = keep.iter().map(|&row| old_arms[row]).collect::<Vec<_>>();
    arms.extend(new_arms);
    Ok(PendingProposal {
        ids,
        arms: Some(arms),
        x,
        outcomes,
        ordered_tell_allowed: false,
    })
}

fn allocate_idle(
    state: &mut MultiTrustRegionState,
    num_arms: usize,
    rng: &mut dyn RngCore,
) -> Result<Vec<crate::optimizer::RegionBatch>, ENNError> {
    if num_arms == 0 {
        return Err(ENNError::InvalidParameter(
            "regional allocation inputs are invalid".into(),
        ));
    }
    for slot in 0..state.num_regions() {
        if state.is_pending(slot) == Some(false) && !state.active_mask[slot] {
            restart_slot(state, slot, rng)?;
        }
    }
    if state.allocation_policy() == crate::optimizer::AllocationPolicy::JointAcquisition {
        let mut batches = Vec::new();
        let mut start = 0;
        for slot in 0..state.num_regions() {
            if state.active_mask[slot] && state.is_pending(slot) == Some(false) {
                batches.push(crate::optimizer::RegionBatch {
                    region: slot,
                    start,
                    len: num_arms,
                });
                start += num_arms;
            }
        }
        if batches.is_empty() {
            return Err(ENNError::InvalidParameter(
                "all regional slots have outstanding evaluations".into(),
            ));
        }
        return Ok(batches);
    }
    state
        .allocate(num_arms)
        .map_err(|error| ENNError::InvalidParameter(error.to_string()))
}

fn ask_warmup(
    optimizer: BayesOptimizer,
    num_arms: usize,
    rng: &mut dyn RngCore,
) -> Result<(BayesOptimizer, Option<Array2<f64>>), ENNError> {
    if matches!(
        optimizer.regions.as_ref(),
        TrustRegionState::Multi {
            pending: Some(PendingProposal { arms: None, .. }),
            ..
        }
    ) {
        return Err(ENNError::InvalidParameter(
            "finish the outstanding warm-up batch before asking again".into(),
        ));
    }
    if !optimizer.warmup.in_init() {
        return Ok((optimizer, None));
    }
    let state = InitStrategyState::new(optimizer.warmup.kind, optimizer.warmup.budget);
    let (optimizer, x) = ask_init(&state, optimizer, num_arms, rng)?;
    let (mut optimizer, ids) = optimizer.issue_ids(x.nrows())?;
    if let TrustRegionState::Multi { pending, .. } = optimizer.trust_mut() {
        *pending = Some(PendingProposal {
            outcomes: vec![EvaluationStatus::Pending; x.nrows()],
            ids,
            arms: None,
            x: x.clone(),
            ordered_tell_allowed: true,
        });
    }
    Ok((optimizer, Some(x)))
}

fn record_proposal(
    optimizer: BayesOptimizer,
    segments: &[CandidateSegment],
    selected: &Array2<f64>,
) -> Result<BayesOptimizer, ENNError> {
    let (mut optimizer, ids) =
        optimizer.issue_ids(segments.iter().map(|segment| segment.arms).sum())?;
    let TrustRegionState::Multi { state, pending } = optimizer.trust_mut() else {
        unreachable!("ask_multi requires TrustRegionState::Multi");
    };
    let mut tickets = Vec::with_capacity(segments.len());
    for segment in segments {
        match state.issue(segment.region) {
            Ok(ticket) => tickets.push(ticket),
            Err(error) => {
                for ticket in tickets {
                    let _ = state.retire(ticket);
                }
                return Err(ENNError::InvalidParameter(error.to_string()));
            }
        }
    }
    let arms = segments
        .iter()
        .zip(tickets.iter().copied())
        .flat_map(|(segment, ticket)| {
            std::iter::repeat_n(
                crate::optimizer::RegionProvenance {
                    slot: segment.region,
                    generation: ticket.region.generation,
                    ticket,
                },
                segment.arms,
            )
        })
        .collect::<Vec<_>>();
    let next = PendingProposal {
        ids,
        outcomes: vec![EvaluationStatus::Pending; arms.len()],
        arms: Some(arms),
        x: selected.clone(),
        ordered_tell_allowed: true,
    };
    if let Some(existing) = pending.as_ref() {
        match append_pending(state, existing, next) {
            Ok(combined) => *pending = Some(combined),
            Err(error) => {
                for ticket in tickets {
                    let _ = state.retire(ticket);
                }
                return Err(error);
            }
        }
    } else {
        *pending = Some(next);
    }
    Ok(optimizer)
}

pub(in crate::strategy) fn ask_multi(
    mut optimizer: BayesOptimizer,
    num_arms: usize,
    rng: &mut dyn RngCore,
) -> Result<(BayesOptimizer, Array2<f64>), ENNError> {
    let (next_opt, warmup_x) = ask_warmup(optimizer, num_arms, rng)?;
    optimizer = next_opt;
    if let Some(x) = warmup_x {
        return Ok((optimizer, x));
    }

    let batches = match optimizer.trust_mut() {
        TrustRegionState::Multi { state, .. } => allocate_idle(state, num_arms, rng)?,
        _ => unreachable!("ask_multi requires TrustRegionState::Multi"),
    };

    if let Some(models) = optimizer.local_models.take() {
        let TrustRegionState::Multi { state, .. } = optimizer.regions.as_ref() else {
            unreachable!("ask_multi requires TrustRegionState::Multi");
        };
        optimizer.local_models = Some(models.sync(state, rng)?);
    }

    let lengthscales = optimizer.surrogate().and_then(|s| s.lengthscales());
    let ls_ref: Option<ArrayView1<f64>> = lengthscales.as_ref().map(|ls| ls.view());
    optimizer.telemetry.num_candidates = 0;
    optimizer.telemetry.dt_sel = 0.0;
    let (specs, pending_points) = collect_specs(&mut optimizer, batches, ls_ref, rng)?;

    if specs.is_empty() {
        return Err(ENNError::InvalidParameter(
            "experimental strategy produced no candidates".to_string(),
        ));
    }

    let is_joint = matches!(
        optimizer.regions.as_ref(),
        TrustRegionState::Multi { state, .. }
            if state.allocation_policy() == crate::optimizer::AllocationPolicy::JointAcquisition
    );
    let start = std::time::Instant::now();
    let (selected, segments) = if is_joint {
        joint_regions(&optimizer, &specs, pending_points.as_ref(), num_arms, rng)?
    } else {
        select_regions(&optimizer, &specs, pending_points.as_ref(), rng)?
    };
    optimizer.telemetry.dt_sel += start.elapsed().as_secs_f64();

    optimizer = record_proposal(optimizer, &segments, &selected)?;
    Ok((optimizer, selected))
}

#[cfg(test)]
mod tests {
    use ndarray::array;

    use super::exclude_pending;

    #[test]
    fn pending_exclusion() {
        let candidates = array![[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]];
        let pending = array![[0.3, 0.4]];
        let eligible = exclude_pending(candidates.clone(), &pending, 2).unwrap();
        assert_eq!(eligible, array![[0.1, 0.2], [0.5, 0.6]]);
        assert!(exclude_pending(candidates, &pending, 3).is_err());
    }
}
