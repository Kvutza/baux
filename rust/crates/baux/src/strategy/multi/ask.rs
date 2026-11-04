//! Issue regional proposals while respecting per-slot capacity.

use super::*;

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

fn build_segments(
    optimizer: &mut BayesOptimizer,
    batches: Vec<crate::optimizer::RegionBatch>,
    ls_ref: Option<ArrayView1<f64>>,
    rng: &mut dyn RngCore,
) -> Result<(Vec<Array2<f64>>, Vec<CandidateSegment>), ENNError> {
    let num_dim = optimizer.num_dim();
    let config = optimizer.proposals.config().clone();
    let mut blocks = Vec::with_capacity(batches.len());
    let mut segments = Vec::with_capacity(batches.len());
    let mut offset = 0;
    let pending_points = match &optimizer.regions {
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

    for batch in batches {
        let (x_center, (lower_1d, upper_1d)) = match &optimizer.regions {
            TrustRegionState::Multi { state, .. } => (
                state.centers.row(batch.region).to_owned(),
                state.compute_bounds(batch.region, ls_ref.as_ref()),
            ),
            _ => unreachable!("multi-region ask requires TrustRegionState::Multi"),
        };
        let num_candidates = config.num_candidates(num_dim, batch.len);
        optimizer.telemetry.num_candidates += num_candidates;

        let generated = perturb_candidates(
            &x_center.view(),
            &lower_1d,
            &upper_1d,
            num_candidates,
            config.candidate_rv,
            config.coord_system,
            &config.tied_dims,
            false,
            config.raasp_fast,
            rng,
            optimizer.proposals.sobol_mut(),
            config.num_pert,
        )?;
        let x_cand = match pending_points.as_ref() {
            Some(points) => exclude_pending(generated, points, batch.len)?,
            None => generated,
        };
        let end = offset + x_cand.nrows();
        segments.push(CandidateSegment {
            start: offset,
            end,
            arms: batch.len,
            region: batch.region,
        });
        offset = end;
        blocks.push(x_cand);
    }
    Ok((blocks, segments))
}

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

fn append_pending(existing: &mut PendingProposal, next: PendingProposal) -> Result<(), ENNError> {
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
                || !existing.settled_regions[arm.slot])
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
    let mut settled_regions = existing.settled_regions.clone();
    for arm in &new_arms {
        settled_regions[arm.slot] = false;
    }
    arms.extend(new_arms);
    *existing = PendingProposal {
        ids,
        settled_regions,
        arms: Some(arms),
        x,
        outcomes,
        ordered_tell_allowed: false,
    };
    Ok(())
}

fn allocate_idle(
    state: &mut MultiTrustRegionState,
    pending: Option<&PendingProposal>,
    num_arms: usize,
    rng: &mut dyn RngCore,
) -> Result<Vec<crate::optimizer::RegionBatch>, ENNError> {
    let mut busy = vec![false; state.num_regions()];
    if let Some(proposal) = pending {
        let arms = proposal
            .arms
            .as_ref()
            .expect("regional ask cannot overlap warm-up work");
        for (arm, outcome) in arms.iter().zip(&proposal.outcomes) {
            if matches!(outcome, EvaluationStatus::Pending) {
                busy[arm.slot] = true;
            }
        }
    }
    for (slot, &is_busy) in busy.iter().enumerate() {
        if !is_busy && !state.active_mask[slot] {
            restart_slot(state, slot, rng)?;
        }
    }
    let active = state.active_mask.clone();
    for (slot, &is_busy) in busy.iter().enumerate() {
        if is_busy {
            state.active_mask[slot] = false;
        }
    }
    let result = if state.active_count() == 0 {
        Err(ENNError::InvalidParameter(
            "all regional slots have outstanding evaluations".into(),
        ))
    } else {
        state
            .allocate(num_arms)
            .map_err(|error| ENNError::InvalidParameter(error.to_string()))
    };
    state.active_mask = active;
    result
}

pub(in crate::strategy) fn ask_multi(
    optimizer: &mut BayesOptimizer,
    num_arms: usize,
    rng: &mut dyn RngCore,
) -> Result<Array2<f64>, ENNError> {
    if matches!(
        &optimizer.regions,
        TrustRegionState::Multi {
            pending: Some(PendingProposal { arms: None, .. }),
            ..
        }
    ) {
        return Err(ENNError::InvalidParameter(
            "finish the outstanding warm-up batch before asking again".into(),
        ));
    }
    if optimizer.warmup.in_init() {
        let state = InitStrategyState::new(optimizer.warmup.kind, optimizer.warmup.budget);
        let x = ask_init(&state, optimizer, num_arms, rng)?;
        let ids = issue_ids(x.nrows())?;
        if let TrustRegionState::Multi { pending, .. } = &mut optimizer.regions {
            *pending = Some(PendingProposal {
                outcomes: vec![EvaluationStatus::Pending; x.nrows()],
                ids,
                arms: None,
                settled_regions: Vec::new(),
                x: x.clone(),
                ordered_tell_allowed: true,
            });
        }
        return Ok(x);
    }

    let batches = match &mut optimizer.regions {
        TrustRegionState::Multi { state, pending } => {
            allocate_idle(state, pending.as_ref(), num_arms, rng)?
        }
        _ => unreachable!("ask_multi requires TrustRegionState::Multi"),
    };

    if let (Some(models), TrustRegionState::Multi { state, .. }) =
        (optimizer.local_models.as_mut(), &optimizer.regions)
    {
        if let Err(error) = models.sync(state, rng) {
            optimizer.ingestion_error = Some(error.to_string());
            return Err(error);
        }
    }

    let lengthscales = optimizer.surrogate().and_then(|s| s.lengthscales());
    let ls_ref: Option<ArrayView1<f64>> = lengthscales.as_ref().map(|ls| ls.view());
    optimizer.telemetry.num_candidates = 0;
    optimizer.telemetry.dt_sel = 0.0;
    let (blocks, segments) = build_segments(optimizer, batches, ls_ref, rng)?;

    if blocks.is_empty() {
        return Err(ENNError::InvalidParameter(
            "experimental strategy produced no candidates".to_string(),
        ));
    }

    let views = blocks.iter().map(|b| b.view()).collect::<Vec<_>>();
    let candidates = ndarray::concatenate(Axis(0), &views)
        .map_err(|error| ENNError::InvalidParameter(error.to_string()))?;
    let start = std::time::Instant::now();
    let indices = select_segment(optimizer, &candidates.view(), &segments, rng)?;
    optimizer.telemetry.dt_sel += start.elapsed().as_secs_f64();
    let selected = select_indices(&candidates.view(), &indices);

    if let TrustRegionState::Multi { state, pending } = &mut optimizer.regions {
        let arms: Vec<crate::optimizer::RegionProvenance> = segments
            .iter()
            .flat_map(|segment| {
                let generation = state
                    .generation(segment.region)
                    .expect("candidate segment refers to an active region slot");
                std::iter::repeat_n(
                    crate::optimizer::RegionProvenance {
                        slot: segment.region,
                        generation,
                    },
                    segment.arms,
                )
            })
            .collect();
        let ids = issue_ids(arms.len())?;
        let next = PendingProposal {
            ids,
            outcomes: vec![EvaluationStatus::Pending; arms.len()],
            arms: Some(arms),
            settled_regions: vec![false; state.num_regions()],
            x: selected.clone(),
            ordered_tell_allowed: true,
        };
        if let Some(existing) = pending.as_mut() {
            append_pending(existing, next)?;
        } else {
            *pending = Some(next);
        }
    }
    Ok(selected)
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
