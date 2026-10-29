//! Multi-trust-region ask and tell execution over `TrustRegionState::Multi`.

use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};
use rand::RngCore;

use super::{CandidateSegment, ask_init, select_indices, select_segment, tell_common};
use crate::candidates::generate_candidates;
use crate::error::ENNError;
use crate::optimizer::{
    BayesOptimizer, InitStrategyState, MultiTrustRegionState, RegionBatch, TrustRegionState,
};

fn ensure_active(
    multi_tr: &mut MultiTrustRegionState,
    center: &Array1<f64>,
) -> Result<(), ENNError> {
    if multi_tr.active_count() > 0 {
        return Ok(());
    }
    let center_view = center.view();
    for region in 0..multi_tr.num_regions() {
        multi_tr
            .restart_region(region, &center_view)
            .map_err(|e| ENNError::InvalidParameter(e.to_string()))?;
    }
    Ok(())
}

fn restart_inactive(
    multi_tr: &mut MultiTrustRegionState,
    center: &Array1<f64>,
) -> Result<(), ENNError> {
    if multi_tr.active_count() == 0 {
        return ensure_active(multi_tr, center);
    }
    let center_view = center.view();
    for region in 0..multi_tr.num_regions() {
        if !multi_tr.active_mask[region] {
            multi_tr
                .restart_region(region, &center_view)
                .map_err(|e| ENNError::InvalidParameter(e.to_string()))?;
        }
    }
    Ok(())
}

fn build_segments(
    optimizer: &mut BayesOptimizer,
    batches: Vec<RegionBatch>,
    ls_ref: Option<ArrayView1<f64>>,
    rng: &mut dyn RngCore,
) -> Result<(Vec<Array2<f64>>, Vec<CandidateSegment>), ENNError> {
    let num_dim = optimizer.num_dim();
    let config = optimizer.proposals.config().clone();
    let mut blocks = Vec::with_capacity(batches.len());
    let mut segments = Vec::with_capacity(batches.len());
    let mut offset = 0;

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

        let x_cand = generate_candidates(
            || (lower_1d.clone(), upper_1d.clone()),
            &x_center.view(),
            ls_ref.as_ref(),
            num_candidates,
            config.candidate_rv,
            rng,
            optimizer.proposals.sobol_mut(),
            config.num_pert,
        )?;
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

pub(super) fn ask_multi(
    optimizer: &mut BayesOptimizer,
    num_arms: usize,
    rng: &mut dyn RngCore,
) -> Result<Array2<f64>, ENNError> {
    if optimizer.warmup.in_init() {
        if let TrustRegionState::Multi { pending, .. } = &mut optimizer.regions {
            *pending = None;
        }
        let state = InitStrategyState::new(optimizer.warmup.kind, optimizer.warmup.budget);
        return ask_init(&state, optimizer, num_arms, rng);
    }

    let center = optimizer
        .unit_incumbent()
        .cloned()
        .unwrap_or_else(|| optimizer.domain.unit_center());
    let batches = match &mut optimizer.regions {
        TrustRegionState::Multi { state, .. } => {
            ensure_active(state, &center)?;
            state
                .allocate(num_arms)
                .map_err(|e| ENNError::InvalidParameter(e.to_string()))?
        }
        _ => unreachable!("ask_multi requires TrustRegionState::Multi"),
    };

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

    if let TrustRegionState::Multi { pending, .. } = &mut optimizer.regions {
        *pending = Some(
            segments
                .iter()
                .flat_map(|s| std::iter::repeat_n(s.region, s.arms))
                .collect(),
        );
    }
    Ok(select_indices(&candidates.view(), &indices))
}

pub(super) fn tell_multi(
    optimizer: &mut BayesOptimizer,
    x: &ArrayView2<f64>,
    y: &ArrayView2<f64>,
    yvar: Option<&ArrayView2<f64>>,
    rng: &mut dyn RngCore,
) -> Result<(), ENNError> {
    if y.ncols() != 1 {
        return Err(ENNError::InvalidParameter(format!(
            "experimental multi-trust-region strategy expects scalar y, got {} columns",
            y.ncols()
        )));
    }

    let mut telemetry = std::mem::take(&mut optimizer.telemetry);
    let res = tell_common(optimizer, x, y, yvar, Some(&mut telemetry), rng);
    optimizer.telemetry = telemetry;
    res?;

    optimizer.warmup.advance_init(x.nrows());
    optimizer.warmup.check_transition();

    let center = optimizer
        .unit_incumbent()
        .cloned()
        .unwrap_or_else(|| optimizer.domain.unit_center());
    let y_scalar = y.column(0);

    if let TrustRegionState::Multi { state, pending } = &mut optimizer.regions {
        let pending_regions = pending.take();
        if let Some(regions) = pending_regions.as_deref() {
            state
                .tell(x, &y_scalar, Some(regions))
                .map_err(|e| ENNError::InvalidParameter(e.to_string()))?;
        } else {
            state
                .tell_update(x, &y_scalar)
                .map_err(|e| ENNError::InvalidParameter(e.to_string()))?;
        }
        restart_inactive(state, &center)?;
    }

    Ok(())
}
