//! Multi-trust-region ask and tell execution over `TrustRegionState::Multi`.

use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};
use rand::RngCore;

use super::selection::{CandidateSegment, select_indices, select_segment};
use crate::candidates::{generate_candidates, generate_lhd, generate_uniform};
use crate::config::InitStrategy;
use crate::error::ENNError;
use crate::optimizer::{
    BayesOptimizer, InitStrategyState, MultiTrustRegionState, RegionBatch, Telemetry,
    TrustRegionState,
};

/// Generate initialization candidates in the unit cube.
pub(super) fn ask_init(
    state: &InitStrategyState,
    optimizer: &mut BayesOptimizer,
    num_arms: usize,
    rng: &mut dyn RngCore,
) -> Result<Array2<f64>, ENNError> {
    let num_dim = optimizer.num_dim();
    let lower = Array1::zeros(num_dim);
    let upper = Array1::ones(num_dim);

    match state.strategy_type {
        InitStrategy::LHD => {
            let mut unit_bounds = Array2::zeros((num_dim, 2));
            for j in 0..num_dim {
                unit_bounds[[j, 1]] = 1.0;
            }
            Ok(generate_lhd(num_arms, num_dim, &unit_bounds.view(), rng))
        }
        InitStrategy::Random => generate_uniform(&lower, &upper, num_arms, rng),
    }
}

/// Common tell logic: add observations, fit surrogate, update incumbent.
pub(super) fn tell_common(
    optimizer: &mut BayesOptimizer,
    x: &ArrayView2<f64>,
    y: &ArrayView2<f64>,
    yvar: Option<&ArrayView2<f64>>,
    telemetry: Option<&mut Telemetry>,
    rng: &mut dyn RngCore,
) -> Result<(), ENNError> {
    if optimizer.regions.is_morbo() {
        let metrics = optimizer.regions.num_metrics();
        if y.ncols() != metrics {
            return Err(ENNError::InvalidParameter(format!(
                "y has {} metric columns but Morbo expects {metrics}",
                y.ncols()
            )));
        }
    }

    let delta = optimizer.add_observations(x, y)?;
    if let Some(dt_fit) = optimizer.posterior.fit_delta(&delta, yvar, rng)? {
        if let Some(tel) = telemetry {
            tel.dt_fit = dt_fit;
        }
    }

    if optimizer.regions.is_morbo() && delta.new_n > delta.old_n {
        optimizer.regions.morbo_only(&delta.y_view())?;
    }

    optimizer.update_incumbent(rng)?;

    if optimizer.regions.is_morbo() && delta.new_n > 0 {
        let y_inc = optimizer
            .y_scalar()
            .ok_or_else(|| ENNError::InvalidParameter("Missing incumbent y".to_string()))?
            .to_owned();
        optimizer.regions.update_morbo(&y_inc.view(), delta.new_n)?;
    }

    if x.nrows() >= 64 {
        optimizer.posterior.release_pages()?;
    }

    Ok(())
}

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
