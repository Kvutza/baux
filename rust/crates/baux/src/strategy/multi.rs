//! Multi-trust-region ask and tell execution over `TrustRegionState::Multi`.

use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};
use rand::RngCore;

mod ask;
pub(super) use ask::ask_multi;
use ask::restart_inactive;

use super::Submission;
use crate::config::InitStrategy;
use crate::error::ENNError;
use crate::optimizer::multi_tr::EvaluationStatus;
use crate::optimizer::{BayesOptimizer, InitStrategyState, PendingProposal, TrustRegionState};
use crate::posterior::materialize_selected;
use baux::semantics::key::Key;
use baux::semantics::pool::DelayedPool;

/// Generate initialization candidates in the unit cube via zero-materialization
/// [`DelayedPool`] (`Lhd` / `Uniform`).
pub(super) fn ask_init(
    state: &InitStrategyState,
    optimizer: BayesOptimizer,
    num_arms: usize,
    rng: &mut dyn RngCore,
) -> Result<(BayesOptimizer, Array2<f64>), ENNError> {
    let num_dim = optimizer.num_dim();
    let lower = vec![0.0; num_dim];
    let upper = vec![1.0; num_dim];
    let key = Key::new(rng.next_u64());
    let pool = match state.strategy_type {
        InitStrategy::LHD => DelayedPool::Lhd {
            key,
            lower: &lower,
            upper: &upper,
            num_candidates: num_arms,
        },
        InitStrategy::Random => DelayedPool::Uniform {
            key,
            lower: &lower,
            upper: &upper,
            num_candidates: num_arms,
        },
    };
    let indices: Vec<usize> = (0..num_arms).collect();
    let x = materialize_selected(&pool, &indices);
    Ok((optimizer, x))
}

/// Common tell logic: add observations, fit surrogate, update incumbent.
pub(super) fn tell_common(
    mut optimizer: BayesOptimizer,
    x: &ArrayView2<f64>,
    y: &ArrayView2<f64>,
    yvar: Option<&ArrayView2<f64>>,
    mut telemetry: Option<&mut crate::optimizer::Telemetry>,
    rng: &mut dyn RngCore,
) -> Result<BayesOptimizer, ENNError> {
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
    let (posterior, dt_fit) = optimizer.posterior.fit_delta(&delta, yvar, rng)?;
    optimizer.posterior = posterior;
    if let Some(dt_fit) = dt_fit {
        if let Some(tel) = telemetry.as_deref_mut() {
            tel.dt_fit = dt_fit;
        }
    }

    if optimizer.regions.is_morbo() && delta.new_n > delta.old_n {
        optimizer.map_regions(|r| r.morbo_only(&delta.y_view()))?;
    }

    optimizer.update_incumbent(rng)?;

    if optimizer.regions.is_morbo() && delta.new_n > 0 {
        let y_inc = optimizer
            .y_scalar()
            .ok_or_else(|| ENNError::InvalidParameter("Missing incumbent y".to_string()))?
            .to_owned();
        optimizer.map_regions(|r| r.update_morbo(&y_inc.view(), delta.new_n))?;
    }

    if x.nrows() >= 64 {
        optimizer.posterior.release_pages()?;
    }

    Ok(optimizer)
}

/// Validate before changing the variance mode, observation store, or regions.
/// baux's legacy ordered-batch guard is not a paper-prescribed attribution
/// rule.
pub(super) fn validate_multi(
    optimizer: &BayesOptimizer,
    x: &ArrayView2<f64>,
    _y: &ArrayView2<f64>,
) -> Result<(), ENNError> {
    match optimizer.regions.as_ref() {
        TrustRegionState::Multi {
            state,
            pending: Some(proposal),
        } => {
            if !proposal.ordered_tell_allowed {
                return Err(ENNError::InvalidParameter(
                    "overlapping regional asks require completion by evaluation ID".into(),
                ));
            }
            if proposal
                .outcomes
                .iter()
                .any(|outcome| !matches!(outcome, EvaluationStatus::Pending))
            {
                return Err(ENNError::InvalidParameter(
                    "batch has partial completions; finish it with evaluation IDs".into(),
                ));
            }
            if proposal.ids.len() != x.nrows() {
                return Err(ENNError::InvalidParameter(format!(
                    "pending proposal has {} arms but tell contains {} observations",
                    proposal.ids.len(),
                    x.nrows()
                )));
            }
            if proposal.x.view() != *x {
                return Err(ENNError::InvalidParameter(
                    "tell must contain the pending proposal coordinates in their original order"
                        .to_string(),
                ));
            }
            if let Some(arms) = &proposal.arms {
                for provenance in arms {
                    if state.validate(provenance.ticket).is_err() {
                        return Err(ENNError::InvalidParameter(format!(
                            "observation belongs to stale region {} generation {}",
                            provenance.slot, provenance.generation
                        )));
                    }
                }
            }
        }
        TrustRegionState::Multi { pending: None, .. } => {}
        _ => unreachable!("tell_multi requires TrustRegionState::Multi"),
    }
    Ok(())
}

pub(super) fn select_pending(
    optimizer: &BayesOptimizer,
    ids: &[crate::optimizer::EvaluationId],
) -> Result<(Array2<f64>, Option<Vec<usize>>), ENNError> {
    let TrustRegionState::Multi {
        state,
        pending: Some(proposal),
    } = optimizer.regions.as_ref()
    else {
        return Err(ENNError::InvalidParameter(
            "no pending regional evaluations".into(),
        ));
    };
    if ids.is_empty() {
        return Err(ENNError::InvalidParameter(
            "completion needs evaluation IDs".into(),
        ));
    }
    let mut rows = Vec::with_capacity(ids.len());
    let mut selected = vec![false; proposal.ids.len()];
    let mut origins = proposal
        .arms
        .as_ref()
        .map(|_| Vec::with_capacity(ids.len()));
    for &id in ids {
        let row = proposal
            .ids
            .iter()
            .position(|&candidate| candidate == id)
            .ok_or_else(|| ENNError::InvalidParameter("foreign or stale evaluation ID".into()))?;
        if selected[row] || !matches!(proposal.outcomes[row], EvaluationStatus::Pending) {
            return Err(ENNError::InvalidParameter(
                "duplicate completed evaluation ID".into(),
            ));
        }
        selected[row] = true;
        if let Some(arms) = &proposal.arms {
            let provenance = arms[row];
            if state.validate(provenance.ticket).is_err() {
                return Err(ENNError::InvalidParameter(
                    "evaluation belongs to a stale region generation".into(),
                ));
            }
            if let Some(origins) = &mut origins {
                origins.push(provenance.slot);
            }
        }
        rows.push(row);
    }
    Ok((proposal.x.select(Axis(0), &rows), origins))
}

fn record_pending(
    proposal: &mut PendingProposal,
    y: &ArrayView1<f64>,
    submission: Submission<'_>,
) -> Result<bool, ENNError> {
    if let Submission::Identified { ids, .. } = submission {
        for (&id, &value) in ids.iter().zip(y.iter()) {
            let row = proposal
                .ids
                .iter()
                .position(|&candidate| candidate == id)
                .ok_or_else(|| {
                    ENNError::InvalidParameter("completed evaluation ID disappeared".into())
                })?;
            proposal.outcomes[row] = EvaluationStatus::Observed(value);
        }
    } else {
        for (slot, &value) in proposal.outcomes.iter_mut().zip(y.iter()) {
            *slot = EvaluationStatus::Observed(value);
        }
    }
    Ok(proposal
        .outcomes
        .iter()
        .all(|outcome| !matches!(outcome, EvaluationStatus::Pending)))
}

fn finish_warmup(mut optimizer: BayesOptimizer) -> Result<BayesOptimizer, ENNError> {
    let count = if let TrustRegionState::Multi { pending, .. } = optimizer.trust_mut() {
        let proposal = pending.as_ref().ok_or_else(|| {
            ENNError::InvalidParameter("warm-up completion lost its pending batch".into())
        })?;
        if proposal
            .outcomes
            .iter()
            .any(|outcome| matches!(outcome, EvaluationStatus::Pending))
        {
            return Ok(optimizer);
        }
        let count = proposal
            .outcomes
            .iter()
            .filter(|outcome| matches!(outcome, EvaluationStatus::Observed(_)))
            .count();
        *pending = None;
        count
    } else {
        unreachable!("warm-up completion requires a multi-region optimizer");
    };
    optimizer.warmup.advance_init(count);
    optimizer.warmup.check_transition();
    Ok(optimizer)
}

fn finish_multi(
    mut optimizer: BayesOptimizer,
    ls_ref: Option<&ArrayView1<f64>>,
    rng: &mut dyn RngCore,
) -> Result<BayesOptimizer, ENNError> {
    {
        let TrustRegionState::Multi { state, pending } = optimizer.trust_mut() else {
            unreachable!("regional completion requires a multi-region optimizer");
        };
        let proposal = pending.as_mut().ok_or_else(|| {
            ENNError::InvalidParameter("regional completion lost its pending batch".into())
        })?;
        let arms = proposal
            .arms
            .as_ref()
            .expect("regional completion retains provenance for every ask row");
        // A slot's controller sees the entire issued group once that group's
        // results are resolved. Other slots in the same ask batch need not
        // wait.
        for slot in 0..state.num_regions() {
            let rows_for_slot = arms
                .iter()
                .enumerate()
                .filter_map(|(row, arm)| (arm.slot == slot).then_some(row))
                .collect::<Vec<_>>();
            if rows_for_slot.is_empty() {
                continue;
            }
            let ticket = arms[rows_for_slot[0]].ticket;
            if state.validate(ticket).is_err() {
                continue;
            }
            if rows_for_slot
                .iter()
                .any(|&row| matches!(proposal.outcomes[row], EvaluationStatus::Pending))
            {
                continue;
            }
            let observed = rows_for_slot
                .into_iter()
                .filter_map(|row| match proposal.outcomes[row] {
                    EvaluationStatus::Observed(value) => Some((row, value)),
                    _ => None,
                })
                .collect::<Vec<_>>();
            if !observed.is_empty() {
                let rows = observed.iter().map(|(row, _)| *row).collect::<Vec<_>>();
                let values = observed.iter().map(|(_, value)| *value).collect::<Vec<_>>();
                let origins = vec![slot; rows.len()];
                state
                    .tell_scaled(
                        &proposal.x.select(Axis(0), &rows).view(),
                        &Array1::from(values).view(),
                        Some(&origins),
                        ls_ref,
                    )
                    .map_err(|e| ENNError::InvalidParameter(e.to_string()))?;
                state
                    .finish(ticket)
                    .map_err(|error| ENNError::InvalidParameter(error.to_string()))?;
            } else {
                state
                    .retire(ticket)
                    .map_err(|error| ENNError::InvalidParameter(error.to_string()))?;
            }
        }
        if proposal
            .outcomes
            .iter()
            .any(|outcome| matches!(outcome, EvaluationStatus::Pending))
        {
            return Ok(optimizer);
        }
        *pending = None;
        restart_inactive(state, rng)?;
    }
    if let Some(models) = optimizer.local_models.take() {
        let TrustRegionState::Multi { state, .. } = optimizer.regions.as_ref() else {
            unreachable!("regional completion requires a multi-region optimizer");
        };
        optimizer.local_models = Some(models.sync(state, rng)?);
    }
    Ok(optimizer)
}

pub(super) fn cancel_pending(
    mut optimizer: BayesOptimizer,
    ids: &[crate::optimizer::EvaluationId],
    rng: &mut dyn RngCore,
) -> Result<BayesOptimizer, ENNError> {
    {
        let TrustRegionState::Multi {
            pending: Some(proposal),
            ..
        } = optimizer.trust_mut()
        else {
            unreachable!("cancelled IDs were validated against a pending batch");
        };
        for &id in ids {
            let row = proposal
                .ids
                .iter()
                .position(|&candidate| candidate == id)
                .expect("cancelled ID was validated before mutation");
            proposal.outcomes[row] = EvaluationStatus::Cancelled;
        }
    }
    if optimizer.warmup.in_init() {
        finish_warmup(optimizer)
    } else {
        let lengthscales = optimizer.surrogate().and_then(|s| s.lengthscales());
        let ls_ref = lengthscales.as_ref().map(|ls| ls.view());
        finish_multi(optimizer, ls_ref.as_ref(), rng)
    }
}

pub(super) fn tell_warmup(
    mut optimizer: BayesOptimizer,
    x: &ArrayView2<f64>,
    y: &ArrayView2<f64>,
    yvar: Option<&ArrayView2<f64>>,
    rng: &mut dyn RngCore,
    submission: Submission<'_>,
) -> Result<BayesOptimizer, ENNError> {
    optimizer = tell_common(optimizer, x, y, yvar, None, rng)?;
    if let Some(models) = optimizer.local_models.take() {
        optimizer.local_models = Some(models.warmup(x, y, yvar, rng)?);
    }
    if matches!(submission, Submission::Imported) {
        return Ok(optimizer);
    }
    if let TrustRegionState::Multi { pending, .. } = optimizer.trust_mut() {
        let proposal = pending.as_mut().ok_or_else(|| {
            ENNError::InvalidParameter("warm-up completion lost its pending batch".into())
        })?;
        // Regional adaptation has one scalar control channel. Pareto
        // acquisition retains every objective in the surrogate; the
        // first objective is the explicit control value until the host
        // API accepts a separate control.
        record_pending(proposal, &y.column(0), submission)?;
    }
    finish_warmup(optimizer)
}

pub(super) fn tell_multi(
    mut optimizer: BayesOptimizer,
    x: &ArrayView2<f64>,
    y: &ArrayView2<f64>,
    yvar: Option<&ArrayView2<f64>>,
    rng: &mut dyn RngCore,
    submission: Submission<'_>,
) -> Result<BayesOptimizer, ENNError> {
    let mut telemetry = std::mem::take(&mut optimizer.telemetry);
    let mut optimizer = tell_common(optimizer, x, y, yvar, Some(&mut telemetry), rng)?;
    optimizer.telemetry = telemetry;

    let pending_regions = match (optimizer.regions.as_ref(), submission) {
        (_, Submission::Identified { origins, .. }) => origins.map(<[usize]>::to_vec),
        (
            TrustRegionState::Multi {
                pending: Some(proposal),
                ..
            },
            Submission::Ordered,
        ) => proposal
            .arms
            .as_ref()
            .map(|arms| arms.iter().map(|arm| arm.slot).collect::<Vec<_>>()),
        _ => None,
    };
    let lengthscales = optimizer.surrogate().and_then(|s| s.lengthscales());
    let ls_ref: Option<ArrayView1<f64>> = lengthscales.as_ref().map(|ls| ls.view());
    if let Some(models) = optimizer.local_models.take() {
        let TrustRegionState::Multi { state, .. } = optimizer.regions.as_ref() else {
            unreachable!("tell_multi requires TrustRegionState::Multi");
        };
        let updated = models.route(
            state,
            x,
            y,
            yvar,
            pending_regions.as_deref(),
            ls_ref.as_ref(),
            rng,
        )?;
        optimizer.local_models = Some(updated);
    }

    // Imports have no generating region to credit, but shared and
    // nearest-center relevance can still update their regional incumbents.
    let Some(_pending_regions) = pending_regions else {
        if let TrustRegionState::Multi { state, .. } = optimizer.trust_mut() {
            if state.sharing_policy() != crate::optimizer::SharingPolicy::OriginOnly {
                state
                    .tell_scaled(x, &y.column(0), None, ls_ref.as_ref())
                    .map_err(|e| ENNError::InvalidParameter(e.to_string()))?;
            }
        }
        return Ok(optimizer);
    };

    if let TrustRegionState::Multi { pending, .. } = optimizer.trust_mut() {
        let proposal = pending.as_mut().ok_or_else(|| {
            ENNError::InvalidParameter("regional completion lost its pending batch".into())
        })?;
        // Preserve the full matrix above for the surrogate and Pareto selector;
        // only the explicit first-objective control reaches region adaptation.
        record_pending(proposal, &y.column(0), submission)?;
    }
    finish_multi(optimizer, ls_ref.as_ref(), rng)
}
