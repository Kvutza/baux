//! Optimization execution strategies for the role-decomposed ask/tell state
//! machine.

use ndarray::{Array2, ArrayView1, ArrayView2, Axis, s};
use rand::RngCore;

mod multi;
mod selection;
use multi::{ask_init, ask_multi, select_pending, tell_common, tell_multi, validate_multi};
use selection::*;

use crate::candidates::perturb_candidates;
use crate::config::InitStrategy;
use crate::error::ENNError;
use crate::optimizer::multi_tr::EvaluationStatus;
use crate::optimizer::{BayesOptimizer, EvaluationId, Telemetry, TrustRegionState};
pub use crate::optimizer::{InitStrategyState, Strategy, StrategyState, TurboStrategyState};
use crate::optimizer_dispatch::{AskGate, TellGate};

#[derive(Clone, Copy)]
enum Submission<'a> {
    Ordered,
    Imported,
    Identified {
        ids: &'a [EvaluationId],
        origins: Option<&'a [usize]>,
    },
}

impl BayesOptimizer {
    /// Public-boundary ask in natural coordinates.
    pub fn ask_natural(
        mut self,
        num_arms: usize,
        rng: &mut dyn RngCore,
    ) -> Result<(Array2<f64>, Self), ENNError> {
        let (unit, next) = self.ask(num_arms, rng)?;
        self = next;
        let natural = self.domain.from_unit(&unit.view());
        if let TrustRegionState::Multi {
            pending: Some(proposal),
            ..
        } = &mut self.regions
        {
            // Compare the exact values that a natural-coordinate tell will
            // reconstruct; floating-point normalization need not round-trip.
            let start = proposal.x.nrows() - natural.nrows();
            proposal
                .x
                .slice_mut(s![start.., ..])
                .assign(&self.domain.to_unit(&natural.view()));
        }
        Ok((natural, self))
    }

    /// Public-boundary tell in natural coordinates.
    pub fn tell_natural(
        self,
        x: &ArrayView2<f64>,
        y: &ArrayView2<f64>,
        yvar: Option<&ArrayView2<f64>>,
        rng: &mut dyn RngCore,
    ) -> Result<Self, ENNError> {
        if x.ncols() != self.num_dim() {
            return Err(ENNError::InvalidShape {
                expected: vec![x.nrows(), self.num_dim()],
                got: vec![x.nrows(), x.ncols()],
            });
        }
        if y.nrows() != x.nrows() {
            return Err(ENNError::InvalidShape {
                expected: vec![x.nrows(), y.ncols()],
                got: vec![y.nrows(), y.ncols()],
            });
        }
        let unit = self.domain.to_unit(x);
        self.tell_variance(&unit.view(), y, yvar, rng)
    }

    /// Add an externally evaluated point without consuming a pending region
    /// proposal. Its relevance is still determined by the sharing policy.
    pub fn import_natural(
        self,
        x: &ArrayView2<f64>,
        y: &ArrayView2<f64>,
        yvar: Option<&ArrayView2<f64>>,
        rng: &mut dyn RngCore,
    ) -> Result<Self, ENNError> {
        if x.ncols() != self.num_dim() || x.nrows() != y.nrows() {
            return Err(ENNError::InvalidShape {
                expected: vec![x.nrows(), self.num_dim()],
                got: vec![x.nrows(), x.ncols()],
            });
        }
        let unit = self.domain.to_unit(x);
        self.import_variance(&unit.view(), y, yvar, rng)
    }

    /// Add externally evaluated observations in unit coordinates without
    /// consuming a pending region proposal.
    pub fn import_observations(
        self,
        x: &ArrayView2<f64>,
        y: &ArrayView2<f64>,
        yvar: Option<&ArrayView2<f64>>,
        rng: &mut dyn RngCore,
    ) -> Result<Self, ENNError> {
        self.import_variance(x, y, yvar, rng)
    }

    pub fn import_variance(
        self,
        x: &ArrayView2<f64>,
        y: &ArrayView2<f64>,
        yvar: Option<&ArrayView2<f64>>,
        rng: &mut dyn RngCore,
    ) -> Result<Self, ENNError> {
        self.submit(x, y, yvar, rng, Submission::Imported)
    }

    /// Ask for candidate points in unit coordinates `[0, 1]^d`.
    pub fn ask(
        mut self,
        num_arms: usize,
        rng: &mut dyn RngCore,
    ) -> Result<(Array2<f64>, Self), ENNError> {
        self.check_ingestion()?;
        match crate::optimizer_dispatch::ask_gate(
            num_arms,
            self.regions.is_morbo(),
            self.regions.needs_restart(),
        ) {
            AskGate::ZeroArms => {
                return Err(ENNError::InvalidParameter(
                    "num_arms must be > 0, got 0".to_string(),
                ));
            }
            AskGate::RestartLocal => {
                self = self.restart_local()?;
            }
            AskGate::Dispatch => {}
        }
        let start = std::time::Instant::now();
        let (candidates, mut this) = self.dispatch_ask(num_arms, rng)?;
        this.telemetry.dt_gen = start.elapsed().as_secs_f64();
        if let Err(error) = this.posterior.schedule_flush() {
            if this.regions.is_multi() {
                this.ingestion_error = Some(error.to_string());
            }
            return Err(error);
        }
        Ok((candidates, this))
    }

    fn dispatch_ask(
        mut self,
        num_arms: usize,
        rng: &mut dyn RngCore,
    ) -> Result<(Array2<f64>, Self), ENNError> {
        if self.reinit_left().is_some() {
            return ask_reinit(self, num_arms, rng);
        }
        if self.regions.is_multi() {
            return ask_multi(self, num_arms, rng);
        }
        if self.warmup.in_init() {
            let state = InitStrategyState::new(self.warmup.kind, self.warmup.budget);
            return ask_init(&state, self, num_arms, rng);
        }
        let mut telemetry = std::mem::take(&mut self.telemetry);
        let (result, mut this) = ask_turbo(self, num_arms, &mut telemetry, rng)?;
        this.telemetry = telemetry;
        Ok((result, this))
    }

    /// Tell observations with optional known observation variance.
    pub fn tell(
        self,
        x: &ArrayView2<f64>,
        y: &ArrayView2<f64>,
        yvar: Option<&ArrayView2<f64>>,
        rng: &mut dyn RngCore,
    ) -> Result<Self, ENNError> {
        self.submit(x, y, yvar, rng, Submission::Ordered)
    }

    /// Tell observations with optional known observation variance.
    pub fn tell_variance(
        self,
        x: &ArrayView2<f64>,
        y: &ArrayView2<f64>,
        yvar: Option<&ArrayView2<f64>>,
        rng: &mut dyn RngCore,
    ) -> Result<Self, ENNError> {
        self.tell(x, y, yvar, rng)
    }

    /// Complete any nonempty subset of a pending host batch by evaluation
    /// identity. IDs and result rows may arrive in any order.
    pub fn complete_ids(
        self,
        ids: &[EvaluationId],
        y: &ArrayView2<f64>,
        yvar: Option<&ArrayView2<f64>>,
        rng: &mut dyn RngCore,
    ) -> Result<Self, ENNError> {
        self.check_ingestion()?;
        if y.ncols() != 1 || y.nrows() != ids.len() {
            return Err(ENNError::InvalidParameter(
                "ID completion requires one scalar result per ID".into(),
            ));
        }
        let (x, origins) = select_pending(&self, ids)?;
        self.submit(
            &x.view(),
            y,
            yvar,
            rng,
            Submission::Identified {
                ids,
                origins: origins.as_deref(),
            },
        )
    }

    /// Retire unevaluated rows without adding observations or controller
    /// credit. A cancelled warm-up row does not consume the initialization
    /// budget.
    pub fn cancel_ids(self, ids: &[EvaluationId], rng: &mut dyn RngCore) -> Result<Self, ENNError> {
        self.check_ingestion()?;
        select_pending(&self, ids)?;
        multi::cancel_pending(self, ids, rng)
    }

    /// Restart one region, abandoning its unresolved evaluations without
    /// controller credit. Already ingested observations remain model evidence.
    /// The returned IDs must not be completed on this optimizer; a late result
    /// can only be submitted as an explicit external import.
    pub fn restart_region(
        mut self,
        slot: usize,
        center: &ArrayView1<f64>,
        rng: &mut dyn RngCore,
    ) -> Result<(Vec<EvaluationId>, Self), ENNError> {
        self.check_ingestion()?;
        if self.warmup.in_init() {
            return Err(ENNError::InvalidParameter(
                "regional restart requires completed warm-up".into(),
            ));
        }
        if center.len() != self.num_dim()
            || center
                .iter()
                .any(|&value| !value.is_finite() || !(0.0..=1.0).contains(&value))
        {
            return Err(ENNError::InvalidParameter(
                "regional restart center must be finite unit coordinates".into(),
            ));
        }
        if let Err(error) = self.posterior.wait_flush() {
            self.ingestion_error = Some(error.to_string());
            return Err(error);
        }
        let abandoned = {
            let (state, pending) = match &mut self.regions {
                TrustRegionState::Multi { state, pending } => (state, pending),
                _ => {
                    return Err(ENNError::InvalidParameter(
                        "regional restart requires a multi-region optimizer".into(),
                    ));
                }
            };
            state
                .restart_region(slot, center)
                .map_err(|error| ENNError::InvalidParameter(error.to_string()))?;
            let mut abandoned = Vec::new();
            if let Some(proposal) = pending.as_mut() {
                if let Some(arms) = proposal.arms.as_ref() {
                    for (row, arm) in arms.iter().enumerate() {
                        if arm.slot == slot
                            && matches!(proposal.outcomes[row], EvaluationStatus::Pending)
                        {
                            abandoned.push(proposal.ids[row]);
                            proposal.outcomes[row] = EvaluationStatus::Cancelled;
                        }
                    }
                }
                if proposal
                    .outcomes
                    .iter()
                    .all(|outcome| !matches!(outcome, EvaluationStatus::Pending))
                {
                    *pending = None;
                }
            }
            abandoned
        };
        if let Some(models) = self.local_models.take() {
            let state = match &self.regions {
                TrustRegionState::Multi { state, .. } => state,
                _ => unreachable!(),
            };
            match models.sync(state, rng) {
                Ok(synced) => self.local_models = Some(synced),
                Err(error) => {
                    self.ingestion_error = Some(error.to_string());
                    return Err(error);
                }
            }
        }
        Ok((abandoned, self))
    }

    /// Natural-coordinate form of [`Self::restart_region`].
    pub fn restart_natural(
        self,
        slot: usize,
        center: &ArrayView1<f64>,
        rng: &mut dyn RngCore,
    ) -> Result<(Vec<EvaluationId>, Self), ENNError> {
        if center.len() != self.num_dim() {
            return Err(ENNError::InvalidShape {
                expected: vec![self.num_dim()],
                got: vec![center.len()],
            });
        }
        let natural = center.to_owned().insert_axis(Axis(0));
        let unit = self.domain.to_unit(&natural.view());
        self.restart_region(slot, &unit.row(0), rng)
    }

    /// Outstanding warm-up or regional evaluation identities in ask order.
    pub fn pending_ids(&self) -> Option<Vec<EvaluationId>> {
        match &self.regions {
            TrustRegionState::Multi {
                pending: Some(proposal),
                ..
            } => Some(
                proposal
                    .ids
                    .iter()
                    .zip(&proposal.outcomes)
                    .filter_map(|(&id, outcome)| {
                        matches!(outcome, EvaluationStatus::Pending).then_some(id)
                    })
                    .collect(),
            ),
            _ => None,
        }
    }

    fn submit(
        mut self,
        x: &ArrayView2<f64>,
        y: &ArrayView2<f64>,
        yvar: Option<&ArrayView2<f64>>,
        rng: &mut dyn RngCore,
        submission: Submission<'_>,
    ) -> Result<Self, ENNError> {
        self.check_ingestion()?;
        self.check_submission(x, y, submission)?;
        match crate::optimizer_dispatch::tell_gate(self.num_dim(), x.nrows(), x.ncols(), y.nrows())
        {
            TellGate::InvalidX => {
                return Err(ENNError::InvalidShape {
                    expected: vec![x.nrows(), self.num_dim()],
                    got: vec![x.nrows(), x.ncols()],
                });
            }
            TellGate::InvalidRows => {
                return Err(ENNError::InvalidShape {
                    expected: vec![x.nrows(), y.ncols()],
                    got: vec![y.nrows(), y.ncols()],
                });
            }
            TellGate::Empty => return Ok(self),
            TellGate::Dispatch => {}
        }
        if y.ncols() == 0 || x.iter().chain(y.iter()).any(|v| !v.is_finite()) {
            return Err(ENNError::InvalidParameter(
                "observations require finite coordinates and at least one finite objective column"
                    .to_string(),
            ));
        }
        if let Some(variance) = yvar {
            if variance.dim() != y.dim() || variance.iter().any(|v| !v.is_finite() || *v < 0.0) {
                return Err(ENNError::InvalidParameter(
                    "y_var must match y and contain finite nonnegative variances".to_string(),
                ));
            }
        }
        self.posterior = self.posterior.check_variance(yvar.is_some())?;
        let start = std::time::Instant::now();
        if let Err(error) = self.posterior.wait_flush() {
            // A deferred flush belongs to a previous accepted submission.
            // Its failure leaves that submission's storage outcome unknown,
            // so this optimizer cannot safely accept a retry or new result.
            if self.regions.is_multi() {
                self.ingestion_error = Some(error.to_string());
            }
            return Err(error);
        }
        let mut this = self.dispatch_tell(x, y, yvar, rng, submission)?;
        if x.nrows() < 64 {
            if let Err(error) = this.posterior.schedule_flush() {
                if this.regions.is_multi() {
                    this.ingestion_error = Some(error.to_string());
                }
                return Err(error);
            }
        }
        this.telemetry.dt_tell = start.elapsed().as_secs_f64();
        Ok(this)
    }

    fn check_submission(
        &self,
        x: &ArrayView2<f64>,
        y: &ArrayView2<f64>,
        submission: Submission<'_>,
    ) -> Result<(), ENNError> {
        if matches!(submission, Submission::Imported) && !self.regions.is_multi() {
            return Err(ENNError::InvalidParameter(
                "explicit imports currently require a multi-region optimizer".into(),
            ));
        }
        if self.regions.is_multi() && matches!(submission, Submission::Ordered) {
            if matches!(&self.regions, TrustRegionState::Multi { pending: None, .. }) {
                return Err(ENNError::InvalidParameter(
                    "no pending region proposal; use explicit import_observations for external data"
                        .into(),
                ));
            }
            validate_multi(self, x, y)?;
        }
        Ok(())
    }

    fn check_ingestion(&self) -> Result<(), ENNError> {
        if let Some(error) = &self.ingestion_error {
            return Err(ENNError::InvalidParameter(format!(
                "multi-region ingestion failed after mutation may have begun ({error}); do not retry this observation on this optimizer"
            )));
        }
        Ok(())
    }

    pub fn ingestion_failure(&self) -> Option<&str> {
        self.ingestion_error.as_deref()
    }

    fn dispatch_tell(
        mut self,
        x: &ArrayView2<f64>,
        y: &ArrayView2<f64>,
        yvar: Option<&ArrayView2<f64>>,
        rng: &mut dyn RngCore,
        submission: Submission<'_>,
    ) -> Result<Self, ENNError> {
        if self.reinit_left().is_some() {
            return tell_reinit(self, x, y, yvar, rng);
        }
        if self.warmup.in_init() && self.regions.is_multi() {
            return multi::tell_warmup(self, x, y, yvar, rng, submission);
        }
        if self.warmup.in_init() {
            self.warmup.advance_init(x.nrows());
            let mut optimizer = tell_common(self, x, y, yvar, None, rng)?;
            if let Some(models) = optimizer.local_models.take() {
                optimizer.local_models = Some(models.warmup(x, y, yvar, rng)?);
            }
            optimizer.warmup.check_transition();
            if let Some(y_inc) = optimizer.y_scalar().map(|y| y[0]) {
                let obs_count = optimizer.obs_count();
                optimizer.regions = optimizer.regions.seed_baseline(y_inc, obs_count);
            }
            return Ok(optimizer);
        }
        if self.regions.is_multi() {
            return tell_multi(self, x, y, yvar, rng, submission);
        }
        let mut telemetry = std::mem::take(&mut self.telemetry);
        let mut this = tell_turbo(self, x, y, yvar, &mut telemetry, rng)?;
        this.telemetry = telemetry;
        Ok(this)
    }
}

impl Strategy {
    pub fn init_plan(&self) -> (usize, InitStrategy) {
        match self {
            Strategy::Init(state) => (state.num_init, state.strategy_type),
            Strategy::Hybrid { init, .. } => (init.num_init, init.strategy_type),
            Strategy::Experimental(state) => (state.init.num_init, state.init.strategy_type),
            Strategy::Turbo(_) => (0, InitStrategy::LHD),
        }
    }
}

fn ask_reinit(
    optimizer: BayesOptimizer,
    num_arms: usize,
    rng: &mut dyn RngCore,
) -> Result<(Array2<f64>, BayesOptimizer), ENNError> {
    let count = num_arms.max(1);
    let state = InitStrategyState::new(optimizer.init_kind(), count);
    ask_init(&state, optimizer, count, rng)
}

fn tell_reinit(
    optimizer: BayesOptimizer,
    x: &ArrayView2<f64>,
    y: &ArrayView2<f64>,
    yvar: Option<&ArrayView2<f64>>,
    rng: &mut dyn RngCore,
) -> Result<BayesOptimizer, ENNError> {
    let mut optimizer = tell_common(optimizer, x, y, yvar, None, rng)?;
    optimizer = optimizer.consume_reinit(x.nrows());
    if let Some(y_inc) = optimizer.y_scalar().map(|y| y[0]) {
        let obs_count = optimizer.obs_count();
        optimizer.regions = optimizer.regions.seed_baseline(y_inc, obs_count);
    }
    Ok(optimizer)
}

/// Generate initialization candidates in hybrid mode.
#[allow(dead_code)]
fn ask_hybrid(
    state: &InitStrategyState,
    optimizer: BayesOptimizer,
    num_arms: usize,
    rng: &mut dyn RngCore,
) -> Result<(Array2<f64>, BayesOptimizer), ENNError> {
    ask_init(state, optimizer, num_arms, rng)
}

fn observe_morbo(mut optimizer: BayesOptimizer) -> Result<BayesOptimizer, ENNError> {
    if !optimizer.regions.is_morbo() {
        return Ok(optimizer);
    }
    let Some(y_all) = optimizer.y_obs() else {
        return Ok(optimizer);
    };
    if y_all.nrows() == 0 {
        return Ok(optimizer);
    }
    optimizer.regions = optimizer.regions.morbo_only(&y_all.view())?;
    Ok(optimizer)
}

/// Tell for initialization phase.
#[allow(dead_code)]
fn tell_init(
    state: &mut InitStrategyState,
    optimizer: BayesOptimizer,
    x: &ArrayView2<f64>,
    y: &ArrayView2<f64>,
    yvar: Option<&ArrayView2<f64>>,
    rng: &mut dyn RngCore,
) -> Result<BayesOptimizer, ENNError> {
    state.completed += x.nrows();
    tell_common(optimizer, x, y, yvar, None, rng)
}

/// Generate and select candidates for the active single-region TuRBO/MORBO
/// phase.
fn ask_turbo(
    mut optimizer: BayesOptimizer,
    num_arms: usize,
    telemetry: &mut Telemetry,
    rng: &mut dyn RngCore,
) -> Result<(Array2<f64>, BayesOptimizer), ENNError> {
    let (should_rescale, regions) = optimizer.regions.resample_propose(rng);
    optimizer.regions = regions;
    if should_rescale {
        optimizer.update_incumbent(rng)?;
        let y_inc = optimizer.y_scalar().cloned();
        let obs_count = optimizer.obs_count();
        optimizer.regions = optimizer.regions.rescale_morbo(obs_count, y_inc)?;
    }
    optimizer.regions = optimizer.regions.set_arms(num_arms);

    let x_center = optimizer
        .unit_incumbent()
        .cloned()
        .unwrap_or_else(|| optimizer.domain.unit_center());
    let lengthscales = optimizer.surrogate().and_then(|s| s.lengthscales());
    let ls_ref: Option<ArrayView1<f64>> = lengthscales.as_ref().map(|ls| ls.view());
    let (lower_1d, upper_1d) = optimizer
        .regions
        .compute_bounds(&x_center.view(), ls_ref.as_ref());

    let num_dim = optimizer.num_dim();
    let config = optimizer.proposals.config().clone();
    let num_candidates = config.num_candidates(num_dim, num_arms);
    telemetry.num_candidates = num_candidates;
    if optimizer.regions.is_morbo()
        && config.candidate_rv == crate::candidates::CandidateSampler::Sobol
    {
        optimizer = optimizer.reseed_sobol(num_arms)?;
    }

    let is_morbo = optimizer.regions.is_morbo();
    let x_cand_unit = perturb_candidates(
        &x_center.view(),
        &lower_1d,
        &upper_1d,
        num_candidates,
        config.candidate_rv,
        config.coord_system,
        &config.tied_dims,
        is_morbo,
        config.raasp_fast,
        rng,
        optimizer.proposals.sobol_mut(),
        config.num_pert,
    )?;

    let start = std::time::Instant::now();
    let selected = select_arms(&optimizer, &x_cand_unit.view(), num_arms, rng)?;
    telemetry.dt_sel = start.elapsed().as_secs_f64();

    Ok((selected, optimizer))
}

/// Tell for active single-region TuRBO/MORBO phase.
fn tell_turbo(
    optimizer: BayesOptimizer,
    x: &ArrayView2<f64>,
    y: &ArrayView2<f64>,
    yvar: Option<&ArrayView2<f64>>,
    telemetry: &mut Telemetry,
    rng: &mut dyn RngCore,
) -> Result<BayesOptimizer, ENNError> {
    let mut optimizer = tell_common(optimizer, x, y, yvar, Some(telemetry), rng)?;

    let num_obs = optimizer.obs_count();
    let y_incumbent = optimizer
        .y_scalar()
        .ok_or_else(|| ENNError::InvalidParameter("Missing incumbent y".to_string()))?
        .to_owned();
    optimizer.regions = optimizer.regions.set_arms(x.nrows());
    if !optimizer.regions.is_morbo() {
        if optimizer.regions.turbo_obs() == 0 {
            let prev = num_obs.saturating_sub(y.nrows());
            optimizer.regions = optimizer.regions.set_obs(prev);
        }
        optimizer.regions = optimizer
            .regions
            .tell_batch(y, &y_incumbent.view(), num_obs)?;
    }
    if optimizer.regions.needs_restart() && optimizer.regions.is_morbo() {
        optimizer.regions = optimizer.regions.restart(Some(rng));
        optimizer = optimizer.increment_generation();
        optimizer = observe_morbo(optimizer)?;
        optimizer.update_incumbent(rng)?;
        if let Some(y_inc) = optimizer.y_scalar().cloned() {
            optimizer.regions = optimizer.regions.update_morbo(&y_inc.view(), num_obs)?;
        }
    }
    Ok(optimizer)
}

#[cfg(test)]
mod tests_incremental;
#[cfg(test)]
mod tests_init;
#[cfg(test)]
mod tests_morbo;
#[cfg(test)]
mod tests_selection;
