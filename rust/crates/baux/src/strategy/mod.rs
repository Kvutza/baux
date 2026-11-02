//! Optimization execution strategies for the role-decomposed ask/tell state
//! machine.

use ndarray::{Array2, ArrayView1, ArrayView2};
use rand::RngCore;

mod multi;
mod selection;
use multi::{ask_init, ask_multi, tell_common, tell_multi};
use selection::*;

use crate::candidates::draw_turbo;
use crate::config::InitStrategy;
use crate::error::ENNError;
use crate::optimizer::{BayesOptimizer, Telemetry};
pub use crate::optimizer::{InitStrategyState, Strategy, StrategyState, TurboStrategyState};
use crate::optimizer_dispatch::{AskGate, TellGate};

impl BayesOptimizer {
    /// Public-boundary ask in natural coordinates.
    pub fn ask_natural(
        &mut self,
        num_arms: usize,
        rng: &mut dyn RngCore,
    ) -> Result<Array2<f64>, ENNError> {
        let unit = self.ask(num_arms, rng)?;
        Ok(self.domain.from_unit(&unit.view()))
    }

    /// Public-boundary tell in natural coordinates.
    pub fn tell_natural(
        &mut self,
        x: &ArrayView2<f64>,
        y: &ArrayView2<f64>,
        yvar: Option<&ArrayView2<f64>>,
        rng: &mut dyn RngCore,
    ) -> Result<(), ENNError> {
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

    /// Ask for candidate points in unit coordinates `[0, 1]^d`.
    pub fn ask(&mut self, num_arms: usize, rng: &mut dyn RngCore) -> Result<Array2<f64>, ENNError> {
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
            AskGate::RestartLocal => self.restart_local()?,
            AskGate::Dispatch => {}
        }
        let start = std::time::Instant::now();
        let result = self.dispatch_ask(num_arms, rng);
        self.telemetry.dt_gen = start.elapsed().as_secs_f64();
        if result.is_ok() {
            self.posterior.schedule_flush()?;
        }
        result
    }

    fn dispatch_ask(
        &mut self,
        num_arms: usize,
        rng: &mut dyn RngCore,
    ) -> Result<Array2<f64>, ENNError> {
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
        let result = ask_turbo(self, num_arms, &mut telemetry, rng);
        self.telemetry = telemetry;
        result
    }

    /// Tell observations.
    pub fn tell(
        &mut self,
        x: &ArrayView2<f64>,
        y: &ArrayView2<f64>,
        rng: &mut dyn RngCore,
    ) -> Result<(), ENNError> {
        self.tell_variance(x, y, None, rng)
    }

    /// Tell observations with optional known observation variance.
    pub fn tell_variance(
        &mut self,
        x: &ArrayView2<f64>,
        y: &ArrayView2<f64>,
        yvar: Option<&ArrayView2<f64>>,
        rng: &mut dyn RngCore,
    ) -> Result<(), ENNError> {
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
            TellGate::Empty => return Ok(()),
            TellGate::Dispatch => {}
        }
        self.posterior.check_variance(yvar.is_some())?;
        let start = std::time::Instant::now();
        self.posterior.wait_flush()?;
        let result = self.dispatch_tell(x, y, yvar, rng);
        self.telemetry.dt_tell = start.elapsed().as_secs_f64();
        if result.is_ok() && x.nrows() < 64 {
            self.posterior.schedule_flush()?;
        }
        result
    }

    fn dispatch_tell(
        &mut self,
        x: &ArrayView2<f64>,
        y: &ArrayView2<f64>,
        yvar: Option<&ArrayView2<f64>>,
        rng: &mut dyn RngCore,
    ) -> Result<(), ENNError> {
        if self.reinit_left().is_some() {
            return tell_reinit(self, x, y, yvar, rng);
        }
        if self.regions.is_multi() {
            return tell_multi(self, x, y, yvar, rng);
        }
        if self.warmup.in_init() {
            self.warmup.advance_init(x.nrows());
            tell_common(self, x, y, yvar, None, rng)?;
            self.warmup.check_transition();
            return Ok(());
        }
        let mut telemetry = std::mem::take(&mut self.telemetry);
        let result = tell_turbo(self, x, y, yvar, &mut telemetry, rng);
        self.telemetry = telemetry;
        result
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
    optimizer: &mut BayesOptimizer,
    num_arms: usize,
    rng: &mut dyn RngCore,
) -> Result<Array2<f64>, ENNError> {
    let left = optimizer.reinit_left().unwrap_or(num_arms);
    let count = num_arms.min(left).max(1);
    let state = InitStrategyState::new(optimizer.init_kind(), count);
    ask_init(&state, optimizer, count, rng)
}

fn tell_reinit(
    optimizer: &mut BayesOptimizer,
    x: &ArrayView2<f64>,
    y: &ArrayView2<f64>,
    yvar: Option<&ArrayView2<f64>>,
    rng: &mut dyn RngCore,
) -> Result<(), ENNError> {
    tell_common(optimizer, x, y, yvar, None, rng)?;
    optimizer.consume_reinit(x.nrows());
    Ok(())
}

/// Generate initialization candidates in hybrid mode.
#[allow(dead_code)]
fn ask_hybrid(
    state: &InitStrategyState,
    optimizer: &mut BayesOptimizer,
    num_arms: usize,
    rng: &mut dyn RngCore,
) -> Result<Array2<f64>, ENNError> {
    ask_init(state, optimizer, num_arms, rng)
}

fn observe_morbo(optimizer: &mut BayesOptimizer) -> Result<(), ENNError> {
    if !optimizer.regions.is_morbo() {
        return Ok(());
    }
    let Some(y_all) = optimizer.y_obs() else {
        return Ok(());
    };
    if y_all.nrows() == 0 {
        return Ok(());
    }
    optimizer.regions.morbo_only(&y_all.view())
}

/// Tell for initialization phase.
#[allow(dead_code)]
fn tell_init(
    state: &mut InitStrategyState,
    optimizer: &mut BayesOptimizer,
    x: &ArrayView2<f64>,
    y: &ArrayView2<f64>,
    yvar: Option<&ArrayView2<f64>>,
    rng: &mut dyn RngCore,
) -> Result<(), ENNError> {
    state.completed += x.nrows();
    tell_common(optimizer, x, y, yvar, None, rng)
}

/// Generate and select candidates for the active single-region TuRBO/MORBO
/// phase.
fn ask_turbo(
    optimizer: &mut BayesOptimizer,
    num_arms: usize,
    telemetry: &mut Telemetry,
    rng: &mut dyn RngCore,
) -> Result<Array2<f64>, ENNError> {
    optimizer.regions.resample_propose(rng);
    optimizer.regions.set_arms(num_arms);

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
        optimizer.reseed_sobol(num_arms)?;
    }

    let is_morbo = optimizer.regions.is_morbo();
    let x_cand_unit = draw_turbo(
        is_morbo,
        config.candidate_rv,
        config.raasp_fast,
        &x_center.view(),
        &lower_1d,
        &upper_1d,
        num_candidates,
        rng,
        optimizer.proposals.sobol_mut(),
        config.num_pert,
    )?;

    let start = std::time::Instant::now();
    let selected = select_arms(optimizer, &x_cand_unit.view(), num_arms, rng)?;
    telemetry.dt_sel = start.elapsed().as_secs_f64();

    Ok(selected)
}

fn seed_history(optimizer: &mut BayesOptimizer, previous: usize) {
    if previous == 0 {
        return;
    }
    let Some(y_all) = optimizer.y_obs() else {
        return;
    };
    if y_all.ncols() != 1 || y_all.nrows() < previous {
        return;
    }
    let prefix = y_all.column(0).slice(ndarray::s![..previous]).to_owned();
    optimizer.regions.seed_history(&prefix.view());
}

/// Tell for active single-region TuRBO/MORBO phase.
fn tell_turbo(
    optimizer: &mut BayesOptimizer,
    x: &ArrayView2<f64>,
    y: &ArrayView2<f64>,
    yvar: Option<&ArrayView2<f64>>,
    telemetry: &mut Telemetry,
    rng: &mut dyn RngCore,
) -> Result<(), ENNError> {
    tell_common(optimizer, x, y, yvar, Some(telemetry), rng)?;

    let num_obs = optimizer.obs_count();
    let y_incumbent = optimizer
        .y_scalar()
        .ok_or_else(|| ENNError::InvalidParameter("Missing incumbent y".to_string()))?
        .to_owned();
    optimizer.regions.set_arms(x.nrows());
    if !optimizer.regions.is_morbo() {
        if optimizer.regions.turbo_obs() == 0 {
            let prev = num_obs.saturating_sub(y.nrows());
            seed_history(optimizer, prev);
            optimizer.regions.set_obs(prev);
        }
        optimizer
            .regions
            .tell_batch(y, &y_incumbent.view(), num_obs)?;
    }
    if optimizer.regions.needs_restart() && optimizer.regions.is_morbo() {
        optimizer.regions.restart(Some(rng));
        optimizer.increment_generation();
        observe_morbo(optimizer)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests_incremental;
#[cfg(test)]
mod tests_init;
#[cfg(test)]
mod tests_morbo;
#[cfg(test)]
mod tests_selection;
