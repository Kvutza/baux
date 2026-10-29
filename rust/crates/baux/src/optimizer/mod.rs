//! BayesOptimizer state machine decomposed into orthogonal scientific roles.

mod incumbent;
mod inspection;
pub mod multi_tr;
pub mod obs_access;
mod observation_delta;
mod tr_state;

pub use incumbent::{IncumbentLedger, IncumbentPoint, ProposalEngine};
pub use inspection::DomainFrame;
pub use multi_tr::{
    MultiTrustRegionConfig, MultiTrustRegionState, RegionBatch, RegionCandidate, SharingPolicy,
};
pub use obs_access::{PosteriorBackend, PosteriorStore};
pub use observation_delta::{ObservationDelta, WarmupMode, WarmupSchedule};
pub use tr_state::TrustRegionState;

use ndarray::{Array1, Array2, ArrayView2};
use rand::RngCore;

use crate::config::{InitStrategy, OptimizerConfig, OptimizerSpec};
use crate::error::ENNError;
use crate::surrogate::{BoxedSurrogate, Surrogate};

/// Strategy state for initialization phase.
#[derive(Debug, Clone)]
pub struct InitStrategyState {
    pub strategy_type: InitStrategy,
    pub num_init: usize,
    pub completed: usize,
}

impl InitStrategyState {
    pub fn new(strategy_type: InitStrategy, num_init: usize) -> Self {
        Self {
            strategy_type,
            num_init,
            completed: 0,
        }
    }
}

/// Strategy state for TuRBO normal phase.
#[derive(Debug, Clone, Default)]
pub struct TurboStrategyState;

/// Strategy state for the experimental multi-trust-region phase.
#[derive(Debug, Clone)]
pub struct StrategyState {
    pub init: InitStrategyState,
    pub multi_tr: MultiTrustRegionState,
    pub in_init: bool,
    pub pending_regions: Option<Vec<usize>>,
}

impl StrategyState {
    pub fn new(
        num_dim: usize,
        num_regions: usize,
        num_init: usize,
        rng: &mut dyn RngCore,
    ) -> Result<Self, ENNError> {
        let mut config = MultiTrustRegionConfig::new(num_regions, Default::default());
        config.sharing_policy = SharingPolicy::Shared;
        let multi_tr = MultiTrustRegionState::new(num_dim, config, None, rng)
            .map_err(|e| ENNError::InvalidParameter(e.to_string()))?;
        Ok(Self {
            init: InitStrategyState::new(InitStrategy::LHD, num_init),
            multi_tr,
            in_init: num_init > 0,
            pending_regions: None,
        })
    }
}

/// Strategy specification builder compiling into `WarmupSchedule` and
/// `TrustRegionState`.
#[derive(Debug, Clone)]
pub enum Strategy {
    /// Initialization-only strategy.
    Init(InitStrategyState),
    /// TuRBO normal strategy.
    Turbo(TurboStrategyState),
    /// Hybrid: initialization then TuRBO.
    Hybrid {
        init: InitStrategyState,
        turbo: TurboStrategyState,
        in_init: bool,
    },
    /// Experimental multi-trust-region strategy.
    Experimental(StrategyState),
}

impl Strategy {
    /// Create a new initialization-only strategy.
    pub fn init(strategy_type: InitStrategy, num_init: usize) -> Self {
        Strategy::Init(InitStrategyState::new(strategy_type, num_init))
    }

    /// Create a new TuRBO strategy.
    pub fn turbo() -> Self {
        Strategy::Turbo(TurboStrategyState)
    }

    /// Create a new experimental multi-trust-region strategy.
    pub fn experimental(
        num_dim: usize,
        num_regions: usize,
        num_init: usize,
        rng: &mut dyn RngCore,
    ) -> Result<Self, ENNError> {
        Ok(Strategy::Experimental(StrategyState::new(
            num_dim,
            num_regions,
            num_init,
            rng,
        )?))
    }

    /// Create a new hybrid strategy.
    pub fn hybrid(init_strategy: InitStrategy, num_init: usize) -> Self {
        Strategy::Hybrid {
            init: InitStrategyState::new(init_strategy, num_init),
            turbo: TurboStrategyState,
            in_init: true,
        }
    }

    /// Get initialization progress if applicable.
    pub fn init_progress(&self) -> Option<(usize, usize)> {
        match self {
            Strategy::Init(state) => Some((state.completed, state.num_init)),
            Strategy::Hybrid {
                init,
                in_init: true,
                ..
            } => Some((init.completed, init.num_init)),
            Strategy::Experimental(state) if state.in_init => {
                Some((state.init.completed, state.init.num_init))
            }
            _ => None,
        }
    }

    pub(crate) fn into_roles(
        self,
    ) -> (
        WarmupSchedule,
        Option<(MultiTrustRegionState, Option<Vec<usize>>)>,
    ) {
        match self {
            Strategy::Init(s) => (
                WarmupSchedule::design_only(s.strategy_type, s.num_init, s.completed),
                None,
            ),
            Strategy::Turbo(_) => (WarmupSchedule::active(InitStrategy::LHD, 0, 0, false), None),
            Strategy::Hybrid { init, in_init, .. } => (
                WarmupSchedule::active(init.strategy_type, init.num_init, init.completed, in_init),
                None,
            ),
            Strategy::Experimental(s) => (
                WarmupSchedule::active(
                    s.init.strategy_type,
                    s.init.num_init,
                    s.init.completed,
                    s.in_init,
                ),
                Some((s.multi_tr, s.pending_regions)),
            ),
        }
    }
}

/// Telemetry for timing.
#[derive(Debug, Clone, Default)]
pub struct Telemetry {
    pub dt_fit: f64,
    pub dt_gen: f64,
    pub dt_sel: f64,
    pub dt_tell: f64,
    pub num_candidates: usize,
}

/// Role-decomposed Bayesian optimizer state machine.
pub struct BayesOptimizer {
    pub(crate) domain: DomainFrame,
    pub(crate) config: OptimizerConfig,
    pub(crate) warmup: WarmupSchedule,
    pub(crate) posterior: PosteriorStore,
    pub(crate) incumbent: IncumbentLedger,
    pub(crate) regions: TrustRegionState,
    pub(crate) proposals: ProposalEngine,
    pub(crate) telemetry: Telemetry,
}

impl BayesOptimizer {
    /// Create a new optimizer.
    pub fn new(
        bounds: Array2<f64>,
        config: OptimizerConfig,
        rng: &mut dyn RngCore,
    ) -> Result<Self, ENNError> {
        Self::new_strategy(bounds, config, Strategy::hybrid(InitStrategy::LHD, 10), rng)
    }

    /// Create a new optimizer directly from a closed v4 [`OptimizerSpec`].
    pub fn from_spec(
        bounds: Array2<f64>,
        spec: &OptimizerSpec,
        num_init: usize,
        rng: &mut dyn RngCore,
    ) -> Result<Self, ENNError> {
        let config = OptimizerConfig::from_spec(spec)?;
        let strategy = if num_init > 0 {
            Strategy::hybrid(InitStrategy::LHD, num_init)
        } else {
            Strategy::turbo()
        };
        Self::new_strategy(bounds, config, strategy, rng)
    }

    /// Create a new optimizer with an explicit strategy.
    pub fn new_strategy(
        bounds: Array2<f64>,
        config: OptimizerConfig,
        strategy: Strategy,
        rng: &mut dyn RngCore,
    ) -> Result<Self, ENNError> {
        Self::new_inner(bounds, config, strategy, None, rng)
    }

    /// Create an optimizer whose surrogate is supplied by an external frontend.
    pub fn new_surrogate(
        bounds: Array2<f64>,
        config: OptimizerConfig,
        strategy: Strategy,
        surrogate: BoxedSurrogate,
        rng: &mut dyn RngCore,
    ) -> Result<Self, ENNError> {
        Self::new_inner(bounds, config, strategy, Some(surrogate), rng)
    }

    fn new_inner(
        bounds: Array2<f64>,
        config: OptimizerConfig,
        strategy: Strategy,
        provided_surrogate: Option<BoxedSurrogate>,
        rng: &mut dyn RngCore,
    ) -> Result<Self, ENNError> {
        let domain = DomainFrame::new(bounds)?;
        let num_dim = domain.num_dim();
        let (warmup, multi_override) = strategy.into_roles();

        let mut regions = match multi_override {
            Some((multi_tr, pending)) => TrustRegionState::multi(multi_tr, pending),
            None => TrustRegionState::from_config(num_dim, &config.trust_region, rng)
                .map_err(|e| ENNError::InvalidParameter(e.to_string()))?,
        };
        if let Some(dim) = config.failure_tolerance_dim {
            if let TrustRegionState::Turbo(t) = &mut regions {
                t.set_dim(dim);
            }
        }

        let posterior = PosteriorStore::new(&config.surrogate, provided_surrogate);
        let proposals = ProposalEngine::new(num_dim, config.candidates.clone(), rng)?;
        let incumbent = IncumbentLedger::new(&config, regions.num_metrics());

        Ok(Self {
            domain,
            config,
            warmup,
            posterior,
            incumbent,
            regions,
            proposals,
            telemetry: Telemetry::default(),
        })
    }

    /// Add observations (internal).
    pub fn add_observations(
        &mut self,
        x: &ArrayView2<f64>,
        y: &ArrayView2<f64>,
    ) -> Result<ObservationDelta, ENNError> {
        let delta = self.prepare_observations(x, y)?;
        self.commit_observations(&delta);
        Ok(delta)
    }

    pub(crate) fn prepare_observations(
        &self,
        x: &ArrayView2<f64>,
        y: &ArrayView2<f64>,
    ) -> Result<ObservationDelta, ENNError> {
        if x.nrows() != y.nrows() {
            return Err(ENNError::InvalidShape {
                expected: vec![x.nrows(), y.ncols()],
                got: vec![y.nrows(), y.ncols()],
            });
        }
        let old_n = self.posterior.obs_count();
        observation_delta::observation_batch(old_n, x, y)
    }

    pub(crate) fn commit_observations(&mut self, delta: &ObservationDelta) {
        for i in 0..delta.x_new.nrows() {
            let y_row: Array1<f64> = delta.y_new.row(i).to_owned();
            self.incumbent.tell_row(delta.old_n + i, &y_row);
            self.posterior
                .push_empirical(delta.x_new.row(i).to_owned(), y_row);
        }
    }
}

#[cfg(test)]
mod tests;
