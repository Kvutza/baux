//! Stochastic trust-region step acceptance, TuRBO radius control, and
//! defunctionalized round planning.

pub mod controller;
pub mod plan;
pub mod pool;

pub use baux::semantics::neighbor::MAX_HISTORY;

pub use baux::semantics::optimizer::surrogate::{
    AcquisitionRule, BetterScore, DEFAULT_BETA, DistanceMetric, FitState, INITIAL_OBSERVATIONS,
    MetricKind, NeighborValueEstimator, PoolSelection, Posterior, Surrogate, SurrogateSpec, TopK,
    UncertaintyLaw, ValuePrediction,
};
pub use controller::{
    NoTrustState, RadiusAction, RadiusController, StepDecision, TrustRegionController, TurboState,
    accept_bits, accept_candidate, accept_decision, turbo_adapt, turbo_tolerance,
};
pub use plan::{
    LifecyclePhase, PoolSampler, RegionPlan, RoundPlan, SurrogateMode, WarmupKind, default_pool,
};
pub use pool::{PoolShape, arm_seed, candidate_seed, legacy_seed, splitmix64, stream_word};

#[cfg(test)]
mod tests;
