//! Stochastic trust-region step acceptance and TuRBO radius control.

pub mod controller;
pub mod pool;

pub const CANDIDATES: usize = 4;
pub use baux::semantics::neighbor::MAX_HISTORY;

pub use baux::semantics::optimizer::surrogate::{
    AcquisitionRule, DistanceMetric, FitState, INITIAL_OBSERVATIONS, NeighborValueEstimator,
    PoolSelection, Surrogate, SurrogateSpec, UncertaintyLaw, ValuePrediction,
};
pub use controller::{
    RadiusAction, RadiusController, StepDecision, TrustRegionController, TurboState, accept_bits,
    accept_candidate, accept_decision, turbo_adapt, turbo_tolerance,
};
pub use pool::{PoolShape, candidate_seed, splitmix64};

#[cfg(test)]
mod tests;
