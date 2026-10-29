//! Stochastic trust-region step acceptance, model agreement, and radius
//! control.

pub mod agreement;
pub mod controller;
pub mod pool;

pub const CANDIDATES: usize = 4;
pub const INITIAL_OBSERVATIONS: usize = 32;
pub use baux::semantics::neighbor::MAX_HISTORY;

pub use agreement::{
    ModelAgreement, RadiusControl, ReliabilityEvidence, ReliabilityPolicy, ReliabilityTelemetry,
    TrialStep, TrustRegionSpec, TrustRegionStatus,
};
pub use baux::semantics::optimizer::surrogate::{
    AcquisitionRule, DistanceMetric, FitState, NeighborValueEstimator, PoolSelection, Surrogate,
    SurrogateSpec, UncertaintyLaw, ValuePrediction,
};
pub use controller::{
    RadiusAction, RadiusController, ReliabilityAction, ReliabilityDecision, StepDecision,
    TrustRegionController, accept_bits, accept_candidate, accept_decision, decide_action,
};
pub use pool::{PoolShape, candidate_seed, splitmix64};

#[cfg(test)]
mod tests;
