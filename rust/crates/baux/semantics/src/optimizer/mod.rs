//! Execution-target-neutral optimizer semantics for BAUX.
//!
//! Provides modular optimizer components:
//! - `surrogate`: local precision-weighted neighbor value estimation
//!   (`Surrogate` / `NeighborValueEstimator`)
//! - `trust_region`: stochastic trust-region model agreement and radius control
//!   (`TrustRegionController`)

pub mod calibration;
pub mod core {
    pub use super::surrogate::ValuePrediction;
}
pub mod param_tree;
pub mod provenance;
mod response;
pub mod reuse;
pub use reuse::{antipodal_slot, conditioned_sign, conditioned_slot};
pub mod settings {
    pub use super::surrogate::SurrogateSpec;
}
pub mod surrogate;
pub mod transaction;
pub mod trust_region;
pub use trust_region as reliability;
pub mod window_plan;
pub use response::response_rms;
pub use window_plan::{WindowPlan, advance_clock, verify_coverage, verify_full};

// Canonical exports directly at crate root
pub use param_tree::{ParamFamily, ParamTree};
pub use provenance::{ChainedObservation, PairedImprovement, WorldId, WorldMismatchError};
pub use surrogate::{
    AcquisitionRule, DistanceMetric, FitState, NeighborValueEstimator, PoolSelection, Surrogate,
    SurrogateSpec, UncertaintyLaw, ValuePrediction,
};
pub use transaction::{AskPlan, EvalReceipt, Evaluated, RoundId, TellDecision};
pub use trust_region::{
    CANDIDATES, INITIAL_OBSERVATIONS, MAX_HISTORY, ModelAgreement, PoolShape, RadiusAction,
    RadiusControl, RadiusController, ReliabilityAction, ReliabilityDecision, ReliabilityEvidence,
    ReliabilityPolicy, ReliabilityTelemetry, StepDecision, TrialStep, TrustRegionController,
    TrustRegionSpec, TrustRegionStatus, accept_bits, accept_candidate, accept_decision,
    candidate_seed, decide_action, splitmix64,
};
