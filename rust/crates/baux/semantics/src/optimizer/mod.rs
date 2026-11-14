//! Backend-neutral optimizer semantics for BAUX.
//!
//! Provides modular optimizer components:
//! - `surrogate`: local precision-weighted neighbor value estimation
//!   (`Surrogate` / `NeighborValueEstimator`)
//! - `trust_region`: stochastic trust-region model agreement and radius control
//!   (`TrustRegionController`)

#[cfg(feature = "cuda-host")]
pub use baux_cuda_device::semantics::optimizer::param_tree;
#[cfg(not(feature = "cuda-host"))]
pub mod param_tree;
pub use param_tree::{ParamFamily, ParamTree};

#[cfg(not(feature = "cuda-device"))]
pub mod core {
    pub use super::surrogate::ValuePrediction;
}
#[cfg(not(feature = "cuda-device"))]
pub mod portfolio;
#[cfg(not(feature = "cuda-device"))]
pub mod provenance;
#[cfg(not(feature = "cuda-device"))]
pub mod settings {
    pub use super::surrogate::SurrogateSpec;
}
#[cfg(not(feature = "cuda-device"))]
pub mod surrogate;
#[cfg(not(feature = "cuda-device"))]
pub mod transaction;
#[cfg(not(feature = "cuda-device"))]
pub mod trust_region;

#[cfg(not(feature = "cuda-device"))]
pub use portfolio::{
    Allocation, AllocationPolicy, RegionBatch, RegionController, RegionId, RegionTicket,
    RegionalError, RegionalLedger, RestartCenterPolicy, RestartCenterSelection,
    RestartMemoryPolicy, SharingPolicy, contains_ball, contains_box,
};
#[cfg(not(feature = "cuda-device"))]
pub use provenance::{ChainedObservation, PairedImprovement, WorldId, WorldMismatchError};
#[cfg(not(feature = "cuda-device"))]
pub use surrogate::{
    AcquisitionRule, Dissimilarity, DistanceMetric, FitState, NeighborValueEstimator,
    PoolSelection, Surrogate, SurrogateSpec, UncertaintyLaw, ValuePrediction,
};
#[cfg(not(feature = "cuda-device"))]
pub use transaction::{AskPlan, EvalReceipt, Evaluated, RoundId, TellDecision};
#[cfg(not(feature = "cuda-device"))]
pub use trust_region::{
    CANDIDATES, INITIAL_OBSERVATIONS, MAX_HISTORY, PoolShape, RadiusAction, RadiusController,
    StepDecision, TrustRegionController, TurboState, accept_bits, accept_candidate,
    accept_decision, arm_seed, candidate_seed, legacy_seed, splitmix64, stream_word, turbo_adapt,
    turbo_tolerance,
};
