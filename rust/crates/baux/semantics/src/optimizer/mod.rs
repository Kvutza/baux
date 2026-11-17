//! Backend-neutral optimizer semantics for baux.
//!
//! Provides modular `#![no_std]`-compatible optimizer primitives and host
//! adapters:
//! - `surrogate`: local precision-weighted neighbor value estimation, streaming
//!   `TopK`, slice LOO, and NaN-safe acquisition rules
//! - `trust_region`: stochastic step acceptance, TuRBO adaptation, and
//!   defunctionalized `RoundPlan`

#[cfg(feature = "cuda-host")]
pub use baux_cuda_device::semantics::optimizer::{
    core, param_tree, portfolio, provenance, settings, surrogate, transaction, trust_region,
};

#[cfg(not(feature = "cuda-host"))]
pub mod core {
    pub use super::surrogate::ValuePrediction;
}
#[cfg(not(feature = "cuda-host"))]
pub mod param_tree;
#[cfg(not(feature = "cuda-host"))]
pub mod portfolio;
#[cfg(not(feature = "cuda-host"))]
pub mod provenance;
#[cfg(not(feature = "cuda-host"))]
pub mod settings {
    pub use super::surrogate::SurrogateSpec;
}
#[cfg(not(feature = "cuda-host"))]
pub mod surrogate;
#[cfg(not(feature = "cuda-host"))]
pub mod transaction;
#[cfg(not(feature = "cuda-host"))]
pub mod trust_region;

pub use param_tree::{ParamFamily, ParamTree};
pub use portfolio::{
    Allocation, AllocationPolicy, Portfolio, RegionBatch, RegionController, RegionId,
    RegionSlotState, RegionTicket, RegionalError, RegionalLedger, RestartCenterPolicy,
    RestartCenterSelection, RestartMemoryPolicy, SharingPolicy, contains_ball, contains_box,
    joint_select, maximin_idx, retain_mask,
};
pub use provenance::{ChainedObservation, PairedImprovement, WorldId, WorldMismatchError};
pub use surrogate::{
    AcquisitionRule, BetterScore, DEFAULT_BETA, Dissimilarity, DistanceMetric, FitState,
    MetricKind, NeighborValueEstimator, PoolSelection, Posterior, Surrogate, SurrogateSpec, TopK,
    UncertaintyLaw, ValuePrediction, loo_point, loo_slice, pareto_dom, scale_prior, score_thompson,
    score_ucb, select_pareto, select_random, select_topk,
};
pub use transaction::{AskPlan, EvalReceipt, Evaluated, RoundId, TellDecision};
pub use trust_region::{
    INITIAL_OBSERVATIONS, LifecyclePhase, MAX_HISTORY, NoTrustState, PoolSampler, PoolShape,
    RadiusAction, RadiusController, RegionPlan, RoundPlan, StepDecision, SurrogateMode,
    TrustRegionController, TurboState, WarmupKind, accept_bits, accept_candidate, accept_decision,
    arm_seed, candidate_seed, default_pool, legacy_seed, splitmix64, stream_word, turbo_adapt,
    turbo_tolerance,
};
