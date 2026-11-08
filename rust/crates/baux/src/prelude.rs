//! Curated stable Rust API surface.
//!
//! Lower-level native, packed-weight, and accelerator-frontier APIs live under
//! [`crate::experimental`].

pub use crate::acquisition::{
    AcquisitionError, ParetoAcquisition, RandomAcquisition, ThompsonAcquisition, UCBAcquisition,
};
pub use crate::backend::{DiskStore, EnnStorage, MemoryStore, Storage, Store};
pub use crate::candidates::{
    CandidateSampler, from_unit, generate_candidates, generate_lhd, to_unit,
};
pub use crate::config::{
    AcquisitionConfig, CandidateConfig, DataRecipe, DistanceMetric, ExperimentKind, ExperimentSpec,
    InitStrategy, ModelArchitecture, ObjectiveReference, OptimizerConfig, SurrogateConfig,
    TrustRegionKind, lhd_only, turbo_enn, turbo_zero,
};
pub use crate::device::{Device, DeviceMemory};
pub use crate::device_posterior::{Posterior, PosteriorDraw};
pub use crate::draw::{Candidates, ConditionalDraw, DrawInternals, NeighborData};
pub use crate::error::{ENNError, EPS_VAR};
pub use crate::file_config::{IndexConfig, TreeConfig, index_config};
pub use crate::fit::{row_loglik, subsample_loglik, subsample_model};
pub use crate::fitter::ENNFitter;
pub use crate::hash::{Prng, domain_prng, domain_seed, prng, splitmix64, stream_prng, stream_seed};
pub use crate::hypervolume::{hypervolume_max, hypervolume2d_max};
pub use crate::index::{Index, IndexDriver, IndexKind};
pub use crate::model::{ENN, ModelOptions};
pub use crate::optimizer::{BayesOptimizer, Telemetry};
pub use crate::optimizer_factory::{create_lhd, create_optimizer, enn_optimizer};
pub use crate::params::{ENNNormal, ENNParams, ParamsError, PosteriorFlags};
pub use crate::pipeline::{
    AcqStage, Active, InitStage, ModelStage, PerturbStage, Pipeline, ProposalTicket,
    TypedOptimizer, TypedSurrogate, Unfitted, Warmup,
};
pub use crate::posterior::{WeightedPosteriorData, compute_internals, conditional_internals};
pub use crate::scratch::{AlignedBuffer, CandidateSoa, ScratchArena, UnifiedBuffer};
pub use crate::sense::{
    BetterScore, EvalId, Lexicographic, Maximize, Minimize, ObjectiveSense, ObsId, RegionId, Sense,
    Target,
};
pub use crate::stats::WeightedStats;
pub use crate::strategy::Strategy;
pub use crate::surrogate::{ENNSurrogate, ENNSurrogateConfig, Surrogate, SurrogatePrediction};
pub use crate::surrogate_api::{
    IncrementalSurrogate, LengthscaleProvider, ObservationView, Predictor,
};
pub use crate::traits::PosteriorComputation;
pub use crate::trust_region::{
    NoTrustRegion, TRLengthConfig, TrustRegionConfig, TrustRegionError, TurboTrustRegion,
};
pub use crate::util::{argmax_tie, pareto2d_max, sobol_indices, standardize_y};
