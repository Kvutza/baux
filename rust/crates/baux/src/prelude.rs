//! Curated stable Rust API surface.
//!
//! Lower-level Rust, packed-weight, and accelerator-frontier APIs live under
//! [`crate::experimental`].

pub use crate::acquisition::{
    AcquisitionError, ParetoAcquisition, RandomAcquisition, ThompsonAcquisition, UCBAcquisition,
};
pub use crate::candidates::{
    CandidateSampler, from_unit, generate_candidates, generate_lhd, to_unit,
};
pub use crate::config::{
    AcquisitionConfig, CandidateConfig, DataRecipe, DistanceScaling, ExperimentKind,
    ExperimentSpec, InitStrategy, ModelArchitecture, ObjectiveReference, OptimizerConfig,
    SurrogateConfig, TrustRegionKind, lhd_only, turbo_enn, turbo_zero,
};
pub use crate::draw::{Candidates, ConditionalDraw, DrawInternals, NeighborData};
pub use crate::error::{ENNError, EPS_VAR};
pub use crate::fit::{row_loglik, subsample_loglik, subsample_model};
pub use crate::fitter::ENNFitter;
pub use crate::hypervolume::{hypervolume_max, hypervolume2d_max};
pub use crate::index::IndexDriver;
pub use crate::model::{ENN, ModelOptions};
pub use crate::optimizer::{BayesOptimizer, Telemetry};
pub use crate::optimizer_factory::{create_lhd, create_optimizer, enn_optimizer};
pub use crate::params::{ENNNormal, ENNParams, ParamsError, PosteriorFlags};
pub use crate::posterior::{WeightedPosteriorData, compute_internals, conditional_internals};
pub use crate::stats::WeightedStats;
pub use crate::store::EnnStorage;
pub use crate::strategy::Strategy;
pub use crate::surrogate::{ENNSurrogate, ENNSurrogateConfig, Surrogate, SurrogatePrediction};
pub use crate::traits::PosteriorComputation;
pub use crate::trust_region::{
    NoTrustRegion, TRLengthConfig, TrustRegionConfig, TrustRegionError, TurboTrustRegion,
};
pub use crate::util::{argmax_tie, pareto2d_max, sobol_indices, standardize_y};
