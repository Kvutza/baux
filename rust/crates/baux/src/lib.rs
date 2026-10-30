//! Core neighbor-surrogate optimization algorithms in Rust.
//!
//! This crate owns the optimizer semantics shared by the Python API and the
//! accelerator backends.

#![allow(clippy::pedantic, clippy::nursery, clippy::cargo)]

extern crate self as baux;

pub mod acquisition;
pub mod agent_contract;
#[cfg(all(any(target_os = "macos", target_os = "ios"), feature = "metal"))]
#[allow(dead_code)]
mod apple_gpu;
#[path = "../artifact/src/lib.rs"]
pub mod artifact;
pub use artifact as baux_artifact;
pub mod base_policy;
#[path = "../neighbor-index/src/lib.rs"]
pub mod baux_neighbor_index;
pub mod store;
pub use baux_neighbor_index as neighbor_index;
#[cfg(all(feature = "cuda", target_os = "linux", target_arch = "x86_64"))]
#[allow(dead_code)]
mod resident_cuda;
#[cfg(all(any(target_os = "macos", target_os = "ios"), feature = "metal"))]
#[allow(dead_code)]
mod resident_metal;
pub use baux::semantics::optimizer::SurrogateSpec;
pub use baux::semantics::optimizer::calibration;
pub mod calibrator;
pub mod candidates;
mod candidates_fast;
// Compile and execute the backend-independent CUDA mHC4 controller tests on
// Metal development hosts too. CUDA device kernels remain NVIDIA-only, while
// streams, posterior updates, ENN fitting, acceptance, and radius trajectories
// are verified in the ordinary cross-backend test suite.
pub mod capability;
#[cfg(any(
    test,
    all(any(target_os = "macos", target_os = "ios"), feature = "metal")
))]
mod code_overlap;
pub mod coding_outcome;
pub mod config;
mod coordinate;
mod coordinate_flat;
mod coordinate_scalar;
pub mod device_posterior;
pub use baux::semantics::context;
#[cfg(all(any(target_os = "macos", target_os = "ios"), feature = "metal"))]
mod context_metal;
mod dense;
pub mod disk_bpann;
pub mod draw;
pub mod error;
pub mod experimental;
#[cfg(feature = "extraction")]
pub mod extraction;
pub mod fbt;
#[cfg(all(any(target_os = "macos", target_os = "ios"), feature = "metal"))]
#[allow(dead_code)]
mod fbt_attention;
#[cfg(all(any(target_os = "macos", target_os = "ios"), feature = "metal"))]
#[allow(dead_code)]
mod fbt_metal;
#[cfg(all(any(target_os = "macos", target_os = "ios"), feature = "metal"))]
mod fbt_micro;
#[cfg(all(any(target_os = "macos", target_os = "ios"), feature = "metal"))]
#[allow(dead_code)]
mod fbt_model;
#[cfg(all(any(target_os = "macos", target_os = "ios"), feature = "metal"))]
#[allow(dead_code)]
mod fbt_moe;
#[cfg(all(any(target_os = "macos", target_os = "ios"), feature = "metal"))]
mod fbt_mps;
#[cfg(all(any(target_os = "macos", target_os = "ios"), feature = "metal"))]
#[allow(dead_code)]
mod fbt_pisa1;
#[cfg(all(any(target_os = "macos", target_os = "ios"), feature = "metal"))]
#[path = "fbt_pisa1/grad.rs"]
mod fbt_pisa1_grad;
pub mod file_config;
pub mod fit;
pub mod fitter;
#[cfg(all(any(target_os = "macos", target_os = "ios"), feature = "metal"))]
mod flame_metal;
#[cfg(all(any(target_os = "macos", target_os = "ios"), feature = "metal"))]
mod forward_metal;
#[allow(dead_code)]
mod forward_program;
mod forward_weights;
pub mod hash;
#[cfg(all(any(target_os = "macos", target_os = "ios"), feature = "metal"))]
pub mod hnet_metal;
pub mod hypervolume;
pub mod incumbent_tracker;
pub mod index;
mod knn;
pub mod metric_auto;
pub mod metric_loo;
mod metric_rng;
#[allow(dead_code)]
mod metric_rows;
mod metric_seed;
mod metric_sobol;
pub mod metric_weights;
pub mod model;
pub mod morbo;
pub mod muon;
pub mod ndtri;
#[cfg(all(any(target_os = "macos", target_os = "ios"), feature = "metal"))]
#[allow(dead_code)]
mod nn;
pub use baux::semantics::neighbor as neighbor_rule;
pub mod objective_observation;
pub mod optimizer;
mod optimizer_config;
mod optimizer_dispatch;
pub mod optimizer_factory;
mod optimizer_mu;
pub mod params;
pub mod posterior;
pub mod prelude;
mod pretrain_data;
pub mod procedural_pool;
pub mod proposal_kernel;
mod quantization;
#[cfg(all(any(target_os = "macos", target_os = "ios"), feature = "metal"))]
mod qwen_metal;
#[cfg(any(test, all(target_os = "macos", feature = "metal")))]
mod reconstruction;
pub mod reliability_region;
pub mod response_observation;
pub mod search;
#[path = "../semantics/src/lib.rs"]
pub mod semantics;
pub use semantics as baux_semantics;
pub mod stats;
pub mod strategy;
pub mod surrogate;
mod surrogate_api;
#[cfg(all(any(target_os = "macos", target_os = "ios"), feature = "metal"))]
#[allow(dead_code)]
mod tensor;
pub mod tensor_store;
pub mod text;
pub mod traits;
mod trials;
pub mod trust_region;
pub mod util;
mod weights;
pub mod y_bounds;

#[cfg(test)]
pub(crate) mod test_helpers;

pub use acquisition::{
    AcquisitionError, ParetoAcquisition, RandomAcquisition, ThompsonAcquisition, UCBAcquisition,
};
pub use agent_contract::{
    AGENT_SCHEMA, AgentMessage, AgentTranscript, AssistantSegment, DataSplit, EpisodeTermination,
    TaskProvenance,
};
pub use base_policy::{BasePolicyManifest, POLICY_SCHEMA, PolicyQualification};
pub use calibrator::{
    AffineCalibrator, CalibrationMode, Calibrator, PosteriorCalibrator, ScaleCalibrator,
    fit_calibrator,
};
pub use candidates::{
    CandidateRV, CandidateSampler, draw_turbo, from_unit, generate_candidates, generate_lhd,
    generate_uniform, sobol_masked, sobol_sequence, to_unit,
};
pub use capability::{
    Capability, ExecutionTarget, Operation, Support, execution_targets, matrix, operations, support,
};
pub use coding_outcome::{
    CODING_SCHEMA, CandidateEfficiency, CandidateEligibility, CodingOutcome, ExecutableChecks,
};
pub use config::{
    AcquisitionConfig, CandidateConfig, DataRecipe, DistanceScaling, ExperimentKind,
    ExperimentSpec, InitStrategy, ModelArchitecture, ObjectiveReference, OptimizerConfig,
    ResidentEnnConfig, SurrogateConfig, TrustRegionKind, lhd_only, turbo_enn, turbo_zero,
};
pub use draw::{Candidates, ConditionalDraw, DrawInternals, NeighborData};
pub use error::{ENNError, EPS_VAR};
pub use file_config::{BpannConfig, Config, ConfigFile, bpann_config, config_path, set_path};
pub use fit::{row_loglik, subsample_loglik, subsample_model};
pub use fitter::ENNFitter;
pub use hash::normal_hash;
pub use hypervolume::{hypervolume_max, hypervolume2d_max};
pub use incumbent_tracker::IncumbentTracker;
pub use index::{ENNIndex, IndexDriver, IndexError};
pub use metric_auto::MetricLearner;
pub use model::ENN;
pub use model::{EnnIndexAccess, EnnRowAccess, ModelOptions};
pub use morbo::{MorboTRSettings, MorboTrustRegion, Rescalarize};
pub use ndtri::{ndtri, z_crit};
pub use optimizer::obs_access::ObsAccess;
pub use optimizer::{BayesOptimizer, Telemetry};
pub use optimizer_factory::{create_lhd, create_optimizer, enn_optimizer};
pub use params::{ENNNormal, ENNParams, ParamsError, PosteriorFlags};
pub use posterior::{WeightedPosteriorData, compute_internals, conditional_internals};
pub use pretrain_data::load_tokens as pretrain_tokens;
pub use proposal_kernel::ProposalKernel;
pub use reliability_region::{
    ReliabilityAction, ReliabilityController, ReliabilityEvidence, ReliabilityPolicy,
    ReliabilityTelemetry,
};
pub use stats::WeightedStats;
pub use store::DiskBpannStore;
pub use store::{EnnStorage, EnnStore, MemoryEnnStore};
pub use strategy::Strategy;
pub use surrogate::{
    ENNSurrogate, ENNSurrogateConfig, MetricLearning, Surrogate, SurrogatePrediction,
};
pub use traits::PosteriorComputation;
pub use trust_region::{
    NoTrustRegion, TRLengthConfig, TrustRegionConfig, TrustRegionError, TurboTrustRegion,
};
pub use util::{argmax_tie, pareto2d_max, sobol_indices, standardize_y};
pub use y_bounds::OutputTransform;
