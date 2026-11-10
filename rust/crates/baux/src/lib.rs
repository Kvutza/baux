//! Core neighbor-surrogate optimization algorithms in Rust.
//!
//! This crate owns the optimizer semantics shared by the Python API and the
//! accelerator backends.

#![allow(
    clippy::pedantic,
    clippy::nursery,
    clippy::cargo,
    incomplete_features,
    internal_features
)]
#![feature(
    adt_const_params,
    contracts,
    diagnostic_on_move,
    diagnostic_on_type_error,
    negative_impls,
    portable_simd
)]

extern crate self as baux;

pub mod acquisition;
pub mod agent_contract;
#[cfg(all(any(target_os = "macos", target_os = "ios"), feature = "metal"))]
#[allow(dead_code)]
pub mod apple_gpu;
#[path = "../artifact/src/lib.rs"]
pub mod artifact;
pub use artifact as baux_artifact;
pub mod backend;
pub mod base_policy;
#[path = "../neighbor-index/src/lib.rs"]
pub mod baux_neighbor_index;
pub use baux_neighbor_index as neighbor_index;
#[cfg(all(any(target_os = "macos", target_os = "ios"), feature = "metal"))]
#[allow(dead_code)]
pub mod resident_metal;
pub use baux::semantics::optimizer::SurrogateSpec;
#[cfg(all(any(target_os = "macos", target_os = "ios"), feature = "metal"))]
pub use resident_metal as resident;
pub mod candidates;
mod candidates_fast;
// Compile and execute the backend-independent CUDA mHC4 controller tests on
// Metal development hosts too. CUDA device kernels remain NVIDIA-only, while
// streams, posterior updates, ENN fitting, acceptance, and radius trajectories
// are verified in the ordinary cross-backend test suite.
pub mod capability;
pub mod coding_outcome;
pub mod config;
pub mod coord_space;
mod coordinate;
mod coordinate_flat;
mod coordinate_scalar;
pub mod device;
pub mod device_posterior;
pub use baux::semantics::context;
pub mod disk_bpann;
pub mod draw;
pub mod error;
pub mod experimental;
#[cfg(feature = "extraction")]
pub mod extraction;
#[cfg(all(any(target_os = "macos", target_os = "ios"), feature = "metal"))]
pub mod mps;
#[cfg(all(any(target_os = "macos", target_os = "ios"), feature = "metal"))]
pub use mps as fbt_mps;
pub mod file_config;
pub mod fit;
pub mod fitter;
#[allow(dead_code)]
pub mod forward_program;
pub mod hash;
#[cfg(all(any(target_os = "macos", target_os = "ios"), feature = "metal"))]
pub mod hnet_metal;
pub mod hypervolume;
pub mod incumbent_tracker;
pub mod index;
pub mod knn;
pub mod layout;
pub mod metric_auto;
pub mod metric_loo;
pub mod metric_provider;
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
pub use baux::semantics::neighbor as neighbor_rule;
pub mod objective_observation;
pub mod obs_view;
pub mod optimizer;
pub mod optimizer_config;
mod optimizer_dispatch;
pub mod optimizer_factory;
mod optimizer_mu;
pub mod params;
pub mod pipeline;
#[cfg(target_os = "macos")]
pub mod portable;
pub mod posterior;
pub mod predictor;
pub mod prelude;
pub mod pretrain_data;
pub mod procedural_pool;
pub mod proposal_kernel;
pub mod quantization;
#[cfg(any(test, all(target_os = "macos", feature = "metal")))]
pub mod reconstruction;
pub mod response_observation;
pub mod scratch;
pub mod search;
#[path = "../semantics/src/lib.rs"]
pub mod semantics;
pub use semantics as baux_semantics;
pub mod sense;
pub mod stats;
pub mod strategy;
pub mod surrogate;
pub mod surrogate_api;
pub mod tensor_store;
pub mod text;
pub mod ticket;
pub mod tokamak;
pub mod traits;
pub mod trials;
pub mod trust_region;
pub mod typestate;
pub mod unified_mem;
pub mod util;
pub mod weights;
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
pub use backend::DiskBpannEnnBackend;
pub use backend::{
    DiskStore, EnnBackend, EnnStorage, InMemoryEnnBackend, MemoryStore, Storage, Store,
};
pub use base_policy::{BasePolicyManifest, POLICY_SCHEMA, PolicyQualification};
pub use candidates::{
    CandidateRV, CandidateSampler, draw_turbo, from_unit, generate_candidates, generate_lhd,
    generate_uniform, perturb_candidates, sobol_masked, sobol_sequence, to_unit,
};
pub use capability::{
    Backend, Capability, Operation, Support, backends, matrix, operations, support,
};
pub use coding_outcome::{
    CODING_SCHEMA, CandidateEfficiency, CandidateEligibility, CodingOutcome, ExecutableChecks,
};
pub use config::{
    DataRecipe, DistanceMetric, ExperimentKind, ExperimentSpec, ModelArchitecture,
    ObjectiveReference, ResidentEnnConfig, TrustRegionKind,
};
pub use coordinate::{CoordPoint, Coords, DomainFrame, DomainSpace, OwnedCoords, UnitSpace};
pub use device::{Device, DeviceMemory};
pub use device_posterior::{Posterior, PosteriorDraw};
pub use draw::{Candidates, ConditionalDraw, DrawInternals, NeighborData};
pub use error::{ENNError, EPS_VAR};
pub use file_config::{
    BpannConfig, Config, ConfigFile, IndexConfig, TreeConfig, bpann_config, config_path,
    index_config, set_path,
};
pub use fit::{row_loglik, subsample_loglik, subsample_model};
pub use fitter::{ENNFitter, fit_model};
pub use hash::{
    AffineKey, Key, Prng, auto_seed, domain_prng, domain_seed, normal_hash, prng, resolve_seed,
    splitmix_f32, splitmix64, stream_prng, stream_seed, tune_seed,
};
pub use hypervolume::{hypervolume_max, hypervolume2d_max};
pub use incumbent_tracker::{
    GenericTracker, IncumbentTracker, MaxTracker, MinTracker, TargetTracker,
};
pub use index::{ENNIndex, Index, IndexDriver, IndexError, IndexKind};
pub use layout::EnnLayout;
pub use metric_auto::MetricLearner;
pub use model::ENN;
pub use model::{EnnIndexAccess, EnnRowAccess, ModelOptions};
pub use morbo::{MorboTRSettings, MorboTrustRegion, Rescalarize};
pub use ndtri::{ndtri, z_crit};
pub use objective_observation::ConstObservation;
pub use optimizer::obs_access::ObsAccess;
pub use optimizer::{BayesOptimizer, Telemetry};
pub use optimizer_config::{
    AcquisitionConfig, CandidateConfig, InitStrategy, OptimizerConfig, SurrogateConfig, lhd_only,
    turbo_enn, turbo_zero,
};
pub use optimizer_factory::{create_lhd, create_optimizer, enn_optimizer};
pub use params::{ENNNormal, ENNParams, ParamsError, PosteriorFlags};
pub use pipeline::{
    AcqStage, Active, DesignPipe, Fitted, InitStage, ModelStage, PerturbStage, Pipeline,
    ProposalTicket, SurrogatePipe, TypedOptimizer, TypedSurrogate, Unfitted, Warmup,
};
pub use posterior::{Pool, TopK, WeightedPosteriorData, compute_internals, conditional_internals};
pub use pretrain_data::load_tokens as pretrain_tokens;
pub use proposal_kernel::{CoordSystem, ProposalKernel};
pub use scratch::{
    AlignedBuffer, CacheLine, CandidateSoa, PageBlock, ScratchArena, ScratchScope, UnifiedBuffer,
    UnifiedRead, UnifiedWrite,
};
pub use semantics::affine::Affine;
pub use semantics::categorical::{CatEncoding, CatSpec};
pub use semantics::stiefel::Stiefel;
pub use sense::{
    BetterScore, EvalId, Lexicographic, Maximize, Minimize, ObjectiveSense, ObsId, RegionId, Sense,
    Target,
};
pub use stats::WeightedStats;
pub use strategy::Strategy;
pub use surrogate::{
    ENNSurrogate, ENNSurrogateConfig, MetricLearning, Surrogate, SurrogatePrediction,
};
pub use surrogate_api::{LengthscaleProvider, ObservationView, Predictor};
pub use traits::PosteriorComputation;
pub use trust_region::{
    NoTrustRegion, TRLengthConfig, TrustRegionConfig, TrustRegionError, TurboTrustRegion,
};
pub use util::{argmax_tie, pareto2d_max, sobol_indices, standardize_y};
pub use y_bounds::OutputTransform;
