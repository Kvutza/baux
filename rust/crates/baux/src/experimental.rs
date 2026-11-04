//! Experimental BAUX APIs.
//!
//! This module is the staging area for unstable lower-level surface area.
//! Keep stable user-facing Rust entry points in [`crate::prelude`].

#[cfg(all(target_os = "macos", feature = "metal"))]
pub use crate::apple_gpu::{DeviceInfo as AppleGpuInfo, Target, device_info as apple_gpu_info};
pub use crate::forward_program::{
    Backend, BackendSchedule, Compiler, MHC4_COEFFS, MHC4_PARAMS, MHC4_PATCH, ModelProgram,
    PISA_BYTES, ProgramKind, ScheduledKernel, perf_digest, perf_sources,
};
pub use crate::knn::{KnnIndex, KnnPlan, KnnProfile};
pub use crate::optimizer::{
    BayesOptimizer, EvaluationId, MultiTrustRegionConfig, MultiTrustRegionState, ObservationDelta,
    RegionBatch, RegionCandidate, SharingPolicy, Telemetry,
};
pub use crate::optimizer_factory::enn_tr;
pub use crate::quantization::{FP4_LUT, quantize_e2m1, quantize_int4};
#[cfg(all(target_os = "macos", feature = "metal"))]
pub use crate::resident_metal::{
    MetalMemoryInfo, ParamBlock as MetalParamBlock, ProposalProfile as MetalProfileInfo,
    Proposals as MetalProposals, RegionalProposal, RegionalSearch, SearchState as MetalSearchState,
    memory_info as metal_memory_info, upload as metal_upload,
};
pub use crate::response_observation::{
    ProposalMeasure, RESPONSE_SCHEMA, ResponseObservation, ResponseReference, TokenAccounting,
};
pub use crate::trials::{
    AcquisitionParams, EncodingType, ProposalParams, RoundConfig, SurrogateParams,
};
pub use crate::weights::{AcquisitionKind, ComputeDevice};
#[cfg(all(feature = "native-flame", target_os = "linux", target_arch = "x86_64"))]
pub use baux_cuda::flame::{FlameConfig, FlameEvaluator};
#[cfg(all(feature = "cuda", target_os = "linux", target_arch = "x86_64"))]
pub use baux_cuda::{
    device_target as cuda_target, initialized_check as cuda_trace, model_check as cuda_check,
};
