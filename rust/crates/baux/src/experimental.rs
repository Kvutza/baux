//! Experimental BAUX APIs.
//!
//! This module is the staging area for unstable lower-level surface area.
//! Keep stable user-facing Rust entry points in [`crate::prelude`].

#[cfg(all(target_os = "macos", feature = "metal"))]
pub use crate::apple_gpu::{
    DeviceInfo as AppleGpuInfo, MemoryInfo as AppleGpuMemoryInfo, Target,
    device_info as apple_gpu_info, memory_info as metal_memory_info,
    upload_bf16 as metal_upload_bf16,
};
pub use crate::compute::{AcquisitionKind, ComputeDevice};
pub use crate::dense::{
    DenseLeaf, DenseLinear, DenseResult, DenseTerm, DenseView, METAL_OPS, OPENCL_OPS, ParamBuffer,
    apply as apply_dense, dist2 as dense_dist2, linear as dense_linear,
};
pub use crate::forward_program::{
    Compiler, Exactness, ExecutionSchedule, ExecutionTarget, Extent, FusionBoundary, LayerVisit,
    MHC4_COEFFS, MHC4_PARAMS, MHC4_PATCH, ModelProgram, PISA_BYTES, ProgramFunction, ProgramKind,
    RecurrentCore, ScheduledKernel, SemanticNode, SemanticOp, TargetCapability, TensorDType,
    TensorLayout, TensorLifetime, TensorSpec, TensorStorage, WeightArenaSpec, perf_digest,
    perf_sources,
};
pub use crate::knn::{KnnIndex, KnnPlan, KnnProfile};
pub use crate::optimizer::{
    BayesOptimizer, MultiTrustRegionConfig, MultiTrustRegionState, ObservationDelta, RegionBatch,
    RegionCandidate, SharingPolicy, Telemetry,
};
pub use crate::optimizer_factory::enn_tr;
#[cfg(all(target_os = "macos", feature = "metal"))]
pub use crate::qwen_metal::{
    QwenCompiledContext as MetalQwenCompiledContext, QwenEvaluator as MetalQwenEvaluator,
};
#[cfg(all(feature = "cuda", target_os = "linux", target_arch = "x86_64"))]
pub use crate::resident_cuda::{ParamBlock, Proposal, Proposals, SearchState};
#[cfg(all(target_os = "macos", feature = "metal"))]
pub use crate::resident_metal::{
    ParamBlock as MetalParamBlock, ProposalProfile as MetalProfileInfo,
    Proposals as MetalProposals, SearchState as MetalSearchState,
};
pub use crate::response_observation::{
    ProposalMeasure, RESPONSE_SCHEMA, ResponseObservation, ResponseReference, TokenAccounting,
};
pub use crate::trials::{
    AcquisitionParams, BpannHistory, IndexedObservation, ObservationId, ProposalParams,
    RoundConfig, SurrogateParams,
};
pub use crate::weights::{
    apply_sparse, blocks_words, draw_sparse, merge_values, missing_words, sparse_union, sparse_xor,
    take_words,
};
#[cfg(all(feature = "cuda", target_os = "linux", target_arch = "x86_64"))]
pub use baux_cuda::{
    device_target as cuda_target, initialized_check as cuda_trace, model_check as cuda_check,
};

#[cfg(all(target_os = "macos", feature = "metal"))]
pub use crate::context_metal::ContextCache;
#[cfg(all(target_os = "macos", feature = "metal"))]
pub use crate::fbt_micro::{MicroMetrics, benchmark_micro};
#[cfg(all(target_os = "macos", feature = "metal"))]
pub use crate::fbt_model::{
    GateUpProbe, RoundLatencyExperiment, RoundLatencyRecord, gateup_probe, round_experiment,
};
#[cfg(all(target_os = "macos", feature = "metal"))]
pub use crate::fbt_moe::{
    ActualBoResult, GroupedMoeProbe, context_loop, moe_dataset, moe_probe, run_generated,
    run_generation, run_pretrain,
};
