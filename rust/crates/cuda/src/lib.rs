extern crate self as baux;

macro_rules! buck_mod {
    ($item:item) => {
        $item
    };
}

#[cfg(buck_build)]
buck_mod! {
    #[path = "../baux-source/artifact/src/lib.rs"]
    pub mod artifact;
}
#[cfg(not(buck_build))]
#[path = "../../baux/artifact/src/lib.rs"]
pub mod artifact;
#[cfg(buck_build)]
buck_mod! {
    #[path = "../baux-source/semantics/src/lib.rs"]
    pub mod semantics;
}
#[cfg(not(buck_build))]
#[path = "../../baux/semantics/src/lib.rs"]
pub mod semantics;
pub use baux_cuda_device::{
    BatchValue, FbtShape, FeedbackShape, KNN_K, MAX_DEPTH, MAX_HISTORY, MatmulShape, MoeShape,
    ObjectiveShape, PisaShape, RouteShape, RoutedTile, ScoreStats, THREADS, fbt_model, trials,
};
pub use cuda_core::{
    CudaContext, CudaEvent, CudaStream, DeviceBuffer, DeviceCopy, IntoResult, LaunchConfig1D,
    simt::LaunchConfig,
};

mod capture;
pub use capture::{CudaResult, device_target, target_name};
pub(crate) use capture::{
    Runtime, copy_prefix, cuda_error, load_module, read_prefix, timing_event, to_u32,
};
pub mod gemm;
mod graph;
mod mhc4_probe;
pub mod muon;
pub mod neighbors;
mod precision;

#[cfg(feature = "native-flame")]
pub mod flame;

pub mod context;
mod knn;
mod model;
mod prefill;
pub use knn::{CUVS_MAX_K, CudaIndex, CudaKnnMode, KnnProfile};
pub use model::diffusion_run::{check as diffusion_check, run as model_diffusion};
pub use model::run::initialized_check;
pub use model::run::{
    bench as generation_bench, check as model_check, generate as model_generate, mhc4_bench,
    mhc4_bo, run as model_run,
};
pub use model::weights::proposal_check;
pub use model::{FbtModel, GenerationOutput, ModelOutput, WaveProfile};
pub use prefill::{FbtPrefill, FbtPrefillProfile, embed_parity, feedback_parity};
