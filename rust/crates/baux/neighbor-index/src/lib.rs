#[cfg(all(feature = "disk-bpann", not(target_arch = "wasm32")))]
#[path = "backend.rs"]
pub mod backend;
#[cfg(all(feature = "disk-bpann", not(target_arch = "wasm32")))]
#[path = "distance.rs"]
pub mod distance;
#[path = "error.rs"]
pub mod error;
#[cfg(all(feature = "disk-bpann", not(target_arch = "wasm32")))]
#[path = "index/mod.rs"]
pub mod index;
#[cfg(all(feature = "disk-bpann", not(target_arch = "wasm32")))]
#[path = "lgnsrch.rs"]
pub mod lgnsrch;
#[cfg(all(feature = "disk-bpann", not(target_arch = "wasm32")))]
#[path = "merge.rs"]
pub mod merge;
#[cfg(all(feature = "disk-bpann", not(target_arch = "wasm32")))]
#[path = "mmap_store.rs"]
pub mod mmap_store;
#[cfg(all(feature = "disk-bpann", not(target_arch = "wasm32")))]
#[path = "observation.rs"]
pub mod observation;
#[path = "parallel.rs"]
pub mod parallel;
#[path = "residency.rs"]
pub mod residency;
#[cfg(all(feature = "disk-bpann", not(target_arch = "wasm32")))]
#[path = "smnsrch.rs"]
pub mod smnsrch;
#[path = "tuning.rs"]
pub mod tuning;

#[cfg(all(feature = "disk-bpann", not(target_arch = "wasm32")))]
pub use backend::{BpannBackend, PAPER_URL, soft_build, soft_publish};
pub use error::BpannError;
#[cfg(all(feature = "disk-bpann", not(target_arch = "wasm32")))]
pub use index::{BpannIndex, IncrementalIndex};
#[cfg(all(feature = "disk-bpann", not(target_arch = "wasm32")))]
pub use lgnsrch::{SearchPendingArgs, search_pending};
#[cfg(all(feature = "disk-bpann", not(target_arch = "wasm32")))]
pub use observation::{MAX_DIM, MAX_STRIDE, remove_store};
#[cfg(not(all(feature = "disk-bpann", not(target_arch = "wasm32"))))]
pub fn remove_store(work_dir: &std::path::Path) -> std::io::Result<()> {
    if work_dir.exists() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::Unsupported,
            "Disk storage is unavailable in wasm32 or without disk-bpann",
        ));
    }
    Ok(())
}
pub use residency::{Residency, ResidencyPolicy};
#[cfg(all(feature = "disk-bpann", not(target_arch = "wasm32")))]
pub use smnsrch::{N_LIMIT, OrderedF32, ScoreQueriesFlat, resident_cache, score_queries, topk_l2};
pub use tuning::{
    BpannTuning, EXHAUSTIVE_LIMIT, INDEX_MIN, PENDING_HARD, PENDING_SOFT, SKIP_LIMIT,
    STRUCTURED_LIMIT, clear_provider, current_tuning, set_provider,
};

#[cfg(all(feature = "disk-bpann", not(target_arch = "wasm32")))]
pub type TreeBackend = BpannBackend;
#[cfg(all(feature = "disk-bpann", not(target_arch = "wasm32")))]
pub type TreeIndex = BpannIndex;
pub type TreeTuning = BpannTuning;
pub type TreeError = BpannError;

#[cfg(all(test, feature = "disk-bpann", not(target_arch = "wasm32")))]
#[path = "acceptance_tests.rs"]
mod acceptance_tests;
