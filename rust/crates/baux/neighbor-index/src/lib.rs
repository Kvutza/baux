#[path = "backend.rs"]
pub mod backend;
#[path = "distance.rs"]
pub mod distance;
#[path = "error.rs"]
pub mod error;
#[path = "index/mod.rs"]
pub mod index;
#[path = "lgnsrch.rs"]
pub mod lgnsrch;
#[path = "merge.rs"]
pub mod merge;
#[path = "mmap_store.rs"]
pub mod mmap_store;
#[path = "observation.rs"]
pub mod observation;
#[path = "parallel.rs"]
pub mod parallel;
#[path = "residency.rs"]
pub mod residency;
#[path = "smnsrch.rs"]
pub mod smnsrch;
#[path = "tuning.rs"]
pub mod tuning;

pub use backend::{BpannBackend, PAPER_URL, PENDING_HARD, PENDING_SOFT, soft_build, soft_publish};
pub use error::BpannError;
pub use index::{BpannIndex, IncrementalIndex};
pub use lgnsrch::{SearchPendingArgs, search_pending};
pub use observation::{MAX_DIM, MAX_STRIDE, remove_store};
pub use residency::{Residency, ResidencyPolicy};
pub use smnsrch::{N_LIMIT, OrderedF32, ScoreQueriesFlat, resident_cache, score_queries, topk_l2};
pub use tuning::{
    BpannTuning, EXHAUSTIVE_LIMIT, INDEX_MIN, SKIP_LIMIT, STRUCTURED_LIMIT, clear_provider,
    current_tuning, set_provider,
};

#[cfg(test)]
#[path = "acceptance_tests.rs"]
mod acceptance_tests;
