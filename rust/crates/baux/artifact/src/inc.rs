//! Incremental memoization, append-only trajectory logs, and dynamic `bind`
//! rungs.

#[path = "inc/db.rs"]
pub mod db;
#[path = "inc/paired.rs"]
pub mod paired;
#[path = "inc/types.rs"]
pub mod types;

#[cfg(test)]
#[path = "inc/tests.rs"]
mod tests;

pub use db::{IncStore, default_dir, eval_key, traj_key};
pub use paired::{
    ArmInput, PairedArmSummary, PairedReport, SeedLossTrace, active_cells, paired_report,
};
pub use types::{EvalOutcome, StepRecord, StoreStats, TrajHead, TrajSlice};
