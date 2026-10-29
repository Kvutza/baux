#[path = "radar/performance.rs"]
mod performance;

pub use baux_tooling::radar::{PerformanceAction, PerformanceBackend, RadarAction};

pub fn run(root: &std::path::Path, action: RadarAction) -> Result<(), String> {
    baux_tooling::radar::run(root, action, performance::run)
}
