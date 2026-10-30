pub use baux_tooling::radar::{PerformanceAction, PerformanceTarget, RadarAction};

pub fn run(root: &std::path::Path, action: RadarAction) -> Result<(), String> {
    baux_tooling::radar::run(root, action, crate::radar_perf::run)
}
