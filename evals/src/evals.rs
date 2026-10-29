//! Continuous-control evaluation benchmark library.

pub mod bench_chain;
pub mod bench_experiment;
pub mod bench_optimizer;
pub mod bench_oracle;
pub mod bench_schedule;
pub mod bench_task;
#[cfg(test)]
mod bench_tests;

use baux::artifact::toml;
pub use bench_experiment::{ControlArch, ControlEnv, ExperimentConfig};
pub use bench_optimizer::{
    AcquisitionConfig, BayesOptimizerExt, OptimizerConfig, ProposalConfig, ProposalGenerator,
    SurrogateConfig, SurrogateKind, TrustRegionConfig,
};
use deser::{Deserialize, Serialize};

/// Top-level TOML configuration for continuous-control benchmark runs.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[deser(default, rename_all = "kebab-case")]
pub struct ControlConfig {
    pub experiment: ExperimentConfig,
    pub optimizer: OptimizerConfig,
}

impl ControlConfig {
    /// Parse a continuous-control benchmark recipe from a TOML string.
    pub fn from_toml(content: &str) -> Result<Self, String> {
        toml::from_str(content).map_err(|err| format!("parse control config: {err}"))
    }
}
