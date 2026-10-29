//! Experiment configuration types for continuous-control evaluations.

use deser::{Deserialize, Serialize};

/// High-level benchmark environment choice.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, Default)]
#[deser(rename_all = "lowercase")]
pub enum ControlEnv {
    #[default]
    Cheetah,
    Hopper,
}

/// Policy parameterization choice.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, Default)]
#[deser(rename_all = "lowercase")]
pub enum ControlArch {
    Linear,
    #[default]
    Mlp,
    Simba,
}

/// Execution parameters for the control evaluation experiment.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[deser(default, rename_all = "kebab-case")]
pub struct ExperimentConfig {
    pub env: ControlEnv,
    pub policy: ControlArch,
    #[deser(alias = "num_arms")]
    pub num_arms: usize,
    #[deser(alias = "num_rounds")]
    pub num_rounds: usize,
    #[deser(alias = "num_denoise_passive")]
    pub num_denoise_passive: usize,
    #[deser(alias = "heldout_episodes")]
    pub heldout_episodes: usize,
    pub repetitions: usize,
}

impl Default for ExperimentConfig {
    fn default() -> Self {
        Self {
            env: ControlEnv::Cheetah,
            policy: ControlArch::Mlp,
            num_arms: 1,
            num_rounds: 50,
            num_denoise_passive: 1,
            heldout_episodes: 10,
            repetitions: 1,
        }
    }
}
