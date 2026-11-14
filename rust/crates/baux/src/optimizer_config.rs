//! Configuration surface used directly by the ask/tell optimizer.

use deser::{Deserialize, Serialize};

use crate::candidates::CandidateSampler;
use crate::proposal_distribution::CoordSystem;
use crate::surrogate::ENNSurrogateConfig;
use crate::trust_region::TrustRegionConfig;

/// Optimizer configuration decomposed by role.
#[derive(Debug, Clone)]
pub struct OptimizerConfig {
    pub surrogate: SurrogateConfig,
    pub trust_region: TrustRegionConfig,
    pub candidates: CandidateConfig,
    pub acquisition: AcquisitionConfig,
    pub noise_aware: bool,
    pub failure_tolerance_dim: Option<f64>,
}

impl Default for OptimizerConfig {
    fn default() -> Self {
        Self {
            surrogate: SurrogateConfig::ENN(ENNSurrogateConfig::default()),
            trust_region: TrustRegionConfig::default(),
            candidates: CandidateConfig::default(),
            acquisition: AcquisitionConfig::default(),
            noise_aware: false,
            failure_tolerance_dim: None,
        }
    }
}

#[derive(Debug, Clone)]
pub enum SurrogateConfig {
    ENN(ENNSurrogateConfig),
    None,
}

impl Default for SurrogateConfig {
    fn default() -> Self {
        SurrogateConfig::ENN(ENNSurrogateConfig::default())
    }
}

#[derive(Debug, Clone)]
pub struct CandidateConfig {
    pub num_candidates_factor: f64,
    pub min_candidates: usize,
    pub max_candidates: Option<usize>,
    pub num_candidates_per_arm: Option<usize>,
    pub num_pert: usize,
    pub candidate_rv: CandidateSampler,
    pub raasp_fast: bool,
    pub coord_system: CoordSystem,
    pub tied_dims: Vec<Vec<usize>>,
}

impl Default for CandidateConfig {
    fn default() -> Self {
        Self {
            num_candidates_factor: 100.0,
            min_candidates: 10,
            max_candidates: Some(5000),
            num_candidates_per_arm: Some(0),
            num_pert: 20,
            candidate_rv: CandidateSampler::Uniform,
            raasp_fast: false,
            coord_system: CoordSystem::Cartesian,
            tied_dims: Vec::new(),
        }
    }
}

impl CandidateConfig {
    pub fn num_candidates(&self, num_dim: usize, num_arms: usize) -> usize {
        let is_exact_fixed = self.num_candidates_factor == 1.0
            && self.max_candidates == Some(self.min_candidates)
            && self.num_candidates_per_arm.is_none();

        let mut base = if is_exact_fixed {
            self.min_candidates
        } else {
            let raw = (self.num_candidates_factor * num_dim as f64) as usize;
            let formula = match self.max_candidates {
                Some(cap) if self.num_candidates_per_arm.is_some() => raw.min(cap),
                Some(cap) if (self.num_candidates_factor - 100.0).abs() < f64::EPSILON => {
                    raw.min(cap)
                }
                _ => raw,
            };
            formula.max(self.min_candidates)
        };

        if let Some(multiplier) = self.num_candidates_per_arm {
            base = base.max(num_arms * multiplier);
        }
        base
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[deser(rename_all = "snake_case")]
pub enum AcquisitionConfig {
    #[deser(rename = "ucb")]
    UCB {
        beta: f64,
    },
    Thompson,
    Random,
    Pareto,
}

impl Default for AcquisitionConfig {
    fn default() -> Self {
        AcquisitionConfig::UCB { beta: 2.0 }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum InitStrategy {
    #[default]
    LHD,
    Random,
}

pub fn turbo_enn() -> OptimizerConfig {
    OptimizerConfig {
        surrogate: SurrogateConfig::ENN(ENNSurrogateConfig {
            k: 10,
            num_candidates: 30,
            num_samples: 10,
            ..Default::default()
        }),
        trust_region: TrustRegionConfig::default(),
        candidates: CandidateConfig {
            candidate_rv: CandidateSampler::RAASP,
            ..CandidateConfig::default()
        },
        acquisition: AcquisitionConfig::UCB { beta: 2.0 },
        noise_aware: false,
        failure_tolerance_dim: None,
    }
}

pub fn turbo_zero() -> OptimizerConfig {
    OptimizerConfig {
        surrogate: SurrogateConfig::None,
        trust_region: TrustRegionConfig::default(),
        candidates: CandidateConfig {
            candidate_rv: CandidateSampler::RAASP,
            ..CandidateConfig::default()
        },
        acquisition: AcquisitionConfig::Random,
        noise_aware: false,
        failure_tolerance_dim: None,
    }
}

pub fn lhd_only() -> OptimizerConfig {
    OptimizerConfig {
        surrogate: SurrogateConfig::None,
        trust_region: TrustRegionConfig::default(),
        candidates: CandidateConfig {
            num_candidates_factor: 1.0,
            min_candidates: 1,
            max_candidates: None,
            num_candidates_per_arm: None,
            num_pert: 20,
            candidate_rv: CandidateSampler::Uniform,
            raasp_fast: false,
            coord_system: CoordSystem::Cartesian,
            tied_dims: Vec::new(),
        },
        acquisition: AcquisitionConfig::Random,
        noise_aware: false,
        failure_tolerance_dim: None,
    }
}
