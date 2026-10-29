//! Configuration surface used directly by the ask/tell optimizer.

use deser::{Deserialize, Serialize};

use crate::candidates::CandidateSampler;
use crate::config::{
    AcquisitionSpec, CandidateGenerationSpec, OptimizerSpec, RadiusControlSpec, SurrogateEstimator,
    SurrogateFit, SurrogateRecipe, TrustRegionBounds, TrustRegionSpec,
};
use crate::error::ENNError;
use crate::morbo::MorboTRSettings;
use crate::procedural_pool::ProposalMethod;
use crate::surrogate::ENNSurrogateConfig;
use crate::trust_region::{TRLengthConfig, TrustRegionConfig};

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

fn length_bounds(bounds: &TrustRegionBounds) -> Result<TRLengthConfig, ENNError> {
    bounds.validate().map_err(ENNError::InvalidParameter)?;
    Ok(TRLengthConfig {
        length_init: bounds.initial,
        length_min: bounds.min,
        length_max: bounds.max,
    })
}

impl OptimizerConfig {
    /// Compile an ask/tell [`OptimizerConfig`] directly from a v4
    /// [`OptimizerSpec`].
    pub fn from_spec(spec: &OptimizerSpec) -> Result<Self, ENNError> {
        match spec {
            OptimizerSpec::Muon | OptimizerSpec::Adamw => Err(ENNError::InvalidParameter(
                "Muon and AdamW are matrix pretraining optimizers, not ask/tell BO recipes"
                    .to_string(),
            )),
            OptimizerSpec::OperatorBo {
                candidate_generation,
                surrogate,
                radius_control,
                ..
            } => {
                let candidates = CandidateConfig::from_spec(candidate_generation)?;
                let surrogate_cfg = SurrogateConfig::from_recipe(surrogate)?;
                let RadiusControlSpec::Turbo { bounds } = radius_control;
                let trust_region = TrustRegionConfig::Turbo(length_bounds(bounds)?);
                Ok(Self {
                    surrogate: surrogate_cfg,
                    trust_region,
                    candidates,
                    acquisition: AcquisitionConfig::UCB { beta: 1.0 },
                    noise_aware: false,
                    failure_tolerance_dim: None,
                })
            }
            OptimizerSpec::TurboEnn {
                candidate_generation,
                surrogate,
                acquisition,
                trust_region,
            } => {
                let candidates = CandidateConfig::from_spec(candidate_generation)?;
                let surrogate_cfg = SurrogateConfig::from_recipe(surrogate)?;
                let acq_cfg = AcquisitionConfig::from_spec(acquisition);
                let tr_cfg = TrustRegionConfig::from_spec(trust_region, acquisition, false)?;
                Ok(Self {
                    surrogate: surrogate_cfg,
                    trust_region: tr_cfg,
                    candidates,
                    acquisition: acq_cfg,
                    noise_aware: false,
                    failure_tolerance_dim: None,
                })
            }
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

impl SurrogateConfig {
    pub fn from_recipe(recipe: &SurrogateRecipe) -> Result<Self, ENNError> {
        if recipe.neighbors < 1 {
            return Err(ENNError::InvalidParameter(format!(
                "k (number of neighbors) must be > 0, got {}",
                recipe.neighbors
            )));
        }
        let (num_candidates, num_samples) = match recipe.fit {
            SurrogateFit::Fixed => (0, 0),
            SurrogateFit::LeaveOneOut {
                fit_candidates,
                fit_samples,
            } => (fit_candidates, fit_samples),
        };
        let _ = SurrogateEstimator::Enn;
        Ok(SurrogateConfig::ENN(ENNSurrogateConfig {
            k: recipe.neighbors,
            num_candidates,
            num_samples,
            ..Default::default()
        }))
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
        }
    }
}

impl CandidateConfig {
    pub fn from_spec(spec: &CandidateGenerationSpec) -> Result<Self, ENNError> {
        if spec.arms == 0 || spec.candidates == 0 || !spec.candidates.is_multiple_of(spec.arms) {
            return Err(ENNError::InvalidParameter(
                "candidates must be positive and divisible by arms".to_string(),
            ));
        }
        let per_arm = (spec.candidates / spec.arms) as usize;
        let total = spec.candidates as usize;
        let candidate_rv = match spec.generator {
            ProposalMethod::Additive => CandidateSampler::RAASP,
            ProposalMethod::PairMix | ProposalMethod::QuadraticFlow => CandidateSampler::Sobol,
        };
        Ok(Self {
            num_candidates_factor: 1.0,
            min_candidates: total,
            max_candidates: Some(total),
            num_candidates_per_arm: Some(per_arm),
            num_pert: 20,
            candidate_rv,
            raasp_fast: false,
        })
    }

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

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
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

impl AcquisitionConfig {
    pub fn from_spec(spec: &AcquisitionSpec) -> Self {
        match spec {
            AcquisitionSpec::Ucb { beta } => Self::UCB { beta: *beta },
            AcquisitionSpec::Thompson => Self::Thompson,
            AcquisitionSpec::Random => Self::Random,
            AcquisitionSpec::Pareto { .. } => Self::Pareto,
            AcquisitionSpec::AugmentedChebyshev { .. } => Self::Pareto,
        }
    }
}

impl TrustRegionConfig {
    pub fn from_spec(
        spec: &TrustRegionSpec,
        acquisition: &AcquisitionSpec,
        noise_aware: bool,
    ) -> Result<Self, ENNError> {
        match spec {
            TrustRegionSpec::Turbo { bounds } => Ok(Self::Turbo(length_bounds(bounds)?)),
            TrustRegionSpec::Morbo {
                bounds,
                rescalarize,
                ..
            } => {
                let (num_metrics, alpha) = match acquisition {
                    AcquisitionSpec::AugmentedChebyshev { scales, alpha, .. } => {
                        (scales.len().max(2), f64::from(*alpha))
                    }
                    AcquisitionSpec::Pareto { scales } => (scales.len().max(2), 0.05),
                    _ => (2, 0.05),
                };
                Ok(Self::Morbo(MorboTRSettings {
                    num_metrics,
                    alpha,
                    length: length_bounds(bounds)?,
                    rescalarize: *rescalarize,
                    noise_aware,
                }))
            }
        }
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
        },
        acquisition: AcquisitionConfig::Random,
        noise_aware: false,
        failure_tolerance_dim: None,
    }
}
