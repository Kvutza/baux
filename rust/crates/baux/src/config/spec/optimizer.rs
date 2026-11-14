//! Scientific roles in the TOML optimizer recipe, not executable algorithms.
//!
//! Proposal generation constructs trials; the surrogate estimates their values;
//! acquisition selects a trial; acceptance decides whether it replaces the
//! incumbent; radius control sets subsequent step sizes. Fitting parameters is
//! distinct from adapting a query's neighbor count. Definitions here describe
//! the configuration; `conversion` and runtime guards establish supported
//! paths. Deserialization success does not certify a theorem or learning
//! performance.

use super::*;

#[path = "optimizer/regional.rs"]
mod regional;
pub use regional::*;

/// A closed optimizer recipe. The selected algorithm determines which
/// scientific roles exist; unrelated sections are rejected by deserialization.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[deser(
    tag = "algorithm",
    rename_all = "kebab-case",
    rename_all_fields = "kebab-case",
    deny_unknown_fields
)]
pub enum OptimizerSpec {
    /// Acquisition-free single-candidate (`K = 1`) Bayesian optimization backed
    /// by the ENN surrogate (Bafna, Jadhav, and Sweet, "Taking the GP Out of
    /// the Loop", ICML 2026, PMLR 306:5179-5199, arXiv:2506.12818).
    #[deser(alias = "operator-bo")]
    SingleStateBo {
        #[deser(default)]
        distribution: crate::ProposalDistribution,
        #[deser(default)]
        perturbation: Option<PerturbationSpec>,
        #[deser(default)]
        surrogate: SurrogateRecipe,
        #[deser(default)]
        trust_region: TrustRegionBounds,
        regional_search: Option<RegionalSearchSpec>,
    },
    #[deser(rename = "bayesian-search")]
    BayesianSearch {
        #[deser(default = turbo_candidates())]
        candidates: u32,
        #[deser(default = default_arms())]
        arms: u32,
        distribution: Option<crate::ProposalDistribution>,
        #[deser(default)]
        perturbation: Option<PerturbationSpec>,
        #[deser(flatten)]
        acquisition: AcquisitionSpec,
        #[deser(default)]
        surrogate: SurrogateRecipe,
        #[deser(default)]
        trust_region: TrustRegionBounds,
        regional_search: Option<RegionalSearchSpec>,
    },
    #[deser(rename = "muon")]
    Muon,
    #[deser(rename = "adamw")]
    Adamw,
    #[deser(rename = "ppo")]
    Ppo,
}

impl OptimizerSpec {
    pub fn candidate_generation(&self) -> Option<CandidateGenerationSpec> {
        match self {
            Self::SingleStateBo {
                distribution,
                perturbation,
                ..
            } => Some(CandidateGenerationSpec {
                candidates: 1,
                arms: 1,
                distribution: Some(*distribution),
                perturbation: perturbation.clone(),
            }),
            Self::BayesianSearch {
                candidates,
                arms,
                distribution,
                perturbation,
                ..
            } => Some(CandidateGenerationSpec {
                candidates: *candidates,
                arms: *arms,
                distribution: *distribution,
                perturbation: perturbation.clone(),
            }),
            Self::Muon | Self::Adamw | Self::Ppo => None,
        }
    }

    pub fn regional_search(&self) -> Option<&RegionalSearchSpec> {
        match self {
            Self::SingleStateBo {
                regional_search, ..
            }
            | Self::BayesianSearch {
                regional_search, ..
            } => regional_search.as_ref(),
            Self::Muon | Self::Adamw | Self::Ppo => None,
        }
    }

    pub(crate) fn strategy(
        &self,
        dimensions: usize,
        num_init: usize,
        rng: &mut dyn rand::RngCore,
    ) -> Result<crate::optimizer::Strategy, crate::error::ENNError> {
        use crate::optimizer::{Strategy, StrategyState};
        use crate::optimizer_config::InitStrategy;
        let Some(regions) = self.regional_search() else {
            return Ok(if num_init > 0 {
                Strategy::hybrid(InitStrategy::LHD, num_init)
            } else {
                Strategy::turbo()
            });
        };
        let bounds = match self {
            Self::SingleStateBo { trust_region, .. }
            | Self::BayesianSearch { trust_region, .. } => trust_region,
            Self::Muon | Self::Adamw | Self::Ppo => unreachable!("matrix optimizer handled above"),
        };
        let config = regions
            .runtime(bounds)
            .map_err(crate::error::ENNError::InvalidParameter)?;
        Ok(Strategy::Experimental(StrategyState::configured(
            dimensions, config, num_init, rng,
        )?))
    }
}

const fn default_arms() -> u32 {
    1
}

const fn turbo_candidates() -> u32 {
    4
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[deser(deny_unknown_fields, rename_all = "kebab-case")]
/// Pool and transformation used to construct trials around an incumbent.
/// This does not set the number of forward evaluations or committed updates.
pub struct CandidateGenerationSpec {
    /// TOTAL proposals across arms, distinct from fitting trials or objective
    /// calls.
    #[deser(default = default_arms())]
    pub candidates: u32,
    /// Seed namespaces sharing a center, not independent trust regions.
    #[deser(default = default_arms())]
    pub arms: u32,
    /// Underlying sampler selection; program transformations can change the
    /// distribution of the resulting parameter displacement.
    pub distribution: Option<crate::ProposalDistribution>,
    #[deser(default)]
    pub perturbation: Option<PerturbationSpec>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[deser(default, deny_unknown_fields, rename_all = "kebab-case")]
/// Geometry, uncertainty scales, and fitting policy for retained data.
/// These settings do not by themselves define how a trial direction is
/// generated.
pub struct SurrogateRecipe {
    /// Fixed k, or the configured neighbor budget when a local rule is enabled.
    pub neighbors: i32,
    /// Query-local neighbor policy, independent of hyperparameter fitting.
    pub neighbor_rule: NeighborRule,
    pub epistemic_scale: f64,
    pub aleatoric_scale: f64,
    pub y_scale: f64,
    pub metric: DistanceMetric,
    pub local_neighbors: Option<usize>,
    /// Scale fitting rule. This is independent of query-local neighbor choice.
    pub fit: SurrogateFit,
    /// Hyperparameter trials in the fitter, not candidate parameter vectors.
    pub fit_candidates: Option<usize>,
    /// Fitter's leave-one-out scoring budget, not fresh model evaluations.
    pub fit_samples: Option<usize>,
}

impl Default for SurrogateRecipe {
    fn default() -> Self {
        Self {
            neighbors: default_neighbors(),
            neighbor_rule: NeighborRule::default(),
            epistemic_scale: epistemic_default(),
            aleatoric_scale: aleatoric_default(),
            y_scale: default_scale(),
            metric: DistanceMetric::default(),
            local_neighbors: None,
            fit: SurrogateFit::default(),
            fit_candidates: None,
            fit_samples: None,
        }
    }
}

impl SurrogateRecipe {
    pub fn fit_budget(&self) -> Result<(usize, usize), String> {
        match (self.fit, self.fit_candidates, self.fit_samples) {
            (SurrogateFit::Fixed, None, None) => Ok((0, 0)),
            (SurrogateFit::Fixed, _, _) => {
                Err("fit='fixed' cannot specify fit-candidates or fit-samples".into())
            }
            (SurrogateFit::LeaveOneOut, Some(candidates), Some(samples))
                if candidates > 0 && samples > 0 =>
            {
                Ok((candidates, samples))
            }
            (SurrogateFit::LeaveOneOut, _, _) => {
                Err("fit='leave-one-out' requires positive fit-candidates and fit-samples".into())
            }
        }
    }
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[deser(rename_all = "kebab-case")]
pub enum NeighborRule {
    /// Use the configured neighbor count, clamped to available history at ask.
    #[default]
    Fixed,
    /// Request the selection kernel's query-local risk-prefix rule.
    RiskPrefix,
}

/// Keep schema defaults aligned with the resident ask's initial neighbor
/// budget.
pub fn default_neighbors() -> i32 {
    crate::trials::RoundConfig::default().surrogate.neighbors as i32
}

/// Initial epistemic scale from the same ask defaults used by the runtime.
pub fn epistemic_default() -> f64 {
    f64::from(
        crate::trials::RoundConfig::default()
            .surrogate
            .epistemic_scale,
    )
}

/// Initial aleatoric scale from the resident ask, before optional fitting.
pub fn aleatoric_default() -> f64 {
    f64::from(
        crate::trials::RoundConfig::default()
            .surrogate
            .aleatoric_scale,
    )
}

/// Initial output scale from the resident ask, before optional fitting.
pub fn default_scale() -> f64 {
    f64::from(crate::trials::RoundConfig::default().surrogate.output_scale)
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[deser(rename_all = "kebab-case")]
pub enum SurrogateFit {
    /// Keep supplied hyperparameters; neighbor adaptation remains independent.
    #[default]
    Fixed,
    #[deser(rename = "leave-one-out")]
    LeaveOneOut,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[deser(
    tag = "acquisition",
    rename_all = "kebab-case",
    rename_all_fields = "kebab-case",
    deny_unknown_fields
)]
pub enum AcquisitionSpec {
    #[deser(rename = "upper-confidence-bound")]
    Ucb {
        #[deser(default = default_beta())]
        beta: f64,
    },
    #[deser(rename = "thompson")]
    Thompson,
    /// Uniform selection control over the generated candidate pool.
    Random,
    Pareto {
        scales: Vec<f32>,
    },
}

#[path = "optimizer/bounds.rs"]
mod bounds;
pub use bounds::*;

/// Reuse the shared UCB exploration default rather than a second local
/// constant.
fn default_beta() -> f64 {
    let AcquisitionConfig::UCB { beta } = AcquisitionConfig::default() else {
        unreachable!()
    };
    beta
}

impl crate::optimizer_config::OptimizerConfig {
    /// Compile an ask/tell [`OptimizerConfig`] directly from a v4
    /// [`OptimizerSpec`].
    pub fn from_spec(spec: &OptimizerSpec) -> Result<Self, crate::error::ENNError> {
        use crate::error::ENNError;
        use crate::optimizer_config::{AcquisitionConfig, CandidateConfig, SurrogateConfig};
        use crate::trust_region::TrustRegionConfig;
        match spec {
            OptimizerSpec::Muon | OptimizerSpec::Adamw | OptimizerSpec::Ppo => {
                Err(ENNError::InvalidParameter(
                    "Muon, AdamW, and PPO are gradient optimizers, not ask/tell BO recipes"
                        .to_string(),
                ))
            }
            OptimizerSpec::SingleStateBo {
                surrogate,
                trust_region,
                ..
            } => {
                let candidate_generation = spec.candidate_generation().unwrap();
                let candidates = CandidateConfig::from_spec(&candidate_generation)?;
                let surrogate_cfg = SurrogateConfig::from_recipe(surrogate)?;
                let trust_region = TrustRegionConfig::from_spec(trust_region)?;
                Ok(Self {
                    surrogate: surrogate_cfg,
                    trust_region,
                    candidates,
                    acquisition: AcquisitionConfig::UCB { beta: 1.0 },
                    noise_aware: false,
                    failure_tolerance_dim: None,
                })
            }
            OptimizerSpec::BayesianSearch {
                surrogate,
                acquisition,
                trust_region,
                ..
            } => {
                let candidate_generation = spec.candidate_generation().unwrap();
                let candidates = CandidateConfig::from_spec(&candidate_generation)?;
                let surrogate_cfg = if matches!(acquisition, AcquisitionSpec::Random) {
                    SurrogateConfig::None
                } else {
                    SurrogateConfig::from_recipe(surrogate)?
                };
                let acq_cfg = AcquisitionConfig::from_spec(acquisition);
                let tr_cfg = TrustRegionConfig::from_spec(trust_region)?;
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

impl crate::optimizer_config::SurrogateConfig {
    pub fn from_recipe(recipe: &SurrogateRecipe) -> Result<Self, crate::error::ENNError> {
        use crate::error::ENNError;
        use crate::semantics::neighbor::DistanceMetric as GeomMetric;
        use crate::surrogate::ENNSurrogateConfig;
        if recipe.neighbors < 1 {
            return Err(ENNError::InvalidParameter(format!(
                "k (number of neighbors) must be > 0, got {}",
                recipe.neighbors
            )));
        }
        crate::params::ENNParams::new(
            recipe.neighbors,
            recipe.epistemic_scale,
            recipe.aleatoric_scale,
        )
        .map_err(|e| ENNError::InvalidParameter(e.to_string()))?;
        if !recipe.y_scale.is_finite() || recipe.y_scale <= 0.0 {
            return Err(ENNError::InvalidParameter(format!(
                "y_scale must be finite and positive, got {}",
                recipe.y_scale
            )));
        }
        let local_k = recipe.local_neighbors.unwrap_or(8);
        if local_k == 0 || local_k > 128 {
            return Err(ENNError::InvalidParameter(
                "local-neighbors must be in 1..=128".into(),
            ));
        }
        if recipe.metric == DistanceMetric::Euclidean && recipe.local_neighbors.is_some() {
            return Err(ENNError::InvalidParameter(
                "local-neighbors requires metric = 'conformal', 'hyperbolic', or 'wasserstein'"
                    .into(),
            ));
        }
        let geometry = match recipe.metric {
            DistanceMetric::Euclidean => GeomMetric::Euclidean,
            DistanceMetric::Conformal => GeomMetric::Conformal { local_k },
            DistanceMetric::Hyperbolic => GeomMetric::Hyperbolic { local_k },
            DistanceMetric::Wasserstein => GeomMetric::Wasserstein { local_k },
        };
        let (num_candidates, num_samples) =
            recipe.fit_budget().map_err(ENNError::InvalidParameter)?;
        let y_scale = (recipe.fit == SurrogateFit::Fixed
            && (recipe.y_scale - default_scale()).abs() > 1.0e-12)
            .then_some(recipe.y_scale);
        Ok(Self::ENN(ENNSurrogateConfig {
            k: recipe.neighbors,
            num_candidates,
            num_samples,
            epistemic_scale: recipe.epistemic_scale,
            aleatoric_scale: recipe.aleatoric_scale,
            y_scale,
            adapt_neighbors: recipe.neighbor_rule == NeighborRule::RiskPrefix,
            geometry,
            ..Default::default()
        }))
    }
}

impl crate::optimizer_config::CandidateConfig {
    pub fn from_spec(spec: &CandidateGenerationSpec) -> Result<Self, crate::error::ENNError> {
        use crate::error::ENNError;
        if spec.arms == 0 || spec.candidates == 0 || !spec.candidates.is_multiple_of(spec.arms) {
            return Err(ENNError::InvalidParameter(
                "candidates must be positive and divisible by arms".to_string(),
            ));
        }
        let pert = spec.perturbation.clone().unwrap_or_default();
        pert.validate().map_err(ENNError::InvalidParameter)?;
        let per_arm = (spec.candidates / spec.arms) as usize;
        let total = spec.candidates as usize;
        Ok(Self {
            num_candidates_factor: 1.0,
            min_candidates: total,
            max_candidates: Some(total),
            num_candidates_per_arm: Some(per_arm),
            num_pert: pert.num_pert,
            candidate_rv: pert.sampler,
            raasp_fast: pert.raasp_fast,
            coord_system: pert.coord_system,
            tied_dims: Vec::new(),
        })
    }
}

impl crate::optimizer_config::AcquisitionConfig {
    pub fn from_spec(spec: &AcquisitionSpec) -> Self {
        match spec {
            AcquisitionSpec::Ucb { beta } => Self::UCB { beta: *beta },
            AcquisitionSpec::Thompson => Self::Thompson,
            AcquisitionSpec::Random => Self::Random,
            AcquisitionSpec::Pareto { .. } => Self::Pareto,
        }
    }
}

impl crate::trust_region::TrustRegionConfig {
    pub fn from_spec(bounds: &TrustRegionBounds) -> Result<Self, crate::error::ENNError> {
        bounds
            .validate()
            .map_err(crate::error::ENNError::InvalidParameter)?;
        Ok(Self::Turbo(crate::trust_region::TRLengthConfig {
            length_init: bounds.initial,
            length_min: bounds.min,
            length_max: bounds.max,
        }))
    }
}
