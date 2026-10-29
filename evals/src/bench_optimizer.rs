//! Optimizer configuration types and factory helpers for continuous-control
//! evaluations.

use baux::BayesOptimizer;
use baux::semantics::neighbor::DistanceMetric;
use baux::semantics::optimizer::surrogate::{AcquisitionRule, Surrogate, SurrogateSpec};
use baux::semantics::optimizer::trust_region::{TrustRegionController, TrustRegionSpec};
use deser::{Deserialize, Serialize};
use ndarray::{Array1, Array2};
use rand::RngCore;

/// Extension trait exposing incumbent coordinates and warm-start seeding on
/// `baux::BayesOptimizer`.
pub trait BayesOptimizerExt {
    /// Returns the current incumbent coordinates in unit space `[0, 1]^D`, if
    /// an incumbent exists.
    fn incumbent_x(&self) -> Option<Array1<f64>>;

    /// Warm-starts the optimizer with an initial incumbent parameter vector and
    /// active return.
    fn seed_incumbent(
        &mut self,
        params: &[f32],
        ret: f64,
        rng: &mut dyn RngCore,
    ) -> Result<(), String>;
}

impl BayesOptimizerExt for BayesOptimizer {
    fn incumbent_x(&self) -> Option<Array1<f64>> {
        self.unit_incumbent().cloned()
    }

    fn seed_incumbent(
        &mut self,
        params: &[f32],
        ret: f64,
        rng: &mut dyn RngCore,
    ) -> Result<(), String> {
        let dim = params.len();
        let bounds = self.bounds();
        let mut x = Array2::zeros((1, dim));
        for (j, &p) in params.iter().enumerate() {
            let low = bounds[[j, 0]];
            let high = bounds[[j, 1]];
            let u = ((f64::from(p) - low) / (high - low)).clamp(0.0, 1.0);
            x[[0, j]] = u;
        }
        let y = Array2::from_elem((1, 1), ret);
        let yvar = Array2::zeros((1, 1));
        self.tell_variance(&x.view(), &y.view(), Some(&yvar.view()), rng)
            .map_err(|e| e.to_string())
    }
}

/// Proposal and perturbation generator choice.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, Default)]
#[deser(rename_all = "kebab-case")]
pub enum ProposalGenerator {
    #[default]
    AdjointChain,
    Independent,
    PairRotation,
    Compass,
}

/// Candidate and proposal generation configuration.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[deser(default, rename_all = "kebab-case")]
pub struct ProposalConfig {
    pub generator: ProposalGenerator,
    pub distribution: String,
}

impl Default for ProposalConfig {
    fn default() -> Self {
        Self {
            generator: ProposalGenerator::AdjointChain,
            distribution: "gaussian".to_string(),
        }
    }
}

/// Surrogate estimator algorithm choice.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, Default)]
#[deser(rename_all = "kebab-case")]
pub enum SurrogateKind {
    #[default]
    Knn,
    Enn,
}

/// Detailed surrogate configuration matching full repository capabilities.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[deser(default, rename_all = "kebab-case")]
pub struct SurrogateConfig {
    pub estimator: SurrogateKind,
    pub index: String,
    pub neighbors: usize,
    #[deser(alias = "neighbor_rule")]
    pub neighbor_rule: String,
    #[deser(alias = "epistemic_scale")]
    pub epistemic_scale: f32,
    #[deser(alias = "aleatoric_scale")]
    pub aleatoric_scale: f32,
    #[deser(alias = "y_scale")]
    pub y_scale: f32,
    pub fit: String,
    #[deser(alias = "fit_candidates")]
    pub fit_candidates: usize,
    #[deser(alias = "fit_samples")]
    pub fit_samples: usize,
}

impl Default for SurrogateConfig {
    fn default() -> Self {
        Self {
            estimator: SurrogateKind::Knn,
            index: "exact".to_string(),
            neighbors: 8,
            neighbor_rule: "fixed".to_string(),
            epistemic_scale: 0.05,
            aleatoric_scale: 0.05,
            y_scale: 1.0,
            fit: "fixed".to_string(),
            fit_candidates: 4,
            fit_samples: 4,
        }
    }
}

fn resolve_index(
    name: &str,
) -> Result<
    (
        baux::IndexDriver,
        Option<baux::EnnStorage>,
        Option<std::path::PathBuf>,
    ),
    String,
> {
    match name {
        "" | "exact" => Ok((baux::IndexDriver::Exact, None, None)),
        "auto" => Ok((baux::IndexDriver::Auto, None, None)),
        "usearch" => Ok((baux::IndexDriver::USearch, None, None)),
        "metal" => Ok((baux::IndexDriver::Metal, None, None)),
        "opencl" => Ok((baux::IndexDriver::OpenCl, None, None)),
        "cuda" => Ok((baux::IndexDriver::Cuda, None, None)),
        "bpann" | "bp-ann" | "bp_ann_disk" => {
            let pid = std::process::id();
            let nanos = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0);
            let dir = std::env::temp_dir().join(format!("baux-bpann-{pid}-{nanos}"));
            Ok((
                baux::IndexDriver::BpAnnDisk,
                Some(baux::EnnStorage::Disk),
                Some(dir),
            ))
        }
        other => Err(format!("unsupported surrogate index: {other}")),
    }
}

/// Acquisition rule choice.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[deser(default, rename_all = "kebab-case")]
pub struct AcquisitionConfig {
    pub function: String,
    pub beta: f32,
}

impl Default for AcquisitionConfig {
    fn default() -> Self {
        Self {
            function: "upper-confidence-bound".to_string(),
            beta: 2.0,
        }
    }
}

/// Trust region radius adaptation configuration.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[deser(default, rename_all = "kebab-case")]
pub struct TrustRegionConfig {
    pub policy: String,
    pub initial: f32,
    pub min: f32,
    pub max: f32,
    #[deser(alias = "success_threshold")]
    pub success_threshold: usize,
    #[deser(alias = "failure_threshold")]
    pub failure_threshold: usize,
}

impl Default for TrustRegionConfig {
    fn default() -> Self {
        Self {
            policy: "turbo".to_string(),
            initial: 0.05,
            min: 0.001,
            max: 0.2,
            success_threshold: 3,
            failure_threshold: 4,
        }
    }
}

/// Complete optimizer configuration supporting both short-hand and full tables.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[deser(default, rename_all = "kebab-case")]
pub struct OptimizerConfig {
    #[deser(alias = "name")]
    pub algorithm: String,
    pub proposal: ProposalConfig,
    pub surrogate: SurrogateConfig,
    pub acquisition: AcquisitionConfig,
    #[deser(alias = "trust_region")]
    pub trust_region: TrustRegionConfig,
}

impl Default for OptimizerConfig {
    fn default() -> Self {
        Self {
            algorithm: "operator-bo".to_string(),
            proposal: ProposalConfig::default(),
            surrogate: SurrogateConfig::default(),
            acquisition: AcquisitionConfig::default(),
            trust_region: TrustRegionConfig::default(),
        }
    }
}

impl OptimizerConfig {
    /// Returns true when the configured algorithm selects Coordinate-Space
    /// TuRBO (`BayesOptimizer`).
    pub fn is_turbo(&self) -> bool {
        matches!(
            self.algorithm.as_str(),
            "turbo-enn" | "turbo-zero" | "lhd-only"
        )
    }

    /// Constructs Engine A (`baux::BayesOptimizer`) over `[-1.0, 1.0]^D`.
    pub fn build_turbo(&self, dim: usize, rng: &mut dyn RngCore) -> Result<BayesOptimizer, String> {
        let bounds = Array2::from_shape_fn((dim, 2), |(_, col)| if col == 0 { -1.0 } else { 1.0 });
        let k = self.surrogate.neighbors.max(1) as i32;
        let num_init = 1;
        let acq = if self.acquisition.function == "thompson" {
            baux::AcquisitionConfig::Thompson
        } else if self.acquisition.function == "random" {
            baux::AcquisitionConfig::Random
        } else {
            baux::AcquisitionConfig::UCB {
                beta: f64::from(self.acquisition.beta),
            }
        };
        let (driver, storage, work_dir) = resolve_index(&self.surrogate.index)?;
        let sampler = if dim <= 21_201 {
            baux::CandidateSampler::Sobol
        } else {
            baux::CandidateSampler::Uniform
        };
        let num_cands = if dim > 10_000 { 64 } else { 256 };
        let overrides = baux::config::RunConfig {
            acquisition: Some(acq),
            candidate_rv: Some(sampler),
            num_candidates_factor: Some(1.0),
            min_candidates: Some(num_cands),
            max_candidates: Some(num_cands),
            num_pert: Some((dim / 8).clamp(16, 128)),
            length_init: Some(f64::from(self.trust_region.initial)),
            length_min: Some(f64::from(self.trust_region.min)),
            length_max: Some(f64::from(self.trust_region.max)),
            index_driver: Some(driver),
            enn_storage: storage,
            work_dir,
            ..Default::default()
        };
        match self.algorithm.as_str() {
            "turbo-enn" => {
                baux::optimizer_factory::enn_configured(bounds, k, num_init, rng, Some(&overrides))
            }
            "turbo-zero" => {
                baux::optimizer_factory::create_configured(bounds, num_init, rng, Some(&overrides))
            }
            "lhd-only" => {
                baux::optimizer_factory::lhd_configured(bounds, num_init, rng, Some(&overrides))
            }
            other => {
                return Err(format!(
                    "unsupported coordinate-space optimizer algorithm: {other}"
                ));
            }
        }
        .map_err(|e| e.to_string())
    }

    /// Constructs Engine B's neighbor-value surrogate
    /// (`NeighborValueEstimator`).
    pub fn build_surrogate(
        &self,
        base_mean: f32,
        base_var: f32,
        seed: u64,
    ) -> Result<Surrogate, String> {
        let cfg = &self.surrogate;
        let spec = SurrogateSpec {
            neighbors: cfg.neighbors,
            epistemic: cfg.epistemic_scale,
            aleatoric: cfg.aleatoric_scale,
            scale: cfg.y_scale,
            fit_candidates: cfg.fit_candidates.max(1),
            fit_samples: cfg.fit_samples.max(1),
            fit: cfg.fit == "leave-one-out",
            adaptive: cfg.neighbor_rule == "risk-prefix",
        };
        let metric = match cfg.estimator {
            SurrogateKind::Enn => DistanceMetric::Euclidean,
            SurrogateKind::Knn => DistanceMetric::SelfTuning {
                local_k: cfg.neighbors.clamp(1, 8),
            },
        };
        let sur = Surrogate::with_spec(base_mean, base_var, seed, spec)?;
        Ok(sur.with_metric(metric))
    }

    /// Constructs Engine B's trust-region controller supporting both `turbo`
    /// and `reliability`.
    pub fn build_controller(&self) -> Result<TrustRegionController, String> {
        let cfg = &self.trust_region;
        let fail_limit = cfg.failure_threshold.max(1);
        match cfg.policy.as_str() {
            "turbo" => TrustRegionController::with_turbo(cfg.initial, cfg.min, cfg.max, fail_limit),
            "reliability" | "model-agreement" => {
                let spec = TrustRegionSpec {
                    failure_tolerance: cfg.failure_threshold,
                    ..Default::default()
                };
                TrustRegionController::from_policy(
                    cfg.initial,
                    cfg.min,
                    cfg.max,
                    fail_limit,
                    Some(spec),
                )
            }
            other => Err(format!("unsupported trust-region policy: {other}")),
        }
    }

    /// Constructs Engine B's acquisition rule for the given round.
    pub fn acquisition_rule(&self, seed: u64, round: usize) -> AcquisitionRule {
        if self.acquisition.function == "thompson" {
            AcquisitionRule::Thompson {
                seed: seed.wrapping_add(round as u64),
            }
        } else {
            AcquisitionRule::Ucb {
                beta: self.acquisition.beta,
            }
        }
    }
}
