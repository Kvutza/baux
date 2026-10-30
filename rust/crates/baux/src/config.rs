//! Configuration types for the optimizer.

use crate::candidates::CandidateSampler;
use crate::index::IndexDriver;
use crate::morbo::{MorboTRSettings, Rescalarize};
use crate::store::EnnStorage;
use crate::trust_region::{TRLengthConfig, TrustRegionConfig};
use deser::{Deserialize, Serialize};
use std::path::PathBuf;

pub use crate::optimizer_config::{
    AcquisitionConfig, CandidateConfig, InitStrategy, OptimizerConfig, SurrogateConfig, lhd_only,
    turbo_enn, turbo_zero,
};

#[path = "config/experiment.rs"]
mod cfg_experiment;
mod diffusion;
mod feedback;
mod generation;
pub use diffusion::{DiffusionConfig, IndexMode};
mod initialization;
mod kernel;
mod objectives;
mod resident;
mod seeds;
pub use resident::ResidentEnnConfig;
mod procedural;
mod spec;
#[path = "config/spec/conversion.rs"]
mod spec_conversion;
#[path = "config/spec/execution.rs"]
mod spec_execution;
#[path = "config/spec/optimizer.rs"]
mod spec_optimizer;
mod validation;
pub use feedback::FeedbackTransition;
pub use generation::{
    FrozenAttention, FrozenComputeMode, FrozenReadout, GenerationConfig, GenerationPurpose,
    GenerationReward, GenerationTask, SignalGate, VerifyConfig, VerifyMode,
};
pub use initialization::ModelInitialization;
pub use kernel::KernelTrial;
pub use objectives::{
    ObjectiveAcquisition, ObjectivePolicy, ResidentObjectiveConfig, ResidentObjectiveMode,
};
pub use spec::ExperimentSpec;
pub use spec_execution::{
    DataSpec, DiagnosticSpec, EvaluationSpec, ExecutionSpec, ObjectiveFunction, ObjectiveSampling,
    ObjectiveSpec,
};
pub use spec_optimizer::{
    AcceptanceCriterion, AcceptanceSpec, AcquisitionSpec, CandidateGenerationSpec, OptimizerSpec,
    ProposalKernelSpec, RadiusControlSpec, SurrogateEstimator, SurrogateFit, SurrogateRecipe,
    TrustRegionBounds, TrustRegionSpec,
};
#[cfg(test)]
#[path = "config/spec/tests.rs"]
mod spec_tests;
pub use cfg_experiment::{
    DistanceScaling, HistoryGeometry, ObjectiveReference, PretrainSelection, TrainingOptimizer,
    TrustRegionShape,
};

/// Concrete runtime configuration compiled from a validated experiment recipe.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[deser(default, deny_unknown_fields)]
pub struct RunConfig {
    /// Parameter update rule used by pretraining.
    pub optimizer: Option<TrainingOptimizer>,
    /// Explicit independent objective acquisition; never inferred from scalar
    /// rewards.
    pub objective_acquisition: Option<ResidentObjectiveConfig>,
    pub generation: Option<GenerationConfig>,
    /// Logical procedural proposal layout. Current resident kernels accept the
    /// legacy one-arm/four-slot layout only; larger requests fail validation.
    pub proposal_pool: Option<crate::procedural_pool::ProceduralPoolConfig>,
    /// Program used to turn procedural random bits into a full-coordinate
    /// direction.
    pub proposal_method: Option<crate::procedural_pool::ProposalMethod>,
    /// File-driven experiment to run. Programmatic callers may omit it.
    pub experiment: Option<ExperimentKind>,
    pub acquisition: Option<AcquisitionConfig>,
    /// Neighbors used by the resident accelerator ENN.
    pub k_neighbors: Option<i32>,
    /// Fixed epistemic distance scale for the resident accelerator ENN.
    pub epistemic_scale: Option<f64>,
    /// Fixed aleatoric variance floor for the resident accelerator ENN.
    pub aleatoric_scale: Option<f64>,
    /// Output uncertainty scale used by accelerator acquisition kernels.
    pub y_scale: Option<f64>,
    /// Geometry normalization used before ENN neighbor weighting.
    pub distance_scaling: Option<DistanceScaling>,
    /// Coordinate system used for resident ENN history distances.
    pub history_geometry: Option<HistoryGeometry>,
    /// Neighbor rank defining each point's local radius under self-tuning
    /// scaling.
    pub local_scale_neighbors: Option<usize>,
    /// Fit resident ENN variance scales by leave-one-out predictive likelihood.
    pub fit_params: Option<bool>,
    /// Select a query-local neighbor prefix using the configured risk rule.
    pub adapt_neighbors: Option<bool>,
    pub candidate_rv: Option<CandidateSampler>,
    pub raasp_fast: Option<bool>,
    pub num_candidates_factor: Option<f64>,
    pub min_candidates: Option<usize>,
    pub max_candidates: Option<usize>,
    pub num_candidates_per_arm: Option<usize>,
    pub num_pert: Option<usize>,
    pub length_init: Option<f64>,
    pub length_min: Option<f64>,
    pub length_max: Option<f64>,
    pub index_driver: Option<IndexDriver>,
    pub num_samples: Option<usize>,
    pub num_candidates: Option<usize>,
    pub scale_x: Option<bool>,
    pub noise_aware: Option<bool>,
    pub failure_tolerance_dim: Option<f64>,
    pub enn_storage: Option<EnnStorage>,
    pub work_dir: Option<PathBuf>,
    pub y_bounds: Option<Vec<[f64; 2]>>,
    pub trust_region_kind: Option<TrustRegionKind>,
    pub num_metrics: Option<usize>,
    pub alpha: Option<f64>,
    pub rescalarize: Option<Rescalarize>,
    /// Artifact root for a file-driven optimizer run.
    pub output: Option<PathBuf>,
    /// Number of complete optimization rounds to measure.
    pub rounds: Option<u32>,
    /// Deterministic repetitions of the same experiment.
    pub reps: Option<u32>,
    /// Required maximum latency for every measured round.
    pub target_round_ms: Option<u32>,
    /// Emit perturbative per-operation timing records for the fixed round
    /// experiment.
    pub trace: Option<bool>,
    /// Opt-in shader replacement and full-loop numerical capture for kernel
    /// search.
    pub kernel_trial: Option<KernelTrial>,
    /// Timestamp complete scorer stages; also profiles the selected gate
    /// experiment.
    pub scorer_stage_samples: Option<u32>,
    /// Score every member of one guided four-candidate pool on one frozen
    /// corpus condition, without updating between candidates.
    pub pool_rank_audit: Option<u32>,
    /// Seed for model initialization.
    pub model_seed: Option<u64>,
    /// Seed for the persistent proposal direction.
    pub reference_seed: Option<u64>,
    /// Base seed for per-round proposals.
    pub proposal_seed: Option<u64>,
    /// Base seed for per-round acquisition.
    pub acquisition_seed: Option<u64>,
    /// Fixed BAUXPTN1 token stream for causal pretraining.
    pub dataset: Option<PathBuf>,
    /// Immutable held-out stream, never supplied to optimizer acceptance.
    pub validation_dataset: Option<PathBuf>,
    /// Tokens in one fixed-corpus causal-likelihood evaluation window.
    pub window_tokens: Option<u32>,
    pub validation_interval: Option<u32>,
    /// Candidate-selection ablation; the remaining ENN policy stays unchanged.
    pub selection: Option<PretrainSelection>,
    /// Versioned model architecture preset for a pretraining experiment.
    pub model: Option<ModelArchitecture>,
    /// Versioned immutable data recipe for a pretraining experiment.
    #[deser(alias = "corpus")]
    pub data_recipe: Option<DataRecipe>,
    /// Full-coordinate perturbation distribution generated by the accelerator.
    pub perturbation: Option<crate::ProposalKernel>,
    /// Trust-region shape policy for the full-weight pretrain experiment.
    pub trust_region_shape: Option<TrustRegionShape>,
    /// Per-minibatch control variate used by the pretraining objective.
    pub objective_reference: Option<ObjectiveReference>,
    /// Scalar token loss function used by fixed-corpus pretraining.
    pub objective_function: Option<ObjectiveFunction>,
    /// Reliability-aware controller parameters for full-weight pretraining.
    pub reliability_controller: Option<crate::ReliabilityPolicy>,
}

/// Complete workload selected by a file-driven optimizer configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[deser(rename_all = "kebab-case")]
pub enum ExperimentKind {
    EndToEnd,
    MoeLayer,
    Pretrain,
    Generation,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[deser(rename_all = "kebab-case")]
#[repr(u8)]
pub enum ModelArchitecture {
    #[deser(rename = "fbt-pisa1-legacy-v1")]
    FbtPisa1MoeV1 = 0,
    FbtPisa1Residual1V1 = 1,
    FbtPisa1ProjectedBoundaryV1 = 2,
    FbtPisa1Hc4V1 = 3,
    FbtPisa1Mhc4V1 = 4,
    FbtPisa1LoopedMhc4V1 = 5,
    FbtPisa1DiffusionMhc4V1 = 6,
    FbtPisa1HnetMhc4V1 = 7,
}

impl ModelArchitecture {
    pub const ALL: [Self; 8] = [
        Self::FbtPisa1MoeV1,
        Self::FbtPisa1Residual1V1,
        Self::FbtPisa1ProjectedBoundaryV1,
        Self::FbtPisa1Hc4V1,
        Self::FbtPisa1Mhc4V1,
        Self::FbtPisa1LoopedMhc4V1,
        Self::FbtPisa1DiffusionMhc4V1,
        Self::FbtPisa1HnetMhc4V1,
    ];

    const IDS: [&'static str; 8] = [
        "fbt-pisa1-legacy-v1",
        "fbt-pisa1-residual1-v1",
        "fbt-pisa1-projected-boundary-v1",
        "fbt-pisa1-hc4-v1",
        "fbt-pisa1-mhc4-v1",
        "fbt-pisa1-looped-mhc4-v1",
        "fbt-pisa1-diffusion-mhc4-v1",
        "fbt-pisa1-hnet-mhc4-v1",
    ];

    pub const fn id(self) -> &'static str {
        Self::IDS[self as usize]
    }
}

/// Checked-in immutable data recipe identifier.
pub type DataRecipe = String;

pub fn valid_recipe(value: &str) -> bool {
    (1..=64).contains(&value.len())
        && value
            .bytes()
            .next()
            .is_some_and(|byte| byte.is_ascii_lowercase())
        && !value.ends_with('-')
        && !value.contains("--")
        && value
            .bytes()
            .all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit() || byte == b'-')
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[deser(rename_all = "snake_case")]
pub enum TrustRegionKind {
    Turbo,
    Morbo,
    Reliability,
}

fn apply_enn(config: &mut OptimizerConfig, overrides: &RunConfig) {
    let SurrogateConfig::ENN(enn_cfg) = &config.surrogate else {
        return;
    };
    let mut enn = enn_cfg.clone();
    if let Some(k) = overrides.k_neighbors {
        enn.k = k;
    }
    if let Some(driver) = overrides.index_driver {
        enn.index_driver = driver;
    }
    if let Some(nfs) = overrides.num_samples {
        enn.num_samples = nfs;
    }
    if let Some(nfc) = overrides.num_candidates {
        enn.num_candidates = nfc;
    }
    if let Some(sx) = overrides.scale_x {
        enn.scale_x = sx;
    }
    if let Some(storage) = overrides.enn_storage {
        enn.storage = storage;
    }
    if let Some(dir) = overrides.work_dir.clone() {
        enn.work_dir = Some(dir);
    }
    if let Some(rows) = overrides.y_bounds.as_ref() {
        let flat = rows.iter().flatten().copied().collect::<Vec<_>>();
        enn.y_bounds = Some(
            ndarray::Array2::from_shape_vec((rows.len(), 2), flat)
                .expect("fixed-width y_bounds rows"),
        );
    }
    config.surrogate = SurrogateConfig::ENN(enn);
}

fn apply_region(overrides: &RunConfig, config: &mut OptimizerConfig) {
    if let Some(kind) = overrides.trust_region_kind {
        if kind == TrustRegionKind::Morbo {
            let num_metrics = overrides.num_metrics.unwrap_or(2);
            let alpha = overrides.alpha.unwrap_or(0.05);
            let length = TRLengthConfig {
                length_init: overrides.length_init.unwrap_or(0.8),
                length_min: overrides.length_min.unwrap_or(0.5f64.powi(7)),
                length_max: overrides.length_max.unwrap_or(1.6),
            };
            let rescalarize = overrides.rescalarize.unwrap_or(Rescalarize::OnPropose);
            config.trust_region = TrustRegionConfig::Morbo(MorboTRSettings {
                num_metrics,
                alpha,
                length,
                rescalarize,
                noise_aware: overrides.noise_aware.unwrap_or(false),
            });
            return;
        }
    }
    if overrides.length_init.is_none()
        && overrides.length_min.is_none()
        && overrides.length_max.is_none()
    {
        return;
    }
    let TRLengthConfig {
        length_init,
        length_min,
        length_max,
    } = match &config.trust_region {
        TrustRegionConfig::Turbo(cfg) => *cfg,
        TrustRegionConfig::Morbo(m) => m.length,
    };
    let updated = TRLengthConfig {
        length_init: overrides.length_init.unwrap_or(length_init),
        length_min: overrides.length_min.unwrap_or(length_min),
        length_max: overrides.length_max.unwrap_or(length_max),
    };
    config.trust_region = match &config.trust_region {
        TrustRegionConfig::Turbo(_) => TrustRegionConfig::Turbo(updated),
        TrustRegionConfig::Morbo(m) => {
            let mut morbo = m.clone();
            morbo.length = updated;
            TrustRegionConfig::Morbo(morbo)
        }
    };
}

impl RunConfig {
    /// Materialize the optimizer configuration used by the runtime.
    pub fn apply_to(&self, mut config: OptimizerConfig) -> OptimizerConfig {
        if let Some(acq) = self.acquisition {
            config.acquisition = acq;
        }
        if let Some(rv) = self.candidate_rv {
            config.candidates.candidate_rv = rv;
        }
        if let Some(fast) = self.raasp_fast {
            config.candidates.raasp_fast = fast;
        }
        if let Some(f) = self.num_candidates_factor {
            config.candidates.num_candidates_factor = f;
        }
        if let Some(m) = self.min_candidates {
            config.candidates.min_candidates = m;
        }
        if let Some(cap) = self.max_candidates {
            config.candidates.max_candidates = Some(cap);
        }
        if let Some(m) = self.num_candidates_per_arm {
            config.candidates.num_candidates_per_arm = Some(m);
        }
        if let Some(n) = self.num_pert {
            config.candidates.num_pert = n.max(1);
        }
        apply_region(self, &mut config);
        if self.k_neighbors.is_some()
            || self.index_driver.is_some()
            || self.num_samples.is_some()
            || self.num_candidates.is_some()
            || self.scale_x.is_some()
            || self.enn_storage.is_some()
            || self.work_dir.is_some()
            || self.y_bounds.is_some()
        {
            apply_enn(&mut config, self);
        }
        if let Some(na) = self.noise_aware {
            config.noise_aware = na;
        }
        if let Some(d) = self.failure_tolerance_dim {
            config.failure_tolerance_dim = Some(d);
        }
        config
    }
}

#[cfg(test)]
mod serialization_tests {
    use super::ExperimentSpec;
    use super::{AcquisitionConfig, ExperimentKind, RunConfig, TrustRegionKind, parse_tune};
    use crate::{CandidateSampler, Rescalarize};

    #[test]
    fn config_names() {
        let parsed: RunConfig = baux::artifact::toml::from_str(
            r#"
            acquisition = { ucb = { beta = 3.5 } }
            candidate_rv = "sobol"
            num_samples = 14
            trust_region_kind = "morbo"
            rescalarize = "on-propose"
            "#,
        )
        .unwrap();

        assert!(matches!(
            parsed.acquisition,
            Some(AcquisitionConfig::UCB { beta: 3.5 })
        ));
        assert_eq!(parsed.candidate_rv, Some(CandidateSampler::Sobol));
        assert_eq!(parsed.num_samples, Some(14));
        assert_eq!(parsed.trust_region_kind, Some(TrustRegionKind::Morbo));
        assert_eq!(parsed.rescalarize, Some(Rescalarize::OnPropose));
    }

    #[test]
    fn config_fields() {
        let error = baux::artifact::toml::from_str::<RunConfig>("fit_samples = 10").unwrap_err();
        assert!(error.to_string().contains("unknown field"));
        let parsed = baux::artifact::toml::from_str::<RunConfig>("rounds = 3").unwrap();
        assert_eq!(parsed.rounds(), 3);
        assert_eq!(crate::ExperimentKind::EndToEnd, ExperimentKind::EndToEnd);
        assert_eq!(
            crate::prelude::ExperimentKind::EndToEnd,
            ExperimentKind::EndToEnd
        );
    }

    #[test]
    fn recipe_seeds() {
        let input = "version=4\nexperiment='pretrain'\nmodel='fbt-pisa1-mhc4-v1'\n[data]\nrecipe='\
                     stack-v3-rust-go-v1'\n[evaluation]\ninterval=4\n[optimizer]\nalgorithm='muon'\
                     \n[objective]\nfunction='cross-entropy'\nsampling='natural'";
        let run = parse_tune(input).unwrap();
        assert_ne!(run.model_seeded(0), run.model_seeded(1));
        assert!(parse_tune(&input.replace("interval=4", "interval=0")).is_err());
    }
}

/// Parse one closed, typed version 4 experiment document.
pub fn parse_tune(text: &str) -> Result<RunConfig, String> {
    ExperimentSpec::parse(text)?.compile()
}

pub fn load_tune(path: &std::path::Path) -> Result<(RunConfig, String), String> {
    let result = || {
        let text = std::fs::read_to_string(path).map_err(|error| error.to_string())?;
        let mut config = parse_tune(&text)?;
        let absolute = std::fs::canonicalize(path).map_err(|error| error.to_string())?;
        let parent = absolute
            .parent()
            .ok_or("configuration path has no parent")?;
        let output = parent.join(config.output());
        config.output = Some(output.clone());
        if let Some(dataset) = config.dataset.as_ref() {
            let dataset = if dataset.is_absolute() {
                dataset.clone()
            } else {
                parent.join(dataset)
            };
            let dataset = std::fs::canonicalize(&dataset)
                .map_err(|error| format!("dataset path {}: {error}", dataset.display()))?;
            config.dataset = Some(dataset);
        }
        if let Some(dataset) = config.validation_dataset.as_ref() {
            config.validation_dataset = Some(std::fs::canonicalize(parent.join(dataset)).map_err(
                |error| format!("held-out dataset path {}: {error}", dataset.display()),
            )?);
        }
        if let Some(trial) = &mut config.kernel_trial {
            trial.resolve(parent)?;
        }
        if let Some(generation) = &mut config.generation {
            generation.resolve(parent)?;
        }
        let resolved = ExperimentSpec::from_runtime(&config)?.to_toml()?;
        Ok((config, resolved))
    };
    result().map_err(|error: String| format!("optimizer config {}: {error}", path.display()))
}

fn validate_lengths(length: TRLengthConfig) -> Result<(), String> {
    for value in [length.length_init, length.length_min, length.length_max] {
        if !value.is_finite() || !(value as f32).is_finite() || value as f32 <= 0.0 {
            return Err("trust-region lengths must be positive finite FP32 values".into());
        }
    }
    if length.length_min >= length.length_max
        || length.length_init < length.length_min
        || length.length_init > length.length_max
    {
        return Err("trust-region lengths must satisfy min <= init <= max and min < max".into());
    }
    Ok(())
}

impl RunConfig {
    pub fn validate_experiment(&self) -> Result<(), String> {
        let pretrain = self.validate_selection()?;
        if self.experiment == Some(ExperimentKind::Generation) {
            let generation = self
                .generation
                .as_ref()
                .ok_or("generation experiment requires [generation]")?;
            generation.validate()?;
            if self.reps() != 1
                || self.data_recipe.is_some()
                || self.dataset.is_some()
                || self.objective_reference.is_some()
                || self.kernel_trial.is_some()
                || self.trace == Some(true)
            {
                return Err(
                    "generation uses explicit tasks, one repetition, and no pretrain objective or \
                     diagnostics"
                        .into(),
                );
            }
            if !generation.corpus_prompt.is_empty() || generation.corpus_prompt_tokens.is_some() {
                return Err(
                    "generation experiment requires explicit tasks; corpus_prompt is for \
                     generated pretraining"
                        .into(),
                );
            }
            let fields = baux::artifact::json::to_value(self).map_err(|error| error.to_string())?;
            if fields.as_map().unwrap().iter().any(|(key, value)| {
                !value.is_null()
                    && (key.as_str().is_some_and(|key| key.ends_with("_pairs"))
                        || key == "scorer_stage_samples")
            }) {
                return Err("generation cannot contain scorer diagnostics".into());
            }
        } else if let Some(generation) = &self.generation {
            if self.experiment != Some(ExperimentKind::Pretrain) {
                return Err("[generation] requires a generation or pretrain experiment".into());
            }
            generation.validate()?;
            if (generation.corpus_prompt.is_empty() && generation.corpus_prompt_tokens.is_none())
                || !generation.tasks.is_empty()
                || self.data_recipe.is_none()
            {
                return Err(
                    "generated pretraining requires data.recipe, corpus_prompt, and no explicit \
                     generation tasks"
                        .into(),
                );
            }
            if self.kernel_trial.is_some() {
                return Err("generated pretraining cannot include kernel trials".into());
            }
        }
        self.validate_fields(pretrain)?;
        self.validate_seeds()?;
        self.validate_resident(pretrain)?;
        self.validate_controller()
    }

    pub fn length(&self) -> TRLengthConfig {
        let effective = self.apply_to(turbo_enn());
        let TrustRegionConfig::Turbo(length) = effective.trust_region else {
            unreachable!("validated resident optimizer configuration")
        };
        length
    }

    pub fn reliability_controller(&self) -> Option<crate::ReliabilityPolicy> {
        (self.trust_region_kind == Some(TrustRegionKind::Reliability))
            .then_some(self.reliability_controller)
            .flatten()
    }

    pub fn output(&self) -> PathBuf {
        self.output
            .clone()
            .unwrap_or_else(|| PathBuf::from("results/baux-tune"))
    }

    pub fn rounds(&self) -> u32 {
        self.rounds.unwrap_or(3)
    }

    pub fn reps(&self) -> u32 {
        self.reps.unwrap_or(1)
    }

    pub fn target_ms(&self) -> u32 {
        self.target_round_ms.unwrap_or(1000)
    }

    pub fn optimizer(&self) -> TrainingOptimizer {
        self.optimizer.unwrap_or_default()
    }

    pub fn trace(&self) -> bool {
        self.trace.unwrap_or(false)
    }

    pub fn objective_reference(&self) -> ObjectiveReference {
        self.objective_reference.unwrap_or_default()
    }

    pub fn objective_function(&self) -> ObjectiveFunction {
        self.objective_function.unwrap_or_default()
    }

    pub fn dataset(&self) -> Option<&std::path::Path> {
        self.dataset.as_deref()
    }

    pub fn proposal_kernel(&self) -> crate::ProposalKernel {
        self.perturbation.unwrap_or_default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::candidates::CandidateSampler;
    use crate::store::EnnStorage;
    use std::path::Path;

    #[test]
    fn test_003() {
        let config = CandidateConfig::default();

        // Basic case: 2D, 1 arm
        let n = config.num_candidates(2, 1);
        assert_eq!(n, 200);

        // Larger dimension
        let n_large = config.num_candidates(10, 1);
        assert_eq!(n_large, 1000);

        // More arms
        let n_arms = config.num_candidates(2, 10);
        assert_eq!(n_arms, 200);
    }

    #[test]
    fn test_004() {
        // ENN 0.4.1 default: max(10, min(5000, 100*num_dim)).
        let config = CandidateConfig {
            num_candidates_factor: 100.0,
            min_candidates: 10,
            max_candidates: Some(5000),
            num_candidates_per_arm: None,
            num_pert: 20,
            candidate_rv: CandidateSampler::Uniform,
            raasp_fast: false,
        };
        assert_eq!(config.num_candidates(60, 1), 5000);
        assert_eq!(config.num_candidates(100, 1), 5000);
        assert_eq!(config.num_candidates(10, 1), 1000);
    }

    #[test]
    fn test_configdefaults() {
        let config = OptimizerConfig::default();
        assert!(matches!(config.acquisition, AcquisitionConfig::UCB { .. }));
    }

    #[test]
    fn test_turboennconfig() {
        let config = turbo_enn();
        assert!(matches!(config.surrogate, SurrogateConfig::ENN(_)));
        assert!(matches!(config.acquisition, AcquisitionConfig::UCB { .. }));
        assert_eq!(config.candidates.candidate_rv, CandidateSampler::RAASP);
        assert_eq!(config.candidates.num_candidates(2, 1), 200);
        assert_eq!(config.candidates.num_candidates(100, 1), 5000);
    }

    #[test]
    fn test_turbozeroconfig() {
        let config = turbo_zero();
        assert!(matches!(config.surrogate, SurrogateConfig::None));
        assert!(matches!(config.acquisition, AcquisitionConfig::Random));
        assert_eq!(config.candidates.candidate_rv, CandidateSampler::RAASP);
    }

    #[test]
    fn test_lhdonlyconfig() {
        let config = lhd_only();
        assert!(matches!(config.surrogate, SurrogateConfig::None));
        let n = config.candidates.num_candidates(10, 1);
        assert_eq!(n, 10);
    }

    #[test]
    fn test_initstrategyenum() {
        let init_default = InitStrategy::default();
        assert_eq!(init_default, InitStrategy::LHD);
        assert_eq!(InitStrategy::Random as u8, 1);
    }

    #[test]
    fn test_010() {
        use crate::index::IndexDriver;

        let overrides = RunConfig {
            acquisition: Some(AcquisitionConfig::Thompson),
            candidate_rv: Some(CandidateSampler::Sobol),
            index_driver: Some(IndexDriver::Exact),
            num_samples: Some(123),
            num_candidates: Some(456),
            scale_x: Some(true),
            ..Default::default()
        };

        let config = turbo_enn();
        let applied = overrides.apply_to(config);

        assert!(matches!(applied.acquisition, AcquisitionConfig::Thompson));
        assert_eq!(applied.candidates.candidate_rv, CandidateSampler::Sobol);
        if let SurrogateConfig::ENN(enn) = &applied.surrogate {
            assert_eq!(enn.index_driver, IndexDriver::Exact);
            assert_eq!(enn.num_samples, 123);
            assert_eq!(enn.num_candidates, 456);
            assert!(enn.scale_x);
        } else {
            panic!("expected ENN surrogate");
        }
    }

    #[test]
    fn test_011() {
        let overrides = RunConfig {
            scale_x: Some(true),
            ..Default::default()
        };
        let applied = overrides.apply_to(turbo_enn());
        let SurrogateConfig::ENN(enn) = applied.surrogate else {
            panic!("expected ENN surrogate");
        };
        assert!(enn.scale_x);
    }

    #[test]
    fn morbo_metrics() {
        use crate::morbo::MorboTrustRegion;
        use crate::trust_region::TrustRegionConfig;
        use rand::SeedableRng;
        use rand::rngs::StdRng;

        let overrides = RunConfig {
            trust_region_kind: Some(TrustRegionKind::Morbo),
            num_metrics: Some(1),
            ..Default::default()
        };
        let applied = overrides.apply_to(turbo_enn());
        let TrustRegionConfig::Morbo(settings) = applied.trust_region else {
            panic!("expected Morbo trust region");
        };
        let mut rng = StdRng::seed_from_u64(8);
        let result = MorboTrustRegion::new(2, settings, &mut rng);
        assert!(
            result.is_err(),
            "PyO3/override path must reject num_metrics=1 like Python Morbo config"
        );
    }

    #[test]
    fn candidate_arms() {
        let cfg = CandidateConfig {
            num_candidates_factor: 1.0,
            min_candidates: 10,
            max_candidates: None,
            num_candidates_per_arm: Some(25),
            num_pert: 20,
            candidate_rv: CandidateSampler::Uniform,
            raasp_fast: false,
        };
        assert_eq!(cfg.num_candidates(2, 3), 75);
        assert_eq!(cfg.num_candidates(2, 8), 200);
    }

    #[test]
    fn config_pool() {
        let overrides = RunConfig {
            num_candidates_factor: Some(1.0),
            min_candidates: Some(10),
            num_candidates_per_arm: Some(40),
            ..Default::default()
        };
        let applied = overrides.apply_to(turbo_zero());
        assert_eq!(applied.candidates.num_candidates(2, 3), 120);
        assert_eq!(applied.candidates.num_candidates(2, 8), 320);
    }

    #[test]
    fn enn_fields() {
        let overrides = RunConfig {
            num_samples: Some(7),
            num_candidates: Some(11),
            scale_x: Some(true),
            ..Default::default()
        };
        let applied = overrides.apply_to(turbo_enn());
        let SurrogateConfig::ENN(enn) = applied.surrogate else {
            panic!("expected ENN surrogate");
        };
        assert_eq!(enn.num_samples, 7);
        assert_eq!(enn.num_candidates, 11);
        assert!(enn.scale_x);
    }

    #[test]
    fn enn_dir() {
        use crate::index::IndexDriver;
        use std::path::PathBuf;

        let overrides = RunConfig {
            index_driver: Some(IndexDriver::BpAnnDisk),
            enn_storage: Some(EnnStorage::Disk),
            work_dir: Some(PathBuf::from("/tmp/enn_work")),
            ..Default::default()
        };
        let applied = overrides.apply_to(turbo_enn());
        let SurrogateConfig::ENN(enn) = applied.surrogate else {
            panic!("expected ENN surrogate");
        };
        assert_eq!(enn.index_driver, IndexDriver::BpAnnDisk);
        assert_eq!(enn.storage, EnnStorage::Disk);
        assert_eq!(enn.work_dir.as_deref(), Some(Path::new("/tmp/enn_work")));
    }

    #[test]
    fn morbo_propose() {
        let overrides = RunConfig {
            trust_region_kind: Some(TrustRegionKind::Morbo),
            num_metrics: Some(2),
            ..Default::default()
        };
        let applied = overrides.apply_to(turbo_enn());
        let TrustRegionConfig::Morbo(settings) = applied.trust_region else {
            panic!("expected Morbo trust region");
        };
        assert_eq!(
            settings.rescalarize,
            Rescalarize::OnPropose,
            "missing rescalarize should match Python MorboTRConfig default ON_PROPOSE"
        );
    }
}
