//! Typed adaptation from authored experiment recipes to the runtime API.
use super::*;
use super::{spec_execution::*, spec_optimizer::*};

impl ExperimentSpec {
    pub fn from_runtime(config: &RunConfig) -> Result<Self, String> {
        config.validate_experiment()?;
        let matrix = matches!(
            config.optimizer(),
            TrainingOptimizer::MuonV1 | TrainingOptimizer::AdamwV1
        ) && config.experiment == Some(ExperimentKind::Pretrain);
        let fixed = config.window_tokens.is_some();
        Ok(Self {
            version: 4,
            experiment: config.experiment.unwrap(),
            model: config.model,
            output: config.output.clone(),
            recipe: config.data_recipe.clone(),
            train: config.dataset.clone(),
            validation: config.validation_dataset.clone(),
            objective: (matrix || fixed).then(|| config.objective_function()),
            window_tokens: config.window_tokens,
            rounds: config.rounds(),
            repetitions: config.reps(),
            round_target_ms: config.target_ms(),
            validation_interval: config.validation_interval,
            trace: config.trace(),
            stage_samples: config.scorer_stage_samples,
            optimizer: OptimizerSpec::from_config(config)?,
            generation: (!matrix).then(|| config.generation.clone()).flatten(),
        })
    }

    pub fn compile(&self) -> Result<RunConfig, String> {
        let optimizer = self.optimizer.training_optimizer();
        let generation = self.objective_generation(optimizer)?;
        let mut config = RunConfig {
            experiment: Some(self.experiment),
            model: self.model,
            data_recipe: self.recipe.clone(),
            output: self.output.clone(),
            optimizer: Some(optimizer),
            rounds: Some(self.rounds),
            reps: Some(self.repetitions),
            target_round_ms: Some(self.round_target_ms),
            validation_interval: self.validation_interval,
            dataset: self.train.clone(),
            validation_dataset: self.validation.clone(),
            window_tokens: self.window_tokens,
            objective_function: self.objective,
            trace: self.trace.then_some(true),
            scorer_stage_samples: self.stage_samples,
            generation,
            ..Default::default()
        };
        self.optimizer.apply(&mut config)?;
        config.validate_experiment()?;
        Ok(config)
    }

    fn objective_generation(
        &self,
        optimizer: TrainingOptimizer,
    ) -> Result<Option<GenerationConfig>, String> {
        let matrix = matches!(
            optimizer,
            TrainingOptimizer::MuonV1 | TrainingOptimizer::AdamwV1
        );
        if self.experiment != ExperimentKind::Pretrain {
            if self.objective.is_some() || self.window_tokens.is_some() {
                return Err("pretraining objective fields require experiment='pretrain'".into());
            }
            return Ok(self.generation.clone());
        }
        if matrix {
            if self.window_tokens.is_some() {
                if self.generation.is_some()
                    || self.objective != Some(ObjectiveFunction::CrossEntropy)
                {
                    return Err("fixed-corpus matrix training requires cross-entropy without generation fields".into());
                }
                return Ok(None);
            }
            if self.generation.is_some()
                || self.objective != Some(ObjectiveFunction::CrossEntropy)
                || self.window_tokens.is_some()
            {
                return Err(
                    "Muon and AdamW pretraining require objective='cross-entropy', \
                     with no generation fields"
                        .into(),
                );
            }
            return matrix_generation(self.model).map(Some);
        }
        if self.generation.is_none()
            && (!matches!(
                self.objective,
                Some(ObjectiveFunction::CrossEntropy | ObjectiveFunction::BoundedLogLikelihood)
            ) || self.window_tokens.is_none())
        {
            return Err(
                "Bayesian pretraining requires objective='cross-entropy' or \
                 'bounded-log-likelihood' and window-tokens"
                    .into(),
            );
        }
        if self.generation.is_some() && (self.objective.is_some() || self.window_tokens.is_some()) {
            return Err(
                "choose either a generated objective or a corpus cross-entropy objective".into(),
            );
        }
        Ok(self.generation.clone())
    }
}

fn matrix_generation(model: Option<ModelArchitecture>) -> Result<GenerationConfig, String> {
    if !matches!(
        model,
        Some(ModelArchitecture::FbtPisa1Mhc4V1 | ModelArchitecture::FbtPisa1HnetMhc4V1)
    ) {
        return Err("natural Muon and AdamW pretraining require fbt-pisa1-mhc4-v1".into());
    }
    Ok(GenerationConfig {
        purpose: GenerationPurpose::Pretrain,
        initialization: ModelInitialization::FbtPisa1Mhc4,
        feedback_transition: FeedbackTransition::default(),
        max_tokens: crate::context::MAX_TOKENS,
        temperature: 1.0,
        temperature_bounds: None,
        temperature_step: None,
        eos_token: None,
        seed: None,
        checkpoint: None,
        qualification_manifest: None,
        save_checkpoint: None,
        save_final_checkpoint: false,
        record_tensor_updates: false,
        signal_gate: None,
        verify: VerifyConfig {
            mode: VerifyMode::AcceptedPrefix,
            window: crate::pretrain_data::CONTEXT,
            max_window: crate::pretrain_data::CONTEXT,
            passes: 1,
            unroll: 1,
            audit_candidate: false,
        },
        draft: None,
        reward: GenerationReward::ConditionalCrossEntropy,
        corpus_prompt: Vec::new(),
        corpus_prompt_tokens: Some(crate::context::MAX_TOKENS),
        episode_dataset: None,
        tasks: Vec::new(),
    })
}

impl OptimizerSpec {
    fn from_config(config: &RunConfig) -> Result<Self, String> {
        if config.optimizer() == TrainingOptimizer::MuonV1 {
            return Ok(Self::Muon);
        }
        if config.optimizer() == TrainingOptimizer::AdamwV1 {
            return Ok(Self::Adamw);
        }
        let candidate_generation = CandidateGenerationSpec::from_config(config)?;
        let surrogate = SurrogateRecipe::from_config(config)?;
        let trust_region = TrustRegionBounds::from_config(config);
        if config.optimizer() == TrainingOptimizer::OperatorBo {
            if candidate_generation.candidates != 1 || candidate_generation.arms != 1 {
                return Err("operator-bo requires 1 candidate and 1 arm".into());
            }
            return Ok(Self::OperatorBo {
                proposal: candidate_generation.proposal,
                distribution: candidate_generation.distribution.unwrap_or_default(),
                surrogate,
                trust_region,
            });
        }
        let acquisition = AcquisitionSpec::from_config(config);
        Ok(Self::TurboEnn {
            proposal: candidate_generation.proposal,
            candidates: candidate_generation.candidates,
            arms: candidate_generation.arms,
            distribution: candidate_generation.distribution,
            acquisition,
            surrogate,
            trust_region,
        })
    }

    const fn training_optimizer(&self) -> TrainingOptimizer {
        match self {
            Self::OperatorBo { .. } => TrainingOptimizer::OperatorBo,
            Self::TurboEnn { .. } => TrainingOptimizer::ResidentEnnV1,
            Self::Muon => TrainingOptimizer::MuonV1,
            Self::Adamw => TrainingOptimizer::AdamwV1,
        }
    }

    fn apply(&self, config: &mut RunConfig) -> Result<(), String> {
        match self {
            Self::Muon | Self::Adamw => Ok(()),
            Self::OperatorBo {
                proposal,
                distribution,
                surrogate,
                trust_region,
            } => {
                let candidate_generation = CandidateGenerationSpec {
                    proposal: *proposal,
                    candidates: 1,
                    arms: 1,
                    distribution: Some(*distribution),
                };
                apply_bounds(config, trust_region)?;
                config.trust_region_kind = Some(TrustRegionKind::Turbo);
                candidate_generation.apply(config)?;
                surrogate.apply(config)?;
                Ok(())
            }
            Self::TurboEnn {
                proposal,
                candidates,
                arms,
                distribution,
                acquisition,
                surrogate,
                trust_region,
            } => {
                let candidate_generation = CandidateGenerationSpec {
                    proposal: *proposal,
                    candidates: *candidates,
                    arms: *arms,
                    distribution: *distribution,
                };
                apply_bounds(config, trust_region)?;
                config.trust_region_kind = Some(TrustRegionKind::Turbo);
                candidate_generation.apply(config)?;
                surrogate.apply(config)?;
                acquisition.apply(config);
                Ok(())
            }
        }
    }
}

impl CandidateGenerationSpec {
    fn from_config(config: &RunConfig) -> Result<Self, String> {
        let pool = config.resident_pool()?;
        Ok(Self {
            proposal: config.proposal_method.unwrap_or_default(),
            candidates: pool.count(),
            arms: pool.arms(),
            distribution: config.perturbation,
        })
    }

    fn apply(&self, config: &mut RunConfig) -> Result<(), String> {
        if self.arms == 0 || self.candidates == 0 || !self.candidates.is_multiple_of(self.arms) {
            return Err("optimizer.candidates must be positive and divisible by arms".into());
        }
        config.proposal_pool = Some(crate::procedural_pool::ProceduralPoolConfig {
            arms: self.arms,
            candidates_per_arm: self.candidates / self.arms,
        });
        config.proposal_method = Some(self.proposal);
        config.perturbation = self.distribution;
        Ok(())
    }
}

impl SurrogateRecipe {
    fn from_config(config: &RunConfig) -> Result<Self, String> {
        let resident = config.resident_enn(config.acquisition_seed())?;
        Ok(Self {
            neighbors: config.k_neighbors.unwrap_or_else(default_neighbors),
            neighbor_rule: if resident.adapt_neighbors {
                NeighborRule::RiskPrefix
            } else {
                NeighborRule::Fixed
            },
            epistemic_scale: config.epistemic_scale.unwrap_or_else(epistemic_default),
            aleatoric_scale: config.aleatoric_scale.unwrap_or_else(aleatoric_default),
            y_scale: config.y_scale.unwrap_or_else(default_scale),
            metric: resident.distance_metric,
            local_neighbors: (resident.distance_metric != DistanceMetric::Euclidean)
                .then_some(resident.local_scale_neighbors),
            fit: if resident.fit_params {
                SurrogateFit::LeaveOneOut
            } else {
                SurrogateFit::Fixed
            },
            fit_candidates: resident.fit_params.then_some(resident.num_candidates),
            fit_samples: resident.fit_params.then_some(resident.num_samples),
        })
    }

    fn apply(&self, config: &mut RunConfig) -> Result<(), String> {
        let (fit_candidates, fit_samples) = self.fit_budget()?;
        config.k_neighbors = Some(self.neighbors);
        config.local_scale_neighbors = self.local_neighbors;
        config.epistemic_scale = Some(self.epistemic_scale);
        config.aleatoric_scale = Some(self.aleatoric_scale);
        config.y_scale = Some(self.y_scale);
        config.adapt_neighbors = Some(self.neighbor_rule == NeighborRule::RiskPrefix);
        config.distance_metric = (self.metric != DistanceMetric::default()).then_some(self.metric);
        match self.fit {
            SurrogateFit::Fixed => {
                config.fit_params = Some(false);
            }
            SurrogateFit::LeaveOneOut => {
                config.fit_params = Some(true);
                config.num_candidates = Some(fit_candidates);
                config.num_samples = Some(fit_samples);
            }
        }
        Ok(())
    }
}

impl AcquisitionSpec {
    fn from_config(config: &RunConfig) -> Self {
        if config.selection == Some(PretrainSelection::Random) {
            return Self::Random;
        }
        match config.objective_acquisition.as_ref() {
            Some(vector) => Self::Pareto {
                scales: vector.scales.clone(),
            },
            None => match config.acquisition.unwrap_or_default() {
                AcquisitionConfig::UCB { beta } => Self::Ucb { beta },
                AcquisitionConfig::Thompson => Self::Thompson,
                AcquisitionConfig::Random | AcquisitionConfig::Pareto => {
                    unreachable!("validated resident acquisition")
                }
            },
        }
    }

    fn apply(&self, config: &mut RunConfig) {
        config.selection = matches!(self, Self::Random).then_some(PretrainSelection::Random);
        config.acquisition = Some(match self {
            Self::Ucb { beta } => AcquisitionConfig::UCB { beta: *beta },
            Self::Thompson => AcquisitionConfig::Thompson,
            Self::Random | Self::Pareto { .. } => AcquisitionConfig::default(),
        });
        config.objective_acquisition = match self {
            Self::Pareto { scales } => Some(ResidentObjectiveConfig {
                mode: ResidentObjectiveMode::Pareto,
                scales: scales.clone(),
                preferences: None,
                regions: None,
                alpha: None,
                seed_domain: None,
                rescalarize: None,
                clip: None,
            }),
            Self::Ucb { .. } | Self::Thompson | Self::Random => None,
        };
    }
}

impl TrustRegionBounds {
    fn from_config(config: &RunConfig) -> Self {
        let length = config.length();
        Self {
            initial: length.length_init,
            min: length.length_min,
            max: length.length_max,
            shape: config.trust_region_shape,
        }
    }
}

fn apply_bounds(config: &mut RunConfig, bounds: &TrustRegionBounds) -> Result<(), String> {
    bounds.validate()?;
    config.length_init = Some(bounds.initial);
    config.length_min = Some(bounds.min);
    config.length_max = Some(bounds.max);
    config.trust_region_shape = bounds.shape;
    Ok(())
}
