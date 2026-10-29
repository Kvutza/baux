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
            execution: ExecutionSpec {
                rounds: config.rounds(),
                repetitions: config.reps(),
                round_target_ms: config.target_ms(),
            },
            evaluation: EvaluationSpec {
                interval: config.validation_interval,
            },
            data: DataSpec {
                recipe: config.data_recipe.clone(),
                train: config.dataset.clone(),
                validation: config.validation_dataset.clone(),
            },
            optimizer: OptimizerSpec::from_config(config)?,
            objective: ObjectiveSpec {
                function: (matrix || fixed).then(|| config.objective_function()),
                sampling: if matrix {
                    Some(ObjectiveSampling::Natural)
                } else if fixed {
                    Some(ObjectiveSampling::Natural)
                } else {
                    None
                },
                window_tokens: config.window_tokens,
                reference: config.objective_reference,
            },
            diagnostics: DiagnosticSpec {
                trace: config.trace(),
                stage_samples: config.scorer_stage_samples,
                pool_rank: config.pool_rank_audit,
                kernels: config.kernel_trial.clone(),
            },
            generation: (!matrix).then(|| config.generation.clone()).flatten(),
        })
    }

    pub fn compile(&self) -> Result<RunConfig, String> {
        let optimizer = self.optimizer.training_optimizer();
        let generation = self.objective_generation(optimizer)?;
        let mut config = RunConfig {
            experiment: Some(self.experiment),
            model: self.model,
            data_recipe: self.data.recipe.clone(),
            output: self.output.clone(),
            optimizer: Some(optimizer),
            rounds: Some(self.execution.rounds),
            reps: Some(self.execution.repetitions),
            target_round_ms: Some(self.execution.round_target_ms),
            validation_interval: self.evaluation.interval,
            dataset: self.data.train.clone(),
            validation_dataset: self.data.validation.clone(),
            window_tokens: self.objective.window_tokens,
            objective_reference: self.objective.reference,
            objective_function: self.objective.function,
            trace: self.diagnostics.trace.then_some(true),
            scorer_stage_samples: self.diagnostics.stage_samples,
            pool_rank_audit: self.diagnostics.pool_rank,
            kernel_trial: self.diagnostics.kernels.clone(),
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
            if self.objective.function.is_some()
                || self.objective.sampling.is_some()
                || self.objective.window_tokens.is_some()
            {
                return Err("pretraining objective fields require experiment='pretrain'".into());
            }
            return Ok(self.generation.clone());
        }
        if matrix {
            if self.generation.is_some()
                || self.objective.function != Some(ObjectiveFunction::CrossEntropy)
                || self.objective.sampling != Some(ObjectiveSampling::Natural)
                || self.objective.window_tokens.is_some()
                || self.objective.reference.is_some()
            {
                return Err("Muon and AdamW pretraining require objective \
                            function='cross-entropy' and sampling='natural', with no generation \
                            or window fields"
                    .into());
            }
            return matrix_generation(self.model).map(Some);
        }
        if self.generation.is_none()
            && (!matches!(
                self.objective.function,
                Some(ObjectiveFunction::CrossEntropy | ObjectiveFunction::BoundedLogLikelihood)
            ) || self.objective.sampling != Some(ObjectiveSampling::Natural)
                || self.objective.window_tokens.is_none())
        {
            return Err(
                "Bayesian pretraining requires objective function='cross-entropy' or \
                 'bounded-log-likelihood', sampling='natural', and window-tokens"
                    .into(),
            );
        }
        if self.generation.is_some()
            && (self.objective.function.is_some()
                || self.objective.sampling.is_some()
                || self.objective.window_tokens.is_some())
        {
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
        if config.optimizer() == TrainingOptimizer::OperatorBo {
            return Ok(Self::OperatorBo {
                candidate_generation,
                surrogate: SurrogateRecipe {
                    estimator: SurrogateEstimator::NeighborValue,
                    ..surrogate
                },
                acceptance: AcceptanceSpec {
                    criterion: AcceptanceCriterion::PositiveImprovement,
                },
                radius_control: RadiusControlSpec::from_config(config),
            });
        }
        let acquisition = AcquisitionSpec::from_config(config);
        let bounds = TrustRegionBounds::from_config(config);
        let trust_region = if config
            .objective_acquisition
            .as_ref()
            .is_some_and(|objective| objective.mode == ResidentObjectiveMode::Morbo)
        {
            let objective = config.objective_acquisition.as_ref().unwrap();
            TrustRegionSpec::Morbo {
                bounds,
                regions: objective.regions.unwrap(),
                rescalarize: objective.rescalarize.unwrap(),
                clip: objective.clip.unwrap(),
            }
        } else {
            TrustRegionSpec::Turbo { bounds }
        };
        Ok(Self::TurboEnn {
            candidate_generation,
            surrogate,
            acquisition,
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
        let (candidate, surrogate, acquisition) = match self {
            Self::Muon | Self::Adamw => return Ok(()),
            Self::OperatorBo {
                candidate_generation,
                surrogate,
                acceptance,
                radius_control,
            } => {
                if candidate_generation.candidates != 1
                    || candidate_generation.arms != 1
                    || candidate_generation.perturbation.is_none()
                    || surrogate.estimator != SurrogateEstimator::NeighborValue
                    || acceptance.criterion != AcceptanceCriterion::PositiveImprovement
                {
                    return Err("operator-bo requires 1 candidate, 1 arm, perturbation, \
                                estimator='knn', and positive-improvement acceptance"
                        .into());
                }
                radius_control.apply(config)?;
                (candidate_generation, surrogate, None)
            }
            Self::TurboEnn {
                candidate_generation,
                surrogate,
                acquisition,
                trust_region,
            } => {
                if surrogate.estimator != SurrogateEstimator::Enn {
                    return Err("turbo-enn requires surrogate estimator='enn'".into());
                }
                apply_bounds(config, trust_region.bounds())?;
                match trust_region {
                    TrustRegionSpec::Turbo { .. } => {
                        if matches!(acquisition, AcquisitionSpec::AugmentedChebyshev { .. }) {
                            return Err("augmented-chebyshev acquisition requires trust-region \
                                        policy='morbo'"
                                .into());
                        }
                        config.trust_region_kind = Some(TrustRegionKind::Turbo);
                    }
                    TrustRegionSpec::Morbo {
                        regions,
                        rescalarize,
                        clip,
                        ..
                    } => {
                        if !matches!(acquisition, AcquisitionSpec::AugmentedChebyshev { .. }) {
                            return Err("trust-region policy='morbo' requires \
                                        augmented-chebyshev acquisition"
                                .into());
                        }
                        config.trust_region_kind = Some(TrustRegionKind::Turbo);
                        acquisition.apply(config);
                        let objective = config.objective_acquisition.as_mut().unwrap();
                        objective.regions = Some(*regions);
                        objective.rescalarize = Some(*rescalarize);
                        objective.clip = Some(*clip);
                    }
                }
                (candidate_generation, surrogate, Some(acquisition))
            }
        };
        candidate.apply(config)?;
        surrogate.apply(config);
        if let Some(acquisition) = acquisition {
            acquisition.apply(config);
        }
        Ok(())
    }
}

impl CandidateGenerationSpec {
    fn from_config(config: &RunConfig) -> Result<Self, String> {
        let pool = config.resident_pool()?;
        Ok(Self {
            generator: config.proposal_method.unwrap_or_default(),
            candidates: pool.count(),
            arms: pool.arms(),
            perturbation: config
                .perturbation
                .map(|distribution| ProposalKernelSpec { distribution }),
        })
    }

    fn apply(&self, config: &mut RunConfig) -> Result<(), String> {
        if self.arms == 0 || self.candidates == 0 || !self.candidates.is_multiple_of(self.arms) {
            return Err(
                "optimizer.candidate-generation.candidates must be positive and divisible by arms"
                    .into(),
            );
        }
        config.proposal_pool = Some(crate::procedural_pool::ProceduralPoolConfig {
            arms: self.arms,
            candidates_per_arm: self.candidates / self.arms,
        });
        config.proposal_method = Some(self.generator);
        config.perturbation = self
            .perturbation
            .as_ref()
            .map(|perturbation| perturbation.distribution);
        Ok(())
    }
}

impl SurrogateRecipe {
    fn from_config(config: &RunConfig) -> Result<Self, String> {
        let resident = config.resident_enn(config.acquisition_seed())?;
        let estimator = SurrogateEstimator::Enn;
        Ok(Self {
            estimator,
            neighbors: config.k_neighbors.unwrap_or_else(default_neighbors),
            neighbor_rule: if resident.adapt_neighbors {
                NeighborRule::RiskPrefix
            } else {
                NeighborRule::Fixed
            },
            epistemic_scale: config.epistemic_scale.unwrap_or_else(epistemic_default),
            aleatoric_scale: config.aleatoric_scale.unwrap_or_else(aleatoric_default),
            y_scale: config.y_scale.unwrap_or_else(default_scale),
            geometry: resident.history_geometry,
            scaling: resident.distance_scaling,
            local_neighbors: (resident.distance_scaling != DistanceScaling::Global)
                .then_some(resident.local_scale_neighbors),
            fit: if resident.fit_params {
                SurrogateFit::LeaveOneOut {
                    fit_candidates: resident.num_candidates,
                    fit_samples: resident.num_samples,
                }
            } else {
                SurrogateFit::Fixed
            },
        })
    }

    fn apply(&self, config: &mut RunConfig) {
        config.k_neighbors = Some(self.neighbors);
        config.local_scale_neighbors = self.local_neighbors;
        config.epistemic_scale = Some(self.epistemic_scale);
        config.aleatoric_scale = Some(self.aleatoric_scale);
        config.y_scale = Some(self.y_scale);
        config.adapt_neighbors = Some(self.neighbor_rule == NeighborRule::RiskPrefix);
        config.history_geometry =
            (self.geometry != HistoryGeometry::default()).then_some(self.geometry);
        config.distance_scaling =
            (self.scaling != DistanceScaling::default()).then_some(self.scaling);
        match &self.fit {
            SurrogateFit::Fixed => {
                config.fit_params = Some(false);
            }
            SurrogateFit::LeaveOneOut {
                fit_candidates,
                fit_samples,
            } => {
                config.fit_params = Some(true);
                config.num_candidates = Some(*fit_candidates);
                config.num_samples = Some(*fit_samples);
            }
        }
    }
}

impl AcquisitionSpec {
    fn from_config(config: &RunConfig) -> Self {
        if config.selection == Some(PretrainSelection::Random) {
            return Self::Random;
        }
        match config.objective_acquisition.as_ref() {
            Some(vector) if vector.mode == ResidentObjectiveMode::Pareto => Self::Pareto {
                scales: vector.scales.clone(),
            },
            Some(vector) => Self::AugmentedChebyshev {
                scales: vector.scales.clone(),
                preferences: vector.preferences.clone().unwrap(),
                alpha: vector.alpha.unwrap(),
                seed_domain: vector.seed_domain.clone().unwrap(),
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
            Self::Random => AcquisitionConfig::default(),
            Self::Pareto { .. } | Self::AugmentedChebyshev { .. } => AcquisitionConfig::default(),
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
            Self::AugmentedChebyshev {
                scales,
                preferences,
                alpha,
                seed_domain,
            } => Some(ResidentObjectiveConfig {
                mode: ResidentObjectiveMode::Morbo,
                scales: scales.clone(),
                preferences: Some(preferences.clone()),
                regions: None,
                alpha: Some(*alpha),
                seed_domain: Some(seed_domain.clone()),
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

impl RadiusControlSpec {
    fn from_config(config: &RunConfig) -> Self {
        let bounds = TrustRegionBounds::from_config(config);
        Self::Turbo { bounds }
    }

    fn apply(&self, config: &mut RunConfig) -> Result<(), String> {
        apply_bounds(config, self.bounds())?;
        let Self::Turbo { .. } = self;
        config.trust_region_kind = Some(TrustRegionKind::Turbo);
        Ok(())
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
