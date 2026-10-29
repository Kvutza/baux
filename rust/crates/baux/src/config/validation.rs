//! Experiment validation stays independent of configuration defaults and
//! lowering.

use super::*;

impl RunConfig {
    pub(super) fn validate_selection(&self) -> Result<bool, String> {
        if !matches!(
            self.experiment,
            Some(
                ExperimentKind::EndToEnd
                    | ExperimentKind::MoeLayer
                    | ExperimentKind::Pretrain
                    | ExperimentKind::Generation
            )
        ) {
            return Err(
                "experiment must be 'end-to-end', 'moe-layer', 'pretrain', or 'generation'".into(),
            );
        }
        if self.rounds() == 0 {
            return Err("rounds must be positive".into());
        }
        if self.reps() == 0 {
            return Err("reps must be positive".into());
        }
        if self.target_ms() == 0 {
            return Err("target_round_ms must be positive".into());
        }
        if self.output().as_os_str().is_empty() {
            return Err("output must be nonempty".into());
        }
        Ok(matches!(
            self.experiment,
            Some(ExperimentKind::Pretrain | ExperimentKind::Generation)
        ))
    }

    pub(super) fn validate_fields(&self, pretrain: bool) -> Result<(), String> {
        self.validate_objectives()?;
        self.resident_pool()?;
        self.validate_identity(pretrain)?;
        if self.validation_dataset.is_some() && !pretrain {
            return Err("data.validation requires a pretraining experiment".into());
        }
        if self.experiment == Some(ExperimentKind::Pretrain)
            && self.generation.is_some()
            && !matches!(
                self.optimizer(),
                TrainingOptimizer::MuonV1 | TrainingOptimizer::AdamwV1
            )
        {
            return Err(
                "Bayesian pretraining uses objective.window-tokens; generation configuration is \
                 not allowed"
                    .into(),
            );
        }
        self.validate_window()?;
        if self.validation_interval == Some(0)
            || (self.generation.is_none()
                && self.validation_interval.is_some()
                && self.validation_dataset.is_none()
                && self.dataset.is_some())
        {
            return Err(
                "validation_interval requires a held-out dataset and must be positive".into(),
            );
        }
        if (self.validation_interval.is_some() || self.selection.is_some()) && !pretrain {
            return Err(
                "validation intervals and selection ablations require pretrain experiments".into(),
            );
        }
        if self
            .generation
            .as_ref()
            .and_then(|config| config.signal_gate.as_ref())
            .is_some_and(|gate| gate.rounds > self.rounds())
        {
            return Err("signal gate must finish within the configured rounds".into());
        }
        self.validate_geometry(pretrain)
    }

    fn validate_window(&self) -> Result<(), String> {
        let Some(tokens) = self.window_tokens else {
            return Ok(());
        };
        if self.experiment != Some(ExperimentKind::Pretrain) {
            return Err("objective.window-tokens requires a pretrain experiment".into());
        }
        if !(128..=crate::context::MAX_TOKENS).contains(&tokens) || tokens % 128 != 0 {
            return Err(
                "objective.window-tokens must be a multiple of 128 from 128 through 1048576".into(),
            );
        }
        if tokens > crate::pretrain_data::CONTEXT
            && !matches!(
                self.model,
                Some(ModelArchitecture::FbtPisa1Mhc4V1 | ModelArchitecture::FbtPisa1HnetMhc4V1)
            )
        {
            return Err("pretraining sequences above 4096 tokens require fbt-pisa1-mhc4-v1".into());
        }
        if tokens > crate::pretrain_data::CONTEXT
            && !matches!(
                self.optimizer(),
                TrainingOptimizer::ResidentEnnV1 | TrainingOptimizer::OperatorBo
            )
        {
            return Err("long fixed-corpus likelihood requires turbo-enn or operator-bo".into());
        }
        if self.generation.is_some() {
            return Err("fixed causal pretraining cannot contain generation configuration".into());
        }
        if self.objective_reference.is_some() {
            return Err(
                "fixed causal pretraining has no reference evaluation; remove objective.reference"
                    .into(),
            );
        }
        Ok(())
    }

    fn validate_identity(&self, pretrain: bool) -> Result<(), String> {
        let diffusion = self
            .generation
            .as_ref()
            .is_some_and(|generation| generation.draft.is_some());
        let diffusion_model = self.model == Some(ModelArchitecture::FbtPisa1DiffusionMhc4V1);
        if diffusion != diffusion_model {
            return Err(
                "fbt-pisa1-diffusion-mhc4-v1 requires generation.draft; causal models cannot run \
                 diffusion"
                    .into(),
            );
        }
        if self.generation.as_ref().is_some_and(|generation| {
            generation.initialization == ModelInitialization::FbtPisa1Mhc4
        }) && !matches!(
            self.model,
            Some(ModelArchitecture::FbtPisa1Mhc4V1 | ModelArchitecture::FbtPisa1HnetMhc4V1)
        ) {
            return Err("initialization fbt-pisa1-mhc4 requires model fbt-pisa1-mhc4-v1".into());
        }
        if self.kernel_trial.is_some() && !pretrain {
            return Err("kernel trials require a pretrain experiment".into());
        }
        if !pretrain && (self.scorer_stage_samples.is_some() || self.pool_rank_audit.is_some()) {
            return Err("scorer timing and pool rank audits require a pretrain experiment".into());
        }
        if self.scorer_stage_samples == Some(0) {
            return Err("scorer_stage_samples must be positive when configured".into());
        }
        if self.pool_rank_audit == Some(0) {
            return Err("pool_rank_audit must be positive when configured".into());
        }
        if pretrain && self.model.is_none() {
            let models = ModelArchitecture::ALL.map(ModelArchitecture::id).join(", ");
            return Err(format!("pretrain model must be one of: {models}"));
        }
        if self.experiment == Some(ExperimentKind::Pretrain)
            && self
                .data_recipe
                .as_deref()
                .is_none_or(|recipe| !valid_recipe(recipe))
        {
            return Err("pretrain data.recipe must be a lowercase kebab-case identifier".into());
        }
        Ok(())
    }

    fn validate_geometry(&self, pretrain: bool) -> Result<(), String> {
        self.validate_program(pretrain)?;
        if !pretrain && self.perturbation.is_some() {
            return Err("perturbation is supported only by pretrain experiments".into());
        }
        if !pretrain && self.objective_reference.is_some() {
            return Err("objective reference is supported only by pretrain experiments".into());
        }
        if !pretrain
            && (self.distance_scaling.is_some()
                || self.history_geometry.is_some()
                || self.local_scale_neighbors.is_some())
        {
            return Err("distance scaling is supported only by pretrain experiments".into());
        }
        if !pretrain && self.reps() > 1 {
            return Err("reps greater than one are supported only by pretrain".into());
        }
        if self.experiment == Some(ExperimentKind::MoeLayer) {
            if self.model.is_some() || self.data_recipe.is_some() {
                return Err("model and corpus presets are supported only by pretrain".into());
            }
            return Ok(());
        }
        if !pretrain
            && (self.dataset.is_some() || self.model.is_some() || self.data_recipe.is_some())
        {
            return Err(
                "dataset, model, and corpus are supported only by pretrain experiments".into(),
            );
        }
        Ok(())
    }

    fn validate_program(&self, pretrain: bool) -> Result<(), String> {
        let lie_or_rotation = matches!(
            self.proposal_method,
            Some(
                crate::procedural_pool::ProposalMethod::PairMix
                    | crate::procedural_pool::ProposalMethod::QuadraticFlow
            )
        );
        if lie_or_rotation
            && (!pretrain || self.history_geometry.unwrap_or_default() != HistoryGeometry::Latent)
        {
            return Err(
                "Lie and pair-rotation require pretraining, Gaussian or Rademacher perturbations, \
                 and geometry='latent' (sampled parameter history)"
                    .into(),
            );
        }
        Ok(())
    }

    pub(super) fn validate_seeds(&self) -> Result<(), String> {
        if self.reps() > 1
            && [
                self.model_seed,
                self.reference_seed,
                self.proposal_seed,
                self.acquisition_seed,
                self.generation
                    .as_ref()
                    .and_then(|generation| generation.seed),
            ]
            .iter()
            .any(Option::is_some)
        {
            return Err("reps greater than one require internally derived seeds".into());
        }
        let fixed_seeds = [self.model_seed(), self.reference_seed()];
        if fixed_seeds.iter().any(|&seed| seed > i64::MAX as u64) {
            return Err("seeds must fit TOML signed 64-bit integers".into());
        }
        for rep in 0..self.reps() {
            let seeds = [self.proposal_seeded(rep), self.acquisition_seeded(rep)];
            if seeds.iter().any(|&seed| seed > i64::MAX as u64) {
                return Err("seeds must fit TOML signed 64-bit integers".into());
            }
            for seed in [self.proposal_seeded(rep), self.acquisition_seeded(rep)] {
                seed.checked_add(u64::from(self.rounds() - 1))
                    .ok_or("round seed overflow")?;
            }
        }
        Ok(())
    }

    pub(super) fn validate_resident(&self, pretrain: bool) -> Result<(), String> {
        if self.trust_region_kind == Some(TrustRegionKind::Morbo) {
            return Err("the LocalV1 FBT latency run requires the TuRBO trust region".into());
        }
        if !pretrain && self.trust_region_shape.is_some() {
            return Err("trust-region shape is supported only by pretrain experiments".into());
        }
        match (self.trust_region_kind, self.reliability_controller) {
            (Some(TrustRegionKind::Reliability), Some(config)) if pretrain => {
                config.validate()?;
            }
            (Some(TrustRegionKind::Reliability), _) if !pretrain => {
                return Err(
                    "the reliability controller is supported only by pretrain experiments".into(),
                );
            }
            (Some(TrustRegionKind::Reliability), None) => {
                return Err("reliability radius control requires its policy parameters".into());
            }
            (_, Some(_)) => {
                return Err(
                    "reliability policy parameters require reliability radius control".into(),
                );
            }
            (Some(TrustRegionKind::Turbo) | Some(TrustRegionKind::Morbo) | None, None) => {}
        }
        let effective = self.apply_to(turbo_enn());
        match effective.acquisition {
            AcquisitionConfig::Random | AcquisitionConfig::Pareto => {
                return Err(
                    "the resident round experiment supports only UCB or Thompson acquisition"
                        .into(),
                );
            }
            AcquisitionConfig::UCB { beta } if !beta.is_finite() || !(beta as f32).is_finite() => {
                return Err("turbo_enn UCB beta must be finite FP32".into());
            }
            AcquisitionConfig::UCB { .. } | AcquisitionConfig::Thompson => {}
        }
        self.resident_enn(self.acquisition_seed())?;
        let TrustRegionConfig::Turbo(length) = effective.trust_region else {
            return Err("the LocalV1 FBT latency run requires the TuRBO trust region".into());
        };
        validate_lengths(length)
    }

    pub(super) fn validate_controller(&self) -> Result<(), String> {
        let unsupported = self.candidate_rv.is_some()
            || self.num_candidates_factor.is_some()
            || self.min_candidates.is_some()
            || self.max_candidates.is_some()
            || self.num_candidates_per_arm.is_some()
            || self.num_pert.is_some()
            || self.index_driver.is_some()
            || self.scale_x.is_some()
            || self.noise_aware.is_some()
            || self.failure_tolerance_dim.is_some()
            || self.enn_storage.is_some()
            || self.work_dir.is_some()
            || self.y_bounds.is_some()
            || self.num_metrics.is_some()
            || self.alpha.is_some()
            || self.rescalarize.is_some();
        if unsupported {
            return Err(
                "the resident round experiment received optimizer fields that its fixed \
                 controller does not implement"
                    .into(),
            );
        }
        Ok(())
    }
}

#[cfg(test)]
mod program_tests {
    use super::*;

    #[test]
    fn pair_rotation() {
        let mut config: RunConfig = baux::artifact::toml::from_str(
            "proposal_method='pair-rotation'\nperturbation='gaussian'\nhistory_geometry='latent'",
        )
        .unwrap();
        assert_eq!(config.proposal_method.unwrap().name(), "pair-rotation");
        assert!(config.validate_program(true).is_ok());
        assert!(config.validate_program(false).is_err());
        config.history_geometry = Some(HistoryGeometry::Realized);
        assert!(config.validate_program(true).is_err());
        config.history_geometry = Some(HistoryGeometry::Latent);
        config.perturbation = Some(crate::ProposalKernel::Rademacher);
        assert!(config.validate_program(true).is_ok());
        assert!(
            baux::artifact::toml::from_str::<RunConfig>("proposal_method='lie-operator'").is_err()
        );
    }
}
