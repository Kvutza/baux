use super::generation::*;
pub(super) struct LoopState {
    pub(super) search: SearchState,
    pub(super) updates: UpdateLog,
    pub(super) journal: journal::Journal,
    pub(super) records: BufWriter<File>,
    pub(super) walls: Vec<f64>,
    pub(super) minimum_changed_weights: u64,
    pub(super) minimum_generated_tokens: usize,
    pub(super) minimum_context_tokens: usize,
    pub(super) accepted: u32,
    pub(super) incumbent_version: u64,
    pub(super) incumbent_rollouts: Vec<decode::Rollout>,
    pub(super) regions: RegionState,
    pub(super) temperature: f32,
    pub(super) loop_seconds: f64,
    pub(super) rank_audit: Option<RankAudit>,
    pub(super) pool_rank_audited: u32,
    pub(super) pool_rank_report: Option<baux::artifact::json::Value>,
    pub(super) causal_value: Option<f32>,
    pub(super) condition_baselines: Vec<Option<ConditionBaseline>>,
}
pub(super) struct RegionState {
    pub(super) versions: Vec<u64>,
    pub(super) rollouts: Vec<Vec<decode::Rollout>>,
    pub(super) temperatures: Vec<f32>,
}
pub(super) struct RoundCondition {
    pub(super) config: GenerationConfig,
    pub(super) condition_seed: u64,
    pub(super) evaluation_seed: u64,
    pub(super) corpus_token_offset: usize,
}
#[derive(Clone)]
pub(super) struct ConditionBaseline {
    pub(super) condition_seed: u64,
    pub(super) reward: f32,
    pub(super) variance: f32,
    pub(super) rollouts: Vec<decode::Rollout>,
}
impl ConditionBaseline {
    fn measured(condition_seed: u64, evaluation: &Evaluation) -> Self {
        Self {
            condition_seed,
            reward: evaluation.mean,
            variance: evaluation.variance,
            rollouts: evaluation.rollouts.clone(),
        }
    }
    fn accepted(condition_seed: u64, evaluation: &Evaluation) -> Result<Self, String> {
        let paired = evaluation
            .paired
            .ok_or("accepted paired evaluation lost its measured candidate response")?;
        Ok(Self {
            condition_seed,
            reward: paired.candidate_reward,
            variance: paired.candidate_variance,
            rollouts: evaluation.rollouts.clone(),
        })
    }
    fn gpu_seconds(&self) -> f64 {
        self.rollouts
            .iter()
            .map(|rollout| rollout.gpu_seconds)
            .sum()
    }
}
impl RegionState {
    pub(super) fn initial(
        search: &SearchState,
        model_seed: u64,
        rollouts: &[decode::Rollout],
        temperature: f32,
    ) -> Self {
        let count = search.morbo_regions().max(1);
        let version = crate::hash::splitmix64(model_seed ^ 0x656e_6e78_2d62_6173);
        Self {
            versions: vec![version; count],
            rollouts: vec![rollouts.to_vec(); count],
            temperatures: vec![temperature; count],
        }
    }
}
impl LoopState {
    pub(super) fn commit(
        &mut self,
        evaluation: &Evaluation,
        accepted: bool,
        region: usize,
        version: u64,
        temperature: f32,
    ) {
        let generated = evaluation
            .rollouts
            .iter()
            .map(|rollout| rollout.tokens.len())
            .min()
            .unwrap_or(0);
        self.minimum_generated_tokens = self.minimum_generated_tokens.min(generated);
        self.minimum_context_tokens = self
            .minimum_context_tokens
            .min(evaluation.prompt_tokens + generated);
        if accepted {
            self.accept_region(region, version, &evaluation.rollouts, temperature);
            if let Some(causal) = evaluation.causal {
                self.causal_value = Some(causal.optimizer_value);
            }
        }
        self.temperature = self.regions.temperatures[region];
        self.accepted += u32::from(accepted);
    }
    fn accept_region(
        &mut self,
        region: usize,
        version: u64,
        rollouts: &[decode::Rollout],
        temperature: f32,
    ) {
        self.incumbent_version = version;
        self.incumbent_rollouts = rollouts.to_vec();
        self.regions.versions[region] = version;
        self.regions.rollouts[region] = rollouts.to_vec();
        self.regions.temperatures[region] = temperature;
    }
    pub(super) fn finalize_state(
        &mut self,
        record: bool,
        output: &Path,
        weights: &CandidateWeights,
    ) -> Result<metal::Buffer, String> {
        if let Some(region) = self.search.finalize_morbo()? {
            self.incumbent_version = self.regions.versions[region];
            self.incumbent_rollouts = self.regions.rollouts[region].clone();
            self.temperature = self.regions.temperatures[region];
        }
        let final_weights = self.search.base_buffer();
        if record {
            weights.finalize_updates(&final_weights, &mut self.updates)?;
            self.updates.write(&output.join("tensor-updates.bin"))?;
            weights.write_stats(&final_weights, &output.join("tensor-state.jsonl"))?;
        }
        self.records.flush().map_err(|error| error.to_string())?;
        self.journal.finish()?;
        Ok(final_weights)
    }

    pub(super) fn record_rank(
        &mut self,
        evaluation: &Evaluation,
        record: &mut baux::artifact::json::Value,
    ) {
        if let (Some(audit), Some(nll)) = (
            self.rank_audit.as_mut(),
            evaluation
                .rollouts
                .first()
                .and_then(|rollout| rollout.conditional_target_nll),
        ) {
            record["model_selection_audit"] = audit.observe(-nll, evaluation.mean);
        }
    }

    pub(super) fn record_updates(
        &mut self,
        enabled: bool,
        step: u32,
        proposal: &crate::resident_metal::Proposals,
        accepted: bool,
        changes: &[(u64, f64)],
    ) -> Result<(), String> {
        if enabled {
            self.updates.push_scaled(
                step,
                proposal.seed,
                proposal.length,
                accepted,
                changes,
                &proposal.block_scales,
            )?;
        }
        Ok(())
    }

    pub(super) fn write_record(
        &mut self,
        record: &baux::artifact::json::Value,
        flush: bool,
    ) -> Result<(), String> {
        baux::artifact::json::write_line(&mut self.records, record)
            .map_err(|error| error.to_string())?;
        if flush {
            self.records.flush().map_err(|error| error.to_string())?;
        }
        Ok(())
    }
}

pub(super) struct LoopContext<'a> {
    pub(super) run: &'a RunConfig,
    pub(super) config: &'a GenerationConfig,
    pub(super) output: &'a Path,
    pub(super) runtime: &'a Runtime,
    pub(super) decoder: &'a decode::Decoder,
    pub(super) verifier: &'a Option<block_decode::BlockDecoder>,
    pub(super) weights: &'a CandidateWeights,
    pub(super) sampling_seed: u64,
    pub(super) tokenizer: Option<&'a ByteDecoder>,
    pub(super) evaluator: &'a mut Option<Evaluator>,
    pub(super) validation_tasks: &'a [GenerationTask],
    pub(super) dataset: Option<&'a crate::pretrain_data::PretrainDataset>,
    pub(super) corpus_prompt_cap: usize,
    pub(super) corpus_output_cap: usize,
    pub(super) policy: crate::config::ResidentEnnConfig,
    pub(super) proposal_seed: u64,
    pub(super) acquisition_seed: u64,
    pub(super) causal: &'a mut Option<CausalScorer>,
    pub(super) causal_log: &'a mut Option<CausalLog>,
}

pub(super) fn attach_pair(
    candidate: &mut Evaluation,
    incumbent: &ConditionBaseline,
    baseline: f32,
    baseline_variance: f32,
    condition_seed: u64,
    sampling_seed: u64,
    corpus_token_offset: usize,
    incumbent_gpu_seconds: f64,
    incumbent_reused: bool,
) -> Result<(), String> {
    if candidate.observation.is_vector() {
        return Err(
            "paired corpus evaluation currently requires one scalar control objective".into(),
        );
    }
    let candidate_reward = candidate.mean;
    let candidate_variance = candidate.variance;
    let incumbent_reward = incumbent.reward;
    let incumbent_variance = incumbent.variance;
    let paired = candidate
        .rollouts
        .iter()
        .map(|rollout| {
            rollout.target_quality.and_then(|quality| {
                Some((
                    quality.paired_improvement?,
                    quality.paired_improvement_variance?,
                ))
            })
        })
        .collect::<Option<Vec<_>>>();
    let response_components = candidate
        .rollouts
        .iter()
        .map(|rollout| rollout.paired_patch_improvements.len())
        .sum();
    let (improvement, improvement_variance, difference_statistic) =
        if let Some(paired) = paired.filter(|values| !values.is_empty()) {
            let count = paired.len() as f32;
            (
                paired.iter().map(|value| value.0).sum::<f32>() / count,
                paired.iter().map(|value| value.1).sum::<f32>() / (count * count),
                if response_components > 0 {
                    "causal-patch-vector"
                } else {
                    "causal-block-means"
                },
            )
        } else {
            (
                candidate_reward - incumbent_reward,
                candidate_variance + incumbent_variance,
                "independent-marginals",
            )
        };
    let optimizer_value = baseline + improvement;
    let optimizer_variance = baseline_variance + improvement_variance;
    if !improvement.is_finite()
        || !improvement_variance.is_finite()
        || improvement_variance < 0.0
        || !optimizer_value.is_finite()
        || !optimizer_variance.is_finite()
        || optimizer_variance < 0.0
    {
        return Err("paired generation produced a non-finite response".into());
    }
    candidate.mean = optimizer_value;
    candidate.variance = optimizer_variance;
    candidate.observation = crate::objective_observation::ObjectiveObservation::scalar(
        optimizer_value,
        optimizer_variance,
    )?;
    candidate.paired = Some(PairedGeneration {
        condition_seed,
        sampling_seed,
        corpus_token_offset,
        candidate_reward,
        candidate_variance,
        incumbent_reward,
        incumbent_variance,
        improvement,
        improvement_variance,
        response_components,
        difference_statistic,
        optimizer_value,
        optimizer_variance,
        incumbent_generated_tokens: incumbent
            .rollouts
            .iter()
            .map(|rollout| rollout.tokens.len())
            .sum(),
        incumbent_evaluated_positions: incumbent
            .rollouts
            .iter()
            .map(|rollout| rollout.evaluated_positions)
            .sum(),
        incumbent_committed_tokens: incumbent
            .rollouts
            .iter()
            .map(|rollout| rollout.committed_tokens)
            .sum(),
        incumbent_gpu_seconds,
        incumbent_reused,
        incumbent_expert_mask: decode::route_union(&incumbent.rollouts),
        candidate_expert_mask: decode::route_union(&candidate.rollouts),
    });
    Ok(())
}

impl LoopContext<'_> {
    fn alignment(
        &mut self,
        probe: &mut gen_alignment::Probe,
        step: u32,
        temperature: f32,
        candidate: &BufferRef,
        candidate_nll: f32,
        path: &Path,
    ) -> Result<gen_alignment::Alignment, String> {
        let condition = self.round_condition(step, temperature)?;
        let verifier = self
            .verifier
            .as_ref()
            .ok_or("optimizer alignment requires accepted-prefix verification")?;
        let incumbent = score(
            self.runtime,
            self.decoder,
            Some((verifier, None)),
            self.weights.row(probe.incumbent())?,
            &condition.config,
            condition.evaluation_seed,
            &path.join("alignment-incumbent"),
            self.evaluator.as_mut(),
            self.tokenizer,
            false,
        )?;
        let incumbent_nll = conditional_nll(&incumbent)?;
        let mut alignment = probe.update(
            self.runtime,
            verifier,
            &condition.config.tasks[0],
            condition.config.max_tokens,
            candidate,
            incumbent_nll,
            candidate_nll,
        )?;
        let muon = score(
            self.runtime,
            self.decoder,
            Some((verifier, None)),
            self.weights.row(probe.muon())?,
            &condition.config,
            condition.evaluation_seed,
            &path.join("alignment-muon"),
            self.evaluator.as_mut(),
            self.tokenizer,
            false,
        )?;
        let muon_nll = conditional_nll(&muon)?;
        alignment.muon_nll = Some(muon_nll);
        alignment.muon_nll_change = Some(muon_nll - incumbent_nll);
        Ok(alignment)
    }

    fn round_condition(&self, step: u32, temperature: f32) -> Result<RoundCondition, String> {
        let mut config = self.config.clone();
        config.temperature = temperature;
        // Acquisition ranks the complete candidate pool, then evaluates only
        // its selected member. Each completed evaluation therefore advances
        // the corpus condition exactly once.
        let condition = step;
        let condition_seed = crate::hash::splitmix64(
            self.sampling_seed ^ u64::from(condition + 1) ^ 0x636f_7270_7573_2d78,
        );
        let sampling_seed = crate::hash::splitmix64(condition_seed ^ 0x7361_6d70_6c65_2d78);
        let corpus_token_offset = if let Some(dataset) = self.dataset {
            let (task, offset) = setup::corpus_window(
                dataset,
                &config.tasks[0],
                self.corpus_prompt_cap,
                self.corpus_output_cap,
                self.sampling_seed,
                condition,
            )?;
            config.max_tokens = u32::try_from(task.expected.len())
                .map_err(|_| "sampled natural target exceeds u32 token capacity")?;
            config.tasks[0] = task;
            offset
        } else {
            0
        };
        Ok(RoundCondition {
            config,
            condition_seed,
            evaluation_seed: if self.dataset.is_some() {
                sampling_seed
            } else {
                self.sampling_seed
            },
            corpus_token_offset,
        })
    }

    fn preview(
        &self,
        retain: bool,
        evaluation: &Evaluation,
        path: &Path,
    ) -> Result<Option<String>, String> {
        if !retain {
            return Ok(None);
        }
        decoded_completion(self.tokenizer, &evaluation.rollouts[0], path, false)
    }

    pub(super) fn retain_round(
        &self,
        step: u32,
        path: &Path,
        evaluation: &mut Evaluation,
    ) -> Result<Option<String>, String> {
        if !self.checkpoint_step(step) {
            return Ok(None);
        }
        if transient(&self.config.reward) {
            retain(&path.join("candidate"), evaluation)?;
        }
        self.preview(true, evaluation, path)
    }

    pub(super) fn checkpoint_step(&self, step: u32) -> bool {
        step == 0
            || step + 1 == self.run.rounds()
            || learning_checkpoint(step + 1)
            || self
                .run
                .validation_interval
                .is_some_and(|interval| (step + 1) % interval == 0)
    }
    pub(super) fn annotate_evaluation(
        evaluation: &Evaluation,
        record: &mut baux::artifact::json::Value,
    ) -> Result<(), String> {
        if let Some(causal) = evaluation.causal {
            record["causal_pretraining"] =
                baux::artifact::json::to_value(causal).map_err(|error| error.to_string())?;
        }
        if let Some(paired) = evaluation.paired {
            record["paired_generation"] =
                baux::artifact::json::to_value(paired).map_err(|error| error.to_string())?;
        }
        evaluation.annotate(record)?;
        Ok(())
    }

    fn validate_causal(&mut self, state: &LoopState, round: u32) -> Result<(), String> {
        if let (Some(scorer), Some(log)) = (self.causal.as_ref(), self.causal_log.as_mut()) {
            let base = state.search.base_buffer();
            log.observe(scorer, self.runtime, self.weights.row(&base)?, round)?;
        }
        Ok(())
    }

    fn observe_incumbent(
        &mut self,
        state: &LoopState,
        config: &GenerationConfig,
        buffer: &metal::Buffer,
        path: &Path,
        seed: u64,
        step: u32,
    ) -> Result<Option<Evaluation>, String> {
        if self.causal.is_some() || matches!(config.reward, GenerationReward::CausalObjectives) {
            return Ok(None);
        }
        let region = state.search.morbo_region().unwrap_or(0);
        let mut config = config.clone();
        config.temperature = state.regions.temperatures[region];
        let persist = !transient(&config.reward) || self.checkpoint_step(step);
        Ok(Some(score(
            self.runtime,
            self.decoder,
            self.verifier.as_ref().map(|verifier| (verifier, None)),
            self.weights.row(buffer)?,
            &config,
            seed,
            &path.join("incumbent"),
            self.evaluator.as_mut(),
            self.tokenizer,
            persist,
        )?))
    }

    fn condition_baseline(
        &mut self,
        state: &mut LoopState,
        region: usize,
        condition: &RoundCondition,
        incumbent: &metal::Buffer,
        path: &Path,
        step: u32,
    ) -> Result<Option<(ConditionBaseline, f64, bool)>, String> {
        if self.causal.is_some()
            || matches!(condition.config.reward, GenerationReward::CausalObjectives)
        {
            return Ok(None);
        }
        if let Some(baseline) = state.condition_baselines[region]
            .as_ref()
            .filter(|baseline| baseline.condition_seed == condition.condition_seed)
        {
            return Ok(Some((baseline.clone(), 0.0, true)));
        }
        if let Some(verifier) = self.verifier {
            verifier.clear_pair();
        }
        let measured = self
            .observe_incumbent(
                state,
                &condition.config,
                incumbent,
                path,
                condition.evaluation_seed,
                step,
            )?
            .ok_or("pretraining did not produce an incumbent observation")?;
        if let Some(verifier) = self.verifier {
            let _ = verifier.capture_pair(self.runtime, condition.config.max_tokens as usize)?;
        }
        let baseline = ConditionBaseline::measured(condition.condition_seed, &measured);
        let gpu_seconds = baseline.gpu_seconds();
        state.condition_baselines[region] = Some(baseline.clone());
        Ok(Some((baseline, gpu_seconds, false)))
    }

    fn evaluate(
        &mut self,
        state: &mut LoopState,
        row: &metal::Buffer,
        proposal: &crate::resident_metal::Proposals,
        path: &Path,
        initial: bool,
        step: u32,
    ) -> Result<
        (
            Evaluation,
            f32,
            crate::resident_metal::NoisyDecision,
            Duration,
        ),
        String,
    > {
        let region = state.search.morbo_region().unwrap_or(0);
        let temperature = proposal
            .axis_value()
            .unwrap_or(state.regions.temperatures[region]);
        let condition = self.round_condition(step, temperature)?;
        let incumbent_buffer = state.search.base_buffer();
        let incumbent =
            self.condition_baseline(state, region, &condition, &incumbent_buffer, path, step)?;
        let candidate_path = path.join("candidate");
        let oracle_path = if incumbent.is_some() {
            candidate_path.as_path()
        } else {
            path
        };
        let evaluation = if self
            .run
            .pool_rank_audit
            .is_some_and(|points| state.pool_rank_audited < points)
            && !initial
        {
            self.audit_pool(
                state,
                proposal,
                &condition,
                incumbent.as_ref(),
                region,
                path,
                step,
            )?
        } else {
            self.score_candidate(
                state,
                row,
                &condition,
                incumbent.as_ref(),
                region,
                oracle_path,
                step,
            )?
        };
        let observation = evaluation.observation;
        let tell_start = Instant::now();
        let decision = if observation.is_vector() {
            state.search.modeled_objectives(proposal, observation)?
        } else if evaluation.paired.is_some() {
            super::gen_pool::tell_paired(&mut state.search, proposal, &evaluation)?
        } else if initial {
            state.search.initial_objectives(proposal, observation)?
        } else if let Some(causal) = evaluation.causal {
            state.search.paired_modeled(
                proposal,
                causal.optimizer_value,
                causal.variance,
                state.search.best()?,
                state.search.best_variance()?,
                causal.improvement,
                causal.variance,
            )?
        } else {
            state.search.modeled_objectives(proposal, observation)?
        };
        if state.search.sync()? != vec![decision.accepted] {
            return Err("resident search acceptance mismatch".into());
        }
        if decision.accepted && evaluation.paired.is_some() {
            if let Some(verifier) = self.verifier {
                let _ =
                    verifier.capture_pair(self.runtime, condition.config.max_tokens as usize)?;
            }
            state.condition_baselines[region] = Some(ConditionBaseline::accepted(
                condition.condition_seed,
                &evaluation,
            )?);
        }
        let tell_time = tell_start.elapsed();
        Ok((evaluation, temperature, decision, tell_time))
    }

    fn propose_step(
        &self,
        state: &mut LoopState,
        step: u32,
    ) -> Result<(metal::Buffer, crate::resident_metal::Proposals, bool), String> {
        let mut ask = self.policy.ask;
        ask.acquisition.seed = self.acquisition_seed + u64::from(step);
        let pool_count = state.search.pool_layout().count() as usize;
        let forced = (self.run.selection == Some(crate::config::PretrainSelection::Random))
            .then(|| (crate::hash::splitmix64(ask.acquisition.seed) as usize) % pool_count);
        state.search.propose(
            self.proposal_seed + u64::from(step),
            ask,
            step as usize % pool_count,
            forced,
        )
    }

    fn validate_checkpoint(&mut self, state: &mut LoopState, round: u32) -> Result<(), String> {
        let scheduled = learning_checkpoint(round)
            || self
                .run
                .validation_interval
                .is_some_and(|interval| round % interval == 0);
        if !scheduled || round == self.run.rounds() {
            return Ok(());
        }
        let started = Instant::now();
        let mut config = self.config.clone();
        config.temperature = state.temperature;
        let rewards = validate_generation(
            self.runtime,
            self.decoder,
            self.verifier.as_ref(),
            self.weights,
            &state.search.base_buffer(),
            &config,
            self.sampling_seed,
            &self.output.join(format!("validation-{round:06}")),
            self.evaluator.as_mut(),
            self.tokenizer,
            self.validation_tasks,
        )?;
        state
            .journal
            .validation(round, &rewards, started.elapsed().as_secs_f64())?;
        self.validate_causal(state, round)
    }

    fn maybe_align(
        &mut self,
        diagnose: bool,
        probe: Option<&mut gen_alignment::Probe>,
        step: u32,
        candidate_temperature: f32,
        row: &BufferRef,
        evaluation: &Evaluation,
        round_path: &Path,
    ) -> Result<(Option<gen_alignment::Alignment>, f64), String> {
        if !diagnose {
            return Ok((None, 0.0));
        }
        let Some(probe) = probe else {
            return Ok((None, 0.0));
        };
        let diagnostic_start = Instant::now();
        let mut alignment = self.alignment(
            probe,
            step,
            candidate_temperature,
            row,
            conditional_nll(evaluation)?,
            round_path,
        )?;
        let seconds = diagnostic_start.elapsed().as_secs_f64();
        alignment.diagnostic_ms = seconds * 1000.0;
        Ok((Some(alignment), seconds))
    }

    pub(super) fn run(&mut self, state: &mut LoopState) -> Result<(), String> {
        let mut alignment = self
            .config
            .record_tensor_updates
            .then(|| {
                let rows = self
                    .verifier
                    .as_ref()
                    .ok_or("optimizer alignment requires accepted-prefix verification")?
                    .context;
                gen_alignment::Probe::new(self.runtime, self.weights, rows)
            })
            .transpose()?;
        let rounds = if std::env::var_os("BAUX_UPDATE_ALIGNMENT_SINGLE").is_some() {
            1
        } else {
            self.run.rounds()
        };
        for step in 0..rounds {
            let start = Instant::now();
            let (row, proposal, initializing) = self.propose_step(state, step)?;
            let region = state.search.morbo_region().unwrap_or(0);
            let tensor_version = state
                .search
                .tensor_version(state.regions.versions[region], &proposal)?;
            let proposal_seconds = start.elapsed().as_secs_f64();
            let changes = state.search.describe(&proposal)?.remove(0).3;
            let round_path = self.output.join(format!("round-{:04}", step + 1));
            let diagnose = self.checkpoint_step(step);
            if diagnose && let Some(probe) = alignment.as_ref() {
                probe.preserve(self.runtime, &state.search.base_buffer())?;
            }
            let evaluation_start = Instant::now();
            let (mut evaluation, candidate_temperature, decision, tell_time) =
                self.evaluate(state, &row, &proposal, &round_path, initializing, step)?;
            let evaluation_seconds =
                evaluation_start.elapsed().as_secs_f64() - tell_time.as_secs_f64();
            let (round_align, diagnostic_seconds) = self.maybe_align(
                diagnose,
                alignment.as_mut(),
                step,
                candidate_temperature,
                &row,
                &evaluation,
                &round_path,
            )?;
            let changed_weights = changes.iter().map(|change| change.0).sum::<u64>();
            self.finish_step(
                state,
                step,
                initializing,
                region,
                &proposal,
                decision,
                &mut evaluation,
                &tensor_version,
                round_align,
                &changes,
                &round_path,
                RoundMetrics {
                    candidate_temperature,
                    proposal_seconds,
                    evaluation_seconds,
                    tell_seconds: tell_time.as_secs_f64(),
                    wall_seconds: start.elapsed().as_secs_f64() - diagnostic_seconds,
                    changed_weights,
                    changed_fraction: changed_weights as f64
                        / self.weights.architecture.parameter_count() as f64,
                },
            )?;
            self.validate_checkpoint(state, step + 1)?;
        }
        Ok(())
    }
}

fn conditional_nll(evaluation: &Evaluation) -> Result<f32, String> {
    evaluation
        .rollouts
        .first()
        .and_then(|rollout| rollout.conditional_target_nll)
        .ok_or_else(|| "optimizer comparison requires conditional target NLL".into())
}

pub(super) fn transient(reward: &GenerationReward) -> bool {
    matches!(
        reward,
        GenerationReward::ConditionalCrossEntropy
            | GenerationReward::CausalObjectives
            | GenerationReward::BoundedCrossEntropy
            | GenerationReward::SequenceKernel
    )
}
