use super::generation::*;

pub(super) struct RoundMetrics {
    pub(super) candidate_temperature: f32,
    pub(super) proposal_seconds: f64,
    pub(super) evaluation_seconds: f64,
    pub(super) tell_seconds: f64,
    pub(super) wall_seconds: f64,
    pub(super) changed_weights: u64,
    pub(super) changed_fraction: f64,
}

fn token_sum(
    rollouts: &[decode::Rollout],
    field: fn(&decode::Rollout) -> usize,
) -> Result<u64, String> {
    let total = rollouts
        .iter()
        .try_fold(0usize, |total, rollout| total.checked_add(field(rollout)))
        .ok_or("Generation token count overflow")?;
    u64::try_from(total).map_err(|_| "Generation token count exceeds u64".to_string())
}

fn token_accounting(
    context: &LoopContext<'_>,
    evaluation: &Evaluation,
) -> Result<crate::response_observation::TokenAccounting, String> {
    let prompt =
        u64::try_from(evaluation.prompt_tokens).map_err(|_| "Prompt token count exceeds u64")?;
    let drafted = evaluation
        .rollouts
        .iter()
        .filter_map(|rollout| rollout.draft.as_ref())
        .try_fold(0usize, |total, draft| {
            total.checked_add(draft.generated_tokens)
        })
        .ok_or("Drafted token count overflow")?;
    let drafted = u64::try_from(drafted).map_err(|_| "Drafted token count exceeds u64")?;
    let generated = token_sum(&evaluation.rollouts, |rollout| rollout.tokens.len())?;
    let committed = token_sum(&evaluation.rollouts, |rollout| rollout.committed_tokens)?;
    let _ = context;
    let evaluated = token_sum(&evaluation.rollouts, |rollout| rollout.evaluated_positions)?;
    let repaired = token_sum(&evaluation.rollouts, |rollout| rollout.repaired_tokens)?;
    let output = evaluation
        .rollouts
        .iter()
        .map(|rollout| rollout.token_count)
        .max()
        .ok_or("Generation response has no rollout")?;
    let output = u64::try_from(output).map_err(|_| "Context token count exceeds u64")?;
    let context_tokens = prompt
        .checked_add(output)
        .ok_or("Context token count overflow")?;
    crate::response_observation::TokenAccounting {
        samples: evaluation.rollouts.len() as u64,
        context: context_tokens,
        prompt,
        drafted,
        evaluated,
        repaired,
        generated,
        committed,
    }
    .validate()
}

fn response_observation(
    context: &LoopContext<'_>,
    proposal: &crate::resident_metal::Proposals,
    evaluation: &Evaluation,
    decision: crate::resident_metal::NoisyDecision,
    initializing: bool,
) -> Result<crate::response_observation::ResponseObservation, String> {
    use crate::objective_observation::ObjectiveEstimate;
    use crate::response_observation::{ResponseObservation, ResponseReference};

    let reference = if evaluation.paired.is_some() {
        ResponseReference::PairedIncumbent
    } else if initializing {
        ResponseReference::Initial
    } else if context.run.objective_acquisition.is_some() {
        ResponseReference::VectorPolicy
    } else if evaluation.causal.is_some() {
        ResponseReference::PairedIncumbent
    } else {
        ResponseReference::SurrogateIncumbent
    };
    let improvement_variance = evaluation.paired.map_or_else(
        || {
            evaluation.causal.map_or_else(
                || evaluation.variance + decision.incumbent_variance,
                |causal| causal.variance,
            )
        },
        |paired| paired.improvement_variance,
    );
    let coordinates = u64::try_from(context.weights.architecture.parameter_count())
        .map_err(|_| "Model parameter count exceeds u64")?;
    ResponseObservation {
        proposal: proposal.measure(coordinates)?,
        reference,
        candidate: ObjectiveEstimate {
            mean: evaluation.mean,
            variance: evaluation.variance,
        },
        baseline: ObjectiveEstimate {
            mean: decision.incumbent_value,
            variance: decision.incumbent_variance,
        },
        improvement: ObjectiveEstimate {
            mean: decision.improvement as f32,
            variance: improvement_variance,
        },
        accepted: decision.accepted,
        tokens: Some(token_accounting(context, evaluation)?),
    }
    .validate()
}

fn response_json(
    context: &LoopContext<'_>,
    response: crate::response_observation::ResponseObservation,
) -> Result<baux::artifact::json::Value, String> {
    use crate::response_observation::{RESPONSE_SCHEMA, ResponseReference};

    let proposal = response.proposal;
    let tokens = response
        .tokens
        .ok_or("Generation response lost token accounting")?;
    let reference =
        baux::artifact::json::to_value(response.reference).map_err(|error| error.to_string())?;
    let distribution = match context.run.proposal_method {
        Some(crate::procedural_pool::ProposalMethod::QuadraticFlow) => "lie_algebra_flow_v1",
        Some(crate::procedural_pool::ProposalMethod::PairMix) => "pair_rotation_mix_v1",
        _ => context.run.proposal_kernel().name(),
    };
    Ok(baux::artifact::json::json!({
        "schema": RESPONSE_SCHEMA,
        "reference": reference,
        "measured_pair": matches!(response.reference, ResponseReference::PairedIncumbent),
        "proposal": {
            "distribution": distribution,
            "seed": proposal.seed,
            "candidate": proposal.candidate,
            "coordinates": proposal.coordinates,
            "changed_coordinates": proposal.changed_coordinates,
            "quantized_survival": proposal.quantized_survival(),
            "squared_displacement": proposal.squared_displacement,
            "metric_radius": proposal.metric_radius,
            "trust_radius": proposal.trust_radius,
        },
        "candidate": {
            "mean": response.candidate.mean,
            "variance": response.candidate.variance,
        },
        "baseline": {
            "mean": response.baseline.mean,
            "variance": response.baseline.variance,
        },
        "improvement": {
            "mean": response.improvement.mean,
            "variance": response.improvement.variance,
        },
        "accepted": response.accepted,
        "tokens": {
            "samples": tokens.samples,
            "context": tokens.context,
            "prompt": tokens.prompt,
            "drafted": tokens.drafted,
            "evaluated": tokens.evaluated,
            "repaired": tokens.repaired,
            "generated": tokens.generated,
            "committed": tokens.committed,
        },
    }))
}

pub(super) fn response_record(
    context: &LoopContext<'_>,
    proposal: &crate::resident_metal::Proposals,
    evaluation: &Evaluation,
    decision: crate::resident_metal::NoisyDecision,
    initializing: bool,
) -> Result<baux::artifact::json::Value, String> {
    let mut record = response_json(
        context,
        response_observation(context, proposal, evaluation, decision, initializing)?,
    )?;
    record["numerical"] = baux::artifact::json::json!(decision.numerical.label());
    record["posterior_response"] = baux::artifact::json::json!({
        "components": evaluation
            .paired
            .map_or(0, |paired| paired.response_components),
    });
    Ok(record)
}

impl LoopContext<'_> {
    fn generation_json(
        &self,
        state: &LoopState,
        evaluation: &Evaluation,
        metrics: &RoundMetrics,
    ) -> Result<baux::artifact::json::Value, String> {
        let code = codelength(&evaluation.rollouts);
        let rollouts = &evaluation.rollouts;
        let eval_pos: usize = rollouts.iter().map(|r| r.evaluated_positions).sum();
        let gen_tokens: usize = rollouts.iter().map(|r| r.tokens.len()).sum();
        let mut generation = baux::artifact::json::json!({
            "reward": evaluation.reported_reward(),
            "optimizer_value": evaluation.mean,
            "variance": evaluation.variance,
            "diagnostics": evaluation.diagnostics,
            "selection": self.run.selection.unwrap_or_default(),
            "temperature": metrics.candidate_temperature,
            "incumbent_temperature": state.temperature,
            "proposal_seconds": metrics.proposal_seconds,
            "reward_seconds": evaluation.reward_seconds,
            "evaluation_seconds": metrics.evaluation_seconds,
            "tell_seconds": metrics.tell_seconds,
            "draft": rollouts.iter().map(|r| &r.draft).collect::<Vec<_>>(),
            "target_evaluated_positions": rollouts.iter().map(|r| r.evaluated_positions - r.draft.as_ref().map_or(0, |d| d.evaluated_positions)).sum::<usize>(),
            "target_gpu_seconds": rollouts.iter().map(|r| r.gpu_seconds - r.draft.as_ref().map_or(0.0, |d| d.generation_gpu_seconds + d.cache_gpu_seconds)).sum::<f64>(),
            "position_amplification": eval_pos as f64 / gen_tokens.max(1) as f64,
            "changed_weights": metrics.changed_weights,
            "changed_fraction": metrics.changed_fraction,
            "generated_tokens": gen_tokens,
            "prompt_tokens": evaluation.prompt_tokens,
            "context_tokens": evaluation.prompt_tokens + rollouts.iter().map(|r| r.token_count).max().unwrap_or(0),
            "generation_samples": rollouts.len(),
            "drafted_tokens": rollouts.iter().filter_map(|r| r.draft.as_ref()).map(|draft| draft.generated_tokens).sum::<usize>(),
            "scored_tokens": if code.tokens == 0 && matches!(self.config.reward, GenerationReward::SequenceKernel) { rollouts.iter().map(|r| r.token_count).sum::<usize>() } else { code.tokens },
            "codelength_bits": (code.tokens != 0).then_some(code.bits),
            "bits_per_token": (code.tokens != 0).then(|| code.bits / code.tokens as f64),
            "codelength_scope": "complete-reference-continuation-under-true-causal-prefix",
            "evaluated_core_positions": eval_pos,
            "evaluated_tokens": eval_pos,
            "evaluated_positions": eval_pos,
            "committed_tokens": rollouts.iter().map(|r| r.committed_tokens).sum::<usize>(),
            "rollout_gpu_seconds": rollouts.iter().map(|r| r.gpu_seconds).sum::<f64>(),
            "broad_passes": rollouts.iter().map(|r| r.broad_passes).sum::<usize>(),
            "correction_waves": rollouts.iter().map(|r| r.correction_waves).sum::<usize>(),
            "repair_batches": rollouts.iter().map(|r| r.repair_batches).sum::<usize>(),
            "repaired_tokens": rollouts.iter().map(|r| r.repaired_tokens).sum::<usize>(),
            "accepted_tokens": rollouts.iter().map(|r| r.accepted_tokens).sum::<usize>(),
            "first_mismatch": rollouts.iter().map(|r| r.first_mismatch).collect::<Vec<_>>(),
            "evaluated_lengths": rollouts.iter().map(|r| &r.evaluated_lengths).collect::<Vec<_>>(),
            "accepted_lengths": rollouts.iter().map(|r| &r.accepted_lengths).collect::<Vec<_>>(),
            "committed_lengths": rollouts.iter().map(|r| &r.committed_lengths).collect::<Vec<_>>(),
            "route_samples": rollouts.iter().map(|r| &r.route_samples).collect::<Vec<_>>(),
        });
        Self::annotate_evaluation(evaluation, &mut generation)?;
        Ok(generation)
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn round_record(
        &self,
        state: &mut LoopState,
        step: u32,
        initializing: bool,
        region: usize,
        proposal: &crate::resident_metal::Proposals,
        decision: crate::resident_metal::NoisyDecision,
        evaluation: &Evaluation,
        tensor_version: &crate::tensor_store::TensorVersion,
        alignment: Option<gen_alignment::Alignment>,
        metrics: &RoundMetrics,
    ) -> Result<baux::artifact::json::Value, String> {
        let controller = state.search.controller_info()?;
        let mut record = optimization_record(
            step + 1,
            initializing,
            proposal,
            decision,
            controller,
            state.search.reliability_info()?,
            metrics.wall_seconds,
            metrics.proposal_seconds + metrics.tell_seconds,
        );
        annotate_proposal(&state.search, proposal, &mut record);
        if let Some(alignment) = alignment {
            record["muon_alignment"] =
                baux::artifact::json::to_value(alignment).map_err(|error| error.to_string())?;
        }
        record["response"] = response_record(self, proposal, evaluation, decision, initializing)?;
        record["generation"] = self.generation_json(state, evaluation, metrics)?;
        if let Some(report) = state.pool_rank_report.take() {
            record["pool_rank_audit"] = report;
        }
        state.record_rank(evaluation, &mut record);
        if let Some(report) = state.search.objective_report()? {
            let width = report.width as usize;
            record["acquisition"]["objectives"] = baux::artifact::json::json!({
                "method": "vector",
                "values": report.values.map(|values| values[..width].to_vec()),
                "weights": report.weights[..width].to_vec(),
                "nondominated_mask": report.nondominated_mask,
                "selected": report.selected,
                "region": region,
                "regions": state.search.morbo_regions(),
            });
        }
        record["tensor_version"] = baux::artifact::json::json!({
            "id": tensor_version.id,
            "parent": tensor_version.parent,
            "seed": tensor_version.seed,
            "radius": tensor_version.radius,
            "distribution": tensor_version.distribution,
            "blocks": tensor_version.blocks.len(),
            "basis_seed": tensor_version.basis_seed,
            "accepted_parent": state.incumbent_version,
        });
        if let Some(bounds) = proposal.bound_report() {
            record["history_bounds"] = baux::artifact::json::json!({
                "history": bounds.history,
                "neighbors": bounds.neighbors,
                "raw_survivors": bounds.raw_survivors,
                "final_survivors": bounds.final_survivors,
                "bound_violations": bounds.bound_violations,
                "rank_violations": bounds.rank_violations,
                "pruning_enabled": false,
            });
        }
        Ok(record)
    }
    #[allow(clippy::too_many_arguments)]
    pub(super) fn finish_step(
        &self,
        state: &mut LoopState,
        step: u32,
        initializing: bool,
        region: usize,
        proposal: &crate::resident_metal::Proposals,
        decision: crate::resident_metal::NoisyDecision,
        evaluation: &mut Evaluation,
        tensor_version: &crate::tensor_store::TensorVersion,
        round_align: Option<gen_alignment::Alignment>,
        changes: &[(u64, f64)],
        round_path: &Path,
        metrics: RoundMetrics,
    ) -> Result<(), String> {
        state.minimum_changed_weights = state.minimum_changed_weights.min(metrics.changed_weights);
        state.commit(
            evaluation,
            decision.accepted,
            region,
            tensor_version.id,
            metrics.candidate_temperature,
        );
        state.record_updates(
            self.config.record_tensor_updates,
            step + 1,
            proposal,
            decision.accepted,
            changes,
        )?;
        let text = self.retain_round(step, round_path, evaluation)?;
        let controller = state.search.controller_info()?;
        let record = self.round_record(
            state,
            step,
            initializing,
            region,
            proposal,
            decision,
            evaluation,
            tensor_version,
            round_align,
            &metrics,
        )?;
        state.write_record(&record, self.checkpoint_step(step))?;
        state.loop_seconds += metrics.wall_seconds;
        state.walls.push(metrics.wall_seconds);
        state.journal.training(
            step + 1,
            metrics.wall_seconds,
            evaluation,
            decision.accepted,
            if initializing {
                "initialization"
            } else {
                "optimization"
            },
        )?;
        report_round(
            step + 1,
            self.run.rounds(),
            RoundTiming {
                wall: metrics.wall_seconds,
                proposal: metrics.proposal_seconds,
                evaluation: metrics.evaluation_seconds,
                tell: metrics.tell_seconds,
            },
            RoundTrust {
                proposed: proposal.length,
                next: controller.length,
                outcome: decision.trust_outcome,
                successes: controller.success_counter,
                failures: controller.failure_counter,
            },
            metrics.candidate_temperature,
            state.temperature,
            metrics.changed_weights,
            metrics.changed_fraction,
            evaluation,
            decision,
            text.as_deref(),
        );
        state
            .journal
            .gate(step + 1, self.config.signal_gate.as_ref())
    }
}

pub(super) struct InitialReport {
    pub(super) observation: Evaluation,
    pub(super) journal: journal::Journal,
    pub(super) validation: Vec<f32>,
    pub(super) reference_reward: Option<Vec<f32>>,
    pub(super) reward_audit: Option<baux::artifact::json::Value>,
}

pub(super) struct InitialContext<'a> {
    pub(super) runtime: &'a Runtime,
    pub(super) decoder: &'a decode::Decoder,
    pub(super) verifier: Option<&'a block_decode::BlockDecoder>,
    pub(super) weights: &'a CandidateWeights,
    pub(super) base: &'a metal::Buffer,
    pub(super) config: &'a GenerationConfig,
    pub(super) sampling_seed: u64,
    pub(super) output: &'a Path,
    pub(super) evaluator: &'a mut Option<Evaluator>,
    pub(super) tokenizer: Option<&'a ByteDecoder>,
    pub(super) validation_tasks: &'a [GenerationTask],
    pub(super) causal: Option<&'a CausalScorer>,
    pub(super) causal_log: Option<&'a mut CausalLog>,
    pub(super) vector_control: bool,
    pub(super) experiment_start: Instant,
}

fn initial_causal(
    scorer: Option<&CausalScorer>,
    runtime: &Runtime,
    weights: &CandidateWeights,
    base: &metal::Buffer,
    initial: &mut Evaluation,
    vector_control: bool,
) -> Result<(), String> {
    let Some(scorer) = scorer else {
        return Ok(());
    };
    let evidence = scorer.initial(runtime, weights.row(base)?)?;
    attach_causal(initial, evidence, vector_control)
}

impl InitialContext<'_> {
    pub(super) fn run(self) -> Result<InitialReport, String> {
        let Self {
            runtime,
            decoder,
            verifier,
            weights,
            base,
            config,
            sampling_seed,
            output,
            evaluator,
            tokenizer,
            validation_tasks,
            causal,
            causal_log,
            vector_control,
            experiment_start,
        } = self;
        let mut observation = observe_base(
            runtime,
            decoder,
            verifier,
            weights,
            base,
            config,
            sampling_seed,
            output,
            evaluator.as_mut(),
            tokenizer,
        )?;
        initial_causal(
            causal,
            runtime,
            weights,
            base,
            &mut observation,
            vector_control,
        )?;
        if let (Some(scorer), Some(log)) = (causal, causal_log) {
            log.observe(scorer, runtime, weights.row(base)?, 0)?;
        }
        display_initial(tokenizer, &observation, &config.reward, output)?;
        let (reference_reward, reward_audit) =
            reference_controls(&observation, config, output, evaluator, tokenizer)?;
        let mut journal = journal::Journal::new(output, experiment_start, &observation)?;
        let validation = if std::env::var_os("BAUX_UPDATE_ALIGNMENT_SINGLE").is_some() {
            Vec::new()
        } else {
            validate_initial(
                runtime,
                decoder,
                verifier,
                weights,
                base,
                config,
                sampling_seed,
                output,
                evaluator.as_mut(),
                tokenizer,
                validation_tasks,
                &mut journal,
            )?
        };
        Ok(InitialReport {
            observation,
            journal,
            validation,
            reference_reward,
            reward_audit,
        })
    }
}

pub(super) struct FinalContext<'a> {
    pub(super) run: &'a RunConfig,
    pub(super) config: &'a GenerationConfig,
    pub(super) output: &'a Path,
    pub(super) runtime: &'a Runtime,
    pub(super) decoder: &'a decode::Decoder,
    pub(super) verifier: &'a Option<block_decode::BlockDecoder>,
    pub(super) weights: &'a CandidateWeights,
    pub(super) tokenizer: Option<&'a ByteDecoder>,
    pub(super) validation_tasks: &'a [GenerationTask],
    pub(super) evaluator: &'a mut Option<Evaluator>,
    pub(super) causal: &'a Option<CausalScorer>,
    pub(super) causal_log: &'a mut Option<CausalLog>,
    pub(super) seeds: GenerationSeeds,
    pub(super) experiment_start: Instant,
}

impl FinalContext<'_> {
    pub(super) fn run(
        self,
        state: &mut LoopState,
        initial_validation: Vec<f32>,
        reference_reward: Option<Vec<f32>>,
        reward_audit: Option<baux::artifact::json::Value>,
    ) -> Result<(), String> {
        let final_weights =
            state.finalize_state(self.config.record_tensor_updates, self.output, self.weights)?;
        if let (Some(scorer), Some(log)) = (self.causal.as_ref(), self.causal_log.as_mut()) {
            log.observe(
                scorer,
                self.runtime,
                self.weights.row(&final_weights)?,
                self.run.rounds(),
            )?;
        }
        let validation_start = Instant::now();
        let final_validation = validate_final(
            self.runtime,
            self.decoder,
            self.verifier.as_ref(),
            self.weights,
            &state.search.base_buffer(),
            self.config,
            state.temperature,
            self.seeds.sampling,
            &self.output.join("validation-final"),
            self.evaluator.as_mut(),
            self.tokenizer,
            self.validation_tasks,
        )?;
        state.journal.validation(
            self.run.rounds(),
            &final_validation,
            validation_start.elapsed().as_secs_f64(),
        )?;
        finish_run(
            self.run,
            self.config,
            self.output,
            self.tokenizer,
            state,
            FinishReport {
                experiment_start: self.experiment_start,
                initial_validation,
                final_validation: final_validation.rewards,
                reference_reward,
                reward_audit,
            },
        )
    }
}
