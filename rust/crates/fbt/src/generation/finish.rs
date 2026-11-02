use super::generation::*;

pub(super) fn display_completion(header: &str, text: &str) {
    eprintln!("\n{header}");
    let clean = text.trim();
    let limit = clean.floor_char_boundary(clean.len().min(140));
    let snippet = clean[..limit].replace('\n', " ");
    eprintln!("generation.preview {snippet:?}");
}

pub(super) struct RoundTiming {
    pub(super) wall: f64,
    pub(super) proposal: f64,
    pub(super) evaluation: f64,
    pub(super) tell: f64,
}

pub(super) struct RoundTrust {
    pub(super) proposed: f32,
    pub(super) next: f64,
    pub(super) outcome: crate::trust_region::TrustRegionOutcome,
    pub(super) successes: i32,
    pub(super) failures: i32,
}

pub(super) fn report_round(
    step: u32,
    rounds: u32,
    timing: RoundTiming,
    trust: RoundTrust,
    candidate_temperature: f32,
    incumbent_temperature: f32,
    changed_weights: u64,
    changed_fraction: f64,
    evaluation: &Evaluation,
    decision: crate::resident_metal::NoisyDecision,
    text: Option<&str>,
) {
    let reward = evaluation.reported_reward();
    let code = codelength(&evaluation.rollouts);
    let (paired_gain, paired_se) = evaluation.paired.map_or_else(
        || ("unavailable".to_owned(), "unavailable".to_owned()),
        |paired| {
            (
                paired.improvement.to_string(),
                paired.improvement_variance.sqrt().to_string(),
            )
        },
    );
    eprintln!(
        "generation.round round={} total={} wall_ms={:.3} proposal_ms={:.3} evaluation_ms={:.3} \
         tell_ms={:.3} proposal_radius={:.9} next_radius={:.9} trust_outcome={:?} \
         trust_successes={} trust_failures={} temperature={} incumbent_temperature={} \
         changed_weights={} changed_fraction={:.9} generated_tokens={} generation_samples={} \
         evaluated_positions={} correction_waves={} repair_batches={} first_mismatch={:?} \
         reward={} paired_gain={} paired_se={} accepted={} prompt_tokens={} context_tokens={} \
         scored_tokens={} codelength_bits={} bits_per_token={} nll={}",
        step,
        rounds,
        timing.wall * 1000.0,
        timing.proposal * 1000.0,
        timing.evaluation * 1000.0,
        timing.tell * 1000.0,
        trust.proposed,
        trust.next,
        trust.outcome,
        trust.successes,
        trust.failures,
        candidate_temperature,
        incumbent_temperature,
        changed_weights,
        changed_fraction,
        evaluation
            .rollouts
            .iter()
            .map(|rollout| rollout.tokens.len())
            .sum::<usize>(),
        evaluation.rollouts.len(),
        evaluation
            .rollouts
            .iter()
            .map(|rollout| rollout.evaluated_positions)
            .sum::<usize>(),
        evaluation
            .rollouts
            .iter()
            .map(|rollout| rollout.correction_waves)
            .sum::<usize>(),
        evaluation
            .rollouts
            .iter()
            .map(|rollout| rollout.repair_batches)
            .sum::<usize>(),
        evaluation
            .rollouts
            .first()
            .and_then(|rollout| rollout.first_mismatch),
        reward,
        paired_gain,
        paired_se,
        decision.accepted,
        evaluation.prompt_tokens,
        evaluation.prompt_tokens
            + evaluation
                .rollouts
                .iter()
                .map(|rollout| rollout.token_count)
                .max()
                .unwrap_or(0),
        code.tokens,
        if code.tokens == 0 {
            "unavailable".into()
        } else {
            code.bits.to_string()
        },
        if code.tokens == 0 {
            "unavailable".into()
        } else {
            (code.bits / code.tokens as f64).to_string()
        },
        evaluation
            .rollouts
            .first()
            .and_then(|rollout| rollout.conditional_target_nll)
            .map_or_else(|| "unavailable".into(), |nll| nll.to_string()),
    );
    if let Some(text) = text {
        display_completion(
            &format!(
                "generation.sample round={} tokens={} temperature={} reward={} accepted={} \
                 bits_per_token={}",
                step,
                evaluation.rollouts[0].tokens.len(),
                candidate_temperature,
                reward,
                decision.accepted,
                if code.tokens == 0 {
                    "unavailable".into()
                } else {
                    (code.bits / code.tokens as f64).to_string()
                },
            ),
            text,
        );
    }
}

pub(super) struct FinishReport {
    pub(super) experiment_start: Instant,
    pub(super) initial_validation: Vec<f32>,
    pub(super) final_validation: Vec<f32>,
    pub(super) reference_reward: Option<Vec<f32>>,
    pub(super) reward_audit: Option<baux::artifact::json::Value>,
}

struct Truth {
    architecture: ResidualArchitecture,
    patch: usize,
    learned: bool,
    direct: usize,
    hashed: usize,
}

impl Truth {
    fn new(model: crate::config::ModelArchitecture, tokens: usize) -> Self {
        let architecture = ResidualArchitecture::from_model(model);
        let patch = architecture.patch_size();
        let learned = false;
        let direct = tokens.div_ceil(patch);
        Self {
            architecture,
            patch,
            learned,
            direct,
            hashed: tokens - direct,
        }
    }

    fn exact(&self) -> bool {
        self.patch == 1 || self.learned
    }

    fn engine(&self, draft: bool) -> &'static str {
        if self.learned {
            "coarse-plan-local-causal-block-decoder"
        } else if self.patch > 1 {
            "fixed-patch-hash-expansion"
        } else if draft {
            "diffusion-draft-causal-target"
        } else {
            "causal-target"
        }
    }

    fn scored_output(&self, reward: &GenerationReward) -> &'static str {
        match reward {
            GenerationReward::SequenceKernel => "conditional-sequence-kernel",
            GenerationReward::CodeExecution { .. } => "conditional-code-kernel",
            GenerationReward::ConditionalCrossEntropy => "complete-episode-codelength",
            GenerationReward::CausalObjectives => "complete-episode-causal-objectives",
            GenerationReward::BoundedCrossEntropy => "complete-episode-bounded-likelihood",
            _ if self.patch > 1 => "macro-loss-repeated-across-patch",
            _ => "target-completion",
        }
    }
}

pub(super) fn finish_run(
    run: &RunConfig,
    config: &GenerationConfig,
    output: &Path,
    tokenizer: Option<&ByteDecoder>,
    state: &mut LoopState,
    report: FinishReport,
) -> Result<(), String> {
    let FinishReport {
        experiment_start,
        initial_validation,
        final_validation,
        reference_reward,
        reward_audit,
    } = report;
    let completion = state
        .incumbent_rollouts
        .first()
        .ok_or("generation produced no final completion")?;
    baux::artifact::json::pretty_writer(
        File::create(output.join("completion.tokens.json")).map_err(|e| e.to_string())?,
        &baux::artifact::json::json!({
            "schema": "baux.generated_completion.v1",
            "tokens": completion.tokens,
            "token_count": completion.tokens.len(),
            "finish_reason": completion.finish_reason,
            "conditional_target_nll": completion.conditional_target_nll,
            "target_quality": completion.target_quality,
        }),
    )
    .map_err(|e| e.to_string())?;
    decoded_completion(tokenizer, completion, output, true)?;
    if let Some(path) = &config.save_checkpoint {
        let row = state.search.base_buffer();
        let architecture =
            ResidualArchitecture::from_model(run.model.ok_or("generation model is required")?);
        save_checkpoint(&row, path, architecture)?;
    }
    state.walls.sort_by(f64::total_cmp);
    let maximum = *state.walls.last().ok_or("no generation rounds")?;
    let middle = state.walls.len() / 2;
    let median = (state.walls[(state.walls.len() - 1) / 2] + state.walls[middle]) * 0.5;
    let mean = state.loop_seconds / state.walls.len() as f64;
    let latency_target_met = maximum * 1000.0 <= f64::from(run.target_ms());
    let effective_rounds = state.minimum_changed_weights > 0;
    let truth = Truth::new(
        run.model.ok_or("generation model is required")?,
        config.max_tokens as usize,
    );
    let context_target_tokens = config
        .tasks
        .iter()
        .map(|task| task.prompt.len() + config.max_tokens as usize)
        .max()
        .ok_or("missing generation task")?;
    let generation_target_met =
        state.minimum_context_tokens == context_target_tokens && truth.exact();
    let parameters = truth.architecture.parameter_count();
    let stage = if config.draft.is_some() {
        "diffusion_draft_target_bo"
    } else if config.purpose == crate::config::GenerationPurpose::Pretrain {
        "free_running_pretraining_bo"
    } else {
        "generated_reward_bo"
    };
    let result = baux::artifact::json::json!({
        "status": "completed",
        "stage": stage,
        "parameters": parameters,
        "rounds": state.walls.len(),
        "requested_rounds": run.rounds(),
        "accepted": state.accepted,
        "loop_seconds": state.loop_seconds,
        "elapsed_seconds": experiment_start.elapsed().as_secs_f64(),
        "generation_in_loop": true,
        "mean_wall_ms": mean * 1000.0,
        "median_wall_ms": median * 1000.0,
        "max_wall_ms": maximum * 1000.0,
        "target_round_ms": run.target_ms(),
        "latency_target_met": latency_target_met,
        "minimum_changed_weights": state.minimum_changed_weights,
        "effective_rounds": effective_rounds,
        "generation_target_met": generation_target_met,
        "minimum_generated_tokens": state.minimum_generated_tokens,
        "minimum_context_tokens": state.minimum_context_tokens,
        "context_target_tokens": context_target_tokens,
        "target_met": latency_target_met && effective_rounds && generation_target_met,
        "generation_engine": truth.engine(config.draft.is_some()),
        "scored_output": truth.scored_output(&config.reward),
        "exact_per_token_generation": truth.exact(),
        "exact_per_token_generation_meaning": "Each output token has a model decision; this does not assert equivalence to a dense autoregressive reference.",
        "learned_per_token_generation": truth.learned || truth.patch == 1,
        "fine_token_objective_exact": truth.patch == 1 || truth.learned,
        "direct_token_decisions_per_candidate": truth.direct,
        "hash_expanded_tokens_per_candidate": truth.hashed,
        "causal_output_verification": truth.patch == 1,
        "timing_scope": "proposal-generation-scoring-tell-completion-write-controller-write",
        "timing_excludes": ["setup", "initial-incumbent", "curve-write", "console-display", "validation", "final-checkpoint"],
        "token_counter_scope": "Final incumbent rollouts across tasks; not cumulative work over optimization rounds. See controller records for per-round work.",
        "minimum_generated_tokens_scope":"Minimum generated token count per candidate across measured rounds.",
        "validation_scope":"Configured validation tasks; dataset provenance determines whether these are held out.",
        "validation_initial":initial_validation,"validation_final":final_validation,
        "incumbent_temperature":state.temperature,"temperature_searched":config.temperature_bounds.is_some()});
    let mut result = result;
    let prompt = config
        .tasks
        .iter()
        .map(|task| task.prompt.len())
        .max()
        .ok_or("missing prompt")?;
    result["prompt_tokens"] = baux::artifact::json::json!(prompt);
    result["retained_context_tokens"] = baux::artifact::json::json!(state.minimum_context_tokens);
    result["target_input_positions"] =
        baux::artifact::json::json!(prompt + config.max_tokens as usize - 1);
    result["draft_input_positions"] =
        baux::artifact::json::json!(config.draft.map(|_| prompt + config.max_tokens as usize));
    result["requested_generated_tokens"] = baux::artifact::json::json!(config.max_tokens);
    add_counts(&mut result, state);
    result["drafted_tokens"] = baux::artifact::json::json!(
        state
            .incumbent_rollouts
            .iter()
            .filter_map(|rollout| rollout.draft.as_ref())
            .map(|draft| draft.generated_tokens)
            .sum::<usize>()
    );
    result["evaluated_core_positions"] = baux::artifact::json::json!(
        state
            .incumbent_rollouts
            .iter()
            .map(|rollout| rollout.evaluated_positions)
            .sum::<usize>()
    );
    result["evaluated_tokens"] = if truth.learned {
        baux::artifact::json::json!(
            state
                .incumbent_rollouts
                .iter()
                .map(|rollout| rollout.token_count)
                .sum::<usize>()
        )
    } else {
        result["evaluated_core_positions"].clone()
    };
    result["scored_tokens"] = baux::artifact::json::json!(if matches!(
        config.reward,
        GenerationReward::SequenceKernel | GenerationReward::CodeExecution { .. }
    ) {
        state
            .incumbent_rollouts
            .iter()
            .map(|rollout| rollout.token_count)
            .sum::<usize>()
    } else {
        state
            .incumbent_rollouts
            .iter()
            .filter_map(|rollout| rollout.target_quality)
            .map(|quality| quality.scored_tokens)
            .sum::<usize>()
    });
    if truth.learned {
        result["decoder_causal_waves"] = baux::artifact::json::json!(64);
    }
    result["repaired_tokens"] = baux::artifact::json::json!(
        state
            .incumbent_rollouts
            .iter()
            .map(|rollout| rollout.repaired_tokens)
            .sum::<usize>()
    );
    result["committed_tokens"] = baux::artifact::json::json!(
        state
            .incumbent_rollouts
            .iter()
            .map(|rollout| rollout.committed_tokens)
            .sum::<usize>()
    );
    result["generation_uses_reference_tokens"] = baux::artifact::json::json!(false);
    result["objective_uses_reference_tokens"] = baux::artifact::json::json!(matches!(
        config.reward,
        GenerationReward::DraftedCode { .. }
            | GenerationReward::CodeContrastive { .. }
            | GenerationReward::CodeReconstruction
            | GenerationReward::CodeObjectives { .. }
            | GenerationReward::ExactMatch
            | GenerationReward::TokenAccuracy
            | GenerationReward::ConditionalCrossEntropy
            | GenerationReward::CausalObjectives
            | GenerationReward::BoundedCrossEntropy
            | GenerationReward::SequenceKernel
            | GenerationReward::CodeExecution { .. }
    ));
    result["incumbent_control"] =
        baux::artifact::json::to_value(state.search.best()?).map_err(|e| e.to_string())?;
    result["purpose"] =
        baux::artifact::json::to_value(config.purpose).map_err(|e| e.to_string())?;
    result["initialization"] =
        baux::artifact::json::to_value(config.initialization).map_err(|e| e.to_string())?;
    result["corpus_reference_reward"] =
        baux::artifact::json::to_value(reference_reward).map_err(|e| e.to_string())?;
    result["reward_audit"] = reward_audit.unwrap_or(baux::artifact::json::Value::null());
    write_report(
        output,
        &result,
        state,
        latency_target_met && effective_rounds && generation_target_met,
    )
}

fn add_codelength(result: &mut baux::artifact::json::Value, rollouts: &[decode::Rollout]) {
    let code = codelength(rollouts);
    result["codelength_bits"] = baux::artifact::json::json!(code.bits);
    result["bits_per_token"] = if code.tokens == 0 {
        baux::artifact::json::Value::null()
    } else {
        baux::artifact::json::json!(code.bits / code.tokens as f64)
    };
    result["codelength_scope"] = baux::artifact::json::json!(
        "Complete reference continuation under true causal prefixes; prompt tokens are \
         conditioning context and are not charged."
    );
}

fn add_counts(result: &mut baux::artifact::json::Value, state: &LoopState) {
    add_codelength(result, &state.incumbent_rollouts);
    result["generation_samples"] = baux::artifact::json::json!(state.incumbent_rollouts.len());
    result["generated_tokens_per_sample"] =
        baux::artifact::json::json!(state.minimum_generated_tokens);
    result["generated_tokens_per_candidate"] = baux::artifact::json::json!(
        state
            .incumbent_rollouts
            .iter()
            .map(|rollout| rollout.token_count)
            .sum::<usize>()
    );
}

fn write_report(
    output: &Path,
    result: &baux::artifact::json::Value,
    state: &LoopState,
    target_met: bool,
) -> Result<(), String> {
    baux::artifact::json::pretty_writer(
        File::create(output.join("result.json")).map_err(|e| e.to_string())?,
        result,
    )
    .map_err(|e| e.to_string())?;
    eprintln!(
        "generation.summary rounds={} accepted={} loop_s={:.3} mean_round_ms={:.3} \
         max_round_ms={:.3} target_met={} result={}",
        state.walls.len(),
        state.accepted,
        state.loop_seconds,
        state.loop_seconds * 1000.0 / state.walls.len() as f64,
        state.walls.last().ok_or("no generation rounds")? * 1000.0,
        target_met,
        output.join("result.json").display(),
    );
    Ok(())
}
