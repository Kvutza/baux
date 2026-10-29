use super::generation::*;
use crate::traits::Oracle;

pub(super) fn score(
    runtime: &Runtime,
    decoder: &decode::Decoder,
    verifier: Option<(&block_decode::BlockDecoder, Option<&[decode::Rollout]>)>,
    row: CandidateRow<'_>,
    config: &GenerationConfig,
    seed: u64,
    path: &Path,
    evaluator: Option<&mut Evaluator>,
    tokenizer: Option<&ByteDecoder>,
    persist: bool,
) -> Result<Evaluation, String> {
    if persist || diagnostics_required(&config.reward) {
        std::fs::create_dir_all(path).map_err(|e| e.to_string())?;
    }
    let first_drafts = verifier
        .and_then(|(_, drafts)| drafts)
        .map(|drafts| {
            drafts
                .get(..1)
                .ok_or("generation requires an incumbent continuation")
        })
        .transpose()?;
    let first = verifier.map(|(verifier, _)| (verifier, first_drafts));
    let mut rollouts = generate(runtime, decoder, first, row, config, seed)?;
    if matches!(
        config.reward,
        GenerationReward::SequenceKernel | GenerationReward::CodeExecution { .. }
    ) {
        let drafts = verifier
            .and_then(|(_, drafts)| drafts)
            .map(|drafts| {
                drafts
                    .get(1..2)
                    .ok_or("distributional rewards require two incumbent continuations")
            })
            .transpose()?;
        let second = generate(
            runtime,
            decoder,
            verifier.map(|(verifier, _)| (verifier, drafts)),
            row,
            config,
            crate::hash::splitmix64(seed ^ 0x7365_7175_656e_6365),
        )?;
        rollouts.extend(second);
    }
    score_rollouts(config, rollouts, path, evaluator, tokenizer, persist)
}

fn generate(
    runtime: &Runtime,
    decoder: &decode::Decoder,
    verifier: Option<(&block_decode::BlockDecoder, Option<&[decode::Rollout]>)>,
    row: CandidateRow<'_>,
    config: &GenerationConfig,
    seed: u64,
) -> Result<Vec<decode::Rollout>, String> {
    if let Some((verifier, drafts)) = verifier {
        if let Some(drafts) = drafts {
            verifier.verify(runtime, decoder, row, &config.tasks, config, seed, drafts)
        } else {
            verifier.generate(runtime, decoder, row, &config.tasks, config, seed)
        }
    } else {
        config
            .tasks
            .iter()
            .enumerate()
            .map(|(index, task)| {
                decoder.generate(
                    runtime,
                    row,
                    task,
                    config,
                    crate::hash::splitmix64(seed ^ index as u64),
                )
            })
            .collect::<Result<Vec<_>, _>>()
    }
}

fn diagnostics_required(reward: &GenerationReward) -> bool {
    matches!(
        reward,
        GenerationReward::Command { .. } | GenerationReward::CommandObjectives { .. }
    )
}

fn score_rollouts(
    config: &GenerationConfig,
    rollouts: Vec<decode::Rollout>,
    path: &Path,
    evaluator: Option<&mut Evaluator>,
    tokenizer: Option<&ByteDecoder>,
    persist: bool,
) -> Result<Evaluation, String> {
    if persist || diagnostics_required(&config.reward) {
        if let Some(tokenizer) = tokenizer {
            std::fs::create_dir_all(path).map_err(|error| error.to_string())?;
            for (index, rollout) in rollouts.iter().enumerate() {
                let bytes = tokenizer.decode_bytes(&rollout.tokens)?;
                let text = String::from_utf8_lossy(&bytes);
                let name = if index == 0 {
                    "generated.txt".to_owned()
                } else {
                    format!("generated-{:04}.txt", index + 1)
                };
                std::fs::write(path.join(name), text.as_bytes())
                    .map_err(|error| error.to_string())?;
            }
        }
    }
    let diagnostics = if persist || diagnostics_required(&config.reward) {
        rollouts.iter().map(rollout_diagnostics).collect::<Vec<_>>()
    } else {
        Vec::new()
    };
    let start = Instant::now();
    let reward = evaluate(config, &rollouts, &diagnostics, path, evaluator, tokenizer)?;
    let observation = reward.observation()?;
    let control = observation.control();
    let result = Evaluation {
        prompt_tokens: config.tasks.iter().map(|task| task.prompt.len()).sum(),
        diagnostics,
        rollouts,
        rewards: reward.rewards,
        mean: control.mean,
        variance: control.variance,
        observation,
        objectives: reward.vector,
        reward_seconds: start.elapsed().as_secs_f64(),
        causal: None,
        paired: None,
    };
    if persist {
        write_evaluation(path, &result)?;
    }
    Ok(result)
}

fn write_evaluation(path: &Path, evaluation: &Evaluation) -> Result<(), String> {
    std::fs::create_dir_all(path).map_err(|e| e.to_string())?;
    baux::artifact::json::pretty_writer(
        File::create(path.join("evaluation.json")).map_err(|e| e.to_string())?,
        evaluation,
    )
    .map_err(|e| e.to_string())
}

pub(super) fn retain(path: &Path, evaluation: &mut Evaluation) -> Result<(), String> {
    if evaluation.diagnostics.is_empty() {
        evaluation.diagnostics = evaluation
            .rollouts
            .iter()
            .map(rollout_diagnostics)
            .collect();
    }
    write_evaluation(path, evaluation)
}

#[derive(Serialize)]
struct DecodeAudit {
    schema: &'static str,
    exact_tokens: bool,
    block_tokens: usize,
    direct_tokens: usize,
    #[deser(skip_serializing_if = Option::is_none)]
    first_difference: Option<usize>,
    #[deser(skip_serializing_if = Option::is_none)]
    block_token: Option<u32>,
    #[deser(skip_serializing_if = Option::is_none)]
    direct_token: Option<u32>,
    block_wall_ms: f64,
    block_gpu_ms: f64,
    direct_wall_ms: f64,
    direct_gpu_ms: f64,
}

pub(super) struct FbtOracle<'a> {
    pub runtime: &'a Runtime,
    pub decoder: &'a decode::Decoder,
    pub verifier: Option<(
        &'a block_decode::BlockDecoder,
        Option<&'a [decode::Rollout]>,
    )>,
    pub weights: &'a CandidateWeights,
    pub config: &'a GenerationConfig,
    pub seed: u64,
    pub path: &'a Path,
    pub evaluator: Option<&'a mut Evaluator>,
    pub tokenizer: Option<&'a ByteDecoder>,
    pub audit: bool,
    pub causal: Option<CausalObjective<'a>>,
    pub vector_control: bool,
    pub persist: bool,
}

pub(super) struct CausalObjective<'a> {
    pub scorer: &'a CausalScorer,
    pub incumbent: CandidateRow<'a>,
    pub batch: u32,
    pub baseline: f32,
}

fn first_difference(block: &[u32], direct: &[u32]) -> Option<usize> {
    block
        .iter()
        .zip(direct)
        .position(|(left, right)| left != right)
        .or_else(|| (block.len() != direct.len()).then_some(block.len().min(direct.len())))
}

fn audit_decode(
    runtime: &Runtime,
    decoder: &decode::Decoder,
    weights: CandidateRow<'_>,
    config: &GenerationConfig,
    seed: u64,
    path: &Path,
    block: &decode::Rollout,
) -> Result<(), String> {
    let direct = decoder.generate(
        runtime,
        weights,
        &config.tasks[0],
        config,
        crate::hash::splitmix64(seed),
    )?;
    let difference = first_difference(&block.tokens, &direct.tokens);
    let audit = DecodeAudit {
        schema: "baux.decode_audit.v1",
        exact_tokens: difference.is_none(),
        block_tokens: block.tokens.len(),
        direct_tokens: direct.tokens.len(),
        first_difference: difference,
        block_token: difference.and_then(|index| block.tokens.get(index).copied()),
        direct_token: difference.and_then(|index| direct.tokens.get(index).copied()),
        block_wall_ms: block.wall_seconds * 1_000.0,
        block_gpu_ms: block.gpu_seconds * 1_000.0,
        direct_wall_ms: direct.wall_seconds * 1_000.0,
        direct_gpu_ms: direct.gpu_seconds * 1_000.0,
    };
    baux::artifact::json::pretty_writer(
        File::create(path.join("direct-audit.json")).map_err(|error| error.to_string())?,
        &audit,
    )
    .map_err(|error| error.to_string())?;
    eprintln!(
        "decode.audit exact={} block_ms={:.3} direct_ms={:.3} first_difference={:?}",
        audit.exact_tokens, audit.block_wall_ms, audit.direct_wall_ms, audit.first_difference
    );
    if !audit.exact_tokens {
        return Err("accepted-prefix output differs from direct candidate decoding".into());
    }
    Ok(())
}

impl Oracle for FbtOracle<'_> {
    type Evidence = Evaluation;

    fn observe(
        &mut self,
        candidate: crate::search::DeviceView<'_>,
    ) -> Result<
        (
            crate::objective_observation::ObjectiveObservation,
            Self::Evidence,
        ),
        String,
    > {
        let (buffer, offset) = candidate
            .as_metal()
            .ok_or("FBT oracle requires a Metal candidate")?;
        let weights = self
            .weights
            .row_view(buffer, offset, candidate.row_bytes())?;
        let mut evaluation = score(
            self.runtime,
            self.decoder,
            self.verifier,
            weights,
            self.config,
            self.seed,
            self.path,
            self.evaluator.as_deref_mut(),
            self.tokenizer,
            self.persist,
        )?;
        if let Some(causal) = &self.causal {
            let evidence = causal.scorer.paired(
                self.runtime,
                weights,
                causal.incumbent,
                causal.batch,
                causal.baseline,
            )?;
            attach_causal(&mut evaluation, evidence, self.vector_control)?;
        }
        if self.audit {
            audit_decode(
                self.runtime,
                self.decoder,
                weights,
                self.config,
                self.seed,
                self.path,
                &evaluation.rollouts[0],
            )?;
        }
        Ok((evaluation.observation, evaluation))
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn observe_base(
    runtime: &Runtime,
    decoder: &decode::Decoder,
    verifier: Option<&block_decode::BlockDecoder>,
    weights: &CandidateWeights,
    base: &metal::Buffer,
    config: &GenerationConfig,
    seed: u64,
    output: &Path,
    evaluator: Option<&mut Evaluator>,
    tokenizer: Option<&ByteDecoder>,
) -> Result<Evaluation, String> {
    let path = output.join("initial");
    let candidate = crate::search::DeviceView::metal(base.clone(), 0, base.length() as usize);
    let mut oracle = FbtOracle {
        runtime,
        decoder,
        verifier: verifier.map(|verifier| (verifier, None)),
        weights,
        config,
        seed,
        path: &path,
        evaluator,
        tokenizer,
        audit: false,
        causal: None,
        vector_control: false,
        persist: true,
    };
    Ok(oracle.observe(candidate)?.1)
}
