use super::generation::*;

pub(super) fn evaluate(
    config: &GenerationConfig,
    rollouts: &[decode::Rollout],
    diagnostics: &[baux::artifact::json::Value],
    path: &Path,
    evaluator: Option<&mut Evaluator>,
    tokenizer: Option<&ByteDecoder>,
) -> Result<RewardResult, String> {
    if matches!(config.reward, GenerationReward::DraftedCode { .. }) {
        return drafted(rollouts, config, path, evaluator, tokenizer);
    }
    if matches!(config.reward, GenerationReward::CodeExecution { .. }) {
        return evaluator.ok_or("missing execution evaluator")?.execute(
            tokenizer.ok_or("execution requires a tokenizer")?,
            rollouts,
            path,
        );
    }
    if matches!(config.reward, GenerationReward::SequenceKernel) {
        return sequence_kernel(
            config,
            rollouts,
            tokenizer.ok_or("sequence kernel requires the corpus tokenizer")?,
        );
    }
    if matches!(config.reward, GenerationReward::CodeObjectives { .. }) {
        let likelihood = rollouts
            .iter()
            .map(|rollout| {
                rollout
                    .conditional_target_nll
                    .map(|loss| -loss)
                    .ok_or("block decoder did not produce conditional target NLL")
            })
            .collect::<Result<Vec<_>, _>>()?;
        let reconstruction = evaluator
            .ok_or("missing code reconstruction evaluator")?
            .score(tokenizer, &config.tasks, rollouts, path)?;
        let critical = rollouts
            .iter()
            .map(|rollout| {
                rollout
                    .target_quality
                    .map(|quality| -quality.worst_window_nll)
                    .ok_or("block decoder did not produce localized target loss")
            })
            .collect::<Result<Vec<_>, _>>()?;
        return protocol::native_vector(
            reconstruction.clone(),
            &[likelihood, critical, reconstruction],
            &[
                "mean_target_likelihood",
                "worst_window_target_likelihood",
                "byte_reconstruction",
            ],
        );
    }
    if matches!(config.reward, GenerationReward::CausalObjectives) {
        let quality = rollouts
            .iter()
            .map(|rollout| {
                rollout
                    .target_quality
                    .ok_or("block decoder did not produce causal target quality")
            })
            .collect::<Result<Vec<_>, _>>()?;
        let mean = quality
            .iter()
            .map(|quality| -quality.mean_nll)
            .collect::<Vec<_>>();
        let minimum = quality
            .iter()
            .map(|quality| -quality.maximum_nll)
            .collect::<Vec<_>>();
        let mut result = protocol::native_vector(
            mean.clone(),
            &[mean, minimum],
            &["mean-token-log-likelihood", "minimum-token-log-likelihood"],
        )?;
        result.variance = target_variance(config, rollouts)?;
        result.control_only = true;
        return Ok(result);
    }
    if let GenerationReward::CommandObjectives {
        program,
        args,
        timeout_ms,
    } = &config.reward
    {
        let response = command_response(
            config,
            rollouts,
            diagnostics,
            path,
            program,
            args,
            *timeout_ms,
            "baux.generation_objectives.v2",
        )?;
        return protocol::parse_vector(&response, config.tasks.len());
    }
    let rewards = scalar_reward(config, rollouts, diagnostics, path, evaluator, tokenizer)?;
    if rewards.len() != config.tasks.len() || rewards.iter().any(|score| !score.is_finite()) {
        return Err("evaluator must return one finite reward per task; higher is better".into());
    }
    Ok(RewardResult {
        variance: target_variance(config, rollouts)?,
        rewards,
        vector: None,
        control_only: false,
    })
}

fn sequence_kernel(
    config: &GenerationConfig,
    rollouts: &[decode::Rollout],
    tokenizer: &ByteDecoder,
) -> Result<RewardResult, String> {
    if config.tasks.len() != 1 || rollouts.len() != 2 {
        return Err("sequence kernel requires one target and two generated continuations".into());
    }
    let target = &config.tasks[0].expected;
    let target_bytes = tokenizer.decode_bytes(target)?;
    let left_bytes = tokenizer.decode_bytes(&rollouts[0].tokens)?;
    let right_bytes = tokenizer.decode_bytes(&rollouts[1].tokens)?;
    let mut scores = Vec::with_capacity(6);
    scores.push(kernel_score(
        f32::from(left_bytes == target_bytes),
        f32::from(right_bytes == target_bytes),
        f32::from(left_bytes == right_bytes),
    ));
    scores.push(kernel_score(
        token_kernel(&rollouts[0].tokens, target),
        token_kernel(&rollouts[1].tokens, target),
        token_kernel(&rollouts[0].tokens, &rollouts[1].tokens),
    ));
    for order in 1..=4 {
        scores.push(byte_score(&target_bytes, &left_bytes, &right_bytes, order));
    }
    let reward = scores.iter().sum::<f32>() / scores.len() as f32;
    Ok(RewardResult {
        rewards: vec![reward],
        vector: None,
        // Candidate and incumbent use identical prompts and random streams.
        // Kernel components are distinct measurements, not noise replicates.
        variance: Some(0.0),
        control_only: false,
    })
}

pub(crate) fn kernel_score(left: f32, right: f32, self_similarity: f32) -> f32 {
    left + right - self_similarity
}

fn token_kernel(left: &[u32], right: &[u32]) -> f32 {
    let total = left.len().max(right.len()).max(1);
    let mismatches = left
        .iter()
        .zip(right)
        .filter(|(left, right)| left != right)
        .count()
        + left.len().abs_diff(right.len());
    (-2.0 * mismatches as f32 / total as f32).exp()
}

const SKETCH: usize = 2048;

pub(crate) fn byte_score(target: &[u8], left: &[u8], right: &[u8], order: usize) -> f32 {
    let target = byte_sketch(target, order);
    let left = byte_sketch(left, order);
    let right = byte_sketch(right, order);
    kernel_score(
        rbf(&left, &target),
        rbf(&right, &target),
        rbf(&left, &right),
    )
}

fn rbf(left: &[f32], right: &[f32]) -> f32 {
    let distance = left
        .iter()
        .zip(right)
        .map(|(left, right)| (left - right).powi(2))
        .sum::<f32>();
    (-distance).exp()
}

fn byte_sketch(bytes: &[u8], order: usize) -> Vec<f32> {
    let mut sketch = vec![0.0f32; SKETCH];
    for gram in bytes.windows(order) {
        let key = gram
            .iter()
            .fold(order as u64, |key, byte| (key << 8) ^ u64::from(*byte));
        let hash = crate::hash::splitmix64(key);
        let sign = if hash >> 63 == 0 { 1.0 } else { -1.0 };
        sketch[hash as usize & (SKETCH - 1)] += sign;
    }
    let norm = sketch.iter().map(|value| value * value).sum::<f32>().sqrt();
    if norm > 0.0 {
        for value in &mut sketch {
            *value /= norm;
        }
    }
    sketch
}

fn target_variance(
    config: &GenerationConfig,
    rollouts: &[decode::Rollout],
) -> Result<Option<f32>, String> {
    if !matches!(
        config.reward,
        GenerationReward::ConditionalCrossEntropy
            | GenerationReward::CausalObjectives
            | GenerationReward::BoundedCrossEntropy
    ) {
        return Ok(None);
    }
    let is_bounded = matches!(config.reward, GenerationReward::BoundedCrossEntropy);
    let count = rollouts.len() as f32;
    let variance = rollouts.iter().try_fold(0.0, |sum, rollout| {
        rollout
            .target_quality
            .map(|quality| {
                if is_bounded {
                    let nll = rollout.conditional_target_nll.unwrap_or(0.0);
                    sum + (-2.0 * nll).exp() * quality.mean_nll_variance
                } else {
                    sum + quality.mean_nll_variance
                }
            })
            .ok_or("block decoder did not produce target-loss variance")
    })? / (count * count);
    Ok(Some(variance))
}

fn drafted(
    rollouts: &[decode::Rollout],
    config: &GenerationConfig,
    path: &Path,
    evaluator: Option<&mut Evaluator>,
    tokenizer: Option<&ByteDecoder>,
) -> Result<RewardResult, String> {
    let quality = evaluator.ok_or("missing generated-code scorer")?.score(
        tokenizer,
        &config.tasks,
        rollouts,
        path,
    )?;
    let agreement = rollouts
        .iter()
        .map(|rollout| {
            let draft = rollout
                .draft
                .as_ref()
                .ok_or("missing learned draft evidence")?;
            Ok(draft.matching_tokens as f32 / rollout.tokens.len().max(1) as f32)
        })
        .collect::<Result<Vec<_>, String>>()?;
    let work = rollouts
        .iter()
        .map(|rollout| {
            -(rollout.evaluated_positions as f32 / rollout.committed_tokens.max(1) as f32)
        })
        .collect::<Vec<_>>();
    protocol::native_vector(
        quality.clone(),
        &[quality, agreement, work],
        &[
            "generated-code",
            "draft-agreement",
            "negative-position-amplification",
        ],
    )
}

fn conditional_nll(rollouts: &[decode::Rollout]) -> Result<Vec<f32>, String> {
    rollouts
        .iter()
        .map(|rollout| {
            rollout
                .conditional_target_nll
                .map(|loss| -loss)
                .ok_or_else(|| "block decoder did not produce conditional target NLL".into())
        })
        .collect()
}

fn bounded_nll(rollouts: &[decode::Rollout]) -> Result<Vec<f32>, String> {
    rollouts
        .iter()
        .map(|rollout| {
            rollout
                .conditional_target_nll
                .map(|loss| (-loss).exp())
                .ok_or_else(|| "block decoder did not produce conditional target NLL".into())
        })
        .collect()
}

fn scalar_reward(
    config: &GenerationConfig,
    rollouts: &[decode::Rollout],
    diagnostics: &[baux::artifact::json::Value],
    path: &Path,
    evaluator: Option<&mut Evaluator>,
    tokenizer: Option<&ByteDecoder>,
) -> Result<Vec<f32>, String> {
    Ok(match &config.reward {
        GenerationReward::ExactMatch => config
            .tasks
            .iter()
            .zip(rollouts)
            .map(|(task, rollout)| {
                if task.expected == rollout.tokens {
                    1.0
                } else {
                    0.0
                }
            })
            .collect(),
        GenerationReward::TokenAccuracy => config
            .tasks
            .iter()
            .zip(rollouts)
            .map(|(task, rollout)| {
                let matches = task
                    .expected
                    .iter()
                    .zip(&rollout.tokens)
                    .filter(|(a, b)| a == b)
                    .count();
                matches as f32 / task.expected.len().max(rollout.tokens.len()) as f32
            })
            .collect(),
        GenerationReward::ConditionalCrossEntropy => conditional_nll(rollouts)?,
        GenerationReward::BoundedCrossEntropy => bounded_nll(rollouts)?,
        GenerationReward::FrozenQwen { .. }
        | GenerationReward::CodeReconstruction
        | GenerationReward::CodeContrastive { .. } => evaluator
            .ok_or("missing generation evaluator")?
            .score(tokenizer, &config.tasks, rollouts, path)?,
        GenerationReward::DraftedCode { .. } => {
            return Err("drafted-code requires vector scoring".into());
        }
        GenerationReward::Command {
            program,
            args,
            timeout_ms,
        } => command_reward(
            config,
            rollouts,
            diagnostics,
            path,
            program,
            args,
            *timeout_ms,
        )?,
        GenerationReward::CausalObjectives
        | GenerationReward::SequenceKernel
        | GenerationReward::CommandObjectives { .. }
        | GenerationReward::CodeObjectives { .. }
        | GenerationReward::CodeExecution { .. } => unreachable!("handled above"),
    })
}

fn command_reward(
    config: &GenerationConfig,
    rollouts: &[decode::Rollout],
    diagnostics: &[baux::artifact::json::Value],
    path: &Path,
    program: &Path,
    args: &[String],
    timeout_ms: u64,
) -> Result<Vec<f32>, String> {
    let response = command_response(
        config,
        rollouts,
        diagnostics,
        path,
        program,
        args,
        timeout_ms,
        "baux.generation_reward.v1",
    )?;
    #[derive(Deserialize)]
    #[deser(deny_unknown_fields)]
    struct Response {
        rewards: Vec<f32>,
    }
    Ok(baux::artifact::json::from_reader::<Response, _>(
        File::open(response).map_err(|e| e.to_string())?,
    )
    .map_err(|e| format!("invalid reward response: {e}"))?
    .rewards)
}

fn command_response(
    config: &GenerationConfig,
    rollouts: &[decode::Rollout],
    diagnostics: &[baux::artifact::json::Value],
    path: &Path,
    program: &Path,
    args: &[String],
    timeout_ms: u64,
    schema: &str,
) -> Result<std::path::PathBuf, String> {
    let request = path.join("request.json");
    let response = path.join("response.json");
    let payload = baux::artifact::json::json!({
        "schema": schema,
        "tasks": config.tasks,
        "rollouts": rollouts,
        "diagnostics": diagnostics,
    });
    baux::artifact::json::to_writer(File::create(&request).map_err(|e| e.to_string())?, &payload)
        .map_err(|e| e.to_string())?;
    let log = File::create(path.join("evaluator.log")).map_err(|e| e.to_string())?;
    let mut child = Command::new(program)
        .args(args)
        .arg(&request)
        .arg(&response)
        .stdin(Stdio::null())
        .stdout(Stdio::from(log.try_clone().map_err(|e| e.to_string())?))
        .stderr(Stdio::from(log))
        .spawn()
        .map_err(|e| format!("reward evaluator: {e}"))?;
    let start = Instant::now();
    loop {
        if let Some(status) = child.try_wait().map_err(|e| e.to_string())? {
            if !status.success() {
                return Err(format!("reward evaluator exited {status}"));
            }
            break;
        }
        if start.elapsed() >= Duration::from_millis(timeout_ms) {
            let _ = child.kill();
            let _ = child.wait();
            return Err("reward evaluator timed out; candidate not accepted".into());
        }
        std::thread::sleep(Duration::from_millis(2));
    }
    Ok(response)
}

use super::gen_protocol as protocol;
pub(super) use protocol::{RewardResult, VectorResult};
