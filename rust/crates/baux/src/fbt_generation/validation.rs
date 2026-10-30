use super::*;
use crate::traits::Oracle;

#[allow(clippy::too_many_arguments)]
pub(crate) fn validate_final(
    runtime: &Runtime,
    decoder: &decode::Decoder,
    verifier: Option<&block_decode::BlockDecoder>,
    weights: &CandidateWeights,
    buffer: &metal::Buffer,
    config: &GenerationConfig,
    temperature: f32,
    seed: u64,
    path: &Path,
    evaluator: Option<&mut Evaluator>,
    tokenizer: Option<&ByteDecoder>,
    tasks: &[GenerationTask],
) -> Result<Validation, String> {
    let mut config = config.clone();
    config.temperature = temperature;
    validate_generation(
        runtime, decoder, verifier, weights, buffer, &config, seed, path, evaluator, tokenizer,
        tasks,
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn validate_generation(
    runtime: &Runtime,
    decoder: &decode::Decoder,
    verifier: Option<&block_decode::BlockDecoder>,
    weights: &CandidateWeights,
    buffer: &metal::Buffer,
    config: &GenerationConfig,
    seed: u64,
    path: &Path,
    mut evaluator: Option<&mut Evaluator>,
    tokenizer: Option<&ByteDecoder>,
    tasks: &[GenerationTask],
) -> Result<Validation, String> {
    let started = Instant::now();
    if tasks.is_empty() {
        return Ok(Validation {
            rewards: Vec::new(),
            objectives: Vec::new(),
        });
    }
    std::fs::create_dir(path).map_err(|e| e.to_string())?;
    let mut rewards = Vec::new();
    let mut objectives = Vec::new();
    for (index, task) in tasks.iter().enumerate() {
        let mut evaluation_config = config.clone();
        evaluation_config.tasks = vec![task.clone()];
        evaluation_config.max_tokens = u32::try_from(task.expected.len())
            .map_err(|_| "validation action exceeds u32 token capacity")?;
        evaluation_config.validate()?;
        let task_path = path.join(format!("task-{:04}", index + 1));
        let candidate =
            crate::search::DeviceView::metal(buffer.clone(), 0, buffer.length() as usize);
        let mut oracle = FbtOracle {
            runtime,
            decoder,
            weights,
            verifier: verifier.map(|verifier| (verifier, None)),
            config: &evaluation_config,
            seed,
            path: &task_path,
            evaluator: evaluator.as_deref_mut(),
            tokenizer,
            audit: false,
            causal: None,
            vector_control: false,
            persist: false,
        };
        let (_, result) = oracle.observe(candidate)?;
        rewards.push(result.mean);
        if let Some(vector) = result.objectives.as_ref() {
            let row = vector.means();
            if objectives
                .first()
                .is_some_and(|previous: &Vec<f32>| previous.len() != row.len())
            {
                return Err("held-out objective width changed between tasks".into());
            }
            objectives.push(row);
        }
    }
    let phase = path.file_name().unwrap_or_default().to_string_lossy();
    eprintln!("generation.validation phase={phase} rewards={rewards:?} used_for_acceptance=false");
    let mean_reward =
        rewards.iter().map(|&reward| f64::from(reward)).sum::<f64>() / rewards.len() as f64;
    let (metric, value) = if matches!(
        config.reward,
        GenerationReward::ConditionalCrossEntropy | GenerationReward::CausalObjectives
    ) {
        ("bits_per_token", -mean_reward / std::f64::consts::LN_2)
    } else if matches!(config.reward, GenerationReward::BoundedCrossEntropy) {
        ("bounded_likelihood", mean_reward)
    } else {
        ("reward", mean_reward)
    };
    eprintln!(
        "generation.validation_summary phase={phase} tasks={} metric={metric} value={value:.9} \
         elapsed_ms={:.3}",
        rewards.len(),
        started.elapsed().as_secs_f64() * 1000.0,
    );
    if !objectives.is_empty() && objectives.len() != rewards.len() {
        return Err("held-out tasks disagree on scalar versus vector objectives".into());
    }
    Ok(Validation {
        rewards,
        objectives,
    })
}
