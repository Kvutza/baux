use super::*;

pub(crate) struct Validation {
    pub(crate) rewards: Vec<f32>,
    pub(crate) objectives: Vec<Vec<f32>>,
}

impl Validation {
    pub(crate) fn mean(&self) -> f64 {
        self.rewards
            .iter()
            .map(|&reward| f64::from(reward))
            .sum::<f64>()
            / self.rewards.len() as f64
    }

    pub(crate) fn stddev(&self) -> f64 {
        if self.rewards.len() < 2 {
            return 0.0;
        }
        let mean = self.mean();
        let variance = self
            .rewards
            .iter()
            .map(|&reward| (f64::from(reward) - mean).powi(2))
            .sum::<f64>()
            / (self.rewards.len() - 1) as f64;
        variance.sqrt()
    }

    pub(crate) fn standard_error(&self) -> f64 {
        self.stddev() / (self.rewards.len() as f64).sqrt()
    }

    pub(crate) fn objective_means(&self) -> Vec<f32> {
        let Some(width) = self.objectives.first().map(Vec::len) else {
            return Vec::new();
        };
        let mut means = vec![0.0; width];
        for row in &self.objectives {
            for (mean, value) in means.iter_mut().zip(row) {
                *mean += *value;
            }
        }
        for mean in &mut means {
            *mean /= self.objectives.len() as f32;
        }
        means
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn validate_initial(
    runtime: &Runtime,
    decoder: &decode::Decoder,
    verifier: Option<&block_decode::BlockDecoder>,
    weights: &CandidateWeights,
    buffer: &metal::Buffer,
    config: &GenerationConfig,
    sampling_seed: u64,
    output: &Path,
    evaluator: Option<&mut Evaluator>,
    tokenizer: Option<&ByteDecoder>,
    validation_tasks: &[GenerationTask],
    journal: &mut journal::Journal,
) -> Result<Vec<f32>, String> {
    let validation_start = Instant::now();
    let initial_validation = validate_generation(
        runtime,
        decoder,
        verifier,
        weights,
        buffer,
        config,
        sampling_seed,
        &output.join("validation-initial"),
        evaluator,
        tokenizer,
        validation_tasks,
    )?;
    journal.validation(
        0,
        &initial_validation,
        validation_start.elapsed().as_secs_f64(),
    )?;
    Ok(initial_validation.rewards)
}

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
