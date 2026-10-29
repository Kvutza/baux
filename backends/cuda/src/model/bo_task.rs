use super::VOCAB;
use crate::{CudaResult, cuda_error};
use baux::semantics::mhc4::ScaleShape;
use baux::semantics::optimizer::{PoolShape, ReliabilityPolicy, SurrogateSpec, splitmix64};
use std::fs;
use std::path::Path;

#[derive(Clone)]
pub(super) struct Mhc4Task {
    pub prompt: Vec<u32>,
    pub targets: Vec<u32>,
    pub corpus: Option<baux::artifact::pretrain::CorpusBatch>,
}

impl Mhc4Task {
    pub(super) fn causal(batch: baux::artifact::pretrain::CorpusBatch) -> Self {
        Self {
            prompt: Vec::new(),
            targets: Vec::new(),
            corpus: Some(batch),
        }
    }

    pub(super) fn input(&self) -> &[u32] {
        self.corpus
            .as_ref()
            .map_or(self.prompt.as_slice(), |c| c.batch.tokens.as_slice())
    }

    pub(super) fn targets(&self) -> &[u32] {
        self.corpus.as_ref().map_or(self.targets.as_slice(), |c| {
            &c.batch.labels[baux::semantics::mhc4::PATCH - 1..]
        })
    }

    pub(super) fn capacity(&self) -> CudaResult<usize> {
        if let Some(corpus) = &self.corpus {
            Ok(corpus.batch.tokens.len())
        } else {
            self.prompt
                .len()
                .checked_add(self.targets.len())
                .ok_or_else(|| "CUDA mHC-4 context overflow".into())
        }
    }

    pub(super) fn sample_seed(&self, fallback: u64) -> u64 {
        self.corpus
            .as_ref()
            .map_or(fallback, |c| splitmix64(c.batch.offset as u64))
    }

    pub(super) fn weight(&self) -> u32 {
        self.corpus.as_ref().map_or(1, |c| c.weight)
    }
}

pub(super) enum Mhc4Source {
    Rows(Vec<Mhc4Task>),
    Corpus {
        data: baux::artifact::pretrain::PretrainDataset,
        prompt: usize,
        output: usize,
        seed: u64,
    },
    Causal {
        data: baux::artifact::pretrain::CorpusSet,
        window: usize,
        seed: u64,
    },
}

impl Mhc4Source {
    pub(super) fn causal(
        input: &str,
        window: usize,
        seed: u64,
        validation: bool,
    ) -> CudaResult<Self> {
        let path = Path::new(input);
        let data = if validation {
            baux::artifact::pretrain::CorpusSet::load_eval(path)?
        } else {
            baux::artifact::pretrain::CorpusSet::load(path)?
        };
        Ok(Self::Causal { data, window, seed })
    }

    pub(super) fn load(input: &str, target: &str, output: usize, seed: u64) -> CudaResult<Self> {
        if input.ends_with(".bauxptn") {
            let prompt = target
                .parse::<usize>()
                .map_err(|error| format!("invalid mHC-4 corpus prompt cap: {error}"))?;
            return Ok(Self::Corpus {
                data: baux::artifact::pretrain::PretrainDataset::load(Path::new(input))?,
                prompt,
                output,
                seed,
            });
        }
        Ok(Self::Rows(mhc4_tasks(input, target)?))
    }

    pub(super) fn task(&self, round: usize) -> CudaResult<Mhc4Task> {
        match self {
            Self::Rows(tasks) => {
                let index = if tasks.len() == 1 { 0 } else { round };
                tasks
                    .get(index)
                    .cloned()
                    .ok_or_else(|| "CUDA mHC-4 training task schedule ended early".into())
            }
            Self::Corpus {
                data,
                prompt,
                output,
                seed,
            } => {
                let round = u32::try_from(round).map_err(|_| "CUDA mHC-4 round exceeds u32")?;
                let (prompt, targets, _) = data.window(*prompt, *output, *seed, round)?;
                Ok(Mhc4Task {
                    prompt,
                    targets,
                    corpus: None,
                })
            }
            Self::Causal { data, window, seed } => data
                .causal_batch(*window, *seed, u32::try_from(round).map_err(cuda_error)?)
                .map(Mhc4Task::causal),
        }
    }

    pub(super) fn fixed(&self) -> bool {
        matches!(self, Self::Rows(tasks) if tasks.len() == 1)
    }

    pub(super) fn count(&self, rounds: usize) -> usize {
        match self {
            Self::Rows(tasks) => tasks.len(),
            Self::Corpus { .. } | Self::Causal { .. } => rounds,
        }
    }

    pub(super) fn capacity(&self) -> CudaResult<usize> {
        match self {
            Self::Rows(tasks) => tasks
                .iter()
                .map(Mhc4Task::capacity)
                .collect::<CudaResult<Vec<_>>>()?
                .into_iter()
                .max()
                .ok_or_else(|| String::from("CUDA mHC-4 BO requires a training task")),
            Self::Corpus { prompt, output, .. } => prompt
                .checked_add(*output)
                .ok_or_else(|| String::from("CUDA mHC-4 BO context overflow")),
            Self::Causal { window, .. } => Ok(*window),
        }
    }

    pub(super) fn validation(&self) -> CudaResult<Vec<Mhc4Task>> {
        match self {
            Self::Rows(tasks) => Ok(tasks.clone()),
            Self::Causal { data, window, seed } => data
                .causal_batches(*window, *seed)
                .map(|batches| batches.into_iter().map(Mhc4Task::causal).collect()),
            Self::Corpus {
                data,
                prompt,
                output,
                seed,
            } => corpus_tasks(data, *prompt, *output, *seed),
        }
    }
}

fn corpus_tasks(
    data: &baux::artifact::pretrain::PretrainDataset,
    prompt: usize,
    output: usize,
    seed: u64,
) -> CudaResult<Vec<Mhc4Task>> {
    const TASKS: usize = 32;
    let mut tasks = Vec::with_capacity(TASKS);
    let mut offsets = std::collections::HashSet::with_capacity(TASKS);
    for round in 0..data.sequences() {
        let Some((task_prompt, targets, offset)) =
            data.natural_task(prompt, output, seed, round)?
        else {
            break;
        };
        if offsets.insert(offset) {
            tasks.push(Mhc4Task {
                prompt: task_prompt,
                targets,
                corpus: None,
            });
        }
        if tasks.len() == TASKS {
            return Ok(tasks);
        }
    }
    if !tasks.is_empty() {
        return Err(format!(
            "CUDA mHC-4 validation corpus yielded only {} distinct natural tasks; {TASKS} required",
            tasks.len()
        ));
    }
    let count = TASKS.min(data.sequences() as usize);
    let total = prompt
        .checked_add(output)
        .ok_or("CUDA mHC-4 validation token range overflow")?
        .min(baux::artifact::pretrain::CONTEXT as usize);
    let prompt_len = prompt.min(total);
    if prompt_len == total {
        return Err("CUDA mHC-4 validation context leaves no target tokens".into());
    }
    for index in 0..count {
        let source = data.sequence(index as u32)?;
        tasks.push(Mhc4Task {
            prompt: source[..prompt_len].to_vec(),
            targets: source[prompt_len..total].to_vec(),
            corpus: None,
        });
    }
    Ok(tasks)
}

fn token_rows(path: &str) -> CudaResult<Vec<Vec<u32>>> {
    let value: baux::artifact::json::Value =
        baux::artifact::json::from_slice(&fs::read(path).map_err(|e| e.to_string())?)
            .map_err(|e| e.to_string())?;
    let rows = value
        .as_seq()
        .ok_or_else(|| format!("mHC-4 token file {path} must contain a JSON array"))?;
    if rows.first().is_some_and(|row| row.is_seq()) {
        baux::artifact::json::from_value(value).map_err(|e| e.to_string())
    } else {
        let row = baux::artifact::json::from_value(value).map_err(|e| e.to_string())?;
        Ok(vec![row])
    }
}

fn mhc4_tasks(prompt_path: &str, target_path: &str) -> CudaResult<Vec<Mhc4Task>> {
    let prompts = token_rows(prompt_path)?;
    let targets = token_rows(target_path)?;
    if prompts.len() != targets.len() || prompts.is_empty() {
        return Err(
            "mHC-4 prompt and target files must contain the same positive row count".into(),
        );
    }
    Ok(prompts
        .into_iter()
        .zip(targets)
        .map(|(prompt, targets)| Mhc4Task {
            prompt,
            targets,
            corpus: None,
        })
        .collect())
}

pub(super) fn validate_task(task: &Mhc4Task, generated_cap: usize) -> CudaResult<()> {
    if let Some(corpus) = &task.corpus {
        let batch = &corpus.batch;
        baux::semantics::mhc4::CausalPlan::new(
            batch.tokens.len(),
            batch.labels.len(),
            &batch.loss_mask,
            usize::MAX,
        )?;
        if batch.tokens.len() > generated_cap
            || batch
                .tokens
                .iter()
                .chain(&batch.labels)
                .any(|&t| t >= VOCAB as u32)
        {
            return Err("CUDA causal mHC-4 batch exceeds its window or vocabulary".into());
        }
        return Ok(());
    }
    if task.prompt.is_empty()
        || task.prompt.len() % 64 != 0
        || task.targets.is_empty()
        || task.targets.len() % 64 != 0
        || task.targets.len() > generated_cap
        || task
            .prompt
            .iter()
            .chain(&task.targets)
            .any(|&token| token >= VOCAB as u32)
    {
        return Err(
            "CUDA mHC-4 BO tasks require vocabulary-valid, nonempty 64-token prompt/target \
             patches within GENERATED_TOKENS"
                .into(),
        );
    }
    Ok(())
}

pub(super) const fn learning_checkpoint(round: usize) -> bool {
    (round <= 512 && round.is_power_of_two()) || (round >= 1000 && round % 1000 == 0)
}

pub(super) fn condition_seed(root: u64, round: usize) -> CudaResult<u64> {
    let round = u64::try_from(round).map_err(|_| "CUDA mHC-4 round exceeds u64")?;
    Ok(splitmix64(
        root ^ round.wrapping_add(1) ^ 0x636f_7270_7573_2d78,
    ))
}

pub(super) fn stage_sample(round: usize, rounds: usize, samples: usize) -> Option<usize> {
    let samples = samples.min(rounds);
    match samples {
        0 => None,
        1 => (round == 0).then_some(0),
        _ => (0..samples).find(|&sample| sample * (rounds - 1) / (samples - 1) == round),
    }
}

pub(super) fn parse_arg<T: std::str::FromStr>(
    args: &[String],
    index: usize,
    default: T,
) -> CudaResult<T> {
    args.get(index).map_or(Ok(default), |value| {
        value
            .parse()
            .map_err(|_| format!("invalid argument {index}: {value}"))
    })
}

pub(super) struct BoConfig {
    pub model_seed: u64,
    pub output_dir: String,
    pub generated: usize,
    pub visits: usize,
    pub temperature: f32,
    pub proposal_seed: u64,
    pub radius: f32,
    pub bo_rounds: usize,
    pub acquisition: String,
    pub acquisition_seed: u64,
    pub evaluation_seed: u64,
    pub arms: usize,
    pub candidates: usize,
    pub beta: f32,
    pub radius_min: f32,
    pub radius_max: f32,
    pub stage_samples: usize,
    pub reference_seed: u64,
    pub window_tokens: Option<usize>,
    pub trust_shape: ScaleShape,
    pub reliability: ReliabilityPolicy,
    pub surrogate: SurrogateSpec,
    pub pool_shape: PoolShape,
    pub training: Mhc4Source,
    pub validation: Vec<Mhc4Task>,
    pub validation_interval: Option<usize>,
    pub primary: Mhc4Task,
    pub core_positions: usize,
    pub core_rows: usize,
    pub core_context: usize,
}

fn parse_flags(
    mut args: &[String],
) -> CudaResult<(
    &[String],
    ReliabilityPolicy,
    SurrogateSpec,
    Option<usize>,
    Option<u64>,
    Option<ScaleShape>,
)> {
    let mut reliability = ReliabilityPolicy::default();
    let mut window_tokens = None;
    let mut reference_seed = None;
    let mut trust_shape = None;
    let mut surrogate = SurrogateSpec {
        fit: true,
        ..Default::default()
    };
    while args.len() >= 2 {
        let flag = args[args.len() - 2].as_str();
        let value = &args[args.len() - 1];
        match flag {
            "--trust-shape" => trust_shape = Some(ScaleShape::parse(value)?),
            "--window-tokens" => {
                window_tokens = Some(
                    value
                        .parse::<usize>()
                        .map_err(|error| format!("invalid causal window: {error}"))?,
                )
            }
            "--reference-seed" => {
                reference_seed = Some(
                    value
                        .parse::<u64>()
                        .map_err(|error| format!("invalid corpus seed: {error}"))?,
                )
            }
            "--reliability" => {
                reliability = baux::artifact::json::from_str::<ReliabilityPolicy>(value)
                    .map_err(|error| format!("invalid reliability policy: {error}"))?
                    .validate()?
            }
            "--surrogate" => {
                surrogate = baux::artifact::json::from_str::<SurrogateSpec>(value)
                    .map_err(|error| format!("invalid surrogate settings: {error}"))?
                    .validate()?
            }
            _ => break,
        }
        args = &args[..args.len() - 2];
    }
    Ok((
        args,
        reliability,
        surrogate,
        window_tokens,
        reference_seed,
        trust_shape,
    ))
}

fn parse_sources(
    args: &[String],
    generated: usize,
    evaluation_seed: u64,
    validation_seed: u64,
    reference_seed: u64,
    window_tokens: Option<usize>,
) -> CudaResult<(Mhc4Source, Vec<Mhc4Task>, Option<usize>)> {
    let training = if let Some(window) = window_tokens {
        Mhc4Source::causal(&args[1], window, reference_seed, false)?
    } else {
        Mhc4Source::load(&args[1], &args[2], generated, evaluation_seed)?
    };
    let validation_start = match args.len() {
        16 => Some(13),
        22 => Some(19),
        23 => Some(20),
        _ => None,
    };
    let validation = if let Some(start) = validation_start {
        if let Some(window) = window_tokens {
            Mhc4Source::causal(&args[start], window, reference_seed, true)?.validation()?
        } else {
            Mhc4Source::load(&args[start], &args[start + 1], generated, validation_seed)?
                .validation()?
        }
    } else {
        Vec::new()
    };
    let validation_interval = if let Some(start) = validation_start {
        Some(
            args[start + 2]
                .parse::<usize>()
                .map_err(|error| format!("invalid mHC-4 validation interval: {error}"))?,
        )
    } else {
        None
    };
    Ok((training, validation, validation_interval))
}

fn check_config(
    training: &Mhc4Source,
    validation: &[Mhc4Task],
    generated: usize,
    temperature: f32,
    radius: f32,
    bo_rounds: usize,
    window_tokens: Option<usize>,
    validation_interval: Option<usize>,
) -> CudaResult<(Mhc4Task, usize, usize, usize)> {
    if generated == 0
        || (window_tokens.is_none() && generated % 64 != 0)
        || window_tokens.is_some_and(|window| {
            window < 64 || window > 1_048_576 || generated != window || temperature != 0.0
        })
        || !radius.is_finite()
        || radius <= 0.0
        || bo_rounds < 128
        || validation_interval == Some(0)
    {
        return Err(
            "CUDA mHC-4 BO requires valid 64-token prompt/target patches and a positive radius"
                .into(),
        );
    }
    let primary = training.task(0)?;
    for task in std::iter::once(&primary).chain(validation) {
        validate_task(task, generated)?;
    }
    if let Mhc4Source::Rows(tasks) = training
        && tasks.len() != 1
        && tasks.len() < bo_rounds
    {
        return Err(
            "CUDA mHC-4 rotating training inputs require one pre-materialized task per BO round"
                .into(),
        );
    }
    if validation
        .iter()
        .any(|heldout| primary.input() == heldout.input() && primary.targets() == heldout.targets())
    {
        return Err(
            "CUDA mHC-4 held-out validation tasks must be disjoint from training tasks".into(),
        );
    }
    let core_positions = std::iter::once(training.capacity()?)
        .chain(
            validation
                .iter()
                .map(Mhc4Task::capacity)
                .collect::<CudaResult<Vec<_>>>()?,
        )
        .max()
        .ok_or("CUDA mHC-4 BO requires a training task")?
        .div_ceil(64);
    let core_rows = core_positions
        .next_power_of_two()
        .max(if window_tokens.is_some() { 128 } else { 64 });
    let core_context = core_rows.max(4096);
    if core_context > 65_536 {
        return Err("CUDA mHC-4 BO coarse context exceeds the resident CUDA limit".into());
    }
    Ok((primary, core_positions, core_rows, core_context))
}

impl BoConfig {
    pub(super) fn parse(args: &[String]) -> CudaResult<Self> {
        let flags = parse_flags(args)?;
        let args = flags.0;
        if args.len() < 5 || !matches!(args.len(), 5..=13 | 16 | 19 | 20 | 22 | 23) {
            return Err(
                "usage: mhc4-bo-bench MODEL_SEED TRAIN_INPUT TARGET_INPUT OUTPUT_DIR \
                 GENERATED_TOKENS [VISITS=2] [TEMPERATURE=1.0] [PROPOSAL_SEED=17] [RADIUS=0.001] \
                 [ROUNDS=128] [ACQUISITION=ucb] [ACQUISITION_SEED=PROPOSAL_SEED] \
                 [EVALUATION_SEED=PROPOSAL_SEED] [ARMS=1 CANDIDATES=4 UCB_BETA=1 \
                 RADIUS_MIN=RADIUS RADIUS_MAX=2 VALIDATION_SEED=EVALUATION_SEED] \
                 [STAGE_SAMPLES=0] [VALIDATION_INPUT VALIDATION_TARGET VALIDATION_INTERVAL] \
                 [--reliability JSON] [--surrogate JSON] [--window-tokens N --reference-seed N] \
                 [--trust-shape scalar|tensor-family-static] (for an BAUXPTN input, TARGET_INPUT \
                 is the prompt cap; JSON inputs may be one flat task or ordered task rows)"
                    .into(),
            );
        }
        let generated = parse_arg(args, 4, 1_048_576_usize)?;
        let temperature = parse_arg(args, 6, 1.0_f32)?;
        let proposal_seed = parse_arg(args, 7, 17_u64)?;
        let radius = parse_arg(args, 8, 0.001_f32)?;
        let bo_rounds = parse_arg(args, 9, 128_usize)?;
        let acquisition = args.get(10).map(String::as_str).unwrap_or("ucb").to_owned();
        if !matches!(acquisition.as_str(), "ucb" | "thompson") {
            return Err("CUDA mHC-4 operator-BO requires UCB or Thompson acquisition".into());
        }
        let evaluation_seed = parse_arg(args, 12, proposal_seed)?;
        let semantic = matches!(args.len(), 19 | 20 | 22 | 23);
        let arms = if semantic { parse_arg(args, 13, 1)? } else { 1 };
        let candidates = if semantic { parse_arg(args, 14, 4)? } else { 4 };
        let validation_seed = if semantic {
            parse_arg(args, 18, evaluation_seed)?
        } else {
            evaluation_seed
        };
        let reference_seed = flags.4.unwrap_or(evaluation_seed);
        let pool_shape = PoolShape::new(arms, candidates)?;
        if !matches!(pool_shape.count(), 1 | 4) {
            return Err("CUDA mHC-4 resident search requires one or four candidates".into());
        }
        let sources = parse_sources(
            args,
            generated,
            evaluation_seed,
            validation_seed,
            reference_seed,
            flags.3,
        )?;
        let core = check_config(
            &sources.0,
            &sources.1,
            generated,
            temperature,
            radius,
            bo_rounds,
            flags.3,
            sources.2,
        )?;
        Ok(Self {
            model_seed: args[0]
                .parse::<u64>()
                .map_err(|error| format!("invalid mHC-4 model seed: {error}"))?,
            output_dir: args[3].clone(),
            generated,
            visits: parse_arg(args, 5, 2_usize)?,
            temperature,
            proposal_seed,
            radius,
            bo_rounds,
            acquisition,
            acquisition_seed: parse_arg(args, 11, proposal_seed)?,
            evaluation_seed,
            arms,
            candidates,
            beta: if semantic {
                parse_arg(args, 15, 1.0)?
            } else {
                1.0
            },
            radius_min: if semantic {
                parse_arg(args, 16, radius)?
            } else {
                radius
            },
            radius_max: if semantic {
                parse_arg(args, 17, 2.0)?
            } else {
                2.0
            },
            stage_samples: if matches!(args.len(), 20 | 23) {
                parse_arg(args, 19, 0)?
            } else {
                0
            },
            reference_seed,
            window_tokens: flags.3,
            trust_shape: flags.5.unwrap_or(if flags.3.is_some() {
                ScaleShape::TensorFamilyStatic
            } else {
                ScaleShape::Scalar
            }),
            reliability: flags.1,
            surrogate: flags.2,
            pool_shape,
            training: sources.0,
            validation: sources.1,
            validation_interval: sources.2,
            primary: core.0,
            core_positions: core.1,
            core_rows: core.2,
            core_context: core.3,
        })
    }
}
