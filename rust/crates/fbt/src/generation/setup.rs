use super::generation::*;

#[derive(Clone, Copy)]
pub(super) struct CorpusCaps {
    pub(super) prompt: usize,
    pub(super) output: usize,
}

#[derive(Clone, Copy)]
pub(super) struct CorpusSchedule<'a> {
    pub(super) path: &'a Path,
    pub(super) caps: CorpusCaps,
    pub(super) origins: u32,
}

pub(super) const VALIDATION_TASKS: usize = 32;

pub(super) fn validation_tasks(
    dataset_path: &Path,
    validation_path: Option<&Path>,
    prompt_cap: usize,
    output_cap: usize,
    seed: u64,
) -> Result<Vec<GenerationTask>, String> {
    let validation_path = validation_path
        .ok_or("generation pretraining requires a separate data.validation corpus")?;
    let train = dataset_path
        .canonicalize()
        .map_err(|error| format!("training corpus {}: {error}", dataset_path.display()))?;
    let validation = validation_path
        .canonicalize()
        .map_err(|error| format!("validation corpus {}: {error}", validation_path.display()))?;
    if train == validation {
        return Err("training and validation corpora must be separate files".into());
    }
    let dataset = crate::pretrain_data::PretrainDataset::load(&validation)?;
    let mut tasks = Vec::with_capacity(VALIDATION_TASKS);
    let mut offsets = std::collections::HashSet::with_capacity(VALIDATION_TASKS);
    for round in 0..dataset.sequences() {
        let Some((prompt, expected, offset)) =
            dataset.natural_task(prompt_cap, output_cap, seed, round)?
        else {
            break;
        };
        if offsets.insert(offset) {
            tasks.push(GenerationTask {
                prompt,
                expected,
                decoys: Vec::new(),
            });
        }
        if tasks.len() == VALIDATION_TASKS {
            return Ok(tasks);
        }
    }
    if !tasks.is_empty() {
        return Err(format!(
            "validation corpus yielded only {} distinct natural tasks; {} required",
            tasks.len(),
            VALIDATION_TASKS
        ));
    }
    let count = VALIDATION_TASKS.min(dataset.sequences() as usize);
    let total = prompt_cap
        .checked_add(output_cap)
        .ok_or("validation token range overflow")?
        .min(crate::pretrain_data::CONTEXT as usize);
    let prompt = if prompt_cap >= total {
        total / 2
    } else {
        prompt_cap
    };
    if prompt >= total || prompt == 0 {
        return Err("validation context leaves no target tokens".into());
    }
    for index in 0..count {
        let source = dataset.sequence(index as u32)?;
        tasks.push(GenerationTask {
            prompt: source[..prompt].to_vec(),
            expected: source[prompt..total].to_vec(),
            decoys: Vec::new(),
        });
    }
    Ok(tasks)
}

pub(super) fn corpus_task(
    config: &mut GenerationConfig,
    dataset_path: &Path,
    task_seed: u64,
    coding: bool,
) -> Result<(Vec<u32>, Vec<u32>, u32), String> {
    if coding {
        return Err(
            "code_reconstruction requires document-aligned corpus_prompt_tokens; packed \
             cross-file targets are not coding tasks"
                .into(),
        );
    }
    let dataset = crate::pretrain_data::PretrainDataset::load(dataset_path)?;
    let prompt_len = if let Some(tokens) = config.corpus_prompt_tokens {
        tokens as usize
    } else if config.corpus_prompt.is_empty() {
        config.max_tokens as usize
    } else {
        config.corpus_prompt.len()
    };
    let generated = config.max_tokens as usize;
    if let Some((prompt, expected, offset)) =
        dataset.natural_task(prompt_len, generated, task_seed, 0)?
    {
        config.max_tokens = u32::try_from(expected.len())
            .map_err(|_| "natural target exceeds u32 token capacity")?;
        return Ok((
            prompt,
            expected,
            u32::try_from(offset).map_err(|_| "corpus target offset exceeds u32")?,
        ));
    }
    if config.max_tokens > crate::pretrain_data::CONTEXT
        || prompt_len > crate::pretrain_data::CONTEXT as usize
    {
        let total = prompt_len + config.max_tokens as usize;
        let source = dataset.prefix(total)?;
        let p = if config.corpus_prompt.is_empty() {
            source[..prompt_len].to_vec()
        } else {
            std::mem::take(&mut config.corpus_prompt)
        };
        Ok((p, source[prompt_len..total].to_vec(), 0))
    } else {
        let index = (task_seed % u64::from(dataset.sequences())) as u32;
        let expected = dataset.sequence(index)?.to_vec();
        let prompt = if let Some(tokens) = config.corpus_prompt_tokens {
            expected[..tokens as usize].to_vec()
        } else {
            std::mem::take(&mut config.corpus_prompt)
        };
        Ok((prompt, expected, index))
    }
}

pub(super) fn corpus_window(
    dataset: &crate::pretrain_data::PretrainDataset,
    template: &GenerationTask,
    prompt_cap: usize,
    output_cap: usize,
    distribution_seed: u64,
    round: u32,
) -> Result<(GenerationTask, usize), String> {
    let (prompt, expected, offset) =
        dataset.window(prompt_cap, output_cap, distribution_seed, round)?;
    Ok((
        GenerationTask {
            prompt,
            expected,
            decoys: template.decoys.clone(),
        },
        offset,
    ))
}

pub(super) struct Resources {
    pub(super) runtime: std::sync::Arc<Runtime>,
    pub(super) decoder: decode::Decoder,
    pub(super) verifier: Option<block_decode::BlockDecoder>,
    pub(super) evaluator: Option<Evaluator>,
    pub(super) context: u32,
    pub(super) validation_tasks: Vec<GenerationTask>,
}

pub(super) fn prepare_resources(
    config: &GenerationConfig,
    output: &Path,
    validation_tasks: &[GenerationTask],
    tokenizer: Option<&ByteDecoder>,
    architecture: ResidualArchitecture,
    stage_samples: Option<u32>,
    context_capacity: Option<u32>,
) -> Result<Resources, String> {
    let runtime = Runtime::shared()?;
    let patch = architecture.patch_size();
    let raw_positions = config
        .tasks
        .iter()
        .map(|task| {
            task.prompt.len() + config.max_tokens as usize - usize::from(config.draft.is_none())
        })
        .max()
        .ok_or("generation requires tasks")?;
    let core_positions = raw_positions.div_ceil(patch);
    let context = context_capacity.map_or_else(
        || core_positions.next_power_of_two().max(CONTEXT as usize) as u32,
        |tokens| tokens.div_ceil(patch as u32),
    );
    if core_positions > context as usize {
        return Err("generation task exceeds retained-context capacity".into());
    }
    let visits = config
        .draft
        .map(|diffusion| {
            crate::forward_program::RecurrentCore::selective_fbt()
                .layer_visits(MODEL_LAYERS as usize, diffusion.visits[1] as usize)
                .map(|steps| steps.len().max(architecture.layer_steps().len()))
        })
        .transpose()?
        .unwrap_or_else(|| architecture.layer_steps().len());
    let decoder = decode::Decoder::with_visits(
        &runtime,
        context,
        context > ROWS || config.draft.is_some(),
        visits,
    )?;
    let verify = config.draft.is_some()
        || match config.verify.mode {
            VerifyMode::Auto => config.max_tokens >= 32 && config.tasks.len() == 1,
            VerifyMode::Serial => false,
            VerifyMode::AcceptedPrefix => true,
        };
    if context > ROWS && !verify {
        return Err("contexts above 8192 require chunked accepted-prefix verification".into());
    }
    if verify && config.tasks.len() != 1 {
        return Err("accepted-prefix verification requires exactly one task".into());
    }
    let mut verifier = verify
        .then(|| block_decode::BlockDecoder::with_architecture(&runtime, context, architecture))
        .transpose()?;
    if let Some(verifier) = verifier.as_mut() {
        verifier.profile(&runtime, stage_samples)?;
    }
    let mut evaluator = Evaluator::new(&config.reward, &config.tasks, tokenizer)?;
    if let Some(evaluator) = evaluator.as_mut() {
        let mut audit_tasks = config.tasks.clone();
        audit_tasks.extend_from_slice(validation_tasks);
        if let Some(audit) = evaluator.audit(&audit_tasks, tokenizer)? {
            baux::artifact::json::pretty_writer(
                File::create(output.join("reward-controls.json")).map_err(|e| e.to_string())?,
                &audit,
            )
            .map_err(|e| e.to_string())?;
            if audit["passed"] != true {
                return Err(
                    "generation reward failed reference-versus-negative controls; \
                     reward-controls.json retained"
                        .into(),
                );
            }
            eprintln!(
                "generation.controls passed=true tasks={} functional_correctness_established=false",
                audit_tasks.len()
            );
        }
    }
    Ok(Resources {
        runtime,
        decoder,
        verifier,
        evaluator,
        context,
        validation_tasks: validation_tasks.to_vec(),
    })
}

pub(super) fn reference_controls(
    _initial: &Evaluation,
    _config: &GenerationConfig,
    _output: &Path,
    _evaluator: &mut Option<Evaluator>,
    _tokenizer: Option<&ByteDecoder>,
) -> Result<(Option<Vec<f32>>, Option<baux::artifact::json::Value>), String> {
    Ok((None, None))
}

#[derive(Clone, Copy)]
pub(super) struct GenerationSeeds {
    pub(super) sampling: u64,
    pub(super) proposal: u64,
    pub(super) acquisition: u64,
    pub(super) model: u64,
}

/// Log-spaced early checkpoints followed by regular long-run measurements.
pub(super) const fn learning_checkpoint(round: u32) -> bool {
    (round <= 512 && round.is_power_of_two()) || (round >= 1000 && round % 1000 == 0)
}

impl GenerationSeeds {
    pub(super) fn new(run: &RunConfig, config: &GenerationConfig) -> Self {
        Self {
            sampling: config
                .seed
                .unwrap_or_else(|| run.derived_seed(0, "sampling", 0)),
            proposal: run.proposal_seed(),
            acquisition: run.acquisition_seed(),
            model: run.model_seed(),
        }
    }
}

pub(super) fn episode_task(
    rows: &[&baux::artifact::json::Value],
    index: usize,
    count: usize,
) -> Result<GenerationTask, String> {
    let target = rows[index];
    let mut decoys = Vec::new();
    let mut documents = std::collections::HashSet::new();
    for offset in 1..rows.len() {
        let row = rows[(index + offset) % rows.len()];
        let key = (
            row["pool"].as_str().unwrap_or_default(),
            row["document_offset"].as_u64().unwrap_or_default(),
        );
        if row["expected"] != target["expected"]
            && (row["pool"] != target["pool"]
                || row["document_offset"] != target["document_offset"])
            && documents.insert(key)
            && decoys.len() < count
        {
            decoys.push(row["expected"].clone());
        }
    }
    if decoys.len() != count {
        return Err(format!(
            "episode has only {} distinct-document decoys; requires {count}",
            decoys.len()
        ));
    }
    baux::artifact::json::from_value(baux::artifact::json::json!({"prompt":target["prompt"],"expected":target["expected"],"decoys":decoys})).map_err(|e| e.to_string())
}

pub(super) fn episode_rows<'a>(
    document: &'a baux::artifact::json::Value,
    split: &str,
    coding: bool,
) -> Result<Vec<&'a baux::artifact::json::Value>, String> {
    let rows = baux::artifact::json::pointer(document, &format!("/splits/{split}/episodes"))
        .and_then(|value| value.as_seq())
        .ok_or_else(|| format!("missing {split} generation episodes"))?;
    let rows = rows
        .iter()
        .filter(|row| !coding || matches!(row["bucket"].as_str(), Some("implementation" | "tests")))
        .collect::<Vec<_>>();
    if rows.is_empty() {
        return Err(format!("no eligible {split} generation episodes"));
    }
    Ok(rows)
}

#[allow(clippy::too_many_arguments)]
pub(super) fn write_config(
    run: &RunConfig,
    config: &GenerationConfig,
    output: &Path,
    objective: &baux::artifact::json::Value,
    runtime: &Runtime,
    context: u32,
    model_seed: u64,
    proposal_seed: u64,
    acquisition_seed: u64,
    sampling_seed: u64,
) -> Result<(), String> {
    let parameters =
        ResidualArchitecture::from_model(run.model.ok_or("generation model is required")?)
            .parameter_count();
    let task_shapes = config.tasks.iter().map(|task| baux::artifact::json::json!({
        "prompt_tokens":task.prompt.len(),"generated_tokens":task.expected.len(),"decoys":task.decoys.len()
    })).collect::<Vec<_>>();
    let mut persisted = config.clone();
    persisted.tasks.clear();
    let resolved = baux::artifact::json::json!({"model_seed":model_seed,"proposal_seed":proposal_seed,
        "acquisition_seed":acquisition_seed,"sampling_seed":sampling_seed,
        "config":persisted,"task_shapes":task_shapes,
        "objective":objective,
        "selection":run.selection.unwrap_or_default(),
        "history_distance_semantics":"latent procedural geometry; independent directions use expected squared norms and omit sample cross terms and weight rounding",
        "generation_in_loop":true,
        "parameters":parameters,"cache_policy":"rebuild_per_candidate_per_task",
        "device":runtime.info().name,
        "low_power_mode":crate::apple_gpu::power_mode().ok(),
        "student_cache_context":context,
        "position_encoding":{"kind":"rope","base":ROPE_BASE,"dimensions":HEAD_DIM},
        "sampling_policy":if matches!(config.reward, GenerationReward::CausalObjectives) {
            "one candidate evaluation per round; mean causal likelihood is compared with the surrogate incumbent; tail measurements are diagnostics"
        } else if run.objective_acquisition.is_none() {
            "one procedural candidate cycle shares a corpus task and random stream; the measured incumbent is reused exactly within that cycle"
        } else {
            "fresh_round_condition; vector control observation"
        },
        "backend":"native_metal_mlx_derived_gemvt_pisa",
        "timing":"proposal+prefill+decode+reward+tell+artifact_writes; initial incumbent excluded"});
    baux::artifact::json::pretty_writer(
        File::create(output.join("generation.json")).map_err(|e| e.to_string())?,
        &resolved,
    )
    .map_err(|e| e.to_string())?;
    Ok(())
}

pub(super) fn display_initial(
    tokenizer: Option<&ByteDecoder>,
    initial: &Evaluation,
    reward: &GenerationReward,
    output: &Path,
) -> Result<(), String> {
    if let Some(text) = decoded_completion(
        tokenizer,
        &initial.rollouts[0],
        &output.join("initial"),
        false,
    )? {
        let metric = if matches!(
            reward,
            GenerationReward::ConditionalCrossEntropy | GenerationReward::CausalObjectives
        ) {
            format!(
                "bits_per_token={}",
                -f64::from(initial.mean) / std::f64::consts::LN_2
            )
        } else if matches!(reward, GenerationReward::BoundedCrossEntropy) {
            let bpt = if initial.mean > 0.0 {
                -f64::from(initial.mean).ln() / std::f64::consts::LN_2
            } else {
                f64::INFINITY
            };
            format!("reward={:.4} bits_per_token={:.3}", initial.mean, bpt)
        } else {
            format!("reward={}", initial.mean)
        };
        display_completion(
            &format!(
                "generation.sample phase=initial tokens={} {metric}",
                initial.rollouts[0].tokens.len(),
            ),
            &text,
        );
    }
    Ok(())
}
