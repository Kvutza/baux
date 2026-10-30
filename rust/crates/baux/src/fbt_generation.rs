//! Generate/evaluate/tell with the existing full-coordinate resident ENN.
pub(super) use super::causal::{CausalEvidence, CausalLog, CausalScorer};
pub(super) use super::generation_reward::{Evaluator, reference_audit};
pub(super) use super::*;

#[path = "fbt_generation/context.rs"]
mod gen_context;
#[path = "fbt_generation/validation.rs"]
mod validation;
pub(super) use crate::config::{
    GenerationConfig, GenerationReward, GenerationTask, RunConfig, VerifyMode,
};
pub(super) use crate::text::ByteDecoder;
pub(super) use deser::{Deserialize, Serialize};
pub use gen_context::context_loop;
pub(super) use std::collections::BTreeMap;
pub(super) use std::fs::{File, OpenOptions};
pub(super) use std::io::{BufWriter, Write};
pub(super) use std::path::Path;
pub(super) use std::process::{Command, Stdio};
pub(super) use std::time::Duration;

pub(super) use super::gen_metrics::{
    RankAudit, annotate_proposal, codelength, rank_audit, rollout_diagnostics,
};
pub(super) use super::{
    gen_alignment,
    gen_checkpoint::*,
    gen_finish::*,
    gen_journal as journal,
    gen_oracle::{CausalObjective, FbtOracle, observe_base, retain, score},
    gen_record::*,
    gen_reward as reward,
    gen_reward::evaluate,
    gen_rounds::*,
    gen_setup as setup,
    gen_setup::*,
};
pub(super) use validation::{validate_final, validate_generation};

#[derive(Serialize)]
pub(super) struct Evaluation {
    pub(super) prompt_tokens: usize,
    pub(super) rollouts: Vec<decode::Rollout>,
    pub(super) rewards: Vec<f32>,
    pub(super) mean: f32,
    pub(super) variance: f32,
    #[deser(skip_serializing)]
    pub(super) observation: crate::objective_observation::ObjectiveObservation,
    #[deser(skip_serializing_if = Option::is_none)]
    pub(super) objectives: Option<reward::VectorResult>,
    pub(super) reward_seconds: f64,
    pub(super) diagnostics: Vec<baux::artifact::json::Value>,
    #[deser(skip_serializing_if = Option::is_none)]
    pub(super) causal: Option<CausalEvidence>,
    #[deser(skip_serializing_if = Option::is_none)]
    pub(super) paired: Option<PairedGeneration>,
}

#[derive(Clone, Copy, Serialize)]
pub(super) struct PairedGeneration {
    pub(super) condition_seed: u64,
    pub(super) sampling_seed: u64,
    pub(super) corpus_token_offset: usize,
    pub(super) candidate_reward: f32,
    pub(super) candidate_variance: f32,
    pub(super) incumbent_reward: f32,
    pub(super) incumbent_variance: f32,
    pub(super) improvement: f32,
    pub(super) improvement_variance: f32,
    pub(super) response_components: usize,
    pub(super) difference_statistic: &'static str,
    pub(super) optimizer_value: f32,
    pub(super) optimizer_variance: f32,
    pub(super) incumbent_generated_tokens: usize,
    pub(super) incumbent_evaluated_positions: usize,
    pub(super) incumbent_committed_tokens: usize,
    pub(super) incumbent_gpu_seconds: f64,
    pub(super) incumbent_reused: bool,
    pub(super) incumbent_expert_mask: [u64; routing::ROUTE_WORDS],
    pub(super) candidate_expert_mask: [u64; routing::ROUTE_WORDS],
}

pub(super) struct Validation {
    pub(super) rewards: Vec<f32>,
    pub(super) objectives: Vec<Vec<f32>>,
}

impl Validation {
    pub(super) fn mean(&self) -> f64 {
        self.rewards
            .iter()
            .map(|&reward| f64::from(reward))
            .sum::<f64>()
            / self.rewards.len() as f64
    }

    pub(super) fn stddev(&self) -> f64 {
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

    pub(super) fn standard_error(&self) -> f64 {
        self.stddev() / (self.rewards.len() as f64).sqrt()
    }

    pub(super) fn objective_means(&self) -> Vec<f32> {
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

impl Evaluation {
    pub(super) fn reported_reward(&self) -> f32 {
        self.paired
            .map_or(self.mean, |paired| paired.candidate_reward)
    }

    pub(super) fn annotate(&self, record: &mut baux::artifact::json::Value) -> Result<(), String> {
        if let Some(objectives) = &self.objectives {
            record["objectives"] =
                baux::artifact::json::to_value(objectives).map_err(|error| error.to_string())?;
        }
        Ok(())
    }
}

pub(super) fn decoded_completion(
    tokenizer: Option<&ByteDecoder>,
    rollout: &decode::Rollout,
    path: &Path,
    persist: bool,
) -> Result<Option<String>, String> {
    let Some(tokenizer) = tokenizer else {
        return Ok(None);
    };
    let bytes = tokenizer.decode_bytes(&rollout.tokens)?;
    let text = String::from_utf8_lossy(&bytes).into_owned();
    if persist {
        std::fs::write(path.join("completion.bin"), &bytes).map_err(|error| error.to_string())?;
        std::fs::write(path.join("completion.txt"), &text).map_err(|error| error.to_string())?;
    }
    Ok(Some(text))
}

pub(super) fn attach_causal(
    evaluation: &mut Evaluation,
    evidence: CausalEvidence,
    vector_control: bool,
) -> Result<(), String> {
    if !vector_control || !evaluation.observation.is_vector() {
        evaluation.mean = evidence.optimizer_value;
        evaluation.variance = evidence.variance;
        evaluation.observation = crate::objective_observation::ObjectiveObservation::scalar(
            evidence.optimizer_value,
            evidence.variance,
        )?;
    }
    evaluation.causal = Some(evidence);
    Ok(())
}

pub fn run_generation(run: &RunConfig, output: &Path) -> Result<(), String> {
    run.validate_experiment()?;
    let config = run
        .generation
        .as_ref()
        .ok_or("missing generation configuration")?;
    if let GenerationReward::CodeExecution { environment, .. } = &config.reward {
        let spec = super::code_environment::Environment::load(environment)?;
        let tokenizer = ByteDecoder::load(&spec.tokenizer)?;
        let mut config = config.clone();
        let task = spec.task(&tokenizer)?;
        if !config.tasks.is_empty() && config.tasks != [task.clone()] {
            return Err("code_execution tasks must match the environment prompt exactly".into());
        }
        config.tasks = vec![task];
        if config.save_final_checkpoint && config.save_checkpoint.is_none() {
            config.save_checkpoint = Some(output.join("checkpoint.safetensors"));
        }
        config.validate()?;
        return run_config(
            run,
            &config,
            output,
            spec.provenance(),
            Some(&tokenizer),
            &[],
            None,
        );
    }
    if config.tasks.is_empty() {
        return Err("standalone generation requires explicit tasks".into());
    }
    run_config(
        run,
        config,
        output,
        baux::artifact::json::json!({"kind":"explicit_generation_tasks"}),
        None,
        &[],
        None,
    )
}

pub fn run_generated(run: &RunConfig, dataset_path: &Path, output: &Path) -> Result<(), String> {
    run.validate_experiment()?;
    let mut config = run
        .generation
        .clone()
        .ok_or("missing generated-pretraining configuration")?;
    let tokenizer_path = dataset_path
        .parent()
        .ok_or("generated-pretraining dataset has no parent directory")?
        .join("tokenizer.json");
    let tokenizer = ByteDecoder::load(&tokenizer_path)?;
    let task_seed = run.derived_seed(0, "generated-pretrain-task", 0);
    let coding = matches!(
        config.reward,
        GenerationReward::CodeReconstruction
            | GenerationReward::CodeContrastive { .. }
            | GenerationReward::DraftedCode { .. }
            | GenerationReward::CodeObjectives { .. }
    );
    let decoy_count = match config.reward {
        GenerationReward::CodeContrastive { decoys, .. }
        | GenerationReward::DraftedCode { decoys, .. } => decoys,
        _ => 0,
    };
    let mut decoys = Vec::new();
    let mut validation_tasks = Vec::<GenerationTask>::new();
    let caps = CorpusCaps {
        prompt: config
            .corpus_prompt_tokens
            .map(|tokens| tokens as usize)
            .unwrap_or_else(|| config.corpus_prompt.len().max(config.max_tokens as usize)),
        output: config.max_tokens as usize,
    };
    let (prompt, expected, sequence_index, episode) = if let Some(path) = &config.episode_dataset {
        let document: baux::artifact::json::Value =
            baux::artifact::json::from_reader(File::open(path).map_err(|e| e.to_string())?)
                .map_err(|e| e.to_string())?;
        if document["schema"] != "baux.document_episodes.v2" {
            return Err("unsupported generation episode schema".into());
        }
        let episodes = episode_rows(&document, "train", coding)?;
        let index = (task_seed % episodes.len() as u64) as usize;
        let episode = episodes[index].clone();
        let heldout = episode_rows(&document, "validation", coding)?;
        let first =
            (run.derived_seed(0, "generated-validation-task", 0) % heldout.len() as u64) as usize;
        for offset in 0..heldout.len().min(setup::VALIDATION_TASKS) {
            validation_tasks.push(episode_task(
                &heldout,
                (first + offset) % heldout.len(),
                decoy_count,
            )?);
        }
        let task = episode_task(&episodes, index, decoy_count)?;
        decoys = task.decoys;
        (task.prompt, task.expected, index as u32, Some(episode))
    } else {
        let (p, e, idx) = corpus_task(&mut config, dataset_path, task_seed, coding)?;
        validation_tasks.extend(setup::validation_tasks(
            dataset_path,
            run.validation_dataset.as_deref(),
            caps.prompt,
            caps.output,
            run.derived_seed(0, "generated-validation-task", 0),
        )?);
        (p, e, idx, None)
    };
    let schedule = episode
        .is_none()
        .then(|| {
            crate::pretrain_data::PretrainDataset::load(dataset_path).map(|dataset| {
                CorpusSchedule {
                    path: dataset_path,
                    caps,
                    origins: dataset.sequences(),
                }
            })
        })
        .transpose()?;
    if expected.len() != config.max_tokens as usize {
        return Err(format!(
            "generated pretraining requires max_tokens={}, matching the corpus sequence; found {}",
            expected.len(),
            config.max_tokens
        ));
    }
    config.corpus_prompt.clear();
    config.corpus_prompt_tokens = None;
    config.episode_dataset = None;
    config.tasks = vec![GenerationTask {
        prompt,
        expected,
        decoys,
    }];
    if config.save_final_checkpoint && config.save_checkpoint.is_none() {
        config.save_checkpoint = Some(output.join("checkpoint.safetensors"));
    }
    config.validate()?;
    run_config(
        run,
        &config,
        output,
        baux::artifact::json::json!({
            "kind":"free_running_corpus_continuation",
            "dataset":dataset_path,
            "initial_sequence_index":sequence_index,
            "task_seed":task_seed,
            "episode":episode,
            "training_conditions":{"schedule":if schedule.is_some() { "repository_stream_full_context" } else { "fixed_episode" },
                "sequence_alignment":crate::pretrain_data::CONTEXT,"origins":schedule.map(|value| value.origins),
                "configured_prompt_tokens":caps.prompt,"configured_output_tokens":caps.output,
                "context_budget":(crate::context::MAX_TOKENS as usize).min(caps.prompt.saturating_add(caps.output)),
                "maximum_context_tokens":crate::context::MAX_TOKENS,
                "paired_candidate_incumbent":!matches!(config.reward, GenerationReward::CausalObjectives) && run.objective_acquisition.is_none()},
            "validation":{"kind":if episode.is_some() { "separate_episode" } else { "separate_validation_corpus" },"tasks":validation_tasks.len(),
                "prompt_tokens":validation_tasks.iter().map(|task| task.prompt.len()).sum::<usize>(),
                "generated_tokens":validation_tasks.iter().map(|task| task.expected.len()).sum::<usize>()},
        }),
        Some(&tokenizer),
        &validation_tasks,
        schedule.map(|value| (value.path, value.caps)),
    )
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
        "history_geometry":run.history_geometry.unwrap_or_default().name(),
        "history_distance_semantics":if run.history_geometry.unwrap_or_default() == crate::config::HistoryGeometry::Latent {
            "latent procedural geometry; independent directions use expected squared norms and omit sample cross terms and weight rounding"
        } else { "realized weight distances" },
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
        "execution_target":"metal_mlx_derived_gemvt_pisa",
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

pub(super) fn validate_initial(
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

fn timed_phase<T>(name: &str, work: impl FnOnce() -> Result<T, String>) -> Result<T, String> {
    eprintln!("experiment.phase name={name} status=start");
    let started = Instant::now();
    let result = work()?;
    eprintln!(
        "experiment.phase name={name} status=done elapsed_ms={:.3}",
        started.elapsed().as_secs_f64() * 1000.0
    );
    Ok(result)
}

fn initialize_search(
    run: &RunConfig,
    config: &GenerationConfig,
    runtime: &Runtime,
    seeds: GenerationSeeds,
) -> Result<
    (
        CandidateWeights,
        SearchState,
        UpdateLog,
        crate::config::ResidentEnnConfig,
    ),
    String,
> {
    let weights = CandidateWeights::initialized(
        runtime,
        seeds.model,
        config.initialization,
        ResidualArchitecture::from_model(run.model.ok_or("generation model is required")?),
    )?;
    if let Some(path) = &config.checkpoint {
        load_checkpoint(&weights, path)?;
    } else {
        eprintln!(
            "[model] initialized {}; no checkpoint loaded",
            run.model.ok_or("generation model is required")?.id()
        );
    }
    let (mut search, updates) = weights.search_shaped(
        run.proposal_kernel(),
        run.length(),
        run.trust_region_shape
            .unwrap_or(crate::config::TrustRegionShape::TensorFamilyStatic),
    )?;
    configure_temperature(&mut search, config)?;
    let policy = run.resident_enn(seeds.acquisition)?;
    search.configure_enn(policy)?;
    search.set_profiling(run.trace());
    if let Some(controller) = run.reliability_controller() {
        search.configure_controller(controller)?;
    }
    Ok((weights, search, updates, policy))
}

fn run_preflight(
    runtime: &Runtime,
    decoder: &decode::Decoder,
    verifier: Option<&block_decode::BlockDecoder>,
    config: &GenerationConfig,
    seeds: GenerationSeeds,
    architecture: ResidualArchitecture,
    output: &Path,
    evaluator: &mut Option<Evaluator>,
    tokenizer: Option<&ByteDecoder>,
) -> Result<(), String> {
    let verifier =
        verifier.ok_or("optimizer comparison preflight requires accepted-prefix verification")?;
    let weights =
        CandidateWeights::initialized(runtime, seeds.model, config.initialization, architecture)?;
    let mut probe = gen_alignment::Probe::new(runtime, &weights, verifier.context)?;
    let evaluation = score(
        runtime,
        decoder,
        Some((verifier, None)),
        weights.row(probe.muon())?,
        config,
        seeds.sampling,
        &output.join("update-alignment-preflight"),
        evaluator.as_mut(),
        tokenizer,
        false,
    )?;
    let nll = evaluation
        .rollouts
        .first()
        .and_then(|rollout| rollout.conditional_target_nll)
        .ok_or("optimizer comparison preflight requires conditional target NLL")?;
    let result = probe.preflight(runtime, verifier, &config.tasks[0], config.max_tokens, nll)?;
    baux::artifact::json::pretty_writer(
        File::create(output.join("update-alignment-preflight.json"))
            .map_err(|error| error.to_string())?,
        &result,
    )
    .map_err(|error| error.to_string())
}

fn build_state(
    run: &RunConfig,
    config: &GenerationConfig,
    output: &Path,
    seeds: GenerationSeeds,
    mut search: SearchState,
    updates: UpdateLog,
    initial: InitialReport,
) -> Result<
    (
        LoopState,
        Vec<f32>,
        Option<Vec<f32>>,
        Option<baux::artifact::json::Value>,
    ),
    String,
> {
    search.observe_objectives(initial.observation.observation)?;
    search.configure_objectives(run.resident_objectives(0)?)?;
    let region_count = search.morbo_regions().max(1);
    let state = LoopState {
        updates,
        journal: initial.journal,
        records: BufWriter::new(
            OpenOptions::new()
                .create_new(true)
                .write(true)
                .open(output.join("optimization.jsonl"))
                .map_err(|e| e.to_string())?,
        ),
        walls: Vec::new(),
        minimum_changed_weights: u64::MAX,
        minimum_generated_tokens: usize::MAX,
        minimum_context_tokens: usize::MAX,
        accepted: 0,
        incumbent_version: crate::hash::splitmix64(seeds.model ^ 0x656e_6e78_2d62_6173),
        incumbent_rollouts: initial.observation.rollouts.clone(),
        regions: RegionState::initial(
            &search,
            seeds.model,
            &initial.observation.rollouts,
            config.temperature,
        ),
        search,
        temperature: config.temperature,
        loop_seconds: 0.0,
        rank_audit: rank_audit(config, &initial.observation)?,
        pool_rank_audited: 0,
        pool_rank_report: None,
        causal_value: initial
            .observation
            .causal
            .map(|evidence| evidence.optimizer_value),
        condition_baselines: vec![None; region_count],
    };
    Ok((
        state,
        initial.validation,
        initial.reference_reward,
        initial.reward_audit,
    ))
}

fn single_alignment(state: &mut LoopState, output: &Path) -> Result<(), String> {
    state.records.flush().map_err(|error| error.to_string())?;
    baux::artifact::json::pretty_writer(
        File::create(output.join("result.json")).map_err(|error| error.to_string())?,
        &baux::artifact::json::json!({
            "schema": "baux.update-alignment-single.v1",
            "status": "completed",
            "rounds": 1,
            "context_tokens": crate::context::MAX_TOKENS,
        }),
    )
    .map_err(|error| error.to_string())
}

pub(super) fn run_config(
    run: &RunConfig,
    config: &GenerationConfig,
    output: &Path,
    objective: baux::artifact::json::Value,
    tokenizer: Option<&ByteDecoder>,
    validation_tasks: &[GenerationTask],
    dataset: Option<(&Path, CorpusCaps)>,
) -> Result<(), String> {
    if let Some(result) = super::muon_train::select(
        run,
        config,
        output,
        &objective,
        tokenizer,
        validation_tasks,
        dataset,
    ) {
        return result;
    }
    let experiment_start = Instant::now();
    verify_policy(config)?;
    let architecture = model_arch(run)?;
    let Resources {
        runtime,
        decoder,
        verifier,
        mut evaluator,
        context,
        validation_tasks,
    } = timed_phase("compile-and-allocate", || {
        prepare_resources(
            config,
            output,
            validation_tasks,
            tokenizer,
            architecture,
            run.scorer_stage_samples,
            dataset.map(|_| crate::context::MAX_TOKENS),
        )
    })?;
    let mut causal: Option<CausalScorer> = None;
    let mut causal_log = causal
        .as_ref()
        .map(|_| CausalLog::new(output))
        .transpose()?;
    let corpus_caps = dataset
        .map(|(_, caps)| (caps.prompt, caps.output))
        .unwrap_or((0, 0));
    let dataset = timed_phase("load-corpus", || {
        dataset
            .map(|(path, _)| crate::pretrain_data::PretrainDataset::load(path))
            .transpose()
    })?;
    let seeds = GenerationSeeds::new(run, config);
    if std::env::var_os("BAUX_UPDATE_ALIGNMENT_PREFLIGHT").is_some() {
        return run_preflight(
            &runtime,
            &decoder,
            verifier.as_ref(),
            config,
            seeds,
            architecture,
            output,
            &mut evaluator,
            tokenizer,
        );
    }
    let (weights, search, updates, policy) = timed_phase("initialize-weights-and-search", || {
        initialize_search(run, config, &runtime, seeds)
    })?;
    write_config(
        run,
        config,
        output,
        &objective,
        &runtime,
        context,
        seeds.model,
        seeds.proposal,
        seeds.acquisition,
        seeds.sampling,
    )?;
    let base = search.base_buffer();
    let initial = timed_phase("initial-score-and-validation", || {
        InitialContext {
            runtime: &runtime,
            decoder: &decoder,
            verifier: verifier.as_ref(),
            weights: &weights,
            base: &base,
            config,
            sampling_seed: seeds.sampling,
            output,
            evaluator: &mut evaluator,
            tokenizer,
            validation_tasks: &validation_tasks,
            causal: causal.as_ref(),
            causal_log: causal_log.as_mut(),
            vector_control: run.objective_acquisition.is_some(),
            experiment_start,
        }
        .run()
    })?;
    let (mut state, initial_validation, reference_reward, reward_audit) =
        build_state(run, config, output, seeds, search, updates, initial)?;
    timed_phase("optimization-loop", || {
        LoopContext {
            run,
            config,
            output,
            runtime: &runtime,
            decoder: &decoder,
            verifier: &verifier,
            weights: &weights,
            sampling_seed: seeds.sampling,
            tokenizer,
            evaluator: &mut evaluator,
            validation_tasks: &validation_tasks,
            dataset: dataset.as_ref(),
            corpus_prompt_cap: corpus_caps.0,
            corpus_output_cap: corpus_caps.1,
            policy,
            proposal_seed: seeds.proposal,
            acquisition_seed: seeds.acquisition,
            causal: &mut causal,
            causal_log: &mut causal_log,
        }
        .run(&mut state)
    })?;
    if std::env::var_os("BAUX_UPDATE_ALIGNMENT_SINGLE").is_some() {
        return single_alignment(&mut state, output);
    }
    FinalContext {
        run,
        config,
        output,
        runtime: &runtime,
        decoder: &decoder,
        verifier: &verifier,
        weights: &weights,
        tokenizer,
        validation_tasks: &validation_tasks,
        evaluator: &mut evaluator,
        causal: &causal,
        causal_log: &mut causal_log,
        seeds,
        experiment_start,
    }
    .run(
        &mut state,
        initial_validation,
        reference_reward,
        reward_audit,
    )
}

pub(super) fn model_arch(run: &RunConfig) -> Result<ResidualArchitecture, String> {
    run.model
        .map(ResidualArchitecture::from_model)
        .ok_or_else(|| "generation model is required".into())
}

fn configure_temperature(
    search: &mut SearchState,
    config: &GenerationConfig,
) -> Result<(), String> {
    match (config.temperature_bounds, config.temperature_step) {
        (Some(bounds), Some(step)) => search.configure_axis(config.temperature, bounds, step),
        (None, None) => Ok(()),
        _ => Err("temperature search configuration was not validated".into()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validation_means() {
        let validation = Validation {
            rewards: vec![0.2, 0.4],
            objectives: vec![vec![-8.0, -6.0, 0.25], vec![-10.0, -8.0, 0.75]],
        };
        assert_eq!(validation.objective_means(), [-9.0, -7.0, 0.5]);
    }

    #[test]
    fn checkpoint_shapes() {
        let layout = checkpoint_layout(ResidualArchitecture::Legacy);
        assert_eq!(layout.len(), 11);
        assert_eq!(
            layout
                .iter()
                .map(|(_, shape)| shape.iter().product::<usize>())
                .sum::<usize>(),
            FULL_PARAMETERS
        );
        let mut names = layout.iter().map(|(name, _)| *name).collect::<Vec<_>>();
        names.sort_unstable();
        names.dedup();
        assert_eq!(names.len(), layout.len());

        let mhc = checkpoint_layout(ResidualArchitecture::Mhc4);
        assert_eq!(mhc.len(), 15);
        assert_eq!(
            mhc.iter()
                .map(|(_, shape)| shape.iter().product::<usize>())
                .sum::<usize>(),
            MHC_FULLPARAMS
        );
    }

    fn evaluated(control: f32, objective: f32) -> Evaluation {
        use crate::objective_observation::{ObjectiveEstimate, ObjectiveObservation};
        let estimate = |mean| ObjectiveEstimate {
            mean,
            variance: 0.0,
        };
        let observation = ObjectiveObservation::new(
            &[estimate(objective), estimate(objective)],
            estimate(control),
        )
        .unwrap();
        Evaluation {
            prompt_tokens: 0,
            rollouts: Vec::new(),
            rewards: vec![900.0],
            mean: control,
            variance: 0.0,
            observation,
            objectives: None,
            reward_seconds: 0.0,
            diagnostics: Vec::new(),
            causal: None,
            paired: None,
        }
    }

    #[test]
    fn vector_routing() {
        metal::objc::rc::autoreleasepool(|| {
            let base = [0x3e80; 17];
            let block =
                crate::resident_metal::ParamBlock::new(71, 0, base.len(), 0.25, 1.0).unwrap();
            let mut search = SearchState::new_implicit(
                &base,
                vec![block],
                3,
                crate::TRLengthConfig::new(0.1, 0.001, 0.4),
                crate::ProposalKernel::Gaussian,
            )
            .unwrap();
            let mut policy = RunConfig::default().resident_enn(19).unwrap();
            policy.ask.surrogate.neighbors = 2;
            search.configure_enn(policy).unwrap();
            search
                .observe_objectives(evaluated(0.0, 0.0).observation)
                .unwrap();
            search
                .configure_objectives(Some(crate::resident_metal::ObjectivePolicy {
                    acquisition: crate::resident_metal::ObjectiveAcquisition::Pareto,
                    scales: vec![1.0, 1.0],
                }))
                .unwrap();
            let (_row, proposal, initializing) = search.propose(29, policy.ask, 0, None).unwrap();
            assert!(initializing);
            let accepted = evaluated(-1.0, 1000.0);
            let decision = search
                .modeled_objectives(&proposal, accepted.observation)
                .unwrap();
            assert!(decision.accepted);
            assert_eq!(search.sync().unwrap(), vec![true]);
            let row = search.objective_observations().unwrap().last().unwrap();
            assert_eq!(row.observation, accepted.observation);
            let mut ask = policy.ask;
            ask.surrogate.neighbors = 2;
            let (_row, proposal, initializing) = search.propose(31, ask, 0, None).unwrap();
            assert!(!initializing);
            let control = proposal.incumbent_mean + 4.0 * proposal.incumbent_standard_error + 1.0;
            let rejected = evaluated(control, -1000.0);
            let decision = search
                .modeled_objectives(&proposal, rejected.observation)
                .unwrap();
            assert!(!decision.accepted);
            assert_eq!(search.sync().unwrap(), vec![false]);
            assert_eq!(
                search.incumbent_objectives().unwrap().unwrap().observation,
                accepted.observation
            );
        });
    }

    #[test]
    fn causal_vector() {
        let mut evaluation = evaluated(2.0, 3.0);
        let observation = evaluation.observation;
        attach_causal(
            &mut evaluation,
            CausalEvidence {
                batch: 0,
                candidate_nll: 9.0,
                incumbent_nll: 9.1,
                improvement: 0.1,
                optimizer_value: -9.0,
                variance: 0.25,
                gpu_seconds: 0.0,
            },
            true,
        )
        .unwrap();
        assert_eq!(evaluation.observation, observation);
        assert_eq!(evaluation.mean, 2.0);
        assert_eq!(evaluation.causal.unwrap().optimizer_value, -9.0);
    }
}
