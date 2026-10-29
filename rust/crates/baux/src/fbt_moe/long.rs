//! Fixed-corpus long-context pretraining for the mHC4 forward program.
//!
//! The objective is corpus next-token cross-entropy, evaluated under a
//! single-candidate proposal pipeline. In each round:
//! 1. A candidate row is proposed by the optimizer search state
//!    (`search.propose`).
//! 2. A corpus batch window is selected from the pretraining dataset.
//! 3. The candidate model is evaluated on this window (`score_batch`),
//!    obtaining the negative NLL (in nats/token), variance, codelength (in
//!    bits), and bits_per_token.
//! 4. Full window coverage is enforced via verification before reporting
//!    metrics.
//! 5. The optimizer accepts or rejects the candidate based on modeled
//!    evaluation, synchronizes the search state, and commits parameter updates.
//!
//! # Batch Metrics and Units
//!
//! - **Negative NLL (`reward`)**: Negative log-likelihood $-\log p(x)$ per
//!   token, measured in natural units (`nats/token`). Higher values represent
//!   better predictions (lower cross-entropy loss).
//! - **Codelength (`bits` / `codelength_bits`)**: Total Shannon codelength for
//!   the evaluated window, measured in `bits` ($\text{codelength} =
//!   \text{mean\_nll} \times \text{tokens} \times \log_2(e)$).
//! - **Bits per token (`bits_per_token`)**: Window codelength normalized by
//!   scored tokens ($\text{bits} / \text{tokens}$), measured in `bits/token`.
//! - **Variance (`variance`)**: Estimated variance of the mean cross-entropy
//!   metric across the window, provided as telemetry from the scorer.
//!
//! Validation scores the current incumbent on fixed per-source windows without
//! contributing observations or acceptance decisions. Training window scores,
//! modeled incumbent values, and held-out mixture means are distinct
//! quantities. This module does not establish calibration, regret, or a latency
//! guarantee.
//!
//! Per-round wall time excludes validation and initial setup. `loop_seconds`
//! includes interleaved validation; GPU round time covers single-candidate
//! scoring, not proposal kernels. Artifact serialization is performed by the
//! worker.

use super::*;

/// One evaluated corpus window batch metrics:
/// - `reward`: Negative NLL in nats/token ($-\log p(x)$), higher is better.
/// - `variance`: Estimated mean variance across the scored window tokens
///   (telemetry).
/// - `bits`: Total window Shannon codelength in bits ($-\sum \log_2 p(x)$).
/// - `tokens`: Actual number of scored causal next-token positions.
/// - `gpu_seconds`: Total scorer GPU execution time in seconds (prefill + chunk
///   scoring).
struct BatchScore {
    reward: f32,
    variance: f32,
    bits: f64,
    tokens: usize,
    gpu_seconds: f64,
    observation: crate::objective_observation::ObjectiveObservation,
}

/// Run the mHC4 causal scorer with supplied weights and corpus tokens.
/// Converts mean NLL to higher-is-better reward without changing parameter
/// storage or search state. Mutates scorer/decoder scratch; the caller clears
/// stale paired-loss state when no paired measurement is intended.
fn score_batch(
    scorer: &block_decode::BlockDecoder,
    runtime: &Runtime,
    decoder: &decode::Decoder,
    weights: CandidateRow<'_>,
    batch: &crate::pretrain_data::CausalBatch,
    objective: crate::config::ObjectiveFunction,
) -> Result<BatchScore, String> {
    let prefill = scorer.prefill_prefix(runtime, decoder, weights, &batch.tokens)?;
    let result = scorer.score_objective(runtime, decoder, weights, batch, objective)?;
    Ok(BatchScore {
        reward: -result.quality.mean_nll,
        variance: result.quality.mean_nll_variance,
        bits: result.quality.codelength_bits,
        tokens: result.quality.scored_tokens,
        gpu_seconds: prefill + result.gpu_seconds,
        observation: result.observation,
    })
}

/// Score a fixed window per validation source and weight source means.
/// The seed is reused independently of `round`, which is only a record label.
/// Requires an evaluation `CorpusSet`; it does not send observations to ENN.
/// `scored_tokens` in the record is the requested window size, whereas
/// `evaluated_tokens` sums actual scored positions over sources. These counts
/// can differ because of boundary context and multiple source evaluations.
fn validate(
    scorer: &block_decode::BlockDecoder,
    runtime: &Runtime,
    decoder: &decode::Decoder,
    weights: &CandidateWeights,
    row: &BufferRef,
    dataset: &crate::pretrain_data::CorpusSet,
    window_tokens: usize,
    seed: u64,
    round: u32,
) -> Result<baux::artifact::json::Value, String> {
    scorer.clear_pair();
    let mut nll = 0.0f64;
    let mut bits = 0.0f64;
    let mut gpu_seconds = 0.0f64;
    let mut evaluated_tokens = 0usize;
    let mut sources = Vec::new();
    for batch in dataset.causal_batches(window_tokens, seed)? {
        let score = score_batch(
            scorer,
            runtime,
            decoder,
            weights.row(row)?,
            &batch.batch,
            crate::config::ObjectiveFunction::CrossEntropy,
        )?;
        let weight = f64::from(batch.weight);
        nll += f64::from(-score.reward) * weight;
        bits += score.bits / score.tokens as f64 * weight;
        gpu_seconds += score.gpu_seconds;
        evaluated_tokens += score.tokens;
        sources.push(baux::artifact::json::json!({
            "source":batch.source,
            "weight":batch.weight,
            "nll":-score.reward,
            "bits_per_token":score.bits / score.tokens as f64,
            "corpus_offset":batch.batch.offset,
            "scored_tokens":score.tokens,
            "gpu_seconds":score.gpu_seconds,
        }));
    }
    let total = dataset.total_weight() as f64;
    Ok(baux::artifact::json::json!({
        "round":round,
        "nll":nll / total,
        "bits_per_token":bits / total,
        "scored_tokens":window_tokens,
        "evaluated_tokens":evaluated_tokens,
        "gpu_seconds":gpu_seconds,
        "used_for_acceptance":false,
        "scope":"mixture-weighted-causal-likelihood",
        "sources":sources,
    }))
}

/// Print a completed validation record without measuring or updating a model.
/// Missing reporting fields fall back to zero/NaN; this display adapter is not
/// schema validation and never makes an acceptance decision.
fn emit_validation(record: &baux::artifact::json::Value) {
    let sources = record["sources"]
        .as_seq()
        .map(|sources| {
            sources
                .iter()
                .map(|source| {
                    format!(
                        "{}:{:.9}",
                        source["source"].as_str().unwrap_or("unknown"),
                        source["nll"].as_f64().unwrap_or(f64::NAN),
                    )
                })
                .collect::<Vec<_>>()
                .join(",")
        })
        .unwrap_or_default();
    eprintln!(
        "pretrain.validation round={} nll={:.9} bits_per_token={:.9} scored_tokens={} \
         evaluated_tokens={} sources={sources} used_for_acceptance=false",
        record["round"].as_u64().unwrap_or(0),
        record["nll"].as_f64().unwrap_or(f64::NAN),
        record["bits_per_token"].as_f64().unwrap_or(f64::NAN),
        record["scored_tokens"].as_u64().unwrap_or(0),
        record["evaluated_tokens"].as_u64().unwrap_or(0),
    );
}

struct LongState {
    window_tokens: usize,
    architecture: ResidualArchitecture,
    dataset: crate::pretrain_data::CorpusSet,
    training_batch: crate::pretrain_data::CorpusBatch,
    validation: Option<crate::pretrain_data::CorpusSet>,
    runtime: std::sync::Arc<Runtime>,
    decoder: decode::Decoder,
    scorer: block_decode::BlockDecoder,
    weights: CandidateWeights,
    search: SearchState,
    updates: UpdateLog,
    trust_region: baux::semantics::optimizer::TrustRegionController,
    surrogate: baux::semantics::optimizer::Surrogate,
    incumbent_nll: f32,
    accepted: u32,
    validated_accepts: u32,
    cumulative_tokens: usize,
    cumulative_wall: f64,
    cumulative_gpu: f64,
    validation_records: Vec<baux::artifact::json::Value>,
}

fn init_surrogate(
    run: &crate::config::RunConfig,
    policy: &crate::config::ResidentEnnConfig,
    search: &mut SearchState,
    initial: BatchScore,
) -> Result<
    (
        baux::semantics::optimizer::TrustRegionController,
        baux::semantics::optimizer::Surrogate,
    ),
    String,
> {
    search.observe_objectives(initial.observation)?;
    let ctrl_info = search.controller_info()?;
    let trust_region = baux::semantics::optimizer::TrustRegionController::policy_f64(
        ctrl_info.length,
        ctrl_info.length_min,
        ctrl_info.length_max,
        8,
        run.reliability_controller(),
    )?;
    let distance_metric = match policy.distance_scaling {
        crate::config::DistanceScaling::SelfTuning => {
            baux::semantics::neighbor::DistanceMetric::SelfTuning {
                local_k: policy.local_scale_neighbors.max(1),
            }
        }
        crate::config::DistanceScaling::Hyperbolic => {
            baux::semantics::neighbor::DistanceMetric::Hyperbolic {
                local_k: policy.local_scale_neighbors.max(1),
            }
        }
        crate::config::DistanceScaling::Global => {
            baux::semantics::neighbor::DistanceMetric::Euclidean
        }
    };
    let surrogate = baux::semantics::optimizer::Surrogate::with_spec(
        initial.reward,
        initial.variance,
        run.acquisition_seed(),
        policy.surrogate_spec()?,
    )?
    .with_metric(distance_metric);
    Ok((trust_region, surrogate))
}

fn setup_state(
    run: &crate::config::RunConfig,
    dataset_path: &std::path::Path,
) -> Result<LongState, String> {
    let window_tokens =
        run.window_tokens
            .ok_or("long pretraining requires objective.window-tokens")? as usize;
    let architecture =
        ResidualArchitecture::from_model(run.model.ok_or("long pretraining requires a model")?);
    if !architecture.is_mhc4() {
        return Err("long pretraining requires fbt-pisa1-mhc4-v1".into());
    }
    let dataset = crate::pretrain_data::CorpusSet::load(dataset_path)?;
    let validation = run
        .validation_dataset
        .as_deref()
        .map(crate::pretrain_data::CorpusSet::load_eval)
        .transpose()?;
    let runtime = Runtime::shared()?;
    let core_rows = window_tokens
        .div_ceil(architecture.patch_size())
        .max(window_tokens)
        .next_power_of_two()
        .max(CONTEXT as usize);
    let context = u32::try_from(core_rows).map_err(|error| error.to_string())?;
    let decoder = decode::Decoder::with_visits(
        &runtime,
        context,
        context > ROWS,
        architecture.layer_steps().len(),
    )?;
    let mut scorer =
        block_decode::BlockDecoder::with_architecture(&runtime, context, architecture)?;
    scorer.profile(&runtime, run.scorer_stage_samples)?;
    let weights = CandidateWeights::initialized(
        &runtime,
        run.model_seed(),
        crate::config::ModelInitialization::FbtPisa1Mhc4,
        architecture,
    )?;
    let shape = run
        .trust_region_shape
        .unwrap_or(crate::config::TrustRegionShape::TensorFamilyStatic);
    let (mut search, updates) =
        weights.search_shaped(run.proposal_kernel(), run.length(), shape)?;
    let policy = run.resident_enn(run.acquisition_seed())?;
    search.configure_enn(policy)?;
    if let Some(controller) = run.reliability_controller() {
        search.configure_controller(controller)?;
    }
    search.set_profiling(run.trace());

    let initial_batch = dataset.causal_batch(window_tokens, run.reference_seed(), 0)?;
    scorer.clear_pair();
    let initial = score_batch(
        &scorer,
        &runtime,
        &decoder,
        weights.row(&search.base_buffer())?,
        &initial_batch.batch,
        run.objective_function(),
    )?;
    let incumbent_nll = -initial.reward;
    eprintln!(
        "pretrain.initial source={} window_tokens={window_tokens} scored_tokens={} \
         corpus_offset={} nll={:.9} bits_per_token={:.9} gpu_ms={:.3}",
        initial_batch.source,
        initial.tokens,
        initial_batch.batch.offset,
        incumbent_nll,
        initial.bits / initial.tokens as f64,
        initial.gpu_seconds * 1000.0,
    );
    let (trust_region, surrogate) = init_surrogate(run, &policy, &mut search, initial)?;
    let mut validation_records = Vec::new();
    if let Some(validation) = validation.as_ref()
        && run.validation_interval == Some(0)
    {
        let record = validate(
            &scorer,
            &runtime,
            &decoder,
            &weights,
            &search.base_buffer(),
            validation,
            window_tokens,
            run.reference_seed(),
            0,
        )?;
        emit_validation(&record);
        validation_records.push(record);
    }
    Ok(LongState {
        window_tokens,
        architecture,
        dataset,
        training_batch: initial_batch.clone(),
        validation,
        runtime,
        decoder,
        scorer,
        weights,
        search,
        updates,
        trust_region,
        surrogate,
        incumbent_nll,
        accepted: 0,
        validated_accepts: 0,
        cumulative_tokens: 0,
        cumulative_wall: 0.0,
        cumulative_gpu: 0.0,
        validation_records,
    })
}

struct RoundProposal {
    geometry: Vec<Vec<f64>>,
    selection: baux::semantics::optimizer::PoolSelection,
    row: Buffer,
    proposal: Proposals,
    ask_seconds: f64,
}

impl LongState {
    fn propose_round(
        &mut self,
        run: &crate::config::RunConfig,
        step: u32,
    ) -> Result<RoundProposal, String> {
        let prop_seed = run
            .proposal_seed()
            .checked_add(u64::from(step))
            .ok_or("proposal seed overflow")?;
        let ask_start = Instant::now();
        let geometry = self.search.sample_distances(prop_seed)?;
        if geometry.len() != 1 {
            return Err("operator-bo requires a single-candidate proposal geometry".into());
        }
        let selection = self.surrogate.evaluate_candidate(&geometry[0])?;
        let (row, proposal) = self.search.materialize_selection(prop_seed, &selection)?;
        Ok(RoundProposal {
            geometry,
            selection,
            row,
            proposal,
            ask_seconds: ask_start.elapsed().as_secs_f64(),
        })
    }

    fn tell_round(
        &mut self,
        step: u32,
        proposed: &RoundProposal,
        candidate: &BatchScore,
    ) -> Result<(baux::semantics::optimizer::ReliabilityDecision, u64, f64), String> {
        let changes = self.search.describe(&proposed.proposal)?.remove(0).3;
        let tell_start = Instant::now();
        let decision = self.trust_region.evaluate_selection(
            candidate.reward,
            0.0,
            proposed.proposal.length,
            &proposed.selection,
        )?;
        let target_slot = self.surrogate.append(
            &proposed.geometry[proposed.selection.selected],
            candidate.reward,
            candidate.variance,
            decision.accepted,
        )?;
        let next_radius = self.trust_region.radius_f64();
        self.search.commit_host(
            &proposed.proposal,
            target_slot,
            candidate.observation,
            decision.accepted,
            next_radius,
        )?;
        if self.search.sync()? != vec![decision.accepted] {
            return Err("long pretraining acceptance synchronization failed".into());
        }
        let tell_seconds = tell_start.elapsed().as_secs_f64();
        self.accepted += u32::from(decision.accepted);
        let changed_coords: u64 = changes.iter().map(|(count, _)| *count).sum();
        self.updates.push_scaled(
            step + 1,
            proposed.proposal.seed,
            proposed.proposal.length,
            decision.accepted,
            &changes,
            &proposed.proposal.block_scales,
        )?;
        if decision.accepted {
            self.incumbent_nll = -candidate.reward;
        }
        Ok((decision, changed_coords, tell_seconds))
    }

    fn maybe_validate(&mut self, run: &crate::config::RunConfig, round: u32) -> Result<(), String> {
        let Some(validation) = self.validation.as_ref() else {
            return Ok(());
        };
        if round % run.validation_interval.unwrap_or(run.rounds()) != 0 && round != run.rounds() {
            return Ok(());
        }
        let record = if self.accepted == self.validated_accepts
            && let Some(prev) = self.validation_records.last()
        {
            baux::artifact::json::json!({
                "round":round,
                "nll":prev["nll"].clone(),
                "bits_per_token":prev["bits_per_token"].clone(),
                "scored_tokens":prev["scored_tokens"].clone(),
                "evaluated_tokens":prev["evaluated_tokens"].clone(),
                "gpu_seconds":0.0,
                "used_for_acceptance":false,
                "scope":"mixture-weighted-causal-likelihood",
                "sources":prev["sources"].clone(),
            })
        } else {
            self.validated_accepts = self.accepted;
            validate(
                &self.scorer,
                &self.runtime,
                &self.decoder,
                &self.weights,
                &self.search.base_buffer(),
                validation,
                self.window_tokens,
                run.reference_seed(),
                round,
            )?
        };
        emit_validation(&record);
        self.validation_records.push(record);
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn record_round(
        &self,
        step: u32,
        total_rounds: u32,
        tell_seconds: f64,
        wall_seconds: f64,
        changed_coords: u64,
        next_radius: f64,
        proposed: &RoundProposal,
        batch: &crate::pretrain_data::CorpusBatch,
        candidate: &BatchScore,
        decision: &baux::semantics::optimizer::ReliabilityDecision,
    ) -> baux::artifact::json::Value {
        let gain = decision.improvement as f32;
        let threshold = decision.threshold as f32;
        let predicted_gain = decision.predicted_improvement as f32;
        eprintln!(
            "pretrain.round round={} total={} accepted={} nll={:.6} gain={:+.6} pred_gain={:+.6} \
             proposal_radius={:.6} next_radius={:.6} scored_tokens={} gpu_ms={:.1} wall_ms={:.1}",
            step + 1,
            total_rounds,
            decision.accepted,
            -candidate.reward,
            gain,
            predicted_gain,
            proposed.proposal.length,
            next_radius,
            candidate.tokens,
            candidate.gpu_seconds * 1000.0,
            wall_seconds * 1000.0,
        );
        baux::artifact::json::json!({
            "round": step + 1,
            "initializing": proposed.selection.initializing,
            "transported": false,
            "selected_slot": proposed.selection.selected,
            "accepted": decision.accepted,
            "numerical": decision.numerical.label(),
            "candidate_nll": -candidate.reward,
            "incumbent_nll": self.incumbent_nll,
            "gain": gain,
            "threshold": threshold,
            "predicted_gain": predicted_gain,
            "candidate_se": proposed.selection.candidate.standard_error,
            "candidate_neighbors": proposed.selection.candidate.neighbors,
            "incumbent_neighbors": proposed.selection.incumbent.neighbors,
            "optimizer_value": candidate.reward,
            "optimizer_variance": candidate.variance,
            "scored_tokens": candidate.tokens,
            "cumulative_tokens": self.cumulative_tokens,
            "codelength_bits": candidate.bits,
            "bits_per_token": candidate.bits / candidate.tokens as f64,
            "source": &batch.source,
            "corpus_offset": batch.batch.offset,
            "window_tokens": self.window_tokens,
            "objective": "fixed-corpus-causal-likelihood",
            "candidate_gpu_ms": candidate.gpu_seconds * 1000.0,
            "base_gpu_ms": 0.0,
            "round_gpu_ms": candidate.gpu_seconds * 1000.0,
            "round_wall_ms": wall_seconds * 1000.0,
            "cumulative_wall_seconds": self.cumulative_wall,
            "cumulative_gpu_seconds": self.cumulative_gpu,
            "ask_ms": proposed.ask_seconds * 1000.0,
            "tell_ms": tell_seconds * 1000.0,
            "proposal_radius": proposed.proposal.length,
            "next_radius": next_radius,
            "changed_coords": changed_coords,
        })
    }
}

/// Execute the fixed-corpus Metal/mHC4 experiment and return measurements.
/// Requires a validated nonzero-round configuration, window size, supported
/// mHC4 architecture, and loadable corpora. Initializes weights and optimizer
/// search; each round proposes one candidate via `propose`, evaluates it on the
/// sampled corpus window with `score_batch`, informs the optimizer
/// (`tell_initial` / `tell_modeled`), and commits accepted updates via `sync`.
///
/// Batch metrics are computed in:
/// - Negative NLL: `nats/token` ($-\text{mean\_nll}$)
/// - Codelength: `bits`
/// - Bits per token: `bits / scored_tokens`
///
/// Structured lowercase dotted event lines (`pretrain.initial`,
/// `pretrain.round`, `pretrain.validation`) log full window coverage and
/// telemetry. Optional validation is held out of optimizer decisions. Returned
/// timing samples use the scopes in the module contract; no learned checkpoint
/// is saved here. Errors propagate rather than starting a replacement
/// experiment or rolling back completed rounds. Successful execution does not
/// imply successful learning.
pub(super) fn run(
    run: &crate::config::RunConfig,
    dataset_path: &std::path::Path,
) -> Result<ActualBoResult, String> {
    let mut state = setup_state(run, dataset_path)?;
    let training_batch = state.training_batch.clone();
    let started = Instant::now();
    let mut wall = Vec::with_capacity(run.rounds() as usize);
    let mut gpu = Vec::with_capacity(run.rounds() as usize);
    let mut optimizer_seconds = Vec::with_capacity(run.rounds() as usize);
    let mut optimization_records = Vec::with_capacity(run.rounds() as usize);
    for step in 0..run.rounds() {
        let round_start = Instant::now();
        let proposed = state.propose_round(run, step)?;
        let batch = &training_batch;
        state.scorer.clear_pair();
        let candidate = score_batch(
            &state.scorer,
            &state.runtime,
            &state.decoder,
            state.weights.row(&proposed.row)?,
            &batch.batch,
            run.objective_function(),
        )?;
        state.cumulative_tokens += candidate.tokens;

        let (decision, changed_coords, tell_seconds) =
            state.tell_round(step, &proposed, &candidate)?;
        let wall_seconds = round_start.elapsed().as_secs_f64();
        wall.push(wall_seconds);
        gpu.push(candidate.gpu_seconds);
        state.cumulative_wall += wall_seconds;
        state.cumulative_gpu += candidate.gpu_seconds;
        let next_radius = state.search.length()?;
        optimizer_seconds.push(proposed.ask_seconds + tell_seconds);
        optimization_records.push(state.record_round(
            step,
            run.rounds(),
            tell_seconds,
            wall_seconds,
            changed_coords,
            next_radius,
            &proposed,
            &batch,
            &candidate,
            &decision,
        ));
        state.maybe_validate(run, step + 1)?;
    }
    let loop_seconds = started.elapsed().as_secs_f64();
    state
        .weights
        .finalize_updates(&state.search.base_buffer(), &mut state.updates)?;
    wall.sort_by(f64::total_cmp);
    gpu.sort_by(f64::total_cmp);
    Ok(ActualBoResult {
        parameters: state.architecture.parameter_count(),
        loop_seconds,
        median_wall_seconds: wall[wall.len() / 2],
        min_wall_seconds: wall[0],
        max_wall_seconds: wall[wall.len() - 1],
        median_gpu_seconds: gpu[gpu.len() / 2],
        accepted: state.accepted,
        learning_seconds: loop_seconds,
        validation: state.validation_records,
        optimizer_seconds,
        optimization_records,
        updates: state.updates,
    })
}
