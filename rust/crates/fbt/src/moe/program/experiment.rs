//! Full-model integration check. Synthetic tokens test execution, not learning.
use super::*;
use baux::artifact::json::{Value, json};
use deser::Deserialize;
use std::{fs, path::Path, time::Instant};

#[derive(Deserialize)]
#[deser(deny_unknown_fields)]
struct Config {
    version: u32,
    experiment: String,
    model: crate::config::ModelArchitecture,
    initialization: crate::config::ModelInitialization,
    seed: u64,
    rounds: u32,
    radius: f64,
    #[deser(default)]
    backend: Backend,
}

#[derive(Deserialize)]
#[deser(deny_unknown_fields)]
struct ForwardConfig {
    version: u32,
    experiment: String,
    context: u32,
    samples: usize,
    profile: bool,
    #[deser(default)]
    backend: Backend,
}

struct Batch {
    tokens: Vec<u32>,
    labels: Vec<u32>,
    mask: Vec<u8>,
    resident: Vec<TensorBuffer>,
}

impl Batch {
    fn new(model: &Resident, seed: u64) -> Result<Self, String> {
        // A reproducible non-language workload, independent of proposal RNG.
        let mut state = seed ^ 0xd1b54a32d192ed03;
        let tokens = (0..model.rows())
            .map(|_| {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                ((state >> 32) % u64::from(VOCAB)) as u32
            })
            .collect::<Vec<_>>();
        let labels = tokens
            .chunks(CONTEXT as usize)
            .flat_map(|sequence| sequence.iter().skip(1).copied().chain(std::iter::once(0)))
            .collect::<Vec<_>>();
        let mask = (0..tokens.len())
            .map(|i| u8::from((i + 1) % CONTEXT as usize != 0))
            .collect::<Vec<_>>();
        let resident = model.batch(&tokens, &labels, &mask)?;
        Ok(Self {
            tokens,
            labels,
            mask,
            resident,
        })
    }
}

// Independently bind ordinary Metal inputs and read ordinary scorer outputs.
// This checks the runtime handoff, not the mathematical correctness of PISA.
fn reference(
    model: &Resident,
    weights: &metal::BufferRef,
    batch: &Batch,
) -> Result<Vec<Vec<f32>>, String> {
    metal::objc::rc::autoreleasepool(|| {
        let mut state = model
            .state
            .lock()
            .map_err(|_| "FBT scorer mutex poisoned")?;
        state.buffers.tokens = state.runtime.buffer_with(&batch.tokens);
        state.buffers.labels = state.runtime.buffer_with(&batch.labels);
        state.buffers.score_mask = state.runtime.buffer_with(&batch.mask);
        let command = state.runtime.queue.new_command_buffer();
        scorer::objective_fused(
            command,
            &state.pipelines,
            &state.tensorops,
            &state.pisa,
            &state.buffers,
            state.layout.view(weights),
        )?;
        complete(command)?;
        Ok(vec![
            crate::apple_gpu::buffer_slice::<f32>(&state.buffers.losses, ROWS as usize)?.to_vec(),
            crate::apple_gpu::buffer_slice::<f32>(&state.buffers.sequence_scores, BATCH as usize)?
                .to_vec(),
        ])
    })
}

fn compare(outputs: &[TensorBuffer], expected: &[Vec<f32>]) -> Result<f32, String> {
    let mut error = 0.0f32;
    if outputs.len() != expected.len() {
        return Err("reference output count differs".into());
    }
    for (buffer, expected) in outputs.iter().zip(expected) {
        let actual = floats(buffer)?;
        if actual.len() != expected.len() {
            return Err("reference output shape differs".into());
        }
        for (a, b) in actual.iter().zip(expected) {
            if !a.is_finite() || !b.is_finite() {
                return Err("nonfinite parity observation".into());
            }
            error = error.max((a - b).abs());
        }
    }
    if error > 1e-5 {
        return Err(format!("native forward parity error {error}"));
    }
    Ok(error)
}

fn policy(radius: f32, seed: u64) -> Result<crate::config::ResidentEnnConfig, String> {
    let mut ask = baux::trials::RoundConfig::default();
    ask.proposal.radius = radius;
    ask.surrogate.neighbors = 1;
    ask.acquisition.seed = seed;
    Ok(crate::config::ResidentEnnConfig {
        pool: baux::procedural_pool::ProceduralPoolConfig::default().resolve_resident()?,
        ask,
        num_candidates: 1,
        num_samples: 1,
        fit_params: false,
        adapt_neighbors: false,
        distance_metric: crate::config::Dissimilarity::Euclidean,
        local_scale_neighbors: 8,
    })
}

fn round(
    model: &mut Resident,
    batch: &Batch,
    seed: u64,
    ask: baux::trials::RoundConfig,
) -> Result<(Value, Vec<TensorBuffer>), String> {
    let started = Instant::now();
    let (weights, proposal, initializing) = model.search.propose(seed, ask, 0, None)?;
    let proposal_seconds = started.elapsed().as_secs_f64();
    let changed: u64 = model.search.describe(&proposal)?[0]
        .3
        .iter()
        .map(|value| value.0)
        .sum();
    if changed == 0 {
        return Err("proposal did not change any model parameters".into());
    }
    let input = model.search.program_candidate(&model.forward, &proposal)?;
    let (handle, offset) = unsafe { input.metal()? };
    if handle != weights.as_ptr().cast() || offset != 0 {
        return Err("parameter import copied or shifted the row".into());
    }
    // Hold an explicit alias so this check cannot race with fast completion.
    let alias = input.clone();
    let pending = model.submit(input, &batch.resident)?;
    for error in [
        model.search.cancel_round(&proposal).err(),
        model.search.tell_modeled(&proposal, 0.0, 0.0).err(),
    ] {
        if !error.is_some_and(|error| error.contains("still leased")) {
            return Err("parameter mutation did not reject the active lease".into());
        }
    }
    drop(alias);
    let outputs = pending.wait()?;
    let observation = model.observe(&outputs, &batch.mask)?;
    let forward_seconds = started.elapsed().as_secs_f64() - proposal_seconds;
    let gpu_seconds = model.gpu_seconds()?;
    let tell = Instant::now();
    let decision = if initializing {
        model
            .search
            .tell_initial(&proposal, observation.reward, observation.variance)?
    } else {
        model
            .search
            .tell_modeled(&proposal, observation.reward, observation.variance)?
    };
    if model.search.sync()? != [decision.accepted] {
        return Err("resident decision failed to settle".into());
    }
    let tell_seconds = tell.elapsed().as_secs_f64();
    let round_seconds = started.elapsed().as_secs_f64();
    // Audit only after the complete uninterrupted optimizer round. Settlement
    // rotates retained allocations; it does not modify this candidate's bytes.
    let audit = Instant::now();
    let error = compare(&outputs, &reference(model, &weights, batch)?)?;
    let audit_seconds = audit.elapsed().as_secs_f64();
    Ok((
        json!({"seed": seed, "initializing": initializing, "changed_parameters": changed,
        "selected": proposal.index, "predicted_mean": proposal.predicted_mean,
        "reward": observation.reward, "variance": observation.variance,
        "scored_tokens": observation.scored_tokens, "sequence_nlls": observation.sequence_nlls,
        "accepted": decision.accepted, "numerical": decision.numerical.label(),
        "history": model.search.history(), "forward_gpu_seconds": gpu_seconds,
        "proposal_seconds": proposal_seconds, "forward_wall_seconds": forward_seconds,
        "tell_seconds": tell_seconds, "round_seconds": round_seconds,
        "audit_seconds": audit_seconds, "parity_max_abs": error,
        "parameter_allocation_identical": true, "lease_rejected_mutation": true}),
        outputs,
    ))
}

pub fn run(path: &Path, output: &Path) -> Result<(), String> {
    let source = fs::read_to_string(path).map_err(|e| e.to_string())?;
    let value: Value = baux::artifact::toml::from_str(&source).map_err(|e| e.to_string())?;
    if value["experiment"].as_str() == Some("resident-forward") {
        let config: ForwardConfig =
            baux::artifact::toml::from_str(&source).map_err(|e| e.to_string())?;
        if config.version != 1 || config.samples == 0 {
            return Err("resident-forward requires version 1 and positive samples".into());
        }
        if config.experiment != "resident-forward" {
            return Err("invalid resident-forward experiment".into());
        }
        let report = crate::moe::forward_probe::run_backend(
            config.context,
            config.samples,
            config.profile,
            &output.join("forward.json"),
            config.backend,
        )?;
        let bytes = baux::artifact::json::pretty_vec(&report).map_err(|e| e.to_string())?;
        fs::write(output.join("forward.json"), &bytes).map_err(|e| e.to_string())?;
        fs::write(output.join("result.json"), bytes).map_err(|e| e.to_string())?;
        let mut stream =
            fs::File::create(output.join("rounds.jsonl")).map_err(|e| e.to_string())?;
        for measurement in report["measurements"]
            .as_seq()
            .ok_or("resident-forward measurements are missing")?
        {
            baux::artifact::json::write_line(&mut stream, measurement)
                .map_err(|e| e.to_string())?;
        }
        return Ok(());
    }
    let config: Config = baux::artifact::toml::from_str(&source).map_err(|e| e.to_string())?;
    if config.version != 1 || config.experiment != "resident-program" || config.rounds == 0 {
        return Err("resident-program requires version 1 and positive rounds".into());
    }
    let length = crate::trust_region::TRLengthConfig::try_new(
        config.radius,
        config.radius / 128.0,
        config.radius * 2.0,
    )
    .map_err(|e| e.to_string())?;
    let mut model = Resident::with_backend(
        config.model,
        config.initialization,
        config.seed,
        length,
        crate::ProposalDistribution::Gaussian,
        config.backend,
    )?;
    let policy = policy(config.radius as f32, config.seed)?;
    model.search.configure_enn(policy)?;
    let batch = Batch::new(&model, config.seed)?;
    let baseline = model
        .submit(model.search.program_base(&model.forward)?, &batch.resident)?
        .wait()?;
    let initial = model.observe(&baseline, &batch.mask)?;
    let saved = baseline
        .iter()
        .map(TensorBuffer::read)
        .collect::<Result<Vec<_>, _>>()?;
    model
        .search
        .observe_initial(initial.reward, initial.variance)?;
    let mut records = Vec::new();
    let mut stream = fs::File::create(output.join("rounds.jsonl")).map_err(|e| e.to_string())?;
    for index in 0..config.rounds {
        let (mut record, _outputs) = round(
            &mut model,
            &batch,
            config.seed.wrapping_add(u64::from(index) + 1),
            policy.ask,
        )?;
        record["round"] = (index + 1).into();
        record["mean_nll"] = (-record["reward"].as_f64().ok_or("missing observed reward")?).into();
        record["wall_ms"] = (record["round_seconds"]
            .as_f64()
            .ok_or("missing round time")?
            * 1000.0)
            .into();
        baux::artifact::json::write_line(&mut stream, &record).map_err(|e| e.to_string())?;
        eprintln!(
            "resident-program round={} reward={:?} seconds={:?}",
            index + 1,
            record["reward"],
            record["round_seconds"]
        );
        records.push(record);
    }
    if baseline
        .iter()
        .map(TensorBuffer::read)
        .collect::<Result<Vec<_>, _>>()?
        != saved
    {
        return Err("a subsequent round overwrote published observations".into());
    }
    let report = json!({"schema": "baux.resident_program.v1", "model": config.model.id(),
        "runtime": if config.backend == Backend::Tvm { "tvm-relax" } else { "native" },
        "kernels": "production-metal",
        "parameters": model.parameters, "parameter_dtype": "float16", "device": Runtime::shared()?.device.name(),
        "batch": BATCH, "context": CONTEXT, "data": "deterministic synthetic vocabulary IDs; no learning-quality claim",
        "initial_reward": initial.reward, "optimizer_evaluations": config.rounds + 1,
        "audit_evaluations": config.rounds, "round_parameter_readbacks": 0,
        "timing": "continuous proposal through settlement; audit follows each completed round; initialization excluded",
        "old_observations_preserved": true, "rounds": records});
    let bytes = baux::artifact::json::pretty_vec(&report).map_err(|e| e.to_string())?;
    fs::write(output.join("program.json"), bytes).map_err(|e| e.to_string())
}
