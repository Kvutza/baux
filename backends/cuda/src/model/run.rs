use super::VOCAB;
pub use super::bo_run::mhc4_bo;
use crate::model::{FbtModel, GenerationOutput, ModelOutput};
use crate::{CudaResult, cuda_error, read_prefix, target_name};
use baux::artifact::json::{Value, json};
use baux_cuda_device::{ObjectiveShape, ScoreStats};
use cuda_core::{DeviceBuffer, LaunchConfig1D};
use half::f16;
use std::fs;
use std::path::Path;

fn initialized_trace(model: &mut FbtModel) -> CudaResult<baux::artifact::forward::ForwardTrace> {
    use baux::artifact::forward::ForwardTrace;
    use baux::semantics::mhc4::probe::initialized as case;
    let weights = model.weights.logical_hashes(&model.stream)?;
    let sequence = case::sequence();
    let batch = baux::artifact::pretrain::CausalBatch {
        tokens: sequence[..case::TOKENS].to_vec(),
        labels: sequence[1..].to_vec(),
        loss_mask: vec![1; case::TOKENS],
        offset: 0,
    };
    model.configure_linear(&batch)?;
    model.evaluate_mhc4(2, 0.0, baux::semantics::random::splitmix64(0), false)?;
    let state = model
        .mhc4
        .as_ref()
        .ok_or("CUDA initialized trace lost its state")?;
    let trace = ForwardTrace {
        format: ForwardTrace::FORMAT.into(),
        seed: case::SEED,
        input: batch.tokens,
        labels: batch.labels,
        rows: case::ROWS,
        width: 512,
        weights,
        encoded: read_prefix(&model.scratch.embedded, &model.stream, case::WORDS)?,
        hidden: read_prefix(&model.scratch.normalized, &model.stream, case::WORDS)?,
        predictions: read_prefix(&state.output, &model.stream, case::TARGETS)?,
        loss_bits: read_prefix(&state.losses, &model.stream, case::TARGETS)?
            .iter()
            .map(|v| v.to_bits())
            .collect(),
    };
    trace.validate()?;
    Ok(trace)
}

/// Finite lifecycle fixture through the production encoder, core, readout,
/// loss capture and graph replay. Uniform fixture weights have uniform
/// target likelihood; this is not an optimizer or learning experiment.
fn check_causal(model: &mut FbtModel) -> CudaResult<()> {
    model.weights.mhc4 = Some(super::weights::Mhc4Weights {
        mask: DeviceBuffer::zeroed(&model.stream, 512).map_err(cuda_error)?,
        positions: DeviceBuffer::zeroed(&model.stream, 64 * 512).map_err(cuda_error)?,
    });
    for (index, length) in [128, 128, 64].into_iter().enumerate() {
        let mut batch = baux::artifact::pretrain::CausalBatch {
            tokens: vec![index as u32; length],
            labels: vec![u32::from(index != 0); length],
            loss_mask: vec![1; length],
            offset: index * length,
        };
        model.configure_causal(&batch)?;
        if index == 1
            && model
                .mhc4
                .as_ref()
                .is_none_or(|s| s.generation_graph.is_none())
        {
            return Err("same-shape corpus switch discarded its readout graph".into());
        }
        let first = model.evaluate_mhc4(2, 0.0, 17, false)?;
        let estimate = first.estimate.ok_or("causal fixture omitted estimate")?;
        let stats = first.stats.ok_or("causal fixture omitted counts")?;
        let targets = length - 63;
        if estimate.improvement.is_some()
            || first.patch_gains.is_some()
            || stats.tokens as usize != targets
            || stats.mismatches as usize != if index == 0 { 0 } else { targets }
            || (estimate.mean - (VOCAB as f32).ln()).abs() > 0.001
        {
            return Err(format!(
                "causal fixture differs at condition {index}: mean={} tokens={} mismatches={}",
                estimate.mean, stats.tokens, stats.mismatches
            ));
        }
        model.capture_pair()?;
        let repeat = model.evaluate_mhc4(2, 0.0, 17, false)?;
        let paired = repeat.estimate.ok_or("causal repeat omitted estimate")?;
        if paired.mean.to_bits() != estimate.mean.to_bits()
            || paired.improvement != Some(0.0)
            || paired.paired_variance != Some(0.0)
            || repeat
                .patch_gains
                .as_ref()
                .is_none_or(|v| v.len() != targets.div_ceil(64) || v.iter().any(|&v| v != 0.0))
        {
            return Err("causal repeated condition changed or lost token-aligned capture".into());
        }
        model.clear_pair()?;
        batch.loss_mask[0] = 0;
        if model.configure_causal(&batch).is_ok() {
            return Err("causal fixture accepted an unsupported loss mask".into());
        }
    }
    println!(
        "MHC4_CAUSAL_PARITY ok=true targets=65,65,1 graph_reused=true labels_reloaded=true \
         capture_reset=true identical_pair=exact"
    );
    Ok(())
}

pub fn check() -> CudaResult<()> {
    crate::precision::check()?;
    crate::mhc4_probe::check()?;
    super::routing::check()?;
    let mut model = FbtModel::fixture()?;
    score_check(&model)?;
    mhc4_check(&model)?;
    let tokens = vec![0; 4096];
    for (visits, value) in [(1, 0x4600), (2, 0x4800), (4, 0x4a00)] {
        let result = model.run(&tokens, visits, 0.0, 17)?;
        if result.tokens.iter().any(|&token| token != 0)
            || result.hidden.iter().any(|&v| v != 0x3c00)
            || result.stream_probe.iter().any(|&v| v != value)
        {
            return Err(format!(
                "full model fixture mismatch at {visits} recurrent visits; probe={:?}",
                &result.stream_probe[..8]
            ));
        }
        report(&result);
    }
    let sampled = model.run(&tokens, 2, 0.8, 17)?;
    let expected = model.generate(&[0; 128], 2, 0.8, 17, 1)?;
    let unrolled = model.generate(&[0; 128], 2, 0.8, 17, 2)?;
    if expected.tokens[128..] != sampled.tokens[127..4095]
        || expected.tokens != unrolled.tokens
        || expected.tokens[128..]
            .iter()
            .all(|&token| token == expected.tokens[128])
    {
        return Err(
            "accepted-prefix generation changed with unrolling or stochastic sampling collapsed"
                .into(),
        );
    }
    println!(
        "GENERATION_PARITY ok=true prompt=128 generated=3968 temperature=0.8 unroll=1,2 \
         passes={}/{}",
        expected.passes, unrolled.passes
    );
    println!(
        "MODEL_PARITY ok=true layers=5 experts=625 shared=1 streams=4 recurrence=1,2,4 rope=true \
         readout=true"
    );
    check_causal(&mut model)?;
    Ok(())
}

/// Capture full initialized weights and real forward outputs, without BO
/// updates. Allocation dimensions match the shared finite case explicitly.
pub fn initialized_check(path: &Path) -> CudaResult<()> {
    use baux::semantics::mhc4::probe::initialized as case;
    let mut model = FbtModel::initialize_mhc4(case::SEED, case::ROWS, 4096)?;
    let trace = initialized_trace(&mut model)?;
    fs::write(
        path,
        baux::artifact::json::to_vec(&trace).map_err(cuda_error)?,
    )
    .map_err(cuda_error)?;
    println!(
        "INITIALIZED_FORWARD backend=cuda seed={} parameters={} rows={} targets={} output={}",
        case::SEED,
        baux::semantics::shape::MHC4_PARAMS,
        case::ROWS,
        case::TARGETS,
        path.display()
    );
    Ok(())
}

fn mhc4_check(model: &FbtModel) -> CudaResult<()> {
    let macros = [3_u32, 7];
    let (tokens, _) = model.expand_patches(&macros)?;
    let mut expected = Vec::with_capacity(128);
    for macro_token in macros {
        let mut previous = macro_token;
        expected.push(previous);
        for intra in 1..64_u32 {
            let mut mixed = (u64::from(previous) << 32)
                ^ (u64::from(macro_token) << 16)
                ^ (u64::from(intra) << 8)
                ^ 0x3c00;
            mixed = mixed.wrapping_add(0x9e3779b97f4a7c15);
            mixed = (mixed ^ (mixed >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
            mixed = (mixed ^ (mixed >> 27)).wrapping_mul(0x94d049bb133111eb);
            mixed ^= mixed >> 31;
            previous = (mixed % VOCAB as u64) as u32;
            expected.push(previous);
        }
    }
    if tokens != expected {
        return Err("CUDA mHC-4 patch expansion mismatch".into());
    }
    println!("MHC4_PARITY ok=true patches=2 patch-size=64 tokens=128");
    Ok(())
}

fn score_check(model: &FbtModel) -> CudaResult<()> {
    const ROWS: usize = 2;

    let mut logits = vec![f16::from_f32(-2.0).to_bits(); ROWS * VOCAB];
    let targets = [3_u32, 7];
    for (row, target) in targets.iter().enumerate() {
        logits[row * VOCAB + *target as usize] = f16::from_f32(2.0).to_bits();
    }
    let generated = [3_u32, 9];
    let logits = DeviceBuffer::from_host(&model.stream, &logits).map_err(cuda_error)?;
    let targets = DeviceBuffer::from_host(&model.stream, &targets).map_err(cuda_error)?;
    let generated = DeviceBuffer::from_host(&model.stream, &generated).map_err(cuda_error)?;
    let mut losses = DeviceBuffer::zeroed(&model.stream, ROWS).map_err(cuda_error)?;
    let mut output = DeviceBuffer::zeroed(&model.stream, 1).map_err(cuda_error)?;
    let launch = model
        .module
        .prepare_score(LaunchConfig1D::new(ROWS as u32, 256, 0))
        .map_err(cuda_error)?;
    model
        .module
        .score(
            &model.stream,
            &launch,
            &logits,
            &targets,
            &mut losses,
            ROWS as u32,
            0,
            0,
        )
        .map_err(cuda_error)?;
    let launch = model
        .module
        .prepare_objective(LaunchConfig1D::new(1, 256, 0))
        .map_err(cuda_error)?;
    model
        .module
        .objective(
            &model.stream,
            &launch,
            &losses,
            &generated,
            &targets,
            &mut output,
            ObjectiveShape {
                losses: ROWS as u32,
                tokens: ROWS as u32,
                patch: 1,
            },
        )
        .map_err(cuda_error)?;
    model.context.check_err().map_err(cuda_error)?;
    let loss = (2.0_f32.exp() + (VOCAB - 1) as f32 * (-2.0_f32).exp()).ln() - 2.0;
    let stats: Vec<ScoreStats> = read_prefix(&output, &model.stream, 1)?;
    let stats = stats[0];
    if (stats.mean_nll - loss).abs() > 0.002
        || (stats.maximum_nll - loss).abs() > 0.002
        || (stats.reward + loss).abs() > 0.002
        || stats.positional_accuracy != 0.5
        || stats.tokens != 2
        || stats.mismatches != 1
    {
        return Err(format!("CUDA generation scoring mismatch: {stats:?}"));
    }
    println!(
        "SCORING_PARITY ok=true rows={} reward={:.6} accuracy={:.3}",
        stats.tokens, stats.reward, stats.positional_accuracy
    );
    Ok(())
}

pub fn run(args: &[String]) -> CudaResult<()> {
    if args.len() < 3 || args.len() > 7 {
        return Err(
            "usage: ./baux cuda model CHECKPOINT TOKENS_JSON OUTPUT_DIR [CONTEXT=4096] [VISITS=2] \
             [TEMPERATURE=0.8] [SEED=17]"
                .into(),
        );
    }
    let tokens: Vec<u32> =
        baux::artifact::json::from_slice(&fs::read(&args[1]).map_err(|e| e.to_string())?)
            .map_err(|e| e.to_string())?;
    let context = arg(args, 3, 4096_usize)?;
    let visits = arg(args, 4, 2_usize)?;
    let temperature = arg(args, 5, 0.8_f32)?;
    let seed = arg(args, 6, 17_u64)?;
    let output = Path::new(&args[2]);
    fs::create_dir(output).map_err(|e| format!("create fresh output directory: {e}"))?;
    let mut model = FbtModel::open(Path::new(&args[0]), tokens.len(), context)?;
    let result = model.run(&tokens, visits, temperature, seed)?;
    fs::write(
        output.join("tokens.json"),
        baux::artifact::json::to_string(&result.tokens).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    let hidden = result
        .hidden
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect::<Vec<_>>();
    fs::write(output.join("hidden.f16"), hidden).map_err(|e| e.to_string())?;
    let record = json!({ "stage": "cuda-full-model-forward", "checkpoint": args[0], "rows": tokens.len(), "context": context, "physical-layers": 5, "layer-visits": result.visits, "temperature": temperature, "seed": seed, "device-ms": result.device_ms, "wall-ms": result.wall_ms });
    fs::write(
        output.join("result.json"),
        baux::artifact::json::pretty_string(&record).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    report(&result);
    Ok(())
}

pub fn generate(args: &[String]) -> CudaResult<()> {
    if args.len() < 3 || args.len() > 9 {
        return Err(
            "usage: ./baux cuda generate CHECKPOINT PROMPT_JSON OUTPUT_DIR [CONTEXT=4096] \
             [VISITS=2] [TEMPERATURE=0.8] [SEED=17] [UNROLL=2] [CHUNK=0]"
                .into(),
        );
    }
    let prompt: Vec<u32> =
        baux::artifact::json::from_slice(&fs::read(&args[1]).map_err(|e| e.to_string())?)
            .map_err(|e| e.to_string())?;
    let context = arg(args, 3, 4096_usize)?;
    let visits = arg(args, 4, 2_usize)?;
    let temperature = arg(args, 5, 0.8_f32)?;
    let seed = arg(args, 6, 17_u64)?;
    let unroll = arg(args, 7, 2_usize)?;
    let chunk = arg(args, 8, 0_usize)?;
    let output = Path::new(&args[2]);
    let mut model = if chunk == 0 {
        FbtModel::open(Path::new(&args[0]), context, context)?
    } else {
        FbtModel::open_streamed(Path::new(&args[0]), context, chunk)?
    };
    let (result, evaluated) = if chunk == 0 {
        let result = model.generate(&prompt, visits, temperature, seed, unroll)?;
        let evaluated = context * result.passes;
        (result, evaluated)
    } else {
        model.generate_streamed(&prompt, visits, temperature, seed, unroll, chunk)?
    };
    fs::create_dir(output).map_err(|e| format!("create fresh output directory: {e}"))?;
    fs::write(
        output.join("tokens.json"),
        baux::artifact::json::to_string(&result.tokens).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    let record = json!({ "stage": "cuda-accepted-prefix-generation", "checkpoint": args[0], "context": context, "prompt-tokens": result.prompt_tokens, "drafted-tokens": 0, "evaluated-positions": evaluated, "repaired-tokens": context - result.prompt_tokens, "generated-tokens": context - result.prompt_tokens, "committed-tokens": context - result.prompt_tokens, "physical-layers": 5, "layer-visits": result.visits, "temperature": temperature, "seed": seed, "unroll": unroll, "chunk-rows": chunk, "passes": result.passes, "device-ms": result.device_ms, "wall-ms": result.wall_ms, "full-bo-loop": false });
    fs::write(
        output.join("result.json"),
        baux::artifact::json::pretty_string(&record).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    println!(
        "GENERATE prompt={} generated={} passes={} executions={} device_ms={:.3} wall_ms={:.3}",
        result.prompt_tokens,
        context - result.prompt_tokens,
        result.passes,
        result.visits,
        result.device_ms,
        result.wall_ms
    );
    Ok(())
}

pub fn mhc4_bench(args: &[String]) -> CudaResult<()> {
    if args.len() < 4 || args.len() > 7 {
        return Err(
            "usage: mhc4-bench MODEL_SEED PROMPT_JSON OUTPUT_DIR GENERATED_TOKENS [VISITS=2] \
             [TEMPERATURE=1.0] [SEED=17]"
                .into(),
        );
    }
    let model_seed = args[0]
        .parse::<u64>()
        .map_err(|error| format!("invalid mHC-4 model seed: {error}"))?;
    let prompt: Vec<u32> =
        baux::artifact::json::from_slice(&fs::read(&args[1]).map_err(|e| e.to_string())?)
            .map_err(|e| e.to_string())?;
    let generated = arg(args, 3, 1_048_576_usize)?;
    let visits = arg(args, 4, 2_usize)?;
    let temperature = arg(args, 5, 1.0_f32)?;
    let seed = arg(args, 6, 17_u64)?;
    if prompt.is_empty()
        || prompt.len() % 64 != 0
        || generated == 0
        || generated % 64 != 0
        || prompt.iter().any(|&token| token >= VOCAB as u32)
    {
        return Err(
            "mHC-4 prompt and output must contain vocabulary-valid, nonempty 64-token patches"
                .into(),
        );
    }
    let core_positions = prompt
        .len()
        .checked_add(generated)
        .ok_or("mHC-4 context overflow")?
        .div_ceil(64);
    let core_rows = core_positions.next_power_of_two().max(64);
    let core_context = core_rows.max(4096);
    if core_context > 65_536 {
        return Err("mHC-4 coarse context exceeds the resident CUDA limit".into());
    }
    let output = Path::new(&args[2]);
    fs::create_dir(output).map_err(|e| format!("create fresh output directory: {e}"))?;
    let wall = std::time::Instant::now();
    let mut model = FbtModel::initialize_mhc4(model_seed, core_rows, core_context)?;
    model.configure_mhc4(&prompt, generated, None)?;
    let result = model.evaluate_mhc4(
        visits,
        temperature,
        seed,
        std::env::var_os("BAUX_CUDA_PROFILE").is_some(),
    )?;
    let tokens = model.mhc4_tokens()?;
    let throughput = generated as f64 * 1000.0 / f64::from(result.device_ms);
    let architecture_weights = model.weights.format == "mhc4";
    let record = json!({
        "schema": "baux.cuda.mhc4-generation.v1",
        "target": target_name(&model.context)?,
        "hardware": model.context.device_name().map_err(cuda_error)?,
        "model_seed": model_seed,
        "checkpoint_format": model.weights.format,
        "execution_architecture": "mhc4",
        "architecture_weights_match": architecture_weights,
        "patch_size": 64,
        "context_window_tokens": prompt.len() + generated,
        "prompt_tokens": prompt.len(),
        "drafted_tokens": 0,
        "core_context_positions": core_context,
        "evaluated_core_positions": core_rows,
        "core_active_positions": core_positions,
        "evaluated_tokens": generated,
        "repaired_tokens": 0,
        "generated_tokens": generated,
        "committed_tokens": generated,
        "forward_passes": 1,
        "layer_visits_per_wave": result.visits,
        "temperature": temperature,
        "seed": seed,
        "core_device_ms": result.core_ms,
        "generation_device_ms": result.generation_ms,
        "objective_device_ms": result.objective_ms,
        "device_ms": result.device_ms,
        "generated_tokens_per_device_second": throughput,
        "wall_ms": wall.elapsed().as_secs_f64() * 1000.0,
        "exact_within_patch_causal": true,
        "generation_method": "learned-mhc4-64-wave",
        "full_generation_loop": true,
        "full_bo_loop": false
    });
    fs::write(
        output.join("tokens.json"),
        baux::artifact::json::to_string(&tokens).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    let json = baux::artifact::json::pretty_string(&record).map_err(|e| e.to_string())?;
    fs::write(output.join("result.json"), &json).map_err(|e| e.to_string())?;
    println!("{json}");
    Ok(())
}

pub fn bench(args: &[String]) -> CudaResult<()> {
    if args.len() < 2 || args.len() > 10 {
        return Err(
            "usage: generation-bench CHECKPOINT OUTPUT_DIR [CONTEXT=4096] [PROMPT=128] [VISITS=2] \
             [TEMPERATURE=0.8] [SEED=17] [UNROLL=2] [CHUNK=0] [PROMPT_JSON]"
                .into(),
        );
    }
    let context = arg(args, 2, 4096_usize)?;
    let prompt_tokens = arg(args, 3, 128_usize)?;
    let visits = arg(args, 4, 2_usize)?;
    let temperature = arg(args, 5, 0.8_f32)?;
    let seed = arg(args, 6, 17_u64)?;
    let unroll = arg(args, 7, 2_usize)?;
    let chunk = arg(args, 8, 0_usize)?;
    if prompt_tokens == 0 || prompt_tokens >= context {
        return Err("generation benchmark prompt must be nonempty and shorter than context".into());
    }
    let prompt: Vec<u32> = if let Some(path) = args.get(9) {
        baux::artifact::json::from_slice(&fs::read(path).map_err(|e| e.to_string())?)
            .map_err(|e| e.to_string())?
    } else {
        vec![0; prompt_tokens]
    };
    if prompt.len() != prompt_tokens || prompt.iter().any(|&t| t >= VOCAB as u32) {
        return Err("benchmark prompt file must match the declared count and vocabulary".into());
    }
    let output = Path::new(&args[1]);
    fs::create_dir(output).map_err(|e| format!("create fresh output directory: {e}"))?;
    if std::env::var_os("BAUX_SYNTH_GEMM").is_some() {
        fs::write(output.join("gemm.json"), crate::gemm::measure()?).map_err(|e| e.to_string())?;
    }
    let precision_cases = crate::precision::check()?;
    let (mut model, profile, result, evaluated, load_ms) = exec_bench(
        &args[0],
        &prompt,
        context,
        visits,
        temperature,
        seed,
        unroll,
        chunk,
    )?;
    let mut comparisons = compare_chunks(
        &args[0],
        &prompt,
        &result,
        context,
        visits,
        temperature,
        seed,
        unroll,
        chunk,
        evaluated,
    )?;
    if context == 4096 && chunk == 0 {
        compare_cases(
            &args[0],
            &mut model,
            context,
            prompt_tokens,
            visits,
            temperature,
            seed,
            unroll,
            &mut comparisons,
        )?;
    }
    let record = bench_record(
        args,
        &model,
        &result,
        profile,
        comparisons,
        (context, prompt_tokens, temperature, seed, unroll, chunk),
        (evaluated, load_ms, precision_cases),
    )?;
    let json = baux::artifact::json::pretty_string(&record).map_err(|e| e.to_string())?;
    fs::write(output.join("result.json"), &json).map_err(|e| e.to_string())?;
    fs::write(
        output.join("tokens.json"),
        baux::artifact::json::to_string(&result.tokens).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    println!("{json}");
    Ok(())
}

fn exec_bench(
    checkpoint: &str,
    prompt: &[u32],
    context: usize,
    visits: usize,
    temperature: f32,
    seed: u64,
    unroll: usize,
    chunk: usize,
) -> CudaResult<(
    FbtModel,
    Option<super::WaveProfile>,
    GenerationOutput,
    usize,
    f64,
)> {
    let loaded_at = std::time::Instant::now();
    let mut model = if chunk == 0 {
        FbtModel::open(Path::new(checkpoint), context, context)?
    } else {
        FbtModel::open_streamed(Path::new(checkpoint), context, chunk)?
    };
    let profile = (chunk == 0)
        .then(|| model.profile_wave(&vec![0; context], visits, temperature, seed))
        .transpose()?;
    let (result, evaluated) = if chunk == 0 {
        let result = model.generate(prompt, visits, temperature, seed, unroll)?;
        let evaluated = context * result.passes;
        (result, evaluated)
    } else {
        model.generate_streamed(prompt, visits, temperature, seed, unroll, chunk)?
    };
    let load_ms = loaded_at.elapsed().as_secs_f64() * 1000.0;
    Ok((model, profile, result, evaluated, load_ms))
}

fn compare_chunks(
    checkpoint: &str,
    prompt: &[u32],
    result: &GenerationOutput,
    context: usize,
    visits: usize,
    temperature: f32,
    seed: u64,
    unroll: usize,
    chunk: usize,
    evaluated: usize,
) -> CudaResult<Vec<Value>> {
    let mut comparisons = Vec::new();
    if context > 65536 && chunk != 0 {
        let other_chunk = if chunk == 512 { 1024 } else { 512 };
        let mut reference = FbtModel::open_streamed(Path::new(checkpoint), context, other_chunk)?;
        let (other, other_evaluated) =
            reference.generate_streamed(prompt, visits, temperature, seed, unroll, other_chunk)?;
        if other.tokens != result.tokens || other.frontiers != result.frontiers {
            return Err("million-context generation changed with chunk size".into());
        }
        comparisons.push(json!({
            "case": "chunk-invariance", "reference_chunk_rows": other_chunk,
            "reference_device_ms": other.device_ms,
            "reference_evaluated_positions": other_evaluated,
            "token_parity": true, "frontier_parity": true,
            "packed_reference": false
        }));
    }
    if chunk != 0 && context <= 65536 {
        let mut reference = FbtModel::open(Path::new(checkpoint), context, context)?;
        let packed = reference.generate(prompt, visits, temperature, seed, unroll)?;
        if packed.tokens != result.tokens || packed.frontiers != result.frontiers {
            return Err("streamed generation differs from packed reference".into());
        }
        comparisons.push(json!({
            "case": "selected-chunk", "chunk_rows": chunk,
            "packed_device_ms": packed.device_ms,
            "streamed_device_ms": result.device_ms,
            "packed_evaluated_positions": context * packed.passes,
            "streamed_evaluated_positions": evaluated,
            "token_parity": true, "frontier_parity": true
        }));
    }
    Ok(comparisons)
}

fn compare_cases(
    checkpoint: &str,
    model: &mut FbtModel,
    context: usize,
    prompt_tokens: usize,
    visits: usize,
    temperature: f32,
    seed: u64,
    unroll: usize,
    comparisons: &mut Vec<Value>,
) -> CudaResult<()> {
    let prompt = vec![0; prompt_tokens];
    let varied = (0..context)
        .map(|i| ((i * 7919 + 17) % VOCAB) as u32)
        .collect::<Vec<_>>();
    let mut streamed = FbtModel::open(Path::new(checkpoint), context, context)?;
    for (case, input, chunk, case_seed, case_unroll) in [
        ("zero-512", prompt.as_slice(), 512, seed, unroll),
        ("zero-2048", prompt.as_slice(), 2048, seed, unroll),
        (
            "varied-512",
            &varied[..prompt_tokens],
            512,
            seed.wrapping_add(1),
            2,
        ),
    ] {
        let packed = model.generate(input, visits, temperature, case_seed, case_unroll)?;
        let (cached, evaluated) = streamed.generate_streamed(
            input,
            visits,
            temperature,
            case_seed,
            case_unroll,
            chunk,
        )?;
        if packed.tokens != cached.tokens || packed.frontiers != cached.frontiers {
            let mismatch = packed
                .tokens
                .iter()
                .zip(&cached.tokens)
                .position(|(a, b)| a != b);
            return Err(format!(
                "streamed parity failed: case={case} first_token={mismatch:?} \
                 packed_frontiers={:?} cached_frontiers={:?}",
                packed.frontiers, cached.frontiers
            ));
        }
        comparisons.push(json!({
            "case": case,
            "chunk_rows": chunk,
            "seed": case_seed,
            "unroll": case_unroll,
            "packed_device_ms": packed.device_ms,
            "streamed_device_ms": cached.device_ms,
            "streamed_wall_ms": cached.wall_ms,
            "packed_evaluated_positions": context * packed.passes,
            "streamed_evaluated_positions": evaluated,
            "committed_tokens": context - input.len(),
            "token_parity": true,
            "frontier_parity": true
        }));
        println!(
            "STREAMED_PARITY case={case} chunk={chunk} evaluated={evaluated} packed_ms={:.3} \
             streamed_ms={:.3}",
            packed.device_ms, cached.device_ms
        );
    }
    Ok(())
}

fn bench_record(
    args: &[String],
    model: &FbtModel,
    result: &GenerationOutput,
    profile: Option<super::WaveProfile>,
    comparisons: Vec<Value>,
    params: (usize, usize, f32, u64, usize, usize),
    metrics: (usize, f64, usize),
) -> CudaResult<Value> {
    let (context, prompt_tokens, temperature, seed, unroll, chunk) = params;
    let (evaluated, load_ms, precision_cases) = metrics;
    let generated = context - prompt_tokens;
    Ok(json!({
        "schema": "baux.cuda.generation.v1",
        "fp16_narrowing_cases": precision_cases,
        "fp16_widening_patterns": 65536,
        "synth_gemm_tile_k": std::env::var("BAUX_SYNTH_GEMM").ok(),
        "synth_scope": std::env::var_os("BAUX_SYNTH_GEMM").map(|_| "readout_only"),
        "streamed_comparison": comparisons,
        "chunk_rows": chunk,
        "repair_chunk_rows": if chunk == 0 { 0 } else { super::cached::repair_chunk(chunk)? },
        "logit_scratch_bytes": model.scratch.logits.len() * 2,
        "readout_chunk_rows": model.scratch.logits.len() / VOCAB,
        "target": target_name(&model.context)?,
        "hardware": model.context.device_name().map_err(crate::cuda_error)?,
        "prompt_source": if args.get(9).is_some() { "token_file" } else { "zero_token_fixture" },
        "checkpoint": args[0],
        "context_tokens": context,
        "prompt_tokens": prompt_tokens,
        "drafted_tokens": 0,
        "evaluated_positions": evaluated,
        "repaired_tokens": generated,
        "generated_tokens": generated,
        "committed_tokens": generated,
        "repair_waves": result.passes,
        "frontiers": result.frontiers,
        "layer_visits_per_wave": result.visits,
        "temperature": temperature,
        "seed": seed,
        "unroll": unroll,
        "device_ms": result.device_ms,
        "wall_ms": result.wall_ms,
        "first_wave_ms": result.first_wave_ms,
        "checkpoint_load_and_generation_ms": load_ms,
        "first_wave_chunk_profiles": result.chunk_profiles.iter().map(|(first, p)| json!({
            "first_position": first, "rows": chunk,
            "embed_ms": p.embed_ms, "mhc_ms": p.mhc_ms, "qkv_ms": p.qkv_ms,
            "index_ms": p.index_ms, "pisa_ms": p.pisa_ms, "output_ms": p.output_ms,
            "moe_ms": p.moe_ms, "readout_ms": p.readout_ms, "total_ms": p.total_ms
        })).collect::<Vec<_>>(),
        "wave_profile": profile.map(|profile| json!({
            "embed_ms": profile.embed_ms,
            "mhc_ms": profile.mhc_ms,
            "qkv_ms": profile.qkv_ms,
            "index_ms": profile.index_ms,
            "pisa_ms": profile.pisa_ms,
            "output_ms": profile.output_ms,
            "attention_ms": profile.attention_ms,
            "moe_router_ms": profile.moe_router_ms,
            "moe_route_ms": profile.moe_route_ms,
            "moe_experts_ms": profile.moe_experts_ms,
            "moe_combine_ms": profile.moe_combine_ms,
            "moe_ms": profile.moe_ms,
            "readout_ms": profile.readout_ms,
            "total_ms": profile.total_ms
        })),
        "full_generation_loop": true,
        "full_bo_loop": false
    }))
}

fn arg<T: std::str::FromStr>(args: &[String], index: usize, default: T) -> CudaResult<T> {
    args.get(index).map_or(Ok(default), |value| {
        value
            .parse()
            .map_err(|_| format!("invalid argument {index}: {value}"))
    })
}

fn report(result: &ModelOutput) {
    println!(
        "MODEL rows={} executions={} device_ms={:.3} wall_ms={:.3}",
        result.tokens.len(),
        result.visits,
        result.device_ms,
        result.wall_ms
    );
}
