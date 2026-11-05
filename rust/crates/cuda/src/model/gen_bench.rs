use super::VOCAB;
use super::run::arg;
use crate::capture::{CudaResult, cuda_error, target_name};
use crate::model::{FbtModel, GenerationOutput};
use baux::artifact::json::{Value, json};
use std::fs;
use std::path::Path;

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
        "streamed_comparison": comparisons,
        "chunk_rows": chunk,
        "repair_chunk_rows": if chunk == 0 { 0 } else { super::cached::repair_chunk(chunk)? },
        "logit_scratch_bytes": model.scratch.logits.len() * 2,
        "readout_chunk_rows": model.scratch.logits.len() / VOCAB,
        "target": target_name(&model.context)?,
        "hardware": model.context.device_name().map_err(cuda_error)?,
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
