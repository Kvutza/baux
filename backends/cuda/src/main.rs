use std::error::Error;
use std::io;

use baux_cuda::{FbtPrefill, FbtPrefillProfile};

mod main_bench;
mod ptx_probe;

use main_bench::{benchmark, bf16_benchmark, bf16_curve};

type AppResult<T> = Result<T, Box<dyn Error + Send + Sync>>;

fn main() {
    if let Err(error) = run() {
        eprintln!("ERROR: {error}");
        std::process::exit(1);
    }
}

fn run() -> AppResult<()> {
    let args: Vec<String> = std::env::args().collect();
    let tail = args.get(2..).unwrap_or(&[]);
    match args.get(1).map(String::as_str).unwrap_or("model-check") {
        "bench" => benchmark(tail),
        "bf16-bench" => bf16_benchmark(tail),
        "bf16-curve" => bf16_curve(tail),
        "prefill" => prefill(),
        "gemm-bench" => baux_cuda::gemm::bench().map_err(|e| io::Error::other(e).into()),
        "ptx-probe" => ptx_probe::run(tail),
        "context" => baux_cuda::context::run(tail).map_err(|e| io::Error::other(e).into()),
        command => run_model(command, tail),
    }
}

fn run_model(command: &str, args: &[String]) -> AppResult<()> {
    match command {
        "model-check" => check_model(args),
        "diffusion" => baux_cuda::model_diffusion(args).map_err(|e| io::Error::other(e).into()),
        "diffusion-check" => {
            baux_cuda::diffusion_check().map_err(|e| io::Error::other(e).into())
        }
        "generate" => baux_cuda::model_generate(args).map_err(|e| io::Error::other(e).into()),
        "generation-bench" => {
            baux_cuda::generation_bench(args).map_err(|e| io::Error::other(e).into())
        }
        "mhc4-bench" => baux_cuda::mhc4_bench(args).map_err(|e| io::Error::other(e).into()),
        "mhc4-bo-bench" => baux_cuda::mhc4_bo(args).map_err(|e| io::Error::other(e).into()),
        "model" => baux_cuda::model_run(args).map_err(|e| io::Error::other(e).into()),
        _ => Err(io::Error::other(format!(
            "unknown command {command:?}; expected prefill, model-check, model, generate, generation-bench, mhc4-bench, mhc4-bo-bench, diffusion, diffusion-check, context, ptx-probe, bench, bf16-bench, or bf16-curve"
        ))
        .into()),
    }
}

fn check_model(args: &[String]) -> AppResult<()> {
    let initialized = match args {
        [] => None,
        [flag, path] if flag == "--initialized" => Some(std::path::Path::new(path)),
        _ => {
            return Err(io::Error::other("usage: model-check [--initialized OUTPUT]").into());
        }
    };
    baux_cuda::model_check().map_err(io::Error::other)?;
    if let Some(path) = initialized {
        baux_cuda::initialized_check(path).map_err(io::Error::other)?;
    }
    Ok(())
}

fn prefill() -> AppResult<()> {
    let embedding = vec![0x3c00_u16; baux_cuda_device::VOCAB as usize * 512];
    let rms = vec![0x3c00_u16; 512];
    let qkv = vec![0x1800_u16; 512 * 640];
    let output_weight = vec![0x1800_u16; 512 * 512];
    let router_weight = vec![0x1800_u16; 512 * 128];
    let mut executor = FbtPrefill::new(&embedding, &rms, &qkv, &output_weight, &router_weight)
        .map_err(|error| io::Error::other(format!("prefill init: {error}")))?;
    let tokens = (0..4096)
        .map(|index| (index % baux_cuda_device::VOCAB) as u32)
        .collect::<Vec<_>>();
    // Warm-up absorbs first-launch/module state. Keep all weights and scratch
    // buffers resident for the measured repetitions.
    let _ = executor
        .run(&tokens)
        .map_err(|error| io::Error::other(format!("prefill warmup: {error}")))?;
    let mut samples = Vec::with_capacity(5);
    let mut output = None;
    let mut experts = None;
    for _ in 0..5 {
        let (result, selected, profile) = executor
            .run(&tokens)
            .map_err(|error| io::Error::other(format!("prefill run: {error}")))?;
        output = Some(result);
        experts = Some(selected);
        samples.push(profile);
    }
    let output = output.ok_or_else(|| io::Error::other("prefill produced no measured output"))?;
    // Unit input with 1/512 projections gives unit attention and output
    // branches; the first residual is exactly two.
    if output.len() != 4096 * 512 || output.iter().any(|&value| value != 0x4000) {
        return Err(io::Error::other("prefill output parity failed").into());
    }
    let experts = experts.ok_or_else(|| io::Error::other("prefill produced no routes"))?;
    if experts.chunks_exact(3).any(|route| route != [0, 1, 2]) {
        return Err(io::Error::other("prefill route parity failed").into());
    }
    print_prefill(&samples);
    Ok(())
}

fn print_prefill(samples: &[FbtPrefillProfile]) {
    let median = |select: fn(&FbtPrefillProfile) -> f32| {
        let mut values: Vec<f32> = samples.iter().map(select).collect();
        values.sort_by(f32::total_cmp);
        values[values.len() / 2]
    };
    println!(
        "FBT routed-attention slice: embedding -> RMS -> QKV -> PISA -> residual -> top-k; rows=4096 width=512 qkv=640 warmup=1 measured=5 median_embed_ms={:.3} median_rms_ms={:.3} median_projection_ms={:.3} median_tree_ms={:.3} median_selection_ms={:.3} median_attention_ms={:.3} median_residual_ms={:.3} median_routing_ms={:.3} median_device_ms={:.3} median_end_to_end_ms={:.3}",
        median(|p| p.embed_ms),
        median(|p| p.rms_ms),
        median(|p| p.projection_ms),
        median(|p| p.tree_ms),
        median(|p| p.selection_ms),
        median(|p| p.attention_ms),
        median(|p| p.residual_ms),
        median(|p| p.routing_ms),
        median(|p| p.device_ms),
        median(|p| p.end_to_end_ms),
    );
}

fn parse_arg<T>(args: &[String], index: usize, default: T, name: &str) -> AppResult<T>
where
    T: std::str::FromStr,
    T::Err: Error + Send + Sync + 'static,
{
    args.get(index)
        .map(|value| value.parse().map_err(Into::into))
        .unwrap_or(Ok(default))
        .map_err(|error: Box<dyn Error + Send + Sync>| {
            io::Error::other(format!("invalid {name}: {error}")).into()
        })
}
