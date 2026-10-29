use std::error::Error;
use std::io;

use baux_cuda::{FbtPrefill, FbtPrefillProfile, Leaf as ResidentLeaf};
use baux_cuda_device::trials;
use cuda_core::{CudaContext, CudaStream, DeviceBuffer, simt::LaunchConfig};

mod main_bench;
mod main_parity;
mod ptx_probe;

use main_bench::{benchmark, bf16_benchmark, bf16_curve, trial_benchmark};
use main_parity::{parity, resident_parity};

type AppResult<T> = Result<T, Box<dyn Error + Send + Sync>>;

#[derive(Debug, Clone, Copy)]
struct Leaf {
    offset: usize,
    length: usize,
    bits: u8,
    scale: f32,
    radius: f32,
}

impl Leaf {
    fn new(offset: usize, length: usize, bits: u8, scale: f32, radius: f32) -> AppResult<Self> {
        if length == 0 {
            return Err(io::Error::other("leaf length must be positive").into());
        }
        if bits != 4 && bits != 8 {
            return Err(io::Error::other("leaf bits must be 4 or 8").into());
        }
        if !scale.is_finite() || scale <= 0.0 || !radius.is_finite() || radius <= 0.0 {
            return Err(io::Error::other("leaf scale and radius must be positive").into());
        }
        Ok(Self {
            offset,
            length,
            bits,
            scale,
            radius,
        })
    }
}

#[derive(Debug, Clone, Copy)]
struct LaunchLeaf {
    byte_offset: u32,
    element_offset: u32,
    length: u32,
    bits: u32,
    whole: u32,
    threshold: u32,
}

impl LaunchLeaf {
    fn new(leaf: Leaf, byte_offset: usize, length: f32) -> AppResult<Self> {
        let max_code = (1_u32 << leaf.bits) - 1;
        let amplitude = (length * leaf.radius / leaf.scale).clamp(0.0, max_code as f32);
        let whole = amplitude.floor() as u32;
        let threshold = if whole == max_code {
            0
        } else {
            ((amplitude - whole as f32) * (u32::MAX as f32)) as u32
        };
        Ok(Self {
            byte_offset: byte_offset.try_into()?,
            element_offset: leaf.offset.try_into()?,
            length: leaf.length.try_into()?,
            bits: u32::from(leaf.bits),
            whole,
            threshold,
        })
    }

    fn work_items(self) -> u32 {
        if self.bits == 4 {
            self.length.div_ceil(2)
        } else {
            self.length
        }
    }

    fn row_bytes(self) -> usize {
        self.work_items() as usize
    }
}

fn main() {
    if let Err(error) = run() {
        eprintln!("ERROR: {error}");
        std::process::exit(1);
    }
}

fn run() -> AppResult<()> {
    let args: Vec<String> = std::env::args().collect();
    let tail = args.get(2..).unwrap_or(&[]);
    match args.get(1).map(String::as_str).unwrap_or("parity") {
        "parity" => parity(),
        "resident" => resident_parity(),
        "bench" => benchmark(tail),
        "trial-bench" => trial_benchmark(tail),
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
            "unknown command {command:?}; expected parity, resident, prefill, model-check, model, generate, generation-bench, mhc4-bench, mhc4-bo-bench, diffusion, diffusion-check, context, ptx-probe, bench, trial-bench, bf16-bench, or bf16-curve"
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

fn resident_steps(leaves: &[Leaf], length: f32) -> AppResult<Vec<ResidentLeaf>> {
    let mut byte_offset = 0;
    leaves
        .iter()
        .map(|&leaf| {
            let step = LaunchLeaf::new(leaf, byte_offset, length)?;
            byte_offset += step.row_bytes();
            Ok(ResidentLeaf {
                byte_offset: step.byte_offset,
                element_offset: step.element_offset,
                length: step.length,
                bits: step.bits,
                encoding: if step.bits == 4 { 0 } else { 1 },
                scale: leaf.scale,
                weight: 1.0,
                whole: step.whole,
                threshold: step.threshold,
            })
        })
        .collect()
}

fn check_case(
    context: &CudaContext,
    stream: &CudaStream,
    module: &trials::LoadedModule,
    base: &[u8],
    leaves: &[Leaf],
    length: f32,
    seed: u64,
) -> AppResult<()> {
    let base_device = DeviceBuffer::from_host(stream, base)?;
    let mut output_device = DeviceBuffer::<u8>::zeroed(stream, base.len())?;
    launch_materialize(
        module,
        stream,
        &base_device,
        &mut output_device,
        leaves,
        length,
        seed,
    )?;
    let output = output_device.to_host_vec(stream)?;
    let expected = expected_row(base, leaves, length, seed)?;
    compare_rows(&expected, &output, seed, length)?;
    context.check_err()?;
    Ok(())
}

fn launch_materialize(
    module: &trials::LoadedModule,
    stream: &CudaStream,
    base: &DeviceBuffer<u8>,
    output: &mut DeviceBuffer<u8>,
    leaves: &[Leaf],
    length: f32,
    seed: u64,
) -> AppResult<()> {
    let mut byte_offset = 0;
    for &leaf in leaves {
        let launch = LaunchLeaf::new(leaf, byte_offset, length)?;
        // SAFETY: there is one thread per leaf byte, leaf byte ranges do not
        // overlap, and the input and output buffers contain the
        // complete encoded row.
        unsafe {
            module.materialize(
                stream,
                LaunchConfig::for_num_elems(launch.work_items()),
                base,
                output,
                launch.byte_offset,
                launch.element_offset,
                launch.length,
                launch.bits,
                seed as u32,
                (seed >> 32) as u32,
                launch.whole,
                launch.threshold,
            )
        }?;
        byte_offset += launch.row_bytes();
    }
    Ok(())
}

fn expected_row(base: &[u8], leaves: &[Leaf], length: f32, seed: u64) -> AppResult<Vec<u8>> {
    let mut row = vec![0_u8; base.len()];
    let mut byte_offset = 0;
    for &leaf in leaves {
        let step = LaunchLeaf::new(leaf, byte_offset, length)?;
        for element in 0..leaf.length {
            let byte = byte_offset + if leaf.bits == 4 { element / 2 } else { element };
            let shift = if leaf.bits == 4 { (element & 1) * 4 } else { 0 };
            let mask = if leaf.bits == 4 { 0x0f } else { 0xff };
            let code = u32::from((base[byte] >> shift) & mask);
            let value = cpu_perturb(
                code,
                seed,
                leaf.offset as u32 + element as u32,
                step.bits,
                step.whole,
                step.threshold,
            );
            row[byte] |= (value as u8) << shift;
        }
        byte_offset += step.row_bytes();
    }
    Ok(row)
}

fn cpu_perturb(code: u32, seed: u64, element: u32, bits: u32, whole: u32, threshold: u32) -> u32 {
    let random = cpu_hash(seed, element);
    let amount = whole + u32::from((random >> 1) < (threshold >> 1));
    if amount == 0 {
        return code;
    }
    let max_code = (1_u32 << bits) - 1;
    if random & 1 == 0 {
        if code >= amount {
            code - amount
        } else {
            (code + amount).min(max_code)
        }
    } else if code + amount <= max_code {
        code + amount
    } else {
        code.saturating_sub(amount)
    }
}

fn cpu_hash(seed: u64, element: u32) -> u32 {
    let mut value = (seed as u32) ^ element.wrapping_mul(0x9e37_79b9);
    value ^= value >> 16;
    value = value.wrapping_mul(0x7feb_352d);
    value ^= (seed >> 32) as u32;
    value = value.wrapping_mul(0x846c_a68b);
    value ^ (value >> 15)
}

fn compare_rows(expected: &[u8], actual: &[u8], seed: u64, length: f32) -> AppResult<()> {
    if expected == actual {
        return Ok(());
    }
    let mismatch = expected
        .iter()
        .zip(actual)
        .position(|(expected, actual)| expected != actual)
        .unwrap_or(expected.len().min(actual.len()));
    Err(io::Error::other(format!(
        "CUDA parity failed at byte {mismatch} for seed={seed} length={length}: \
         expected={:?} actual={:?}",
        expected.get(mismatch),
        actual.get(mismatch)
    ))
    .into())
}

fn row_bytes(leaves: &[Leaf]) -> usize {
    leaves
        .iter()
        .map(|leaf| {
            if leaf.bits == 4 {
                leaf.length.div_ceil(2)
            } else {
                leaf.length
            }
        })
        .sum()
}

fn make_base(bytes: usize) -> Vec<u8> {
    (0..bytes)
        .map(|index| (index.wrapping_mul(37).wrapping_add(11) & 0xff) as u8)
        .collect()
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
