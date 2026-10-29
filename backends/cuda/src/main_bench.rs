use std::io;
use std::time::Instant;

use baux_cuda::{
    Bf16Leaf, Bf16SearchEngine, Leaf as ResidentLeaf, MAX_HISTORY, RoundConfig as ResidentAsk,
    Tile as ResidentTile, TrialEngine, device_target, target_name,
};
use baux_cuda_device::trials;
use cuda_core::{CudaContext, DeviceBuffer};

use super::main_parity::pair_parity;
use super::*;

pub(super) fn benchmark(args: &[String]) -> AppResult<()> {
    if args.first().is_some_and(|value| value == "muon") {
        if args.len() != 1 {
            return Err(io::Error::other("bench muon accepts no additional arguments").into());
        }
        let record = baux_cuda::muon::measure().map_err(io::Error::other)?;
        println!("{}", baux::artifact::json::pretty_string(&record)?);
        return Ok(());
    }
    let elements = parse_arg(args, 0, 16 * 1024 * 1024, "elements")?;
    let iterations = parse_arg(args, 1, 100, "iterations")?;
    let bits: u8 = parse_arg(args, 2, 4, "bits")?;
    if iterations == 0 {
        return Err(io::Error::other("iterations must be positive").into());
    }
    if bits != 4 && bits != 8 {
        return Err(io::Error::other("bits must be 4 or 8").into());
    }

    let scale = if bits == 4 { 0.125 } else { 0.03125 };
    let leaves = vec![Leaf::new(0, elements, bits, scale, 0.75)?];
    let base = make_base(row_bytes(&leaves));
    let context = CudaContext::new(0)?;
    let stream = context.default_stream();
    // SAFETY: the generated bindings load the matching embedded kernel
    // artifact.
    let module = unsafe { trials::load(&context) }?;
    let base_device = DeviceBuffer::from_host(&stream, &base)?;
    let mut output_device = DeviceBuffer::<u8>::zeroed(&stream, base.len())?;
    let length = 0.8;

    for seed in 0..5 {
        launch_materialize(
            &module,
            &stream,
            &base_device,
            &mut output_device,
            &leaves,
            length,
            seed,
        )?;
    }
    stream.synchronize()?;

    let event_flags = cuda_core::sys::CUevent_flags_enum_CU_EVENT_DEFAULT;
    let start = stream.record_event(Some(event_flags))?;
    for iteration in 0..iterations {
        launch_materialize(
            &module,
            &stream,
            &base_device,
            &mut output_device,
            &leaves,
            length,
            10_000 + iteration as u64,
        )?;
    }
    let end = stream.record_event(Some(event_flags))?;
    let total_ms = f64::from(start.elapsed_ms(&end)?);
    let output = output_device.to_host_vec(&stream)?;
    let final_seed = 10_000 + iterations as u64 - 1;
    let expected = expected_row(&base, &leaves, length, final_seed)?;
    compare_rows(&expected, &output, final_seed, length)?;

    let kernel_ms = total_ms / iterations as f64;
    let bytes = (base.len() as f64) * 2.0 * iterations as f64;
    let gib_s = bytes / (total_ms / 1_000.0) / (1024.0 * 1024.0 * 1024.0);
    let target = target_name(&context).map_err(io::Error::other)?;
    println!(
        "BENCH ok=true target={target} elements={elements} bits={bits} iterations={iterations} \
         row_bytes={} kernel_ms={kernel_ms:.6} gib_s={gib_s:.3}",
        base.len()
    );
    Ok(())
}

pub(super) fn trial_benchmark(args: &[String]) -> AppResult<()> {
    let candidates: usize = parse_arg(args, 0, 1_024, "candidates")?;
    let history: usize = parse_arg(args, 1, 32, "history")?;
    let elements: usize = parse_arg(args, 2, 8_192, "elements")?;
    let iterations: usize = parse_arg(args, 3, 50, "iterations")?;
    bf16_args(candidates, history, elements, iterations)?;
    let (mut engine, steps) = init_trial(history, elements)?;

    let history_slots = (0..history as u32).collect::<Vec<_>>();
    let outcomes = (0..history)
        .map(|index| ((index.wrapping_mul(37) % 101) as f32 - 50.0) / 25.0)
        .collect::<Vec<_>>();
    let seeds = (0..candidates)
        .map(|index| 0x9e37_79b9_7f4a_7c15_u64.wrapping_mul(index as u64 + 1))
        .collect::<Vec<_>>();
    let draws = (0..history)
        .map(|index| ((index.wrapping_mul(29) % 97) as f32 - 48.0) / 24.0)
        .collect::<Vec<_>>();
    let config = ResidentAsk {
        neighbors: history.min(8),
        acquisition: 0,
        epistemic_scale: 0.7,
        aleatoric_scale: 0.05,
        y_scale: 1.0,
        beta: 1.2,
        adaptive_neighbors: false,
    };
    engine.set_profiling(true);

    for _ in 0..3 {
        engine
            .ask(
                0,
                &history_slots,
                &outcomes,
                history,
                &seeds,
                &draws,
                &steps,
                config,
                true,
            )
            .map_err(|error| io::Error::other(format!("trial benchmark warmup: {error}")))?;
    }

    let started = Instant::now();
    let mut totals = [0.0_f64; 4];
    for _ in 0..iterations {
        let (choice, score) = engine
            .ask(
                0,
                &history_slots,
                &outcomes,
                history,
                &seeds,
                &draws,
                &steps,
                config,
                true,
            )
            .map_err(|error| io::Error::other(format!("trial benchmark ask: {error}")))?;
        if choice >= candidates || !score.is_finite() {
            return Err(io::Error::other(format!(
                "trial benchmark returned choice={choice}, score={score}"
            ))
            .into());
        }
        let profile = engine
            .last_profile()
            .ok_or_else(|| io::Error::other("trial benchmark profile was not recorded"))?;
        totals[0] += f64::from(profile.score_ms);
        totals[1] += f64::from(profile.pick_ms);
        totals[2] += f64::from(profile.materialize_ms);
        totals[3] += f64::from(profile.total_ms);
    }
    let wall_seconds = started.elapsed().as_secs_f64();
    let count = iterations as f64;
    let asks_s = count / wall_seconds;
    let target = device_target().map_err(io::Error::other)?;
    println!(
        "{{\"schema\":\"baux.cuda.packed-bo-ask.v1\",\"target\":\"{target}\",\
         \"storage\":\"packed-quantized\",\"candidates\":{candidates},\"history\":{history},\
         \"parameter_elements\":{elements},\"iterations\":{iterations},\
         \"proposal_score_ms\":{:.6},\"pick_ms\":{:.6},\"materialize_ms\":{:.6},\
         \"gpu_ms\":{:.6},\"wall_ms\":{:.6},\"asks_per_second\":{asks_s:.3},\
         \"candidates_per_second\":{:.3},\"context_tokens\":0,\"drafted_tokens\":0,\
         \"evaluated_tokens\":0,\"repaired_tokens\":0,\"generated_tokens\":0,\
         \"committed_tokens\":0,\"full_generation_loop\":false}}",
        totals[0] / count,
        totals[1] / count,
        totals[2] / count,
        totals[3] / count,
        wall_seconds * 1_000.0 / count,
        candidates as f64 * asks_s,
    );
    Ok(())
}

fn init_trial(history: usize, elements: usize) -> AppResult<(TrialEngine, Vec<ResidentLeaf>)> {
    let leaves = vec![Leaf::new(0, elements, 4, 0.125, 0.75)?];
    let steps = resident_steps(&leaves, 0.8)?;
    let tiles = steps
        .iter()
        .enumerate()
        .map(|(leaf, step)| ResidentTile {
            leaf: leaf as u32,
            start: 0,
            length: step.length,
            pad: 0,
        })
        .collect::<Vec<_>>();
    let base = make_base(row_bytes(&leaves));
    let mut engine = TrialEngine::new(&base, &steps, &tiles, history + 1)
        .map_err(|error| io::Error::other(format!("trial benchmark init: {error}")))?;
    for slot in 1..history {
        engine
            .materialize(0, slot, 0x1_0000 + slot as u64, &steps)
            .map_err(|error| io::Error::other(format!("trial benchmark history: {error}")))?;
    }
    Ok((engine, steps))
}

#[derive(Debug)]
struct Bf16Run {
    candidates: usize,
    history: usize,
    elements: usize,
    iterations: usize,
    score_ms: f64,
    pick_ms: f64,
    materialize_ms: f64,
    gpu_ms: f64,
    wall_ms: f64,
}

impl Bf16Run {
    fn write_json(&self, output: &mut String) {
        use std::fmt::Write as _;
        write!(
            output,
            "{{\"candidates\":{},\"history\":{},\"parameter_elements\":{},\"iterations\":{},\"\
             proposal_score_ms\":{:.6},\"pick_ms\":{:.6},\"materialize_ms\":{:.6},\"gpu_ms\":{:.\
             6},\"wall_ms\":{:.6},\"asks_per_second\":{:.3},\"candidates_per_second\":{:.3}}}",
            self.candidates,
            self.history,
            self.elements,
            self.iterations,
            self.score_ms,
            self.pick_ms,
            self.materialize_ms,
            self.gpu_ms,
            self.wall_ms,
            1_000.0 / self.wall_ms,
            self.candidates as f64 * 1_000.0 / self.wall_ms,
        )
        .expect("writing JSON to a string cannot fail");
    }
}

pub(super) fn bf16_benchmark(args: &[String]) -> AppResult<()> {
    let candidates: usize = parse_arg(args, 0, 4, "candidates")?;
    let history: usize = parse_arg(args, 1, 32, "history")?;
    let elements: usize = parse_arg(args, 2, 65_536, "elements")?;
    let iterations: usize = parse_arg(args, 3, 50, "iterations")?;
    bf16_args(candidates, history, elements, iterations)?;
    pair_parity()?;
    let run = measure_bf16(candidates, history, elements, iterations)?;
    let mut point = String::new();
    run.write_json(&mut point);
    let target = device_target().map_err(io::Error::other)?;
    println!(
        "{{\"schema\":\"baux.cuda.bf16-bo-ask.v1\",\"target\":\"{target}\",\"storage\":\"bf16\",\"\
         distance_fusion\":\"candidate-pairs\",\"fused_pair_parity\":true,\"measurement\":{point},\
         \"context_tokens\":0,\"drafted_tokens\":0,\"evaluated_tokens\":0,\"repaired_tokens\":0,\"\
         generated_tokens\":0,\"committed_tokens\":0,\"full_generation_loop\":false}}"
    );
    Ok(())
}

pub(super) fn bf16_curve(args: &[String]) -> AppResult<()> {
    let candidates: usize = parse_arg(args, 0, 4, "candidates")?;
    let history: usize = parse_arg(args, 1, 32, "history")?;
    let iterations: usize = parse_arg(args, 2, 50, "iterations")?;
    const ELEMENTS: [usize; 3] = [4_096, 65_536, 1_048_576];
    for &elements in &ELEMENTS {
        bf16_args(candidates, history, elements, iterations)?;
    }
    pair_parity()?;
    let mut points = String::new();
    for (index, elements) in ELEMENTS.into_iter().enumerate() {
        if index != 0 {
            points.push(',');
        }
        measure_bf16(candidates, history, elements, iterations)?.write_json(&mut points);
    }
    let target = device_target().map_err(io::Error::other)?;
    println!(
        "{{\"schema\":\"baux.cuda.bf16-bo-curve.v1\",\"target\":\"{target}\",\"storage\":\"bf16\",\
         \"distance_fusion\":\"candidate-pairs\",\"fused_pair_parity\":true,\"points\":[{points}],\
         \"context_tokens\":0,\"drafted_tokens\":0,\"evaluated_tokens\":0,\"repaired_tokens\":0,\"\
         generated_tokens\":0,\"committed_tokens\":0,\"full_generation_loop\":false}}"
    );
    Ok(())
}

fn bf16_args(
    candidates: usize,
    history: usize,
    elements: usize,
    iterations: usize,
) -> AppResult<()> {
    if candidates == 0 || elements == 0 || iterations == 0 {
        return Err(
            io::Error::other("candidates, elements, and iterations must be positive").into(),
        );
    }
    if history == 0 || history > MAX_HISTORY {
        return Err(io::Error::other(format!("history must be in 1..={MAX_HISTORY}")).into());
    }
    Ok(())
}

fn make_seeds(round: usize, candidates: usize) -> Vec<u64> {
    (0..candidates)
        .map(|candidate| {
            0x9e37_79b9_7f4a_7c15_u64.wrapping_mul((round * candidates + candidate + 1) as u64)
        })
        .collect()
}

fn fill_history(
    engine: &mut Bf16SearchEngine,
    candidates: usize,
    history: usize,
    trial_slot: u32,
    config: ResidentAsk,
) -> AppResult<()> {
    let mut resident_history = 1usize;
    while resident_history < history {
        let seeds = make_seeds(resident_history, candidates);
        let history_config = ResidentAsk {
            neighbors: resident_history.min(config.neighbors),
            ..config
        };
        let selections = engine
            .ask(
                0,
                resident_history,
                &[trial_slot],
                &seeds,
                candidates,
                0.125,
                0x5eed_u64.wrapping_add(resident_history as u64),
                history_config,
            )
            .map_err(|error| io::Error::other(format!("BF16 benchmark history ask: {error}")))?;
        if selections.len() != 1 || selections[0].index as usize >= candidates {
            return Err(io::Error::other("BF16 benchmark history selection is invalid").into());
        }
        let outcome = ((resident_history.wrapping_mul(37) % 101) as f32 - 50.0) / 25.0;
        let summary = engine
            .tell(&[trial_slot], &[outcome], &[0.0], history, history)
            .map_err(|error| io::Error::other(format!("BF16 benchmark history tell: {error}")))?;
        resident_history = summary.history;
    }
    if resident_history != history {
        return Err(io::Error::other("BF16 benchmark did not fill resident history").into());
    }
    Ok(())
}

fn measure_bf16(
    candidates: usize,
    history: usize,
    elements: usize,
    iterations: usize,
) -> AppResult<Bf16Run> {
    let one = (1.0_f32.to_bits() >> 16) as u16;
    let base = vec![one; elements];
    let leaves = [Bf16Leaf {
        key: 0x6279_6573_6961_6e_u64,
        offset: 0,
        length: elements as u64,
        scale: 0.0078125,
        weight: 1.0,
    }];
    let slots = history
        .checked_add(2)
        .ok_or_else(|| io::Error::other("BF16 benchmark slot count overflow"))?;
    let trial_slot = (history + 1) as u32;
    let mut engine = Bf16SearchEngine::new(&base, &leaves, slots)
        .map_err(|error| io::Error::other(format!("BF16 benchmark init: {error}")))?;
    engine
        .copy_row(0, 1)
        .map_err(|error| io::Error::other(format!("BF16 benchmark anchor: {error}")))?;
    engine
        .init_search(0.0, 0.0, history, 0.125, 0.001, 1.0)
        .map_err(|error| io::Error::other(format!("BF16 benchmark search state: {error}")))?;
    let config = ResidentAsk {
        neighbors: history.min(8),
        acquisition: 0,
        epistemic_scale: 0.7,
        aleatoric_scale: 0.05,
        y_scale: 1.0,
        beta: 1.2,
        adaptive_neighbors: true,
    };
    fill_history(&mut engine, candidates, history, trial_slot, config)?;

    let seeds = make_seeds(history + 1, candidates);
    engine.set_profiling(true);
    for warmup in 0..3 {
        let selections = engine
            .ask(
                0,
                history,
                &[trial_slot],
                &seeds,
                candidates,
                0.125,
                0xcafe_u64.wrapping_add(warmup),
                config,
            )
            .map_err(|error| io::Error::other(format!("BF16 benchmark warmup: {error}")))?;
        if selections.len() != 1 || selections[0].index as usize >= candidates {
            return Err(io::Error::other("BF16 benchmark warmup selection is invalid").into());
        }
    }

    let started = Instant::now();
    let mut totals = [0.0_f64; 4];
    for iteration in 0..iterations {
        let selections = engine
            .ask(
                0,
                history,
                &[trial_slot],
                &seeds,
                candidates,
                0.125,
                0xface_u64.wrapping_add(iteration as u64),
                config,
            )
            .map_err(|error| io::Error::other(format!("BF16 benchmark ask: {error}")))?;
        if selections.len() != 1
            || selections[0].index as usize >= candidates
            || !selections[0].score.is_finite()
        {
            return Err(io::Error::other("BF16 benchmark selection is invalid").into());
        }
        let profile = engine
            .last_profile()
            .ok_or_else(|| io::Error::other("BF16 benchmark profile was not recorded"))?;
        totals[0] += f64::from(profile.score_ms);
        totals[1] += f64::from(profile.pick_ms);
        totals[2] += f64::from(profile.materialize_ms);
        totals[3] += f64::from(profile.total_ms);
    }
    let wall_seconds = started.elapsed().as_secs_f64();
    let count = iterations as f64;
    Ok(Bf16Run {
        candidates,
        history,
        elements,
        iterations,
        score_ms: totals[0] / count,
        pick_ms: totals[1] / count,
        materialize_ms: totals[2] / count,
        gpu_ms: totals[3] / count,
        wall_ms: wall_seconds * 1_000.0 / count,
    })
}
