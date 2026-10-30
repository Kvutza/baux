use std::io;
use std::time::Instant;

use baux_cuda::{
    Bf16Leaf, Bf16SearchEngine, MAX_HISTORY, RoundConfig as ResidentAsk, device_target, target_name,
};

use super::*;

pub(super) fn benchmark(args: &[String]) -> AppResult<()> {
    if args != ["muon"] {
        return Err(io::Error::other("bench accepts only the 'muon' benchmark").into());
    }
    let record = baux_cuda::muon::measure().map_err(io::Error::other)?;
    println!("{}", baux::artifact::json::pretty_string(&record)?);
    Ok(())
}

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
