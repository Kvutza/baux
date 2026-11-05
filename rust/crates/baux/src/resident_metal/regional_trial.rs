//! Small resident workload for checking the complete multi-start lifecycle.
//! This synthetic experiment is not evidence of training quality or scaling.
use super::*;
use crate::hash::{Prng, prng};
use crate::optimizer::multi_tr::SharingPolicy;
use baux::semantics::optimizer::portfolio::Allocation;
use rand::Rng;
use std::time::Duration;

#[derive(Debug, Clone, Copy)]
pub struct RegionalTrial {
    pub dimensions: usize,
    pub regions: usize,
    pub evaluations: usize,
    pub seed: u64,
    pub allocation: Allocation,
    pub sharing: SharingPolicy,
    pub clustered: bool,
}

pub struct RegionalEvidence {
    pub trace: Vec<(usize, f64)>,
    pub seconds: f64,
    pub proposal_seconds: f64,
    pub objective_seconds: f64,
    pub tell_seconds: f64,
    pub final_sync_seconds: f64,
    pub resident_bytes: u64,
    pub region_evaluations: Vec<usize>,
    pub restarts: usize,
    pub delivered_imports: u64,
    pub device: String,
}

struct Objective {
    runtime: Arc<Runtime>,
    pipeline: ComputePipelineState,
    output: Buffer,
    dimensions: usize,
}

impl Objective {
    fn evaluate(&self, row: &Buffer) -> Result<f32, String> {
        let command = self.runtime.queue.new_command_buffer();
        let encoder = command.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.pipeline);
        encoder.set_buffer(0, Some(row), 0);
        encoder.set_buffer(1, Some(&self.output), 0);
        let count = self.dimensions as u32;
        encoder.set_bytes(2, size_of::<u32>() as u64, (&count as *const u32).cast());
        encoder.dispatch_threads(metal::MTLSize::new(1, 1, 1), metal::MTLSize::new(1, 1, 1));
        encoder.end_encoding();
        finish(command)?;
        let value = read::<f32>(&self.output, 1)[0];
        if !value.is_finite() {
            return Err("nonfinite regional objective".into());
        }
        Ok(value)
    }

    fn start(&self, rng: &mut Prng) -> Result<SearchState, String> {
        let base: Vec<u16> = (0..self.dimensions)
            .map(|_| {
                let value: f32 = rng.gen_range(-0.75..0.75);
                (value.to_bits() >> 16) as u16
            })
            .collect();
        let value = self.evaluate(&self.runtime.buffer_with(&base))?;
        SearchState::new(
            &base,
            value,
            0.0,
            vec![ParamBlock::new(1, 0, base.len(), 0.25, 1.0)?],
            8,
            1,
            TRLengthConfig::new(0.2, 0.005, 0.8),
        )
    }
}

pub fn regional_trial(config: RegionalTrial) -> Result<RegionalEvidence, String> {
    if config.dimensions == 0
        || config.dimensions > 65536
        || config.regions == 0
        || config.regions > 32
        || config.evaluations < config.regions + 4
    {
        return Err("regional trial requires dimensions 1..=65536, regions 1..=32 and at least regions+4 evaluations".into());
    }
    autoreleasepool(|| run_trial(config))
}

fn run_trial(config: RegionalTrial) -> Result<RegionalEvidence, String> {
    let runtime = Runtime::shared()?;
    let objective = Objective {
        pipeline: runtime.precise(
            include_str!("regional_objective.metal"),
            "regional-objective",
            "regional_objective",
        )?,
        output: runtime.buffer::<f32>(1),
        runtime,
        dimensions: config.dimensions,
    };
    let mut starts_rng = prng(config.seed);
    let states = (0..config.regions)
        .map(|_| {
            if config.clustered {
                objective.start(&mut prng(config.seed))
            } else {
                objective.start(&mut starts_rng)
            }
        })
        .collect::<Result<Vec<_>, _>>()?;
    let mut search = RegionalSearch::with_sharing(states, config.allocation, config.sharing)?;
    let mut best = search.best().1;
    let mut trace = vec![(config.regions, best)];
    let mut counts = vec![1; config.regions];
    let mut restarts = 0;
    let start = Instant::now();
    let mut proposal_time = Duration::ZERO;
    let mut objective_time = Duration::ZERO;
    let mut tell_time = Duration::ZERO;
    for evaluation in config.regions..config.evaluations {
        if evaluation == config.evaluations / 2 {
            let slot = (search.best().0 + 1) % config.regions;
            search.restart(slot, objective.start(&mut starts_rng)?)?;
            counts[slot] += 1;
            restarts += 1;
        } else {
            let mut ask = RoundConfig::default();
            ask.surrogate.neighbors = 1;
            ask.acquisition.seed = config.seed.wrapping_add(evaluation as u64);
            let phase = Instant::now();
            let proposal = search.ask(crate::hash::splitmix64(ask.acquisition.seed), ask)?;
            proposal_time += phase.elapsed();
            let phase = Instant::now();
            let value = objective.evaluate(&search.buffer(&proposal)?)?;
            objective_time += phase.elapsed();
            let phase = Instant::now();
            search.tell(&proposal, value, 0.0)?;
            tell_time += phase.elapsed();
            counts[proposal.ticket().region.slot] += 1;
        }
        best = best.max(search.best().1);
        trace.push((evaluation + 1, best));
    }
    let phase = Instant::now();
    for slot in 0..config.regions {
        search.flush_sharing(slot)?;
    }
    let final_sync_time = phase.elapsed();
    Ok(RegionalEvidence {
        trace,
        seconds: start.elapsed().as_secs_f64(),
        proposal_seconds: proposal_time.as_secs_f64(),
        objective_seconds: objective_time.as_secs_f64(),
        tell_seconds: tell_time.as_secs_f64(),
        final_sync_seconds: final_sync_time.as_secs_f64(),
        resident_bytes: search.memory_bytes(),
        region_evaluations: counts,
        restarts,
        delivered_imports: search.delivered_imports(),
        device: objective.runtime.info().name.clone(),
    })
}
