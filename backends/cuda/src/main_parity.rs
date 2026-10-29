use std::io;

use baux_cuda::{
    Bf16Leaf, Bf16SearchEngine, Leaf as ResidentLeaf, RoundConfig as ResidentAsk,
    Tile as ResidentTile, TrialEngine, device_target, embed_parity, feedback_parity, target_name,
};
use baux_cuda_device::{Bf16Score, SearchState, trials};
use cuda_core::{CudaContext, CudaStream, DeviceBuffer, LaunchConfig1D};

use super::*;

pub(super) fn resident_parity() -> AppResult<()> {
    let leaves = vec![
        Leaf::new(0, 257, 4, 0.125, 0.75)?,
        Leaf::new(257, 259, 8, 0.03125, 0.5)?,
    ];
    let base = make_base(row_bytes(&leaves));
    let length = 0.8;
    let steps = resident_steps(&leaves, length)?;
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
    let mut engine = TrialEngine::new(&base, &steps, &tiles, 8)
        .map_err(|error| io::Error::other(format!("resident init: {error}")))?;

    let history_seeds = [11_u64, 0x0123_4567_89ab_cdef, 42_u64];
    let mut history_rows = vec![base.clone()];
    for (slot, seed) in history_seeds.into_iter().enumerate() {
        engine
            .materialize(0, slot + 1, seed, &steps)
            .map_err(|error| io::Error::other(format!("resident history write: {error}")))?;
        let actual = engine
            .read(slot + 1)
            .map_err(|error| io::Error::other(format!("resident history read: {error}")))?;
        let expected = expected_row(&base, &leaves, length, seed)?;
        compare_rows(&expected, &actual, seed, length)?;
        history_rows.push(expected);
    }

    let history_slots = [0_u32, 1, 2, 3];
    let outcomes = [-0.75_f32, 1.25, 0.5, 2.0];
    let seeds = [3_u64, 17, 0xdead_beef_cafe_babe, u64::MAX - 9, 99, 1001];
    let draws = [0.1_f32, -0.5, 0.8, -0.2];

    for acquisition in [0_u32, 1, 2] {
        let config = ResidentAsk {
            neighbors: 3,
            acquisition,
            epistemic_scale: 0.7,
            aleatoric_scale: 0.05,
            y_scale: 1.0,
            beta: 1.2,
            adaptive_neighbors: false,
        };
        let expected_scores = seeds
            .iter()
            .map(|&seed| {
                cpu_score(
                    &base,
                    &history_rows,
                    &outcomes,
                    &leaves,
                    &steps,
                    seed,
                    &draws,
                    config,
                )
            })
            .collect::<Vec<_>>();
        let expected_choice = expected_scores
            .iter()
            .enumerate()
            .max_by(|(left_index, left), (right_index, right)| {
                left.total_cmp(right)
                    .then_with(|| right_index.cmp(left_index))
            })
            .map(|(index, _)| index)
            .expect("resident parity has candidates");
        let (choice, scores) = engine
            .ask_scores(
                0,
                &history_slots,
                &outcomes,
                4,
                &seeds,
                &draws,
                &steps,
                config,
                true,
            )
            .map_err(|error| io::Error::other(format!("resident ask: {error}")))?;
        if choice != expected_choice {
            return Err(io::Error::other(format!(
                "resident choice mismatch for acq={acquisition}: expected {expected_choice}, got \
                 {choice}; expected_scores={expected_scores:?}, scores={scores:?}"
            ))
            .into());
        }
        for (index, (&expected, &actual)) in expected_scores.iter().zip(&scores).enumerate() {
            let tolerance = 2.0e-5 * expected.abs().max(1.0);
            if !actual.is_finite() || (expected - actual).abs() > tolerance {
                return Err(io::Error::other(format!(
                    "resident score {index} mismatch for acq={acquisition}: expected {expected}, \
                     got {actual}, tolerance={tolerance}"
                ))
                .into());
            }
        }
        let actual = engine
            .read(4)
            .map_err(|error| io::Error::other(format!("resident trial read: {error}")))?;
        let expected = expected_row(&base, &leaves, length, seeds[choice])?;
        compare_rows(&expected, &actual, seeds[choice], length)?;

        if acquisition == 1 {
            check_thompson(
                &mut engine,
                &history_slots,
                &outcomes,
                &seeds,
                &draws,
                &steps,
                config,
                &scores,
            )?;
        }
    }

    bf16_scoreparity()?;
    let target = device_target().map_err(io::Error::other)?;
    println!(
        "RESIDENT ok=true target={target} history={} candidates={} choice=passed bf16=passed",
        history_rows.len(),
        seeds.len()
    );
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn check_thompson(
    engine: &mut TrialEngine,
    history_slots: &[u32],
    outcomes: &[f32],
    seeds: &[u64],
    draws: &[f32],
    steps: &[ResidentLeaf],
    config: ResidentAsk,
    scores: &[f32],
) -> AppResult<()> {
    let mut orders = vec![vec![5, 4, 3, 2, 1, 0], vec![2, 3, 4, 5, 0, 1]];
    for index in 0..seeds.len() {
        orders.push(vec![index]);
        orders.push(vec![index; 3]);
    }
    // Reuse the same observations and draws, including when history exceeds
    // the candidate count. Trial slot 4 is outside the unchanged history.
    for order in orders {
        let candidates = order.iter().map(|&index| seeds[index]).collect::<Vec<_>>();
        let (actual_choice, actual_scores) = engine
            .ask_scores(
                0,
                history_slots,
                outcomes,
                4,
                &candidates,
                draws,
                steps,
                config,
                false,
            )
            .map_err(|error| io::Error::other(format!("resident coherent ask: {error}")))?;
        let expected_choice = (0..order.len()).fold(0, |best, index| {
            if scores[order[index]] > scores[order[best]] {
                index
            } else {
                best
            }
        });
        if actual_scores.len() != order.len()
            || actual_choice != expected_choice
            || actual_scores
                .iter()
                .zip(&order)
                .any(|(&actual, &index)| actual.to_bits() != scores[index].to_bits())
        {
            return Err(io::Error::other(format!(
                "resident Thompson coherence mismatch: order={order:?}, \
                 expected_choice={expected_choice}, actual_choice={actual_choice}, \
                 original_scores={scores:?}, actual_scores={actual_scores:?}"
            ))
            .into());
        }
    }
    Ok(())
}

fn neighbor_reference(
    params: Bf16Score,
    outcomes: &[f32],
    variances: &[f32],
    draws: &[f32],
) -> f64 {
    let y2 = f64::from(params.y_scale).powi(2);
    let mut sum = 0.0;
    let mut noise_sum = 0.0;
    let mut count = 1;
    let mut risk = f64::INFINITY;
    let points = [(1.0, 0), (2.0, 2)];
    for (rank, &(distance, index)) in points.iter().enumerate() {
        let noise = f64::from(params.aleatoric_scale) + f64::from(variances[index]) / y2;
        let precision = 1.0 / (1.0e-9 + f64::from(params.epistemic_scale) * distance + noise);
        sum += precision;
        noise_sum += precision * precision * noise;
        let next = f64::from(params.epistemic_scale) * distance + noise_sum / (sum * sum);
        if next < risk {
            risk = next;
            count = rank + 1;
        }
    }
    let mut sum = 0.0;
    let mut mean = 0.0;
    let mut aleatoric = 0.0;
    let mut sample = 0.0;
    let mut squared = 0.0;
    for &(distance, index) in points.iter().take(count) {
        let noise = f64::from(params.aleatoric_scale) + f64::from(variances[index]) / y2;
        let precision = 1.0 / (1.0e-9 + f64::from(params.epistemic_scale) * distance + noise);
        sum += precision;
        mean += precision * f64::from(outcomes[index]);
        aleatoric += precision * noise;
        sample += precision * f64::from(draws[index]);
        squared += precision * precision;
    }
    let error = ((1.0 + aleatoric) / sum).sqrt() * f64::from(params.y_scale);
    mean / sum
        + error
            * match params.acquisition {
                1 => sample / squared.sqrt(),
                2 => 1.0,
                _ => f64::from(params.beta),
            }
}

fn bf16_scoreparity() -> AppResult<()> {
    let context = CudaContext::new(0)?;
    let stream = context.default_stream();
    // SAFETY: the generated bindings load the matching embedded kernel
    // artifact.
    let module = unsafe { trials::load(&context) }?;
    let params = Bf16Score {
        row_stride: 1,
        coefficient: 1.0,
        epistemic_scale: 0.7,
        aleatoric_scale: 0.05,
        y_scale: 1.7,
        beta: 1.2,
        history: 3,
        candidates: 2,
        base_slot: 0,
        neighbors: 2,
        acquisition: 1,
        tiles: 1,
        resident: 0,
        correlated: 0,
        adaptive_neighbors: 0,
    };
    let outcomes = [0.5_f32, 10.0, -0.25];
    let variances = [8.0_f32, 0.5, 0.125];
    let draws = [0.1_f32, -0.5, 0.8];
    // Duplicate queries select observations 0 and 2. Observation 0 is nearer,
    // but observation 2 has the strongest weight because its variance is lower.
    let partials = DeviceBuffer::from_host(&stream, &[1.0_f32, 9.0, 2.0, 1.0, 9.0, 2.0])?;
    let outcomes_device = DeviceBuffer::from_host(&stream, &outcomes)?;
    let variances_device = DeviceBuffer::from_host(&stream, &variances)?;
    let draws_device = DeviceBuffer::from_host(&stream, &draws)?;
    let state = DeviceBuffer::<SearchState>::zeroed(&stream, 1)?;
    let tile_status = DeviceBuffer::<u32>::zeroed(&stream, 2)?;
    let mut scores = DeviceBuffer::<f32>::zeroed(&stream, 2)?;
    let launch = module.prepare_score_bf16(LaunchConfig1D::new(2, 256, 0))?;
    module.score_bf16(
        &stream,
        &launch,
        &partials,
        &tile_status,
        &outcomes_device,
        &variances_device,
        &state,
        &draws_device,
        &mut scores,
        params,
    )?;
    let actual = scores.to_host_vec(&stream)?;

    let mut weight_sum = 0.0_f64;
    let mut weighted_value = 0.0_f64;
    let mut weighted_noise = 0.0_f64;
    let mut squared_weight_sum = 0.0_f64;
    let mut weighted_aleatoric = 0.0_f64;
    for (distance, index) in [(1.0, 0), (2.0, 2)] {
        let weight = 1.0
            / (1.0e-9
                + f64::from(params.epistemic_scale) * distance
                + f64::from(params.aleatoric_scale)
                + f64::from(variances[index]) / f64::from(params.y_scale).powi(2));
        weight_sum += weight;
        weighted_value += weight * f64::from(outcomes[index]);
        weighted_noise += weight * f64::from(draws[index]);
        squared_weight_sum += weight * weight;
        weighted_aleatoric += weight
            * (f64::from(params.aleatoric_scale)
                + f64::from(variances[index]) / f64::from(params.y_scale).powi(2));
    }
    let se = f64::from(params.y_scale) * ((1.0 + weighted_aleatoric) / weight_sum).sqrt();
    let expected = weighted_value / weight_sum + se * weighted_noise / squared_weight_sum.sqrt();
    if actual[0].to_bits() != actual[1].to_bits()
        || actual.iter().any(|&score| {
            !score.is_finite()
                || (f64::from(score) - expected).abs() > 2.0e-5 * expected.abs().max(1.0)
        })
    {
        return Err(io::Error::other(format!(
            "BF16 Thompson score mismatch: expected {expected}, duplicate scores={actual:?}"
        ))
        .into());
    }

    check_adaptive(
        &module,
        &stream,
        &partials,
        &tile_status,
        &outcomes_device,
        &variances_device,
        &state,
        &draws_device,
        &mut scores,
        params,
        &outcomes,
        &variances,
    )?;
    context.check_err()?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn check_adaptive(
    module: &trials::LoadedModule,
    stream: &CudaStream,
    partials: &DeviceBuffer<f32>,
    tile_status: &DeviceBuffer<u32>,
    outcomes_device: &DeviceBuffer<f32>,
    variances_device: &DeviceBuffer<f32>,
    state: &DeviceBuffer<SearchState>,
    draws_device: &DeviceBuffer<f32>,
    scores: &mut DeviceBuffer<f32>,
    params: Bf16Score,
    outcomes: &[f32],
    variances: &[f32],
) -> AppResult<()> {
    let draws = [0.1_f32, -0.5, 0.8];
    let launch = module.prepare_score_bf16(LaunchConfig1D::new(2, 256, 0))?;
    for acquisition in 0..3 {
        let adaptive = Bf16Score {
            acquisition,
            adaptive_neighbors: 1,
            ..params
        };
        module.score_bf16(
            stream,
            &launch,
            partials,
            tile_status,
            outcomes_device,
            variances_device,
            state,
            draws_device,
            scores,
            adaptive,
        )?;
        let actual = scores.to_host_vec(stream)?;
        let expected = neighbor_reference(adaptive, outcomes, variances, &draws);
        if actual.iter().any(|&value| {
            !value.is_finite()
                || (f64::from(value) - expected).abs() > 2.0e-5 * expected.abs().max(1.0)
        }) {
            return Err(io::Error::other(format!(
                "Adaptive ENN acquisition mismatch: {actual:?} vs {expected}"
            ))
            .into());
        }
    }
    println!("NEIGHBOR_RULE parity=true acquisitions=ucb,thompson,pareto adaptive=true");

    let launch = module.prepare_draw_bf16(LaunchConfig1D::new(1, 256, 0))?;
    let mut sampled = DeviceBuffer::<f32>::zeroed(stream, 3)?;
    let mut original = Vec::new();
    for slots in [[11_u32, 3, 27], [27, 11, 3]] {
        let slots = DeviceBuffer::from_host(stream, &slots)?;
        module.draw_bf16(stream, &launch, &mut sampled, &slots, state, 42, 3, 0)?;
        let actual = sampled.to_host_vec(stream)?;
        if original.is_empty() {
            original = actual;
        } else if actual != [original[2], original[0], original[1]]
            || actual.iter().any(|draw| !draw.is_finite())
            || actual.iter().all(|&draw| draw == actual[0])
        {
            return Err(io::Error::other(format!(
                "BF16 Thompson draw slot mismatch: original={original:?}, reordered={actual:?}"
            ))
            .into());
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn cpu_score(
    base: &[u8],
    history_rows: &[Vec<u8>],
    outcomes: &[f32],
    leaves: &[Leaf],
    steps: &[ResidentLeaf],
    seed: u64,
    draws: &[f32],
    config: ResidentAsk,
) -> f32 {
    let mut nearest = vec![(f32::INFINITY, 0_usize); config.neighbors];
    for (history_index, observation) in history_rows.iter().enumerate() {
        let distance = cpu_distance(base, observation, leaves, steps, seed);
        let insert_at = nearest.iter().position(|&(other, other_index)| {
            distance < other || (distance == other && history_index < other_index)
        });
        if let Some(insert_at) = insert_at {
            nearest.insert(insert_at, (distance, history_index));
            nearest.pop();
        }
    }
    let mut weight_sum = 0.0_f64;
    let mut weighted_value = 0.0_f64;
    let mut weighted_noise = 0.0_f64;
    let mut weight_squared_sum = 0.0_f64;
    for &(distance, history_index) in &nearest {
        let variance = 1.0e-9
            + f64::from(config.epistemic_scale) * f64::from(distance)
            + f64::from(config.aleatoric_scale);
        let weight = 1.0 / variance.max(1.0e-12);
        weight_sum += weight;
        weighted_value += weight * f64::from(outcomes[history_index]);
        if config.acquisition == 1 {
            weighted_noise += weight * f64::from(draws[history_index]);
            weight_squared_sum += weight * weight;
        }
    }
    let mean = weighted_value / weight_sum.max(1.0e-12);
    let se = (1.0 / weight_sum.max(1.0e-12)).sqrt() * f64::from(config.y_scale);
    let score = match config.acquisition {
        1 => mean + se * weighted_noise / weight_squared_sum.sqrt(),
        2 => mean + se,
        _ => mean + f64::from(config.beta) * se,
    };
    score as f32
}

fn cpu_distance(
    base: &[u8],
    observation: &[u8],
    leaves: &[Leaf],
    steps: &[ResidentLeaf],
    seed: u64,
) -> f32 {
    let mut distance = 0.0_f32;
    for (&leaf, &step) in leaves.iter().zip(steps) {
        for element in 0..leaf.length {
            let byte =
                step.byte_offset as usize + if leaf.bits == 4 { element / 2 } else { element };
            let shift = if leaf.bits == 4 { (element & 1) * 4 } else { 0 };
            let mask = if leaf.bits == 4 { 0x0f } else { 0xff };
            let code = u32::from((base[byte] >> shift) & mask);
            let candidate = cpu_perturb(
                code,
                seed,
                leaf.offset as u32 + element as u32,
                step.bits,
                step.whole,
                step.threshold,
            );
            let observed = u32::from((observation[byte] >> shift) & mask);
            let delta = (candidate as f32 - observed as f32) * step.scale;
            distance = delta.mul_add(delta * step.weight, distance);
        }
    }
    distance
}

pub(super) fn parity() -> AppResult<()> {
    baux_cuda::proposal_check().map_err(io::Error::other)?;
    baux_cuda::neighbors::parity().map_err(io::Error::other)?;
    let context = CudaContext::new(0)?;
    let stream = context.default_stream();
    // SAFETY: the generated bindings load the matching embedded kernel
    // artifact.
    let module = unsafe { trials::load(&context) }?;
    let layouts = [
        vec![Leaf::new(0, 257, 4, 0.125, 0.75)?],
        vec![Leaf::new(0, 259, 8, 0.03125, 0.5)?],
        vec![
            Leaf::new(0, 257, 4, 0.125, 0.75)?,
            Leaf::new(257, 259, 8, 0.03125, 0.5)?,
        ],
    ];
    let mut cases = 0;
    for leaves in layouts {
        let base = make_base(row_bytes(&leaves));
        for length in [0.0, 0.8, 8.0] {
            for seed in [0_u64, 1, 0x0123_4567_89ab_cdef, u64::MAX] {
                check_case(&context, &stream, &module, &base, &leaves, length, seed)?;
                cases += 1;
            }
        }
    }
    embed_parity().map_err(io::Error::other)?;
    feedback_parity().map_err(io::Error::other)?;
    let target = target_name(&context).map_err(io::Error::other)?;
    println!("PARITY ok=true cases={cases} target={target}");
    Ok(())
}

pub(super) fn pair_parity() -> AppResult<()> {
    const ELEMENTS: usize = 4_097;
    const HISTORY: usize = 4;
    const CANDIDATES: usize = 4;
    const TRIAL_SLOT: u32 = (HISTORY + 1) as u32;

    let one = (1.0_f32.to_bits() >> 16) as u16;
    let base = vec![one; ELEMENTS];
    let leaves = [Bf16Leaf {
        key: 0x7061_6972_7465_7374,
        offset: 0,
        length: ELEMENTS as u64,
        scale: 0.0078125,
        weight: 1.0,
    }];
    let build = || -> Result<Bf16SearchEngine, String> {
        let mut engine = Bf16SearchEngine::new(&base, &leaves, HISTORY + 2)?;
        engine.copy_row(0, 1)?;
        engine.init_search(0.0, 0.0, HISTORY, 0.125, 0.001, 1.0)?;
        Ok(engine)
    };
    let mut baseline = build().map_err(io::Error::other)?;
    baseline.set_fused_pairs(false);
    let mut fused = build().map_err(io::Error::other)?;
    fused.set_fused_pairs(true);

    let mut resident_history = 1usize;
    for round in 1..=HISTORY {
        let seeds = (0..CANDIDATES)
            .map(|candidate| {
                0x9e37_79b9_7f4a_7c15_u64.wrapping_mul((round * CANDIDATES + candidate + 1) as u64)
            })
            .collect::<Vec<_>>();
        let config = ResidentAsk {
            neighbors: resident_history.min(3),
            acquisition: 0,
            epistemic_scale: 0.7,
            aleatoric_scale: 0.05,
            y_scale: 1.0,
            beta: 1.2,
            adaptive_neighbors: false,
        };
        let baseline_selection = baseline
            .ask(
                0,
                resident_history,
                &[TRIAL_SLOT],
                &seeds,
                CANDIDATES,
                0.125,
                0x600d_u64 + round as u64,
                config,
            )
            .map_err(|error| io::Error::other(format!("BF16 pair baseline parity: {error}")))?;
        let fused_selection = fused
            .ask(
                0,
                resident_history,
                &[TRIAL_SLOT],
                &seeds,
                CANDIDATES,
                0.125,
                0x600d_u64 + round as u64,
                config,
            )
            .map_err(|error| io::Error::other(format!("BF16 pair fused parity: {error}")))?;
        if baseline_selection.len() != 1
            || fused_selection.len() != 1
            || baseline_selection[0].index != fused_selection[0].index
            || baseline_selection[0].score.to_bits() != fused_selection[0].score.to_bits()
        {
            return Err(io::Error::other(format!(
                "BF16 fused-pair selection mismatch at round {round}: \
                 baseline={baseline_selection:?}, fused={fused_selection:?}"
            ))
            .into());
        }
        let baseline_row = baseline
            .read(TRIAL_SLOT as usize)
            .map_err(io::Error::other)?;
        let fused_row = fused.read(TRIAL_SLOT as usize).map_err(io::Error::other)?;
        if baseline_row != fused_row {
            return Err(io::Error::other(format!(
                "BF16 fused-pair materialization mismatch at round {round}"
            ))
            .into());
        }
        if round < HISTORY {
            let outcome = (round as f32 - 2.0) * 0.25;
            let baseline_summary = baseline
                .tell(&[TRIAL_SLOT], &[outcome], &[0.0], HISTORY, HISTORY)
                .map_err(io::Error::other)?;
            let fused_summary = fused
                .tell(&[TRIAL_SLOT], &[outcome], &[0.0], HISTORY, HISTORY)
                .map_err(io::Error::other)?;
            if baseline_summary.accepted != fused_summary.accepted
                || baseline_summary.history != fused_summary.history
                || baseline_summary.length.to_bits() != fused_summary.length.to_bits()
                || baseline_summary.best.to_bits() != fused_summary.best.to_bits()
            {
                return Err(io::Error::other(format!(
                    "BF16 fused-pair tell mismatch at round {round}"
                ))
                .into());
            }
            resident_history = baseline_summary.history;
        }
    }
    Ok(())
}
