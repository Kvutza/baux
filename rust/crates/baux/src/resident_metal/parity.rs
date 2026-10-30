use super::*;
use crate::params::ENNParams;
use crate::procedural_pool::ProceduralPoolConfig;
use crate::reliability_region::{
    ReliabilityAction as MetalAction, ReliabilityController, ReliabilityEvidence, ReliabilityPolicy,
};
use crate::trust_region::TRLengthConfig;
use baux::semantics::optimizer::{
    INITIAL_OBSERVATIONS, NeighborValueEstimator, PoolShape, RadiusController, ReliabilityAction,
};
use ndarray::Array2;
use rand::{SeedableRng, rngs::StdRng};

#[test]
fn rounded_proposal() {
    autoreleasepool(|| {
        let base = vec![0x3c00; 1025];
        let ask = {
            let mut config = RoundConfig::default();
            config.surrogate.neighbors = 1;
            config
        };
        let mut state = SearchState::new_implicit(
            &base,
            vec![ParamBlock::new(1, 0, base.len(), 0.25, 1.0).unwrap()],
            1,
            TRLengthConfig::new(1e-20, 1e-21, 0.4),
            ProposalKernel::Gaussian,
        )
        .unwrap();
        state
            .configure_enn(crate::config::ResidentEnnConfig {
                pool: ProceduralPoolConfig {
                    arms: 1,
                    candidates_per_arm: 1,
                }
                .resolve_resident()
                .unwrap(),
                proposal_method: crate::procedural_pool::ProposalMethod::Additive,
                ask,
                num_candidates: 4,
                num_samples: 4,
                fit_params: false,
                adapt_neighbors: false,
                distance_scaling: crate::config::DistanceScaling::Global,
                history_geometry: crate::config::HistoryGeometry::Latent,
                local_scale_neighbors: 8,
            })
            .unwrap();
        state.observe_initial(0.0, 0.0).unwrap();
        let error = match state.propose(17, ask, 0, None) {
            Ok(_) => panic!("rounded-away Metal proposal was published"),
            Err(error) => error,
        };
        assert_eq!(error, "selected proposal did not change");
        assert_eq!(read::<u16>(&state.proposal, base.len()), base);
        assert_eq!(state.read_best().unwrap(), base);
        assert_eq!(state.history_len().unwrap(), 1);
    });
}

#[test]
fn posterior_parity() {
    use crate::apple_gpu::thread_group;
    use baux::semantics::neighbor::query::Query;
    autoreleasepool(|| {
        let base = vec![0x3c00; 17];
        let mut state = SearchState::new_implicit(
            &base,
            vec![ParamBlock::new(71, 0, base.len(), 0.25, 1.0).unwrap()],
            1,
            TRLengthConfig::new(0.1, 0.001, 0.4),
            ProposalKernel::Gaussian,
        )
        .unwrap();
        let source = initialization::search_source(true, ProposalKernel::Gaussian);
        let pipeline = state
            .runtime
            .precise(&source, "Posterior parity", "bf16_posterior_probe")
            .unwrap();
        let mut random = 73_u64;
        let mut next = || {
            random = crate::hash::splitmix64(random);
            (random >> 40) as f32 / 16_777_216.0
        };
        for len in [1, 2, 10, 32, MAX_HISTORY] {
            state.history = len;
            for i in 0..len {
                state.outcomes[i] = 8.0 * next() - 4.0;
                state.variances[i] = 0.01 + next();
                state.identities[i] = i as i64 + 1;
            }
            for adaptive in [false, true] {
                for thompson in [false, true] {
                    state.adapt_neighbors = adaptive;
                    let ask = {
                        let mut config = RoundConfig::default();
                        config.surrogate.neighbors = 10;
                        config.surrogate.epistemic_scale = 0.25;
                        config.surrogate.aleatoric_scale = 0.1;
                        config.surrogate.output_scale = 2.0;
                        config.acquisition.beta = 3.0;
                        config.acquisition.kind = if thompson {
                            AcquisitionKind::Thompson
                        } else {
                            AcquisitionKind::Ucb
                        };
                        config
                    };
                    let params = state.select_params(73, ask, None);
                    let distances = (0..len)
                        .map(|_| ((next() * 8.0).floor() * 0.125) as f64)
                        .collect::<Vec<_>>();
                    for excluded in [None, Some(0)] {
                        let expected = Query {
                            distances: &distances,
                            outcomes: &params.outcomes[..len],
                            variances: &params.variances[..len],
                            draws: thompson.then_some(&params.draws[..len]),
                            neighbors: ask.surrogate.neighbors,
                            epistemic: ask.surrogate.epistemic_scale,
                            aleatoric: ask.surrogate.aleatoric_scale,
                            scale: ask.surrogate.output_scale,
                            beta: ask.acquisition.beta,
                            excluded,
                            incumbent: params.incumbent_index as usize,
                            adaptive,
                        }
                        .predict()
                        .unwrap();
                        let dist_f32: Vec<f32> = distances.iter().map(|&d| d as f32).collect();
                        let input = state.runtime.buffer_with(&dist_f32);
                        let output = state.runtime.buffer::<f32>(4);
                        let command = state.runtime.queue.new_command_buffer();
                        let encoder = command.new_compute_command_encoder();
                        encoder.set_compute_pipeline_state(&pipeline);
                        encoder.set_bytes(
                            0,
                            size_of::<SelectionParams>() as u64,
                            std::ptr::from_ref(&params).cast(),
                        );
                        encoder.set_buffer(1, Some(&input), 0);
                        encoder.set_buffer(2, Some(&output), 0);
                        let omitted = excluded.map_or(MAX_HISTORY as u32, |i| i as u32);
                        encoder.set_bytes(3, 4, std::ptr::from_ref(&omitted).cast());
                        encoder.dispatch_threads(thread_group(1), thread_group(1));
                        encoder.end_encoding();
                        finish(command).unwrap();
                        let actual = read::<f32>(&output, 4);
                        assert_eq!(actual[3] as usize, expected.neighbors);
                        for (actual, expected) in actual[..3].iter().zip([
                            expected.score,
                            expected.mean,
                            expected.standard_error,
                        ]) {
                            assert!(
                                (actual - expected).abs()
                                    <= 8.0 * f32::EPSILON * expected.abs().max(1.0),
                                "history={len} adaptive={adaptive} thompson={thompson} \
                                 excluded={excluded:?}: {actual} != {expected}"
                            );
                        }
                    }
                }
            }
        }
    });
}

#[test]
fn pool_parity() {
    for arms in [1, 2, 4] {
        let host = PoolShape::new(arms, 4).unwrap();
        let metal = ProceduralPoolConfig {
            arms: arms as u32,
            candidates_per_arm: 4 / arms as u32,
        }
        .resolve()
        .unwrap();
        for candidate in 0..4 {
            let identity = metal.identity(candidate).unwrap();
            assert_eq!(
                host.seed(73, candidate as usize),
                metal.seed(73, identity).unwrap()
            );
        }
    }
}

#[test]
fn fit_parity() {
    let fit_seed = 0x1234_5678_9abc_def0_u64;
    let mut host = NeighborValueEstimator::configured(-0.4, 0.002, fit_seed, true, false).unwrap();
    let mut matrix = [[0.0; INITIAL_OBSERVATIONS]; INITIAL_OBSERVATIONS];
    let mut outcomes = [-0.4; INITIAL_OBSERVATIONS];
    let mut variances = [0.002; INITIAL_OBSERVATIONS];
    for row in 1..INITIAL_OBSERVATIONS {
        let distances = (0..row)
            .map(|column| {
                let delta = (row as f64 - column as f64) * 0.03125;
                delta * delta + ((row * 13 + column * 7) % 11) as f64 * 0.0001
            })
            .collect::<Vec<_>>();
        let value = (row as f32 * 0.37).sin() * 0.2 - 0.3;
        let variance = 0.001 + (row % 5) as f32 * 0.0002;
        for (column, &distance) in distances.iter().enumerate() {
            matrix[row][column] = distance;
            matrix[column][row] = distance;
        }
        outcomes[row] = value;
        variances[row] = variance;
        host.append(&distances, value, variance, row % 7 == 0)
            .unwrap();
    }

    let distances = Array2::from_shape_fn(
        (INITIAL_OBSERVATIONS, INITIAL_OBSERVATIONS),
        |(row, column)| matrix[row][column],
    );
    let outcomes = Array2::from_shape_fn((INITIAL_OBSERVATIONS, 1), |(row, _)| {
        f64::from(outcomes[row])
    });
    let variances = Array2::from_shape_fn((INITIAL_OBSERVATIONS, 1), |(row, _)| {
        f64::from(variances[row])
    });
    let mut metal = ENNFitter::new(INITIAL_OBSERVATIONS as i32, true);
    metal.set_params(ENNParams::new(INITIAL_OBSERVATIONS as i32, 0.05, 0.05).unwrap());
    let mut rng = StdRng::seed_from_u64(fit_seed.wrapping_add(INITIAL_OBSERVATIONS as u64));
    let expected = metal
        .ask_distances(
            &distances.view(),
            &outcomes.view(),
            Some(&variances.view()),
            4,
            4,
            None,
            &mut rng,
        )
        .unwrap();
    let actual = host.fit_state();
    assert_eq!(actual.neighbors, expected.k_neighbors as usize);
    assert_eq!(
        actual.epistemic.to_bits(),
        expected.epistemic_scale.to_bits()
    );
    assert_eq!(
        actual.aleatoric.to_bits(),
        expected.aleatoric_scale.to_bits()
    );
    assert_eq!(actual.scale.to_bits(), metal.y_std()[0].to_bits());
}

#[test]
fn fitting_trace() {
    for (k, candidates, samples) in [(10, 30, 10), (3, 7, 4), (32, 4, 4)] {
        let seed = 0x1234_5678_9abc_def0;
        let spec = baux::semantics::optimizer::SurrogateSpec {
            neighbors: k,
            fit_candidates: candidates,
            fit_samples: samples,
            fit: true,
            ..Default::default()
        };
        let mut host = NeighborValueEstimator::with_spec(-0.4, 0.002, seed, spec).unwrap();
        let mut metal = ENNFitter::new(k as i32, true);
        metal.set_params(
            ENNParams::new(
                k as i32,
                f64::from(spec.epistemic),
                f64::from(spec.aleatoric),
            )
            .unwrap(),
        );
        let mut matrix = [[0.0; MAX_HISTORY]; MAX_HISTORY];
        let mut values = vec![-0.4_f32];
        let mut errors = vec![0.002_f32];
        for row in 1..(2 * MAX_HISTORY + 17) {
            let slot = host.next_slot().unwrap();
            let distances = host
                .identities()
                .iter()
                .map(|&identity| {
                    let delta = (row as f64 - (identity - 1) as f64) * 0.03125;
                    delta * delta + ((row * 13 + identity as usize * 7) % 11) as f64 * 0.0001
                })
                .collect::<Vec<_>>();
            for (col, &d) in distances.iter().enumerate() {
                matrix[slot][col] = d;
                matrix[col][slot] = d;
            }
            matrix[slot][slot] = 0.0;
            let value = (row as f32 * 0.37).sin() * 0.2 - 0.3;
            let variance = 0.001 + (row % 5) as f32 * 0.0002;
            if slot == values.len() {
                values.push(value);
                errors.push(variance);
            } else {
                values[slot] = value;
                errors[slot] = variance;
            }
            host.append(&distances, value, variance, row % 7 == 0)
                .unwrap();
            let n = host.len();
            if n < k {
                continue;
            }
            let distances = Array2::from_shape_fn((n, n), |(r, c)| matrix[r][c]);
            let outcomes = Array2::from_shape_fn((n, 1), |(r, _)| f64::from(values[r]));
            let variances = Array2::from_shape_fn((n, 1), |(r, _)| f64::from(errors[r]));
            let mut random = StdRng::seed_from_u64(seed.wrapping_add(n as u64));
            let expected = metal
                .ask_distances(
                    &distances.view(),
                    &outcomes.view(),
                    Some(&variances.view()),
                    candidates,
                    samples,
                    None,
                    &mut random,
                )
                .unwrap();
            let actual = host.fit_state();
            assert_eq!(actual.neighbors, expected.k_neighbors as usize);
            assert_eq!(
                actual.epistemic.to_bits(),
                expected.epistemic_scale.to_bits(),
                "k={k} n={n}"
            );
            assert_eq!(
                actual.aleatoric.to_bits(),
                expected.aleatoric_scale.to_bits(),
                "k={k} n={n}"
            );
            assert_eq!(
                actual.scale.to_bits(),
                metal.y_std()[0].to_bits(),
                "k={k} n={n}"
            );
        }
    }
}

#[test]
fn radius_parity() {
    let mut host = RadiusController::new(0.1, 0.1, 2.0).unwrap();
    let mut metal = ReliabilityController::new(
        ReliabilityPolicy::default(),
        TRLengthConfig::new(0.1, 0.1, 2.0),
    )
    .unwrap();
    for step in 0..48 {
        let observed = (step as f32 * 0.017).sin() * 0.2;
        let observed_variance = 0.002 + (step % 5) as f32 * 0.0005;
        let predicted = (step as f32 * 0.011).cos() * 0.15;
        let predicted_variance = 0.003 + (step % 7) as f32 * 0.0004;
        let response_radius = 0.03 + (step % 4) as f32 * 0.01;
        let before = host.radius();
        let actual = host
            .observe(
                observed,
                observed_variance,
                predicted,
                predicted_variance,
                response_radius,
                false,
            )
            .unwrap();
        let expected = metal
            .update(ReliabilityEvidence {
                observed_improvement: f64::from(observed),
                observation_variance: f64::from(observed_variance),
                predicted_improvement: f64::from(predicted),
                prediction_variance: f64::from(predicted_variance),
                accepted: actual.accepted,
                nominal_radius: f64::from(before),
                response_radius: f64::from(response_radius),
            })
            .unwrap();
        assert_eq!(
            actual.action,
            match expected.action {
                MetalAction::Hold => ReliabilityAction::Hold,
                MetalAction::Expand => ReliabilityAction::Expand,
                MetalAction::Shrink => ReliabilityAction::Shrink,
            }
        );
        assert_eq!(
            actual.next_radius.to_bits(),
            (metal.length() as f32).to_bits()
        );
        assert_eq!(
            actual.standardized_error.to_bits(),
            expected.standardized_error.unwrap().to_bits()
        );
        assert_eq!(
            actual.sharpness_ratio.to_bits(),
            expected.sharpness_ratio.unwrap().to_bits()
        );
    }
}

/// Exercise Metal's real tell path. Unlike radius_parity, acceptance is not
/// supplied by the other controller: each side independently makes its
/// decision.
#[test]
fn warmup_trace() {
    autoreleasepool(|| {
        let bounds = TRLengthConfig::new(0.125, 0.03125, 2.0);
        let ask = {
            let mut config = RoundConfig::default();
            config.surrogate.neighbors = 10;
            config
        };
        let base = vec![0x3c00; 17];
        let mut state = SearchState::new_implicit(
            &base,
            vec![ParamBlock::new(73, 0, base.len(), 0.25, 1.0).unwrap()],
            1,
            bounds,
            ProposalKernel::Gaussian,
        )
        .unwrap();
        state
            .configure_enn(crate::config::ResidentEnnConfig {
                pool: ProceduralPoolConfig {
                    arms: 1,
                    candidates_per_arm: 1,
                }
                .resolve_resident()
                .unwrap(),
                proposal_method: crate::procedural_pool::ProposalMethod::Additive,
                ask,
                num_candidates: 4,
                num_samples: 4,
                fit_params: false,
                adapt_neighbors: false,
                distance_scaling: crate::config::DistanceScaling::Global,
                history_geometry: crate::config::HistoryGeometry::Latent,
                local_scale_neighbors: 8,
            })
            .unwrap();
        state.observe_initial(1_048_576.0, 0.0).unwrap();
        let host = RadiusController::configured(0.125, 0.03125, 2.0, Default::default()).unwrap();
        for (step, gain) in [0.01, 0.125, -0.5, 0.01, 0.25, -0.125, 0.01, 0.125, -0.25]
            .into_iter()
            .enumerate()
        {
            let (_, round, initializing) = state.propose(73 + step as u64, ask, 0, None).unwrap();
            assert!(initializing);
            let incumbent = state.best().unwrap();
            let value = incumbent + gain;
            let expected = host
                .initial_evidence(f64::from(value) - f64::from(incumbent), 0.0, 0.0)
                .unwrap();
            let actual = state.tell_initial(&round, value, 0.0).unwrap();
            state.sync().unwrap();
            assert_eq!(actual.accepted, expected.accepted);
            if gain == 0.01 {
                assert!(
                    !actual.accepted,
                    "a gain rounded away in stored FP32 is not accepted during warm-up"
                );
            }
        }
    });
}

#[test]
fn paired_trace() {
    autoreleasepool(|| {
        let base = vec![0x3c00; 17];
        // Keep this tiny FP16 fixture above its rounding floor while
        // exercising contraction, expansion and multiple full-history windows.
        let bounds = TRLengthConfig::new(0.125, 0.03125, 2.0);
        let policy = ReliabilityPolicy {
            failure_tolerance: 4,
            ..Default::default()
        };
        let ask = {
            let mut config = RoundConfig::default();
            config.surrogate.neighbors = 1;
            config
        };
        let mut state = SearchState::new_implicit(
            &base,
            vec![ParamBlock::new(73, 0, base.len(), 0.25, 1.0).unwrap()],
            1,
            bounds,
            ProposalKernel::Gaussian,
        )
        .unwrap();
        state
            .configure_enn(crate::config::ResidentEnnConfig {
                pool: ProceduralPoolConfig {
                    arms: 1,
                    candidates_per_arm: 1,
                }
                .resolve_resident()
                .unwrap(),
                proposal_method: crate::procedural_pool::ProposalMethod::Additive,
                ask,
                num_candidates: 4,
                num_samples: 4,
                fit_params: false,
                adapt_neighbors: false,
                distance_scaling: crate::config::DistanceScaling::Global,
                history_geometry: crate::config::HistoryGeometry::Latent,
                local_scale_neighbors: 8,
            })
            .unwrap();
        state.configure_controller(policy).unwrap();
        state.observe_initial(0.0, 0.0).unwrap();
        let mut host = RadiusController::configured(0.125, 0.03125, 2.0, policy).unwrap();
        let mut history = NeighborValueEstimator::with_spec(
            0.0,
            0.0,
            ask.acquisition.seed,
            baux::semantics::optimizer::SurrogateSpec {
                neighbors: 1,
                ..Default::default()
            },
        )
        .unwrap();
        let mut accepted = 0;
        for step in 0..384 {
            let (_, mut round, initializing) = state
                .propose(73 + step, ask, 0, None)
                .unwrap_or_else(|error| panic!("step={step}: {error}"));
            assert!(!initializing);
            assert_eq!(history.identities(), &state.identities[..state.history]);
            let distances = round
                .history_distances
                .iter()
                .map(|&(_, distance)| f64::from(distance))
                .collect::<Vec<_>>();
            let slot = history.next_slot().unwrap();
            round.predicted_mean = ((step * 7) % 31) as f32 * 0.011 - 0.1;
            round.incumbent_mean = ((step * 13) % 29) as f32 * 0.009 - 0.1;
            round.predicted_standard_error = 0.03 + (step % 5) as f32 * 0.001;
            round.incumbent_standard_error = 0.02 + (step % 7) as f32 * 0.002;
            let measured = ((step * 11) % 37) as f32 * 0.017 - 0.2;
            let variance = 0.001 + (step % 3) as f32 * 0.0001;
            let blocks = [measured, measured * 0.5];
            let evidence = baux::semantics::optimizer::ReliabilityEvidence {
                observed_improvement: f64::from(measured),
                observation_variance: f64::from(variance),
                predicted_improvement: f64::from(round.predicted_mean)
                    - f64::from(round.incumbent_mean),
                prediction_variance: f64::from(round.predicted_standard_error).powi(2)
                    + f64::from(round.incumbent_standard_error).powi(2),
                nominal_radius: f64::from(round.length.abs()),
                response_radius: baux::semantics::optimizer::response_rms(Some(&blocks), 0.0),
                accepted: false,
            };
            let actual = host.observe_evidence(evidence, true).unwrap();
            let incumbent = state.best().unwrap();
            let value = incumbent + measured;
            let expected = state
                .paired_blocks(
                    &round,
                    value,
                    variance,
                    incumbent,
                    state.best_variance().unwrap(),
                    measured,
                    variance,
                    &blocks,
                )
                .unwrap();
            state.sync().unwrap();
            history
                .append(&distances, value, variance, actual.accepted)
                .unwrap();
            assert_eq!(
                history.identities(),
                &state.identities[..state.history],
                "step={step} slot={slot}"
            );
            assert_eq!(
                history.base_value().to_bits(),
                state.best().unwrap().to_bits()
            );
            for row in 0..state.history {
                let reference: Vec<f64> = state.pairwise_distances
                    [row * MAX_HISTORY..row * MAX_HISTORY + state.history]
                    .iter()
                    .map(|&d| f64::from(d))
                    .collect();
                assert_eq!(
                    history.distance_row(row).unwrap(),
                    &reference,
                    "step={step} row={row}"
                );
            }
            accepted += usize::from(expected.accepted);
            assert_eq!(actual.accepted, expected.accepted, "step={step}");
            assert_eq!(
                actual.threshold.to_bits(),
                expected.threshold.to_bits(),
                "step={step}"
            );
            assert_eq!(
                actual.predicted_improvement.to_bits(),
                expected.predicted_improvement.unwrap().to_bits()
            );
            assert_eq!(
                actual.next_radius.to_bits(),
                (state.length().unwrap() as f32).to_bits(),
                "step={step}"
            );
        }
        assert!(accepted > 0 && accepted < 384);
    });
}
