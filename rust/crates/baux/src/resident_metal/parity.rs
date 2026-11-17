use super::*;
use crate::params::ENNParams;
use crate::procedural_pool::ProceduralPoolConfig;
use crate::trust_region::TRLengthConfig;
use baux::semantics::optimizer::{
    INITIAL_OBSERVATIONS, NeighborValueEstimator, PoolShape, RadiusController,
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
        let state = SearchState::new_implicit(
            &base,
            vec![ParamBlock::new(1, 0, base.len(), 0.25, 1.0).unwrap()],
            1,
            TRLengthConfig::new(1e-20, 1e-21, 0.4),
            ProposalDistribution::Gaussian,
        )
        .unwrap();
        let state = state
            .configure_enn(crate::config::ResidentEnnConfig {
                pool: ProceduralPoolConfig {
                    arms: 1,
                    candidates_per_arm: 1,
                }
                .resolve_resident()
                .unwrap(),
                ask,
                num_candidates: 4,
                num_samples: 4,
                fit_params: false,
                adapt_neighbors: false,
                distance_metric: crate::config::DistanceMetric::Euclidean,
                local_scale_neighbors: 8,
            })
            .unwrap();
        let state = state.observe_initial(0.0, 0.0).unwrap();
        let error = match state.clone().propose(17, ask, 0, None) {
            Ok(_) => panic!("rounded-away Metal proposal was published"),
            Err(error) => error,
        };
        assert_eq!(error, "selected proposal did not change");
        assert_eq!(
            read::<u16>(&*state.proposal, base.len()),
            vec![0u16; base.len()]
        );
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
            ProposalDistribution::Gaussian,
        )
        .unwrap();
        let source = initialization::search_source(true, ProposalDistribution::Gaussian);
        let pipeline = state
            .runtime
            .precise(&source, "Posterior parity", "dense_posterior_probe")
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
    let (metal, expected) = metal
        .ask_distances(
            &distances.view(),
            &outcomes.view(),
            Some(&variances.view()),
            30,
            10,
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
    for (k, candidates, samples, adaptive) in [
        (10, 30, 10, false),
        (3, 7, 4, false),
        (32, 4, 4, false),
        (10, 16, 6, true),
    ] {
        let seed = 0x1234_5678_9abc_def0;
        let spec = baux::semantics::optimizer::SurrogateSpec {
            neighbors: k,
            fit_candidates: candidates,
            fit_samples: samples,
            fit: true,
            adaptive,
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
            let (next_metal, expected) = if adaptive {
                metal
                    .ask_adaptive(
                        &distances.view(),
                        &outcomes.view(),
                        Some(&variances.view()),
                        k as i32,
                        candidates,
                        samples,
                        None,
                        &mut random,
                    )
                    .unwrap()
            } else {
                metal
                    .ask_distances(
                        &distances.view(),
                        &outcomes.view(),
                        Some(&variances.view()),
                        candidates,
                        samples,
                        None,
                        &mut random,
                    )
                    .unwrap()
            };
            metal = next_metal;
            let actual = host.fit_state();
            assert_eq!(actual.neighbors, expected.k_neighbors as usize);
            assert_eq!(
                (actual.epistemic as f32).to_bits(),
                (expected.epistemic_scale as f32).to_bits(),
                "k={k} n={n}"
            );
            assert!(
                (actual.epistemic - expected.epistemic_scale).abs()
                    <= 128.0 * f64::EPSILON * expected.epistemic_scale.abs().max(1.0),
                "k={k} n={n}: actual={} expected={}",
                actual.epistemic,
                expected.epistemic_scale
            );
            assert_eq!(
                (actual.aleatoric as f32).to_bits(),
                (expected.aleatoric_scale as f32).to_bits(),
                "k={k} n={n}"
            );
            assert!(
                (actual.aleatoric - expected.aleatoric_scale).abs()
                    <= 128.0 * f64::EPSILON * expected.aleatoric_scale.abs().max(1.0),
                "k={k} n={n}: actual={} expected={}",
                actual.aleatoric,
                expected.aleatoric_scale
            );
            assert_eq!(
                (actual.scale as f32).to_bits(),
                (metal.y_std()[0] as f32).to_bits(),
                "k={k} n={n}"
            );
            assert!(
                (actual.scale - metal.y_std()[0]).abs()
                    <= 128.0 * f64::EPSILON * metal.y_std()[0].abs().max(1.0),
                "k={k} n={n}: actual={} expected={}",
                actual.scale,
                metal.y_std()[0]
            );
        }
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
        let state = SearchState::new_implicit(
            &base,
            vec![ParamBlock::new(73, 0, base.len(), 0.25, 1.0).unwrap()],
            1,
            bounds,
            ProposalDistribution::Gaussian,
        )
        .unwrap();
        let state = state
            .configure_enn(crate::config::ResidentEnnConfig {
                pool: ProceduralPoolConfig {
                    arms: 1,
                    candidates_per_arm: 1,
                }
                .resolve_resident()
                .unwrap(),
                ask,
                num_candidates: 4,
                num_samples: 4,
                fit_params: false,
                adapt_neighbors: false,
                distance_metric: crate::config::DistanceMetric::Euclidean,
                local_scale_neighbors: 8,
            })
            .unwrap();
        let mut state = state.observe_initial(1_048_576.0, 0.0).unwrap();
        let host = RadiusController::new(0.125, 0.03125, 2.0).unwrap();
        for (step, gain) in [0.01, 0.125, -0.5, 0.01, 0.25, -0.125, 0.01, 0.125, -0.25]
            .into_iter()
            .enumerate()
        {
            let (s, (_, round, initializing)) =
                state.propose(73 + step as u64, ask, 0, None).unwrap();
            assert!(initializing);
            let incumbent = s.best().unwrap();
            let value = incumbent + gain;
            let expected = host
                .initial_evidence(f64::from(value) - f64::from(incumbent), 0.0, 0.0)
                .unwrap();
            let (s, actual) = s.tell_initial(&round, value, 0.0).unwrap();
            let (s, _) = s.sync().unwrap();
            assert_eq!(actual.accepted, expected.accepted);
            if gain == 0.01 {
                assert!(
                    !actual.accepted,
                    "a gain rounded away in stored FP32 is not accepted during warm-up"
                );
            }
            state = s;
        }
    });
}

#[test]
fn large_parity() {
    use baux::semantics::neighbor::query::Query;
    autoreleasepool(|| {
        let dims = 33usize;
        let base = vec![0x3800u16; dims];
        let block = ParamBlock::new(73, 0, dims, 0.5, 1.0 / (dims as f32)).unwrap();
        for &(arms, slots) in &[
            (1u32, 1u32),
            (1, 4),
            (1, 64),
            (1, 512),
            (2, 1024),
            (1, 5000),
        ] {
            let count = (arms * slots) as usize;
            let pool = ProceduralPoolConfig {
                arms,
                candidates_per_arm: slots,
            }
            .resolve_resident()
            .unwrap();
            let mut ask = RoundConfig::default();
            ask.surrogate.neighbors = 3;
            ask.surrogate.epistemic_scale = 0.4;
            ask.surrogate.aleatoric_scale = 0.05;
            ask.surrogate.output_scale = 1.5;
            ask.acquisition.beta = 1.0;
            ask.acquisition.seed = 0x5eed_cafe_1234_5678;

            let mut state = SearchState::new_implicit(
                &base,
                vec![block],
                8,
                TRLengthConfig::new(0.6, 0.01, 1.6),
                ProposalDistribution::Gaussian,
            )
            .unwrap()
            .configure_enn(crate::config::ResidentEnnConfig {
                pool,
                ask,
                num_candidates: 4,
                num_samples: 2,
                fit_params: false,
                adapt_neighbors: false,
                distance_metric: crate::config::DistanceMetric::Euclidean,
                local_scale_neighbors: 3,
            })
            .unwrap()
            .configure_sampler(ResidentSampler::Raasp, Some(12), true)
            .unwrap()
            .observe_initial(0.25, 0.01)
            .unwrap();

            for w in 0..3u64 {
                let (s, (_buf, prop, init)) =
                    state.propose(300 + w, ask, w as usize, None).unwrap();
                assert!(init || w >= 2);
                let y = (w as f32 * 0.7).sin();
                let (s, _) = if init {
                    s.tell_initial(&prop, y, 0.02).unwrap()
                } else {
                    s.tell_modeled(&prop, y, 0.02).unwrap()
                };
                let (s, _) = s.sync().unwrap();
                state = s;
            }

            for kind in [AcquisitionKind::Ucb, AcquisitionKind::Thompson] {
                let mut q_ask = ask;
                q_ask.acquisition.kind = kind;
                let root = 9001u64;
                let (state_after_dist, dist_matrix) = state.clone().sample_distances(root).unwrap();
                assert_eq!(dist_matrix.len(), count);
                let sel_params = state_after_dist.select_params(root, q_ask, None);
                let len = state_after_dist.history;

                let (state_gpu, prop) = state
                    .clone()
                    .ask_round(arms as usize, slots as usize, root, q_ask)
                    .unwrap();
                assert_eq!(prop.pool_predictions.len(), count);

                let mut best_cpu = (f32::NEG_INFINITY, 0usize);
                for (cand, row_dists) in dist_matrix.iter().enumerate() {
                    let expected = Query {
                        distances: row_dists,
                        outcomes: &sel_params.outcomes[..len],
                        variances: &sel_params.variances[..len],
                        draws: (kind == AcquisitionKind::Thompson)
                            .then_some(&sel_params.draws[..len]),
                        neighbors: q_ask.surrogate.neighbors.min(len),
                        epistemic: q_ask.surrogate.epistemic_scale,
                        aleatoric: q_ask.surrogate.aleatoric_scale,
                        scale: q_ask.surrogate.output_scale,
                        beta: q_ask.acquisition.beta,
                        excluded: None,
                        incumbent: sel_params.incumbent_index as usize,
                        adaptive: false,
                    }
                    .predict()
                    .unwrap();
                    let (gm, gse, gsc) = prop.pool_predictions[cand];
                    let tol = 32.0 * f32::EPSILON * expected.score.abs().max(1.0);
                    assert!(
                        (gm - expected.mean).abs() <= tol
                            && (gse - expected.standard_error).abs() <= tol
                            && (gsc - expected.score).abs() <= tol,
                        "count={count} kind={kind:?} cand={cand}: gpu=({gm},{gse},{gsc}) != cpu=({},{},{})",
                        expected.mean,
                        expected.standard_error,
                        expected.score
                    );
                    if expected.score > best_cpu.0 {
                        best_cpu = (expected.score, cand);
                    }
                }
                let (_wm, _wse, win_gpu_score) = prop.pool_predictions[prop.index];
                assert!(
                    (win_gpu_score - best_cpu.0).abs()
                        <= 32.0 * f32::EPSILON * best_cpu.0.abs().max(1.0),
                    "winner score mismatch for count={count}: gpu={} cpu={}",
                    win_gpu_score,
                    best_cpu.0
                );
                let expected_row = state.test_candidate(root, prop.index).unwrap();
                assert_eq!(
                    prop.read_row().unwrap(),
                    read::<u16>(&expected_row, dims),
                    "materialized row mismatch at count={count} winner={}",
                    prop.index
                );
                drop(state_gpu);
            }
        }
    });
}

#[test]
fn delayed_parity() {
    use baux::semantics::key::Key;
    use baux::semantics::pool::{DelayedPool, joe_kuo};
    autoreleasepool(|| {
        let dims = 24usize;
        let base_bf16 = vec![0x3f00u16; dims]; // 0.5 in BF16
        let center = vec![0.5f32; dims];
        let lower = vec![0.2f32; dims];
        let upper = vec![0.8f32; dims];
        let unit_lo = vec![0.0f32; dims];
        let unit_hi = vec![1.0f32; dims];
        let dirs: Vec<[u32; 32]> = (0..dims).map(joe_kuo).collect();
        let block = ParamBlock::new(97, 0, dims, 0.5, 1.0 / (dims as f32)).unwrap();
        let pool_cfg = ProceduralPoolConfig {
            arms: 1,
            candidates_per_arm: 64,
        }
        .resolve_resident()
        .unwrap();
        let bf16_word = |x: f32| -> u16 {
            let bits = x.to_bits();
            (bits.wrapping_add(0x7fff + ((bits >> 16) & 1)) >> 16) as u16
        };
        let root = 0xfeed_beef_cafe_0001u64;
        let key = Key::new(root);

        for sampler in [
            ResidentSampler::Uniform,
            ResidentSampler::Raasp,
            ResidentSampler::Sobol,
            ResidentSampler::SobolMasked,
            ResidentSampler::Lhd,
        ] {
            let state = SearchState::new(
                &base_bf16,
                1.0,
                0.0,
                vec![block],
                8,
                1,
                TRLengthConfig::new(0.6, 0.05, 1.6),
            )
            .unwrap()
            .configure_pool(pool_cfg)
            .unwrap()
            .configure_sampler(sampler, Some(8), true)
            .unwrap()
            .configure_lhd(64)
            .unwrap();

            let cpu_pool = match sampler {
                ResidentSampler::Uniform => DelayedPool::Uniform {
                    key,
                    lower: &lower,
                    upper: &upper,
                    num_candidates: 64,
                },
                ResidentSampler::Raasp => DelayedPool::Raasp {
                    key,
                    center: &center,
                    lower: &lower,
                    upper: &upper,
                    num_candidates: 64,
                    num_pert: 8,
                },
                ResidentSampler::Sobol => DelayedPool::Sobol {
                    key,
                    lower: &lower,
                    upper: &upper,
                    dirs: &dirs,
                    base_idx: 0,
                    num_candidates: 64,
                    scramble: true,
                },
                ResidentSampler::SobolMasked => DelayedPool::SobolMasked {
                    key,
                    center: &center,
                    lower: &lower,
                    upper: &upper,
                    dirs: &dirs,
                    base_idx: 0,
                    num_candidates: 64,
                    num_pert: 8,
                    scramble: true,
                },
                ResidentSampler::Lhd => DelayedPool::Lhd {
                    key,
                    lower: &unit_lo,
                    upper: &unit_hi,
                    num_candidates: 64,
                },
                _ => unreachable!(),
            };

            for cand in [0usize, 1, 7, 31, 63] {
                let gpu_buf = state.test_candidate(root, cand).unwrap();
                let gpu_words = read::<u16>(&gpu_buf, dims);
                let mut cpu_row = vec![0.0f32; dims];
                cpu_pool.write_row(cand, &mut cpu_row);
                let cpu_words: Vec<u16> = cpu_row.iter().copied().map(bf16_word).collect();
                assert_eq!(
                    gpu_words, cpu_words,
                    "DelayedPool<f32> bit parity mismatch for sampler={sampler:?} cand={cand}"
                );
            }
        }
    });
}

#[test]
fn sobol_cursor() {
    use baux::semantics::key::Key;
    use baux::semantics::pool::{DelayedPool, joe_kuo};
    autoreleasepool(|| {
        let dims = 20usize;
        let base_bf16 = vec![0x3f00u16; dims]; // 0.5 in BF16
        let center = vec![0.5f32; dims];
        let lower = vec![0.2f32; dims];
        let upper = vec![0.8f32; dims];
        let dirs: Vec<[u32; 32]> = (0..dims).map(joe_kuo).collect();
        let block = ParamBlock::new(97, 0, dims, 0.5, 1.0 / (dims as f32)).unwrap();
        let pool_cfg = ProceduralPoolConfig {
            arms: 2,
            candidates_per_arm: 16,
        }
        .resolve_resident()
        .unwrap();
        let pool_count = pool_cfg.count() as usize;
        let bf16_word = |x: f32| -> u16 {
            let bits = x.to_bits();
            (bits.wrapping_add(0x7fff + ((bits >> 16) & 1)) >> 16) as u16
        };
        let mut ask = RoundConfig::default();
        ask.surrogate.neighbors = 1;
        ask.acquisition.beta = 1.0;

        for sampler in [ResidentSampler::Sobol, ResidentSampler::SobolMasked] {
            let mut state = SearchState::new(
                &base_bf16,
                1.0,
                0.0,
                vec![block],
                8,
                1,
                TRLengthConfig::new(0.6, 0.05, 1.6),
            )
            .unwrap()
            .fix_radius(true)
            .unwrap()
            .configure_pool(pool_cfg)
            .unwrap()
            .configure_sampler(sampler, Some(8), true)
            .unwrap();

            for round_idx in 0..2u32 {
                let root = 0x50b0_1000_0000_0000u64 + u64::from(round_idx);
                let key = Key::new(root);
                let expected_base = round_idx * (pool_count as u32);
                let (next_state, prop) = state.ask_round(2, 16, root, ask).unwrap();
                assert_eq!(prop.sobol_offset, expected_base);
                assert_eq!(prop.selected_indices.len(), 2);

                let cpu_pool = match sampler {
                    ResidentSampler::Sobol => DelayedPool::Sobol {
                        key,
                        lower: &lower,
                        upper: &upper,
                        dirs: &dirs,
                        base_idx: expected_base,
                        num_candidates: pool_count,
                        scramble: true,
                    },
                    ResidentSampler::SobolMasked => DelayedPool::SobolMasked {
                        key,
                        center: &center,
                        lower: &lower,
                        upper: &upper,
                        dirs: &dirs,
                        base_idx: expected_base,
                        num_candidates: pool_count,
                        num_pert: 8,
                        scramble: true,
                    },
                    _ => unreachable!(),
                };

                let batch = next_state.materialize_batch(&prop).unwrap();
                assert_eq!(batch.len(), 2);
                for (arm_pos, &cand_idx) in prop.selected_indices.iter().enumerate() {
                    let mut cpu_row = vec![0.0f32; dims];
                    cpu_pool.write_row(cand_idx, &mut cpu_row);
                    let cpu_words: Vec<u16> = cpu_row.iter().copied().map(bf16_word).collect();
                    assert_eq!(
                        batch[arm_pos].read_u16().unwrap(),
                        cpu_words,
                        "materialize_batch mismatch for sampler={sampler:?} round={round_idx} arm={arm_pos} cand={cand_idx}"
                    );
                }

                let diag_cand = (prop.selected_indices[0] + 5) % pool_count;
                let diag_buf = next_state
                    .diagnostic_row(&prop, root, ask, diag_cand)
                    .unwrap();
                let mut cpu_diag = vec![0.0f32; dims];
                cpu_pool.write_row(diag_cand, &mut cpu_diag);
                let cpu_diag_words: Vec<u16> = cpu_diag.iter().copied().map(bf16_word).collect();
                assert_eq!(
                    read::<u16>(&diag_buf, dims),
                    cpu_diag_words,
                    "diagnostic_row mismatch for sampler={sampler:?} round={round_idx} cand={diag_cand}"
                );

                // Verify diagnostic_row never overwrote round.device_row()!
                let mut cpu_arm0 = vec![0.0f32; dims];
                cpu_pool.write_row(prop.selected_indices[0], &mut cpu_arm0);
                let cpu_arm0_words: Vec<u16> = cpu_arm0.iter().copied().map(bf16_word).collect();
                assert_eq!(prop.read_row().unwrap(), cpu_arm0_words);

                let (told, _dec) = next_state.tell(&prop, 0.5, 0.01).unwrap();
                let (synced, _) = told.sync().unwrap();
                state = synced;
            }
        }
    });
}
