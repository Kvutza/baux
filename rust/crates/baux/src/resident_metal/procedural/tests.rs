use super::*;

fn state(pool: ProceduralPool, perturbation: ProposalDistribution) -> SearchState {
    let base = vec![0x3400; TILE_ELEMENTS + 17];
    let block = ParamBlock::new(71, 0, base.len(), 0.25, 1.0).unwrap();
    let state = SearchState::new_implicit(
        &base,
        vec![block],
        1,
        TRLengthConfig::new(0.1, 0.001, 0.4),
        perturbation,
    )
    .unwrap();
    let state = state
        .configure_enn(crate::config::ResidentEnnConfig {
            pool,
            ask: {
                let mut config = RoundConfig::default();
                config.surrogate.neighbors = 1;
                config
            },
            num_candidates: 4,
            num_samples: 2,
            fit_params: true,
            adapt_neighbors: false,
            distance_metric: crate::config::DistanceMetric::Euclidean,
            local_scale_neighbors: 1,
        })
        .unwrap();
    state.observe_initial(0.0, 0.0).unwrap()
}

fn check_layout(pool: ProceduralPool, perturbation: ProposalDistribution) {
    let mut search = state(pool, perturbation);
    let mut elapsed = Vec::new();
    for round in 0..8 {
        let candidate = round % (pool.count() as usize);
        let root = crate::hash::splitmix64(round as u64);
        let expected = search.test_candidate(root, candidate).unwrap();
        let bits = read::<u16>(&expected, search.dimensions).to_vec();
        let radius = search.radius(candidate);
        let start = std::time::Instant::now();
        let (s, _row) = search
            .begin_forced(root, RoundConfig::default(), candidate)
            .unwrap();
        let (s, proposal) = s.finish_ask().unwrap();
        elapsed.push(start.elapsed().as_secs_f64() * 1000.0);
        assert_eq!(proposal.seed, pool.seed(root, proposal.identity()).unwrap());
        assert_eq!(proposal.length.to_bits(), radius.to_bits());
        assert_eq!(proposal.arms(), pool.arms() as usize);
        assert_eq!(read::<u16>(&*s.proposal, s.dimensions), bits);
        assert_eq!(proposal.pool_predictions.len(), pool.count() as usize);
        assert_eq!(
            proposal.procedural_candidates().count(),
            pool.count() as usize
        );
        let (s, _decision) = s.tell_noisy(&proposal, -1.0, 0.0).unwrap();
        let (s, _synced) = s.sync().unwrap();
        search = s;
    }
    elapsed.sort_by(f64::total_cmp);
    eprintln!(
        "procedural {}x{} {:?}: {} weights, tiny ask p50={:.3}ms max={:.3}ms (not full BO)",
        pool.arms(),
        pool.slots(),
        perturbation,
        search.dimensions,
        elapsed[4],
        elapsed[7]
    );
}

#[test]
fn resident_layouts() {
    autoreleasepool(|| {
        for (arms, slots) in [(1, 1), (1, 4), (2, 2), (4, 1), (1, 64), (4, 128)] {
            let pool = ProceduralPoolConfig {
                arms,
                candidates_per_arm: slots,
            }
            .resolve_resident()
            .unwrap();
            for perturbation in [
                ProposalDistribution::Rademacher,
                ProposalDistribution::Gaussian,
            ] {
                check_layout(pool, perturbation);
            }
        }
    });
}

#[test]
fn large_samplers() {
    autoreleasepool(|| {
        let dims = 64usize;
        let base = vec![0x3800u16; dims]; // 0.5 in FP16
        let block = ParamBlock::new(71, 0, dims, 0.5, 1.0 / (dims as f32)).unwrap();
        for count in [64u32, 512, 2048] {
            let pool = ProceduralPoolConfig {
                arms: if count == 2048 { 4 } else { 1 },
                candidates_per_arm: if count == 2048 { 512 } else { count },
            }
            .resolve_resident()
            .unwrap();
            for sampler in [
                ResidentSampler::Uniform,
                ResidentSampler::Raasp,
                ResidentSampler::Sobol,
                ResidentSampler::Lhd,
            ] {
                let mut ask = RoundConfig::default();
                ask.surrogate.neighbors = 2;
                ask.acquisition.beta = 1.0;
                let mut st = SearchState::new_implicit(
                    &base,
                    vec![block],
                    8,
                    TRLengthConfig::new(0.8, 0.01, 1.6),
                    ProposalDistribution::Gaussian,
                )
                .unwrap()
                .configure_enn(crate::config::ResidentEnnConfig {
                    pool,
                    ask,
                    num_candidates: 4,
                    num_samples: 2,
                    fit_params: true,
                    adapt_neighbors: true,
                    distance_metric: crate::config::DistanceMetric::Euclidean,
                    local_scale_neighbors: 2,
                })
                .unwrap()
                .configure_sampler(sampler, Some(8), true)
                .unwrap()
                .configure_lhd(count as usize)
                .unwrap()
                .configure_manifold(ResidentManifold {
                    scale_x: true,
                    metric_learning: crate::layout::MetricLearning::Auto,
                    tied_dims: vec![vec![0, 1]],
                    unscaled_dims: vec![2],
                    categories: vec![(3, 4)],
                    coord_system: crate::proposal_distribution::CoordSystem::Cartesian,
                    stiefel_rank: None,
                    calibrate_alpha: Some(0.2),
                    y_bounds: Some(vec![(-100.0, 100.0)]),
                    noise_aware: true,
                    no_trust_region: false,
                })
                .unwrap()
                .observe_initial(0.0, 0.01)
                .unwrap();

                for step in 0..5u64 {
                    let (next_st, (_buf, prop, init)) =
                        st.propose(1000 + step, ask, step as usize, None).unwrap();
                    let words = prop.read_row().unwrap();
                    assert_eq!(words.len(), dims);
                    // Tied dims 0 and 1 must match identically!
                    assert_eq!(words[0], words[1]);
                    // All coordinates must stay clamped inside [0, 1]!
                    for &w in &words {
                        let v = crate::unified_mem::decode_half(w);
                        assert!((-1e-4..=1.0001).contains(&v), "out of box: {v}");
                    }
                    if !init {
                        assert_eq!(prop.pool_predictions.len(), count as usize);
                        let m = prop.measure(dims as u64).unwrap();
                        assert_eq!(m.candidate, prop.index as u32);
                        if count == 2048 {
                            assert_eq!(prop.selected_indices.len(), 4);
                            let batch_rows = next_st.materialize_batch(&prop).unwrap();
                            assert_eq!(batch_rows.len(), 4);
                            assert_ne!(
                                batch_rows[0].read_u16().unwrap(),
                                batch_rows[1].read_u16().unwrap()
                            );
                        }
                    }
                    let y = -((step as f32 - 2.0).powi(2));
                    let (next_st, _dec) = if init {
                        next_st.tell_initial(&prop, y, 0.01).unwrap()
                    } else {
                        next_st.tell_modeled(&prop, y, 0.01).unwrap()
                    };
                    let (next_st, _) = next_st.sync().unwrap();
                    st = next_st;
                }
            }
        }
    });
}

#[test]
fn coord_lhd() {
    autoreleasepool(|| {
        let dims = 32usize;
        let base = vec![0x3800u16; dims];
        let block = ParamBlock::new(19, 0, dims, 0.4, 1.0 / (dims as f32)).unwrap();
        let pool = ProceduralPoolConfig {
            arms: 1,
            candidates_per_arm: 64,
        }
        .resolve_resident()
        .unwrap();
        for cs in [
            crate::proposal_distribution::CoordSystem::Cartesian,
            crate::proposal_distribution::CoordSystem::Spherical,
            crate::proposal_distribution::CoordSystem::Polar,
            crate::proposal_distribution::CoordSystem::OriginSphere,
        ] {
            let mut ask = RoundConfig::default();
            ask.acquisition.kind = AcquisitionKind::Random;
            ask.surrogate.neighbors = 1;
            let st = SearchState::new_fp16(
                &base,
                vec![block],
                0,
                TRLengthConfig::new(0.5, 0.05, 1.0),
                ProposalDistribution::Gaussian,
            )
            .unwrap()
            .configure_pool(pool)
            .unwrap()
            .configure_manifold(ResidentManifold {
                coord_system: cs,
                no_trust_region: true,
                ..ResidentManifold::default()
            })
            .unwrap()
            .observe_initial(0.0, 0.0)
            .unwrap();
            assert!(!st.has_surrogate());
            let (st, prop) = st.ask_round(1, 64, 555, ask).unwrap();
            assert_eq!(prop.length, 2.0);
            let words = prop.read_row().unwrap();
            for &w in &words {
                let v = crate::unified_mem::decode_half(w);
                assert!((0.0..=1.0).contains(&v), "cs={cs:?} out of [0,1]: {v}");
            }
            let (st, _dec) = st.tell_noisy(&prop, -10.0, 0.0).unwrap();
            let (st, _) = st.sync().unwrap();
            // `no_trust_region` keeps radius invariant across failures!
            assert_eq!(st.radius(0), 2.0);
            assert_eq!(st.restarts().unwrap(), 0);
        }
    });
}

#[test]
fn single_pass() {
    autoreleasepool(|| {
        let pool = ProceduralPoolConfig {
            arms: 1,
            candidates_per_arm: 1,
        }
        .resolve_resident()
        .unwrap();
        let search = state(pool, ProposalDistribution::Gaussian);
        let search = search.set_profiling(true);
        let root = crate::hash::splitmix64(99);
        let expected = search.test_candidate(root, 0).unwrap();
        let bits = read::<u16>(&expected, search.dimensions).to_vec();

        let (search, _row) = search
            .begin_ask(1, 1, root, RoundConfig::default())
            .unwrap();
        let (search, proposal) = search.finish_ask().unwrap();
        let profile = search.last_profile().unwrap();
        assert_eq!(profile.materialize_ms, 0.0);
        assert_eq!(proposal.pool_predictions.len(), 1);
        assert_eq!(read::<u16>(&*search.proposal, search.dimensions), bits);
        let (search, _dec) = search.tell_noisy(&proposal, -0.5, 0.0).unwrap();
        let (search, _synced) = search.sync().unwrap();

        let root2 = crate::hash::splitmix64(100);
        let expected2 = search.test_candidate(root2, 0).unwrap();
        let bits2 = read::<u16>(&expected2, search.dimensions).to_vec();
        let (search, distances) = search.sample_distances(root2).unwrap();
        assert_eq!(distances.len(), 1);

        let mut surrogate =
            baux::semantics::optimizer::NeighborValueEstimator::new(0.0, 0.0, 7).unwrap();
        surrogate
            .append(&distances[0][..1], -0.5, 0.0, false)
            .unwrap();
        let selection = surrogate
            .select_pool(
                &distances,
                1,
                baux::semantics::optimizer::surrogate::AcquisitionRule::Ucb { beta: 1.0 },
                None,
            )
            .unwrap();
        assert_eq!(selection.pool_predictions.len(), 1);
        let (search, (_buf, proposal2)) = search.materialize_selection(root2, &selection).unwrap();
        assert_eq!(proposal2.pool_predictions.len(), 1);
        assert_eq!(read::<u16>(&*search.proposal, search.dimensions), bits2);

        let eight_way: Vec<Vec<f64>> = (0..8).map(|i| vec![1.0e-4 * (i + 1) as f64; 2]).collect();
        let sel8 = surrogate
            .select_pool(
                &eight_way,
                2,
                baux::semantics::optimizer::surrogate::AcquisitionRule::Ucb { beta: 1.0 },
                None,
            )
            .unwrap();
        assert_eq!(sel8.pool_predictions.len(), 8);
    });
}
