use super::*;

fn state(pool: ProceduralPool, perturbation: ProposalKernel) -> SearchState {
    let base = vec![0x3400; TILE_ELEMENTS + 17];
    let block = ParamBlock::new(71, 0, base.len(), 0.25, 1.0).unwrap();
    let mut state = SearchState::new_implicit(
        &base,
        vec![block],
        1,
        TRLengthConfig::new(0.1, 0.001, 0.4),
        perturbation,
    )
    .unwrap();
    state
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
    state.observe_initial(0.0, 0.0).unwrap();
    state
}

fn check_layout(pool: ProceduralPool, perturbation: ProposalKernel) {
    let mut search = state(pool, perturbation);
    let mut elapsed = Vec::new();
    for round in 0..8 {
        let candidate = round % (pool.count() as usize);
        let root = crate::hash::splitmix64(round as u64);
        let expected = search.test_candidate(root, candidate).unwrap();
        let bits = read::<u16>(&expected, search.dimensions).to_vec();
        let radius = search.radius(candidate);
        let start = std::time::Instant::now();
        search
            .begin_forced(root, RoundConfig::default(), candidate)
            .unwrap();
        let proposal = search.finish_ask().unwrap();
        elapsed.push(start.elapsed().as_secs_f64() * 1000.0);
        assert_eq!(proposal.seed, pool.seed(root, proposal.identity()).unwrap());
        assert_eq!(proposal.length.to_bits(), radius.to_bits());
        assert_eq!(proposal.arms(), pool.arms() as usize);
        assert_eq!(read::<u16>(&search.proposal, search.dimensions), bits);
        assert_eq!(proposal.pool_predictions.len(), pool.count() as usize);
        assert_eq!(
            proposal.procedural_candidates().count(),
            pool.count() as usize
        );
        search.tell_noisy(&proposal, -1.0, 0.0).unwrap();
        search.sync().unwrap();
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
        for (arms, slots) in [(1, 1), (1, 4), (2, 2), (4, 1)] {
            let pool = ProceduralPoolConfig {
                arms,
                candidates_per_arm: slots,
            }
            .resolve_resident()
            .unwrap();
            for perturbation in [ProposalKernel::Rademacher, ProposalKernel::Gaussian] {
                check_layout(pool, perturbation);
            }
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
        let mut search = state(pool, ProposalKernel::Gaussian);
        search.set_profiling(true);
        let root = crate::hash::splitmix64(99);
        let expected = search.test_candidate(root, 0).unwrap();
        let bits = read::<u16>(&expected, search.dimensions).to_vec();

        search
            .begin_ask(1, 1, root, RoundConfig::default())
            .unwrap();
        let proposal = search.finish_ask().unwrap();
        let profile = search.last_profile().unwrap();
        assert_eq!(profile.materialize_ms, 0.0);
        assert_eq!(proposal.pool_predictions.len(), 1);
        assert_eq!(read::<u16>(&search.proposal, search.dimensions), bits);
        search.tell_noisy(&proposal, -0.5, 0.0).unwrap();
        search.sync().unwrap();

        let root2 = crate::hash::splitmix64(100);
        let expected2 = search.test_candidate(root2, 0).unwrap();
        let bits2 = read::<u16>(&expected2, search.dimensions).to_vec();
        let distances = search.sample_distances(root2).unwrap();
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
        let (_buf, proposal2) = search.materialize_selection(root2, &selection).unwrap();
        assert_eq!(proposal2.pool_predictions.len(), 1);
        assert_eq!(read::<u16>(&search.proposal, search.dimensions), bits2);

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
