use super::tell::retire_aftertell;
use super::*;
use crate::objective_observation::{ObjectiveEstimate, ObjectiveObservation};

fn state() -> SearchState {
    SearchState::new_fp16(
        &[0x3400; 4],
        vec![ParamBlock::new(1, 0, 4, 0.25, 1.0).unwrap()],
        2,
        TRLengthConfig::new(0.1, 0.001, 0.4),
        ProposalDistribution::Gaussian,
    )
    .unwrap()
    .observe_initial(-1.0, 0.0)
    .unwrap()
}

fn vector(mean: f32, first: f32, second: f32) -> ObjectiveObservation {
    let estimate = |value| ObjectiveEstimate {
        mean: value,
        variance: 0.0,
    };
    ObjectiveObservation::new(&[estimate(first), estimate(second)], estimate(mean)).unwrap()
}

fn vector_state() -> SearchState {
    SearchState::new_objectives(
        &[0x3400; 4],
        vector(0.0, 0.0, 0.0),
        vec![ParamBlock::new(1, 0, 4, 0.25, 1.0).unwrap()],
        3,
        1,
        TRLengthConfig::new(0.1, 0.001, 0.4),
    )
    .unwrap()
}

#[test]
fn typed_policy() {
    let spec = RegionalSearchSpec {
        count: 2,
        sharing: ObservationSharing::OriginOnly,
        allocation: RegionAllocation::LeastTrials,
        controller: RegionController::Turbo,
        success_tolerance: 3,
        failure_tolerance: 5,
        restart_memory: RegionMemory::Fresh,
        restart_candidates: 1,
        restart_center: RestartCenter::First,
        acquisitions: Vec::new(),
    };
    let search = RegionalSearch::from_spec(vec![state(), state()], &spec).unwrap();
    assert_eq!(search.len(), 2);
    assert_eq!(search.trials(), [0, 0]);
    let fixed = RegionalSearchSpec {
        controller: RegionController::FixedRadius,
        allocation: RegionAllocation::JointAcquisition,
        restart_memory: RegionMemory::RetainRelevant,
        restart_candidates: 4,
        restart_center: RestartCenter::Maximin,
        success_tolerance: 5,
        ..spec
    };
    let fixed_search = RegionalSearch::from_spec(vec![state(), state()], &fixed).unwrap();
    assert_eq!(fixed_search.len(), 2);
    let center = fixed_search.sample_center(0, 42).unwrap();
    assert_eq!(center.len(), 4);
    let (fixed_search, prop) = fixed_search.ask(99, RoundConfig::default()).unwrap();
    let (fixed_search, _dec) = fixed_search.tell(&prop, 2.0, 0.0).unwrap();
    assert_eq!(fixed_search.trials().iter().sum::<u64>(), 1);
}

#[test]
fn completion_cleanup() {
    let state = state();
    let mut ledger = RegionalLedger::new(&[-1.0], Allocation::LeastTrials).unwrap();
    let ticket = ledger.issue(0).unwrap();
    let mut states = vec![state];
    let mut proposals = vec![None];
    let mut tickets = vec![Some(ticket)];
    let failure = retire_aftertell(
        &mut ledger,
        &mut states,
        &mut proposals,
        &mut tickets,
        0,
        ticket,
        "injected ledger completion failure".into(),
    );
    assert!(matches!(failure, TellFailure::Retired(_)));
    assert!(states[0].check_healthy().is_ok());
    assert!(ledger.validate(ticket).is_err());
    assert!(proposals[0].is_none());
    assert!(tickets[0].is_none());
}

#[test]
fn regional_pareto() {
    autoreleasepool(|| {
        let spec = RegionalSearchSpec {
            count: 2,
            sharing: ObservationSharing::OriginOnly,
            allocation: RegionAllocation::IncumbentUcb1,
            controller: RegionController::Turbo,
            success_tolerance: 3,
            failure_tolerance: 5,
            restart_memory: RegionMemory::Fresh,
            restart_candidates: 1,
            restart_center: RestartCenter::First,
            acquisitions: vec![
                Some(crate::config::AcquisitionSpec::Pareto {
                    scales: vec![1.0, 1.0],
                });
                2
            ],
        };
        let search =
            RegionalSearch::from_spec(vec![vector_state(), vector_state()], &spec).unwrap();
        let (search, proposal) = search.ask_region(0, 7, RoundConfig::default()).unwrap();
        let (search, decision) =
            objectives::tell(search, &proposal, vector(1.0, 1.0, 1.0)).unwrap();
        assert!(decision.accepted);
        assert_eq!(search.trials(), [1, 0]);
        assert!(search.best().1 > 0.0);
    });
}

#[test]
fn ellipsoid_zero() {
    autoreleasepool(|| {
        let blocks = vec![
            ParamBlock::new(1, 0, 2, 0.5, 4.0).unwrap(),
            ParamBlock::new(2, 2, 2, 0.25, 1.0).unwrap(),
        ];
        let length = TRLengthConfig::new(0.4, 0.01, 0.8);
        let first = SearchState::new(&[0x3f80; 4], 0.0, 0.0, blocks.clone(), 0, 1, length).unwrap();
        let second = SearchState::new(&[0x3f80; 4], 0.0, 0.0, blocks, 0, 1, length).unwrap();
        assert!(!first.has_surrogate());
        assert!(!second.has_surrogate());
        let search = RegionalSearch::with_sharing(
            vec![first, second],
            Allocation::LeastTrials,
            SharingPolicy::EllipsoidBall,
        )
        .unwrap();
        let mut ask = RoundConfig::default();
        ask.acquisition.kind = AcquisitionKind::Random;
        ask.surrogate.neighbors = 1;
        let (search, round0) = search.ask_region(0, 11, ask).unwrap();
        let accepted = search.buffer(&round0).unwrap();
        let bytes = read::<u16>(&accepted, 4);
        let (search, decision) = search.tell(&round0, 5.0, 0.0).unwrap();
        assert!(decision.accepted);
        assert_eq!(search.sharing_backlog(), [0, 1]);
        let search = search.flush_sharing(1).unwrap();
        assert_eq!(search.sharing_backlog(), [0, 0]);
        assert_eq!(search.delivered_imports(), 1);
        let adopted = read::<u16>(&search.incumbent(1).unwrap(), 4);
        assert_eq!(adopted, bytes);
        let (search, round1) = search.ask_region(1, 19, ask).unwrap();
        let (search, decision1) = search.tell(&round1, 6.0, 0.0).unwrap();
        assert!(decision1.accepted);
        assert_eq!(search.best(), (1, 6.0));
    });
}

#[test]
fn fork_isolation() {
    autoreleasepool(|| {
        let blocks = vec![ParamBlock::new(1, 0, 4, 0.25, 1.0).unwrap()];
        let length = TRLengthConfig::new(0.2, 0.01, 0.8);
        let first = SearchState::new(&[0x3f80; 4], 0.0, 0.0, blocks.clone(), 0, 1, length).unwrap();
        let second = SearchState::new(&[0x3f80; 4], 0.0, 0.0, blocks, 0, 1, length).unwrap();
        let search = RegionalSearch::with_sharing(
            vec![first, second],
            Allocation::LeastTrials,
            SharingPolicy::OriginOnly,
        )
        .unwrap();

        // Fork the search state (O(1) COW)
        let forked = search.fork().unwrap();
        assert_eq!(search.trials(), forked.trials());
        assert_eq!(search.best(), forked.best());

        // Evolve primary search
        let mut ask = RoundConfig::default();
        ask.acquisition.kind = AcquisitionKind::Random;
        ask.surrogate.neighbors = 1;

        let (search, prop_a) = search.ask_region(0, 101, ask).unwrap();
        let (search, dec_a) = search.tell(&prop_a, 10.0, 0.0).unwrap();
        assert!(dec_a.accepted);

        // Verify primary changed and fork remained untouched
        assert_eq!(search.trials(), [1, 0]);
        assert_eq!(search.best(), (0, 10.0));
        assert_eq!(forked.trials(), [0, 0]);
        assert_eq!(forked.best(), (0, 0.0));

        // Evolve fork independently
        let (forked, prop_b) = forked.ask_region(1, 202, ask).unwrap();
        let (forked, dec_b) = forked.tell(&prop_b, 25.0, 0.0).unwrap();
        assert!(dec_b.accepted);

        // Verify fork has its own independent state
        assert_eq!(forked.trials(), [0, 1]);
        assert_eq!(forked.best(), (1, 25.0));
        // Verify primary is still unaffected by fork's progress
        assert_eq!(search.trials(), [1, 0]);
        assert_eq!(search.best(), (0, 10.0));
    });
}

#[test]
fn cancel_restart() {
    autoreleasepool(|| {
        let spec = RegionalSearchSpec {
            count: 2,
            sharing: ObservationSharing::OriginOnly,
            allocation: RegionAllocation::LeastTrials,
            controller: RegionController::Turbo,
            success_tolerance: 3,
            failure_tolerance: 5,
            restart_memory: RegionMemory::Fresh,
            restart_candidates: 1,
            restart_center: RestartCenter::First,
            acquisitions: Vec::new(),
        };
        let search = RegionalSearch::from_spec(vec![state(), state()], &spec).unwrap();
        let (search, prop) = search.ask_region(0, 5, RoundConfig::default()).unwrap();
        let search = search.cancel(&prop).unwrap();
        assert_eq!(search.trials(), [0, 0]);

        // Restart slot 0 with a fresh state
        let fresh = state();
        let search = search.restart(0, fresh).unwrap();
        assert_eq!(search.trials(), [0, 0]);
    });
}

#[test]
fn tell_batch() {
    autoreleasepool(|| {
        let blocks = vec![ParamBlock::new(1, 0, 4, 0.25, 1.0).unwrap()];
        let length = TRLengthConfig::new(0.2, 0.01, 0.8);
        let first = SearchState::new(&[0x3f80; 4], 0.0, 0.0, blocks.clone(), 0, 1, length).unwrap();
        let second = SearchState::new(&[0x3f80; 4], 0.0, 0.0, blocks, 0, 1, length).unwrap();
        let search = RegionalSearch::with_sharing(
            vec![first, second],
            Allocation::LeastTrials,
            SharingPolicy::OriginOnly,
        )
        .unwrap();

        let mut ask = RoundConfig::default();
        ask.acquisition.kind = AcquisitionKind::Random;
        ask.surrogate.neighbors = 1;

        let (search, prop0) = search.ask_region(0, 11, ask).unwrap();
        let (search, prop1) = search.ask_region(1, 22, ask).unwrap();

        let batch = vec![(prop0, 3.0, 0.0), (prop1, 7.0, 0.0)];
        let (search, decisions) = search.tell_batch(&batch).unwrap();
        assert_eq!(decisions.len(), 2);
        assert_eq!(search.trials(), [1, 1]);
        assert_eq!(search.best(), (1, 7.0));
    });
}

#[test]
fn joint_retain() {
    autoreleasepool(|| {
        let make_region = |center_bits: u16, init_val: f32| {
            let mut ask = RoundConfig::default();
            ask.surrogate.neighbors = 1;
            ask.acquisition.beta = 1.0;
            let pool = crate::procedural_pool::ProceduralPoolConfig {
                arms: 2,
                candidates_per_arm: 16,
            }
            .resolve_resident()
            .unwrap();
            SearchState::new_implicit(
                &[center_bits; 8],
                vec![ParamBlock::new(11, 0, 8, 0.25, 0.125).unwrap()],
                8,
                TRLengthConfig::new(0.8, 0.05, 1.6),
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
                local_scale_neighbors: 1,
            })
            .unwrap()
            .configure_sampler(ResidentSampler::Raasp, Some(4), true)
            .unwrap()
            .observe_initial(init_val, 0.0)
            .unwrap()
        };

        // Region 2 has strictly highest incumbent/posterior (100.0 vs 0.0 and
        // 1.0)
        let r0 = make_region(0x3400, 0.0);
        let r1 = make_region(0x3800, 1.0);
        let r2 = make_region(0x3a00, 100.0);
        let spec = RegionalSearchSpec {
            count: 3,
            sharing: ObservationSharing::BoxContain,
            allocation: RegionAllocation::JointAcquisition,
            controller: RegionController::FixedRadius,
            success_tolerance: 4,
            failure_tolerance: 8,
            restart_memory: RegionMemory::RetainRelevant,
            restart_candidates: 8,
            restart_center: RestartCenter::Maximin,
            acquisitions: vec![
                Some(crate::config::AcquisitionSpec::Ucb { beta: 1.0 }),
                Some(crate::config::AcquisitionSpec::Ucb { beta: 1.0 }),
                Some(crate::config::AcquisitionSpec::Ucb { beta: 1.0 }),
            ],
        };
        let search = RegionalSearch::from_spec(vec![r0, r1, r2], &spec).unwrap();
        let mut ask = RoundConfig::default();
        ask.surrogate.neighbors = 1;
        ask.acquisition.beta = 1.0;

        // JointAcquisition must pick slot 2 (score ~ 100+) rather than slot 0!
        let (search, prop) = search.ask(77, ask).unwrap();
        assert_eq!(prop.ticket().region.slot, 2);
        let batch_rows = search.materialize_batch(&prop).unwrap();
        assert_eq!(batch_rows.len(), 2);
        assert_eq!(batch_rows[0].read_u16().unwrap().len(), 8);
        assert_eq!(batch_rows[1].read_u16().unwrap().len(), 8);
        let (search, _dec) = search.tell(&prop, 105.0, 0.0).unwrap();
        assert_eq!(search.trials(), [0, 0, 1]);

        // Restart slot 0 near 0x3a00: RetainRelevant imports relevant archived
        // points!
        let fresh_r0 = make_region(0x3a00, 50.0);
        assert_eq!(fresh_r0.history_len().unwrap(), 1);
        let search = search.restart(0, fresh_r0).unwrap();
        assert!(
            search.states[0].history_len().unwrap() > 1,
            "RetainRelevant must import relevant archived observations into restarted region"
        );

        // Maximin center selection must prefer a candidate far from active
        // centers (near 0.0 vs 0.625)
        let runtime = Runtime::shared().unwrap();
        let near_active = runtime.buffer_with(&[0x3900u16; 8]);
        let far_corner = runtime.buffer_with(&[0x0000u16; 8]);
        let chosen = search.choose_center(1, &[near_active, far_corner]).unwrap();
        assert_eq!(chosen, 1);
    });
}

#[test]
fn arms_winner() {
    use crate::config::{ObjectiveAcquisition, ObjectivePolicy};
    use crate::morbo::Rescalarize;
    use crate::objective_observation::{ObjectiveEstimate, ObjectiveObservation};
    autoreleasepool(|| {
        // 1. Scalar 4-arm batch where arm 2 (not arm 0) strictly improves
        //    incumbent!
        let dims = 8usize;
        let pool = crate::procedural_pool::ProceduralPoolConfig {
            arms: 4,
            candidates_per_arm: 8,
        }
        .resolve_resident()
        .unwrap();
        let mut ask = RoundConfig::default();
        ask.surrogate.neighbors = 1;
        ask.acquisition.beta = 1.0;

        let scalar_state = SearchState::new_implicit(
            &[0x3800u16; 8],
            vec![ParamBlock::new(11, 0, dims, 0.3, 0.125).unwrap()],
            16,
            TRLengthConfig::new(0.6, 0.05, 1.6),
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
            local_scale_neighbors: 1,
        })
        .unwrap()
        .configure_sampler(ResidentSampler::Raasp, Some(4), true)
        .unwrap()
        .observe_initial(1.0, 0.0)
        .unwrap();

        let search = RegionalSearch::new(vec![scalar_state], Allocation::LeastTrials).unwrap();
        let (search, prop) = search.ask(31337, ask).unwrap();
        let rows = search.materialize_batch(&prop).unwrap();
        assert_eq!(rows.len(), 4);
        let arm2_words = rows[2].read_u16().unwrap();
        assert_ne!(arm2_words, rows[0].read_u16().unwrap());

        let scalar_obs = vec![
            ObjectiveObservation::scalar_adapter(-5.0, 0.0),
            ObjectiveObservation::scalar_adapter(-2.0, 0.0),
            ObjectiveObservation::scalar_adapter(10.0, 0.0),
            ObjectiveObservation::scalar_adapter(3.0, 0.0),
        ];
        let (search, dec) = search.tell_arms(&prop, &rows, &scalar_obs).unwrap();
        assert!(dec.accepted, "arm 2 improvement must accept the batch");
        assert_eq!(
            search.states[0].read_best().unwrap(),
            arm2_words,
            "incumbent base buffer must advance to winning arm 2"
        );
        assert_eq!(
            search.states[0].history_len().unwrap(),
            5,
            "all 4 arms plus initial observation must be retained in history"
        );

        // 2. Vector 2-objective MORBO + Pareto 4-arm batches where arm 2 wins!
        let est = |mean| ObjectiveEstimate {
            mean,
            variance: 0.0,
        };
        for policy in [
            ObjectivePolicy {
                acquisition: ObjectiveAcquisition::Pareto,
                scales: vec![1.0, 1.0],
            },
            ObjectivePolicy {
                acquisition: ObjectiveAcquisition::Morbo {
                    regions: 2,
                    weights: vec![0.5, 0.5],
                    seed: 42,
                    alpha: 0.05,
                    rescalarize: Rescalarize::OnPropose,
                    clip: true,
                },
                scales: vec![1.0, 1.0],
            },
        ] {
            let init_obs = ObjectiveObservation::new(&[est(0.0), est(0.0)], est(0.0)).unwrap();
            let vec_state = SearchState::new_implicit(
                &[0x3800u16; 8],
                vec![ParamBlock::new(11, 0, dims, 0.3, 0.125).unwrap()],
                16,
                TRLengthConfig::new(0.6, 0.05, 1.6),
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
                local_scale_neighbors: 1,
            })
            .unwrap()
            .configure_sampler(ResidentSampler::Sobol, Some(8), true)
            .unwrap()
            .observe_objectives(init_obs)
            .unwrap()
            .configure_objectives(Some(policy))
            .unwrap();

            let vsearch = RegionalSearch::new(vec![vec_state], Allocation::LeastTrials).unwrap();
            let (vsearch, vprop) = vsearch.ask(999, ask).unwrap();
            let vrows = vsearch.materialize_batch(&vprop).unwrap();
            assert_eq!(vrows.len(), 4);
            let win_words = vrows[2].read_u16().unwrap();

            let vec_obs = vec![
                ObjectiveObservation::new(&[est(-2.0), est(-2.0)], est(-2.0)).unwrap(),
                ObjectiveObservation::new(&[est(-1.0), est(-1.0)], est(-1.0)).unwrap(),
                ObjectiveObservation::new(&[est(5.0), est(6.0)], est(5.0)).unwrap(),
                ObjectiveObservation::new(&[est(1.0), est(1.0)], est(1.0)).unwrap(),
            ];
            let (vsearch, vdec) = vsearch.tell_arms(&vprop, &vrows, &vec_obs).unwrap();
            assert!(vdec.accepted, "vector arm 2 winner must accept");
            assert_eq!(
                vsearch.states[0].read_best().unwrap(),
                win_words,
                "vector incumbent base must advance to arm 2"
            );
            assert_eq!(
                vsearch.states[0].history_len().unwrap(),
                5,
                "all 4 vector arms plus initial observation must enter history"
            );
            // Also verify `import_vec` directly!
            let extra_buf = vrows[1].wait().unwrap();
            let vsearch = vsearch
                .import_vec(
                    0,
                    extra_buf,
                    ObjectiveObservation::new(&[est(0.5), est(0.5)], est(0.5)).unwrap(),
                )
                .unwrap();
            assert_eq!(vsearch.states[0].history_len().unwrap(), 6);
        }
    });
}
