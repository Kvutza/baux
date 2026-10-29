use super::*;
use baux::semantics::optimizer::{AskPlan, EvalReceipt, WorldId};

fn state() -> (SearchState, RoundConfig) {
    let base = [0x3c00; 17];
    let mut ask = RoundConfig::default();
    ask.surrogate.neighbors = 1;
    let mut search = SearchState::new_implicit(
        &base,
        vec![ParamBlock::new(73, 0, base.len(), 0.25, 1.0).unwrap()],
        1,
        TRLengthConfig::new(0.1, 0.001, 0.4),
        ProposalKernel::Gaussian,
    )
    .unwrap();
    search
        .configure_enn(crate::config::ResidentEnnConfig {
            pool: crate::procedural_pool::ProceduralPoolConfig {
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
    search.observe_initial(0.0, 0.0).unwrap();
    (search, ask)
}

fn world(step: u64) -> WorldId {
    WorldId::RotatingCondition {
        step,
        window_id: 42,
        tokens: 16,
    }
}

#[test]
fn candidate_receipts() {
    autoreleasepool(|| {
        let (mut search, ask) = state();
        for (step, value) in [1e6_f32, -1e6].into_iter().enumerate() {
            let before = search.read_best().unwrap();
            let base_id = search.base_id;
            let (row, proposal, initializing) =
                search.propose(73 + step as u64, ask, 0, None).unwrap();
            assert!(!initializing);
            let proposed = read::<u16>(&row, before.len());
            let plan = search
                .plan_evaluation(&proposal, world(step as u64))
                .unwrap();
            assert!(search.plan_evaluation(&proposal, world(99)).is_err());
            let receipt = EvalReceipt::new(world(step as u64), 16, f64::from(-value), 0.0, None);
            let evaluation = plan.record(receipt).unwrap();
            let mut foreign = evaluation.round();
            foreign.owner += 1;
            let wrong = AskPlan::new(foreign, receipt.world)
                .record(receipt)
                .unwrap();
            assert!(search.tell_evaluated(&proposal, wrong).is_err());
            assert_eq!(search.read_best().unwrap(), before);
            assert_eq!(search.base_id, base_id);
            let duplicate = search
                .plan_evaluation(&proposal, receipt.world)
                .unwrap()
                .record(receipt)
                .unwrap();
            let mut edited = proposal.clone();
            edited.incumbent_mean = -value;
            edited.seed = u64::MAX;
            edited.length = 1000.0;
            let result = search.tell_evaluated(&edited, evaluation).unwrap();
            assert_eq!(result.incumbent_value, proposal.incumbent_mean);
            assert_eq!(result.accepted, step == 0);
            assert_eq!(
                search.read_best().unwrap(),
                if result.accepted { proposed } else { before }
            );
            assert_eq!(
                search.base_id,
                if result.accepted {
                    search.observation
                } else {
                    base_id
                }
            );
            assert!(search.tell_evaluated(&proposal, duplicate).is_err());
            assert_eq!(search.sync().unwrap(), vec![result.accepted]);
        }
    });
}

#[test]
fn partial_commit() {
    autoreleasepool(|| {
        let (mut search, ask) = state();
        let (row, proposal, _) = search.propose(73, ask, 0, None).unwrap();
        let proposed = read::<u16>(&row, 17);
        let evaluation = search
            .plan_evaluation(&proposal, world(0))
            .unwrap()
            .record(EvalReceipt::new(world(0), 16, -1e6, 0.0, None))
            .unwrap();
        // Inject a bookkeeping failure after the physical weight commit.
        // The public contract is poison-and-stop, not rollback or retry.
        search.exact_history = true;
        search.pending.as_mut().unwrap().block_scales.clear();
        let error = search.tell_evaluated(&proposal, evaluation).unwrap_err();
        assert_eq!(
            error,
            "Exact replay proposal has the wrong block-scale count"
        );
        assert_eq!(read::<u16>(&search.base, 17), proposed);
        assert!(search.poisoned);
        assert!(search.sync().is_err());
        assert!(search.propose(74, ask, 0, None).is_err());
    });
}

#[test]
fn seam_roundtrip() {
    autoreleasepool(|| {
        let (mut search, _ask) = state();
        let mut surrogate = baux::semantics::optimizer::Surrogate::new(0.0, 0.0, 73)
            .unwrap()
            .with_metric(baux::semantics::neighbor::DistanceMetric::SelfTuning { local_k: 2 });
        let mut trust_region =
            baux::semantics::optimizer::TrustRegionController::new(0.1, 0.001, 0.4).unwrap();
        for (step, value) in [1.5_f32, -0.5_f32].into_iter().enumerate() {
            let before = search.read_best().unwrap();
            let geometry = search.sample_distances(91 + step as u64).unwrap();
            assert_eq!(geometry.len(), 1);
            assert_eq!(geometry[0].len(), surrogate.len());
            let selection = surrogate.evaluate_candidate(&geometry[0]).unwrap();
            let (row, proposal) = search
                .materialize_selection(91 + step as u64, &selection)
                .unwrap();
            let proposed = read::<u16>(&row, before.len());
            let decision = trust_region
                .evaluate_selection(value, 0.0, proposal.length, &selection)
                .unwrap();
            assert_eq!(decision.accepted, step == 0);
            let target_slot = surrogate
                .append(&geometry[selection.selected], value, 0.0, decision.accepted)
                .unwrap();
            search
                .commit_host(
                    &proposal,
                    target_slot,
                    crate::objective_observation::ObjectiveObservation::scalar_adapter(value, 0.0),
                    decision.accepted,
                    f64::from(decision.next_radius),
                )
                .unwrap();
            assert_eq!(search.sync().unwrap(), vec![decision.accepted]);
            assert_eq!(
                search.read_best().unwrap(),
                if decision.accepted { proposed } else { before }
            );
        }
    });
}
