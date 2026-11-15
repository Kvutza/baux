use super::*;
use baux::semantics::optimizer::{AskPlan, EvalReceipt, WorldId};

fn state() -> (SearchState, RoundConfig) {
    let base = [0x3c00; 17];
    let mut ask = RoundConfig::default();
    ask.surrogate.neighbors = 1;
    let search = SearchState::new_implicit(
        &base,
        vec![ParamBlock::new(73, 0, base.len(), 0.25, 1.0).unwrap()],
        1,
        TRLengthConfig::new(0.1, 0.001, 0.4),
        ProposalDistribution::Gaussian,
    )
    .unwrap();
    let search = search
        .configure_enn(crate::config::ResidentEnnConfig {
            pool: crate::procedural_pool::ProceduralPoolConfig {
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
    let search = search.observe_initial(0.0, 0.0).unwrap();
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
            let (search_step, (row, proposal, initializing)) =
                search.propose(73 + step as u64, ask, 0, None).unwrap();
            assert!(!initializing);
            let proposed = read::<u16>(&row, before.len());
            let (search_step, plan) = search_step
                .plan_evaluation(&proposal, world(step as u64))
                .unwrap();
            assert!(
                search_step
                    .clone()
                    .plan_evaluation(&proposal, world(99))
                    .is_err()
            );
            let receipt = EvalReceipt::new(world(step as u64), 16, f64::from(-value), 0.0, None);
            let evaluation = plan.record(receipt).unwrap();
            let mut foreign = evaluation.round();
            foreign.proposal = foreign.proposal.wrapping_add(1);
            let wrong = AskPlan::new(foreign, receipt.world)
                .record(receipt)
                .unwrap();
            assert!(
                search_step
                    .clone()
                    .tell_evaluated(&proposal, wrong)
                    .is_err()
            );
            assert_eq!(search_step.read_best().unwrap(), before);
            assert_eq!(search_step.base_id, base_id);
            let (search_step, duplicate_plan) = search_step
                .plan_evaluation(&proposal, receipt.world)
                .unwrap();
            let duplicate = duplicate_plan.record(receipt).unwrap();
            let mut edited = proposal.clone();
            edited.incumbent_mean = -value;
            edited.seed = u64::MAX;
            edited.length = 1000.0;
            let (search_step, result) = search_step.tell_evaluated(&edited, evaluation).unwrap();
            assert_eq!(result.incumbent_value, proposal.incumbent_mean);
            assert_eq!(result.accepted, step == 0);
            assert_eq!(
                search_step.read_best().unwrap(),
                if result.accepted { proposed } else { before }
            );
            assert_eq!(
                search_step.base_id,
                if result.accepted {
                    search_step.observation
                } else {
                    base_id
                }
            );
            assert!(
                search_step
                    .clone()
                    .tell_evaluated(&proposal, duplicate)
                    .is_err()
            );
            let (search_step, synced) = search_step.sync().unwrap();
            assert_eq!(synced, vec![result.accepted]);
            search = search_step;
        }
    });
}

#[test]
fn partial_commit() {
    autoreleasepool(|| {
        let (search, ask) = state();
        let (search, (row, proposal, _)) = search.propose(73, ask, 0, None).unwrap();
        let _proposed = read::<u16>(&row, 17);
        let (search, plan) = search.plan_evaluation(&proposal, world(0)).unwrap();
        let evaluation = plan
            .record(EvalReceipt::new(world(0), 16, -1e6, 0.0, None))
            .unwrap();
        // Inject a bookkeeping failure after the physical weight commit on an
        // isolated branch. The public contract is poison-and-stop, not
        // rollback or retry.
        let mut doomed = search.clone();
        doomed.fit_params = true;
        doomed.fitter = None;
        let error = match doomed.tell_evaluated(&proposal, evaluation) {
            Err(e) => e,
            Ok(_) => panic!("expected tell_evaluated to fail"),
        };
        assert_eq!(error, "Metal implicit ENN is not configured");
        // COW isolation: the parent search snapshot retains its exact prior
        // unpoisoned state
        assert!(!search.poisoned);
        assert!(search.check_healthy().is_ok());
    });
}

#[test]
fn seam_roundtrip() {
    autoreleasepool(|| {
        let (mut search, _ask) = state();
        let mut surrogate = baux::semantics::optimizer::Surrogate::new(0.0, 0.0, 73)
            .unwrap()
            .with_geometry(baux::semantics::neighbor::DistanceMetric::Conformal { local_k: 2 });
        let mut trust_region =
            baux::semantics::optimizer::TrustRegionController::new(0.1, 0.001, 0.4).unwrap();
        for (step, value) in [1.5_f32, -0.5_f32].into_iter().enumerate() {
            let before = search.read_best().unwrap();
            let (search_step, geometry) = search.sample_distances(91 + step as u64).unwrap();
            assert_eq!(geometry.len(), 1);
            assert_eq!(geometry[0].len(), surrogate.len());
            let selection = surrogate
                .select_pool(
                    &geometry,
                    step as u32,
                    baux::semantics::optimizer::AcquisitionRule::Ucb { beta: 0.0 },
                    None,
                )
                .unwrap();
            let (search_step, (row, proposal)) = search_step
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
            let search_step = search_step
                .commit_host(
                    &proposal,
                    target_slot,
                    crate::objective_observation::ObjectiveObservation::scalar_adapter(value, 0.0),
                    decision.accepted,
                    f64::from(decision.next_radius),
                )
                .unwrap();
            let (search_step, synced) = search_step.sync().unwrap();
            assert_eq!(synced, vec![decision.accepted]);
            assert_eq!(
                search_step.read_best().unwrap(),
                if decision.accepted { proposed } else { before }
            );
            search = search_step;
        }
    });
}

#[test]
fn branch_isolation() {
    autoreleasepool(|| {
        let (search, ask) = state();
        let base_snapshot = search.read_best().unwrap();
        assert_eq!(search.history_len().unwrap(), 1);

        // Branch A: fork/clone search and advance it with an accepted update
        let branch_a = search.clone();
        let (branch_a, (row_a, prop_a, _)) = branch_a.propose(101, ask, 0, None).unwrap();
        let proposed_a = read::<u16>(&row_a, base_snapshot.len());
        let (branch_a, plan_a) = branch_a.plan_evaluation(&prop_a, world(10)).unwrap();
        let receipt_a = EvalReceipt::new(world(10), 16, -100.0, 0.0, None);
        let eval_a = plan_a.record(receipt_a).unwrap();
        let (branch_a, dec_a) = branch_a.tell_evaluated(&prop_a, eval_a).unwrap();
        let (branch_a, sync_a) = branch_a.sync().unwrap();
        assert!(dec_a.accepted);
        assert_eq!(sync_a, vec![true]);
        assert_eq!(branch_a.read_best().unwrap(), proposed_a);
        assert_ne!(branch_a.read_best().unwrap(), base_snapshot);
        assert_eq!(branch_a.history_len().unwrap(), 2);

        // Branch B: fork/clone from root search independently and advance it
        // with another update
        let branch_b = search.clone();
        let (branch_b, (row_b, prop_b, _)) = branch_b.propose(202, ask, 0, None).unwrap();
        let proposed_b = read::<u16>(&row_b, base_snapshot.len());
        let (branch_b, plan_b) = branch_b.plan_evaluation(&prop_b, world(20)).unwrap();
        let receipt_b = EvalReceipt::new(world(20), 16, -200.0, 0.0, None);
        let eval_b = plan_b.record(receipt_b).unwrap();
        let (branch_b, dec_b) = branch_b.tell_evaluated(&prop_b, eval_b).unwrap();
        let (branch_b, sync_b) = branch_b.sync().unwrap();
        assert!(dec_b.accepted);
        assert_eq!(sync_b, vec![true]);
        assert_eq!(branch_b.read_best().unwrap(), proposed_b);
        assert_ne!(branch_b.read_best().unwrap(), base_snapshot);
        assert_ne!(branch_a.read_best().unwrap(), branch_b.read_best().unwrap());
        assert_eq!(branch_b.history_len().unwrap(), 2);

        // Invariant: The original root snapshot MUST retain its exact prior GPU
        // buffer and history!
        assert_eq!(search.read_best().unwrap(), base_snapshot);
        assert_eq!(search.history_len().unwrap(), 1);
        assert_eq!(search.best().unwrap(), 0.0);
        assert_eq!(branch_a.best().unwrap(), 100.0);
        assert_eq!(branch_b.best().unwrap(), 200.0);
    });
}

#[test]
fn buffer_isolation() {
    autoreleasepool(|| {
        let (search, ask) = state();
        let original_weights = search.read_best().unwrap();

        // Fork branch
        let branch = search.clone();

        // Propose on branch
        let (branch, (row, proposal, _)) = branch.propose(303, ask, 0, None).unwrap();
        let branch_proposed = read::<u16>(&row, original_weights.len());

        // Root search must not have any pending proposal or corrupted buffers
        assert_eq!(search.read_best().unwrap(), original_weights);
        assert_eq!(search.history_len().unwrap(), 1);

        // Reject update on branch
        let (branch, plan) = branch.plan_evaluation(&proposal, world(30)).unwrap();
        let receipt = EvalReceipt::new(world(30), 16, 1e6, 0.0, None);
        let eval = plan.record(receipt).unwrap();
        let (branch, dec) = branch.tell_evaluated(&proposal, eval).unwrap();
        let (branch, sync) = branch.sync().unwrap();
        assert!(!dec.accepted);
        assert_eq!(sync, vec![false]);

        // Branch best should still be original, but branch history must have
        // grown to 2
        assert_eq!(branch.read_best().unwrap(), original_weights);
        assert_eq!(branch.history_len().unwrap(), 2);

        // Root search history must still be 1, weights intact
        assert_eq!(search.history_len().unwrap(), 1);
        assert_eq!(search.read_best().unwrap(), original_weights);
        assert_ne!(branch_proposed, original_weights);
    });
}
