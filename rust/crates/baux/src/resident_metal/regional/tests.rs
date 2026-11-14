use super::tell::retire_aftertell;
use super::*;
use crate::objective_observation::{ObjectiveEstimate, ObjectiveObservation};

fn state() -> SearchState {
    let mut state = SearchState::new_fp16(
        &[0x3400; 4],
        vec![ParamBlock::new(1, 0, 4, 0.25, 1.0).unwrap()],
        2,
        TRLengthConfig::new(0.1, 0.001, 0.4),
        ProposalDistribution::Gaussian,
    )
    .unwrap();
    state.observe_initial(-1.0, 0.0).unwrap();
    state
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
    let unsupported = RegionalSearchSpec {
        controller: RegionController::FixedRadius,
        ..spec
    };
    assert!(RegionalSearch::from_spec(vec![state(), state()], &unsupported).is_err());
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
    assert!(states[0].poisoned);
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
        let mut search =
            RegionalSearch::from_spec(vec![vector_state(), vector_state()], &spec).unwrap();
        let proposal = search.ask_region(0, 7, RoundConfig::default()).unwrap();
        let decision = objectives::tell(&mut search, &proposal, vector(1.0, 1.0, 1.0)).unwrap();
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
        let mut search = RegionalSearch::with_sharing(
            vec![first, second],
            Allocation::LeastTrials,
            SharingPolicy::EllipsoidBall,
        )
        .unwrap();
        let mut ask = RoundConfig::default();
        ask.acquisition.kind = AcquisitionKind::Random;
        ask.surrogate.neighbors = 1;
        let round0 = search.ask_region(0, 11, ask).unwrap();
        let accepted = search.buffer(&round0).unwrap();
        let bytes = read::<u16>(&accepted, 4);
        let decision = search.tell(&round0, 5.0, 0.0).unwrap();
        assert!(decision.accepted);
        assert_eq!(search.sharing_backlog(), [0, 1]);
        search.flush_sharing(1).unwrap();
        assert_eq!(search.sharing_backlog(), [0, 0]);
        assert_eq!(search.delivered_imports(), 1);
        let adopted = read::<u16>(&search.incumbent(1).unwrap(), 4);
        assert_eq!(adopted, bytes);
        let round1 = search.ask_region(1, 19, ask).unwrap();
        let decision1 = search.tell(&round1, 6.0, 0.0).unwrap();
        assert!(decision1.accepted);
        assert_eq!(search.best(), (1, 6.0));
    });
}
