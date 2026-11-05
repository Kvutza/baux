use super::tell::retire_aftertell;
use super::*;
use crate::objective_observation::{ObjectiveEstimate, ObjectiveObservation};

fn state() -> SearchState {
    let mut state = SearchState::new_fp16(
        &[0x3400; 4],
        vec![ParamBlock::new(1, 0, 4, 0.25, 1.0).unwrap()],
        2,
        TRLengthConfig::new(0.1, 0.001, 0.4),
        ProposalKernel::Gaussian,
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
        sharing: ObservationSharing::Independent,
        allocation: RegionAllocation::Balanced,
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
    let mut ledger = RegionalLedger::new(&[-1.0], Allocation::Balanced).unwrap();
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
            sharing: ObservationSharing::Independent,
            allocation: RegionAllocation::UtilityUcb,
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
