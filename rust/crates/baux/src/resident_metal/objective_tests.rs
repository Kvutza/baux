use super::*;
use crate::objective_observation::{ObjectiveEstimate, ObjectiveObservation};

fn measured(mean: f32, first: f32, second: f32) -> ObjectiveObservation {
    let point = |value| ObjectiveEstimate {
        mean: value,
        variance: 0.0,
    };
    ObjectiveObservation::new(&[point(first), point(second)], point(mean)).unwrap()
}

fn state(initial: ObjectiveObservation) -> SearchState {
    let base = [0x3e80; 17];
    let blocks = vec![ParamBlock::new(71, 0, base.len(), 0.25, 1.0).unwrap()];
    SearchState::new_objectives(
        &base,
        initial,
        blocks,
        3,
        1,
        TRLengthConfig::new(0.1, 0.001, 0.4),
    )
    .unwrap()
}

#[test]
fn vector_history() {
    autoreleasepool(|| {
        let mut search = state(measured(0.0, 10.0, -10.0));
        for step in 0..5 {
            let (s, proposal) = search
                .ask_round(1, 4, step, RoundConfig::default())
                .unwrap();
            let outcome = measured(step as f32 + 1.0, -100.0 - step as f32, 100.0 + step as f32);
            let (s, decision) = s.noisy_objectives(&proposal, outcome).unwrap();
            assert!(decision.accepted);
            let (s, synced) = s.sync().unwrap();
            assert_eq!(synced, vec![true]);
            let record = s.objective_observations().unwrap().last().unwrap();
            assert_eq!(record.identity, s.observation);
            assert_eq!(record.observation, outcome);
            assert_eq!(s.incumbent_objectives().unwrap(), Some(record));
            search = s;
        }
        assert_eq!(search.objective_observations().unwrap().len(), 3);
        for (row, objective) in search.objective_observations().unwrap().enumerate() {
            assert_eq!(objective.identity, search.identities[row]);
            assert_eq!(objective.observation.control().mean, search.outcomes[row]);
        }
    });
}

#[test]
fn scalar_adapter() {
    autoreleasepool(|| {
        let base = [0x3e80; 17];
        let block = ParamBlock::new(71, 0, base.len(), 0.25, 1.0).unwrap();
        let length = TRLengthConfig::new(0.1, 0.001, 0.4);
        let mut scalar = SearchState::new(&base, 0.0, 0.0, vec![block], 3, 1, length).unwrap();
        let mut vector = SearchState::new_objectives(
            &base,
            ObjectiveObservation::scalar(0.0, 0.0).unwrap(),
            vec![block],
            3,
            1,
            length,
        )
        .unwrap();
        for (step, value) in [1.0, -1.0, 2.0, -2.0].into_iter().enumerate() {
            let (s_a, a) = scalar
                .ask_round(1, 4, step as u64, RoundConfig::default())
                .unwrap();
            let (s_b, b) = vector
                .ask_round(1, 4, step as u64, RoundConfig::default())
                .unwrap();
            assert_eq!(
                (a.seed, a.index, a.score.to_bits()),
                (b.seed, b.index, b.score.to_bits())
            );
            let (s_a, decision_a) = s_a.tell_noisy(&a, value, 0.0).unwrap();
            let (s_b, decision_b) = s_b
                .noisy_objectives(&b, ObjectiveObservation::scalar(value, 0.0).unwrap())
                .unwrap();
            assert_eq!(decision_a, decision_b);
            let (s_a, sync_a) = s_a.sync().unwrap();
            let (s_b, sync_b) = s_b.sync().unwrap();
            assert_eq!(sync_a, sync_b);
            assert_eq!(
                s_a.length().unwrap().to_bits(),
                s_b.length().unwrap().to_bits()
            );
            let bits = |buffer: &Buffer| unsafe {
                std::slice::from_raw_parts(buffer.contents().cast::<u16>(), base.len())
            };
            assert_eq!(bits(&*s_a.base), bits(&*s_b.base));
            assert_eq!(s_a.outcomes, s_b.outcomes);
            assert_eq!(s_a.variances, s_b.variances);
            scalar = s_a;
            vector = s_b;
        }
    });
}

#[test]
fn retryable_schema() {
    autoreleasepool(|| {
        let search = state(measured(0.0, 1.0, 2.0));
        let (search, proposal) = search.ask_round(1, 4, 0, RoundConfig::default()).unwrap();
        let identity = search.observation;
        assert!(
            search
                .clone()
                .noisy_objectives(&proposal, ObjectiveObservation::scalar(1.0, 0.0).unwrap())
                .is_err()
        );
        assert_eq!(search.observation, identity);
        assert_eq!(search.objective_observations().unwrap().len(), 1);
        let (search, result) = search
            .noisy_objectives(&proposal, measured(1.0, 3.0, 4.0))
            .unwrap();
        assert!(result.accepted);
        let (_search, synced) = search.sync().unwrap();
        assert_eq!(synced, vec![true]);
    });
}
