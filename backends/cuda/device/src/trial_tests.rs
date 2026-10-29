use crate::{SearchState, TellParams, bf16};

fn state() -> SearchState {
    SearchState {
        length: 0.01,
        length_init: 0.01,
        length_min: 0.0001,
        length_max: 0.08,
        trust_best: -3.0,
        hist_min: -3.0,
        hist_max: -3.0,
        best: -3.0,
        best_variance: 0.5,
        prev_obs: 1,
        successes: 0,
        failures: 0,
        restarts: 0,
        history: 1,
        status: 0,
    }
}

#[test]
fn fifo_anchor() {
    for capacity in [2, 3, 128] {
        let mut local = state();
        let mut history_slots = vec![0; capacity];
        let mut outcomes = vec![0.0; capacity];
        let mut variances = vec![0.0; capacity];
        let mut destinations = [0];
        history_slots[0] = 1;
        let mut params = TellParams {
            row_stride: 128,
            row_len: 1,
            trials: 1,
            capacity: capacity as u32,
            failure_tolerance: 4,
            status_count: 1,
            correlated: 1,
            paired: 2,
            accept: 0,
            incumbent_value: -3.0,
            incumbent_variance: 0.5,
            improvement: 0.0,
            improvement_variance: 0.0,
            reject_is_failure: 1,
            trial_slots: [0; 32],
        };
        for epoch in 0..2 {
            params.reject_is_failure = epoch;
            for step in 1..=capacity * 3 {
                params.improvement = -(step as f32) * 0.01;
                params.improvement_variance = step as f32 * 0.001;
                tell_relativehistory!(
                    local,
                    history_slots,
                    outcomes,
                    variances,
                    destinations,
                    params
                );
                assert_eq!(local.history as usize, (step + 1).min(capacity));
                assert_eq!((history_slots[0], outcomes[0], variances[0]), (1, 0.0, 0.0));
                assert!((2..=capacity as u32).contains(&destinations[0]));
                let len = local.history as usize;
                let mut resident = history_slots[..len].to_vec();
                resident.sort_unstable();
                resident.dedup();
                assert_eq!(resident.len(), len);
                for index in 1..len {
                    let observed_step = step - (len - 1 - index);
                    assert_eq!(outcomes[index], -(observed_step as f32) * 0.01);
                    assert_eq!(variances[index], observed_step as f32 * 0.001);
                }
            }
            params.accept = 1;
            tell_relativehistory!(
                local,
                history_slots,
                outcomes,
                variances,
                destinations,
                params
            );
            assert_eq!(local.history, 1, "epoch {epoch}");
            assert_eq!(destinations[0], 1);
            assert_eq!((history_slots[0], outcomes[0], variances[0]), (1, 0.0, 0.0));
            params.accept = 0;
        }
    }
}

#[test]
fn relative_state() {
    let mut local = state();
    for rejection in 1..=40 {
        let previous = local.length;
        let selected = bf16::candidate_radius(local, 1);
        bf16::paired_radius(&mut local, selected, false, 4, true);
        assert_eq!(local.failures, rejection % 4);
        assert_eq!(
            local.length,
            if rejection % 4 == 0 {
                (previous * 0.5).max(local.length_min)
            } else {
                previous
            }
        );
        assert_eq!(local.restarts, 0);
    }
    assert_eq!(local.length, local.length_min);
    for candidate in 0..4 {
        let selected = bf16::candidate_radius(local, candidate);
        local.failures = 3;
        bf16::paired_radius(&mut local, selected, true, 4, true);
        assert_eq!(local.length, f64::from(selected));
        assert_eq!(local.failures, 0);
        assert_eq!(local.restarts, 0);
    }
    local.length = local.length_max;
    let selected = bf16::candidate_radius(local, 1);
    bf16::paired_radius(&mut local, selected, false, 1, true);
    assert_eq!(local.length, local.length_max * 0.5);
    local.failures = u32::MAX - 1;
    bf16::paired_radius(&mut local, selected, false, u32::MAX, true);
    assert_eq!(local.failures, 0);
    assert_eq!(local.restarts, 0);
}

#[test]
fn inconclusive_state() {
    for failures in [0, 3, u32::MAX] {
        let mut local = state();
        local.failures = failures;
        let initial_length = local.length;
        let selected = bf16::candidate_radius(local, 1);
        for _ in 0..40 {
            bf16::paired_radius(&mut local, selected, false, 4, false);
            assert_eq!(local.length, initial_length);
            assert_eq!(local.failures, failures);
            assert_eq!(local.restarts, 0);
        }
        if failures == 3 {
            bf16::paired_radius(&mut local, selected, false, 4, true);
            assert_eq!(local.length, initial_length * 0.5);
            assert_eq!(local.failures, 0);
        }
    }
}

#[test]
fn radius_reset() {
    for reject_is_failure in [false, true] {
        for candidate in 0..4 {
            let mut local = state();
            local.failures = 3;
            let selected = bf16::candidate_radius(local, candidate);
            bf16::paired_radius(&mut local, selected, true, 4, reject_is_failure);
            assert_eq!(local.length, f64::from(selected));
            assert_eq!(local.failures, 0);
            assert_eq!(local.restarts, 0);
        }
    }
}
