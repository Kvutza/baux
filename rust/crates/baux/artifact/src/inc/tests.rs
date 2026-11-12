use super::*;

fn sample_trace(rep: u32, init_loss: f64, final_loss: f64, len: usize) -> SeedLossTrace {
    let mut loss_curve = Vec::with_capacity(len);
    for i in 0..len {
        let frac = (i + 1) as f64 / len as f64;
        loss_curve.push(init_loss + frac * (final_loss - init_loss));
    }
    SeedLossTrace {
        rep,
        seed: rep as u64 * 42,
        init_loss,
        final_loss,
        accepts: 5,
        elapsed_s: 1.0,
        loss_curve,
    }
}

fn sample_head(completed_round: u32) -> TrajHead {
    TrajHead {
        env_digest: "env1".into(),
        opt_digest: "opt1".into(),
        rep: 0,
        seed: 12345,
        num_dim: 10,
        completed_round,
        has_eval_trace: true,
        initial_error: 10.0,
        best_score: 1.0,
        total_accepts: 3,
        elapsed_s: 2.0,
        worker_s: 1.8,
    }
}

#[test]
fn test_memo() {
    let tmp = tempfile::tempdir().unwrap();
    let store = IncStore::open(tmp.path()).unwrap();
    let env = "env-sphere";
    let o1 = EvalOutcome {
        scores: vec![1.0, 2.0],
        viol: 0.0,
        cost_us: 100,
    };
    let o2 = EvalOutcome {
        scores: vec![3.0],
        viol: 0.1,
        cost_us: 200,
    };
    store
        .put_evals(env, &[("h1".into(), o1.clone()), ("h2".into(), o2.clone())])
        .unwrap();

    let batch = store
        .get_evals(env, &["h1".into(), "h2".into(), "h3".into()])
        .unwrap();
    assert_eq!(batch[0], Some(o1.clone()));
    assert_eq!(batch[1], Some(o2));
    assert_eq!(batch[2], None);

    assert_eq!(store.get_eval(env, "h1").unwrap(), Some(o1));
    assert_eq!(store.get_eval(env, "h3").unwrap(), None);
}

#[test]
fn test_prefix() {
    let tmp = tempfile::tempdir().unwrap();
    let store = IncStore::open(tmp.path()).unwrap();
    let traj_id = traj_key("env1", "opt1", 0);
    let mut steps = Vec::new();
    for r in 1..=10 {
        steps.push(StepRecord {
            eval_scores: vec![r as f64],
            eval_viol: 0.0,
            best_score: 10.0 - (r as f64),
            best_viol: 0.0,
            accepted: r % 2 == 0,
            cost_us: 50,
        });
    }
    let head10 = sample_head(10);
    store.append_steps(&traj_id, &head10, 1, &steps).unwrap();

    let slice5 = store.slice_trajectory(&traj_id, 5).unwrap().unwrap();
    assert_eq!(slice5.horizon, 5);
    assert_eq!(slice5.best_scores.len(), 5);
    assert_eq!(slice5.final_score, 5.0);
    assert_eq!(slice5.accepts_at_horizon, 2);

    assert_eq!(store.slice_trajectory(&traj_id, 15).unwrap(), None);

    let (head, loaded_steps) = store.load_prefix(&traj_id).unwrap().unwrap();
    assert_eq!(head.completed_round, 10);
    assert_eq!(loaded_steps.len(), 10);

    let mut head4 = sample_head(4);
    head4.best_score = 99.0;
    store
        .append_steps(&traj_id, &head4, 1, &steps[..4])
        .unwrap();

    let (head_after, _) = store.load_prefix(&traj_id).unwrap().unwrap();
    assert_eq!(head_after.completed_round, 10);
    assert_eq!(head_after.best_score, 1.0);

    let stats = store.stats().unwrap();
    assert_eq!(stats.trajectories, 1);
    assert!(stats.step_rows >= 10);
}

#[test]
fn test_paired() {
    let b_traces = vec![
        sample_trace(1, 10.0, 5.0, 10),
        sample_trace(2, 10.0, 6.0, 10),
        sample_trace(3, 10.0, 7.0, 10),
    ];
    let a1_traces = vec![
        sample_trace(1, 10.0, 4.0, 10),
        sample_trace(2, 10.0, 4.0, 10),
        sample_trace(3, 10.0, 4.0, 10),
    ];
    let a2_traces = vec![
        sample_trace(1, 10.0, 5.0, 10),
        sample_trace(2, 10.0, 6.0, 10),
        sample_trace(3, 10.0, 7.0, 10),
    ];
    let arms = vec![
        ArmInput {
            cell_id: "c_base".into(),
            arm_id: "base".into(),
            baseline: true,
            factors: vec![],
            traces: b_traces,
        },
        ArmInput {
            cell_id: "c_arm1".into(),
            arm_id: "arm1".into(),
            baseline: false,
            factors: vec![],
            traces: a1_traces,
        },
        ArmInput {
            cell_id: "c_arm2".into(),
            arm_id: "arm2".into(),
            baseline: false,
            factors: vec![],
            traces: a2_traces,
        },
    ];
    let report = paired_report("env_test", 10, &arms).unwrap();
    assert_eq!(report.common_reps, vec![1, 2, 3]);
    assert_eq!(report.max_paired_init_drift, 0.0);
    assert_eq!(report.arms[0].p_raw, 1.0);
    assert_eq!(report.arms[0].p_holm, 1.0);
    assert!((report.arms[1].p_raw - 0.25).abs() < 1e-6);
    assert_eq!(report.arms[2].p_raw, 1.0);
    assert!((report.arms[1].p_holm - 0.50).abs() < 1e-6);
    assert_eq!(report.arms[2].p_holm, 1.0);
}

#[test]
fn test_pruning() {
    let b_traces = vec![
        sample_trace(1, 10.0, 5.0, 5),
        sample_trace(2, 10.0, 5.0, 5),
        sample_trace(3, 10.0, 5.0, 5),
    ];
    let dom_traces = vec![
        sample_trace(1, 10.0, 15.0, 5),
        sample_trace(2, 10.0, 15.0, 5),
        sample_trace(3, 10.0, 15.0, 5),
    ];
    let good_traces = vec![
        sample_trace(1, 10.0, 3.0, 5),
        sample_trace(2, 10.0, 3.0, 5),
        sample_trace(3, 10.0, 3.0, 5),
    ];
    let arms = vec![
        ArmInput {
            cell_id: "base".into(),
            arm_id: "b".into(),
            baseline: true,
            factors: vec![],
            traces: b_traces,
        },
        ArmInput {
            cell_id: "dom".into(),
            arm_id: "d".into(),
            baseline: false,
            factors: vec![],
            traces: dom_traces,
        },
        ArmInput {
            cell_id: "good".into(),
            arm_id: "g".into(),
            baseline: false,
            factors: vec![],
            traces: good_traces,
        },
    ];
    let report = paired_report("env_test", 5, &arms).unwrap();

    let active = active_cells(&report, true, 2.0);
    assert!(active.contains("base"));
    assert!(active.contains("good"));
    assert!(!active.contains("dom"));

    let all = active_cells(&report, false, 2.0);
    assert_eq!(all.len(), 3);
}
