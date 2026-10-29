//! Unit and regression tests for MuJoCo control evaluation framework.

use crate::bench_chain::{ChainPolicy, PerturbMode};
use crate::bench_oracle::ControlOracle;
use crate::bench_schedule::{ActiveSampling, RolloutSchedule, control_seed};
use crate::bench_task::{ControlTask, HalfCheetahTask, HopperTask, MjDataState};
use crate::{BayesOptimizerExt, ControlConfig, ExperimentConfig, OptimizerConfig};
use ndarray::Array2;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

#[test]
fn schedule_invariants() {
    let rep = 42;
    let frozen = RolloutSchedule::new(rep, ActiveSampling::Frozen { episodes: 4 }, 10);
    for ep in 0..4 {
        assert_eq!(frozen.active_seed(0, ep), frozen.active_seed(1, ep));
        assert_eq!(frozen.active_seed(0, ep), frozen.active_seed(5, ep));
    }

    let natural = RolloutSchedule::new(rep, ActiveSampling::Natural { episodes: 4 }, 10);
    assert_ne!(natural.active_seed(0, 0), natural.active_seed(1, 0));
    assert_eq!(natural.active_seed(0, 1), natural.active_seed(0, 1));

    for ep in 0..4 {
        assert_ne!(frozen.active_seed(0, ep), frozen.heldout_seed(ep));
        assert_ne!(natural.active_seed(0, ep), natural.heldout_seed(ep));
    }
}

#[test]
fn domain_seeds() {
    let s1 = control_seed(1, "mujoco-active", 0);
    let s2 = control_seed(1, "mujoco-heldout", 0);
    let s3 = control_seed(1, "mujoco-init", 0);
    let s4 = control_seed(2, "mujoco-active", 0);
    assert_ne!(s1, s2);
    assert_ne!(s1, s3);
    assert_ne!(s2, s3);
    assert_ne!(s1, s4);
}

#[test]
fn cheetah_rollout() {
    let task = HalfCheetahTask::new();
    let mut state1 = MjDataState::new(task.model());
    let mut state2 = MjDataState::new(task.model());
    task.reset(&mut state1, 12345);
    task.reset(&mut state2, 12345);
    assert_eq!(state1.qpos, state2.qpos);
    assert_eq!(state1.qvel, state2.qvel);

    let action = vec![0.5f32; 6];
    let out1 = task.step(&mut state1, &action);
    let out2 = task.step(&mut state2, &action);
    assert_eq!(out1, out2);
    assert_eq!(state1.qpos, state2.qpos);

    let mut zero_state = MjDataState::new(task.model());
    task.reset(&mut zero_state, 77);
    let zero_out = task.step(&mut zero_state, &[0.0; 6]);
    for act_idx in 0..6 {
        let mut s = MjDataState::new(task.model());
        task.reset(&mut s, 77);
        let mut act = [0.0f32; 6];
        act[act_idx] = 1.0;
        let out = task.step(&mut s, &act);
        assert!(
            (out.reward_forward - zero_out.reward_forward).abs() > 1e-5,
            "actuator {act_idx} must contribute to propulsion"
        );
    }
}

#[test]
fn hopper_rollout() {
    let task = HopperTask::new();
    let mut state1 = MjDataState::new(task.model());
    let mut state2 = MjDataState::new(task.model());
    task.reset(&mut state1, 54321);
    task.reset(&mut state2, 54321);
    assert_eq!(state1.qpos, state2.qpos);
    assert_eq!(state1.qvel, state2.qvel);

    let action = vec![0.1f32, 0.4f32, 0.2f32];
    let out1 = task.step(&mut state1, &action);
    let out2 = task.step(&mut state2, &action);
    assert_eq!(out1, out2);
    assert_eq!(state1.qpos, state2.qpos);

    let mut survive_state = MjDataState::new(task.model());
    task.reset(&mut survive_state, 54321);
    for _ in 0..75 {
        let out = task.step(&mut survive_state, &[0.0, 0.0, 0.0]);
        assert!(
            !out.terminated,
            "capable Hopper control must survive past step 42 (terminated at step {}, z={}, \
             angle={})",
            survive_state.steps, survive_state.qpos[1], survive_state.qpos[2]
        );
    }
    assert_eq!(survive_state.steps, 75);
}

#[test]
fn policy_shape() {
    let mut policy = ChainPolicy::new(&[17, 32, 16, 6]);
    let obs = vec![0.1f32; 17];
    let mut act = vec![0.0f32; 6];
    policy.forward(&obs, &mut act);
    for &a in &act {
        assert!((-1.0..=1.0).contains(&a));
    }

    let mut linear = ChainPolicy::new(&[2, 1]);
    linear.weights = vec![1.0_f64, 1.0_f64, 0.0_f64];
    let mut out_const = [0.0f32; 1];
    linear.forward(&[0.5, 0.5], &mut out_const);
    assert!(
        out_const[0] > 0.7,
        "forward must not apply cross-coordinate norm_layer that zeroes constant inputs"
    );

    let mut simba = ChainPolicy::simba(17, 128, 6);
    assert!(simba.simba);
    assert_eq!(simba.weights.len(), 134_790);
    let zeros = simba.weights.clone();
    simba.apply_perturbation(&zeros, 77, 0.25, PerturbMode::AdjointChain);
    simba.observe_norm(&obs);
    simba.forward(&obs, &mut act);
    for &a in &act {
        assert!(a.is_finite() && (-1.0..=1.0).contains(&a));
    }
}

#[test]
fn perturbations_mode() {
    let mut policy_uncoupled = ChainPolicy::new(&[17, 32, 16, 6]);
    let mut policy_adjoint = ChainPolicy::new(&[17, 32, 16, 6]);
    let base = vec![0.05_f64; policy_uncoupled.weights.len()];

    policy_uncoupled.apply_perturbation(&base, 999, 0.1, PerturbMode::Uncoupled);
    policy_adjoint.apply_perturbation(&base, 999, 0.1, PerturbMode::AdjointChain);

    assert_ne!(policy_uncoupled.weights, policy_adjoint.weights);
    for link in &policy_adjoint.links {
        let b_start = link.offset + link.in_dim * link.out_dim;
        let b_slice = &policy_adjoint.weights[b_start..b_start + link.out_dim];
        assert!(
            b_slice.iter().any(|&b| (b - 0.05).abs() > 1e-6),
            "biases must be perturbed in every layer"
        );
    }

    let mut policy_neg = ChainPolicy::new(&[17, 32, 16, 6]);
    policy_neg.apply_signed(&base, 999, 0.1, -1.0, PerturbMode::AdjointChain);
    for ((&w_pos, &w_neg), &b) in policy_adjoint
        .weights
        .iter()
        .zip(policy_neg.weights.iter())
        .zip(base.iter())
    {
        let d_pos = w_pos - b;
        let d_neg = w_neg - b;
        assert!(
            (d_pos + d_neg).abs() < 1e-6,
            "signed perturbation must be exact antipodal ray"
        );
    }
}

#[test]
fn oracle_eval() {
    assert_eq!(ExperimentConfig::default().heldout_episodes, 10);
    let task = HalfCheetahTask::new();
    let schedule = RolloutSchedule::new(1, ActiveSampling::Natural { episodes: 3 }, 3);
    let mut oracle = ControlOracle::new(task, schedule);
    let mut policy = ChainPolicy::new(&[17, 16, 6]);
    let zeros = policy.weights.clone();
    policy.apply_perturbation(&zeros, 42, 0.2, PerturbMode::Uncoupled);

    let (obs, evidence) = oracle.evaluate_candidate(&mut policy).unwrap();
    assert_eq!(evidence.episodes, 3);
    assert_eq!(evidence.round, 0);
    assert!(evidence.mean_steps > 0.0);
    assert_eq!(evidence.nonfinite_steps, 0);
    assert_eq!(obs.estimates().len(), 1);
    let inc_mean = obs.estimates()[0].mean;
    oracle.set_incumbent(inc_mean);

    let mut bad_cand = policy.clone();
    bad_cand.apply_perturbation(&zeros, 999, 0.9, PerturbMode::Uncoupled);
    let (cand_obs, _) = oracle.evaluate_candidate(&mut bad_cand).unwrap();
    assert_ne!(cand_obs.estimates()[0].mean, inc_mean);

    let heldout = oracle.evaluate_heldout(&mut policy).unwrap();
    assert_eq!(heldout.episodes, 3);
    assert!(
        (heldout.active_minus_heldout_gap - (inc_mean - heldout.mean_return)).abs() < 1e-5,
        "gap must be computed against incumbent_active_mean_return, not rejected candidate"
    );

    oracle.advance_round();
    let (_, ev_round1) = oracle.evaluate_candidate(&mut policy).unwrap();
    assert_eq!(ev_round1.round, 1);
}

#[test]
fn engine_configs() {
    let turbo_toml = include_str!("../experiments/cheetah-turbo.toml");
    let op_toml = include_str!("../experiments/cheetah-operator.toml");
    let turbo_cfg = ControlConfig::from_toml(turbo_toml).unwrap();
    let op_cfg = ControlConfig::from_toml(op_toml).unwrap();
    assert_eq!(turbo_cfg.optimizer.algorithm, "turbo-enn");
    assert!(turbo_cfg.optimizer.is_turbo());
    assert_eq!(op_cfg.optimizer.algorithm, "operator-bo");
    assert!(!op_cfg.optimizer.is_turbo());

    let mut rng = ChaCha8Rng::seed_from_u64(control_seed(0, "control-turbo", 0));
    let mut opt = turbo_cfg.optimizer.build_turbo(8, &mut rng).unwrap();
    assert!(opt.incumbent_x().is_none());
    opt.seed_incumbent(&[0.2f32; 8], 1.25, &mut rng).unwrap();
    assert_eq!(opt.obs_count(), 1);
    assert!(opt.incumbent_x().is_some());
    let x = opt.ask(1, &mut rng).unwrap();
    let y = Array2::from_elem((1, 1), 1.5);
    let yvar = Array2::from_elem((1, 1), 0.01);
    opt.tell_variance(&x.view(), &y.view(), Some(&yvar.view()), &mut rng)
        .unwrap();
    assert!(opt.incumbent_x().is_some());

    let mut rel_cfg = OptimizerConfig::default();
    rel_cfg.trust_region.policy = "turbo".to_string();
    let mut tr = rel_cfg.build_controller().unwrap();
    let mut sur = rel_cfg.build_surrogate(1.0, 0.05, 123).unwrap();
    let rule = rel_cfg.acquisition_rule(456, 0);
    let sel = sur.select_pool(&[vec![0.1]], 0, rule, None).unwrap();
    let dec = tr.evaluate_selection(0.5, 0.05, tr.radius(), &sel).unwrap();
    assert_eq!(dec.accepted, false);
}
