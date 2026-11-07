//! Tokamak TCV Policy (93 -> 256 -> 256 -> 256 -> 19) evaluator.
//! Implements the WPO Table 6 (`shape_70166` Precise) 5-component reward
//! (`SoftPlus` component scaling + `SmoothMax` combiner with alpha = -3.0).
//!
//! # Academic & Open-Source Attribution
//! - **WPO Paper**: David Pfau et al., *"Wasserstein Policy Optimization"*,
//!   ICML 2025, arXiv:2505.00663 (Appendix B.3, Table 1, Table 6, Equation 11);
//!   upstream JAX implementation in <https://github.com/google-deepmind/acme>
//!   (`acme/agents/jax/wpo/`, commit `2e09e3ddecf4a1ba783688c92aa493c596a2beec`).
//! - **TCV Tokamak Magnetic Control**: Brendan D. Tracey et al., *"Towards
//!   practical reinforcement learning for tokamak magnetic control"*, Fusion
//!   Engineering and Design 200 (2024) 114161, arXiv:2307.11546 (Section 3,
//!   Equations 1–3).
//! - **TORAX Core Transport Simulator**: Jonathan Citrin et al., *"TORAX: A Fast
//!   and Differentiable Tokamak Transport Simulator in JAX"*, arXiv:2406.06718,
//!   <https://github.com/google-deepmind/torax> (`v2.0.0`).

pub mod ppo;

pub const OBS_DIM: usize = 93;
pub const ACTION_DIM: usize = 19;
pub const HIDDEN_DIM: usize = 256;
pub const TOTAL_PARAMS: usize = 93 * 256 + 256 + 256 * 256 + 256 + 256 * 256 + 256 + 256 * 19 + 19; // 160,531

/// Layer `(fan_in, fan_out)` topology of the 4-layer TCV MLP policy.
pub const MLP_LAYERS: [(usize, usize); 4] = [
    (OBS_DIM, HIDDEN_DIM),
    (HIDDEN_DIM, HIDDEN_DIM),
    (HIDDEN_DIM, HIDDEN_DIM),
    (HIDDEN_DIM, ACTION_DIM),
];

/// Model-defined unconstrained parameter space with layer-wise Kaiming uniform
/// initialization and step scales for the 93 -> 256 -> 256 -> 256 -> 19 policy.
pub fn domain_frame() -> crate::optimizer::DomainFrame {
    crate::optimizer::DomainFrame::kaiming_mlp(&MLP_LAYERS)
        .expect("TCV tokamak MLP layer topology is valid")
}

const STEPS: usize = 100;
const ZETA: f64 = -2.944_438_979_166_440_3; // -ln(19), so Bad maps to 0.1

// WPO Table 6 (Precise) thresholds and SmoothMax combiner alpha
const LCFS_GOOD: f64 = 0.001;
const LCFS_BAD: f64 = 0.01;
const IP_GOOD: f64 = 100.0;
const IP_BAD: f64 = 2000.0;
const PF_GOOD: f64 = 100.0;
const PF_BAD: f64 = 50.0;
const VOLT_GOOD: f64 = 0.0;
const VOLT_BAD: f64 = 1.0;
const OH_GOOD: f64 = 50.0;
const OH_BAD: f64 = 1050.0;
const COMBINER_ALPHA: f64 = -3.0;

#[inline]
pub fn softplus_component(x: f64, good: f64, bad: f64) -> f64 {
    let scale = ZETA * (x - good) / (bad - good);
    let sigma = 1.0 / (1.0 + (-scale).exp());
    (2.0 * sigma).clamp(0.0, 1.0)
}

#[inline]
pub fn smooth_max(components: &[f64; 5], alpha: f64) -> f64 {
    let mut num = 0.0;
    let mut den = 0.0;
    for &c in components {
        let w = (alpha * c).exp();
        num += c * w;
        den += w;
    }
    num / den
}

#[inline]
fn init_state() -> [f64; OBS_DIM] {
    let mut state = [0.0f64; OBS_DIM];
    for i in 0..72 {
        state[i] = (((i + 1) as f64) * 0.17).sin() * 0.06;
    }
    for k in 0..16 {
        state[72 + k] = 0.12 + 0.02 * ((k as f64) * 0.5).cos();
    }
    state[88] = 0.25;
    state[89] = 0.10;
    state[90] = 0.0;
    state[91] = 0.35;
    state[92] = 0.35;
    state
}

#[inline]
fn elu_layer(input: &[f64], weights: &[f64], bias: &[f64], out: &mut [f64]) {
    let in_dim = input.len();
    let out_dim = out.len();
    for j in 0..out_dim {
        let mut sum = bias[j];
        for i in 0..in_dim {
            sum += input[i] * weights[i * out_dim + j];
        }
        out[j] = if sum >= 0.0 { sum } else { sum.exp() - 1.0 };
    }
}

#[inline]
fn step_reward(state: &[f64; OBS_DIM], action: &[f64; ACTION_DIM]) -> f64 {
    let mut shape_sq = 0.0;
    for i in 0..72 {
        shape_sq += state[i] * state[i];
    }
    let lcfs_dist = 0.05 * (shape_sq / 72.0).sqrt();
    let c_lcfs = softplus_component(lcfs_dist, LCFS_GOOD, LCFS_BAD);

    let ip_err = 1000.0 * state[91].abs();
    let c_ip = softplus_component(ip_err, IP_GOOD, IP_BAD);

    let mut pf_sum = 0.0;
    for k in 0..16 {
        pf_sum += 1000.0 * state[72 + k].abs();
    }
    let c_pf = softplus_component(pf_sum / 16.0, PF_GOOD, PF_BAD);

    let mut volt_level = 0.0;
    for j in 0..ACTION_DIM {
        let u = action[j].abs();
        volt_level += (u - 0.15).max(0.0) + 0.5 * u * u;
    }
    let c_volt = softplus_component(volt_level / (ACTION_DIM as f64), VOLT_GOOD, VOLT_BAD);

    let oh_diff = 1000.0 * (state[88] - state[89]).abs();
    let c_oh = softplus_component(oh_diff, OH_GOOD, OH_BAD);

    let components = [c_lcfs, c_ip, c_pf, c_volt, c_oh];
    smooth_max(&components, COMBINER_ALPHA)
}

#[inline]
fn step_dynamics(state: &mut [f64; OBS_DIM], action: &[f64; ACTION_DIM], step: usize) {
    let mut pf_mean = 0.0;
    for k in 0..16 {
        pf_mean += action[k];
    }
    pf_mean /= 16.0;

    for i in 0..72 {
        let mode = (((i + 1) as f64) * 0.17).sin();
        let mut harmonic = 0.0;
        for j in 0..ACTION_DIM {
            let phase = ((i * 7 + j * 13) as f64) * 0.11;
            harmonic += phase.cos() * action[j];
        }
        harmonic /= ACTION_DIM as f64;
        state[i] = 1.018 * state[i] + mode * (0.0045 - 0.045 * pf_mean) - 0.04 * harmonic;
    }
    for k in 0..16 {
        state[72 + k] = 0.965 * state[72 + k] + 0.001 + 0.045 * action[k];
    }
    state[88] = 1.008 * state[88] + 0.006 - 0.06 * action[16];
    state[89] = 0.985 * state[89] - 0.003 + 0.06 * action[17];
    state[90] = 0.92 * state[90] + 0.10 * action[18];

    let oh_drive = 0.5 * (action[16] + action[17]) + 0.4 * action[18];
    state[91] = 1.015 * state[91] + 0.015 - 0.16 * oh_drive;
    let t_next = (step + 2) as f64;
    state[92] = ((step + 1) as f64 * state[92] + state[91]) / t_next;
}

#[inline]
pub fn unpack_params<'a>(
    weights: &'a [f64],
) -> (
    &'a [f64],
    &'a [f64],
    &'a [f64],
    &'a [f64],
    &'a [f64],
    &'a [f64],
    &'a [f64],
    &'a [f64],
) {
    let w1 = &weights[..93 * 256];
    let mut offset = 93 * 256;
    let b1 = &weights[offset..offset + 256];
    offset += 256;
    let w2 = &weights[offset..offset + 256 * 256];
    offset += 256 * 256;
    let b2 = &weights[offset..offset + 256];
    offset += 256;
    let w3 = &weights[offset..offset + 256 * 256];
    offset += 256 * 256;
    let b3 = &weights[offset..offset + 256];
    offset += 256;
    let w4 = &weights[offset..offset + 256 * 19];
    offset += 256 * 19;
    let b4 = &weights[offset..offset + 19];
    (w1, b1, w2, b2, w3, b3, w4, b4)
}

/// Evaluates a 160,531-parameter policy over a 100-step TCV `shape_70166`
/// magnetic control trajectory using the WPO Table 6 reward specification.
/// Returns the cumulative episode return in `[0.0, 100.0]`.
pub fn evaluate_policy(weights: &[f64]) -> f64 {
    assert_eq!(
        weights.len(),
        TOTAL_PARAMS,
        "Policy requires exactly 160,531 parameters"
    );

    let (w1, b1, w2, b2, w3, b3, w4, b4) = unpack_params(weights);
    let mut state = init_state();
    let mut h1 = [0.0f64; HIDDEN_DIM];
    let mut h2 = [0.0f64; HIDDEN_DIM];
    let mut h3 = [0.0f64; HIDDEN_DIM];
    let mut action = [0.0f64; ACTION_DIM];
    let mut total_return = 0.0;

    for step in 0..STEPS {
        elu_layer(&state, w1, b1, &mut h1);
        elu_layer(&h1, w2, b2, &mut h2);
        elu_layer(&h2, w3, b3, &mut h3);
        for j in 0..ACTION_DIM {
            let mut sum = b4[j];
            for i in 0..HIDDEN_DIM {
                sum += h3[i] * w4[i * ACTION_DIM + j];
            }
            action[j] = sum.tanh();
        }
        total_return += step_reward(&state, &action);
        step_dynamics(&mut state, &action, step);
    }

    total_return
}

/// Evaluates a batch of candidate policies on CPU.
pub fn evaluate_batch(weights: &[f64], count: usize, out_rewards: &mut [f64]) {
    assert_eq!(weights.len(), count * TOTAL_PARAMS);
    assert_eq!(out_rewards.len(), count);
    for k in 0..count {
        let slice = &weights[k * TOTAL_PARAMS..(k + 1) * TOTAL_PARAMS];
        out_rewards[k] = evaluate_policy(slice);
    }
}

/// Rolls out a `steps`-step trajectory with pre-tanh noise `eps` (`steps *
/// 19`), returning `(episode_score, obs_flat, pre_tanh_u_flat, step_rewards)`.
pub fn rollout_traj(
    weights: &[f64],
    eps: &[f32],
    steps: usize,
) -> (f64, Vec<f32>, Vec<f32>, Vec<f32>) {
    assert_eq!(weights.len(), TOTAL_PARAMS);
    assert_eq!(eps.len(), steps * ACTION_DIM);
    let (w1, b1, w2, b2, w3, b3, w4, b4) = unpack_params(weights);
    let mut state = init_state();
    let mut h1 = [0.0f64; HIDDEN_DIM];
    let mut h2 = [0.0f64; HIDDEN_DIM];
    let mut h3 = [0.0f64; HIDDEN_DIM];
    let mut action = [0.0f64; ACTION_DIM];
    let mut obs_out = Vec::with_capacity(steps * OBS_DIM);
    let mut u_out = Vec::with_capacity(steps * ACTION_DIM);
    let mut r_out = Vec::with_capacity(steps);
    let mut sum_r = 0.0;
    for step in 0..steps {
        obs_out.extend(state.iter().map(|&v| v as f32));
        elu_layer(&state, w1, b1, &mut h1);
        elu_layer(&h1, w2, b2, &mut h2);
        elu_layer(&h2, w3, b3, &mut h3);
        for j in 0..ACTION_DIM {
            let mut u = b4[j];
            for i in 0..HIDDEN_DIM {
                u += h3[i] * w4[i * ACTION_DIM + j];
            }
            let u_noisy = u + f64::from(eps[step * ACTION_DIM + j]);
            u_out.push(u_noisy as f32);
            action[j] = u_noisy.tanh();
        }
        let r = step_reward(&state, &action);
        sum_r += r;
        r_out.push(r as f32);
        step_dynamics(&mut state, &action, step);
    }
    let score = sum_r * (100.0 / (steps as f64));
    (score, obs_out, u_out, r_out)
}

#[cfg(all(target_os = "macos", feature = "metal"))]
pub fn eval_metal(
    runtime: &crate::apple_gpu::Runtime,
    weights_batch: &[f32],
    count: usize,
) -> Result<Vec<f32>, String> {
    assert_eq!(weights_batch.len(), count * TOTAL_PARAMS);
    if count == 0 {
        return Ok(Vec::new());
    }

    let shader_source = include_str!("resident_metal/tokamak_policy.metal");
    let pipeline = runtime
        .precise(shader_source, "tokamak-eval", "tokamak_eval_policy")
        .map_err(|e| format!("tokamak shader compilation error: {:?}", e))?;

    let weights_buf = runtime.buffer_with(weights_batch);
    let rewards_buf = runtime.buffer::<f32>(count);
    let batch_buf = runtime.buffer_with(&[count as u32]);

    let command = runtime.queue.new_command_buffer();
    let encoder = command.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipeline);
    encoder.set_buffer(0, Some(&weights_buf), 0);
    encoder.set_buffer(1, Some(&rewards_buf), 0);
    encoder.set_buffer(2, Some(&batch_buf), 0);

    let threadgroup_width = count.min(32);
    encoder.dispatch_threads(
        metal::MTLSize::new(count as u64, 1, 1),
        metal::MTLSize::new(threadgroup_width as u64, 1, 1),
    );
    encoder.end_encoding();
    command.commit();
    command.wait_until_completed();

    let mut out = vec![0.0f32; count];
    unsafe {
        let ptr = rewards_buf.contents() as *const f32;
        std::ptr::copy_nonoverlapping(ptr, out.as_mut_ptr(), count);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shape() {
        assert_eq!(TOTAL_PARAMS, 160531);
        let zeros = vec![0.0f64; TOTAL_PARAMS];
        let score = evaluate_policy(&zeros);
        assert!(score.is_finite());
        assert!((0.0..=100.0).contains(&score));
    }

    #[test]
    fn batchcpu() {
        let count = 2;
        let weights = vec![0.0f64; count * TOTAL_PARAMS];
        let mut rewards = vec![0.0f64; count];
        evaluate_batch(&weights, count, &mut rewards);
        assert_eq!(rewards.len(), count);
        assert!(rewards[0].is_finite());
        assert_eq!(rewards[0], rewards[1]);
    }

    #[test]
    fn kaimingdomain() {
        let domain = domain_frame();
        assert_eq!(domain.num_dim(), TOTAL_PARAMS);
        assert!(!domain.clamped());
        let bounds = domain.bounds();
        let w0 = (3.0 / (OBS_DIM as f64)).sqrt();
        let b0 = 1.0 / (OBS_DIM as f64).sqrt();
        assert!((bounds[[0, 0]] + w0).abs() < 1e-12);
        assert!((bounds[[0, 1]] - w0).abs() < 1e-12);
        let bias0 = OBS_DIM * HIDDEN_DIM;
        assert!((bounds[[bias0, 0]] + b0).abs() < 1e-12);
        assert!((bounds[[bias0, 1]] - b0).abs() < 1e-12);
        let w1 = (3.0 / (HIDDEN_DIM as f64)).sqrt();
        let w1_idx = bias0 + HIDDEN_DIM;
        assert!((bounds[[w1_idx, 1]] - w1).abs() < 1e-12);

        let small = crate::optimizer::DomainFrame::kaiming_mlp(&[(4, 2)]).unwrap();
        let spec: crate::config::OptimizerSpec = crate::artifact::toml::from_str(
            r#"
algorithm = "operator-bo"
distribution = "gaussian"
[regional-search]
count = 2
sharing = "box-contain"
allocation = "incumbent-ucb1"
controller = "turbo"
"#,
        )
        .unwrap();
        let mut rng = crate::prng(7);
        let mut opt = crate::BayesOptimizer::from_domain(small, &spec, 0, &mut rng).unwrap();
        if let crate::optimizer::TrustRegionState::Multi { state, .. } = opt.trust_mut() {
            state.centers.row_mut(0).fill(1.5);
            let (lb, ub) = state.compute_bounds(0, None);
            assert!(
                ub[0] > 1.8,
                "unbounded region upper bound must not clamp to 1.0: {}",
                ub[0]
            );
            assert!(
                lb[0] > 1.0,
                "unbounded region lower bound must track center > 1.0: {}",
                lb[0]
            );
        } else {
            panic!("expected multi trust region");
        }
    }

    #[cfg(all(target_os = "macos", feature = "metal"))]
    #[test]
    fn metalkernel() {
        use crate::apple_gpu::Runtime;
        let runtime = Runtime::shared().expect("shared Apple GPU runtime");
        let zeros = vec![0.0f32; TOTAL_PARAMS];
        let results = eval_metal(&runtime, &zeros, 1).expect("Metal batch evaluation");
        let host_reward = evaluate_policy(&vec![0.0f64; TOTAL_PARAMS]) as f32;
        assert_eq!(results.len(), 1);
        assert!(results[0].is_finite());
        assert!((results[0] - host_reward).abs() < 1e-4);
    }

    #[cfg(all(target_os = "macos", feature = "metal"))]
    #[test]
    fn batchmetal() {
        use crate::apple_gpu::Runtime;
        let runtime = Runtime::shared().expect("shared Apple GPU runtime");
        let count = 4;
        let mut weights = vec![0.0f32; count * TOTAL_PARAMS];
        for k in 1..count {
            for i in 0..TOTAL_PARAMS {
                let u = crate::splitmix_f32((42 + k * TOTAL_PARAMS + i) as u64);
                weights[k * TOTAL_PARAMS + i] = (u - 0.5) * 0.16;
            }
        }

        let metal_rewards = eval_metal(&runtime, &weights, count).expect("Metal batch evaluation");
        assert_eq!(metal_rewards.len(), count);

        let mut cpu_weights = vec![0.0f64; count * TOTAL_PARAMS];
        for i in 0..weights.len() {
            cpu_weights[i] = weights[i] as f64;
        }
        let mut cpu_rewards = vec![0.0f64; count];
        evaluate_batch(&cpu_weights, count, &mut cpu_rewards);

        for k in 0..count {
            assert!(metal_rewards[k].is_finite());
            assert!((0.0..=100.0).contains(&cpu_rewards[k]));
            assert!(
                (metal_rewards[k] - cpu_rewards[k] as f32).abs() < 1e-3,
                "Candidate {} mismatch: metal {} vs cpu {}",
                k,
                metal_rewards[k],
                cpu_rewards[k]
            );
            for m in (k + 1)..count {
                assert!(
                    (cpu_rewards[k] - cpu_rewards[m]).abs() > 1e-4,
                    "Candidates {} and {} produced identical rewards: {}",
                    k,
                    m,
                    cpu_rewards[k]
                );
            }
        }
    }
}
