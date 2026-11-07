//! HuggingFace Candle (`candle-core` + `candle-nn`) Actor-Critic PPO baseline
//! on Apple Silicon Metal GPU (`Device::new_metal(0)`) for the
//! 160,531-parameter TCV tokamak (`93 -> 256 -> 256 -> 256 -> 19`) magnetic &
//! transport control task.
//!
//! # Academic & Open-Source Attribution
//! - **PPO Paper**: John Schulman et al., *"Proximal Policy Optimization
//!   Algorithms"*, arXiv:1707.06347 (2017), Equations 7–12 (clipped surrogate
//!   objective + GAE).
//! - **GAE Paper**: John Schulman et al., *"High-Dimensional Continuous Control
//!   Using Generalized Advantage Estimation"*, ICLR 2016, arXiv:1506.02438.
//! - **Candle Framework**: HuggingFace Candle (<https://github.com/huggingface/candle>,
//!   `candle-core` & `candle-nn` `v0.9.2` with Apple `metal` backend).

use super::{
    ACTION_DIM, MLP_LAYERS, OBS_DIM, TOKAGYM_ACT, TOKAGYM_LAYERS, TOKAGYM_OBS, TOKAGYM_PARAMS,
    TOTAL_PARAMS,
};
use candle_core::{DType, Device, Tensor, Var};
use candle_nn::{AdamW, Optimizer, ParamsAdamW};
use rand::Rng;
use rand_distr::StandardNormal;

const CRITIC_HIDDEN: usize = 128;
const GAMMA: f32 = 0.99;
const GAE_LAMBDA: f32 = 0.95;
const CLIP_EPS: f64 = 0.2;
const VF_COEF: f64 = 0.5;
const ENT_COEF: f64 = 0.005;
const PPO_EPOCHS: usize = 4;
const BATCH_EPISODES: usize = 8;
const INIT_LOGSTD: f32 = -2.302_585_1; // ln(0.10)

fn metal_device() -> Result<Device, String> {
    #[cfg(all(target_os = "macos", feature = "metal"))]
    {
        return Device::new_metal(0).map_err(|e| format!("Candle Metal device: {e}"));
    }
    #[cfg(not(all(target_os = "macos", feature = "metal")))]
    {
        Ok(Device::Cpu)
    }
}

/// Metal-compatible ELU activation: `relu(x) + (exp(-relu(-x)) - 1)`.
fn elu(x: &Tensor) -> candle_core::Result<Tensor> {
    let pos = x.relu()?;
    let neg_in = x.neg()?.relu()?.neg()?;
    let neg = neg_in.exp()?.affine(1.0, -1.0)?;
    pos.add(&neg)
}

fn linear(x: &Tensor, w: &Var, b: &Var) -> candle_core::Result<Tensor> {
    x.matmul(w.as_tensor())?.broadcast_add(b.as_tensor())
}

struct EpisodeStepBatch {
    obs: Vec<f32>,
    u: Vec<f32>,
    rewards: Vec<f32>,
    steps: usize,
}

/// On-policy Actor-Critic PPO agent backed by HuggingFace `candle` on Metal
/// GPU.
pub struct CandlePpo {
    device: Device,
    obs_dim: usize,
    action_dim: usize,
    total_params: usize,
    actor_w: [Var; 4],
    actor_b: [Var; 4],
    log_std: Var,
    critic_w: [Var; 3],
    critic_b: [Var; 3],
    opt: AdamW,
    buffer: Vec<EpisodeStepBatch>,
}

impl CandlePpo {
    /// Initializes Actor from `init_weights` (`160,531` TORAX or `136,712`
    /// TokaGym parameters) and Critic with Kaiming uniform initialization
    /// on the Metal GPU device.
    pub fn new(init_weights: &[f64], rng: &mut dyn rand::RngCore) -> Result<Self, String> {
        let (obs_dim, action_dim, layers) = match init_weights.len() {
            TOTAL_PARAMS => (OBS_DIM, ACTION_DIM, &MLP_LAYERS),
            TOKAGYM_PARAMS => (TOKAGYM_OBS, TOKAGYM_ACT, &TOKAGYM_LAYERS),
            n => return Err(format!("unsupported tokamak policy parameter count: {n}")),
        };
        let device = metal_device()?;
        let (actor_w, actor_b) = build_actor(init_weights, layers, &device)?;
        let log_std = Var::from_slice(&vec![INIT_LOGSTD; action_dim], (1, action_dim), &device)
            .map_err(|e| e.to_string())?;
        let (critic_w, critic_b) = build_critic(obs_dim, rng, &device)?;

        let mut vars = Vec::with_capacity(15);
        vars.extend(actor_w.iter().cloned());
        vars.extend(actor_b.iter().cloned());
        vars.push(log_std.clone());
        vars.extend(critic_w.iter().cloned());
        vars.extend(critic_b.iter().cloned());

        let params = ParamsAdamW {
            lr: 3e-4,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.0,
        };
        let opt = AdamW::new(vars, params).map_err(|e| e.to_string())?;
        Ok(Self {
            device,
            obs_dim,
            action_dim,
            total_params: init_weights.len(),
            actor_w,
            actor_b,
            log_std,
            critic_w,
            critic_b,
            opt,
            buffer: Vec::with_capacity(BATCH_EPISODES),
        })
    }

    /// Returns whether the agent is executing on an Apple Metal GPU device.
    pub fn is_metal(&self) -> bool {
        self.device.is_metal()
    }

    /// Exports the current Actor parameters from Metal GPU memory to host
    /// `f64`.
    pub fn actor_weights(&self) -> Result<Vec<f64>, String> {
        let mut out = Vec::with_capacity(self.total_params);
        for layer in 0..4 {
            let w_vec = self.actor_w[layer]
                .as_tensor()
                .flatten_all()
                .and_then(|t| t.to_vec1::<f32>())
                .map_err(|e| e.to_string())?;
            out.extend(w_vec.into_iter().map(f64::from));
            let b_vec = self.actor_b[layer]
                .as_tensor()
                .flatten_all()
                .and_then(|t| t.to_vec1::<f32>())
                .map_err(|e| e.to_string())?;
            out.extend(b_vec.into_iter().map(f64::from));
        }
        Ok(out)
    }

    /// Samples pre-tanh Gaussian exploration noise `eps` of length `steps *
    /// action_dim` using the current learnable `log_std` tensor.
    pub fn sample_noise(
        &self,
        steps: usize,
        deterministic: bool,
        rng: &mut dyn rand::RngCore,
    ) -> Result<Vec<f32>, String> {
        if deterministic {
            return Ok(vec![0.0f32; steps * self.action_dim]);
        }
        let log_std_vec = self
            .log_std
            .as_tensor()
            .flatten_all()
            .and_then(|t| t.to_vec1::<f32>())
            .map_err(|e| e.to_string())?;
        let stds: Vec<f32> = log_std_vec
            .into_iter()
            .map(|ls| ls.clamp(-4.0, -0.5).exp())
            .collect();
        let mut eps = Vec::with_capacity(steps * self.action_dim);
        for _ in 0..steps {
            for &s in &stds {
                let z: f32 = rng.sample(StandardNormal);
                eps.push(z * s);
            }
        }
        Ok(eps)
    }

    /// Records a completed episode trajectory and runs a multi-epoch PPO update
    /// on the Metal GPU once `BATCH_EPISODES` episodes have been collected.
    pub fn record_step(
        &mut self,
        obs: Vec<f32>,
        u: Vec<f32>,
        rewards: Vec<f32>,
    ) -> Result<bool, String> {
        let steps = rewards.len();
        assert_eq!(obs.len(), steps * self.obs_dim);
        assert_eq!(u.len(), steps * self.action_dim);
        self.buffer.push(EpisodeStepBatch {
            obs,
            u,
            rewards,
            steps,
        });
        if self.buffer.len() < BATCH_EPISODES {
            return Ok(false);
        }
        self.ppo_update()?;
        self.buffer.clear();
        Ok(true)
    }

    fn actor_forward(&self, obs: &Tensor) -> candle_core::Result<Tensor> {
        let mut h = obs.clone();
        for idx in 0..3 {
            h = elu(&linear(&h, &self.actor_w[idx], &self.actor_b[idx])?)?;
        }
        linear(&h, &self.actor_w[3], &self.actor_b[3])
    }

    fn critic_forward(&self, obs: &Tensor) -> candle_core::Result<Tensor> {
        let h_first = elu(&linear(obs, &self.critic_w[0], &self.critic_b[0])?)?;
        let h_sec = elu(&linear(&h_first, &self.critic_w[1], &self.critic_b[1])?)?;
        linear(&h_sec, &self.critic_w[2], &self.critic_b[2])?.squeeze(1)
    }

    fn gauss_logprob(&self, u: &Tensor, mean: &Tensor) -> candle_core::Result<(Tensor, Tensor)> {
        let log_std = self.log_std.as_tensor().clamp(-4.0f32, -0.5f32)?;
        let std = log_std.exp()?;
        let diff = u.sub(mean)?.broadcast_div(&std)?;
        let sq = diff.sqr()?;
        let term = sq.broadcast_add(&log_std.affine(2.0, 0.0)?)?;
        let log_prob = term.sum(1)?.affine(-0.5, 0.0)?;
        let entropy = log_std.sum_all()?;
        Ok((log_prob, entropy))
    }

    fn ppo_epoch(
        &mut self,
        obs_t: &Tensor,
        u_t: &Tensor,
        lp_old: &Tensor,
        adv_t: &Tensor,
        ret_t: &Tensor,
    ) -> Result<(), String> {
        let mean = self.actor_forward(obs_t).map_err(|e| e.to_string())?;
        let (lp_new, entropy) = self.gauss_logprob(u_t, &mean).map_err(|e| e.to_string())?;
        let ratio = lp_new
            .sub(lp_old)
            .and_then(|d| d.clamp(-10.0f32, 10.0f32))
            .and_then(|c| c.exp())
            .map_err(|e| e.to_string())?;
        let surr_unclipped = ratio.mul(adv_t).map_err(|e| e.to_string())?;
        let surr_clipped = ratio
            .clamp((1.0 - CLIP_EPS) as f32, (1.0 + CLIP_EPS) as f32)
            .and_then(|r| r.mul(adv_t))
            .map_err(|e| e.to_string())?;
        let policy_loss = surr_unclipped
            .minimum(&surr_clipped)
            .and_then(|m| m.mean_all())
            .and_then(|m| m.neg())
            .map_err(|e| e.to_string())?;
        let v_loss = self
            .critic_forward(obs_t)
            .and_then(|v| v.sub(ret_t))
            .and_then(|d| d.sqr())
            .and_then(|s| s.mean_all())
            .and_then(|m| m.affine(VF_COEF, 0.0))
            .map_err(|e| e.to_string())?;
        let ent_loss = entropy.affine(-ENT_COEF, 0.0).map_err(|e| e.to_string())?;
        let total_loss = policy_loss
            .add(&v_loss)
            .and_then(|l| l.add(&ent_loss))
            .map_err(|e| e.to_string())?;
        self.opt
            .backward_step(&total_loss)
            .map_err(|e| format!("Candle PPO backward_step: {e}"))
    }

    fn ppo_update(&mut self) -> Result<(), String> {
        let total_steps: usize = self.buffer.iter().map(|ep| ep.steps).sum();
        let mut all_obs = Vec::with_capacity(total_steps * self.obs_dim);
        let mut all_u = Vec::with_capacity(total_steps * self.action_dim);
        for ep in &self.buffer {
            all_obs.extend_from_slice(&ep.obs);
            all_u.extend_from_slice(&ep.u);
        }
        let obs_t = Tensor::from_vec(all_obs, (total_steps, self.obs_dim), &self.device)
            .map_err(|e| e.to_string())?;
        let u_t = Tensor::from_vec(all_u, (total_steps, self.action_dim), &self.device)
            .map_err(|e| e.to_string())?;
        let values_old = self
            .critic_forward(&obs_t)
            .and_then(|v| v.to_vec1::<f32>())
            .map_err(|e| e.to_string())?;
        let mean_old = self.actor_forward(&obs_t).map_err(|e| e.to_string())?;
        let (lp_old_t, _) = self
            .gauss_logprob(&u_t, &mean_old)
            .map_err(|e| e.to_string())?;
        let lp_old = lp_old_t.detach();
        let (adv_vec, ret_vec) = compute_gae(&self.buffer, &values_old);
        let adv_t =
            Tensor::from_vec(adv_vec, total_steps, &self.device).map_err(|e| e.to_string())?;
        let ret_t =
            Tensor::from_vec(ret_vec, total_steps, &self.device).map_err(|e| e.to_string())?;
        for _ in 0..PPO_EPOCHS {
            self.ppo_epoch(&obs_t, &u_t, &lp_old, &adv_t, &ret_t)?;
        }
        Ok(())
    }
}

fn build_actor(
    init_weights: &[f64],
    layers: &[(usize, usize); 4],
    device: &Device,
) -> Result<([Var; 4], [Var; 4]), String> {
    let mut offset = 0usize;
    let mut ws = Vec::with_capacity(4);
    let mut bs = Vec::with_capacity(4);
    for &(fan_in, fan_out) in layers {
        let w_len = fan_in * fan_out;
        let w_f32: Vec<f32> = init_weights[offset..offset + w_len]
            .iter()
            .map(|&v| v as f32)
            .collect();
        offset += w_len;
        let b_f32: Vec<f32> = init_weights[offset..offset + fan_out]
            .iter()
            .map(|&v| v as f32)
            .collect();
        offset += fan_out;
        let w_var =
            Var::from_slice(&w_f32, (fan_in, fan_out), device).map_err(|e| e.to_string())?;
        let b_var = Var::from_slice(&b_f32, fan_out, device).map_err(|e| e.to_string())?;
        ws.push(w_var);
        bs.push(b_var);
    }
    Ok((
        ws.try_into().map_err(|_| "actor_w len")?,
        bs.try_into().map_err(|_| "actor_b len")?,
    ))
}

fn kaiming_var(
    fan_in: usize,
    fan_out: usize,
    rng: &mut dyn rand::RngCore,
    device: &Device,
) -> Result<(Var, Var), String> {
    let w_bound = (3.0 / (fan_in as f32)).sqrt();
    let b_bound = 1.0 / (fan_in as f32).sqrt();
    let w_vec: Vec<f32> = (0..fan_in * fan_out)
        .map(|_| rng.gen_range(-w_bound..w_bound))
        .collect();
    let b_vec: Vec<f32> = (0..fan_out)
        .map(|_| rng.gen_range(-b_bound..b_bound))
        .collect();
    let w = Var::from_slice(&w_vec, (fan_in, fan_out), device)
        .and_then(|v| {
            if v.dtype() == DType::F32 {
                Ok(v)
            } else {
                Err(candle_core::Error::Msg("expected F32".into()))
            }
        })
        .map_err(|e| e.to_string())?;
    let b = Var::from_slice(&b_vec, fan_out, device).map_err(|e| e.to_string())?;
    Ok((w, b))
}

fn build_critic(
    obs_dim: usize,
    rng: &mut dyn rand::RngCore,
    device: &Device,
) -> Result<([Var; 3], [Var; 3]), String> {
    let (w_in, b_in) = kaiming_var(obs_dim, CRITIC_HIDDEN, rng, device)?;
    let (w_mid, b_mid) = kaiming_var(CRITIC_HIDDEN, CRITIC_HIDDEN, rng, device)?;
    let (w_out, b_out) = kaiming_var(CRITIC_HIDDEN, 1, rng, device)?;
    Ok(([w_in, w_mid, w_out], [b_in, b_mid, b_out]))
}

fn compute_gae(buffer: &[EpisodeStepBatch], values: &[f32]) -> (Vec<f32>, Vec<f32>) {
    let total = values.len();
    let mut advantages = vec![0.0f32; total];
    let mut returns = vec![0.0f32; total];
    let mut cursor = 0usize;
    for ep in buffer {
        let len = ep.steps;
        let mut gae = 0.0f32;
        for step in (0..len).rev() {
            let idx = cursor + step;
            let next_val = if step + 1 < len { values[idx + 1] } else { 0.0 };
            let delta = ep.rewards[step] + GAMMA * next_val - values[idx];
            gae = delta + GAMMA * GAE_LAMBDA * gae;
            advantages[idx] = gae;
            returns[idx] = gae + values[idx];
        }
        cursor += len;
    }
    let mean = advantages.iter().sum::<f32>() / (total.max(1) as f32);
    let var = advantages
        .iter()
        .map(|&a| (a - mean) * (a - mean))
        .sum::<f32>()
        / (total.max(1) as f32);
    let std = (var + 1e-8).sqrt();
    for adv in &mut advantages {
        *adv = (*adv - mean) / std;
    }
    (advantages, returns)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tokamak::rollout_traj;

    #[test]
    fn candle_metal() {
        let mut rng = crate::prng(42);
        let init_weights = vec![0.0f64; TOTAL_PARAMS];
        let mut agent = CandlePpo::new(&init_weights, &mut rng).expect("init CandlePpo");
        #[cfg(all(target_os = "macos", feature = "metal"))]
        assert!(agent.is_metal(), "CandlePpo must run on Apple Metal GPU");

        let steps = 20;
        let mut updated = false;
        for round in 0..BATCH_EPISODES {
            let weights = agent.actor_weights().expect("export actor weights");
            let eps = agent
                .sample_noise(steps, round == 0, &mut rng)
                .expect("sample noise");
            let (score, obs, u, rewards) = rollout_traj(&weights, &eps, steps);
            assert!(score.is_finite());
            if agent.record_step(obs, u, rewards).expect("ppo step") {
                updated = true;
            }
        }
        assert!(updated, "PPO update must trigger after BATCH_EPISODES");
        let after_weights = agent.actor_weights().expect("export updated weights");
        let diff: f64 = after_weights.iter().map(|w| w.abs()).sum();
        assert!(
            diff > 0.0,
            "Actor weights must update after Candle PPO step"
        );
    }
}
