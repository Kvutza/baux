//! Evaluation oracle orchestrating rollouts across active rounds and heldout
//! checks.

use crate::bench_chain::ChainPolicy;
use crate::bench_schedule::RolloutSchedule;
use crate::bench_task::{ControlTask, MjDataState};
use baux::objective_observation::{MAX_OBJECTIVES, ObjectiveEstimate, ObjectiveObservation};

/// Diagnostic rollout telemetry captured across evaluation episodes.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RolloutEvidence {
    pub round: u32,
    pub episodes: u32,
    pub mean_steps: f32,
    pub x_displacement: f32,
    pub mean_x_velocity: f32,
    pub action_saturation_frac: f32,
    pub nonfinite_steps: u32,
}

/// Evaluation summary on the heldout set of random seeds.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HeldoutSummary {
    pub round: u32,
    pub episodes: u32,
    pub mean_return: f32,
    pub se_return: f32,
    pub mean_forward_reward: f32,
    pub mean_ctrl_cost: f32,
    pub survival_fraction: f32,
    pub active_minus_heldout_gap: f32,
    pub used_for_acceptance: bool,
}

struct EpisodeMetrics {
    total_reward: f32,
    fwd_reward: f32,
    ctrl_cost: f32,
    steps: usize,
    x_disp: f32,
    sat_count: usize,
    action_count: usize,
    nonfinite: u32,
    survived: bool,
}

/// Closed-loop oracle coordinating benchmark rollouts for candidate policies.
pub struct ControlOracle<T: ControlTask> {
    task: T,
    schedule: RolloutSchedule,
    current_round: u32,
    data: MjDataState,
    obs_buffer: Vec<f32>,
    act_buffer: Vec<f32>,
    incumbent_active_mean_return: Option<f32>,
    cached_heldout: Option<HeldoutSummary>,
}

impl<T: ControlTask> ControlOracle<T> {
    /// Creates a new control evaluation oracle.
    pub fn new(task: T, schedule: RolloutSchedule) -> Self {
        let obs_dim = task.obs_dim();
        let nu = task.model().nu;
        let data = MjDataState::new(task.model());
        Self {
            task,
            schedule,
            current_round: 0,
            data,
            obs_buffer: vec![0.0; obs_dim],
            act_buffer: vec![0.0; nu],
            incumbent_active_mean_return: None,
            cached_heldout: None,
        }
    }

    /// Access the underlying benchmark task.
    pub fn task(&self) -> &T {
        &self.task
    }

    /// Updates the tracked active mean return of the current incumbent policy.
    pub fn set_incumbent(&mut self, mean_return: f32) {
        self.incumbent_active_mean_return = Some(mean_return);
        self.cached_heldout = None;
    }

    /// Returns cached heldout summary if available, otherwise evaluates and
    /// caches it.
    pub fn heldout_cached(&mut self, policy: &mut ChainPolicy) -> Result<HeldoutSummary, String> {
        if let Some(mut h) = self.cached_heldout {
            h.round = self.current_round;
            let inc = self.incumbent_active_mean_return.unwrap_or(h.mean_return);
            h.active_minus_heldout_gap = inc - h.mean_return;
            return Ok(h);
        }
        let h = self.evaluate_heldout(policy)?;
        self.cached_heldout = Some(h);
        Ok(h)
    }

    /// Returns the tracked active mean return of the current incumbent policy,
    /// if set.
    pub fn incumbent_return(&self) -> Option<f32> {
        self.incumbent_active_mean_return
    }

    /// Evaluates a candidate policy on the current active round's episodes.
    pub fn evaluate_candidate(
        &mut self,
        policy: &mut ChainPolicy,
    ) -> Result<(ObjectiveObservation, RolloutEvidence), String> {
        let num_episodes = self.schedule.active.episodes();
        if num_episodes == 0 {
            return Err("Number of active episodes must be positive".into());
        }
        let round = self.current_round;
        let mut returns = Vec::with_capacity(num_episodes as usize);
        let mut total_steps = 0;
        let mut total_x_disp = 0.0;
        let mut total_sat = 0;
        let mut total_act = 0;
        let mut total_nonfinite = 0;

        for ep in 0..num_episodes {
            let seed = self.schedule.active_seed(round, ep);
            let m = self.rollout_episode(policy, seed);
            returns.push(m.total_reward);
            total_steps += m.steps;
            total_x_disp += m.x_disp;
            total_sat += m.sat_count;
            total_act += m.action_count;
            total_nonfinite += m.nonfinite;
        }

        let mean_ret = returns.iter().sum::<f32>() / (num_episodes as f32);
        let var_ret = if num_episodes > 1 {
            let s: f32 = returns
                .iter()
                .map(|&r| (r - mean_ret) * (r - mean_ret))
                .sum();
            (s / ((num_episodes * (num_episodes - 1)) as f32)).max(0.0)
        } else {
            0.0
        };

        if self.incumbent_active_mean_return.is_none() {
            self.incumbent_active_mean_return = Some(mean_ret);
        }
        let control_est = ObjectiveEstimate {
            mean: mean_ret,
            variance: var_ret,
        };
        let mut estimates = [ObjectiveEstimate::default(); MAX_OBJECTIVES];
        estimates[0] = control_est;
        let observation = ObjectiveObservation::new(&estimates[..1], control_est)?;

        let mean_steps = (total_steps as f32) / (num_episodes as f32);
        let x_displacement = total_x_disp / (num_episodes as f32);
        let dt = self.task.model().dt * (self.task.model().frame_skip as f32);
        let time = mean_steps * dt;
        let mean_x_velocity = if time > 1e-6 {
            x_displacement / time
        } else {
            0.0
        };
        let action_saturation_frac = if total_act > 0 {
            (total_sat as f32) / (total_act as f32)
        } else {
            0.0
        };

        let evidence = RolloutEvidence {
            round,
            episodes: num_episodes,
            mean_steps,
            x_displacement,
            mean_x_velocity,
            action_saturation_frac,
            nonfinite_steps: total_nonfinite,
        };

        Ok((observation, evidence))
    }

    /// Advances the oracle to the next round.
    pub fn advance_round(&mut self) {
        self.current_round += 1;
    }

    /// Evaluates the policy on the heldout set of random seeds without mutating
    /// running statistics.
    pub fn evaluate_heldout(&mut self, policy: &mut ChainPolicy) -> Result<HeldoutSummary, String> {
        let saved_norm = policy.rsnorm.clone();
        let num_episodes = self.schedule.heldout_episodes;
        if num_episodes == 0 {
            return Err("Number of heldout episodes must be positive".into());
        }
        let round = self.current_round;
        let mut returns = Vec::with_capacity(num_episodes as usize);
        let mut sum_fwd = 0.0;
        let mut sum_ctrl = 0.0;
        let mut survived_count = 0;

        for ep in 0..num_episodes {
            let seed = self.schedule.heldout_seed(ep);
            let m = self.rollout_episode(policy, seed);
            returns.push(m.total_reward);
            sum_fwd += m.fwd_reward;
            sum_ctrl += m.ctrl_cost;
            if m.survived {
                survived_count += 1;
            }
        }

        let mean_return = returns.iter().sum::<f32>() / (num_episodes as f32);
        let se_return = if num_episodes > 1 {
            let s: f32 = returns
                .iter()
                .map(|&r| (r - mean_return) * (r - mean_return))
                .sum();
            (s / ((num_episodes * (num_episodes - 1)) as f32)).sqrt()
        } else {
            0.0
        };

        let mean_forward_reward = sum_fwd / (num_episodes as f32);
        let mean_ctrl_cost = sum_ctrl / (num_episodes as f32);
        let survival_fraction = (survived_count as f32) / (num_episodes as f32);
        let incumbent_active = self.incumbent_active_mean_return.unwrap_or(mean_return);
        let active_minus_heldout_gap = incumbent_active - mean_return;
        policy.rsnorm = saved_norm;

        Ok(HeldoutSummary {
            round,
            episodes: num_episodes,
            mean_return,
            se_return,
            mean_forward_reward,
            mean_ctrl_cost,
            survival_fraction,
            active_minus_heldout_gap,
            used_for_acceptance: false,
        })
    }

    fn rollout_episode(&mut self, policy: &mut ChainPolicy, seed: u64) -> EpisodeMetrics {
        self.task.reset(&mut self.data, seed);
        let x_start = self.data.qpos[0];
        let mut total_reward = 0.0;
        let mut fwd_reward = 0.0;
        let mut ctrl_cost = 0.0;
        let mut sat_count = 0;
        let mut action_count = 0;
        let mut nonfinite = 0;
        let mut terminated = false;

        while self.data.steps < self.task.model().max_episode_steps {
            self.task.extract_obs(&self.data, &mut self.obs_buffer);
            if self.obs_buffer.iter().any(|&x| !x.is_finite()) {
                nonfinite += 1;
            }
            policy.observe_norm(&self.obs_buffer);
            policy.forward(&self.obs_buffer, &mut self.act_buffer);
            for &a in &self.act_buffer {
                action_count += 1;
                if a.abs() >= 0.99 {
                    sat_count += 1;
                }
            }
            let outcome = self.task.step(&mut self.data, &self.act_buffer);
            total_reward += outcome.reward_total;
            fwd_reward += outcome.reward_forward;
            ctrl_cost += outcome.reward_ctrl;
            if outcome.terminated {
                terminated = true;
                break;
            }
            if outcome.truncated {
                break;
            }
        }

        let x_end = self.data.qpos[0];
        let x_disp = x_end - x_start;
        EpisodeMetrics {
            total_reward,
            fwd_reward,
            ctrl_cost,
            steps: self.data.steps,
            x_disp,
            sat_count,
            action_count,
            nonfinite,
            survived: !terminated,
        }
    }
}
