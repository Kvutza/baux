//! Observation-driven trajectory seed-mixing and Online EM family scale
//! updates.
//!
//! Implements the five machine-verified generator mechanisms from
//! `BauxResearch.Optimization`:
//! - `VariableMetric`: full-rank seed-reconstructed covariance stretching with
//!   exact unit-variance preservation (`alpha_fresh^2 + sum_j alpha_j^2 = 1`).
//! - `SelfInteracting`: shift-invariant Boltzmann weighting over accepted
//!   tangent velocities (`gain > 0`).
//! - `KernelBandit`: antipodal deflection away from rejected local probes
//!   (`gain < 0`) evaluated at the current incumbent `base_id`.
//! - `TensorEm`: Cappé–Moulines Online EM sufficient-statistic recursion on the
//!   4-family simplex with closed-form M-step scale updates.
use super::*;

pub(super) const TRAJECTORY_SEEDS: usize = 4;

#[derive(Clone, Copy, Debug)]
pub(super) struct TrajectoryStep {
    pub(super) seed: u64,
    pub(super) base_id: i64,
    pub(super) gain: f32,
    pub(super) accepted: bool,
    pub(super) family_share: [f32; FAMILIES],
}

#[derive(Clone, Debug)]
pub(super) struct TrajectoryGenerator {
    pub(super) steps: Vec<TrajectoryStep>,
    pub(super) online_em_stats: [f32; FAMILIES],
    pub(super) tt_core: [[f32; FAMILIES]; FAMILIES],
    pub(super) local_streak: usize,
}

impl Default for TrajectoryGenerator {
    fn default() -> Self {
        Self::new()
    }
}

impl TrajectoryGenerator {
    pub(super) fn new() -> Self {
        let uniform = 1.0 / FAMILIES as f32;
        Self {
            steps: Vec::with_capacity(MAX_HISTORY),
            online_em_stats: [uniform; FAMILIES],
            tt_core: [[uniform; FAMILIES]; FAMILIES],
            local_streak: 0,
        }
    }

    pub(super) fn reset(&mut self) {
        let uniform = 1.0 / FAMILIES as f32;
        self.steps.clear();
        self.online_em_stats = [uniform; FAMILIES];
        self.tt_core = [[uniform; FAMILIES]; FAMILIES];
        self.local_streak = 0;
    }

    /// Deterministic seed-specific latent excitation on the 4-family simplex,
    /// pushed forward through the row-stochastic Tensor-Train core
    /// (`BauxResearch.Optimization.TensorEm.pushForward`).
    fn chain_share(&self, seed: u64) -> ([f32; FAMILIES], [f32; FAMILIES]) {
        let mut raw = [0.0_f32; FAMILIES];
        let mut sum = 0.0_f32;
        for (i, slot) in raw.iter_mut().enumerate() {
            let h = crate::hash::splitmix64(seed ^ (i as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15));
            let unit = (h >> 40) as f32 * (1.0 / 16_777_216.0);
            let w = (0.45 * (unit - 0.5)).exp();
            *slot = w;
            sum += w;
        }
        let u = raw.map(|w| w / sum.max(1e-12));
        let mut pushed = [0.0_f32; FAMILIES];
        let mut pushed_sum = 0.0_f32;
        for g in 0..FAMILIES {
            let mut acc = 0.0_f32;
            for i in 0..FAMILIES {
                acc += u[i] * self.tt_core[i][g];
            }
            pushed[g] = acc.max(1e-6);
            pushed_sum += pushed[g];
        }
        let pi = pushed.map(|p| p / pushed_sum.max(1e-12));
        (u, pi)
    }

    /// Record one completed trial evaluation `(seed, base_id, gain, accepted)`
    /// along with its per-family canonical displacement energy, updating both
    /// the Online EM Tensor-Train core + marginal sufficient statistics and the
    /// bounded trajectory archive.
    pub(super) fn observe(
        &mut self,
        seed: u64,
        base_id: i64,
        gain: f32,
        accepted: bool,
        family_energy: [f32; FAMILIES],
    ) {
        if !gain.is_finite() {
            return;
        }
        let uniform = 1.0 / FAMILIES as f32;
        let (u_seed, pi_seed) = self.chain_share(seed);
        let total_energy: f32 = family_energy
            .iter()
            .copied()
            .filter(|e| e.is_finite() && *e > 0.0)
            .sum();
        let raw_share = if total_energy > 1e-20 {
            family_energy.map(|e| {
                if e.is_finite() && e > 0.0 {
                    e / total_energy
                } else {
                    0.0
                }
            })
        } else {
            [uniform; FAMILIES]
        };
        let family_share = if raw_share.iter().all(|&s| (s - uniform).abs() < 1e-3) {
            pi_seed
        } else {
            let mut blended = [0.0_f32; FAMILIES];
            let mut b_sum = 0.0_f32;
            for g in 0..FAMILIES {
                blended[g] = (0.70 * raw_share[g] + 0.30 * pi_seed[g]).max(1e-6);
                b_sum += blended[g];
            }
            blended.map(|b| b / b_sum.max(1e-12))
        };

        // Cappé & Moulines Online EM + E²M Tensor-Train conditional-count
        // update (BauxResearch.Optimization.TensorEm):
        // S_{t+1} = (1 - gamma) S_t + gamma * s_hat_{t+1} on the 4-family
        // simplex and the 4x4 row-stochastic Tensor-Train core.
        let s_hat = if gain > 0.0 {
            std::array::from_fn(|g| 0.4 * uniform + 0.6 * family_share[g])
        } else if gain < 0.0 {
            std::array::from_fn(|g| ((1.0 - family_share[g]).max(0.0)) / (FAMILIES as f32 - 1.0))
        } else {
            [uniform; FAMILIES]
        };
        let gamma = if accepted {
            0.20_f32
        } else if self.local_streak >= 10 {
            0.0_f32
        } else if gain < 0.0 {
            0.03_f32 * (200.0 * gain).clamp(-20.0, 0.0).exp()
        } else {
            0.02_f32
        };
        for i in 0..FAMILIES {
            let row_weight = (0.5 * uniform + 0.5 * u_seed[i]) * FAMILIES as f32;
            let step_gamma = (gamma * row_weight).clamp(0.0, 0.35);
            let mut row_sum = 0.0_f32;
            for g in 0..FAMILIES {
                let target = (self.tt_core[i][g] * s_hat[g] / uniform).clamp(1e-4, 4.0);
                self.tt_core[i][g] =
                    ((1.0 - step_gamma) * self.tt_core[i][g] + step_gamma * target).max(1e-4);
                row_sum += self.tt_core[i][g];
            }
            if row_sum > 0.0 {
                for g in 0..FAMILIES {
                    self.tt_core[i][g] /= row_sum;
                }
            }
        }
        let mut stat_sum = 0.0_f32;
        for g in 0..FAMILIES {
            let pushed_g: f32 = (0..FAMILIES).map(|i| uniform * self.tt_core[i][g]).sum();
            let target_g = 0.5 * s_hat[g] + 0.5 * pushed_g;
            self.online_em_stats[g] =
                ((1.0 - gamma) * self.online_em_stats[g] + gamma * target_g).max(1e-4);
            stat_sum += self.online_em_stats[g];
        }
        if stat_sum > 0.0 {
            for g in 0..FAMILIES {
                self.online_em_stats[g] /= stat_sum;
            }
        }

        self.push_step(TrajectoryStep {
            seed,
            base_id,
            gain,
            accepted,
            family_share,
        });
    }

    fn least_informative(&self, base_id: i64) -> Option<usize> {
        let local: Vec<(usize, f32)> = self
            .steps
            .iter()
            .enumerate()
            .filter(|(_, s)| !s.accepted && s.base_id == base_id)
            .map(|(i, s)| (i, s.gain))
            .collect();
        if local.is_empty() {
            return None;
        }
        let mean = local.iter().map(|(_, g)| *g).sum::<f32>() / local.len() as f32;
        local
            .into_iter()
            .min_by(|a, b| (a.1 - mean).abs().total_cmp(&(b.1 - mean).abs()))
            .map(|(i, _)| i)
    }

    fn push_step(&mut self, step: TrajectoryStep) {
        if step.accepted {
            self.local_streak = 0;
        } else {
            let same_base = self
                .steps
                .last()
                .is_none_or(|prev| prev.accepted || prev.base_id == step.base_id);
            self.local_streak = if same_base { self.local_streak + 1 } else { 1 };
        }
        if self.steps.len() == MAX_HISTORY {
            let stale = self
                .steps
                .iter()
                .position(|s| !s.accepted && s.base_id != step.base_id);
            let rej_count = self.steps.iter().filter(|s| !s.accepted).count();
            let first_rej = if rej_count > 8 {
                if self.local_streak > 10 {
                    self.least_informative(step.base_id)
                } else {
                    self.steps.iter().position(|s| !s.accepted)
                }
            } else {
                None
            };
            let evict_idx = stale.or(first_rej).unwrap_or(0);
            self.steps.remove(evict_idx);
        }
        self.steps.push(step);
    }

    /// Closed-form Online EM M-step update for the 4-family metric weights
    /// (`FamilyHistory::weights`, where proposal scale is `1 / sqrt(weight)`).
    pub(super) fn em_weights(&self, current: [f32; FAMILIES]) -> [f32; FAMILIES] {
        if self.local_streak > 10 {
            return current;
        }
        let uniform = 1.0 / FAMILIES as f32;
        let mut logs = [0.0_f32; FAMILIES];
        for g in 0..FAMILIES {
            let resp = self.online_em_stats[g].max(1e-3);
            // Higher Online EM responsibility -> larger proposal scale ->
            // smaller metric weight.
            let target_log = (uniform / resp).ln().clamp(-4.0f32.ln(), 4.0f32.ln());
            let current_log = current[g].max(1e-3).ln();
            logs[g] = 0.80 * current_log + 0.20 * target_log;
        }
        let mean_log = logs.iter().sum::<f32>() / FAMILIES as f32;
        for g in 0..FAMILIES {
            logs[g] = (logs[g] - mean_log).clamp(-4.0f32.ln(), 4.0f32.ln());
        }
        logs.map(f32::exp)
    }

    fn select_centered(&self, current_base_id: i64) -> Vec<(TrajectoryStep, f32, f32)> {
        if self.local_streak % 2 == 0 {
            return Vec::new();
        }
        let mut selected: Vec<(TrajectoryStep, f32, f32)> = Vec::with_capacity(TRAJECTORY_SEEDS);
        if let Some(best_acc) = self
            .steps
            .iter()
            .rev()
            .filter(|s| s.accepted && s.gain >= 0.0)
            .take(4)
            .max_by(|a, b| a.gain.total_cmp(&b.gain))
        {
            selected.push((*best_acc, 0.75, 1.0));
        }
        let local_rej: Vec<(usize, TrajectoryStep)> = self
            .steps
            .iter()
            .rev()
            .filter(|s| !s.accepted && s.base_id == current_base_id && s.gain < 0.0)
            .copied()
            .enumerate()
            .collect();
        if local_rej.is_empty() {
            return selected;
        }
        let mean_gain = local_rej.iter().map(|(_, s)| s.gain).sum::<f32>() / local_rej.len() as f32;
        let max_dev = local_rej
            .iter()
            .map(|(_, s)| (s.gain - mean_gain).abs())
            .fold(1e-6_f32, f32::max);
        let mut scored: Vec<(TrajectoryStep, f32, f32)> = local_rej
            .into_iter()
            .map(|(rank, s)| {
                let delta = s.gain - mean_gain;
                let sign = if delta >= 0.0 { 1.0 } else { -1.0 };
                let recency = 0.90_f32.powi(rank as i32);
                let score = (recency * (delta.abs() / max_dev)).max(1e-4);
                (s, score, sign)
            })
            .collect();
        scored.sort_by(|a, b| b.1.total_cmp(&a.1));
        for (step, score, sign) in scored {
            if selected.len() >= TRAJECTORY_SEEDS {
                break;
            }
            if !selected.iter().any(|(prev, _, _)| prev.seed == step.seed) {
                selected.push((step, score, sign));
            }
        }
        selected
    }

    fn select_steps(&self, current_base_id: i64) -> Vec<(TrajectoryStep, f32, f32)> {
        if self.local_streak >= 10 {
            return self.select_centered(current_base_id);
        }
        let mut selected: Vec<(TrajectoryStep, f32, f32)> = Vec::with_capacity(TRAJECTORY_SEEDS);
        let trailing_rej = self
            .steps
            .iter()
            .rev()
            .take_while(|s| !s.accepted && s.base_id == current_base_id)
            .count();
        let accepted_recent: Vec<(usize, TrajectoryStep)> = self
            .steps
            .iter()
            .rev()
            .filter(|s| s.accepted && s.gain >= 0.0)
            .take(8)
            .copied()
            .enumerate()
            .collect();
        let rejected_local: Vec<(usize, TrajectoryStep)> = self
            .steps
            .iter()
            .rev()
            .filter(|s| !s.accepted && s.base_id == current_base_id && s.gain < 0.0)
            .take(2)
            .copied()
            .enumerate()
            .collect();
        let rej_cap = rejected_local.len().min(TRAJECTORY_SEEDS / 2);
        let acc_cap = (TRAJECTORY_SEEDS - rej_cap).min(accepted_recent.len());
        if acc_cap > 0 {
            let max_gain = accepted_recent
                .iter()
                .map(|(_, s)| s.gain)
                .fold(0.0_f32, f32::max);
            let scale_gain = max_gain.max(1e-6);
            let mom_decay = 0.75_f32.powi(trailing_rej as i32);
            let mut scored_acc: Vec<(TrajectoryStep, f32)> = accepted_recent
                .into_iter()
                .map(|(rank, s)| {
                    let recency = 0.85_f32.powi(rank as i32) * mom_decay;
                    let boltzmann = (1.5 * (s.gain - max_gain) / scale_gain).exp();
                    (s, (recency * boltzmann).max(1e-4))
                })
                .collect();
            scored_acc.sort_by(|a, b| b.1.total_cmp(&a.1));
            for (step, score) in scored_acc {
                if selected.len() >= acc_cap {
                    break;
                }
                if !selected.iter().any(|(prev, _, _)| prev.seed == step.seed) {
                    selected.push((step, score, 1.0));
                }
            }
        }
        if !rejected_local.is_empty() {
            let max_loss = rejected_local
                .iter()
                .map(|(_, s)| (-s.gain).max(0.0))
                .fold(0.0_f32, f32::max);
            let scale_loss = max_loss.max(1e-6);
            let mut scored_rej: Vec<(TrajectoryStep, f32)> = rejected_local
                .into_iter()
                .map(|(rank, s)| {
                    let recency = 0.50_f32.powi(rank as i32);
                    let loss = (-s.gain).max(0.0);
                    let boltzmann = ((loss - max_loss) / scale_loss).exp();
                    (s, (0.45 * recency * boltzmann).max(1e-4))
                })
                .collect();
            scored_rej.sort_by(|a, b| b.1.total_cmp(&a.1));
            for (step, score) in scored_rej {
                if selected.len() >= TRAJECTORY_SEEDS {
                    break;
                }
                if !selected.iter().any(|(prev, _, _)| prev.seed == step.seed) {
                    selected.push((step, score, -1.0));
                }
            }
        }
        selected
    }

    /// Pack up to `TRAJECTORY_SEEDS` historical seeds and module-conditional
    /// unit-norm mixing coefficients for the next proposal at
    /// `current_base_id`.
    pub(super) fn pack(
        &self,
        current_base_id: i64,
    ) -> (
        u32,
        [u64; TRAJECTORY_SEEDS],
        [[f32; TRAJECTORY_SEEDS]; FAMILIES],
        [f32; FAMILIES],
    ) {
        let selected = self.select_steps(current_base_id);
        if selected.is_empty() {
            return (
                0,
                [0; TRAJECTORY_SEEDS],
                [[0.0; TRAJECTORY_SEEDS]; FAMILIES],
                [1.0; FAMILIES],
            );
        }

        let k = selected.len();
        let mut seeds = [0_u64; TRAJECTORY_SEEDS];
        for (j, (step, _, _)) in selected.iter().enumerate() {
            seeds[j] = step.seed;
        }

        let mut weights = [[0.0_f32; TRAJECTORY_SEEDS]; FAMILIES];
        let mut fresh = [1.0_f32; FAMILIES];
        let trailing_rej = self
            .steps
            .iter()
            .rev()
            .take_while(|s| !s.accepted && s.base_id == current_base_id)
            .count();
        let streak_damp = 0.70_f32.powi(trailing_rej.saturating_sub(1) as i32);
        let rho_base = if self.local_streak >= 10 {
            (0.12 * k as f32).clamp(0.24, 0.48)
        } else {
            ((0.08 * k as f32).clamp(0.08, 0.28) * streak_damp).clamp(0.03, 0.28)
        };
        let (max_rho, min_fresh) = if self.local_streak >= 10 {
            (0.56_f32, 0.44_f32)
        } else {
            (0.36_f32, 0.64_f32)
        };

        for g in 0..FAMILIES {
            let rho = (rho_base * (0.5 + 2.0 * self.online_em_stats[g])).clamp(0.02, max_rho);
            let mut raw = [0.0_f32; TRAJECTORY_SEEDS];
            let mut raw_sum = 0.0_f32;
            for (j, (step, score, _)) in selected.iter().enumerate() {
                let mod_factor = 0.5 + 0.5 * (FAMILIES as f32) * step.family_share[g];
                let w = (score * mod_factor).max(1e-6);
                raw[j] = w;
                raw_sum += w;
            }
            let mut sq_sum = 0.0_f32;
            if raw_sum > 0.0 {
                for (j, (_, _, sign)) in selected.iter().enumerate() {
                    let norm_w = raw[j] / raw_sum;
                    let alpha = sign * (rho * norm_w).sqrt();
                    weights[g][j] = alpha;
                    sq_sum += alpha * alpha;
                }
            }
            fresh[g] = (1.0 - sq_sum).max(min_fresh).sqrt();
        }

        (k as u32, seeds, weights, fresh)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn norm_simplex() {
        let mut traj = TrajectoryGenerator::new();
        assert_eq!(traj.pack(1).0, 0);

        // Observe an accepted step and a rejected probe at incumbent 2.
        traj.observe(101, 1, 0.05, true, [0.4, 0.3, 0.2, 0.1]);
        traj.observe(102, 2, -0.03, false, [0.1, 0.2, 0.3, 0.4]);

        let stat_sum: f32 = traj.online_em_stats.iter().sum();
        assert!((stat_sum - 1.0).abs() < 1e-6);
        assert!(traj.online_em_stats.iter().all(|&s| s > 0.0));

        let (count, seeds, weights, fresh) = traj.pack(2);
        assert_eq!(count, 2);
        assert_eq!(seeds[0], 101);
        assert_eq!(seeds[1], 102);

        for g in 0..FAMILIES {
            // Accepted seed has positive weight; local rejected seed has
            // negative (antipodal) weight.
            assert!(weights[g][0] > 0.0);
            assert!(weights[g][1] < 0.0);
            let norm_sq =
                fresh[g] * fresh[g] + weights[g][0] * weights[g][0] + weights[g][1] * weights[g][1];
            assert!(
                (norm_sq - 1.0).abs() < 1e-5,
                "family {g} norm_sq={norm_sq} != 1.0"
            );
        }

        // Advancing incumbent to 3 drops the local rejected probe from
        // incumbent 2 while keeping the accepted velocity step.
        let (count_next, seeds_next, weights_next, _) = traj.pack(3);
        assert_eq!(count_next, 1);
        assert_eq!(seeds_next[0], 101);
        for g in 0..FAMILIES {
            assert!(weights_next[g][0] > 0.0);
        }

        // Even after multiple accepted steps, local rejected probes at the
        // current incumbent are retained and weighted by loss magnitude.
        traj.observe(103, 3, 0.04, true, [0.25; 4]);
        traj.observe(104, 4, 0.06, true, [0.25; 4]);
        traj.observe(105, 5, -0.01, false, [0.25; 4]);
        traj.observe(106, 5, -0.08, false, [0.25; 4]);
        let (count5, seeds5, weights5, fresh5) = traj.pack(5);
        assert_eq!(count5, 4);
        assert_eq!(seeds5[2], 106);
        assert_eq!(seeds5[3], 105);
        for g in 0..FAMILIES {
            assert!(weights5[g][0] > 0.0 && weights5[g][1] > 0.0);
            assert!(weights5[g][2] < weights5[g][3] && weights5[g][3] < 0.0);
            let sq: f32 = fresh5[g] * fresh5[g] + weights5[g].iter().map(|w| w * w).sum::<f32>();
            assert!((sq - 1.0).abs() < 1e-5);
        }
    }

    #[test]
    fn trajectory_em() -> Result<(), String> {
        autoreleasepool(|| {
            let len = 4096;
            let base = vec![0x3c00u16; 4 * len];
            let blocks = (0..4)
                .map(|g| ParamBlock::new(g as u64, g * len, len, 0.25, 16.0 / len as f32))
                .collect::<Result<Vec<_>, _>>()?;
            let mut s = SearchState::new_implicit(
                &base,
                blocks,
                2,
                TRLengthConfig::new(0.05, 0.001, 0.2),
                ProposalKernel::Gaussian,
            )?;
            s.enable_family(vec![0, 1, 2, 3])?;
            let mut config = crate::config::RunConfig {
                k_neighbors: Some(4),
                history_geometry: Some(crate::config::HistoryGeometry::Latent),
                proposal_method: Some(crate::procedural_pool::ProposalMethod::QuadraticFlow),
                ..Default::default()
            }
            .resident_enn(77)?;
            config.pool = crate::procedural_pool::ProceduralPoolConfig {
                arms: 1,
                candidates_per_arm: 1,
            }
            .resolve_resident()?;
            s.configure_enn(config)?;
            assert!(s.trajectory.is_some());
            s.observe_initial(-2.5, 0.001)?;

            // Step 1: evaluate and reject a probe at the initial incumbent.
            let _ = s.sample_distances(1001)?;
            let (_, prop1) = s.materialize_selected(
                1001,
                0,
                -2.5,
                -2.5,
                0.01,
                -2.5,
                0.01,
                1,
                1,
                [(-2.5, 0.01, -2.5); 4],
            )?;
            s.commit_host(
                &prop1,
                1,
                crate::objective_observation::ObjectiveObservation::scalar_adapter(-2.6, 0.001),
                false,
                0.05,
            )?;
            assert_eq!(s.sync()?, vec![false]);
            assert_eq!(s.trajectory.as_ref().unwrap().pack(s.base_id).0, 1);
            // Rejected local probe has negative antipodal mixing weight.
            assert!(s.trajectory.as_ref().unwrap().pack(s.base_id).2[0][0] < 0.0);

            // Step 2: evaluate and accept a step at the incumbent; verify GPU
            // materialization matches single-candidate `params` and
            // updates Online EM family scales.
            let direct = read::<u16>(&s.test_candidate(1002, 0)?, base.len());
            let _ = s.sample_distances(1002)?;
            let (_, prop2) = s.materialize_selected(
                1002,
                0,
                -2.5,
                -2.5,
                0.01,
                -2.5,
                0.01,
                1,
                1,
                [(-2.5, 0.01, -2.5); 4],
            )?;
            let materialized = read::<u16>(&s.proposal, base.len());
            assert_eq!(materialized, direct);

            s.commit_host(
                &prop2,
                2,
                crate::objective_observation::ObjectiveObservation::scalar_adapter(-2.3, 0.001),
                true,
                0.05,
            )?;
            assert_eq!(s.sync()?, vec![true]);
            let (count, seeds, weights, _) = s.trajectory.as_ref().unwrap().pack(s.base_id);
            assert_eq!(count, 1);
            assert_eq!(seeds[0], prop2.seed);
            assert!(weights[0][0] > 0.0);
            let (f_weights, f_scales) = s.family_shape().unwrap();
            assert!((f_weights.iter().product::<f32>() - 1.0).abs() < 1e-4);
            assert!(f_scales.iter().all(|&sc| sc.is_finite() && sc > 0.0));
            Ok(())
        })
    }
}
