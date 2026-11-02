//! Curvature-centered Bayesian linear-quadratic Kalman filtering over
//! elementary Ziggurat seeds.
//!
//! Deconvolves mixed trajectory steps into elementary fresh seeds `z(seed)`,
//! estimates online isotropic quadratic curvature `curvature` (`kappa_hat`),
//! attributes curvature-centered directional derivatives
//! `mu_obs = (gain + kappa_hat * r^2) / r`, updates active historical seed
//! slopes via Kalman innovation, and packs up to `TRAJECTORY_CAPACITY`
//! unit-norm mixing weights (`alpha_fresh^2 + sum_j alpha_j^2 = 1`).
use super::*;

pub const TRAJECTORY_CAPACITY: usize = 8;

#[derive(Clone, Copy, Debug)]
pub struct ElementarySeed {
    pub seed: u64,
    pub base_id: i64,
    pub slope: f32,
    pub precision: f32,
    pub accepted: bool,
}

#[derive(Clone, Debug)]
pub struct TrajectoryGenerator {
    pub steps: Vec<ElementarySeed>,
    pub curvature: f32,
    pub slope_scale: f32,
    pub last_radius: f32,
}

impl Default for TrajectoryGenerator {
    fn default() -> Self {
        Self::new()
    }
}

fn damp_innovation(delta: f32, scale: f32, accepted: bool) -> f32 {
    if !accepted && delta < 0.0 && scale > 0.0 {
        let ratio = delta.abs() / scale;
        -scale * (ratio / (1.0 + 0.5 * ratio))
    } else {
        delta
    }
}

fn update_slope(old: f32, updated: f32, precision: f32, mu_obs: f32, accepted: bool) -> f32 {
    if accepted {
        return if old >= 0.0 {
            updated.max(0.50 * old)
        } else {
            updated.min(0.50 * old)
        };
    }
    if (old < 0.0 && mu_obs < 0.0) || precision > 1.15 {
        return 0.0;
    }
    if old >= 0.0 {
        updated.max(0.25 * old)
    } else {
        updated.min(0.25 * old)
    }
}

impl TrajectoryGenerator {
    pub fn new() -> Self {
        Self {
            steps: Vec::with_capacity(MAX_HISTORY),
            curvature: 0.0,
            slope_scale: 0.0,
            last_radius: 0.0,
        }
    }

    pub fn reset(&mut self) {
        self.steps.clear();
        self.curvature = 0.0;
        self.slope_scale = 0.0;
        self.last_radius = 0.0;
    }

    /// Record one completed trial evaluation `(seed, base_id, radius, gain,
    /// accepted)` and update the curvature-centered Kalman posterior over
    /// elementary seeds.
    pub fn observe(&mut self, seed: u64, base_id: i64, radius: f32, gain: f32, accepted: bool) {
        if !gain.is_finite() || !radius.is_finite() || radius <= 0.0 {
            return;
        }
        if self.last_radius > 0.0 && (radius - self.last_radius).abs() > 1e-6 {
            self.curvature *= self.last_radius / radius;
        }
        self.last_radius = radius;
        let (count, hist_seeds, hist_weights, fresh_w) = self.pack(base_id);
        let mut mu_pred = 0.0_f32;
        for j in 0..count as usize {
            if let Some(s) = self.steps.iter().rev().find(|s| s.seed == hist_seeds[j]) {
                mu_pred += hist_weights[j] * s.slope;
            }
        }
        let r_sq = radius * radius;
        let mu_obs = (gain + 0.5 * self.curvature * r_sq) / radius;
        if gain < 0.0 {
            let kappa_obs = (-gain / r_sq).max(1.0);
            self.curvature = if self.curvature == 0.0 {
                kappa_obs
            } else {
                let ratio = (kappa_obs / self.curvature).clamp(0.5, 1.25);
                self.curvature * ratio.powf(0.15 * fresh_w * fresh_w)
            };
        } else if self.curvature == 0.0 && radius * mu_pred > gain {
            self.curvature = ((radius * mu_pred - gain) / r_sq).max(1.0);
        }
        let delta = mu_obs - mu_pred;
        let delta_eff = damp_innovation(delta, self.slope_scale, accepted);
        self.update_history(
            count as usize,
            &hist_seeds,
            &hist_weights,
            mu_obs,
            mu_pred,
            delta_eff,
            radius,
            accepted,
        );
        let mu_fresh =
            self.fresh_slope(fresh_w, mu_obs, mu_pred, delta_eff, radius, gain, accepted);
        let abs_slope = mu_fresh.abs();
        if abs_slope > 0.0 {
            self.slope_scale = if self.slope_scale == 0.0 {
                abs_slope
            } else {
                0.80 * self.slope_scale + 0.20 * abs_slope
            };
        }
        self.push_seed(ElementarySeed {
            seed,
            base_id,
            slope: mu_fresh,
            precision: fresh_w * fresh_w,
            accepted,
        });
    }

    #[allow(clippy::too_many_arguments)]
    fn update_history(
        &mut self,
        count: usize,
        hist_seeds: &[u64; TRAJECTORY_CAPACITY],
        hist_weights: &[f32; TRAJECTORY_CAPACITY],
        mu_obs: f32,
        mu_pred: f32,
        delta_eff: f32,
        radius: f32,
        accepted: bool,
    ) {
        let step_shift = if accepted {
            delta_eff - 0.5 * self.curvature * radius
        } else {
            delta_eff.min(-0.35 * mu_pred.abs())
        };
        for j in 0..count {
            let w = hist_weights[j];
            if let Some(s) = self
                .steps
                .iter_mut()
                .rev()
                .find(|s| s.seed == hist_seeds[j])
            {
                let old = s.slope;
                let updated = old + w * step_shift;
                s.slope = update_slope(old, updated, s.precision, mu_obs, accepted);
                s.precision += w * w;
            }
        }
        if accepted {
            for s in &mut self.steps {
                s.slope *= 0.85;
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn fresh_slope(
        &self,
        fresh_w: f32,
        mu_obs: f32,
        mu_pred: f32,
        delta_eff: f32,
        radius: f32,
        gain: f32,
        accepted: bool,
    ) -> f32 {
        if accepted {
            let raw_acc = (gain + 0.5 * self.curvature * radius * radius) / radius;
            let cap = if self.slope_scale > 0.0 {
                (2.5 * self.slope_scale).max(1.0)
            } else {
                raw_acc.abs().max(1.0)
            };
            let mu_acc = raw_acc.min(cap);
            if fresh_w >= 0.999 {
                mu_acc.max(1e-4)
            } else {
                (fresh_w * delta_eff).max(0.5 * fresh_w * mu_acc).max(1e-4)
            }
        } else if mu_obs >= 0.0 {
            (fresh_w * (mu_obs - 0.25 * mu_pred.max(0.0))).max(0.1 * fresh_w * mu_obs)
        } else {
            let raw = mu_obs.min(mu_obs - 0.25 * mu_pred.max(0.0));
            fresh_w * damp_innovation(raw, self.slope_scale, false)
        }
    }

    fn push_seed(&mut self, entry: ElementarySeed) {
        if let Some(existing) = self.steps.iter_mut().find(|s| s.seed == entry.seed) {
            *existing = entry;
            return;
        }
        if self.steps.len() == MAX_HISTORY {
            let stale = self
                .steps
                .iter()
                .position(|s| !s.accepted && s.base_id != entry.base_id)
                .unwrap_or(0);
            self.steps.remove(stale);
        }
        self.steps.push(entry);
    }

    fn select_seeds(&self, current_base: i64) -> Vec<(u64, f32)> {
        let mut candidates: Vec<(u64, f32)> = Vec::with_capacity(self.steps.len());
        let mut acc_rank = 0_i32;
        for step in self.steps.iter().rev() {
            if candidates.iter().any(|&(s, _)| s == step.seed) {
                continue;
            }
            let eff_slope = if step.base_id == current_base {
                if step.accepted {
                    acc_rank += 1;
                }
                step.slope
            } else if step.accepted && step.slope.abs() > 1e-6 {
                let decayed = step.slope * 0.90_f32.powi(acc_rank);
                acc_rank += 1;
                decayed
            } else {
                continue;
            };
            if eff_slope.abs() > 1e-6 {
                candidates.push((step.seed, eff_slope));
            }
        }
        candidates.sort_by(|a, b| b.1.abs().total_cmp(&a.1.abs()));
        if let Some(&(_, first_mu)) = candidates.first() {
            let cutoff = (0.04 * first_mu.abs()).max(1e-5);
            candidates.retain(|&(_, mu)| mu.abs() >= cutoff);
        }
        candidates.truncate(TRAJECTORY_CAPACITY);
        candidates
    }

    /// Pack up to `TRAJECTORY_CAPACITY` historical elementary seeds and
    /// unit-norm mixing coefficients for the next proposal at `current_base`.
    pub fn pack(
        &self,
        current_base: i64,
    ) -> (
        u32,
        [u64; TRAJECTORY_CAPACITY],
        [f32; TRAJECTORY_CAPACITY],
        f32,
    ) {
        let selected = self.select_seeds(current_base);
        if selected.is_empty() {
            return (0, [0; TRAJECTORY_CAPACITY], [0.0; TRAJECTORY_CAPACITY], 1.0);
        }
        let k = selected.len();
        let mut seeds = [0_u64; TRAJECTORY_CAPACITY];
        let mut weights = [0.0_f32; TRAJECTORY_CAPACITY];
        let drift_sq: f32 = selected.iter().map(|(_, mu)| mu * mu).sum();
        let mu_drift = drift_sq.sqrt();
        if mu_drift <= 1e-12 {
            return (0, [0; TRAJECTORY_CAPACITY], [0.0; TRAJECTORY_CAPACITY], 1.0);
        }
        let prior_var = (self.slope_scale * self.slope_scale).max(1e-6);
        let cap = (0.20 * k as f32).clamp(0.20, 0.65);
        let rho = ((drift_sq / (drift_sq + prior_var)) * cap).clamp(0.05, 0.65);
        let amp = rho.sqrt();
        let mut sq_sum = 0.0_f32;
        for (j, &(seed, mu)) in selected.iter().enumerate() {
            seeds[j] = seed;
            let w = amp * (mu / mu_drift);
            weights[j] = w;
            sq_sum += w * w;
        }
        let fresh = (1.0 - sq_sum).max(0.25).sqrt();
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

        // Observe an accepted step at incumbent 1 and a rejected probe at
        // incumbent 2.
        traj.observe(101, 1, 0.05, 0.05, true);
        traj.observe(102, 2, 0.05, -0.03, false);

        let (count, seeds, weights, fresh) = traj.pack(2);
        assert_eq!(count, 2);
        assert!(seeds[..2].contains(&101));
        assert!(seeds[..2].contains(&102));

        let idx_101 = seeds[..2].iter().position(|&s| s == 101).unwrap();
        let idx_102 = seeds[..2].iter().position(|&s| s == 102).unwrap();
        assert!(weights[idx_101] > 0.0);
        assert!(weights[idx_102] < 0.0);

        let norm_sq = fresh * fresh + weights[0] * weights[0] + weights[1] * weights[1];
        assert!((norm_sq - 1.0).abs() < 1e-5, "norm_sq={norm_sq} != 1.0");

        // Advancing incumbent to 3 drops the local rejected probe from
        // incumbent 2 while keeping the accepted velocity step.
        let (count_next, seeds_next, weights_next, fresh_next) = traj.pack(3);
        assert_eq!(count_next, 1);
        assert_eq!(seeds_next[0], 101);
        assert!(weights_next[0] > 0.0);
        assert!((fresh_next * fresh_next + weights_next[0] * weights_next[0] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn curvature_sign() {
        let mut traj = TrajectoryGenerator::new();
        let r = 0.05_f32;
        // Initialize curvature with a rejected step at incumbent 1:
        // kappa_hat = (-(-0.04) / 0.05^2) = 16.0, so kappa_hat * r^2 = 0.04.
        traj.observe(200, 1, r, -0.04, false);
        assert!((traj.curvature * r * r - 0.04).abs() < 1e-5);

        // At incumbent 2, observe a rejected probe with small loss:
        // 0 < -gain (0.01) < kappa_hat * r^2 (0.04) -> positive slope &
        // positive weight!
        traj.observe(201, 2, r, -0.01, false);
        let (count_a, seeds_a, weights_a, fresh_a) = traj.pack(2);
        assert_eq!(count_a, 1);
        assert_eq!(seeds_a[0], 201);
        assert!(
            weights_a[0] > 0.0,
            "small-loss rejected probe should get positive weight, got {}",
            weights_a[0]
        );
        assert!((fresh_a * fresh_a + weights_a[0] * weights_a[0] - 1.0).abs() < 1e-5);

        // Now observe a rejected probe with large loss at incumbent 2:
        // -gain (0.12) > kappa_hat * r^2 -> negative slope & negative weight!
        traj.observe(202, 2, r, -0.12, false);
        let (count_b, seeds_b, weights_b, fresh_b) = traj.pack(2);
        assert_eq!(count_b, 2);
        let idx_small = seeds_b[..2].iter().position(|&s| s == 201).unwrap();
        let idx_large = seeds_b[..2].iter().position(|&s| s == 202).unwrap();
        assert!(weights_b[idx_small] > 0.0);
        assert!(
            weights_b[idx_large] < 0.0,
            "large-loss rejected probe should get negative weight, got {}",
            weights_b[idx_large]
        );
        let sq: f32 = fresh_b * fresh_b + weights_b.iter().map(|w| w * w).sum::<f32>();
        assert!((sq - 1.0).abs() < 1e-5);
    }

    #[test]
    fn kalman_attenuation() {
        let mut traj = TrajectoryGenerator::new();
        let r = 0.05_f32;
        traj.observe(301, 1, r, 0.06, true);
        let slope_before = traj.steps.iter().find(|s| s.seed == 301).unwrap().slope;
        assert!(slope_before > 0.0);

        // Rejecting a mixed step at incumbent 2 attenuates the historical
        // seed's slope.
        traj.observe(302, 2, r, -0.08, false);
        let slope_after = traj.steps.iter().find(|s| s.seed == 301).unwrap().slope;
        assert!(
            slope_after > 0.0 && slope_after < slope_before,
            "expected 0 < slope_after ({slope_after}) < slope_before ({slope_before})"
        );
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
            // Rejected initial probe has negative antipodal mixing weight.
            assert!(s.trajectory.as_ref().unwrap().pack(s.base_id).2[0] < 0.0);

            // Step 2: evaluate and accept a step at the incumbent; verify GPU
            // materialization matches single-candidate `params`.
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
            assert!(weights[0] > 0.0);
            Ok(())
        })
    }
}
