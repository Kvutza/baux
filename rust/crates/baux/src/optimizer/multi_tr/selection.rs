//! Candidate budgets, acquisition selection, and region bounds.

use super::*;

impl MultiTrustRegionState {
    /// Divide a flat candidate budget among active regions with a UCB policy.
    pub fn allocate(&self, budget: usize) -> Result<Vec<RegionBatch>, TrustRegionError> {
        self.allocate_with(budget, &self.mean_y.view())
    }

    /// Allocate with objective-specific region utility.
    ///
    /// Utilities may be scalar trajectory values, hypervolume contributions,
    /// or feasibility scores. Larger values receive more candidates.
    pub fn allocate_with(
        &self,
        budget: usize,
        utility: &ArrayView1<f64>,
    ) -> Result<Vec<RegionBatch>, TrustRegionError> {
        if utility.len() != self.num_regions || utility.iter().any(|value| !value.is_finite()) {
            return Err(TrustRegionError::InvalidParameter(
                "region utility must contain one finite value per region".to_string(),
            ));
        }
        let active: Vec<usize> = self
            .active_mask
            .iter()
            .enumerate()
            .filter_map(|(r, &is_active)| is_active.then_some(r))
            .collect();
        if active.is_empty() {
            return Err(TrustRegionError::InvalidState(
                "no active trust regions".to_string(),
            ));
        }
        if budget == 0 {
            return Err(TrustRegionError::InvalidParameter(
                "candidate budget must be > 0".to_string(),
            ));
        }

        if self.config.allocation_policy == AllocationPolicy::Balanced {
            let mut counts = vec![0usize; self.num_regions];
            for _ in 0..budget {
                let region = active
                    .iter()
                    .copied()
                    .min_by_key(|&region| {
                        (
                            u128::from(self.trial_counts[region]) + counts[region] as u128,
                            region,
                        )
                    })
                    .expect("active regions are non-empty");
                counts[region] += 1;
            }
            let mut start = 0;
            return Ok(active
                .into_iter()
                .filter_map(|region| {
                    let len = counts[region];
                    if len == 0 {
                        return None;
                    }
                    let batch = RegionBatch { region, start, len };
                    start += len;
                    Some(batch)
                })
                .collect());
        }

        let total_trials = self.trial_counts.iter().sum::<u64>().max(1) as f64;
        if budget < active.len() {
            let mut ranked = active;
            ranked.sort_by(|&a, &b| {
                self.priority(b, 0, total_trials, utility[b])
                    .total_cmp(&self.priority(a, 0, total_trials, utility[a]))
                    .then_with(|| a.cmp(&b))
            });
            let mut start = 0;
            return Ok(ranked
                .into_iter()
                .take(budget)
                .map(|region| {
                    let batch = RegionBatch {
                        region,
                        start,
                        len: 1,
                    };
                    start += 1;
                    batch
                })
                .collect());
        }

        let mut counts = vec![1usize; self.num_regions];
        for _ in active.len()..budget {
            let region = active
                .iter()
                .copied()
                .max_by(|&a, &b| {
                    self.priority(a, counts[a], total_trials, utility[a])
                        .total_cmp(&self.priority(b, counts[b], total_trials, utility[b]))
                        .then_with(|| b.cmp(&a))
                })
                .expect("active regions are non-empty");
            counts[region] += 1;
        }

        let mut start = 0;
        Ok(active
            .into_iter()
            .map(|region| {
                let batch = RegionBatch {
                    region,
                    start,
                    len: counts[region],
                };
                start += batch.len;
                batch
            })
            .collect())
    }

    /// Sample variance of completed objectives for a region.
    pub fn variance(&self, region: usize) -> Option<f64> {
        let count = *self.trial_counts.get(region)?;
        (count > 1).then(|| self.m2_y[region] / (count - 1) as f64)
    }

    /// Select `num_arms` candidates globally while avoiding duplicate
    /// perturbations.
    pub fn select(
        &self,
        candidates: &[RegionCandidate],
        num_arms: usize,
    ) -> Result<Vec<RegionCandidate>, TrustRegionError> {
        if num_arms == 0 || num_arms > candidates.len() {
            return Err(TrustRegionError::InvalidParameter(format!(
                "num_arms {num_arms} is outside 1..={}",
                candidates.len()
            )));
        }
        if candidates
            .iter()
            .any(|c| c.region >= self.num_regions || !c.score.is_finite())
        {
            return Err(TrustRegionError::InvalidParameter(
                "candidate has an invalid region or score".to_string(),
            ));
        }

        let mut ranked = candidates.to_vec();
        ranked.sort_by(|a, b| {
            b.score
                .total_cmp(&a.score)
                .then_with(|| a.index.cmp(&b.index))
        });

        let mut selected = Vec::with_capacity(num_arms);
        for candidate in &ranked {
            if selected
                .iter()
                .all(|other: &RegionCandidate| self.seed_distance(candidate.seed, other.seed))
            {
                selected.push(*candidate);
                if selected.len() == num_arms {
                    return Ok(selected);
                }
            }
        }

        for candidate in ranked {
            if !selected.iter().any(|other| other.index == candidate.index) {
                selected.push(candidate);
                if selected.len() == num_arms {
                    break;
                }
            }
        }
        Ok(selected)
    }

    /// Compute lower and upper bounds for a specific trust region `r`.
    pub fn compute_bounds(
        &self,
        r: usize,
        lengthscales: Option<&ArrayView1<f64>>,
    ) -> (Array1<f64>, Array1<f64>) {
        let center = self.centers.row(r);
        let len = self.lengths[r];
        let mut lb = Array1::zeros(self.num_dim);
        let mut ub = Array1::zeros(self.num_dim);

        if let Some(ls) = lengthscales {
            let ls_mean = ls.mean().unwrap_or(1.0);
            for d in 0..self.num_dim {
                let half = 0.5 * len * (ls[d] / ls_mean);
                lb[d] = (center[d] - half).clamp(0.0, 1.0);
                ub[d] = (center[d] + half).clamp(0.0, 1.0);
            }
        } else {
            let half = 0.5 * len;
            for d in 0..self.num_dim {
                lb[d] = (center[d] - half).clamp(0.0, 1.0);
                ub[d] = (center[d] + half).clamp(0.0, 1.0);
            }
        }

        (lb, ub)
    }

    fn priority(&self, region: usize, queued: usize, total_trials: f64, utility: f64) -> f64 {
        if self.config.exploration == 0.0 {
            return utility;
        }
        let trials = self.trial_counts[region] as f64;
        let effective_trials = (trials + queued as f64).max(1e-6);
        let uncertainty =
            ((total_trials + self.num_regions as f64 + 1.0).ln() / effective_trials).sqrt();
        utility + self.config.exploration * uncertainty
    }

    pub(super) fn seed_distance(&self, a: u64, b: u64) -> bool {
        (a ^ b).count_ones() as f64 / u64::BITS as f64 >= self.config.min_seed_distance
    }
}
