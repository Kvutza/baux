//! Candidate budgets, acquisition selection, and region bounds.

use super::*;

impl MultiTrustRegionState {
    /// Divide a flat candidate budget among active regions with a UCB policy.
    pub fn allocate(&self, budget: usize) -> Result<Vec<RegionBatch>, TrustRegionError> {
        self.ledger
            .allocate(budget, &self.active_mask)
            .map_err(|error| TrustRegionError::InvalidState(error.to_string()))
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
        let utility = utility.iter().copied().collect::<Vec<_>>();
        self.ledger
            .allocate_with(budget, &self.active_mask, &utility)
            .map_err(|error| TrustRegionError::InvalidState(error.to_string()))
    }

    /// Sample variance of completed objectives for a region.
    pub fn variance(&self, region: usize) -> Option<f64> {
        self.ledger.variance(region)
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
                let lo = center[d] - half;
                let hi = center[d] + half;
                lb[d] = if self.config.clamp_domain {
                    lo.clamp(0.0, 1.0)
                } else {
                    lo
                };
                ub[d] = if self.config.clamp_domain {
                    hi.clamp(0.0, 1.0)
                } else {
                    hi
                };
            }
        } else {
            let half = 0.5 * len;
            let c_slice = center.as_slice().expect("center row contiguous");
            let lb_slice = lb.as_slice_mut().expect("lb contiguous");
            let ub_slice = ub.as_slice_mut().expect("ub contiguous");
            if self.config.clamp_domain {
                for d in 0..self.num_dim {
                    lb_slice[d] = (c_slice[d] - half).clamp(0.0, 1.0);
                    ub_slice[d] = (c_slice[d] + half).clamp(0.0, 1.0);
                }
            } else {
                for d in 0..self.num_dim {
                    lb_slice[d] = c_slice[d] - half;
                    ub_slice[d] = c_slice[d] + half;
                }
            }
        }

        (lb, ub)
    }

    pub(super) fn seed_distance(&self, a: u64, b: u64) -> bool {
        (a ^ b).count_ones() as f64 / u64::BITS as f64 >= self.config.min_seed_distance
    }
}
