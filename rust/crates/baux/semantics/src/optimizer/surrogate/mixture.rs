//! Surrogate-informed temporal mutation mixtures, an experimental proposal law.
//!
//! Ermakov and Zhigljavskii, "On a random search of a global extremum" (1983),
//! Algorithm 1, https://www.mathnet.ru/eng/tvp2160, uses outcome-weighted mutation
//! kernels. Huang and Kouhkouh, "Self-interacting CBO" (2025), equation (6),
//! https://arxiv.org/abs/2411.10295, supplies temporal rather than population memory.
//! Bungert, Roith and Wacker, "Polarized consensus-based dynamics" (2025),
//! equation (4), https://doi.org/10.1007/s10107-024-02095-y, localizes interactions.
//! ENN precision weighting follows "Taking the GP Out of the Loop", section
//! 4.1, https://arxiv.org/abs/2506.12818.
//!
//! Here ENN denoises archived rewards and supplies the precision kernel. One
//! categorical draw selects a complete archived state before drawing one
//! full-dimensional mutation. This is our combination, not a convergence result
//! from those papers. It requires neither candidate ranking nor covariance
//! construction. Retaining full states and computing their distances still
//! costs memory and bandwidth; this scalar controller does not eliminate those
//! costs.

use super::{Mutation, NeighborValueEstimator, UncertaintyLaw, loo_fit};

/// Distribution over retained, already evaluated states; not new candidates.
#[derive(Debug, Clone)]
pub struct Mixture {
    probabilities: Vec<f64>,
    means: Vec<f32>,
}

/// A single shared state identity for every coordinate of the next mutation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Anchor {
    pub slot: usize,
    pub probability: f64,
    pub mean: f32,
}

impl Mixture {
    pub fn probabilities(&self) -> &[f64] {
        &self.probabilities
    }

    /// Draw one full-dimensional candidate, sharing its anchor across every
    /// coordinate. Archived states are already evaluated observations. This
    /// CPU reference performs no objective calls, gradient estimation, or
    /// acceptance test; the caller evaluates this one returned candidate.
    pub fn draw(
        &self,
        current: &[f32],
        archive: &[&[f32]],
        mutation: Mutation,
        seed: u64,
    ) -> Result<(Anchor, Vec<f32>), String> {
        if archive.len() != self.probabilities.len()
            || archive.iter().any(|row| row.len() != current.len())
        {
            return Err("temporal mixture archive shape differs from its weights".into());
        }
        let anchor = self.sample(seed);
        let candidate = mutation.apply(current, archive[anchor.slot], seed)?;
        Ok((anchor, candidate))
    }

    /// Draw one whole-state anchor. Per-coordinate anchor draws would define a
    /// different, factorized law and discard this mixture's shared dependence.
    pub fn sample(&self, seed: u64) -> Anchor {
        let uniform = crate::semantics::random::u64_f53(crate::semantics::random::splitmix64(
            seed ^ 0x2d35_8dcc_aa6c_78a5,
        ));
        let mut cumulative = 0.0;
        let mut selected = self
            .probabilities
            .iter()
            .rposition(|&probability| probability > 0.0)
            .expect("validated mixture has positive mass");
        for (slot, &probability) in self.probabilities.iter().enumerate() {
            cumulative += probability;
            if uniform < cumulative {
                selected = slot;
                break;
            }
        }
        Anchor {
            slot: selected,
            probability: self.probabilities[selected],
            mean: self.means[selected],
        }
    }
}

impl NeighborValueEstimator {
    /// Form `p_i ∝ precision_i(query) exp(beta * ENN_mean(state_i))` over the
    /// nearest retained states. Rewards are maximized; beta has inverse-reward
    /// units. Each archived state uses an inclusive ENN prediction, matching
    /// the noisy-observation denoising convention rather than claiming
    /// calibration.
    ///
    /// The fitted metric and uncertainty scales are the same as for value
    /// prediction. A stable log normalization handles large reward offsets.
    pub fn mixture(&self, distances: &[f64], beta: f64) -> Result<Mixture, String> {
        if distances.len() != self.len
            || distances
                .iter()
                .any(|value| !value.is_finite() || *value < 0.0)
            || !beta.is_finite()
            || beta < 0.0
        {
            return Err("invalid temporal mixture distances or inverse temperature".into());
        }
        let params = if self.fitted {
            self.params
        } else {
            self.defaults
        };
        let transformed = loo_fit::metric_row(self, distances, None, self.distance_metric);
        let count = loo_fit::build_query(
            self,
            distances,
            None,
            0.0,
            None,
            self.distance_metric,
            self.uncertainty_law,
        )?
        .neighbors;
        let mut nearest: Vec<_> = (0..self.len).collect();
        nearest.sort_by(|&left, &right| transformed[left].total_cmp(&transformed[right]));
        nearest.truncate(count);
        let mut means = vec![0.0; self.len];
        for &slot in &nearest {
            means[slot] = loo_fit::build_query(
                self,
                &self.distances[slot][..self.len],
                None,
                0.0,
                None,
                self.distance_metric,
                UncertaintyLaw::InversePrecision,
            )?
            .mean;
        }
        let best = nearest
            .iter()
            .map(|&slot| f64::from(means[slot]))
            .max_by(f64::total_cmp)
            .ok_or("temporal mixture has no neighbors")?;
        let scale_sq = (self.y_scale * self.y_scale).max(1.0e-12);
        let mut logits = vec![f64::NEG_INFINITY; self.len];
        for &slot in &nearest {
            let variance = 1.0e-9
                + params.epistemic * transformed[slot]
                + params.aleatoric
                + f64::from(self.variances[slot]) / scale_sq;
            logits[slot] = beta * (f64::from(means[slot]) - best) - variance.max(1.0e-12).ln();
        }
        let maximum = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        if !maximum.is_finite() {
            return Err("temporal mixture has no finite positive weight".into());
        }
        let mut probabilities: Vec<_> =
            logits.iter().map(|logit| (logit - maximum).exp()).collect();
        let total: f64 = probabilities.iter().sum();
        for probability in &mut probabilities {
            *probability /= total;
        }
        Ok(Mixture {
            probabilities,
            means,
        })
    }
}

#[cfg(test)]
mod tests;
