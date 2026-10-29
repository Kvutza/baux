//! Shared streaming prefix selection for precision-weighted neighbours.
//! The distance coefficient is a fitted bias proxy, not a certified Lipschitz
//! constant. No covariance matrix or gradient estimate is constructed.

/// Resident observation capacity, shared by host state and device scratch.
pub const MAX_HISTORY: usize = 128;
#[path = "neighbor/query.rs"]
pub mod query;

/// Posterior variance law for precision-weighted neighbor estimation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum UncertaintyLaw {
    /// Inverse total precision `1 / sum(w_i)` plus precision-weighted
    /// observation noise.
    #[default]
    InversePrecision,
    /// Exact linear-estimator noise variance `sum(w_i^2 * sigma_i^2) /
    /// sum(w_i)^2` plus geometric bias.
    BiasVariance,
    /// Pure geometric distance bias `epistemic * sum(w_i * d_i^2) / sum(w_i)`.
    Geometric,
}

/// Pairwise distance transformation applied to stored and queried rows.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum DistanceMetric {
    #[default]
    Euclidean,
    SelfTuning {
        local_k: usize,
    },
    Hyperbolic {
        local_k: usize,
    },
}

/// Append until full, then replace the oldest observation other than the
/// incumbent. Slots retain their order: Thompson draws follow identities,
/// not the current slot number. Parameter storage must use the same slot.
pub fn retention(identities: &[i64], incumbent: i64) -> Result<usize, &'static str> {
    if identities.len() > MAX_HISTORY {
        return Err("observation history exceeds its capacity");
    }
    if identities.len() < MAX_HISTORY {
        return Ok(identities.len());
    }
    identities
        .iter()
        .enumerate()
        .filter(|(_, identity)| **identity != incumbent)
        .min_by_key(|(_, identity)| **identity)
        .map(|(slot, _)| slot)
        .ok_or("observation history has no replaceable slot")
}

/// Append until full, then replace the oldest cone-dominated observation
/// (`Theorem 3.1` / `SlopeEnvelope.dominates`) other than the incumbent,
/// falling back to `retention` when no slot is cone-dominated.
pub fn retention_cone(
    identities: &[i64],
    incumbent: i64,
    outcomes: &[f32],
    variances: &[f32],
    pairwise_sq_dists: &[f32],
    stride: usize,
    slope: f32,
) -> Result<usize, &'static str> {
    if identities.len() < MAX_HISTORY {
        return retention(identities, incumbent);
    }
    let n = identities.len();
    let lip = slope.max(0.0);
    if lip > 0.0
        && outcomes.len() >= n
        && variances.len() >= n
        && pairwise_sq_dists.len() >= n * stride
    {
        let lower = |k: usize| outcomes[k] - variances[k].max(0.0).sqrt();
        if let Some((slot, _)) = identities
            .iter()
            .enumerate()
            .filter(|&(j, &id)| {
                id != incumbent
                    && (0..n).any(|i| {
                        i != j && {
                            let dist = pairwise_sq_dists[i * stride + j].max(0.0).sqrt();
                            lower(j) <= lower(i) - lip * dist
                        }
                    })
            })
            .min_by_key(|&(_, &id)| id)
        {
            return Ok(slot);
        }
    }
    retention(identities, incumbent)
}

/// Self-tuning distance normalization (`sq_dist / (scale_a * scale_b)`).
#[inline(always)]
pub fn selftuning_delta(sq_dist: f64, scale_a: f64, scale_b: f64) -> f64 {
    let a = scale_a.max(1.0e-12);
    let b = scale_b.max(1.0e-12);
    sq_dist.max(0.0) / (a * b)
}

/// Nonnegative hyperbolic vertical scale-mismatch penalty (`(scale_a -
/// scale_b)^2 / (2 * scale_a * scale_b)`).
#[inline(always)]
pub fn scale_penalty(scale_a: f64, scale_b: f64) -> f64 {
    let a = scale_a.max(1.0e-12);
    let b = scale_b.max(1.0e-12);
    let diff = a - b;
    (diff * diff) / (2.0 * a * b)
}

/// Upper half-space Poincare dissimilarity (`arXiv:2305.01356`):
/// `0.5 * selftuning_delta(sq_dist, scale_a, scale_b) + scale_penalty(scale_a,
/// scale_b)`.
#[inline(always)]
pub fn halfspace_delta(sq_dist: f64, scale_a: f64, scale_b: f64) -> f64 {
    0.5 * selftuning_delta(sq_dist, scale_a, scale_b) + scale_penalty(scale_a, scale_b)
}

/// Boroczky binary-tiling L-order cell key (`arXiv:2305.01356`) packing
/// integer level `floor(log2(scale))` and dilated grid coordinate `floor(coord
/// * 2^-level)`.
#[inline(always)]
pub fn grid_cell(coord: f32, scale: f32) -> u64 {
    let level = scale.max(1.0e-12).log2().floor() as i32;
    let grid = (coord * (-level as f32).exp2()).floor() as i32;
    ((level as u32 as u64) << 32) | (grid as u32 as u64)
}

#[derive(Clone, Copy, Default)]
pub struct Prefix {
    weight: f64,
    noise: f64,
    best: f64,
    pub count: u32,
}

impl Prefix {
    #[inline(always)]
    pub fn push(&mut self, rank: u32, distance: f64, epistemic: f64, noise: f64) {
        let precision = 1.0 / (1.0e-9 + epistemic * distance + noise).max(1.0e-12);
        self.weight += precision;
        self.noise += precision * precision * noise;
        let variance_error = self.noise / (self.weight * self.weight).max(1.0e-24);
        let risk = epistemic * distance + variance_error.sqrt();
        if self.count == 0 || risk < self.best {
            self.best = risk;
            self.count = rank;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bias_noise() {
        let mut noisy = Prefix::default();
        for rank in 1..=8 {
            noisy.push(rank, 1.0, 0.0, 1.0);
        }
        assert_eq!(noisy.count, 8);
        let mut separated = Prefix::default();
        separated.push(1, 0.0, 1.0, 0.01);
        separated.push(2, 100.0, 1.0, 0.01);
        assert_eq!(separated.count, 1);

        let horo = halfspace_delta(4.0_f64, 2.0_f64, 2.0_f64);
        assert!((horo - 0.5).abs() < 1.0e-6);
        let dilated = halfspace_delta(16.0_f64, 4.0_f64, 4.0_f64);
        assert!((horo - dilated).abs() < 1.0e-6);
        assert_eq!(grid_cell(1.5, 0.5), grid_cell(1.75, 0.5));
        assert_ne!(grid_cell(1.5, 0.5), grid_cell(1.5, 2.0));

        let ids: Vec<i64> = (1..=MAX_HISTORY as i64).collect();
        let mut outcomes = vec![-2.0f32; MAX_HISTORY];
        let variances = vec![0.0f32; MAX_HISTORY];
        let mut sq_dists = vec![100.0f32; MAX_HISTORY * MAX_HISTORY];
        for i in 0..MAX_HISTORY {
            sq_dists[i * MAX_HISTORY + i] = 0.0;
        }
        outcomes[0] = -1.0;
        outcomes[5] = -5.0;
        sq_dists[0 * MAX_HISTORY + 5] = 0.01;
        sq_dists[5 * MAX_HISTORY + 0] = 0.01;
        let slot =
            retention_cone(&ids, 1, &outcomes, &variances, &sq_dists, MAX_HISTORY, 1.0).unwrap();
        assert_eq!(slot, 5);
    }
}
