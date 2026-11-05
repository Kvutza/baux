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
#[cfg_attr(not(cabi_parity), derive(deser::Serialize, deser::Deserialize))]
#[cfg_attr(
    not(cabi_parity),
    deser(
        tag = "kind",
        rename_all = "kebab-case",
        rename_all_fields = "kebab-case"
    )
)]
pub enum DistanceMetric {
    #[default]
    Euclidean,
    Conformal {
        local_k: usize,
    },
    Hyperbolic {
        local_k: usize,
    },
    Wasserstein {
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

/// Conformal local-scale distance normalization (`sq_dist / (scale_a *
/// scale_b)`).
#[inline(always)]
pub fn conformal_delta(sq_dist: f64, scale_a: f64, scale_b: f64) -> f64 {
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
/// `0.5 * conformal_delta(sq_dist, scale_a, scale_b) + scale_penalty(scale_a,
/// scale_b)`.
#[inline(always)]
pub fn halfspace_delta(sq_dist: f64, scale_a: f64, scale_b: f64) -> f64 {
    0.5 * conformal_delta(sq_dist, scale_a, scale_b) + scale_penalty(scale_a, scale_b)
}

/// Entropic Wasserstein metric on local observation scales.
/// Combines scale-normalized distance with 1D Wasserstein-2 transport cost
/// `|scale_a - scale_b|`.
#[inline(always)]
pub fn wasserstein_delta(sq_dist: f64, scale_a: f64, scale_b: f64) -> f64 {
    let diff = (scale_a - scale_b).abs();
    let norm = (scale_a * scale_b).max(1.0e-12).sqrt();
    sq_dist.max(0.0) / norm + diff * diff
}

/// Apply the configured pairwise distance metric transformation to a squared
/// Euclidean distance given two local lengthscales.
#[inline(always)]
pub fn metric_delta(metric: DistanceMetric, sq_dist: f64, scale_a: f64, scale_b: f64) -> f64 {
    match metric {
        DistanceMetric::Euclidean => sq_dist,
        DistanceMetric::Conformal { .. } => conformal_delta(sq_dist, scale_a, scale_b),
        DistanceMetric::Hyperbolic { .. } => halfspace_delta(sq_dist, scale_a, scale_b),
        DistanceMetric::Wasserstein { .. } => wasserstein_delta(sq_dist, scale_a, scale_b),
    }
}

/// Per-row `k`th off-diagonal squared distance (floored at `1e-12`) for an
/// `n x n` distance matrix.
pub fn local_scales(n: usize, k: usize, dist_at: impl Fn(usize, usize) -> f64) -> Vec<f64> {
    (0..n)
        .map(|row| {
            let mut values = (0..n)
                .filter(|&col| col != row)
                .map(|col| dist_at(row, col))
                .filter(|&d| d > 0.0)
                .collect::<Vec<_>>();
            if values.is_empty() {
                1.0e-12
            } else {
                values.sort_by(f64::total_cmp);
                values[k.min(values.len()) - 1].max(1.0e-12)
            }
        })
        .collect()
}

/// Mean positive finite off-diagonal squared distance across an `n x n`
/// distance matrix, floored at `1e-12` (fallback `1.0`).
pub fn characteristic_distance(n: usize, dist_at: impl Fn(usize, usize) -> f64) -> f64 {
    let mut sum = 0.0_f64;
    let mut count = 0usize;
    for row in 0..n {
        for col in 0..n {
            let d = dist_at(row, col);
            if row != col && d.is_finite() && d > 0.0 {
                sum += d;
                count += 1;
            }
        }
    }
    if count > 0 {
        (sum / count as f64).max(1.0e-12)
    } else {
        1.0
    }
}

/// Draw `(epistemic, aleatoric)` variance scale candidates from the canonical
/// log-uniform hyperparameter prior (`10^[-3, 3] / d_scale` and `10^[-6, 0]`).
#[cfg(not(cabi_parity))]
pub fn draw_scales<R: rand::Rng>(
    num_candidates: usize,
    d_scale: f64,
    infer_aleatoric: bool,
    rng: &mut R,
) -> Vec<(f64, f64)> {
    if num_candidates == 0 {
        return Vec::new();
    }
    let d_scale = if d_scale.is_finite() && d_scale > 0.0 {
        d_scale.max(1.0e-12)
    } else {
        1.0
    };
    let epi: Vec<f64> = (0..num_candidates)
        .map(|_| 10.0_f64.powf(rng.gen_range(-3.0..=3.0)) / d_scale)
        .collect();
    let ale: Vec<f64> = if infer_aleatoric {
        (0..num_candidates)
            .map(|_| 10.0_f64.powf(rng.gen_range(-6.0..=0.0)))
            .collect()
    } else {
        vec![0.0; num_candidates]
    };
    epi.into_iter().zip(ale).collect()
}

/// Sample `count` distinct row indices from `0..n` without replacement (or all
/// `0..n` in order when `count >= n`).
#[cfg(not(cabi_parity))]
pub fn sampled_rows<R: rand::Rng>(n: usize, count: usize, rng: &mut R) -> Vec<usize> {
    if count >= n {
        (0..n).collect()
    } else {
        rand::seq::index::sample(rng, n, count)
            .into_iter()
            .collect()
    }
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
        let risk = epistemic * distance + variance_error;
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

        let ot_delta = wasserstein_delta(4.0_f64, 2.0_f64, 2.0_f64);
        assert!((ot_delta - 2.0).abs() < 1.0e-6);
        let ot_diff = wasserstein_delta(4.0_f64, 3.0_f64, 1.0_f64);
        assert!(ot_diff > ot_delta);
    }
}
