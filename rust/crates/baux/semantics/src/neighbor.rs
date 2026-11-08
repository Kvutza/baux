//! Shared streaming prefix selection for precision-weighted neighbours.
//! The distance coefficient is a fitted bias proxy, not a certified Lipschitz
//! constant. No covariance matrix or gradient estimate is constructed.
//!
//! Non-Euclidean local-scale dissimilarities and certified ANN bounds draw on:
//! - Poincare (1882); Cannon, Floyd, Kenyon, and Parry (1997), *Hyperbolic
//!   Geometry*, MSRI Publ. 31; Nickel and Kiela (2017), *Poincare Embeddings
//!   for Learning Hierarchical Representations*, NeurIPS (`arXiv:1705.08039`);
//!   local-scale half-space dissimilarity (`arXiv:2305.01356`).
//! - Zelnik-Manor and Perona (2004), *Self-Tuning Spectral Clustering*, NeurIPS
//!   17 (conformal local-scale normalization).
//! - Kantorovich (1942); Villani (2009), *Optimal Transport: Old and New*,
//!   Springer; Peyre and Cuturi (2019), *Computational Optimal Transport*
//!   (`arXiv:1803.00567`) (Gaussian/entropic Wasserstein-2 local scale
//!   transport).

/// Resident observation capacity, shared by host state and device scratch.
pub const MAX_HISTORY: usize = 128;
#[path = "neighbor/query.rs"]
pub mod query;
#[path = "neighbor/simd.rs"]
pub mod simd;
pub use simd::{MetricKind, certify_const, refine_const, simd_dot, simd_l2, simd_scaled};

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

/// Pairwise local-scale dissimilarity transformation applied to stored and
/// queried rows.
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

/// Pairwise local-scale dissimilarity geometry alias for [`DistanceMetric`].
pub type Dissimilarity = DistanceMetric;

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

/// Analytical boundary parameters for certifying an `f32` candidate pool in
/// `f64`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PoolBound {
    /// Unrefined `f32` boundary distance `b_32` (radius of the `(k + extra)`-th
    /// candidate).
    pub b32: f64,
    /// Worst-case `f32 -> f64` coordinate/scale quantization drift `eps_scale`.
    pub eps_scale: f64,
}

/// Certified squared-distance lower bound `b_64^2 = max(0, b_32 - eps_scale)^2`
/// on any omitted point outside an `f32` candidate pool.
#[inline(always)]
pub fn refine_bound(bound: PoolBound) -> f64 {
    if !bound.b32.is_finite() {
        return f64::INFINITY;
    }
    let b64 = (bound.b32 - bound.eps_scale.max(0.0)).max(0.0);
    b64 * b64
}

/// Return true if the `k`-th refined `f64` squared distance `d2_k` is certified
/// not to exceed the analytical lower bound of any unvisited point.
#[inline(always)]
pub fn is_certified(d2_k: f64, bound: PoolBound) -> bool {
    d2_k.is_finite() && d2_k <= refine_bound(bound)
}

/// Analytical boundary parameters for certifying a non-Euclidean
/// `DistanceMetric` candidate pool in `f64`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MetricBound {
    pub pool: PoolBound,
    pub s_query: f64,
    pub s_min: f64,
    pub s_max: f64,
}

/// Certified lower bound on `metric_delta(metric, d^2, s_query, s)` over all
/// omitted points (`d^2 >= refine_bound(bound.pool)`) and training scales
/// `s in [s_min, s_max]`.
#[inline(always)]
pub fn refine_metric(metric: DistanceMetric, bound: MetricBound) -> f64 {
    let b64_sq = refine_bound(bound.pool);
    if !b64_sq.is_finite() {
        return f64::INFINITY;
    }
    let sq = bound.s_query.max(1.0e-12);
    let s_min = bound.s_min.max(1.0e-12);
    let s_max = bound.s_max.max(s_min);
    match metric {
        DistanceMetric::Euclidean => b64_sq,
        DistanceMetric::Conformal { .. } => conformal_delta(b64_sq, sq, s_max),
        DistanceMetric::Hyperbolic { .. } => {
            let s_star = (b64_sq + sq * sq).sqrt().clamp(s_min, s_max);
            halfspace_delta(b64_sq, sq, s_star)
        }
        DistanceMetric::Wasserstein { .. } => {
            let diff = sq - sq.clamp(s_min, s_max);
            b64_sq.max(0.0) / (sq * s_max).max(1.0e-12).sqrt() + diff * diff
        }
    }
}

pub(super) fn rank_pool(candidates: &[(f64, i64)], nonneg: bool) -> Vec<(f64, i64)> {
    let mut valid: Vec<(f64, i64)> = candidates
        .iter()
        .copied()
        .filter(|&(d, idx)| idx >= 0 && d.is_finite() && (!nonneg || d >= 0.0))
        .collect();
    valid.sort_by(|a, b| a.1.cmp(&b.1).then(a.0.total_cmp(&b.0)));
    valid.dedup_by_key(|pair| pair.1);
    valid.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
    valid
}

/// Refine a candidate pool `(delta_64, index)` under `metric`, sort finite
/// nonnegative candidates with deterministic index tie-breaking, and return
/// `Some(top_k)` iff `top[k - 1].0 <= refine_metric(metric, bound)`.
pub fn certify_metric(
    metric: DistanceMetric,
    candidates: &[(f64, i64)],
    k: usize,
    bound: MetricBound,
) -> Option<Vec<(f64, i64)>> {
    if k == 0 {
        return Some(Vec::new());
    }
    let mut valid = rank_pool(candidates, true);
    if valid.len() < k {
        return (!bound.pool.b32.is_finite()).then_some(valid);
    }
    if valid[k - 1].0 > refine_metric(metric, bound) {
        return None;
    }
    valid.truncate(k);
    Some(valid)
}

/// Refine a candidate pool `(d2_64, index)` in `f64`, sort with deterministic
/// tie-breaking by index, and return `Some(top_k)` iff the margin bound holds.
pub fn certify_topk(
    candidates: &[(f64, i64)],
    k: usize,
    bound: PoolBound,
) -> Option<Vec<(f64, i64)>> {
    if k == 0 {
        return Some(Vec::new());
    }
    let mut valid = rank_pool(candidates, false);
    if valid.len() < k {
        return (!bound.b32.is_finite()).then_some(valid);
    }
    if !is_certified(valid[k - 1].0, bound) {
        return None;
    }
    valid.truncate(k);
    Some(valid)
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
        let mut sep = Prefix::default();
        sep.push(1, 0.0, 1.0, 0.01);
        sep.push(2, 100.0, 1.0, 0.01);
        assert_eq!(sep.count, 1);
        let horo = halfspace_delta(4.0, 2.0, 2.0);
        assert!((horo - 0.5).abs() < 1.0e-6);
        assert!((horo - halfspace_delta(16.0, 4.0, 4.0)).abs() < 1.0e-6);
        let ot = wasserstein_delta(4.0, 2.0, 2.0);
        assert!((ot - 2.0).abs() < 1.0e-6 && wasserstein_delta(4.0, 3.0, 1.0) > ot);
    }

    #[test]
    fn pool_bound() {
        let bound = PoolBound {
            b32: 5.0,
            eps_scale: 0.5,
        };
        assert!((refine_bound(bound) - 20.25).abs() < 1.0e-12);
        assert!(is_certified(20.25, bound) && !is_certified(20.26, bound));
        let clamped = PoolBound {
            b32: 0.2,
            eps_scale: 1.0,
        };
        assert!(refine_bound(clamped) == 0.0 && is_certified(0.0, clamped));
        let inf = PoolBound {
            b32: f64::INFINITY,
            eps_scale: 0.5,
        };
        assert!(refine_bound(inf).is_infinite() && is_certified(1.0e12, inf));
    }

    #[test]
    fn certify_pass() {
        let pool = [(4.0, 3), (1.0, 1), (1.0, 2), (1.0, 1), (9.0, 4)];
        let pass = PoolBound {
            b32: 4.0,
            eps_scale: 0.5,
        };
        assert_eq!(certify_topk(&pool, 0, pass), Some(Vec::new()));
        assert_eq!(certify_topk(&pool, 2, pass), Some(vec![(1.0, 1), (1.0, 2)]));
        let fail = PoolBound {
            b32: 2.0,
            eps_scale: 0.5,
        };
        assert_eq!(certify_topk(&pool, 3, fail), None);
        assert_eq!(certify_topk(&pool, 10, pass), None);
    }

    #[test]
    fn metric_bounds() {
        let pool = PoolBound {
            b32: 4.0,
            eps_scale: 0.0,
        };
        let mb = MetricBound {
            pool,
            s_query: 3.0,
            s_min: 1.0,
            s_max: 9.0,
        };
        for m in [
            DistanceMetric::Euclidean,
            DistanceMetric::Conformal { local_k: 4 },
            DistanceMetric::Hyperbolic { local_k: 4 },
            DistanceMetric::Wasserstein { local_k: 4 },
        ] {
            let lb = refine_metric(m, mb);
            for s in [1.0, 3.0, 9.0] {
                assert!(lb <= metric_delta(m, 16.0, 3.0, s) + 1.0e-12);
            }
            let cands = [(lb * 0.5, 2), (lb, 1), (lb + 1.0, 3), (-1.0, 0)];
            assert!(certify_metric(m, &cands, 2, mb).is_some());
            assert_eq!(certify_metric(m, &cands, 3, mb), None);
        }
    }
}
