//! Portable SIMD (`f64x4`) distance kernels and `ConstParamTy` metric
//! specialization.

use core::marker::ConstParamTy;
use core::simd::f64x4;
use core::simd::num::SimdFloat;

use super::{
    DistanceMetric, MetricBound, conformal_delta, halfspace_delta, rank_pool, refine_bound,
};

/// Compile-time metric tag for `adt_const_params` specialization.
#[derive(Clone, Copy, Debug, PartialEq, Eq, ConstParamTy)]
pub enum MetricKind {
    Euclidean,
    Conformal,
    Hyperbolic,
    Wasserstein,
}

impl MetricKind {
    /// Convert a runtime [`DistanceMetric`] into its [`MetricKind`] tag.
    #[inline(always)]
    pub const fn from_metric(m: DistanceMetric) -> Self {
        match m {
            DistanceMetric::Euclidean => Self::Euclidean,
            DistanceMetric::Conformal { .. } => Self::Conformal,
            DistanceMetric::Hyperbolic { .. } => Self::Hyperbolic,
            DistanceMetric::Wasserstein { .. } => Self::Wasserstein,
        }
    }
}

/// Compile-time specialized lower bound on `metric_delta` over omitted points.
#[inline(always)]
#[core::contracts::ensures(|lb: &f64| *lb >= 0.0)]
pub fn refine_const<const M: MetricKind>(bound: MetricBound) -> f64 {
    let b64_sq = refine_bound(bound.pool);
    if !b64_sq.is_finite() {
        return f64::INFINITY;
    }
    let sq = bound.s_query.max(1.0e-12);
    let s_min = bound.s_min.max(1.0e-12);
    let s_max = bound.s_max.max(s_min);
    match M {
        MetricKind::Euclidean => b64_sq,
        MetricKind::Conformal => conformal_delta(b64_sq, sq, s_max),
        MetricKind::Hyperbolic => {
            let s_star = (b64_sq + sq * sq).sqrt().clamp(s_min, s_max);
            halfspace_delta(b64_sq, sq, s_star)
        }
        MetricKind::Wasserstein => {
            let diff = sq - sq.clamp(s_min, s_max);
            b64_sq.max(0.0) / (sq * s_max).max(1.0e-12).sqrt() + diff * diff
        }
    }
}

/// Compile-time specialized candidate pool certification under metric `M`.
#[core::contracts::requires(k <= cands.len())]
#[core::contracts::ensures(move |ret: &Option<Vec<(f64, i64)>>| ret.as_ref().is_none_or(|v| v.len() == k))]
pub fn certify_const<const M: MetricKind>(
    cands: &[(f64, i64)],
    k: usize,
    bound: MetricBound,
) -> Option<Vec<(f64, i64)>> {
    if k == 0 {
        return Some(Vec::new());
    }
    let mut valid = rank_pool(cands, true);
    if valid.len() < k {
        return None;
    }
    if valid[k - 1].0 > refine_const::<M>(bound) {
        return None;
    }
    valid.truncate(k);
    Some(valid)
}

/// SIMD-accelerated squared Euclidean distance `sum_j (x_j - y_j)^2` using
/// `f64x4`.
#[inline]
#[core::contracts::ensures(|d: &f64| *d >= 0.0)]
pub fn simd_l2(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len().min(y.len());
    let xc = x[..n].chunks_exact(4);
    let yc = y[..n].chunks_exact(4);
    let (xr, yr) = (xc.remainder(), yc.remainder());
    let mut acc = f64x4::splat(0.0);
    for (a, b) in xc.zip(yc) {
        let d = f64x4::from_slice(a) - f64x4::from_slice(b);
        acc += d * d;
    }
    let tail: f64 = xr
        .iter()
        .zip(yr.iter())
        .map(|(&a, &b)| {
            let d = a - b;
            d * d
        })
        .sum();
    (acc.reduce_sum() + tail).max(0.0)
}

/// SIMD-accelerated scaled squared Euclidean distance
/// `sum_j (x_j / scale_j - y_j / scale_j)^2` using `f64x4`.
#[inline]
#[core::contracts::ensures(|d: &f64| *d >= 0.0)]
pub fn simd_scaled(x: &[f64], y: &[f64], scale: &[f64]) -> f64 {
    let n = x.len().min(y.len()).min(scale.len());
    let xc = x[..n].chunks_exact(4);
    let yc = y[..n].chunks_exact(4);
    let sc = scale[..n].chunks_exact(4);
    let (xr, yr, sr) = (xc.remainder(), yc.remainder(), sc.remainder());
    let mut acc = f64x4::splat(0.0);
    for ((a, b), s) in xc.zip(yc).zip(sc) {
        let vs = f64x4::from_slice(s);
        let d = f64x4::from_slice(a) / vs - f64x4::from_slice(b) / vs;
        acc += d * d;
    }
    let tail: f64 = xr
        .iter()
        .zip(yr.iter())
        .zip(sr.iter())
        .map(|((&a, &b), &s)| {
            let d = a / s - b / s;
            d * d
        })
        .sum();
    (acc.reduce_sum() + tail).max(0.0)
}

/// SIMD-accelerated weighted dot product `sum_j w_j * a_j * b_j` using `f64x4`.
#[inline]
#[core::contracts::requires(a.len() == b.len() && a.len() == w.len())]
pub fn simd_dot(a: &[f64], b: &[f64], w: &[f64]) -> f64 {
    let n = a.len().min(b.len()).min(w.len());
    let ac = a[..n].chunks_exact(4);
    let bc = b[..n].chunks_exact(4);
    let wc = w[..n].chunks_exact(4);
    let (ar, br, wr) = (ac.remainder(), bc.remainder(), wc.remainder());
    let mut acc = f64x4::splat(0.0);
    for ((x, y), g) in ac.zip(bc).zip(wc) {
        acc += f64x4::from_slice(g) * f64x4::from_slice(x) * f64x4::from_slice(y);
    }
    let tail: f64 = ar
        .iter()
        .zip(br.iter())
        .zip(wr.iter())
        .map(|((&x, &y), &g)| g * x * y)
        .sum();
    acc.reduce_sum() + tail
}

#[cfg(test)]
mod tests {
    use super::super::{PoolBound, certify_metric, refine_metric};
    use super::*;

    #[test]
    fn simd_parity() {
        for d in [7usize, 13, 19] {
            let x: Vec<f64> = (0..d).map(|i| (i as f64) * 0.37 - 1.2).collect();
            let y: Vec<f64> = (0..d).map(|i| (i as f64) * -0.23 + 0.8).collect();
            let s: Vec<f64> = (0..d).map(|i| 0.5 + (i as f64) * 0.15).collect();
            let ref_l2: f64 = x.iter().zip(&y).map(|(a, b)| (a - b).powi(2)).sum();
            let ref_sl2: f64 = x
                .iter()
                .zip(&y)
                .zip(&s)
                .map(|((a, b), sc)| (a / sc - b / sc).powi(2))
                .sum();
            let ref_dot: f64 = x.iter().zip(&y).zip(&s).map(|((a, b), w)| w * a * b).sum();
            assert!((simd_l2(&x, &y) - ref_l2).abs() < 1.0e-12);
            assert!((simd_scaled(&x, &y, &s) - ref_sl2).abs() < 1.0e-12);
            assert!((simd_dot(&x, &y, &s) - ref_dot).abs() < 1.0e-12);
        }
    }

    #[test]
    fn const_certify() {
        let mb = MetricBound {
            pool: PoolBound {
                b32: 4.0,
                eps_scale: 0.0,
            },
            s_query: 3.0,
            s_min: 1.0,
            s_max: 9.0,
        };
        let m = DistanceMetric::Hyperbolic { local_k: 4 };
        assert_eq!(MetricKind::from_metric(m), MetricKind::Hyperbolic);
        let lb_const = refine_const::<{ MetricKind::Hyperbolic }>(mb);
        let lb_dyn = refine_metric(m, mb);
        assert!((lb_const - lb_dyn).abs() < 1.0e-12);
        let cands = [(lb_const * 0.5, 2), (lb_const, 1), (lb_const + 1.0, 3)];
        assert_eq!(
            certify_const::<{ MetricKind::Hyperbolic }>(&cands, 2, mb),
            certify_metric(m, &cands, 2, mb)
        );
        assert_eq!(
            certify_const::<{ MetricKind::Hyperbolic }>(&cands, 3, mb),
            None
        );
    }
}
