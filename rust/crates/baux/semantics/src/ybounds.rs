//! `#![no_std]` scalar and slice output-space boundary warps (`y_bounds`).
//!
//! Maps bounded natural outcomes `y in (lo, hi)` bijectively onto unconstrained
//! `z in R` via log / logit charts, with exact inverse Jacobian `|dy/dz|`
//! standard-error scaling and open-interval boundary clamping over generic
//! [`Real`] scalars (`f32` and `f64`).

use super::real::Real;

/// Check whether `(lo, hi)` is a valid `(T::NEG_INFINITY | finite,
/// T::INFINITY | finite)` interval with `lo < hi` when both endpoints are
/// finite.
#[inline(always)]
pub fn valid_bound<T: Real>(lo: T, hi: T) -> bool {
    if lo.is_nan() || hi.is_nan() {
        return false;
    }
    if (!lo.is_finite() && lo != T::NEG_INFINITY) || (!hi.is_finite() && hi != T::INFINITY) {
        return false;
    }
    !(lo.is_finite() && hi.is_finite() && lo >= hi)
}

/// Return `true` iff every metric row is unbounded `(-inf, +inf)`.
#[inline]
pub fn is_identity<T: Real>(bounds: &[[T; 2]]) -> bool {
    bounds
        .iter()
        .all(|b| b[0] == T::NEG_INFINITY && b[1] == T::INFINITY)
}

/// Forward output warp `y -> z` on `(lo, hi)`.
#[inline(always)]
pub fn warp_scalar<T: Real>(y: T, lo: T, hi: T) -> T {
    match (lo.is_finite(), hi.is_finite()) {
        (false, false) => y,
        (true, false) => (y - lo).ln(),
        (false, true) => -(hi - y).ln(),
        (true, true) => {
            let u = (y - lo) / (hi - lo);
            (u / (T::ONE - u)).ln()
        }
    }
}

/// Checked forward output warp returning `None` if `y` is non-finite or lies on
/// or outside open bounds `(lo, hi)`.
#[inline(always)]
pub fn warp_checked<T: Real>(y: T, lo: T, hi: T) -> Option<T> {
    if !y.is_finite() || !valid_bound(lo, hi) {
        return None;
    }
    if (lo.is_finite() && y <= lo) || (hi.is_finite() && y >= hi) {
        return None;
    }
    Some(warp_scalar(y, lo, hi))
}

/// Inverse output warp `z -> y` into `(lo, hi)`.
#[inline(always)]
pub fn inv_scalar<T: Real>(z: T, lo: T, hi: T) -> T {
    match (lo.is_finite(), hi.is_finite()) {
        (false, false) => z,
        (true, false) => lo + z.exp(),
        (false, true) => hi - (-z).exp(),
        (true, true) => {
            let s = T::ONE / (T::ONE + (-z).exp());
            lo + (hi - lo) * s
        }
    }
}

/// Absolute inverse Jacobian `|dy / dz|` at latent value `z`.
#[inline(always)]
pub fn jac_inv<T: Real>(z: T, lo: T, hi: T) -> T {
    match (lo.is_finite(), hi.is_finite()) {
        (false, false) => T::ONE,
        (true, false) => z.exp(),
        (false, true) => (-z).exp(),
        (true, true) => {
            let s = T::ONE / (T::ONE + (-z).exp());
            (hi - lo) * s * (T::ONE - s)
        }
    }
    .abs()
}

/// Project natural mean `mu` strictly inside `(lo, hi)`.
#[inline(always)]
pub fn project_mu<T: Real>(mu: T, lo: T, hi: T) -> T {
    match (lo.is_finite(), hi.is_finite()) {
        (false, false) => mu,
        (true, false) => mu.max(lo + T::BOUND_EPS * lo.abs().max(T::ONE)),
        (false, true) => mu.min(hi - T::BOUND_EPS * hi.abs().max(T::ONE)),
        (true, true) => {
            let eps = T::BOUND_EPS * (hi - lo);
            mu.clamp(lo + eps, hi - eps)
        }
    }
}

/// Delta-method observation variance warp `(dz / dy)^2 * yvar` into latent `z`
/// space, returning `None` if out of bounds or non-finite.
#[inline(always)]
pub fn warp_var<T: Real>(y: T, yvar: T, lo: T, hi: T) -> Option<T> {
    if !y.is_finite() || !yvar.is_finite() || yvar < T::ZERO || !valid_bound(lo, hi) {
        return None;
    }
    if (lo.is_finite() && y <= lo) || (hi.is_finite() && y >= hi) {
        return None;
    }
    let jac = match (lo.is_finite(), hi.is_finite()) {
        (false, false) => T::ONE,
        (true, false) => T::ONE / (y - lo),
        (false, true) => T::ONE / (hi - y),
        (true, true) => {
            let u = (y - lo) / (hi - lo);
            T::ONE / ((hi - lo) * u * (T::ONE - u))
        }
    };
    let v = jac * jac * yvar;
    v.is_finite().then_some(v)
}

/// Delta-method latent-to-natural variance `(dy / dz)^2 * zvar`.
#[inline(always)]
pub fn natural_var<T: Real>(z: T, zvar: T, lo: T, hi: T) -> T {
    let jac = jac_inv(z, lo, hi);
    zvar * jac * jac
}

/// Naturalize a single `(z_mu, z_se, z_epi, z_ale)` posterior tuple into
/// natural space `(mu, se, se_epi, se_ale)`.
#[inline(always)]
pub fn natural_point<T: Real>(z_mu: T, z_se: T, z_epi: T, z_ale: T, lo: T, hi: T) -> (T, T, T, T) {
    if !lo.is_finite() && !hi.is_finite() {
        return (z_mu, z_se, z_epi, z_ale);
    }
    let jac = jac_inv(z_mu, lo, hi);
    let mu = project_mu(inv_scalar(z_mu, lo, hi), lo, hi);
    (mu, z_se * jac, z_epi * jac, z_ale * jac)
}

/// In-place naturalization over row-major `(rows, cols)` slices.
#[inline]
pub fn natural_slice<T: Real>(
    mu: &mut [T],
    se: &mut [T],
    se_epi: &mut [T],
    se_ale: &mut [T],
    bounds: &[[T; 2]],
) {
    let cols = bounds.len();
    if cols == 0 || is_identity(bounds) {
        return;
    }
    let total = mu.len().min(se.len()).min(se_epi.len()).min(se_ale.len());
    let rows = total / cols;
    for r in 0..rows {
        for (j, &[lo, hi]) in bounds.iter().enumerate() {
            let idx = r * cols + j;
            let (m, s, e, a) = natural_point(mu[idx], se[idx], se_epi[idx], se_ale[idx], lo, hi);
            mu[idx] = m;
            se[idx] = s;
            se_epi[idx] = e;
            se_ale[idx] = a;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_bounds() {
        for (lo, hi, y) in [
            (f64::NEG_INFINITY, f64::INFINITY, -3.5_f64),
            (0.0, f64::INFINITY, 2.25),
            (f64::NEG_INFINITY, 10.0, 4.5),
            (0.0, 1.0, 0.25),
            (-2.0, 5.0, 1.3),
        ] {
            let z = warp_checked(y, lo, hi).unwrap();
            let back = inv_scalar(z, lo, hi);
            assert!((back - y).abs() < 1e-12);
            let wv = warp_var(y, 0.04, lo, hi).unwrap();
            let nv = natural_var(z, wv, lo, hi);
            assert!((nv - 0.04).abs() < 1e-12);
        }
        let z32 = warp_checked(0.25_f32, 0.0_f32, 1.0_f32).unwrap();
        assert!((inv_scalar(z32, 0.0_f32, 1.0_f32) - 0.25_f32).abs() < 1e-6);
        assert!(warp_checked(0.0, 0.0, 1.0).is_none());
        assert!(warp_checked(1.0, 0.0, 1.0).is_none());
        assert!(!valid_bound(2.0, 1.0));
    }
}
