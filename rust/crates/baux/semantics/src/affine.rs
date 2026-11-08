//! Affine posterior location-scale group elements (`mu' = a + b * mu`, `se' =
//! c * se` in `Aff(1)^M`), inverse-variance Aitken Weighted Least Squares
//! (WLS), and split conformal scaling.
//!
//! References:
//! - Aitken (1935), *On Least Squares and Linear Combination of Observations*,
//!   Proc. R. Soc. Edinb. 55, 42-48.
//! - Vovk, Gammerman, and Shafer (2005), *Algorithmic Learning in a Random
//!   World*, Springer.
//! - Lei, G'Sell, Rinaldo, Tibshirani, and Wasserman (2018), *Distribution-Free
//!   Predictive Inference for Regression*, JASA 113(523), 1094-1111
//!   (`arXiv:1604.04173`).

/// Immutable affine location-scale transformation across `M` output metrics.
#[derive(Clone, Debug, PartialEq)]
pub struct Affine {
    pub a: Vec<f64>,
    pub b: Vec<f64>,
    pub c: Vec<f64>,
}

fn col_moments(
    mu: &[f64],
    y: &[f64],
    rows: usize,
    cols: usize,
    col: usize,
) -> (f64, f64, f64, f64) {
    let mut s_mu = 0.0;
    let mut s_y = 0.0;
    let mut s_mumu = 0.0;
    let mut s_muy = 0.0;
    for r in 0..rows {
        let m = mu[r * cols + col];
        let t = y[r * cols + col];
        s_mu += m;
        s_y += t;
        s_mumu += m * m;
        s_muy += m * t;
    }
    (s_mu, s_y, s_mumu, s_muy)
}

fn fit_scale(
    mu: &[f64],
    y: &[f64],
    se: &[f64],
    a: &[f64],
    b: &[f64],
    rows: usize,
    cols: usize,
) -> Vec<f64> {
    let mut c = vec![1.0; cols];
    if rows == 0 {
        return c;
    }
    for j in 0..cols {
        let mut z2_sum = 0.0;
        let mut count = 0usize;
        for r in 0..rows {
            let idx = r * cols + j;
            let s = se[idx];
            if s.is_finite() && s > 1.0e-12 {
                let pred = a[j] + b[j] * mu[idx];
                let z = (y[idx] - pred) / s;
                if z.is_finite() {
                    z2_sum += z * z;
                    count += 1;
                }
            }
        }
        if count > 0 {
            let scale = (z2_sum / count as f64).sqrt();
            if scale.is_finite() && scale > 1.0e-6 {
                c[j] = scale;
            }
        }
    }
    c
}

/// Finite-sample split conformal quantile `s_{(ceil((N + 1)(1 - alpha)))}`
/// clamped to `0..N - 1`.
#[core::contracts::ensures(|ret: &Result<f64, String>| match ret { Ok(q) => q.is_finite() && *q >= 0.0, Err(_) => true })]
pub fn conformal_quantile(scores: &[f64], alpha: f64) -> Result<f64, String> {
    if scores.is_empty() || scores.iter().any(|s| !s.is_finite()) {
        return Err("conformal scores must be non-empty and finite".to_string());
    }
    if !alpha.is_finite() || alpha <= 0.0 || alpha >= 1.0 {
        return Err("alpha must be in (0, 1)".to_string());
    }
    let mut sorted = scores.to_vec();
    sorted.sort_by(f64::total_cmp);
    let n = sorted.len();
    let rank = ((n + 1) as f64 * (1.0 - alpha)).ceil() as isize;
    let idx = (rank - 1).clamp(0, (n - 1) as isize) as usize;
    Ok(sorted[idx].max(0.0))
}

fn wls_col(mu: &[f64], y: &[f64], se: &[f64], rows: usize, cols: usize, col: usize) -> (f64, f64) {
    let mut s_w = 0.0;
    let mut s_mu = 0.0;
    let mut s_y = 0.0;
    let mut s_mumu = 0.0;
    let mut s_muy = 0.0;
    for r in 0..rows {
        let idx = r * cols + col;
        let s = se[idx];
        let w = if s.is_finite() && s > 0.0 {
            1.0 / (s * s).max(1.0e-12)
        } else {
            1.0
        };
        let m = mu[idx];
        let t = y[idx];
        s_w += w;
        s_mu += w * m;
        s_y += w * t;
        s_mumu += w * m * m;
        s_muy += w * m * t;
    }
    let det = s_w * s_mumu - s_mu * s_mu;
    let scale = (s_w * s_mumu).abs().max(1.0);
    if det.is_finite() && det.abs() >= 1.0e-18 * scale {
        let aj = (s_mumu * s_y - s_mu * s_muy) / det;
        let bj = (s_w * s_muy - s_mu * s_y) / det;
        if aj.is_finite() && bj.is_finite() {
            return (aj, bj);
        }
    }
    (0.0, 1.0)
}

impl Affine {
    /// Identity affine map (`a = 0`, `b = 1`, `c = 1`).
    pub fn identity(cols: usize) -> Self {
        Self {
            a: vec![0.0; cols],
            b: vec![1.0; cols],
            c: vec![1.0; cols],
        }
    }

    /// Construct from explicit coefficient vectors `(a, b, c)`.
    pub fn new(a: Vec<f64>, b: Vec<f64>, c: Vec<f64>) -> Result<Self, String> {
        if a.len() != b.len() || b.len() != c.len() {
            return Err("Affine coefficient lengths must match".to_string());
        }
        if a.iter()
            .chain(b.iter())
            .chain(c.iter())
            .any(|v| !v.is_finite())
        {
            return Err("Affine coefficients must be finite".to_string());
        }
        Ok(Self { a, b, c })
    }

    /// Fit ordinary-least-squares `(a, b)` and residual uncertainty multiplier
    /// `c`.
    pub fn fit(
        mu: &[f64],
        y: &[f64],
        se: Option<&[f64]>,
        rows: usize,
        cols: usize,
    ) -> Result<Self, String> {
        if mu.len() != rows * cols || y.len() != rows * cols {
            return Err("mu and y shape mismatch".to_string());
        }
        let mut a = vec![0.0; cols];
        let mut b = vec![1.0; cols];
        if rows >= 2 {
            let n = rows as f64;
            for j in 0..cols {
                let (s_mu, s_y, s_mumu, s_muy) = col_moments(mu, y, rows, cols, j);
                let det = n * s_mumu - s_mu * s_mu;
                let scale = (n * s_mumu).abs().max(1.0);
                if det.is_finite() && det.abs() >= 1.0e-18 * scale {
                    let aj = (s_mumu * s_y - s_mu * s_muy) / det;
                    let bj = (n * s_muy - s_mu * s_y) / det;
                    if aj.is_finite() && bj.is_finite() {
                        a[j] = aj;
                        b[j] = bj;
                    }
                }
            }
        }
        let c = match se {
            Some(s) if s.len() == rows * cols => fit_scale(mu, y, s, &a, &b, rows, cols),
            Some(_) => return Err("se shape mismatch".to_string()),
            None => vec![1.0; cols],
        };
        Ok(Self { a, b, c })
    }

    /// Fit inverse-variance Weighted Least Squares (WLS, `w_i = se_i^-2`) `(a,
    /// b)` and standardized residual multiplier `c`.
    pub fn fit_wls(
        mu: &[f64],
        y: &[f64],
        se: &[f64],
        rows: usize,
        cols: usize,
    ) -> Result<Self, String> {
        if mu.len() != rows * cols || y.len() != rows * cols || se.len() != rows * cols {
            return Err("mu, y, and se shape mismatch".to_string());
        }
        let mut a = vec![0.0; cols];
        let mut b = vec![1.0; cols];
        if rows >= 2 {
            for j in 0..cols {
                let (aj, bj) = wls_col(mu, y, se, rows, cols, j);
                a[j] = aj;
                b[j] = bj;
            }
        }
        let c = fit_scale(mu, y, se, &a, &b, rows, cols);
        Ok(Self { a, b, c })
    }

    /// Finite-sample split conformal quantile `s_{(ceil((N + 1)(1 - alpha)))}`.
    pub fn conformal_quantile(scores: &[f64], alpha: f64) -> Result<f64, String> {
        conformal_quantile(scores, alpha)
    }

    /// Compute per-column split conformal residual multipliers `q_{1 - alpha,
    /// j}` on `|y_{r, j} - (a_j + b_j * mu_{r, j})| / max(se_{r, j},
    /// 1e-12)`.
    pub fn conformal_scale(
        &self,
        mu: &[f64],
        y: &[f64],
        se: &[f64],
        rows: usize,
        cols: usize,
        alpha: f64,
    ) -> Result<Vec<f64>, String> {
        if rows == 0 || cols != self.a.len() {
            return Err("invalid rows or cols for conformal_scale".to_string());
        }
        if mu.len() != rows * cols || y.len() != rows * cols || se.len() != rows * cols {
            return Err("mu, y, and se shape mismatch".to_string());
        }
        let mut out = Vec::with_capacity(cols);
        for j in 0..cols {
            let mut scores = Vec::with_capacity(rows);
            for r in 0..rows {
                let idx = r * cols + j;
                let s = se[idx];
                if !s.is_finite() || s < 0.0 {
                    return Err("se must be nonnegative and finite".to_string());
                }
                let pred = self.a[j] + self.b[j] * mu[idx];
                scores.push((y[idx] - pred).abs() / s.max(1.0e-12));
            }
            out.push(conformal_quantile(&scores, alpha)?);
        }
        Ok(out)
    }

    /// Apply `mu' = a + b * mu` to a row-major `(rows, cols)` mean matrix.
    pub fn apply_mu(&self, mu: &[f64], rows: usize) -> Result<Vec<f64>, String> {
        let cols = self.a.len();
        if mu.len() != rows * cols {
            return Err("mu length mismatch".to_string());
        }
        let mut out = vec![0.0; rows * cols];
        for r in 0..rows {
            for j in 0..cols {
                let idx = r * cols + j;
                out[idx] = self.a[j] + self.b[j] * mu[idx];
            }
        }
        Ok(out)
    }

    /// Apply `se' = c * se` to a row-major `(rows, cols)` standard-error
    /// matrix.
    pub fn apply_se(&self, se: &[f64], rows: usize) -> Result<Vec<f64>, String> {
        let cols = self.c.len();
        if se.len() != rows * cols {
            return Err("se length mismatch".to_string());
        }
        let mut out = vec![0.0; rows * cols];
        for r in 0..rows {
            for j in 0..cols {
                let idx = r * cols + j;
                out[idx] = self.c[j] * se[idx];
            }
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn affine_fit() {
        let mu = [1.0, 2.0, 3.0];
        let y = [3.0, 5.0, 7.0];
        let se = [0.5, 0.5, 0.5];
        let cal = Affine::fit(&mu, &y, Some(&se), 3, 1).unwrap();
        assert!((cal.a[0] - 1.0).abs() < 1e-12);
        assert!((cal.b[0] - 2.0).abs() < 1e-12);
        let mapped = cal.apply_mu(&mu, 3).unwrap();
        for (m, target) in mapped.iter().zip(y.iter()) {
            assert!((m - target).abs() < 1e-12);
        }
    }

    #[test]
    fn affine_collinear() {
        let id = Affine::identity(2);
        assert_eq!(id.a, vec![0.0, 0.0]);
        assert_eq!(id.b, vec![1.0, 1.0]);
        assert_eq!(id.c, vec![1.0, 1.0]);
        let mu = [4.0, 1.0, 4.0, 2.0, 4.0, 3.0];
        let y = [9.0, -1.0, 9.0, -3.0, 9.0, -5.0];
        let cal = Affine::fit(&mu, &y, None, 3, 2).unwrap();
        assert_eq!(cal.a[0], 0.0);
        assert_eq!(cal.b[0], 1.0);
        assert!((cal.a[1] - 1.0).abs() < 1.0e-12);
        assert!((cal.b[1] + 2.0).abs() < 1.0e-12);
        assert_eq!(cal.c, vec![1.0, 1.0]);
        let single = Affine::fit(&[2.0], &[5.0], None, 1, 1).unwrap();
        assert_eq!(single.a, vec![0.0]);
        assert_eq!(single.b, vec![1.0]);
    }

    #[test]
    fn affine_scale() {
        let mu = [0.0, 0.0, 1.0, 1.0];
        let y = [1.5, -1.5, 2.5, -0.5];
        let se = [0.5, 0.5, 0.5, 0.5];
        let cal = Affine::fit(&mu, &y, Some(&se), 4, 1).unwrap();
        assert!((cal.a[0] - 0.0).abs() < 1.0e-12);
        assert!((cal.b[0] - 1.0).abs() < 1.0e-12);
        assert!((cal.c[0] - 3.0).abs() < 1.0e-12);
        let scaled = cal.apply_se(&se, 4).unwrap();
        for s in scaled {
            assert!((s - 1.5).abs() < 1.0e-12);
        }
    }

    #[test]
    fn affine_errors() {
        assert!(Affine::new(vec![0.0], vec![1.0, 2.0], vec![1.0]).is_err());
        assert!(Affine::new(vec![f64::NAN], vec![1.0], vec![1.0]).is_err());
        assert!(Affine::new(vec![0.0], vec![f64::INFINITY], vec![1.0]).is_err());
        assert!(Affine::fit(&[1.0, 2.0], &[1.0], None, 2, 1).is_err());
        assert!(Affine::fit(&[1.0, 2.0], &[1.0, 2.0], Some(&[0.5]), 2, 1).is_err());
        let cal = Affine::new(vec![1.0], vec![2.0], vec![3.0]).unwrap();
        assert!(cal.apply_mu(&[1.0, 2.0], 1).is_err());
        assert!(cal.apply_se(&[1.0, 2.0], 1).is_err());
    }

    #[test]
    fn wls_conformal() {
        let mu = [0.0, 1.0, 2.0, 2.0];
        let y = [1.0, 3.0, 5.0, 50.0];
        let se = [0.01, 0.01, 0.01, 100.0];
        let wls = Affine::fit_wls(&mu, &y, &se, 4, 1).unwrap();
        assert!((wls.a[0] - 1.0).abs() < 1.0e-4);
        assert!((wls.b[0] - 2.0).abs() < 1.0e-4);

        let q = conformal_quantile(&[1.0, 2.0, 3.0, 4.0], 0.2).unwrap();
        assert!((q - 4.0).abs() < 1.0e-12);
        let cs = wls
            .conformal_scale(
                &[0.0, 1.0, 2.0],
                &[1.0, 3.0, 5.02],
                &[0.01, 0.01, 0.01],
                3,
                1,
                0.25,
            )
            .unwrap();
        assert!((cs[0] - 2.0).abs() < 1.0e-2);
    }
}
