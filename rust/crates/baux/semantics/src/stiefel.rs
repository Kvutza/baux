//! Empirical-Bayes multi-output factor contraction (`P_w = L diag(w_N) L^T`)
//! and subspace-targeted James-Stein contraction on the Stiefel manifold
//! `V_r(R^M)`.
//!
//! References:
//! - Stein (1956), *Inadmissibility of the Usual Estimator for the Mean of a
//!   Multivariate Normal Distribution*, Proc. Third Berkeley Symp. 1, 197-206.
//! - James and Stein (1961), *Estimation with Quadratic Loss*, Proc. Fourth
//!   Berkeley Symp. 1, 361-379.
//! - Efron and Morris (1973), *Stein's Estimation Rule and Its Competitors - An
//!   Empirical Bayes Approach*, JASA 68(341), 117-130.
//! - Higham (1986), *Computing the Polar Decomposition - with Applications*,
//!   SIAM J. Sci. Stat. Comput. 7(4), 1160-1174.

const NORM_TOL: f64 = 1.0e-6;
const PRIOR_PSEUDO: f64 = 12.0;

/// Immutable orthonormal subspace frame `L in V_r(R^M)` on the Stiefel
/// manifold.
#[derive(Clone, Debug, PartialEq)]
pub struct Stiefel {
    loadings: Vec<f64>,
    proj: Vec<f64>,
    p1: Vec<f64>,
    rows: usize,
    cols: usize,
    count: usize,
}

fn check_cols(loadings: &[f64], rows: usize, cols: usize) -> Result<(), String> {
    if rows == 0 || cols == 0 {
        return Err("Stiefel loadings must be non-empty".to_string());
    }
    if loadings.len() != rows * cols {
        return Err(format!(
            "loadings length {} does not match ({rows}, {cols})",
            loadings.len()
        ));
    }
    if loadings.iter().any(|v| !v.is_finite()) {
        return Err("loadings must be finite".to_string());
    }
    for f in 0..cols {
        let norm2: f64 = (0..rows)
            .map(|m| {
                let v = loadings[m * cols + f];
                v * v
            })
            .sum();
        if (norm2.sqrt() - 1.0).abs() > NORM_TOL {
            return Err(format!("factor column {f} must have unit norm"));
        }
    }
    Ok(())
}

fn build_proj(loadings: &[f64], rows: usize, cols: usize) -> (Vec<f64>, Vec<f64>) {
    let mut proj = vec![0.0; rows * rows];
    let mut p1 = vec![0.0; rows];
    for i in 0..rows {
        let mut row_sum = 0.0;
        for j in 0..rows {
            let mut dot = 0.0;
            for f in 0..cols {
                dot += loadings[i * cols + f] * loadings[j * cols + f];
            }
            proj[i * rows + j] = dot;
            row_sum += dot;
        }
        p1[i] = row_sum;
    }
    (proj, p1)
}

/// Optimal Empirical-Bayes James-Stein orthogonal retention factor
/// `alpha* = signal_sq / (signal_sq + noise_var)` in `[0, 1]`.
#[inline(always)]
#[core::contracts::ensures(|a: &f64| *a >= 0.0 && *a <= 1.0)]
pub fn stein_alpha(signal_sq: f64, noise_var: f64) -> f64 {
    if !signal_sq.is_finite() || !noise_var.is_finite() || noise_var <= 0.0 {
        return 1.0;
    }
    let sig = signal_sq.max(0.0);
    let denom = sig + noise_var;
    if denom <= 0.0 {
        1.0
    } else {
        (sig / denom).clamp(0.0, 1.0)
    }
}

fn stiefel_mgs(factors: &[f64], rows: usize, cols: usize) -> Result<Vec<f64>, String> {
    if rows == 0 || cols == 0 || cols > rows {
        return Err("invalid Stiefel dimensions".to_string());
    }
    if factors.len() != rows * cols || factors.iter().any(|v| !v.is_finite()) {
        return Err("factors must be finite with shape (rows, cols)".to_string());
    }
    let mut q = factors.to_vec();
    for f in 0..cols {
        for prev in 0..f {
            let dot: f64 = (0..rows)
                .map(|m| q[m * cols + prev] * q[m * cols + f])
                .sum();
            for m in 0..rows {
                q[m * cols + f] -= dot * q[m * cols + prev];
            }
        }
        let norm = (0..rows)
            .map(|m| q[m * cols + f].powi(2))
            .sum::<f64>()
            .sqrt();
        if !norm.is_finite() || norm <= 1.0e-12 {
            return Err(format!("factor column {f} is rank-deficient"));
        }
        for m in 0..rows {
            q[m * cols + f] /= norm;
        }
    }
    Ok(q)
}

fn stein_row(proj: &[f64], m_dim: usize, mu_r: &[f64], se_r: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let mut p_r = vec![0.0; m_dim];
    let mut sig_sq = 0.0;
    let mut noise_var = 0.0;
    for i in 0..m_dim {
        let p_i: f64 = (0..m_dim).map(|j| proj[i * m_dim + j] * mu_r[j]).sum();
        p_r[i] = p_i;
        sig_sq += (mu_r[i] - p_i).powi(2);
        noise_var += (1.0 - proj[i * m_dim + i]).max(0.0) * se_r[i].powi(2);
    }
    let alpha = stein_alpha(sig_sq, noise_var);
    let mu_out = (0..m_dim)
        .map(|i| p_r[i] + alpha * (mu_r[i] - p_r[i]))
        .collect();
    let se_out = (0..m_dim)
        .map(|i| {
            (0..m_dim)
                .map(|j| {
                    let diag = if i == j { alpha } else { 0.0 };
                    let t_ij = (1.0 - alpha) * proj[i * m_dim + j] + diag;
                    (t_ij * se_r[j]).powi(2)
                })
                .sum::<f64>()
                .sqrt()
        })
        .collect();
    (mu_out, se_out)
}

impl Stiefel {
    /// Construct a validated Stiefel frame from row-major `(rows, cols)`
    /// orthonormal loadings `L`.
    pub fn new(loadings: Vec<f64>, rows: usize, cols: usize) -> Result<Self, String> {
        check_cols(&loadings, rows, cols)?;
        let (proj, p1) = build_proj(&loadings, rows, cols);
        Ok(Self {
            loadings,
            proj,
            p1,
            rows,
            cols,
            count: 0,
        })
    }

    /// Orthonormalize the columns of a row-major `(rows, cols)` factor matrix
    /// onto the Stiefel manifold `V_cols(R^rows)` via modified Gram-Schmidt.
    pub fn from_factors(factors: &[f64], rows: usize, cols: usize) -> Result<Self, String> {
        let q = stiefel_mgs(factors, rows, cols)?;
        Self::new(q, rows, cols)
    }

    /// Optimal Empirical-Bayes James-Stein orthogonal retention factor.
    #[inline(always)]
    pub fn stein_alpha(signal_sq: f64, noise_var: f64) -> f64 {
        stein_alpha(signal_sq, noise_var)
    }

    /// Apply subspace-targeted Empirical-Bayes James-Stein contraction to
    /// row-major `(rows, self.rows)` mean `mu` and standard-error `se`
    /// matrices.
    pub fn transform_stein(
        &self,
        mu: &[f64],
        se: &[f64],
        rows: usize,
    ) -> Result<(Vec<f64>, Vec<f64>), String> {
        let m_dim = self.rows;
        if mu.len() != rows * m_dim || se.len() != rows * m_dim {
            return Err("mu and se shape mismatch".to_string());
        }
        if mu.iter().any(|v| !v.is_finite()) || se.iter().any(|&s| !s.is_finite() || s < 0.0) {
            return Err("mu must be finite and se must be nonnegative finite".to_string());
        }
        let mut mu_out = Vec::with_capacity(rows * m_dim);
        let mut se_out = Vec::with_capacity(rows * m_dim);
        for r in 0..rows {
            let (m_r, s_r) = stein_row(
                &self.proj,
                m_dim,
                &mu[r * m_dim..(r + 1) * m_dim],
                &se[r * m_dim..(r + 1) * m_dim],
            );
            mu_out.extend(m_r);
            se_out.extend(s_r);
        }
        Ok((mu_out, se_out))
    }

    #[inline(always)]
    pub fn rows(&self) -> usize {
        self.rows
    }

    #[inline(always)]
    pub fn cols(&self) -> usize {
        self.cols
    }

    #[inline(always)]
    pub fn count(&self) -> usize {
        self.count
    }

    #[inline(always)]
    pub fn loadings(&self) -> &[f64] {
        &self.loadings
    }

    #[inline(always)]
    pub fn projector(&self) -> &[f64] {
        &self.proj
    }

    /// Return a new `Stiefel` frame updated to observation count `count`.
    #[inline(always)]
    pub fn with_count(&self, count: usize) -> Self {
        let mut next = self.clone();
        next.count = count;
        next
    }

    /// Empirical-Bayes contraction weight `lambda_N = N / (N + 12)`.
    #[inline(always)]
    pub fn lambda(&self) -> f64 {
        let n = self.count as f64;
        n / (n + PRIOR_PSEUDO)
    }

    /// Blended per-metric weight vector `w_N = (1 - lambda_N) * 1 + lambda_N *
    /// (L L^T 1)`.
    pub fn weights(&self) -> Vec<f64> {
        let lam = self.lambda();
        self.p1.iter().map(|&p| (1.0 - lam) + lam * p).collect()
    }

    /// Apply the blended operator `Y * (L^T diag(w_N) L)` to row-major `(batch,
    /// rows)` predictions.
    pub fn project(&self, y: &[f64], batch: usize) -> Result<Vec<f64>, String> {
        if y.len() != batch * self.rows {
            return Err(format!(
                "input length {} does not match ({batch}, {})",
                y.len(),
                self.rows
            ));
        }
        let w = self.weights();
        let mut out = vec![0.0; batch * self.rows];
        for b in 0..batch {
            let row = &y[b * self.rows..(b + 1) * self.rows];
            for m in 0..self.rows {
                let mut acc = 0.0;
                for j in 0..self.rows {
                    acc += row[j] * w[j] * self.proj[j * self.rows + m];
                }
                out[b * self.rows + m] = acc;
            }
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stiefel_project() {
        let s2 = 0.5_f64.sqrt();
        let sh = Stiefel::new(vec![s2, s2], 2, 1).unwrap().with_count(12);
        assert!((sh.lambda() - 0.5).abs() < 1e-12);
        let w = sh.weights();
        assert_eq!(w.len(), 2);
        let out = sh.project(&[2.0, 4.0], 1).unwrap();
        assert!((out[0] - 3.0).abs() < 1e-12);
        assert!((out[1] - 3.0).abs() < 1e-12);
    }

    #[test]
    fn stiefel_errors() {
        assert!(Stiefel::new(vec![], 0, 1).is_err());
        assert!(Stiefel::new(vec![], 1, 0).is_err());
        assert!(Stiefel::new(vec![1.0, 0.0], 3, 1).is_err());
        assert!(Stiefel::new(vec![f64::NAN, 0.0], 2, 1).is_err());
        assert!(Stiefel::new(vec![f64::INFINITY, 0.0], 2, 1).is_err());
        assert!(Stiefel::new(vec![0.5, 0.5], 2, 1).is_err());
        let sh = Stiefel::new(vec![1.0, 0.0], 2, 1).unwrap();
        assert!(sh.project(&[1.0, 2.0, 3.0], 1).is_err());
    }

    #[test]
    fn stiefel_limits() {
        let sh0 = Stiefel::new(vec![1.0, 0.0], 2, 1).unwrap();
        assert_eq!(sh0.rows(), 2);
        assert_eq!(sh0.cols(), 1);
        assert_eq!(sh0.count(), 0);
        assert_eq!(sh0.loadings(), &[1.0, 0.0]);
        assert_eq!(sh0.projector(), &[1.0, 0.0, 0.0, 0.0]);
        assert_eq!(sh0.lambda(), 0.0);
        assert_eq!(sh0.weights(), vec![1.0, 1.0]);
        assert_eq!(sh0.project(&[3.0, 5.0], 1).unwrap(), vec![3.0, 0.0]);
        let sh_inf = sh0.with_count(12_000_000);
        assert_eq!(sh0.count(), 0);
        assert_eq!(sh_inf.count(), 12_000_000);
        assert!((sh_inf.lambda() - 1.0).abs() < 1.0e-6);
        let w_inf = sh_inf.weights();
        assert!((w_inf[0] - 1.0).abs() < 1.0e-6);
        assert!(w_inf[1].abs() < 1.0e-6);
    }

    #[test]
    fn stiefel_multicol() {
        let s2 = 0.5_f64.sqrt();
        let loadings = vec![s2, 0.0, s2, 0.0, 0.0, 1.0];
        let sh = Stiefel::new(loadings, 3, 2).unwrap().with_count(12);
        assert_eq!(sh.rows(), 3);
        assert_eq!(sh.cols(), 2);
        let y = [2.0, 4.0, 7.0, 6.0, 2.0, -3.0];
        let out = sh.project(&y, 2).unwrap();
        assert_eq!(out.len(), 6);
        assert!((out[0] - 3.0).abs() < 1.0e-12);
        assert!((out[1] - 3.0).abs() < 1.0e-12);
        assert!((out[2] - 7.0).abs() < 1.0e-12);
        assert!((out[3] - 4.0).abs() < 1.0e-12);
        assert!((out[4] - 4.0).abs() < 1.0e-12);
        assert!((out[5] + 3.0).abs() < 1.0e-12);
    }

    #[test]
    fn stiefel_stein() {
        let sh = Stiefel::from_factors(&[3.0, 1.0, 4.0, 2.0, 0.0, 5.0], 3, 2).unwrap();
        assert_eq!(sh.rows(), 3);
        assert_eq!(sh.cols(), 2);
        let q = sh.loadings();
        let dot01: f64 = (0..3).map(|m| q[m * 2] * q[m * 2 + 1]).sum();
        assert!(dot01.abs() < 1.0e-12);
        assert!(Stiefel::from_factors(&[1.0, 2.0, 2.0, 4.0], 2, 2).is_err());

        assert_eq!(stein_alpha(0.0, 0.0), 1.0);
        assert_eq!(Stiefel::stein_alpha(0.0, 1.0), 0.0);
        assert!((stein_alpha(3.0, 1.0) - 0.75).abs() < 1.0e-12);

        let sh1 = Stiefel::from_factors(&[1.0, 1.0, 1.0], 3, 1).unwrap();
        let truth = [2.0, 2.0, 2.0];
        let noisy_mu = [2.6, 1.4, 2.0];
        let se = [1.0, 1.0, 1.0];
        let (mu_s, se_s) = sh1.transform_stein(&noisy_mu, &se, 1).unwrap();
        let raw_mse: f64 = noisy_mu
            .iter()
            .zip(truth)
            .map(|(a, b)| (a - b).powi(2))
            .sum();
        let shr_mse: f64 = mu_s.iter().zip(truth).map(|(a, b)| (a - b).powi(2)).sum();
        assert!(shr_mse < raw_mse);
        assert!(se_s.iter().all(|&s| s < 1.0));
    }
}
