//! Extensible post-hoc posterior calibration (`Calibrator`, `ScaleCalibrator`,
//! `AffineCalibrator`).

use deser::{Deserialize, Serialize};
use ndarray::{Array1, Array2, Array3, ArrayView1, ArrayView2, Ix2, Ix3};
use rand::Rng;
use rand::seq::index::sample;

use crate::error::ENNError;
use crate::model::ENN;
use crate::params::{ENNNormal, ENNParams, PosteriorFlags};
use crate::surrogate::SurrogatePrediction;
use crate::traits::PosteriorComputation;
use crate::y_bounds::OutputTransform;

const MIN_OBS: usize = 4;
const EPS_VAR: f64 = 1e-12;
const HUBER_TAU: f64 = 16.0;
const RIDGE_REL: f64 = 1e-8;
const SLOPE_MIN: f64 = 0.25;
const SLOPE_MAX: f64 = 4.0;
const SCALE_MIN: f64 = 0.25;
const SCALE_MAX: f64 = 16.0;

/// Calibration law selection for post-hoc posterior adjustment.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[deser(rename_all = "snake_case")]
pub enum CalibrationMode {
    #[default]
    None,
    Scale,
    Affine,
}

/// General post-hoc calibration law for predictive moments and posterior
/// function draws.
pub trait Calibrator: Send + Sync {
    /// Calibrate a single metric's `(mu, se, se_epi, se_ale)`.
    fn calibrate_scalar(
        &self,
        metric: usize,
        mu: f64,
        se_epi: f64,
        se_ale: f64,
        bounds: Option<&dyn OutputTransform>,
    ) -> (f64, f64, f64, f64);

    /// Calibrate a single posterior draw value given the uncalibrated mean
    /// `mu0`.
    fn calibrate_draw(
        &self,
        metric: usize,
        draw: f64,
        mu0: f64,
        bounds: Option<&dyn OutputTransform>,
    ) -> f64;

    /// Apply calibration to a 2D `(N, M)` or 3D `(P, N, M)` `ENNNormal`
    /// posterior.
    fn calibrate_normal(&self, normal: ENNNormal, bounds: Option<&Array2<f64>>) -> ENNNormal {
        let active_bounds = bounds
            .or(normal.y_bounds.as_ref())
            .filter(|b| !b.is_identity());
        let transform = active_bounds.map(|b| b as &dyn OutputTransform);
        match normal.mu.ndim() {
            2 => {
                let mut mu = normal.mu.into_dimensionality::<Ix2>().unwrap();
                let mut se = normal.se.into_dimensionality::<Ix2>().unwrap();
                let mut se_epi = normal.se_epi.into_dimensionality::<Ix2>().unwrap();
                let mut se_ale = normal.se_ale.into_dimensionality::<Ix2>().unwrap();
                let (n, m_dim) = mu.dim();
                for i in 0..n {
                    for m in 0..m_dim {
                        let (mu_hat, se_hat, epi_hat, ale_hat) = self.calibrate_scalar(
                            m,
                            mu[[i, m]],
                            se_epi[[i, m]],
                            se_ale[[i, m]],
                            transform,
                        );
                        mu[[i, m]] = mu_hat;
                        se[[i, m]] = se_hat;
                        se_epi[[i, m]] = epi_hat;
                        se_ale[[i, m]] = ale_hat;
                    }
                }
                let mut out = ENNNormal::new(
                    mu.into_dyn(),
                    se.into_dyn(),
                    se_epi.into_dyn(),
                    se_ale.into_dyn(),
                    normal.idx,
                );
                if let Some(b) = active_bounds {
                    out = out.with_bounds(b.clone());
                }
                out
            }
            3 => {
                let mut mu = normal.mu.into_dimensionality::<Ix3>().unwrap();
                let mut se = normal.se.into_dimensionality::<Ix3>().unwrap();
                let mut se_epi = normal.se_epi.into_dimensionality::<Ix3>().unwrap();
                let mut se_ale = normal.se_ale.into_dimensionality::<Ix3>().unwrap();
                let (p_dim, n, m_dim) = mu.dim();
                for p in 0..p_dim {
                    for i in 0..n {
                        for m in 0..m_dim {
                            let (mu_hat, se_hat, epi_hat, ale_hat) = self.calibrate_scalar(
                                m,
                                mu[[p, i, m]],
                                se_epi[[p, i, m]],
                                se_ale[[p, i, m]],
                                transform,
                            );
                            mu[[p, i, m]] = mu_hat;
                            se[[p, i, m]] = se_hat;
                            se_epi[[p, i, m]] = epi_hat;
                            se_ale[[p, i, m]] = ale_hat;
                        }
                    }
                }
                let mut out = ENNNormal::new(
                    mu.into_dyn(),
                    se.into_dyn(),
                    se_epi.into_dyn(),
                    se_ale.into_dyn(),
                    normal.idx,
                );
                if let Some(b) = active_bounds {
                    out = out.with_bounds(b.clone());
                }
                out
            }
            _ => normal,
        }
    }

    /// Apply calibration to a `SurrogatePrediction`.
    fn calibrate_prediction(
        &self,
        mut pred: SurrogatePrediction,
        bounds: Option<&Array2<f64>>,
    ) -> SurrogatePrediction {
        let transform = bounds
            .filter(|b| !b.is_identity())
            .map(|b| b as &dyn OutputTransform);
        let (n, m_dim) = pred.mu.dim();
        for i in 0..n {
            for m in 0..m_dim {
                let (mu_hat, se_hat, _, _) =
                    self.calibrate_scalar(m, pred.mu[[i, m]], pred.se[[i, m]], 0.0, transform);
                pred.mu[[i, m]] = mu_hat;
                pred.se[[i, m]] = se_hat;
            }
        }
        pred
    }

    /// Apply calibration in-place to posterior function draws `(S, N, M)` given
    /// uncalibrated `mu0`.
    fn calibrate_draws(
        &self,
        draws: &mut Array3<f64>,
        mu0: &ArrayView2<f64>,
        bounds: Option<&Array2<f64>>,
    ) {
        let transform = bounds
            .filter(|b| !b.is_identity())
            .map(|b| b as &dyn OutputTransform);
        let (s_dim, n, m_dim) = draws.dim();
        for s in 0..s_dim {
            for i in 0..n {
                for m in 0..m_dim {
                    draws[[s, i, m]] =
                        self.calibrate_draw(m, draws[[s, i, m]], mu0[[i, m]], transform);
                }
            }
        }
    }
}

/// Pure uncertainty scale calibrator: $\hat\mu = \mu$, $\hat\sigma = c_m
/// \sigma$.
#[derive(Debug, Clone, PartialEq)]
pub struct ScaleCalibrator {
    pub c: Array1<f64>,
}

impl ScaleCalibrator {
    pub fn identity(num_metrics: usize) -> Self {
        Self {
            c: Array1::ones(num_metrics),
        }
    }

    pub fn fit(
        mu_pred: ArrayView2<f64>,
        se_pred: ArrayView2<f64>,
        y: ArrayView2<f64>,
    ) -> Result<Self, ENNError> {
        validate_shapes(mu_pred, se_pred, y)?;
        let (n, m_dim) = y.dim();
        if n < MIN_OBS {
            return Ok(Self::identity(m_dim));
        }
        let mut c = Array1::ones(m_dim);
        for m in 0..m_dim {
            c[m] = fit_scale(mu_pred.column(m), se_pred.column(m), y.column(m), 1.0, 0.0);
        }
        Ok(Self { c })
    }
}

impl Calibrator for ScaleCalibrator {
    fn calibrate_scalar(
        &self,
        metric: usize,
        mu: f64,
        se_epi: f64,
        se_ale: f64,
        bounds: Option<&dyn OutputTransform>,
    ) -> (f64, f64, f64, f64) {
        let cm = self.c[metric];
        let mu_hat = match bounds {
            Some(b) => b.project_val(metric, mu),
            None => mu,
        };
        let epi_hat = cm * se_epi;
        let ale_hat = cm * se_ale;
        let se_hat = epi_hat.hypot(ale_hat);
        (mu_hat, se_hat, epi_hat, ale_hat)
    }

    fn calibrate_draw(
        &self,
        metric: usize,
        draw: f64,
        mu0: f64,
        bounds: Option<&dyn OutputTransform>,
    ) -> f64 {
        let cm = self.c[metric];
        let val = mu0 + cm * (draw - mu0);
        match bounds {
            Some(b) => b.project_val(metric, val),
            None => val,
        }
    }
}

/// Per-metric affine location-scale calibrator: $\hat\mu = a_m \mu + b_m$,
/// $\hat\sigma = c_m \sigma$.
#[derive(Debug, Clone, PartialEq)]
pub struct AffineCalibrator {
    pub a: Array1<f64>,
    pub b: Array1<f64>,
    pub c: Array1<f64>,
}

impl AffineCalibrator {
    pub fn identity(num_metrics: usize) -> Self {
        Self {
            a: Array1::ones(num_metrics),
            b: Array1::zeros(num_metrics),
            c: Array1::ones(num_metrics),
        }
    }

    pub fn fit(
        mu_pred: ArrayView2<f64>,
        se_pred: ArrayView2<f64>,
        y: ArrayView2<f64>,
    ) -> Result<Self, ENNError> {
        validate_shapes(mu_pred, se_pred, y)?;
        let (n, m_dim) = y.dim();
        if n < MIN_OBS {
            return Ok(Self::identity(m_dim));
        }
        let mut a = Array1::ones(m_dim);
        let mut b = Array1::zeros(m_dim);
        let mut c = Array1::ones(m_dim);
        for m in 0..m_dim {
            let (am, bm, cm) = fit_column(mu_pred.column(m), se_pred.column(m), y.column(m));
            a[m] = am;
            b[m] = bm;
            c[m] = cm;
        }
        Ok(Self { a, b, c })
    }
}

impl Calibrator for AffineCalibrator {
    fn calibrate_scalar(
        &self,
        metric: usize,
        mu: f64,
        se_epi: f64,
        se_ale: f64,
        bounds: Option<&dyn OutputTransform>,
    ) -> (f64, f64, f64, f64) {
        let (am, bm, cm) = (self.a[metric], self.b[metric], self.c[metric]);
        let raw_mu = am * mu + bm;
        let mu_hat = match bounds {
            Some(b) => b.project_val(metric, raw_mu),
            None => raw_mu,
        };
        let epi_hat = cm * se_epi;
        let ale_hat = cm * se_ale;
        let se_hat = epi_hat.hypot(ale_hat);
        (mu_hat, se_hat, epi_hat, ale_hat)
    }

    fn calibrate_draw(
        &self,
        metric: usize,
        draw: f64,
        mu0: f64,
        bounds: Option<&dyn OutputTransform>,
    ) -> f64 {
        let (am, bm, cm) = (self.a[metric], self.b[metric], self.c[metric]);
        let mu_hat = match bounds {
            Some(b) => b.project_val(metric, am * mu0 + bm),
            None => am * mu0 + bm,
        };
        let val = mu_hat + cm * (draw - mu0);
        match bounds {
            Some(b) => b.project_val(metric, val),
            None => val,
        }
    }
}

/// Extensible sum type over built-in posterior calibrators.
#[derive(Debug, Clone, PartialEq)]
pub enum PosteriorCalibrator {
    Identity,
    Scale(ScaleCalibrator),
    Affine(AffineCalibrator),
}

impl Calibrator for PosteriorCalibrator {
    fn calibrate_scalar(
        &self,
        metric: usize,
        mu: f64,
        se_epi: f64,
        se_ale: f64,
        bounds: Option<&dyn OutputTransform>,
    ) -> (f64, f64, f64, f64) {
        match self {
            Self::Identity => (mu, se_epi.hypot(se_ale), se_epi, se_ale),
            Self::Scale(cal) => cal.calibrate_scalar(metric, mu, se_epi, se_ale, bounds),
            Self::Affine(cal) => cal.calibrate_scalar(metric, mu, se_epi, se_ale, bounds),
        }
    }

    fn calibrate_draw(
        &self,
        metric: usize,
        draw: f64,
        mu0: f64,
        bounds: Option<&dyn OutputTransform>,
    ) -> f64 {
        match self {
            Self::Identity => draw,
            Self::Scale(cal) => cal.calibrate_draw(metric, draw, mu0, bounds),
            Self::Affine(cal) => cal.calibrate_draw(metric, draw, mu0, bounds),
        }
    }
}

fn validate_shapes(
    mu_pred: ArrayView2<f64>,
    se_pred: ArrayView2<f64>,
    y: ArrayView2<f64>,
) -> Result<(), ENNError> {
    if mu_pred.dim() != y.dim() || se_pred.dim() != y.dim() {
        return Err(ENNError::InvalidShape {
            expected: y.shape().to_vec(),
            got: mu_pred.shape().to_vec(),
        });
    }
    Ok(())
}

fn fit_scale(mu: ArrayView1<f64>, se: ArrayView1<f64>, y: ArrayView1<f64>, a: f64, b: f64) -> f64 {
    let n = y.len();
    let mut r2_sum = 0.0;
    let mut valid_n = 0usize;
    for i in 0..n {
        let (mui, sei, yi) = (mu[i], se[i], y[i]);
        if !mui.is_finite() || !sei.is_finite() || !yi.is_finite() || sei < 0.0 {
            continue;
        }
        let s = sei.hypot(EPS_VAR.sqrt());
        let ri = ((yi - (a * mui + b)) / s).clamp(-HUBER_TAU, HUBER_TAU);
        r2_sum += ri * ri;
        valid_n += 1;
    }
    if valid_n == 0 {
        return 1.0;
    }
    let c_raw = (r2_sum / (valid_n as f64)).sqrt();
    if c_raw.is_finite() {
        c_raw.clamp(SCALE_MIN, SCALE_MAX)
    } else {
        1.0
    }
}

fn fit_column(mu: ArrayView1<f64>, se: ArrayView1<f64>, y: ArrayView1<f64>) -> (f64, f64, f64) {
    let n = y.len();
    let mut w_sum = 0.0;
    let mut w_mu = 0.0;
    let mut w_y = 0.0;
    let mut valid_n = 0usize;
    let mut weights = vec![0.0; n];
    for i in 0..n {
        let (mui, sei, yi) = (mu[i], se[i], y[i]);
        if !mui.is_finite() || !sei.is_finite() || !yi.is_finite() || sei < 0.0 {
            continue;
        }
        let wi = 1.0 / (sei * sei + EPS_VAR);
        if !wi.is_finite() || wi <= 0.0 {
            continue;
        }
        weights[i] = wi;
        w_sum += wi;
        w_mu += wi * mui;
        w_y += wi * yi;
        valid_n += 1;
    }
    if valid_n < MIN_OBS || !w_sum.is_finite() || w_sum <= EPS_VAR {
        return (1.0, 0.0, 1.0);
    }
    let mu_bar = w_mu / w_sum;
    let y_bar = w_y / w_sum;
    let mut s_mm = 0.0;
    let mut s_my = 0.0;
    for i in 0..n {
        let wi = weights[i];
        if wi <= 0.0 {
            continue;
        }
        let dm = mu[i] - mu_bar;
        let dy = y[i] - y_bar;
        s_mm += wi * dm * dm;
        s_my += wi * dm * dy;
    }
    let ridge = RIDGE_REL * s_mm.max(w_sum);
    let denom = s_mm + ridge;
    let a_raw = if denom.is_finite() && denom > 0.0 {
        (s_my + ridge) / denom
    } else {
        1.0
    };
    let a = if a_raw.is_finite() {
        a_raw.clamp(SLOPE_MIN, SLOPE_MAX)
    } else {
        1.0
    };
    let b_raw = y_bar - a * mu_bar;
    let b = if b_raw.is_finite() { b_raw } else { 0.0 };
    let c = fit_scale(mu, se, y, a, b);
    (a, b, c)
}

/// Fit a `PosteriorCalibrator` from leave-one-out subsampled predictions on
/// `model`.
pub fn fit_calibrator<R: Rng>(
    model: &ENN,
    params: &ENNParams,
    num_samples: usize,
    rng: &mut R,
    mode: CalibrationMode,
    observation_noise: bool,
) -> Result<PosteriorCalibrator, ENNError> {
    if mode == CalibrationMode::None {
        return Ok(PosteriorCalibrator::Identity);
    }
    let n = model.len();
    let m_dim = model.num_metrics();
    if n < 4 {
        return Ok(match mode {
            CalibrationMode::None => PosteriorCalibrator::Identity,
            CalibrationMode::Scale => PosteriorCalibrator::Scale(ScaleCalibrator::identity(m_dim)),
            CalibrationMode::Affine => {
                PosteriorCalibrator::Affine(AffineCalibrator::identity(m_dim))
            }
        });
    }
    let p_actual = num_samples.max(4).min(n);
    let indices: Vec<usize> = if p_actual == n {
        (0..n).collect()
    } else {
        sample(rng, n, p_actual).into_iter().collect()
    };
    let (x_sub, y_sub, _) = model.natural_rows(&indices)?;
    let flags = PosteriorFlags::new()
        .exclude_nearest(true)
        .observation_noise(observation_noise);
    let post = model.posterior(&x_sub.view(), params, &flags)?;
    let mu2 = post
        .mu
        .view()
        .into_dimensionality::<Ix2>()
        .map_err(|_| ENNError::InvalidParameter("expected 2D posterior mu".to_string()))?;
    let se2 = post
        .se
        .view()
        .into_dimensionality::<Ix2>()
        .map_err(|_| ENNError::InvalidParameter("expected 2D posterior se".to_string()))?;
    match mode {
        CalibrationMode::None => Ok(PosteriorCalibrator::Identity),
        CalibrationMode::Scale => Ok(PosteriorCalibrator::Scale(ScaleCalibrator::fit(
            mu2,
            se2,
            y_sub.view(),
        )?)),
        CalibrationMode::Affine => Ok(PosteriorCalibrator::Affine(AffineCalibrator::fit(
            mu2,
            se2,
            y_sub.view(),
        )?)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::index::IndexDriver;
    use ndarray::{ArrayD, IxDyn, array};
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    #[test]
    fn small_identity() {
        let mu = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let se = array![[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]];
        let y = array![[2.0, 4.0], [6.0, 8.0], [10.0, 12.0]];
        let cal = AffineCalibrator::fit(mu.view(), se.view(), y.view()).unwrap();
        assert_eq!(cal, AffineCalibrator::identity(2));
    }

    #[test]
    fn affine_recovery() {
        let mut mu = Array2::zeros((40, 1));
        let se = Array2::from_elem((40, 1), 0.1);
        let mut y = Array2::zeros((40, 1));
        for i in 0..40 {
            let m = (i as f64) * 0.25 - 5.0;
            let sign = if i % 2 == 0 { 1.0 } else { -1.0 };
            mu[[i, 0]] = m;
            y[[i, 0]] = 2.0 * m + 3.0 + sign * 0.3;
        }
        let cal = AffineCalibrator::fit(mu.view(), se.view(), y.view()).unwrap();
        assert!((cal.a[0] - 2.0).abs() < 0.05);
        assert!((cal.b[0] - 3.0).abs() < 0.05);
        assert!((cal.c[0] - 3.0).abs() < 0.2);

        let scale_cal = ScaleCalibrator::fit(mu.view(), se.view(), y.view()).unwrap();
        assert!(scale_cal.c[0] > 1.0);
    }

    #[test]
    fn bounded_projection() {
        let cal = PosteriorCalibrator::Affine(AffineCalibrator {
            a: array![2.0],
            b: array![1.0],
            c: array![1.5],
        });
        let bounds = array![[0.0, 1.0]];
        let normal = ENNNormal::new(
            array![[0.8]].into_dyn(),
            array![[0.5]].into_dyn(),
            array![[0.3]].into_dyn(),
            array![[0.4]].into_dyn(),
            None,
        )
        .with_bounds(bounds.clone());
        let out = cal.calibrate_normal(normal, Some(&bounds));
        assert!(out.mu[[0, 0]] > 0.0 && out.mu[[0, 0]] < 1.0);
        assert!((out.se_epi[[0, 0]] - 0.45).abs() < 1e-12);
        assert!((out.se_ale[[0, 0]] - 0.60).abs() < 1e-12);
        assert!((out.se[[0, 0]] - 0.75).abs() < 1e-12);

        let mut draws = Array3::from_shape_vec((2, 1, 1), vec![0.7, 0.9]).unwrap();
        let mu0 = array![[0.8]];
        cal.calibrate_draws(&mut draws, &mu0.view(), Some(&bounds));
        assert!(draws.iter().all(|&v| v > 0.0 && v < 1.0));
        let _ = ArrayD::<f64>::zeros(IxDyn(&[1, 1]));
    }

    #[test]
    fn model_calibration() {
        let x = array![
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [0.5, 0.5],
            [0.2, 0.8]
        ];
        let y = array![[0.0], [1.0], [1.0], [2.0], [1.0], [1.0]];
        let model = ENN::new(x, y, None, false, IndexDriver::Exact).unwrap();
        let params = ENNParams::new(3, 1.0, 0.1).unwrap();
        let mut rng = StdRng::seed_from_u64(42);
        let cal =
            fit_calibrator(&model, &params, 6, &mut rng, CalibrationMode::Affine, false).unwrap();
        assert!(matches!(cal, PosteriorCalibrator::Affine(_)));
    }
}
