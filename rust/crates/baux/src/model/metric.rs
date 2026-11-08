//! Optional upstream AUTO metric on ordinary disk ENN, not full model-weight
//! rows.
use ndarray::{Array1, ArrayView2};

use super::inspection::ENN;
use crate::backend::EnnBackend;
use crate::error::ENNError;
use crate::index::disk_driver;
use crate::metric_auto::{AutoMetric, DiagonalMetric, ScaleUpdate};

const SCALE_TOL: f64 = 0.01;
const SCALE_DRIFT: f64 = std::f64::consts::LN_2;

fn log_ratio(a: &Array1<f64>, b: &Array1<f64>) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x / y.max(1e-12)).ln().abs())
        .fold(0.0_f64, f64::max)
}

impl ENN {
    /// Enable the pinned upstream dependence estimator, Algorithm R reservoir,
    /// LOO gate and geometric refit schedule. The caller supplies the seed.
    /// This does not select the resident optimizer's tied-family metric policy.
    pub fn learn_metric(&mut self, tied: Vec<Vec<usize>>, seed: u64) -> Result<(), ENNError> {
        if !matches!(self.backend, EnnBackend::Disk(_)) {
            return Err(ENNError::InvalidParameter(
                "Enable AUTO once on disk BPANN".into(),
            ));
        }
        self.enable_metric(tied, seed)
    }

    pub(crate) fn enable_metric(
        &mut self,
        tied: Vec<Vec<usize>>,
        seed: u64,
    ) -> Result<(), ENNError> {
        if self.metric_fixed || self.metric.is_some() {
            return Err(ENNError::InvalidParameter(
                "Enable AUTO metric once per model".into(),
            ));
        }
        let mut uniq: Vec<usize> = tied.iter().flatten().copied().collect();
        uniq.sort_unstable();
        uniq.dedup();
        self.set_unscaled(uniq)?;
        self.tied_groups.clone_from(&tied);
        let mut metric = AutoMetric::new(self.num_dim, self.num_metrics, tied, seed)?;
        // Observe in bounded chunks; never gather the whole persisted dataset.
        for start in (0..self.num_obs).step_by(64) {
            let ids = (start..(start + 64).min(self.num_obs)).collect::<Vec<_>>();
            let (x, y, _) = self.backend.train_rows(&ids)?;
            metric.observe(x.as_slice().unwrap(), y.as_slice().unwrap(), ids.len())?;
        }
        let weights = metric.weights();
        let scale = Array1::from_iter(weights.iter().map(|v| 1.0 / v.sqrt()));
        if matches!(self.backend, EnnBackend::Disk(_)) {
            self.backend.set_metric(&scale, true)?;
        } else {
            self.backend.mark_stale();
        }
        self.built_scale = scale.clone();
        self.x_scale = scale;
        self.scale_x = true;
        self.metric_fixed = true;
        self.metric = Some(metric);
        Ok(())
    }

    pub fn diagonal_metric(&self) -> Option<DiagonalMetric> {
        self.metric.as_ref().map(AutoMetric::diagonal_metric)
    }

    pub fn current_weights(&self) -> Option<&[f64]> {
        self.metric.as_ref().map(AutoMetric::weights)
    }

    pub fn trust_sides(&self) -> Option<Vec<f64>> {
        self.current_weights()
            .map(crate::metric_weights::trust_sides)
    }

    pub fn set_unscaled(&mut self, dims: Vec<usize>) -> Result<(), ENNError> {
        if dims.iter().any(|&j| j >= self.num_dim) {
            return Err(ENNError::InvalidParameter(
                "unscaled dimension out of range".into(),
            ));
        }
        if let Some(metric) = self.metric.as_mut() {
            metric.set_unscaled(dims.clone())?;
        }
        self.unscaled_dims = dims;
        if self.scale_x && !self.metric_fixed {
            self.refresh_scale(self.num_obs)?;
        }
        Ok(())
    }

    pub fn set_tied(&mut self, groups: Vec<Vec<usize>>) -> Result<(), ENNError> {
        crate::metric_weights::validate_tied(&groups, self.num_dim)?;
        self.tied_groups = groups;
        Ok(())
    }

    pub fn tied_groups(&self) -> &[Vec<usize>] {
        if let Some(m) = self.metric.as_ref() {
            m.tied()
        } else {
            &self.tied_groups
        }
    }

    pub fn set_metric(&mut self, weights: &[f64]) -> Result<(), ENNError> {
        if self.metric.is_some() {
            return self.metric_weights(weights);
        }
        if weights.len() != self.num_dim {
            return Err(ENNError::InvalidShape {
                expected: vec![self.num_dim],
                got: vec![weights.len()],
            });
        }
        if weights.iter().any(|&w| !w.is_finite() || w <= 0.0) {
            return Err(ENNError::InvalidParameter(
                "metric weights must be finite and positive".into(),
            ));
        }
        let scale = Array1::from_iter(weights.iter().map(|&w| 1.0 / w.sqrt()));
        self.scale_x = true;
        self.metric_fixed = true;
        if disk_driver(self.backend.driver()) {
            self.incremental_scale(scale)
        } else {
            self.x_scale = scale;
            self.backend.mark_stale();
            Ok(())
        }
    }

    pub(super) fn data_scale(&self, n: usize) -> Array1<f64> {
        let mut x_scale = super::scale_moments(n, self.num_dim, &self.x_sum, &self.x_sumsq, 1e-12);
        for &j in &self.unscaled_dims {
            x_scale[j] = 1.0;
        }
        x_scale
    }

    pub(super) fn refresh_scale(&mut self, n: usize) -> Result<(), ENNError> {
        let x_scale = self.data_scale(n);
        if disk_driver(self.backend.driver()) {
            self.incremental_scale(x_scale)
        } else {
            self.x_scale = x_scale;
            self.backend.mark_stale();
            Ok(())
        }
    }

    fn incremental_scale(&mut self, x_scale: Array1<f64>) -> Result<(), ENNError> {
        if log_ratio(&x_scale, &self.x_scale) <= SCALE_TOL {
            return Ok(());
        }
        let rebuild = log_ratio(&x_scale, &self.built_scale) > SCALE_DRIFT;
        self.backend.set_metric(&x_scale, rebuild)?;
        if rebuild {
            self.built_scale = x_scale.clone();
        }
        self.x_scale = x_scale;
        Ok(())
    }

    pub fn metric_config(
        &mut self,
        growth: Option<f64>,
        drift: Option<f64>,
        seed: Option<u64>,
        capacity: Option<usize>,
    ) -> Result<(), ENNError> {
        self.metric
            .as_mut()
            .ok_or_else(|| ENNError::InvalidParameter("AUTO metric is not enabled".into()))?
            .configure(growth, drift, seed, capacity)
    }

    pub fn metric_weights(&mut self, weights: &[f64]) -> Result<(), ENNError> {
        let update = self
            .metric
            .as_ref()
            .ok_or_else(|| ENNError::InvalidParameter("AUTO metric is not enabled".into()))?
            .preview(weights)?;
        self.apply_metric(update)?;
        self.metric.as_mut().unwrap().set_weights(weights)?;
        Ok(())
    }

    pub(super) fn observe_metric(
        &mut self,
        x: &ArrayView2<f64>,
        y: &ArrayView2<f64>,
    ) -> Result<(), ENNError> {
        if self.metric.is_none() {
            return Ok(());
        }
        let x = x.to_owned();
        let y = if self.bounded_outputs {
            crate::y_bounds::warp_y(*y, &self.y_bounds)?
        } else {
            y.to_owned()
        };
        let ready = self.metric.as_mut().unwrap().ingest(
            x.as_slice().unwrap(),
            y.as_slice().unwrap(),
            x.nrows(),
        )?;
        if ready {
            if let Some(weights) = self.metric.as_mut().unwrap().propose()? {
                self.metric_weights(&weights)?;
            }
        }
        Ok(())
    }

    fn apply_metric(&mut self, update: ScaleUpdate) -> Result<(), ENNError> {
        let scale = Array1::from(update.x_scale);
        if matches!(self.backend, EnnBackend::Disk(_)) {
            self.backend.set_metric(&scale, update.rebuild)?;
        } else {
            self.backend.mark_stale();
        }
        if update.rebuild {
            self.built_scale = scale.clone();
        }
        self.x_scale = scale;
        Ok(())
    }

    pub(crate) fn refresh_finite(&mut self, k: usize, unscaled: &[usize]) -> Result<(), ENNError> {
        let n = self.num_obs;
        let d = self.num_dim;
        let m = self.num_metrics;
        if self.metric.is_none() || !(9..crate::metric_sobol::DEPENDENCE_ROWS).contains(&n) || d < 2
        {
            return Ok(());
        }
        let ids: Vec<usize> = (0..n).collect();
        let (x, y, _) = self.backend.train_rows(&ids)?;
        let mut s = vec![0.0; d];
        for col in 0..m {
            let y_col = y.column(col);
            let sb = crate::util::sobol_indices(&x.view(), &y_col);
            for j in 0..d {
                s[j] += sb[j] / m as f64;
            }
        }
        for g in self.tied_groups().to_vec() {
            let avg = g.iter().map(|&j| s[j]).sum::<f64>() / g.len().max(1) as f64;
            for j in g {
                s[j] = avg;
            }
        }
        let s_max = s.iter().copied().fold(0.0_f64, f64::max);
        if s_max <= 1.0e-12 {
            return Ok(());
        }
        let mut w: Vec<f64> = s.into_iter().map(|v| v.max(0.05 * s_max)).collect();
        let mean = w.iter().sum::<f64>() / d as f64;
        for v in &mut w {
            *v /= mean.max(1.0e-12);
        }
        for &j in unscaled {
            if j < d {
                w[j] = 1.0;
            }
        }
        let k_use = k.min(n - 1).max(1);
        let xs = x.as_slice().expect("standard layout");
        let ys = y.as_slice().expect("standard layout");
        let ones = vec![1.0; d];
        if crate::metric_loo::loo_loglik(xs, n, d, ys, m, &w, k_use)
            > crate::metric_loo::loo_loglik(xs, n, d, ys, m, &ones, k_use)
        {
            self.metric_weights(&w)?;
        }
        Ok(())
    }

    pub(crate) fn refresh_codomain(
        &mut self,
        params: &crate::params::ENNParams,
        stiefel_rank: Option<usize>,
        calibrate_alpha: Option<f64>,
    ) -> Result<(), ENNError> {
        let n = self.num_obs;
        let m = self.num_metrics;
        if let Some(rank) = stiefel_rank {
            if n >= 2 && rank >= 1 && rank <= m {
                let ids: Vec<usize> = (0..n).collect();
                let (_, y, _) = self.natural_rows(&ids)?;
                let ys = y.as_standard_layout();
                if let Ok(st) = baux::semantics::stiefel::Stiefel::from_cov(
                    ys.as_slice().expect("standard layout"),
                    n,
                    m,
                    rank,
                ) {
                    self.stiefel = Some(st.with_count(n));
                }
            }
        }
        if let Some(alpha) = calibrate_alpha {
            if n >= 4 {
                let ids: Vec<usize> = (0..n).collect();
                let (x, y, _) = self.natural_rows(&ids)?;
                self.affine = None;
                let flags = crate::params::PosteriorFlags::new().exclude_nearest(true);
                let loo = <Self as crate::traits::PosteriorComputation>::posterior(
                    self,
                    &x.view(),
                    params,
                    &flags,
                )?;
                let mu_2d = loo
                    .mu
                    .into_dimensionality::<ndarray::Ix2>()
                    .map_err(|e| ENNError::InvalidParameter(e.to_string()))?;
                let se_2d = loo
                    .se
                    .into_dimensionality::<ndarray::Ix2>()
                    .map_err(|e| ENNError::InvalidParameter(e.to_string()))?;
                let f_mu = mu_2d.as_standard_layout();
                let f_se = se_2d.as_standard_layout();
                let f_y = y.as_standard_layout();
                let mu_s = f_mu.as_slice().expect("standard layout");
                let se_s = f_se.as_slice().expect("standard layout");
                let y_s = f_y.as_slice().expect("standard layout");
                if let Ok(mut aff) = baux::semantics::affine::Affine::fit_wls(mu_s, y_s, se_s, n, m)
                {
                    if alpha > 0.0 && alpha < 1.0 {
                        if let Ok(c) = aff.conformal_scale(mu_s, y_s, se_s, n, m, alpha) {
                            aff.c = c;
                        }
                    }
                    self.affine = Some(aff);
                }
            }
        }
        Ok(())
    }
}
