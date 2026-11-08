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
}
