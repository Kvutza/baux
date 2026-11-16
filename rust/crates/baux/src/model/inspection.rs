//! Observation shape and output bounds.

use ndarray::{Array1, Array2};
use std::path::PathBuf;

use crate::backend::{EnnBackend, EnnStorage};
use crate::error::ENNError;
use crate::index::IndexDriver;

#[derive(Debug, Clone)]
pub struct ModelOptions {
    pub scale_x: bool,
    pub driver: IndexDriver,
    pub storage: EnnStorage,
    pub work_dir: Option<PathBuf>,
    pub y_bounds: Option<Array2<f64>>,
}

impl Default for ModelOptions {
    fn default() -> Self {
        Self {
            scale_x: false,
            driver: IndexDriver::Exact,
            storage: EnnStorage::InMemory,
            work_dir: None,
            y_bounds: None,
        }
    }
}

#[derive(Clone, Debug)]
pub(crate) struct PureScaleCache(
    pub(crate) std::sync::Arc<crate::posterior::neighbor_dist::ScaleInfo>,
);

impl Default for PureScaleCache {
    fn default() -> Self {
        Self(std::sync::Arc::new((std::sync::Arc::from([]), 0, 1.0, 1.0)))
    }
}

impl PureScaleCache {
    pub(crate) fn get(&self) -> Option<crate::posterior::neighbor_dist::ScaleInfo> {
        if self.0.0.is_empty() {
            None
        } else {
            Some((*self.0).clone())
        }
    }

    pub(crate) fn set(
        &self,
        _info: crate::posterior::neighbor_dist::ScaleInfo,
    ) -> Result<(), crate::posterior::neighbor_dist::ScaleInfo> {
        Ok(())
    }
}

pub(crate) fn compute_scales(
    num_obs: usize,
    dist_metric: baux::semantics::neighbor::DistanceMetric,
    backend: &EnnBackend,
    scale_x: bool,
    x_scale: &Array1<f64>,
) -> Result<std::sync::Arc<crate::posterior::neighbor_dist::ScaleInfo>, ENNError> {
    use crate::posterior::neighbor_dist::distance_matrix;
    use baux::semantics::neighbor::{DistanceMetric, local_scales};
    use std::sync::Arc;

    let local_k = match dist_metric {
        DistanceMetric::Euclidean => return Ok(Arc::new((Arc::from([]), 0, 1.0, 1.0))),
        DistanceMetric::Conformal { local_k }
        | DistanceMetric::Hyperbolic { local_k }
        | DistanceMetric::Wasserstein { local_k } => local_k.max(1),
    };
    if num_obs == 0 {
        return Ok(Arc::new((Arc::from([]), local_k, 1.0, 1.0)));
    }
    let d2 = if let Some(tx) = backend.x_view() {
        distance_matrix(&tx, &tx, scale_x, &x_scale.view())
    } else {
        let ids: Vec<usize> = (0..num_obs).collect();
        let (tx, _, _) = backend.train_rows(&ids)?;
        distance_matrix(&tx.view(), &tx.view(), scale_x, &x_scale.view())
    };
    let sc: Arc<[f64]> = local_scales(num_obs, local_k, |r, c| d2[[r, c]])
        .into_iter()
        .map(|s2| s2.max(1.0e-12).sqrt())
        .collect::<Vec<_>>()
        .into();
    let s_min = sc.iter().copied().fold(f64::INFINITY, f64::min).max(1.0e-6);
    let s_max = sc.iter().copied().fold(1.0e-6_f64, f64::max).max(s_min);
    Ok(Arc::new((sc, local_k, s_min, s_max)))
}

/// Epistemic Nearest Neighbors model.
pub struct ENN {
    pub(crate) backend: EnnBackend,
    pub(crate) num_obs: usize,
    pub(crate) num_dim: usize,
    pub(crate) num_metrics: usize,
    pub(crate) scale_x: bool,
    pub(crate) x_scale: Array1<f64>,
    pub(crate) y_scale: Array1<f64>,
    pub(crate) y_bounds: Array2<f64>,
    pub(crate) bounded_outputs: bool,
    pub(crate) stiefel: Option<baux::semantics::stiefel::Stiefel>,
    pub(crate) affine: Option<baux::semantics::affine::Affine>,
    pub(crate) y_sum: Array1<f64>,
    pub(crate) y_sumsq: Array1<f64>,
    pub(super) x_sum: Array1<f64>,
    pub(super) x_sumsq: Array1<f64>,
    pub(super) work_dir: Option<PathBuf>,
    pub(super) metric: Option<crate::metric_auto::AutoMetric>,
    pub(super) metric_fixed: bool,
    pub(super) built_scale: Array1<f64>,
    pub(super) unscaled_dims: Vec<usize>,
    pub(super) tied_groups: Vec<Vec<usize>>,
    pub(crate) dist_metric: baux::semantics::neighbor::DistanceMetric,
    pub(crate) scale_info: std::sync::Arc<crate::posterior::neighbor_dist::ScaleInfo>,
    pub(crate) scale_cache: PureScaleCache,
}

impl Clone for ENN {
    fn clone(&self) -> Self {
        let metric = self.metric.as_ref().and_then(|m| {
            let mut next = crate::metric_auto::AutoMetric::new(
                self.num_dim,
                self.num_metrics,
                m.tied().to_vec(),
                m.seed(),
            )
            .ok()?;
            let _ = next.set_unscaled(self.unscaled_dims.clone());
            let _ = next.set_weights(m.weights());
            Some(next)
        });
        Self {
            backend: self.backend.clone(),
            num_obs: self.num_obs,
            num_dim: self.num_dim,
            num_metrics: self.num_metrics,
            scale_x: self.scale_x,
            x_scale: self.x_scale.clone(),
            y_scale: self.y_scale.clone(),
            y_bounds: self.y_bounds.clone(),
            bounded_outputs: self.bounded_outputs,
            stiefel: self.stiefel.clone(),
            affine: self.affine.clone(),
            y_sum: self.y_sum.clone(),
            y_sumsq: self.y_sumsq.clone(),
            x_sum: self.x_sum.clone(),
            x_sumsq: self.x_sumsq.clone(),
            work_dir: self.work_dir.clone(),
            metric,
            metric_fixed: self.metric_fixed,
            built_scale: self.built_scale.clone(),
            unscaled_dims: self.unscaled_dims.clone(),
            tied_groups: self.tied_groups.clone(),
            dist_metric: self.dist_metric,
            scale_info: self.scale_info.clone(),
            scale_cache: self.scale_cache.clone(),
        }
    }
}

impl ENN {
    /// Return a new `ENN` model configured with a local-scale pairwise
    /// [`baux::semantics::neighbor::Dissimilarity`] geometry.
    pub fn with_geometry(mut self, geometry: baux::semantics::neighbor::DistanceMetric) -> Self {
        self.dist_metric = geometry;
        self = self.refresh_scales();
        self
    }

    pub(crate) fn refresh_scales(mut self) -> Self {
        if let Ok(info) = compute_scales(
            self.num_obs,
            self.dist_metric,
            &self.backend,
            self.scale_x,
            &self.x_scale,
        ) {
            self.scale_info = info.clone();
            self.scale_cache = PureScaleCache(info);
        }
        self
    }

    pub fn scale_info(&self) -> &std::sync::Arc<crate::posterior::neighbor_dist::ScaleInfo> {
        &self.scale_info
    }

    pub fn with_thresholds(mut self, soft: usize, hard: usize) -> Result<Self, ENNError> {
        if soft == 0 || hard == 0 || hard < soft {
            return Err(ENNError::InvalidParameter(
                "thresholds require 1 <= soft <= hard".to_string(),
            ));
        }
        self.backend = self.backend.with_thresholds(soft, hard)?;
        Ok(self)
    }

    /// Return the configured pairwise
    /// [`baux::semantics::neighbor::Dissimilarity`] geometry.
    pub fn geometry(&self) -> baux::semantics::neighbor::DistanceMetric {
        self.dist_metric
    }

    pub fn with_stiefel(mut self, st: baux::semantics::stiefel::Stiefel) -> Result<Self, ENNError> {
        if st.rows() != self.num_metrics {
            return Err(ENNError::InvalidParameter(format!(
                "Stiefel rows {} != num_metrics {}",
                st.rows(),
                self.num_metrics
            )));
        }
        self.stiefel = Some(st.with_count(self.num_obs));
        Ok(self)
    }

    pub fn with_affine(mut self, aff: baux::semantics::affine::Affine) -> Result<Self, ENNError> {
        if aff.a.len() != self.num_metrics
            || aff.b.len() != self.num_metrics
            || aff.c.len() != self.num_metrics
        {
            return Err(ENNError::InvalidParameter(format!(
                "Affine cols {} != num_metrics {}",
                aff.a.len(),
                self.num_metrics
            )));
        }
        self.affine = Some(aff);
        Ok(self)
    }

    pub fn stiefel(&self) -> Option<&baux::semantics::stiefel::Stiefel> {
        self.stiefel.as_ref()
    }

    pub fn affine(&self) -> Option<&baux::semantics::affine::Affine> {
        self.affine.as_ref()
    }

    pub(crate) fn apply_codomain(
        &self,
        mu: &mut Array2<f64>,
        se: &mut Array2<f64>,
        se_epi: &mut Array2<f64>,
        se_ale: &mut Array2<f64>,
    ) -> Result<(), ENNError> {
        let nrows = mu.nrows();
        if nrows == 0 {
            return Ok(());
        }
        let dim = mu.raw_dim();
        if let Some(st) = self.stiefel.as_ref() {
            let f_mu = mu.as_standard_layout();
            let f_se = se.as_standard_layout();
            let s_old = f_se.as_slice().expect("standard layout");
            let (m_out, s_out) = st
                .transform_stein(f_mu.as_slice().expect("standard layout"), s_old, nrows)
                .map_err(ENNError::InvalidParameter)?;
            for (k, (&so, &sn)) in s_old.iter().zip(&s_out).enumerate() {
                let r = if so > 1.0e-12 { sn / so } else { 1.0 };
                se_epi.as_slice_memory_order_mut().expect("contiguous")[k] *= r;
                se_ale.as_slice_memory_order_mut().expect("contiguous")[k] *= r;
            }
            *mu = Array2::from_shape_vec(dim, m_out).unwrap();
            *se = Array2::from_shape_vec(dim, s_out).unwrap();
        }
        if let Some(aff) = self.affine.as_ref() {
            let flat_mu = mu.as_standard_layout();
            let cal_mu = aff
                .apply_mu(flat_mu.as_slice().expect("standard layout"), nrows)
                .map_err(ENNError::InvalidParameter)?;
            *mu = Array2::from_shape_vec(dim, cal_mu)
                .map_err(|e| ENNError::InvalidParameter(e.to_string()))?;
            for arr in [se, se_epi, se_ale] {
                let flat = arr.as_standard_layout();
                let scaled = aff
                    .apply_se(flat.as_slice().expect("standard layout"), nrows)
                    .map_err(ENNError::InvalidParameter)?;
                *arr = Array2::from_shape_vec(dim, scaled)
                    .map_err(|e| ENNError::InvalidParameter(e.to_string()))?;
            }
        }
        Ok(())
    }
    pub fn len(&self) -> usize {
        self.num_obs
    }

    pub fn is_empty(&self) -> bool {
        self.num_obs == 0
    }

    pub fn num_outputs(&self) -> usize {
        self.num_metrics
    }

    pub fn y_bounds(&self) -> &Array2<f64> {
        &self.y_bounds
    }

    pub fn bounded_outputs(&self) -> bool {
        self.bounded_outputs
    }

    pub(crate) fn output_scale(&self) -> &Array1<f64> {
        &self.y_scale
    }

    pub fn x_scale(&self) -> Array2<f64> {
        self.x_scale.clone().insert_axis(ndarray::Axis(0))
    }

    pub fn y_scale(&self) -> Array2<f64> {
        self.y_scale.clone().insert_axis(ndarray::Axis(0))
    }

    pub(crate) fn num_obs(&self) -> usize {
        self.num_obs
    }

    pub fn num_dim(&self) -> usize {
        self.num_dim
    }

    pub fn num_metrics(&self) -> usize {
        self.num_metrics
    }

    pub fn has_yvar(&self) -> bool {
        self.num_obs > 0 && self.backend.row_yvar(0).ok().flatten().is_some()
    }
}
