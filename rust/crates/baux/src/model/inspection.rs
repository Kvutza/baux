//! Observation shape and output bounds.

use ndarray::{Array1, Array2};
use std::path::PathBuf;

use crate::backend::{EnnBackend, EnnStorage};
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
    pub(super) y_sum: Array1<f64>,
    pub(super) y_sumsq: Array1<f64>,
    pub(super) x_sum: Array1<f64>,
    pub(super) x_sumsq: Array1<f64>,
    pub(super) work_dir: Option<PathBuf>,
    pub(super) metric: Option<crate::metric_auto::AutoMetric>,
}

impl ENN {
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
