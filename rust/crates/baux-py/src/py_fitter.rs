//! Stateful ENN fitter Python bindings.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::py_model::PyENN;

/// Wrapper for ENNParams
#[pyclass(name = "ENNParams", from_py_object)]
#[derive(Clone, Copy)]
pub struct PyENNParams {
    pub(crate) inner: baux::ENNParams,
}

#[pymethods]
impl PyENNParams {
    #[new]
    #[pyo3(signature = (k_neighbors, epistemic_scale, aleatoric_scale))]
    fn new(k_neighbors: i32, epistemic_scale: f64, aleatoric_scale: f64) -> PyResult<Self> {
        let inner = baux::ENNParams::new(k_neighbors, epistemic_scale, aleatoric_scale)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self { inner })
    }

    #[getter]
    fn k_neighbors(&self) -> i32 {
        self.inner.k_neighbors
    }

    #[getter]
    fn epistemic_scale(&self) -> f64 {
        self.inner.epistemic_scale
    }

    #[getter]
    fn aleatoric_scale(&self) -> f64 {
        self.inner.aleatoric_scale
    }

    fn __repr__(&self) -> String {
        format!(
            "ENNParams(k={}, epi={:.4}, ale={:.4})",
            self.inner.k_neighbors, self.inner.epistemic_scale, self.inner.aleatoric_scale
        )
    }
}

impl From<baux::ENNParams> for PyENNParams {
    fn from(inner: baux::ENNParams) -> Self {
        Self { inner }
    }
}

/// Pure stateless hyperparameter fit over an ENN model.
#[pyfunction(name = "fit_model")]
#[pyo3(signature = (model, k, num_candidates, num_samples, infer_aleatoric=true, seed=0, warm=None))]
pub fn fit_model(
    model: &PyENN,
    k: i32,
    num_candidates: usize,
    num_samples: usize,
    infer_aleatoric: bool,
    seed: u64,
    warm: Option<PyENNParams>,
) -> PyResult<PyENNParams> {
    let inner = baux::fit_model(
        &model.inner,
        k,
        num_candidates,
        num_samples,
        infer_aleatoric,
        warm.map(|w| w.inner),
        baux::Key::new(seed),
    )
    .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(PyENNParams { inner })
}
