//! Stateful ENN fitter Python bindings.

use numpy::{PyArray1, PyReadonlyArray2, ToPyArray};
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

#[pyclass(name = "ENNFitter")]
pub struct PyENNFitter {
    inner: baux::ENNFitter,
    rng: baux::Prng,
}

#[pymethods]
impl PyENNFitter {
    #[new]
    #[pyo3(signature = (k, seed=None, infer_aleatoric_variance_scale=true))]
    fn new(k: i32, seed: Option<u64>, infer_aleatoric_variance_scale: bool) -> Self {
        Self {
            inner: baux::ENNFitter::new(k, infer_aleatoric_variance_scale),
            rng: baux::prng(baux::resolve_seed(seed)),
        }
    }

    #[pyo3(signature = (x, y, yvar=None))]
    fn tell(
        &mut self,
        x: PyReadonlyArray2<f64>,
        y: PyReadonlyArray2<f64>,
        yvar: Option<PyReadonlyArray2<f64>>,
    ) -> PyResult<()> {
        let yvar_arr = yvar.as_ref().map(|v| v.as_array());
        self.inner
            .tell(&x.as_array(), &y.as_array(), yvar_arr.as_ref())
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    fn y_std<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        Ok(self.inner.y_std().to_pyarray(py))
    }

    #[pyo3(signature = (model, num_candidates, num_samples, params_warm_start=None))]
    fn ask(
        &mut self,
        model: &PyENN,
        num_candidates: usize,
        num_samples: usize,
        params_warm_start: Option<PyENNParams>,
    ) -> PyResult<PyENNParams> {
        let warm = params_warm_start.as_ref().map(|p| p.inner);
        let result = self
            .inner
            .ask(
                &model.inner,
                num_candidates,
                num_samples,
                warm.as_ref(),
                &mut self.rng,
            )
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(PyENNParams { inner: result })
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
