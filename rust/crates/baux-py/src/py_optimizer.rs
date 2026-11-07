//! Optimizer Python bindings.

#![allow(clippy::too_many_arguments)]

use numpy::{IntoPyArray, PyArrayDyn, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

mod build;
mod config;
mod regional;
pub use build::{bind_lhd, bind_regions, bind_zero, create_py, enn_py};

/// Python wrapper for BayesOptimizer
#[pyclass(name = "BayesOptimizer")]
pub struct PyBayesOptimizer {
    inner: baux::BayesOptimizer,
    rng: baux::Prng,
}

#[pymethods]
impl PyBayesOptimizer {
    /// Ask for candidate points
    #[pyo3(signature = (num_arms, seed=None))]
    fn ask<'py>(
        &mut self,
        py: Python<'py>,
        num_arms: usize,
        seed: Option<u64>,
    ) -> PyResult<Bound<'py, PyArrayDyn<f64>>> {
        let result = match seed {
            Some(seed) => {
                let mut rng = baux::prng(seed);
                self.inner.ask_natural(num_arms, &mut rng)
            }
            None => self.inner.ask_natural(num_arms, &mut self.rng),
        }
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(result.into_dyn().into_pyarray(py))
    }

    /// Outstanding warm-up or regional evaluation IDs in ask order.
    fn pending_ids(&self) -> Option<Vec<u64>> {
        self.inner
            .pending_ids()
            .map(|ids| ids.into_iter().map(|id| id.0).collect())
    }

    /// Complete a subset of the outstanding evaluations by identity.
    #[pyo3(signature = (ids, y, seed=None, y_var=None))]
    fn complete_ids(
        &mut self,
        ids: Vec<u64>,
        y: PyReadonlyArray2<f64>,
        seed: Option<u64>,
        y_var: Option<PyReadonlyArray2<f64>>,
    ) -> PyResult<()> {
        let ids = ids
            .into_iter()
            .map(baux::optimizer::EvaluationId)
            .collect::<Vec<_>>();
        let y_view = y.as_array();
        let yvar_view = y_var.as_ref().map(PyReadonlyArray2::as_array);
        match seed {
            Some(seed) => {
                let mut rng = baux::prng(seed);
                self.inner
                    .complete_ids(&ids, &y_view, yvar_view.as_ref(), &mut rng)
            }
            None => self
                .inner
                .complete_ids(&ids, &y_view, yvar_view.as_ref(), &mut self.rng),
        }
        .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Retire objective evaluations that produced no measured result.
    #[pyo3(signature = (ids, seed=None))]
    fn cancel_ids(&mut self, ids: Vec<u64>, seed: Option<u64>) -> PyResult<()> {
        let ids = ids
            .into_iter()
            .map(baux::optimizer::EvaluationId)
            .collect::<Vec<_>>();
        match seed {
            Some(seed) => {
                let mut rng = baux::prng(seed);
                self.inner.cancel_ids(&ids, &mut rng)
            }
            None => self.inner.cancel_ids(&ids, &mut self.rng),
        }
        .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Restart a regional slot using a center in natural parameter coordinates.
    /// Returns IDs abandoned from that slot; late results must be imported.
    #[pyo3(signature = (slot, center, seed=None))]
    fn restart_region(
        &mut self,
        slot: usize,
        center: PyReadonlyArray1<f64>,
        seed: Option<u64>,
    ) -> PyResult<Vec<u64>> {
        let center_view = center.as_array();
        let abandoned = match seed {
            Some(seed) => {
                let mut rng = baux::prng(seed);
                self.inner.restart_natural(slot, &center_view, &mut rng)
            }
            None => self
                .inner
                .restart_natural(slot, &center_view, &mut self.rng),
        }
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
        Ok(abandoned.into_iter().map(|identity| identity.0).collect())
    }

    /// Tell observations
    #[pyo3(signature = (x, y, seed=None, y_var=None))]
    fn tell(
        &mut self,
        x: PyReadonlyArray2<f64>,
        y: PyReadonlyArray2<f64>,
        seed: Option<u64>,
        y_var: Option<PyReadonlyArray2<f64>>,
    ) -> PyResult<()> {
        let x_view = x.as_array();
        let y_view = y.as_array();
        let yvar_view = y_var.as_ref().map(PyReadonlyArray2::as_array);
        match seed {
            Some(seed) => {
                let mut rng = baux::prng(seed);
                self.inner
                    .tell_natural(&x_view, &y_view, yvar_view.as_ref(), &mut rng)
            }
            None => self
                .inner
                .tell_natural(&x_view, &y_view, yvar_view.as_ref(), &mut self.rng),
        }
        .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Externally evaluated observations never complete a pending proposal.
    #[pyo3(signature = (x, y, seed=None, y_var=None))]
    fn import_observations(
        &mut self,
        x: PyReadonlyArray2<f64>,
        y: PyReadonlyArray2<f64>,
        seed: Option<u64>,
        y_var: Option<PyReadonlyArray2<f64>>,
    ) -> PyResult<()> {
        let x_view = x.as_array();
        let y_view = y.as_array();
        let yvar_view = y_var.as_ref().map(PyReadonlyArray2::as_array);
        match seed {
            Some(seed) => {
                let mut rng = baux::prng(seed);
                self.inner
                    .import_natural(&x_view, &y_view, yvar_view.as_ref(), &mut rng)
            }
            None => self
                .inner
                .import_natural(&x_view, &y_view, yvar_view.as_ref(), &mut self.rng),
        }
        .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Posterior mean of the fitted surrogate at `x` in natural units.
    fn posterior_mu<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<f64>,
    ) -> PyResult<Bound<'py, PyArrayDyn<f64>>> {
        let mu = self
            .inner
            .posterior_mu(&x.as_array())
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(mu.into_dyn().into_pyarray(py))
    }

    /// Whether this optimizer fits a surrogate.
    fn has_surrogate(&self) -> bool {
        self.inner.has_surrogate()
    }

    #[getter]
    fn ingestion_failure(&self) -> Option<String> {
        self.inner.ingestion_failure().map(str::to_owned)
    }

    /// Get init progress if in initialization phase
    fn init_progress(&self) -> Option<(usize, usize)> {
        self.inner.init_progress()
    }

    /// Get current telemetry
    fn telemetry(&self) -> PyTelemetry {
        let t = self.inner.telemetry();
        PyTelemetry {
            dt_fit: t.dt_fit,
            dt_gen: t.dt_gen,
            dt_sel: t.dt_sel,
            dt_tell: t.dt_tell,
            num_candidates: t.num_candidates,
        }
    }

    /// Number of retained trust-region observations.
    fn region_count(&self) -> usize {
        self.inner.obs_count()
    }

    fn num_regions(&self) -> usize {
        self.inner.num_regions()
    }

    /// Current trust-region length.
    fn tr_length(&self) -> f64 {
        self.inner.tr_length()
    }

    /// Get observations x in natural coordinates (if any).
    fn x_obs<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArrayDyn<f64>>> {
        self.inner.x_obs().map(|x| x.into_dyn().into_pyarray(py))
    }

    /// Get observation values y (if any).
    fn y_obs<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArrayDyn<f64>>> {
        self.inner.y_obs().map(|y| y.into_dyn().into_pyarray(py))
    }

    /// Get incumbent x in unit space (if any).
    fn unit_incumbent<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArrayDyn<f64>>> {
        self.inner
            .unit_incumbent()
            .map(|x| x.view().to_owned().into_dyn().into_pyarray(py))
    }

    /// Get incumbent x in natural coordinates (if any).
    fn incumbent_x<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArrayDyn<f64>>> {
        self.inner
            .incumbent_x()
            .map(|x| x.into_dyn().into_pyarray(py))
    }

    /// Get optimizer bounds.
    fn bounds<'py>(&self, py: Python<'py>) -> Bound<'py, PyArrayDyn<f64>> {
        self.inner
            .bounds()
            .view()
            .to_owned()
            .into_dyn()
            .into_pyarray(py)
    }
}

/// Telemetry data structure for Python
#[pyclass(name = "Telemetry", from_py_object)]
#[derive(Clone, Copy)]
pub struct PyTelemetry {
    #[pyo3(get)]
    pub dt_fit: f64,
    #[pyo3(get)]
    pub dt_gen: f64,
    #[pyo3(get)]
    pub dt_sel: f64,
    #[pyo3(get)]
    pub dt_tell: f64,
    #[pyo3(get)]
    pub num_candidates: usize,
}
