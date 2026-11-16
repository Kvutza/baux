//! Optimizer Python bindings.

#![allow(clippy::too_many_arguments)]

use numpy::{IntoPyArray, PyArrayDyn, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

mod build;
mod config;
mod dense;
mod regional;
pub use build::{bind_regions, create_py};
pub use dense::{PyDenseLinear, dense_apply, dense_dist2};

/// Python wrapper for BayesOptimizer
#[pyclass(name = "BayesOptimizer", frozen)]
#[derive(Clone)]
pub struct PyBayesOptimizer {
    pub(crate) inner: baux::BayesOptimizer,
    pub(crate) rng: baux::Prng,
}

const _: () = {
    fn assert_sync<T: Send + Sync>() {}
    fn check() {
        assert_sync::<PyBayesOptimizer>();
    }
};

#[pymethods]
impl PyBayesOptimizer {
    /// Ask for candidate points
    #[pyo3(signature = (num_arms=1, seed=None))]
    fn ask<'py>(
        &self,
        py: Python<'py>,
        num_arms: usize,
        seed: Option<u64>,
    ) -> PyResult<(Self, Bound<'py, PyArrayDyn<f64>>)> {
        let opt = self.inner.clone();
        let mut rng = self.rng.clone();
        let (next_opt, result) = py
            .detach(|| match seed {
                Some(seed) => {
                    let mut temp_rng = baux::prng(seed);
                    opt.ask_natural(num_arms, &mut temp_rng)
                }
                None => opt.ask_natural(num_arms, &mut rng),
            })
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let next_self = Self {
            inner: next_opt,
            rng,
        };
        Ok((next_self, result.into_dyn().into_pyarray(py)))
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
        &self,
        py: Python<'_>,
        ids: Vec<u64>,
        y: PyReadonlyArray2<f64>,
        seed: Option<u64>,
        y_var: Option<PyReadonlyArray2<f64>>,
    ) -> PyResult<Self> {
        let opt = self.inner.clone();
        let mut rng = self.rng.clone();
        let ids = ids
            .into_iter()
            .map(baux::optimizer::EvaluationId)
            .collect::<Vec<_>>();
        let y_view = y.as_array();
        let yvar_view = y_var.as_ref().map(PyReadonlyArray2::as_array);
        let next_opt = py
            .detach(|| match seed {
                Some(seed) => {
                    let mut temp_rng = baux::prng(seed);
                    opt.complete_ids(&ids, &y_view, yvar_view.as_ref(), &mut temp_rng)
                }
                None => opt.complete_ids(&ids, &y_view, yvar_view.as_ref(), &mut rng),
            })
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self {
            inner: next_opt,
            rng,
        })
    }

    /// Retire objective evaluations that produced no measured result.
    #[pyo3(signature = (ids, seed=None))]
    fn cancel_ids(&self, py: Python<'_>, ids: Vec<u64>, seed: Option<u64>) -> PyResult<Self> {
        let opt = self.inner.clone();
        let mut rng = self.rng.clone();
        let ids = ids
            .into_iter()
            .map(baux::optimizer::EvaluationId)
            .collect::<Vec<_>>();
        let next_opt = py
            .detach(|| match seed {
                Some(seed) => {
                    let mut temp_rng = baux::prng(seed);
                    opt.cancel_ids(&ids, &mut temp_rng)
                }
                None => opt.cancel_ids(&ids, &mut rng),
            })
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self {
            inner: next_opt,
            rng,
        })
    }

    /// Restart a regional slot using a center in natural parameter coordinates.
    /// Returns IDs abandoned from that slot; late results must be imported.
    #[pyo3(signature = (slot, center, seed=None))]
    fn restart_region(
        &self,
        py: Python<'_>,
        slot: usize,
        center: PyReadonlyArray1<f64>,
        seed: Option<u64>,
    ) -> PyResult<(Self, Vec<u64>)> {
        let opt = self.inner.clone();
        let mut rng = self.rng.clone();
        let center_view = center.as_array();
        let (next_opt, abandoned) = py
            .detach(|| match seed {
                Some(seed) => {
                    let mut temp_rng = baux::prng(seed);
                    opt.restart_natural(slot, &center_view, &mut temp_rng)
                }
                None => opt.restart_natural(slot, &center_view, &mut rng),
            })
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        let next_self = Self {
            inner: next_opt,
            rng,
        };
        Ok((
            next_self,
            abandoned.into_iter().map(|identity| identity.0).collect(),
        ))
    }

    /// Tell observations
    #[pyo3(signature = (x, y, seed=None, y_var=None))]
    fn tell(
        &self,
        py: Python<'_>,
        x: PyReadonlyArray2<f64>,
        y: PyReadonlyArray2<f64>,
        seed: Option<u64>,
        y_var: Option<PyReadonlyArray2<f64>>,
    ) -> PyResult<Self> {
        let opt = self.inner.clone();
        let mut rng = self.rng.clone();
        let x_view = x.as_array();
        let y_view = y.as_array();
        let yvar_view = y_var.as_ref().map(PyReadonlyArray2::as_array);
        let next_opt = py
            .detach(|| match seed {
                Some(seed) => {
                    let mut temp_rng = baux::prng(seed);
                    opt.tell_natural(&x_view, &y_view, yvar_view.as_ref(), &mut temp_rng)
                }
                None => opt.tell_natural(&x_view, &y_view, yvar_view.as_ref(), &mut rng),
            })
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self {
            inner: next_opt,
            rng,
        })
    }

    /// Externally evaluated observations never complete a pending proposal.
    #[pyo3(signature = (x, y, seed=None, y_var=None))]
    fn import_observations(
        &self,
        py: Python<'_>,
        x: PyReadonlyArray2<f64>,
        y: PyReadonlyArray2<f64>,
        seed: Option<u64>,
        y_var: Option<PyReadonlyArray2<f64>>,
    ) -> PyResult<Self> {
        let opt = self.inner.clone();
        let mut rng = self.rng.clone();
        let x_view = x.as_array();
        let y_view = y.as_array();
        let yvar_view = y_var.as_ref().map(PyReadonlyArray2::as_array);
        let next_opt = py
            .detach(|| match seed {
                Some(seed) => {
                    let mut temp_rng = baux::prng(seed);
                    opt.import_natural(&x_view, &y_view, yvar_view.as_ref(), &mut temp_rng)
                }
                None => opt.import_natural(&x_view, &y_view, yvar_view.as_ref(), &mut rng),
            })
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self {
            inner: next_opt,
            rng,
        })
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

    /// Number of retained observations across the optimizer.
    fn num_observations(&self) -> usize {
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

    /// Get learned surrogate lengthscales (if any).
    fn lengthscales<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArrayDyn<f64>>> {
        self.inner
            .lengthscales()
            .map(|ls| ls.into_dyn().into_pyarray(py))
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

    /// Fork an isolated optimizer clone sharing immutable storage.
    fn fork(&self) -> PyResult<Self> {
        Ok(self.clone())
    }

    /// Hex-encoded BLAKE3 digest of the optimizer state.
    fn digest_hex(&self) -> String {
        self.inner.digest_hex()
    }

    /// Hex-encoded BLAKE3 digest folding Python cursor state.
    fn digest_host(&self, seed: u64, epoch: u64, identified: bool) -> String {
        self.inner.digest_host(seed, epoch, identified)
    }

    /// Serialize optimizer state to a portable binary checkpoint.
    fn freeze<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, pyo3::types::PyBytes>> {
        let bytes = self
            .inner
            .freeze()
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(pyo3::types::PyBytes::new(py, &bytes))
    }

    /// Serialize optimizer state and Python cursor metadata to a portable
    /// binary checkpoint.
    fn freeze_host<'py>(
        &self,
        py: Python<'py>,
        seed: u64,
        epoch: u64,
        identified: bool,
        id_map: Vec<(u64, u64)>,
    ) -> PyResult<Bound<'py, pyo3::types::PyBytes>> {
        let host = baux::optimizer::HostCursor {
            seed,
            epoch,
            identified,
            id_map,
        };
        let bytes = self
            .inner
            .freeze_host(&host)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(pyo3::types::PyBytes::new(py, &bytes))
    }

    /// Restore optimizer state from a binary checkpoint.
    fn thaw(&self, data: &[u8]) -> PyResult<Self> {
        let next = self
            .inner
            .thaw(data)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self {
            inner: next,
            rng: self.rng.clone(),
        })
    }

    /// Restore optimizer state and Python cursor metadata from a binary
    /// checkpoint.
    #[pyo3(signature = (data, default_seed=0, default_epoch=0, default_identified=false))]
    fn thaw_host(
        &self,
        data: &[u8],
        default_seed: u64,
        default_epoch: u64,
        default_identified: bool,
    ) -> PyResult<(Self, (u64, u64, bool, Vec<(u64, u64)>))> {
        let (next, host) = self
            .inner
            .thaw_host(data)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let meta = match host {
            Some(h) => (h.seed, h.epoch, h.identified, h.id_map),
            None => (default_seed, default_epoch, default_identified, Vec::new()),
        };
        let next_self = Self {
            inner: next,
            rng: self.rng.clone(),
        };
        Ok((next_self, meta))
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
