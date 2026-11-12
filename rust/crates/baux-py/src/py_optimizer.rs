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
        let result = py
            .detach(|| match seed {
                Some(seed) => {
                    let mut rng = baux::prng(seed);
                    self.inner.ask_natural(num_arms, &mut rng)
                }
                None => self.inner.ask_natural(num_arms, &mut self.rng),
            })
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
        py: Python<'_>,
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
        py.detach(|| match seed {
            Some(seed) => {
                let mut rng = baux::prng(seed);
                self.inner
                    .complete_ids(&ids, &y_view, yvar_view.as_ref(), &mut rng)
            }
            None => self
                .inner
                .complete_ids(&ids, &y_view, yvar_view.as_ref(), &mut self.rng),
        })
        .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Retire objective evaluations that produced no measured result.
    #[pyo3(signature = (ids, seed=None))]
    fn cancel_ids(&mut self, py: Python<'_>, ids: Vec<u64>, seed: Option<u64>) -> PyResult<()> {
        let ids = ids
            .into_iter()
            .map(baux::optimizer::EvaluationId)
            .collect::<Vec<_>>();
        py.detach(|| match seed {
            Some(seed) => {
                let mut rng = baux::prng(seed);
                self.inner.cancel_ids(&ids, &mut rng)
            }
            None => self.inner.cancel_ids(&ids, &mut self.rng),
        })
        .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Restart a regional slot using a center in natural parameter coordinates.
    /// Returns IDs abandoned from that slot; late results must be imported.
    #[pyo3(signature = (slot, center, seed=None))]
    fn restart_region(
        &mut self,
        py: Python<'_>,
        slot: usize,
        center: PyReadonlyArray1<f64>,
        seed: Option<u64>,
    ) -> PyResult<Vec<u64>> {
        let center_view = center.as_array();
        let abandoned = py
            .detach(|| match seed {
                Some(seed) => {
                    let mut rng = baux::prng(seed);
                    self.inner.restart_natural(slot, &center_view, &mut rng)
                }
                None => self
                    .inner
                    .restart_natural(slot, &center_view, &mut self.rng),
            })
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        Ok(abandoned.into_iter().map(|identity| identity.0).collect())
    }

    /// Tell observations
    #[pyo3(signature = (x, y, seed=None, y_var=None))]
    fn tell(
        &mut self,
        py: Python<'_>,
        x: PyReadonlyArray2<f64>,
        y: PyReadonlyArray2<f64>,
        seed: Option<u64>,
        y_var: Option<PyReadonlyArray2<f64>>,
    ) -> PyResult<()> {
        let x_view = x.as_array();
        let y_view = y.as_array();
        let yvar_view = y_var.as_ref().map(PyReadonlyArray2::as_array);
        py.detach(|| match seed {
            Some(seed) => {
                let mut rng = baux::prng(seed);
                self.inner
                    .tell_natural(&x_view, &y_view, yvar_view.as_ref(), &mut rng)
            }
            None => self
                .inner
                .tell_natural(&x_view, &y_view, yvar_view.as_ref(), &mut self.rng),
        })
        .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Externally evaluated observations never complete a pending proposal.
    #[pyo3(signature = (x, y, seed=None, y_var=None))]
    fn import_observations(
        &mut self,
        py: Python<'_>,
        x: PyReadonlyArray2<f64>,
        y: PyReadonlyArray2<f64>,
        seed: Option<u64>,
        y_var: Option<PyReadonlyArray2<f64>>,
    ) -> PyResult<()> {
        let x_view = x.as_array();
        let y_view = y.as_array();
        let yvar_view = y_var.as_ref().map(PyReadonlyArray2::as_array);
        py.detach(|| match seed {
            Some(seed) => {
                let mut rng = baux::prng(seed);
                self.inner
                    .import_natural(&x_view, &y_view, yvar_view.as_ref(), &mut rng)
            }
            None => self
                .inner
                .import_natural(&x_view, &y_view, yvar_view.as_ref(), &mut self.rng),
        })
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
        let inner = self
            .inner
            .fork()
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self {
            inner,
            rng: self.rng.clone(),
        })
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
    fn thaw(&mut self, data: &[u8]) -> PyResult<()> {
        self.inner = self
            .inner
            .thaw(data)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(())
    }

    /// Restore optimizer state and Python cursor metadata from a binary
    /// checkpoint.
    fn thaw_host(
        &mut self,
        data: &[u8],
        default_seed: u64,
        default_epoch: u64,
        default_identified: bool,
    ) -> PyResult<(u64, u64, bool, Vec<(u64, u64)>)> {
        let (next, host) = self
            .inner
            .thaw_host(data)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        self.inner = next;
        match host {
            Some(h) => Ok((h.seed, h.epoch, h.identified, h.id_map)),
            None => Ok((default_seed, default_epoch, default_identified, Vec::new())),
        }
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
