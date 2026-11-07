//! Synchronous resident Metal buffers; no DLPack or framework-owned GPU
//! tensors.

#![allow(clippy::too_many_arguments)]

use std::cell::Cell;
use std::rc::Rc;

use baux::TRLengthConfig;
use baux::experimental::{
    AcquisitionKind, AcquisitionParams, MetalSearchState, ProposalParams, RoundConfig,
    SurrogateParams,
};
use numpy::{IntoPyArray, PyArray1};
use pyo3::exceptions::{PyBufferError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;

mod py_buf;

use py_buf::err;
pub use py_buf::{PyMetalParamBlock, PyMetalProposals, PyMetalWeights};

#[pyclass(name = "Region", module = "baux", unsendable)]
pub struct PyMetalSearchState {
    pub(super) inner: MetalSearchState,
    pub(super) leases: Rc<Cell<usize>>,
}

impl PyMetalSearchState {
    fn unleased(&self) -> PyResult<()> {
        if self.leases.get() != 0 {
            return Err(PyBufferError::new_err(
                "Release incumbent Metal weight views before mutating search",
            ));
        }
        Ok(())
    }
}

#[pymethods]
impl PyMetalSearchState {
    #[new]
    #[pyo3(signature=(base,base_value,blocks,capacity,max_pending=1,base_variance=0.0,length_init=0.01,length_min=0.0001,length_max=0.08,failure_tolerance=None,sampler="gaussian",reference_seed=0))]
    fn new(
        base: PyRef<'_, PyMetalWeights>,
        base_value: f32,
        blocks: Vec<PyRef<'_, PyMetalParamBlock>>,
        capacity: usize,
        max_pending: usize,
        base_variance: f32,
        length_init: f64,
        length_min: f64,
        length_max: f64,
        failure_tolerance: Option<usize>,
        sampler: &str,
        reference_seed: u64,
    ) -> PyResult<Self> {
        if sampler != "gaussian" || reference_seed != 0 {
            return Err(PyValueError::new_err(
                "Metal search supports sampler='gaussian' with reference_seed=0; correlated reference search is unavailable",
            ));
        }
        let values =
            unsafe { std::slice::from_raw_parts(base.buffer.contents().cast::<u16>(), base.len) };
        let mut inner = MetalSearchState::new(
            values,
            base_value,
            base_variance,
            blocks.iter().map(|b| b.inner).collect(),
            capacity,
            max_pending,
            TRLengthConfig::new(length_init, length_min, length_max),
        )
        .map_err(err)?;
        if let Some(tolerance) = failure_tolerance {
            inner.set_tolerance(tolerance).map_err(err)?;
        }
        Ok(Self {
            inner,
            leases: Rc::new(Cell::new(0)),
        })
    }

    #[pyo3(name = "enable_paired_relative", signature=(failure_tolerance=4))]
    fn enable_relative(&mut self, failure_tolerance: usize) -> PyResult<()> {
        self.unleased()?;
        let _ = failure_tolerance;
        Err(PyValueError::new_err(
            "paired-relative Metal search is unavailable",
        ))
    }

    #[pyo3(signature=(arms=1,candidates=4,neighbors=1,seed=None,epistemic_scale=1.0,aleatoric_scale=0.0,y_scale=1.0,beta=1.0,acquisition="thompson",draw_seed=0))]
    fn ask(
        mut slf: PyRefMut<'_, Self>,
        arms: usize,
        candidates: usize,
        neighbors: usize,
        seed: Option<u64>,
        epistemic_scale: f32,
        aleatoric_scale: f32,
        y_scale: f32,
        beta: f32,
        acquisition: &str,
        draw_seed: u64,
    ) -> PyResult<PyMetalProposals> {
        slf.unleased()?;
        let seed = baux::resolve_seed(seed);
        let kind = AcquisitionKind::parse(acquisition).map_err(err)?;
        let config = RoundConfig {
            proposal: ProposalParams { radius: 0.0 },
            surrogate: SurrogateParams {
                neighbors,
                epistemic_scale,
                aleatoric_scale,
                output_scale: y_scale,
            },
            acquisition: AcquisitionParams {
                kind,
                beta,
                seed: draw_seed,
            },
        };
        let inner = slf
            .inner
            .ask_round(arms, candidates, seed, config)
            .map_err(err)?;
        Ok(PyMetalProposals {
            owner: slf.into(),
            inner,
        })
    }

    fn incumbent(&self) -> PyResult<PyMetalWeights> {
        // The returned buffer is read-only in Python and retains its
        // allocation.
        self.inner.best().map_err(err)?;
        self.leases.set(self.leases.get() + 1);
        Ok(PyMetalWeights {
            buffer: self.inner.base_buffer(),
            len: self.inner.len(),
            lease: Some(self.leases.clone()),
        })
    }

    #[pyo3(signature=(proposals, value, variance=0.0))]
    fn tell(
        &mut self,
        proposals: PyRef<'_, PyMetalProposals>,
        value: f32,
        variance: f32,
    ) -> PyResult<bool> {
        self.unleased()?;
        self.inner
            .tell_noisy(&proposals.inner, value, variance)
            .map(|d| d.accepted)
            .map_err(err)
    }

    #[pyo3(signature=(proposals, value, variance, incumbent_value, incumbent_variance, accept))]
    fn tell_paired(
        &mut self,
        proposals: PyRef<'_, PyMetalProposals>,
        value: f64,
        variance: f64,
        incumbent_value: f64,
        incumbent_variance: f64,
        accept: bool,
    ) -> PyResult<()> {
        self.unleased()?;
        if [value, variance, incumbent_value, incumbent_variance]
            .iter()
            .any(|v| !v.is_finite() || !(*v as f32).is_finite())
            || variance < 0.0
            || incumbent_variance < 0.0
        {
            return Err(PyValueError::new_err(
                "Paired values must be finite FP32; variances must be nonnegative",
            ));
        }
        let _ = (incumbent_value, incumbent_variance, accept);
        self.inner
            .tell_noisy(&proposals.inner, value as f32, variance as f32)
            .map(|_| ())
            .map_err(err)
    }

    fn sync(&mut self) -> PyResult<Vec<bool>> {
        self.unleased()?;
        self.inner.sync().map_err(err)
    }

    fn set_profiling(&mut self, enabled: bool) -> PyResult<()> {
        self.unleased()?;
        self.inner.set_profiling(enabled);
        Ok(())
    }

    #[getter]
    fn last_profile(&self) -> Option<(f32, f32, f32, f32)> {
        crate::py_util::format_profile(self.inner.last_profile())
    }

    fn memory_info<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let memory = self.inner.memory_info();
        let info = PyDict::new(py);
        info.set_item("row_bytes", memory.row_bytes)?;
        info.set_item("search_resident_bytes", memory.resident_bytes)?;
        info.set_item("device_allocated_bytes", memory.current_allocated_size)?;
        info.set_item(
            "recommended_working_set_bytes",
            memory.recommended_max_working_set_size,
        )?;
        info.set_item("max_buffer_bytes", memory.max_buffer_length)?;
        Ok(info)
    }

    fn controller_info<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let controller = self.inner.controller_info().map_err(err)?;
        let info = PyDict::new(py);
        info.set_item("dimensions", controller.dimensions)?;
        info.set_item("evaluated_arms", controller.evaluated_arms)?;
        info.set_item("length", controller.length)?;
        info.set_item("length_min", controller.length_min)?;
        info.set_item("length_max", controller.length_max)?;
        info.set_item("success_tolerance", controller.success_tolerance)?;
        info.set_item("failure_tolerance", controller.failure_tolerance)?;
        info.set_item("success_counter", controller.success_counter)?;
        info.set_item("failure_counter", controller.failure_counter)?;
        info.set_item("restarts", controller.restarts)?;
        Ok(info)
    }

    fn read_best<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<u16>>> {
        Ok(self.inner.read_best().map_err(err)?.into_pyarray(py))
    }

    fn read_reference<'py>(&mut self, _py: Python<'py>) -> PyResult<Bound<'py, PyArray1<u16>>> {
        self.unleased()?;
        Err(PyValueError::new_err(
            "Gaussian Metal search has no correlated reference",
        ))
    }

    #[getter]
    fn length(&self) -> PyResult<f64> {
        self.inner.length().map_err(err)
    }
    #[getter]
    fn best(&self) -> PyResult<f32> {
        self.inner.best().map_err(err)
    }
    #[getter]
    fn best_variance(&self) -> PyResult<f32> {
        self.inner.best_variance().map_err(err)
    }
    #[getter]
    fn history_len(&self) -> PyResult<usize> {
        self.inner.history_len().map_err(err)
    }
    #[getter]
    fn restarts(&self) -> PyResult<usize> {
        self.inner.restarts().map_err(err)
    }
}
