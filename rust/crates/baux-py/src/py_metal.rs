//! Synchronous resident Metal buffers; no DLPack or framework-owned GPU
//! tensors.

#![allow(clippy::too_many_arguments)]

use std::cell::Cell;
use std::rc::Rc;

use baux::TRLengthConfig;
use baux::experimental::{
    AcquisitionKind, AcquisitionParams, MetalProposals, MetalSearchState, ProposalParams,
    RoundConfig, SurrogateParams,
};
use numpy::{IntoPyArray, PyArray1};
use pyo3::exceptions::{PyBufferError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;

mod py_buf;

pub(crate) use py_buf::weight_buffer;
pub use py_buf::{PyMetalFlameEvaluator, PyMetalParamBlock, PyMetalWeights};

use crate::py_qwen::PyMetalQwenEvaluator;

pub(super) fn err(message: String) -> PyErr {
    PyValueError::new_err(message)
}

fn round_config(
    neighbors: usize,
    epistemic_scale: f32,
    aleatoric_scale: f32,
    output_scale: f32,
    beta: f32,
    kind: AcquisitionKind,
    seed: u64,
) -> RoundConfig {
    RoundConfig {
        proposal: ProposalParams { radius: 0.0 },
        surrogate: SurrogateParams {
            neighbors,
            epistemic_scale,
            aleatoric_scale,
            output_scale,
        },
        acquisition: AcquisitionParams { kind, beta, seed },
    }
}

#[pyclass(name = "MetalSearchState", module = "baux.experimental", unsendable)]
pub struct PyMetalSearchState {
    pub(super) inner: MetalSearchState,
    leases: Rc<Cell<usize>>,
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
    #[pyo3(signature=(base,base_value,blocks,capacity,max_pending=1,base_variance=0.0,length_init=0.01,length_min=0.0001,length_max=0.08,failure_tolerance=None,sampler="correlated",reference_seed=0))]
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
        if sampler != "correlated" || failure_tolerance.is_some() {
            return Err(PyValueError::new_err(
                "Metal search requires sampler='correlated' and failure_tolerance=None; the \
                 shared TuRBO controller derives its tolerance from dimension",
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
        inner.correlate(reference_seed).map_err(err)?;
        Ok(Self {
            inner,
            leases: Rc::new(Cell::new(0)),
        })
    }

    #[pyo3(name = "enable_paired_relative", signature=(failure_tolerance=4))]
    fn enable_relative(&mut self, failure_tolerance: usize) -> PyResult<()> {
        self.unleased()?;
        self.inner.enable_relative(failure_tolerance).map_err(err)
    }

    #[pyo3(signature=(arms,candidates,neighbors,seed,epistemic_scale=1.0,aleatoric_scale=0.0,y_scale=1.0,beta=1.0,acquisition="thompson",draw_seed=0))]
    fn ask(
        mut slf: PyRefMut<'_, Self>,
        arms: usize,
        candidates: usize,
        neighbors: usize,
        seed: u64,
        epistemic_scale: f32,
        aleatoric_scale: f32,
        y_scale: f32,
        beta: f32,
        acquisition: &str,
        draw_seed: u64,
    ) -> PyResult<PyMetalProposals> {
        slf.unleased()?;
        let inner = slf
            .inner
            .ask_round(
                arms,
                candidates,
                seed,
                round_config(
                    neighbors,
                    epistemic_scale,
                    aleatoric_scale,
                    y_scale,
                    beta,
                    AcquisitionKind::parse(acquisition).map_err(err)?,
                    draw_seed,
                ),
            )
            .map_err(err)?;
        Ok(PyMetalProposals {
            owner: slf.into(),
            inner,
        })
    }

    #[pyo3(signature=(evaluator,prompts,max_new_tokens,arms=1,candidates=4,neighbors=2,seed=0,draw_seed=0))]
    fn ask_generate(
        mut slf: PyRefMut<'_, Self>,
        mut evaluator: PyRefMut<'_, PyMetalQwenEvaluator>,
        prompts: Vec<Vec<i32>>,
        max_new_tokens: usize,
        arms: usize,
        candidates: usize,
        neighbors: usize,
        seed: u64,
        draw_seed: u64,
    ) -> PyResult<(PyMetalProposals, Vec<Vec<i32>>)> {
        slf.unleased()?;
        let buffer = slf
            .inner
            .begin_ask(
                arms,
                candidates,
                seed,
                round_config(
                    neighbors,
                    1.0,
                    0.0,
                    1.0,
                    1.0,
                    AcquisitionKind::Thompson,
                    draw_seed,
                ),
            )
            .map_err(err)?;
        let generated = match evaluator
            .inner
            .generate_batch(&buffer, &prompts, max_new_tokens)
        {
            Ok(generated) => generated,
            Err(error) => {
                slf.inner.abort_ask();
                return Err(err(error));
            }
        };
        let inner = slf.inner.finish_ask().map_err(err)?;
        Ok((
            PyMetalProposals {
                owner: slf.into(),
                inner,
            },
            generated,
        ))
    }

    #[pyo3(signature=(evaluator,tokens,masks,arms=1,candidates=4,neighbors=2,seed=0,draw_seed=0))]
    fn ask_losses(
        mut slf: PyRefMut<'_, Self>,
        mut evaluator: PyRefMut<'_, PyMetalQwenEvaluator>,
        tokens: Vec<Vec<i32>>,
        masks: Vec<Vec<bool>>,
        arms: usize,
        candidates: usize,
        neighbors: usize,
        seed: u64,
        draw_seed: u64,
    ) -> PyResult<(PyMetalProposals, Vec<f32>)> {
        slf.unleased()?;
        let buffer = slf
            .inner
            .begin_ask(
                arms,
                candidates,
                seed,
                round_config(
                    neighbors,
                    1.0,
                    0.0,
                    1.0,
                    1.0,
                    AcquisitionKind::Thompson,
                    draw_seed,
                ),
            )
            .map_err(err)?;
        let losses = match evaluator.inner.losses(&buffer, &tokens, &masks) {
            Ok(losses) => losses,
            Err(error) => {
                slf.inner.abort_ask();
                return Err(err(error));
            }
        };
        let inner = slf.inner.finish_ask().map_err(err)?;
        Ok((
            PyMetalProposals {
                owner: slf.into(),
                inner,
            },
            losses,
        ))
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
        self.inner
            .tell_paired(
                &proposals.inner,
                value as f32,
                variance as f32,
                incumbent_value as f32,
                incumbent_variance as f32,
                accept,
            )
            .map_err(err)
    }

    #[pyo3(name = "tell_paired_relative", signature=(proposals, value, variance, incumbent_value, incumbent_variance, improvement, improvement_variance, accept, *, reject_is_failure=true))]
    fn tell_relative(
        &mut self,
        proposals: PyRef<'_, PyMetalProposals>,
        value: f64,
        variance: f64,
        incumbent_value: f64,
        incumbent_variance: f64,
        improvement: f64,
        improvement_variance: f64,
        accept: bool,
        reject_is_failure: bool,
    ) -> PyResult<()> {
        self.unleased()?;
        let values = [
            value,
            variance,
            incumbent_value,
            incumbent_variance,
            improvement,
            improvement_variance,
        ];
        if values
            .iter()
            .any(|v| !v.is_finite() || !(*v as f32).is_finite())
            || [variance, incumbent_variance, improvement_variance]
                .iter()
                .any(|v| *v < 0.0)
        {
            return Err(PyValueError::new_err(
                "Paired values must be finite FP32; variances must be nonnegative",
            ));
        }
        self.inner
            .tell_relative(
                &proposals.inner,
                value as f32,
                variance as f32,
                incumbent_value as f32,
                incumbent_variance as f32,
                improvement as f32,
                improvement_variance as f32,
                accept,
                reject_is_failure,
            )
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
        if let Some(reliability) = self.inner.reliability_info().map_err(err)? {
            info.set_item("reliability_action", format!("{:?}", reliability.action))?;
            info.set_item("standardized_error", reliability.standardized_error)?;
            info.set_item("sharpness_ratio", reliability.sharpness_ratio)?;
            info.set_item("agreement_ratio", reliability.agreement_ratio)?;
            info.set_item("adequacy_mean", reliability.adequacy_mean)?;
            info.set_item("adequacy_lower", reliability.adequacy_lower)?;
            info.set_item("reliability_evidence", reliability.evidence)?;
            info.set_item("response_radius", reliability.response_radius)?;
            info.set_item("response_boundary", reliability.response_boundary)?;
            info.set_item("boundary_fraction", reliability.boundary_fraction)?;
            info.set_item("radius_conversion", reliability.conversion)?;
            info.set_item("reliability_length", reliability.length)?;
        }
        Ok(info)
    }

    fn read_best<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<u16>>> {
        Ok(self.inner.read_best().map_err(err)?.into_pyarray(py))
    }

    fn read_reference<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<u16>>> {
        self.unleased()?;
        Ok(self.inner.read_reference().map_err(err)?.into_pyarray(py))
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

#[pyclass(name = "MetalProposals", module = "baux.experimental", unsendable)]
pub struct PyMetalProposals {
    pub(super) owner: Py<PyMetalSearchState>,
    pub(super) inner: MetalProposals,
}

#[pymethods]
impl PyMetalProposals {
    fn describe(&self, py: Python<'_>) -> PyResult<Vec<(u64, f32, f32, Vec<(u64, f64)>)>> {
        self.owner
            .bind(py)
            .try_borrow()?
            .inner
            .describe(&self.inner)
            .map_err(err)
    }

    fn geometry(&self, py: Python<'_>) -> PyResult<Vec<(usize, f32)>> {
        self.owner
            .bind(py)
            .try_borrow()?
            .inner
            .geometry(&self.inner)
            .map_err(err)
    }

    fn base_id(&self, py: Python<'_>) -> PyResult<i64> {
        self.owner
            .bind(py)
            .try_borrow()?
            .inner
            .base_id(&self.inner)
            .map_err(err)
    }

    fn history_dists(&self, py: Python<'_>) -> PyResult<Vec<(i64, f32)>> {
        self.owner
            .bind(py)
            .try_borrow()?
            .inner
            .history_dists(&self.inner)
            .map_err(err)
    }

    fn pool(&self, py: Python<'_>) -> PyResult<Vec<(usize, u64, f32, f32, Vec<(i64, f32)>)>> {
        self.owner
            .bind(py)
            .try_borrow()?
            .inner
            .pool(&self.inner)
            .map_err(err)
    }

    fn pool_geometry(
        &self,
        py: Python<'_>,
    ) -> PyResult<(Vec<f32>, Vec<(usize, usize, Option<f32>)>, Vec<Option<f32>>)> {
        self.owner
            .bind(py)
            .try_borrow()?
            .inner
            .pool_geometry(&self.inner)
            .map_err(err)
    }

    fn read<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<u16>>> {
        let owner = self.owner.bind(py).try_borrow()?;
        let buffer = owner.inner.propose_buffer(&self.inner).map_err(err)?;
        let values = unsafe {
            std::slice::from_raw_parts(buffer.contents().cast::<u16>(), owner.inner.len())
        };
        Ok(values.to_vec().into_pyarray(py))
    }
}
