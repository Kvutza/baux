#![allow(clippy::too_many_arguments)]

use std::cell::Cell;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use baux::TRLengthConfig;
use baux::experimental::{
    AcquisitionKind, AcquisitionParams, ParamBlock, ProposalParams, Proposals as CoreProposals,
    RoundConfig, SearchState, SurrogateParams,
};
use numpy::{IntoPyArray, PyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

mod checks;

use checks::{Sampler, err};

type PyObject = Py<PyAny>;

#[pyclass(name = "ParamBlock", frozen, from_py_object)]
#[derive(Clone)]
pub struct PyParamBlock {
    pub(crate) inner: ParamBlock,
}

#[pymethods]
impl PyParamBlock {
    #[new]
    #[pyo3(signature=(key,offset,length,scale,weight=1.0))]
    fn new(key: u64, offset: usize, length: usize, scale: f32, weight: f32) -> PyResult<Self> {
        Ok(Self {
            inner: ParamBlock::new(key, offset, length, scale, weight).map_err(err)?,
        })
    }

    #[getter]
    fn key(&self) -> u64 {
        self.inner.key
    }

    #[getter]
    fn offset(&self) -> usize {
        self.inner.offset
    }

    #[getter]
    fn length(&self) -> usize {
        self.inner.len
    }

    #[getter]
    fn scale(&self) -> f32 {
        self.inner.scale
    }

    #[getter]
    fn weight(&self) -> f32 {
        self.inner.weight
    }
}

/// CUDA BF16 search. Independent sign (default) and Gaussian sampling use
/// TuRBO. Correlated Gaussian sampling requires one arm and four candidates;
/// its accepted-radius controller replaces TuRBO failure/success counters.
#[pyclass(name = "SearchState", unsendable)]
pub struct PySearchState {
    inner: SearchState,
    exports: Arc<AtomicUsize>,
    scratch_generation: u64,
    consumer_sync: Cell<bool>,
}

#[pymethods]
impl PySearchState {
    #[new]
    #[pyo3(signature=(base,base_value,blocks,capacity,max_pending=1,base_variance=0.0,length_init=0.8,length_min=0.0078125,length_max=1.6,failure_tolerance=None,sampler="independent",reference_seed=0))]
    fn new(
        base: &Bound<'_, PyAny>,
        base_value: f32,
        blocks: Vec<PyRef<'_, PyParamBlock>>,
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
        let sampler =
            checks::check_sampler(sampler, max_pending, failure_tolerance).map_err(err)?;
        let input = crate::dlpack::Input::new(base)?;
        let blocks = blocks.iter().map(|block| block.inner).collect();
        let mut inner = unsafe {
            SearchState::from_device(
                input.pointer,
                input.len,
                base_value,
                base_variance,
                blocks,
                capacity,
                max_pending,
                TRLengthConfig::new(length_init, length_min, length_max),
            )
        }
        .map_err(err)?;
        match sampler {
            Sampler::Independent => {}
            Sampler::Gaussian => inner.enable_gaussian().map_err(err)?,
            Sampler::Correlated => inner.enable_correlated(reference_seed).map_err(err)?,
        }
        if let Some(tolerance) = failure_tolerance {
            inner.set_tolerance(tolerance).map_err(err)?;
        }
        Ok(Self {
            inner,
            exports: Arc::new(AtomicUsize::new(0)),
            scratch_generation: 0,
            consumer_sync: Cell::new(false),
        })
    }

    #[pyo3(signature=(arms,candidates,neighbors,seed,epistemic_scale=0.7,aleatoric_scale=0.05,y_scale=1.0,beta=1.0,acquisition="ucb",draw_seed=0))]
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
    ) -> PyResult<PyProposals> {
        slf.ensure_idle()?;
        slf.scratch_generation = slf.scratch_generation.wrapping_add(1);
        let inner = slf
            .inner
            .ask_round(
                arms,
                candidates,
                seed,
                RoundConfig {
                    proposal: ProposalParams { radius: 0.0 },
                    surrogate: SurrogateParams {
                        neighbors,
                        epistemic_scale,
                        aleatoric_scale,
                        output_scale: y_scale,
                    },
                    acquisition: AcquisitionParams {
                        kind: AcquisitionKind::parse(acquisition).map_err(err)?,
                        beta,
                        seed: draw_seed,
                    },
                },
            )
            .map_err(err)?;
        let py = slf.py();
        Ok(PyProposals {
            owner: slf.into_pyobject(py).unwrap().into_any().unbind(),
            inner,
        })
    }

    /// Snapshot the incumbent into unused pending storage, with shape (1, N).
    /// The returned exporter is single-use; release the consumer before ask.
    fn incumbent(mut slf: PyRefMut<'_, Self>) -> PyResult<PyIncumbent> {
        slf.ensure_idle()?;
        slf.inner.snapshot_incumbent().map_err(err)?;
        slf.scratch_generation = slf.scratch_generation.wrapping_add(1);
        let generation = slf.scratch_generation;
        let py = slf.py();
        Ok(PyIncumbent {
            owner: slf.into_pyobject(py).unwrap().into_any().unbind(),
            generation,
            exported: false,
        })
    }

    /// Enable zero-anchored paired-relative history before the first round.
    #[pyo3(name = "enable_paired_relative", signature=(failure_tolerance=4))]
    fn enable_relative(&mut self, failure_tolerance: usize) -> PyResult<()> {
        self.ensure_idle()?;
        self.inner.enable_relative(failure_tolerance).map_err(err)
    }

    #[pyo3(name = "tell_paired_relative", signature=(proposals,value,variance,incumbent_value,incumbent_variance,improvement,improvement_variance,accept,*,reject_is_failure=true))]
    fn tell_relative(
        &mut self,
        proposals: PyRef<'_, PyProposals>,
        value: f64,
        variance: f64,
        incumbent_value: f64,
        incumbent_variance: f64,
        improvement: f64,
        improvement_variance: f64,
        accept: bool,
        reject_is_failure: bool,
    ) -> PyResult<()> {
        self.ensure_idle()?;
        let [
            value,
            variance,
            incumbent_value,
            incumbent_variance,
            improvement,
            improvement_variance,
        ] = checks::check_scores(
            value,
            variance,
            incumbent_value,
            incumbent_variance,
            improvement,
            improvement_variance,
        )
        .map_err(err)?;
        self.inner
            .tell_relative(
                &proposals.inner,
                value,
                variance,
                incumbent_value,
                incumbent_variance,
                improvement,
                improvement_variance,
                accept,
                reject_is_failure,
            )
            .map_err(err)
    }

    /// Queue an explicit paired decision; collect its result with sync().
    #[pyo3(signature=(proposals,value,variance,incumbent_value,incumbent_variance,accept))]
    fn tell_paired(
        &mut self,
        proposals: PyRef<'_, PyProposals>,
        value: f64,
        variance: f64,
        incumbent_value: f64,
        incumbent_variance: f64,
        accept: bool,
    ) -> PyResult<()> {
        self.ensure_idle()?;
        let [value, variance, incumbent_value, incumbent_variance] =
            checks::check_pair(value, variance, incumbent_value, incumbent_variance)
                .map_err(err)?;
        self.inner
            .tell_paired(
                &proposals.inner,
                value,
                variance,
                incumbent_value,
                incumbent_variance,
                accept,
            )
            .map_err(err)
    }

    #[pyo3(signature=(proposals,values,variances=None))]
    fn tell(
        &mut self,
        proposals: PyRef<'_, PyProposals>,
        values: &Bound<'_, PyAny>,
        variances: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<()> {
        self.ensure_idle()?;
        if values.hasattr("__dlpack__")? {
            let values = crate::dlpack::Input::f32(values)?;
            if values.len != proposals.inner.arms() {
                return Err(PyValueError::new_err(
                    "device rewards must match the proposal count",
                ));
            }
            let variances = variances.map(crate::dlpack::Input::f32).transpose()?;
            if variances
                .as_ref()
                .is_some_and(|input| input.len != proposals.inner.arms())
            {
                return Err(PyValueError::new_err(
                    "device variances must match the proposal count",
                ));
            }
            unsafe {
                self.inner.finish_round(
                    &proposals.inner,
                    values.pointer,
                    variances.as_ref().map(|input| input.pointer),
                )
            }
            .map_err(err)?;
            return Ok(());
        }
        let values = values.extract::<Vec<f32>>()?;
        let variances = variances
            .map(|input| input.extract::<Vec<f32>>())
            .transpose()?
            .unwrap_or_else(|| vec![0.0; values.len()]);
        self.inner
            .queue_round(&proposals.inner, &values, &variances)
            .map_err(err)?;
        Ok(())
    }

    fn sync(&mut self) -> PyResult<Vec<bool>> {
        self.ensure_idle()?;
        self.inner.sync().map_err(err)
    }

    fn profile(&mut self, enabled: bool) -> PyResult<()> {
        self.ensure_idle()?;
        self.inner.set_profiling(enabled);
        Ok(())
    }

    /// Copy the accepted model to host memory as raw BF16 bits.
    fn read_best<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<u16>>> {
        self.ensure_idle()?;
        Ok(self.inner.read_best().map_err(err)?.into_pyarray(py))
    }

    /// Copy the correlated reference to host as raw BF16 bits (a model-sized
    /// copy). Other sampler modes return an error. Intended for explicit
    /// validation only.
    fn read_reference<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<u16>>> {
        self.ensure_idle()?;
        Ok(self.inner.read_reference().map_err(err)?.into_pyarray(py))
    }

    #[getter]
    fn last_profile(&self) -> Option<(f32, f32, f32, f32)> {
        crate::py_util::format_profile(self.inner.last_profile())
    }

    #[getter]
    fn length(&mut self) -> PyResult<f64> {
        self.ensure_idle()?;
        self.inner.length().map_err(err)
    }

    #[getter]
    fn best(&mut self) -> PyResult<f32> {
        self.ensure_idle()?;
        self.inner.best().map_err(err)
    }

    #[getter]
    fn best_variance(&mut self) -> PyResult<f32> {
        self.ensure_idle()?;
        self.inner.best_variance().map_err(err)
    }

    #[getter]
    fn restarts(&mut self) -> PyResult<usize> {
        self.ensure_idle()?;
        self.inner.restarts().map_err(err)
    }

    #[getter]
    fn history_len(&mut self) -> PyResult<usize> {
        self.ensure_idle()?;
        self.inner.history_len().map_err(err)
    }

    #[getter]
    fn len(&self) -> usize {
        self.inner.len()
    }
}

#[pyclass(name = "Incumbent", unsendable)]
pub struct PyIncumbent {
    owner: PyObject,
    generation: u64,
    exported: bool,
}

#[pymethods]
impl PyIncumbent {
    fn __dlpack_device__(&self) -> (i32, i32) {
        (2, 0)
    }

    #[pyo3(signature=(stream=None,max_version=None,dl_device=None,copy=None))]
    fn __dlpack__(
        mut slf: PyRefMut<'_, Self>,
        stream: Option<i64>,
        max_version: Option<(u32, u32)>,
        dl_device: Option<(i32, i32)>,
        copy: Option<bool>,
    ) -> PyResult<PyObject> {
        checks::check_export(stream, dl_device, copy)?;
        let py = slf.py();
        let (pointer, rows, columns, lease) = {
            let search = slf.owner.bind(py).extract::<PyRef<'_, PySearchState>>()?;
            search.ensure_idle()?;
            checks::check_handle(slf.exported, slf.generation, search.scratch_generation)
                .map_err(err)?;
            let (pointer, rows, columns) = search.inner.device_incumbent(stream).map_err(err)?;
            search.consumer_sync.set(true);
            (pointer, rows, columns, Arc::clone(&search.exports))
        };
        // Even a released legacy consumer could have written to scratch. Never
        // export the same snapshot twice; a fresh incumbent() copies row zero
        // again.
        slf.exported = true;
        let owner = slf.into_pyobject(py).unwrap().into_any().unbind();
        checks::export_batch(py, owner, lease, pointer, rows, columns, max_version)
    }
}

#[pyclass(name = "Proposals", unsendable)]
pub struct PyProposals {
    owner: PyObject,
    inner: CoreProposals,
}

#[pymethods]
impl PyProposals {
    /// Selected candidate index and Gaussian persistence per arm.
    /// Persistence is 0.75 for correlated indices 0/1 and 0 otherwise.
    fn geometry(&self, py: Python<'_>) -> PyResult<Vec<(usize, f32)>> {
        let search = self.owner.bind(py).extract::<PyRef<'_, PySearchState>>()?;
        search.inner.geometry(&self.inner).map_err(err)
    }

    /// Seed, acquisition score, radius, and (changed count, squared L2) per
    /// block.
    fn describe(&self, py: Python<'_>) -> PyResult<Vec<(u64, f32, f32, Vec<(u64, f64)>)>> {
        let search = self.owner.bind(py).extract::<PyRef<'_, PySearchState>>()?;
        search.inner.describe(&self.inner).map_err(err)
    }

    #[getter]
    fn arms(&self) -> usize {
        self.inner.arms()
    }

    fn __dlpack_device__(&self) -> (i32, i32) {
        (2, 0)
    }

    #[pyo3(signature=(stream=None,max_version=None,dl_device=None,copy=None))]
    fn __dlpack__(
        slf: PyRef<'_, Self>,
        stream: Option<i64>,
        max_version: Option<(u32, u32)>,
        dl_device: Option<(i32, i32)>,
        copy: Option<bool>,
    ) -> PyResult<PyObject> {
        checks::check_export(stream, dl_device, copy)?;
        let py = slf.py();
        let (pointer, rows, columns, lease) = {
            let mut search = slf
                .owner
                .bind(py)
                .extract::<PyRefMut<'_, PySearchState>>()?;
            search.ensure_idle()?;
            let (pointer, rows, columns) =
                search.inner.device_round(&slf.inner, stream).map_err(err)?;
            search.consumer_sync.set(true);
            (pointer, rows, columns, Arc::clone(&search.exports))
        };
        let owner = slf.into_pyobject(py).unwrap().into_any().unbind();
        checks::export_batch(py, owner, lease, pointer, rows, columns, max_version)
    }
}

impl PySearchState {
    fn ensure_idle(&self) -> PyResult<()> {
        if self.exports.load(Ordering::Acquire) == 0 {
            if self.consumer_sync.get() {
                self.inner.sync_consumers().map_err(err)?;
                self.consumer_sync.set(false);
            }
            Ok(())
        } else {
            Err(PyValueError::new_err(
                "release live JAX proposals or incumbent exports before mutating the search",
            ))
        }
    }
}
