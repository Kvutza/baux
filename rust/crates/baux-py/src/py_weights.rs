#![allow(clippy::too_many_arguments)]

use crate::py_parameter::{PyParameter, parameters};
use baux::TRLengthConfig;
use baux::experimental::{
    AcquisitionKind, AcquisitionParams, ComputeDevice, ProposalParams, RoundConfig, SurrogateParams,
};
use baux::search;
use numpy::{
    Element, IntoPyArray, PyArray1, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

mod py_dense;
mod py_sparse;

pub use py_dense::*;
pub use py_sparse::*;

pub(super) fn err(error: String) -> PyErr {
    PyValueError::new_err(error)
}

fn round_config(
    radius: f32,
    neighbors: usize,
    epistemic_scale: f32,
    aleatoric_scale: f32,
    output_scale: f32,
    beta: f32,
    kind: AcquisitionKind,
    seed: u64,
) -> RoundConfig {
    RoundConfig {
        proposal: ProposalParams { radius },
        surrogate: SurrogateParams {
            neighbors,
            epistemic_scale,
            aleatoric_scale,
            output_scale,
        },
        acquisition: AcquisitionParams { kind, beta, seed },
    }
}

pub(super) fn array1_vec<T: Copy + Element>(array: PyReadonlyArray1<'_, T>) -> Vec<T> {
    array.as_array().iter().copied().collect()
}

pub(super) fn array2_vec<T: Copy + Element>(array: &PyReadonlyArray2<'_, T>) -> Vec<T> {
    array.as_array().iter().copied().collect()
}

#[pyclass(name = "TrialEngine", module = "baux.search", unsendable)]
pub struct PyTrialEngine {
    pub(crate) inner: search::TrialEngine,
}

#[pymethods]
impl PyTrialEngine {
    #[new]
    #[pyo3(signature=(base,base_value,leaves,capacity,device="auto"))]
    fn new(
        base: PyReadonlyArray1<'_, u8>,
        base_value: f32,
        leaves: Vec<PyRef<'_, PyParameter>>,
        capacity: usize,
        device: &str,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: search::TrialEngine::new(
                &array1_vec(base),
                base_value,
                parameters(leaves),
                capacity,
                ComputeDevice::parse(device).map_err(err)?,
            )
            .map_err(err)?,
        })
    }

    #[pyo3(signature=(seeds,length,neighbors,epistemic_scale=0.7,aleatoric_scale=0.05,y_scale=1.0,beta=1.0,acquisition="ucb",seed=0))]
    fn ask(
        &mut self,
        seeds: PyReadonlyArray1<'_, u64>,
        length: f32,
        neighbors: usize,
        epistemic_scale: f32,
        aleatoric_scale: f32,
        y_scale: f32,
        beta: f32,
        acquisition: &str,
        seed: u64,
    ) -> PyResult<PyTrial> {
        let trial = self
            .inner
            .ask(
                &array1_vec(seeds),
                round_config(
                    length,
                    neighbors,
                    epistemic_scale,
                    aleatoric_scale,
                    y_scale,
                    beta,
                    AcquisitionKind::parse(acquisition).map_err(err)?,
                    seed,
                ),
            )
            .map_err(err)?;
        Ok(PyTrial { inner: trial })
    }

    #[pyo3(signature=(base_seed,count,length,neighbors,epistemic_scale=0.7,aleatoric_scale=0.05,y_scale=1.0,beta=1.0,acquisition="ucb",seed=0))]
    fn ask_stream(
        &mut self,
        base_seed: u64,
        count: usize,
        length: f32,
        neighbors: usize,
        epistemic_scale: f32,
        aleatoric_scale: f32,
        y_scale: f32,
        beta: f32,
        acquisition: &str,
        seed: u64,
    ) -> PyResult<PyTrial> {
        let trial = self
            .inner
            .ask_stream(
                base_seed,
                count,
                round_config(
                    length,
                    neighbors,
                    epistemic_scale,
                    aleatoric_scale,
                    y_scale,
                    beta,
                    AcquisitionKind::parse(acquisition).map_err(err)?,
                    seed,
                ),
            )
            .map_err(err)?;
        Ok(PyTrial { inner: trial })
    }

    /// Select a perturbation seed without writing the full candidate row.
    /// The evaluator can regenerate the row from the returned seed.
    #[pyo3(signature=(seeds,length,neighbors,epistemic_scale=0.7,aleatoric_scale=0.05,y_scale=1.0,beta=1.0,acquisition="ucb",seed=0))]
    fn ask_lazy(
        &mut self,
        seeds: PyReadonlyArray1<'_, u64>,
        length: f32,
        neighbors: usize,
        epistemic_scale: f32,
        aleatoric_scale: f32,
        y_scale: f32,
        beta: f32,
        acquisition: &str,
        seed: u64,
    ) -> PyResult<PyTrial> {
        let trial = self
            .inner
            .ask_lazy(
                &array1_vec(seeds),
                round_config(
                    length,
                    neighbors,
                    epistemic_scale,
                    aleatoric_scale,
                    y_scale,
                    beta,
                    AcquisitionKind::parse(acquisition).map_err(err)?,
                    seed,
                ),
            )
            .map_err(err)?;
        Ok(PyTrial { inner: trial })
    }

    fn row<'py>(
        &self,
        py: Python<'py>,
        trial: PyRef<'_, PyTrial>,
    ) -> PyResult<Bound<'py, PyArray1<u8>>> {
        let trial = trial.inner;
        Ok(self.inner.row(trial).map_err(err)?.into_pyarray(py))
    }

    #[cfg(all(target_os = "linux", target_arch = "x86_64", feature = "cuda"))]
    fn device_row(&self, trial: PyRef<'_, PyTrial>) -> PyResult<(u64, usize, usize)> {
        let trial = trial.inner;
        self.inner.device_row(trial).map_err(err)
    }

    #[cfg(all(target_os = "macos", feature = "metal"))]
    fn bind_model(
        &mut self,
        trial: PyRef<'_, PyTrial>,
        mut model: PyRefMut<'_, crate::py_experimental::PyNativeKdaModel>,
    ) -> PyResult<()> {
        let trial = trial.inner;
        self.inner.materialize_pending(trial).map_err(err)?;
        model.inner.bind_trial(&self.inner, trial).map_err(err)?;
        model.inner.prepare_candidate(0).map_err(err)
    }

    fn tell(&mut self, trial: PyRef<'_, PyTrial>, value: f32, accept: bool) -> PyResult<()> {
        let trial = trial.inner;
        self.inner.tell(trial, value, accept).map_err(err)?;
        Ok(())
    }

    fn replace_history(
        &mut self,
        rows: PyReadonlyArray2<'_, u8>,
        values: PyReadonlyArray1<'_, f32>,
    ) -> PyResult<()> {
        self.inner
            .replace_history(&array2_vec(&rows), &array1_vec(values))
            .map_err(err)
    }

    #[getter]
    fn history_len(&self) -> usize {
        self.inner.history_len()
    }

    #[getter]
    fn history_capacity(&self) -> usize {
        self.inner.history_capacity()
    }

    #[getter]
    fn row_bytes(&self) -> usize {
        self.inner.row_bytes()
    }
}

#[pyclass(name = "ParameterOptimizer", module = "baux.search", unsendable)]
pub struct PyParameterOptimizer {
    inner: search::ParameterOptimizer,
}

#[pymethods]
impl PyParameterOptimizer {
    /// Generate candidates from a scalar seed on the selected backend.
    #[pyo3(signature=(base_seed,count,neighbors,epistemic_scale=0.7,aleatoric_scale=0.05,y_scale=1.0,beta=1.0,acquisition="ucb",seed=0))]
    fn ask_stream(
        &mut self,
        base_seed: u64,
        count: usize,
        neighbors: usize,
        epistemic_scale: f32,
        aleatoric_scale: f32,
        y_scale: f32,
        beta: f32,
        acquisition: &str,
        seed: u64,
    ) -> PyResult<PyTrial> {
        let trial = self
            .inner
            .ask_stream(
                base_seed,
                count,
                round_config(
                    0.0,
                    neighbors,
                    epistemic_scale,
                    aleatoric_scale,
                    y_scale,
                    beta,
                    AcquisitionKind::parse(acquisition).map_err(err)?,
                    seed,
                ),
            )
            .map_err(err)?;
        Ok(PyTrial { inner: trial })
    }

    /// Generate a batch without uploading candidate seed or edit arrays.
    #[pyo3(signature=(base_seed,arms,candidates,neighbors,epistemic_scale=0.7,aleatoric_scale=0.05,y_scale=1.0,beta=1.0,acquisition="ucb",seed=0))]
    fn batch_stream(
        &mut self,
        base_seed: u64,
        arms: usize,
        candidates: usize,
        neighbors: usize,
        epistemic_scale: f32,
        aleatoric_scale: f32,
        y_scale: f32,
        beta: f32,
        acquisition: &str,
        seed: u64,
    ) -> PyResult<Vec<PyTrial>> {
        let trials = self
            .inner
            .batch_stream(
                base_seed,
                arms,
                candidates,
                round_config(
                    0.0,
                    neighbors,
                    epistemic_scale,
                    aleatoric_scale,
                    y_scale,
                    beta,
                    AcquisitionKind::parse(acquisition).map_err(err)?,
                    seed,
                ),
            )
            .map_err(err)?;
        Ok(trials.into_iter().map(|inner| PyTrial { inner }).collect())
    }

    #[new]
    #[pyo3(signature=(base,base_value,leaves,capacity,device="auto",num_pert=20,length_init=0.8,length_min=0.0078125,length_max=1.6,max_pending=1))]
    fn new(
        base: PyReadonlyArray1<'_, u8>,
        base_value: f32,
        leaves: Vec<PyRef<'_, PyParameter>>,
        capacity: usize,
        device: &str,
        num_pert: usize,
        length_init: f64,
        length_min: f64,
        length_max: f64,
        max_pending: usize,
    ) -> PyResult<Self> {
        let device = ComputeDevice::parse(device).map_err(err)?;
        Ok(Self {
            inner: search::ParameterOptimizer::new_batch(
                &array1_vec(base),
                base_value,
                parameters(leaves),
                capacity,
                device,
                num_pert,
                TRLengthConfig::new(length_init, length_min, length_max),
                max_pending,
            )
            .map_err(err)?,
        })
    }

    #[pyo3(signature=(seeds,neighbors,epistemic_scale=0.7,aleatoric_scale=0.05,y_scale=1.0,beta=1.0,acquisition="ucb",seed=0))]
    fn ask(
        &mut self,
        seeds: PyReadonlyArray1<'_, u64>,
        neighbors: usize,
        epistemic_scale: f32,
        aleatoric_scale: f32,
        y_scale: f32,
        beta: f32,
        acquisition: &str,
        seed: u64,
    ) -> PyResult<PyTrial> {
        let trial = self
            .inner
            .ask(
                &array1_vec(seeds),
                round_config(
                    0.0,
                    neighbors,
                    epistemic_scale,
                    aleatoric_scale,
                    y_scale,
                    beta,
                    AcquisitionKind::parse(acquisition).map_err(err)?,
                    seed,
                ),
            )
            .map_err(err)?;
        Ok(PyTrial { inner: trial })
    }

    #[pyo3(signature=(seeds,neighbors,epistemic_scale=0.7,aleatoric_scale=0.05,y_scale=1.0,beta=1.0,acquisition="ucb",seed=0))]
    fn ask_batch(
        &mut self,
        seeds: PyReadonlyArray2<'_, u64>,
        neighbors: usize,
        epistemic_scale: f32,
        aleatoric_scale: f32,
        y_scale: f32,
        beta: f32,
        acquisition: &str,
        seed: u64,
    ) -> PyResult<Vec<PyTrial>> {
        let shape = seeds.shape();
        let arms = shape[0];
        let trials = self
            .inner
            .ask_batch(
                &array2_vec(&seeds),
                arms,
                round_config(
                    0.0,
                    neighbors,
                    epistemic_scale,
                    aleatoric_scale,
                    y_scale,
                    beta,
                    AcquisitionKind::parse(acquisition).map_err(err)?,
                    seed,
                ),
            )
            .map_err(err)?;
        Ok(trials.into_iter().map(|inner| PyTrial { inner }).collect())
    }

    fn row<'py>(
        &self,
        py: Python<'py>,
        trial: PyRef<'_, PyTrial>,
    ) -> PyResult<Bound<'py, PyArray1<u8>>> {
        Ok(self.inner.row(trial.inner).map_err(err)?.into_pyarray(py))
    }

    #[cfg(all(target_os = "linux", target_arch = "x86_64", feature = "cuda"))]
    fn device_row(&self, trial: PyRef<'_, PyTrial>) -> PyResult<(u64, usize, usize)> {
        self.inner.device_row(trial.inner).map_err(err)
    }

    #[cfg(all(target_os = "linux", target_arch = "x86_64", feature = "cuda"))]
    fn device_batch(&self, trials: Vec<PyRef<'_, PyTrial>>) -> PyResult<Vec<(u64, usize, usize)>> {
        let inner = trials.iter().map(|trial| trial.inner).collect::<Vec<_>>();
        self.inner.device_batch(&inner).map_err(err)
    }

    fn observe(&mut self, trial: PyRef<'_, PyTrial>, value: f32) -> PyResult<()> {
        self.inner.observe(trial.inner, value).map_err(err)
    }

    fn tell(&mut self, trial: PyRef<'_, PyTrial>, value: f32) -> PyResult<bool> {
        let accepted = self.inner.tell(trial.inner, value).map_err(err)?;
        Ok(accepted)
    }

    fn tell_batch(
        &mut self,
        trials: Vec<PyRef<'_, PyTrial>>,
        values: PyReadonlyArray1<'_, f32>,
    ) -> PyResult<Vec<bool>> {
        let inner = trials.iter().map(|trial| trial.inner).collect::<Vec<_>>();
        let accepted = self
            .inner
            .tell_batch(&inner, &array1_vec(values))
            .map_err(err)?;
        Ok(accepted)
    }

    #[getter]
    fn length(&self) -> PyResult<f64> {
        self.inner.length().map_err(err)
    }

    #[getter]
    fn probability(&self) -> f64 {
        self.inner.probability()
    }

    #[getter]
    fn best(&self) -> PyResult<f32> {
        self.inner.best().map_err(err)
    }

    #[getter]
    fn restarts(&self) -> PyResult<usize> {
        self.inner.restarts().map_err(err)
    }

    #[getter]
    fn history_len(&self) -> PyResult<usize> {
        self.inner.history_len().map_err(err)
    }
}
