//! Thin Python ownership layer over the native TVM compiler and Relax VM.
use baux::portable::{Pending, Tensor, TvmExecutable, TvmTensor};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use std::sync::Mutex;

#[pyclass(name = "TvmBuffer", frozen)]
pub struct PyTvmBuffer {
    inner: TvmTensor,
}
#[pymethods]
impl PyTvmBuffer {
    #[getter]
    fn shape(&self) -> Vec<usize> {
        self.inner.spec().shape.clone()
    }
    #[getter]
    fn dtype(&self) -> &str {
        &self.inner.spec().dtype
    }
    fn read<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        let bytes = py
            .detach(|| self.inner.read())
            .map_err(PyValueError::new_err)?;
        Ok(PyBytes::new(py, &bytes))
    }
}

fn buffers(py: Python<'_>, values: Vec<Py<PyTvmBuffer>>) -> Vec<TvmTensor> {
    values.iter().map(|v| v.borrow(py).inner.clone()).collect()
}
fn results(values: Vec<TvmTensor>) -> Vec<PyTvmBuffer> {
    values
        .into_iter()
        .map(|inner| PyTvmBuffer { inner })
        .collect()
}

#[pyclass(name = "TvmSession", frozen)]
pub struct PyTvmSession {
    inner: TvmExecutable,
}
#[pymethods]
impl PyTvmSession {
    #[new]
    fn new(py: Python<'_>, source: &str, entry: &str, importer: &str) -> PyResult<Self> {
        let inner = py
            .detach(|| TvmExecutable::compile(source, entry, std::path::Path::new(importer)))
            .map_err(PyRuntimeError::new_err)?;
        Ok(Self { inner })
    }
    fn put(
        &self,
        py: Python<'_>,
        shape: Vec<usize>,
        dtype: &str,
        bytes: &[u8],
    ) -> PyResult<PyTvmBuffer> {
        let spec = Tensor::new(shape, dtype).map_err(PyValueError::new_err)?;
        let inner = py
            .detach(|| TvmTensor::put(spec, bytes))
            .map_err(PyValueError::new_err)?;
        Ok(PyTvmBuffer { inner })
    }
    fn run(&self, py: Python<'_>, inputs: Vec<Py<PyTvmBuffer>>) -> PyResult<Vec<PyTvmBuffer>> {
        let inputs = buffers(py, inputs);
        py.detach(|| self.inner.run(&inputs.iter().collect::<Vec<_>>()))
            .map(results)
            .map_err(PyValueError::new_err)
    }
    fn submit(&self, py: Python<'_>, inputs: Vec<Py<PyTvmBuffer>>) -> PyResult<PyTvmPending> {
        let pending = self
            .inner
            .submit(buffers(py, inputs))
            .map_err(PyValueError::new_err)?;
        Ok(PyTvmPending {
            inner: Mutex::new(Some(pending)),
        })
    }
}

#[pyclass(name = "TvmPending", frozen)]
pub struct PyTvmPending {
    inner: Mutex<Option<Pending<TvmTensor>>>,
}
#[pymethods]
impl PyTvmPending {
    fn ready(&self) -> bool {
        self.inner
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .as_ref()
            .is_some_and(Pending::is_ready)
    }
    fn wait(&self, py: Python<'_>) -> PyResult<Vec<PyTvmBuffer>> {
        let pending = self
            .inner
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .take()
            .ok_or_else(|| PyValueError::new_err("completion was already consumed"))?;
        py.detach(|| pending.wait())
            .map(results)
            .map_err(PyValueError::new_err)
    }
}
