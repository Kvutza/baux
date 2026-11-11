//! Python handles for the shared native program runtime.
use baux::portable::{Buffer, Executable, Pending, Tensor};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use std::sync::Mutex;

type Spec = (Vec<usize>, String);
fn specs(values: Vec<Spec>) -> PyResult<Vec<Tensor>> {
    values
        .into_iter()
        .map(|(shape, dtype)| Tensor::new(shape, &dtype).map_err(PyValueError::new_err))
        .collect()
}
fn buffers(py: Python<'_>, values: Vec<Py<PyNativeBuffer>>) -> Vec<Buffer> {
    values
        .into_iter()
        .map(|value| value.borrow(py).inner.clone())
        .collect()
}
fn results(values: Vec<Buffer>) -> Vec<PyNativeBuffer> {
    values
        .into_iter()
        .map(|inner| PyNativeBuffer { inner })
        .collect()
}

#[pyclass(name = "NativeBuffer", frozen)]
pub struct PyNativeBuffer {
    pub(crate) inner: Buffer,
}
#[pymethods]
impl PyNativeBuffer {
    #[getter]
    fn shape(&self) -> Vec<usize> {
        self.inner.spec().shape.clone()
    }
    #[getter]
    fn dtype(&self) -> &str {
        &self.inner.spec().dtype
    }
    fn read<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        let data = py
            .detach(|| self.inner.read())
            .map_err(PyValueError::new_err)?;
        Ok(PyBytes::new(py, &data))
    }
}

#[pyclass(name = "NativeSession", frozen)]
pub struct PyNativeSession {
    pub(crate) inner: Executable,
}
#[pymethods]
impl PyNativeSession {
    #[staticmethod]
    fn load_tvm(
        py: Python<'_>,
        library: &str,
        entry: &str,
        device: &str,
        inputs: Vec<Spec>,
        outputs: Vec<Spec>,
    ) -> PyResult<Self> {
        let inputs = specs(inputs)?;
        let outputs = specs(outputs)?;
        let inner = py
            .detach(|| Executable::load_tvm(library, entry, device, inputs, outputs))
            .map_err(PyValueError::new_err)?;
        Ok(Self { inner })
    }

    #[new]
    fn new(
        py: Python<'_>,
        bytecode: &[u8],
        entry: &str,
        device: &str,
        inputs: Vec<Spec>,
        outputs: Vec<Spec>,
    ) -> PyResult<Self> {
        let inputs = specs(inputs)?;
        let outputs = specs(outputs)?;
        let inner = py
            .detach(|| Executable::load(bytecode, entry, device, inputs, outputs))
            .map_err(PyValueError::new_err)?;
        Ok(Self { inner })
    }
    fn put(
        &self,
        py: Python<'_>,
        shape: Vec<usize>,
        dtype: &str,
        bytes: &[u8],
    ) -> PyResult<PyNativeBuffer> {
        let spec = Tensor::new(shape, dtype).map_err(PyValueError::new_err)?;
        let inner = py
            .detach(|| self.inner.put(spec, bytes))
            .map_err(PyValueError::new_err)?;
        Ok(PyNativeBuffer { inner })
    }
    fn run(
        &self,
        py: Python<'_>,
        inputs: Vec<Py<PyNativeBuffer>>,
    ) -> PyResult<Vec<PyNativeBuffer>> {
        let inputs = buffers(py, inputs);
        py.detach(|| self.inner.run(&inputs))
            .map(results)
            .map_err(PyValueError::new_err)
    }
    fn submit(&self, py: Python<'_>, inputs: Vec<Py<PyNativeBuffer>>) -> PyResult<PyNativePending> {
        let pending = self
            .inner
            .submit(buffers(py, inputs))
            .map_err(PyValueError::new_err)?;
        Ok(PyNativePending {
            inner: Mutex::new(Some(pending)),
        })
    }
}

#[pyclass(name = "NativePending", frozen)]
pub struct PyNativePending {
    pub(crate) inner: Mutex<Option<Pending>>,
}
#[pymethods]
impl PyNativePending {
    fn ready(&self) -> bool {
        self.inner
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .as_ref()
            .is_some_and(Pending::is_ready)
    }
    fn wait(&self, py: Python<'_>) -> PyResult<Vec<PyNativeBuffer>> {
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
