//! DLPack views retain immutable posterior snapshots independently of the ENN.
use baux::device_posterior::{DeviceDraw, DevicePosterior};
use pyo3::exceptions::{PyBufferError, PyValueError};
use pyo3::prelude::*;
use std::sync::Arc;

#[pyclass(name = "DevicePosterior", unsendable)]
pub struct PyDevicePosterior {
    inner: Arc<DevicePosterior>,
}

impl PyDevicePosterior {
    pub(crate) fn new(value: DevicePosterior) -> Self {
        Self {
            inner: Arc::new(value),
        }
    }
    fn array(&self, error: bool) -> PyDeviceArray {
        PyDeviceArray {
            _owner: ArrayOwner::Posterior(Arc::clone(&self.inner)),
            pointer: self.inner.pointer(error),
            rows: self.inner.rows,
            columns: self.inner.metrics,
            device: self.inner.device(),
            bits: 32,
        }
    }
}

#[pymethods]
impl PyDevicePosterior {
    #[getter]
    /// Predictive means, shape (queries, metrics), on the model's device.
    fn mean(&self) -> PyDeviceArray {
        self.array(false)
    }
    #[getter]
    /// Predictive standard errors, shape (queries, metrics).
    fn se(&self) -> PyDeviceArray {
        self.array(true)
    }
    /// Shared-seed joint draws, flattened to (seeds * queries, metrics).
    /// CUDA uses the existing FP64 draw kernel; Metal returns FP32 draws.
    fn draw(&self, seeds: Vec<u64>) -> PyResult<PyDeviceArray> {
        let rows = seeds
            .len()
            .checked_mul(self.inner.rows)
            .ok_or_else(|| PyValueError::new_err("draw shape overflow"))?;
        let draw = self.inner.draw(&seeds).map_err(PyValueError::new_err)?;
        Ok(PyDeviceArray {
            pointer: draw.pointer(),
            bits: draw.bits(),
            _owner: ArrayOwner::Draw(draw),
            rows,
            columns: self.inner.metrics,
            device: self.inner.device(),
        })
    }
}

#[allow(dead_code)]
enum ArrayOwner {
    Posterior(Arc<DevicePosterior>),
    Draw(DeviceDraw),
}

#[pyclass(name = "DeviceArray", unsendable)]
pub struct PyDeviceArray {
    _owner: ArrayOwner,
    pointer: u64,
    rows: usize,
    columns: usize,
    device: i32,
    bits: u8,
}

#[pymethods]
impl PyDeviceArray {
    fn __dlpack_device__(&self) -> (i32, i32) {
        (self.device, 0)
    }
    #[pyo3(signature=(*,stream=None,max_version=None,dl_device=None,copy=None))]
    fn __dlpack__(
        slf: PyRef<'_, Self>,
        stream: Option<i64>,
        max_version: Option<(u32, u32)>,
        dl_device: Option<(i32, i32)>,
        copy: Option<bool>,
    ) -> PyResult<Py<PyAny>> {
        if copy == Some(true) || dl_device.is_some_and(|v| v != (slf.device, 0)) {
            return Err(PyBufferError::new_err(
                "device posterior exports its own storage without copies or device transfers",
            ));
        }
        if slf.device == 8 && stream.is_some_and(|v| v != -1) {
            return Err(PyValueError::new_err(
                "Metal DLPack stream must be None or -1",
            ));
        }
        if slf.device == 2 && stream == Some(0) {
            return Err(PyValueError::new_err("CUDA DLPack stream cannot be zero"));
        }
        let py = slf.py();
        let (pointer, rows, columns, device, bits) =
            (slf.pointer, slf.rows, slf.columns, slf.device, slf.bits);
        let owner = slf.into_pyobject(py)?.into_any().unbind();
        crate::dlpack::export_array(py, owner, pointer, rows, columns, device, bits, max_version)
    }
}
