use super::super::py_util;
use baux::experimental::MetalParamBlock;
use metal::Buffer;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::cell::Cell;
use std::rc::Rc;

pub(super) fn err(message: String) -> PyErr {
    let _ = py_util::format_profile::<baux::experimental::MetalProfileInfo>;
    PyValueError::new_err(message)
}

#[pyclass(name = "MetalWeights", module = "baux.experimental", unsendable)]
pub struct PyMetalWeights {
    pub(crate) buffer: Buffer,
    pub(crate) len: usize,
    pub(crate) lease: Option<Rc<Cell<usize>>>,
}

impl Drop for PyMetalWeights {
    fn drop(&mut self) {
        if let Some(lease) = &self.lease {
            lease.set(lease.get() - 1);
        }
    }
}

#[pymethods]
impl PyMetalWeights {
    #[new]
    fn new(bits: PyReadonlyArray1<'_, u16>) -> PyResult<Self> {
        let values = bits.as_slice()?;
        Ok(Self {
            buffer: baux::experimental::metal_upload(values).map_err(err)?,
            len: values.len(),
            lease: None,
        })
    }

    #[staticmethod]
    fn device_info(py: Python<'_>) -> PyResult<Bound<'_, PyDict>> {
        let device = baux::experimental::metal_memory_info().map_err(err)?;
        if !device.has_unified_memory {
            return Err(PyValueError::new_err(
                "Metal search requires unified memory",
            ));
        }
        let info = PyDict::new(py);
        info.set_item("name", device.name)?;
        info.set_item(
            "recommended_working_set_bytes",
            device.recommended_max_working_set_size,
        )?;
        info.set_item("allocated_bytes", device.current_allocated_size)?;
        info.set_item("max_buffer_bytes", device.max_buffer_length)?;
        Ok(info)
    }

    #[getter]
    fn size(&self) -> usize {
        self.len
    }

    fn read<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<u16>> {
        // All producers synchronize before publishing a view, and leases
        // prevent mutation.
        let values =
            unsafe { std::slice::from_raw_parts(self.buffer.contents().cast::<u16>(), self.len) };
        values.to_vec().into_pyarray(py)
    }
}

#[pyclass(name = "MetalParamBlock", module = "baux.experimental", frozen)]
pub struct PyMetalParamBlock {
    pub(crate) inner: MetalParamBlock,
}

#[pymethods]
impl PyMetalParamBlock {
    #[new]
    #[pyo3(signature=(key,offset,length,scale,weight=1.0))]
    fn new(key: u64, offset: usize, length: usize, scale: f32, weight: f32) -> PyResult<Self> {
        Ok(Self {
            inner: MetalParamBlock::new(key, offset, length, scale, weight).map_err(err)?,
        })
    }
}
