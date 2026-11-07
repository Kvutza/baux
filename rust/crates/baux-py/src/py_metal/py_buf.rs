use super::super::py_util;
use super::PyMetalSearchState;
use baux::experimental::{MetalParamBlock, MetalProposals};
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

#[pyclass(name = "Var", module = "baux", unsendable)]
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
    pub(crate) fn device_info(py: Python<'_>) -> PyResult<Bound<'_, PyDict>> {
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

    #[getter]
    fn device(&self) -> &'static str {
        "metal"
    }

    #[getter]
    fn shape(&self) -> (usize,) {
        (self.len,)
    }

    fn read<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<u16>> {
        // All producers synchronize before publishing a view, and leases
        // prevent mutation.
        let values =
            unsafe { std::slice::from_raw_parts(self.buffer.contents().cast::<u16>(), self.len) };
        values.to_vec().into_pyarray(py)
    }
}

#[pyclass(name = "Span", module = "baux", frozen)]
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

#[pyclass(name = "Trial", module = "baux", unsendable)]
pub struct PyMetalProposals {
    pub(super) owner: Py<PyMetalSearchState>,
    pub(super) inner: MetalProposals,
}

#[pymethods]
impl PyMetalProposals {
    fn weights(&self, py: Python<'_>) -> PyResult<PyMetalWeights> {
        let owner = self.owner.bind(py).try_borrow()?;
        let buffer = owner.inner.propose_buffer(&self.inner).map_err(err)?;
        let len = owner.inner.len();
        owner.leases.set(owner.leases.get() + 1);
        Ok(PyMetalWeights {
            buffer,
            len,
            lease: Some(owner.leases.clone()),
        })
    }

    fn var(&self, py: Python<'_>) -> PyResult<PyMetalWeights> {
        self.weights(py)
    }

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

    fn pool_geometry(&self, py: Python<'_>) -> PyResult<([f32; 4], [Option<f32>; 6])> {
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
