use std::cell::Cell;
use std::rc::Rc;

use baux::experimental::{MetalFlameEvaluator, MetalParamBlock};
use metal::Buffer;
use ndarray::Array2;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyDict};

use super::{PyMetalProposals, err};
use crate::py_flameconfig::configuration;

#[pyclass(name = "MetalWeights", module = "baux.experimental", unsendable)]
pub struct PyMetalWeights {
    pub(super) buffer: Buffer,
    pub(super) len: usize,
    pub(super) lease: Option<Rc<Cell<usize>>>,
}

impl PyMetalWeights {
    pub(crate) fn owned(buffer: Buffer, len: usize) -> Self {
        Self {
            buffer,
            len,
            lease: None,
        }
    }
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
            buffer: baux::experimental::metal_flame_upload(values).map_err(err)?,
            len: values.len(),
            lease: None,
        })
    }

    #[staticmethod]
    fn device_info(py: Python<'_>) -> PyResult<Bound<'_, PyDict>> {
        let device = baux::experimental::metal_flame_memory_info().map_err(err)?;
        if !device.has_unified_memory {
            return Err(PyValueError::new_err("Metal FLAME requires unified memory"));
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

#[pyclass(name = "MetalFlameEvaluator", module = "baux.experimental", unsendable)]
pub struct PyMetalFlameEvaluator {
    inner: MetalFlameEvaluator,
}

#[pymethods]
impl PyMetalFlameEvaluator {
    #[new]
    #[pyo3(signature=(config,max_tokens))]
    fn new(config: &Bound<'_, PyDict>, max_tokens: &Bound<'_, PyAny>) -> PyResult<Self> {
        if max_tokens.is_instance_of::<PyBool>() {
            return Err(PyTypeError::new_err(
                "max_tokens must be an integer, not bool",
            ));
        }
        Ok(Self {
            inner: MetalFlameEvaluator::new(configuration(config)?, max_tokens.extract()?)
                .map_err(err)?,
        })
    }

    #[getter]
    fn weights_len(&self) -> usize {
        self.inner.weights_len()
    }

    #[getter]
    fn workspace_bytes(&self) -> u64 {
        self.inner.workspace_bytes()
    }

    fn logits<'py>(
        &mut self,
        py: Python<'py>,
        weights: &Bound<'py, PyAny>,
        tokens: Vec<i32>,
    ) -> PyResult<Bound<'py, PyArray2<f32>>> {
        self.inner.check_tokens(&tokens).map_err(err)?;
        let buffer = weight_buffer(weights)?;
        let logits = self.inner.logits(&buffer, &tokens).map_err(err)?;
        let array = Array2::from_shape_vec((tokens.len(), self.inner.vocab()), logits)
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        Ok(array.into_pyarray(py))
    }

    fn next_logits<'py>(
        &mut self,
        py: Python<'py>,
        weights: &Bound<'py, PyAny>,
        tokens: Vec<i32>,
    ) -> PyResult<Bound<'py, PyArray1<f32>>> {
        self.inner.check_tokens(&tokens).map_err(err)?;
        let buffer = weight_buffer(weights)?;
        Ok(self
            .inner
            .next_logits(&buffer, &tokens)
            .map_err(err)?
            .into_pyarray(py))
    }

    fn losses<'py>(
        &mut self,
        py: Python<'py>,
        weights: &Bound<'py, PyAny>,
        tokens: Vec<Vec<i32>>,
        masks: Vec<Vec<bool>>,
    ) -> PyResult<Bound<'py, PyArray1<f32>>> {
        self.inner.check_batch(&tokens, &masks).map_err(err)?;
        let buffer = weight_buffer(weights)?;
        Ok(self
            .inner
            .losses(&buffer, &tokens, &masks)
            .map_err(err)?
            .into_pyarray(py))
    }
}

pub(crate) fn weight_buffer(weights: &Bound<'_, PyAny>) -> PyResult<Buffer> {
    if let Ok(view) = weights.extract::<PyRef<'_, PyMetalWeights>>() {
        return Ok(view.buffer.clone());
    }
    if let Ok(proposal) = weights.extract::<PyRef<'_, PyMetalProposals>>() {
        let owner = proposal.owner.bind(weights.py()).try_borrow()?;
        return owner.inner.propose_buffer(&proposal.inner).map_err(err);
    }
    Err(PyTypeError::new_err(
        "Expected resident MetalWeights or a live Metal proposal",
    ))
}

#[pyclass(name = "MetalParamBlock", module = "baux.experimental", frozen)]
pub struct PyMetalParamBlock {
    pub(super) inner: MetalParamBlock,
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
