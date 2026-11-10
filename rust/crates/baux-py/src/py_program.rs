//! Python ownership of a backend-independent Rust program signature.

use baux::semantics::compute::{Program, Tensor};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

type Spec = (Vec<usize>, String);

#[pyclass(name = "Program", frozen)]
pub struct PyProgram {
    inner: Program,
}

fn tensors(specs: Vec<Spec>) -> PyResult<Vec<Tensor>> {
    specs
        .into_iter()
        .map(|(shape, dtype)| Tensor::new(shape, &dtype).map_err(PyValueError::new_err))
        .collect()
}

fn specs(tensors: &[Tensor]) -> Vec<Spec> {
    tensors
        .iter()
        .map(|tensor| (tensor.shape.clone(), tensor.dtype.clone()))
        .collect()
}

#[pymethods]
impl PyProgram {
    #[new]
    #[pyo3(signature = (source, inputs, outputs, *, entry="main"))]
    fn new(source: String, inputs: Vec<Spec>, outputs: Vec<Spec>, entry: &str) -> PyResult<Self> {
        let inner = Program::new(source, entry.into(), tensors(inputs)?, tensors(outputs)?)
            .map_err(PyValueError::new_err)?;
        Ok(Self { inner })
    }

    #[getter]
    fn source(&self) -> &str {
        self.inner.source()
    }
    #[getter]
    fn entry(&self) -> &str {
        self.inner.entry()
    }
    #[getter]
    fn inputs(&self) -> Vec<Spec> {
        specs(self.inner.inputs())
    }
    #[getter]
    fn outputs(&self) -> Vec<Spec> {
        specs(self.inner.outputs())
    }
}
