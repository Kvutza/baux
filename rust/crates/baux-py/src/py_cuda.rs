use std::path::PathBuf;

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

fn err(error: String) -> PyErr {
    PyRuntimeError::new_err(error)
}

#[pyfunction]
pub fn target() -> PyResult<&'static str> {
    baux::experimental::cuda_target().map_err(err)
}

#[pyfunction]
pub fn check(py: Python<'_>) -> PyResult<()> {
    py.detach(baux::experimental::cuda_check).map_err(err)
}

#[pyfunction]
pub fn trace(py: Python<'_>, output: PathBuf) -> PyResult<()> {
    py.detach(|| baux::experimental::cuda_trace(&output))
        .map_err(err)
}
