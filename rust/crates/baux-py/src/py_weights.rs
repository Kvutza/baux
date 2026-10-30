#![allow(clippy::too_many_arguments)]

use numpy::{Element, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

mod py_dense;
mod py_sparse;

pub use py_dense::*;
pub use py_sparse::*;

pub(super) fn err(error: String) -> PyErr {
    PyValueError::new_err(error)
}

pub(super) fn array1_vec<T: Copy + Element>(array: PyReadonlyArray1<'_, T>) -> Vec<T> {
    array.as_array().iter().copied().collect()
}

pub(super) fn array2_vec<T: Copy + Element>(array: &PyReadonlyArray2<'_, T>) -> Vec<T> {
    array.as_array().iter().copied().collect()
}
