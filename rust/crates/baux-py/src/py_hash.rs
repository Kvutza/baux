//! Hash-based RNG Python bindings.

use ndarray::IxDyn;
use numpy::{IntoPyArray, PyArrayDyn, PyReadonlyArray1, PyReadonlyArrayDyn};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use std::path::{Path, PathBuf};

/// Python wrapper for normal_hash
#[pyfunction(name = "normal_hash")]
pub fn normal_py<'py>(
    py: Python<'py>,
    function_seeds: PyReadonlyArray1<i64>,
    data_indices: PyReadonlyArrayDyn<i64>,
    num_metrics: i64,
) -> PyResult<Bound<'py, PyArrayDyn<f64>>> {
    let seeds_arr = function_seeds.as_array();
    let indices_arr = data_indices.as_array();

    let seeds: Vec<i64> = seeds_arr.iter().copied().collect();
    let indices: Vec<i64> = indices_arr.iter().copied().collect();
    let input_shape = indices_arr.shape().to_vec();
    let mut output_shape = Vec::with_capacity(2 + input_shape.len());
    output_shape.push(seeds.len());
    output_shape.extend(input_shape.iter().copied());
    output_shape.push(num_metrics.max(0) as usize);

    // Release GIL for computation
    let result = py.detach(|| baux::normal_hash(&seeds, &indices, num_metrics));

    match result {
        Ok(arr) => {
            // Reshape from (num_seeds, flattened_indices, num_metrics)
            // to (num_seeds, *data_indices.shape, num_metrics) for API parity.
            let reshaped = arr
                .into_shape_with_order(IxDyn(&output_shape))
                .map_err(|e| PyValueError::new_err(format!("Shape error: {}", e)))?;
            Ok(reshaped.into_pyarray(py))
        }
        Err(e) => Err(PyValueError::new_err(e.to_string())),
    }
}

/// Verify upstream archive and source lockfiles against their Merkle root
/// digests.
#[pyfunction(name = "verify_upstream_lock", signature = (name, override_dir=None, repo_root=None))]
pub fn verify_lock(
    name: &str,
    override_dir: Option<&str>,
    repo_root: Option<&str>,
) -> PyResult<String> {
    let root = match repo_root {
        Some(path) => PathBuf::from(path),
        None => std::env::current_dir().map_err(|e| PyRuntimeError::new_err(e.to_string()))?,
    };
    let dir = override_dir.map(Path::new);
    let lock =
        baux::artifact::merkle::verify_lock(&root, name, dir).map_err(PyRuntimeError::new_err)?;
    baux::artifact::json::pretty_string(&lock).map_err(|e| PyRuntimeError::new_err(e.to_string()))
}

/// Compute the 64-character lowercase hexadecimal BLAKE3 digest of `data`.
#[pyfunction(name = "blake3_hex")]
pub fn py_blake3hex(py: Python<'_>, data: &[u8]) -> String {
    py.detach(|| baux::artifact::blake3::hash_hex(data))
}

/// Compute the 64-character lowercase hexadecimal BLAKE3 digest of a file on
/// disk.
#[pyfunction(name = "blake3_file")]
pub fn py_blake3file(py: Python<'_>, path: &str) -> PyResult<String> {
    py.detach(|| baux::artifact::blake3::hash_file(Path::new(path)))
        .map_err(PyRuntimeError::new_err)
}

/// Splittable 128-bit cryptographic PRNG key for functional pipelines.
#[pyclass(name = "Key", frozen, from_py_object)]
#[derive(Clone, Copy)]
pub struct PyKey {
    pub(crate) inner: baux::Key,
}

#[pymethods]
impl PyKey {
    #[new]
    #[pyo3(signature = (source=None))]
    fn new(source: Option<&Bound<'_, PyAny>>) -> PyResult<Self> {
        let inner = match source {
            None => baux::Key::entropy(),
            Some(obj) => {
                if let Ok(k) = obj.extract::<PyRef<'_, PyKey>>() {
                    k.inner
                } else if let Ok(bytes) = obj.extract::<&[u8]>() {
                    baux::Key::from_bytes(bytes)
                } else if let Ok(seed) = obj.extract::<u64>() {
                    baux::Key::new(seed)
                } else if let Ok(seed_i64) = obj.extract::<i64>() {
                    baux::Key::new(seed_i64 as u64)
                } else {
                    return Err(PyValueError::new_err(
                        "Key expects None (entropy), bytes, an int seed, or an existing Key",
                    ));
                }
            }
        };
        Ok(Self { inner })
    }

    #[staticmethod]
    fn entropy() -> Self {
        Self {
            inner: baux::Key::entropy(),
        }
    }

    #[staticmethod]
    fn from_bytes(data: &[u8]) -> Self {
        Self {
            inner: baux::Key::from_bytes(data),
        }
    }

    #[staticmethod]
    fn from_pair(hi: u64, lo: u64) -> Self {
        Self {
            inner: baux::Key(hi, lo),
        }
    }

    #[getter]
    fn hi(&self) -> u64 {
        self.inner.0
    }

    #[getter]
    fn lo(&self) -> u64 {
        self.inner.1
    }

    fn seed(&self) -> u64 {
        self.inner.seed()
    }

    fn split(&self) -> (Self, Self) {
        let (a, b) = self.inner.split();
        (Self { inner: a }, Self { inner: b })
    }

    fn split_n(&self, n: usize) -> Vec<Self> {
        self.inner
            .split_n(n)
            .into_iter()
            .map(|inner| Self { inner })
            .collect()
    }

    fn fold(&self, tag: u64) -> Self {
        Self {
            inner: self.inner.fold(tag),
        }
    }

    fn __repr__(&self) -> String {
        format!("Key(0x{:016x}, 0x{:016x})", self.inner.0, self.inner.1)
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }
}
