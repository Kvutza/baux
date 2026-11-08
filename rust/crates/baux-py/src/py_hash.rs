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
