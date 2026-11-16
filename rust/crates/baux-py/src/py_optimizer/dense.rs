//! Python bindings for CPU and WebGPU exact `f32` dense linear layers and
//! distances.

use ndarray::{Array1, Array2};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::py_device::PyDevice;

fn extract_2d(obj: &Bound<'_, PyAny>, name: &str) -> PyResult<(usize, usize, Vec<f32>)> {
    if let Ok(arr) = obj.extract::<PyReadonlyArray2<f32>>() {
        let view = arr.as_array();
        return Ok((view.nrows(), view.ncols(), view.iter().copied().collect()));
    }
    if let Ok(arr) = obj.extract::<PyReadonlyArray2<f64>>() {
        let view = arr.as_array();
        return Ok((
            view.nrows(),
            view.ncols(),
            view.iter().map(|&v| v as f32).collect(),
        ));
    }
    Err(PyValueError::new_err(format!(
        "{name} must be a 2D float32 or float64 NumPy array"
    )))
}

fn extract_1d(obj: &Bound<'_, PyAny>, name: &str) -> PyResult<Vec<f32>> {
    if let Ok(arr) = obj.extract::<PyReadonlyArray1<f32>>() {
        return Ok(arr.as_array().iter().copied().collect());
    }
    if let Ok(arr) = obj.extract::<PyReadonlyArray1<f64>>() {
        return Ok(arr.as_array().iter().map(|&v| v as f32).collect());
    }
    Err(PyValueError::new_err(format!(
        "{name} must be a 1D float32 or float64 NumPy array"
    )))
}

fn dense_device(device: Option<&Bound<'_, PyAny>>) -> PyResult<baux::Device> {
    let raw = match device {
        None => "auto".to_string(),
        Some(obj) => {
            if let Ok(dev) = obj.extract::<PyRef<'_, PyDevice>>() {
                dev.kind.to_string()
            } else {
                obj.extract::<String>()?
            }
        }
    };
    let lower = raw.trim().to_ascii_lowercase();
    match lower.as_str() {
        "" | "auto" => {
            if cfg!(target_arch = "wasm32") && baux::webgpu::is_available() {
                Ok(baux::Device::WebGpu)
            } else {
                Ok(baux::Device::Cpu)
            }
        }
        _ => baux::Device::parse(&lower).map_err(PyValueError::new_err),
    }
}

/// Exact `f32` dense linear layer (`Y = X @ weight^T + bias`) on CPU or WebGPU.
#[pyclass(name = "DenseLinear", frozen)]
pub struct PyDenseLinear {
    inner: baux::webgpu::DenseLinear,
}

#[pymethods]
impl PyDenseLinear {
    #[new]
    #[pyo3(signature = (weight, bias=None, device=None))]
    fn new(
        weight: &Bound<'_, PyAny>,
        bias: Option<&Bound<'_, PyAny>>,
        device: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        let (out_features, in_features, w_vec) = extract_2d(weight, "weight")?;
        let b_vec = bias.map(|b| extract_1d(b, "bias")).transpose()?;
        let dev = dense_device(device)?;
        let inner = baux::webgpu::DenseLinear::new(out_features, in_features, w_vec, b_vec, dev)
            .map_err(PyValueError::new_err)?;
        Ok(Self { inner })
    }

    #[getter]
    fn in_features(&self) -> usize {
        self.inner.in_features()
    }

    #[getter]
    fn out_features(&self) -> usize {
        self.inner.out_features()
    }

    #[getter]
    fn device(&self) -> &'static str {
        self.inner.device().name()
    }

    #[getter]
    fn weight<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f32>>> {
        let arr = Array2::from_shape_vec(
            (self.inner.out_features(), self.inner.in_features()),
            self.inner.weight().to_vec(),
        )
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(arr.into_pyarray(py))
    }

    #[getter]
    fn bias<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray1<f32>>> {
        self.inner
            .bias()
            .map(|b| Array1::from_vec(b.to_vec()).into_pyarray(py))
    }

    /// Evaluates `Y = X @ weight^T + bias` for a `(m, in_features)` matrix `x`.
    fn apply<'py>(
        &self,
        py: Python<'py>,
        x: &Bound<'_, PyAny>,
    ) -> PyResult<Bound<'py, PyArray2<f32>>> {
        let (m, k, x_vec) = extract_2d(x, "x")?;
        if k != self.inner.in_features() {
            return Err(PyValueError::new_err(format!(
                "x has {k} columns, expected in_features={}",
                self.inner.in_features()
            )));
        }
        let n = self.inner.out_features();
        let out = py
            .detach(|| self.inner.apply(m, &x_vec))
            .map_err(PyValueError::new_err)?;
        let arr = Array2::from_shape_vec((m, n), out)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(arr.into_pyarray(py))
    }

    fn __call__<'py>(
        &self,
        py: Python<'py>,
        x: &Bound<'_, PyAny>,
    ) -> PyResult<Bound<'py, PyArray2<f32>>> {
        self.apply(py, x)
    }

    fn __repr__(&self) -> String {
        format!(
            "DenseLinear(in_features={}, out_features={}, bias={}, device={:?})",
            self.inner.in_features(),
            self.inner.out_features(),
            self.inner.bias().is_some(),
            self.inner.device().name()
        )
    }
}

/// Evaluates `Y = X @ weight^T + bias` in `f32` precision on CPU or WebGPU.
#[pyfunction]
#[pyo3(signature = (x, weight, bias=None, device=None))]
pub fn dense_apply<'py>(
    py: Python<'py>,
    x: &Bound<'_, PyAny>,
    weight: &Bound<'_, PyAny>,
    bias: Option<&Bound<'_, PyAny>>,
    device: Option<&Bound<'_, PyAny>>,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let (m, k, x_vec) = extract_2d(x, "x")?;
    let (n, wk, w_vec) = extract_2d(weight, "weight")?;
    if k != wk {
        return Err(PyValueError::new_err(format!(
            "x has {k} features but weight has {wk}"
        )));
    }
    let b_vec = bias.map(|b| extract_1d(b, "bias")).transpose()?;
    let dev = dense_device(device)?;
    let out = py
        .detach(|| baux::webgpu::dense_apply(m, n, k, &x_vec, &w_vec, b_vec.as_deref(), dev))
        .map_err(PyValueError::new_err)?;
    let arr =
        Array2::from_shape_vec((m, n), out).map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(arr.into_pyarray(py))
}

/// Evaluates pairwise `f32` squared Euclidean distance `||a - b||^2` on CPU or
/// WebGPU.
#[pyfunction]
#[pyo3(signature = (a, b, device=None))]
pub fn dense_dist2<'py>(
    py: Python<'py>,
    a: &Bound<'_, PyAny>,
    b: &Bound<'_, PyAny>,
    device: Option<&Bound<'_, PyAny>>,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let (q, d, a_vec) = extract_2d(a, "a")?;
    let (n, bd, b_vec) = extract_2d(b, "b")?;
    if d != bd {
        return Err(PyValueError::new_err(format!(
            "a has {d} dimensions but b has {bd}"
        )));
    }
    let dev = dense_device(device)?;
    let out = py
        .detach(|| baux::webgpu::dense_dist2(q, n, d, &a_vec, &b_vec, dev))
        .map_err(PyValueError::new_err)?;
    let arr =
        Array2::from_shape_vec((q, n), out).map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(arr.into_pyarray(py))
}
