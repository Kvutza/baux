//! ENN model Python bindings.

#![allow(clippy::too_many_arguments, clippy::type_complexity)]

use crate::py_device::{apply_metric, metric_dict, parse_storage, py_flags, resolve_driver};
use baux::traits::PosteriorComputation;
use numpy::{IntoPyArray, PyArray2, PyArrayDyn, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::path::PathBuf;

pub(crate) type PosteriorPyOut<'py> = (
    Bound<'py, PyArrayDyn<f64>>,
    Bound<'py, PyArrayDyn<f64>>,
    Bound<'py, PyArrayDyn<f64>>,
    Bound<'py, PyArrayDyn<f64>>,
    Option<Bound<'py, PyArrayDyn<i64>>>,
);

type PyRows<'py> = (
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray2<f64>>,
    Option<Bound<'py, PyArray2<f64>>>,
);

/// Python wrapper for ENN
#[pyclass(unsendable, name = "ENN")]
pub struct PyENN {
    pub(crate) inner: baux::ENN,
    pub(crate) driver: baux::IndexDriver,
}

#[pymethods]
impl PyENN {
    /// Device FP32 queries in; independent device posterior snapshot out.
    #[pyo3(signature=(x,k_neighbors,epistemic_scale,aleatoric_scale,exclude_nearest=false,observation_noise=false,tie_mode=1))]
    fn device_posterior(
        &self,
        x: &Bound<'_, PyAny>,
        k_neighbors: i32,
        epistemic_scale: f64,
        aleatoric_scale: f64,
        exclude_nearest: bool,
        observation_noise: bool,
        tie_mode: u8,
    ) -> PyResult<crate::py_posterior::PyDevicePosterior> {
        let (device, index): (i32, i32) = x.call_method0("__dlpack_device__")?.extract()?;
        if index != 0 {
            return Err(PyValueError::new_err(
                "device posterior currently requires device index 0",
            ));
        }
        let input = crate::dlpack::Input::posterior(x, device)?;
        if input.shape.len() != 2 || input.shape[1] != self.inner.num_dim() {
            return Err(PyValueError::new_err(
                "device queries must have shape (queries, dimensions)",
            ));
        }
        let params = baux::ENNParams::new(k_neighbors, epistemic_scale, aleatoric_scale)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let flags = py_flags(tie_mode, exclude_nearest, observation_noise);
        // SAFETY: validated DLPack dtype, shape and device. Input retains the
        // producer allocation until native compute and its copies complete.
        let posterior = unsafe {
            self.inner.device_posterior(
                input.pointer,
                input.offset,
                device,
                input.shape[0],
                &params,
                &flags,
            )
        }
        .map_err(PyValueError::new_err)?;
        Ok(crate::py_posterior::PyDevicePosterior::new(posterior))
    }

    #[new]
    #[pyo3(signature = (train_x, train_y, train_yvar=None, scale_x=false, index_driver="Exact", work_dir=None, enn_storage=None, y_bounds=None, device=None, index=None, storage=None, metric=None))]
    fn new(
        train_x: PyReadonlyArray2<f64>,
        train_y: PyReadonlyArray2<f64>,
        train_yvar: Option<PyReadonlyArray2<f64>>,
        scale_x: bool,
        index_driver: &str,
        work_dir: Option<&str>,
        enn_storage: Option<&str>,
        y_bounds: Option<PyReadonlyArray2<f64>>,
        device: Option<&Bound<'_, PyAny>>,
        index: Option<&str>,
        storage: Option<&str>,
        metric: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        let driver = resolve_driver(index_driver, device, index)?;
        let storage = parse_storage(enn_storage, storage, work_dir, driver)?;
        let work_dir = work_dir.map(PathBuf::from);
        if driver == baux::IndexDriver::BpAnnDisk {
            baux::bpann_config().map_err(PyValueError::new_err)?;
        }
        let mut model = baux::ENN::new_options(
            train_x.as_array().to_owned(),
            train_y.as_array().to_owned(),
            train_yvar.map(|v| v.as_array().to_owned()),
            baux::ModelOptions {
                scale_x,
                driver,
                storage,
                work_dir,
                y_bounds: y_bounds.map(|v| v.as_array().to_owned()),
            },
        )
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
        apply_metric(&mut model, metric, scale_x)?;
        Ok(Self {
            inner: model,
            driver,
        })
    }

    #[getter]
    fn device(&self) -> &'static str {
        match self.driver {
            baux::IndexDriver::Metal => "metal",
            baux::IndexDriver::Cuda | baux::IndexDriver::Cuvs | baux::IndexDriver::CuvsCagra => {
                "cuda"
            }
            baux::IndexDriver::OpenCl => "opencl",
            _ => "cpu",
        }
    }

    #[pyo3(signature = (x, y, yvar=None))]
    fn add(
        &mut self,
        x: PyReadonlyArray2<f64>,
        y: PyReadonlyArray2<f64>,
        yvar: Option<PyReadonlyArray2<f64>>,
    ) -> PyResult<()> {
        let yvar_arr = yvar.as_ref().map(|v| v.as_array());
        self.inner
            .add(&x.as_array(), &y.as_array(), yvar_arr.as_ref())
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    #[pyo3(signature = (x=None, y=None, yvar=None, shrinkage=None, calibrator=None, metric=None, local_k=None))]
    fn append_cow(
        &self,
        x: Option<PyReadonlyArray2<f64>>,
        y: Option<PyReadonlyArray2<f64>>,
        yvar: Option<PyReadonlyArray2<f64>>,
        shrinkage: Option<PyReadonlyArray2<f64>>,
        calibrator: Option<(Vec<f64>, Vec<f64>, Vec<f64>)>,
        metric: Option<&str>,
        local_k: Option<usize>,
    ) -> PyResult<Self> {
        let mut inner = match (x, y) {
            (Some(xa), Some(ya)) => {
                let yvar_arr = yvar.as_ref().map(|v| v.as_array());
                self.inner
                    .append_cow(&xa.as_array(), &ya.as_array(), yvar_arr.as_ref())
                    .map_err(|e| PyValueError::new_err(e.to_string()))?
            }
            (None, None) => self.inner.clone(),
            _ => return Err(PyValueError::new_err("x and y must both be provided")),
        };
        if let Some(loadings) = shrinkage {
            let arr = loadings.as_array();
            let sh = baux::Shrinkage::new(arr.iter().copied().collect(), arr.nrows(), arr.ncols())
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
            inner = inner
                .with_shrinkage(sh)
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
        }
        if let Some((a, b, c)) = calibrator {
            let cal =
                baux::Calibrator::new(a, b, c).map_err(|e| PyValueError::new_err(e.to_string()))?;
            inner = inner
                .with_calibrator(cal)
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
        }
        if let Some(m) = metric {
            inner = inner.with_geometry(parse_dist(m, local_k)?);
        }
        Ok(Self {
            inner,
            driver: self.driver,
        })
    }

    fn ensure_sync(&self) -> PyResult<()> {
        self.inner
            .index_access()
            .ensure_sync()
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    fn schedule_flush(&self) -> PyResult<()> {
        self.inner
            .schedule_flush()
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    fn persist_index(&self) -> PyResult<()> {
        self.inner
            .persist_index()
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    fn index_bytes(&self) -> PyResult<usize> {
        self.inner
            .index_access()
            .memory_bytes()
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    #[getter]
    fn y_bounds<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        self.inner.y_bounds().clone().into_pyarray(py)
    }

    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (x, k_neighbors, epistemic_scale, aleatoric_scale, exclude_nearest=false, observation_noise=false, tie_mode=1))]
    fn posterior<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<f64>,
        k_neighbors: i32,
        epistemic_scale: f64,
        aleatoric_scale: f64,
        exclude_nearest: bool,
        observation_noise: bool,
        tie_mode: u8,
    ) -> PyResult<PosteriorPyOut<'py>> {
        let params = baux::ENNParams::new(k_neighbors, epistemic_scale, aleatoric_scale)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let flags = py_flags(tie_mode, exclude_nearest, observation_noise);
        let out = self
            .inner
            .posterior(&x.as_array(), &params, &flags)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok((
            out.mu.into_pyarray(py),
            out.se.into_pyarray(py),
            out.se_epi.into_pyarray(py),
            out.se_ale.into_pyarray(py),
            out.idx.map(|idx| idx.into_dyn().into_pyarray(py)),
        ))
    }

    /// Batch posterior with multiple parameter sets.
    #[allow(clippy::too_many_arguments, clippy::type_complexity)]
    #[pyo3(signature = (x, k_values, epistemic_scales, aleatoric_scales, exclude_nearest=false, observation_noise=false, tie_mode=1))]
    fn batch_posterior<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<f64>,
        k_values: Vec<i32>,
        epistemic_scales: Vec<f64>,
        aleatoric_scales: Vec<f64>,
        exclude_nearest: bool,
        observation_noise: bool,
        tie_mode: u8,
    ) -> PyResult<(
        Bound<'py, PyArrayDyn<f64>>,
        Bound<'py, PyArrayDyn<f64>>,
        Bound<'py, PyArrayDyn<f64>>,
        Bound<'py, PyArrayDyn<f64>>,
    )> {
        // Build params list
        let n_params = k_values.len();
        if epistemic_scales.len() != n_params || aleatoric_scales.len() != n_params {
            return Err(PyValueError::new_err(
                "k_values, epistemic_scales, and aleatoric_scales must have same length",
            ));
        }

        let mut paramss = Vec::with_capacity(n_params);
        for i in 0..n_params {
            let params =
                baux::ENNParams::new(k_values[i], epistemic_scales[i], aleatoric_scales[i])
                    .map_err(|e| PyValueError::new_err(e.to_string()))?;
            paramss.push(params);
        }

        let flags = py_flags(tie_mode, exclude_nearest, observation_noise);

        let out = self
            .inner
            .batch_posterior(&x.as_array(), &paramss, &flags)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok((
            out.mu.into_pyarray(py),
            out.se.into_pyarray(py),
            out.se_epi.into_pyarray(py),
            out.se_ale.into_pyarray(py),
        ))
    }

    /// Posterior function draw - sample from posterior predictive.
    #[allow(clippy::too_many_arguments, clippy::type_complexity)]
    #[pyo3(signature = (x, k_neighbors, epistemic_scale, aleatoric_scale, function_seeds, exclude_nearest=false, observation_noise=false, tie_mode=1))]
    fn posterior_draw<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<f64>,
        k_neighbors: i32,
        epistemic_scale: f64,
        aleatoric_scale: f64,
        function_seeds: Vec<i64>,
        exclude_nearest: bool,
        observation_noise: bool,
        tie_mode: u8,
    ) -> PyResult<(Bound<'py, PyArrayDyn<f64>>, Vec<Vec<usize>>)> {
        let params = baux::ENNParams::new(k_neighbors, epistemic_scale, aleatoric_scale)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let flags = py_flags(tie_mode, exclude_nearest, observation_noise);
        let (draws, idx) = self
            .inner
            .posterior_draw(&x.as_array(), &params, &function_seeds, &flags)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok((draws.into_dyn().into_pyarray(py), idx))
    }

    /// Conditional posterior with what-if scenarios.
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (x_whatif, y_whatif, x, k_neighbors, epistemic_scale, aleatoric_scale, exclude_nearest=false, observation_noise=false, tie_mode=1))]
    fn conditional_posterior<'py>(
        &self,
        py: Python<'py>,
        x_whatif: PyReadonlyArray2<f64>,
        y_whatif: PyReadonlyArray2<f64>,
        x: PyReadonlyArray2<f64>,
        k_neighbors: i32,
        epistemic_scale: f64,
        aleatoric_scale: f64,
        exclude_nearest: bool,
        observation_noise: bool,
        tie_mode: u8,
    ) -> PyResult<PosteriorPyOut<'py>> {
        let params = baux::ENNParams::new(k_neighbors, epistemic_scale, aleatoric_scale)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let flags = py_flags(tie_mode, exclude_nearest, observation_noise);
        let out = self
            .inner
            .conditional_posterior(
                &x_whatif.as_array(),
                &y_whatif.as_array(),
                &x.as_array(),
                &params,
                &flags,
            )
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok((
            out.mu.into_pyarray(py),
            out.se.into_pyarray(py),
            out.se_epi.into_pyarray(py),
            out.se_ale.into_pyarray(py),
            out.idx.map(|idx| idx.into_dyn().into_pyarray(py)),
        ))
    }

    /// Conditional posterior function draw.
    #[allow(clippy::too_many_arguments, clippy::type_complexity)]
    #[pyo3(signature = (x_whatif, y_whatif, x, k_neighbors, epistemic_scale, aleatoric_scale, function_seeds, exclude_nearest=false, observation_noise=false, tie_mode=1))]
    fn conditional_draw<'py>(
        &self,
        py: Python<'py>,
        x_whatif: PyReadonlyArray2<f64>,
        y_whatif: PyReadonlyArray2<f64>,
        x: PyReadonlyArray2<f64>,
        k_neighbors: i32,
        epistemic_scale: f64,
        aleatoric_scale: f64,
        function_seeds: Vec<i64>,
        exclude_nearest: bool,
        observation_noise: bool,
        tie_mode: u8,
    ) -> PyResult<(Bound<'py, PyArrayDyn<f64>>, Vec<Vec<usize>>)> {
        let params = baux::ENNParams::new(k_neighbors, epistemic_scale, aleatoric_scale)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let flags = py_flags(tie_mode, exclude_nearest, observation_noise);
        let (draws, idx) = self
            .inner
            .conditional_draw(
                &x_whatif.as_array(),
                &y_whatif.as_array(),
                &x.as_array(),
                &params,
                &function_seeds,
                &flags,
            )
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok((draws.into_dyn().into_pyarray(py), idx))
    }

    /// Get k nearest neighbors for query points.
    #[pyo3(signature = (x, k, exclude_nearest=false))]
    fn neighbors<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<f64>,
        k: i32,
        exclude_nearest: bool,
    ) -> PyResult<Bound<'py, PyArrayDyn<usize>>> {
        let result = self
            .inner
            .neighbors(&x.as_array(), k, exclude_nearest)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(result.into_dyn().into_pyarray(py))
    }

    #[allow(clippy::type_complexity)]
    #[pyo3(signature = (x, search_k, exclude_nearest=false))]
    fn nearest_neighbors<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<f64>,
        search_k: i32,
        exclude_nearest: bool,
    ) -> PyResult<(Bound<'py, PyArrayDyn<f64>>, Bound<'py, PyArrayDyn<i64>>)> {
        let (dist2s, idx) = self
            .inner
            .index_access()
            .nearest_neighbors(&x.as_array(), search_k, exclude_nearest)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok((
            dist2s.into_dyn().into_pyarray(py),
            idx.into_dyn().into_pyarray(py),
        ))
    }

    #[allow(clippy::type_complexity)]
    #[pyo3(signature = (x, search_k, exclude_nearest=false, tie_neighbors=true))]
    fn posterior_neighbors<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<f64>,
        search_k: i32,
        exclude_nearest: bool,
        tie_neighbors: bool,
    ) -> PyResult<(Bound<'py, PyArrayDyn<f64>>, Bound<'py, PyArrayDyn<i64>>)> {
        let (dist2s, idx) = self
            .inner
            .index_access()
            .posterior_neighbors(&x.as_array(), search_k, exclude_nearest, tie_neighbors)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok((
            dist2s.into_dyn().into_pyarray(py),
            idx.into_dyn().into_pyarray(py),
        ))
    }

    fn __len__(&self) -> usize {
        self.inner.len()
    }

    #[getter]
    fn num_outputs(&self) -> usize {
        self.inner.num_outputs()
    }

    #[getter]
    fn num_dim(&self) -> usize {
        self.inner.num_dim()
    }

    #[getter]
    fn diagonal_metric<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<Option<Bound<'py, pyo3::types::PyDict>>> {
        metric_dict(py, &self.inner)
    }

    fn train_rows<'py>(&self, py: Python<'py>, indices: Vec<usize>) -> PyResult<PyRows<'py>> {
        let (x, y, yvar) = self
            .inner
            .natural_rows(&indices)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok((
            x.into_pyarray(py),
            y.into_pyarray(py),
            yvar.map(|a| a.into_pyarray(py)),
        ))
    }

    fn row_y<'py>(&self, py: Python<'py>, i: usize) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let row = self
            .inner
            .natural_y(i)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(row.insert_axis(ndarray::Axis(0)).into_pyarray(py))
    }
}

fn parse_dist(
    metric: &str,
    local_k: Option<usize>,
) -> PyResult<baux::semantics::neighbor::DistanceMetric> {
    use baux::semantics::neighbor::DistanceMetric;
    let k = local_k.unwrap_or(8).max(1);
    match metric.to_ascii_lowercase().as_str() {
        "euclidean" | "l2" | "squared_l2" => Ok(DistanceMetric::Euclidean),
        "conformal" => Ok(DistanceMetric::Conformal { local_k: k }),
        "hyperbolic" | "poincare" | "halfspace" => Ok(DistanceMetric::Hyperbolic { local_k: k }),
        "wasserstein" | "ot" => Ok(DistanceMetric::Wasserstein { local_k: k }),
        _ => Err(PyValueError::new_err(format!(
            "Unknown distance metric: {metric}"
        ))),
    }
}
