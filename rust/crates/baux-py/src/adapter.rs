//! Python surrogate adapter for the Rust optimizer.

use std::borrow::Cow;

use baux::error::ENNError;
use baux::scratch::ImLog;
use baux::surrogate::{BoxedSurrogate, Surrogate, SurrogatePrediction};
use ndarray::{Array1, Array2, Array3, ArrayView2};
use numpy::{IntoPyArray, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rand::RngCore;

fn surrogate_error(error: impl std::fmt::Display) -> ENNError {
    ENNError::InvalidParameter(format!("Python surrogate: {error}"))
}

/// Adapter for a Python object implementing batched `fit`, `predict`, and
/// `draw`.
///
/// Calls cross the language boundary once per full observation or candidate
/// batch. Observation ownership stays in Rust [`ImLog`] chunks so `Clone` /
/// `fork()` is $O(1)$ structural sharing and linear `tell` appends in place.
pub struct PythonSurrogateAdapter {
    provider: Py<PyAny>,
    num_dim: usize,
    x_obs: ImLog,
    y_obs: ImLog,
    yvar: Option<ImLog>,
    has_yvar: Option<bool>,
    lengthscales: Option<Array1<f64>>,
    num_steps: usize,
}

impl Clone for PythonSurrogateAdapter {
    fn clone(&self) -> Self {
        let provider = Python::attach(|py| {
            let bound = self.provider.bind(py);
            if bound.hasattr("fork").unwrap_or(false) {
                if let Ok(forked) = bound.call_method0("fork") {
                    return forked.unbind();
                }
            }
            if bound.hasattr("copy").unwrap_or(false) {
                if let Ok(copied) = bound.call_method0("copy") {
                    return copied.unbind();
                }
            }
            self.provider.clone_ref(py)
        });
        Self {
            provider,
            num_dim: self.num_dim,
            x_obs: self.x_obs.clone(),
            y_obs: self.y_obs.clone(),
            yvar: self.yvar.clone(),
            has_yvar: self.has_yvar,
            lengthscales: self.lengthscales.clone(),
            num_steps: self.num_steps,
        }
    }
}

impl PythonSurrogateAdapter {
    pub fn new(provider: Py<PyAny>, num_dim: usize, num_steps: usize) -> Self {
        Self {
            provider,
            num_dim,
            x_obs: ImLog::with_cols(num_dim),
            y_obs: ImLog::empty(),
            yvar: None,
            has_yvar: None,
            lengthscales: None,
            num_steps,
        }
    }

    fn call_fit(&mut self, x: &ImLog, y: &ImLog, yvar: Option<&ImLog>) -> Result<(), ENNError> {
        let num_dim = self.num_dim;
        let lengthscales = Python::attach(|py| -> PyResult<Option<Array1<f64>>> {
            let x_py = x.to_array().into_pyarray(py);
            let y_py = y.to_array().into_pyarray(py);
            let yvar_py = match yvar {
                Some(values) => values.to_array().into_pyarray(py).into_any().unbind(),
                None => py.None(),
            };
            let kwargs = PyDict::new(py);
            kwargs.set_item("steps", self.num_steps)?;
            let result =
                self.provider
                    .bind(py)
                    .call_method("fit", (x_py, y_py, yvar_py), Some(&kwargs))?;
            let extracted = result
                .getattr("lengthscales")?
                .extract::<Option<PyReadonlyArray1<'_, f64>>>()?;
            Ok(extracted.map(|values| values.as_array().to_owned()))
        })
        .map_err(surrogate_error)?;
        if let Some(values) = lengthscales.as_ref() {
            if values.len() != num_dim
                || values
                    .iter()
                    .any(|value| !value.is_finite() || *value <= 0.0)
            {
                return Err(surrogate_error(format!(
                    "lengthscales must contain {num_dim} finite positive values"
                )));
            }
        }
        self.lengthscales = lengthscales;
        Ok(())
    }

    fn validate_batch(
        &self,
        x: &ArrayView2<'_, f64>,
        y: &ArrayView2<'_, f64>,
        yvar: Option<&ArrayView2<'_, f64>>,
    ) -> Result<(), ENNError> {
        if x.nrows() != y.nrows() || x.ncols() != self.num_dim {
            return Err(ENNError::InvalidShape {
                expected: vec![x.nrows(), self.num_dim],
                got: vec![y.nrows(), x.ncols()],
            });
        }
        if self.y_obs.ncols() != 0 && y.ncols() != self.y_obs.ncols() {
            return Err(ENNError::InvalidParameter(format!(
                "y has {} columns but the fitted surrogate has {}",
                y.ncols(),
                self.y_obs.ncols()
            )));
        }
        if let Some(variance) = yvar {
            if variance.shape() != y.shape() {
                return Err(ENNError::InvalidShape {
                    expected: y.shape().to_vec(),
                    got: variance.shape().to_vec(),
                });
            }
        }
        if let Some(has_yvar) = self.has_yvar {
            if has_yvar != yvar.is_some() {
                return Err(ENNError::InvalidParameter(format!(
                    "y_var must be {} on every tell()",
                    if has_yvar { "provided" } else { "omitted" }
                )));
            }
        }
        Ok(())
    }
}

impl Surrogate for PythonSurrogateAdapter {
    fn with_fit(
        mut self: Box<Self>,
        x: &ArrayView2<f64>,
        y: &ArrayView2<f64>,
        yvar: Option<&ArrayView2<f64>>,
        _rng: &mut dyn RngCore,
    ) -> Result<BoxedSurrogate, ENNError> {
        self.validate_batch(x, y, yvar)?;
        let x_new = ImLog::with_cols(self.num_dim).push_rows(x)?;
        let y_new = ImLog::with_cols(y.ncols()).push_rows(y)?;
        let yvar_new = match yvar {
            Some(values) => Some(ImLog::with_cols(values.ncols()).push_rows(values)?),
            None => None,
        };
        self.call_fit(&x_new, &y_new, yvar_new.as_ref())?;
        self.x_obs = x_new;
        self.y_obs = y_new;
        self.yvar = yvar_new;
        self.has_yvar = Some(yvar.is_some());
        Ok(self as BoxedSurrogate)
    }

    fn with_append(
        mut self: Box<Self>,
        x_new: &ArrayView2<f64>,
        y_new: &ArrayView2<f64>,
        yvar_new: Option<&ArrayView2<f64>>,
        _rng: &mut dyn RngCore,
    ) -> Result<BoxedSurrogate, ENNError> {
        self.validate_batch(x_new, y_new, yvar_new)?;
        let prev_x = std::mem::take(&mut self.x_obs);
        let prev_y = std::mem::take(&mut self.y_obs);
        let prev_yvar = self.yvar.take();

        let next_x = prev_x.push_rows(x_new)?;
        let next_y = prev_y.push_rows(y_new)?;
        let next_yvar = match (prev_yvar, yvar_new) {
            (Some(current), Some(rows)) => Some(current.push_rows(rows)?),
            (None, Some(rows)) => Some(ImLog::with_cols(rows.ncols()).push_rows(rows)?),
            (None, None) => None,
            (Some(_), None) => unreachable!("validate_batch rejects a mixed y_var stream"),
        };
        self.call_fit(&next_x, &next_y, next_yvar.as_ref())?;
        self.x_obs = next_x;
        self.y_obs = next_y;
        self.yvar = next_yvar;
        self.has_yvar = Some(yvar_new.is_some());
        Ok(self as BoxedSurrogate)
    }

    fn predict(&self, x: &ArrayView2<f64>) -> Result<SurrogatePrediction, ENNError> {
        let pred = Python::attach(|py| -> PyResult<SurrogatePrediction> {
            let x_py = x.to_owned().into_pyarray(py);
            let result = self.provider.bind(py).call_method1("predict", (x_py,))?;
            let mu = result
                .getattr("mu")?
                .extract::<PyReadonlyArray2<'_, f64>>()?
                .as_array()
                .to_owned();
            let se = result
                .getattr("sigma")?
                .extract::<PyReadonlyArray2<'_, f64>>()?
                .as_array()
                .to_owned();
            Ok(SurrogatePrediction { mu, se })
        })
        .map_err(surrogate_error)?;
        let shape = [x.nrows(), self.y_obs.ncols()];
        if pred.mu.shape() != shape || pred.se.shape() != shape {
            return Err(surrogate_error(format!(
                "predict must return mu and sigma with shape {shape:?}"
            )));
        }
        if pred.mu.iter().any(|value| !value.is_finite())
            || pred
                .se
                .iter()
                .any(|value| !value.is_finite() || *value < 0.0)
        {
            return Err(surrogate_error(
                "predict returned non-finite values or negative sigma",
            ));
        }
        Ok(pred)
    }

    fn sample(
        &self,
        x: &ArrayView2<f64>,
        num_samples: usize,
        rng: &mut dyn RngCore,
    ) -> Result<Array3<f64>, ENNError> {
        let seed = rng.next_u64() & ((1_u64 << 63) - 1);
        let samples = Python::attach(|py| -> PyResult<Array3<f64>> {
            let x_py = x.to_owned().into_pyarray(py);
            Ok(self
                .provider
                .bind(py)
                .call_method1("draw", (x_py, num_samples, seed))?
                .extract::<numpy::PyReadonlyArray3<'_, f64>>()
                .map(|samples| samples.as_array().to_owned())?)
        })
        .map_err(surrogate_error)?;
        let shape = [num_samples, x.nrows(), self.y_obs.ncols()];
        if samples.shape() != shape || samples.iter().any(|value| !value.is_finite()) {
            return Err(surrogate_error(format!(
                "draw must return finite samples with shape {shape:?}"
            )));
        }
        Ok(samples)
    }

    fn lengthscales(&self) -> Option<Array1<f64>> {
        self.lengthscales.clone()
    }

    fn fitted_metrics(&self) -> Option<usize> {
        (self.y_obs.ncols() > 0).then_some(self.y_obs.ncols())
    }

    fn observation_count(&self) -> Option<usize> {
        Some(self.x_obs.nrows())
    }

    fn x_slice(&self, idx: usize) -> Result<Cow<'_, [f64]>, ENNError> {
        if idx >= self.x_obs.nrows() {
            return Err(ENNError::InvalidParameter(format!(
                "observation index {idx} out of bounds"
            )));
        }
        Ok(Cow::Borrowed(self.x_obs.row_slice(idx)))
    }

    fn y_slice(&self, idx: usize) -> Result<Cow<'_, [f64]>, ENNError> {
        if idx >= self.y_obs.nrows() {
            return Err(ENNError::InvalidParameter(format!(
                "observation index {idx} out of bounds"
            )));
        }
        Ok(Cow::Borrowed(self.y_obs.row_slice(idx)))
    }

    fn x_row(&self, idx: usize) -> Result<Array1<f64>, ENNError> {
        if idx >= self.x_obs.nrows() {
            return Err(ENNError::InvalidParameter(format!(
                "observation index {idx} out of bounds"
            )));
        }
        Ok(self.x_obs.row_vec(idx))
    }

    fn y_row(&self, idx: usize) -> Result<Array1<f64>, ENNError> {
        if idx >= self.y_obs.nrows() {
            return Err(ENNError::InvalidParameter(format!(
                "observation index {idx} out of bounds"
            )));
        }
        Ok(self.y_obs.row_vec(idx))
    }

    fn observations_x(&self) -> Result<Option<Array2<f64>>, ENNError> {
        Ok((self.x_obs.nrows() > 0).then(|| self.x_obs.to_array()))
    }

    fn observations_y(&self) -> Result<Option<Array2<f64>>, ENNError> {
        Ok((self.y_obs.nrows() > 0).then(|| self.y_obs.to_array()))
    }

    fn cleared(mut self: Box<Self>) -> Result<BoxedSurrogate, ENNError> {
        self.x_obs = ImLog::with_cols(self.num_dim);
        self.y_obs = ImLog::empty();
        self.yvar = None;
        self.has_yvar = None;
        self.lengthscales = None;
        Ok(self as BoxedSurrogate)
    }

    fn clone_box(&self) -> Result<BoxedSurrogate, ENNError> {
        Ok(Box::new(self.clone()))
    }
}

const _: () = {
    fn assert_sync<T: Send + Sync>() {}
    fn check() {
        assert_sync::<PythonSurrogateAdapter>();
    }
};
