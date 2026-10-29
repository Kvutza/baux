//! Backend-independent surrogate interface used by the optimizer state machine.

use ndarray::{Array1, Array2, Array3, ArrayView2};
use rand::RngCore;

use crate::error::ENNError;

#[derive(Debug, Clone)]
pub struct SurrogatePrediction {
    pub mu: Array2<f64>,
    pub se: Array2<f64>,
}

pub trait Surrogate: Send + Sync {
    fn fit(
        &mut self,
        x: &ArrayView2<f64>,
        y: &ArrayView2<f64>,
        yvar: Option<&ArrayView2<f64>>,
        rng: &mut dyn RngCore,
    ) -> Result<(), ENNError>;

    fn fit_append(
        &mut self,
        x_new: &ArrayView2<f64>,
        y_new: &ArrayView2<f64>,
        yvar_new: Option<&ArrayView2<f64>>,
        rng: &mut dyn RngCore,
    ) -> Result<(), ENNError> {
        let _ = (x_new, y_new, yvar_new, rng);
        Err(ENNError::InvalidParameter(
            "fit_append not supported for this surrogate".to_string(),
        ))
    }

    fn predict(&self, x: &ArrayView2<f64>) -> Result<SurrogatePrediction, ENNError>;

    fn sample(
        &self,
        x: &ArrayView2<f64>,
        num_samples: usize,
        rng: &mut dyn RngCore,
    ) -> Result<Array3<f64>, ENNError>;

    fn lengthscales(&self) -> Option<Array1<f64>>;

    fn fitted_metrics(&self) -> Option<usize> {
        None
    }

    fn observation_count(&self) -> Option<usize> {
        None
    }

    fn x_row(&self, idx: usize) -> Result<Array1<f64>, ENNError> {
        let _ = idx;
        Err(ENNError::InvalidParameter(
            "x_row not supported for this surrogate".to_string(),
        ))
    }

    fn y_row(&self, idx: usize) -> Result<Array1<f64>, ENNError> {
        let _ = idx;
        Err(ENNError::InvalidParameter(
            "y_row not supported for this surrogate".to_string(),
        ))
    }

    fn observations_y(&self) -> Result<Option<Array2<f64>>, ENNError> {
        Ok(None)
    }

    fn observations_x(&self) -> Result<Option<Array2<f64>>, ENNError> {
        Ok(None)
    }

    fn schedule_flush(&self) -> Result<(), ENNError> {
        Ok(())
    }

    fn wait_flush(&self) -> Result<(), ENNError> {
        Ok(())
    }

    fn release_pages(&self) -> Result<(), ENNError> {
        Ok(())
    }

    fn clear_observations(&mut self) -> Result<(), ENNError> {
        Err(ENNError::InvalidParameter(
            "clear_observations not supported for this surrogate".to_string(),
        ))
    }

    fn y_bounds(&self) -> Option<&Array2<f64>> {
        None
    }

    fn naturalize_y(&self, z: &ArrayView2<f64>) -> Array2<f64> {
        z.to_owned()
    }

    fn warp_y(&self, y: &ArrayView2<f64>) -> Result<Array2<f64>, ENNError> {
        Ok(y.to_owned())
    }
}

pub type BoxedSurrogate = Box<dyn Surrogate + Send + Sync>;
