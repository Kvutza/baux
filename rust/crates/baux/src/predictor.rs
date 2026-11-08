//! Split capability trait: posterior mean and standard error prediction.

use ndarray::ArrayView2;

use crate::error::ENNError;
use crate::surrogate_api::{Surrogate, SurrogatePrediction};

/// Split capability trait: posterior mean and standard error prediction.
pub trait Predictor: Send + Sync {
    fn predict_batch(&self, x: &ArrayView2<f64>) -> Result<SurrogatePrediction, ENNError>;
}

impl<T: Surrogate + ?Sized> Predictor for T {
    #[inline]
    fn predict_batch(&self, x: &ArrayView2<f64>) -> Result<SurrogatePrediction, ENNError> {
        self.predict(x)
    }
}
