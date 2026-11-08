//! Split capability trait: incremental observation ingestion.

use ndarray::ArrayView2;
use rand::RngCore;

use crate::error::ENNError;
use crate::predictor::Predictor;
use crate::surrogate_api::Surrogate;

/// Split capability trait: incremental observation ingestion.
pub trait IncrementalSurrogate: Predictor {
    fn append_batch(
        &mut self,
        x_new: &ArrayView2<f64>,
        y_new: &ArrayView2<f64>,
        yvar_new: Option<&ArrayView2<f64>>,
        rng: &mut dyn RngCore,
    ) -> Result<(), ENNError>;
}

impl<T: Surrogate + ?Sized> IncrementalSurrogate for T {
    #[inline]
    fn append_batch(
        &mut self,
        x_new: &ArrayView2<f64>,
        y_new: &ArrayView2<f64>,
        yvar_new: Option<&ArrayView2<f64>>,
        rng: &mut dyn RngCore,
    ) -> Result<(), ENNError> {
        self.fit_append(x_new, y_new, yvar_new, rng)
    }
}
