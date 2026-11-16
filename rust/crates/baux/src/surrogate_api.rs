//! Backend-independent surrogate interface and split capability traits.

use std::borrow::Cow;

use ndarray::{Array1, Array2, Array3, ArrayView2};
use rand::RngCore;

use crate::error::ENNError;

#[derive(Debug, Clone)]
pub struct SurrogatePrediction {
    pub mu: Array2<f64>,
    pub se: Array2<f64>,
}

pub trait Surrogate: Send + Sync {
    fn with_fit(
        self: Box<Self>,
        x: &ArrayView2<f64>,
        y: &ArrayView2<f64>,
        yvar: Option<&ArrayView2<f64>>,
        rng: &mut dyn RngCore,
    ) -> Result<BoxedSurrogate, ENNError>;

    fn with_append(
        self: Box<Self>,
        x_new: &ArrayView2<f64>,
        y_new: &ArrayView2<f64>,
        yvar_new: Option<&ArrayView2<f64>>,
        rng: &mut dyn RngCore,
    ) -> Result<BoxedSurrogate, ENNError> {
        let _ = (x_new, y_new, yvar_new, rng);
        Err(ENNError::InvalidParameter(
            "with_append not supported for this surrogate".to_string(),
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

    /// Zero-copy borrowed row slice when backed by in-memory storage, or owned
    /// vector when reconstructed from disk.
    fn x_slice(&self, idx: usize) -> Result<Cow<'_, [f64]>, ENNError> {
        Ok(Cow::Owned(self.x_row(idx)?.into_raw_vec_and_offset().0))
    }

    /// Zero-copy borrowed target slice when backed by in-memory storage, or
    /// owned vector when reconstructed from disk.
    fn y_slice(&self, idx: usize) -> Result<Cow<'_, [f64]>, ENNError> {
        Ok(Cow::Owned(self.y_row(idx)?.into_raw_vec_and_offset().0))
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

    fn cleared(self: Box<Self>) -> Result<BoxedSurrogate, ENNError> {
        Err(ENNError::InvalidParameter(
            "cleared not supported for this surrogate".to_string(),
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

    fn clone_box(&self) -> Result<BoxedSurrogate, ENNError> {
        Err(ENNError::InvalidParameter(
            "surrogate does not support cloning".to_string(),
        ))
    }

    fn freeze_surrogate(&self, buf: &mut Vec<u8>) -> Result<(), ENNError> {
        let _ = buf;
        Err(ENNError::InvalidParameter(
            "surrogate does not support binary checkpointing".to_string(),
        ))
    }

    fn with_thawed(
        self: Box<Self>,
        bytes: &[u8],
        pos: &mut usize,
    ) -> Result<BoxedSurrogate, ENNError> {
        let _ = (bytes, pos);
        Err(ENNError::InvalidParameter(
            "with_thawed not supported for this surrogate".to_string(),
        ))
    }

    fn hash_surrogate(&self, buf: &mut Vec<u8>) {
        let _ = buf;
    }
}

impl dyn Surrogate {
    pub fn with_flushed(self: Box<Self>) -> Result<Box<Self>, ENNError> {
        self.schedule_flush()?;
        self.wait_flush()?;
        Ok(self)
    }

    pub fn with_released(self: Box<Self>) -> Result<Box<Self>, ENNError> {
        self.release_pages()?;
        Ok(self)
    }
}

pub type BoxedSurrogate = Box<dyn Surrogate + Send + Sync>;

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

/// Split capability trait with Generic Associated Type (`Row<'a>`) for
/// zero-copy borrowed observation access when rows reside in RAM, or owned
/// buffers when faulted from disk.
pub trait ObservationView: Send + Sync {
    type Row<'a>: core::ops::Deref<Target = [f64]>
    where
        Self: 'a;

    fn obs_len(&self) -> usize;
    fn view_x<'a>(&'a self, idx: usize) -> Result<Self::Row<'a>, ENNError>;
    fn view_y<'a>(&'a self, idx: usize) -> Result<Self::Row<'a>, ENNError>;
}

impl<T: Surrogate + ?Sized> ObservationView for T {
    type Row<'a>
        = Cow<'a, [f64]>
    where
        Self: 'a;

    #[inline]
    fn obs_len(&self) -> usize {
        self.observation_count().unwrap_or(0)
    }

    #[inline]
    fn view_x<'a>(&'a self, idx: usize) -> Result<Self::Row<'a>, ENNError> {
        self.x_slice(idx)
    }

    #[inline]
    fn view_y<'a>(&'a self, idx: usize) -> Result<Self::Row<'a>, ENNError> {
        self.y_slice(idx)
    }
}

/// Split capability trait: anisotropic trust-region metric lengthscales.
pub trait LengthscaleProvider: Send + Sync {
    fn metric_scales(&self) -> Option<Cow<'_, [f64]>>;
}

impl<T: Surrogate + ?Sized> LengthscaleProvider for T {
    #[inline]
    fn metric_scales(&self) -> Option<Cow<'_, [f64]>> {
        self.lengthscales()
            .map(|ls| Cow::Owned(ls.into_raw_vec_and_offset().0))
    }
}
