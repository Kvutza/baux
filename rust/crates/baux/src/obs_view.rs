//! Split capability trait with Generic Associated Type (`Row<'a>`) for
//! zero-copy borrowed observation access.

use std::borrow::Cow;

use crate::error::ENNError;
use crate::surrogate_api::Surrogate;

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
