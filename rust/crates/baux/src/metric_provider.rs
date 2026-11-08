//! Split capability trait: anisotropic trust-region metric lengthscales.

use std::borrow::Cow;

use crate::surrogate_api::Surrogate;

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
