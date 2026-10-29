//! Posterior-mean optimizer transition, isolated for source-bound extraction.

use ndarray::{Array2, ArrayView2};

use crate::coordinate::to_unit;
use crate::error::ENNError;
use crate::surrogate_api::Surrogate;

pub fn posterior_mu(
    num_dim: usize,
    bounds: &ArrayView2<f64>,
    surrogate: Option<&(dyn Surrogate + Send + Sync)>,
    x: &ArrayView2<f64>,
) -> Result<Array2<f64>, ENNError> {
    if x.ncols() != num_dim {
        return Err(ENNError::InvalidShape {
            expected: vec![x.nrows(), num_dim],
            got: vec![x.nrows(), x.ncols()],
        });
    }
    let surrogate =
        surrogate.ok_or_else(|| ENNError::InvalidParameter("No surrogate".to_string()))?;
    let unit = to_unit(x, bounds);
    Ok(surrogate.predict(&unit.view())?.mu)
}
