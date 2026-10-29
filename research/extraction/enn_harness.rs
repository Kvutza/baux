//! Extraction-only entry points for ENN's public optimizer methods.
//!
//! This module is copied into the isolated pinned-source snapshot.  Each
//! function is deliberately a single call so that Charon can start from a free
//! function while preserving the implementation reached through the inherent
//! `Optimizer` methods.

use ndarray::{Array2, ArrayView2};

use crate::error::ENNError;
use crate::optimizer::Optimizer;

pub fn optimizer_ask(optimizer: &mut Optimizer, num_arms: usize) -> Result<Array2<f64>, ENNError> {
    optimizer.ask(num_arms)
}

pub fn optimizer_tell(
    optimizer: &mut Optimizer,
    x: &ArrayView2<f64>,
    y: &ArrayView2<f64>,
    yvar: Option<&ArrayView2<f64>>,
) -> Result<(), ENNError> {
    optimizer.tell(x, y, yvar)
}

pub fn optimizer_mu(optimizer: &Optimizer, x: &ArrayView2<f64>) -> Result<Array2<f64>, ENNError> {
    optimizer.posterior_mu(x)
}
