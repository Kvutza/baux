//! Extraction-only roots for the production optimizer API.
//!
//! The `extraction` Cargo feature is disabled in normal builds. These roots
//! target the same natural-coordinate methods used by the Python adapter.
//! A probe must confirm that their method bodies are retained in LLBC.

use ndarray::{Array2, ArrayView2};
use rand::RngCore;

use crate::{BayesOptimizer, ENNError};

pub fn optimizer_ask(
    optimizer: BayesOptimizer,
    num_arms: usize,
    rng: &mut (impl RngCore + ?Sized),
) -> Result<(BayesOptimizer, Array2<f64>), ENNError> {
    optimizer.ask_natural(num_arms, rng)
}

pub fn optimizer_tell(
    optimizer: BayesOptimizer,
    x: &ArrayView2<f64>,
    y: &ArrayView2<f64>,
    yvar: Option<&ArrayView2<f64>>,
    rng: &mut (impl RngCore + ?Sized),
) -> Result<BayesOptimizer, ENNError> {
    optimizer.tell_natural(x, y, yvar, rng)
}

pub fn optimizer_mu(
    optimizer: &BayesOptimizer,
    x: &ArrayView2<f64>,
) -> Result<Array2<f64>, ENNError> {
    optimizer.posterior_mu(x)
}
