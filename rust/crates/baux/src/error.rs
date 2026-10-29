//! Error types for ENN operations.

use thiserror::Error;

#[derive(Error, Debug, Clone, PartialEq)]
pub enum IndexError {
    #[error("Invalid shape: expected {expected} dims, got {got}")]
    InvalidShape { expected: usize, got: usize },
    #[error("Invalid search parameter: {0}")]
    InvalidParameter(String),
}

/// Errors that can occur in ENN operations.
#[derive(Error, Debug, Clone, PartialEq)]
pub enum ENNError {
    /// Invalid input shape.
    #[error("Invalid shape: expected {expected:?}, got {got:?}")]
    InvalidShape {
        expected: Vec<usize>,
        got: Vec<usize>,
    },
    /// Invalid parameter.
    #[error("Invalid parameter: {0}")]
    InvalidParameter(String),
    /// Index error.
    #[error("Index error: {0}")]
    IndexError(#[from] IndexError),
    /// Not enough observations.
    #[error("Not enough observations: got {0}, need at least {1}")]
    NotEnoughObservations(usize, usize),
    /// Shape error from ndarray.
    #[error("Shape error: {0}")]
    ShapeError(String),
}

impl From<ndarray::ShapeError> for ENNError {
    fn from(e: ndarray::ShapeError) -> Self {
        ENNError::ShapeError(e.to_string())
    }
}

/// Small constant to avoid division by zero.
pub const EPS_VAR: f64 = 1e-9;
