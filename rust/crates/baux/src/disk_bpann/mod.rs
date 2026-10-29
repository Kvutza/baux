use ndarray::Array2;

pub mod enn_backend;
mod inspection;

pub use enn_backend::DiskBpannEnnBackend;

/// Gathered training rows: `x`, `y`, optional `yvar`.
pub(crate) type TrainRows = (Array2<f64>, Array2<f64>, Option<Array2<f64>>);
