use ndarray::Array2;

mod inspection;
pub mod store;

pub use store::DiskBpannStore;

/// Gathered training rows: `x`, `y`, optional `yvar`.
pub(crate) type TrainRows = (Array2<f64>, Array2<f64>, Option<Array2<f64>>);
