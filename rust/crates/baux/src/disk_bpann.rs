pub mod enn_backend;
mod inspection;

pub use enn_backend::DiskBpannEnnBackend;
pub type DiskStore = DiskBpannEnnBackend;

pub(crate) use crate::backend::TrainRows;
