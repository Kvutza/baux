//! Disk backend shape and search status.

use super::enn_backend::DiskBpannEnnBackend;
use crate::index::IndexDriver;

impl DiskBpannEnnBackend {
    pub fn driver(&self) -> IndexDriver {
        self.driver
    }

    pub fn len(&self) -> usize {
        self.inner.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn num_dim(&self) -> usize {
        self.inner.num_dim()
    }

    pub fn num_metrics(&self) -> usize {
        self.num_metrics
    }

    pub fn defers_search(&self) -> bool {
        true
    }

    pub fn index_stale(&self) -> bool {
        false
    }
}
