//! Backend shape and storage inspection.

use super::*;
use crate::index::ENNIndex;

impl EnnBackend {
    pub fn len(&self) -> usize {
        match self {
            Self::InMemory(b) => b.len(),
            #[cfg(all(feature = "disk-bpann", not(target_arch = "wasm32")))]
            Self::Disk(b) => b.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn num_dim(&self) -> usize {
        match self {
            Self::InMemory(b) => b.num_dim(),
            #[cfg(all(feature = "disk-bpann", not(target_arch = "wasm32")))]
            Self::Disk(b) => b.num_dim(),
        }
    }

    pub fn num_metrics(&self) -> usize {
        match self {
            Self::InMemory(b) => b.num_metrics(),
            #[cfg(all(feature = "disk-bpann", not(target_arch = "wasm32")))]
            Self::Disk(b) => b.num_metrics(),
        }
    }

    pub fn driver(&self) -> IndexDriver {
        match self {
            Self::InMemory(b) => b.driver(),
            #[cfg(all(feature = "disk-bpann", not(target_arch = "wasm32")))]
            Self::Disk(b) => b.driver(),
        }
    }

    pub fn index_bytes(&self) -> Result<usize, ENNError> {
        match self {
            Self::InMemory(b) => b.index_bytes(),
            #[cfg(all(feature = "disk-bpann", not(target_arch = "wasm32")))]
            Self::Disk(b) => b.index_bytes(),
        }
    }

    pub fn index_len(&self) -> usize {
        match self {
            Self::InMemory(b) => b.index_len(),
            #[cfg(all(feature = "disk-bpann", not(target_arch = "wasm32")))]
            Self::Disk(b) => b.len(),
        }
    }

    pub fn memory_index(&self) -> Option<&ENNIndex> {
        match self {
            Self::InMemory(b) => Some(b.index()),
            #[cfg(all(feature = "disk-bpann", not(target_arch = "wasm32")))]
            Self::Disk(_) => None,
        }
    }

    pub fn x_view(&self) -> Option<ndarray::ArrayView2<'_, f64>> {
        match self {
            Self::InMemory(b) => b.x_view(),
            #[cfg(all(feature = "disk-bpann", not(target_arch = "wasm32")))]
            Self::Disk(_) => None,
        }
    }

    pub fn y_view(&self) -> Option<ndarray::ArrayView2<'_, f64>> {
        match self {
            Self::InMemory(b) => b.y_view(),
            #[cfg(all(feature = "disk-bpann", not(target_arch = "wasm32")))]
            Self::Disk(_) => None,
        }
    }

    pub fn memory_yvar(&self) -> Option<ndarray::ArrayView2<'_, f64>> {
        match self {
            Self::InMemory(b) => b.yvar_view(),
            #[cfg(all(feature = "disk-bpann", not(target_arch = "wasm32")))]
            Self::Disk(_) => None,
        }
    }

    pub fn memory_x(&self) -> Option<ndarray::Array2<f64>> {
        match self {
            Self::InMemory(b) => Some(b.memory_x()),
            #[cfg(all(feature = "disk-bpann", not(target_arch = "wasm32")))]
            Self::Disk(_) => None,
        }
    }

    pub fn memory_y(&self) -> Option<ndarray::Array2<f64>> {
        match self {
            Self::InMemory(b) => Some(b.memory_y()),
            #[cfg(all(feature = "disk-bpann", not(target_arch = "wasm32")))]
            Self::Disk(_) => None,
        }
    }

    pub fn memory_logs(&self) -> Option<(&ImLog, &ImLog, Option<&ImLog>)> {
        match self {
            Self::InMemory(b) => Some((b.x_log(), b.y_log(), b.yvar_log())),
            #[cfg(all(feature = "disk-bpann", not(target_arch = "wasm32")))]
            Self::Disk(_) => None,
        }
    }

    pub fn slice_x(&self, i: usize) -> Result<Option<&[f64]>, ENNError> {
        match self {
            Self::InMemory(b) => b.slice_x(i).map(Some),
            #[cfg(all(feature = "disk-bpann", not(target_arch = "wasm32")))]
            Self::Disk(_) => Ok(None),
        }
    }

    pub fn slice_y(&self, i: usize) -> Result<Option<&[f64]>, ENNError> {
        match self {
            Self::InMemory(b) => b.slice_y(i).map(Some),
            #[cfg(all(feature = "disk-bpann", not(target_arch = "wasm32")))]
            Self::Disk(_) => Ok(None),
        }
    }

    pub fn slice_yvar(&self, i: usize) -> Result<Option<Option<&[f64]>>, ENNError> {
        match self {
            Self::InMemory(b) => b.slice_yvar(i).map(Some),
            #[cfg(all(feature = "disk-bpann", not(target_arch = "wasm32")))]
            Self::Disk(_) => Ok(None),
        }
    }
}
