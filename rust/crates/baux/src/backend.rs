//! ENN storage and indexing backends.

mod in_memory;
mod inspection;
pub(crate) mod row_storage;

#[cfg(all(feature = "disk-bpann", not(target_arch = "wasm32")))]
pub use crate::disk_bpann::DiskBpannEnnBackend;
pub use in_memory::InMemoryEnnBackend;
pub(crate) use row_storage::TrainRows;

use deser::{Deserialize, Serialize};
use ndarray::{Array1, Array2, ArrayView2};
use std::path::PathBuf;
use std::sync::Arc;

use crate::error::ENNError;
use crate::index::IndexDriver;

#[inline]
pub(crate) fn disk_unavailable() -> ENNError {
    #[cfg(target_arch = "wasm32")]
    {
        ENNError::InvalidParameter(
            "Disk storage is unavailable in wasm32; use InMemory storage".into(),
        )
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        ENNError::InvalidParameter(
            "Disk storage requires the disk-bpann feature; use InMemory storage".into(),
        )
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[deser(rename_all = "snake_case")]
pub enum EnnStorage {
    InMemory,
    Disk,
}

impl EnnStorage {
    #[allow(non_upper_case_globals)]
    pub const Memory: Self = Self::InMemory;
}

#[derive(Clone)]
pub enum EnnBackend {
    InMemory(Arc<InMemoryEnnBackend>),
    #[cfg(all(feature = "disk-bpann", not(target_arch = "wasm32")))]
    Disk(Arc<DiskBpannEnnBackend>),
}

pub type Storage = EnnStorage;
pub type Store = EnnBackend;
pub type MemoryStore = InMemoryEnnBackend;
#[cfg(all(feature = "disk-bpann", not(target_arch = "wasm32")))]
pub type DiskStore = DiskBpannEnnBackend;

pub(crate) fn enn_pages(backend: &EnnBackend) -> Result<(), ENNError> {
    match backend {
        EnnBackend::InMemory(_) => Ok(()),
        #[cfg(all(feature = "disk-bpann", not(target_arch = "wasm32")))]
        EnnBackend::Disk(b) => {
            let inner = Arc::try_unwrap(Arc::clone(b)).unwrap_or_else(|arc| (*arc).clone());
            let _ = inner.release_pages()?;
            Ok(())
        }
    }
}

impl EnnBackend {
    pub(crate) fn is_disk(&self) -> bool {
        #[cfg(all(feature = "disk-bpann", not(target_arch = "wasm32")))]
        {
            matches!(self, Self::Disk(_))
        }
        #[cfg(not(all(feature = "disk-bpann", not(target_arch = "wasm32"))))]
        {
            false
        }
    }

    pub fn new_memory(
        train_x: Array2<f64>,
        train_y: Array2<f64>,
        train_yvar: Option<Array2<f64>>,
        scale_x: bool,
        x_scale: Array1<f64>,
        driver: IndexDriver,
    ) -> Result<Self, ENNError> {
        Ok(Self::InMemory(Arc::new(InMemoryEnnBackend::new(
            train_x, train_y, train_yvar, scale_x, x_scale, driver,
        )?)))
    }

    pub fn new_disk(
        work_dir: PathBuf,
        train_x: Array2<f64>,
        train_y: Array2<f64>,
        train_yvar: Option<Array2<f64>>,
        scale_x: bool,
        x_scale: Array1<f64>,
        driver: IndexDriver,
    ) -> Result<Self, ENNError> {
        #[cfg(all(feature = "disk-bpann", not(target_arch = "wasm32")))]
        {
            if driver != IndexDriver::BpAnnDisk {
                return Err(ENNError::InvalidParameter(
                    "Disk storage requires IndexDriver::BpAnnDisk".to_string(),
                ));
            }
            let inner = DiskBpannEnnBackend::new(
                work_dir, train_x, train_y, train_yvar, scale_x, x_scale, driver,
            )?;
            Ok(Self::Disk(Arc::new(inner)))
        }
        #[cfg(not(all(feature = "disk-bpann", not(target_arch = "wasm32"))))]
        {
            let _ = (
                work_dir, train_x, train_y, train_yvar, scale_x, x_scale, driver,
            );
            Err(disk_unavailable())
        }
    }

    pub fn new_empty(
        num_dim: usize,
        num_metrics: usize,
        driver: IndexDriver,
        storage: EnnStorage,
        work_dir: Option<PathBuf>,
        soft_threshold: Option<usize>,
    ) -> Result<Self, ENNError> {
        match storage {
            EnnStorage::InMemory => Ok(Self::InMemory(Arc::new(InMemoryEnnBackend::new_empty(
                num_dim,
                num_metrics,
                driver,
            )?))),
            EnnStorage::Disk => {
                #[cfg(all(feature = "disk-bpann", not(target_arch = "wasm32")))]
                {
                    let dir = work_dir.ok_or_else(|| {
                        ENNError::InvalidParameter("Disk storage requires work_dir".to_string())
                    })?;
                    if driver != IndexDriver::BpAnnDisk {
                        return Err(ENNError::InvalidParameter(
                            "Disk storage requires IndexDriver::BpAnnDisk".to_string(),
                        ));
                    }
                    let inner = match soft_threshold {
                        Some(threshold) => DiskBpannEnnBackend::new_threshold(
                            dir,
                            num_dim,
                            num_metrics,
                            threshold,
                        )?,
                        None => DiskBpannEnnBackend::new_empty(dir, num_dim, num_metrics)?,
                    };
                    Ok(Self::Disk(Arc::new(inner)))
                }
                #[cfg(not(all(feature = "disk-bpann", not(target_arch = "wasm32"))))]
                {
                    let _ = (work_dir, soft_threshold, num_dim, num_metrics, driver);
                    Err(disk_unavailable())
                }
            }
        }
    }

    pub fn wait_flush(self) -> Result<Self, ENNError> {
        match self {
            Self::InMemory(_) => Ok(self),
            #[cfg(all(feature = "disk-bpann", not(target_arch = "wasm32")))]
            Self::Disk(b) => {
                let inner = Arc::try_unwrap(b).unwrap_or_else(|arc| (*arc).clone());
                let updated = inner.wait_flush()?;
                Ok(Self::Disk(Arc::new(updated)))
            }
        }
    }

    pub fn schedule_flush(self) -> Result<Self, ENNError> {
        match self {
            Self::InMemory(_) => Ok(self),
            #[cfg(all(feature = "disk-bpann", not(target_arch = "wasm32")))]
            Self::Disk(b) => {
                let inner = Arc::try_unwrap(b).unwrap_or_else(|arc| (*arc).clone());
                let updated = inner.schedule_flush()?;
                Ok(Self::Disk(Arc::new(updated)))
            }
        }
    }

    pub fn mark_stale(self) -> Self {
        match self {
            Self::InMemory(b) => {
                let inner = Arc::try_unwrap(b).unwrap_or_else(|arc| (*arc).clone());
                Self::InMemory(Arc::new(inner.mark_stale()))
            }
            #[cfg(all(feature = "disk-bpann", not(target_arch = "wasm32")))]
            Self::Disk(b) => {
                let inner = Arc::try_unwrap(b).unwrap_or_else(|arc| (*arc).clone());
                Self::Disk(Arc::new(inner.mark_stale()))
            }
        }
    }

    pub fn index_stale(&self) -> bool {
        match self {
            Self::InMemory(b) => b.index_stale(),
            #[cfg(all(feature = "disk-bpann", not(target_arch = "wasm32")))]
            Self::Disk(b) => b.index_stale(),
        }
    }

    pub fn append_rows(
        self,
        x: &ArrayView2<f64>,
        y: &ArrayView2<f64>,
        yvar: Option<&ArrayView2<f64>>,
    ) -> Result<Self, ENNError> {
        match self {
            Self::InMemory(b) => {
                let inner = Arc::try_unwrap(b).unwrap_or_else(|arc| (*arc).clone());
                let updated = inner.append_rows(x, y, yvar)?;
                Ok(Self::InMemory(Arc::new(updated)))
            }
            #[cfg(all(feature = "disk-bpann", not(target_arch = "wasm32")))]
            Self::Disk(b) => {
                let inner = Arc::try_unwrap(b).unwrap_or_else(|arc| (*arc).clone());
                let updated = inner.append_rows(x, y, yvar)?;
                Ok(Self::Disk(Arc::new(updated)))
            }
        }
    }

    pub fn defers_search(&self) -> bool {
        match self {
            Self::InMemory(_) => false,
            #[cfg(all(feature = "disk-bpann", not(target_arch = "wasm32")))]
            Self::Disk(b) => b.defers_search(),
        }
    }

    pub fn ensure_sync(self, scale_x: bool, x_scale: &Array1<f64>) -> Result<Self, ENNError> {
        let backend = self.wait_flush()?;
        match backend {
            Self::InMemory(b) => {
                let inner = Arc::try_unwrap(b).unwrap_or_else(|arc| (*arc).clone());
                let updated = inner.ensure_sync(scale_x, x_scale)?;
                Ok(Self::InMemory(Arc::new(updated)))
            }
            #[cfg(all(feature = "disk-bpann", not(target_arch = "wasm32")))]
            Self::Disk(b) => {
                let inner = Arc::try_unwrap(b).unwrap_or_else(|arc| (*arc).clone());
                let updated = inner.ensure_sync(scale_x, x_scale)?;
                Ok(Self::Disk(Arc::new(updated)))
            }
        }
    }

    pub fn with_metric(self, scale: &Array1<f64>, rebuild: bool) -> Result<Self, ENNError> {
        let backend = self.wait_flush()?;
        match backend {
            #[cfg(all(feature = "disk-bpann", not(target_arch = "wasm32")))]
            Self::Disk(b) => {
                let inner = Arc::try_unwrap(b).unwrap_or_else(|arc| (*arc).clone());
                let updated = inner.with_metric(scale, rebuild)?;
                Ok(Self::Disk(Arc::new(updated)))
            }
            Self::InMemory(_) => {
                let _ = (scale, rebuild);
                Err(ENNError::InvalidParameter(
                    "AUTO metric requires disk BPANN".into(),
                ))
            }
        }
    }

    pub fn with_thresholds(self, soft: usize, hard: usize) -> Result<Self, ENNError> {
        match self {
            Self::InMemory(_) => {
                let _ = (soft, hard);
                Ok(self)
            }
            #[cfg(all(feature = "disk-bpann", not(target_arch = "wasm32")))]
            Self::Disk(b) => {
                let inner = Arc::try_unwrap(b).unwrap_or_else(|arc| (*arc).clone());
                let updated = inner.with_thresholds(soft, hard)?;
                Ok(Self::Disk(Arc::new(updated)))
            }
        }
    }

    pub fn persist_index(self) -> Result<Self, ENNError> {
        match self {
            Self::InMemory(_) => Ok(self),
            #[cfg(all(feature = "disk-bpann", not(target_arch = "wasm32")))]
            Self::Disk(b) => {
                let inner = Arc::try_unwrap(b).unwrap_or_else(|arc| (*arc).clone());
                let updated = inner.persist_index()?;
                Ok(Self::Disk(Arc::new(updated)))
            }
        }
    }

    pub fn release_pages(self) -> Result<Self, ENNError> {
        match self {
            Self::InMemory(_) => Ok(self),
            #[cfg(all(feature = "disk-bpann", not(target_arch = "wasm32")))]
            Self::Disk(b) => {
                let inner = Arc::try_unwrap(b).unwrap_or_else(|arc| (*arc).clone());
                let updated = inner.release_pages()?;
                Ok(Self::Disk(Arc::new(updated)))
            }
        }
    }

    pub fn train_rows(&self, indices: &[usize]) -> Result<TrainRows, ENNError> {
        match self {
            Self::InMemory(b) => b.train_rows(indices),
            #[cfg(all(feature = "disk-bpann", not(target_arch = "wasm32")))]
            Self::Disk(b) => b.train_rows(indices),
        }
    }

    pub fn row_x(&self, i: usize) -> Result<Array1<f64>, ENNError> {
        match self {
            Self::InMemory(b) => b.row_x(i),
            #[cfg(all(feature = "disk-bpann", not(target_arch = "wasm32")))]
            Self::Disk(b) => b.row_x(i),
        }
    }

    pub fn row_y(&self, i: usize) -> Result<Array1<f64>, ENNError> {
        match self {
            Self::InMemory(b) => b.row_y(i),
            #[cfg(all(feature = "disk-bpann", not(target_arch = "wasm32")))]
            Self::Disk(b) => b.row_y(i),
        }
    }

    pub fn row_yvar(&self, i: usize) -> Result<Option<Array1<f64>>, ENNError> {
        match self {
            Self::InMemory(b) => b.row_yvar(i),
            #[cfg(all(feature = "disk-bpann", not(target_arch = "wasm32")))]
            Self::Disk(b) => b.row_yvar(i),
        }
    }

    pub fn search(
        &self,
        x: &ArrayView2<f64>,
        search_k: i32,
        exclude_nearest: bool,
    ) -> Result<(Array2<f64>, Array2<i64>), ENNError> {
        match self {
            Self::InMemory(b) => b.search(x, search_k, exclude_nearest),
            #[cfg(all(feature = "disk-bpann", not(target_arch = "wasm32")))]
            Self::Disk(b) => b.search(x, search_k, exclude_nearest),
        }
    }
}
