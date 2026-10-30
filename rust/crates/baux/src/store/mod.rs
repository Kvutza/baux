//! ENN training-data stores and their index lifecycle.

mod flush_controller;
mod inspection;
mod memory;
pub(crate) mod row_storage;

pub use crate::disk_bpann::DiskBpannStore;
pub(crate) use crate::disk_bpann::TrainRows;
pub(crate) use flush_controller::DiskStoreHandle;
pub use memory::MemoryEnnStore;

use deser::{Deserialize, Serialize};
use ndarray::{Array1, Array2, ArrayView2};
use std::path::PathBuf;
use std::sync::{Arc, RwLock};

use crate::error::ENNError;
use crate::index::{ENNIndex, IndexDriver};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[deser(rename_all = "snake_case")]
pub enum EnnStorage {
    InMemory,
    Disk,
}

impl EnnStorage {
    pub fn from_env() -> Self {
        if std::env::var("BAUX_WORK_DIR").is_ok() {
            Self::Disk
        } else {
            Self::InMemory
        }
    }

    pub fn work_env() -> Option<PathBuf> {
        std::env::var("BAUX_WORK_DIR").ok().map(PathBuf::from)
    }
}

pub enum EnnStore {
    InMemory(Box<MemoryEnnStore>),
    Disk(DiskStoreHandle),
}

fn disk_read<'a>(
    b: &'a Arc<RwLock<DiskBpannStore>>,
) -> Result<std::sync::RwLockReadGuard<'a, DiskBpannStore>, ENNError> {
    b.read()
        .map_err(|_| ENNError::InvalidParameter("disk store lock poisoned".to_string()))
}

fn disk_write<'a>(
    b: &'a Arc<RwLock<DiskBpannStore>>,
) -> Result<std::sync::RwLockWriteGuard<'a, DiskBpannStore>, ENNError> {
    b.write()
        .map_err(|_| ENNError::InvalidParameter("disk store lock poisoned".to_string()))
}

pub(crate) fn enn_index(store: &EnnStore) -> Result<(), ENNError> {
    match store {
        EnnStore::InMemory(_) => Ok(()),
        EnnStore::Disk(handle) => handle.persist_index(),
    }
}

impl EnnStore {
    pub fn new_memory(
        train_x: Array2<f64>,
        train_y: Array2<f64>,
        train_yvar: Option<Array2<f64>>,
        scale_x: bool,
        x_scale: Array1<f64>,
        driver: IndexDriver,
    ) -> Result<Self, ENNError> {
        Ok(Self::InMemory(Box::new(MemoryEnnStore::new(
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
        if driver != IndexDriver::BpAnnDisk {
            return Err(ENNError::InvalidParameter(
                "Disk storage requires IndexDriver::BpAnnDisk".to_string(),
            ));
        }
        let inner = DiskBpannStore::new(
            work_dir, train_x, train_y, train_yvar, scale_x, x_scale, driver,
        )?;
        Ok(Self::Disk(DiskStoreHandle::new(inner)))
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
            EnnStorage::InMemory => Ok(Self::InMemory(Box::new(MemoryEnnStore::new_empty(
                num_dim,
                num_metrics,
                driver,
            )?))),
            EnnStorage::Disk => {
                let dir = work_dir.or_else(EnnStorage::work_env).ok_or_else(|| {
                    ENNError::InvalidParameter(
                        "Disk storage requires work_dir or BAUX_WORK_DIR".to_string(),
                    )
                })?;
                if driver != IndexDriver::BpAnnDisk {
                    return Err(ENNError::InvalidParameter(
                        "Disk storage requires IndexDriver::BpAnnDisk".to_string(),
                    ));
                }
                let inner = match soft_threshold {
                    Some(threshold) => {
                        DiskBpannStore::new_threshold(dir, num_dim, num_metrics, threshold)?
                    }
                    None => DiskBpannStore::new_empty(dir, num_dim, num_metrics)?,
                };
                Ok(Self::Disk(DiskStoreHandle::new(inner)))
            }
        }
    }

    pub fn wait_flush(&self) -> Result<(), ENNError> {
        match self {
            Self::InMemory(_) => Ok(()),
            Self::Disk(handle) => handle.wait_flush(),
        }
    }

    pub fn schedule_flush(&self) -> Result<(), ENNError> {
        match self {
            Self::InMemory(_) => Ok(()),
            Self::Disk(handle) => handle.schedule_flush(),
        }
    }

    pub fn mark_stale(&self) {
        match self {
            Self::InMemory(b) => b.mark_stale(),
            Self::Disk(h) => {
                if let Ok(mut g) = disk_write(h.data()) {
                    g.mark_stale();
                }
            }
        }
    }

    pub fn index_stale(&self) -> bool {
        match self {
            Self::InMemory(b) => b.index_stale(),
            Self::Disk(h) => disk_read(h.data())
                .map(|g| g.index_stale())
                .unwrap_or(false),
        }
    }

    pub fn append_rows(
        &mut self,
        x: &ArrayView2<f64>,
        y: &ArrayView2<f64>,
        yvar: Option<&ArrayView2<f64>>,
    ) -> Result<(), ENNError> {
        match self {
            Self::InMemory(b) => b.append_rows(x, y, yvar),
            Self::Disk(h) => disk_write(h.data())?.append_rows(x, y, yvar),
        }
    }

    pub fn defers_search(&self) -> bool {
        match self {
            Self::InMemory(_) => false,
            Self::Disk(h) => disk_read(h.data())
                .map(|g| g.defers_search())
                .unwrap_or(false),
        }
    }

    pub fn ensure_sync(&self, scale_x: bool, x_scale: &Array1<f64>) -> Result<(), ENNError> {
        self.wait_flush()?;
        match self {
            Self::InMemory(b) => b.ensure_sync(scale_x, x_scale),
            Self::Disk(h) => disk_write(h.data())?.ensure_sync(scale_x, x_scale),
        }
    }

    pub(crate) fn set_metric(&self, scale: &Array1<f64>, rebuild: bool) -> Result<(), ENNError> {
        self.wait_flush()?;
        match self {
            Self::Disk(h) => disk_write(h.data())?.set_metric(scale, rebuild),
            Self::InMemory(_) => Err(ENNError::InvalidParameter(
                "AUTO metric requires disk BPANN".into(),
            )),
        }
    }

    pub fn train_rows(&self, indices: &[usize]) -> Result<TrainRows, ENNError> {
        match self {
            Self::InMemory(b) => b.train_rows(indices),
            Self::Disk(h) => disk_read(h.data())?.train_rows(indices),
        }
    }

    pub fn row_x(&self, i: usize) -> Result<Array1<f64>, ENNError> {
        match self {
            Self::InMemory(b) => b.row_x(i),
            Self::Disk(h) => disk_read(h.data())?.row_x(i),
        }
    }

    pub fn row_y(&self, i: usize) -> Result<Array1<f64>, ENNError> {
        match self {
            Self::InMemory(b) => b.row_y(i),
            Self::Disk(h) => disk_read(h.data())?.row_y(i),
        }
    }

    pub fn row_yvar(&self, i: usize) -> Result<Option<Array1<f64>>, ENNError> {
        match self {
            Self::InMemory(b) => b.row_yvar(i),
            Self::Disk(h) => disk_read(h.data())?.row_yvar(i),
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
            Self::Disk(h) => disk_read(h.data())?.search(x, search_k, exclude_nearest),
        }
    }

    pub fn index_bytes(&self) -> Result<usize, ENNError> {
        match self {
            Self::InMemory(b) => b.index_bytes(),
            Self::Disk(h) => disk_read(h.data())?.index_bytes(),
        }
    }

    pub fn index_len(&self) -> usize {
        match self {
            Self::InMemory(b) => b.index_len(),
            Self::Disk(h) => disk_read(h.data()).map(|g| g.len()).unwrap_or(0),
        }
    }

    pub fn memory_index(&self) -> Option<&ENNIndex> {
        match self {
            Self::InMemory(b) => Some(b.index()),
            Self::Disk(_) => None,
        }
    }

    pub fn x_view(&self) -> Option<ndarray::ArrayView2<'_, f64>> {
        match self {
            Self::InMemory(b) => Some(b.x_view()),
            Self::Disk(_) => None,
        }
    }

    pub fn y_view(&self) -> Option<ndarray::ArrayView2<'_, f64>> {
        match self {
            Self::InMemory(b) => Some(b.y_view()),
            Self::Disk(_) => None,
        }
    }

    pub fn memory_yvar(&self) -> Option<ndarray::ArrayView2<'_, f64>> {
        match self {
            Self::InMemory(b) => b.yvar_view(),
            Self::Disk(_) => None,
        }
    }
}

/// Drop faulted observation mmap pages from process RSS (disk store only).
pub(crate) fn enn_pages(store: &EnnStore) -> Result<(), ENNError> {
    match store {
        EnnStore::InMemory(_) => Ok(()),
        EnnStore::Disk(h) => disk_write(h.data())?.release_pages(),
    }
}

impl Drop for EnnStore {
    fn drop(&mut self) {
        if let Err(e) = enn_index(self) {
            eprintln!("baux: persist_index on drop failed: {e}");
        }
    }
}
