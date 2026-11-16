//! Disk-backed ENN backend wrapping the standalone `bpann` crate.

use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use baux::neighbor_index::{BpannBackend, BpannTuning};
use ndarray::{Array1, Array2, ArrayView2};

use super::TrainRows;
use crate::error::ENNError;
use crate::index::IndexDriver;

fn bpann_err(e: baux::neighbor_index::BpannError) -> ENNError {
    match e {
        baux::neighbor_index::BpannError::InvalidShape { expected, got } => {
            ENNError::InvalidShape { expected, got }
        }
        baux::neighbor_index::BpannError::InvalidParameter(s) => ENNError::InvalidParameter(s),
    }
}

fn apply_tuning(inner: BpannBackend, tuning: BpannTuning) -> BpannBackend {
    inner
        .with_tuning(tuning)
        .with_soft(tuning.soft_threshold)
        .with_hard(tuning.hard_threshold)
}

#[derive(Clone)]
pub struct DiskBpannEnnBackend {
    pub(super) inner: BpannBackend,
    pub(super) driver: IndexDriver,
    pub(super) num_metrics: usize,
    pub(super) tuning: BpannTuning,
    pub(super) flush_scheduled: Arc<AtomicBool>,
}

impl std::fmt::Debug for DiskBpannEnnBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DiskBpannEnnBackend")
            .field("driver", &self.driver)
            .field("num_metrics", &self.num_metrics)
            .field("tuning", &self.tuning)
            .finish()
    }
}

impl DiskBpannEnnBackend {
    pub fn new(
        work_dir: PathBuf,
        train_x: Array2<f64>,
        train_y: Array2<f64>,
        train_yvar: Option<Array2<f64>>,
        scale_x: bool,
        x_scale: Array1<f64>,
        driver: IndexDriver,
    ) -> Result<Self, ENNError> {
        Self::from_tuning(
            work_dir,
            train_x,
            train_y,
            train_yvar,
            scale_x,
            x_scale,
            driver,
            BpannTuning::default(),
        )
    }

    pub fn from_tuning(
        work_dir: PathBuf,
        train_x: Array2<f64>,
        train_y: Array2<f64>,
        train_yvar: Option<Array2<f64>>,
        scale_x: bool,
        x_scale: Array1<f64>,
        driver: IndexDriver,
        tuning: BpannTuning,
    ) -> Result<Self, ENNError> {
        if driver != IndexDriver::BpAnnDisk {
            return Err(ENNError::InvalidParameter(
                "DiskBpannEnnBackend requires IndexDriver::BpAnnDisk".to_string(),
            ));
        }
        tuning.validate().map_err(ENNError::InvalidParameter)?;
        let inner = if train_x.nrows() == 0
            && train_y.nrows() == 0
            && work_dir.join("metadata.json").exists()
        {
            BpannBackend::reopen(work_dir.clone()).map_err(bpann_err)?
        } else {
            BpannBackend::new(work_dir, train_x, train_y, train_yvar, scale_x, x_scale)
                .map_err(bpann_err)?
        };
        let inner = apply_tuning(inner, tuning);
        let num_metrics = inner.num_metrics();
        Ok(Self {
            inner,
            driver,
            num_metrics,
            tuning,
            flush_scheduled: Arc::new(AtomicBool::new(false)),
        })
    }

    pub fn new_empty(
        work_dir: PathBuf,
        num_dim: usize,
        num_metrics: usize,
    ) -> Result<Self, ENNError> {
        Self::empty_tuned(work_dir, num_dim, num_metrics, BpannTuning::default())
    }

    pub fn empty_tuned(
        work_dir: PathBuf,
        num_dim: usize,
        num_metrics: usize,
        tuning: BpannTuning,
    ) -> Result<Self, ENNError> {
        tuning.validate().map_err(ENNError::InvalidParameter)?;
        let inner = apply_tuning(
            BpannBackend::new_empty(work_dir, num_dim, num_metrics).map_err(bpann_err)?,
            tuning,
        );
        Ok(Self {
            inner,
            driver: IndexDriver::BpAnnDisk,
            num_metrics,
            tuning,
            flush_scheduled: Arc::new(AtomicBool::new(false)),
        })
    }

    pub fn new_threshold(
        work_dir: PathBuf,
        num_dim: usize,
        num_metrics: usize,
        soft_threshold: usize,
    ) -> Result<Self, ENNError> {
        let default_tuning = BpannTuning::default();
        let hard = default_tuning.hard_threshold;
        Self::new_thresholds(
            work_dir,
            num_dim,
            num_metrics,
            soft_threshold,
            hard.max(soft_threshold),
        )
    }

    pub fn new_thresholds(
        work_dir: PathBuf,
        num_dim: usize,
        num_metrics: usize,
        soft_threshold: usize,
        hard_threshold: usize,
    ) -> Result<Self, ENNError> {
        let tuning = BpannTuning {
            soft_threshold,
            hard_threshold,
            ..BpannTuning::default()
        };
        Self::thresholds_tuned(
            work_dir,
            num_dim,
            num_metrics,
            soft_threshold,
            hard_threshold,
            tuning,
        )
    }

    pub fn thresholds_tuned(
        work_dir: PathBuf,
        num_dim: usize,
        num_metrics: usize,
        soft_threshold: usize,
        hard_threshold: usize,
        tuning: BpannTuning,
    ) -> Result<Self, ENNError> {
        if soft_threshold == 0 {
            return Err(ENNError::InvalidParameter(
                "soft_threshold must be >= 1".to_string(),
            ));
        }
        if hard_threshold == 0 {
            return Err(ENNError::InvalidParameter(
                "hard_threshold must be >= 1".to_string(),
            ));
        }
        if hard_threshold < soft_threshold {
            return Err(ENNError::InvalidParameter(
                "hard_threshold must be >= soft_threshold".to_string(),
            ));
        }
        let inner = BpannBackend::new_empty(work_dir, num_dim, num_metrics)
            .map_err(bpann_err)?
            .with_tuning(tuning)
            .with_soft(soft_threshold)
            .with_hard(hard_threshold)
            .defer_indexing(true);
        Ok(Self {
            inner,
            driver: IndexDriver::BpAnnDisk,
            num_metrics,
            tuning,
            flush_scheduled: Arc::new(AtomicBool::new(false)),
        })
    }

    pub fn with_tuning(mut self, tuning: BpannTuning) -> Self {
        self.inner = self.inner.with_tuning(tuning);
        self.tuning = tuning;
        self
    }

    pub fn tuning(&self) -> &BpannTuning {
        &self.tuning
    }

    pub fn mark_stale(mut self) -> Self {
        self.inner = self.inner.mark_stale();
        self.flush_scheduled.store(false, Ordering::Relaxed);
        self
    }

    pub fn wait_flush(mut self) -> Result<Self, ENNError> {
        if self.flush_scheduled.swap(false, Ordering::Relaxed) {
            self.inner = self.inner.wait_flush().map_err(bpann_err)?;
        }
        Ok(self)
    }

    pub fn schedule_flush(mut self) -> Result<Self, ENNError> {
        self.inner = self.inner.schedule_flush().map_err(bpann_err)?;
        if self.pending_count() > 0 {
            self.flush_scheduled.store(true, Ordering::Relaxed);
        }
        Ok(self)
    }

    pub fn persist_index(mut self) -> Result<Self, ENNError> {
        self.inner = self.inner.persist_index().map_err(bpann_err)?;
        self.flush_scheduled.store(false, Ordering::Relaxed);
        Ok(self)
    }

    pub fn release_pages(mut self) -> Result<Self, ENNError> {
        self.inner = self.inner.release_pages().map_err(bpann_err)?;
        Ok(self)
    }

    pub fn append_rows(
        mut self,
        x: &ArrayView2<f64>,
        y: &ArrayView2<f64>,
        yvar: Option<&ArrayView2<f64>>,
    ) -> Result<Self, ENNError> {
        self.inner = self.inner.append_rows(x, y, yvar).map_err(bpann_err)?;
        Ok(self)
    }

    pub fn ensure_sync(mut self, scale_x: bool, x_scale: &Array1<f64>) -> Result<Self, ENNError> {
        self.inner = self
            .inner
            .ensure_scale(scale_x, x_scale)
            .map_err(bpann_err)?;
        self.flush_scheduled.store(false, Ordering::Relaxed);
        Ok(self)
    }

    pub fn with_metric(mut self, scale: &Array1<f64>, rebuild: bool) -> Result<Self, ENNError> {
        self.inner = self.inner.with_metric(scale, rebuild).map_err(bpann_err)?;
        Ok(self)
    }

    pub fn with_thresholds(mut self, soft: usize, hard: usize) -> Result<Self, ENNError> {
        if soft == 0 {
            return Err(ENNError::InvalidParameter(
                "soft_threshold must be >= 1".to_string(),
            ));
        }
        if hard == 0 {
            return Err(ENNError::InvalidParameter(
                "hard_threshold must be >= 1".to_string(),
            ));
        }
        if hard < soft {
            return Err(ENNError::InvalidParameter(
                "hard_threshold must be >= soft_threshold".to_string(),
            ));
        }
        self.inner = self.inner.with_soft(soft).with_hard(hard);
        self.tuning.soft_threshold = soft;
        self.tuning.hard_threshold = hard;
        Ok(self)
    }

    pub fn train_rows(&self, indices: &[usize]) -> Result<TrainRows, ENNError> {
        self.inner.train_rows(indices).map_err(bpann_err)
    }

    pub fn row_x(&self, i: usize) -> Result<Array1<f64>, ENNError> {
        Ok(Array1::from(
            self.inner.row_slice(i).map_err(bpann_err)?.to_vec(),
        ))
    }

    pub fn slice_x(&self, i: usize) -> Result<&[f64], ENNError> {
        self.inner.row_slice(i).map_err(bpann_err)
    }

    pub fn row_y(&self, i: usize) -> Result<Array1<f64>, ENNError> {
        let (y, _) = self.inner.y_yvar(i).map_err(bpann_err)?;
        Ok(Array1::from(y.to_vec()))
    }

    pub fn row_yvar(&self, i: usize) -> Result<Option<Array1<f64>>, ENNError> {
        let (_, yvar) = self.inner.y_yvar(i).map_err(bpann_err)?;
        Ok(yvar.map(|row| Array1::from(row.to_vec())))
    }

    pub fn search(
        &self,
        x: &ArrayView2<f64>,
        search_k: i32,
        exclude_nearest: bool,
    ) -> Result<(Array2<f64>, Array2<i64>), ENNError> {
        self.inner
            .search(x, search_k as usize, exclude_nearest)
            .map_err(bpann_err)
    }

    pub fn index_bytes(&self) -> Result<usize, ENNError> {
        Ok(self.inner.index_bytes())
    }

    pub fn pending_count(&self) -> usize {
        self.inner.pending_rows()
    }

    pub fn soft_threshold(&self) -> usize {
        self.tuning.soft_threshold
    }

    pub fn hard_threshold(&self) -> usize {
        self.tuning.hard_threshold
    }

    pub fn append_threshold(&self) -> bool {
        !self.inner.index_deferred()
    }

    pub(crate) fn defer_flush(&self) -> bool {
        self.inner.index_deferred()
    }

    /// Build soft-sync fragments without mutating the published index.
    pub(crate) fn build_detached(
        &self,
    ) -> Result<Option<baux::neighbor_index::IncrementalIndex>, ENNError> {
        baux::neighbor_index::soft_build(&self.inner).map_err(bpann_err)
    }

    /// Publish a detached soft-sync result.
    pub fn publish_detached(
        mut self,
        built: baux::neighbor_index::IncrementalIndex,
    ) -> Result<Self, ENNError> {
        baux::neighbor_index::soft_publish(&mut self.inner, built);
        self.flush_scheduled.store(false, Ordering::Relaxed);
        Ok(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;
    use tempfile::TempDir;

    #[test]
    fn new_sync() {
        let dir = TempDir::new().expect("tempdir");
        let backend =
            DiskBpannEnnBackend::new_threshold(dir.path().to_path_buf(), 2, 1, 5).expect("backend");
        assert_eq!(backend.soft_threshold(), 5);
        assert!(!backend.append_threshold());
        assert_eq!(backend.pending_count(), 0);
    }

    #[test]
    fn bpann_driver() {
        let dir = TempDir::new().expect("tempdir");
        let result = DiskBpannEnnBackend::new(
            dir.path().to_path_buf(),
            array![[0.0, 0.0]],
            array![[0.0]],
            None,
            false,
            array![1.0, 1.0],
            IndexDriver::Exact,
        );
        assert!(matches!(result, Err(ENNError::InvalidParameter(_))));
    }

    #[test]
    fn append_maps() {
        let dir = TempDir::new().expect("tempdir");
        let backend =
            DiskBpannEnnBackend::new_empty(dir.path().to_path_buf(), 2, 1).expect("backend");
        let err = backend
            .append_rows(&array![[0.0, 0.0, 0.0]].view(), &array![[0.0]].view(), None)
            .unwrap_err();
        assert!(matches!(err, ENNError::InvalidShape { .. }));
    }

    #[test]
    fn reopen_dir() {
        let dir = TempDir::new().expect("tempdir");
        let path = dir.path().to_path_buf();
        let _fresh = DiskBpannEnnBackend::new(
            path.clone(),
            array![[0.0, 0.0], [1.0, 0.0]],
            array![[0.0], [1.0]],
            None,
            false,
            array![1.0, 1.0],
            IndexDriver::BpAnnDisk,
        )
        .expect("fresh");
        let reopened = DiskBpannEnnBackend::new(
            path,
            Array2::zeros((0, 2)),
            Array2::zeros((0, 1)),
            None,
            false,
            array![1.0, 1.0],
            IndexDriver::BpAnnDisk,
        )
        .expect("reopen");
        assert_eq!(reopened.len(), 2);
    }

    #[test]
    fn y_train() {
        let dir = TempDir::new().expect("tempdir");
        let path = dir.path().to_path_buf();
        let _fresh = DiskBpannEnnBackend::new(
            path.clone(),
            array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
            array![[0.0, 1.0], [1.0, 2.0], [1.0, 3.0]],
            None,
            false,
            array![1.0, 1.0],
            IndexDriver::BpAnnDisk,
        )
        .expect("fresh");
        let reopened = DiskBpannEnnBackend::new(
            path,
            Array2::zeros((0, 2)),
            Array2::zeros((0, 1)),
            None,
            false,
            array![1.0, 1.0],
            IndexDriver::BpAnnDisk,
        )
        .expect("reopen");
        assert_eq!(reopened.num_metrics(), 2);
    }

    #[test]
    fn pending_sync() {
        let dir = TempDir::new().expect("tempdir");
        let backend = DiskBpannEnnBackend::new_threshold(dir.path().to_path_buf(), 2, 1, 100)
            .expect("backend");
        let backend = backend
            .append_rows(&array![[0.0, 0.0]].view(), &array![[1.0]].view(), None)
            .expect("append");
        assert!(backend.pending_count() > 0);
    }

    #[test]
    fn soft_pages2() {
        let dir = TempDir::new().expect("tempdir");
        let backend =
            DiskBpannEnnBackend::new_threshold(dir.path().to_path_buf(), 2, 1, 5).expect("backend");
        assert!(backend.defer_flush());
        let backend = backend
            .append_rows(
                &array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]].view(),
                &array![[0.0], [1.0], [2.0]].view(),
                None,
            )
            .expect("append");
        assert!(backend.pending_count() > 0);
        let backend = backend
            .schedule_flush()
            .expect("schedule")
            .wait_flush()
            .expect("wait");
        assert_eq!(backend.pending_count(), 0);
        assert!(!dir.path().join("index/pages.bin").exists());
        let mapped = bpann_err(baux::neighbor_index::BpannError::InvalidParameter(
            "x".into(),
        ));
        assert!(matches!(mapped, ENNError::InvalidParameter(_)));
        let tuned =
            DiskBpannEnnBackend::empty_tuned(dir.path().join("t"), 2, 1, BpannTuning::default())
                .expect("tuned");
        assert!(tuned.soft_threshold() >= 1);
    }

    #[test]
    fn isolated_tempdir() {
        let dir = TempDir::new().expect("tempdir");
        let work1 = dir.path().join("w1");
        let work2 = dir.path().join("w2");
        let work3 = dir.path().join("w3");

        let b1 = DiskBpannEnnBackend::new(
            work1.clone(),
            array![[0.0, 0.0], [1.0, 0.0]],
            array![[0.0], [1.0]],
            None,
            false,
            array![1.0, 1.0],
            IndexDriver::BpAnnDisk,
        )
        .expect("new in tempdir");
        assert_eq!(b1.len(), 2);
        assert!(work1.join("metadata.json").exists());

        let b2 = DiskBpannEnnBackend::new_empty(work2.clone(), 2, 1).expect("new_empty in tempdir");
        assert_eq!(b2.len(), 0);
        assert!(work2.join("metadata.json").exists());

        let b3 = DiskBpannEnnBackend::new_thresholds(work3.clone(), 2, 1, 10, 20)
            .expect("new_thresholds in tempdir");
        assert_eq!(b3.soft_threshold(), 10);
        assert_eq!(b3.hard_threshold(), 20);
        assert!(work3.join("metadata.json").exists());
    }

    #[test]
    fn clone_transitions() {
        let dir = TempDir::new().expect("tempdir");
        let b = DiskBpannEnnBackend::new_empty(dir.path().to_path_buf(), 2, 1).expect("empty");
        assert!(b.is_empty());
        let b_clone = b.clone();
        assert_eq!(b_clone.len(), 0);
        let b = b.with_thresholds(5, 15).expect("thresholds");
        assert_eq!(b.soft_threshold(), 5);
        assert_eq!(b.hard_threshold(), 15);
        let b = b.mark_stale();
        assert!(!b.index_stale());
        let b = b.release_pages().expect("release");
        let _b = b.persist_index().expect("persist");
    }
}
