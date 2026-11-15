//! Pure soft-sync flush controller utilities for disk BPANN backends.
//!
//! BPANN flush control operates with pure value semantics via consumed-`self`
//! state transitions on `DiskBpannEnnBackend` / `EnnBackend`, without
//! background worker threads, locks, or global static state.

/// Pure helper to check if pending rows have reached the soft flush threshold.
#[inline]
pub fn should_flush(pending: usize, soft_threshold: usize) -> bool {
    pending >= soft_threshold
}

/// Pure helper to check if pending rows have reached the hard flush cap.
#[inline]
pub fn should_sync(pending: usize, hard_threshold: usize) -> bool {
    pending >= hard_threshold
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::{EnnBackend, EnnStorage};
    use crate::disk_bpann::DiskBpannEnnBackend;
    use crate::index::IndexDriver;
    use ndarray::{Array1, Array2};
    use tempfile::TempDir;

    fn disk_model(dir: &TempDir, n: usize, dim: usize) -> EnnBackend {
        let backend = EnnBackend::new_empty(
            dim,
            1,
            IndexDriver::BpAnnDisk,
            EnnStorage::Disk,
            Some(dir.path().to_path_buf()),
            Some(1),
        )
        .expect("new_empty");
        let x = Array2::from_shape_fn((n, dim), |(i, j)| (i + j) as f64);
        let y = Array2::from_shape_fn((n, 1), |(i, _)| i as f64);
        backend
            .append_rows(&x.view(), &y.view(), None)
            .expect("add")
    }

    fn search_neighbors(
        backend: &EnnBackend,
        query: &ndarray::ArrayView2<f64>,
        k: usize,
    ) -> Array2<i64> {
        let scale = Array1::ones(query.ncols());
        let synced = if !backend.defers_search() {
            backend
                .clone()
                .ensure_sync(false, &scale)
                .expect("ensure_sync")
        } else {
            backend.clone()
        };
        synced.search(query, k as i32, false).expect("search").1
    }

    #[test]
    fn schedule_flight() {
        let dir = TempDir::new().expect("tempdir");
        let model = disk_model(&dir, 8, 4);
        let model = model.schedule_flush().expect("schedule");
        let model = model.wait_flush().expect("wait");
        assert!(!dir.path().join("index/pages.bin").exists());
        let _ = model;
    }

    #[test]
    fn search_deadlock() {
        let dir = TempDir::new().expect("tempdir");
        let model = disk_model(&dir, 8, 4);
        let query = Array2::from_shape_fn((1, 4), |(_, j)| j as f64);
        let model = model.schedule_flush().expect("schedule");
        let neighbors = search_neighbors(&model, &query.view(), 1);
        assert_eq!(neighbors.nrows(), 1);
        assert_eq!(neighbors.ncols(), 1);
        let _model = model.wait_flush().expect("wait");
    }

    #[test]
    fn wait_noop() {
        let dir = TempDir::new().expect("tempdir");
        let backend = DiskBpannEnnBackend::new_threshold(dir.path().to_path_buf(), 2, 1, 100)
            .expect("backend");
        let backend = backend.wait_flush().expect("idle wait");
        let backend = backend.schedule_flush().expect("below threshold");
        let _backend = backend.wait_flush().expect("still idle");
    }

    #[test]
    fn schedule_noop() {
        let dir = TempDir::new().expect("tempdir");
        let model = disk_model(&dir, 4, 2);
        let model = model.schedule_flush().expect("first schedule");
        let model = model.schedule_flush().expect("second schedule");
        let _model = model.wait_flush().expect("wait");
    }

    #[test]
    fn search_flight() {
        let dir = TempDir::new().expect("tempdir");
        let model = disk_model(&dir, 6, 3);
        let query = Array2::from_shape_fn((1, 3), |(i, j)| ((5 + i) + j) as f64);
        let model = model.schedule_flush().expect("schedule");
        let neighbors = search_neighbors(&model, &query.view(), 1);
        assert_eq!(neighbors[[0, 0]], 5);
        let _model = model.wait_flush().expect("wait");
    }

    #[test]
    fn soft_metamorphic() {
        let dir_a = TempDir::new().expect("tempdir a");
        let dir_b = TempDir::new().expect("tempdir b");
        let a =
            DiskBpannEnnBackend::new_threshold(dir_a.path().to_path_buf(), 3, 1, 100).expect("a");
        let b =
            DiskBpannEnnBackend::new_threshold(dir_b.path().to_path_buf(), 3, 1, 100).expect("b");
        let x = Array2::from_shape_fn((12, 3), |(i, j)| (i * 3 + j) as f64);
        let y = Array2::from_shape_fn((12, 1), |(i, _)| i as f64);
        let a = a.append_rows(&x.view(), &y.view(), None).expect("append a");
        let b = b.append_rows(&x.view(), &y.view(), None).expect("append b");

        let a = a
            .schedule_flush()
            .expect("schedule a")
            .wait_flush()
            .expect("wait a");
        let b = b
            .ensure_sync(false, &ndarray::Array1::ones(3))
            .expect("ensure b");

        let q = Array2::from_shape_fn((4, 3), |(i, j)| ((i + 2) + j) as f64);
        let (da, ia) = a.search(&q.view(), 3, false).expect("search a");
        let (db, ib) = b.search(&q.view(), 3, false).expect("search b");
        assert_eq!(ia, ib);
        assert_eq!(da.shape(), db.shape());
        for (xa, xb) in da.iter().zip(db.iter()) {
            assert!((xa - xb).abs() < 1e-9, "dist mismatch {xa} vs {xb}");
        }
    }

    #[test]
    fn fuzz_seeds4() {
        use rand::{Rng, SeedableRng};
        use rand_chacha::ChaCha8Rng;
        let seed = std::env::var("ENN_FUZZ_SEED")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(0xC0FFEE_u64);
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        for _trial in 0..8 {
            let dir = TempDir::new().expect("tempdir");
            let n = rng.gen_range(4..=16);
            let dim = rng.gen_range(2..=5);
            let model = disk_model(&dir, n, dim);
            let qrow = rng.gen_range(0..n);
            let query = Array2::from_shape_fn((1, dim), |(_, j)| (qrow + j) as f64);
            let model = model.schedule_flush().expect("schedule");
            let neighbors = search_neighbors(&model, &query.view(), 1);
            assert_eq!(neighbors.ncols(), 1);
            assert!(neighbors[[0, 0]] < n as i64);
            let _model = model.wait_flush().expect("wait");
        }
    }

    #[test]
    fn persist_pages() {
        let dir = TempDir::new().expect("tempdir");
        let model = disk_model(&dir, 6, 2);
        let model = model.schedule_flush().expect("schedule");
        let model = model.wait_flush().expect("wait soft");
        let model = model.persist_index().expect("hard persist");
        assert!(dir.path().join("index/pages.bin").exists());
        let _ = model;
    }

    #[test]
    fn flush_wait() {
        let dir = TempDir::new().expect("tempdir");
        let backend =
            DiskBpannEnnBackend::new_threshold(dir.path().to_path_buf(), 2, 1, 1).expect("backend");
        let x = Array2::from_shape_fn((2, 2), |(i, j)| (i + j) as f64);
        let y = Array2::from_shape_fn((2, 1), |(i, _)| i as f64);
        let backend = backend
            .append_rows(&x.view(), &y.view(), None)
            .expect("append");
        let backend = backend.schedule_flush().expect("schedule ok");
        let backend = backend.wait_flush().expect("drain");
        let _backend = backend.persist_index().expect("persist");
    }

    #[test]
    fn schedule_async() {
        let dir = TempDir::new().expect("tempdir");
        let backend = DiskBpannEnnBackend::new_thresholds(dir.path().to_path_buf(), 2, 1, 2, 5)
            .expect("backend");
        let x = Array2::from_shape_fn((3, 2), |(i, j)| (i + j) as f64);
        let y = Array2::from_shape_fn((3, 1), |(i, _)| i as f64);
        let backend = backend
            .append_rows(&x.view(), &y.view(), None)
            .expect("append");
        assert_eq!(backend.pending_count(), 3);
        let backend = backend.schedule_flush().expect("schedule");
        let backend = backend.wait_flush().expect("wait");
        assert_eq!(backend.pending_count(), 0);
    }

    #[test]
    fn schedule_synchronously() {
        let dir = TempDir::new().expect("tempdir");
        let backend = DiskBpannEnnBackend::new_thresholds(dir.path().to_path_buf(), 2, 1, 2, 100)
            .expect("backend");
        let x = Array2::from_shape_fn((6, 2), |(i, j)| (i + j) as f64);
        let y = Array2::from_shape_fn((6, 1), |(i, _)| i as f64);
        let backend = backend
            .append_rows(&x.view(), &y.view(), None)
            .expect("append");
        assert_eq!(backend.pending_count(), 6);
        let backend = backend.with_thresholds(2, 5).expect("set thresholds");
        let backend = backend.schedule_flush().expect("hard schedule");
        assert_eq!(backend.pending_count(), 0);
        assert!(!dir.path().join("index/pages.bin").exists());
        let _ = backend;
    }

    #[test]
    fn append_add2() {
        let dir = TempDir::new().expect("tempdir");
        let backend = DiskBpannEnnBackend::new_thresholds(dir.path().to_path_buf(), 2, 1, 2, 5)
            .expect("backend");
        let x = Array2::from_shape_fn((5, 2), |(i, j)| (i + j) as f64);
        let y = Array2::from_shape_fn((5, 1), |(i, _)| i as f64);
        let backend = backend
            .append_rows(&x.view(), &y.view(), None)
            .expect("append");
        assert_eq!(backend.pending_count(), 0);
        let _ = backend;
    }
}
