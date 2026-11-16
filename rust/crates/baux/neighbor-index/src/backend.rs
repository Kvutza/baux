use std::fs;
use std::path::PathBuf;
use std::sync::Arc;

use ndarray::{Array1, Array2, ArrayView2};

use baux::neighbor_index::error::BpannError;
use baux::neighbor_index::index::{BpannIndex, IncrementalIndex};
use baux::neighbor_index::lgnsrch::{SearchPendingArgs, search_pending};
use baux::neighbor_index::mmap_store::MmapColumnStore;
use baux::neighbor_index::observation::{
    self as obs, INDEX_BACKEND, MAX_DIM, MAX_STRIDE, TrainRows,
};
use baux::neighbor_index::smnsrch::{N_LIMIT, ScoreQueriesFlat, score_queries};
pub use baux::neighbor_index::tuning::{BpannTuning, PENDING_HARD, PENDING_SOFT};

pub const PAPER_URL: &str = "https://arxiv.org/abs/2511.15557";

#[path = "backend/access.rs"]
mod access;
#[path = "backend/metric.rs"]
mod metric;

#[derive(Clone)]
pub struct BpannBackend {
    pub(crate) work_dir: PathBuf,
    pub(crate) train_x: MmapColumnStore,
    pub(crate) train_y: MmapColumnStore,
    pub(crate) train_yvar: Option<MmapColumnStore>,
    pub(crate) num_dim: usize,
    pub(crate) num_metrics: usize,
    pub(crate) scale_x: bool,
    pub(crate) x_scale: Array1<f64>,
    pub(crate) index: IncrementalIndex,
    pub tuning: BpannTuning,
    pub(crate) index_deferred: bool,
    pub(crate) pending_unindexed: usize,
    pub(crate) index_dirty: bool,
    pub(crate) num_obs_counter: obs::NumObsCounter,
    /// Resident flat `N·D` f32 train cache for the small-N search path.
    /// Invalidated on append. Arc so parallel queries share one buffer.
    pub(crate) small_n_x_cache: Option<(usize, Arc<[f32]>)>,
}

impl std::fmt::Debug for BpannBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BpannBackend")
            .field("work_dir", &self.work_dir)
            .field("num_dim", &self.num_dim)
            .field("num_metrics", &self.num_metrics)
            .field("scale_x", &self.scale_x)
            .field("indexed_rows", &self.index.indexed_rows)
            .field("pending_unindexed", &self.pending_unindexed)
            .finish()
    }
}

impl BpannBackend {
    pub fn new(
        work_dir: PathBuf,
        train_x: Array2<f64>,
        train_y: Array2<f64>,
        train_yvar: Option<Array2<f64>>,
        scale_x: bool,
        x_scale: Array1<f64>,
    ) -> Result<Self, BpannError> {
        obs::check_dims(train_x.ncols())?;
        fs::create_dir_all(&work_dir).map_err(|e| BpannError::InvalidParameter(e.to_string()))?;
        obs::check_backend(&work_dir, INDEX_BACKEND)?;
        let num_obs_counter = obs::NumObsCounter::open(&work_dir)?;

        let num_dim = train_x.ncols();
        let num_metrics = train_y.ncols();
        let known_nrows = obs::bpann_obs(&work_dir);
        let mut train_x_store =
            MmapColumnStore::open_mmap(work_dir.join("train_x.bin"), num_dim, known_nrows)?;
        let mut train_y_store =
            MmapColumnStore::open_mmap(work_dir.join("train_y.bin"), num_metrics, known_nrows)?;
        if train_x_store.nrows == 0 && train_x.nrows() > 0 {
            train_x_store = train_x_store.mmap_append(&train_x.view())?;
            train_y_store = train_y_store.mmap_append(&train_y.view())?;
        }
        if train_x_store.nrows == 0 {
            // Pre-grow and pre-touch fresh stores so the first append pays no
            // file-resize, page-fault, or block-allocation cost.
            train_x_store = train_x_store
                .ensure_capacity(baux::neighbor_index::mmap_store::GROW_ROWS)?
                .pretouch();
            train_y_store = train_y_store
                .ensure_capacity(baux::neighbor_index::mmap_store::GROW_ROWS)?
                .pretouch();
        }
        let train_yvar_store = obs::open_yvar(&work_dir, num_metrics, train_yvar.as_ref())?;

        let n = train_x_store.nrows;
        let index_dir = work_dir.join("index");
        let indexed_rows = if metric::metric_matches(&work_dir, scale_x, &x_scale) {
            obs::load_rows(&work_dir).unwrap_or(0).min(n)
        } else {
            0
        };
        let indices = if index_dir.join("header.json").exists() && indexed_rows > 0 {
            vec![BpannIndex::open(index_dir.clone())?]
        } else {
            Vec::new()
        };
        let persisted_rows = indices.first().map(|i| i.header.indexed_rows).unwrap_or(0);
        let mut index = IncrementalIndex::new(index_dir);
        index.indices = indices;
        index.indexed_rows = persisted_rows.min(indexed_rows);

        let small_n_x_cache = if n <= N_LIMIT && n > 0 {
            let mut flat = Vec::with_capacity(n * num_dim);
            let mut buf = Vec::with_capacity(num_dim);
            let x_scale_slice = x_scale.as_slice().unwrap();
            for i in 0..n {
                let row = train_x_store.row_slice(i)?;
                baux::neighbor_index::distance::bpann_f32(row, scale_x, x_scale_slice, &mut buf);
                flat.extend_from_slice(&buf);
            }
            Some((n, Arc::from(flat.into_boxed_slice())))
        } else {
            None
        };

        let mut backend = Self {
            work_dir,
            train_x: train_x_store,
            train_y: train_y_store,
            train_yvar: train_yvar_store,
            num_dim,
            num_metrics,
            scale_x,
            x_scale,
            index,
            tuning: BpannTuning::default(),
            index_deferred: true,
            pending_unindexed: n.saturating_sub(indexed_rows),
            index_dirty: indexed_rows < n,
            num_obs_counter,
            small_n_x_cache,
        };
        if persisted_rows < indexed_rows {
            let index = backend.index;
            backend.index = index.ensure_backend(
                &backend.train_x,
                backend.num_dim,
                backend.scale_x,
                backend.x_scale.as_slice().unwrap(),
                &backend.work_dir,
                backend.num_metrics,
                indexed_rows,
                &backend.tuning,
            )?;
        }
        backend.pending_unindexed = n.saturating_sub(indexed_rows);
        backend.index.indexed_rows = indexed_rows;
        obs::write_metadata(
            &backend.work_dir,
            n,
            num_dim,
            num_metrics,
            scale_x,
            indexed_rows,
        )?;
        backend.num_obs_counter.set(n);
        Ok(backend)
    }

    pub fn new_empty(
        work_dir: PathBuf,
        num_dim: usize,
        num_metrics: usize,
    ) -> Result<Self, BpannError> {
        Self::new(
            work_dir,
            Array2::zeros((0, num_dim)),
            Array2::zeros((0, num_metrics)),
            None,
            false,
            Array1::ones(num_dim),
        )
    }

    pub fn defer_indexing(mut self, defer: bool) -> Self {
        self.index_deferred = defer;
        self
    }

    pub fn index_deferred(&self) -> bool {
        self.index_deferred
    }

    pub fn pending_rows(&self) -> usize {
        self.pending_unindexed
    }

    #[allow(clippy::len_without_is_empty)]
    pub fn len(&self) -> usize {
        self.train_x.nrows
    }

    pub fn num_dim(&self) -> usize {
        self.num_dim
    }

    pub fn num_metrics(&self) -> usize {
        self.num_metrics
    }

    pub fn mark_stale(self) -> Self {
        self.reset_index()
    }

    pub fn ensure_scale(
        mut self,
        scale_x: bool,
        x_scale: &Array1<f64>,
    ) -> Result<Self, BpannError> {
        if self.scale_x != scale_x || self.x_scale != *x_scale {
            metric::metric_pending(&self.work_dir)?;
            self.scale_x = scale_x;
            self.x_scale = x_scale.to_owned();
            self = self.reset_index();
            self.small_n_x_cache = None;
        }
        self.ensure_sync()
    }

    fn reset_index(mut self) -> Self {
        self.pending_unindexed = self.len();
        self.index = self.index.reset();
        self.index_dirty = true;
        self.small_n_x_cache = None;
        self
    }

    pub fn indexed_rows(&self) -> usize {
        self.index.indexed_rows
    }

    pub fn append_row(
        self,
        x: &Array1<f64>,
        y: &Array1<f64>,
        yvar: Option<&Array1<f64>>,
    ) -> Result<Self, BpannError> {
        let x2 = x.clone().insert_axis(ndarray::Axis(0));
        let y2 = y.clone().insert_axis(ndarray::Axis(0));
        let yv2 = yvar.map(|v| v.clone().insert_axis(ndarray::Axis(0)));
        self.append_rows(
            &x2.view(),
            &y2.view(),
            yv2.as_ref().map(|a| a.view()).as_ref(),
        )
    }

    pub fn append_rows(
        mut self,
        x: &ArrayView2<f64>,
        y: &ArrayView2<f64>,
        yvar: Option<&ArrayView2<f64>>,
    ) -> Result<Self, BpannError> {
        if x.nrows() == 0 {
            return Ok(self);
        }
        if x.ncols() != self.num_dim || y.ncols() != self.num_metrics || x.nrows() != y.nrows() {
            return Err(BpannError::InvalidShape {
                expected: vec![x.nrows(), self.num_dim],
                got: vec![x.nrows(), x.ncols()],
            });
        }
        obs::check_rows(self.len() + x.nrows())?;
        self.train_x = self.train_x.append(x)?;
        self.train_y = self.train_y.append(y)?;
        obs::append_yvar(&self.work_dir, self.num_metrics, &mut self.train_yvar, yvar)?;
        self.index = self
            .index
            .note_rows(x, self.scale_x, self.x_scale.as_slice().unwrap());
        self.pending_unindexed += x.nrows();
        self.index_dirty = true;
        self.num_obs_counter.set(self.len());

        if self.len() <= N_LIMIT {
            let n = self.len();
            let mut flat = Vec::with_capacity(n * self.num_dim);
            let mut buf = Vec::with_capacity(self.num_dim);
            let x_scale = self.x_scale.as_slice().unwrap();
            for i in 0..n {
                let row = self.train_x.row_slice(i)?;
                baux::neighbor_index::distance::bpann_f32(row, self.scale_x, x_scale, &mut buf);
                flat.extend_from_slice(&buf);
            }
            self.small_n_x_cache = Some((n, Arc::from(flat.into_boxed_slice())));
        } else {
            self.small_n_x_cache = None;
        }

        let pending = self.pending_rows();
        // Hard cap: soft-sync on the caller when pending reaches the hard
        // threshold (deferred or not). Soft threshold syncs only on the
        // non-deferred path.
        if pending >= self.tuning.hard_threshold
            || (!self.index_deferred && pending >= self.tuning.soft_threshold)
        {
            self = self.ensure_sync()?;
        }
        Ok(self)
    }

    /// Soft sync: build/compact fragments in memory and update pending
    /// counters. May write `indexed_rows.bin`. Does not write `pages.bin` /
    /// `skip_edges.bin` and does not clear `index_dirty` (hard persist
    /// alone clears disk-dirty).
    ///
    /// Mutates the live index in place. Background flush still uses
    /// [`soft_build`] / [`soft_publish`] so readers can search the
    /// previous snapshot while a detached build runs.
    pub fn ensure_sync(mut self) -> Result<Self, BpannError> {
        let end = self.len();
        if self.index.indexed_rows >= end {
            self.pending_unindexed = 0;
            return Ok(self);
        }
        let index = self.index;
        self.index = index.ensure_backend(
            &self.train_x,
            self.num_dim,
            self.scale_x,
            self.x_scale.as_slice().unwrap(),
            &self.work_dir,
            self.num_metrics,
            end,
            &self.tuning,
        )?;
        self.pending_unindexed = 0;
        // Soft-sync and centroid builds fault train pages; drop them from RSS.
        self.release_pages()
    }

    /// Remap observation mmaps so faulted/dirty pages leave process RSS.
    pub fn release_pages(mut self) -> Result<Self, BpannError> {
        self.train_x = self.train_x.release_pages()?;
        self.train_y = self.train_y.release_pages()?;
        if let Some(store) = self.train_yvar {
            self.train_yvar = Some(store.release_pages()?);
        }
        Ok(self)
    }

    pub fn persist_index(mut self) -> Result<Self, BpannError> {
        self.train_x = self.train_x.flush_disk()?;
        self.train_y = self.train_y.flush_disk()?;
        if let Some(yv) = self.train_yvar {
            self.train_yvar = Some(yv.flush_disk()?);
        }
        obs::write_metadata(
            &self.work_dir,
            self.len(),
            self.num_dim,
            self.num_metrics,
            self.scale_x,
            self.index.indexed_rows,
        )?;
        self.num_obs_counter.set(self.len());

        if !self
            .index
            .needs_rewrite(self.index_dirty, self.train_x.nrows)
        {
            self.pending_unindexed = 0;
            self.index_dirty = false;
            return Ok(self);
        }
        metric::metric_pending(&self.work_dir)?;
        let index = self.index;
        self.index = index.persist_backend(
            &self.train_x,
            self.num_dim,
            self.scale_x,
            self.x_scale.as_slice().unwrap(),
            &self.work_dir,
            self.num_metrics,
            &self.tuning,
        )?;
        metric::persist_metric(&self.work_dir, self.scale_x, &self.x_scale)?;
        obs::write_metadata(
            &self.work_dir,
            self.len(),
            self.num_dim,
            self.num_metrics,
            self.scale_x,
            self.index.indexed_rows,
        )?;
        self.pending_unindexed = 0;
        self.index_dirty = false;
        Ok(self)
    }

    pub fn schedule_flush(self) -> Result<Self, BpannError> {
        if self.pending_rows() >= self.soft_threshold() {
            self.ensure_sync()
        } else {
            Ok(self)
        }
    }

    pub fn wait_flush(self) -> Result<Self, BpannError> {
        self.ensure_sync()
    }

    pub fn train_rows(&self, indices: &[usize]) -> Result<TrainRows, BpannError> {
        obs::train_rows(
            self.len(),
            &self.train_x,
            &self.train_y,
            self.train_yvar.as_ref(),
            indices,
        )
    }

    pub fn search(
        &self,
        queries: &ArrayView2<f64>,
        search_k: usize,
        exclude_nearest: bool,
    ) -> Result<(Array2<f64>, Array2<i64>), BpannError> {
        let total = self.len();
        let n_query = queries.nrows();
        if total == 0 {
            return Ok((Array2::zeros((n_query, 0)), Array2::zeros((n_query, 0))));
        }
        let k_eff = search_k.min(total);
        let pool_k = if exclude_nearest {
            (search_k + 1).min(total)
        } else {
            k_eff
        };
        let mut dist2s = Array2::zeros((n_query, k_eff));
        let mut indices = Array2::zeros((n_query, k_eff));
        let scale_x = self.scale_x;
        let x_scale_vec = self.x_scale.as_slice().unwrap().to_vec();
        let num_dim = self.num_dim;
        let query_rows: Vec<Vec<f64>> = (0..n_query).map(|q| queries.row(q).to_vec()).collect();

        // Small-N: resident flat f32 cache + heap top-k (shared across
        // queries).
        if total <= N_LIMIT {
            let flat = baux::neighbor_index::smnsrch::resident_cache(self, total)?;
            let per_query = score_queries(
                &query_rows,
                &ScoreQueriesFlat {
                    flat: &flat,
                    total,
                    num_dim,
                    scale_x,
                    x_scale: &x_scale_vec,
                    k_eff,
                    pool_k,
                    exclude_nearest,
                },
            );
            for (q, (dist_row, idx_row)) in per_query.into_iter().enumerate() {
                for j in 0..k_eff {
                    dist2s[[q, j]] = dist_row[j];
                    indices[[q, j]] = idx_row[j];
                }
            }
            return Ok((dist2s, indices));
        }

        search_pending(
            self,
            &query_rows,
            &mut dist2s,
            &mut indices,
            SearchPendingArgs {
                total,
                k_eff,
                pool_k,
                exclude_nearest,
                scale_x,
                x_scale: &x_scale_vec,
                num_dim,
            },
        )?;
        Ok((dist2s, indices))
    }
}

/// Build a soft-sync result under a shared borrow (no publish).
/// Readers may search the live index concurrently while this runs.
pub fn soft_build(backend: &BpannBackend) -> Result<Option<IncrementalIndex>, BpannError> {
    let end = backend.len();
    if backend.index.indexed_rows >= end {
        return Ok(None);
    }
    let working = backend.index.clone();
    let working = working.ensure_backend(
        &backend.train_x,
        backend.num_dim,
        backend.scale_x,
        backend.x_scale.as_slice().unwrap(),
        &backend.work_dir,
        backend.num_metrics,
        end,
        &backend.tuning,
    )?;
    Ok(Some(working))
}

/// Publish a detached soft-sync result under exclusive borrow.
pub fn soft_publish(backend: &mut BpannBackend, built: IncrementalIndex) {
    backend.index = built;
    backend.pending_unindexed = 0;
}

pub fn open_dim(num_dim: usize) -> Result<(), BpannError> {
    obs::check_dims(num_dim)
}

pub fn open_stride(num_dim: usize) -> Result<(), BpannError> {
    let record_stride = num_dim * std::mem::size_of::<f64>();
    if record_stride > MAX_STRIDE {
        return Err(BpannError::InvalidParameter(format!(
            "record_stride {record_stride} exceeds maximum {MAX_STRIDE}"
        )));
    }
    if num_dim > MAX_DIM {
        return Err(BpannError::InvalidParameter(format!(
            "num_dim {num_dim} exceeds maximum {MAX_DIM}"
        )));
    }
    Ok(())
}
