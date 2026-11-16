use std::fs;
use std::path::PathBuf;

use baux::neighbor_index::distance::l2_f32;
use baux::neighbor_index::error::BpannError;
use baux::neighbor_index::index::LEAF_CAPACITY;
use baux::neighbor_index::index::build::{BpannIndex, IndexHeader};
use baux::neighbor_index::index::search::{
    MmapSearchStore, refine_store, search_store, search_tree,
};
use baux::neighbor_index::mmap_store::MmapColumnStore;
use baux::neighbor_index::observation as obs;
use baux::neighbor_index::tuning::BpannTuning;

use baux::neighbor_index::index::sync_forest::{
    IndexBuildContext, centroid_rows, load_mmap, range_index, vector_index,
};

const INDEX_MIN: usize = 3;

/// Soft-sync spans in `(2 * structured_limit, MID_MAX]`
/// use one in-RAM vector-leaf forest (leaf size `MID_CAPACITY`) so
/// ask prunes without deep k-means tell. Larger spans use one empty-leaf
/// forest (Internal → row-id leaves of `structured_limit`).
const MID_MAX: usize = 10_000;
/// Vector-leaf size inside the mid-band forest (also the greedy visit size).
const MID_CAPACITY: usize = 256;

fn index_threshold(indexed_rows: usize, tuning: &BpannTuning) -> usize {
    if indexed_rows <= 1000 {
        return 1;
    }
    (indexed_rows / tuning.index_fragment).clamp(INDEX_MIN, tuning.index_max)
}

fn search_budget(fragment_count: usize, indexed_rows: usize, tuning: &BpannTuning) -> usize {
    if fragment_count <= 2 {
        return fragment_count;
    }
    let scaled = (indexed_rows / tuning.search_fragment).max(2);
    scaled.min(fragment_count).min(tuning.search_limit)
}

fn search_width(_indexed_rows: usize, tuning: &BpannTuning) -> usize {
    tuning.search_width
}

#[derive(Clone, Debug)]
pub struct IncrementalIndex {
    pub indices: Vec<BpannIndex>,
    pub indexed_rows: usize,
    pub index_dir: PathBuf,
    pub tuning: BpannTuning,
    pending_centroid_sum: Vec<f64>,
    pending_row_count: usize,
}

const _: () = {
    fn assert_sync<T: Send + Sync>() {}
    fn check() {
        assert_sync::<IncrementalIndex>();
    }
};

impl IncrementalIndex {
    pub(crate) fn rescale_pending(&mut self, factors: &[f64]) {
        for (value, &factor) in self.pending_centroid_sum.iter_mut().zip(factors) {
            *value *= factor;
        }
    }

    pub fn new(index_dir: PathBuf) -> Self {
        Self::tuning_config(index_dir, BpannTuning::default())
    }

    pub fn tuning_config(index_dir: PathBuf, tuning: BpannTuning) -> Self {
        Self {
            indices: Vec::new(),
            indexed_rows: 0,
            index_dir,
            tuning,
            pending_centroid_sum: Vec::new(),
            pending_row_count: 0,
        }
    }

    pub fn with_tuning(mut self, tuning: BpannTuning) -> Self {
        self.tuning = tuning;
        self
    }

    pub fn note_rows(
        mut self,
        x: &ndarray::ArrayView2<f64>,
        scale_x: bool,
        x_scale: &[f64],
    ) -> Self {
        if x.nrows() == 0 {
            return self;
        }
        if self.pending_centroid_sum.len() != x.ncols() {
            self.pending_centroid_sum = vec![0.0; x.ncols()];
        }
        let col_sums = x.sum_axis(ndarray::Axis(0));
        for (j, &s) in col_sums.iter().enumerate() {
            self.pending_centroid_sum[j] += if scale_x { s / x_scale[j] } else { s };
        }
        self.pending_row_count += x.nrows();
        self
    }

    pub(crate) fn take_centroid(&mut self, num_dim: usize) -> Option<Vec<f32>> {
        if self.pending_row_count == 0 {
            return None;
        }
        if self.pending_centroid_sum.len() != num_dim {
            return None;
        }
        let count = self.pending_row_count as f64;
        let centroid = self
            .pending_centroid_sum
            .iter()
            .map(|&s| (s / count) as f32)
            .collect();
        self.pending_centroid_sum.fill(0.0);
        self.pending_row_count = 0;
        Some(centroid)
    }

    pub fn reset(mut self) -> Self {
        self.indices.clear();
        self.indexed_rows = 0;
        self.pending_centroid_sum.clear();
        self.pending_row_count = 0;
        self
    }

    #[allow(clippy::too_many_arguments)]
    pub fn sync_pending(
        mut self,
        train_x: &MmapColumnStore,
        num_dim: usize,
        scale_x: bool,
        x_scale: &[f64],
        work_dir: &std::path::Path,
        num_metrics: usize,
        end: usize,
        tuning: &BpannTuning,
    ) -> Result<Self, BpannError> {
        let ctx = IndexBuildContext {
            train_x,
            num_dim,
            scale_x,
            x_scale,
            work_dir,
            num_metrics,
        };
        self.index_through(&ctx, end, tuning)?;
        Ok(self)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn rebuild(
        mut self,
        train_x: &MmapColumnStore,
        num_dim: usize,
        scale_x: bool,
        x_scale: &[f64],
        work_dir: &std::path::Path,
        num_metrics: usize,
        end: usize,
        tuning: &BpannTuning,
    ) -> Result<Self, BpannError> {
        self = self.reset();
        self.sync_pending(
            train_x,
            num_dim,
            scale_x,
            x_scale,
            work_dir,
            num_metrics,
            end,
            tuning,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn ensure_backend(
        mut self,
        train_x: &MmapColumnStore,
        num_dim: usize,
        scale_x: bool,
        x_scale: &[f64],
        work_dir: &std::path::Path,
        num_metrics: usize,
        end: usize,
        tuning: &BpannTuning,
    ) -> Result<Self, BpannError> {
        let ctx = IndexBuildContext {
            train_x,
            num_dim,
            scale_x,
            x_scale,
            work_dir,
            num_metrics,
        };
        self.index_through(&ctx, end, tuning)?;
        Ok(self)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn ensure_tuned(
        self,
        train_x: &MmapColumnStore,
        num_dim: usize,
        scale_x: bool,
        x_scale: &[f64],
        work_dir: &std::path::Path,
        num_metrics: usize,
        end: usize,
        tuning: &BpannTuning,
    ) -> Result<Self, BpannError> {
        self.ensure_backend(
            train_x,
            num_dim,
            scale_x,
            x_scale,
            work_dir,
            num_metrics,
            end,
            tuning,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn persist_backend(
        mut self,
        train_x: &MmapColumnStore,
        num_dim: usize,
        scale_x: bool,
        x_scale: &[f64],
        work_dir: &std::path::Path,
        num_metrics: usize,
        tuning: &BpannTuning,
    ) -> Result<Self, BpannError> {
        let ctx = IndexBuildContext {
            train_x,
            num_dim,
            scale_x,
            x_scale,
            work_dir,
            num_metrics,
        };
        self.persist_index(&ctx, tuning)?;
        Ok(self)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn persist_tuned(
        self,
        train_x: &MmapColumnStore,
        num_dim: usize,
        scale_x: bool,
        x_scale: &[f64],
        work_dir: &std::path::Path,
        num_metrics: usize,
        tuning: &BpannTuning,
    ) -> Result<Self, BpannError> {
        self.persist_backend(
            train_x,
            num_dim,
            scale_x,
            x_scale,
            work_dir,
            num_metrics,
            tuning,
        )
    }

    pub fn needs_rewrite(&self, index_dirty: bool, nrows: usize) -> bool {
        if self.indices.len() > 1 {
            return true;
        }
        if index_dirty {
            return true;
        }
        let on_disk = match disk_rows(&self.index_dir) {
            Ok(on_disk) => on_disk,
            Err(_) => return true,
        };
        if on_disk != nrows {
            return true;
        }
        match self.indices.first() {
            Some(index) => !index.disk_index().unwrap_or(false),
            None => on_disk != 0 || nrows != 0,
        }
    }

    fn persist_index(
        &mut self,
        ctx: &IndexBuildContext<'_>,
        tuning: &BpannTuning,
    ) -> Result<(), BpannError> {
        self.index_through(ctx, ctx.train_x.nrows, tuning)?;
        if self.indices.is_empty() && self.indexed_rows == 0 {
            return Ok(());
        }
        if self.indices.len() > 1 {
            let merged =
                BpannIndex::concat_merge(self.indices.clone(), self.index_dir.clone(), false)?;
            merged.persist()?;
        } else if let Some(index) = self.indices.first() {
            index.persist()?;
        }
        obs::write_metadata(
            ctx.work_dir,
            ctx.train_x.nrows,
            ctx.num_dim,
            ctx.num_metrics,
            ctx.scale_x,
            self.indexed_rows,
        )?;
        obs::write_rows(ctx.work_dir, self.indexed_rows)?;
        Ok(())
    }

    fn index_through(
        &mut self,
        ctx: &IndexBuildContext<'_>,
        end: usize,
        tuning: &BpannTuning,
    ) -> Result<(), BpannError> {
        if self.indexed_rows >= end {
            return Ok(());
        }
        let limit = tuning.structured_limit;
        let pending = end - self.indexed_rows;
        // Soft-sync spans just above `limit` used to take one structured
        // `build_rows` tree (better ask, slower tell). Very large spans
        // become one empty-leaf forest (Internal → row-id leaves of `limit`).
        // Mid-band spans become one in-RAM vector-leaf forest so ask prunes
        // without k-means tell.
        let chunk_large_spans = pending > limit.saturating_mul(2);
        if chunk_large_spans && pending > MID_MAX {
            let _ = self.take_centroid(ctx.num_dim);
            self.build_ranges(ctx, self.indexed_rows, end, limit)?;
            self.maybe_compact(ctx, tuning)?;
        } else if chunk_large_spans {
            let _ = self.take_centroid(ctx.num_dim);
            self.build_vectors(ctx, self.indexed_rows, end, MID_CAPACITY)?;
            self.maybe_compact(ctx, tuning)?;
        } else {
            self.build_batch(ctx, self.indexed_rows, end, tuning)?;
            self.maybe_compact(ctx, tuning)?;
        }
        obs::write_rows(ctx.work_dir, self.indexed_rows)?;
        Ok(())
    }

    /// RAM amalgamation only: no `persist()`, no metadata rewrite.
    fn maybe_compact(
        &mut self,
        ctx: &IndexBuildContext<'_>,
        tuning: &BpannTuning,
    ) -> Result<(), BpannError> {
        let max_fragments = index_threshold(self.indexed_rows, tuning);
        let compact_limit = max_fragments.saturating_mul(2).max(max_fragments + 1);
        if self.indices.len() > compact_limit {
            self.compact(ctx, tuning)?;
        }
        Ok(())
    }

    fn compact(
        &mut self,
        ctx: &IndexBuildContext<'_>,
        tuning: &BpannTuning,
    ) -> Result<(), BpannError> {
        if self.indexed_rows == 0 {
            self.indices.clear();
            return Ok(());
        }
        let max_fragments = index_threshold(self.indexed_rows, tuning);
        while self.indices.len() > max_fragments {
            let over = self.indices.len() - max_fragments;
            let merge_n = over.clamp(2, 4).min(self.indices.len());
            self.amalgamate_run(ctx, merge_n, tuning)?;
        }
        Ok(())
    }

    fn amalgamate_run(
        &mut self,
        _ctx: &IndexBuildContext<'_>,
        merge_n: usize,
        tuning: &BpannTuning,
    ) -> Result<(), BpannError> {
        if self.indices.len() < 2 {
            return Ok(());
        }
        let merge_n = merge_n.min(self.indices.len());
        let mut best_i = 0usize;
        let mut best_rows = usize::MAX;
        let small_limit = tuning.fragment_rows;
        for i in 0..=self.indices.len().saturating_sub(merge_n) {
            let slice = &self.indices[i..i + merge_n];
            let rows: usize = slice.iter().map(|index| index.header.indexed_rows).sum();
            let all_small = slice.windows(2).all(|pair| {
                pair[0].header.indexed_rows <= small_limit
                    && pair[1].header.indexed_rows <= small_limit
            });
            let rank = if all_small {
                rows
            } else {
                rows + usize::MAX / 2
            };
            if rank < best_rows {
                best_rows = rank;
                best_i = i;
            }
        }
        let removed: Vec<BpannIndex> = self.indices.drain(best_i..best_i + merge_n).collect();
        let merged = BpannIndex::concat_merge(removed, self.index_dir.clone(), false)?;
        self.indices.insert(best_i, merged);
        Ok(())
    }

    #[allow(dead_code)]
    fn amalgamate_pair(
        &mut self,
        ctx: &IndexBuildContext<'_>,
        tuning: &BpannTuning,
    ) -> Result<(), BpannError> {
        self.amalgamate_run(ctx, 2, tuning)
    }

    /// Large soft-sync: one fragment of empty row-id leaves under an Internal
    /// root.
    fn build_ranges(
        &mut self,
        ctx: &IndexBuildContext<'_>,
        start: usize,
        end: usize,
        leaf_rows: usize,
    ) -> Result<(), BpannError> {
        if start >= end {
            return Ok(());
        }
        let index = range_index(ctx, start, end, leaf_rows, self.index_dir.clone())?;
        self.indices.push(index);
        self.indexed_rows = end;
        Ok(())
    }

    /// Mid-band: one fragment of in-RAM vector leaves (no k-means).
    fn build_vectors(
        &mut self,
        ctx: &IndexBuildContext<'_>,
        start: usize,
        end: usize,
        leaf_rows: usize,
    ) -> Result<(), BpannError> {
        if start >= end {
            return Ok(());
        }
        let index = vector_index(ctx, start, end, leaf_rows, self.index_dir.clone())?;
        self.indices.push(index);
        self.indexed_rows = end;
        Ok(())
    }

    fn build_batch(
        &mut self,
        ctx: &IndexBuildContext<'_>,
        start: usize,
        end: usize,
        tuning: &BpannTuning,
    ) -> Result<(), BpannError> {
        if start >= end {
            return Ok(());
        }
        let seed = tuning.build_seed.unwrap_or(start as u64);
        let row_ids: Vec<u32> = (start..end).map(|i| i as u32).collect();
        let batch_len = end - start;
        let index = if batch_len <= tuning.structured_limit {
            let centroid = self
                .take_centroid(ctx.num_dim)
                .unwrap_or_else(|| centroid_rows(ctx, start, end).expect("centroid_rows"));
            BpannIndex::build_centroid(
                &row_ids,
                centroid,
                ctx.num_dim,
                self.index_dir.clone(),
                false,
            )?
        } else {
            // Structured leaf path takes pending for placement; the large path
            // builds its own tree from rows but must still drain the
            // accumulator so the next small sync is not poisoned by
            // already-indexed mass.
            let _ = self.take_centroid(ctx.num_dim);
            let vectors = load_mmap(ctx, start, end)?;
            BpannIndex::build_tree(
                &row_ids,
                &vectors,
                ctx.num_dim,
                LEAF_CAPACITY,
                seed,
                self.index_dir.clone(),
                false,
                tuning,
            )?
        };
        self.indices.push(index);
        self.indexed_rows = end;
        Ok(())
    }

    pub fn search_candidates(
        &self,
        query_f32: &[f32],
        k: usize,
        store: Option<&MmapSearchStore<'_>>,
        tuning: &BpannTuning,
    ) -> Result<Vec<(u32, f32)>, BpannError> {
        let budget = search_budget(self.indices.len(), self.indexed_rows, tuning);
        let indices_to_search: Vec<&BpannIndex> = if self.indices.len() <= budget {
            self.indices.iter().collect()
        } else {
            let mut ranked: Vec<(f32, &BpannIndex)> = self
                .indices
                .iter()
                .map(|index| {
                    let centroid = index.root_centroid();
                    let dist = if centroid.is_empty() {
                        f32::INFINITY
                    } else {
                        l2_f32(query_f32, &centroid)
                    };
                    (dist, index)
                })
                .collect();
            ranked.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
            ranked
                .into_iter()
                .take(budget)
                .map(|(_, index)| index)
                .collect()
        };
        let per_fragment_k = k
            .saturating_mul(self.indices.len())
            .div_ceil(indices_to_search.len().max(1))
            .max(k);
        let mut merged: Vec<(u32, f32)> = Vec::new();
        for index in indices_to_search {
            merged.extend(search_candidates(
                index,
                query_f32,
                per_fragment_k,
                store,
                tuning,
            )?);
        }
        merged.sort_by(|a, b| {
            a.1.partial_cmp(&b.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.0.cmp(&b.0))
        });
        merged.truncate(k);
        Ok(merged)
    }

    pub fn search_default(
        &self,
        query_f32: &[f32],
        k: usize,
        store: Option<&MmapSearchStore<'_>>,
    ) -> Result<Vec<(u32, f32)>, BpannError> {
        self.search_candidates(query_f32, k, store, &self.tuning)
    }

    pub fn index_bytes(&self) -> usize {
        self.indices.iter().map(|i| i.index_bytes()).sum()
    }
}

fn search_candidates(
    index: &BpannIndex,
    query: &[f32],
    k: usize,
    store: Option<&MmapSearchStore<'_>>,
    tuning: &BpannTuning,
) -> Result<Vec<(u32, f32)>, BpannError> {
    let rows = index.header.indexed_rows;
    // Mode cliffs are call-time tuning. On-disk skip edges reflect build-time
    // limits and may be empty if limits changed since the fragment was
    // built.
    if tuning.exhaustive_search(rows) {
        search_store(index, query, k, store)
    } else {
        let beam = search_width(rows, tuning);
        let mut visited = Vec::new();
        if tuning.refine_search(rows) {
            refine_store(index, query, k, beam, &mut visited, store)
        } else {
            search_tree(index, query, k, beam, store)
        }
    }
}

fn disk_rows(index_dir: &std::path::Path) -> Result<usize, BpannError> {
    let header_path = index_dir.join("header.json");
    if !header_path.exists() {
        return Ok(0);
    }
    let text = fs::read_to_string(&header_path)
        .map_err(|e| BpannError::InvalidParameter(e.to_string()))?;
    let header: IndexHeader = baux::artifact::json::from_str(&text)
        .map_err(|e| BpannError::InvalidParameter(e.to_string()))?;
    Ok(header.indexed_rows)
}

#[cfg(test)]
mod tests {
    use super::*;
    use baux::neighbor_index::mmap_store::MmapColumnStore;
    use ndarray::array;
    use tempfile::TempDir;

    #[test]
    fn search_max() {
        let tuning = BpannTuning::default();
        // Default search_limit=1: with many fragments, search only one.
        assert_eq!(search_budget(1, 100_000, &tuning), 1);
        assert_eq!(search_budget(2, 100_000, &tuning), 2);
        assert_eq!(search_budget(8, 100_000, &tuning), 1);
        assert_eq!(search_budget(32, 800_000, &tuning), 1);
    }

    #[test]
    fn sync_behavioral() {
        let dir = TempDir::new().unwrap();
        let tuning = BpannTuning::default();
        let mut idx = IncrementalIndex::tuning_config(dir.path().join("index"), tuning);
        let x_path = dir.path().join("train_x.bin");
        let mut store = MmapColumnStore::open_mmap(x_path, 2, None).unwrap();
        for batch in 0..3 {
            let row0 = batch as f64 * 2.0;
            let chunk = array![[row0, 0.0], [row0 + 1.0, 0.0]];
            store = store.mmap_append(&chunk.view()).unwrap();
            let start = batch * 2;
            let end = start + 2;
            idx = idx
                .sync_pending(&store, 2, false, &[1.0, 1.0], dir.path(), 1, end, &tuning)
                .unwrap();
        }
        idx = idx
            .persist_backend(&store, 2, false, &[1.0, 1.0], dir.path(), 1, &tuning)
            .unwrap();
        assert!(dir.path().join("index/header.json").exists());
        assert_eq!(idx.indexed_rows, 6);
    }

    #[test]
    fn sync_behavioral2() {
        let dir = TempDir::new().unwrap();
        let tuning = BpannTuning::default();
        let index_dir = dir.path().join("index");

        assert_eq!(disk_rows(&index_dir).unwrap(), 0);

        let mut idx = IncrementalIndex::tuning_config(index_dir.clone(), tuning);
        assert!(!idx.needs_rewrite(false, 0));
        assert!(idx.needs_rewrite(true, 0));

        let x_path = dir.path().join("train_x.bin");
        let mut store = MmapColumnStore::open_mmap(x_path, 2, None).unwrap();
        let chunk = array![[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]];
        store = store.mmap_append(&chunk.view()).unwrap();
        idx = idx
            .sync_pending(&store, 2, false, &[1.0, 1.0], dir.path(), 1, 3, &tuning)
            .unwrap();
        idx = idx
            .persist_backend(&store, 2, false, &[1.0, 1.0], dir.path(), 1, &tuning)
            .unwrap();

        assert_eq!(disk_rows(&index_dir).unwrap(), 3);
        assert!(idx.needs_rewrite(false, 999));
    }

    #[test]
    fn api_behavioral() {
        let dir = TempDir::new().unwrap();
        let tuning = BpannTuning::default();
        let mut idx = IncrementalIndex::tuning_config(dir.path().to_path_buf(), tuning);
        let x = array![[0.0, 1.0], [1.0, 0.0]];
        idx = idx.note_rows(&x.view(), false, &[1.0, 1.0]);
        assert!(idx.take_centroid(2).is_some());
        let x_path = dir.path().join("train_x.bin");
        let mut store = MmapColumnStore::open_mmap(x_path, 2, None).unwrap();
        store = store.mmap_append(&x.view()).unwrap();
        idx = idx
            .sync_pending(&store, 2, false, &[1.0, 1.0], dir.path(), 1, 2, &tuning)
            .unwrap();
        let results = idx
            .search_candidates(&[0.0, 0.0], 1, None, &tuning)
            .unwrap();
        let _ = results;
        assert_eq!(idx.indexed_rows, 2);
        assert!(!idx.indices.is_empty());
        // Soft sync leaves pages on disk unwritten; hard persist materializes
        // files.
        assert_eq!(idx.index_bytes(), 0);
        idx = idx
            .persist_backend(&store, 2, false, &[1.0, 1.0], dir.path(), 1, &tuning)
            .unwrap();
        assert!(idx.index_bytes() > 0);
    }

    #[test]
    fn sync_behavioral3() {
        let dir = TempDir::new().unwrap();
        let tuning = BpannTuning::default();
        let mut idx = IncrementalIndex::tuning_config(dir.path().join("index"), tuning);
        let x_path = dir.path().join("train_x.bin");
        let mut store = MmapColumnStore::open_mmap(x_path, 2, None).unwrap();
        for batch in 0..8 {
            let row0 = batch as f64 * 2.0;
            let chunk = array![[row0, 0.0], [row0 + 1.0, 0.0]];
            store = store.mmap_append(&chunk.view()).unwrap();
            let start = batch * 2;
            let end = start + 2;
            idx = idx
                .sync_pending(&store, 2, false, &[1.0, 1.0], dir.path(), 1, end, &tuning)
                .unwrap();
        }
        assert!(idx.indices.len() <= 4);
        let _ = idx
            .search_candidates(&[1.0, 0.0], 2, None, &tuning)
            .unwrap();
    }

    /// Regression: soft-sync spans larger than `structured_limit` must still
    /// consume pending-centroid state (drained before chunked leaf builds).
    /// Otherwise the next structured fragment is placed with a contaminated
    /// centroid.
    #[test]
    fn soft_centroid() {
        let tuning = BpannTuning::default();
        let limit = tuning.structured_limit;
        let n = limit + 76; // force multi-chunk soft-sync
        let dir = TempDir::new().unwrap();
        let mut idx = IncrementalIndex::tuning_config(dir.path().join("index"), tuning);
        let x_path = dir.path().join("train_x.bin");
        let mut store = MmapColumnStore::open_mmap(x_path, 2, None).unwrap();
        let scale = [1.0_f64, 1.0];

        let mut large = ndarray::Array2::<f64>::zeros((n, 2));
        for i in 0..n {
            large[[i, 0]] = (i as f64) * 0.01;
            large[[i, 1]] = 1.0;
        }
        store = store.mmap_append(&large.view()).unwrap();
        idx = idx.note_rows(&large.view(), false, &scale);
        idx = idx
            .sync_pending(&store, 2, false, &scale, dir.path(), 1, n, &tuning)
            .unwrap();

        assert_eq!(
            idx.pending_row_count, 0,
            "large soft-sync batch must clear pending_row_count"
        );
        assert!(
            idx.take_centroid(2).is_none(),
            "large soft-sync batch must leave no pending centroid to take"
        );

        // Metamorphic: next singleton fragment must place at the new row, not a
        // blend with the already-indexed large batch.
        let far = array![[10_000.0, 0.0]];
        store = store.mmap_append(&far.view()).unwrap();
        idx = idx.note_rows(&far.view(), false, &scale);
        idx = idx
            .sync_pending(&store, 2, false, &scale, dir.path(), 1, n + 1, &tuning)
            .unwrap();
        let centroid = idx
            .indices
            .last()
            .expect("singleton fragment")
            .root_centroid();
        assert!(
            (centroid[0] - 10_000.0).abs() < 1.0,
            "post-large-sync fragment must place near the new row, got {centroid:?}"
        );
    }

    #[test]
    fn soft_builds() {
        let tuning = BpannTuning::default();
        let limit = tuning.structured_limit;
        // Must exceed MID_MAX to take the empty-leaf chunk path.
        let n = MID_MAX + limit + 10;
        let dir = TempDir::new().unwrap();
        let mut idx = IncrementalIndex::tuning_config(dir.path().join("index"), tuning);
        let x_path = dir.path().join("train_x.bin");
        let mut store = MmapColumnStore::open_mmap(x_path, 2, None).unwrap();
        let scale = [1.0_f64, 1.0];
        let mut large = ndarray::Array2::<f64>::zeros((n, 2));
        for i in 0..n {
            large[[i, 0]] = i as f64;
            large[[i, 1]] = 0.0;
        }
        store = store.mmap_append(&large.view()).unwrap();
        idx = idx.note_rows(&large.view(), false, &scale);
        idx = idx
            .sync_pending(&store, 2, false, &scale, dir.path(), 1, n, &tuning)
            .unwrap();
        assert_eq!(idx.indexed_rows, n);
        assert_eq!(
            idx.indices.len(),
            1,
            "expected one empty-leaf forest fragment"
        );
        assert!(
            idx.indices[0].pages.len() >= 3,
            "expected multi-page forest, got {} pages",
            idx.indices[0].pages.len()
        );
    }

    #[test]
    fn midband_forest() {
        let tuning = BpannTuning::default();
        let limit = tuning.structured_limit;
        let n = limit * 2 + 100;
        assert!(n > limit * 2 && n <= MID_MAX);
        let dir = TempDir::new().unwrap();
        let mut idx = IncrementalIndex::tuning_config(dir.path().join("index"), tuning);
        let x_path = dir.path().join("train_x.bin");
        let mut store = MmapColumnStore::open_mmap(x_path, 2, None).unwrap();
        let scale = [1.0_f64, 1.0];
        let mut mid = ndarray::Array2::<f64>::zeros((n, 2));
        for i in 0..n {
            mid[[i, 0]] = i as f64;
        }
        store = store.mmap_append(&mid.view()).unwrap();
        idx = idx.note_rows(&mid.view(), false, &scale);
        idx = idx
            .sync_pending(&store, 2, false, &scale, dir.path(), 1, n, &tuning)
            .unwrap();
        assert_eq!(idx.indexed_rows, n);
        assert_eq!(
            idx.indices.len(),
            1,
            "expected one vector-leaf forest fragment"
        );
        assert_eq!(idx.indices[0].header.leaf_capacity, MID_CAPACITY);
        let has_vectors = idx.indices[0].pages.iter().any(
            |p| matches!(p, baux::neighbor_index::index::page::Page::Leaf { vectors, .. } if !vectors.is_empty()),
        );
        assert!(has_vectors);
        assert!(
            !idx.search_candidates(&[1.0, 0.0], 3, None, &tuning)
                .unwrap()
                .is_empty()
        );
    }

    #[test]
    fn midsize_fragment() {
        let tuning = BpannTuning::default();
        let limit = tuning.structured_limit;
        // Between limit and 2*limit: structured single-build path.
        let n = limit + limit / 2;
        assert!(n > limit && n <= limit * 2);
        let dir = TempDir::new().unwrap();
        let mut idx = IncrementalIndex::tuning_config(dir.path().join("index"), tuning);
        let x_path = dir.path().join("train_x.bin");
        let mut store = MmapColumnStore::open_mmap(x_path, 2, None).unwrap();
        let scale = [1.0_f64, 1.0];
        let mut mid = ndarray::Array2::<f64>::zeros((n, 2));
        for i in 0..n {
            mid[[i, 0]] = i as f64;
            mid[[i, 1]] = 0.0;
        }
        store = store.mmap_append(&mid.view()).unwrap();
        idx = idx.note_rows(&mid.view(), false, &scale);
        idx = idx
            .sync_pending(&store, 2, false, &scale, dir.path(), 1, n, &tuning)
            .unwrap();
        assert_eq!(idx.indexed_rows, n);
        assert_eq!(
            idx.indices.len(),
            1,
            "mid-size sync should stay one structured fragment"
        );
        assert_eq!(idx.indices[0].header.indexed_rows, n);
    }
}
