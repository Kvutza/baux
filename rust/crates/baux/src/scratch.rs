//! Cache-aligned Structure-of-Arrays (SoA) and zero-malloc bump arenas.

use core::cmp::Ordering;
use core::ops::{Deref, DerefMut};
use std::sync::Arc;

use ndarray::{Array1, Array2, ArrayView1, ArrayView2, ArrayViewMut1, ArrayViewMut2, Axis};

use crate::error::ENNError;
use crate::sense::{BetterScore, ObjectiveSense, ObsId};
pub use crate::unified_mem::{PageBlock, UnifiedBuffer, UnifiedRead, UnifiedWrite};

/// 64-byte cache line holding 8 contiguous `f64` lanes (NEON 4x2 / AVX-512
/// 8x1).
#[repr(C, align(64))]
#[derive(Clone, Copy, Debug)]
pub struct CacheLine(pub [f64; 8]);

impl Default for CacheLine {
    fn default() -> Self {
        Self([0.0; 8])
    }
}

/// 64-byte cache-line-aligned `f64` buffer for vectorized SIMD reductions.
#[derive(Clone, Debug, Default)]
pub struct AlignedBuffer {
    lines: Vec<CacheLine>,
    len: usize,
}

impl AlignedBuffer {
    pub fn with_capacity(cap: usize) -> Self {
        Self {
            lines: Vec::with_capacity(cap.div_ceil(8)),
            len: 0,
        }
    }

    pub fn resize(&mut self, new_len: usize, value: f64) {
        let num_lines = new_len.div_ceil(8);
        if self.lines.len() < num_lines {
            self.lines.resize(num_lines, CacheLine([value; 8]));
        }
        self.len = new_len;
        self.slice_mut().fill(value);
    }

    #[inline(always)]
    pub fn len(&self) -> usize {
        self.len
    }

    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    #[inline(always)]
    pub fn capacity_floats(&self) -> usize {
        self.lines.capacity() * 8
    }

    #[inline(always)]
    pub fn allocation_ptr(&self) -> *const f64 {
        self.lines.as_ptr().cast::<f64>()
    }

    pub fn clear(&mut self) {
        self.len = 0;
    }

    pub fn reserve_floats(&mut self, total_floats: usize) {
        let target_lines = total_floats.div_ceil(8);
        if self.lines.capacity() < target_lines {
            self.lines.reserve(target_lines - self.lines.len());
        }
    }

    pub fn extend_slice(&mut self, src: &[f64]) {
        if src.is_empty() {
            return;
        }
        let old_len = self.len;
        let new_len = old_len + src.len();
        let num_lines = new_len.div_ceil(8);
        if self.lines.len() < num_lines {
            self.lines.resize(num_lines, CacheLine([0.0; 8]));
        }
        self.len = new_len;
        self.slice_mut()[old_len..new_len].copy_from_slice(src);
    }

    #[inline(always)]
    pub fn as_slice(&self) -> &[f64] {
        // SAFETY: CacheLine is #[repr(C, align(64))] wrapping [f64; 8] with no
        // padding.
        let flat = unsafe {
            core::slice::from_raw_parts(self.lines.as_ptr().cast::<f64>(), self.lines.len() * 8)
        };
        &flat[..self.len]
    }

    #[inline(always)]
    pub fn slice_mut(&mut self) -> &mut [f64] {
        let len = self.len;
        // SAFETY: CacheLine is #[repr(C, align(64))] wrapping [f64; 8] with no
        // padding.
        let flat = unsafe {
            core::slice::from_raw_parts_mut(
                self.lines.as_mut_ptr().cast::<f64>(),
                self.lines.len() * 8,
            )
        };
        &mut flat[..len]
    }
}

impl Deref for AlignedBuffer {
    type Target = [f64];

    #[inline(always)]
    fn deref(&self) -> &Self::Target {
        self.as_slice()
    }
}

impl DerefMut for AlignedBuffer {
    #[inline(always)]
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.slice_mut()
    }
}

/// Structure-of-Arrays (SoA) candidate workspace separating 64-byte-aligned
/// `f64` scores from 32-bit niche-packed [`ObsId`] indices.
#[derive(Clone, Debug, Default)]
pub struct CandidateSoa {
    pub scores: AlignedBuffer,
    pub ids: Vec<ObsId>,
}

impl CandidateSoa {
    pub fn with_capacity(cap: usize) -> Self {
        Self {
            scores: AlignedBuffer::with_capacity(cap),
            ids: Vec::with_capacity(cap),
        }
    }

    /// Vectorized UCB evaluation into the aligned SoA score buffer.
    pub fn evaluate_ucb<O: ObjectiveSense>(
        &mut self,
        mu: &ArrayView1<f64>,
        sigma: &ArrayView1<f64>,
        beta: f64,
        sense: O,
    ) {
        let n = mu.len();
        self.scores.resize(n, 0.0);
        self.ids.clear();
        let out = self.scores.slice_mut();
        if let (Some(m_slice), Some(s_slice)) = (mu.as_slice(), sigma.as_slice()) {
            for i in 0..n {
                out[i] = sense.ucb(m_slice[i], s_slice[i], beta).raw();
            }
        } else {
            for i in 0..n {
                out[i] = sense.ucb(mu[i], sigma[i], beta).raw();
            }
        }
        self.ids.extend((0..n).map(ObsId::new));
    }

    /// Branchless SIMD-friendly top-1 selection without heap allocation.
    pub fn select_best<O: ObjectiveSense>(
        mu: &ArrayView1<f64>,
        sigma: &ArrayView1<f64>,
        beta: f64,
        sense: O,
    ) -> usize {
        let n = mu.len();
        let mut best_idx = 0usize;
        let mut best_score = BetterScore::NEG_INF;
        if let (Some(m_slice), Some(s_slice)) = (mu.as_slice(), sigma.as_slice()) {
            for i in 0..n {
                let score = sense.ucb(m_slice[i], s_slice[i], beta);
                if score > best_score {
                    best_score = score;
                    best_idx = i;
                }
            }
        } else {
            for i in 0..n {
                let score = sense.ucb(mu[i], sigma[i], beta);
                if score > best_score {
                    best_score = score;
                    best_idx = i;
                }
            }
        }
        best_idx
    }

    /// Select top-$k$ candidate indices in $O(N)$ average time using SoA scores
    /// and in-place partial quickselect (`select_nth_unstable_by`).
    pub fn select_topk<O: ObjectiveSense>(
        &mut self,
        mu: &ArrayView1<f64>,
        sigma: &ArrayView1<f64>,
        beta: f64,
        k: usize,
        sense: O,
    ) -> Vec<usize> {
        let n = mu.len();
        if n == 0 || k == 0 {
            return Vec::new();
        }
        let k = k.min(n);
        if k == 1 {
            return vec![Self::select_best(mu, sigma, beta, sense)];
        }
        self.evaluate_ucb(mu, sigma, beta, sense);
        let scores = self.scores.as_slice();
        let cmp = |&a: &ObsId, &b: &ObsId| -> Ordering {
            use baux::semantics::optimizer::surrogate::BetterScore as SafeScore;
            SafeScore(scores[b.index()])
                .cmp(&SafeScore(scores[a.index()]))
                .then_with(|| a.cmp(&b))
        };
        if k < n {
            self.ids.select_nth_unstable_by(k - 1, cmp);
        }
        let top = &mut self.ids[..k];
        top.sort_unstable_by(cmp);
        top.iter().map(|id| id.index()).collect()
    }
}

/// Reusable bump-allocated scratchpad eliminating per-round `malloc`/`free`
/// syscalls on the `ask()` hot path.
#[derive(Clone, Debug, Default)]
pub struct ScratchArena {
    slab: AlignedBuffer,
}

impl ScratchArena {
    pub fn with_capacity(floats: usize) -> Self {
        let mut slab = AlignedBuffer::with_capacity(floats);
        slab.resize(floats.next_multiple_of(8), 0.0);
        Self { slab }
    }

    pub fn capacity(&self) -> usize {
        self.slab.len()
    }

    /// Enter a lifetime-bounded bump allocation scope. When the returned
    /// [`ScratchScope`] drops, the arena automatically resets to offset 0 in
    /// $O(1)$.
    #[inline(always)]
    pub fn scope(&mut self) -> ScratchScope<'_> {
        ScratchScope {
            remaining: self.slab.slice_mut(),
            allocated: 0,
        }
    }
}

/// Lifetime-bound bump allocator capable of handing out multiple disjoint
/// mutable `ndarray` views simultaneously without `unsafe` or heap allocation.
pub struct ScratchScope<'a> {
    remaining: &'a mut [f64],
    allocated: usize,
}

impl<'a> ScratchScope<'a> {
    #[inline(always)]
    pub fn allocated_floats(&self) -> usize {
        self.allocated
    }

    /// Carve a zero-initialized 64-byte-aligned `&'a mut [f64]` slice from the
    /// arena in $O(1)$ pointer-bump time.
    pub fn alloc_slice(&mut self, len: usize) -> Option<&'a mut [f64]> {
        let padded = len.next_multiple_of(8);
        if self.remaining.len() < padded {
            return None;
        }
        let rest = core::mem::take(&mut self.remaining);
        let (head, tail) = rest.split_at_mut(padded);
        self.remaining = tail;
        self.allocated += padded;
        let out = &mut head[..len];
        out.fill(0.0);
        Some(out)
    }

    /// Carve a 1D mutable `ndarray` view from the arena.
    pub fn alloc_vector(&mut self, len: usize) -> Option<ArrayViewMut1<'a, f64>> {
        let slice = self.alloc_slice(len)?;
        Some(ArrayViewMut1::from(slice))
    }

    /// Carve a 2D row-major mutable `ndarray` view from the arena.
    pub fn alloc_matrix(&mut self, rows: usize, cols: usize) -> Option<ArrayViewMut2<'a, f64>> {
        let slice = self.alloc_slice(rows.checked_mul(cols)?)?;
        ArrayViewMut2::from_shape((rows, cols), slice).ok()
    }
}

/// Fixed row capacity per sealed persistent chunk in [`ImLog`].
pub const CHUNK_ROWS: usize = 256;

fn append_view(buf: &mut AlignedBuffer, view: ArrayView2<'_, f64>) {
    if let Some(slice) = view.as_slice() {
        buf.extend_slice(slice);
    } else {
        for row in view.axis_iter(Axis(0)) {
            if let Some(row_slice) = row.as_slice() {
                buf.extend_slice(row_slice);
            } else {
                for &val in row.iter() {
                    buf.extend_slice(core::slice::from_ref(&val));
                }
            }
        }
    }
}

/// Functional-But-In-Place (FBIP) persistent chunked observation log backed by
/// 64-byte-aligned [`AlignedBuffer`] slabs.
///
/// - `Clone` is $O(1)$ (`Arc::clone` on the slab directory and active tail).
/// - Linear `push_rows(self, ...)` mutates the active tail chunk **in place**
///   via `Arc::make_mut` whenever `strong_count == 1`, avoiding heap
///   reallocation until a 256-row chunk fills, and seals completed tail chunks
///   into `slabs` by moving the `Arc<AlignedBuffer>` pointer without copying
///   floats.
/// - Committed rows are write-once immutable across branches.
#[derive(Clone, Debug)]
pub struct ImLog {
    slabs: Arc<Vec<Arc<AlignedBuffer>>>,
    tail: Arc<AlignedBuffer>,
    nrows: usize,
    ncols: usize,
}

impl Default for ImLog {
    fn default() -> Self {
        Self::empty()
    }
}

impl ImLog {
    /// Construct an empty log whose column width is adopted on the first write
    /// (or kept at 0).
    pub fn empty() -> Self {
        Self {
            slabs: Arc::new(Vec::new()),
            tail: Arc::new(AlignedBuffer::default()),
            nrows: 0,
            ncols: 0,
        }
    }

    /// Construct an empty log with fixed column count `ncols` and pre-reserved
    /// 64-byte-aligned tail chunk capacity.
    pub fn with_cols(ncols: usize) -> Self {
        let cap = CHUNK_ROWS.saturating_mul(ncols);
        Self {
            slabs: Arc::new(Vec::new()),
            tail: Arc::new(AlignedBuffer::with_capacity(cap)),
            nrows: 0,
            ncols,
        }
    }

    /// Construct an [`ImLog`] from an owned 2D array.
    pub fn from_array(a: Array2<f64>) -> Self {
        let (nrows, ncols) = a.dim();
        if ncols == 0 {
            return Self {
                slabs: Arc::new(Vec::new()),
                tail: Arc::new(AlignedBuffer::default()),
                nrows,
                ncols: 0,
            };
        }
        let a = a.as_standard_layout().into_owned();
        let slice = a.as_slice().expect("standard layout slice");
        let chunk_elems = CHUNK_ROWS * ncols;
        let full_chunks = nrows / CHUNK_ROWS;
        let mut slabs = Vec::with_capacity(full_chunks);
        let mut offset = 0;
        for _ in 0..full_chunks {
            let mut chunk = AlignedBuffer::with_capacity(chunk_elems);
            chunk.extend_slice(&slice[offset..offset + chunk_elems]);
            slabs.push(Arc::new(chunk));
            offset += chunk_elems;
        }
        let remainder = &slice[offset..];
        let mut tail = AlignedBuffer::with_capacity(chunk_elems.max(remainder.len()));
        tail.extend_slice(remainder);
        Self {
            slabs: Arc::new(slabs),
            tail: Arc::new(tail),
            nrows,
            ncols,
        }
    }

    #[inline(always)]
    pub fn nrows(&self) -> usize {
        self.nrows
    }

    #[inline(always)]
    pub fn len(&self) -> usize {
        self.nrows
    }

    #[inline(always)]
    pub fn ncols(&self) -> usize {
        self.ncols
    }

    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.nrows == 0
    }

    #[inline(always)]
    pub fn num_slabs(&self) -> usize {
        self.slabs.len()
    }

    /// True if `self` and `other` share identical `Arc` slab directory and tail
    /// pointers.
    #[inline(always)]
    pub fn shares_storage(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.slabs, &other.slabs) && Arc::ptr_eq(&self.tail, &other.tail)
    }

    /// True if `self` and `other` share the sealed slab directory `Arc`, or if
    /// all sealed slab `Arc<AlignedBuffer>` entries up to `min(len)` are
    /// pointer-equal.
    pub fn shares_slabs(&self, other: &Self) -> bool {
        if Arc::ptr_eq(&self.slabs, &other.slabs) {
            return true;
        }
        let common = self.slabs.len().min(other.slabs.len());
        (0..common).all(|i| Arc::ptr_eq(&self.slabs[i], &other.slabs[i]))
    }

    /// Raw allocation pointer of the active tail chunk (for verifying zero-copy
    /// FBIP append in unit tests).
    #[inline(always)]
    pub fn tail_ptr(&self) -> *const f64 {
        self.tail.allocation_ptr()
    }

    /// Borrow row `i` directly from its 64-byte-aligned chunk without copying.
    #[inline]
    pub fn row_slice(&self, i: usize) -> &[f64] {
        assert!(i < self.nrows, "row index {i} out of bounds {}", self.nrows);
        if self.ncols == 0 {
            return &[];
        }
        let chunk_idx = i / CHUNK_ROWS;
        let row_in_chunk = i % CHUNK_ROWS;
        let start = row_in_chunk * self.ncols;
        let end = start + self.ncols;
        if chunk_idx < self.slabs.len() {
            &self.slabs[chunk_idx].as_slice()[start..end]
        } else {
            &self.tail.as_slice()[start..end]
        }
    }

    /// Fallible zero-copy row slice accessor.
    #[inline]
    pub fn checked_slice(&self, i: usize) -> Result<&[f64], ENNError> {
        if i >= self.nrows {
            return Err(ENNError::InvalidParameter(format!(
                "ImLog row index {i} out of range [0, {})",
                self.nrows
            )));
        }
        Ok(self.row_slice(i))
    }

    pub fn row_vec(&self, i: usize) -> Array1<f64> {
        Array1::from(self.row_slice(i).to_vec())
    }

    pub fn checked_row(&self, i: usize) -> Result<Array1<f64>, ENNError> {
        Ok(Array1::from(self.checked_slice(i)?.to_vec()))
    }

    pub fn gather_rows(&self, indices: &[usize]) -> Array2<f64> {
        let mut out = Array2::zeros((indices.len(), self.ncols));
        if self.ncols == 0 {
            return out;
        }
        for (new_i, &old_i) in indices.iter().enumerate() {
            let slice = self.row_slice(old_i);
            out.row_mut(new_i).assign(&ArrayView1::from(slice));
        }
        out
    }

    pub fn to_array(&self) -> Array2<f64> {
        let mut out = Array2::<f64>::zeros((self.nrows, self.ncols));
        if self.nrows == 0 || self.ncols == 0 {
            return out;
        }
        if let Some(out_slice) = out.as_slice_mut() {
            let chunk_elems = CHUNK_ROWS * self.ncols;
            let mut offset = 0;
            for slab in self.slabs.iter() {
                out_slice[offset..offset + chunk_elems]
                    .copy_from_slice(&slab.as_slice()[..chunk_elems]);
                offset += chunk_elems;
            }
            let tail_rows = self.nrows - self.slabs.len() * CHUNK_ROWS;
            let tail_len = tail_rows * self.ncols;
            if tail_len > 0 {
                out_slice[offset..offset + tail_len]
                    .copy_from_slice(&self.tail.as_slice()[..tail_len]);
            }
        } else {
            for (i, mut row) in out.axis_iter_mut(Axis(0)).enumerate() {
                row.assign(&ArrayView1::from(self.row_slice(i)));
            }
        }
        out
    }

    /// Zero-copy contiguous 2D view when all rows fit in the active tail chunk
    /// (`nrows <= CHUNK_ROWS`).
    pub fn view(&self) -> Option<ArrayView2<'_, f64>> {
        if self.slabs.is_empty() {
            ArrayView2::from_shape(
                (self.nrows, self.ncols),
                &self.tail.as_slice()[..self.nrows * self.ncols],
            )
            .ok()
        } else {
            None
        }
    }

    pub fn view_owned(&self) -> std::borrow::Cow<'_, [f64]> {
        if self.slabs.is_empty() {
            std::borrow::Cow::Borrowed(&self.tail.as_slice()[..self.nrows * self.ncols])
        } else {
            std::borrow::Cow::Owned(self.to_array().into_raw_vec_and_offset().0)
        }
    }

    /// Iterate over contiguous 64-byte-aligned `(start_row_index, chunk_view)`
    /// blocks without copying.
    pub fn iter_chunks(&self) -> impl Iterator<Item = (usize, ArrayView2<'_, f64>)> + '_ {
        let ncols = self.ncols;
        let slab_iter = self
            .slabs
            .iter()
            .enumerate()
            .filter_map(move |(idx, slab)| {
                let start_row = idx * CHUNK_ROWS;
                let len = CHUNK_ROWS * ncols;
                ArrayView2::from_shape((CHUNK_ROWS, ncols), &slab.as_slice()[..len])
                    .ok()
                    .map(|v| (start_row, v))
            });
        let tail_start = self.slabs.len() * CHUNK_ROWS;
        let tail_rows = self.nrows.saturating_sub(tail_start);
        let tail_iter = (tail_rows > 0).then(|| {
            let len = tail_rows * ncols;
            ArrayView2::from_shape((tail_rows, ncols), &self.tail.as_slice()[..len])
                .ok()
                .map(|v| (tail_start, v))
        });
        slab_iter.chain(tail_iter.flatten())
    }

    /// Append a single row slice using FBIP (`Arc::make_mut` on the active
    /// tail).
    pub fn push_slice(self, row: &[f64]) -> Result<Self, ENNError> {
        let view =
            ArrayView2::from_shape((1, row.len()), row).map_err(|_| ENNError::InvalidShape {
                expected: vec![1, self.ncols],
                got: vec![1, row.len()],
            })?;
        self.push_rows(&view)
    }

    /// Append a 2D batch of rows using FBIP (`Arc::make_mut` on the active tail
    /// chunk when uniquely owned, copy-on-write on only the active tail chunk
    /// when shared across forks).
    pub fn push_rows(mut self, extra: &ArrayView2<f64>) -> Result<Self, ENNError> {
        if self.nrows == 0 && self.ncols == 0 {
            self.ncols = extra.ncols();
            if self.ncols > 0 {
                let cap = CHUNK_ROWS * self.ncols;
                let tail_mut = Arc::make_mut(&mut self.tail);
                tail_mut.reserve_floats(cap);
            }
        } else if extra.ncols() != self.ncols {
            return Err(ENNError::InvalidShape {
                expected: vec![self.nrows, self.ncols],
                got: vec![extra.nrows(), extra.ncols()],
            });
        }

        let n_extra = extra.nrows();
        if n_extra == 0 {
            return Ok(self);
        }
        if self.ncols == 0 {
            self.nrows += n_extra;
            return Ok(self);
        }

        let chunk_elems = CHUNK_ROWS * self.ncols;
        let current_tail_rows = self.tail.len() / self.ncols;
        let needed = CHUNK_ROWS - current_tail_rows;

        if n_extra < needed {
            let tail_mut = Arc::make_mut(&mut self.tail);
            tail_mut.reserve_floats(chunk_elems);
            append_view(tail_mut, *extra);
            self.nrows += n_extra;
            return Ok(self);
        }

        let mut extra_row_start = 0;
        if current_tail_rows > 0 {
            let tail_mut = Arc::make_mut(&mut self.tail);
            tail_mut.reserve_floats(chunk_elems);
            let head_view = extra.slice_axis(Axis(0), ndarray::Slice::from(..needed));
            append_view(tail_mut, head_view);
            let sealed = core::mem::replace(
                &mut self.tail,
                Arc::new(AlignedBuffer::with_capacity(chunk_elems)),
            );
            Arc::make_mut(&mut self.slabs).push(sealed);
            extra_row_start = needed;
        }

        if extra_row_start + CHUNK_ROWS <= n_extra {
            let slabs_mut = Arc::make_mut(&mut self.slabs);
            while extra_row_start + CHUNK_ROWS <= n_extra {
                let end = extra_row_start + CHUNK_ROWS;
                let chunk_view =
                    extra.slice_axis(Axis(0), ndarray::Slice::from(extra_row_start..end));
                let mut chunk_buf = AlignedBuffer::with_capacity(chunk_elems);
                append_view(&mut chunk_buf, chunk_view);
                slabs_mut.push(Arc::new(chunk_buf));
                extra_row_start = end;
            }
        }

        let rem = n_extra - extra_row_start;
        if rem > 0 {
            let tail_mut = Arc::make_mut(&mut self.tail);
            tail_mut.reserve_floats(chunk_elems);
            let rem_view =
                extra.slice_axis(Axis(0), ndarray::Slice::from(extra_row_start..n_extra));
            append_view(tail_mut, rem_view);
        }

        self.nrows += n_extra;
        Ok(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sense::{Maximize, Minimize};
    use ndarray::array;

    #[test]
    fn soa_selection() {
        let mu = array![0.1, 0.9, 0.3, 0.8, 0.2];
        let sigma = array![0.0, 0.0, 0.0, 0.0, 0.0];
        let mut soa = CandidateSoa::with_capacity(8);
        assert_eq!((soa.scores.as_ptr() as usize) % 64, 0);
        assert_eq!(
            CandidateSoa::select_best(&mu.view(), &sigma.view(), 1.0, Maximize),
            1
        );
        assert_eq!(
            CandidateSoa::select_best(&mu.view(), &sigma.view(), 1.0, Minimize),
            0
        );
        assert_eq!(
            soa.select_topk(&mu.view(), &sigma.view(), 1.0, 3, Maximize),
            vec![1, 3, 2]
        );
    }

    #[test]
    fn arena_views() {
        let mut arena = ScratchArena::with_capacity(64);
        let ptr_first = {
            let mut scope = arena.scope();
            let mut v1 = scope.alloc_matrix(4, 3).unwrap();
            let mut v2 = scope.alloc_vector(10).unwrap();
            v1[[2, 1]] = 7.5;
            v2[9] = -3.0;
            assert_eq!(v1[[2, 1]], 7.5);
            assert_eq!(v2[9], -3.0);
            v1.as_ptr()
        };
        let mut scope2 = arena.scope();
        let v3 = scope2.alloc_matrix(4, 3).unwrap();
        assert_eq!(v3.as_ptr(), ptr_first);
        assert_eq!(v3[[2, 1]], 0.0);
    }

    #[test]
    fn imlog_fbip() {
        let mut log = ImLog::with_cols(4);
        assert_eq!((log.tail_ptr() as usize) % 64, 0);
        let initial_ptr = log.tail_ptr();

        // Linear append (strong_count == 1) mutates active 64-byte-aligned tail
        // in place!
        for i in 0..10 {
            let v = i as f64;
            log = log
                .push_rows(&array![[v, v + 1.0, v + 2.0, v + 3.0]].view())
                .unwrap();
            assert_eq!(log.tail_ptr(), initial_ptr);
        }
        assert_eq!(log.nrows(), 10);

        // O(1) structural fork shares slabs and tail pointers.
        let fork_a = log.clone();
        assert!(log.shares_storage(&fork_a));

        // Appending on `log` while `fork_a` is alive CoW-clones only the tail.
        let branch_b = log
            .push_rows(&array![[99.0, 98.0, 97.0, 96.0]].view())
            .unwrap();
        assert_eq!(fork_a.nrows(), 10);
        assert_eq!(branch_b.nrows(), 11);
        assert_eq!(fork_a.row_slice(9), &[9.0, 10.0, 11.0, 12.0]);
        assert_eq!(branch_b.row_slice(10), &[99.0, 98.0, 97.0, 96.0]);
        assert!(fork_a.shares_slabs(&branch_b));

        // Fill across multiple 256-row chunk boundaries and verify zero-copy
        // slab sharing.
        let mut big = branch_b;
        for i in 11..600 {
            let v = i as f64;
            big = big
                .push_rows(&array![[v, v + 1.0, v + 2.0, v + 3.0]].view())
                .unwrap();
        }
        assert_eq!(big.nrows(), 600);
        assert_eq!(big.num_slabs(), 2);
        let big_fork = big.clone();
        assert!(big.shares_storage(&big_fork));

        let big_next = big
            .push_rows(&array![[600.0, 601.0, 602.0, 603.0]].view())
            .unwrap();
        // Sealed slabs 0 and 1 remain pointer-identical across divergent forks!
        assert!(big_fork.shares_slabs(&big_next));
        assert_eq!(big_fork.nrows(), 600);
        assert_eq!(big_next.nrows(), 601);
        assert_eq!(big_next.row_slice(0), &[0.0, 1.0, 2.0, 3.0]);
        assert_eq!(big_next.row_slice(256), &[256.0, 257.0, 258.0, 259.0]);
        assert_eq!(big_next.row_slice(600), &[600.0, 601.0, 602.0, 603.0]);
    }
}
