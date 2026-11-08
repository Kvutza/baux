//! Cache-aligned Structure-of-Arrays (SoA) and zero-malloc bump arenas.

use core::cmp::Ordering;
use core::ops::{Deref, DerefMut};
use ndarray::{ArrayView1, ArrayViewMut1, ArrayViewMut2};

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
            scores[b.index()]
                .total_cmp(&scores[a.index()])
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
}
