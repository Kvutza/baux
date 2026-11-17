//! Zero-materialization streaming candidate evaluation (`PullPool` -> SIMD +
//! sparse-delta `TopK` -> ENN posterior moments and Thompson draws).
//!
//! Never allocates a `(num_candidates, num_dim)` `Array2<f64>` candidate
//! matrix: candidate rows are evaluated on the fly inside per-worker
//! 64-byte-aligned `1 x D` [`AlignedBuffer`] scratch (or via
//! $O(k_{\text{pert}})$ sparse deltas around `x_center` when `k_pert << D`),
//! streaming directly into `(C, M)` predictions or `(S, C, M)` posterior draws.

use core::cmp::Ordering;

use baux::semantics::neighbor::DistanceMetric;
use baux::semantics::neighbor::simd::{simd_l2, simd_scaled};
use ndarray::{Array2, Array3, ArrayView1, ArrayView2, s};
use rand::RngCore;
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;

#[cfg(not(target_arch = "wasm32"))]
use baux::neighbor_index::parallel::use_rayon;

use super::draw_compute::draw_internals;
use super::empty_internals;
use super::weighted::se_components;
use crate::draw::DrawInternals;
use crate::error::{ENNError, EPS_VAR};
use crate::index::IndexDriver;
use crate::model::ENN;
use crate::params::{ENNParams, PosteriorFlags};
use crate::scratch::{AlignedBuffer, ImLog};
use crate::surrogate::{ENNSurrogate, SurrogatePrediction};

/// Maximum tile height for non-exact / disk / non-Euclidean ENN backends so
/// `(num_candidates, num_dim)` is never allocated when `num_candidates > 64`.
pub const TILE_ROWS: usize = 64;

/// Stateless pull-array interface over a `(num_candidates, num_dim)` candidate
/// pool.
pub trait PullPool: Sync {
    fn num_candidates(&self) -> usize;
    fn num_dim(&self) -> usize;

    /// Materialize candidate `cand` into `out[..num_dim]`.
    fn write_row(&self, cand: usize, out: &mut [f64]);

    /// Center vector when candidates are sparse coordinate perturbations around
    /// a shared center.
    fn sparse_center(&self) -> Option<&[f64]> {
        None
    }

    /// Pack perturbed `(dim_idx, new_val)` pairs for `cand` into caller
    /// scratch. Returns `Some(k_pert)` when all perturbations fit in
    /// `out_idx`, or `None` to fall back to dense `write_row` evaluation.
    fn pack_deltas(
        &self,
        _cand: usize,
        _out_idx: &mut [usize],
        _out_val: &mut [f64],
    ) -> Option<usize> {
        None
    }
}

impl PullPool for baux::semantics::pool::DelayedPool<'_> {
    #[inline(always)]
    fn num_candidates(&self) -> usize {
        baux::semantics::pool::DelayedPool::num_candidates(self)
    }

    #[inline(always)]
    fn num_dim(&self) -> usize {
        baux::semantics::pool::DelayedPool::num_dim(self)
    }

    #[inline]
    fn write_row(&self, cand: usize, out: &mut [f64]) {
        baux::semantics::pool::DelayedPool::write_row(self, cand, out);
    }

    #[inline]
    fn sparse_center(&self) -> Option<&[f64]> {
        match *self {
            Self::Raasp { center, .. } | Self::SobolMasked { center, .. } => Some(center),
            _ => None,
        }
    }

    #[inline]
    fn pack_deltas(
        &self,
        cand: usize,
        out_idx: &mut [usize],
        out_val: &mut [f64],
    ) -> Option<usize> {
        match *self {
            Self::Raasp {
                key,
                lower,
                upper,
                num_pert,
                ..
            } => baux::semantics::pool::raasp_pack(
                key, cand, lower, upper, num_pert, out_idx, out_val,
            ),
            Self::SobolMasked {
                key,
                center,
                num_pert,
                ..
            } => {
                let dim = center.len();
                let cap = out_idx.len().min(out_val.len());
                let key_c = baux::semantics::pool::cand_key(key, cand);
                let prob: f64 = baux::semantics::pool::raasp_prob(num_pert, dim);
                let mut k = 0usize;
                for j in 0..dim {
                    if baux::semantics::pool::raasp_perturbs(key_c, j, dim, prob) {
                        if k >= cap {
                            return None;
                        }
                        out_idx[k] = j;
                        out_val[k] = self.eval(cand, j);
                        k += 1;
                    }
                }
                Some(k)
            }
            _ => None,
        }
    }
}

/// Materialize ONLY the winning `indices` (`num_arms << num_candidates`) from a
/// [`PullPool`] into an `(indices.len(), num_dim)` `Array2<f64>`.
pub fn materialize_selected<P: PullPool + ?Sized>(pool: &P, indices: &[usize]) -> Array2<f64> {
    let d = pool.num_dim();
    let mut out = Array2::zeros((indices.len(), d));
    if d == 0 || indices.is_empty() {
        return out;
    }
    for (row_idx, &cand_idx) in indices.iter().enumerate() {
        let mut row = out.row_mut(row_idx);
        if let Some(slice) = row.as_slice_mut() {
            pool.write_row(cand_idx, slice);
        } else {
            let mut tmp = vec![0.0; d];
            pool.write_row(cand_idx, &mut tmp);
            row.assign(&ArrayView1::from(&tmp));
        }
    }
    out
}

/// Index-remapped [`PullPool`] view (used to exclude in-flight pending points
/// in multi-region search without materializing `(C, D)`).
pub struct RemappedPool<'a, P: PullPool + ?Sized> {
    inner: &'a P,
    kept_indices: Vec<usize>,
}

impl<'a, P: PullPool + ?Sized> RemappedPool<'a, P> {
    pub fn exclude_pending(
        inner: &'a P,
        pending: &Array2<f64>,
        min_arms: usize,
    ) -> Result<Self, ENNError> {
        let c = inner.num_candidates();
        let d = inner.num_dim();
        let mut row_buf = AlignedBuffer::with_capacity(d);
        row_buf.resize(d, 0.0);
        let mut kept = Vec::with_capacity(c);
        for cand in 0..c {
            inner.write_row(cand, &mut row_buf.slice_mut()[..d]);
            let cand_view = ArrayView1::from(&row_buf.as_slice()[..d]);
            if pending.rows().into_iter().all(|pt| pt != cand_view) {
                kept.push(cand);
            }
        }
        if kept.len() < min_arms {
            return Err(ENNError::InvalidParameter(
                "candidate pool has too few points distinct from pending evaluations".into(),
            ));
        }
        Ok(Self {
            inner,
            kept_indices: kept,
        })
    }
}

impl<P: PullPool + ?Sized> PullPool for RemappedPool<'_, P> {
    #[inline(always)]
    fn num_candidates(&self) -> usize {
        self.kept_indices.len()
    }

    #[inline(always)]
    fn num_dim(&self) -> usize {
        self.inner.num_dim()
    }

    #[inline]
    fn write_row(&self, cand: usize, out: &mut [f64]) {
        self.inner.write_row(self.kept_indices[cand], out);
    }

    #[inline]
    fn sparse_center(&self) -> Option<&[f64]> {
        self.inner.sparse_center()
    }

    #[inline]
    fn pack_deltas(
        &self,
        cand: usize,
        out_idx: &mut [usize],
        out_val: &mut [f64],
    ) -> Option<usize> {
        self.inner
            .pack_deltas(self.kept_indices[cand], out_idx, out_val)
    }
}

#[inline(always)]
fn cmp_neighbor(a: &(f64, usize), b: &(f64, usize)) -> Ordering {
    a.0.total_cmp(&b.0).then_with(|| a.1.cmp(&b.1))
}

/// Bounded streaming top-$k$ nearest-neighbor accumulator with deterministic
/// `(dist2.total_cmp, obs_idx.cmp)` ordering.
struct TopKScratch {
    k: usize,
    items: Vec<(f64, usize)>,
}

impl TopKScratch {
    fn with_capacity(k: usize) -> Self {
        Self {
            k,
            items: Vec::with_capacity(k.saturating_add(1)),
        }
    }

    #[inline(always)]
    fn reset(&mut self, k: usize) {
        self.k = k;
        self.items.clear();
    }

    #[inline(always)]
    fn push(&mut self, d2: f64, idx: usize) {
        if self.k == 0 || !d2.is_finite() {
            return;
        }
        let cand = (d2, idx);
        let len = self.items.len();
        if len == self.k && cmp_neighbor(&cand, &self.items[len - 1]).is_ge() {
            return;
        }
        let pos = self
            .items
            .binary_search_by(|probe| cmp_neighbor(probe, &cand))
            .unwrap_or_else(|e| e);
        if len < self.k {
            self.items.insert(pos, cand);
        } else if pos < self.k {
            self.items.copy_within(pos..self.k - 1, pos + 1);
            self.items[pos] = cand;
        }
    }

    #[inline(always)]
    fn as_slice(&self) -> &[(f64, usize)] {
        &self.items
    }
}

#[inline(always)]
fn exact_dist2(q: &[f64], t: &[f64], x_scale: Option<&[f64]>) -> f64 {
    match x_scale {
        Some(sc) => simd_scaled(q, t, sc),
        None => simd_l2(q, t),
    }
}

struct WorkerScratch {
    cand_row: AlignedBuffer,
    center_loaded: bool,
    pert_idx: Vec<usize>,
    pert_val: Vec<f64>,
    topk: TopKScratch,
    shortlist: TopKScratch,
    w_buf: Vec<f64>,
}

impl WorkerScratch {
    fn new(num_dim: usize, k: usize, shortlist_k: usize, sparse_cap: usize) -> Self {
        let mut cand_row = AlignedBuffer::with_capacity(num_dim);
        cand_row.resize(num_dim, 0.0);
        Self {
            cand_row,
            center_loaded: false,
            pert_idx: vec![0usize; sparse_cap],
            pert_val: vec![0.0f64; sparse_cap],
            topk: TopKScratch::with_capacity(k),
            shortlist: TopKScratch::with_capacity(shortlist_k),
            w_buf: vec![0.0f64; k],
        }
    }
}

struct StreamCtx<'a, P: PullPool + ?Sized> {
    pool: &'a P,
    x_log: &'a ImLog,
    y_log: &'a ImLog,
    yvar_log: Option<&'a ImLog>,
    x_scale: Option<&'a [f64]>,
    y_scale: &'a [f64],
    y_scale_sq: Vec<f64>,
    d2_center: Option<Vec<f64>>,
    center: Option<&'a [f64]>,
    num_dim: usize,
    num_metrics: usize,
    k: usize,
    shortlist_k: usize,
    sparse_cap: usize,
    epistemic_scale: f64,
    aleatoric_scale: f64,
    adapt_neighbors: bool,
}

impl<'a, P: PullPool + ?Sized> StreamCtx<'a, P> {
    fn build(
        model: &'a ENN,
        params: &ENNParams,
        adapt_neighbors: bool,
        pool: &'a P,
        x_log: &'a ImLog,
        y_log: &'a ImLog,
        yvar_log: Option<&'a ImLog>,
    ) -> Self {
        let n_train = model.num_obs();
        let num_dim = model.num_dim();
        let num_metrics = model.num_metrics();
        let k = (params.k_neighbors as usize).min(n_train);
        let shortlist_k = (k + 8).min(n_train);
        let x_scale = if model.scale_x {
            model.x_scale.as_slice()
        } else {
            None
        };
        let y_scale = model
            .output_scale()
            .as_slice()
            .expect("output_scale contiguous");
        let y_scale_sq: Vec<f64> = y_scale.iter().map(|&v| v * v).collect();

        let center = pool
            .sparse_center()
            .filter(|c| c.len() == num_dim && num_dim >= 32);
        let sparse_cap = if center.is_some() {
            (num_dim / 4).max(1)
        } else {
            0
        };
        let d2_center = center.map(|ctr| {
            let mut d2 = Vec::with_capacity(n_train);
            for (_, chunk) in x_log.iter_chunks() {
                let chunk_slice = chunk.as_slice().expect("ImLog chunk contiguous");
                for r in 0..chunk.nrows() {
                    let t_row = &chunk_slice[r * num_dim..(r + 1) * num_dim];
                    d2.push(exact_dist2(ctr, t_row, x_scale));
                }
            }
            d2
        });

        Self {
            pool,
            x_log,
            y_log,
            yvar_log,
            x_scale,
            y_scale,
            y_scale_sq,
            d2_center,
            center,
            num_dim,
            num_metrics,
            k,
            shortlist_k,
            sparse_cap,
            epistemic_scale: params.epistemic_scale,
            aleatoric_scale: params.aleatoric_scale,
            adapt_neighbors,
        }
    }

    fn find_topk(&self, cand: usize, scratch: &mut WorkerScratch) {
        if let (Some(ctr), Some(d2_base)) = (self.center, self.d2_center.as_deref()) {
            if let Some(kp) =
                self.pool
                    .pack_deltas(cand, &mut scratch.pert_idx, &mut scratch.pert_val)
            {
                if kp * 4 < self.num_dim {
                    if !scratch.center_loaded {
                        scratch.cand_row.slice_mut()[..self.num_dim].copy_from_slice(ctr);
                        scratch.center_loaded = true;
                    }
                    {
                        let row_mut = &mut scratch.cand_row.slice_mut()[..self.num_dim];
                        for p in 0..kp {
                            row_mut[scratch.pert_idx[p]] = scratch.pert_val[p];
                        }
                    }

                    scratch.shortlist.reset(self.shortlist_k);
                    let p_idx = &scratch.pert_idx[..kp];
                    let p_val = &scratch.pert_val[..kp];
                    for (start_row, chunk) in self.x_log.iter_chunks() {
                        let chunk_slice = chunk.as_slice().expect("ImLog chunk contiguous");
                        for r in 0..chunk.nrows() {
                            let t = start_row + r;
                            let target = &chunk_slice[r * self.num_dim..(r + 1) * self.num_dim];
                            let mut acc = d2_base[t];
                            if let Some(sc) = self.x_scale {
                                for p in 0..kp {
                                    let j = p_idx[p];
                                    let s = sc[j];
                                    let dn = p_val[p] / s - target[j] / s;
                                    let do_ = ctr[j] / s - target[j] / s;
                                    acc += dn * dn - do_ * do_;
                                }
                            } else {
                                for p in 0..kp {
                                    let j = p_idx[p];
                                    let dn = p_val[p] - target[j];
                                    let do_ = ctr[j] - target[j];
                                    acc += dn * dn - do_ * do_;
                                }
                            }
                            scratch.shortlist.push(acc.max(0.0), t);
                        }
                    }

                    // Refine the shortlist (`k + 8` finalists) with exact SIMD
                    // `f64` distance so exact tie-breaking
                    // matches dense `f64` evaluation.
                    scratch.topk.reset(self.k);
                    let q_slice = &scratch.cand_row.as_slice()[..self.num_dim];
                    for &(_, t) in scratch.shortlist.as_slice() {
                        let d2_exact = exact_dist2(q_slice, self.x_log.row_slice(t), self.x_scale);
                        scratch.topk.push(d2_exact, t);
                    }

                    {
                        let row_mut = &mut scratch.cand_row.slice_mut()[..self.num_dim];
                        for p in 0..kp {
                            let j = scratch.pert_idx[p];
                            row_mut[j] = ctr[j];
                        }
                    }
                    return;
                }
            }
        }

        scratch.center_loaded = false;
        self.pool
            .write_row(cand, &mut scratch.cand_row.slice_mut()[..self.num_dim]);
        let q_slice = &scratch.cand_row.as_slice()[..self.num_dim];
        scratch.topk.reset(self.k);
        for (start_row, chunk) in self.x_log.iter_chunks() {
            let chunk_slice = chunk.as_slice().expect("ImLog chunk contiguous");
            for r in 0..chunk.nrows() {
                let t = start_row + r;
                let target = &chunk_slice[r * self.num_dim..(r + 1) * self.num_dim];
                let d2 = exact_dist2(q_slice, target, self.x_scale);
                scratch.topk.push(d2, t);
            }
        }
    }

    fn eval_moments(
        &self,
        cand: usize,
        scratch: &mut WorkerScratch,
        mu_out: &mut [f64],
        se_out: &mut [f64],
        epi_out: &mut [f64],
        ale_out: &mut [f64],
    ) {
        self.find_topk(cand, scratch);
        let top = scratch.topk.as_slice();
        let w_buf = &mut scratch.w_buf;
        let valid_k = top.len();
        if valid_k == 0 {
            mu_out.fill(0.0);
            se_out.fill(1.0);
            epi_out.fill(1.0);
            ale_out.fill(0.0);
            return;
        }

        if let Some(yvar_log) = self.yvar_log {
            for m in 0..self.num_metrics {
                let k_used = if self.adapt_neighbors {
                    let mut prefix = crate::semantics::neighbor::Prefix::default();
                    for (j, &(d2, n_idx)) in top.iter().enumerate() {
                        let var_y = yvar_log.row_slice(n_idx)[m] / self.y_scale_sq[m];
                        prefix.push(
                            (j + 1) as u32,
                            d2,
                            self.epistemic_scale,
                            self.aleatoric_scale + var_y,
                        );
                    }
                    (prefix.count as usize).clamp(1, valid_k)
                } else {
                    valid_k
                };
                let mut norm = 0.0;
                for j in 0..valid_k {
                    if j < k_used {
                        let (d2, n_idx) = top[j];
                        let var_y = yvar_log.row_slice(n_idx)[m] / self.y_scale_sq[m];
                        let var_total =
                            EPS_VAR + self.epistemic_scale * d2 + self.aleatoric_scale + var_y;
                        let w = 1.0 / var_total;
                        w_buf[j] = w;
                        norm += w;
                    } else {
                        w_buf[j] = 0.0;
                    }
                }
                let inv_norm = 1.0 / norm;
                let mut mu_val = 0.0;
                for j in 0..k_used {
                    let w_norm = w_buf[j] * inv_norm;
                    mu_val += w_norm * self.y_log.row_slice(top[j].1)[m];
                }
                let (se_v, epi_v, ale_v) = se_components(inv_norm, 0.0, self.y_scale[m]);
                mu_out[m] = mu_val;
                se_out[m] = se_v;
                epi_out[m] = epi_v;
                ale_out[m] = ale_v;
            }
        } else {
            let k_used = if self.adapt_neighbors {
                let mut prefix = crate::semantics::neighbor::Prefix::default();
                for (j, &(d2, _)) in top.iter().enumerate() {
                    prefix.push(
                        (j + 1) as u32,
                        d2,
                        self.epistemic_scale,
                        self.aleatoric_scale,
                    );
                }
                (prefix.count as usize).clamp(1, valid_k)
            } else {
                valid_k
            };
            let mut norm = 0.0;
            for j in 0..k_used {
                let var_total = EPS_VAR + self.epistemic_scale * top[j].0 + self.aleatoric_scale;
                let w = 1.0 / var_total;
                w_buf[j] = w;
                norm += w;
            }
            let inv_norm = 1.0 / norm;
            let se_base = inv_norm.max(EPS_VAR).sqrt();
            for m in 0..self.num_metrics {
                let mut mu_val = 0.0;
                for j in 0..k_used {
                    let w_norm = w_buf[j] * inv_norm;
                    mu_val += w_norm * self.y_log.row_slice(top[j].1)[m];
                }
                let se_v = se_base * self.y_scale[m];
                mu_out[m] = mu_val;
                se_out[m] = se_v;
                epi_out[m] = se_v;
                ale_out[m] = 0.0;
            }
        }
    }

    fn eval_internals(
        &self,
        cand: usize,
        scratch: &mut WorkerScratch,
        idx_out: &mut [usize],
        w_out: &mut [f64],
        l2_out: &mut [f64],
        mu_out: &mut [f64],
        se_out: &mut [f64],
        epi_out: &mut [f64],
        ale_out: &mut [f64],
    ) {
        self.find_topk(cand, scratch);
        let top = scratch.topk.as_slice();
        let w_buf = &mut scratch.w_buf;
        let valid_k = top.len();
        for j in 0..valid_k {
            idx_out[j] = top[j].1;
        }
        if valid_k == 0 {
            w_out.fill(0.0);
            l2_out.fill(1.0);
            mu_out.fill(0.0);
            se_out.fill(1.0);
            epi_out.fill(1.0);
            ale_out.fill(0.0);
            return;
        }

        for m in 0..self.num_metrics {
            let k_used = if self.adapt_neighbors {
                let mut prefix = crate::semantics::neighbor::Prefix::default();
                for (j, &(d2, n_idx)) in top.iter().enumerate() {
                    let var_y = match self.yvar_log {
                        Some(yv) => yv.row_slice(n_idx)[m] / self.y_scale_sq[m],
                        None => 0.0,
                    };
                    prefix.push(
                        (j + 1) as u32,
                        d2,
                        self.epistemic_scale,
                        self.aleatoric_scale + var_y,
                    );
                }
                (prefix.count as usize).clamp(1, valid_k)
            } else {
                valid_k
            };
            let mut norm = 0.0;
            for j in 0..valid_k {
                if j < k_used {
                    let (d2, n_idx) = top[j];
                    let var_y = match self.yvar_log {
                        Some(yv) => yv.row_slice(n_idx)[m] / self.y_scale_sq[m],
                        None => 0.0,
                    };
                    let var_total =
                        EPS_VAR + self.epistemic_scale * d2 + self.aleatoric_scale + var_y;
                    let w = 1.0 / var_total;
                    w_buf[j] = w;
                    norm += w;
                } else {
                    w_buf[j] = 0.0;
                }
            }
            let inv_norm = 1.0 / norm;
            let mut l2_sq = 0.0;
            let mut mu_val = 0.0;
            for j in 0..valid_k {
                let w_norm = w_buf[j] * inv_norm;
                w_out[j * self.num_metrics + m] = w_norm;
                l2_sq += w_norm * w_norm;
                mu_val += w_norm * self.y_log.row_slice(top[j].1)[m];
            }
            l2_out[m] = l2_sq.sqrt();
            mu_out[m] = mu_val;
            let (se_v, epi_v, ale_v) = se_components(inv_norm, 0.0, self.y_scale[m]);
            se_out[m] = se_v;
            epi_out[m] = epi_v;
            ale_out[m] = ale_v;
        }
    }
}

/// Evaluate posterior `(mu, se)` for all candidates in `pool` without
/// materializing a `(num_candidates, num_dim)` matrix.
pub fn predict_pool<P: PullPool + ?Sized>(
    enn: &ENNSurrogate,
    pool: &P,
) -> Result<SurrogatePrediction, ENNError> {
    let model = enn
        .model()
        .ok_or_else(|| ENNError::InvalidParameter("Surrogate not fitted".to_string()))?;
    let c = pool.num_candidates();
    let d = pool.num_dim();
    if d != model.num_dim() {
        return Err(ENNError::InvalidShape {
            expected: vec![c, model.num_dim()],
            got: vec![c, d],
        });
    }
    let m = model.num_metrics();
    if c == 0 {
        return Ok(SurrogatePrediction {
            mu: Array2::zeros((0, m)),
            se: Array2::zeros((0, m)),
        });
    }

    let params = enn.resolved_params()?;
    let config = enn.config();

    if model.backend_driver() == IndexDriver::Exact && model.geometry() == DistanceMetric::Euclidean
    {
        if let Some((x_log, y_log, yvar_log)) = model.backend.memory_logs() {
            let (mut mu, mut se, mut se_epi, mut se_ale) =
                if model.num_obs() == 0 || params.k_neighbors <= 0 {
                    let empty = empty_internals(model, c);
                    (empty.mu, empty.se, empty.se_epi, empty.se_ale)
                } else {
                    let ctx = StreamCtx::build(
                        model,
                        &params,
                        config.adapt_neighbors,
                        pool,
                        x_log,
                        y_log,
                        yvar_log,
                    );
                    let mut mu = Array2::zeros((c, m));
                    let mut se = Array2::zeros((c, m));
                    let mut se_epi = Array2::zeros((c, m));
                    let mut se_ale = Array2::zeros((c, m));

                    let mu_s = mu.as_slice_memory_order_mut().unwrap();
                    let se_s = se.as_slice_memory_order_mut().unwrap();
                    let epi_s = se_epi.as_slice_memory_order_mut().unwrap();
                    let ale_s = se_ale.as_slice_memory_order_mut().unwrap();

                    #[cfg(not(target_arch = "wasm32"))]
                    let ran_par = if c >= 8 && use_rayon() {
                        mu_s.par_chunks_mut(m)
                            .zip(se_s.par_chunks_mut(m))
                            .zip(epi_s.par_chunks_mut(m))
                            .zip(ale_s.par_chunks_mut(m))
                            .enumerate()
                            .for_each_init(
                                || {
                                    WorkerScratch::new(
                                        ctx.num_dim,
                                        ctx.k,
                                        ctx.shortlist_k,
                                        ctx.sparse_cap,
                                    )
                                },
                                |scratch, (cand, (((mu_r, se_r), epi_r), ale_r))| {
                                    ctx.eval_moments(cand, scratch, mu_r, se_r, epi_r, ale_r);
                                },
                            );
                        true
                    } else {
                        false
                    };
                    #[cfg(target_arch = "wasm32")]
                    let ran_par = false;

                    if !ran_par {
                        let mut scratch =
                            WorkerScratch::new(ctx.num_dim, ctx.k, ctx.shortlist_k, ctx.sparse_cap);
                        for cand in 0..c {
                            let lo = cand * m;
                            let hi = lo + m;
                            ctx.eval_moments(
                                cand,
                                &mut scratch,
                                &mut mu_s[lo..hi],
                                &mut se_s[lo..hi],
                                &mut epi_s[lo..hi],
                                &mut ale_s[lo..hi],
                            );
                        }
                    }
                    (mu, se, se_epi, se_ale)
                };

            if model.bounded_outputs() {
                crate::y_bounds::naturalize(
                    &mut mu,
                    &mut se,
                    &mut se_epi,
                    &mut se_ale,
                    model.y_bounds(),
                );
            }
            model.apply_codomain(&mut mu, &mut se, &mut se_epi, &mut se_ale)?;
            if let Some(ys) = config.y_scale {
                if config.num_candidates == 0 || config.num_samples == 0 {
                    let cur = model.output_scale();
                    for col in 0..se.ncols() {
                        let denom = cur[col].max(1.0e-12);
                        let factor = ys / denom;
                        for row in 0..se.nrows() {
                            se[[row, col]] *= factor;
                        }
                    }
                }
            }
            return Ok(SurrogatePrediction { mu, se });
        }
    }

    // Bounded-tile fallback (`<= 64 x D` scratch) for Disk, HNSW, GPU, or
    // non-Euclidean metrics.
    let tile_cap = TILE_ROWS.min(c);
    let mut tile_buf = Array2::zeros((tile_cap, d));
    let mut mu = Array2::zeros((c, m));
    let mut se = Array2::zeros((c, m));
    let mut offset = 0;
    while offset < c {
        let len = (c - offset).min(tile_cap);
        for r in 0..len {
            let mut row = tile_buf.row_mut(r);
            pool.write_row(offset + r, row.as_slice_mut().expect("tile row contiguous"));
        }
        let tile_view: ArrayView2<'_, f64> = tile_buf.slice(s![..len, ..]);
        let pred = enn.predict(&tile_view)?;
        mu.slice_mut(s![offset..offset + len, ..]).assign(&pred.mu);
        se.slice_mut(s![offset..offset + len, ..]).assign(&pred.se);
        offset += len;
    }
    Ok(SurrogatePrediction { mu, se })
}

/// Draw Thompson posterior samples `(num_samples, num_candidates, num_metrics)`
/// for `pool` without materializing a `(num_candidates, num_dim)` matrix.
pub fn sample_pool<P: PullPool + ?Sized>(
    enn: &ENNSurrogate,
    pool: &P,
    num_samples: usize,
    rng: &mut dyn RngCore,
) -> Result<Array3<f64>, ENNError> {
    let model = enn
        .model()
        .ok_or_else(|| ENNError::InvalidParameter("Surrogate not fitted".to_string()))?;
    let c = pool.num_candidates();
    let d = pool.num_dim();
    if d != model.num_dim() {
        return Err(ENNError::InvalidShape {
            expected: vec![c, model.num_dim()],
            got: vec![c, d],
        });
    }
    let m = model.num_metrics();
    let params = enn.resolved_params()?;
    let config = enn.config();

    let mut seed_bytes = [0u8; 8];
    rng.fill_bytes(&mut seed_bytes);
    let base_seed = u64::from_le_bytes(seed_bytes) as i64;
    let function_seeds: Vec<i64> = (0..num_samples as i64).map(|i| base_seed + i).collect();

    if c == 0 || num_samples == 0 {
        return Ok(Array3::zeros((num_samples, c, m)));
    }

    if model.backend_driver() == IndexDriver::Exact && model.geometry() == DistanceMetric::Euclidean
    {
        if let Some((x_log, y_log, yvar_log)) = model.backend.memory_logs() {
            let mut internals = if model.num_obs() == 0 || params.k_neighbors <= 0 {
                empty_internals(model, c)
            } else {
                let ctx = StreamCtx::build(
                    model,
                    &params,
                    config.adapt_neighbors,
                    pool,
                    x_log,
                    y_log,
                    yvar_log,
                );
                let k = ctx.k;
                let km = k * m;
                let mut flat_idx = vec![0usize; c * k];
                let mut w_normalized = Array3::zeros((c, k, m));
                let mut l2 = Array2::zeros((c, m));
                let mut mu = Array2::zeros((c, m));
                let mut se = Array2::zeros((c, m));
                let mut se_epi = Array2::zeros((c, m));
                let mut se_ale = Array2::zeros((c, m));

                let w_s = w_normalized.as_slice_memory_order_mut().unwrap();
                let l2_s = l2.as_slice_memory_order_mut().unwrap();
                let mu_s = mu.as_slice_memory_order_mut().unwrap();
                let se_s = se.as_slice_memory_order_mut().unwrap();
                let epi_s = se_epi.as_slice_memory_order_mut().unwrap();
                let ale_s = se_ale.as_slice_memory_order_mut().unwrap();

                #[cfg(not(target_arch = "wasm32"))]
                let ran_par = if c >= 8 && use_rayon() {
                    flat_idx
                        .par_chunks_mut(k)
                        .zip(w_s.par_chunks_mut(km))
                        .zip(l2_s.par_chunks_mut(m))
                        .zip(mu_s.par_chunks_mut(m))
                        .zip(se_s.par_chunks_mut(m))
                        .zip(epi_s.par_chunks_mut(m))
                        .zip(ale_s.par_chunks_mut(m))
                        .enumerate()
                        .for_each_init(
                            || {
                                WorkerScratch::new(
                                    ctx.num_dim,
                                    ctx.k,
                                    ctx.shortlist_k,
                                    ctx.sparse_cap,
                                )
                            },
                            |scratch,
                             (
                                cand,
                                ((((((idx_r, w_r), l2_r), mu_r), se_r), epi_r), ale_r),
                            )| {
                                ctx.eval_internals(
                                    cand, scratch, idx_r, w_r, l2_r, mu_r, se_r, epi_r, ale_r,
                                );
                            },
                        );
                    true
                } else {
                    false
                };
                #[cfg(target_arch = "wasm32")]
                let ran_par = false;

                if !ran_par {
                    let mut scratch =
                        WorkerScratch::new(ctx.num_dim, ctx.k, ctx.shortlist_k, ctx.sparse_cap);
                    for cand in 0..c {
                        ctx.eval_internals(
                            cand,
                            &mut scratch,
                            &mut flat_idx[cand * k..(cand + 1) * k],
                            &mut w_s[cand * km..(cand + 1) * km],
                            &mut l2_s[cand * m..(cand + 1) * m],
                            &mut mu_s[cand * m..(cand + 1) * m],
                            &mut se_s[cand * m..(cand + 1) * m],
                            &mut epi_s[cand * m..(cand + 1) * m],
                            &mut ale_s[cand * m..(cand + 1) * m],
                        );
                    }
                }

                let idx_vecs: Vec<Vec<usize>> =
                    flat_idx.chunks_exact(k).map(<[usize]>::to_vec).collect();
                DrawInternals::new(idx_vecs, w_normalized, l2, mu, se, se_epi, se_ale)
            };

            if !model.bounded_outputs() {
                model.apply_codomain(
                    &mut internals.mu,
                    &mut internals.se,
                    &mut internals.se_epi,
                    &mut internals.se_ale,
                )?;
            }
            let mut draws = draw_internals(model, &internals, &function_seeds)?;
            if model.bounded_outputs() {
                crate::y_bounds::inverse_draws(&mut draws, model.y_bounds());
            }
            return Ok(draws);
        }
    }

    // Bounded-tile fallback with shared `function_seeds` (preserving exact
    // training-index normal hashes across tiles!).
    use crate::traits::PosteriorComputation;
    let flags = PosteriorFlags::default().adaptive_neighbors(config.adapt_neighbors);
    let tile_cap = TILE_ROWS.min(c);
    let mut tile_buf = Array2::zeros((tile_cap, d));
    let mut draws = Array3::zeros((num_samples, c, m));
    let mut offset = 0;
    while offset < c {
        let len = (c - offset).min(tile_cap);
        for r in 0..len {
            let mut row = tile_buf.row_mut(r);
            pool.write_row(offset + r, row.as_slice_mut().expect("tile row contiguous"));
        }
        let tile_view: ArrayView2<'_, f64> = tile_buf.slice(s![..len, ..]);
        let (tile_draws, _) = model.posterior_draw(&tile_view, &params, &function_seeds, &flags)?;
        draws
            .slice_mut(s![.., offset..offset + len, ..])
            .assign(&tile_draws);
        offset += len;
    }
    Ok(draws)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::surrogate::ENNSurrogateConfig;
    use ndarray::array;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    struct SparseMockPool {
        center: Vec<f64>,
        deltas: Vec<Vec<(usize, f64)>>,
    }

    impl PullPool for SparseMockPool {
        fn num_candidates(&self) -> usize {
            self.deltas.len()
        }

        fn num_dim(&self) -> usize {
            self.center.len()
        }

        fn write_row(&self, cand: usize, out: &mut [f64]) {
            out[..self.center.len()].copy_from_slice(&self.center);
            for &(j, v) in &self.deltas[cand] {
                out[j] = v;
            }
        }

        fn sparse_center(&self) -> Option<&[f64]> {
            Some(&self.center)
        }

        fn pack_deltas(
            &self,
            cand: usize,
            out_idx: &mut [usize],
            out_val: &mut [f64],
        ) -> Option<usize> {
            let d = &self.deltas[cand];
            if d.len() > out_idx.len().min(out_val.len()) {
                return None;
            }
            for (k, &(j, v)) in d.iter().enumerate() {
                out_idx[k] = j;
                out_val[k] = v;
            }
            Some(d.len())
        }
    }

    #[test]
    fn streaming_parity() {
        let d = 64usize;
        let n = 40usize;
        let c = 24usize;
        let train_x =
            Array2::from_shape_fn((n, d), |(i, j)| ((i * 31 + j * 17) % 97) as f64 / 97.0);
        let train_y = Array2::from_shape_fn((n, 2), |(i, m)| {
            ((i as f64) * 0.25 + (m as f64) * 1.5).sin()
        });
        let train_yvar = Array2::from_shape_fn((n, 2), |(i, m)| 0.01 * ((i + m + 1) as f64));

        for (scale_x, adapt, with_yvar) in [
            (false, false, false),
            (true, true, false),
            (true, true, true),
        ] {
            let cfg = ENNSurrogateConfig {
                k: 7,
                scale_x,
                adapt_neighbors: adapt,
                num_candidates: 8,
                num_samples: 4,
                y_bounds: Some(array![[-2.0, 2.0], [-2.0, 2.0]]),
                ..Default::default()
            };
            let mut rng = StdRng::seed_from_u64(123);
            let sur = ENNSurrogate::new(cfg)
                .fit(
                    &train_x.view(),
                    &train_y.view(),
                    with_yvar.then_some(&train_yvar.view()),
                    &mut rng,
                )
                .unwrap();

            let center = vec![0.5; d];
            let deltas: Vec<Vec<(usize, f64)>> = (0..c)
                .map(|cand| {
                    vec![
                        (cand % d, 0.1 + 0.03 * (cand as f64)),
                        ((cand * 7 + 3) % d, 0.9 - 0.02 * (cand as f64)),
                    ]
                })
                .collect();
            let pool = SparseMockPool { center, deltas };
            let dense = materialize_selected(&pool, &(0..c).collect::<Vec<_>>());

            let pred_stream = predict_pool(&sur, &pool).unwrap();
            let pred_dense = sur.predict(&dense.view()).unwrap();
            for (a, b) in pred_stream.mu.iter().zip(pred_dense.mu.iter()) {
                assert!((a - b).abs() < 1e-12, "mu mismatch: {a} vs {b}");
            }
            for (a, b) in pred_stream.se.iter().zip(pred_dense.se.iter()) {
                assert!((a - b).abs() < 1e-12, "se mismatch: {a} vs {b}");
            }

            let mut rng_a = StdRng::seed_from_u64(999);
            let mut rng_b = StdRng::seed_from_u64(999);
            let draws_stream = sample_pool(&sur, &pool, 3, &mut rng_a).unwrap();
            let draws_dense = sur.sample(&dense.view(), 3, &mut rng_b).unwrap();
            for (a, b) in draws_stream.iter().zip(draws_dense.iter()) {
                assert!((a - b).abs() < 1e-12, "draw mismatch: {a} vs {b}");
            }
        }
    }
}
