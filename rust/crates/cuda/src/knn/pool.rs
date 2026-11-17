//! Tiled `DelayedPool<f32>` + ENN posterior + multi-arm selection + fenced
//! winning-row materialization on a single `CudaStream`.

use std::mem::size_of;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, OnceLock};

use baux::semantics::abi::{ArmDecision, PoolDispatch, SelectHeader};
use baux::semantics::affine::Affine;
use baux::semantics::categorical::{CatSpec, code_bit, snap_row};
use baux::semantics::morbo::RescalarizeMode;
use baux::semantics::neighbor::{MetricKind, TopK, ard_dist, ard_sides, region_bounds};
use baux::semantics::optimizer::trust_region::{
    LifecyclePhase, PoolSampler, RegionPlan, RoundPlan, SurrogateMode,
};
use baux::semantics::stiefel::Stiefel;
use baux_cuda_device::{
    KNN_K, KNN_THREADS, KNN_WARPS, MAX_METRICS, MOD_AFFINE, MOD_CORR, MOD_MORBO, MOD_STIEFEL,
    SCORE_THREADS, THREADS, make_pool, pick_host, score_cand,
};
use cuda_core::{CudaEvent, CudaStream, DeviceBuffer, LaunchConfig1D, PinnedHostBuffer};
use cuda_host::kernel_family::{
    KernelFamily, KernelFamilyId, KernelProblem, KernelSelector, KernelVariant,
    NoKernelSelectionCache, SelectionMode,
};

use super::{CudaIndex, CudaScratch, WARP_DIM, next_capacity};
use crate::capture::{CudaResult, cuda_error, read_prefix, timing_event, to_u32};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PoolKind {
    Direct,
    Rows,
    Warps,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PoolMeta {
    pub tile_rows: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PoolProblem {
    pub rows: usize,
    pub dims: usize,
    pub candidates: usize,
    pub arms: usize,
    pub neighbors: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PoolReject(pub &'static str);

impl std::fmt::Display for PoolReject {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.0)
    }
}

impl std::error::Error for PoolReject {}

pub type PoolVariant = KernelVariant<PoolKind, PoolKind, PoolMeta>;
pub type PoolFamily = KernelFamily<PoolKind, PoolKind, PoolMeta, 3>;

impl KernelProblem<PoolVariant> for PoolProblem {
    type Rejection = PoolReject;

    fn validate(&self, variant: &PoolVariant) -> Result<(), Self::Rejection> {
        if self.dims == 0 || self.candidates == 0 || self.arms == 0 || self.arms > self.candidates {
            return Err(PoolReject(
                "CUDA pool shape must have 1 <= arms <= candidates and dims > 0",
            ));
        }
        if self.neighbors == 0 {
            return if *variant.entry() == PoolKind::Direct {
                Ok(())
            } else {
                Err(PoolReject("NoSurrogate/warmup rounds use direct selection"))
            };
        }
        if *variant.entry() == PoolKind::Direct {
            return Err(PoolReject("ENN rounds require tiled neighbor scan"));
        }
        if self.rows == 0 || self.neighbors > KNN_K || self.neighbors > self.rows {
            return Err(PoolReject(
                "CUDA pool neighbor count exceeds history or KNN_K",
            ));
        }
        if variant.metadata().tile_rows == 0 {
            return Err(PoolReject("CUDA pool tile rows must be positive"));
        }
        Ok(())
    }
}

pub struct PoolSelector;

impl KernelSelector<PoolProblem, PoolVariant, PoolKind> for PoolSelector {
    type Error = std::convert::Infallible;

    fn select(
        &mut self,
        _family: KernelFamilyId,
        problem: &PoolProblem,
        _eligible: &[&PoolVariant],
    ) -> Result<PoolKind, Self::Error> {
        if problem.neighbors == 0 {
            Ok(PoolKind::Direct)
        } else if problem.dims >= WARP_DIM {
            Ok(PoolKind::Warps)
        } else {
            Ok(PoolKind::Rows)
        }
    }
}

pub fn pool_family() -> CudaResult<&'static PoolFamily> {
    static FAMILY: OnceLock<Result<PoolFamily, String>> = OnceLock::new();
    FAMILY
        .get_or_init(|| {
            KernelFamily::try_new(
                "baux/delayed-pool",
                1,
                [
                    KernelVariant::new(
                        PoolKind::Direct,
                        PoolKind::Direct,
                        PoolMeta { tile_rows: 0 },
                    ),
                    KernelVariant::new(
                        PoolKind::Rows,
                        PoolKind::Rows,
                        PoolMeta {
                            tile_rows: KNN_THREADS,
                        },
                    ),
                    KernelVariant::new(
                        PoolKind::Warps,
                        PoolKind::Warps,
                        PoolMeta {
                            tile_rows: KNN_WARPS,
                        },
                    ),
                ],
            )
            .map_err(|error| error.to_string())
        })
        .as_ref()
        .map_err(Clone::clone)
}

pub fn select_pool(problem: &PoolProblem) -> CudaResult<(PoolKind, u32)> {
    let family = pool_family()?;
    let mut selector = PoolSelector;
    let mut cache = NoKernelSelectionCache;
    let chosen = family
        .select(problem, SelectionMode::Auto, &mut selector, &mut cache)
        .map_err(|error| error.to_string())?;
    Ok((
        *chosen.variant().entry(),
        chosen.variant().metadata().tile_rows,
    ))
}

#[derive(Clone, Debug, PartialEq)]
pub struct CatSnap {
    pub offset: usize,
    pub card: usize,
    pub width: usize,
    pub codes: Vec<f64>,
    pub one_hot: bool,
}

impl CatSnap {
    pub fn rademacher(offset: usize, card: usize, spec: CatSpec) -> Self {
        let mut codes = Vec::with_capacity(card * spec.width);
        for cat in 0..card {
            for bit in 0..spec.width {
                codes.push(code_bit(
                    cat as i64,
                    spec.variable,
                    spec.seed,
                    bit,
                    spec.sep,
                    spec.width,
                ));
            }
        }
        Self {
            offset,
            card,
            width: spec.width,
            codes,
            one_hot: false,
        }
    }

    pub fn one_hot(offset: usize, card: usize) -> Self {
        let mut codes = vec![0.0; card * card];
        for c in 0..card {
            codes[c * card + c] = 1.0;
        }
        Self {
            offset,
            card,
            width: card,
            codes,
            one_hot: true,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct MorboSpec {
    pub mode: RescalarizeMode,
    pub alpha: f32,
    pub y_min: Vec<f32>,
    pub y_max: Vec<f32>,
    pub weights: Vec<f32>,
}

#[derive(Clone, Debug, Default)]
pub struct PoolModifiers {
    pub weights: Option<Vec<f32>>,
    pub scale_x: Option<Vec<f32>>,
    pub unscaled: Vec<usize>,
    pub tied: Vec<(usize, usize)>,
    pub cats: Vec<CatSnap>,
    pub sides: Option<Vec<f32>>,
    pub ybounds: Option<Vec<[f32; 2]>>,
    pub stiefel: Option<Stiefel>,
    pub affine: Option<Affine>,
    pub morbo: Option<MorboSpec>,
    pub local_scales: Option<Vec<f32>>,
    pub scales: Option<Vec<f32>>,
    pub dirs: Option<Vec<u32>>,
    pub correlated: bool,
    pub adaptive_k: bool,
    pub scale_k: usize,
    pub inc_mean: f32,
    pub inc_se: f32,
}

pub fn eff_weights(dims: usize, mods: Option<&PoolModifiers>) -> Vec<f32> {
    let mut out = vec![1.0_f32; dims];
    let Some(m) = mods else {
        return out;
    };
    if let Some(w) = &m.weights {
        for (j, slot) in out.iter_mut().enumerate().take(w.len().min(dims)) {
            if w[j].is_finite() && w[j] > 0.0 {
                *slot = w[j];
            }
        }
    }
    if let Some(sx) = &m.scale_x {
        for (j, slot) in out.iter_mut().enumerate().take(sx.len().min(dims)) {
            let s = sx[j];
            if s.is_finite() && s > 1.0e-12 {
                *slot /= s * s;
            }
        }
    }
    for &u in &m.unscaled {
        if u < dims {
            out[u] = 1.0;
        }
    }
    out
}

pub fn region_box(
    plan: &RoundPlan,
    center: &[f32],
    mods: Option<&PoolModifiers>,
) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
    let dims = plan.num_dim;
    let mut c_vec = vec![0.5_f32; dims];
    let copy_n = center.len().min(dims);
    c_vec[..copy_n].copy_from_slice(&center[..copy_n]);

    let mut scale = vec![1.0_f32; dims];
    if let Some(m) = mods
        && let Some(sx) = &m.scale_x
    {
        for (j, slot) in scale.iter_mut().enumerate().take(sx.len().min(dims)) {
            if sx[j].is_finite() && sx[j] > 0.0 {
                *slot = sx[j];
            }
        }
    }

    if matches!(plan.phase, LifecyclePhase::Warmup { .. })
        || matches!(plan.region, RegionPlan::Global { .. })
    {
        return (c_vec, scale, vec![0.0_f32; dims], vec![1.0_f32; dims]);
    }

    let mut sides = vec![1.0_f32; dims];
    if let Some(m) = mods {
        if let Some(s) = &m.sides {
            let n = s.len().min(dims);
            sides[..n].copy_from_slice(&s[..n]);
        } else if m.weights.is_some() || m.scale_x.is_some() {
            let w = eff_weights(dims, Some(m));
            ard_sides::<f32>(&w, &mut sides);
        }
        for &u in &m.unscaled {
            if u < dims {
                sides[u] = 1.0;
            }
        }
    }
    let mut lower = vec![0.0_f32; dims];
    let mut upper = vec![1.0_f32; dims];
    region_bounds::<f32>(
        &c_vec,
        plan.region_length() as f32,
        Some(&sides),
        &mut lower,
        &mut upper,
    );
    if let LifecyclePhase::Active {
        sampler: PoolSampler::Gaussian { .. } | PoolSampler::Rademacher { .. },
        ..
    } = plan.phase
    {
        scale.copy_from_slice(&sides);
    }
    (c_vec, scale, lower, upper)
}

pub fn local_radii(rows: &[f32], count: usize, dims: usize, weights: &[f32], k: usize) -> Vec<f32> {
    if count == 0 || dims == 0 {
        return Vec::new();
    }
    let k_eff = k.clamp(1, count.saturating_sub(1).max(1)).min(KNN_K);
    let mut out = vec![1.0_f32; count];
    for i in 0..count {
        let mut tk = TopK::<f32, KNN_K>::new(k_eff);
        let ri = &rows[i * dims..(i + 1) * dims];
        for j in 0..count {
            if i == j {
                continue;
            }
            let rj = &rows[j * dims..(j + 1) * dims];
            tk.push(ard_dist::<f32>(ri, rj, weights), j as u32);
        }
        if let Some(&(d2, _)) = tk.as_slice().last() {
            out[i] = d2.max(1.0e-12);
        }
    }
    out
}

pub fn pack_mods(
    metrics: usize,
    history: usize,
    default_scale: f32,
    mods: Option<&PoolModifiers>,
    computed_local: Option<&[f32]>,
) -> (Vec<f32>, u32, bool) {
    let m = metrics.clamp(1, MAX_METRICS);
    let l_base = 9 * m + m * m;
    let total = l_base + history;
    let mut buf = vec![0.0_f32; total];
    buf[l_base..total].fill(1.0_f32);
    let mut flags = 0_u32;
    let mut has_yb = false;

    for idx in 0..m {
        buf[idx] = default_scale;
        buf[m + 2 * idx] = f32::NEG_INFINITY;
        buf[m + 2 * idx + 1] = f32::INFINITY;
        buf[4 * m + idx] = 1.0;
        buf[5 * m + idx] = 1.0;
        buf[8 * m + idx] = 1.0 / (m as f32);
    }

    if let Some(md) = mods {
        if md.correlated {
            flags |= MOD_CORR;
        }
        if let Some(sc) = &md.scales {
            for (idx, &v) in sc.iter().enumerate().take(m) {
                buf[idx] = v;
            }
        }
        if let Some(yb) = &md.ybounds
            && !baux::semantics::ybounds::is_identity(yb)
        {
            has_yb = true;
            for (idx, &[lo, hi]) in yb.iter().enumerate().take(m) {
                buf[m + 2 * idx] = lo;
                buf[m + 2 * idx + 1] = hi;
            }
        }
        if let Some(aff) = &md.affine {
            flags |= MOD_AFFINE;
            for idx in 0..m {
                buf[3 * m + idx] = aff.a.get(idx).copied().unwrap_or(0.0) as f32;
                buf[4 * m + idx] = aff.b.get(idx).copied().unwrap_or(1.0) as f32;
                buf[5 * m + idx] = aff.c.get(idx).copied().unwrap_or(1.0) as f32;
            }
        }
        if let Some(mb) = &md.morbo
            && m > 1
        {
            flags |= MOD_MORBO;
            for idx in 0..m {
                buf[6 * m + idx] = mb.y_min.get(idx).copied().unwrap_or(0.0);
                buf[7 * m + idx] = mb.y_max.get(idx).copied().unwrap_or(1.0);
                buf[8 * m + idx] = mb.weights.get(idx).copied().unwrap_or(1.0 / (m as f32));
            }
        }
        if let Some(st) = &md.stiefel
            && m > 1
        {
            let proj = st.projector();
            if proj.len() >= m * m {
                flags |= MOD_STIEFEL;
                let s_off = 9 * m;
                for idx in 0..m * m {
                    buf[s_off + idx] = proj[idx] as f32;
                }
            }
        }
        if let Some(ls) = &md.local_scales {
            let l_off = 9 * m + m * m;
            for (idx, &v) in ls.iter().enumerate().take(history) {
                buf[l_off + idx] = v;
            }
        }
    }
    if let Some(ls) = computed_local {
        let l_off = 9 * m + m * m;
        for (idx, &v) in ls.iter().enumerate().take(history) {
            buf[l_off + idx] = v;
        }
    }
    (buf, flags, has_yb)
}

pub fn snap_rows(rows: &mut [f32], arms: usize, dims: usize, mods: Option<&PoolModifiers>) {
    let Some(m) = mods else {
        return;
    };
    if dims == 0 {
        return;
    }
    let w_f64: Option<Vec<f64>> = m
        .weights
        .as_ref()
        .map(|w| w.iter().map(|&v| f64::from(v)).collect());
    for arm in 0..arms {
        let off = arm * dims;
        if off + dims > rows.len() {
            break;
        }
        let row = &mut rows[off..off + dims];
        for &(dst, src) in &m.tied {
            if dst < dims && src < dims {
                row[dst] = row[src];
            }
        }
        for cat in &m.cats {
            if cat.card == 0 || cat.width == 0 || cat.offset + cat.width > dims {
                continue;
            }
            let sub: Vec<f64> = row[cat.offset..cat.offset + cat.width]
                .iter()
                .map(|&v| f64::from(v))
                .collect();
            let w_sub = w_f64.as_ref().and_then(|w| {
                (cat.offset + cat.width <= w.len()).then(|| &w[cat.offset..cat.offset + cat.width])
            });
            let win = snap_row(&sub, &cat.codes, w_sub, cat.card, cat.width);
            for bit in 0..cat.width {
                row[cat.offset + bit] = cat.codes[win * cat.width + bit] as f32;
            }
        }
    }
}

pub fn transpose_cols(row_major: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut out = vec![0.0_f32; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            if r * cols + c < row_major.len() {
                out[c * rows + r] = row_major[r * cols + c];
            }
        }
    }
    out
}

#[derive(Clone, Debug)]
pub struct ProposalHost {
    pub rows: Vec<f32>,
    pub decisions: Vec<ArmDecision>,
    pub selected: Vec<u32>,
    pub num_arms: usize,
    pub num_dim: usize,
    pub kind: &'static str,
}

pub struct FencedProposal {
    rows: DeviceBuffer<f32>,
    decisions: DeviceBuffer<ArmDecision>,
    selected: DeviceBuffer<u32>,
    fence: CudaEvent,
    stream: Arc<CudaStream>,
    _staging: PinnedHostBuffer<f32>,
    _dirs: PinnedHostBuffer<u32>,
    synced: AtomicBool,
    pub num_arms: usize,
    pub num_dim: usize,
    pub kind: &'static str,
}

impl FencedProposal {
    pub fn fence(&self) -> &CudaEvent {
        &self.fence
    }

    pub fn stream(&self) -> &Arc<CudaStream> {
        &self.stream
    }

    pub fn rows_buf(&self) -> &DeviceBuffer<f32> {
        &self.rows
    }

    pub fn arms_buf(&self) -> &DeviceBuffer<ArmDecision> {
        &self.decisions
    }

    pub fn ids_buf(&self) -> &DeviceBuffer<u32> {
        &self.selected
    }

    pub fn rows_ptr(&self) -> u64 {
        self.rows.cu_deviceptr()
    }

    pub fn arms_ptr(&self) -> u64 {
        self.decisions.cu_deviceptr()
    }

    pub fn synchronize(&self) -> CudaResult<()> {
        if !self.synced.load(Ordering::Acquire) {
            self.fence.synchronize().map_err(cuda_error)?;
            self.synced.store(true, Ordering::Release);
        }
        Ok(())
    }

    pub fn materialize(&self, mods: Option<&PoolModifiers>) -> CudaResult<ProposalHost> {
        self.synchronize()?;
        let row_len = self
            .num_arms
            .checked_mul(self.num_dim)
            .ok_or("CUDA proposal row size overflow")?;
        let mut rows = read_prefix(&self.rows, &self.stream, row_len)?;
        let decisions = read_prefix(&self.decisions, &self.stream, self.num_arms)?;
        let selected = read_prefix(&self.selected, &self.stream, self.num_arms)?;
        snap_rows(&mut rows, self.num_arms, self.num_dim, mods);
        Ok(ProposalHost {
            rows,
            decisions,
            selected,
            num_arms: self.num_arms,
            num_dim: self.num_dim,
            kind: self.kind,
        })
    }
}

impl Drop for FencedProposal {
    fn drop(&mut self) {
        if !self.synced.load(Ordering::Acquire) {
            let _ = self.fence.synchronize();
        }
    }
}

pub struct PoolScratch {
    out_cap: usize,
    var_cap: usize,
    mod_cap: usize,
    dim_cap: usize,
    dir_cap: usize,
    score_cap: usize,
    mean_cap: usize,
    err_cap: usize,
    state_cap: usize,
    pub(super) outcomes: DeviceBuffer<f32>,
    pub(super) variances: DeviceBuffer<f32>,
    pub(super) mods: DeviceBuffer<f32>,
    pub(super) weights: DeviceBuffer<f32>,
    pub(super) center: DeviceBuffer<f32>,
    pub(super) scale: DeviceBuffer<f32>,
    pub(super) lower: DeviceBuffer<f32>,
    pub(super) upper: DeviceBuffer<f32>,
    pub(super) dirs: DeviceBuffer<u32>,
    pub(super) scores: DeviceBuffer<f32>,
    pub(super) means: DeviceBuffer<f32>,
    pub(super) errors: DeviceBuffer<f32>,
    pub(super) state: DeviceBuffer<u32>,
}

impl PoolScratch {
    pub(super) fn new(stream: &CudaStream) -> CudaResult<Self> {
        Ok(Self {
            out_cap: 1,
            var_cap: 1,
            mod_cap: 1,
            dim_cap: 1,
            dir_cap: 1,
            score_cap: 1,
            mean_cap: 1,
            err_cap: 1,
            state_cap: 1,
            outcomes: DeviceBuffer::zeroed(stream, 1).map_err(cuda_error)?,
            variances: DeviceBuffer::zeroed(stream, 1).map_err(cuda_error)?,
            mods: DeviceBuffer::zeroed(stream, 1).map_err(cuda_error)?,
            weights: DeviceBuffer::zeroed(stream, 1).map_err(cuda_error)?,
            center: DeviceBuffer::zeroed(stream, 1).map_err(cuda_error)?,
            scale: DeviceBuffer::zeroed(stream, 1).map_err(cuda_error)?,
            lower: DeviceBuffer::zeroed(stream, 1).map_err(cuda_error)?,
            upper: DeviceBuffer::zeroed(stream, 1).map_err(cuda_error)?,
            dirs: DeviceBuffer::zeroed(stream, 1).map_err(cuda_error)?,
            scores: DeviceBuffer::zeroed(stream, 1).map_err(cuda_error)?,
            means: DeviceBuffer::zeroed(stream, 1).map_err(cuda_error)?,
            errors: DeviceBuffer::zeroed(stream, 1).map_err(cuda_error)?,
            state: DeviceBuffer::zeroed(stream, 1).map_err(cuda_error)?,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn ensure(
        &mut self,
        stream: &CudaStream,
        outcomes: usize,
        variances: usize,
        mods: usize,
        dims: usize,
        dirs: usize,
        scores: usize,
        means: usize,
        cands: usize,
    ) -> CudaResult<()> {
        let out_cap = next_capacity(outcomes)?;
        let var_cap = next_capacity(variances)?;
        let mod_cap = next_capacity(mods)?;
        let dim_cap = next_capacity(dims)?;
        let dir_cap = next_capacity(dirs)?;
        let score_cap = next_capacity(scores)?;
        let mean_cap = next_capacity(means)?;
        let err_cap = next_capacity(cands.saturating_mul(4))?;
        let state_cap = next_capacity(cands.saturating_mul(2))?;
        if out_cap > self.out_cap {
            self.outcomes = DeviceBuffer::zeroed(stream, out_cap).map_err(cuda_error)?;
            self.out_cap = out_cap;
        }
        if var_cap > self.var_cap {
            self.variances = DeviceBuffer::zeroed(stream, var_cap).map_err(cuda_error)?;
            self.var_cap = var_cap;
        }
        if mod_cap > self.mod_cap {
            self.mods = DeviceBuffer::zeroed(stream, mod_cap).map_err(cuda_error)?;
            self.mod_cap = mod_cap;
        }
        if dim_cap > self.dim_cap {
            self.weights = DeviceBuffer::zeroed(stream, dim_cap).map_err(cuda_error)?;
            self.center = DeviceBuffer::zeroed(stream, dim_cap).map_err(cuda_error)?;
            self.scale = DeviceBuffer::zeroed(stream, dim_cap).map_err(cuda_error)?;
            self.lower = DeviceBuffer::zeroed(stream, dim_cap).map_err(cuda_error)?;
            self.upper = DeviceBuffer::zeroed(stream, dim_cap).map_err(cuda_error)?;
            self.dim_cap = dim_cap;
        }
        if dir_cap > self.dir_cap {
            self.dirs = DeviceBuffer::zeroed(stream, dir_cap).map_err(cuda_error)?;
            self.dir_cap = dir_cap;
        }
        if score_cap > self.score_cap {
            self.scores = DeviceBuffer::zeroed(stream, score_cap).map_err(cuda_error)?;
            self.score_cap = score_cap;
        }
        if mean_cap > self.mean_cap {
            self.means = DeviceBuffer::zeroed(stream, mean_cap).map_err(cuda_error)?;
            self.mean_cap = mean_cap;
        }
        if err_cap > self.err_cap {
            self.errors = DeviceBuffer::zeroed(stream, err_cap).map_err(cuda_error)?;
            self.err_cap = err_cap;
        }
        if state_cap > self.state_cap {
            self.state = DeviceBuffer::zeroed(stream, state_cap).map_err(cuda_error)?;
            self.state_cap = state_cap;
        }
        Ok(())
    }
}

fn prepare_headers(
    plan: &RoundPlan,
    row_count: usize,
    metrics: usize,
    incumbent: usize,
    has_yvar: bool,
    mods: Option<&PoolModifiers>,
    computed_local: Option<&[f32]>,
) -> CudaResult<(PoolKind, u32, usize, PoolDispatch, SelectHeader, Vec<f32>)> {
    let raw_neighbors = match plan.phase {
        LifecyclePhase::Active {
            surrogate: SurrogateMode::Enn { neighbors, .. },
            ..
        } if row_count > 0 => neighbors.min(row_count).min(KNN_K),
        _ => 0,
    };
    let problem = PoolProblem {
        rows: row_count,
        dims: plan.num_dim,
        candidates: plan.num_candidates,
        arms: plan.num_arms,
        neighbors: raw_neighbors,
    };
    let (kind, tile_rows) = select_pool(&problem)?;

    let (metric, scale) = match plan.phase {
        LifecyclePhase::Active {
            surrogate: SurrogateMode::Enn { metric, scale, .. },
            ..
        } => (metric, scale),
        _ => (MetricKind::Euclidean, 1.0),
    };
    let scale_k = mods
        .map(|m| m.scale_k)
        .filter(|&k| k > 0)
        .unwrap_or(raw_neighbors.max(1))
        .min(row_count.max(1))
        .min(KNN_K);
    let scan_k = if raw_neighbors == 0 {
        0
    } else if metric == MetricKind::Euclidean {
        raw_neighbors
    } else {
        row_count.min(KNN_K).max(raw_neighbors).max(scale_k)
    };

    let lists = if kind == PoolKind::Direct || tile_rows == 0 {
        1
    } else {
        row_count.div_ceil(tile_rows as usize).max(1)
    };
    let mut dispatch = plan.pool_dispatch(
        to_u32(lists, "pool tile lists")?,
        to_u32(row_count, "pool history")?,
        0,
    );
    dispatch.blocks = to_u32(scan_k, "pool scan neighbors")?;

    let (mod_buf, mod_flags, has_yb) = pack_mods(metrics, row_count, scale, mods, computed_local);
    let mut header = plan.select_header(
        to_u32(row_count, "select history")?,
        to_u32(incumbent.min(row_count.saturating_sub(1)), "incumbent")?,
        to_u32(scan_k.max(1), "scan_k")?,
        has_yb,
    );
    header.neighbors = to_u32(raw_neighbors, "used_k")?;
    header.num_metrics = to_u32(metrics.clamp(1, MAX_METRICS), "num_metrics")?;
    header.has_yvar = u32::from(has_yvar);
    header.mode = (header.mode & 0xff) | mod_flags;
    header.scale_neighbors = to_u32(scale_k, "scale_k")?;
    if let Some(m) = mods {
        header.adaptive_k = u32::from(m.adaptive_k);
        header.axis_base = m.inc_mean;
        header.axis_weight = m.inc_se;
        if let Some(mb) = &m.morbo {
            header.morbo_alpha = mb.alpha;
        }
    }
    Ok((kind, tile_rows, scan_k, dispatch, header, mod_buf))
}

unsafe fn stage_slice(
    dst: u64,
    staging: &PinnedHostBuffer<f32>,
    offset: usize,
    len: usize,
    stream: &CudaStream,
) -> CudaResult<()> {
    if len == 0 {
        return Ok(());
    }
    unsafe {
        cuda_core::simt::memory::memcpy_htod_async(
            dst,
            staging.as_slice()[offset..offset + len].as_ptr(),
            len * size_of::<f32>(),
            stream.cu_stream(),
        )
        .map_err(cuda_error)
    }
}

fn auto_scales(
    plan: &RoundPlan,
    host_rows: &[f32],
    row_count: usize,
    weights: &[f32],
    mods: Option<&PoolModifiers>,
) -> Option<Vec<f32>> {
    let LifecyclePhase::Active {
        surrogate: SurrogateMode::Enn {
            neighbors, metric, ..
        },
        ..
    } = plan.phase
    else {
        return None;
    };
    if row_count == 0
        || metric == MetricKind::Euclidean
        || mods.and_then(|m| m.local_scales.as_ref()).is_some()
    {
        return None;
    }
    let raw_k = neighbors.min(row_count).min(KNN_K).max(1);
    let sk = mods.map(|m| m.scale_k).filter(|&k| k > 0).unwrap_or(raw_k);
    Some(local_radii(host_rows, row_count, plan.num_dim, weights, sk))
}

impl CudaIndex {
    #[allow(clippy::too_many_arguments)]
    pub fn propose_fenced(
        &self,
        scratch: &mut CudaScratch,
        plan: &RoundPlan,
        center: &[f32],
        outcomes: &[f32],
        variances: Option<&[f32]>,
        metrics: usize,
        incumbent: usize,
        mods: Option<&PoolModifiers>,
    ) -> CudaResult<FencedProposal> {
        if plan.num_dim != self.dims {
            return Err("CUDA proposal plan dimension does not match index".to_string());
        }
        let m_cnt = metrics.clamp(1, MAX_METRICS);
        if self.row_count > 0 && outcomes.len() < self.row_count * m_cnt {
            return Err(
                "CUDA proposal outcomes shape is smaller than history * metrics".to_string(),
            );
        }
        if let Some(v) = variances
            && v.len() < self.row_count * m_cnt
        {
            return Err(
                "CUDA proposal variances shape is smaller than history * metrics".to_string(),
            );
        }

        let weights = eff_weights(self.dims, mods);
        let (c_vec, s_vec, lo_vec, hi_vec) = region_box(plan, center, mods);
        let auto_local = auto_scales(plan, &self.host_rows, self.row_count, &weights, mods);
        let (kind, tile_rows, scan_k, dispatch, header, mod_buf) = prepare_headers(
            plan,
            self.row_count,
            m_cnt,
            incumbent,
            variances.is_some(),
            mods,
            auto_local.as_deref(),
        )?;

        let out_col = transpose_cols(outcomes, self.row_count, m_cnt);
        let var_col = variances
            .map(|v| transpose_cols(v, self.row_count, m_cnt))
            .unwrap_or_default();
        let dir_slice: &[u32] = mods
            .and_then(|m| m.dirs.as_deref())
            .filter(|d| !d.is_empty())
            .unwrap_or(&[0_u32]);

        let cands = plan.num_candidates;
        let arms = plan.num_arms;
        let score_arms = if header.acquisition == baux::semantics::abi::ACQ_THOMPSON {
            arms.max(1)
        } else {
            1
        };
        let nds_mode = header.acquisition == baux::semantics::abi::ACQ_PARETO
            && m_cnt > 1
            && (header.mode & MOD_MORBO) == 0;
        let mean_cols = if nds_mode { m_cnt } else { 1 };

        let stream = &self.runtime.stream;
        stream.context().bind_to_thread().map_err(cuda_error)?;

        scratch.pool.ensure(
            stream,
            out_col.len().max(1),
            var_col.len().max(1),
            mod_buf.len().max(1),
            self.dims,
            dir_slice.len().max(1),
            cands.saturating_mul(score_arms).max(1),
            cands.saturating_mul(mean_cols).max(1),
            cands.max(1),
        )?;

        if kind != PoolKind::Direct {
            let lists = self.row_count.div_ceil(tile_rows as usize).max(1);
            let partials = cands
                .checked_mul(lists)
                .and_then(|v| v.checked_mul(scan_k))
                .ok_or("CUDA pool partial buffer overflow")?;
            scratch.knn.ensure(stream, 1, partials)?;
        }

        let mut host_pack =
            Vec::with_capacity(out_col.len() + var_col.len() + mod_buf.len() + 5 * self.dims);
        let o_off = 0usize;
        host_pack.extend_from_slice(&out_col);
        let v_off = host_pack.len();
        host_pack.extend_from_slice(&var_col);
        let m_off = host_pack.len();
        host_pack.extend_from_slice(&mod_buf);
        let w_off = host_pack.len();
        host_pack.extend_from_slice(&weights);
        let c_off = host_pack.len();
        host_pack.extend_from_slice(&c_vec);
        let s_off = host_pack.len();
        host_pack.extend_from_slice(&s_vec);
        let l_off = host_pack.len();
        host_pack.extend_from_slice(&lo_vec);
        let u_off = host_pack.len();
        host_pack.extend_from_slice(&hi_vec);
        if host_pack.is_empty() {
            host_pack.push(0.0);
        }

        let staging =
            PinnedHostBuffer::from_slice(&self.runtime.context, &host_pack).map_err(cuda_error)?;
        let dirs_pinned =
            PinnedHostBuffer::from_slice(&self.runtime.context, dir_slice).map_err(cuda_error)?;

        unsafe {
            stage_slice(
                scratch.pool.outcomes.cu_deviceptr(),
                &staging,
                o_off,
                out_col.len(),
                stream,
            )?;
            stage_slice(
                scratch.pool.variances.cu_deviceptr(),
                &staging,
                v_off,
                var_col.len(),
                stream,
            )?;
            stage_slice(
                scratch.pool.mods.cu_deviceptr(),
                &staging,
                m_off,
                mod_buf.len(),
                stream,
            )?;
            stage_slice(
                scratch.pool.weights.cu_deviceptr(),
                &staging,
                w_off,
                weights.len(),
                stream,
            )?;
            stage_slice(
                scratch.pool.center.cu_deviceptr(),
                &staging,
                c_off,
                c_vec.len(),
                stream,
            )?;
            stage_slice(
                scratch.pool.scale.cu_deviceptr(),
                &staging,
                s_off,
                s_vec.len(),
                stream,
            )?;
            stage_slice(
                scratch.pool.lower.cu_deviceptr(),
                &staging,
                l_off,
                lo_vec.len(),
                stream,
            )?;
            stage_slice(
                scratch.pool.upper.cu_deviceptr(),
                &staging,
                u_off,
                hi_vec.len(),
                stream,
            )?;
            cuda_core::simt::memory::memcpy_htod_async(
                scratch.pool.dirs.cu_deviceptr(),
                dirs_pinned.as_slice().as_ptr(),
                std::mem::size_of_val(dirs_pinned.as_slice()),
                stream.cu_stream(),
            )
            .map_err(cuda_error)?;
        }

        let mut rows_out = DeviceBuffer::zeroed(stream, arms.saturating_mul(self.dims).max(1))
            .map_err(cuda_error)?;
        let mut decisions_out = DeviceBuffer::zeroed(stream, arms.max(1)).map_err(cuda_error)?;
        let mut selected_out = DeviceBuffer::zeroed(stream, arms.max(1)).map_err(cuda_error)?;

        let plan_name = match kind {
            PoolKind::Direct => "cuda_pool_direct",
            PoolKind::Rows => "cuda_pool_rows",
            PoolKind::Warps => "cuda_pool_warps",
        };

        if kind != PoolKind::Direct {
            let total_lists = dispatch
                .num_candidates
                .checked_mul(dispatch.tiles)
                .ok_or("CUDA pool scan grid overflow")?;
            match kind {
                PoolKind::Rows => {
                    let launch = self
                        .runtime
                        .module
                        .prepare_poolscan(LaunchConfig1D::new(total_lists, THREADS, 0))
                        .map_err(cuda_error)?;
                    self.runtime
                        .module
                        .poolscan(
                            stream,
                            &launch,
                            &self.rows,
                            &scratch.pool.weights,
                            &scratch.pool.center,
                            &scratch.pool.scale,
                            &scratch.pool.lower,
                            &scratch.pool.upper,
                            &scratch.pool.dirs,
                            &mut scratch.knn.distances_a,
                            &mut scratch.knn.indices_a,
                            dispatch,
                        )
                        .map_err(cuda_error)?;
                }
                PoolKind::Warps => {
                    let launch = self
                        .runtime
                        .module
                        .prepare_poolwarp(LaunchConfig1D::new(total_lists, THREADS, 0))
                        .map_err(cuda_error)?;
                    self.runtime
                        .module
                        .poolwarp(
                            stream,
                            &launch,
                            &self.rows,
                            &scratch.pool.weights,
                            &scratch.pool.center,
                            &scratch.pool.scale,
                            &scratch.pool.lower,
                            &scratch.pool.upper,
                            &scratch.pool.dirs,
                            &mut scratch.knn.distances_a,
                            &mut scratch.knn.indices_a,
                            dispatch,
                        )
                        .map_err(cuda_error)?;
                }
                PoolKind::Direct => {}
            }

            let input_a = self.launch_merge(scratch, cands, dispatch.tiles as usize, scan_k)?;
            let score_blocks = to_u32(cands, "pool candidates")?.div_ceil(SCORE_THREADS);
            let score_launch = self
                .runtime
                .module
                .prepare_poolscore(LaunchConfig1D::new(score_blocks, SCORE_THREADS, 0))
                .map_err(cuda_error)?;
            let (dist_buf, idx_buf) = if input_a {
                (&scratch.knn.distances_a, &scratch.knn.indices_a)
            } else {
                (&scratch.knn.distances_b, &scratch.knn.indices_b)
            };
            self.runtime
                .module
                .poolscore(
                    stream,
                    &score_launch,
                    dist_buf,
                    idx_buf,
                    &scratch.pool.outcomes,
                    &scratch.pool.variances,
                    &scratch.pool.mods,
                    &mut scratch.pool.scores,
                    &mut scratch.pool.means,
                    &mut scratch.pool.errors,
                    header,
                )
                .map_err(cuda_error)?;
        }

        let pick_launch = self
            .runtime
            .module
            .prepare_poolpick(LaunchConfig1D::new(1, THREADS, 0))
            .map_err(cuda_error)?;
        self.runtime
            .module
            .poolpick(
                stream,
                &pick_launch,
                &scratch.pool.scores,
                &scratch.pool.means,
                &scratch.pool.errors,
                &mut scratch.pool.state,
                &mut selected_out,
                &mut decisions_out,
                header,
            )
            .map_err(cuda_error)?;

        let row_elems = dispatch
            .num_arms
            .checked_mul(dispatch.num_dim)
            .ok_or("CUDA winning row element overflow")?;
        let row_blocks = row_elems.div_ceil(THREADS).max(1);
        let row_launch = self
            .runtime
            .module
            .prepare_poolrows(LaunchConfig1D::new(row_blocks, THREADS, 0))
            .map_err(cuda_error)?;
        self.runtime
            .module
            .poolrows(
                stream,
                &row_launch,
                &selected_out,
                &scratch.pool.center,
                &scratch.pool.scale,
                &scratch.pool.lower,
                &scratch.pool.upper,
                &scratch.pool.dirs,
                &mut rows_out,
                dispatch,
            )
            .map_err(cuda_error)?;

        let fence = timing_event(stream)?;
        Ok(FencedProposal {
            rows: rows_out,
            decisions: decisions_out,
            selected: selected_out,
            fence,
            stream: Arc::clone(stream),
            _staging: staging,
            _dirs: dirs_pinned,
            synced: AtomicBool::new(false),
            num_arms: arms,
            num_dim: self.dims,
            kind: plan_name,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn propose_pool(
        &self,
        scratch: &mut CudaScratch,
        plan: &RoundPlan,
        center: &[f32],
        outcomes: &[f32],
        variances: Option<&[f32]>,
        metrics: usize,
        incumbent: usize,
        mods: Option<&PoolModifiers>,
    ) -> CudaResult<ProposalHost> {
        let fenced = self.propose_fenced(
            scratch, plan, center, outcomes, variances, metrics, incumbent, mods,
        )?;
        fenced.materialize(mods)
    }
}

#[allow(clippy::too_many_arguments)]
pub fn simulate_round(
    history_rows: &[f32],
    row_count: usize,
    plan: &RoundPlan,
    center: &[f32],
    outcomes: &[f32],
    variances: Option<&[f32]>,
    metrics: usize,
    incumbent: usize,
    mods: Option<&PoolModifiers>,
) -> CudaResult<ProposalHost> {
    let dims = plan.num_dim;
    let m_cnt = metrics.clamp(1, MAX_METRICS);
    if row_count > 0 && history_rows.len() < row_count * dims {
        return Err("history rows smaller than row_count * dims".to_string());
    }
    if row_count > 0 && outcomes.len() < row_count * m_cnt {
        return Err("outcomes smaller than row_count * metrics".to_string());
    }
    let weights = eff_weights(dims, mods);
    let (c_vec, s_vec, lo_vec, hi_vec) = region_box(plan, center, mods);
    let auto_local = auto_scales(plan, history_rows, row_count, &weights, mods);

    let (kind, _, scan_k, dispatch, header, mod_buf) = prepare_headers(
        plan,
        row_count,
        m_cnt,
        incumbent,
        variances.is_some(),
        mods,
        auto_local.as_deref(),
    )?;

    let out_col = transpose_cols(outcomes, row_count, m_cnt);
    let var_col = variances
        .map(|v| transpose_cols(v, row_count, m_cnt))
        .unwrap_or_default();
    let dir_slice: &[u32] = mods
        .and_then(|m| m.dirs.as_deref())
        .filter(|d| !d.is_empty())
        .unwrap_or(&[]);
    let pool = make_pool(
        &dispatch,
        &c_vec,
        &s_vec,
        &lo_vec,
        &hi_vec,
        dir_slice.as_chunks::<32>().0,
    );

    let cands = plan.num_candidates;
    let arms = plan.num_arms.min(cands);
    let score_arms = if header.acquisition == baux::semantics::abi::ACQ_THOMPSON {
        arms.max(1)
    } else {
        1
    };
    let nds_mode = header.acquisition == baux::semantics::abi::ACQ_PARETO
        && m_cnt > 1
        && (header.mode & MOD_MORBO) == 0;
    let mean_cols = if nds_mode { m_cnt } else { 1 };

    let mut scores = vec![0.0_f32; cands * score_arms];
    let mut means = vec![0.0_f32; cands * mean_cols];
    let mut errors = vec![0.0_f32; cands * 4];

    if kind != PoolKind::Direct {
        let mut cand_row = vec![0.0_f32; dims];
        let mut arm_buf = vec![0.0_f32; score_arms];
        let mut mean_buf = vec![0.0_f32; mean_cols];
        for cand in 0..cands {
            pool.write_row(cand, &mut cand_row);
            let mut tk = TopK::<f32, KNN_K>::new(scan_k);
            for r in 0..row_count {
                let h_row = &history_rows[r * dims..(r + 1) * dims];
                tk.push(ard_dist::<f32>(&cand_row, h_row, &weights), r as u32);
            }
            let raw_d: Vec<f32> = tk.as_slice().iter().map(|&(d, _)| d).collect();
            let raw_i: Vec<u32> = tk.as_slice().iter().map(|&(_, i)| i).collect();
            let (tot, epi, ale, count) = score_cand(
                cand,
                &raw_d,
                &raw_i,
                &out_col,
                &var_col,
                &mod_buf,
                &header,
                &mut arm_buf,
                &mut mean_buf,
            );
            for arm in 0..score_arms {
                scores[arm * cands + cand] = arm_buf[arm];
            }
            means[cand * mean_cols..(cand + 1) * mean_cols].copy_from_slice(&mean_buf);
            errors[cand] = tot;
            errors[cands + cand] = epi;
            errors[2 * cands + cand] = ale;
            errors[3 * cands + cand] = count as f32;
        }
    }

    let mut state = vec![0_u32; 2 * cands];
    let mut selected = vec![0_u32; arms];
    let mut decisions = vec![ArmDecision::default(); arms];
    pick_host(
        &scores,
        &means,
        &errors,
        &mut state,
        &mut selected,
        &mut decisions,
        &header,
    );

    let mut rows = vec![0.0_f32; arms * dims];
    for (arm, &win) in selected.iter().enumerate() {
        pool.write_row(win as usize, &mut rows[arm * dims..(arm + 1) * dims]);
    }
    snap_rows(&mut rows, arms, dims, mods);

    let plan_name = match kind {
        PoolKind::Direct => "cuda_pool_direct",
        PoolKind::Rows => "cuda_pool_rows",
        PoolKind::Warps => "cuda_pool_warps",
    };
    Ok(ProposalHost {
        rows,
        decisions,
        selected,
        num_arms: arms,
        num_dim: dims,
        kind: plan_name,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use baux::semantics::abi::DEFAULT_BETA;
    use baux::semantics::categorical::CatSpec;
    use baux::semantics::key::Key;
    use baux::semantics::neighbor::query::posterior_eval;
    use baux::semantics::neighbor::{DistanceMetric, UncertaintyLaw};
    use baux::semantics::optimizer::surrogate::{
        AcquisitionRule, score_ucb, select_pareto, select_random, select_topk,
    };
    use baux::semantics::optimizer::trust_region::{TurboState, WarmupKind};

    #[test]
    fn family_select() {
        let (k0, _) = select_pool(&PoolProblem {
            rows: 0,
            dims: 600,
            candidates: 512,
            arms: 8,
            neighbors: 0,
        })
        .unwrap();
        assert_eq!(k0, PoolKind::Direct);

        let (k1, t1) = select_pool(&PoolProblem {
            rows: 64,
            dims: 16,
            candidates: 512,
            arms: 4,
            neighbors: 16,
        })
        .unwrap();
        assert_eq!((k1, t1), (PoolKind::Rows, KNN_THREADS));

        let (k2, t2) = select_pool(&PoolProblem {
            rows: 64,
            dims: 600,
            candidates: 2048,
            arms: 8,
            neighbors: 32,
        })
        .unwrap();
        assert_eq!((k2, t2), (PoolKind::Warps, KNN_WARPS));

        assert!(
            select_pool(&PoolProblem {
                rows: 64,
                dims: 16,
                candidates: 4,
                arms: 8,
                neighbors: 8,
            })
            .is_err()
        );
    }

    #[test]
    fn large_pools() {
        let dims = 40usize;
        let hist = 36usize;
        let rows: Vec<f32> = (0..hist * dims)
            .map(|i| ((i * 29 + 11) % 97) as f32 / 97.0)
            .collect();
        let outcomes: Vec<f32> = (0..hist).map(|r| ((r * 13) % 23) as f32 * 0.1).collect();
        let center = rows[..dims].to_vec();

        for (cands, arms) in [(1usize, 1usize), (4, 2), (64, 4), (512, 8)] {
            let state = TurboState::for_dims(dims, arms, 0.8, 0.01, 1.6, None);
            let surr = SurrogateMode::Enn {
                neighbors: 10,
                epistemic: 0.05,
                aleatoric: 0.03,
                scale: 1.2,
                metric: MetricKind::Euclidean,
                law: UncertaintyLaw::InversePrecision,
                noise_aware: false,
            };
            let plan = RoundPlan::turbo_enn(
                Key::new(700 + cands as u64),
                2,
                dims,
                cands,
                arms,
                state,
                surr,
                AcquisitionRule::default(),
            );
            assert_eq!(
                plan.acquisition,
                AcquisitionRule::Ucb { beta: DEFAULT_BETA }
            );
            let out =
                simulate_round(&rows, hist, &plan, &center, &outcomes, None, 1, 0, None).unwrap();
            assert_eq!(out.kind, "cuda_pool_warps");
            assert_eq!(out.selected.len(), arms);
            assert_eq!(out.rows.len(), arms * dims);

            let (_, _, scan_k, dispatch, header, _) =
                prepare_headers(&plan, hist, 1, 0, false, None, None).unwrap();
            let (_, s_vec, lo_vec, hi_vec) = region_box(&plan, &center, None);
            let pool = make_pool(&dispatch, &center, &s_vec, &lo_vec, &hi_vec, &[]);
            let weights = vec![1.0_f32; dims];
            let mut ref_scores = vec![0.0_f32; cands];
            let mut crow = vec![0.0_f32; dims];
            for c in 0..cands {
                pool.write_row(c, &mut crow);
                let dists: Vec<f32> = (0..hist)
                    .map(|r| ard_dist::<f32>(&crow, &rows[r * dims..(r + 1) * dims], &weights))
                    .collect();
                let post = posterior_eval::<f32>(
                    DistanceMetric::Euclidean,
                    UncertaintyLaw::InversePrecision,
                    &dists,
                    None,
                    1.0,
                    &outcomes,
                    None,
                    scan_k,
                    0.05,
                    0.03,
                    1.2,
                    None,
                    false,
                );
                ref_scores[c] = score_ucb::<f32>(post.mu, post.se_epi, DEFAULT_BETA);
            }
            let mut ref_sel = vec![0usize; arms];
            select_topk::<f32>(&ref_scores, Key::new(header.acq_seed), &mut ref_sel);
            let exp_sel: Vec<u32> = ref_sel.iter().map(|&i| i as u32).collect();
            assert_eq!(out.selected, exp_sel);
            for (arm_i, &win) in out.selected.iter().enumerate() {
                pool.write_row(win as usize, &mut crow);
                assert_eq!(&out.rows[arm_i * dims..(arm_i + 1) * dims], crow.as_slice());
            }
        }
    }

    #[test]
    fn all_samplers() {
        let dims = 12usize;
        let hist = 16usize;
        let rows: Vec<f32> = (0..hist * dims).map(|i| (i % 19) as f32 / 19.0).collect();
        let outcomes: Vec<f32> = (0..hist).map(|r| r as f32 * 0.25).collect();
        let center = vec![0.5_f32; dims];
        let state = TurboState::for_dims(dims, 3, 0.6, 0.01, 1.6, None);
        let surr = SurrogateMode::Enn {
            neighbors: 6,
            epistemic: 0.05,
            aleatoric: 0.05,
            scale: 1.0,
            metric: MetricKind::Euclidean,
            law: UncertaintyLaw::InversePrecision,
            noise_aware: false,
        };

        for sampler in [
            PoolSampler::Raasp { prob: 0.5 },
            PoolSampler::Sobol { base_idx: 4 },
            PoolSampler::SobolMasked {
                base_idx: 8,
                prob: 0.5,
            },
            PoolSampler::Uniform,
            PoolSampler::Gaussian { sigma: 0.15 },
            PoolSampler::Rademacher { step_size: 0.1 },
        ] {
            let plan = RoundPlan {
                key: Key::new(42),
                step: 1,
                num_dim: dims,
                num_candidates: 96,
                num_arms: 3,
                phase: LifecyclePhase::Active {
                    sampler,
                    surrogate: surr,
                },
                acquisition: AcquisitionRule::Pareto { seed: 777 },
                region: RegionPlan::Single { state },
                forced_slot: Some(13),
            };
            let out =
                simulate_round(&rows, hist, &plan, &center, &outcomes, None, 1, 0, None).unwrap();
            assert_eq!(out.selected[0], 13);
            assert_eq!(out.selected.len(), 3);
            assert!(
                out.rows
                    .iter()
                    .all(|v| v.is_finite() && (0.0..=1.0).contains(v))
            );
        }

        for wk in [
            WarmupKind::Lhd,
            WarmupKind::Uniform,
            WarmupKind::Sobol { base_idx: 0 },
        ] {
            let wplan = RoundPlan::warmup_round(Key::new(91), 0, dims, 64, 4, wk);
            let out = simulate_round(&[], 0, &wplan, &center, &[], None, 1, 0, None).unwrap();
            assert_eq!(out.kind, "cuda_pool_direct");
            let mut r_idx = [0usize; 4];
            let acq_seed = match wplan.acquisition {
                AcquisitionRule::Random { seed } => seed,
                _ => unreachable!(),
            };
            select_random(64, Key::new(acq_seed), &mut r_idx);
            let exp: Vec<u32> = r_idx.iter().map(|&i| i as u32).collect();
            assert_eq!(out.selected, exp);
        }
    }

    #[test]
    fn metrics_laws() {
        let dims = 8usize;
        let hist = 20usize;
        let rows: Vec<f32> = (0..hist * dims)
            .map(|i| ((i * 41 + 3) % 89) as f32 / 89.0)
            .collect();
        let outcomes: Vec<f32> = (0..hist).map(|r| (r as f32 * 0.3).sin()).collect();
        let variances: Vec<f32> = (0..hist).map(|r| 0.01 + (r % 5) as f32 * 0.02).collect();
        let center = rows[..dims].to_vec();
        let state = TurboState::for_dims(dims, 2, 0.8, 0.01, 1.6, None);

        for dm in [
            DistanceMetric::Euclidean,
            DistanceMetric::Conformal { local_k: 4 },
            DistanceMetric::Hyperbolic { local_k: 4 },
            DistanceMetric::Wasserstein { local_k: 4 },
        ] {
            for law in [
                UncertaintyLaw::InversePrecision,
                UncertaintyLaw::BiasVariance,
                UncertaintyLaw::Geometric,
            ] {
                let mut mods = PoolModifiers::default();
                mods.scale_k = 4;
                mods.adaptive_k = true;
                let plan = RoundPlan::turbo_enn(
                    Key::new(123),
                    3,
                    dims,
                    72,
                    2,
                    state,
                    SurrogateMode::Enn {
                        neighbors: 8,
                        epistemic: 0.08,
                        aleatoric: 0.02,
                        scale: 1.0,
                        metric: MetricKind::from_metric(dm),
                        law,
                        noise_aware: true,
                    },
                    AcquisitionRule::Pareto { seed: 555 },
                );
                let out = simulate_round(
                    &rows,
                    hist,
                    &plan,
                    &center,
                    &outcomes,
                    Some(&variances),
                    1,
                    0,
                    Some(&mods),
                )
                .unwrap();

                let weights = vec![1.0_f32; dims];
                let ls = local_radii(&rows, hist, dims, &weights, 4);
                let (_, s_vec, lo_vec, hi_vec) = region_box(&plan, &center, Some(&mods));
                let (_, _, _, dispatch, header, _) =
                    prepare_headers(&plan, hist, 1, 0, true, Some(&mods), Some(&ls)).unwrap();
                let pool = make_pool(&dispatch, &center, &s_vec, &lo_vec, &hi_vec, &[]);
                let mut ref_mu = vec![0.0_f32; 72];
                let mut ref_epi = vec![0.0_f32; 72];
                let mut crow = vec![0.0_f32; dims];
                for c in 0..72 {
                    pool.write_row(c, &mut crow);
                    let dists: Vec<f32> = (0..hist)
                        .map(|r| ard_dist::<f32>(&crow, &rows[r * dims..(r + 1) * dims], &weights))
                        .collect();
                    let mut sorted_d = dists.clone();
                    sorted_d.sort_by(f32::total_cmp);
                    let q_scale = sorted_d[3];
                    let post = posterior_eval::<f32>(
                        dm,
                        law,
                        &dists,
                        Some(&ls),
                        q_scale,
                        &outcomes,
                        Some(&variances),
                        8,
                        0.08,
                        0.02,
                        1.0,
                        None,
                        true,
                    );
                    ref_mu[c] = post.mu;
                    ref_epi[c] = post.se_epi;
                }
                let mut ref_sel = [0usize; 2];
                select_pareto::<f32>(&ref_mu, &ref_epi, Key::new(header.acq_seed), &mut ref_sel);
                assert_eq!(out.selected, vec![ref_sel[0] as u32, ref_sel[1] as u32]);
            }
        }
    }

    #[test]
    fn manifolds_parity() {
        let dims = 10usize;
        let hist = 18usize;
        let m_cnt = 3usize;
        let rows: Vec<f32> = (0..hist * dims)
            .map(|i| ((i * 17 + 5) % 61) as f32 / 61.0)
            .collect();
        let outcomes: Vec<f32> = (0..hist * m_cnt)
            .map(|i| -0.5 + ((i * 23) % 41) as f32 / 41.0)
            .collect();
        let center = rows[..dims].to_vec();
        let state = TurboState::for_dims(dims, 4, 0.8, 0.01, 1.6, None);

        let y_f64: Vec<f64> = outcomes.iter().map(|&v| f64::from(v)).collect();
        let stiefel = Stiefel::from_cov(&y_f64, hist, m_cnt, 2).unwrap();
        let affine = Affine {
            a: vec![0.1, -0.2, 0.05],
            b: vec![1.2, 0.9, 1.1],
            c: vec![1.05, 0.95, 1.0],
        };
        let mut mods = PoolModifiers {
            weights: Some(vec![2.0, 0.5, 1.0, 3.0, 0.8, 1.5, 1.0, 1.0, 1.0, 1.0]),
            scale_x: Some(vec![0.5; dims]),
            unscaled: vec![2],
            tied: vec![(5, 0)],
            cats: vec![
                CatSnap::rademacher(
                    6,
                    4,
                    CatSpec {
                        variable: 0,
                        width: 2,
                        seed: 99,
                        sep: 1.0,
                    },
                ),
                CatSnap::one_hot(8, 2),
            ],
            ybounds: Some(vec![[0.0, 2.0], [-5.0, 5.0], [0.1, f32::INFINITY]]),
            stiefel: Some(stiefel),
            affine: Some(affine),
            morbo: Some(MorboSpec {
                mode: RescalarizeMode::OnPropose,
                alpha: 0.05,
                y_min: vec![0.0, -2.0, 0.1],
                y_max: vec![2.0, 2.0, 4.0],
                weights: vec![0.5, 0.3, 0.2],
            }),
            correlated: true,
            ..PoolModifiers::default()
        };

        let plan = RoundPlan {
            key: Key::new(808),
            step: 4,
            num_dim: dims,
            num_candidates: 128,
            num_arms: 4,
            phase: LifecyclePhase::Active {
                sampler: PoolSampler::Raasp { prob: 0.6 },
                surrogate: SurrogateMode::Enn {
                    neighbors: 8,
                    epistemic: 0.05,
                    aleatoric: 0.02,
                    scale: 1.0,
                    metric: MetricKind::Euclidean,
                    law: UncertaintyLaw::InversePrecision,
                    noise_aware: false,
                },
            },
            acquisition: AcquisitionRule::Thompson { seed: 31337 },
            region: RegionPlan::Morbo {
                state,
                num_metrics: m_cnt,
            },
            forced_slot: None,
        };
        let out_corr = simulate_round(
            &rows,
            hist,
            &plan,
            &center,
            &outcomes,
            None,
            m_cnt,
            0,
            Some(&mods),
        )
        .unwrap();
        assert_eq!(out_corr.selected.len(), 4);
        for arm in 0..4 {
            let r = &out_corr.rows[arm * dims..(arm + 1) * dims];
            assert_eq!(r[5], r[0]);
            assert!((r[8] == 1.0 && r[9] == 0.0) || (r[8] == 0.0 && r[9] == 1.0));
            assert!(out_corr.decisions[arm].score.is_finite());
        }

        mods.correlated = false;
        let out_marg = simulate_round(
            &rows,
            hist,
            &plan,
            &center,
            &outcomes,
            None,
            m_cnt,
            0,
            Some(&mods),
        )
        .unwrap();
        assert_eq!(out_marg.selected.len(), 4);
    }

    #[test]
    fn local_defaults() {
        let (buf, flags, has_yb) = pack_mods(2, 16, 1.0, None, None);
        assert_eq!(flags, 0);
        assert!(!has_yb);
        let l_off = 9 * 2 + 4;
        assert_eq!(&buf[l_off..l_off + 16], &[1.0_f32; 16]);

        let dims = 6usize;
        let hist = 14usize;
        let rows: Vec<f32> = (0..hist * dims)
            .map(|i| ((i * 31 + 7) % 73) as f32 / 73.0)
            .collect();
        let outcomes: Vec<f32> = (0..hist).map(|r| (r as f32 * 0.4).cos()).collect();
        let center = rows[..dims].to_vec();
        let state = TurboState::for_dims(dims, 2, 0.8, 0.01, 1.6, None);

        for (dm, mk) in [
            (
                DistanceMetric::Conformal { local_k: 6 },
                MetricKind::Conformal,
            ),
            (
                DistanceMetric::Hyperbolic { local_k: 6 },
                MetricKind::Hyperbolic,
            ),
            (
                DistanceMetric::Wasserstein { local_k: 6 },
                MetricKind::Wasserstein,
            ),
        ] {
            let plan = RoundPlan::turbo_enn(
                Key::new(909),
                1,
                dims,
                48,
                2,
                state,
                SurrogateMode::Enn {
                    neighbors: 6,
                    epistemic: 0.05,
                    aleatoric: 0.02,
                    scale: 1.0,
                    metric: mk,
                    law: UncertaintyLaw::InversePrecision,
                    noise_aware: false,
                },
                AcquisitionRule::default(),
            );
            let out =
                simulate_round(&rows, hist, &plan, &center, &outcomes, None, 1, 0, None).unwrap();
            let weights = vec![1.0_f32; dims];
            let ls = auto_scales(&plan, &rows, hist, &weights, None).unwrap();
            assert_eq!(ls.len(), hist);
            assert!(ls.iter().all(|&v| v > 1.0e-6));

            let (_, s_vec, lo_vec, hi_vec) = region_box(&plan, &center, None);
            let (_, _, _, dispatch, header, _) =
                prepare_headers(&plan, hist, 1, 0, false, None, Some(&ls)).unwrap();
            let pool = make_pool(&dispatch, &center, &s_vec, &lo_vec, &hi_vec, &[]);
            let mut ref_scores = vec![0.0_f32; 48];
            let mut crow = vec![0.0_f32; dims];
            for c in 0..48 {
                pool.write_row(c, &mut crow);
                let dists: Vec<f32> = (0..hist)
                    .map(|r| ard_dist::<f32>(&crow, &rows[r * dims..(r + 1) * dims], &weights))
                    .collect();
                let mut sorted_d = dists.clone();
                sorted_d.sort_by(f32::total_cmp);
                let post = posterior_eval::<f32>(
                    dm,
                    UncertaintyLaw::InversePrecision,
                    &dists,
                    Some(&ls),
                    sorted_d[5],
                    &outcomes,
                    None,
                    6,
                    0.05,
                    0.02,
                    1.0,
                    None,
                    false,
                );
                ref_scores[c] = score_ucb::<f32>(post.mu, post.se_epi, DEFAULT_BETA);
            }
            let mut ref_sel = [0usize; 2];
            select_topk::<f32>(&ref_scores, Key::new(header.acq_seed), &mut ref_sel);
            assert_eq!(out.selected, vec![ref_sel[0] as u32, ref_sel[1] as u32]);
        }
    }
}
