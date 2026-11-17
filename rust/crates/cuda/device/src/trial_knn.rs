use core::cmp::Ordering;

use baux::semantics::Real;
use baux::semantics::abi::{
    ACQ_PARETO, ACQ_RANDOM, ACQ_THOMPSON, ArmDecision, NO_FORCED, SelectHeader,
};
use baux::semantics::affine::map_point;
use baux::semantics::key::Key;
use baux::semantics::morbo::{chebyshev, dominates};
use baux::semantics::neighbor::query::{Posterior, error_law, neighbor_weight, posterior_topk};
use baux::semantics::neighbor::{MetricKind, TopK, UncertaintyLaw, eval_metric};
use baux::semantics::optimizer::surrogate::{BetterScore, pareto_dom, score_thompson, score_ucb};
use baux::semantics::optimizer::trust_region::arm_seed;
use baux::semantics::random::normal_metric;
use baux::semantics::stiefel::contract_row;
use baux::semantics::ybounds::natural_point;
use cuda_device::{DisjointSlice, SharedArray, thread};

use crate::THREADS;
use crate::knn::{
    BatchParams, BatchValue, DrawParams, KNN_K, PosteriorParams, WeightedParams, block_sum,
};

pub const SCORE_THREADS: u32 = 32;
pub const MAX_METRICS: usize = 32;
pub const MOD_STIEFEL: u32 = 1 << 8;
pub const MOD_AFFINE: u32 = 1 << 9;
pub const MOD_MORBO: u32 = 1 << 10;
pub const MOD_CORR: u32 = 1 << 11;

const TAG_PARETO: u64 = 0x5041_5245_544f_524b;

#[derive(Clone, Copy)]
pub struct ModSlices<'a> {
    pub scales: &'a [f32],
    pub ybounds: &'a [f32],
    pub aff_a: &'a [f32],
    pub aff_b: &'a [f32],
    pub aff_c: &'a [f32],
    pub y_min: &'a [f32],
    pub y_max: &'a [f32],
    pub morbo_w: &'a [f32],
    pub stiefel: &'a [f32],
    pub local_scales: &'a [f32],
}

#[inline(always)]
pub fn unpack_mods(mods: &[f32], metrics: usize, history: usize) -> ModSlices<'_> {
    let m = metrics.max(1);
    let take = |start: usize, len: usize| -> &[f32] {
        if start < mods.len() {
            let end = (start + len).min(mods.len());
            &mods[start..end]
        } else {
            &[]
        }
    };
    let s_off = 9 * m;
    let l_off = s_off + m * m;
    ModSlices {
        scales: take(0, m),
        ybounds: take(m, 2 * m),
        aff_a: take(3 * m, m),
        aff_b: take(4 * m, m),
        aff_c: take(5 * m, m),
        y_min: take(6 * m, m),
        y_max: take(7 * m, m),
        morbo_w: take(8 * m, m),
        stiefel: take(s_off, m * m),
        local_scales: take(l_off, history),
    }
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
pub fn eval_law(
    law_code: u32,
    denom: f32,
    inv_denom: f32,
    weighted_noise: f32,
    sq_noise: f32,
    dist_var: f32,
    epistemic: f32,
    scale: f32,
) -> (f32, f32, f32) {
    match UncertaintyLaw::from_code(law_code) {
        UncertaintyLaw::InversePrecision => error_law::<{ UncertaintyLaw::InversePrecision }, f32>(
            denom,
            inv_denom,
            weighted_noise,
            sq_noise,
            dist_var,
            epistemic,
            scale,
        ),
        UncertaintyLaw::BiasVariance => error_law::<{ UncertaintyLaw::BiasVariance }, f32>(
            denom,
            inv_denom,
            weighted_noise,
            sq_noise,
            dist_var,
            epistemic,
            scale,
        ),
        UncertaintyLaw::Geometric => error_law::<{ UncertaintyLaw::Geometric }, f32>(
            denom,
            inv_denom,
            weighted_noise,
            sq_noise,
            dist_var,
            epistemic,
            scale,
        ),
    }
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
pub fn eval_topk(
    law_code: u32,
    topk: &[(f32, u32)],
    outcomes: &[f32],
    variances: Option<&[f32]>,
    epistemic: f32,
    aleatoric: f32,
    scale: f32,
    adaptive: bool,
) -> Posterior<f32> {
    match UncertaintyLaw::from_code(law_code) {
        UncertaintyLaw::InversePrecision => {
            posterior_topk::<{ UncertaintyLaw::InversePrecision }, f32>(
                topk, outcomes, variances, epistemic, aleatoric, scale, adaptive,
            )
        }
        UncertaintyLaw::BiasVariance => posterior_topk::<{ UncertaintyLaw::BiasVariance }, f32>(
            topk, outcomes, variances, epistemic, aleatoric, scale, adaptive,
        ),
        UncertaintyLaw::Geometric => posterior_topk::<{ UncertaintyLaw::Geometric }, f32>(
            topk, outcomes, variances, epistemic, aleatoric, scale, adaptive,
        ),
    }
}

#[inline(always)]
pub fn metric_pair(code: u32, sq_dist: f32, scale_a: f32, scale_b: f32) -> f32 {
    match MetricKind::from_code(code) {
        MetricKind::Euclidean => {
            eval_metric::<{ MetricKind::Euclidean }, f32>(sq_dist, scale_a, scale_b)
        }
        MetricKind::Conformal => {
            eval_metric::<{ MetricKind::Conformal }, f32>(sq_dist, scale_a, scale_b)
        }
        MetricKind::Hyperbolic => {
            eval_metric::<{ MetricKind::Hyperbolic }, f32>(sq_dist, scale_a, scale_b)
        }
        MetricKind::Wasserstein => {
            eval_metric::<{ MetricKind::Wasserstein }, f32>(sq_dist, scale_a, scale_b)
        }
    }
}

#[inline(always)]
pub(crate) fn run_postlight(
    distances: &[f32],
    indices: &[u32],
    outcomes: &[f32],
    scales: &[f32],
    means: &mut DisjointSlice<f32>,
    errors: &mut DisjointSlice<f32>,
    selected: &mut DisjointSlice<u32>,
    params: PosteriorParams,
) {
    static mut WEIGHTS: SharedArray<f32, KNN_K> = SharedArray::UNINIT;
    static mut SUMS: SharedArray<f32, { THREADS as usize }> = SharedArray::UNINIT;

    let query = thread::blockIdx_x();
    if query >= params.queries {
        return;
    }
    let tid = thread::threadIdx_x();
    let weights = unsafe { SharedArray::as_raw_mut_ptr(&raw mut WEIGHTS) };
    let sums = unsafe { SharedArray::as_raw_mut_ptr(&raw mut SUMS) };
    let eps_shift = (params.epsilon - f32::PREC_EPS).max(0.0);
    let mut weight = 0.0_f32;
    if tid < params.used_k {
        let source = query as usize * params.input_k as usize + params.skip as usize + tid as usize;
        weight = neighbor_weight::<f32>(
            distances[source],
            params.epistemic_scale,
            params.aleatoric_scale + eps_shift,
            0.0,
        );
        unsafe {
            weights.add(tid as usize).write(weight);
            *selected.get_unchecked_mut(query as usize * params.used_k as usize + tid as usize) =
                indices[source];
        }
    }
    unsafe { sums.add(tid as usize).write(weight) };
    let denom = unsafe { block_sum(sums) }.max(params.epsilon.max(f32::NOISE_FLOOR));
    let inverse = 1.0 / denom;
    let mut metric = tid;
    while metric < params.metrics {
        let mut weighted = 0.0_f32;
        let mut neighbor = 0_u32;
        while neighbor < params.used_k {
            let source =
                query as usize * params.input_k as usize + params.skip as usize + neighbor as usize;
            let row = indices[source] as usize;
            weighted += unsafe { weights.add(neighbor as usize).read() }
                * outcomes[row * params.metrics as usize + metric as usize];
            neighbor += 1;
        }
        let (_, epi, _) = eval_law(
            0,
            denom,
            inverse.max(params.epsilon),
            0.0,
            0.0,
            0.0,
            params.epistemic_scale,
            scales[metric as usize],
        );
        let output = query as usize * params.metrics as usize + metric as usize;
        unsafe {
            *means.get_unchecked_mut(output) = weighted * inverse;
            *errors.get_unchecked_mut(output) = epi;
        }
        metric += THREADS;
    }
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn full_neighbor(
    distances: &[f32],
    indices: &[u32],
    variances: &[f32],
    selected: &mut DisjointSlice<u32>,
    query: u32,
    metric: u32,
    tid: u32,
    scale_sq: f32,
    params: WeightedParams,
) -> (u32, f32, f32, f32) {
    if tid >= params.used_k {
        return (0, 0.0, 0.0, 0.0);
    }
    let source = query as usize * params.input_k as usize + params.skip as usize + tid as usize;
    let row = indices[source];
    let dist = distances[source];
    let variance = if params.has_yvar != 0 {
        variances[row as usize * params.metrics as usize + metric as usize]
            / scale_sq.max(f32::NOISE_FLOOR)
    } else {
        0.0
    };
    let eps_shift = (params.epsilon - f32::PREC_EPS).max(0.0);
    let weight = neighbor_weight::<f32>(
        dist,
        params.epistemic_scale,
        params.aleatoric_scale + eps_shift,
        variance,
    );
    if metric == 0 {
        unsafe {
            *selected.get_unchecked_mut(query as usize * params.used_k as usize + tid as usize) =
                row;
        }
    }
    (row, dist, variance, weight)
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn run_postfull(
    distances: &[f32],
    indices: &[u32],
    outcomes: &[f32],
    variances: &[f32],
    scales: &[f32],
    weights_out: &mut DisjointSlice<f32>,
    l2_out: &mut DisjointSlice<f32>,
    means: &mut DisjointSlice<f32>,
    errors: &mut DisjointSlice<f32>,
    epistemic: &mut DisjointSlice<f32>,
    aleatoric: &mut DisjointSlice<f32>,
    selected: &mut DisjointSlice<u32>,
    params: WeightedParams,
) {
    static mut SUMS: SharedArray<f32, { THREADS as usize }> = SharedArray::UNINIT;

    let block = thread::blockIdx_x();
    let total = params.queries * params.metrics;
    if block >= total {
        return;
    }
    let query = block / params.metrics;
    let metric = block - query * params.metrics;
    let tid = thread::threadIdx_x();
    let sums = unsafe { SharedArray::as_raw_mut_ptr(&raw mut SUMS) };
    let scale = scales[metric as usize];
    let (row, dist, variance, weight) = full_neighbor(
        distances,
        indices,
        variances,
        selected,
        query,
        metric,
        tid,
        scale * scale,
        params,
    );
    unsafe { sums.add(tid as usize).write(weight) };
    let denom = unsafe { block_sum(sums) }.max(params.epsilon.max(f32::NOISE_FLOOR));
    let inverse = 1.0 / denom;
    let normalized = if tid < params.used_k {
        let value = weight * inverse;
        let output = (query as usize * params.used_k as usize + tid as usize)
            * params.metrics as usize
            + metric as usize;
        unsafe { *weights_out.get_unchecked_mut(output) = value };
        value
    } else {
        0.0
    };
    unsafe { sums.add(tid as usize).write(normalized * normalized) };
    let l2 = unsafe { block_sum(sums) }.sqrt();
    let weighted = if tid < params.used_k {
        normalized * outcomes[row as usize * params.metrics as usize + metric as usize]
    } else {
        0.0
    };
    unsafe { sums.add(tid as usize).write(weighted) };
    let mean = unsafe { block_sum(sums) };
    let nz = params.aleatoric_scale + variance;
    let noise = if tid < params.used_k && params.observation_noise != 0 {
        normalized * nz
    } else {
        0.0
    };
    unsafe { sums.add(tid as usize).write(noise) };
    let noise_sum = unsafe { block_sum(sums) };
    let sq_n = if tid < params.used_k && params.observation_noise != 0 {
        weight * weight * nz
    } else {
        0.0
    };
    unsafe { sums.add(tid as usize).write(sq_n) };
    let sq_noise = unsafe { block_sum(sums) };
    let d_term = if tid < params.used_k {
        normalized * dist
    } else {
        0.0
    };
    unsafe { sums.add(tid as usize).write(d_term) };
    let dist_var = unsafe { block_sum(sums) };
    let (mut error, mut epi, mut ale) = eval_law(
        params.pad,
        denom,
        inverse,
        noise_sum,
        sq_noise,
        dist_var,
        params.epistemic_scale,
        scale,
    );
    if params.pad == 0 && inverse + noise_sum < params.epsilon {
        let floor = params.epsilon.sqrt() * scale;
        error = floor;
        epi = floor;
        ale = 0.0;
    }
    if tid == 0 {
        unsafe {
            *l2_out.get_unchecked_mut(block as usize) = l2;
            *means.get_unchecked_mut(block as usize) = mean;
            *errors.get_unchecked_mut(block as usize) = error;
            *epistemic.get_unchecked_mut(block as usize) = epi;
            *aleatoric.get_unchecked_mut(block as usize) = ale;
        }
    }
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn run_postbatch(
    distances: &[f32],
    indices: &[u32],
    outcomes: &[f32],
    variances: &[f32],
    scales: &[f32],
    values: &[BatchValue],
    means: &mut DisjointSlice<f32>,
    errors: &mut DisjointSlice<f32>,
    epistemic: &mut DisjointSlice<f32>,
    aleatoric: &mut DisjointSlice<f32>,
    params: BatchParams,
) {
    static mut SUMS: SharedArray<f32, { THREADS as usize }> = SharedArray::UNINIT;

    let block = thread::blockIdx_x();
    let query_values = params.queries * params.metrics;
    if block >= params.param_count * query_values {
        return;
    }
    let param_index = block / query_values;
    let query_metric = block - param_index * query_values;
    let query = query_metric / params.metrics;
    let metric = query_metric - query * params.metrics;
    let value = values[param_index as usize];
    let tid = thread::threadIdx_x();
    let sums = unsafe { SharedArray::as_raw_mut_ptr(&raw mut SUMS) };
    let scale = scales[metric as usize];
    let eps_shift = (params.epsilon - f32::PREC_EPS).max(0.0);
    let mut row = 0_u32;
    let mut dist = 0.0_f32;
    let mut variance = 0.0_f32;
    let mut weight = 0.0_f32;
    if tid < value.used_k {
        let source = query as usize * params.input_k as usize + value.skip as usize + tid as usize;
        row = indices[source];
        dist = distances[source];
        if params.has_yvar != 0 {
            variance = variances[row as usize * params.metrics as usize + metric as usize]
                / (scale * scale).max(f32::NOISE_FLOOR);
        }
        weight = neighbor_weight::<f32>(
            dist,
            value.epistemic_scale,
            value.aleatoric_scale + eps_shift,
            variance,
        );
    }
    unsafe { sums.add(tid as usize).write(weight) };
    let denom = unsafe { block_sum(sums) }.max(params.epsilon.max(f32::NOISE_FLOOR));
    let inverse = 1.0 / denom;
    let weighted = if tid < value.used_k {
        weight * outcomes[row as usize * params.metrics as usize + metric as usize]
    } else {
        0.0
    };
    unsafe { sums.add(tid as usize).write(weighted) };
    let mean = unsafe { block_sum(sums) } * inverse;
    let nz = value.aleatoric_scale + variance;
    let noise = if tid < value.used_k && params.observation_noise != 0 {
        weight * inverse * nz
    } else {
        0.0
    };
    unsafe { sums.add(tid as usize).write(noise) };
    let noise_sum = unsafe { block_sum(sums) };
    let sq_n = if tid < value.used_k && params.observation_noise != 0 {
        weight * weight * nz
    } else {
        0.0
    };
    unsafe { sums.add(tid as usize).write(sq_n) };
    let sq_noise = unsafe { block_sum(sums) };
    let d_term = if tid < value.used_k {
        weight * inverse * dist
    } else {
        0.0
    };
    unsafe { sums.add(tid as usize).write(d_term) };
    let dist_var = unsafe { block_sum(sums) };
    let (mut error, mut epi, mut ale) = eval_law(
        params.pad0,
        denom,
        inverse,
        noise_sum,
        sq_noise,
        dist_var,
        value.epistemic_scale,
        scale,
    );
    if params.pad0 == 0 && inverse + noise_sum < params.epsilon {
        let floor = params.epsilon.sqrt() * scale;
        error = floor;
        epi = floor;
        ale = 0.0;
    }
    if tid == 0 {
        unsafe {
            *means.get_unchecked_mut(block as usize) = mean;
            *errors.get_unchecked_mut(block as usize) = error;
            *epistemic.get_unchecked_mut(block as usize) = epi;
            *aleatoric.get_unchecked_mut(block as usize) = ale;
        }
    }
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn run_postdraw(
    weights: &[f32],
    l2: &[f32],
    means: &[f32],
    errors: &[f32],
    indices: &[u32],
    seeds: &[u64],
    draws: &mut DisjointSlice<f64>,
    params: DrawParams,
) {
    let output = thread::blockIdx_x() * THREADS + thread::threadIdx_x();
    let query_values = params.queries * params.metrics;
    if output >= params.seed_count * query_values {
        return;
    }
    let seed_index = output / query_values;
    let query_metric = output - seed_index * query_values;
    let query = query_metric / params.metrics;
    let metric = query_metric - query * params.metrics;
    let mut weighted = 0.0_f64;
    let mut neighbor = 0_u32;
    while neighbor < params.neighbors {
        let neighbor_offset = query as usize * params.neighbors as usize + neighbor as usize;
        let weight_offset = neighbor_offset * params.metrics as usize + metric as usize;
        weighted += f64::from(weights[weight_offset])
            * normal_metric(
                seeds[seed_index as usize],
                i64::from(indices[neighbor_offset]),
                metric as usize,
            );
        neighbor += 1;
    }
    let value_offset = query as usize * params.metrics as usize + metric as usize;
    let scale = f64::from(errors[value_offset]) / f64::from(l2[value_offset]).max(1.0e-12);
    unsafe {
        *draws.get_unchecked_mut(output as usize) =
            f64::from(means[value_offset]) + scale * weighted;
    }
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn corr_draw(
    pairs: &[(f32, u32)],
    variances: Option<&[f32]>,
    epistemic: f32,
    aleatoric: f32,
    scale: f32,
    mean: f32,
    se_epi: f32,
    seed: u64,
    metric: usize,
) -> f32 {
    if pairs.is_empty() {
        return mean + se_epi.max(0.0) * (normal_metric(seed, 0, metric) as f32);
    }
    let scale_sq = (scale * scale).max(f32::NOISE_FLOOR);
    let mut denom = 0.0_f32;
    for &(dist, idx) in pairs {
        let yv = variances
            .and_then(|v| v.get(idx as usize))
            .copied()
            .unwrap_or(0.0)
            .max(0.0);
        denom += neighbor_weight::<f32>(dist, epistemic, aleatoric + yv / scale_sq, 0.0);
    }
    let inv = 1.0 / denom.max(f32::NOISE_FLOOR);
    let mut weighted = 0.0_f32;
    let mut sq_sum = 0.0_f32;
    for &(dist, idx) in pairs {
        let yv = variances
            .and_then(|v| v.get(idx as usize))
            .copied()
            .unwrap_or(0.0)
            .max(0.0);
        let nw = neighbor_weight::<f32>(dist, epistemic, aleatoric + yv / scale_sq, 0.0) * inv;
        let z = normal_metric(seed, i64::from(idx), metric) as f32;
        weighted += nw * z;
        sq_sum += nw * nw;
    }
    mean + se_epi.max(0.0) * (weighted / sq_sum.sqrt().max(f32::NOISE_FLOOR))
}

#[inline]
#[allow(clippy::too_many_arguments)]
pub fn score_cand(
    cand: usize,
    raw_dists: &[f32],
    raw_idxs: &[u32],
    outcomes: &[f32],
    variances: &[f32],
    mods: &[f32],
    header: &SelectHeader,
    arm_scores: &mut [f32],
    cand_means: &mut [f32],
) -> (f32, f32, f32, u32) {
    let hist = header.history as usize;
    let scan_k = (header.tiles as usize)
        .min(KNN_K)
        .min(raw_dists.len())
        .min(raw_idxs.len());
    let used_k = (header.neighbors as usize).min(scan_k);
    if used_k == 0 || hist == 0 {
        arm_scores.fill(0.0);
        cand_means.fill(0.0);
        return (0.0, 0.0, 0.0, 0);
    }
    let m_cnt = (header.num_metrics as usize).clamp(1, MAX_METRICS);
    let ms = unpack_mods(mods, m_cnt, hist);

    let mut pairs = [(f32::INFINITY, u32::MAX); KNN_K];
    let pair_len = if MetricKind::from_code(header.distance_metric) == MetricKind::Euclidean {
        for r in 0..used_k {
            pairs[r] = (raw_dists[r], raw_idxs[r]);
        }
        used_k
    } else {
        let s_idx = (header.scale_neighbors.saturating_sub(1) as usize).min(scan_k - 1);
        let s_q = raw_dists[s_idx].max(f32::NOISE_FLOOR).sqrt();
        let mut tk = TopK::<f32, KNN_K>::new(used_k);
        for r in 0..scan_k {
            let row = raw_idxs[r] as usize;
            let s_h = ms
                .local_scales
                .get(row)
                .copied()
                .unwrap_or(1.0)
                .max(f32::NOISE_FLOOR)
                .sqrt();
            let d = metric_pair(header.distance_metric, raw_dists[r], s_q, s_h);
            tk.push(d, raw_idxs[r]);
        }
        let s = tk.as_slice();
        pairs[..s.len()].copy_from_slice(s);
        s.len()
    };

    let mut mu_vec = [0.0_f32; MAX_METRICS];
    let mut tot_vec = [0.0_f32; MAX_METRICS];
    let mut epi_vec = [0.0_f32; MAX_METRICS];
    let mut ale_vec = [0.0_f32; MAX_METRICS];
    let mut used_count = 0_u32;

    for m in 0..m_cnt {
        let y_start = m * hist;
        let y_end = ((m + 1) * hist).min(outcomes.len());
        let y_col = if y_start < y_end {
            &outcomes[y_start..y_end]
        } else {
            &[]
        };
        let v_col = if header.has_yvar != 0 && variances.len() >= (m + 1) * hist {
            Some(&variances[m * hist..(m + 1) * hist])
        } else {
            None
        };
        let scale_m = ms.scales.get(m).copied().unwrap_or(header.y_scale);
        let post = eval_topk(
            header.uncertainty_law,
            &pairs[..pair_len],
            y_col,
            v_col,
            header.epistemic_scale,
            header.aleatoric_scale,
            scale_m,
            header.adaptive_k != 0,
        );
        if m == 0 {
            used_count = post.count as u32;
        }
        let (mu, tot, epi, ale) = if header.has_ybounds != 0 && ms.ybounds.len() >= 2 * (m + 1) {
            natural_point::<f32>(
                post.mu,
                post.se_total,
                post.se_epi,
                post.se_ale,
                ms.ybounds[2 * m],
                ms.ybounds[2 * m + 1],
            )
        } else {
            (post.mu, post.se_total, post.se_epi, post.se_ale)
        };
        mu_vec[m] = mu;
        tot_vec[m] = tot;
        epi_vec[m] = epi;
        ale_vec[m] = ale;
    }

    if (header.mode & MOD_STIEFEL) != 0 && m_cnt > 1 && ms.stiefel.len() >= m_cnt * m_cnt {
        let mut mu_tmp = [0.0_f32; MAX_METRICS];
        let mut epi_tmp = [0.0_f32; MAX_METRICS];
        contract_row::<f32>(
            &ms.stiefel[..m_cnt * m_cnt],
            m_cnt,
            &mu_vec[..m_cnt],
            &epi_vec[..m_cnt],
            &mut mu_tmp[..m_cnt],
            &mut epi_tmp[..m_cnt],
        );
        for m in 0..m_cnt {
            mu_vec[m] = mu_tmp[m];
            epi_vec[m] = epi_tmp[m];
            tot_vec[m] = (epi_tmp[m] * epi_tmp[m] + ale_vec[m] * ale_vec[m]).sqrt();
        }
    }

    if (header.mode & MOD_AFFINE) != 0 {
        for m in 0..m_cnt {
            let a = ms.aff_a.get(m).copied().unwrap_or(0.0);
            let b = ms.aff_b.get(m).copied().unwrap_or(1.0);
            let c = ms.aff_c.get(m).copied().unwrap_or(1.0);
            let (m2, e2) = map_point::<f32>(a, b, c, mu_vec[m], epi_vec[m]);
            let (_, t2) = map_point::<f32>(a, b, c, mu_vec[m], tot_vec[m]);
            let (_, a2) = map_point::<f32>(0.0, 1.0, c, 0.0, ale_vec[m]);
            mu_vec[m] = m2;
            epi_vec[m] = e2;
            tot_vec[m] = t2;
            ale_vec[m] = a2;
        }
    }

    let is_morbo = (header.mode & MOD_MORBO) != 0 && m_cnt > 1;
    let use_corr = (header.mode & MOD_CORR) != 0;
    let pred_mean = if is_morbo && ms.morbo_w.len() >= m_cnt {
        chebyshev::<f32>(
            &mu_vec[..m_cnt],
            &ms.y_min[..m_cnt],
            &ms.y_max[..m_cnt],
            &ms.morbo_w[..m_cnt],
            header.morbo_alpha,
            false,
        )
    } else {
        mu_vec[0]
    };

    if cand_means.len() >= m_cnt && m_cnt > 1 && !is_morbo && header.acquisition == ACQ_PARETO {
        cand_means[..m_cnt].copy_from_slice(&mu_vec[..m_cnt]);
    } else if !cand_means.is_empty() {
        cand_means[0] = pred_mean;
    }

    match header.acquisition {
        ACQ_THOMPSON => {
            for (arm, dst) in arm_scores.iter_mut().enumerate() {
                let a_seed = arm_seed(header.acq_seed, arm as u32);
                if is_morbo && ms.morbo_w.len() >= m_cnt {
                    let mut draws = [0.0_f32; MAX_METRICS];
                    for m in 0..m_cnt {
                        let v_col = if header.has_yvar != 0 && variances.len() >= (m + 1) * hist {
                            Some(&variances[m * hist..(m + 1) * hist])
                        } else {
                            None
                        };
                        let scale_m = ms.scales.get(m).copied().unwrap_or(header.y_scale);
                        draws[m] = if use_corr {
                            corr_draw(
                                &pairs[..pair_len],
                                v_col,
                                header.epistemic_scale,
                                header.aleatoric_scale,
                                scale_m,
                                mu_vec[m],
                                epi_vec[m],
                                a_seed,
                                m,
                            )
                        } else {
                            score_thompson::<f32>(
                                Key::new(a_seed).fold(m as u64),
                                cand,
                                mu_vec[m],
                                epi_vec[m],
                            )
                        };
                    }
                    *dst = chebyshev::<f32>(
                        &draws[..m_cnt],
                        &ms.y_min[..m_cnt],
                        &ms.y_max[..m_cnt],
                        &ms.morbo_w[..m_cnt],
                        header.morbo_alpha,
                        false,
                    );
                } else {
                    let v_col = if header.has_yvar != 0 && variances.len() >= hist {
                        Some(&variances[..hist])
                    } else {
                        None
                    };
                    let scale_0 = ms.scales.first().copied().unwrap_or(header.y_scale);
                    *dst = if use_corr {
                        corr_draw(
                            &pairs[..pair_len],
                            v_col,
                            header.epistemic_scale,
                            header.aleatoric_scale,
                            scale_0,
                            mu_vec[0],
                            epi_vec[0],
                            a_seed,
                            0,
                        )
                    } else {
                        score_thompson::<f32>(Key::new(a_seed), cand, mu_vec[0], epi_vec[0])
                    };
                }
            }
        }
        ACQ_RANDOM => {
            if !arm_scores.is_empty() {
                arm_scores[0] = 0.0;
            }
        }
        _ => {
            if !arm_scores.is_empty() {
                arm_scores[0] = if is_morbo && ms.morbo_w.len() >= m_cnt {
                    let mut u_vec = [0.0_f32; MAX_METRICS];
                    for m in 0..m_cnt {
                        u_vec[m] = score_ucb::<f32>(mu_vec[m], epi_vec[m], header.beta);
                    }
                    chebyshev::<f32>(
                        &u_vec[..m_cnt],
                        &ms.y_min[..m_cnt],
                        &ms.y_max[..m_cnt],
                        &ms.morbo_w[..m_cnt],
                        header.morbo_alpha,
                        false,
                    )
                } else {
                    score_ucb::<f32>(mu_vec[0], epi_vec[0], header.beta)
                };
            }
        }
    }

    (tot_vec[0], epi_vec[0], ale_vec[0], used_count)
}

#[inline(always)]
pub fn better_arm(sc_a: f32, prio_a: u64, idx_a: u32, sc_b: f32, prio_b: u64, idx_b: u32) -> bool {
    if idx_a == u32::MAX {
        return false;
    }
    if idx_b == u32::MAX {
        return true;
    }
    match BetterScore(sc_a).cmp(&BetterScore(sc_b)) {
        Ordering::Greater => true,
        Ordering::Less => false,
        Ordering::Equal => match prio_a.cmp(&prio_b) {
            Ordering::Greater => true,
            Ordering::Less => false,
            Ordering::Equal => idx_a < idx_b,
        },
    }
}

#[inline(always)]
fn make_arm(
    winner: u32,
    arm: u32,
    score: f32,
    means: &[f32],
    errors: &[f32],
    mean_cols: usize,
    header: &SelectHeader,
) -> ArmDecision {
    let cands = header.num_candidates as usize;
    let w = winner as usize;
    let valid = u32::from(winner < header.num_candidates);
    let pred_mean = if valid != 0 && w * mean_cols < means.len() {
        means[w * mean_cols]
    } else {
        0.0
    };
    let pred_se = if valid != 0 && w < errors.len() {
        errors[w]
    } else {
        0.0
    };
    let pred_epi = if valid != 0 && cands + w < errors.len() {
        errors[cands + w]
    } else {
        0.0
    };
    let pred_ale = if valid != 0 && 2 * cands + w < errors.len() {
        errors[2 * cands + w]
    } else {
        0.0
    };
    let neighbors = if valid != 0 && 3 * cands + w < errors.len() {
        errors[3 * cands + w] as u32
    } else {
        header.neighbors
    };
    ArmDecision {
        root_seed: header.root_seed,
        seed: Key::new(header.acq_seed).fold(u64::from(winner)).seed(),
        index: if valid != 0 { winner } else { 0 },
        arm,
        valid,
        mode: header.mode & 0xff,
        radius: header.radius,
        score,
        pred_mean,
        pred_se,
        pred_epi,
        pred_ale,
        incumbent_mean: header.axis_base,
        incumbent_se: header.axis_weight,
        neighbors,
        incumbent_k: header.neighbors,
    }
}

pub fn pick_host(
    scores: &[f32],
    means: &[f32],
    errors: &[f32],
    state: &mut [u32],
    selected: &mut [u32],
    decisions: &mut [ArmDecision],
    header: &SelectHeader,
) {
    let cands = header.num_candidates as usize;
    let arms = (header.num_arms as usize)
        .min(cands)
        .min(selected.len())
        .min(decisions.len());
    if cands == 0 || arms == 0 || state.len() < 2 * cands {
        return;
    }
    state[..2 * cands].fill(0);
    let m_cnt = (header.num_metrics as usize).clamp(1, MAX_METRICS);
    let nds_mode = header.acquisition == ACQ_PARETO && m_cnt > 1 && (header.mode & MOD_MORBO) == 0;
    let mean_cols = if nds_mode { m_cnt } else { 1 };
    let acq_key = if nds_mode {
        Key::new(header.acq_seed).fold(TAG_PARETO)
    } else {
        Key::new(header.acq_seed)
    };

    let mut chosen = 0usize;
    if header.forced_candidate != NO_FORCED && (header.forced_candidate as usize) < cands {
        let w = header.forced_candidate;
        state[w as usize] = 1;
        selected[0] = w;
        let sc = scores.get(w as usize).copied().unwrap_or(0.0);
        decisions[0] = make_arm(w, 0, sc, means, errors, mean_cols, header);
        chosen = 1;
    }

    if header.acquisition == ACQ_PARETO {
        while chosen < arms {
            let mut any_front = false;
            for cand in 0..cands {
                if state[cand] != 0 {
                    state[cands + cand] = 0;
                    continue;
                }
                let mut dom = false;
                for other in 0..cands {
                    if other == cand || state[other] != 0 {
                        continue;
                    }
                    let d = if nds_mode {
                        dominates::<f32>(
                            &means[other * m_cnt..(other + 1) * m_cnt],
                            &means[cand * m_cnt..(cand + 1) * m_cnt],
                        )
                    } else {
                        pareto_dom::<f32>(
                            means[other],
                            errors[cands + other],
                            means[cand],
                            errors[cands + cand],
                        )
                    };
                    if d {
                        dom = true;
                        break;
                    }
                }
                state[cands + cand] = u32::from(!dom);
                any_front |= !dom;
            }
            if !any_front {
                break;
            }
            let front_start = chosen;
            while chosen < arms {
                let mut best_idx = u32::MAX;
                let mut best_prio = 0_u64;
                for cand in 0..cands {
                    if state[cands + cand] == 0 {
                        continue;
                    }
                    let prio = acq_key.fold(cand as u64).seed();
                    if better_arm(0.0, prio, cand as u32, 0.0, best_prio, best_idx) {
                        best_idx = cand as u32;
                        best_prio = prio;
                    }
                }
                if best_idx == u32::MAX {
                    break;
                }
                state[cands + best_idx as usize] = 0;
                selected[chosen] = best_idx;
                let sc = scores.get(best_idx as usize).copied().unwrap_or(0.0);
                decisions[chosen] = make_arm(
                    best_idx,
                    chosen as u32,
                    sc,
                    means,
                    errors,
                    mean_cols,
                    header,
                );
                chosen += 1;
            }
            for &w in &selected[front_start..chosen] {
                state[w as usize] = 1;
            }
        }
        return;
    }

    while chosen < arms {
        let mut best_sc = f32::NAN;
        let mut best_prio = 0_u64;
        let mut best_idx = u32::MAX;
        let score_off = if header.acquisition == ACQ_THOMPSON {
            chosen * cands
        } else {
            0
        };
        for cand in 0..cands {
            if state[cand] != 0 {
                continue;
            }
            let sc = if header.acquisition == ACQ_RANDOM {
                0.0
            } else {
                scores.get(score_off + cand).copied().unwrap_or(f32::NAN)
            };
            let prio = acq_key.fold(cand as u64).seed();
            if better_arm(sc, prio, cand as u32, best_sc, best_prio, best_idx) {
                best_sc = sc;
                best_prio = prio;
                best_idx = cand as u32;
            }
        }
        if best_idx == u32::MAX {
            break;
        }
        state[best_idx as usize] = 1;
        selected[chosen] = best_idx;
        decisions[chosen] = make_arm(
            best_idx,
            chosen as u32,
            best_sc,
            means,
            errors,
            mean_cols,
            header,
        );
        chosen += 1;
    }
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn run_poolscore(
    distances: &[f32],
    indices: &[u32],
    outcomes: &[f32],
    variances: &[f32],
    mods: &[f32],
    scores: &mut DisjointSlice<f32>,
    means: &mut DisjointSlice<f32>,
    errors: &mut DisjointSlice<f32>,
    header: SelectHeader,
) {
    let cand = (thread::blockIdx_x() * SCORE_THREADS + thread::threadIdx_x()) as usize;
    let cands = header.num_candidates as usize;
    if cand >= cands {
        return;
    }
    let scan_k = header.tiles as usize;
    let d_off = cand * scan_k;
    let d_end = (d_off + scan_k).min(distances.len()).min(indices.len());
    let raw_d = if d_off < d_end {
        &distances[d_off..d_end]
    } else {
        &[]
    };
    let raw_i = if d_off < d_end {
        &indices[d_off..d_end]
    } else {
        &[]
    };

    let m_cnt = (header.num_metrics as usize).clamp(1, MAX_METRICS);
    let nds_mode = header.acquisition == ACQ_PARETO && m_cnt > 1 && (header.mode & MOD_MORBO) == 0;
    let mean_cols = if nds_mode { m_cnt } else { 1 };
    let score_arms = if header.acquisition == ACQ_THOMPSON {
        (header.num_arms as usize).max(1)
    } else {
        1
    };

    let mut cand_means = [0.0_f32; MAX_METRICS];
    let mut first_score = [0.0_f32; 1];

    if header.acquisition == ACQ_THOMPSON && score_arms > 1 {
        let mut arm_buf = [0.0_f32; 64];
        if score_arms <= arm_buf.len() {
            let (tot, epi, ale, count) = score_cand(
                cand,
                raw_d,
                raw_i,
                outcomes,
                variances,
                mods,
                &header,
                &mut arm_buf[..score_arms],
                &mut cand_means[..mean_cols],
            );
            for (arm, &val) in arm_buf.iter().enumerate().take(score_arms) {
                unsafe { *scores.get_unchecked_mut(arm * cands + cand) = val };
            }
            for col in 0..mean_cols {
                unsafe { *means.get_unchecked_mut(cand * mean_cols + col) = cand_means[col] };
            }
            unsafe {
                *errors.get_unchecked_mut(cand) = tot;
                *errors.get_unchecked_mut(cands + cand) = epi;
                *errors.get_unchecked_mut(2 * cands + cand) = ale;
                *errors.get_unchecked_mut(3 * cands + cand) = count as f32;
            }
            return;
        }
    }

    let (tot, epi, ale, count) = score_cand(
        cand,
        raw_d,
        raw_i,
        outcomes,
        variances,
        mods,
        &header,
        &mut first_score,
        &mut cand_means[..mean_cols],
    );
    unsafe {
        *scores.get_unchecked_mut(cand) = first_score[0];
        for col in 0..mean_cols {
            *means.get_unchecked_mut(cand * mean_cols + col) = cand_means[col];
        }
        *errors.get_unchecked_mut(cand) = tot;
        *errors.get_unchecked_mut(cands + cand) = epi;
        *errors.get_unchecked_mut(2 * cands + cand) = ale;
        *errors.get_unchecked_mut(3 * cands + cand) = count as f32;
    }
    if header.acquisition == ACQ_THOMPSON && score_arms > 1 {
        let mut hdr = header;
        hdr.num_arms = 1;
        for arm in 1..score_arms {
            hdr.acq_seed = arm_seed(header.acq_seed, arm as u32);
            let mut single = [0.0_f32; 1];
            let mut dummy = [0.0_f32; MAX_METRICS];
            let _ = score_cand(
                cand,
                raw_d,
                raw_i,
                outcomes,
                variances,
                mods,
                &hdr,
                &mut single,
                &mut dummy[..mean_cols],
            );
            unsafe { *scores.get_unchecked_mut(arm * cands + cand) = single[0] };
        }
    }
}

#[inline(always)]
pub(crate) fn run_poolpick(
    scores: &[f32],
    means: &[f32],
    errors: &[f32],
    state: &mut DisjointSlice<u32>,
    selected: &mut DisjointSlice<u32>,
    decisions: &mut DisjointSlice<ArmDecision>,
    header: SelectHeader,
) {
    static mut BEST_SC: SharedArray<f32, { THREADS as usize }> = SharedArray::UNINIT;
    static mut BEST_HI: SharedArray<u32, { THREADS as usize }> = SharedArray::UNINIT;
    static mut BEST_LO: SharedArray<u32, { THREADS as usize }> = SharedArray::UNINIT;
    static mut BEST_ID: SharedArray<u32, { THREADS as usize }> = SharedArray::UNINIT;
    static mut FRONT_ANY: SharedArray<u32, 1> = SharedArray::UNINIT;

    let cands = header.num_candidates as usize;
    let arms = (header.num_arms as usize).min(cands);
    if cands == 0 || arms == 0 {
        return;
    }
    let tid = thread::threadIdx_x() as usize;
    let sc_sh = unsafe { SharedArray::as_raw_mut_ptr(&raw mut BEST_SC) };
    let hi_sh = unsafe { SharedArray::as_raw_mut_ptr(&raw mut BEST_HI) };
    let lo_sh = unsafe { SharedArray::as_raw_mut_ptr(&raw mut BEST_LO) };
    let id_sh = unsafe { SharedArray::as_raw_mut_ptr(&raw mut BEST_ID) };
    let any_sh = unsafe { SharedArray::as_raw_mut_ptr(&raw mut FRONT_ANY) };
    let s_ptr = state.as_mut_ptr();
    let sel_ptr = selected.as_mut_ptr();
    let dec_ptr = decisions.as_mut_ptr();

    let mut idx = tid;
    while idx < 2 * cands {
        unsafe { s_ptr.add(idx).write(0) };
        idx += THREADS as usize;
    }
    thread::sync_threads();

    let m_cnt = (header.num_metrics as usize).clamp(1, MAX_METRICS);
    let nds_mode = header.acquisition == ACQ_PARETO && m_cnt > 1 && (header.mode & MOD_MORBO) == 0;
    let mean_cols = if nds_mode { m_cnt } else { 1 };
    let acq_key = if nds_mode {
        Key::new(header.acq_seed).fold(TAG_PARETO)
    } else {
        Key::new(header.acq_seed)
    };

    let mut chosen = 0usize;
    if header.forced_candidate != NO_FORCED && (header.forced_candidate as usize) < cands {
        if tid == 0 {
            let w = header.forced_candidate;
            let sc = scores.get(w as usize).copied().unwrap_or(0.0);
            unsafe {
                s_ptr.add(w as usize).write(1);
                sel_ptr.write(w);
                dec_ptr.write(make_arm(w, 0, sc, means, errors, mean_cols, &header));
            }
        }
        chosen = 1;
        thread::sync_threads();
    }

    if header.acquisition == ACQ_PARETO {
        while chosen < arms {
            if tid == 0 {
                unsafe { any_sh.write(0) };
            }
            thread::sync_threads();

            let mut cand = tid;
            let mut local_any = false;
            while cand < cands {
                let is_picked = unsafe { s_ptr.add(cand).read() } != 0;
                if is_picked {
                    unsafe { s_ptr.add(cands + cand).write(0) };
                } else {
                    let mut dom = false;
                    let mut other = 0usize;
                    while other < cands {
                        if other != cand && unsafe { s_ptr.add(other).read() } == 0 {
                            let d = if nds_mode {
                                dominates::<f32>(
                                    &means[other * m_cnt..(other + 1) * m_cnt],
                                    &means[cand * m_cnt..(cand + 1) * m_cnt],
                                )
                            } else {
                                pareto_dom::<f32>(
                                    means[other],
                                    errors[cands + other],
                                    means[cand],
                                    errors[cands + cand],
                                )
                            };
                            if d {
                                dom = true;
                                break;
                            }
                        }
                        other += 1;
                    }
                    unsafe { s_ptr.add(cands + cand).write(u32::from(!dom)) };
                    local_any |= !dom;
                }
                cand += THREADS as usize;
            }
            if local_any {
                unsafe { any_sh.write(1) };
            }
            thread::sync_threads();
            if unsafe { any_sh.read() } == 0 {
                break;
            }

            let front_start = chosen;
            while chosen < arms {
                let mut best_prio = 0_u64;
                let mut best_idx = u32::MAX;
                let mut c = tid;
                while c < cands {
                    if unsafe { s_ptr.add(cands + c).read() } != 0 {
                        let prio = acq_key.fold(c as u64).seed();
                        if better_arm(0.0, prio, c as u32, 0.0, best_prio, best_idx) {
                            best_prio = prio;
                            best_idx = c as u32;
                        }
                    }
                    c += THREADS as usize;
                }
                unsafe {
                    sc_sh.add(tid).write(0.0);
                    hi_sh.add(tid).write((best_prio >> 32) as u32);
                    lo_sh.add(tid).write(best_prio as u32);
                    id_sh.add(tid).write(best_idx);
                }
                thread::sync_threads();
                let mut stride = (THREADS as usize) / 2;
                while stride > 0 {
                    if tid < stride {
                        unsafe {
                            let id_a = id_sh.add(tid).read();
                            let pr_a = (u64::from(hi_sh.add(tid).read()) << 32)
                                | u64::from(lo_sh.add(tid).read());
                            let id_b = id_sh.add(tid + stride).read();
                            let pr_b = (u64::from(hi_sh.add(tid + stride).read()) << 32)
                                | u64::from(lo_sh.add(tid + stride).read());
                            if better_arm(0.0, pr_b, id_b, 0.0, pr_a, id_a) {
                                hi_sh.add(tid).write((pr_b >> 32) as u32);
                                lo_sh.add(tid).write(pr_b as u32);
                                id_sh.add(tid).write(id_b);
                            }
                        }
                    }
                    thread::sync_threads();
                    stride /= 2;
                }
                let winner = unsafe { id_sh.read() };
                if winner == u32::MAX {
                    break;
                }
                if tid == 0 {
                    let sc = scores.get(winner as usize).copied().unwrap_or(0.0);
                    unsafe {
                        s_ptr.add(cands + winner as usize).write(0);
                        sel_ptr.add(chosen).write(winner);
                        dec_ptr.add(chosen).write(make_arm(
                            winner,
                            chosen as u32,
                            sc,
                            means,
                            errors,
                            mean_cols,
                            &header,
                        ));
                    }
                }
                chosen += 1;
                thread::sync_threads();
            }
            let mut slot = front_start + tid;
            while slot < chosen {
                let w = unsafe { sel_ptr.add(slot).read() } as usize;
                unsafe { s_ptr.add(w).write(1) };
                slot += THREADS as usize;
            }
            thread::sync_threads();
        }
        return;
    }

    while chosen < arms {
        let mut best_sc = f32::NAN;
        let mut best_prio = 0_u64;
        let mut best_idx = u32::MAX;
        let score_off = if header.acquisition == ACQ_THOMPSON {
            chosen * cands
        } else {
            0
        };
        let mut cand = tid;
        while cand < cands {
            if unsafe { s_ptr.add(cand).read() } == 0 {
                let sc = if header.acquisition == ACQ_RANDOM {
                    0.0
                } else {
                    scores.get(score_off + cand).copied().unwrap_or(f32::NAN)
                };
                let prio = acq_key.fold(cand as u64).seed();
                if better_arm(sc, prio, cand as u32, best_sc, best_prio, best_idx) {
                    best_sc = sc;
                    best_prio = prio;
                    best_idx = cand as u32;
                }
            }
            cand += THREADS as usize;
        }
        unsafe {
            sc_sh.add(tid).write(best_sc);
            hi_sh.add(tid).write((best_prio >> 32) as u32);
            lo_sh.add(tid).write(best_prio as u32);
            id_sh.add(tid).write(best_idx);
        }
        thread::sync_threads();

        let mut stride = (THREADS as usize) / 2;
        while stride > 0 {
            if tid < stride {
                unsafe {
                    let sc_a = sc_sh.add(tid).read();
                    let pr_a =
                        (u64::from(hi_sh.add(tid).read()) << 32) | u64::from(lo_sh.add(tid).read());
                    let id_a = id_sh.add(tid).read();
                    let sc_b = sc_sh.add(tid + stride).read();
                    let pr_b = (u64::from(hi_sh.add(tid + stride).read()) << 32)
                        | u64::from(lo_sh.add(tid + stride).read());
                    let id_b = id_sh.add(tid + stride).read();
                    if better_arm(sc_b, pr_b, id_b, sc_a, pr_a, id_a) {
                        sc_sh.add(tid).write(sc_b);
                        hi_sh.add(tid).write((pr_b >> 32) as u32);
                        lo_sh.add(tid).write(pr_b as u32);
                        id_sh.add(tid).write(id_b);
                    }
                }
            }
            thread::sync_threads();
            stride /= 2;
        }

        let winner = unsafe { id_sh.read() };
        if winner == u32::MAX {
            break;
        }
        if tid == 0 {
            let win_sc = unsafe { sc_sh.read() };
            unsafe {
                s_ptr.add(winner as usize).write(1);
                sel_ptr.add(chosen).write(winner);
                dec_ptr.add(chosen).write(make_arm(
                    winner,
                    chosen as u32,
                    win_sc,
                    means,
                    errors,
                    mean_cols,
                    &header,
                ));
            }
        }
        chosen += 1;
        thread::sync_threads();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use baux::semantics::abi::{ACQ_PARETO, ACQ_UCB, DEFAULT_BETA, MODE_RAASP};
    use baux::semantics::optimizer::surrogate::{select_pareto, select_topk};

    #[test]
    fn device_parity() {
        assert_eq!(DEFAULT_BETA, 1.0);
        let topk = [(0.04_f32, 0_u32), (0.09_f32, 1_u32), (0.25_f32, 2_u32)];
        let outcomes = [1.5_f32, 0.8, -0.2];
        for law in 0..3_u32 {
            let p_dev = eval_topk(law, &topk, &outcomes, None, 0.05, 0.02, 1.0, false);
            let p_ref = match UncertaintyLaw::from_code(law) {
                UncertaintyLaw::InversePrecision => {
                    posterior_topk::<{ UncertaintyLaw::InversePrecision }, f32>(
                        &topk, &outcomes, None, 0.05, 0.02, 1.0, false,
                    )
                }
                UncertaintyLaw::BiasVariance => {
                    posterior_topk::<{ UncertaintyLaw::BiasVariance }, f32>(
                        &topk, &outcomes, None, 0.05, 0.02, 1.0, false,
                    )
                }
                UncertaintyLaw::Geometric => posterior_topk::<{ UncertaintyLaw::Geometric }, f32>(
                    &topk, &outcomes, None, 0.05, 0.02, 1.0, false,
                ),
            };
            assert_eq!(p_dev, p_ref);
        }

        let cands = 128usize;
        let arms = 5usize;
        let scores: Vec<f32> = (0..cands)
            .map(|i| ((i * 37) % 53) as f32 * 0.1 - 1.2)
            .collect();
        let means: Vec<f32> = (0..cands).map(|i| (i % 17) as f32 * 0.2).collect();
        let mut errors = vec![0.1_f32; 4 * cands];
        for i in 0..cands {
            errors[cands + i] = ((cands - i) % 13) as f32 * 0.15;
        }
        let mut header = SelectHeader {
            root_seed: 11,
            acq_seed: 99,
            beta: DEFAULT_BETA,
            num_candidates: cands as u32,
            num_arms: arms as u32,
            neighbors: 3,
            history: 3,
            acquisition: ACQ_UCB,
            tiles: 3,
            mode: MODE_RAASP,
            forced_candidate: NO_FORCED,
            num_metrics: 1,
            ..SelectHeader::default()
        };
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
        let mut ref_idx = [0usize; 5];
        select_topk::<f32>(&scores, Key::new(header.acq_seed), &mut ref_idx);
        let exp: Vec<u32> = ref_idx.iter().map(|&i| i as u32).collect();
        assert_eq!(selected, exp);

        header.acquisition = ACQ_PARETO;
        pick_host(
            &scores,
            &means,
            &errors,
            &mut state,
            &mut selected,
            &mut decisions,
            &header,
        );
        select_pareto::<f32>(
            &means,
            &errors[cands..2 * cands],
            Key::new(header.acq_seed),
            &mut ref_idx,
        );
        let exp_p: Vec<u32> = ref_idx.iter().map(|&i| i as u32).collect();
        assert_eq!(selected, exp_p);
    }

    #[test]
    fn nonfinite_arms() {
        let scores = [
            f32::NAN,
            -f32::NAN,
            f32::NEG_INFINITY,
            -1.0_f32,
            2.5_f32,
            f32::NAN,
        ];
        let means = [
            f32::NAN,
            f32::NEG_INFINITY,
            0.5_f32,
            1.0_f32,
            -2.0_f32,
            f32::NAN,
        ];
        let mut errors = vec![0.0_f32; 24];
        errors[6..12].copy_from_slice(&[f32::NAN, f32::NEG_INFINITY, 0.2, 0.4, 0.1, f32::NAN]);
        let mut header = SelectHeader {
            root_seed: 7,
            acq_seed: 19,
            beta: 1.0,
            num_candidates: 6,
            num_arms: 6,
            neighbors: 2,
            history: 2,
            acquisition: ACQ_UCB,
            tiles: 2,
            mode: MODE_RAASP,
            forced_candidate: NO_FORCED,
            num_metrics: 1,
            ..SelectHeader::default()
        };
        let mut state = vec![0_u32; 12];
        let mut selected = vec![u32::MAX; 6];
        let mut decisions = vec![ArmDecision::default(); 6];
        pick_host(
            &scores,
            &means,
            &errors,
            &mut state,
            &mut selected,
            &mut decisions,
            &header,
        );
        assert_eq!(selected[0], 4);
        assert_eq!(selected[1], 3);
        assert_eq!(selected[2], 2);
        assert!(selected.iter().all(|&i| i < 6));

        header.acquisition = ACQ_PARETO;
        pick_host(
            &scores,
            &means,
            &errors,
            &mut state,
            &mut selected,
            &mut decisions,
            &header,
        );
        assert_eq!(selected[0], 3);
        assert!(selected.iter().all(|&i| i < 6));
    }
}
