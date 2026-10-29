use cuda_device::{DisjointSlice, SharedArray, thread};

use crate::knn::{
    BatchParams, BatchValue, DrawParams, KNN_K, PosteriorParams, WeightedParams, block_sum,
};
use crate::{THREADS, draw_normal};

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
    let thread_index = thread::threadIdx_x();
    let weights = unsafe { SharedArray::as_raw_mut_ptr(&raw mut WEIGHTS) };
    let sums = unsafe { SharedArray::as_raw_mut_ptr(&raw mut SUMS) };
    let mut weight = 0.0_f32;
    if thread_index < params.used_k {
        let source =
            query as usize * params.input_k as usize + params.skip as usize + thread_index as usize;
        let variance =
            params.epsilon + params.epistemic_scale * distances[source] + params.aleatoric_scale;
        weight = 1.0 / variance.max(params.epsilon);
        unsafe {
            weights.add(thread_index as usize).write(weight);
            *selected.get_unchecked_mut(
                query as usize * params.used_k as usize + thread_index as usize,
            ) = indices[source];
        }
    }
    unsafe { sums.add(thread_index as usize).write(weight) };
    let inverse = 1.0 / unsafe { block_sum(sums) }.max(params.epsilon);
    let error_base = inverse.max(params.epsilon).sqrt();
    let mut metric = thread_index;
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
        let output = query as usize * params.metrics as usize + metric as usize;
        unsafe {
            *means.get_unchecked_mut(output) = weighted * inverse;
            *errors.get_unchecked_mut(output) = error_base * scales[metric as usize];
        }
        metric += THREADS;
    }
}

#[inline(always)]
fn split_error(
    epistemic_var: f32,
    aleatoric_var: f32,
    epsilon: f32,
    scale: f32,
) -> (f32, f32, f32) {
    let total_variance = epistemic_var + aleatoric_var;
    if total_variance < epsilon {
        let floor = epsilon.sqrt() * scale;
        (floor, floor, 0.0)
    } else {
        (
            total_variance.sqrt() * scale,
            epistemic_var.sqrt() * scale,
            aleatoric_var.sqrt() * scale,
        )
    }
}

#[inline(always)]
fn full_neighbor(
    distances: &[f32],
    indices: &[u32],
    variances: &[f32],
    selected: &mut DisjointSlice<u32>,
    query: u32,
    metric: u32,
    thread_index: u32,
    scale_sq: f32,
    params: WeightedParams,
) -> (u32, f32, f32) {
    if thread_index >= params.used_k {
        return (0, 0.0, 0.0);
    }
    let source =
        query as usize * params.input_k as usize + params.skip as usize + thread_index as usize;
    let row = indices[source];
    let variance = if params.has_yvar != 0 {
        variances[row as usize * params.metrics as usize + metric as usize] / scale_sq
    } else {
        0.0
    };
    let total_variance = params.epsilon
        + params.epistemic_scale * distances[source]
        + params.aleatoric_scale
        + variance;
    let weight = 1.0 / total_variance.max(params.epsilon);
    if metric == 0 {
        unsafe {
            *selected.get_unchecked_mut(
                query as usize * params.used_k as usize + thread_index as usize,
            ) = row;
        }
    }
    (row, variance, weight)
}

#[inline(always)]
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
    let thread_index = thread::threadIdx_x();
    let sums = unsafe { SharedArray::as_raw_mut_ptr(&raw mut SUMS) };
    let scale = scales[metric as usize];
    let (row, variance, weight) = full_neighbor(
        distances,
        indices,
        variances,
        selected,
        query,
        metric,
        thread_index,
        scale * scale,
        params,
    );
    unsafe { sums.add(thread_index as usize).write(weight) };
    let inverse = 1.0 / unsafe { block_sum(sums) }.max(params.epsilon);
    let normalized = if thread_index < params.used_k {
        let value = weight * inverse;
        let output = (query as usize * params.used_k as usize + thread_index as usize)
            * params.metrics as usize
            + metric as usize;
        unsafe { *weights_out.get_unchecked_mut(output) = value };
        value
    } else {
        0.0
    };
    unsafe {
        sums.add(thread_index as usize)
            .write(normalized * normalized)
    };
    let l2 = unsafe { block_sum(sums) }.sqrt();
    let weighted = if thread_index < params.used_k {
        normalized * outcomes[row as usize * params.metrics as usize + metric as usize]
    } else {
        0.0
    };
    unsafe { sums.add(thread_index as usize).write(weighted) };
    let mean = unsafe { block_sum(sums) };
    let noise = if thread_index < params.used_k && params.observation_noise != 0 {
        normalized * (params.aleatoric_scale + variance)
    } else {
        0.0
    };
    unsafe { sums.add(thread_index as usize).write(noise) };
    let (error, epi, ale) = split_error(inverse, unsafe { block_sum(sums) }, params.epsilon, scale);
    if thread_index == 0 {
        unsafe {
            *l2_out.get_unchecked_mut(block as usize) = l2;
            *means.get_unchecked_mut(block as usize) = mean;
            *errors.get_unchecked_mut(block as usize) = error;
            *epistemic.get_unchecked_mut(block as usize) = epi;
            *epistemic.get_unchecked_mut((total + block) as usize) = ale;
        }
    }
}

#[inline(always)]
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
    let thread_index = thread::threadIdx_x();
    let sums = unsafe { SharedArray::as_raw_mut_ptr(&raw mut SUMS) };
    let scale = scales[metric as usize];
    let mut row = 0_u32;
    let mut variance = 0.0_f32;
    let mut weight = 0.0_f32;
    if thread_index < value.used_k {
        let source =
            query as usize * params.input_k as usize + value.skip as usize + thread_index as usize;
        row = indices[source];
        if params.has_yvar != 0 {
            variance = variances[row as usize * params.metrics as usize + metric as usize]
                / (scale * scale);
        }
        let total_variance = params.epsilon
            + value.epistemic_scale * distances[source]
            + value.aleatoric_scale
            + variance;
        weight = 1.0 / total_variance.max(params.epsilon);
    }
    unsafe { sums.add(thread_index as usize).write(weight) };
    let inverse = 1.0 / unsafe { block_sum(sums) }.max(params.epsilon);
    let weighted = if thread_index < value.used_k {
        weight * outcomes[row as usize * params.metrics as usize + metric as usize]
    } else {
        0.0
    };
    unsafe { sums.add(thread_index as usize).write(weighted) };
    let mean = unsafe { block_sum(sums) } * inverse;
    let noise = if thread_index < value.used_k && params.observation_noise != 0 {
        weight * inverse * (value.aleatoric_scale + variance)
    } else {
        0.0
    };
    unsafe { sums.add(thread_index as usize).write(noise) };
    let (error, epi, ale) = split_error(inverse, unsafe { block_sum(sums) }, params.epsilon, scale);
    if thread_index == 0 {
        unsafe {
            *means.get_unchecked_mut(block as usize) = mean;
            *errors.get_unchecked_mut(block as usize) = error;
            *epistemic.get_unchecked_mut(block as usize) = epi;
            *aleatoric.get_unchecked_mut(block as usize) = ale;
        }
    }
}

#[inline(always)]
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
        weighted += weights[weight_offset] as f64
            * draw_normal(seeds[seed_index as usize], indices[neighbor_offset], metric);
        neighbor += 1;
    }
    let value_offset = query as usize * params.metrics as usize + metric as usize;
    let scale = errors[value_offset] as f64 / (l2[value_offset] as f64).max(1.0e-12);
    unsafe {
        *draws.get_unchecked_mut(output as usize) = means[value_offset] as f64 + scale * weighted;
    }
}
