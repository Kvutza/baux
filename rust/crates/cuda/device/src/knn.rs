use baux::semantics::abi::{
    MODE_GAUSSIAN, MODE_MASKED, MODE_RAASP, MODE_RADEMACHER, MODE_SOBOL, MODE_UNIFORM, PoolDispatch,
};
use baux::semantics::key::Key;
use baux::semantics::neighbor::ard_dist;
use baux::semantics::pool::DelayedPool;
use cuda_device::{DisjointSlice, SharedArray, thread, warp};

use crate::{NeighborShape, THREADS, WARPS};

pub const KNN_K: usize = 128;
pub const POST_MAX_K: usize = 1024;
pub const KNN_THREADS: u32 = THREADS;
pub const KNN_WARPS: u32 = THREADS / 32;
pub const CAND_TILE: usize = 512;

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct KnnParams {
    pub rows: u32,
    pub dims: u32,
    pub queries: u32,
    pub lists: u32,
    pub neighbors: u32,
    pub pad0: u32,
    pub pad1: u32,
    pub pad2: u32,
}

// SAFETY: KnnParams is repr(C) and contains only DeviceCopy scalars.
unsafe impl cuda_core::DeviceCopy for KnnParams {}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct MergeParams {
    pub queries: u32,
    pub input_lists: u32,
    pub output_lists: u32,
    pub neighbors: u32,
}

// SAFETY: MergeParams is repr(C) and contains only DeviceCopy scalars.
unsafe impl cuda_core::DeviceCopy for MergeParams {}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PosteriorParams {
    pub queries: u32,
    pub input_k: u32,
    pub used_k: u32,
    pub skip: u32,
    pub metrics: u32,
    pub epistemic_scale: f32,
    pub aleatoric_scale: f32,
    pub epsilon: f32,
}

// SAFETY: PosteriorParams is repr(C) and contains only DeviceCopy scalars.
unsafe impl cuda_core::DeviceCopy for PosteriorParams {}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct WeightedParams {
    pub queries: u32,
    pub input_k: u32,
    pub used_k: u32,
    pub skip: u32,
    pub metrics: u32,
    pub has_yvar: u32,
    pub observation_noise: u32,
    pub pad: u32,
    pub epistemic_scale: f32,
    pub aleatoric_scale: f32,
    pub epsilon: f32,
    pub padf: f32,
}

// SAFETY: WeightedParams is repr(C) and contains only DeviceCopy scalars.
unsafe impl cuda_core::DeviceCopy for WeightedParams {}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct BatchParams {
    pub queries: u32,
    pub input_k: u32,
    pub metrics: u32,
    pub param_count: u32,
    pub has_yvar: u32,
    pub observation_noise: u32,
    pub pad0: u32,
    pub pad1: u32,
    pub epsilon: f32,
    pub padf0: f32,
    pub padf1: f32,
    pub padf2: f32,
}

// SAFETY: BatchParams is repr(C) and contains only DeviceCopy scalars.
unsafe impl cuda_core::DeviceCopy for BatchParams {}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct BatchValue {
    pub used_k: u32,
    pub skip: u32,
    pub epistemic_scale: f32,
    pub aleatoric_scale: f32,
}

// SAFETY: BatchValue is repr(C) and contains only DeviceCopy scalars.
unsafe impl cuda_core::DeviceCopy for BatchValue {}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct DrawParams {
    pub queries: u32,
    pub neighbors: u32,
    pub metrics: u32,
    pub seed_count: u32,
}

// SAFETY: DrawParams is repr(C) and contains only DeviceCopy scalars.
unsafe impl cuda_core::DeviceCopy for DrawParams {}

#[inline(always)]
pub(super) fn pair_before(
    left_distance: f32,
    left_index: u32,
    right_distance: f32,
    right_index: u32,
) -> bool {
    left_distance < right_distance || (left_distance == right_distance && left_index < right_index)
}

#[inline(always)]
pub(super) unsafe fn order_pair(
    distances: *mut f32,
    indices: *mut u32,
    left: usize,
    right: usize,
    ascending: bool,
) {
    let left_distance = unsafe { distances.add(left).read() };
    let left_index = unsafe { indices.add(left).read() };
    let right_distance = unsafe { distances.add(right).read() };
    let right_index = unsafe { indices.add(right).read() };
    let swap = if ascending {
        pair_before(right_distance, right_index, left_distance, left_index)
    } else {
        pair_before(left_distance, left_index, right_distance, right_index)
    };
    if swap {
        unsafe {
            distances.add(left).write(right_distance);
            indices.add(left).write(right_index);
            distances.add(right).write(left_distance);
            indices.add(right).write(left_index);
        }
    }
}

#[inline(always)]
pub(super) unsafe fn sort_pairs(distances: *mut f32, indices: *mut u32, width: u32) {
    let thread_index = thread::threadIdx_x();
    let mut size = 2_u32;
    while size <= width {
        let mut stride = size / 2;
        while stride > 0 {
            let mut index = thread_index;
            while index < width {
                let partner = index ^ stride;
                if partner > index {
                    unsafe {
                        order_pair(
                            distances,
                            indices,
                            index as usize,
                            partner as usize,
                            index & size == 0,
                        );
                    }
                }
                index += THREADS;
            }
            thread::sync_threads();
            stride /= 2;
        }
        size *= 2;
    }
}

#[inline(always)]
pub(super) fn row_distance(rows: &[f32], queries: &[f32], row: u32, query: u32, dims: u32) -> f32 {
    let row_start = row as usize * dims as usize;
    let query_start = query as usize * dims as usize;
    let mut sum = 0.0_f32;
    let mut dimension = 0_u32;
    while dimension < dims {
        let delta =
            rows[row_start + dimension as usize] - queries[query_start + dimension as usize];
        sum = delta.mul_add(delta, sum);
        dimension += 1;
    }
    sum
}

#[inline(always)]
pub(super) fn warp_distance(rows: &[f32], queries: &[f32], row: u32, query: u32, dims: u32) -> f32 {
    let row_start = row as usize * dims as usize;
    let query_start = query as usize * dims as usize;
    let mut sum = 0.0_f32;
    let mut dimension = warp::lane_id();
    while dimension < dims {
        let delta =
            rows[row_start + dimension as usize] - queries[query_start + dimension as usize];
        sum = delta.mul_add(delta, sum);
        dimension += 32;
    }
    warp::reduce_sum_f32(sum)
}

#[inline(always)]
pub(super) unsafe fn init_pairs(distances: *mut f32, indices: *mut u32, width: u32) {
    let mut index = thread::threadIdx_x();
    while index < width {
        unsafe {
            distances.add(index as usize).write(f32::INFINITY);
            indices.add(index as usize).write(u32::MAX);
        }
        index += THREADS;
    }
}

#[inline(always)]
pub(super) unsafe fn write_list(
    distances: *const f32,
    indices: *const u32,
    output_distances: &mut cuda_device::DisjointSlice<f32>,
    output_indices: &mut cuda_device::DisjointSlice<u32>,
    list: u32,
    neighbors: u32,
) {
    let mut neighbor = thread::threadIdx_x();
    let output_start = list as usize * neighbors as usize;
    while neighbor < neighbors {
        unsafe {
            *output_distances.get_unchecked_mut(output_start + neighbor as usize) =
                distances.add(neighbor as usize).read();
            *output_indices.get_unchecked_mut(output_start + neighbor as usize) =
                indices.add(neighbor as usize).read();
        }
        neighbor += THREADS;
    }
}

#[inline(always)]
pub(crate) unsafe fn block_sum(values: *mut f32) -> f32 {
    let thread_index = thread::threadIdx_x();
    thread::sync_threads();
    let mut stride = THREADS / 2;
    while stride > 0 {
        if thread_index < stride {
            unsafe {
                let value = values.add(thread_index as usize).read()
                    + values.add((thread_index + stride) as usize).read();
                values.add(thread_index as usize).write(value);
            }
        }
        thread::sync_threads();
        stride /= 2;
    }
    unsafe { values.read() }
}

#[inline(always)]
pub(crate) fn run_neighbors(
    pool: &[f32],
    history: &[f32],
    weights: &[f32],
    distances: &mut DisjointSlice<f32>,
    shape: NeighborShape,
) {
    static mut SUMS: SharedArray<f32, WARPS> = SharedArray::UNINIT;
    let pair = thread::blockIdx_x();
    if pair >= shape.history * shape.candidates {
        return;
    }
    let candidate = pair / shape.history;
    let row = pair % shape.history;
    let tid = thread::threadIdx_x();
    let mut index = tid;
    let mut sum = 0.0;
    while index < shape.samples {
        let delta = pool[(candidate * shape.samples + index) as usize]
            - history[(row * shape.samples + index) as usize];
        sum += delta * delta * weights[index as usize];
        index += THREADS;
    }
    let reduced = warp::reduce_sum_f32(sum);
    let sums = unsafe { SharedArray::as_raw_mut_ptr(&raw mut SUMS) };
    if warp::lane_id() == 0 {
        unsafe {
            sums.add((tid / 32) as usize).write(reduced);
        }
    }
    thread::sync_threads();
    if tid == 0 {
        let mut value = 0.0;
        let mut warp_idx = 0;
        while warp_idx < WARPS {
            value += unsafe { sums.add(warp_idx).read() };
            warp_idx += 1;
        }
        unsafe {
            *distances.get_unchecked_mut(pair as usize) = value;
        }
    }
}

#[inline(always)]
pub(crate) fn run_scanrows(
    rows: &[f32],
    queries: &[f32],
    distances: &mut DisjointSlice<f32>,
    indices: &mut DisjointSlice<u32>,
    params: KnnParams,
) {
    static mut VALUES: SharedArray<f32, { KNN_THREADS as usize }> = SharedArray::UNINIT;
    static mut INDICES: SharedArray<u32, { KNN_THREADS as usize }> = SharedArray::UNINIT;

    let list = thread::blockIdx_x();
    if list >= params.queries * params.lists {
        return;
    }
    let thread_index = thread::threadIdx_x();
    let values = unsafe { SharedArray::as_raw_mut_ptr(&raw mut VALUES) };
    let local_indices = unsafe { SharedArray::as_raw_mut_ptr(&raw mut INDICES) };
    unsafe { init_pairs(values, local_indices, KNN_THREADS) };

    let query = list / params.lists;
    let tile = list - query * params.lists;
    let row = tile * KNN_THREADS + thread_index;
    if row < params.rows {
        let distance = row_distance(rows, queries, row, query, params.dims);
        unsafe {
            values.add(thread_index as usize).write(distance);
            local_indices.add(thread_index as usize).write(row);
        }
    }
    thread::sync_threads();

    unsafe {
        sort_pairs(values, local_indices, KNN_THREADS);
        write_list(
            values,
            local_indices,
            distances,
            indices,
            list,
            params.neighbors,
        );
    }
}

#[inline(always)]
pub(crate) fn run_scanwarps(
    rows: &[f32],
    queries: &[f32],
    distances: &mut DisjointSlice<f32>,
    indices: &mut DisjointSlice<u32>,
    params: KnnParams,
) {
    static mut VALUES: SharedArray<f32, { KNN_THREADS as usize }> = SharedArray::UNINIT;
    static mut INDICES: SharedArray<u32, { KNN_THREADS as usize }> = SharedArray::UNINIT;

    let list = thread::blockIdx_x();
    if list >= params.queries * params.lists {
        return;
    }
    let values = unsafe { SharedArray::as_raw_mut_ptr(&raw mut VALUES) };
    let local_indices = unsafe { SharedArray::as_raw_mut_ptr(&raw mut INDICES) };
    unsafe { init_pairs(values, local_indices, KNN_THREADS) };
    thread::sync_threads();

    let thread_index = thread::threadIdx_x();
    let warp_index = thread_index / 32;
    let query = list / params.lists;
    let tile = list - query * params.lists;
    let row = tile * KNN_WARPS + warp_index;
    if row < params.rows {
        let distance = warp_distance(rows, queries, row, query, params.dims);
        if warp::lane_id() == 0 {
            unsafe {
                values.add(warp_index as usize).write(distance);
                local_indices.add(warp_index as usize).write(row);
            }
        }
    }
    thread::sync_threads();

    unsafe {
        sort_pairs(values, local_indices, KNN_WARPS);
        write_list(
            values,
            local_indices,
            distances,
            indices,
            list,
            params.neighbors,
        );
    }
}

#[inline(always)]
pub(crate) fn run_mergetopk(
    input_distances: &[f32],
    input_indices: &[u32],
    output_distances: &mut DisjointSlice<f32>,
    output_indices: &mut DisjointSlice<u32>,
    params: MergeParams,
) {
    static mut VALUES: SharedArray<f32, { KNN_THREADS as usize }> = SharedArray::UNINIT;
    static mut INDICES: SharedArray<u32, { KNN_THREADS as usize }> = SharedArray::UNINIT;

    let output_list = thread::blockIdx_x();
    if output_list >= params.queries * params.output_lists {
        return;
    }
    let thread_index = thread::threadIdx_x();
    let values = unsafe { SharedArray::as_raw_mut_ptr(&raw mut VALUES) };
    let local_indices = unsafe { SharedArray::as_raw_mut_ptr(&raw mut INDICES) };
    unsafe { init_pairs(values, local_indices, KNN_THREADS) };

    let query = output_list / params.output_lists;
    let local_list = output_list - query * params.output_lists;
    let left_list = query * params.input_lists + 2 * local_list;
    let right_local = 2 * local_list + 1;
    if thread_index < params.neighbors {
        let source = left_list as usize * params.neighbors as usize + thread_index as usize;
        unsafe {
            values
                .add(thread_index as usize)
                .write(input_distances[source]);
            local_indices
                .add(thread_index as usize)
                .write(input_indices[source]);
        }
    } else if thread_index < 2 * params.neighbors && right_local < params.input_lists {
        let neighbor = thread_index - params.neighbors;
        let source = (left_list + 1) as usize * params.neighbors as usize + neighbor as usize;
        unsafe {
            values
                .add(thread_index as usize)
                .write(input_distances[source]);
            local_indices
                .add(thread_index as usize)
                .write(input_indices[source]);
        }
    }
    thread::sync_threads();

    let mut width = 2_u32;
    while width < 2 * params.neighbors {
        width *= 2;
    }
    unsafe {
        sort_pairs(values, local_indices, width);
        write_list(
            values,
            local_indices,
            output_distances,
            output_indices,
            output_list,
            params.neighbors,
        );
    }
}

#[inline(always)]
pub fn make_pool<'a>(
    dispatch: &PoolDispatch,
    center: &'a [f32],
    scale: &'a [f32],
    lower: &'a [f32],
    upper: &'a [f32],
    dirs: &'a [[u32; 32]],
) -> DelayedPool<'a, f32> {
    let key = Key(dispatch.key_hi, dispatch.key_lo);
    let num_candidates = dispatch.num_candidates as usize;
    let num_pert = dispatch.num_pert as usize;
    let scramble = dispatch.scramble != 0;
    match dispatch.mode {
        MODE_RAASP => DelayedPool::Raasp {
            key,
            center,
            lower,
            upper,
            num_candidates,
            num_pert,
        },
        MODE_SOBOL => DelayedPool::Sobol {
            key,
            lower,
            upper,
            dirs,
            base_idx: dispatch.base_idx,
            num_candidates,
            scramble,
        },
        MODE_MASKED => DelayedPool::SobolMasked {
            key,
            center,
            lower,
            upper,
            dirs,
            base_idx: dispatch.base_idx,
            num_candidates,
            num_pert,
            scramble,
        },
        MODE_UNIFORM => DelayedPool::Uniform {
            key,
            lower,
            upper,
            num_candidates,
        },
        MODE_GAUSSIAN => DelayedPool::Gaussian {
            key,
            center,
            scale,
            bounds: Some((lower, upper)),
            radius: dispatch.radius,
            num_candidates,
        },
        MODE_RADEMACHER => DelayedPool::Rademacher {
            key,
            center,
            scale,
            bounds: Some((lower, upper)),
            radius: dispatch.radius,
            num_candidates,
        },
        _ => DelayedPool::Lhd {
            key,
            lower,
            upper,
            num_candidates,
        },
    }
}

#[inline(always)]
unsafe fn load_tile(
    pool: &DelayedPool<'_, f32>,
    weights: &[f32],
    coords: *mut f32,
    w_tile: *mut f32,
    cand: usize,
    base_dim: usize,
    chunk: usize,
) {
    let mut idx = thread::threadIdx_x() as usize;
    while idx < chunk {
        let dim = base_dim + idx;
        let val = pool.eval(cand, dim);
        let weight = if dim < weights.len() {
            weights[dim]
        } else {
            1.0
        };
        unsafe {
            coords.add(idx).write(val);
            w_tile.add(idx).write(weight);
        }
        idx += THREADS as usize;
    }
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn run_poolscan(
    rows: &[f32],
    weights: &[f32],
    center: &[f32],
    scale: &[f32],
    lower: &[f32],
    upper: &[f32],
    dirs: &[u32],
    distances: &mut DisjointSlice<f32>,
    indices: &mut DisjointSlice<u32>,
    dispatch: PoolDispatch,
) {
    static mut VALUES: SharedArray<f32, { KNN_THREADS as usize }> = SharedArray::UNINIT;
    static mut INDICES: SharedArray<u32, { KNN_THREADS as usize }> = SharedArray::UNINIT;
    static mut COORDS: SharedArray<f32, CAND_TILE> = SharedArray::UNINIT;
    static mut WTILE: SharedArray<f32, CAND_TILE> = SharedArray::UNINIT;

    let list = thread::blockIdx_x();
    if list >= dispatch.num_candidates * dispatch.tiles {
        return;
    }
    let tid = thread::threadIdx_x();
    let values = unsafe { SharedArray::as_raw_mut_ptr(&raw mut VALUES) };
    let local_indices = unsafe { SharedArray::as_raw_mut_ptr(&raw mut INDICES) };
    let coords = unsafe { SharedArray::as_raw_mut_ptr(&raw mut COORDS) };
    let w_tile = unsafe { SharedArray::as_raw_mut_ptr(&raw mut WTILE) };

    let cand = (list / dispatch.tiles) as usize;
    let tile = list - (cand as u32) * dispatch.tiles;
    let row = tile * KNN_THREADS + tid;
    let dims = dispatch.num_dim as usize;
    let pool = make_pool(
        &dispatch,
        center,
        scale,
        lower,
        upper,
        dirs.as_chunks::<32>().0,
    );

    let mut acc = 0.0_f32;
    let mut base_dim = 0usize;
    while base_dim < dims {
        let chunk = (dims - base_dim).min(CAND_TILE);
        unsafe { load_tile(&pool, weights, coords, w_tile, cand, base_dim, chunk) };
        thread::sync_threads();
        if row < dispatch.history {
            let row_off = row as usize * dims + base_dim;
            let cand_s = unsafe { core::slice::from_raw_parts(coords, chunk) };
            let w_s = unsafe { core::slice::from_raw_parts(w_tile, chunk) };
            acc += ard_dist::<f32>(cand_s, &rows[row_off..row_off + chunk], w_s);
        }
        thread::sync_threads();
        base_dim += CAND_TILE;
    }

    unsafe { init_pairs(values, local_indices, KNN_THREADS) };
    thread::sync_threads();
    if row < dispatch.history {
        unsafe {
            values.add(tid as usize).write(acc);
            local_indices.add(tid as usize).write(row);
        }
    }
    thread::sync_threads();

    unsafe {
        sort_pairs(values, local_indices, KNN_THREADS);
        write_list(
            values,
            local_indices,
            distances,
            indices,
            list,
            dispatch.blocks,
        );
    }
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn run_poolwarp(
    rows: &[f32],
    weights: &[f32],
    center: &[f32],
    scale: &[f32],
    lower: &[f32],
    upper: &[f32],
    dirs: &[u32],
    distances: &mut DisjointSlice<f32>,
    indices: &mut DisjointSlice<u32>,
    dispatch: PoolDispatch,
) {
    static mut VALUES: SharedArray<f32, { KNN_THREADS as usize }> = SharedArray::UNINIT;
    static mut INDICES: SharedArray<u32, { KNN_THREADS as usize }> = SharedArray::UNINIT;
    static mut COORDS: SharedArray<f32, CAND_TILE> = SharedArray::UNINIT;
    static mut WTILE: SharedArray<f32, CAND_TILE> = SharedArray::UNINIT;

    let list = thread::blockIdx_x();
    if list >= dispatch.num_candidates * dispatch.tiles {
        return;
    }
    let tid = thread::threadIdx_x();
    let warp_idx = tid / 32;
    let lane = warp::lane_id() as usize;
    let values = unsafe { SharedArray::as_raw_mut_ptr(&raw mut VALUES) };
    let local_indices = unsafe { SharedArray::as_raw_mut_ptr(&raw mut INDICES) };
    let coords = unsafe { SharedArray::as_raw_mut_ptr(&raw mut COORDS) };
    let w_tile = unsafe { SharedArray::as_raw_mut_ptr(&raw mut WTILE) };

    let cand = (list / dispatch.tiles) as usize;
    let tile = list - (cand as u32) * dispatch.tiles;
    let row = tile * KNN_WARPS + warp_idx;
    let dims = dispatch.num_dim as usize;
    let pool = make_pool(
        &dispatch,
        center,
        scale,
        lower,
        upper,
        dirs.as_chunks::<32>().0,
    );

    let mut acc = 0.0_f32;
    let mut base_dim = 0usize;
    while base_dim < dims {
        let chunk = (dims - base_dim).min(CAND_TILE);
        unsafe { load_tile(&pool, weights, coords, w_tile, cand, base_dim, chunk) };
        thread::sync_threads();
        if row < dispatch.history {
            let row_off = row as usize * dims + base_dim;
            let mut sum = 0.0_f32;
            let mut offset = lane;
            while offset < chunk {
                let c = unsafe { coords.add(offset).read() };
                let w = unsafe { w_tile.add(offset).read() };
                let delta = c - rows[row_off + offset];
                sum = delta.mul_add(delta * w, sum);
                offset += 32;
            }
            acc += warp::reduce_sum_f32(sum);
        }
        thread::sync_threads();
        base_dim += CAND_TILE;
    }

    unsafe { init_pairs(values, local_indices, KNN_THREADS) };
    thread::sync_threads();
    if row < dispatch.history && lane == 0 {
        unsafe {
            values.add(warp_idx as usize).write(acc);
            local_indices.add(warp_idx as usize).write(row);
        }
    }
    thread::sync_threads();

    unsafe {
        sort_pairs(values, local_indices, KNN_WARPS);
        write_list(
            values,
            local_indices,
            distances,
            indices,
            list,
            dispatch.blocks,
        );
    }
}

#[inline(always)]
pub(crate) fn run_poolrows(
    selected: &[u32],
    center: &[f32],
    scale: &[f32],
    lower: &[f32],
    upper: &[f32],
    dirs: &[u32],
    rows_out: &mut DisjointSlice<f32>,
    dispatch: PoolDispatch,
) {
    let idx = thread::blockIdx_x() * THREADS + thread::threadIdx_x();
    let total = dispatch.num_arms * dispatch.num_dim;
    if idx >= total || dispatch.num_dim == 0 {
        return;
    }
    let arm = (idx / dispatch.num_dim) as usize;
    let dim = (idx - (arm as u32) * dispatch.num_dim) as usize;
    if arm >= selected.len() {
        return;
    }
    let cand = selected[arm] as usize;
    if cand >= dispatch.num_candidates as usize {
        return;
    }
    let pool = make_pool(
        &dispatch,
        center,
        scale,
        lower,
        upper,
        dirs.as_chunks::<32>().0,
    );
    let val = pool.eval(cand, dim);
    unsafe {
        *rows_out.get_unchecked_mut(idx as usize) = val;
    }
}
