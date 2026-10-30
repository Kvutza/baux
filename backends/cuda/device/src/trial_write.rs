use cuda_device::{DisjointSlice, SharedArray, thread, warp};

use crate::{
    DenseLeaf, DenseLinearParams, DenseTerm, DenseTile, Selection, THREADS, dense_value,
    reduce_best,
};

#[inline(always)]
pub(crate) fn run_dense(
    base: &[f32],
    leaves: &[DenseLeaf],
    terms: &[DenseTerm],
    tiles: &[DenseTile],
    output: &mut DisjointSlice<f32>,
    term_count: u32,
) {
    let tile_index = thread::blockIdx_x();
    if tile_index as usize >= tiles.len() {
        return;
    }
    let tile = tiles[tile_index as usize];
    let leaf = leaves[tile.leaf as usize];
    let mut item = thread::threadIdx_x();
    while item < tile.length {
        let coordinate = u64::from(tile.start + item);
        let index = leaf.offset + coordinate;
        unsafe {
            *output.get_unchecked_mut(index as usize) = dense_value(
                base[index as usize],
                leaf.key,
                coordinate,
                leaf.scale,
                terms,
                term_count,
            );
        }
        item += THREADS;
    }
}

#[inline(always)]
pub(crate) fn run_linear(
    input: &[f32],
    weight: &[f32],
    bias: &[f32],
    terms: &[DenseTerm],
    output: &mut DisjointSlice<f32>,
    params: DenseLinearParams,
) {
    static mut WARP_SUMS: SharedArray<f32, 8> = SharedArray::UNINIT;

    let row = thread::blockIdx_x();
    if row >= params.rows {
        return;
    }
    let thread_index = thread::threadIdx_x();
    let lane = warp::lane_id();
    let warp_index = thread_index / 32;
    let row_start = u64::from(row) * u64::from(params.columns);
    let mut sum = 0.0_f32;
    let mut column = thread_index;
    while column < params.columns {
        let index = row_start + u64::from(column);
        let value = dense_value(
            weight[index as usize],
            params.weight_key,
            params.weight_start + index,
            params.weight_scale,
            terms,
            params.term_count,
        );
        sum = input[column as usize].mul_add(value, sum);
        column += THREADS;
    }

    let warp_sum = warp::reduce_sum_f32(sum);
    if lane == 0 {
        unsafe {
            WARP_SUMS[warp_index as usize] = warp_sum;
        }
    }
    thread::sync_threads();
    if warp_index == 0 {
        let part = if lane < 8 {
            unsafe { WARP_SUMS[lane as usize] }
        } else {
            0.0
        };
        let mut total = warp::reduce_sum_f32(part);
        if lane == 0 {
            if params.has_bias != 0 {
                total += dense_value(
                    bias[row as usize],
                    params.bias_key,
                    params.bias_start + u64::from(row),
                    params.bias_scale,
                    terms,
                    params.term_count,
                );
            }
            unsafe {
                *output.get_unchecked_mut(row as usize) = total;
            }
        }
    }
}

#[inline(always)]
pub(crate) fn run_pick(
    scores: &[f32],
    selection: &mut DisjointSlice<Selection>,
    regions: u32,
    candidates_per_region: u32,
) {
    static mut WARP_BEST_SCORES: SharedArray<f32, 8> = SharedArray::UNINIT;
    static mut WARP_BEST_INDICES: SharedArray<u32, 8> = SharedArray::UNINIT;

    let thread_index = thread::threadIdx_x();
    let lane = warp::lane_id();
    let warp_index = thread_index / 32;
    let region = thread::blockIdx_x();
    if region >= regions {
        return;
    }
    let first_candidate = region * candidates_per_region;
    let end_candidate = first_candidate + candidates_per_region;

    let mut best_score = f32::NEG_INFINITY;
    let mut best_index = u32::MAX;
    let mut candidate = first_candidate + thread_index;
    while candidate < end_candidate {
        let score = scores[candidate as usize];
        if score > best_score || (score == best_score && candidate < best_index) {
            best_score = score;
            best_index = candidate;
        }
        candidate += THREADS;
    }

    let (best_score, best_index) = reduce_best(best_score, best_index, 16);
    if lane == 0 {
        unsafe {
            WARP_BEST_SCORES[warp_index as usize] = best_score;
            WARP_BEST_INDICES[warp_index as usize] = best_index;
        }
    }
    thread::sync_threads();

    if warp_index == 0 {
        let warp0_score = if lane < 8 {
            unsafe { WARP_BEST_SCORES[lane as usize] }
        } else {
            f32::NEG_INFINITY
        };
        let warp0_index = if lane < 8 {
            unsafe { WARP_BEST_INDICES[lane as usize] }
        } else {
            u32::MAX
        };
        let (final_score, final_index) = reduce_best(warp0_score, warp0_index, 4);
        if lane == 0 {
            unsafe {
                *selection.get_unchecked_mut(region as usize) = Selection {
                    index: final_index,
                    score: final_score,
                };
            }
        }
    }
}
