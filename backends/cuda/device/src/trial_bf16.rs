use cuda_device::{DisjointSlice, SharedArray, thread, warp};

use crate::bf16::{
    acquisition_score, bf16_candidate, bf16_decode, bf16_finite, candidate_radius, tile_distances,
    warp_invalid,
};
use crate::{Bf16Leaf, Bf16Score, DenseTile, MAX_HISTORY, SearchState, Seed, THREADS, WARPS};

#[inline(always)]
pub(crate) fn run_distbf16(
    rows: &[u16],
    history_slots: &[u32],
    reference: &[u16],
    reference_scales: &[f32],
    state: &[SearchState],
    seeds: &[Seed],
    leaves: &[Bf16Leaf],
    tiles: &[DenseTile],
    partials: &mut DisjointSlice<f32>,
    tile_status: &mut DisjointSlice<u32>,
    mut params: Bf16Score,
) {
    static mut VALUES: SharedArray<f32, 256> = SharedArray::UNINIT;
    static mut DISTANCES: SharedArray<f32, MAX_HISTORY> = SharedArray::UNINIT;
    static mut WARP_STATUS: SharedArray<u32, WARPS> = SharedArray::UNINIT;

    if params.resident != 0 {
        params.history = state[0].history;
        params.coefficient = state[0].length as f32;
        params.neighbors = params.neighbors.min(params.history);
    }
    let block_index = thread::blockIdx_x();
    if block_index >= params.candidates * params.tiles {
        return;
    }
    let candidate = block_index / params.tiles;
    if params.correlated == 1 {
        params.coefficient = candidate_radius(state[0], candidate);
    }
    let tile_index = block_index % params.tiles;
    let thread_index = thread::threadIdx_x();
    let lane = warp::lane_id();
    let warp_index = thread_index / 32;
    let values = unsafe { SharedArray::as_raw_mut_ptr(&raw mut VALUES) };
    let distances = unsafe { SharedArray::as_raw_mut_ptr(&raw mut DISTANCES) };
    let warp_status = unsafe { SharedArray::as_raw_mut_ptr(&raw mut WARP_STATUS) };
    if thread_index < params.history {
        unsafe { distances.add(thread_index as usize).write(0.0) };
    }
    if lane == 0 {
        unsafe { warp_status.add(warp_index as usize).write(0) };
    }
    thread::sync_threads();

    let tile = tiles[tile_index as usize];
    tile_distances(
        rows,
        history_slots,
        reference,
        if params.correlated == 1 {
            reference_scales[tile.leaf as usize]
        } else {
            1.0
        },
        candidate,
        seeds[candidate as usize],
        leaves[tile.leaf as usize],
        tile,
        values,
        distances,
        warp_status,
        params,
    );
    let invalid = warp_invalid(warp_status);
    if thread_index == 0 {
        let mut flags = 0;
        let mut index = 0;
        while index < WARPS {
            flags |= unsafe { warp_status.add(index).read() };
            index += 1;
        }
        unsafe { *tile_status.get_unchecked_mut(block_index as usize) = flags };
    }
    if thread_index < params.history {
        let output = block_index as usize * params.history as usize + thread_index as usize;
        unsafe {
            *partials.get_unchecked_mut(output) = if invalid {
                f32::INFINITY
            } else {
                distances.add(thread_index as usize).read()
            };
        }
    }
}

#[inline(always)]
unsafe fn pair_history(
    rows: &[u16],
    history_slots: &[u32],
    leaf: Bf16Leaf,
    tile: DenseTile,
    tile_offset: u32,
    values: *const f32,
    distances: *mut f32,
    params: Bf16Score,
) {
    let thread_index = thread::threadIdx_x();
    let lane = warp::lane_id();
    let warp_index = thread_index / 32;
    let tile_length = (tile.length - tile_offset).min(THREADS);
    let mut history_base = 0u32;
    while history_base < params.history {
        let observation = history_base + warp_index;
        let mut first_sum = 0.0f32;
        let mut second_sum = 0.0f32;
        if observation < params.history {
            let row_offset =
                history_slots[observation as usize] as usize * params.row_stride as usize;
            let mut item = lane;
            while item < tile_length {
                let element = u64::from(tile.start + tile_offset + item);
                let observed = bf16_decode(rows[row_offset + (leaf.offset + element) as usize]);
                let first_delta = unsafe { values.add(item as usize).read() } - observed;
                let second_delta =
                    unsafe { values.add(THREADS as usize + item as usize).read() } - observed;
                first_sum += first_delta * first_delta * leaf.weight;
                second_sum += second_delta * second_delta * leaf.weight;
                item += 32;
            }
        }
        let first_partial = warp::reduce_sum_f32(first_sum);
        let second_partial = warp::reduce_sum_f32(second_sum);
        if lane == 0 && observation < params.history {
            unsafe {
                let first = distances.add(observation as usize);
                first.write(first.read() + first_partial);
                let second = distances.add(MAX_HISTORY + observation as usize);
                second.write(second.read() + second_partial);
            }
        }
        history_base += WARPS as u32;
    }
}

#[inline(always)]
unsafe fn pair_accumulate(
    rows: &[u16],
    history_slots: &[u32],
    reference: &[u16],
    seeds: &[Seed],
    leaf: Bf16Leaf,
    tile: DenseTile,
    pair_spec: (u32, f32, f32, f32),
    values: *mut f32,
    distances: *mut f32,
    warp_status: *mut u32,
    params: Bf16Score,
) {
    let (first_candidate, first_coefficient, second_coefficient, inverse_rms) = pair_spec;
    let thread_index = thread::threadIdx_x();
    let lane = warp::lane_id();
    let warp_index = thread_index / 32;
    let second_candidate = first_candidate + 1;
    let base_offset = params.base_slot as usize * params.row_stride as usize;
    let mut tile_offset = 0u32;
    while tile_offset < tile.length {
        let local = tile_offset + thread_index;
        let mut first_invalid = false;
        let mut second_invalid = false;
        let mut first_changed = false;
        let mut second_changed = false;
        if local < tile.length {
            let element = u64::from(tile.start + local);
            let index = leaf.offset + element;
            let base = rows[base_offset + index as usize];
            let first = bf16_candidate(
                base,
                leaf,
                element,
                seeds[first_candidate as usize],
                first_coefficient,
                reference,
                inverse_rms,
                params.correlated,
                first_candidate,
            );
            let second = bf16_candidate(
                base,
                leaf,
                element,
                seeds[second_candidate as usize],
                second_coefficient,
                reference,
                inverse_rms,
                params.correlated,
                second_candidate,
            );
            first_invalid = !bf16_finite(first);
            second_invalid = !bf16_finite(second);
            first_changed = first != base;
            second_changed = second != base;
            unsafe {
                values.add(thread_index as usize).write(bf16_decode(first));
                values
                    .add(THREADS as usize + thread_index as usize)
                    .write(bf16_decode(second));
            }
        }
        let first_flags =
            u32::from(warp::any(first_invalid)) | (u32::from(warp::any(first_changed)) << 1);
        let second_flags =
            u32::from(warp::any(second_invalid)) | (u32::from(warp::any(second_changed)) << 1);
        if lane == 0 {
            unsafe {
                let status = warp_status.add(warp_index as usize);
                status.write(status.read() | first_flags);
                let status = warp_status.add(WARPS + warp_index as usize);
                status.write(status.read() | second_flags);
            }
        }
        thread::sync_threads();
        unsafe {
            pair_history(
                rows,
                history_slots,
                leaf,
                tile,
                tile_offset,
                values,
                distances,
                params,
            );
        }
        thread::sync_threads();
        tile_offset += THREADS;
    }
}

#[inline(always)]
unsafe fn pair_export(
    distances: *const f32,
    warp_status: *const u32,
    partials: &mut DisjointSlice<f32>,
    tile_status: &mut DisjointSlice<u32>,
    first_candidate: u32,
    tile_index: u32,
    params: Bf16Score,
) {
    let thread_index = thread::threadIdx_x();
    let second_candidate = first_candidate + 1;
    let first_invalid = warp_invalid(warp_status);
    let second_invalid = warp_invalid(unsafe { warp_status.add(WARPS) });
    if thread_index == 0 {
        let mut first_flags = 0;
        let mut second_flags = 0;
        let mut index = 0;
        while index < WARPS {
            unsafe {
                first_flags |= warp_status.add(index).read();
                second_flags |= warp_status.add(WARPS + index).read();
            }
            index += 1;
        }
        unsafe {
            *tile_status
                .get_unchecked_mut((first_candidate * params.tiles + tile_index) as usize) =
                first_flags;
            *tile_status
                .get_unchecked_mut((second_candidate * params.tiles + tile_index) as usize) =
                second_flags;
        }
    }
    if thread_index < params.history {
        let first_output = ((first_candidate * params.tiles + tile_index) * params.history
            + thread_index) as usize;
        let second_output = ((second_candidate * params.tiles + tile_index) * params.history
            + thread_index) as usize;
        unsafe {
            *partials.get_unchecked_mut(first_output) = if first_invalid {
                f32::INFINITY
            } else {
                distances.add(thread_index as usize).read()
            };
            *partials.get_unchecked_mut(second_output) = if second_invalid {
                f32::INFINITY
            } else {
                distances.add(MAX_HISTORY + thread_index as usize).read()
            };
        }
    }
}

#[inline(always)]
pub(crate) fn run_dist2bf16(
    rows: &[u16],
    history_slots: &[u32],
    reference: &[u16],
    reference_scales: &[f32],
    state: &[SearchState],
    seeds: &[Seed],
    leaves: &[Bf16Leaf],
    tiles: &[DenseTile],
    partials: &mut DisjointSlice<f32>,
    tile_status: &mut DisjointSlice<u32>,
    mut params: Bf16Score,
) {
    static mut VALUES: SharedArray<f32, 512> = SharedArray::UNINIT;
    static mut DISTANCES: SharedArray<f32, 256> = SharedArray::UNINIT;
    static mut WARP_STATUS: SharedArray<u32, 16> = SharedArray::UNINIT;

    if params.resident != 0 {
        params.history = state[0].history;
        params.coefficient = state[0].length as f32;
        params.neighbors = params.neighbors.min(params.history);
    }
    let block_index = thread::blockIdx_x();
    let pairs = params.candidates.div_ceil(2);
    if block_index >= pairs * params.tiles {
        return;
    }
    let first_candidate = (block_index / params.tiles) * 2;
    let second_candidate = first_candidate + 1;
    let tile_index = block_index % params.tiles;
    let thread_index = thread::threadIdx_x();
    let values = unsafe { SharedArray::as_raw_mut_ptr(&raw mut VALUES) };
    let distances = unsafe { SharedArray::as_raw_mut_ptr(&raw mut DISTANCES) };
    let warp_status = unsafe { SharedArray::as_raw_mut_ptr(&raw mut WARP_STATUS) };
    if thread_index < params.history {
        unsafe {
            distances.add(thread_index as usize).write(0.0);
            distances
                .add(MAX_HISTORY + thread_index as usize)
                .write(0.0);
        }
    }
    if warp::lane_id() == 0 {
        let warp_index = (thread_index / 32) as usize;
        unsafe {
            warp_status.add(warp_index).write(0);
            warp_status.add(WARPS + warp_index).write(0);
        }
    }
    thread::sync_threads();

    let tile = tiles[tile_index as usize];
    let leaf = leaves[tile.leaf as usize];
    let inverse_rms = if params.correlated == 1 {
        reference_scales[tile.leaf as usize]
    } else {
        1.0
    };
    let first_coefficient = if params.correlated == 1 {
        candidate_radius(state[0], first_candidate)
    } else {
        params.coefficient
    };
    let second_coefficient = if params.correlated == 1 {
        candidate_radius(state[0], second_candidate)
    } else {
        params.coefficient
    };
    unsafe {
        pair_accumulate(
            rows,
            history_slots,
            reference,
            seeds,
            leaf,
            tile,
            (
                first_candidate,
                first_coefficient,
                second_coefficient,
                inverse_rms,
            ),
            values,
            distances,
            warp_status,
            params,
        );
        pair_export(
            distances,
            warp_status,
            partials,
            tile_status,
            first_candidate,
            tile_index,
            params,
        );
    }
}

#[inline(always)]
fn scan_changes(tile_status: &[u32], candidate: u32, params: Bf16Score) -> u32 {
    let mut changes = 0u32;
    if params.correlated == 1 {
        let mut tile = thread::threadIdx_x();
        while tile < params.tiles {
            let mut other = 0;
            while other < params.candidates {
                let flags = tile_status[(other * params.tiles + tile) as usize];
                if flags & 2 != 0 {
                    changes |= if other == candidate { 3 } else { 1 };
                }
                other += 1;
            }
            tile += THREADS;
        }
    }
    let any_changed = warp::any(changes & 1 != 0);
    let own_changed = warp::any(changes & 2 != 0);
    u32::from(any_changed) | (u32::from(own_changed) << 1)
}

#[inline(always)]
pub(crate) fn run_scorebf16(
    partials: &[f32],
    tile_status: &[u32],
    outcomes: &[f32],
    variances: &[f32],
    state: &[SearchState],
    draws: &[f32],
    scores: &mut DisjointSlice<f32>,
    mut params: Bf16Score,
) {
    static mut DISTANCES: SharedArray<f32, MAX_HISTORY> = SharedArray::UNINIT;
    static mut NEAREST_DISTANCES: SharedArray<f32, MAX_HISTORY> = SharedArray::UNINIT;
    static mut NEAREST_INDICES: SharedArray<u32, MAX_HISTORY> = SharedArray::UNINIT;
    static mut CHANGED: SharedArray<u32, WARPS> = SharedArray::UNINIT;

    if params.resident != 0 {
        params.history = state[0].history;
        params.coefficient = state[0].length as f32;
        params.neighbors = params.neighbors.min(params.history);
    }
    let candidate = thread::blockIdx_x();
    if candidate >= params.candidates {
        return;
    }
    let thread_index = thread::threadIdx_x();
    let distances = unsafe { SharedArray::as_raw_mut_ptr(&raw mut DISTANCES) };
    let nearest_distances = unsafe { SharedArray::as_raw_mut_ptr(&raw mut NEAREST_DISTANCES) };
    let nearest_indices = unsafe { SharedArray::as_raw_mut_ptr(&raw mut NEAREST_INDICES) };
    let warp_changes = scan_changes(tile_status, candidate, params);
    if warp::lane_id() == 0 {
        unsafe { CHANGED[(thread_index / 32) as usize] = warp_changes };
    }
    if thread_index < params.history {
        let mut distance = 0.0f32;
        let mut tile_index = 0u32;
        while tile_index < params.tiles {
            let block_index = candidate * params.tiles + tile_index;
            let input = block_index as usize * params.history as usize + thread_index as usize;
            distance += partials[input];
            tile_index += 1;
        }
        unsafe { distances.add(thread_index as usize).write(distance) };
    }
    thread::sync_threads();

    if thread_index == 0 {
        let mut changed = 0;
        let mut index = 0;
        while index < WARPS {
            changed |= unsafe { CHANGED[index] };
            index += 1;
        }
        let score = if unsafe { !distances.read().is_finite() } || changed == 1 {
            f32::NEG_INFINITY
        } else {
            acquisition_score(
                distances,
                nearest_distances,
                nearest_indices,
                outcomes,
                variances,
                draws,
                params,
            )
        };
        unsafe {
            *scores.get_unchecked_mut(candidate as usize) = score;
        }
    }
}
