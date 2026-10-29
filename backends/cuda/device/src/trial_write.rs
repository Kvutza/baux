use cuda_device::{DisjointSlice, SharedArray, thread, warp};

use crate::{
    DenseLeaf, DenseLinearParams, DenseTerm, DenseTile, Leaf, Seed, Selection, SparseEdit, THREADS,
    Tile, code_at, code_ptr, decode_code, dense_value, perturb_code, reduce_best, sparse_code,
};

#[inline(always)]
pub(crate) fn run_writetrial(
    rows: &mut DisjointSlice<u8>,
    seeds: &[Seed],
    selection: &[Selection],
    leaves: &[Leaf],
    tiles: &[Tile],
    row_stride: u32,
    base_slot: u32,
    trial_slot: u32,
) {
    let tile_index = thread::blockIdx_x();
    if tile_index as usize >= tiles.len() {
        return;
    }
    let tile = tiles[tile_index as usize];
    let leaf = leaves[tile.leaf as usize];
    let seed = seeds[selection[0].index as usize];
    let thread_index = thread::threadIdx_x();
    let base_offset = base_slot as usize * row_stride as usize;
    let trial_offset = trial_slot as usize * row_stride as usize;
    let rows_ptr = rows.as_mut_ptr() as *const u8;

    if leaf.bits == 4 {
        let first_byte = tile.start / 2;
        let bytes = tile.length.div_ceil(2);
        let mut local_byte = thread_index;
        while local_byte < bytes {
            let first = tile.start + local_byte * 2;
            let low = perturb_code(
                unsafe { code_ptr(rows_ptr, base_offset, leaf, first) },
                seed.low,
                seed.high,
                leaf.element_offset + first,
                leaf.bits,
                leaf.whole,
                leaf.threshold,
            );
            let high = if first + 1 < leaf.length {
                perturb_code(
                    unsafe { code_ptr(rows_ptr, base_offset, leaf, first + 1) },
                    seed.low,
                    seed.high,
                    leaf.element_offset + first + 1,
                    leaf.bits,
                    leaf.whole,
                    leaf.threshold,
                )
            } else {
                0
            };
            let output = trial_offset
                + leaf.byte_offset as usize
                + first_byte as usize
                + local_byte as usize;
            unsafe {
                *rows.get_unchecked_mut(output) = (low | (high << 4)) as u8;
            }
            local_byte += THREADS;
        }
    } else {
        let end = tile.start + tile.length;
        let mut element = tile.start + thread_index;
        while element < end {
            let value = perturb_code(
                unsafe { code_ptr(rows_ptr, base_offset, leaf, element) },
                seed.low,
                seed.high,
                leaf.element_offset + element,
                leaf.bits,
                leaf.whole,
                leaf.threshold,
            );
            let output = trial_offset + leaf.byte_offset as usize + element as usize;
            unsafe {
                *rows.get_unchecked_mut(output) = value as u8;
            }
            element += THREADS;
        }
    }
}

#[inline(always)]
pub(crate) fn run_writesparse(
    rows: &mut DisjointSlice<u8>,
    seeds: &[Seed],
    selection: &[Selection],
    leaves: &[Leaf],
    edits: &[SparseEdit],
    row_stride: u32,
    trial_slot: u32,
    num_pert: u32,
) {
    if thread::index_1d().get() != 0 {
        return;
    }
    let candidate = selection[0].index;
    let seed = seeds[candidate as usize];
    let trial_offset = trial_slot as usize * row_stride as usize;
    let rows_ptr = rows.as_mut_ptr() as *const u8;
    let mut edit_index = 0;
    while edit_index < num_pert {
        let edit = edits[(candidate * num_pert + edit_index) as usize];
        let leaf = leaves[edit.leaf as usize];
        let byte = trial_offset
            + leaf.byte_offset as usize
            + if leaf.bits == 4 {
                (edit.element / 2) as usize
            } else {
                edit.element as usize
            };
        let current = unsafe { rows_ptr.add(byte).read() };
        let shift = if leaf.bits == 4 {
            (edit.element & 1) * 4
        } else {
            0
        };
        let code = if leaf.bits == 4 {
            u32::from((current >> shift) & 0x0f)
        } else {
            u32::from(current)
        };
        let value = sparse_code(code, seed, leaf.element_offset + edit.element, leaf);
        unsafe {
            *rows.get_unchecked_mut(byte) = if leaf.bits == 4 {
                (current & !(0x0f << shift)) | ((value as u8) << shift)
            } else {
                value as u8
            };
        }
        edit_index += 1;
    }
}

#[inline(always)]
pub(crate) fn run_materialize(
    base: &[u8],
    output: &mut DisjointSlice<u8>,
    byte_offset: u32,
    element_offset: u32,
    length: u32,
    bits: u32,
    seed_low: u32,
    seed_high: u32,
    whole: u32,
    threshold: u32,
) {
    let thread_index = thread::index_1d().get();
    let local = thread_index as u32;

    if bits == 4 {
        let byte_count = length.div_ceil(2);
        if local >= byte_count {
            return;
        }
        let first = local * 2;
        let byte_index = byte_offset as usize + thread_index;
        let base_byte = base[byte_index];
        let low = perturb_code(
            u32::from(base_byte & 0x0f),
            seed_low,
            seed_high,
            element_offset + first,
            bits,
            whole,
            threshold,
        );
        let high = if first + 1 < length {
            perturb_code(
                u32::from(base_byte >> 4),
                seed_low,
                seed_high,
                element_offset + first + 1,
                bits,
                whole,
                threshold,
            )
        } else {
            0
        };
        unsafe {
            *output.get_unchecked_mut(byte_index) = (low | (high << 4)) as u8;
        }
    } else {
        if local >= length {
            return;
        }
        let byte_index = byte_offset as usize + thread_index;
        let value = perturb_code(
            u32::from(base[byte_index]),
            seed_low,
            seed_high,
            element_offset + local,
            bits,
            whole,
            threshold,
        );
        unsafe {
            *output.get_unchecked_mut(byte_index) = value as u8;
        }
    }
}

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
pub(crate) fn run_basedist(
    rows: &[u8],
    history_slots: &[u32],
    leaves: &[Leaf],
    distances: &mut DisjointSlice<f32>,
    row_stride: u32,
    history: u32,
    base_slot: u32,
) {
    static mut PARTIALS: SharedArray<f32, 256> = SharedArray::UNINIT;
    let observation = thread::blockIdx_x();
    if observation >= history {
        return;
    }
    let thread_index = thread::threadIdx_x();
    let base_offset = base_slot as usize * row_stride as usize;
    let row_offset = history_slots[observation as usize] as usize * row_stride as usize;
    let mut sum = 0.0_f32;
    let mut leaf_index = 0usize;
    while leaf_index < leaves.len() {
        let leaf = leaves[leaf_index];
        let mut element = thread_index;
        while element < leaf.length {
            let delta = decode_code(code_at(rows, base_offset, leaf, element), leaf)
                - decode_code(code_at(rows, row_offset, leaf, element), leaf);
            sum += delta * delta * leaf.weight;
            element += THREADS;
        }
        leaf_index += 1;
    }
    unsafe { PARTIALS[thread_index as usize] = sum };
    thread::sync_threads();
    let mut width = THREADS / 2;
    while width > 0 {
        if thread_index < width {
            unsafe {
                PARTIALS[thread_index as usize] += PARTIALS[(thread_index + width) as usize];
            }
        }
        thread::sync_threads();
        width /= 2;
    }
    if thread_index == 0 {
        unsafe {
            *distances.get_unchecked_mut(observation as usize) = PARTIALS[0];
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
