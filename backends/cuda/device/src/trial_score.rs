use cuda_device::{DisjointSlice, SharedArray, thread, warp};

use crate::{
    MAX_HISTORY, THREADS, code_at, decode_code, perturb_code, resolve_center, sparse_code,
};

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct Seed {
    pub low: u32,
    pub high: u32,
}

// SAFETY: Seed is repr(C) and contains only DeviceCopy scalars.
unsafe impl cuda_core::DeviceCopy for Seed {}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct CenterStep {
    pub parent: u32,
    pub seed: Seed,
}

// SAFETY: CenterStep is repr(C) and contains only DeviceCopy fields.
unsafe impl cuda_core::DeviceCopy for CenterStep {}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct Selection {
    pub index: u32,
    pub score: f32,
}

// SAFETY: Selection is repr(C) and contains only DeviceCopy scalars.
unsafe impl cuda_core::DeviceCopy for Selection {}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct SparseEdit {
    pub leaf: u32,
    pub element: u32,
}

// SAFETY: SparseEdit is repr(C) and contains only DeviceCopy scalars.
unsafe impl cuda_core::DeviceCopy for SparseEdit {}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct Leaf {
    pub byte_offset: u32,
    pub element_offset: u32,
    pub length: u32,
    pub bits: u32,
    pub encoding: u32,
    pub scale: f32,
    pub weight: f32,
    pub whole: u32,
    pub threshold: u32,
}

// SAFETY: Leaf is repr(C) and contains only DeviceCopy scalars.
unsafe impl cuda_core::DeviceCopy for Leaf {}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct Tile {
    pub leaf: u32,
    pub start: u32,
    pub length: u32,
    pub pad: u32,
}

// SAFETY: Tile is repr(C) and contains only DeviceCopy scalars.
unsafe impl cuda_core::DeviceCopy for Tile {}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct TrialParams {
    pub row_stride: u32,
    pub history: u32,
    pub candidates: u32,
    pub base_slot: u32,
    pub center_count: u32,
    pub neighbors: u32,
    pub acquisition: u32,
    pub epistemic_scale: f32,
    pub aleatoric_scale: f32,
    pub y_scale: f32,
    pub beta: f32,
}

// SAFETY: TrialParams is repr(C) and contains only DeviceCopy scalars.
unsafe impl cuda_core::DeviceCopy for TrialParams {}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct SparseParams {
    pub row_stride: u32,
    pub history: u32,
    pub candidates: u32,
    pub num_pert: u32,
    pub base_slot: u32,
    pub neighbors: u32,
    pub acquisition: u32,
    pub epistemic_scale: f32,
    pub aleatoric_scale: f32,
    pub y_scale: f32,
    pub beta: f32,
}

// SAFETY: SparseParams is repr(C) and contains only DeviceCopy scalars.
unsafe impl cuda_core::DeviceCopy for SparseParams {}

#[inline(always)]
unsafe fn nibble_history(
    rows: &[u8],
    history_slots: &[u32],
    candidate_values: *const f32,
    distances: *mut f32,
    leaf: Leaf,
    tile_byte: u32,
    params: TrialParams,
) {
    let thread_index = thread::threadIdx_x();
    let lane = warp::lane_id();
    let warp_index = thread_index / 32;
    let tile_elements = (leaf.length - tile_byte * 2).min(THREADS * 2);
    let tile_bytes = tile_elements.div_ceil(2);
    let mut history_base = 0;
    while history_base < params.history {
        let h = history_base + warp_index;
        let mut sum = 0.0_f32;
        if h < params.history {
            let observation_offset =
                history_slots[h as usize] as usize * params.row_stride as usize;
            let mut local = lane;
            while local < tile_bytes {
                let observed = rows[observation_offset
                    + leaf.byte_offset as usize
                    + tile_byte as usize
                    + local as usize];
                let first = local * 2;
                let low_delta = unsafe { candidate_values.add(first as usize).read() }
                    - decode_code(u32::from(observed & 0x0f), leaf);
                sum += low_delta * low_delta * leaf.weight;
                if first + 1 < tile_elements {
                    let high_delta = unsafe { candidate_values.add((first + 1) as usize).read() }
                        - decode_code(u32::from(observed >> 4), leaf);
                    sum += high_delta * high_delta * leaf.weight;
                }
                local += 32;
            }
        }
        let partial = warp::reduce_sum_f32(sum);
        if lane == 0 && h < params.history {
            unsafe {
                let slot = distances.add(h as usize);
                slot.write(slot.read() + partial);
            }
        }
        history_base += 8;
    }
}

#[inline(always)]
unsafe fn score_nibble(
    rows: &[u8],
    history_slots: &[u32],
    centers: &[CenterStep],
    candidate_values: *mut f32,
    distances: *mut f32,
    leaf: Leaf,
    seed: Seed,
    center: u32,
    params: TrialParams,
) {
    let thread_index = thread::threadIdx_x();
    let base_offset = params.base_slot as usize * params.row_stride as usize;
    let bytes = leaf.length.div_ceil(2);
    let mut tile_byte = 0;
    while tile_byte < bytes {
        let byte = tile_byte + thread_index;
        if byte < bytes {
            let first = byte * 2;
            let base_byte = rows[base_offset + leaf.byte_offset as usize + byte as usize];
            let low_base = resolve_center(
                u32::from(base_byte & 0x0f),
                centers,
                center,
                leaf.element_offset + first,
                leaf,
            );
            let low = perturb_code(
                low_base,
                seed.low,
                seed.high,
                leaf.element_offset + first,
                leaf.bits,
                leaf.whole,
                leaf.threshold,
            );
            unsafe {
                candidate_values
                    .add((thread_index * 2) as usize)
                    .write(decode_code(low, leaf));
            }
            if first + 1 < leaf.length {
                let high_base = resolve_center(
                    u32::from(base_byte >> 4),
                    centers,
                    center,
                    leaf.element_offset + first + 1,
                    leaf,
                );
                let high = perturb_code(
                    high_base,
                    seed.low,
                    seed.high,
                    leaf.element_offset + first + 1,
                    leaf.bits,
                    leaf.whole,
                    leaf.threshold,
                );
                unsafe {
                    candidate_values
                        .add((thread_index * 2 + 1) as usize)
                        .write(decode_code(high, leaf));
                }
            }
        }
        thread::sync_threads();
        unsafe {
            nibble_history(
                rows,
                history_slots,
                candidate_values,
                distances,
                leaf,
                tile_byte,
                params,
            );
        }
        thread::sync_threads();
        tile_byte += THREADS;
    }
}

#[inline(always)]
unsafe fn score_byte(
    rows: &[u8],
    history_slots: &[u32],
    centers: &[CenterStep],
    candidate_values: *mut f32,
    distances: *mut f32,
    leaf: Leaf,
    seed: Seed,
    center: u32,
    params: TrialParams,
) {
    let thread_index = thread::threadIdx_x();
    let lane = warp::lane_id();
    let warp_index = thread_index / 32;
    let base_offset = params.base_slot as usize * params.row_stride as usize;
    let mut tile_element = 0;
    while tile_element < leaf.length {
        let element = tile_element + thread_index;
        if element < leaf.length {
            let base_code = resolve_center(
                code_at(rows, base_offset, leaf, element),
                centers,
                center,
                leaf.element_offset + element,
                leaf,
            );
            let value = perturb_code(
                base_code,
                seed.low,
                seed.high,
                leaf.element_offset + element,
                leaf.bits,
                leaf.whole,
                leaf.threshold,
            );
            unsafe {
                candidate_values
                    .add(thread_index as usize)
                    .write(decode_code(value, leaf));
            }
        }
        thread::sync_threads();

        let tile_elements = (leaf.length - tile_element).min(THREADS);
        let mut history_base = 0;
        while history_base < params.history {
            let h = history_base + warp_index;
            let mut sum = 0.0_f32;
            if h < params.history {
                let observation_offset =
                    history_slots[h as usize] as usize * params.row_stride as usize;
                let mut local = lane;
                while local < tile_elements {
                    let observed_code =
                        code_at(rows, observation_offset, leaf, tile_element + local);
                    let delta = unsafe { candidate_values.add(local as usize).read() }
                        - decode_code(observed_code, leaf);
                    sum += delta * delta * leaf.weight;
                    local += 32;
                }
            }
            let partial = warp::reduce_sum_f32(sum);
            if lane == 0 && h < params.history {
                unsafe {
                    let slot = distances.add(h as usize);
                    slot.write(slot.read() + partial);
                }
            }
            history_base += 8;
        }
        thread::sync_threads();
        tile_element += THREADS;
    }
}

#[inline(always)]
unsafe fn insert_trial(
    nearest_distances: *mut f32,
    nearest_indices: *mut u32,
    neighbors: u32,
    observation: u32,
    distance: f32,
) {
    let mut insert_at = neighbors;
    let mut nearest = 0;
    while nearest < neighbors {
        let nearest_distance = unsafe { nearest_distances.add(nearest as usize).read() };
        let nearest_index = unsafe { nearest_indices.add(nearest as usize).read() };
        if distance < nearest_distance
            || (distance == nearest_distance && observation < nearest_index)
        {
            insert_at = nearest;
            break;
        }
        nearest += 1;
    }
    if insert_at < neighbors {
        let mut move_index = neighbors - 1;
        while move_index > insert_at {
            unsafe {
                nearest_distances
                    .add(move_index as usize)
                    .write(nearest_distances.add((move_index - 1) as usize).read());
                nearest_indices
                    .add(move_index as usize)
                    .write(nearest_indices.add((move_index - 1) as usize).read());
            }
            move_index -= 1;
        }
        unsafe {
            nearest_distances.add(insert_at as usize).write(distance);
            nearest_indices.add(insert_at as usize).write(observation);
        }
    }
}

#[inline(always)]
unsafe fn trial_acquisition(
    distances: *const f32,
    nearest_distances: *mut f32,
    nearest_indices: *mut u32,
    outcomes: &[f32],
    draws: &[f32],
    history: u32,
    neighbors: u32,
    acquisition: u32,
    epistemic_scale: f32,
    aleatoric_scale: f32,
    y_scale: f32,
    beta: f32,
) -> f32 {
    let mut k = 0;
    while k < neighbors {
        unsafe {
            nearest_distances.add(k as usize).write(f32::INFINITY);
            nearest_indices.add(k as usize).write(0);
        }
        k += 1;
    }
    let mut h = 0;
    while h < history {
        unsafe {
            insert_trial(
                nearest_distances,
                nearest_indices,
                neighbors,
                h,
                distances.add(h as usize).read(),
            );
        }
        h += 1;
    }
    let mut weight_sum = 0.0_f32;
    let mut weighted_value = 0.0_f32;
    let mut weighted_noise = 0.0_f32;
    let mut weight_squared_sum = 0.0_f32;
    let reference_weight = if acquisition == 1 {
        let distance = unsafe { nearest_distances.read() };
        let variance = 1.0e-9 + epistemic_scale * distance + aleatoric_scale;
        (1.0_f32 / variance.max(1.0e-12)).max(f32::MIN_POSITIVE)
    } else {
        1.0
    };
    k = 0;
    while k < neighbors {
        let distance = unsafe { nearest_distances.add(k as usize).read() };
        let index = unsafe { nearest_indices.add(k as usize).read() } as usize;
        let variance = 1.0e-9 + epistemic_scale * distance + aleatoric_scale;
        let weight = 1.0 / variance.max(1.0e-12);
        weight_sum += weight;
        weighted_value += weight * outcomes[index];
        if acquisition == 1 {
            let draw_weight = weight / reference_weight;
            weighted_noise += draw_weight * draws[index];
            weight_squared_sum += draw_weight * draw_weight;
        }
        k += 1;
    }
    let mean = weighted_value / weight_sum.max(1.0e-12);
    let se = (1.0 / weight_sum.max(1.0e-12)).sqrt() * y_scale;
    match acquisition {
        1 => mean + se * (weighted_noise / weight_squared_sum.sqrt().max(1.0e-12)),
        2 => mean + se,
        _ => mean + beta * se,
    }
}

#[inline(always)]
pub(crate) fn run_trials(
    rows: &[u8],
    history_slots: &[u32],
    outcomes: &[f32],
    seeds: &[Seed],
    draws: &[f32],
    leaves: &[Leaf],
    centers: &[CenterStep],
    candidate_centers: &[u32],
    scores: &mut DisjointSlice<f32>,
    params: TrialParams,
) {
    static mut CANDIDATE_VALUES: SharedArray<f32, 512> = SharedArray::UNINIT;
    static mut DISTANCES: SharedArray<f32, MAX_HISTORY> = SharedArray::UNINIT;
    static mut NEAREST_DISTANCES: SharedArray<f32, MAX_HISTORY> = SharedArray::UNINIT;
    static mut NEAREST_INDICES: SharedArray<u32, MAX_HISTORY> = SharedArray::UNINIT;

    let candidate = thread::blockIdx_x();
    if candidate >= params.candidates {
        return;
    }
    let thread_index = thread::threadIdx_x();
    let candidate_values = unsafe { SharedArray::as_raw_mut_ptr(&raw mut CANDIDATE_VALUES) };
    let distances = unsafe { SharedArray::as_raw_mut_ptr(&raw mut DISTANCES) };
    let nearest_distances = unsafe { SharedArray::as_raw_mut_ptr(&raw mut NEAREST_DISTANCES) };
    let nearest_indices = unsafe { SharedArray::as_raw_mut_ptr(&raw mut NEAREST_INDICES) };
    if thread_index < params.history {
        unsafe { distances.add(thread_index as usize).write(0.0) };
    }
    thread::sync_threads();

    let seed = seeds[candidate as usize];
    let center = if params.center_count == 0 {
        u32::MAX
    } else {
        candidate_centers[candidate as usize]
    };

    let mut leaf_index = 0;
    while leaf_index < leaves.len() {
        let leaf = leaves[leaf_index];
        unsafe {
            if leaf.bits == 4 {
                score_nibble(
                    rows,
                    history_slots,
                    centers,
                    candidate_values,
                    distances,
                    leaf,
                    seed,
                    center,
                    params,
                );
            } else {
                score_byte(
                    rows,
                    history_slots,
                    centers,
                    candidate_values,
                    distances,
                    leaf,
                    seed,
                    center,
                    params,
                );
            }
        }
        leaf_index += 1;
    }

    if thread_index == 0 {
        let score = unsafe {
            trial_acquisition(
                distances,
                nearest_distances,
                nearest_indices,
                outcomes,
                draws,
                params.history,
                params.neighbors,
                params.acquisition,
                params.epistemic_scale,
                params.aleatoric_scale,
                params.y_scale,
                params.beta,
            )
        };
        unsafe {
            *scores.get_unchecked_mut(candidate as usize) = score;
        }
    }
}

#[inline(always)]
pub(crate) fn run_sparse(
    rows: &[u8],
    history_slots: &[u32],
    outcomes: &[f32],
    seeds: &[Seed],
    draws: &[f32],
    leaves: &[Leaf],
    edits: &[SparseEdit],
    base_distances: &[f32],
    scores: &mut DisjointSlice<f32>,
    params: SparseParams,
) {
    static mut DISTANCES: SharedArray<f32, MAX_HISTORY> = SharedArray::UNINIT;
    static mut NEAREST_DISTANCES: SharedArray<f32, MAX_HISTORY> = SharedArray::UNINIT;
    static mut NEAREST_INDICES: SharedArray<u32, MAX_HISTORY> = SharedArray::UNINIT;

    let candidate = thread::blockIdx_x();
    if candidate >= params.candidates {
        return;
    }
    let thread_index = thread::threadIdx_x();
    let distances = unsafe { SharedArray::as_raw_mut_ptr(&raw mut DISTANCES) };
    let nearest_distances = unsafe { SharedArray::as_raw_mut_ptr(&raw mut NEAREST_DISTANCES) };
    let nearest_indices = unsafe { SharedArray::as_raw_mut_ptr(&raw mut NEAREST_INDICES) };
    if thread_index < params.history {
        let observation_offset =
            history_slots[thread_index as usize] as usize * params.row_stride as usize;
        let base_offset = params.base_slot as usize * params.row_stride as usize;
        let seed = seeds[candidate as usize];
        let mut distance = base_distances[thread_index as usize];
        let mut edit_index = 0;
        while edit_index < params.num_pert {
            let edit = edits[(candidate * params.num_pert + edit_index) as usize];
            let leaf = leaves[edit.leaf as usize];
            let base_code = code_at(rows, base_offset, leaf, edit.element);
            let observed_code = code_at(rows, observation_offset, leaf, edit.element);
            let candidate_code =
                sparse_code(base_code, seed, leaf.element_offset + edit.element, leaf);
            let base_delta = decode_code(base_code, leaf) - decode_code(observed_code, leaf);
            let candidate_delta =
                decode_code(candidate_code, leaf) - decode_code(observed_code, leaf);
            distance += (candidate_delta * candidate_delta - base_delta * base_delta) * leaf.weight;
            edit_index += 1;
        }
        unsafe {
            distances
                .add(thread_index as usize)
                .write(distance.max(0.0))
        };
    }
    thread::sync_threads();
    if thread_index == 0 {
        let score = unsafe {
            trial_acquisition(
                distances,
                nearest_distances,
                nearest_indices,
                outcomes,
                draws,
                params.history,
                params.neighbors,
                params.acquisition,
                params.epistemic_scale,
                params.aleatoric_scale,
                params.y_scale,
                params.beta,
            )
        };
        unsafe { *scores.get_unchecked_mut(candidate as usize) = score };
    }
}
