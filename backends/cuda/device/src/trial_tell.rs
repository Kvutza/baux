use cuda_device::{DisjointSlice, SharedArray, thread, warp};

use crate::{
    Bf16Change, Bf16Leaf, DenseLeaf, DenseTerm, DenseTile, SearchState, Seed, Selection, THREADS,
    TellParams, TellSummary, WARPS, WriteParams, bf16, rotate_history, update_correlated_radius,
    validate_tell_params, write_tell_state,
};

macro_rules! tell_validate {
    ($values:ident, $trial_variances:ident, $accepted:ident, $trials:expr) => {{
        let mut status = 0_u32;
        let mut trial = 0_u32;
        while trial < $trials {
            let value = $values[trial as usize];
            let variance = $trial_variances[trial as usize];
            if !value.is_finite() || !variance.is_finite() || variance < 0.0 {
                status = 1;
            }
            unsafe { *$accepted.get_unchecked_mut(trial as usize) = 0 };
            trial += 1;
        }
        status
    }};
}

macro_rules! tell_history {
    (
        $local:ident,
        $history_slots:ident,
        $outcomes:ident,
        $variances:ident,
        $destinations:ident,
        $values:ident,
        $trial_variances:ident,
        $accepted:ident,
        $params:ident
    ) => {{
        let slots_ptr = $history_slots.as_mut_ptr() as *const u32;
        let outcomes_ptr = $outcomes.as_mut_ptr() as *const f32;
        let variances_ptr = $variances.as_mut_ptr() as *const f32;
        let mut best_slot = u32::MAX;
        let mut trial = 0_u32;
        while trial < $params.trials {
            let value = $values[trial as usize];
            let variance = $trial_variances[trial as usize];
            let slot = $params.trial_slots[trial as usize];
            if value > $local.best {
                $local.best = value;
                $local.best_variance = variance;
                best_slot = slot;
                unsafe { *$accepted.get_unchecked_mut(trial as usize) = 1 };
            }
            let destination = if $local.history == $params.capacity {
                let oldest = unsafe { slots_ptr.read() };
                let mut index = 1_u32;
                while index < $local.history {
                    unsafe {
                        *$history_slots.get_unchecked_mut((index - 1) as usize) =
                            slots_ptr.add(index as usize).read();
                        *$outcomes.get_unchecked_mut((index - 1) as usize) =
                            outcomes_ptr.add(index as usize).read();
                        *$variances.get_unchecked_mut((index - 1) as usize) =
                            variances_ptr.add(index as usize).read();
                    }
                    index += 1;
                }
                $local.history -= 1;
                oldest
            } else {
                $local.history + 1
            };
            let index = $local.history as usize;
            unsafe {
                *$destinations.get_unchecked_mut(trial as usize) = destination;
                *$history_slots.get_unchecked_mut(index) = destination;
                *$outcomes.get_unchecked_mut(index) = value;
                *$variances.get_unchecked_mut(index) = variance;
            }
            $local.history += 1;
            trial += 1;
        }
        best_slot
    }};
}

// Slot 1 stays pinned to the incumbent; only rejected relative observations
// rotate.
macro_rules! tell_relativehistory {
    (
        $local:ident,
        $history_slots:ident,
        $outcomes:ident,
        $variances:ident,
        $destinations:ident,
        $params:ident
    ) => {{
        if $params.accept != 0 {
            $local.history = 1;
            unsafe {
                *$history_slots.get_unchecked_mut(0) = 1;
                *$outcomes.get_unchecked_mut(0) = 0.0;
                *$variances.get_unchecked_mut(0) = 0.0;
                *$destinations.get_unchecked_mut(0) = 1;
            }
        } else {
            let destination = if $local.history == $params.capacity {
                let slots_ptr = $history_slots.as_mut_ptr() as *const u32;
                let outcomes_ptr = $outcomes.as_mut_ptr() as *const f32;
                let variances_ptr = $variances.as_mut_ptr() as *const f32;
                let oldest = unsafe { slots_ptr.add(1).read() };
                let mut index = 2_u32;
                while index < $local.history {
                    unsafe {
                        *$history_slots.get_unchecked_mut((index - 1) as usize) =
                            slots_ptr.add(index as usize).read();
                        *$outcomes.get_unchecked_mut((index - 1) as usize) =
                            outcomes_ptr.add(index as usize).read();
                        *$variances.get_unchecked_mut((index - 1) as usize) =
                            variances_ptr.add(index as usize).read();
                    }
                    index += 1;
                }
                $local.history -= 1;
                oldest
            } else {
                $local.history + 1
            };
            let index = $local.history as usize;
            unsafe {
                *$destinations.get_unchecked_mut(0) = destination;
                *$history_slots.get_unchecked_mut(index) = destination;
                *$outcomes.get_unchecked_mut(index) = $params.improvement;
                *$variances.get_unchecked_mut(index) = $params.improvement_variance;
            }
            $local.history += 1;
        }
    }};
}

macro_rules! tell_adapt {
    (
        $local:ident,
        $values:ident,
        $history_slots:ident,
        $outcomes:ident,
        $variances:ident,
        $params:ident
    ) => {{
        let previous_best = $local.trust_best;
        if previous_best.is_finite() {
            let scale = if $local.prev_obs >= 2 {
                ($local.hist_max - $local.hist_min).max(1.0e-6)
            } else {
                0.0
            };
            if f64::from($local.best) > previous_best + 1.0e-3 * scale {
                $local.successes += 1;
                $local.failures = 0;
            } else {
                $local.failures += 1;
                $local.successes = 0;
            }
            if $local.successes >= 3 {
                $local.length = ($local.length * 2.0).min($local.length_max);
                $local.successes = 0;
            } else if $local.failures >= $params.failure_tolerance {
                $local.length *= 0.5;
                $local.failures = 0;
            }
        }
        $local.trust_best = f64::from($local.best);
        let mut trial = 0_u32;
        while trial < $params.trials {
            let value = f64::from($values[trial as usize]);
            $local.hist_min = $local.hist_min.min(value);
            $local.hist_max = $local.hist_max.max(value);
            trial += 1;
        }
        $local.prev_obs += u64::from($params.trials);
        let mut restarted = 0_u32;
        if $local.length < $local.length_min {
            $local.length = $local.length_init;
            $local.successes = 0;
            $local.failures = 0;
            $local.trust_best = f64::NEG_INFINITY;
            $local.hist_min = f64::INFINITY;
            $local.hist_max = f64::NEG_INFINITY;
            $local.history = 1;
            $local.restarts += 1;
            unsafe {
                *$history_slots.get_unchecked_mut(0) = 1;
                *$outcomes.get_unchecked_mut(0) = $local.best;
                *$variances.get_unchecked_mut(0) = $local.best_variance;
            }
            restarted = 1;
        }
        restarted
    }};
}

#[inline(always)]
fn write_tile(
    rows: &mut DisjointSlice<u16>,
    batch: &mut DisjointSlice<u16>,
    reference: &[u16],
    leaf: Bf16Leaf,
    tile: DenseTile,
    seed: Seed,
    candidate: u32,
    region: u32,
    coefficient: f32,
    inverse_rms: f32,
    mut invalid: bool,
    params: WriteParams,
) -> (bool, f32, f32) {
    let base_offset = params.base_slot as usize * params.row_stride as usize;
    let row_offset = params.trial_slots[region as usize] as usize * params.row_stride as usize;
    let rows_ptr = rows.as_mut_ptr() as *const u16;
    let mut squared = 0.0_f32;
    let mut changed = 0_u32;
    let mut item = thread::threadIdx_x();
    while item < tile.length {
        let element = u64::from(tile.start + item);
        let index = leaf.offset + element;
        let base = unsafe { rows_ptr.add(base_offset + index as usize).read() };
        let value = bf16::bf16_candidate(
            base,
            leaf,
            element,
            seed,
            coefficient,
            reference,
            inverse_rms,
            params.correlated,
            candidate,
        );
        invalid |= !bf16::bf16_finite(value);
        if params.verify_only != 0 {
            invalid |= unsafe { rows_ptr.add(row_offset + index as usize).read() } != value;
        }
        let delta = bf16::bf16_decode(value) - bf16::bf16_decode(base);
        squared += delta * delta;
        changed += u32::from(value != base);
        if params.verify_only == 0 {
            unsafe {
                *rows.get_unchecked_mut(row_offset + index as usize) = value;
                if params.export_batch != 0 {
                    *batch.get_unchecked_mut(
                        region as usize * params.row_len as usize + index as usize,
                    ) = value;
                }
            }
        }
        item += THREADS;
    }
    (
        warp::any(invalid),
        warp::reduce_sum_f32(squared),
        warp::reduce_sum_f32(changed as f32),
    )
}

#[inline(always)]
pub(crate) fn run_writebf16(
    rows: &mut DisjointSlice<u16>,
    batch: &mut DisjointSlice<u16>,
    changes: &mut DisjointSlice<Bf16Change>,
    state: &[SearchState],
    seeds: &[Seed],
    reference: &[u16],
    reference_scales: &[f32],
    selection: &[Selection],
    leaves: &[Bf16Leaf],
    tiles: &[DenseTile],
    status: &mut DisjointSlice<u32>,
    params: WriteParams,
) {
    static mut WARP_STATUS: SharedArray<u32, WARPS> = SharedArray::UNINIT;
    static mut WARP_SQUARED: SharedArray<f32, WARPS> = SharedArray::UNINIT;
    static mut WARP_CHANGED: SharedArray<f32, WARPS> = SharedArray::UNINIT;

    let block = thread::blockIdx_x();
    let region = block / params.tile_count;
    let tile_index = block % params.tile_count;
    if region as usize >= selection.len() || tile_index as usize >= tiles.len() {
        return;
    }
    let tile = tiles[tile_index as usize];
    let leaf = leaves[tile.leaf as usize];
    let candidate = selection[region as usize].index;
    let coefficient = if params.correlated == 1 {
        bf16::candidate_radius(state[0], candidate)
    } else if params.resident != 0 {
        state[0].length as f32
    } else {
        params.coefficient
    };
    let inverse_rms = if params.correlated == 1 {
        reference_scales[tile.leaf as usize]
    } else {
        1.0
    };
    let (warp_invalid, warp_squared, warp_changed) = write_tile(
        rows,
        batch,
        reference,
        leaf,
        tile,
        seeds[candidate as usize],
        candidate,
        region,
        coefficient,
        inverse_rms,
        params.resident != 0 && state[0].status != 0,
        params,
    );
    let thread_index = thread::threadIdx_x();
    let warp_index = (thread_index / 32) as usize;
    if warp::lane_id() == 0 {
        unsafe {
            WARP_STATUS[warp_index] = u32::from(warp_invalid);
            WARP_SQUARED[warp_index] = warp_squared;
            WARP_CHANGED[warp_index] = warp_changed;
        }
    }
    thread::sync_threads();
    if thread_index == 0 {
        let mut block_status = 0;
        let mut total_squared = 0.0_f32;
        let mut total_changed = 0_u32;
        let mut idx = 0;
        while idx < WARPS {
            block_status |= unsafe { WARP_STATUS[idx] };
            total_squared += unsafe { WARP_SQUARED[idx] };
            total_changed += unsafe { WARP_CHANGED[idx] } as u32;
            idx += 1;
        }
        unsafe {
            *status.get_unchecked_mut(block as usize) = block_status;
            *changes.get_unchecked_mut(block as usize) = Bf16Change {
                squared: total_squared,
                changed: total_changed,
            };
        }
    }
}

#[inline(always)]
pub(crate) fn run_tellbf16(
    history_slots: &mut DisjointSlice<u32>,
    outcomes: &mut DisjointSlice<f32>,
    variances: &mut DisjointSlice<f32>,
    destinations: &mut DisjointSlice<u32>,
    values: &[f32],
    trial_variances: &[f32],
    proposal_status: &[u32],
    accepted: &mut DisjointSlice<u32>,
    state: &mut DisjointSlice<SearchState>,
    summary: &mut DisjointSlice<TellSummary>,
    selection: &[Selection],
    params: TellParams,
) {
    if thread::threadIdx_x() != 0 {
        return;
    }
    let state_ptr = state.as_mut_ptr() as *const SearchState;
    let mut status = tell_validate!(values, trial_variances, accepted, params.trials);
    status |= validate_tell_params(&params, proposal_status);
    let mut best_slot = u32::MAX;
    let mut restarted = 0_u32;
    if status == 0 {
        let mut local = unsafe { state_ptr.read() };
        let selected_radius = bf16::candidate_radius(local, selection[0].index);
        if params.paired == 2 {
            local.best = params.incumbent_value;
            local.best_variance = params.incumbent_variance;
            if params.accept != 0 {
                local.best = values[0];
                local.best_variance = trial_variances[0];
                best_slot = params.trial_slots[0];
                unsafe { *accepted.get_unchecked_mut(0) = 1 };
            }
            tell_relativehistory!(
                local,
                history_slots,
                outcomes,
                variances,
                destinations,
                params
            );
        } else if params.paired != 0 {
            local.best = params.incumbent_value;
            local.best_variance = params.incumbent_variance;
            if params.accept != 0 {
                local.best = values[0];
                local.best_variance = trial_variances[0];
                best_slot = params.trial_slots[0];
                unsafe { *accepted.get_unchecked_mut(0) = 1 };
            }
            let destination = rotate_history(
                history_slots,
                outcomes,
                variances,
                &mut local,
                params.capacity,
            );
            let index = local.history as usize;
            unsafe {
                *destinations.get_unchecked_mut(0) = destination;
                *history_slots.get_unchecked_mut(index) = destination;
                *outcomes.get_unchecked_mut(index) = values[0];
                *variances.get_unchecked_mut(index) = trial_variances[0];
            }
            local.history += 1;
        } else {
            best_slot = tell_history!(
                local,
                history_slots,
                outcomes,
                variances,
                destinations,
                values,
                trial_variances,
                accepted,
                params
            );
        }
        if params.correlated == 1 {
            update_correlated_radius(&mut local, &params, selected_radius, best_slot);
        } else {
            restarted = tell_adapt!(local, values, history_slots, outcomes, variances, params);
        }
        local.status = 0;
        write_tell_state(state, summary, local, restarted, 0, best_slot);
    } else {
        let mut local = unsafe { state_ptr.read() };
        local.status = status;
        write_tell_state(state, summary, local, 0, status, best_slot);
    }
}

#[inline(always)]
pub(crate) fn run_copytell(
    rows: &mut DisjointSlice<u16>,
    destinations: &[u32],
    summary: &[TellSummary],
    params: TellParams,
) {
    if summary[0].status != 0 {
        return;
    }
    let best_slot = summary[0].best_slot;
    let rows_ptr = rows.as_mut_ptr() as *const u16;
    let mut index =
        u64::from(thread::blockIdx_x()) * u64::from(THREADS) + u64::from(thread::threadIdx_x());
    let stride = u64::from(thread::gridDim_x()) * u64::from(THREADS);
    while index < params.row_len {
        if best_slot != u32::MAX {
            let source = u64::from(best_slot) * params.row_stride + index;
            let value = unsafe { rows_ptr.add(source as usize).read() };
            unsafe { *rows.get_unchecked_mut(index as usize) = value };
        }
        let mut trial = 0;
        while trial < params.trials {
            let source = u64::from(params.trial_slots[trial as usize]) * params.row_stride + index;
            let destination = u64::from(destinations[trial as usize]) * params.row_stride + index;
            let value = unsafe { rows_ptr.add(source as usize).read() };
            unsafe { *rows.get_unchecked_mut(destination as usize) = value };
            trial += 1;
        }
        if summary[0].restarted != 0 {
            let value = unsafe { rows_ptr.add(index as usize).read() };
            unsafe { *rows.get_unchecked_mut((params.row_stride + index) as usize) = value };
        }
        index += stride;
    }
}

#[inline(always)]
pub(crate) fn run_refbf16(
    reference: &mut DisjointSlice<u16>,
    partials: &mut DisjointSlice<f32>,
    reference_scales: &[f32],
    leaves: &[Bf16Leaf],
    tiles: &[DenseTile],
    selection: &[Selection],
    seeds: &[Seed],
    accepted: &[u32],
    summary: &[TellSummary],
    initial_seed: u64,
    initialize: u32,
) {
    static mut SUMS: SharedArray<f32, WARPS> = SharedArray::UNINIT;
    if initialize == 0 && (summary[0].status != 0 || accepted[0] == 0) {
        return;
    }
    let candidate = if initialize != 0 {
        0
    } else {
        selection[0].index
    };
    let seed = if initialize != 0 {
        initial_seed ^ 0xe703_7ed1_a0b4_28db
    } else {
        let seed = seeds[candidate as usize];
        u64::from(seed.low) | (u64::from(seed.high) << 32)
    };
    let block = thread::blockIdx_x() as usize;
    let tile = tiles[block];
    let leaf = leaves[tile.leaf as usize];
    let thread_index = thread::threadIdx_x();
    let pointer = reference.as_mut_ptr() as *const u16;
    let mut item = thread_index;
    let mut sum = 0.0;
    while item < tile.length {
        let element = u64::from(tile.start + item);
        let index = (leaf.offset + element) as usize;
        let value = if initialize != 0 {
            bf16::dense_normal(seed, leaf.key, element)
        } else {
            let noise = bf16::dense_normal(seed ^ 0x8ebc_6af0_9c88_c6e3, leaf.key, element);
            bf16::correlated_value(
                bf16::bf16_decode(unsafe { pointer.add(index).read() }),
                reference_scales[tile.leaf as usize],
                noise,
                candidate,
            )
        };
        let bits = bf16::bf16_encode(value);
        let stored = bf16::bf16_decode(bits);
        sum += stored * stored;
        unsafe { *reference.get_unchecked_mut(index) = bits };
        item += THREADS;
    }
    let sum = warp::reduce_sum_f32(sum);
    if warp::lane_id() == 0 {
        unsafe { SUMS[(thread_index / 32) as usize] = sum };
    }
    thread::sync_threads();
    if thread_index == 0 {
        let mut total = 0.0;
        let mut index = 0;
        while index < WARPS {
            total += unsafe { SUMS[index] };
            index += 1;
        }
        unsafe { *partials.get_unchecked_mut(block) = total };
    }
}

#[inline(always)]
pub(crate) fn run_rmsbf16(
    partials: &[f32],
    leaves: &[Bf16Leaf],
    tile_offsets: &[u32],
    scales: &mut DisjointSlice<f32>,
    accepted: &[u32],
    summary: &[TellSummary],
    initialize: u32,
) {
    static mut SUMS: SharedArray<f32, WARPS> = SharedArray::UNINIT;
    if initialize == 0 && (summary[0].status != 0 || accepted[0] == 0) {
        return;
    }
    let leaf = thread::blockIdx_x() as usize;
    let thread_index = thread::threadIdx_x();
    let mut tile = tile_offsets[leaf] + thread_index;
    let mut sum = 0.0;
    while tile < tile_offsets[leaf + 1] {
        sum += partials[tile as usize];
        tile += THREADS;
    }
    let sum = warp::reduce_sum_f32(sum);
    if warp::lane_id() == 0 {
        unsafe { SUMS[(thread_index / 32) as usize] = sum };
    }
    thread::sync_threads();
    if thread_index == 0 {
        let mut total = 0.0;
        let mut index = 0;
        while index < WARPS {
            total += unsafe { SUMS[index] };
            index += 1;
        }
        let inverse_rms = (leaves[leaf].length as f32 / total).sqrt();
        unsafe { *scales.get_unchecked_mut(leaf) = inverse_rms };
    }
}

#[inline(always)]
pub(crate) fn run_applybf16(
    base: &[u16],
    leaves: &[DenseLeaf],
    terms: &[DenseTerm],
    tiles: &[DenseTile],
    output: &mut DisjointSlice<u16>,
    status: &mut DisjointSlice<u32>,
    term_count: u32,
) {
    static mut WARP_STATUS: SharedArray<u32, WARPS> = SharedArray::UNINIT;

    let tile_index = thread::blockIdx_x();
    if tile_index as usize >= tiles.len() {
        return;
    }
    let tile = tiles[tile_index as usize];
    let leaf = leaves[tile.leaf as usize];
    let thread_index = thread::threadIdx_x();
    let mut invalid = false;
    let mut item = thread_index;
    while item < tile.length {
        let coordinate = u64::from(tile.start + item);
        let index = leaf.offset + coordinate;
        unsafe {
            let value = bf16::bf16_value(
                base[index as usize],
                leaf.key,
                coordinate,
                leaf.scale,
                terms,
                term_count,
            );
            invalid |= !bf16::bf16_finite(value);
            *output.get_unchecked_mut(index as usize) = value;
        }
        item += THREADS;
    }
    let warp_invalid = warp::any(invalid);
    if warp::lane_id() == 0 {
        unsafe { WARP_STATUS[(thread_index / 32) as usize] = u32::from(warp_invalid) };
    }
    thread::sync_threads();
    if thread_index == 0 {
        let mut block_status = 0;
        let mut warp_index = 0;
        while warp_index < WARPS {
            block_status |= unsafe { WARP_STATUS[warp_index] };
            warp_index += 1;
        }
        unsafe { *status.get_unchecked_mut(tile_index as usize) = block_status };
    }
}
