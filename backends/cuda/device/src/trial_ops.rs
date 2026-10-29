use baux::semantics::random::splitmix64 as dense_mix64;
use cuda_device::{DisjointSlice, SharedArray, thread, warp};

use super::{
    Bf16Leaf, CenterStep, Leaf, MAX_DEPTH, SearchState, Seed, THREADS, TellParams, TellSummary,
    WARPS, bf16,
};

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct DenseLeaf {
    pub key: u64,
    pub offset: u64,
    pub length: u64,
    pub scale: f32,
    pub pad: u32,
}

// SAFETY: DenseLeaf is repr(C) and contains only DeviceCopy scalars.
unsafe impl cuda_core::DeviceCopy for DenseLeaf {}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct DenseTerm {
    pub seed: u64,
    pub coefficient: f32,
    pub pad: u32,
}

// SAFETY: DenseTerm is repr(C) and contains only DeviceCopy scalars.
unsafe impl cuda_core::DeviceCopy for DenseTerm {}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct DenseTile {
    pub leaf: u32,
    pub start: u32,
    pub length: u32,
    pub pad: u32,
}

// SAFETY: DenseTile is repr(C) and contains only DeviceCopy scalars.
unsafe impl cuda_core::DeviceCopy for DenseTile {}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct DenseLinearParams {
    pub rows: u32,
    pub columns: u32,
    pub has_bias: u32,
    pub term_count: u32,
    pub weight_key: u64,
    pub weight_start: u64,
    pub bias_key: u64,
    pub bias_start: u64,
    pub weight_scale: f32,
    pub bias_scale: f32,
    pub pad0: u32,
    pub pad1: u32,
}

// SAFETY: DenseLinearParams is repr(C) and contains only DeviceCopy scalars.
unsafe impl cuda_core::DeviceCopy for DenseLinearParams {}

#[inline(always)]
pub(crate) fn validate_tell_params(params: &TellParams, proposal_status: &[u32]) -> u32 {
    let mut status = 0_u32;
    if params.paired != 0
        && (params.correlated != 1
            || params.trials != 1
            || params.accept > 1
            || !params.incumbent_value.is_finite()
            || !params.incumbent_variance.is_finite()
            || params.incumbent_variance < 0.0)
    {
        status |= 1;
    }
    if params.paired > 2
        || (params.paired == 2
            && (params.capacity < 2
                || params.failure_tolerance == 0
                || params.reject_is_failure > 1
                || !params.improvement.is_finite()
                || !params.improvement_variance.is_finite()
                || params.improvement_variance < 0.0))
    {
        status |= 1;
    }
    let mut status_index = 0_u32;
    while status_index < params.status_count {
        status |= proposal_status[status_index as usize];
        status_index += 1;
    }
    status
}

pub(crate) const FP4_E2M1: [f32; 16] = [
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
];

#[inline(always)]
pub(crate) fn trial_hash(seed_low: u32, seed_high: u32, element: u32) -> u32 {
    let mut value = seed_low ^ element.wrapping_mul(0x9e37_79b9);
    value ^= value >> 16;
    value = value.wrapping_mul(0x7feb_352d);
    value ^= seed_high;
    value = value.wrapping_mul(0x846c_a68b);
    value ^ (value >> 15)
}

#[inline(always)]
pub(crate) fn decode_e4m3(code: u32) -> f32 {
    let sign = if code & 0x80 != 0 { -1.0 } else { 1.0 };
    let exp = (code >> 3) & 0x0f;
    let mant = code & 0x07;
    if exp == 0 {
        sign * (mant as f32 / 8.0) * 2.0_f32.powi(-6)
    } else if exp == 15 && mant == 7 {
        f32::NAN
    } else {
        sign * (1.0 + mant as f32 / 8.0) * 2.0_f32.powi(exp as i32 - 7)
    }
}

#[inline(always)]
pub(crate) fn decode_e5m2(code: u32) -> f32 {
    let sign = if code & 0x80 != 0 { -1.0 } else { 1.0 };
    let exp = (code >> 2) & 0x1f;
    let mant = code & 0x03;
    if exp == 0 {
        sign * (mant as f32 / 4.0) * 2.0_f32.powi(-14)
    } else if exp == 31 {
        if mant == 0 {
            sign * f32::INFINITY
        } else {
            f32::NAN
        }
    } else {
        sign * (1.0 + mant as f32 / 4.0) * 2.0_f32.powi(exp as i32 - 15)
    }
}

#[inline(always)]
pub(crate) fn decode_code(code: u32, leaf: Leaf) -> f32 {
    let value = match leaf.encoding {
        2 => FP4_E2M1[(code & 0x0f) as usize],
        3 => decode_e4m3(code),
        4 => decode_e5m2(code),
        _ => code as f32,
    };
    value * leaf.scale
}

#[inline(always)]
pub(crate) fn code_at(rows: &[u8], row_offset: usize, leaf: Leaf, element: u32) -> u32 {
    if leaf.bits == 4 {
        let byte = rows[row_offset + leaf.byte_offset as usize + (element / 2) as usize];
        u32::from((byte >> ((element & 1) * 4)) & 0x0f)
    } else {
        u32::from(rows[row_offset + leaf.byte_offset as usize + element as usize])
    }
}

#[inline(always)]
pub(crate) unsafe fn code_ptr(rows: *const u8, row_offset: usize, leaf: Leaf, element: u32) -> u32 {
    let index = row_offset
        + leaf.byte_offset as usize
        + if leaf.bits == 4 {
            (element / 2) as usize
        } else {
            element as usize
        };
    let byte = unsafe { rows.add(index).read() };
    if leaf.bits == 4 {
        u32::from((byte >> ((element & 1) * 4)) & 0x0f)
    } else {
        u32::from(byte)
    }
}

#[inline(always)]
pub(crate) fn perturb_code(
    code: u32,
    seed_low: u32,
    seed_high: u32,
    element: u32,
    bits: u32,
    whole: u32,
    threshold: u32,
) -> u32 {
    let random = trial_hash(seed_low, seed_high, element);
    let amount = whole + u32::from((random >> 1) < (threshold >> 1));
    if amount == 0 {
        return code;
    }

    let max_code = (1_u32 << bits) - 1;
    if random & 1 == 0 {
        if code >= amount {
            code - amount
        } else {
            (code + amount).min(max_code)
        }
    } else if code + amount <= max_code {
        code + amount
    } else {
        code.saturating_sub(amount)
    }
}

#[inline(always)]
pub(crate) fn sparse_code(code: u32, seed: Seed, element: u32, leaf: Leaf) -> u32 {
    if leaf.whole == 0 && leaf.threshold == 0 {
        return code;
    }
    perturb_code(
        code,
        seed.low,
        seed.high,
        element,
        leaf.bits,
        leaf.whole.max(1),
        0,
    )
}

#[inline(always)]
pub(crate) fn resolve_center(
    mut code: u32,
    centers: &[CenterStep],
    mut center: u32,
    element: u32,
    leaf: Leaf,
) -> u32 {
    let mut chain = [Seed { low: 0, high: 0 }; MAX_DEPTH];
    let mut depth = 0_usize;
    while center != u32::MAX && depth < MAX_DEPTH {
        let step = centers[center as usize];
        chain[depth] = step.seed;
        center = step.parent;
        depth += 1;
    }
    while depth > 0 {
        depth -= 1;
        let seed = chain[depth];
        code = perturb_code(
            code,
            seed.low,
            seed.high,
            element,
            leaf.bits,
            leaf.whole,
            leaf.threshold,
        );
    }
    code
}

#[inline(always)]
pub(crate) fn draw_normal(seed: u64, index: u32, metric: u32) -> f64 {
    const SEED_PRIME: u64 = 1_000_003;
    const XOR_OFFSET: u64 = 0xd2b7_4407_b1ce_6e93;
    const INV_2P53: f64 = 1.0 / 9_007_199_254_740_992.0;
    const CLIP_MIN: f64 = 1.0e-12;
    const CLIP_MAX: f64 = 1.0 - 1.0e-12;
    const TAU: f64 = 6.283_185_307_179_586;

    let base = seed
        .wrapping_mul(SEED_PRIME)
        .wrapping_add(index as u64)
        .wrapping_mul(SEED_PRIME);
    let combined = base.wrapping_add(metric as u64);
    let first = dense_mix64(combined);
    let second = dense_mix64(combined ^ XOR_OFFSET);
    let raw = ((first >> 11) as f64) * INV_2P53;
    let u1 = if raw < CLIP_MIN {
        CLIP_MIN
    } else if raw > CLIP_MAX {
        CLIP_MAX
    } else {
        raw
    };
    let u2 = ((second >> 11) as f64) * INV_2P53;
    (-2.0 * u1.ln()).sqrt() * (TAU * u2).cos()
}

#[inline(always)]
pub(crate) fn dense_sign(seed: u64, leaf: u64, element: u64) -> f32 {
    if dense_hash(seed, leaf, element) & 1 == 0 {
        -1.0
    } else {
        1.0
    }
}

#[inline(always)]
pub(crate) fn dense_hash(seed: u64, leaf: u64, element: u64) -> u64 {
    let leaf = dense_mix64(leaf ^ 0xd6e8_feb8_6659_fd93);
    let element = dense_mix64(element ^ 0xa076_1d64_78bd_642f);
    dense_mix64(seed ^ leaf ^ element)
}

#[inline(always)]
pub(crate) fn dense_finite(value: f32, positive: bool) -> f32 {
    if value == 0.0 {
        return f32::from_bits(if positive { 1 } else { 0x8000_0001 });
    }
    let bits = if (value > 0.0) == positive {
        value.to_bits().wrapping_add(1)
    } else {
        value.to_bits().wrapping_sub(1)
    };
    let candidate = f32::from_bits(bits);
    if candidate.is_finite() {
        candidate
    } else if (value > 0.0) == positive {
        f32::from_bits(value.to_bits().wrapping_sub(1))
    } else {
        f32::from_bits(value.to_bits().wrapping_add(1))
    }
}

#[inline(always)]
pub(crate) fn dense_value(
    value: f32,
    leaf_key: u64,
    element: u64,
    scale: f32,
    terms: &[DenseTerm],
    term_count: u32,
) -> f32 {
    let mut sum = 0.0_f32;
    let mut strongest = 0.0_f32;
    let mut positive = true;
    let mut term_index = 0;
    while term_index < term_count {
        let term = terms[term_index as usize];
        if term.coefficient != 0.0 {
            let direction = dense_sign(term.seed, leaf_key, element);
            sum += term.coefficient * direction;
            if term.coefficient.abs() > strongest {
                strongest = term.coefficient.abs();
                positive = (term.coefficient > 0.0) == (direction > 0.0);
            }
        }
        term_index += 1;
    }
    let candidate = value + scale * sum;
    if sum == 0.0 || candidate == value {
        dense_finite(value, positive)
    } else {
        candidate
    }
}

#[inline(always)]
pub(crate) fn reduce_best(mut best_score: f32, mut best_index: u32, mut offset: u32) -> (f32, u32) {
    while offset > 0 {
        let other_score = warp::shuffle_xor_f32(best_score, offset);
        let other_index = warp::shuffle_xor(best_index, offset);
        if other_score > best_score || (other_score == best_score && other_index < best_index) {
            best_score = other_score;
            best_index = other_index;
        }
        offset /= 2;
    }
    (best_score, best_index)
}

#[inline(always)]
pub(crate) fn rotate_history(
    history_slots: &mut DisjointSlice<u32>,
    outcomes: &mut DisjointSlice<f32>,
    variances: &mut DisjointSlice<f32>,
    local: &mut SearchState,
    capacity: u32,
) -> u32 {
    if local.history == capacity {
        let slots_ptr = history_slots.as_mut_ptr() as *const u32;
        let outcomes_ptr = outcomes.as_mut_ptr() as *const f32;
        let variances_ptr = variances.as_mut_ptr() as *const f32;
        let oldest = unsafe { slots_ptr.read() };
        let mut index = 1_u32;
        while index < local.history {
            unsafe {
                *history_slots.get_unchecked_mut((index - 1) as usize) =
                    slots_ptr.add(index as usize).read();
                *outcomes.get_unchecked_mut((index - 1) as usize) =
                    outcomes_ptr.add(index as usize).read();
                *variances.get_unchecked_mut((index - 1) as usize) =
                    variances_ptr.add(index as usize).read();
            }
            index += 1;
        }
        local.history -= 1;
        oldest
    } else {
        local.history + 1
    }
}

#[inline(always)]
pub(crate) fn write_tell_state(
    state: &mut DisjointSlice<SearchState>,
    summary: &mut DisjointSlice<TellSummary>,
    local: SearchState,
    restarted: u32,
    status: u32,
    best_slot: u32,
) {
    unsafe {
        *state.get_unchecked_mut(0) = local;
        *summary.get_unchecked_mut(0) = TellSummary {
            length: local.length,
            best: local.best,
            best_variance: local.best_variance,
            history: local.history,
            restarts: local.restarts,
            restarted,
            status,
            best_slot,
        };
    }
}

#[inline(always)]
pub(crate) fn update_correlated_radius(
    local: &mut SearchState,
    params: &TellParams,
    selected_radius: f32,
    best_slot: u32,
) {
    if params.paired == 2 {
        bf16::paired_radius(
            local,
            selected_radius,
            best_slot != u32::MAX,
            params.failure_tolerance,
            params.reject_is_failure != 0,
        );
    } else if best_slot != u32::MAX {
        local.length = f64::from(selected_radius);
    }
}

#[inline(always)]
pub(crate) fn run_checksearch(
    rows: &[u16],
    leaves: &[Bf16Leaf],
    tiles: &[DenseTile],
    status: &mut DisjointSlice<u32>,
    row_stride: u64,
    slot: u32,
) {
    static mut WARP_STATUS: SharedArray<u32, WARPS> = SharedArray::UNINIT;

    let tile_index = thread::blockIdx_x();
    if tile_index as usize >= tiles.len() {
        return;
    }
    let tile = tiles[tile_index as usize];
    let leaf = leaves[tile.leaf as usize];
    let row_offset = slot as usize * row_stride as usize;
    let thread_index = thread::threadIdx_x();
    let mut invalid = false;
    let mut item = thread_index;
    while item < tile.length {
        let index = leaf.offset + u64::from(tile.start + item);
        invalid |= !bf16::bf16_finite(rows[row_offset + index as usize]);
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

#[inline(always)]
pub(crate) fn run_checkbf16(
    base: &[u16],
    leaves: &[DenseLeaf],
    tiles: &[DenseTile],
    status: &mut DisjointSlice<u32>,
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
        let index = leaf.offset + u64::from(tile.start + item);
        invalid |= !bf16::bf16_finite(base[index as usize]);
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
