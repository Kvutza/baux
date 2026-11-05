use super::fbt_route::attention_pos;
use super::fbt_types::{
    FBT_THREADS, PISA_CANDIDATES, PISA_ROUTING_QUERIES, PISA_SELECTED, PisaShape,
};
use super::pisa_score::{
    PISA_LEAF_PAIR, PISA_SHARED_KEY_STRIDE, PISA_SHARED_KEYS, select_query, select_score,
    select_top, shared_leaf_score,
};
use cuda_device::{DisjointSlice, SharedArray, thread};

// All callers use one 256-thread block. Shared buffers are sized by the
// PISA constants; every lane participates in both barriers around each tile.
#[inline(always)]
unsafe fn load_keys(
    kv: &[f32],
    keys: *mut f32,
    candidates: *const u32,
    sequence: u32,
    current: u32,
    pair: u32,
    tid: u32,
    shape: PisaShape,
) {
    let previous = current.saturating_sub(1);
    let mut index = tid as usize;
    while index < PISA_LEAF_PAIR * 64 * 64 {
        let key_slot = index >> 12;
        let local = index & 4095;
        let token = local >> 6;
        let dim = local & 63;
        let candidate_slot = pair as usize * PISA_LEAF_PAIR + key_slot;
        let leaf = unsafe { candidates.add(candidate_slot).read() };
        let scoreable = leaf != u32::MAX && leaf < current && leaf != 0 && leaf != previous;
        let value = if scoreable {
            let source = sequence * shape.context + (leaf << 6) + token as u32;
            let source_index = u64::from(source) * u64::from(shape.kv_stride)
                + u64::from(shape.key_offset + dim as u32);
            unsafe { *kv.get_unchecked(source_index as usize) }
        } else {
            0.0
        };
        let destination =
            key_slot * 64 * PISA_SHARED_KEY_STRIDE + dim * PISA_SHARED_KEY_STRIDE + token;
        unsafe { keys.add(destination).write(value) };
        index += FBT_THREADS as usize;
    }
}

#[inline(always)]
unsafe fn score_leaves(
    kv: &[f32],
    query: *const f32,
    candidates: *const u32,
    scores: *mut f32,
    keys: *mut f32,
    sequence: u32,
    current: u32,
    tid: u32,
    shape: PisaShape,
) {
    let lane = tid & 31;
    let warp_index = tid >> 5;
    let previous = current.saturating_sub(1);
    let mut pair = 0u32;
    while pair < (PISA_CANDIDATES / PISA_LEAF_PAIR) as u32 {
        unsafe { load_keys(kv, keys, candidates, sequence, current, pair, tid, shape) };
        thread::sync_threads();
        if warp_index < PISA_LEAF_PAIR as u32 {
            let slot = pair * PISA_LEAF_PAIR as u32 + warp_index;
            let leaf = unsafe { candidates.add(slot as usize).read() };
            let score = if leaf == u32::MAX || leaf > current {
                f32::NEG_INFINITY
            } else if leaf == 0 || leaf == previous || leaf == current {
                f32::INFINITY
            } else {
                unsafe { shared_leaf_score(keys, query, warp_index, lane) }
            };
            if lane == 0 {
                unsafe { scores.add(slot as usize).write(score) };
            }
        }
        thread::sync_threads();
        pair += 1;
    }
}

#[inline(always)]
unsafe fn score_nodes(
    kv: &[f32],
    pyramid: &[f32],
    query: *const f32,
    candidates: *const u32,
    scores: *mut f32,
    sequence: u32,
    current: u32,
    level: u32,
    tid: u32,
    shape: PisaShape,
) {
    let lane = tid & 31;
    let mut slot = tid >> 5;
    while slot < PISA_CANDIDATES as u32 {
        let node = unsafe { candidates.add(slot as usize).read() };
        let score = unsafe {
            select_score(
                kv,
                pyramid,
                query,
                node,
                level,
                sequence,
                current,
                shape.context >> 6,
                lane,
                shape,
            )
        };
        if lane == 0 {
            unsafe { scores.add(slot as usize).write(score) };
        }
        slot += FBT_THREADS >> 5;
    }
}

#[inline(always)]
unsafe fn choose(candidates: *const u32, scores: *const f32, chosen: *mut u32, tid: u32) {
    if tid >= 32 {
        return;
    }
    let node = if tid < PISA_CANDIDATES as u32 {
        unsafe { candidates.add(tid as usize).read() }
    } else {
        u32::MAX
    };
    let score = if tid < PISA_CANDIDATES as u32 {
        unsafe { scores.add(tid as usize).read() }
    } else {
        f32::NEG_INFINITY
    };
    let selected = select_top(score, node, tid);
    if tid < PISA_SELECTED as u32 {
        unsafe { chosen.add(tid as usize).write(selected) };
    }
}

#[inline(always)]
unsafe fn expand(candidates: *mut u32, chosen: *const u32, tid: u32) {
    if tid < PISA_CANDIDATES as u32 {
        let parent = unsafe { chosen.add((tid >> 1) as usize).read() };
        let child = if parent == u32::MAX {
            u32::MAX
        } else {
            (parent << 1) | (tid & 1)
        };
        unsafe { candidates.add(tid as usize).write(child) };
    }
}

#[inline(always)]
pub(crate) fn run_select(
    qkv: &[f32],
    kv: &[f32],
    pyramid: &[f32],
    blocks: &mut DisjointSlice<u32>,
    shape: PisaShape,
) {
    static mut QUERY: SharedArray<f32, PISA_ROUTING_QUERIES> = SharedArray::UNINIT;
    static mut CANDIDATES: SharedArray<u32, PISA_CANDIDATES> = SharedArray::UNINIT;
    static mut SCORES: SharedArray<f32, PISA_CANDIDATES> = SharedArray::UNINIT;
    static mut CHOSEN: SharedArray<u32, PISA_SELECTED> = SharedArray::UNINIT;
    static mut KEYS: SharedArray<f32, PISA_SHARED_KEYS> = SharedArray::UNINIT;
    let row = thread::blockIdx_x();
    let tid = thread::threadIdx_x();
    if row >= shape.rows {
        return;
    }
    let query = unsafe { SharedArray::as_raw_mut_ptr(&raw mut QUERY) };
    let candidates = unsafe { SharedArray::as_raw_mut_ptr(&raw mut CANDIDATES) };
    let scores = unsafe { SharedArray::as_raw_mut_ptr(&raw mut SCORES) };
    let chosen = unsafe { SharedArray::as_raw_mut_ptr(&raw mut CHOSEN) };
    let keys = unsafe { SharedArray::as_raw_mut_ptr(&raw mut KEYS) };
    unsafe { select_query(qkv, query, row, tid, shape) };
    if tid < PISA_CANDIDATES as u32 {
        unsafe { candidates.add(tid as usize).write(tid) };
    }
    thread::sync_threads();
    let (sequence, position) = unsafe { attention_pos(row, shape) };
    let current = position >> 6;
    let leaves = shape.context >> 6;
    let mut level = leaves.trailing_zeros() - 4;
    loop {
        if level == 0 {
            unsafe {
                score_leaves(
                    kv, query, candidates, scores, keys, sequence, current, tid, shape,
                )
            };
        } else {
            unsafe {
                score_nodes(
                    kv, pyramid, query, candidates, scores, sequence, current, level, tid, shape,
                )
            };
        }
        thread::sync_threads();
        unsafe { choose(candidates, scores, chosen, tid) };
        thread::sync_threads();
        if level == 0 {
            if tid < PISA_SELECTED as u32 {
                unsafe {
                    *blocks.get_unchecked_mut(((row << 3) + tid) as usize) =
                        chosen.add(tid as usize).read()
                };
            }
            break;
        }
        unsafe { expand(candidates, chosen, tid) };
        thread::sync_threads();
        level -= 1;
    }
}
