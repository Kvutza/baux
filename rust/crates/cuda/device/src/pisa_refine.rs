use cuda_device::{DisjointSlice, SharedArray, thread};

use super::fbt_route;
use super::fbt_types::{
    FBT_THREADS, PISA_CANDIDATES, PISA_ROUTING_QUERIES, PISA_SELECTED, PisaShape,
};
use super::pisa_score::{leaf_score, select_query, select_top};

#[inline(always)]
unsafe fn refine_score(
    kv: &[f32],
    query: *const f32,
    candidates: *const u32,
    sequence: u32,
    current: u32,
    leaves: u32,
    slot: u32,
    lane: u32,
    shape: PisaShape,
) -> (u32, f32) {
    let candidate = unsafe { candidates.add(slot as usize).read() };
    let mut duplicate = false;
    let mut earlier = 0;
    while earlier < slot {
        duplicate |= unsafe { candidates.add(earlier as usize).read() } == candidate;
        earlier += 1;
    }
    if duplicate || candidate >= leaves {
        return (candidate, f32::NEG_INFINITY);
    }
    let previous = current.saturating_sub(1);
    if candidate == 0 || candidate == previous || candidate == current {
        return (candidate, f32::INFINITY);
    }
    if candidate > current {
        return (candidate, f32::NEG_INFINITY);
    }
    (candidate, unsafe {
        leaf_score(kv, query, candidate, sequence, lane, shape)
    })
}

#[inline(always)]
pub(crate) fn run_refine(
    qkv: &[f32],
    kv: &[f32],
    _pyramid: &[f32],
    incumbent: &[u32],
    blocks: &mut DisjointSlice<u32>,
    visit_offset: u32,
    shape: PisaShape,
) {
    static mut QUERY: SharedArray<f32, PISA_ROUTING_QUERIES> = SharedArray::UNINIT;
    static mut CANDIDATES: SharedArray<u32, PISA_CANDIDATES> = SharedArray::UNINIT;
    static mut SCORES: SharedArray<f32, PISA_CANDIDATES> = SharedArray::UNINIT;
    let row = thread::blockIdx_x();
    let tid = thread::threadIdx_x();
    if row >= shape.rows {
        return;
    }
    let query = unsafe { SharedArray::as_raw_mut_ptr(&raw mut QUERY) };
    let candidates = unsafe { SharedArray::as_raw_mut_ptr(&raw mut CANDIDATES) };
    let scores = unsafe { SharedArray::as_raw_mut_ptr(&raw mut SCORES) };
    unsafe { select_query(qkv, query, row, tid, shape) };
    let (sequence, position) = unsafe { fbt_route::attention_pos(row, shape) };
    let current = position >> 6;
    let leaves = shape.context >> 6;
    if tid < 16 {
        let old =
            unsafe { *incumbent.get_unchecked((visit_offset + (row << 3) + (tid & 7)) as usize) };
        let candidate = if tid < 8 {
            old
        } else if tid < 15 {
            old ^ 1
        } else {
            let mut bits = u64::from(row) ^ (u64::from(visit_offset) << 32);
            bits = bits.wrapping_add(0x9e3779b97f4a7c15);
            bits = (bits ^ (bits >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
            bits = (bits ^ (bits >> 27)).wrapping_mul(0x94d049bb133111eb);
            bits ^= bits >> 31;
            (bits as u32) % (current + 1)
        };
        unsafe { candidates.add(tid as usize).write(candidate) };
    }
    thread::sync_threads();
    let lane = tid & 31;
    let warp_index = tid >> 5;
    let mut slot = warp_index;
    while slot < PISA_CANDIDATES as u32 {
        let (_, score) = unsafe {
            refine_score(
                kv, query, candidates, sequence, current, leaves, slot, lane, shape,
            )
        };
        if lane == 0 {
            unsafe { scores.add(slot as usize).write(score) };
        }
        slot += FBT_THREADS >> 5;
    }
    thread::sync_threads();
    unsafe { write_refined(candidates, scores, blocks, row, current, tid) };
}

#[inline(always)]
unsafe fn write_refined(
    candidates: *const u32,
    scores: *const f32,
    blocks: &mut DisjointSlice<u32>,
    row: u32,
    current: u32,
    tid: u32,
) {
    let lane = tid & 31;
    let warp_index = tid >> 5;
    if warp_index == 0 {
        let node = if lane < PISA_CANDIDATES as u32 {
            unsafe { candidates.add(lane as usize).read() }
        } else {
            u32::MAX
        };
        let score = if lane < PISA_CANDIDATES as u32 {
            unsafe { scores.add(lane as usize).read() }
        } else {
            f32::NEG_INFINITY
        };
        let chosen = select_top(score, node, lane);
        if lane < PISA_SELECTED as u32 {
            let out = if chosen == u32::MAX { current } else { chosen };
            unsafe { *blocks.get_unchecked_mut(((row << 3) + lane) as usize) = out };
        }
    }
}
