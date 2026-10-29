use cuda_device::{DisjointSlice, SharedArray, thread};

use super::fbt_cache::{select_query, select_top};
use super::fbt_math::reduce_sum;
use super::fbt_route;
use super::fbt_types::{
    FBT_THREADS, PISA_CANDIDATES, PISA_GROUP_HEADS, PISA_HEAD_GROUPS, PISA_HEAD_SCORES,
    PISA_QUERIES, PISA_SCORES, PISA_SELECTED, PISA_VALUES, PISA_WIDTH, PisaShape, ROUTE_THREADS,
    RouteShape, half,
};

#[inline(always)]
pub(crate) fn run_counts(
    experts: &[u32],
    counts: &mut DisjointSlice<u32>,
    prefixes: &mut DisjointSlice<u32>,
    assignments: u32,
) {
    static mut SUMS: SharedArray<f32, 256> = SharedArray::UNINIT;
    let expert = thread::blockIdx_x();
    let tid = thread::threadIdx_x();
    let sums = unsafe { SharedArray::as_raw_mut_ptr(&raw mut SUMS) };
    let chunks = assignments.div_ceil(1024);
    let mut total = 0;
    for chunk in 0..chunks {
        if tid == 0 {
            unsafe {
                *prefixes.get_unchecked_mut((expert * chunks + chunk) as usize) = total;
            }
        }
        let mut count = 0.0;
        for item in 0..4 {
            let index = chunk * 1024 + tid + item * 256;
            if index < assignments && experts[index as usize] == expert {
                count += 1.0;
            }
        }
        unsafe { sums.add(tid as usize).write(count) };
        total += unsafe { reduce_sum(sums) } as u32;
        thread::sync_threads();
    }
    if tid == 0 {
        unsafe { *counts.get_unchecked_mut(expert as usize) = total };
    }
}

#[inline(always)]
unsafe fn refine_score(
    pyramid: &[u16],
    query: *const f32,
    candidates: *const u32,
    sequence: u32,
    current: u32,
    leaves: u32,
    tid: u32,
) -> (u32, f32) {
    if tid >= 16 {
        return (u32::MAX, f32::NEG_INFINITY);
    }
    let candidate = unsafe { candidates.add(tid as usize).read() };
    let mut duplicate = false;
    let mut earlier = 0;
    while earlier < tid {
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
    let nodes = (leaves << 1) - 1;
    let summary = ((u64::from(sequence) * u64::from(nodes) + u64::from(candidate)) as usize) << 6;
    let mut score = 0.0;
    let mut dim = 0usize;
    while dim < 64 {
        score += unsafe { query.add(dim).read() * half(*pyramid.get_unchecked(summary + dim)) };
        dim += 1;
    }
    (candidate, score)
}

#[inline(always)]
pub(crate) fn run_refine(
    qkv: &[u16],
    pyramid: &[u16],
    incumbent: &[u32],
    blocks: &mut DisjointSlice<u32>,
    visit_offset: u32,
    shape: PisaShape,
) {
    static mut QUERY: SharedArray<f32, PISA_WIDTH> = SharedArray::UNINIT;
    static mut CANDIDATES: SharedArray<u32, PISA_CANDIDATES> = SharedArray::UNINIT;
    let row = thread::blockIdx_x();
    let tid = thread::threadIdx_x();
    if row >= shape.rows {
        return;
    }
    let query = unsafe { SharedArray::as_raw_mut_ptr(&raw mut QUERY) };
    let candidates = unsafe { SharedArray::as_raw_mut_ptr(&raw mut CANDIDATES) };
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
    if tid >= 32 {
        return;
    }
    let (node, score) =
        unsafe { refine_score(pyramid, query, candidates, sequence, current, leaves, tid) };
    let chosen = select_top(score, node, 1, tid);
    if tid < 8 {
        let out = if chosen == u32::MAX { current } else { chosen };
        unsafe { *blocks.get_unchecked_mut(((row << 3) + tid) as usize) = out };
    }
}

#[inline(always)]
pub(crate) fn run_routetopk(
    scores_input: &[u16],
    experts_output: DisjointSlice<u32>,
    weights_output: DisjointSlice<u16>,
    margin_output: DisjointSlice<f32>,
    shape: RouteShape,
) {
    static mut SCORES: SharedArray<f32, 256> = SharedArray::UNINIT;
    static mut IDS: SharedArray<u32, 256> = SharedArray::UNINIT;
    static mut BEST_SCORES: SharedArray<f32, 4> = SharedArray::UNINIT;
    static mut BEST_IDS: SharedArray<u32, 4> = SharedArray::UNINIT;
    let row = thread::blockIdx_x();
    let tid = thread::threadIdx_x();
    if row >= shape.rows {
        return;
    }
    let scores = unsafe { SharedArray::as_raw_mut_ptr(&raw mut SCORES) };
    let ids = unsafe { SharedArray::as_raw_mut_ptr(&raw mut IDS) };
    let best_scores = unsafe { SharedArray::as_raw_mut_ptr(&raw mut BEST_SCORES) };
    let best_ids = unsafe { SharedArray::as_raw_mut_ptr(&raw mut BEST_IDS) };
    let (score, id) = unsafe { fbt_route::find_initial(scores_input, row, tid, shape) };
    unsafe {
        scores.add(tid as usize).write(score);
        ids.add(tid as usize).write(id);
    }
    thread::sync_threads();
    let mut position = 0;
    while position < 4 {
        let mut stride = ROUTE_THREADS / 2;
        while stride > 0 {
            unsafe { fbt_route::reduce_pair(scores, ids, tid, stride) };
            thread::sync_threads();
            stride /= 2;
        }
        let selected = unsafe { ids.read() };
        if tid == 0 {
            unsafe {
                best_scores.add(position as usize).write(scores.read());
                best_ids.add(position as usize).write(selected);
            }
        }
        thread::sync_threads();
        let (next_score, next_id) = unsafe {
            fbt_route::find_next(scores_input, best_ids, row, tid, selected, position, shape)
        };
        unsafe {
            scores.add(tid as usize).write(next_score);
            ids.add(tid as usize).write(next_id);
        }
        thread::sync_threads();
        position += 1;
    }
    if tid == 0 {
        unsafe {
            fbt_route::write_routes(
                experts_output,
                weights_output,
                margin_output,
                best_scores,
                best_ids,
                row,
                shape,
            )
        };
    }
}

#[inline(always)]
pub(crate) fn run_attention(
    qkv: &[u16],
    kv: &[u16],
    blocks: &[u32],
    output: DisjointSlice<u16>,
    shape: PisaShape,
) {
    static mut SCORES: SharedArray<f32, PISA_SCORES> = SharedArray::UNINIT;
    static mut REDUCTION: SharedArray<f32, 256> = SharedArray::UNINIT;
    static mut QUERY: SharedArray<f32, PISA_WIDTH> = SharedArray::UNINIT;
    static mut BLOCKS: SharedArray<u32, PISA_SELECTED> = SharedArray::UNINIT;
    let group = thread::blockIdx_x();
    let row = group / shape.heads;
    let head = group % shape.heads;
    let tid = thread::threadIdx_x();
    if row >= shape.rows {
        return;
    }
    let (sequence, position) = unsafe { fbt_route::attention_pos(row, shape) };
    let scores = unsafe { SharedArray::as_raw_mut_ptr(&raw mut SCORES) };
    let reduction = unsafe { SharedArray::as_raw_mut_ptr(&raw mut REDUCTION) };
    let query = unsafe { SharedArray::as_raw_mut_ptr(&raw mut QUERY) };
    let selected_blocks = unsafe { SharedArray::as_raw_mut_ptr(&raw mut BLOCKS) };
    let selected_tokens = shape.selected << 6;
    unsafe {
        fbt_route::attention_load(qkv, blocks, query, selected_blocks, row, head, tid, shape)
    };
    thread::sync_threads();
    let lane = tid % 32;
    let mut local = tid / 32;
    while local < selected_tokens {
        let score = unsafe {
            fbt_route::attention_dot(
                query,
                kv,
                selected_blocks,
                lane,
                local,
                sequence,
                position,
                shape,
            )
        };
        if lane == 0 {
            unsafe { scores.add(local as usize).write(score) };
        }
        local += 8;
    }
    thread::sync_threads();
    let total = unsafe { fbt_route::attention_norm(scores, reduction, tid, selected_tokens) };
    let dim = tid & 63;
    let groups = FBT_THREADS >> 6;
    let value = unsafe {
        fbt_route::attention_values(
            scores,
            kv,
            selected_blocks,
            dim,
            tid,
            sequence,
            position,
            groups,
            selected_tokens,
            shape,
        )
    };
    thread::sync_threads();
    unsafe { reduction.add(tid as usize).write(value) };
    thread::sync_threads();
    unsafe { fbt_route::attention_head(reduction, output, tid, row, head, total, groups, shape) };
}

#[inline(always)]
pub(crate) fn run_multihead(
    qkv: &[u16],
    kv: &[u16],
    blocks: &[u32],
    output: DisjointSlice<u16>,
    shape: PisaShape,
) {
    static mut SCORES: SharedArray<f32, PISA_HEAD_SCORES> = SharedArray::UNINIT;
    static mut QUERIES: SharedArray<f32, PISA_QUERIES> = SharedArray::UNINIT;
    static mut VALUES: SharedArray<f32, PISA_VALUES> = SharedArray::UNINIT;
    static mut BLOCKS: SharedArray<u32, PISA_SELECTED> = SharedArray::UNINIT;
    static mut TOTALS: SharedArray<f32, PISA_GROUP_HEADS> = SharedArray::UNINIT;
    let group_index = thread::blockIdx_x() % PISA_HEAD_GROUPS;
    let row = thread::blockIdx_x() / PISA_HEAD_GROUPS;
    let head_base = group_index * PISA_GROUP_HEADS as u32;
    let tid = thread::threadIdx_x();
    if row >= shape.rows {
        return;
    }
    let lane = tid % 32;
    let warp_index = tid / 32;
    let scores = unsafe { SharedArray::as_raw_mut_ptr(&raw mut SCORES) };
    let queries = unsafe { SharedArray::as_raw_mut_ptr(&raw mut QUERIES) };
    let values = unsafe { SharedArray::as_raw_mut_ptr(&raw mut VALUES) };
    let selected_blocks = unsafe { SharedArray::as_raw_mut_ptr(&raw mut BLOCKS) };
    let totals = unsafe { SharedArray::as_raw_mut_ptr(&raw mut TOTALS) };
    unsafe {
        fbt_route::multihead_init(
            qkv,
            blocks,
            queries,
            selected_blocks,
            row,
            head_base,
            tid,
            shape,
        )
    };
    thread::sync_threads();
    let selected_tokens = shape.selected << 6;
    let (sequence, position) = unsafe { fbt_route::attention_pos(row, shape) };
    let mut local = warp_index;
    while local < selected_tokens {
        unsafe {
            fbt_route::multihead_step(
                scores,
                queries,
                kv,
                selected_blocks,
                lane,
                local,
                sequence,
                position,
                shape,
            )
        };
        local += 8;
    }
    thread::sync_threads();
    if warp_index < PISA_GROUP_HEADS as u32 {
        unsafe { fbt_route::multihead_norm(scores, totals, warp_index, lane, selected_tokens) };
    }
    thread::sync_threads();
    let dim = tid & 63;
    let group = tid >> 6;
    let groups = FBT_THREADS >> 6;
    unsafe {
        fbt_route::multihead_accum(
            scores,
            values,
            kv,
            selected_blocks,
            dim,
            group,
            groups,
            tid,
            sequence,
            position,
            selected_tokens,
            shape,
        )
    };
    thread::sync_threads();
    unsafe {
        fbt_route::multihead_write(values, totals, output, row, head_base, tid, groups, shape)
    };
}
