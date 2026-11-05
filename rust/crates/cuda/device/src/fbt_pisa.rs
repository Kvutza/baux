use cuda_device::{DisjointSlice, SharedArray, thread};

use super::fbt_route;
use super::fbt_types::{
    FBT_THREADS, PISA_GROUP_HEADS, PISA_HEAD_GROUPS, PISA_HEAD_SCORES, PISA_QUERIES, PISA_SCORES,
    PISA_SELECTED, PISA_VALUES, PISA_WIDTH, PisaShape, ROUTE_THREADS, RouteShape,
};

#[inline(always)]
pub(crate) fn run_routetopk(
    scores_input: &[f32],
    experts_output: DisjointSlice<u32>,
    weights_output: DisjointSlice<f32>,
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
    qkv: &[f32],
    kv: &[f32],
    blocks: &[u32],
    output: DisjointSlice<f32>,
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
    qkv: &[f32],
    kv: &[f32],
    blocks: &[u32],
    output: DisjointSlice<f32>,
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
