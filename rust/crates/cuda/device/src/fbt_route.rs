use super::fbt::{FBT_THREADS, PISA_QUERIES, PisaShape, ROUTE_THREADS, RouteShape};
use super::fbt_math::{reduce_max, reduce_sum, sigmoid};
use cuda_device::{DisjointSlice, thread, warp};

pub const PISA_GROUP_HEADS: usize = 4;
pub const PISA_HEAD_GROUPS: u32 = 2; // 8 / 4

#[inline(always)]
pub unsafe fn reduce_pair(scores: *mut f32, ids: *mut u32, tid: u32, stride: u32) {
    if tid < stride {
        let right = tid + stride;
        let left_score = scores.add(tid as usize).read();
        let right_score = scores.add(right as usize).read();
        let left_id = ids.add(tid as usize).read();
        let right_id = ids.add(right as usize).read();
        if right_score > left_score || (right_score == left_score && right_id < left_id) {
            scores.add(tid as usize).write(right_score);
            ids.add(tid as usize).write(right_id);
        }
    }
}

#[inline(always)]
pub unsafe fn find_initial(
    scores_input: &[f32],
    row: u32,
    tid: u32,
    shape: RouteShape,
) -> (f32, u32) {
    let mut score = f32::NEG_INFINITY;
    let mut id = tid;
    let mut expert = tid;
    while expert < shape.experts {
        let candidate = *scores_input.get_unchecked((row * shape.experts + expert) as usize);
        if candidate > score {
            score = candidate;
            id = expert;
        }
        expert += ROUTE_THREADS;
    }
    (score, id)
}

#[inline(always)]
pub unsafe fn find_next(
    scores_input: &[f32],
    best_ids: *mut u32,
    row: u32,
    tid: u32,
    selected: u32,
    position: u32,
    shape: RouteShape,
) -> (f32, u32) {
    let mut next_score = f32::NEG_INFINITY;
    let mut next_id = tid;
    let mut expert = tid;
    while expert < shape.experts {
        let mut excluded = expert == selected;
        let mut earlier = 0;
        while earlier < position {
            excluded = excluded || expert == best_ids.add(earlier as usize).read();
            earlier += 1;
        }
        let candidate = *scores_input.get_unchecked((row * shape.experts + expert) as usize);
        if !excluded && candidate > next_score {
            next_score = candidate;
            next_id = expert;
        }
        expert += ROUTE_THREADS;
    }
    (next_score, next_id)
}

#[inline(always)]
pub unsafe fn write_routes(
    mut experts_out: DisjointSlice<u32>,
    mut weights_out: DisjointSlice<f32>,
    mut margin_out: DisjointSlice<f32>,
    best_scores: *mut f32,
    best_ids: *mut u32,
    row: u32,
    shape: RouteShape,
) {
    let denominator = sigmoid(best_scores.read())
        + sigmoid(best_scores.add(1).read())
        + sigmoid(best_scores.add(2).read());
    let mut route = 0;
    while route < shape.top_k {
        let destination = (row * shape.top_k + route) as usize;
        *experts_out.get_unchecked_mut(destination) = best_ids.add(route as usize).read();
        *weights_out.get_unchecked_mut(destination) =
            sigmoid(best_scores.add(route as usize).read()) / denominator;
        route += 1;
    }
    *margin_out.get_unchecked_mut(row as usize) =
        best_scores.add(2).read() - best_scores.add(3).read();
}

#[inline(always)]
pub unsafe fn attention_load(
    qkv: &[f32],
    blocks: &[u32],
    query: *mut f32,
    selected_blocks: *mut u32,
    row: u32,
    head: u32,
    tid: u32,
    shape: PisaShape,
) {
    if tid < 64 {
        let q = u64::from(row) * u64::from(shape.query_stride) + u64::from((head << 6) + tid);
        query
            .add(tid as usize)
            .write(*qkv.get_unchecked(q as usize));
    }
    if tid < shape.selected {
        selected_blocks
            .add(tid as usize)
            .write(*blocks.get_unchecked((row * shape.selected + tid) as usize));
    }
}

#[inline(always)]
pub unsafe fn attention_pos(row: u32, shape: PisaShape) -> (u32, u32) {
    let mask = shape.context - 1;
    let sequence = row >> shape.context.trailing_zeros();
    let absolute = shape.query_start + (row & mask);
    let position = (absolute | shape.visibility.saturating_sub(1)).min(mask);
    (sequence, position)
}

#[inline(always)]
pub unsafe fn attention_dot(
    query: *mut f32,
    kv: &[f32],
    selected_blocks: *mut u32,
    lane: u32,
    local: u32,
    sequence: u32,
    position: u32,
    shape: PisaShape,
) -> f32 {
    let slot = local >> 6;
    let within = local & 63;
    let block = selected_blocks.add(slot as usize).read();
    let source_position = block.saturating_mul(64) + within;
    let valid = block != u32::MAX && source_position <= position;
    let dot = if valid {
        let source = sequence * shape.context + source_position;
        let base = (u64::from(source) * u64::from(shape.kv_stride)
            + u64::from(shape.key_offset + lane)) as usize;
        query.add(lane as usize).read() * *kv.get_unchecked(base)
            + query.add((lane + 32) as usize).read() * *kv.get_unchecked(base + 32)
    } else {
        0.0_f32
    };
    let dot = warp::reduce_sum_f32(dot);
    if valid {
        dot * 0.125
    } else {
        f32::NEG_INFINITY
    }
}

#[inline(always)]
pub unsafe fn attention_value(
    scores: *mut f32,
    kv: &[f32],
    selected_blocks: *mut u32,
    dim: u32,
    index: u32,
    sequence: u32,
    position: u32,
    shape: PisaShape,
) -> f32 {
    let slot = index >> 6;
    let within = index & 63;
    let block = selected_blocks.add(slot as usize).read();
    let source_position = block.saturating_mul(64) + within;
    if block != u32::MAX && source_position <= position {
        let source = sequence * shape.context + source_position;
        let v = (u64::from(source) * u64::from(shape.kv_stride)
            + u64::from(shape.key_offset + 64 + dim)) as usize;
        scores.add(index as usize).read() * *kv.get_unchecked(v)
    } else {
        0.0
    }
}

#[inline(always)]
pub unsafe fn attention_values(
    scores: *mut f32,
    kv: &[f32],
    selected_blocks: *mut u32,
    dim: u32,
    tid: u32,
    sequence: u32,
    position: u32,
    groups: u32,
    selected_tokens: u32,
    shape: PisaShape,
) -> f32 {
    let mut value = 0.0;
    let mut index = tid >> 6;
    while index < selected_tokens {
        value += attention_value(
            scores,
            kv,
            selected_blocks,
            dim,
            index,
            sequence,
            position,
            shape,
        );
        index += groups;
    }
    value
}

#[inline(always)]
pub unsafe fn attention_norm(
    scores: *mut f32,
    reduction: *mut f32,
    tid: u32,
    selected_tokens: u32,
) -> f32 {
    let first = if tid < selected_tokens {
        scores.add(tid as usize).read()
    } else {
        f32::NEG_INFINITY
    };
    let second_idx = tid + FBT_THREADS;
    let second = if second_idx < selected_tokens {
        scores.add(second_idx as usize).read()
    } else {
        f32::NEG_INFINITY
    };
    reduction.add(tid as usize).write(first.max(second));
    let maximum = reduce_max(reduction);
    let mut subtotal = 0.0;
    let mut index = tid;
    while index <= second_idx {
        if index < selected_tokens {
            let score = scores.add(index as usize).read();
            let probability = if score.is_finite() {
                (score - maximum).exp()
            } else {
                0.0
            };
            scores.add(index as usize).write(probability);
            subtotal += probability;
        }
        index += FBT_THREADS;
    }
    reduction.add(tid as usize).write(subtotal);
    reduce_sum(reduction).max(f32::MIN_POSITIVE)
}

#[inline(always)]
pub unsafe fn attention_head(
    reduction: *mut f32,
    mut output: DisjointSlice<f32>,
    tid: u32,
    row: u32,
    head: u32,
    total: f32,
    groups: u32,
    shape: PisaShape,
) {
    if tid < 64 {
        let mut sum = 0.0;
        let mut group = 0;
        while group < groups {
            sum += reduction.add(((group << 6) + tid) as usize).read();
            group += 1;
        }
        let destination = (u64::from(row) * u64::from(shape.query_width())
            + u64::from((head << 6) + tid)) as usize;
        *output.get_unchecked_mut(destination) = sum / total;
    }
}

#[inline(always)]
pub unsafe fn multihead_step(
    scores: *mut f32,
    queries: *mut f32,
    kv: &[f32],
    selected_blocks: *mut u32,
    lane: u32,
    local: u32,
    sequence: u32,
    position: u32,
    shape: PisaShape,
) {
    let slot = local >> 6;
    let within = local & 63;
    let block = selected_blocks.add(slot as usize).read();
    let source_pos = block.saturating_mul(64) + within;
    let valid = block != u32::MAX && source_pos <= position;
    let source = sequence * shape.context + source_pos;
    let (key0, key1) = if valid {
        let base =
            u64::from(source) * u64::from(shape.kv_stride) + u64::from(shape.key_offset + lane);
        let i0 = base as usize;
        let i1 = i0 + 32;
        (*kv.get_unchecked(i0), *kv.get_unchecked(i1))
    } else {
        (0.0, 0.0)
    };
    let selected_tokens = shape.selected << 6;
    let mut head = 0;
    while head < PISA_GROUP_HEADS as u32 {
        let query = head << 6;
        let dot = queries.add((query + lane) as usize).read() * key0
            + queries.add((query + lane + 32) as usize).read() * key1;
        let dot = warp::reduce_sum_f32(dot);
        if lane == 0 {
            let score = if valid {
                dot * 0.125
            } else {
                f32::NEG_INFINITY
            };
            scores.add(((head << 9) + local) as usize).write(score);
        }
        head += 1;
    }
}

#[inline(always)]
pub unsafe fn multihead_norm(
    scores: *mut f32,
    totals: *mut f32,
    head: u32,
    lane: u32,
    selected_tokens: u32,
) {
    let head_base = head << 9;
    let mut maximum = f32::NEG_INFINITY;
    let mut index = lane;
    while index < selected_tokens {
        maximum = maximum.max(scores.add((head_base + index) as usize).read());
        index += 32;
    }
    let mut offset = 16;
    while offset > 0 {
        maximum = maximum.max(warp::shuffle_xor_f32(maximum, offset));
        offset /= 2;
    }
    let mut subtotal = 0.0;
    index = lane;
    while index < selected_tokens {
        let address = (head_base + index) as usize;
        let score = scores.add(address).read();
        let probability = if score.is_finite() {
            (score - maximum).exp()
        } else {
            0.0
        };
        scores.add(address).write(probability);
        subtotal += probability;
        index += 32;
    }
    let total = warp::reduce_sum_f32(subtotal).max(f32::MIN_POSITIVE);
    if lane == 0 {
        totals.add(head as usize).write(total);
    }
}

#[inline(always)]
pub unsafe fn multihead_init(
    qkv: &[f32],
    blocks: &[u32],
    queries: *mut f32,
    selected_blocks: *mut u32,
    row: u32,
    head_base: u32,
    tid: u32,
    shape: PisaShape,
) {
    if tid < (PISA_QUERIES as u32) {
        let index = (head_base << 6) + tid;
        let source = u64::from(row) * u64::from(shape.query_stride) + u64::from(index);
        queries
            .add(tid as usize)
            .write(*qkv.get_unchecked(source as usize));
    }
    if tid < 8 {
        selected_blocks
            .add(tid as usize)
            .write(*blocks.get_unchecked(((row << 3) + tid) as usize));
    }
}

#[inline(always)]
pub unsafe fn multihead_accum(
    scores: *mut f32,
    values: *mut f32,
    kv: &[f32],
    selected_blocks: *mut u32,
    dim: u32,
    group: u32,
    groups: u32,
    tid: u32,
    sequence: u32,
    position: u32,
    selected_tokens: u32,
    shape: PisaShape,
) {
    let mut sum0 = 0.0_f32;
    let mut sum1 = 0.0_f32;
    let mut sum2 = 0.0_f32;
    let mut sum3 = 0.0_f32;
    let mut index = group;
    while index < selected_tokens {
        let slot = index >> 6;
        let within = index & 63;
        let block = selected_blocks.add(slot as usize).read();
        let source_position = block.saturating_mul(64) + within;
        if block != u32::MAX && source_position <= position {
            let source = sequence * shape.context + source_position;
            let value_index = u64::from(source) * u64::from(shape.kv_stride)
                + u64::from(shape.key_offset + 64 + dim);
            let value = *kv.get_unchecked(value_index as usize);
            let idx = index as usize;
            sum0 += scores.add(idx).read() * value;
            sum1 += scores.add(idx + 512).read() * value;
            sum2 += scores.add(idx + 1024).read() * value;
            sum3 += scores.add(idx + 1536).read() * value;
        }
        index += groups;
    }
    let t = tid as usize;
    values.add(t).write(sum0);
    values.add(t + 256).write(sum1);
    values.add(t + 512).write(sum2);
    values.add(t + 768).write(sum3);
}

#[inline(always)]
pub unsafe fn multihead_write(
    values: *mut f32,
    totals: *mut f32,
    mut output: DisjointSlice<f32>,
    row: u32,
    head_base: u32,
    tid: u32,
    _groups: u32,
    _shape: PisaShape,
) {
    if tid < 64 {
        let mut head = 0;
        while head < PISA_GROUP_HEADS as u32 {
            let base = ((head << 8) + tid) as usize;
            let sum = values.add(base).read()
                + values.add(base + 64).read()
                + values.add(base + 128).read()
                + values.add(base + 192).read();
            let destination =
                ((u64::from(row) << 9) + u64::from(((head_base + head) << 6) + tid)) as usize;
            let total = totals.add(head as usize).read();
            *output.get_unchecked_mut(destination) = sum / total;
            head += 1;
        }
    }
}
