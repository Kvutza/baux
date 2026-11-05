use super::fbt_math::contains_leaf;
use super::fbt_types::{FBT_THREADS, PISA_ROUTING_QUERIES, PisaShape};
use cuda_device::warp;
pub(crate) const PISA_LEAF_PAIR: usize = 2;
pub(crate) const PISA_SHARED_KEY_STRIDE: usize = 65;
pub(crate) const PISA_SHARED_KEYS: usize = PISA_LEAF_PAIR * 64 * PISA_SHARED_KEY_STRIDE;

#[inline(always)]
pub(crate) unsafe fn select_query(
    qkv: &[f32],
    query: *mut f32,
    row: u32,
    tid: u32,
    shape: PisaShape,
) {
    let base = (u64::from(row) * u64::from(shape.query_stride)) as usize;
    let mut index = tid as usize;
    while index < PISA_ROUTING_QUERIES {
        unsafe { query.add(index).write(*qkv.get_unchecked(base + index)) };
        index += FBT_THREADS as usize;
    }
}

#[inline(always)]
unsafe fn intermediate_score(
    pyramid: &[f32],
    query: *const f32,
    node: u32,
    level: u32,
    sequence: u32,
    leaves: u32,
    lane: u32,
) -> f32 {
    let child_level = level - 1;
    let child_offset = if child_level == 0 {
        0
    } else {
        (leaves << 1) - ((leaves << 1) >> child_level)
    };
    let nodes = (leaves << 1) - 1;
    let tree = u64::from(sequence) * u64::from(nodes);
    let mut grouped = 0.0;
    let mut head = 0usize;
    while head < 8 {
        let query_head = head << 6;
        let mut child_score = [0.0_f32; 2];
        let mut child = 0usize;
        while child < 2 {
            let summary =
                ((tree + u64::from(child_offset + (node << 1) + child as u32)) << 6) as usize;
            let dim = lane as usize;
            let mut partial = unsafe {
                query.add(query_head + dim).read() * *pyramid.get_unchecked(summary + dim)
            };
            partial += unsafe {
                query.add(query_head + dim + 32).read() * *pyramid.get_unchecked(summary + dim + 32)
            };
            child_score[child] = warp::reduce_sum_f32(partial) * 0.125;
            child += 1;
        }
        let maximum = child_score[0].max(child_score[1]);
        grouped +=
            maximum + ((child_score[0] - maximum).exp() + (child_score[1] - maximum).exp()).ln();
        head += 1;
    }
    grouped
}

#[inline(always)]
pub(crate) unsafe fn leaf_score(
    kv: &[f32],
    query: *const f32,
    leaf: u32,
    sequence: u32,
    lane: u32,
    shape: PisaShape,
) -> f32 {
    let first = sequence * shape.context + (leaf << 6);
    let mut grouped = 0.0;
    let mut head = 0usize;
    while head < 8 {
        let query_head = head << 6;
        let mut logits = [0.0_f32; 2];
        let mut token_slot = 0usize;
        while token_slot < 2 {
            let token = lane + ((token_slot as u32) << 5);
            let key = (u64::from(first + token) * u64::from(shape.kv_stride)
                + u64::from(shape.key_offset)) as usize;
            let mut score = 0.0;
            let mut dim = 0usize;
            while dim < 64 {
                score +=
                    unsafe { query.add(query_head + dim).read() * *kv.get_unchecked(key + dim) };
                dim += 1;
            }
            logits[token_slot] = score * 0.125;
            token_slot += 1;
        }
        let maximum = warp::reduce_max_f32(logits[0].max(logits[1]));
        let total = warp::reduce_sum_f32((logits[0] - maximum).exp() + (logits[1] - maximum).exp());
        grouped += maximum + total.ln();
        head += 1;
    }
    grouped
}

#[inline(always)]
pub(crate) unsafe fn shared_leaf_score(
    keys: *const f32,
    query: *const f32,
    key_slot: u32,
    lane: u32,
) -> f32 {
    let key_base = key_slot as usize * 64 * PISA_SHARED_KEY_STRIDE;
    let mut grouped = 0.0;
    let mut head = 0usize;
    while head < 8 {
        let query_head = head << 6;
        let mut logits = [0.0_f32; 2];
        let mut token_slot = 0usize;
        while token_slot < 2 {
            let token = lane as usize + (token_slot << 5);
            let mut score = 0.0;
            let mut dim = 0usize;
            while dim < 64 {
                score += unsafe {
                    query.add(query_head + dim).read()
                        * keys
                            .add(key_base + dim * PISA_SHARED_KEY_STRIDE + token)
                            .read()
                };
                dim += 1;
            }
            logits[token_slot] = score * 0.125;
            token_slot += 1;
        }
        let maximum = warp::reduce_max_f32(logits[0].max(logits[1]));
        let total = warp::reduce_sum_f32((logits[0] - maximum).exp() + (logits[1] - maximum).exp());
        grouped += maximum + total.ln();
        head += 1;
    }
    grouped
}

#[inline(always)]
pub(crate) unsafe fn select_score(
    kv: &[f32],
    pyramid: &[f32],
    query: *const f32,
    node: u32,
    level: u32,
    sequence: u32,
    current: u32,
    leaves: u32,
    lane: u32,
    shape: PisaShape,
) -> f32 {
    if node == u32::MAX {
        return f32::NEG_INFINITY;
    }
    let previous = current.saturating_sub(1);
    if contains_leaf(node, level, 0)
        || contains_leaf(node, level, previous)
        || contains_leaf(node, level, current)
    {
        return f32::INFINITY;
    }
    if (node + 1) * (1_u32 << level) - 1 >= current {
        return f32::NEG_INFINITY;
    }
    if level == 0 {
        unsafe { leaf_score(kv, query, node, sequence, lane, shape) }
    } else {
        unsafe { intermediate_score(pyramid, query, node, level, sequence, leaves, lane) }
    }
}

#[inline(always)]
pub(crate) fn select_top(mut score: f32, node: u32, tid: u32) -> u32 {
    let mut my_chosen = u32::MAX;
    let mut slot = 0_u32;
    while slot < 8 {
        let mut best_score = score;
        for mask in [8, 4, 2, 1] {
            best_score = best_score.max(warp::shuffle_xor_f32(best_score, mask));
        }
        let mut best_node = if score == best_score && best_score > f32::NEG_INFINITY {
            node
        } else {
            u32::MAX
        };
        for mask in [8, 4, 2, 1] {
            best_node = best_node.min(warp::shuffle_xor(best_node, mask));
        }
        if tid == slot {
            my_chosen = best_node;
        }
        if best_node != u32::MAX && node == best_node {
            score = f32::NEG_INFINITY;
        }
        slot += 1;
    }
    my_chosen
}
