use cuda_device::{DisjointSlice, SharedArray, thread, warp};

use super::fbt_math::contains_leaf;
use super::fbt_route::attention_pos;
use super::fbt_types::{PISA_WIDTH, PisaShape, half, half_bits};

#[inline(always)]
pub(crate) fn run_pisaleaves(
    qkv: &[u16],
    pyramid: &mut DisjointSlice<u16, thread::Runtime2DIndex>,
    shape: PisaShape,
) {
    let leaf = thread::blockIdx_x();
    let sequence = thread::blockIdx_y();
    let dim = thread::threadIdx_x();
    let leaves = shape.context >> 6;
    let sequences = shape.rows >> shape.context.trailing_zeros();
    if leaf >= leaves || sequence >= sequences || dim >= 64 {
        return;
    }
    let first = sequence * shape.context + (leaf << 6);
    let mut sum = 0.0;
    let mut token = 0;
    while token < 64 {
        let offset = u64::from(first + token) * u64::from(shape.kv_stride)
            + u64::from(shape.key_offset + dim);
        sum += half(unsafe { *qkv.get_unchecked(offset as usize) });
        token += 1;
    }
    let nodes = (leaves << 1) - 1;
    let destination =
        ((u64::from(sequence) * u64::from(nodes) + u64::from(leaf)) << 6) + u64::from(dim);
    unsafe { *pyramid.get_unchecked_mut(destination as usize) = half_bits(sum * (1.0 / 64.0)) };
}

#[inline(always)]
pub(crate) fn run_cacheleaves(kv: &[u16], tree: &mut DisjointSlice<u16>, first: u32, count: u32) {
    let group = thread::blockIdx_x();
    let dim = thread::threadIdx_x();
    if group >= count || dim >= 64 {
        return;
    }
    let leaf = first + group;
    let base_token = u64::from(leaf) << 6;
    let mut sum = 0.0;
    for token in 0..64 {
        let idx = (((base_token + token) << 7) + u64::from(dim)) as usize;
        sum += half(unsafe { *kv.get_unchecked(idx) });
    }
    unsafe {
        *tree.get_unchecked_mut(((u64::from(leaf) << 6) + u64::from(dim)) as usize) =
            half_bits(sum * (1.0 / 64.0))
    };
}

#[inline(always)]
pub(crate) fn run_cachesplit(
    qkv: &[u16],
    queries: &mut DisjointSlice<u16>,
    kv: &mut DisjointSlice<u16>,
    rows: u32,
    first: u32,
) {
    let index = thread::blockIdx_x() * 256 + thread::threadIdx_x();
    if index >= rows * 640 {
        return;
    }
    let row = index / 640;
    let column = index - row * 640;
    unsafe {
        let val = *qkv.get_unchecked(index as usize);
        if column < 512 {
            *queries.get_unchecked_mut(((row << 9) + column) as usize) = val;
        } else {
            *kv.get_unchecked_mut(
                ((u64::from(first + row) << 7) + u64::from(column - 512)) as usize,
            ) = val;
        }
    }
}

#[inline(always)]
pub(crate) fn run_ancestors(
    tree: &mut DisjointSlice<u16>,
    leaves: u32,
    mut first: u32,
    mut end: u32,
) {
    let dim = thread::threadIdx_x();
    let data = tree.as_mut_ptr();
    let (mut child, mut count, mut parent) = (0, leaves, leaves);
    while count > 1 {
        first >>= 1;
        end = end.div_ceil(2);
        for node in first..end {
            let left = (((child + (node << 1)) << 6) + dim) as usize;
            let output = (((parent + node) << 6) + dim) as usize;
            unsafe {
                let value = 0.5 * (half(data.add(left).read()) + half(data.add(left + 64).read()));
                data.add(output).write(half_bits(value));
            }
        }
        child = parent;
        count >>= 1;
        parent += count;
    }
}

#[inline(always)]
pub(crate) fn run_parents(
    tree: &mut DisjointSlice<u16>,
    child: u32,
    parent: u32,
    first: u32,
    count: u32,
) {
    let group = thread::blockIdx_x();
    let dim = thread::threadIdx_x();
    if group >= count || dim >= 64 {
        return;
    }
    let node = first + group;
    let left = (((child + (node << 1)) << 6) + dim) as usize;
    let output = (((parent + node) << 6) + dim) as usize;
    let data = tree.as_mut_ptr();
    let value = unsafe { 0.5 * (half(data.add(left).read()) + half(data.add(left + 64).read())) };
    unsafe { *tree.get_unchecked_mut(output) = half_bits(value) };
}

#[inline(always)]
pub(crate) fn run_pisaupper(pyramid: &mut DisjointSlice<u16>, shape: PisaShape) {
    let sequence = thread::blockIdx_x();
    let dim = thread::threadIdx_x();
    let leaves = shape.context >> 6;
    let sequences = shape.rows >> shape.context.trailing_zeros();
    if sequence >= sequences || dim >= 64 {
        return;
    }
    let nodes = (leaves << 1) - 1;
    let base = (u64::from(sequence) * u64::from(nodes)) << 6;
    let mut child_offset = 0;
    let mut parent_offset = leaves;
    let mut child_count = leaves;
    while child_count > 1 {
        let parent_count = child_count >> 1;
        let mut parent = 0;
        while parent < parent_count {
            let left = base + (u64::from(child_offset + (parent << 1)) << 6) + u64::from(dim);
            let right = left + 64;
            let destination = base + (u64::from(parent_offset + parent) << 6) + u64::from(dim);
            let data = pyramid.as_mut_ptr();
            let value = unsafe {
                0.5 * (half(data.add(left as usize).read()) + half(data.add(right as usize).read()))
            };
            unsafe { *pyramid.get_unchecked_mut(destination as usize) = half_bits(value) };
            parent += 1;
        }
        thread::sync_threads();
        child_offset = parent_offset;
        parent_offset += parent_count;
        child_count = parent_count;
    }
}

#[inline(always)]
pub(crate) unsafe fn select_query(
    qkv: &[u16],
    query: *mut f32,
    row: u32,
    tid: u32,
    shape: PisaShape,
) {
    if tid < 64 {
        let base = (u64::from(row) * u64::from(shape.query_stride) + u64::from(tid)) as usize;
        let mut sum = 0.0_f32;
        let mut head = 0usize;
        while head < 8 {
            sum += half(unsafe { *qkv.get_unchecked(base + (head << 6)) });
            head += 1;
        }
        unsafe { query.add(tid as usize).write(sum) };
    }
}

#[inline(always)]
unsafe fn select_score(
    pyramid: &[u16],
    query: *const f32,
    node: u32,
    level: u32,
    sequence: u32,
    current: u32,
    leaves: u32,
) -> f32 {
    if thread::threadIdx_x() >= 16 || node == u32::MAX {
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
    let nodes = (leaves << 1) - 1;
    let offset = (leaves << 1) - ((leaves << 1) >> level);
    let summary =
        ((u64::from(sequence) * u64::from(nodes) + u64::from(offset + node)) as usize) << 6;
    let mut score = 0.0;
    let mut dim = 0usize;
    while dim < 64 {
        score += unsafe { query.add(dim).read() * half(*pyramid.get_unchecked(summary + dim)) };
        dim += 1;
    }
    score
}

#[inline(always)]
pub(crate) fn select_top(mut score: f32, node: u32, level: u32, tid: u32) -> u32 {
    let mut max_finite = f32::NEG_INFINITY;
    let mut my_chosen = u32::MAX;
    let mut slot = 0_u32;
    while slot < 8 {
        let mut best_score = score;
        for mask in [8, 4, 2, 1] {
            best_score = best_score.max(warp::shuffle_xor_f32(best_score, mask));
        }
        if best_score.is_finite() && max_finite == f32::NEG_INFINITY {
            max_finite = best_score;
        }
        let drop = level == 0 && max_finite > f32::NEG_INFINITY && best_score < max_finite - 12.0;
        let mut best_node = if score == best_score && best_score > f32::NEG_INFINITY {
            node
        } else {
            u32::MAX
        };
        for mask in [8, 4, 2, 1] {
            best_node = best_node.min(warp::shuffle_xor(best_node, mask));
        }
        if tid == slot {
            my_chosen = if drop { u32::MAX } else { best_node };
        }
        if best_node != u32::MAX && node == best_node {
            score = f32::NEG_INFINITY;
        }
        slot += 1;
    }
    my_chosen
}

#[inline(always)]
pub(crate) fn run_select(
    qkv: &[u16],
    pyramid: &[u16],
    blocks: &mut DisjointSlice<u32>,
    shape: PisaShape,
) {
    static mut QUERY: SharedArray<f32, PISA_WIDTH> = SharedArray::UNINIT;
    let row = thread::blockIdx_x();
    let tid = thread::threadIdx_x();
    if row >= shape.rows {
        return;
    }
    let query = unsafe { SharedArray::as_raw_mut_ptr(&raw mut QUERY) };
    unsafe { select_query(qkv, query, row, tid, shape) };
    thread::sync_threads();
    if tid >= 32 {
        return;
    }
    let (sequence, position) = unsafe { attention_pos(row, shape) };
    let current = position >> 6;
    let leaves = shape.context >> 6;
    let mut level = leaves.trailing_zeros() - 4;
    let mut node = if tid < 16 { tid } else { u32::MAX };
    loop {
        let score = unsafe { select_score(pyramid, query, node, level, sequence, current, leaves) };
        let my_chosen = select_top(score, node, level, tid);
        if level == 0 {
            if tid < 8 {
                unsafe { *blocks.get_unchecked_mut(((row << 3) + tid) as usize) = my_chosen };
            }
            break;
        }
        let parent =
            warp::shuffle_f32_sync(u32::MAX, f32::from_bits(my_chosen), tid >> 1).to_bits();
        node = if tid < 16 && parent != u32::MAX {
            (parent << 1) | (tid & 1)
        } else {
            u32::MAX
        };
        level -= 1;
    }
}
