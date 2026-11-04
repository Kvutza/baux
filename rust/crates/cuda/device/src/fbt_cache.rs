use super::fbt_types::PisaShape;
use cuda_device::{DisjointSlice, thread};

#[inline(always)]
pub(crate) fn run_pisaleaves(
    qkv: &[f32],
    pyramid: &mut DisjointSlice<f32, thread::Runtime2DIndex>,
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
        sum += unsafe { *qkv.get_unchecked(offset as usize) };
        token += 1;
    }
    let nodes = (leaves << 1) - 1;
    let destination =
        ((u64::from(sequence) * u64::from(nodes) + u64::from(leaf)) << 6) + u64::from(dim);
    unsafe { *pyramid.get_unchecked_mut(destination as usize) = sum * (1.0 / 64.0) };
}

#[inline(always)]
pub(crate) fn run_cacheleaves(kv: &[f32], tree: &mut DisjointSlice<f32>, first: u32, count: u32) {
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
        sum += unsafe { *kv.get_unchecked(idx) };
    }
    unsafe {
        *tree.get_unchecked_mut(((u64::from(leaf) << 6) + u64::from(dim)) as usize) =
            sum * (1.0 / 64.0)
    };
}

#[inline(always)]
pub(crate) fn run_cachesplit(
    qkv: &[f32],
    queries: &mut DisjointSlice<f32>,
    kv: &mut DisjointSlice<f32>,
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
    tree: &mut DisjointSlice<f32>,
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
                let value = 0.5 * (data.add(left).read() + data.add(left + 64).read());
                data.add(output).write(value);
            }
        }
        child = parent;
        count >>= 1;
        parent += count;
    }
}

#[inline(always)]
pub(crate) fn run_parents(
    tree: &mut DisjointSlice<f32>,
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
    let value = unsafe { 0.5 * (data.add(left).read() + data.add(left + 64).read()) };
    unsafe { *tree.get_unchecked_mut(output) = value };
}

#[inline(always)]
pub(crate) fn run_pisaupper(pyramid: &mut DisjointSlice<f32>, shape: PisaShape) {
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
            let value =
                unsafe { 0.5 * (data.add(left as usize).read() + data.add(right as usize).read()) };
            unsafe { *pyramid.get_unchecked_mut(destination as usize) = value };
            parent += 1;
        }
        thread::sync_threads();
        child_offset = parent_offset;
        parent_offset += parent_count;
        child_count = parent_count;
    }
}
