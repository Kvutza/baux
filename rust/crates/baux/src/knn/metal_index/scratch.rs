use super::Plan;
use super::pad_k;
use crate::apple_gpu::Runtime;
use crate::error::IndexError;
extern crate metal as mtl;
use mtl::MTLSize;
use ndarray::Array2;
use std::ffi::c_void;

pub(super) const THREADS: u64 = 256;
pub(super) const TILE_ROWS: usize = 1024;
pub(super) const GRAM_ROWS: usize = 64;
pub(super) const GRAM_QUERIES: usize = 8;
pub(super) const MAX_K: usize = 2048;

#[repr(C)]
#[derive(Clone, Copy)]
pub(super) struct Params {
    pub(super) rows: u32,
    pub(super) dim: u32,
    pub(super) queries: u32,
    pub(super) tile_start: u32,
    pub(super) tile_rows: u32,
    pub(super) k: u32,
    pub(super) lanes: u32,
    pub(super) tiles: u32,
    pub(super) lists: u32,
    pub(super) groups: u32,
}

pub(super) struct Scratch {
    pub(super) query: metal::Buffer,
    pub(super) query_norms: metal::Buffer,
    pub(super) tile_distances: metal::Buffer,
    pub(super) local_distances: metal::Buffer,
    pub(super) local_indices: metal::Buffer,
    pub(super) result_distances: metal::Buffer,
    pub(super) result_indices: metal::Buffer,
    pub(super) tree_distances: metal::Buffer,
    pub(super) tree_indices: metal::Buffer,
    pub(super) tree_alt_distances: metal::Buffer,
    pub(super) tree_alt_indices: metal::Buffer,
    pub(super) query_capacity: usize,
    pub(super) k_capacity: usize,
    pub(super) tree_query_capacity: usize,
    pub(super) tree_k_capacity: usize,
    pub(super) tile_capacity: usize,
    pub(super) group_capacity: usize,
}

impl Scratch {
    pub(super) fn new(runtime: &Runtime) -> Self {
        let empty = || runtime.buffer::<u32>(1);
        Self {
            query: empty(),
            query_norms: empty(),
            tile_distances: empty(),
            local_distances: empty(),
            local_indices: empty(),
            result_distances: empty(),
            result_indices: empty(),
            tree_distances: empty(),
            tree_indices: empty(),
            tree_alt_distances: empty(),
            tree_alt_indices: empty(),
            query_capacity: 0,
            k_capacity: 0,
            tree_query_capacity: 0,
            tree_k_capacity: 0,
            tile_capacity: 0,
            group_capacity: 0,
        }
    }

    pub(super) fn ensure(&mut self, runtime: &Runtime, dim: usize, queries: usize, k: usize) {
        if queries <= self.query_capacity && k <= self.k_capacity {
            return;
        }
        self.query_capacity = next_capacity(queries);
        self.k_capacity = next_capacity(k);
        self.query = runtime.buffer::<f32>(self.query_capacity * dim);
        self.query_norms = runtime.buffer::<f32>(self.query_capacity);
        self.tile_distances = runtime.buffer::<f32>(self.query_capacity * TILE_ROWS);
        self.local_distances = runtime.buffer::<f32>(self.query_capacity * self.k_capacity);
        self.local_indices = runtime.buffer::<u32>(self.query_capacity * self.k_capacity);
        self.result_distances = runtime.buffer::<f32>(self.query_capacity * self.k_capacity);
        self.result_indices = runtime.buffer::<u32>(self.query_capacity * self.k_capacity);
    }

    pub(super) fn ensure_tree(
        &mut self,
        runtime: &Runtime,
        queries: usize,
        k: usize,
        tiles: usize,
        fan: usize,
    ) {
        let groups = tiles.div_ceil(fan);
        if queries <= self.tree_query_capacity
            && k <= self.tree_k_capacity
            && tiles <= self.tile_capacity
            && groups <= self.group_capacity
        {
            return;
        }
        self.tree_query_capacity = next_capacity(queries);
        self.tree_k_capacity = next_capacity(k);
        self.tile_capacity = next_capacity(tiles);
        self.group_capacity = next_capacity(groups);
        let lists = self.tree_query_capacity * self.tile_capacity * self.tree_k_capacity;
        let alt = self.tree_query_capacity * self.group_capacity * self.tree_k_capacity;
        self.tree_distances = runtime.buffer::<f32>(lists);
        self.tree_indices = runtime.buffer::<u32>(lists);
        self.tree_alt_distances = runtime.buffer::<f32>(alt);
        self.tree_alt_indices = runtime.buffer::<u32>(alt);
    }
}

pub(super) fn next_capacity(value: usize) -> usize {
    value
        .max(1)
        .checked_next_power_of_two()
        .unwrap_or(value.max(1))
}

pub(super) fn norms(values: &[f32], dim: usize) -> Vec<f32> {
    values
        .chunks_exact(dim)
        .map(|row| {
            row.iter()
                .fold(0.0, |sum, &value| value.mul_add(value, sum))
        })
        .collect()
}

pub(super) fn plan_lists(plan: Plan, rows: usize) -> usize {
    if plan == Plan::Gram {
        rows.div_ceil(GRAM_ROWS)
    } else {
        rows.div_ceil(TILE_ROWS)
    }
}

pub(super) fn plan_fan(k: usize) -> usize {
    if k <= 16 { TILE_ROWS / k } else { 2 }
}

pub(super) fn set_params(encoder: &metal::ComputeCommandEncoderRef, slot: u64, params: &Params) {
    encoder.set_bytes(
        slot,
        std::mem::size_of::<Params>() as u64,
        (params as *const Params).cast::<c_void>(),
    );
}

pub(super) fn dispatch(encoder: &metal::ComputeCommandEncoderRef, count: usize) {
    encoder.dispatch_thread_groups(
        MTLSize {
            width: count.div_ceil(THREADS as usize) as u64,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: THREADS,
            height: 1,
            depth: 1,
        },
    );
}

pub(super) fn same_distances(left: &[f32], right: &[f32]) -> bool {
    left.iter()
        .zip(right)
        .all(|(&left, &right)| same_distance(left, right))
}

pub(super) fn same_distance(left: f32, right: f32) -> bool {
    left == right
        || (left.is_finite()
            && right.is_finite()
            && (left - right).abs() <= 1e-4 * (1.0 + left.abs()))
}

pub(super) fn same_topk(
    left: &(Vec<f32>, Vec<u32>),
    right: &(Vec<f32>, Vec<u32>),
    queries: usize,
    k: usize,
    rows: usize,
) -> bool {
    if left.0.len() != queries * k
        || left.1.len() != queries * k
        || right.0.len() != queries * k
        || right.1.len() != queries * k
        || !same_distances(&left.0, &right.0)
    {
        return false;
    }
    for query in 0..queries {
        let range = query * k..(query + 1) * k;
        let left_distances = &left.0[range.clone()];
        let left_indices = &left.1[range.clone()];
        let right_distances = &right.0[range.clone()];
        let right_indices = &right.1[range];
        if left_indices
            .iter()
            .chain(right_indices)
            .any(|&index| index != u32::MAX && index as usize >= rows)
            || left_indices
                .iter()
                .enumerate()
                .any(|(rank, &index)| index != u32::MAX && left_indices[..rank].contains(&index))
            || right_indices
                .iter()
                .enumerate()
                .any(|(rank, &index)| index != u32::MAX && right_indices[..rank].contains(&index))
        {
            return false;
        }
        for (&distance, &index) in left_distances.iter().zip(left_indices) {
            if index == u32::MAX {
                if distance.is_finite() {
                    return false;
                }
                continue;
            }
            if let Some(rank) = right_indices.iter().position(|&right| right == index) {
                if !same_distance(distance, right_distances[rank]) {
                    return false;
                }
            } else if !same_distance(distance, left_distances[k - 1])
                || !same_distance(distance, right_distances[k - 1])
            {
                return false;
            }
        }
        for (&distance, &index) in right_distances.iter().zip(right_indices) {
            if index == u32::MAX {
                if distance.is_finite() {
                    return false;
                }
                continue;
            }
            if let Some(rank) = left_indices.iter().position(|&left| left == index) {
                if !same_distance(distance, left_distances[rank]) {
                    return false;
                }
            } else if !same_distance(distance, right_distances[k - 1])
                || !same_distance(distance, left_distances[k - 1])
            {
                return false;
            }
        }
    }
    true
}

pub(super) fn write_buffer(buffer: &metal::Buffer, offset: usize, values: &[f32]) {
    if values.is_empty() {
        return;
    }
    unsafe {
        std::ptr::copy_nonoverlapping(
            values.as_ptr(),
            buffer.contents().cast::<f32>().add(offset),
            values.len(),
        );
    }
}

pub(super) fn to_u32(value: usize, name: &str) -> Result<u32, IndexError> {
    u32::try_from(value)
        .map_err(|_| IndexError::InvalidParameter(format!("{name} exceeds u32 range")))
}

pub(super) fn unpack_results(
    scratch: &Scratch,
    queries: usize,
    k_eff: usize,
    search_k: usize,
) -> (Array2<f64>, Array2<i64>) {
    let distances = unsafe {
        std::slice::from_raw_parts(
            scratch.result_distances.contents().cast::<f32>(),
            queries * k_eff,
        )
    };
    let indices = unsafe {
        std::slice::from_raw_parts(
            scratch.result_indices.contents().cast::<u32>(),
            queries * k_eff,
        )
    };
    let mut out_dist = Array2::zeros((queries, k_eff));
    let mut out_idx = Array2::zeros((queries, k_eff));
    for q in 0..queries {
        for k in 0..k_eff {
            out_dist[[q, k]] = f64::from(distances[q * k_eff + k]);
            out_idx[[q, k]] = i64::from(indices[q * k_eff + k]);
        }
    }
    pad_k(out_dist, out_idx, search_k)
}

pub(super) fn direct_plan(request: super::KnnPlan, k: usize) -> Result<Option<Plan>, IndexError> {
    let direct = match request {
        super::KnnPlan::Measured => None,
        super::KnnPlan::Split => Some(Plan::Split),
        super::KnnPlan::Fused if k <= 16 => Some(Plan::Fused),
        super::KnnPlan::Tree if k <= 16 => Some(Plan::Tree),
        super::KnnPlan::Tiled if k <= 16 => Some(Plan::Tiled),
        super::KnnPlan::Simd if k <= 16 => Some(Plan::Simd),
        super::KnnPlan::Gram if k <= 16 => Some(Plan::Gram),
        super::KnnPlan::Wide => Some(Plan::Wide),
        super::KnnPlan::Fused
        | super::KnnPlan::Tree
        | super::KnnPlan::Tiled
        | super::KnnPlan::Simd
        | super::KnnPlan::Gram => {
            return Err(IndexError::InvalidParameter(
                "specialized KNN plans support at most 16 neighbors".to_string(),
            ));
        }
    };
    Ok(direct)
}

pub(super) fn plan_options(gram: bool, k: usize) -> &'static [Plan] {
    if gram {
        &[
            Plan::Split,
            Plan::Fused,
            Plan::Tree,
            Plan::Tiled,
            Plan::Simd,
            Plan::Gram,
        ]
    } else if k <= 16 {
        &[
            Plan::Split,
            Plan::Fused,
            Plan::Tree,
            Plan::Tiled,
            Plan::Simd,
        ]
    } else {
        &[Plan::Split, Plan::Wide]
    }
}
