#![allow(incomplete_features, internal_features)]
#![feature(
    f16,
    adt_const_params,
    contracts,
    diagnostic_on_move,
    diagnostic_on_type_error,
    negative_impls,
    portable_simd
)]

extern crate self as baux;

#[macro_use]
mod macros {
    macro_rules! buck_mod {
        ($item:item) => {
            $item
        };
    }
}

#[cfg(buck_build)]
buck_mod! {
    #[path = "../../baux-source/semantics/src/lib.rs"]
    pub mod semantics;
}
#[cfg(not(buck_build))]
#[path = "../../../baux/semantics/src/lib.rs"]
pub mod semantics;

use cuda_device::{DisjointSlice, SharedArray, cuda_module, kernel, launch_contract, thread};

mod neighbor_shape;
use baux::semantics::neighbor as neighbor_rule;
pub use neighbor_shape::NeighborShape;
mod knn;
pub use knn::{
    BatchParams, BatchValue, DrawParams, KNN_K, KNN_THREADS, KNN_WARPS, KnnParams, MergeParams,
    POST_MAX_K, PosteriorParams, WeightedParams,
};
mod diffusion;
mod fbt;
mod fbt_cache;
mod fbt_eval;
mod fbt_gemm;
mod fbt_math;
mod fbt_moe;
mod fbt_obj;
mod fbt_pack;
mod fbt_pisa;
mod fbt_pool;
mod fbt_route;
mod fbt_stream;
mod fbt_types;
pub mod muon;
mod pisa_refine;
mod pisa_score;
mod pisa_select;
pub use diffusion::diffusion_model;
pub use fbt::{
    FbtShape, FeedbackShape, GaussianShape, MatmulShape, Mhc4EncodeShape, Mhc4Point, MoeShape,
    NeighborPoint, ObjectiveShape, PisaShape, RouteShape, RoutedTile, ScoreStats, ShardStats,
    VOCAB, WaveShape, fbt_model,
};

pub const MODULE_NAME: &str = env!("CARGO_PKG_NAME");
pub const THREADS: u32 = 256;
const WARPS: usize = (THREADS / 32) as usize;
pub use baux::semantics::neighbor::MAX_HISTORY;
pub const MAX_DEPTH: usize = 8;

buck_mod! {
#[rustfmt::skip]
#[cuda_module]
pub mod trials {
    use super::*;

    #[kernel(unchecked_indexing)]
    #[launch_contract(domain = 1, block = (256, 1, 1), dynamic_shared = 0)]
    pub fn neighbor_distances(pool: &[f32], history: &[f32], weights: &[f32], mut distances: DisjointSlice<f32>, shape: NeighborShape) {
        knn::run_neighbors(pool, history, weights, &mut distances, shape);
    }

    #[kernel(unchecked_indexing)]
    #[launch_contract(domain = 1, block = (256, 1, 1), dynamic_shared = 0)]
    pub fn scan_rows(rows: &[f32], queries: &[f32], mut distances: DisjointSlice<f32>, mut indices: DisjointSlice<u32>, params: KnnParams) {
        knn::run_scanrows(rows, queries, &mut distances, &mut indices, params);
    }

    #[kernel(unchecked_indexing)]
    #[launch_contract(domain = 1, block = (256, 1, 1), dynamic_shared = 0)]
    pub fn scan_warps(rows: &[f32], queries: &[f32], mut distances: DisjointSlice<f32>, mut indices: DisjointSlice<u32>, params: KnnParams) {
        knn::run_scanwarps(rows, queries, &mut distances, &mut indices, params);
    }

    #[kernel(unchecked_indexing)]
    #[launch_contract(domain = 1, block = (256, 1, 1), dynamic_shared = 0)]
    pub fn merge_topk(input_distances: &[f32], input_indices: &[u32], mut output_distances: DisjointSlice<f32>, mut output_indices: DisjointSlice<u32>, params: MergeParams) {
        knn::run_mergetopk(input_distances, input_indices, &mut output_distances, &mut output_indices, params);
    }
}
}
