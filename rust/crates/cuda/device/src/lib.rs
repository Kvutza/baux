#![allow(incomplete_features, internal_features, unsafe_op_in_unsafe_fn)]
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
pub use baux::semantics::abi::{ArmDecision, PoolDispatch, SelectHeader};
pub use neighbor_shape::NeighborShape;
mod knn;
pub use knn::{
    BatchParams, BatchValue, CAND_TILE, DrawParams, KNN_K, KNN_THREADS, KNN_WARPS, KnnParams,
    MergeParams, POST_MAX_K, PosteriorParams, WeightedParams, make_pool,
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
mod trial_knn;
pub use diffusion::diffusion_model;
pub use fbt::{
    FbtShape, FeedbackShape, GaussianShape, MatmulShape, Mhc4EncodeShape, Mhc4Point, MoeShape,
    NeighborPoint, ObjectiveShape, PisaShape, RouteShape, RoutedTile, ScoreStats, ShardStats,
    VOCAB, WaveShape, fbt_model,
};
pub use trial_knn::{
    MAX_METRICS, MOD_AFFINE, MOD_CORR, MOD_MORBO, MOD_STIEFEL, ModSlices, SCORE_THREADS,
    better_arm, eval_law, eval_topk, metric_pair, pick_host, score_cand, unpack_mods,
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

    #[kernel(unchecked_indexing)]
    #[launch_contract(domain = 1, block = (256, 1, 1), dynamic_shared = 0)]
    pub fn posterior_light(
        distances: &[f32],
        indices: &[u32],
        outcomes: &[f32],
        scales: &[f32],
        mut means: DisjointSlice<f32>,
        mut errors: DisjointSlice<f32>,
        mut selected: DisjointSlice<u32>,
        params: PosteriorParams,
    ) {
        trial_knn::run_postlight(
            distances,
            indices,
            outcomes,
            scales,
            &mut means,
            &mut errors,
            &mut selected,
            params,
        );
    }

    #[kernel(unchecked_indexing)]
    #[launch_contract(domain = 1, block = (256, 1, 1), dynamic_shared = 0)]
    pub fn posterior_full(
        distances: &[f32],
        indices: &[u32],
        outcomes: &[f32],
        variances: &[f32],
        scales: &[f32],
        mut weights_out: DisjointSlice<f32>,
        mut l2_out: DisjointSlice<f32>,
        mut means: DisjointSlice<f32>,
        mut errors: DisjointSlice<f32>,
        mut epistemic: DisjointSlice<f32>,
        mut aleatoric: DisjointSlice<f32>,
        mut selected: DisjointSlice<u32>,
        params: WeightedParams,
    ) {
        trial_knn::run_postfull(
            distances,
            indices,
            outcomes,
            variances,
            scales,
            &mut weights_out,
            &mut l2_out,
            &mut means,
            &mut errors,
            &mut epistemic,
            &mut aleatoric,
            &mut selected,
            params,
        );
    }

    #[kernel(unchecked_indexing)]
    #[launch_contract(domain = 1, block = (256, 1, 1), dynamic_shared = 0)]
    pub fn posterior_batch(
        distances: &[f32],
        indices: &[u32],
        outcomes: &[f32],
        variances: &[f32],
        scales: &[f32],
        values: &[BatchValue],
        mut means: DisjointSlice<f32>,
        mut errors: DisjointSlice<f32>,
        mut epistemic: DisjointSlice<f32>,
        mut aleatoric: DisjointSlice<f32>,
        params: BatchParams,
    ) {
        trial_knn::run_postbatch(
            distances,
            indices,
            outcomes,
            variances,
            scales,
            values,
            &mut means,
            &mut errors,
            &mut epistemic,
            &mut aleatoric,
            params,
        );
    }

    #[kernel(unchecked_indexing)]
    #[launch_contract(domain = 1, block = (256, 1, 1), dynamic_shared = 0)]
    pub fn posterior_draw(
        weights: &[f32],
        l2: &[f32],
        means: &[f32],
        errors: &[f32],
        indices: &[u32],
        seeds: &[u64],
        mut draws: DisjointSlice<f64>,
        params: DrawParams,
    ) {
        trial_knn::run_postdraw(
            weights, l2, means, errors, indices, seeds, &mut draws, params,
        );
    }

    #[kernel(unchecked_indexing)]
    #[launch_contract(domain = 1, block = (256, 1, 1), dynamic_shared = 0)]
    pub fn poolscan(
        rows: &[f32],
        weights: &[f32],
        center: &[f32],
        scale: &[f32],
        lower: &[f32],
        upper: &[f32],
        dirs: &[u32],
        mut distances: DisjointSlice<f32>,
        mut indices: DisjointSlice<u32>,
        dispatch: PoolDispatch,
    ) {
        knn::run_poolscan(
            rows,
            weights,
            center,
            scale,
            lower,
            upper,
            dirs,
            &mut distances,
            &mut indices,
            dispatch,
        );
    }

    #[kernel(unchecked_indexing)]
    #[launch_contract(domain = 1, block = (256, 1, 1), dynamic_shared = 0)]
    pub fn poolwarp(
        rows: &[f32],
        weights: &[f32],
        center: &[f32],
        scale: &[f32],
        lower: &[f32],
        upper: &[f32],
        dirs: &[u32],
        mut distances: DisjointSlice<f32>,
        mut indices: DisjointSlice<u32>,
        dispatch: PoolDispatch,
    ) {
        knn::run_poolwarp(
            rows,
            weights,
            center,
            scale,
            lower,
            upper,
            dirs,
            &mut distances,
            &mut indices,
            dispatch,
        );
    }

    #[kernel(unchecked_indexing)]
    #[launch_contract(domain = 1, block = (32, 1, 1), dynamic_shared = 0)]
    pub fn poolscore(
        distances: &[f32],
        indices: &[u32],
        outcomes: &[f32],
        variances: &[f32],
        mods: &[f32],
        mut scores: DisjointSlice<f32>,
        mut means: DisjointSlice<f32>,
        mut errors: DisjointSlice<f32>,
        header: SelectHeader,
    ) {
        trial_knn::run_poolscore(
            distances,
            indices,
            outcomes,
            variances,
            mods,
            &mut scores,
            &mut means,
            &mut errors,
            header,
        );
    }

    #[kernel(unchecked_indexing)]
    #[launch_contract(domain = 1, block = (256, 1, 1), dynamic_shared = 0)]
    pub fn poolpick(
        scores: &[f32],
        means: &[f32],
        errors: &[f32],
        mut state: DisjointSlice<u32>,
        mut selected: DisjointSlice<u32>,
        mut decisions: DisjointSlice<ArmDecision>,
        header: SelectHeader,
    ) {
        trial_knn::run_poolpick(
            scores,
            means,
            errors,
            &mut state,
            &mut selected,
            &mut decisions,
            header,
        );
    }

    #[kernel(unchecked_indexing)]
    #[launch_contract(domain = 1, block = (256, 1, 1), dynamic_shared = 0)]
    pub fn poolrows(
        selected: &[u32],
        center: &[f32],
        scale: &[f32],
        lower: &[f32],
        upper: &[f32],
        dirs: &[u32],
        mut rows_out: DisjointSlice<f32>,
        dispatch: PoolDispatch,
    ) {
        knn::run_poolrows(
            selected,
            center,
            scale,
            lower,
            upper,
            dirs,
            &mut rows_out,
            dispatch,
        );
    }
}
}
