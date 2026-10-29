#![feature(f16)]

extern crate self as baux;

#[path = "../../../../rust/crates/baux/semantics/src/lib.rs"]
pub mod semantics;

use cuda_device::{DisjointSlice, cuda_module, kernel, launch_contract, thread};

mod bf16;
use baux::semantics::neighbor as neighbor_rule;
pub use bf16::{
    Bf16Change, Bf16Leaf, Bf16Score, NeighborShape, SearchState, TellParams, TellSummary,
    WriteParams,
};
mod knn;
pub use knn::{
    BatchParams, BatchValue, DrawParams, KNN_K, KNN_THREADS, KNN_WARPS, KnnParams, MergeParams,
    PosteriorParams, WeightedParams,
};
mod diffusion;
mod fbt;
mod fbt_cache;
mod fbt_gemm;
mod fbt_math;
mod fbt_mhc;
mod fbt_obj;
mod fbt_pack;
mod fbt_pisa;
mod fbt_pool;
mod fbt_route;
mod fbt_stats;
mod fbt_stream;
mod fbt_types;
mod fbt_wave;
pub mod muon;
mod trial_bf16;
mod trial_knn;
mod trial_ops;
mod trial_score;
#[macro_use]
mod trial_tell;
#[cfg(test)]
mod trial_tests;
mod trial_write;
pub use diffusion::diffusion_model;
pub use fbt::{
    FbtShape, FeedbackShape, GaussianShape, MatmulShape, Mhc4EncodeShape, Mhc4Point, MoeShape,
    NeighborPoint, ObjectiveShape, PisaShape, RouteShape, RoutedTile, ScoreStats, ShardStats,
    VOCAB, WaveShape, fbt_model, mhc_model,
};
pub use trial_ops::*;
pub use trial_score::{
    CenterStep, Leaf, Seed, Selection, SparseEdit, SparseParams, Tile, TrialParams,
};

pub const MODULE_NAME: &str = env!("CARGO_PKG_NAME");
pub const THREADS: u32 = 256;
const WARPS: usize = (THREADS / 32) as usize;
pub use baux::semantics::neighbor::MAX_HISTORY;
pub const MAX_DEPTH: usize = 8;

#[cuda_module]
pub mod trials {
    use super::*;

    #[kernel(unchecked_indexing)]
    #[launch_contract(domain = 1, block = (256, 1, 1), dynamic_shared = 0)]
    pub fn neighbor_distances(
        pool: &[u16],
        history: &[u16],
        weights: &[f32],
        mut distances: DisjointSlice<f32>,
        shape: NeighborShape,
    ) {
        knn::run_neighbors(pool, history, weights, &mut distances, shape);
    }

    #[kernel(unchecked_indexing)]
    #[launch_contract(domain = 1, block = (256, 1, 1), dynamic_shared = 0)]
    pub fn scan_rows(
        rows: &[f32],
        queries: &[f32],
        mut distances: DisjointSlice<f32>,
        mut indices: DisjointSlice<u32>,
        params: KnnParams,
    ) {
        knn::run_scanrows(rows, queries, &mut distances, &mut indices, params);
    }

    #[kernel(unchecked_indexing)]
    #[launch_contract(domain = 1, block = (256, 1, 1), dynamic_shared = 0)]
    pub fn scan_warps(
        rows: &[f32],
        queries: &[f32],
        mut distances: DisjointSlice<f32>,
        mut indices: DisjointSlice<u32>,
        params: KnnParams,
    ) {
        knn::run_scanwarps(rows, queries, &mut distances, &mut indices, params);
    }

    #[kernel(unchecked_indexing)]
    #[launch_contract(domain = 1, block = (256, 1, 1), dynamic_shared = 0)]
    pub fn merge_topk(
        input_distances: &[f32],
        input_indices: &[u32],
        mut output_distances: DisjointSlice<f32>,
        mut output_indices: DisjointSlice<u32>,
        params: MergeParams,
    ) {
        knn::run_mergetopk(
            input_distances,
            input_indices,
            &mut output_distances,
            &mut output_indices,
            params,
        );
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
    pub fn score_trials(
        rows: &[u8],
        history_slots: &[u32],
        outcomes: &[f32],
        seeds: &[Seed],
        draws: &[f32],
        leaves: &[Leaf],
        centers: &[CenterStep],
        candidate_centers: &[u32],
        mut scores: DisjointSlice<f32>,
        params: TrialParams,
    ) {
        trial_score::run_trials(
            rows,
            history_slots,
            outcomes,
            seeds,
            draws,
            leaves,
            centers,
            candidate_centers,
            &mut scores,
            params,
        );
    }

    #[kernel(unchecked_indexing)]
    #[launch_contract(domain = 1, block = (256, 1, 1), dynamic_shared = 0)]
    pub fn distance_bf16(
        rows: &[u16],
        history_slots: &[u32],
        reference: &[u16],
        reference_scales: &[f32],
        state: &[SearchState],
        seeds: &[Seed],
        leaves: &[Bf16Leaf],
        tiles: &[DenseTile],
        mut partials: DisjointSlice<f32>,
        mut tile_status: DisjointSlice<u32>,
        params: Bf16Score,
    ) {
        trial_bf16::run_distbf16(
            rows,
            history_slots,
            reference,
            reference_scales,
            state,
            seeds,
            leaves,
            tiles,
            &mut partials,
            &mut tile_status,
            params,
        );
    }

    #[kernel(unchecked_indexing)]
    #[launch_contract(domain = 1, block = (256, 1, 1), dynamic_shared = 0)]
    pub fn distance2_bf16(
        rows: &[u16],
        history_slots: &[u32],
        reference: &[u16],
        reference_scales: &[f32],
        state: &[SearchState],
        seeds: &[Seed],
        leaves: &[Bf16Leaf],
        tiles: &[DenseTile],
        mut partials: DisjointSlice<f32>,
        mut tile_status: DisjointSlice<u32>,
        params: Bf16Score,
    ) {
        trial_bf16::run_dist2bf16(
            rows,
            history_slots,
            reference,
            reference_scales,
            state,
            seeds,
            leaves,
            tiles,
            &mut partials,
            &mut tile_status,
            params,
        );
    }

    #[kernel(unchecked_indexing)]
    #[launch_contract(domain = 1, block = (256, 1, 1), dynamic_shared = 0)]
    pub fn score_bf16(
        partials: &[f32],
        tile_status: &[u32],
        outcomes: &[f32],
        variances: &[f32],
        state: &[SearchState],
        draws: &[f32],
        mut scores: DisjointSlice<f32>,
        params: Bf16Score,
    ) {
        trial_bf16::run_scorebf16(
            partials,
            tile_status,
            outcomes,
            variances,
            state,
            draws,
            &mut scores,
            params,
        );
    }

    #[kernel]
    #[launch_contract(domain = 1, block = (256, 1, 1), dynamic_shared = 0)]
    pub fn draw_bf16(
        mut draws: DisjointSlice<f32>,
        history_slots: &[u32],
        state: &[SearchState],
        seed: u64,
        mut count: u32,
        resident: u32,
    ) {
        if resident != 0 {
            count = state[0].history;
        }
        let mut index = thread::threadIdx_x() + thread::blockIdx_x() * THREADS;
        let stride = thread::gridDim_x() * THREADS;
        while index < count {
            unsafe {
                *draws.get_unchecked_mut(index as usize) =
                    draw_normal(seed, history_slots[index as usize], 0) as f32;
            }
            index += stride;
        }
    }

    #[kernel]
    #[launch_contract(domain = 1, block = (256, 1, 1), dynamic_shared = 0)]
    pub fn seed_bf16(mut seeds: DisjointSlice<Seed>, root: u64, count: u32, correlated: u32) {
        let mut index = thread::threadIdx_x() + thread::blockIdx_x() * THREADS;
        let stride = thread::gridDim_x() * THREADS;
        while index < count {
            let root_low = root as u32;
            let root_high = (root >> 32) as u32;
            let stream = if correlated == 1 { index / 2 } else { index };
            unsafe {
                *seeds.get_unchecked_mut(index as usize) = Seed {
                    low: trial_hash(root_low, root_high, stream),
                    high: trial_hash(root_high, root_low, stream ^ 0x9e37_79b9),
                };
            }
            index += stride;
        }
    }

    #[kernel(unchecked_indexing)]
    #[launch_contract(domain = 1, block = (256, 1, 1), dynamic_shared = 0)]
    pub fn base_distance(
        rows: &[u8],
        history_slots: &[u32],
        leaves: &[Leaf],
        mut distances: DisjointSlice<f32>,
        row_stride: u32,
        history: u32,
        base_slot: u32,
    ) {
        trial_write::run_basedist(
            rows,
            history_slots,
            leaves,
            &mut distances,
            row_stride,
            history,
            base_slot,
        );
    }

    #[kernel(unchecked_indexing)]
    #[launch_contract(domain = 1, block = (256, 1, 1), dynamic_shared = 0)]
    pub fn score_sparse(
        rows: &[u8],
        history_slots: &[u32],
        outcomes: &[f32],
        seeds: &[Seed],
        draws: &[f32],
        leaves: &[Leaf],
        edits: &[SparseEdit],
        base_distances: &[f32],
        mut scores: DisjointSlice<f32>,
        params: SparseParams,
    ) {
        trial_score::run_sparse(
            rows,
            history_slots,
            outcomes,
            seeds,
            draws,
            leaves,
            edits,
            base_distances,
            &mut scores,
            params,
        );
    }

    #[kernel]
    #[launch_contract(domain = 1, block = (256, 1, 1), dynamic_shared = 0)]
    pub fn pick_trial(
        scores: &[f32],
        mut selection: DisjointSlice<Selection>,
        regions: u32,
        candidates_per_region: u32,
    ) {
        trial_write::run_pick(scores, &mut selection, regions, candidates_per_region);
    }

    #[kernel]
    #[launch_contract(domain = 1, block = (256, 1, 1), dynamic_shared = 0)]
    pub fn write_trial(
        mut rows: DisjointSlice<u8>,
        seeds: &[Seed],
        selection: &[Selection],
        leaves: &[Leaf],
        tiles: &[Tile],
        row_stride: u32,
        base_slot: u32,
        trial_slot: u32,
    ) {
        trial_write::run_writetrial(
            &mut rows, seeds, selection, leaves, tiles, row_stride, base_slot, trial_slot,
        );
    }

    #[kernel]
    #[launch_contract(domain = 1, block = (1, 1, 1), dynamic_shared = 0)]
    pub fn write_sparse(
        mut rows: DisjointSlice<u8>,
        seeds: &[Seed],
        selection: &[Selection],
        leaves: &[Leaf],
        edits: &[SparseEdit],
        row_stride: u32,
        trial_slot: u32,
        num_pert: u32,
    ) {
        trial_write::run_writesparse(
            &mut rows, seeds, selection, leaves, edits, row_stride, trial_slot, num_pert,
        );
    }

    #[kernel]
    pub fn materialize(
        base: &[u8],
        mut output: DisjointSlice<u8>,
        byte_offset: u32,
        element_offset: u32,
        length: u32,
        bits: u32,
        seed_low: u32,
        seed_high: u32,
        whole: u32,
        threshold: u32,
    ) {
        trial_write::run_materialize(
            base,
            &mut output,
            byte_offset,
            element_offset,
            length,
            bits,
            seed_low,
            seed_high,
            whole,
            threshold,
        );
    }

    #[kernel]
    pub fn apply_dense(
        base: &[f32],
        leaves: &[DenseLeaf],
        terms: &[DenseTerm],
        tiles: &[DenseTile],
        mut output: DisjointSlice<f32>,
        term_count: u32,
    ) {
        trial_write::run_dense(base, leaves, terms, tiles, &mut output, term_count);
    }

    #[kernel]
    pub fn apply_bf16(
        base: &[u16],
        leaves: &[DenseLeaf],
        terms: &[DenseTerm],
        tiles: &[DenseTile],
        mut output: DisjointSlice<u16>,
        mut status: DisjointSlice<u32>,
        term_count: u32,
    ) {
        trial_tell::run_applybf16(
            base,
            leaves,
            terms,
            tiles,
            &mut output,
            &mut status,
            term_count,
        );
    }

    #[kernel]
    #[launch_contract(domain = 1, block = (256, 1, 1), dynamic_shared = 0)]
    pub fn write_bf16(
        mut rows: DisjointSlice<u16>,
        mut batch: DisjointSlice<u16>,
        mut changes: DisjointSlice<Bf16Change>,
        state: &[SearchState],
        seeds: &[Seed],
        reference: &[u16],
        reference_scales: &[f32],
        selection: &[Selection],
        leaves: &[Bf16Leaf],
        tiles: &[DenseTile],
        mut status: DisjointSlice<u32>,
        params: WriteParams,
    ) {
        trial_tell::run_writebf16(
            &mut rows,
            &mut batch,
            &mut changes,
            state,
            seeds,
            reference,
            reference_scales,
            selection,
            leaves,
            tiles,
            &mut status,
            params,
        );
    }

    #[kernel]
    #[launch_contract(domain = 1, block = (256, 1, 1), dynamic_shared = 0)]
    pub fn tell_bf16(
        mut history_slots: DisjointSlice<u32>,
        mut outcomes: DisjointSlice<f32>,
        mut variances: DisjointSlice<f32>,
        mut destinations: DisjointSlice<u32>,
        values: &[f32],
        trial_variances: &[f32],
        proposal_status: &[u32],
        mut accepted: DisjointSlice<u32>,
        mut state: DisjointSlice<SearchState>,
        mut summary: DisjointSlice<TellSummary>,
        selection: &[Selection],
        params: TellParams,
    ) {
        trial_tell::run_tellbf16(
            &mut history_slots,
            &mut outcomes,
            &mut variances,
            &mut destinations,
            values,
            trial_variances,
            proposal_status,
            &mut accepted,
            &mut state,
            &mut summary,
            selection,
            params,
        );
    }

    #[kernel]
    #[launch_contract(domain = 1, block = (256, 1, 1), dynamic_shared = 0)]
    pub fn copy_tellbf16(
        mut rows: DisjointSlice<u16>,
        destinations: &[u32],
        summary: &[TellSummary],
        params: TellParams,
    ) {
        trial_tell::run_copytell(&mut rows, destinations, summary, params);
    }

    #[kernel]
    #[launch_contract(domain = 1, block = (256, 1, 1), dynamic_shared = 0)]
    pub fn reference_bf16(
        mut reference: DisjointSlice<u16>,
        mut partials: DisjointSlice<f32>,
        reference_scales: &[f32],
        leaves: &[Bf16Leaf],
        tiles: &[DenseTile],
        selection: &[Selection],
        seeds: &[Seed],
        accepted: &[u32],
        summary: &[TellSummary],
        initial_seed: u64,
        initialize: u32,
    ) {
        trial_tell::run_refbf16(
            &mut reference,
            &mut partials,
            reference_scales,
            leaves,
            tiles,
            selection,
            seeds,
            accepted,
            summary,
            initial_seed,
            initialize,
        );
    }

    #[kernel]
    #[launch_contract(domain = 1, block = (256, 1, 1), dynamic_shared = 0)]
    pub fn reference_rmsbf16(
        partials: &[f32],
        leaves: &[Bf16Leaf],
        tile_offsets: &[u32],
        mut scales: DisjointSlice<f32>,
        accepted: &[u32],
        summary: &[TellSummary],
        initialize: u32,
    ) {
        trial_tell::run_rmsbf16(
            partials,
            leaves,
            tile_offsets,
            &mut scales,
            accepted,
            summary,
            initialize,
        );
    }

    #[kernel]
    #[launch_contract(domain = 1, block = (256, 1, 1), dynamic_shared = 0)]
    pub fn check_search(
        rows: &[u16],
        leaves: &[Bf16Leaf],
        tiles: &[DenseTile],
        mut status: DisjointSlice<u32>,
        row_stride: u64,
        slot: u32,
    ) {
        trial_ops::run_checksearch(rows, leaves, tiles, &mut status, row_stride, slot);
    }

    #[kernel]
    pub fn check_bf16(
        base: &[u16],
        leaves: &[DenseLeaf],
        tiles: &[DenseTile],
        mut status: DisjointSlice<u32>,
    ) {
        trial_ops::run_checkbf16(base, leaves, tiles, &mut status);
    }

    #[kernel]
    #[launch_contract(domain = 1, block = (256, 1, 1), dynamic_shared = 0)]
    pub fn dense_linear(
        input: &[f32],
        weight: &[f32],
        bias: &[f32],
        terms: &[DenseTerm],
        mut output: DisjointSlice<f32>,
        params: DenseLinearParams,
    ) {
        trial_write::run_linear(input, weight, bias, terms, &mut output, params);
    }
}
