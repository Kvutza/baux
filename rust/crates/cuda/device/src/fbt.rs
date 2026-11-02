use cuda_device::{
    DisjointSlice, cuda_module, kernel, launch_bounds, launch_contract, ptx_asm, thread,
};

pub use super::fbt_math::{GaussianShape, NeighborPoint, reduce_max, reduce_sum};
pub use super::fbt_types::*;
use super::{
    fbt_cache, fbt_eval, fbt_gemm, fbt_math, fbt_moe, fbt_obj, fbt_pack, fbt_pisa, fbt_pool,
    fbt_stream, fbt_types,
};

buck_mod! {
#[rustfmt::skip]
#[cuda_module]
pub mod fbt_model {
    use super::*;

    #[kernel(unchecked_indexing)]
    #[launch_contract(domain = 1, block = (256, 1, 1), dynamic_shared = 0)]
    pub fn rope(mut qkv: DisjointSlice<f32>, table: &[f32], rows: u32, context: u32, first: u32) {
        fbt_math::run_rope(&mut qkv, table, rows, context, first);
    }

    #[kernel(unchecked_indexing)]
    #[launch_contract(domain = 1, block = (256, 1, 1), dynamic_shared = 0)]
    pub fn mhc_replicate(input: &[f32], mut streams: DisjointSlice<f32>, rows: u32) {
        fbt_math::run_replicate(input, &mut streams, rows);
    }

    #[kernel(unchecked_indexing)]
    #[launch_contract(domain = 1, block = (256, 1, 1), dynamic_shared = 0)]
    pub fn mhc_predict(streams: &[f32], predictor: &[f32], bias: &[f32], control: &[f32], mut coefficients: DisjointSlice<f32>, rows: u32, parallel: u32) {
        fbt_stream::run_predict(streams, predictor, bias, control, &mut coefficients, rows, parallel);
    }

    #[kernel(unchecked_indexing)]
    #[launch_contract(domain = 1, block = (256, 1, 1), dynamic_shared = 0)]
    pub fn mhc_mix(streams: &[f32], coefficients: &[f32], weight: &[f32], mut output: DisjointSlice<f32>, rows: u32, mean: u32) {
        fbt_stream::run_mix(streams, coefficients, weight, &mut output, rows, mean);
    }

    #[kernel(unchecked_indexing)]
    #[launch_contract(domain = 1, block = (256, 1, 1), dynamic_shared = 0)]
    pub fn route_counts(experts: &[u32], mut counts: DisjointSlice<u32>, mut prefixes: DisjointSlice<u32>, assignments: u32) {
        fbt_moe::run_counts(experts, &mut counts, &mut prefixes, assignments);
    }

    #[kernel(unchecked_indexing)]
    #[launch_contract(domain = 1, block = (32, 1, 1), dynamic_shared = 0)]
    pub fn route_layout(counts: &[u32], mut offsets: DisjointSlice<u32>, mut tiles: DisjointSlice<RoutedTile>, experts: u32, tile_capacity: u32) {
        fbt_obj::run_layout(counts, &mut offsets, &mut tiles, experts, tile_capacity);
    }

    #[kernel(unchecked_indexing)]
    #[launch_contract(domain = 1, block = (256, 1, 1), dynamic_shared = 0)]
    pub fn route_pack(input: &[f32], experts: &[u32], offsets: &[u32], prefixes: &[u32], mut mapping: DisjointSlice<u32>, mut packed: DisjointSlice<f32>, assignments: u32) {
        fbt_moe::run_pack(input, experts, offsets, prefixes, &mut mapping, &mut packed, assignments);
    }

    #[kernel(unchecked_indexing)]
    #[launch_contract(domain = 1, block = (256, 1, 1), dynamic_shared = 0)]
    pub fn verify_prefix(predicted: &[u32], mut tokens: DisjointSlice<u32>, mut frontier: DisjointSlice<u32>, rows: u32) {
        fbt_pack::run_verify(predicted, &mut tokens, &mut frontier, rows);
    }

    #[kernel(unchecked_indexing)]
    #[launch_contract(domain = 1, block = (256, 1, 1), dynamic_shared = 0)]
    pub fn sample(logits: &[f32], mut tokens: DisjointSlice<u32>, rows: u32, temperature: f32, seed: u64, first_row: u32) {
        fbt_pack::run_sample(logits, &mut tokens, rows, temperature, seed, first_row);
    }

    #[kernel(unchecked_indexing)]
    #[launch_contract(domain = 1, block = (256, 1, 1), dynamic_shared = 0)]
    pub fn expand_patches(macros: &[u32], readout: &[f32], mut tokens: DisjointSlice<u32>, patches: u32, first_patch: u32) {
        fbt_pack::run_expand(macros, readout, &mut tokens, patches, first_patch);
    }

    #[kernel(unchecked_indexing)]
    #[launch_contract(domain = 1, block = (256, 1, 1), dynamic_shared = 0)]
    pub fn score(logits: &[f32], targets: &[u32], mut losses: DisjointSlice<f32>, rows: u32, target_offset: u32, loss_offset: u32) {
        fbt_pack::run_score(logits, targets, &mut losses, rows, target_offset, loss_offset);
    }

    #[kernel(unchecked_indexing)]
    #[launch_contract(domain = 1, block = (256, 1, 1), dynamic_shared = 0)]
    pub fn objective(losses: &[f32], generated: &[u32], targets: &[u32], mut output: DisjointSlice<ScoreStats>, shape: ObjectiveShape) {
        fbt_eval::run_objective(losses, generated, targets, &mut output, shape);
    }

    #[kernel(unchecked_indexing)]
    #[launch_contract(domain = 1, block = (256, 1, 1), dynamic_shared = 0)]
    pub fn embed_packed(weights: &[f32], tokens: &[u32], mut output: DisjointSlice<f32>, shape: FbtShape) {
        fbt_math::run_embed(weights, tokens, &mut output, shape);
    }

    #[kernel(unchecked_indexing)]
    #[launch_contract(domain = 1, block = (256, 1, 1), dynamic_shared = 0)]
    pub fn residual(input: &[f32], branch: &[f32], mut output: DisjointSlice<f32>, elements: u32) {
        fbt_math::run_residual(input, branch, &mut output, elements);
    }

    #[kernel(unchecked_indexing)]
    #[launch_contract(domain = 1, block = (256, 1, 1), dynamic_shared = 0)]
    pub fn rms(input: &[f32], weight: &[f32], mut output: DisjointSlice<f32>, shape: FbtShape) {
        fbt_math::run_rms(input, weight, &mut output, shape);
    }

    #[kernel(unchecked_indexing)]
    #[launch_contract(domain = 2, block = (16, 16, 1), dynamic_shared = 0)]
    pub fn matmul(input: &[f32], weight: &[f32], mut output: DisjointSlice<f32, thread::Runtime2DIndex>, shape: MatmulShape) {
        fbt_gemm::run_matmul(input, weight, &mut output, shape);
    }

    #[kernel(unchecked_indexing)]
    #[launch_contract(domain = 2, block = (256, 1, 1), dynamic_shared = 0)]
    pub fn matmul_turing(input: &[f32], weight: &[f32], mut output: DisjointSlice<f32, thread::Runtime2DIndex>, shape: MatmulShape) {
        fbt_gemm::turing::<256>(input, weight, output.as_mut_ptr(), shape, thread::blockIdx_y() * 64);
    }

    #[kernel(unchecked_indexing)]
    #[launch_contract(domain = 2, block = (256, 1, 1), dynamic_shared = 0)]
    pub fn routed_project(input: &[f32], weights: &[f32], routes: &[RoutedTile], mut output: DisjointSlice<f32, thread::Runtime2DIndex>, mode: u32) {
        let tile = thread::blockIdx_y();
        fbt_moe::routed::<256>(input, weights, routes, output.as_mut_ptr(), mode, tile, 0);
    }

    #[kernel(unchecked_indexing)]
    #[launch_contract(domain = 2, block = (64, 1, 1), dynamic_shared = 0)]
    pub fn routed_compact(input: &[f32], weights: &[f32], routes: &[RoutedTile], mut output: DisjointSlice<f32, thread::Runtime2DIndex>, mode: u32) {
        let block = thread::blockIdx_y();
        fbt_moe::routed::<64>(input, weights, routes, output.as_mut_ptr(), mode, block / 4, (block % 4) * 16);
    }

    #[kernel(unchecked_indexing)]
    #[launch_contract(domain = 1, block = (256, 1, 1), dynamic_shared = 0)]
    pub fn routed_activate(projection: &[f32], mut output: DisjointSlice<f32>, rows: u32) {
        fbt_math::run_activate(projection, &mut output, rows);
    }

    #[kernel(unchecked_indexing)]
    #[launch_contract(domain = 1, block = (64, 1, 1), dynamic_shared = 0)]
    pub fn pisa_select(qkv: &[f32], pyramid: &[f32], mut blocks: DisjointSlice<u32>, shape: PisaShape) {
        fbt_cache::run_select(qkv, pyramid, &mut blocks, shape);
    }

    #[kernel(unchecked_indexing)]
    #[launch_contract(domain = 1, block = (64, 1, 1), dynamic_shared = 0)]
    pub fn pisa_shadow_refine(qkv: &[f32], pyramid: &[f32], incumbent: &[u32], mut blocks: DisjointSlice<u32>, visit_offset: u32, shape: PisaShape) {
        fbt_pisa::run_refine(qkv, pyramid, incumbent, &mut blocks, visit_offset, shape);
    }

    #[kernel(unchecked_indexing)]
    #[launch_contract(domain = 1, block = (256, 1, 1), dynamic_shared = 0)]
    pub fn routetopk(scores_input: &[f32], experts_output: DisjointSlice<u32>, weights_output: DisjointSlice<f32>, margin_output: DisjointSlice<f32>, shape: RouteShape) {
        fbt_pisa::run_routetopk(scores_input, experts_output, weights_output, margin_output, shape);
    }

    #[kernel(unchecked_indexing)]
    #[launch_contract(domain = 1, block = (256, 1, 1), dynamic_shared = 0)]
    pub fn pisa_attention(qkv: &[f32], kv: &[f32], blocks: &[u32], output: DisjointSlice<f32>, shape: PisaShape) {
        fbt_pisa::run_attention(qkv, kv, blocks, output, shape);
    }

    #[kernel(unchecked_indexing)]
    #[launch_contract(domain = 1, block = (256, 1, 1), dynamic_shared = 0)]
    pub fn pisa_multihead(qkv: &[f32], kv: &[f32], blocks: &[u32], output: DisjointSlice<f32>, shape: PisaShape) {
        fbt_pisa::run_multihead(qkv, kv, blocks, output, shape);
    }

    #[kernel(unchecked_indexing)]
    #[launch_contract(domain = 2, block = (64, 1, 1), dynamic_shared = 0)]
    pub fn pisaleaves(qkv: &[f32], mut pyramid: DisjointSlice<f32, thread::Runtime2DIndex>, shape: PisaShape) {
        fbt_cache::run_pisaleaves(qkv, &mut pyramid, shape);
    }

    #[kernel(unchecked_indexing)]
    #[launch_contract(domain = 1, block = (64, 1, 1), dynamic_shared = 0)]
    pub fn cache_leaves(kv: &[f32], mut tree: DisjointSlice<f32>, first: u32, count: u32) {
        fbt_cache::run_cacheleaves(kv, &mut tree, first, count);
    }

    #[kernel(unchecked_indexing)]
    #[launch_contract(domain = 1, block = (256, 1, 1), dynamic_shared = 0)]
    pub fn cache_split(qkv: &[f32], mut queries: DisjointSlice<f32>, mut kv: DisjointSlice<f32>, rows: u32, first: u32) {
        fbt_cache::run_cachesplit(qkv, &mut queries, &mut kv, rows, first);
    }

    #[kernel(unchecked_indexing)]
    #[launch_contract(domain = 1, block = (64, 1, 1), dynamic_shared = 0)]
    pub fn cache_ancestors(mut tree: DisjointSlice<f32>, leaves: u32, first: u32, end: u32) {
        fbt_cache::run_ancestors(&mut tree, leaves, first, end);
    }

    #[kernel(unchecked_indexing)]
    #[launch_contract(domain = 1, block = (64, 1, 1), dynamic_shared = 0)]
    pub fn cache_parents(mut tree: DisjointSlice<f32>, child: u32, parent: u32, first: u32, count: u32) {
        fbt_cache::run_parents(&mut tree, child, parent, first, count);
    }

    #[kernel(unchecked_indexing)]
    #[launch_contract(domain = 1, block = (64, 1, 1), dynamic_shared = 0)]
    pub fn pisaupper(mut pyramid: DisjointSlice<f32>, shape: PisaShape) {
        fbt_cache::run_pisaupper(&mut pyramid, shape);
    }

    #[kernel]
    #[launch_contract(domain = 1, block = (256, 1, 1), dynamic_shared = 0)]
    pub fn convert_check(input: &[f32], mut narrowed: DisjointSlice<u16>, mut widened: DisjointSlice<u32>, count: u32) {
        fbt_pool::run_convcheck(input, &mut narrowed, &mut widened, count);
    }

    #[kernel(unchecked_indexing)]
    #[launch_contract(domain = 1, block = (256, 1, 1), dynamic_shared = 0)]
    pub fn mhc_update(mut streams: DisjointSlice<f32>, branch: &[f32], coefficients: &[f32], rows: u32) {
        fbt_stream::run_update(&mut streams, branch, coefficients, rows);
    }

    #[kernel(unchecked_indexing)]
    #[launch_contract(domain = 2, block = (16, 16, 1), dynamic_shared = 0)]
    pub fn routed_gate(input: &[f32], weights: &[f32], routes: &[RoutedTile], activation: DisjointSlice<f32, thread::Runtime2DIndex>, shape: MoeShape) {
        fbt_moe::run_gate(input, weights, routes, activation, shape);
    }

    #[kernel(unchecked_indexing)]
    #[launch_contract(domain = 2, block = (16, 16, 1), dynamic_shared = 0)]
    pub fn routed_down(activation: &[f32], weights: &[f32], routes: &[RoutedTile], mut output: DisjointSlice<f32, thread::Runtime2DIndex>, shape: MoeShape) {
        fbt_moe::run_down(activation, weights, routes, &mut output, shape);
    }

    #[kernel(unchecked_indexing)]
    #[launch_contract(domain = 1, block = (256, 1, 1), dynamic_shared = 0)]
    pub fn combine_mhc(route_weights: &[f32], packed_rows: &[u32], routed_output: &[f32], shared_output: &[f32], coefficients: &[f32], mut streams: DisjointSlice<f32>, route_shape: RouteShape, moe_shape: MoeShape) {
        fbt_pool::run_combinemhc(route_weights, packed_rows, routed_output, shared_output, coefficients, &mut streams, route_shape, moe_shape);
    }

    #[kernel(unchecked_indexing)]
    #[launch_contract(domain = 1, block = (256, 1, 1), dynamic_shared = 0)]
    pub fn mhc4_encode(readout: &[f32], tokens: &[u32], positions: &[f32], mask: &[f32], mut output: DisjointSlice<f32>, shape: Mhc4EncodeShape) {
        fbt_eval::run_encode(readout, tokens, positions, mask, &mut output, shape);
    }

    #[kernel(unchecked_indexing)]
    #[launch_contract(domain = 1, block = (256, 1, 1), dynamic_shared = 0)]
    pub fn threshold_basis0_fp16(base: &[f32], scales: &[f32], mut candidate: DisjointSlice<f32>, elements: u64, block_len: u32, radius: f32) {
        fbt_pool::run_basis0(base, scales, &mut candidate, elements, block_len, radius);
    }

    #[kernel(unchecked_indexing)]
    #[launch_contract(domain = 1, block = (256, 1, 1), dynamic_shared = 0)]
    pub fn gaussian_fp16(base: &[f32], scales: &[f32], thresholds: &[u32], widths: &[f32], densities: &[f32], mut candidate: DisjointSlice<f32>, mut status: DisjointSlice<u32>, shape: GaussianShape) {
        fbt_pool::run_gaussian(base, scales, thresholds, widths, densities, &mut candidate, &mut status, shape);
    }

    #[kernel(unchecked_indexing)]
    #[launch_contract(domain = 1, block = (256, 1, 1), dynamic_shared = 0)]
    pub fn proposal_status(status: &[u32], mut summary: DisjointSlice<u32>, blocks: u32, tensor: u32) {
        fbt_pool::run_propstatus(status, &mut summary, blocks, tensor);
    }

    #[kernel(unchecked_indexing)]
    #[launch_contract(domain = 1, block = (256, 1, 1), dynamic_shared = 0)]
    pub fn mhc4_pool(base: &[f32], points: &[Mhc4Point], seeds: &[u64], thresholds: &[u32], widths: &[f32], densities: &[f32], mut pool: DisjointSlice<f32>, point_start: u32, point_count: u32, total_points: u32, radius: f32, candidates: u32) {
        fbt_pool::run_pool(base, points, seeds, thresholds, widths, densities, &mut pool, point_start, point_count, total_points, radius, candidates);
    }

    #[kernel(unchecked_indexing)]
    #[launch_contract(domain = 1, block = (64, 1, 1), dynamic_shared = 0)]
    pub fn patch_mean_nll(losses: &[f32], mut patches: DisjointSlice<f32>, tokens: u32) {
        fbt_obj::run_patchnll(losses, &mut patches, tokens);
    }

    #[kernel(unchecked_indexing)]
    #[launch_contract(domain = 1, block = (256, 1, 1), dynamic_shared = 0)]
    pub fn mhc4_objective(losses: &[f32], generated: &[u32], expected: &[u32], mut output: DisjointSlice<fbt_types::ObjectiveStats>, baseline: &[f32], shape: fbt_types::PairShape, mut patches: DisjointSlice<f32>) {
        fbt_obj::run_mhc4obj(losses, generated, expected, &mut output, baseline, shape, &mut patches);
    }

    #[kernel]
    #[launch_bounds(32)]
    pub fn turing_floor() {
        let mut c0 = 0_i32;
        let mut c1 = 0_i32;
        unsafe {
            ptx_asm!(
                "mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 {%0, %1}, {%2}, {%3}, {%0, %1};",
                inout("+r") c0,
                inout("+r") c1,
                in("r") 0_u32,
                in("r") 0_u32,
                options(register_only),
            );
        }
        let _ = (c0, c1);
    }

    #[kernel(unchecked_indexing)]
    #[launch_contract(domain = 1, block = (256, 1, 1), dynamic_shared = 0)]
    pub fn feedback_shift(history: &[u16], mut previous: DisjointSlice<u16>, mut fused: DisjointSlice<u32>, shape: FeedbackShape) {
        fbt_pool::run_fbshift(history, &mut previous, &mut fused, shape);
    }

    #[kernel(unchecked_indexing)]
    #[launch_contract(domain = 1, block = (256, 1, 1), dynamic_shared = 0)]
    pub fn feedback_norm(input: &[u16], mut output: DisjointSlice<u16>, shape: FeedbackShape) {
        fbt_pool::run_fbnorm(input, &mut output, shape);
    }

    #[kernel(unchecked_indexing)]
    #[launch_contract(domain = 1, block = (256, 1, 1), dynamic_shared = 0)]
    pub fn feedback_combine(value: &[u16], logit: &[u16], tokens: &[u16], fused: &[u32], mut output: DisjointSlice<u16>, shape: FeedbackShape) {
        fbt_pool::run_fbcombine(value, logit, tokens, fused, &mut output, shape);
    }

    #[kernel(unchecked_indexing)]
    #[launch_contract(domain = 1, block = (256, 1, 1), dynamic_shared = 0)]
    pub fn combine_routes(route_weights: &[f32], packed_rows: &[u32], routed_output: &[f32], shared_output: &[f32], mut output: DisjointSlice<f32>, route_shape: RouteShape, moe_shape: MoeShape) {
        fbt_pool::run_combroutes(route_weights, packed_rows, routed_output, shared_output, &mut output, route_shape, moe_shape);
    }
}
}
