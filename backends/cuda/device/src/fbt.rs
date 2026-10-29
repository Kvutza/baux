use cuda_device::{DisjointSlice, cuda_module, kernel, launch_contract, thread};

pub use super::fbt_math::{GaussianShape, NeighborPoint};
pub use super::fbt_mhc::mhc_model;
pub use super::fbt_types::*;
use super::{fbt_cache, fbt_gemm, fbt_math, fbt_obj, fbt_pack, fbt_pisa, fbt_stream};

#[cuda_module]
pub mod fbt_model {
    use super::*;

    #[kernel(unchecked_indexing)]
    #[launch_contract(domain = 1, block = (256, 1, 1), dynamic_shared = 0)]
    pub fn rope(mut qkv: DisjointSlice<u16>, table: &[f32], rows: u32, context: u32, first: u32) {
        fbt_stream::run_rope(&mut qkv, table, rows, context, first);
    }

    #[kernel(unchecked_indexing)]
    #[launch_contract(domain = 1, block = (256, 1, 1), dynamic_shared = 0)]
    pub fn mhc_replicate(input: &[u16], mut streams: DisjointSlice<u16>, rows: u32) {
        fbt_stream::run_replicate(input, &mut streams, rows);
    }

    #[kernel(unchecked_indexing)]
    #[launch_contract(domain = 1, block = (256, 1, 1), dynamic_shared = 0)]
    pub fn mhc_predict(
        streams: &[u16],
        predictor: &[u16],
        bias: &[u16],
        control: &[u16],
        mut coefficients: DisjointSlice<f32>,
        rows: u32,
        parallel: u32,
    ) {
        fbt_stream::run_predict(
            streams,
            predictor,
            bias,
            control,
            &mut coefficients,
            rows,
            parallel,
        );
    }

    #[kernel(unchecked_indexing)]
    #[launch_contract(domain = 1, block = (256, 1, 1), dynamic_shared = 0)]
    pub fn mhc_mix(
        streams: &[u16],
        coefficients: &[f32],
        weight: &[u16],
        mut output: DisjointSlice<u16>,
        rows: u32,
        mean: u32,
    ) {
        fbt_stream::run_mix(streams, coefficients, weight, &mut output, rows, mean);
    }

    #[kernel(unchecked_indexing)]
    #[launch_contract(domain = 1, block = (256, 1, 1), dynamic_shared = 0)]
    pub fn route_counts(
        experts: &[u32],
        mut counts: DisjointSlice<u32>,
        mut prefixes: DisjointSlice<u32>,
        assignments: u32,
    ) {
        fbt_pisa::run_counts(experts, &mut counts, &mut prefixes, assignments);
    }

    #[kernel(unchecked_indexing)]
    #[launch_contract(domain = 1, block = (32, 1, 1), dynamic_shared = 0)]
    pub fn route_layout(
        counts: &[u32],
        mut offsets: DisjointSlice<u32>,
        mut tiles: DisjointSlice<RoutedTile>,
        experts: u32,
        tile_capacity: u32,
    ) {
        fbt_obj::run_layout(counts, &mut offsets, &mut tiles, experts, tile_capacity);
    }

    #[kernel(unchecked_indexing)]
    #[launch_contract(domain = 1, block = (256, 1, 1), dynamic_shared = 0)]
    pub fn route_pack(
        input: &[u16],
        experts: &[u32],
        offsets: &[u32],
        prefixes: &[u32],
        mut mapping: DisjointSlice<u32>,
        mut packed: DisjointSlice<u16>,
        assignments: u32,
    ) {
        fbt_obj::run_pack(
            input,
            experts,
            offsets,
            prefixes,
            &mut mapping,
            &mut packed,
            assignments,
        );
    }

    #[kernel(unchecked_indexing)]
    #[launch_contract(domain = 1, block = (256, 1, 1), dynamic_shared = 0)]
    pub fn verify_prefix(
        predicted: &[u32],
        mut tokens: DisjointSlice<u32>,
        mut frontier: DisjointSlice<u32>,
        rows: u32,
    ) {
        fbt_pack::run_verify(predicted, &mut tokens, &mut frontier, rows);
    }

    #[kernel(unchecked_indexing)]
    #[launch_contract(domain = 1, block = (256, 1, 1), dynamic_shared = 0)]
    pub fn sample(
        logits: &[u16],
        mut tokens: DisjointSlice<u32>,
        rows: u32,
        temperature: f32,
        seed: u64,
        first_row: u32,
    ) {
        fbt_pack::run_sample(logits, &mut tokens, rows, temperature, seed, first_row);
    }

    #[kernel(unchecked_indexing)]
    #[launch_contract(domain = 1, block = (256, 1, 1), dynamic_shared = 0)]
    pub fn expand_patches(
        macros: &[u32],
        readout: &[u16],
        mut tokens: DisjointSlice<u32>,
        patches: u32,
        first_patch: u32,
    ) {
        fbt_pack::run_expand(macros, readout, &mut tokens, patches, first_patch);
    }

    #[kernel(unchecked_indexing)]
    #[launch_contract(domain = 1, block = (256, 1, 1), dynamic_shared = 0)]
    pub fn score(
        logits: &[u16],
        targets: &[u32],
        mut losses: DisjointSlice<f32>,
        rows: u32,
        target_offset: u32,
        loss_offset: u32,
    ) {
        fbt_pack::run_score(
            logits,
            targets,
            &mut losses,
            rows,
            target_offset,
            loss_offset,
        );
    }

    #[kernel(unchecked_indexing)]
    #[launch_contract(domain = 1, block = (256, 1, 1), dynamic_shared = 0)]
    pub fn objective(
        losses: &[f32],
        generated: &[u32],
        targets: &[u32],
        mut output: DisjointSlice<ScoreStats>,
        shape: ObjectiveShape,
    ) {
        fbt_pack::run_objective(losses, generated, targets, &mut output, shape);
    }

    #[kernel(unchecked_indexing)]
    #[launch_contract(domain = 1, block = (256, 1, 1), dynamic_shared = 0)]
    pub fn embed_packed(
        weights: &[u16],
        tokens: &[u32],
        mut output: DisjointSlice<u16>,
        shape: FbtShape,
    ) {
        fbt_math::run_embed(weights, tokens, &mut output, shape);
    }

    #[kernel(unchecked_indexing)]
    #[launch_contract(domain = 1, block = (256, 1, 1), dynamic_shared = 0)]
    pub fn residual(input: &[u16], branch: &[u16], mut output: DisjointSlice<u16>, elements: u32) {
        fbt_math::run_residual(input, branch, &mut output, elements);
    }

    #[kernel(unchecked_indexing)]
    #[launch_contract(domain = 1, block = (256, 1, 1), dynamic_shared = 0)]
    pub fn rms(input: &[u16], weight: &[u16], mut output: DisjointSlice<u16>, shape: FbtShape) {
        fbt_math::run_rms(input, weight, &mut output, shape);
    }

    #[kernel(unchecked_indexing)]
    #[launch_contract(domain = 2, block = (16, 16, 1), dynamic_shared = 0)]
    pub fn matmul(
        input: &[u16],
        weight: &[u16],
        mut output: DisjointSlice<u16, thread::Runtime2DIndex>,
        shape: MatmulShape,
    ) {
        fbt_gemm::run_matmul(input, weight, &mut output, shape);
    }

    #[kernel(unchecked_indexing)]
    #[launch_contract(domain = 2, block = (256, 1, 1), dynamic_shared = 0)]
    pub fn matmul_turing(
        input: &[u16],
        weight: &[u16],
        mut output: DisjointSlice<u16, thread::Runtime2DIndex>,
        shape: MatmulShape,
    ) {
        fbt_gemm::turing::<256>(
            input,
            weight,
            output.as_mut_ptr(),
            shape,
            thread::blockIdx_y() * 64,
        );
    }

    #[kernel(unchecked_indexing)]
    #[launch_contract(domain = 2, block = (256, 1, 1), dynamic_shared = 0)]
    pub fn routed_project(
        input: &[u16],
        weights: &[u16],
        routes: &[RoutedTile],
        mut output: DisjointSlice<u16, thread::Runtime2DIndex>,
        mode: u32,
    ) {
        let tile = thread::blockIdx_y();
        fbt_gemm::routed::<256>(input, weights, routes, output.as_mut_ptr(), mode, tile, 0);
    }

    #[kernel(unchecked_indexing)]
    #[launch_contract(domain = 2, block = (64, 1, 1), dynamic_shared = 0)]
    pub fn routed_compact(
        input: &[u16],
        weights: &[u16],
        routes: &[RoutedTile],
        mut output: DisjointSlice<u16, thread::Runtime2DIndex>,
        mode: u32,
    ) {
        let block = thread::blockIdx_y();
        fbt_gemm::routed::<64>(
            input,
            weights,
            routes,
            output.as_mut_ptr(),
            mode,
            block / 4,
            (block % 4) * 16,
        );
    }

    #[kernel(unchecked_indexing)]
    #[launch_contract(domain = 1, block = (256, 1, 1), dynamic_shared = 0)]
    pub fn routed_activate(projection: &[u16], mut output: DisjointSlice<u16>, rows: u32) {
        fbt_math::run_activate(projection, &mut output, rows);
    }

    #[kernel(unchecked_indexing)]
    #[launch_contract(domain = 2, block = (64, 1, 1), dynamic_shared = 0)]
    pub fn pisaleaves(
        qkv: &[u16],
        mut pyramid: DisjointSlice<u16, thread::Runtime2DIndex>,
        shape: PisaShape,
    ) {
        fbt_cache::run_pisaleaves(qkv, &mut pyramid, shape);
    }

    #[kernel(unchecked_indexing)]
    #[launch_contract(domain = 1, block = (64, 1, 1), dynamic_shared = 0)]
    pub fn cache_leaves(kv: &[u16], mut tree: DisjointSlice<u16>, first: u32, count: u32) {
        fbt_cache::run_cacheleaves(kv, &mut tree, first, count);
    }

    #[kernel(unchecked_indexing)]
    #[launch_contract(domain = 1, block = (256, 1, 1), dynamic_shared = 0)]
    pub fn cache_split(
        qkv: &[u16],
        mut queries: DisjointSlice<u16>,
        mut kv: DisjointSlice<u16>,
        rows: u32,
        first: u32,
    ) {
        fbt_cache::run_cachesplit(qkv, &mut queries, &mut kv, rows, first);
    }

    #[kernel(unchecked_indexing)]
    #[launch_contract(domain = 1, block = (64, 1, 1), dynamic_shared = 0)]
    pub fn cache_ancestors(mut tree: DisjointSlice<u16>, leaves: u32, first: u32, end: u32) {
        fbt_cache::run_ancestors(&mut tree, leaves, first, end);
    }

    #[kernel(unchecked_indexing)]
    #[launch_contract(domain = 1, block = (64, 1, 1), dynamic_shared = 0)]
    pub fn cache_parents(
        mut tree: DisjointSlice<u16>,
        child: u32,
        parent: u32,
        first: u32,
        count: u32,
    ) {
        fbt_cache::run_parents(&mut tree, child, parent, first, count);
    }

    #[kernel(unchecked_indexing)]
    #[launch_contract(domain = 1, block = (64, 1, 1), dynamic_shared = 0)]
    pub fn pisaupper(mut pyramid: DisjointSlice<u16>, shape: PisaShape) {
        fbt_cache::run_pisaupper(&mut pyramid, shape);
    }

    #[kernel(unchecked_indexing)]
    #[launch_contract(domain = 1, block = (64, 1, 1), dynamic_shared = 0)]
    pub fn pisa_select(
        qkv: &[u16],
        pyramid: &[u16],
        mut blocks: DisjointSlice<u32>,
        shape: PisaShape,
    ) {
        fbt_cache::run_select(qkv, pyramid, &mut blocks, shape);
    }

    #[kernel(unchecked_indexing)]
    #[launch_contract(domain = 1, block = (64, 1, 1), dynamic_shared = 0)]
    pub fn pisa_shadow_refine(
        qkv: &[u16],
        pyramid: &[u16],
        incumbent: &[u32],
        mut blocks: DisjointSlice<u32>,
        visit_offset: u32,
        shape: PisaShape,
    ) {
        fbt_pisa::run_refine(qkv, pyramid, incumbent, &mut blocks, visit_offset, shape);
    }

    #[kernel(unchecked_indexing)]
    #[launch_contract(domain = 1, block = (256, 1, 1), dynamic_shared = 0)]
    pub fn routetopk(
        scores_input: &[u16],
        experts_output: DisjointSlice<u32>,
        weights_output: DisjointSlice<u16>,
        margin_output: DisjointSlice<f32>,
        shape: RouteShape,
    ) {
        fbt_pisa::run_routetopk(
            scores_input,
            experts_output,
            weights_output,
            margin_output,
            shape,
        );
    }

    #[kernel(unchecked_indexing)]
    #[launch_contract(domain = 1, block = (256, 1, 1), dynamic_shared = 0)]
    pub fn pisa_attention(
        qkv: &[u16],
        kv: &[u16],
        blocks: &[u32],
        output: DisjointSlice<u16>,
        shape: PisaShape,
    ) {
        fbt_pisa::run_attention(qkv, kv, blocks, output, shape);
    }

    #[kernel(unchecked_indexing)]
    #[launch_contract(domain = 1, block = (256, 1, 1), dynamic_shared = 0)]
    pub fn pisa_multihead(
        qkv: &[u16],
        kv: &[u16],
        blocks: &[u32],
        output: DisjointSlice<u16>,
        shape: PisaShape,
    ) {
        fbt_pisa::run_multihead(qkv, kv, blocks, output, shape);
    }
}
