use cuda_device::{
    DisjointSlice, cuda_module, kernel, launch_bounds, launch_contract, ptx_asm, thread,
};

use super::fbt_types::{
    FeedbackShape, GaussianShape, Mhc4EncodeShape, Mhc4Point, MoeShape, RouteShape, RoutedTile,
    ShardStats, WaveShape,
};
use super::{fbt_gemm, fbt_obj, fbt_pool, fbt_stats, fbt_stream, fbt_types, fbt_wave};

#[cuda_module]
pub mod mhc_model {
    use super::*;

    #[kernel]
    #[launch_contract(domain = 1, block = (256, 1, 1), dynamic_shared = 0)]
    pub fn convert_check(
        input: &[f32],
        mut narrowed: DisjointSlice<u16>,
        mut widened: DisjointSlice<u32>,
        count: u32,
    ) {
        fbt_pool::run_convcheck(input, &mut narrowed, &mut widened, count);
    }

    #[kernel(unchecked_indexing)]
    #[launch_contract(domain = 1, block = (256, 1, 1), dynamic_shared = 0)]
    pub fn mhc_update(
        mut streams: DisjointSlice<u16>,
        branch: &[u16],
        coefficients: &[f32],
        rows: u32,
    ) {
        fbt_stream::run_update(&mut streams, branch, coefficients, rows);
    }

    #[kernel(unchecked_indexing)]
    #[launch_contract(domain = 2, block = (16, 16, 1), dynamic_shared = 0)]
    pub fn routed_gate(
        input: &[u16],
        weights: &[u16],
        routes: &[RoutedTile],
        activation: DisjointSlice<u16, thread::Runtime2DIndex>,
        shape: MoeShape,
    ) {
        fbt_gemm::run_gate(input, weights, routes, activation, shape);
    }

    #[kernel(unchecked_indexing)]
    #[launch_contract(domain = 2, block = (16, 16, 1), dynamic_shared = 0)]
    pub fn routed_down(
        activation: &[u16],
        weights: &[u16],
        routes: &[RoutedTile],
        mut output: DisjointSlice<u16, thread::Runtime2DIndex>,
        shape: MoeShape,
    ) {
        fbt_gemm::run_down(activation, weights, routes, &mut output, shape);
    }

    #[kernel(unchecked_indexing)]
    #[launch_contract(domain = 1, block = (256, 1, 1), dynamic_shared = 0)]
    pub fn combine_mhc(
        route_weights: &[u16],
        packed_rows: &[u32],
        routed_output: &[u16],
        shared_output: &[u16],
        coefficients: &[f32],
        mut streams: DisjointSlice<u16>,
        route_shape: RouteShape,
        moe_shape: MoeShape,
    ) {
        fbt_pool::run_combinemhc(
            route_weights,
            packed_rows,
            routed_output,
            shared_output,
            coefficients,
            &mut streams,
            route_shape,
            moe_shape,
        );
    }

    #[kernel(unchecked_indexing)]
    #[launch_contract(domain = 1, block = (256, 1, 1), dynamic_shared = 0)]
    pub fn mhc4_encode(
        readout: &[u16],
        tokens: &[u32],
        positions: &[u16],
        mask: &[u16],
        mut output: DisjointSlice<u16>,
        shape: Mhc4EncodeShape,
    ) {
        fbt_wave::run_encode(readout, tokens, positions, mask, &mut output, shape);
    }

    #[kernel(unchecked_indexing)]
    #[launch_contract(domain = 1, block = (256, 1, 1), dynamic_shared = 0)]
    pub fn mhc4_centroids(readout: &[u16], mut centroids: DisjointSlice<u16>, shape: WaveShape) {
        fbt_wave::run_centroids(readout, &mut centroids, shape);
    }

    #[kernel(unchecked_indexing)]
    #[launch_contract(domain = 1, block = (64, 1, 1), dynamic_shared = 0)]
    pub fn mhc4_expand(
        hidden: &[u16],
        qkv: &[u16],
        positions: &[u16],
        readout: &[u16],
        centroids: &[u16],
        seeds: &[u64],
        expected: &[u32],
        mut output: DisjointSlice<u32>,
        mut losses: DisjointSlice<f32>,
        shape: WaveShape,
    ) {
        fbt_wave::run_mhc4expand(
            hidden,
            qkv,
            positions,
            readout,
            centroids,
            seeds,
            expected,
            &mut output,
            &mut losses,
            shape,
        );
    }

    #[kernel(unchecked_indexing)]
    #[launch_contract(domain = 1, block = (256, 1, 1), dynamic_shared = 0)]
    pub fn mhc4_init(
        hidden: &[u16],
        qkv: &[u16],
        positions: &[u16],
        mut base: DisjointSlice<f32>,
        mut prefix: DisjointSlice<f32>,
        mut query: DisjointSlice<u16>,
        shape: WaveShape,
    ) {
        fbt_wave::run_init(
            hidden,
            qkv,
            positions,
            &mut base,
            &mut prefix,
            &mut query,
            shape,
        );
    }

    #[kernel(unchecked_indexing)]
    #[launch_contract(domain = 1, block = (256, 1, 1), dynamic_shared = 0)]
    pub fn mhc4_project(
        logits: &[f32],
        seeds: &[u64],
        expected: &[u32],
        mut partials: DisjointSlice<ShardStats>,
        shape: WaveShape,
    ) {
        fbt_stats::run_mhc4project(logits, seeds, expected, &mut partials, shape);
    }

    #[kernel(unchecked_indexing)]
    #[launch_contract(domain = 1, block = (64, 1, 1), dynamic_shared = 0)]
    pub fn mhc4_reduce(
        partials: &[ShardStats],
        base: &[f32],
        mut prefix: DisjointSlice<f32>,
        mut query: DisjointSlice<u16>,
        positions: &[u16],
        readout: &[u16],
        seeds: &[u64],
        mut output: DisjointSlice<u32>,
        mut losses: DisjointSlice<f32>,
        shape: WaveShape,
    ) {
        fbt_stats::run_mhc4reduce(
            partials,
            base,
            &mut prefix,
            &mut query,
            positions,
            readout,
            seeds,
            &mut output,
            &mut losses,
            shape,
        );
    }

    #[kernel(unchecked_indexing)]
    #[launch_contract(domain = 1, block = (256, 1, 1), dynamic_shared = 0)]
    pub fn threshold_basis0_fp16(
        base: &[u16],
        scales: &[f32],
        mut candidate: DisjointSlice<u16>,
        elements: u64,
        block_len: u32,
        radius: f32,
    ) {
        fbt_pool::run_basis0(base, scales, &mut candidate, elements, block_len, radius);
    }

    #[kernel(unchecked_indexing)]
    #[launch_contract(domain = 1, block = (256, 1, 1), dynamic_shared = 0)]
    pub fn gaussian_fp16(
        base: &[u16],
        scales: &[f32],
        thresholds: &[u32],
        widths: &[f32],
        densities: &[f32],
        mut candidate: DisjointSlice<u16>,
        mut status: DisjointSlice<u32>,
        shape: GaussianShape,
    ) {
        fbt_pool::run_gaussian(
            base,
            scales,
            thresholds,
            widths,
            densities,
            &mut candidate,
            &mut status,
            shape,
        );
    }

    #[kernel(unchecked_indexing)]
    #[launch_contract(domain = 1, block = (256, 1, 1), dynamic_shared = 0)]
    pub fn proposal_status(
        status: &[u32],
        mut summary: DisjointSlice<u32>,
        blocks: u32,
        tensor: u32,
    ) {
        fbt_pool::run_propstatus(status, &mut summary, blocks, tensor);
    }

    #[kernel(unchecked_indexing)]
    #[launch_contract(domain = 1, block = (256, 1, 1), dynamic_shared = 0)]
    pub fn mhc4_pool(
        base: &[u16],
        points: &[Mhc4Point],
        seeds: &[u64],
        thresholds: &[u32],
        widths: &[f32],
        densities: &[f32],
        mut pool: DisjointSlice<u16>,
        point_start: u32,
        point_count: u32,
        total_points: u32,
        radius: f32,
        candidates: u32,
    ) {
        fbt_pool::run_pool(
            base,
            points,
            seeds,
            thresholds,
            widths,
            densities,
            &mut pool,
            point_start,
            point_count,
            total_points,
            radius,
            candidates,
        );
    }

    #[kernel(unchecked_indexing)]
    #[launch_contract(domain = 1, block = (256, 1, 1), dynamic_shared = 0)]
    pub fn mhc4_capture(
        base: &[u16],
        points: &[Mhc4Point],
        mut history: DisjointSlice<u16>,
        point_start: u32,
        point_count: u32,
        total_points: u32,
        row: u32,
    ) {
        fbt_obj::run_capture(
            base,
            points,
            &mut history,
            point_start,
            point_count,
            total_points,
            row,
        );
    }

    #[kernel(unchecked_indexing)]
    #[launch_contract(domain = 1, block = (256, 1, 1), dynamic_shared = 0)]
    pub fn mhc4_distances(
        pool: &[u16],
        history: &[u16],
        points: &[Mhc4Point],
        mut distances: DisjointSlice<f32>,
        total_points: u32,
        history_count: u32,
    ) {
        fbt_obj::run_distances(
            pool,
            history,
            points,
            &mut distances,
            total_points,
            history_count,
        );
    }

    #[kernel(unchecked_indexing)]
    #[launch_contract(domain = 1, block = (64, 1, 1), dynamic_shared = 0)]
    pub fn patch_mean_nll(losses: &[f32], mut patches: DisjointSlice<f32>, tokens: u32) {
        fbt_obj::run_patchnll(losses, &mut patches, tokens);
    }

    #[kernel(unchecked_indexing)]
    #[launch_contract(domain = 1, block = (256, 1, 1), dynamic_shared = 0)]
    pub fn mhc4_objective(
        losses: &[f32],
        generated: &[u32],
        expected: &[u32],
        mut output: DisjointSlice<fbt_types::ObjectiveStats>,
        baseline: &[f32],
        shape: fbt_types::PairShape,
        mut patches: DisjointSlice<f32>,
    ) {
        fbt_obj::run_mhc4obj(
            losses,
            generated,
            expected,
            &mut output,
            baseline,
            shape,
            &mut patches,
        );
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
    pub fn feedback_shift(
        history: &[u16],
        mut previous: DisjointSlice<u16>,
        mut fused: DisjointSlice<u32>,
        shape: FeedbackShape,
    ) {
        fbt_pool::run_fbshift(history, &mut previous, &mut fused, shape);
    }

    #[kernel(unchecked_indexing)]
    #[launch_contract(domain = 1, block = (256, 1, 1), dynamic_shared = 0)]
    pub fn feedback_norm(input: &[u16], mut output: DisjointSlice<u16>, shape: FeedbackShape) {
        fbt_pool::run_fbnorm(input, &mut output, shape);
    }

    #[kernel(unchecked_indexing)]
    #[launch_contract(domain = 1, block = (256, 1, 1), dynamic_shared = 0)]
    pub fn feedback_combine(
        value: &[u16],
        logit: &[u16],
        tokens: &[u16],
        fused: &[u32],
        mut output: DisjointSlice<u16>,
        shape: FeedbackShape,
    ) {
        fbt_pool::run_fbcombine(value, logit, tokens, fused, &mut output, shape);
    }

    #[kernel(unchecked_indexing)]
    #[launch_contract(domain = 2, block = (256, 1, 1), dynamic_shared = 0)]
    pub fn mhc4_gemm(
        input: &[u16],
        weight: &[u16],
        mut output: DisjointSlice<f32, thread::Runtime2DIndex>,
        shape: WaveShape,
    ) {
        fbt_gemm::turing_f32::<256>(
            input,
            weight,
            output.as_mut_ptr(),
            shape.blocks,
            thread::blockIdx_y() * 64,
        );
    }

    #[kernel(unchecked_indexing)]
    #[launch_contract(domain = 2, block = (256, 1, 1), dynamic_shared = 0)]
    pub fn mhc4_stats(
        input: &[u16],
        weight: &[u16],
        seeds: &[u64],
        expected: &[u32],
        mut partials: DisjointSlice<ShardStats, thread::Runtime2DIndex>,
        shape: WaveShape,
    ) {
        fbt_stats::run_mhc4stats(input, weight, seeds, expected, &mut partials, shape);
    }

    #[kernel(unchecked_indexing)]
    #[launch_contract(domain = 1, block = (256, 1, 1), dynamic_shared = 0)]
    pub fn combine_routes(
        route_weights: &[u16],
        packed_rows: &[u32],
        routed_output: &[u16],
        shared_output: &[u16],
        mut output: DisjointSlice<u16>,
        route_shape: RouteShape,
        moe_shape: MoeShape,
    ) {
        fbt_pool::run_combroutes(
            route_weights,
            packed_rows,
            routed_output,
            shared_output,
            &mut output,
            route_shape,
            moe_shape,
        );
    }
}
