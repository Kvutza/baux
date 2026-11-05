use super::*;

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub(crate) struct Top3RouteShape {
    pub(crate) rows: u32,
    pub(crate) width: u32,
    pub(crate) experts: u32,
    pub(crate) top_k: u32,
    pub(crate) block_tokens: u32,
    pub(crate) blocks: u32,
}

impl Top3RouteShape {
    pub(crate) const fn new(rows: u32, width: u32) -> Self {
        Self {
            rows,
            width,
            experts: ROUTED_EXPERTS,
            top_k: ROUTED_TOPK,
            block_tokens: ROUTE_BLOCK,
            blocks: rows.div_ceil(ROUTE_BLOCK),
        }
    }

    pub(crate) const fn assignments(self) -> u32 {
        self.rows * self.top_k
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct RouteStats {
    pub(crate) active_experts: usize,
    pub(crate) routed_rows: usize,
    pub(crate) routed_tiles: usize,
    /// Bit `i` identifies routed expert tensor `i + 1`; tensor zero is shared.
    pub(crate) expert_mask: [u64; ROUTE_WORDS],
}

/// Exact-size buffers for three assignments per token. Expert segments are
/// represented by offsets and loads, not padded per-expert capacities.
pub(crate) struct FineGrainedMoeBuffers {
    pub(crate) scores: Buffer,
    pub(crate) route_experts: Buffer,
    pub(crate) route_weights: Buffer,
    pub(super) route_margin: Buffer,
    pub(super) block_histogram: Buffer,
    pub(super) block_offsets: Buffer,
    pub(crate) expert_offsets: Buffer,
    pub(crate) expert_loads: Buffer,
    pub(crate) routed_tiles: Buffer,
    pub(crate) routed_dispatch: Buffer,
    pub(super) route_mask: Buffer,
    pub(crate) packed_rows: Buffer,
    pub(crate) packed_input: Buffer,
    pub(super) packed_tokens: Buffer,
    pub(crate) packed_experts: Buffer,
    pub(super) packed_weights: Buffer,
    pub(crate) routed_activation: Buffer,
    pub(crate) routed_output: Buffer,
    pub(crate) shared_activation: Buffer,
    pub(crate) shared_output: Buffer,
}

impl FineGrainedMoeBuffers {
    pub(crate) fn new(runtime: &Runtime) -> Self {
        Self::with_shape(runtime, Top3RouteShape::new(ROWS, WIDTH))
    }

    pub(crate) fn for_rows(runtime: &Runtime, rows: u32) -> Self {
        Self::with_shape(runtime, Top3RouteShape::new(rows, WIDTH))
    }

    pub(super) fn with_shape(runtime: &Runtime, shape: Top3RouteShape) -> Self {
        let rows = shape.rows as usize;
        let width = shape.width as usize;
        let assignments = shape.assignments() as usize;
        let experts = shape.experts as usize;
        let blocks = shape.blocks as usize;
        let max_tiles = shape.assignments().div_ceil(TILE_ROWS) as usize + experts - 1;
        Self {
            scores: runtime.buffer::<u16>(rows * experts),
            route_experts: runtime.buffer::<u32>(assignments),
            route_weights: runtime.buffer::<u16>(assignments),
            route_margin: runtime.buffer::<f32>(rows),
            block_histogram: runtime.buffer::<u32>(blocks * experts),
            block_offsets: runtime.buffer::<u32>(blocks * experts),
            expert_offsets: runtime.buffer::<u32>(experts),
            expert_loads: runtime.buffer::<u32>(experts),
            routed_tiles: runtime.buffer::<u32>(max_tiles * TILE_WORDS),
            routed_dispatch: runtime.buffer::<u32>(DISPATCH_WORDS),
            route_mask: runtime.buffer_with(&vec![0u32; MASK_WORDS]),
            packed_rows: runtime.buffer::<u32>(assignments),
            packed_input: runtime.buffer::<u16>(assignments * width),
            packed_tokens: runtime.buffer::<u32>(assignments),
            packed_experts: runtime.buffer::<u32>(assignments),
            packed_weights: runtime.buffer::<u16>(assignments),
            routed_activation: runtime
                .buffer_with(&vec![0u16; assignments * ACTIVATION_STRIDE as usize]),
            routed_output: runtime.buffer::<u16>(assignments * width),
            shared_activation: runtime.buffer_with(&vec![0u16; rows * ACTIVATION_STRIDE as usize]),
            shared_output: runtime.buffer::<u16>(rows * width),
        }
    }

    pub(crate) fn validate_metadata(&self) -> Result<(), String> {
        let shape = Top3RouteShape::new(ROWS, WIDTH);
        let assignments = shape.assignments() as usize;
        let read = |buffer: &BufferRef, elements: usize| unsafe {
            std::slice::from_raw_parts(buffer.contents().cast::<u32>(), elements)
        };
        let experts = read(&self.route_experts, assignments);
        let packed_rows = read(&self.packed_rows, assignments);
        let tokens = read(&self.packed_tokens, assignments);
        let packed_experts = read(&self.packed_experts, assignments);
        let mut seen = vec![false; assignments];
        for assignment in 0..assignments {
            let expert = experts[assignment] as usize;
            let packed = packed_rows[assignment] as usize;
            if packed >= assignments || seen[packed] {
                return Err(format!(
                    "fused route pack produced invalid or duplicate row {packed} for assignment \
                     {assignment}"
                ));
            }
            seen[packed] = true;
            if tokens[packed] as usize != assignment / shape.top_k as usize
                || packed_experts[packed] as usize != expert
            {
                return Err(format!(
                    "fused route pack metadata mismatch at assignment {assignment}, packed row \
                     {packed}"
                ));
            }
        }
        Ok(())
    }

    /// Counts describe the final site; the mask is the union over every MoE
    /// site in the completed forward pass.
    pub(crate) fn route_stats(&self) -> RouteStats {
        let loads = unsafe {
            std::slice::from_raw_parts(
                self.expert_loads.contents().cast::<u32>(),
                ROUTED_EXPERTS as usize,
            )
        };
        let dispatch = unsafe {
            std::slice::from_raw_parts(
                self.routed_dispatch.contents().cast::<u32>(),
                DISPATCH_WORDS,
            )
        };
        let mask = unsafe {
            std::slice::from_raw_parts(self.route_mask.contents().cast::<u32>(), MASK_WORDS)
        };
        let mut expert_mask = [0; ROUTE_WORDS];
        for (word, pair) in expert_mask.iter_mut().zip(mask.chunks(2)) {
            *word = u64::from(pair[0]) | (u64::from(pair.get(1).copied().unwrap_or(0)) << 32);
        }
        RouteStats {
            active_experts: expert_mask
                .iter()
                .map(|word| word.count_ones() as usize)
                .sum(),
            routed_rows: loads.iter().map(|&load| load as usize).sum(),
            routed_tiles: dispatch[1] as usize,
            expert_mask,
        }
    }

    pub(crate) fn reset_routes(&self, command: &CommandBufferRef) {
        let blit = command.new_blit_command_encoder();
        blit.fill_buffer(
            &self.route_mask,
            metal::NSRange::new(0, self.route_mask.length()),
            0,
        );
        blit.end_encoding();
    }
}

pub(crate) fn tensor_encode(
    encoder: &ComputeCommandEncoderRef,
    pipeline: &ComputePipelineState,
    buffers: &[(&BufferRef, u64)],
    parameters: *const std::ffi::c_void,
    parameter_bytes: u64,
    groups: MTLSize,
) -> Result<(), String> {
    let simd_width = pipeline.thread_execution_width();
    let threads = simd_width * 4;
    if simd_width != 32 || pipeline.max_total_threads_per_threadgroup() < threads {
        return Err(format!(
            "MoE TensorOps kernel requires four 32-lane SIMD groups; pipeline reports SIMD width \
             {simd_width} and max threads {}",
            pipeline.max_total_threads_per_threadgroup()
        ));
    }
    encoder.set_compute_pipeline_state(pipeline);
    for (index, (buffer, offset)) in buffers.iter().enumerate() {
        encoder.set_buffer(index as u64, Some(buffer), *offset);
    }
    encoder.set_bytes(buffers.len() as u64, parameter_bytes, parameters);
    encoder.dispatch_thread_groups(
        groups,
        MTLSize {
            width: threads,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn tensor_indirect(
    encoder: &ComputeCommandEncoderRef,
    pipeline: &ComputePipelineState,
    buffers: &[(&BufferRef, u64)],
    parameters: *const std::ffi::c_void,
    parameter_bytes: u64,
    dispatch: &BufferRef,
    dispatch_offset: u64,
    simdgroups: u64,
) -> Result<(), String> {
    let simd_width = pipeline.thread_execution_width();
    let threads = simd_width * simdgroups;
    if simd_width != 32 || pipeline.max_total_threads_per_threadgroup() < threads {
        return Err(format!(
            "MoE TensorOps kernel requires {simdgroups} 32-lane SIMD groups; pipeline reports \
             SIMD width {simd_width} and max threads {}",
            pipeline.max_total_threads_per_threadgroup()
        ));
    }
    encoder.set_compute_pipeline_state(pipeline);
    for (index, (buffer, offset)) in buffers.iter().enumerate() {
        encoder.set_buffer(index as u64, Some(buffer), *offset);
    }
    encoder.set_bytes(buffers.len() as u64, parameter_bytes, parameters);
    encoder.dispatch_thread_groups_indirect(
        dispatch,
        dispatch_offset,
        MTLSize {
            width: threads,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}
