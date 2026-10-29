use super::routing::*;
use super::*;

const SOURCE: &str = include_str!("../fbt_routing_grad.metal");

pub(crate) struct FineGrainedMoeGrad {
    combine: ComputePipelineState,
    routed_down_input: ComputePipelineState,
    shared_down_input: ComputePipelineState,
    routed_down_weight: ComputePipelineState,
    shared_down_weight: ComputePipelineState,
    routed_projection: ComputePipelineState,
    shared_projection: ComputePipelineState,
    swiglu: ComputePipelineState,
    routed_gate_input: ComputePipelineState,
    shared_gate_input: ComputePipelineState,
    routed_gate_weight: ComputePipelineState,
    shared_gate_weight: ComputePipelineState,
    router_score: ComputePipelineState,
    router_project: ComputePipelineState,
}

pub(crate) struct MoeGradBuffers {
    rows: u32,
    routed_output: Buffer,
    shared_output: Buffer,
    route: Buffer,
    router_score: Buffer,
    routed_activation: Buffer,
    shared_activation: Buffer,
    routed_projection: Buffer,
    shared_projection: Buffer,
    routed_projection_grad: Buffer,
    shared_projection_grad: Buffer,
    packed_input: Buffer,
}

#[derive(Clone, Copy)]
pub(crate) struct MoeGrad<'a> {
    pub input: (&'a BufferRef, u64),
    pub weights: &'a BufferRef,
    pub router_offset: u64,
    pub gate_offset: u64,
    pub down_offset: u64,
    pub branch_grad: (&'a BufferRef, u64),
    pub input_grad: (&'a BufferRef, u64),
    pub router_grad: (&'a BufferRef, u64),
    pub gate_grad: (&'a BufferRef, u64),
    pub down_grad: (&'a BufferRef, u64),
    pub rows: u32,
}

fn shape(encoder: &ComputeCommandEncoderRef, slot: u64, value: &Top3RouteShape) {
    encoder.set_bytes(
        slot,
        size_of::<Top3RouteShape>() as u64,
        std::ptr::from_ref(value).cast(),
    );
}

fn threads(encoder: &ComputeCommandEncoderRef, pipeline: &ComputePipelineState, count: u64) {
    encoder.set_compute_pipeline_state(pipeline);
    encoder.dispatch_threads(
        thread_group(count),
        thread_group(pipeline.max_total_threads_per_threadgroup().min(256)),
    );
}

impl MoeGradBuffers {
    pub(crate) fn new(runtime: &Runtime, rows: u32) -> Self {
        let assignments = rows as usize * ROUTED_TOPK as usize;
        Self {
            rows,
            routed_output: runtime.buffer::<f32>(assignments * WIDTH as usize),
            shared_output: runtime.buffer::<f32>(rows as usize * WIDTH as usize),
            route: runtime.buffer::<f32>(assignments),
            router_score: runtime.buffer::<f32>(assignments),
            routed_activation: runtime.buffer::<f32>(assignments * ACTIVATION_STRIDE as usize),
            shared_activation: runtime.buffer::<f32>(rows as usize * ACTIVATION_STRIDE as usize),
            routed_projection: runtime.buffer::<u16>(assignments * MOE_GATEUP as usize),
            shared_projection: runtime.buffer::<u16>(rows as usize * MOE_GATEUP as usize),
            routed_projection_grad: runtime.buffer::<f32>(assignments * MOE_GATEUP as usize),
            shared_projection_grad: runtime.buffer::<f32>(rows as usize * MOE_GATEUP as usize),
            packed_input: runtime.buffer::<f32>(assignments * WIDTH as usize),
        }
    }

    fn clear(&self, command: &CommandBufferRef) {
        let encoder = command.new_blit_command_encoder();
        for buffer in [
            &self.routed_output,
            &self.shared_output,
            &self.route,
            &self.router_score,
            &self.routed_activation,
            &self.shared_activation,
            &self.routed_projection_grad,
            &self.shared_projection_grad,
            &self.packed_input,
        ] {
            encoder.fill_buffer(buffer, metal::NSRange::new(0, buffer.length()), 0);
        }
        encoder.end_encoding();
    }
}

impl FineGrainedMoeGrad {
    pub(crate) fn new(runtime: &Runtime) -> Result<Self, String> {
        let precise = |name| runtime.precise(SOURCE, "fine-grained MoE gradient", name);
        let tensorops = include_str!("../fbt_moe_routing_tensorops.metal");
        let metal4 = |name| runtime.pipeline_metal4(tensorops, "MoE gradient replay", name, &[]);
        Ok(Self {
            combine: precise("moe_combine_grad")?,
            routed_down_input: precise("moe_routed_down_input_grad")?,
            shared_down_input: precise("moe_shared_down_input_grad")?,
            routed_down_weight: precise("moe_routed_down_weight_grad")?,
            shared_down_weight: precise("moe_shared_down_weight_grad")?,
            routed_projection: metal4("fbt_moe_routed_gate_projection")?,
            shared_projection: metal4("fbt_moe_shared_gate_projection")?,
            swiglu: precise("moe_swiglu_grad")?,
            routed_gate_input: precise("moe_routed_gate_input_grad")?,
            shared_gate_input: precise("moe_shared_gate_input_grad")?,
            routed_gate_weight: precise("moe_routed_gate_weight_grad")?,
            shared_gate_weight: precise("moe_shared_gate_weight_grad")?,
            router_score: precise("moe_router_score_grad")?,
            router_project: precise("moe_router_project_grad")?,
        })
    }

    pub(crate) fn backward(
        &self,
        command: &CommandBufferRef,
        saved: &FineGrainedMoeBuffers,
        scratch: &MoeGradBuffers,
        grad: MoeGrad<'_>,
    ) -> Result<(), String> {
        if grad.rows == 0 || grad.rows > scratch.rows {
            return Err("MoE gradient rows exceed its replay allocation".into());
        }
        let p = Top3RouteShape::new(grad.rows, WIDTH);
        scratch.clear(command);
        let encoder = command.new_compute_command_encoder();
        self.combine(&encoder, saved, scratch, grad, &p);
        self.down(&encoder, saved, scratch, grad, &p);
        self.project(&encoder, saved, scratch, grad, &p)?;
        self.gate(&encoder, saved, scratch, grad, &p);
        self.router(&encoder, saved, scratch, grad, &p);
        encoder.end_encoding();
        Ok(())
    }

    fn combine(
        &self,
        encoder: &ComputeCommandEncoderRef,
        saved: &FineGrainedMoeBuffers,
        scratch: &MoeGradBuffers,
        grad: MoeGrad<'_>,
        p: &Top3RouteShape,
    ) {
        encoder.set_compute_pipeline_state(&self.combine);
        crate::apple_gpu::bind_buffers(
            encoder,
            &[
                (&saved.route_weights, 0),
                (&saved.packed_rows, 0),
                (&saved.routed_output, 0),
                grad.branch_grad,
                (&scratch.routed_output, 0),
                (&scratch.shared_output, 0),
                (&scratch.route, 0),
            ],
        );
        shape(encoder, 7, p);
        encoder.dispatch_thread_groups(thread_group(u64::from(grad.rows)), thread_group(128));
        encoder.memory_barrier_with_resources(&[
            &scratch.routed_output,
            &scratch.shared_output,
            &scratch.route,
        ]);
    }

    fn down(
        &self,
        encoder: &ComputeCommandEncoderRef,
        saved: &FineGrainedMoeBuffers,
        scratch: &MoeGradBuffers,
        grad: MoeGrad<'_>,
        p: &Top3RouteShape,
    ) {
        let assignments = u64::from(p.assignments());
        crate::apple_gpu::bind_buffers(
            encoder,
            &[
                (grad.weights, grad.down_offset),
                (&saved.packed_experts, 0),
                (&scratch.routed_output, 0),
                (&scratch.routed_activation, 0),
            ],
        );
        shape(encoder, 4, p);
        threads(
            encoder,
            &self.routed_down_input,
            assignments * u64::from(ROUTED_WIDTH),
        );
        crate::apple_gpu::bind_buffers(
            encoder,
            &[
                (grad.weights, grad.down_offset),
                (&scratch.shared_output, 0),
                (&scratch.shared_activation, 0),
            ],
        );
        shape(encoder, 3, p);
        threads(
            encoder,
            &self.shared_down_input,
            u64::from(grad.rows * ROUTED_WIDTH),
        );
        crate::apple_gpu::bind_buffers(
            encoder,
            &[
                (&saved.routed_activation, 0),
                (&scratch.routed_output, 0),
                (&saved.expert_offsets, 0),
                (&saved.expert_loads, 0),
                grad.down_grad,
            ],
        );
        shape(encoder, 5, p);
        threads(
            encoder,
            &self.routed_down_weight,
            u64::from(ROUTED_EXPERTS) * u64::from(ROUTED_WIDTH) * u64::from(WIDTH),
        );
        crate::apple_gpu::bind_buffers(
            encoder,
            &[
                (&saved.shared_activation, 0),
                (&scratch.shared_output, 0),
                grad.down_grad,
            ],
        );
        shape(encoder, 3, p);
        threads(
            encoder,
            &self.shared_down_weight,
            u64::from(ROUTED_WIDTH * WIDTH),
        );
        encoder.memory_barrier_with_resources(&[
            &scratch.routed_activation,
            &scratch.shared_activation,
            grad.down_grad.0,
        ]);
    }

    fn project(
        &self,
        encoder: &ComputeCommandEncoderRef,
        saved: &FineGrainedMoeBuffers,
        scratch: &MoeGradBuffers,
        grad: MoeGrad<'_>,
        p: &Top3RouteShape,
    ) -> Result<(), String> {
        let assignments = u64::from(p.assignments());
        tensor_indirect(
            encoder,
            &self.routed_projection,
            &[
                (&saved.packed_input, 0),
                (grad.weights, grad.gate_offset),
                (&saved.routed_tiles, 0),
                (&scratch.routed_projection, 0),
            ],
            std::ptr::from_ref(p).cast(),
            size_of::<Top3RouteShape>() as u64,
            &saved.routed_dispatch,
            0,
            4,
        )?;
        tensor_encode(
            encoder,
            &self.shared_projection,
            &[
                grad.input,
                (grad.weights, grad.gate_offset),
                (&scratch.shared_projection, 0),
            ],
            std::ptr::from_ref(p).cast(),
            size_of::<Top3RouteShape>() as u64,
            MTLSize {
                width: 1,
                height: u64::from(grad.rows.div_ceil(16)),
                depth: 1,
            },
        )?;
        encoder.memory_barrier_with_resources(&[
            &scratch.routed_projection,
            &scratch.shared_projection,
        ]);

        crate::apple_gpu::bind_buffers(
            encoder,
            &[
                (&scratch.routed_projection, 0),
                (&scratch.routed_activation, 0),
                (&scratch.routed_projection_grad, 0),
            ],
        );
        encoder.set_bytes(
            3,
            size_of::<u32>() as u64,
            std::ptr::from_ref(&p.assignments()).cast(),
        );
        threads(encoder, &self.swiglu, assignments * u64::from(ROUTED_WIDTH));
        crate::apple_gpu::bind_buffers(
            encoder,
            &[
                (&scratch.shared_projection, 0),
                (&scratch.shared_activation, 0),
                (&scratch.shared_projection_grad, 0),
            ],
        );
        encoder.set_bytes(
            3,
            size_of::<u32>() as u64,
            std::ptr::from_ref(&grad.rows).cast(),
        );
        threads(encoder, &self.swiglu, u64::from(grad.rows * ROUTED_WIDTH));
        encoder.memory_barrier_with_resources(&[
            &scratch.routed_projection_grad,
            &scratch.shared_projection_grad,
        ]);
        Ok(())
    }

    fn gate(
        &self,
        encoder: &ComputeCommandEncoderRef,
        saved: &FineGrainedMoeBuffers,
        scratch: &MoeGradBuffers,
        grad: MoeGrad<'_>,
        p: &Top3RouteShape,
    ) {
        let assignments = u64::from(p.assignments());
        crate::apple_gpu::bind_buffers(
            encoder,
            &[
                (grad.weights, grad.gate_offset),
                (&saved.packed_experts, 0),
                (&scratch.routed_projection_grad, 0),
                (&scratch.packed_input, 0),
            ],
        );
        shape(encoder, 4, p);
        threads(
            encoder,
            &self.routed_gate_input,
            assignments * u64::from(WIDTH),
        );
        crate::apple_gpu::bind_buffers(
            encoder,
            &[
                (grad.weights, grad.gate_offset),
                (&scratch.shared_projection_grad, 0),
                grad.input_grad,
            ],
        );
        shape(encoder, 3, p);
        threads(
            encoder,
            &self.shared_gate_input,
            u64::from(grad.rows * WIDTH),
        );
        crate::apple_gpu::bind_buffers(
            encoder,
            &[
                (&saved.packed_input, 0),
                (&scratch.routed_projection_grad, 0),
                (&saved.expert_offsets, 0),
                (&saved.expert_loads, 0),
                grad.gate_grad,
            ],
        );
        shape(encoder, 5, p);
        threads(
            encoder,
            &self.routed_gate_weight,
            u64::from(ROUTED_EXPERTS) * u64::from(WIDTH) * u64::from(MOE_GATEUP),
        );
        crate::apple_gpu::bind_buffers(
            encoder,
            &[
                grad.input,
                (&scratch.shared_projection_grad, 0),
                grad.gate_grad,
            ],
        );
        shape(encoder, 3, p);
        threads(
            encoder,
            &self.shared_gate_weight,
            u64::from(WIDTH * MOE_GATEUP),
        );
        encoder.memory_barrier_with_resources(&[&scratch.packed_input, grad.gate_grad.0]);
    }

    fn router(
        &self,
        encoder: &ComputeCommandEncoderRef,
        saved: &FineGrainedMoeBuffers,
        scratch: &MoeGradBuffers,
        grad: MoeGrad<'_>,
        p: &Top3RouteShape,
    ) {
        encoder.set_compute_pipeline_state(&self.router_score);
        crate::apple_gpu::bind_buffers(
            encoder,
            &[
                grad.input,
                (grad.weights, grad.router_offset),
                (&saved.route_experts, 0),
                (&saved.route_weights, 0),
                (&scratch.route, 0),
                (&scratch.router_score, 0),
            ],
        );
        shape(encoder, 6, p);
        encoder.dispatch_thread_groups(thread_group(u64::from(grad.rows)), thread_group(128));
        encoder.memory_barrier_with_resources(&[&scratch.router_score]);
        crate::apple_gpu::bind_buffers(
            encoder,
            &[
                grad.input,
                (grad.weights, grad.router_offset),
                (&saved.route_experts, 0),
                (&scratch.router_score, 0),
                (&scratch.packed_input, 0),
                (&saved.packed_rows, 0),
                grad.input_grad,
                grad.router_grad,
            ],
        );
        shape(encoder, 8, p);
        threads(encoder, &self.router_project, u64::from(grad.rows * WIDTH));
    }
}
