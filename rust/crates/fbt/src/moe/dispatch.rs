use super::*;

pub(super) fn encode(
    pipelines: &Pipelines,
    buffers: &Buffers,
    command: &CommandBufferRef,
    selected: Option<Stage>,
) {
    let shape = MoeShape {
        rows: ROWS,
        width: WIDTH,
        experts: EXPERTS,
        rows_per_expert: EXPERT_ROWS,
        expert_width: EXPERT_WIDTH,
    };
    let shape_ptr = (&shape as *const MoeShape).cast();
    let shape_bytes = std::mem::size_of::<MoeShape>() as u64;
    if selected.is_none_or(|stage| stage == Stage::Gate) {
        dispatch(
            command,
            &pipelines.gate,
            &[&buffers.input, &buffers.router, &buffers.gates],
            shape_ptr,
            shape_bytes,
            u64::from(ROWS) * 32,
        );
    }
    if selected.is_none_or(|stage| stage == Stage::Group) {
        dispatch(
            command,
            &pipelines.group,
            &[&buffers.input, &buffers.grouped],
            shape_ptr,
            shape_bytes,
            u64::from(ROWS) * u64::from(WIDTH),
        );
    }
    if selected.is_none_or(|stage| stage == Stage::Swiglu) {
        dispatch(
            command,
            &pipelines.swiglu,
            &[&buffers.gate_up, &buffers.activation],
            shape_ptr,
            shape_bytes,
            u64::from(ROWS) * u64::from(EXPERT_WIDTH),
        );
    }
    if selected.is_none_or(|stage| stage == Stage::Ungroup) {
        dispatch(
            command,
            &pipelines.ungroup,
            &[
                &buffers.input,
                &buffers.down,
                &buffers.gates,
                &buffers.output,
            ],
            shape_ptr,
            shape_bytes,
            u64::from(ROWS) * u64::from(WIDTH),
        );
    }
}

pub(super) use crate::tensor::{complete, dispatch_on};

pub(super) fn encode_tensorops(
    command: &CommandBufferRef,
    pipeline: &ComputePipelineState,
    buffers: &[&BufferRef],
    output_width: u32,
) -> Result<(), String> {
    tensor_groups(
        command,
        pipeline,
        buffers,
        MTLSize {
            width: u64::from(output_width / 64),
            height: u64::from(EXPERT_ROWS / 128),
            depth: u64::from(EXPERTS),
        },
    )
}

pub(super) fn tensor_groups(
    command: &CommandBufferRef,
    pipeline: &ComputePipelineState,
    buffers: &[&BufferRef],
    groups: MTLSize,
) -> Result<(), String> {
    let simd_width = pipeline.thread_execution_width();
    let threads = simd_width * 4;
    if simd_width != 32 || pipeline.max_total_threads_per_threadgroup() < threads {
        return Err(format!(
            "TensorOps kernel requires four 32-lane SIMD groups; pipeline reports SIMD width \
             {simd_width} and max threads {}",
            pipeline.max_total_threads_per_threadgroup()
        ));
    }
    let encoder = command.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(pipeline);
    for (index, buffer) in buffers.iter().enumerate() {
        encoder.set_buffer(index as u64, Some(buffer), 0);
    }
    encoder.dispatch_thread_groups(
        groups,
        MTLSize {
            width: threads,
            height: 1,
            depth: 1,
        },
    );
    encoder.end_encoding();
    Ok(())
}

pub(super) fn tensor_at(
    command: &CommandBufferRef,
    pipeline: &ComputePipelineState,
    buffers: &[(&BufferRef, u64)],
    groups: MTLSize,
) -> Result<(), String> {
    let encoder = command.new_compute_command_encoder();
    tensor_command(&encoder, pipeline, buffers, groups)?;
    encoder.end_encoding();
    Ok(())
}

pub(super) fn tensor_command(
    encoder: &ComputeCommandEncoderRef,
    pipeline: &ComputePipelineState,
    buffers: &[(&BufferRef, u64)],
    groups: MTLSize,
) -> Result<(), String> {
    tensor_workspace(encoder, pipeline, buffers, groups, 4)
}

pub(super) fn tensor_workspace(
    encoder: &ComputeCommandEncoderRef,
    pipeline: &ComputePipelineState,
    buffers: &[(&BufferRef, u64)],
    groups: MTLSize,
    simdgroups: u64,
) -> Result<(), String> {
    let simd_width = pipeline.thread_execution_width();
    let threads = simd_width * simdgroups;
    if simd_width != 32 || pipeline.max_total_threads_per_threadgroup() < threads {
        return Err(format!(
            "TensorOps kernel requires {simdgroups} 32-lane SIMD groups; pipeline reports SIMD \
             width {simd_width} and max threads {}",
            pipeline.max_total_threads_per_threadgroup()
        ));
    }
    encoder.set_compute_pipeline_state(pipeline);
    for (index, (buffer, offset)) in buffers.iter().enumerate() {
        encoder.set_buffer(index as u64, Some(buffer), *offset);
    }
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

pub(super) fn benchmark_materialize(
    runtime: &Runtime,
    pipeline: &ComputePipelineState,
    buffers: &[&BufferRef],
    output_width: u32,
    input_width: u32,
) -> Result<f64, String> {
    let mut samples = Vec::with_capacity(3);
    for iteration in 0..6 {
        let command = runtime.queue.new_command_buffer();
        let encoder = command.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(pipeline);
        for (index, buffer) in buffers.iter().enumerate() {
            encoder.set_buffer(index as u64, Some(buffer), 0);
        }
        encoder.dispatch_threads(
            MTLSize {
                width: u64::from(output_width / 4),
                height: u64::from(input_width),
                depth: u64::from(EXPERTS),
            },
            MTLSize {
                width: 64,
                height: 1,
                depth: 1,
            },
        );
        encoder.end_encoding();
        let seconds = complete(command)?;
        if iteration >= 3 {
            samples.push(seconds);
        }
    }
    samples.sort_by(f64::total_cmp);
    Ok(samples[1])
}

pub(super) fn mps_reference(runtime: &Runtime, buffers: &Buffers) -> Result<f64, String> {
    let mut matmul = Matmul::default();
    let command = runtime.queue.new_command_buffer();
    matmul.encode(
        &runtime.device,
        command,
        Matrix::half(&buffers.grouped, EXPERT_ROWS, WIDTH).layout(
            EXPERTS,
            u64::from(EXPERT_ROWS) * u64::from(WIDTH),
            0,
        ),
        Matrix::half(&buffers.materialized_gate_up, WIDTH, GATE_UP).layout(
            EXPERTS,
            u64::from(WIDTH) * u64::from(GATE_UP),
            0,
        ),
        Matrix::half(&buffers.mps_gate_up, EXPERT_ROWS, GATE_UP).layout(
            EXPERTS,
            u64::from(EXPERT_ROWS) * u64::from(GATE_UP),
            0,
        ),
        false,
        1.0,
    )?;
    matmul.encode(
        &runtime.device,
        command,
        Matrix::half(&buffers.activation, EXPERT_ROWS, EXPERT_WIDTH).layout(
            EXPERTS,
            u64::from(EXPERT_ROWS) * u64::from(EXPERT_WIDTH),
            0,
        ),
        Matrix::half(&buffers.materialized_down, EXPERT_WIDTH, WIDTH).layout(
            EXPERTS,
            u64::from(EXPERT_WIDTH) * u64::from(WIDTH),
            0,
        ),
        Matrix::half(&buffers.mps_down, EXPERT_ROWS, WIDTH).layout(
            EXPERTS,
            u64::from(EXPERT_ROWS) * u64::from(WIDTH),
            0,
        ),
        false,
        1.0,
    )?;
    complete(command)
}

pub(super) fn dispatch(
    command: &CommandBufferRef,
    pipeline: &ComputePipelineState,
    buffers: &[&BufferRef],
    parameters: *const std::ffi::c_void,
    parameter_bytes: u64,
    threads: u64,
) {
    let encoder = command.new_compute_command_encoder();
    dispatch_on(
        &encoder,
        pipeline,
        buffers,
        parameters,
        parameter_bytes,
        threads,
    );
    encoder.end_encoding();
}

pub(super) fn rms_at(
    command: &CommandBufferRef,
    pipelines: &Pipelines,
    input: &BufferRef,
    weight: &BufferRef,
    weight_offset: u64,
    output: &BufferRef,
) {
    let encoder = command.new_compute_command_encoder();
    rms_command(&encoder, pipelines, input, weight, weight_offset, output);
    encoder.end_encoding();
}

pub(super) fn rms_command(
    encoder: &ComputeCommandEncoderRef,
    pipelines: &Pipelines,
    input: &BufferRef,
    weight: &BufferRef,
    weight_offset: u64,
    output: &BufferRef,
) {
    rms_rows(
        encoder,
        pipelines,
        input,
        weight,
        weight_offset,
        output,
        ROWS,
    );
}

#[allow(clippy::too_many_arguments)]
pub(super) fn rms_rows(
    encoder: &ComputeCommandEncoderRef,
    pipelines: &Pipelines,
    input: &BufferRef,
    weight: &BufferRef,
    weight_offset: u64,
    output: &BufferRef,
    rows: u32,
) {
    rms_offsets(
        encoder,
        pipelines,
        (input, 0),
        (weight, weight_offset),
        (output, 0),
        rows,
    );
}

pub(super) fn rms_offsets(
    encoder: &ComputeCommandEncoderRef,
    pipelines: &Pipelines,
    input: (&BufferRef, u64),
    weight: (&BufferRef, u64),
    output: (&BufferRef, u64),
    rows: u32,
) {
    let shape = model_rows(rows);
    encoder.set_compute_pipeline_state(&pipelines.rms);
    encoder.set_buffer(0, Some(input.0), input.1);
    encoder.set_buffer(1, Some(weight.0), weight.1);
    encoder.set_buffer(2, Some(output.0), output.1);
    encoder.set_bytes(
        3,
        std::mem::size_of::<MoeShape>() as u64,
        (&shape as *const MoeShape).cast(),
    );
    encoder.dispatch_thread_groups(
        thread_group(u64::from(rows)),
        MTLSize {
            width: 128,
            height: 1,
            depth: 1,
        },
    );
}
