use std::ffi::c_void;

use metal::Buffer;

use super::{KdaMoeMetalExecutor, MetalDecoderParams, MetalSeed, set_bytes};
use crate::apple_gpu::thread_group;

pub(super) fn encode_vector(
    exec: &KdaMoeMetalExecutor,
    encoder: &metal::ComputeCommandEncoderRef,
    pipeline: &str,
    output: &Buffer,
    linear: &Buffer,
    seed: &MetalSeed,
    elements: u64,
) -> Result<(), String> {
    encoder.set_compute_pipeline_state(exec.pipeline(pipeline)?);
    exec.set_packed(encoder, 0);
    encoder.set_buffer(1, Some(&exec.scales), 0);
    encoder.set_buffer(2, Some(&exec.biases), 0);
    encoder.set_buffer(3, Some(output), 0);
    encoder.set_buffer(4, Some(linear), 0);
    set_bytes(encoder, 5, seed);
    encoder.dispatch_threads(thread_group(elements), thread_group(256));
    Ok(())
}

pub(super) fn encode_candidate(
    exec: &KdaMoeMetalExecutor,
    encoder: &metal::ComputeCommandEncoderRef,
    seed: u64,
) -> Result<(), String> {
    let seed = exec.seed(seed);
    encode_vector(
        exec,
        encoder,
        "packed_dequantize_row_half",
        &exec.attention_norm,
        &exec.attention_norm_linear,
        &seed,
        u64::from(exec.params.hidden_width),
    )?;
    encode_vector(
        exec,
        encoder,
        "packed_dequantize_row_half",
        &exec.moe_norm,
        &exec.moe_norm_linear,
        &seed,
        u64::from(exec.params.hidden_width),
    )?;
    encode_vector(
        exec,
        encoder,
        "packed_dequantize_row_float",
        &exec.decay,
        &exec.decay_linear,
        &seed,
        u64::from(exec.kda_params.heads),
    )?;
    encode_vector(
        exec,
        encoder,
        "packed_dequantize_row_float",
        &exec.time_bias,
        &exec.time_bias_linear,
        &seed,
        u64::from(exec.params.hidden_width),
    )?;
    encode_vector(
        exec,
        encoder,
        "packed_dequantize_row_float",
        &exec.output_norm,
        &exec.output_norm_linear,
        &seed,
        u64::from(exec.kda_params.value_width),
    )
}

fn encode_qkv(
    exec: &KdaMoeMetalExecutor,
    encoder: &metal::ComputeCommandEncoderRef,
    seed: &MetalSeed,
    tokens: u64,
    qkv_width: u64,
    qkv_elements: u64,
) -> Result<(), String> {
    encoder.set_compute_pipeline_state(exec.pipeline("decoder_rms_norm")?);
    encoder.set_buffer(0, Some(&exec.hidden), 0);
    encoder.set_buffer(1, Some(&exec.attention_norm), 0);
    encoder.set_buffer(2, Some(&exec.normalized), 0);
    set_bytes(encoder, 3, &exec.params);
    encoder.dispatch_threads(thread_group(tokens), thread_group(1));

    exec.encode_projection(
        encoder,
        &exec.normalized,
        &exec.qkv,
        &exec.qkv_linear,
        seed,
        tokens * qkv_width,
    )?;

    encoder.set_compute_pipeline_state(exec.pipeline("kda_short_conv_decode")?);
    encoder.set_buffer(0, Some(&exec.qkv), 0);
    encoder.set_buffer(1, Some(&exec.qkv_conv_history), 0);
    exec.set_packed(encoder, 2);
    encoder.set_buffer(3, Some(&exec.scales), 0);
    encoder.set_buffer(4, Some(&exec.biases), 0);
    encoder.set_buffer(5, Some(&exec.qkv_conv_linear), 0);
    set_bytes(encoder, 6, seed);
    encoder.dispatch_threads(thread_group(qkv_width), thread_group(256));

    encoder.set_compute_pipeline_state(exec.pipeline("kda_split_qkv_16k")?);
    encoder.set_buffer(0, Some(&exec.qkv), 0);
    encoder.set_buffer(1, Some(&exec.query), 0);
    encoder.set_buffer(2, Some(&exec.key), 0);
    encoder.set_buffer(3, Some(&exec.value), 0);
    set_bytes(encoder, 4, &exec.kda_params);
    encoder.dispatch_threads(thread_group(qkv_elements), thread_group(256));

    encoder.set_compute_pipeline_state(exec.pipeline("kda_normalize_qk")?);
    encoder.set_buffer(0, Some(&exec.query), 0);
    encoder.set_buffer(1, Some(&exec.key), 0);
    set_bytes(encoder, 2, &exec.kda_params);
    encoder.dispatch_threads(
        thread_group(u64::from(exec.kda_params.batch) * u64::from(exec.kda_params.heads)),
        thread_group(32),
    );
    Ok(())
}

fn encode_control(
    exec: &KdaMoeMetalExecutor,
    encoder: &metal::ComputeCommandEncoderRef,
    seed: &MetalSeed,
    tokens: u64,
    hidden: u64,
    control_width: u64,
) -> Result<(), String> {
    exec.encode_projection(
        encoder,
        &exec.normalized,
        &exec.control,
        &exec.control_linear,
        seed,
        tokens * control_width,
    )?;

    encoder.set_compute_pipeline_state(exec.pipeline("kda_split_control_16k")?);
    encoder.set_buffer(0, Some(&exec.control), 0);
    encoder.set_buffer(1, Some(&exec.forget_state), 0);
    encoder.set_buffer(2, Some(&exec.output_state), 0);
    encoder.set_buffer(3, Some(&exec.raw_beta), 0);
    set_bytes(encoder, 4, &exec.kda_control_params);
    encoder.dispatch_threads(thread_group(tokens * control_width), thread_group(256));

    exec.encode_projection(
        encoder,
        &exec.forget_state,
        &exec.raw_gate,
        &exec.forget_linear,
        seed,
        hidden,
    )?;
    exec.encode_projection(
        encoder,
        &exec.output_state,
        &exec.output_gate,
        &exec.output_gate_linear,
        seed,
        hidden,
    )?;

    encoder.set_compute_pipeline_state(exec.pipeline("kda_make_gate_beta_16k")?);
    encoder.set_buffer(0, Some(&exec.raw_gate), 0);
    encoder.set_buffer(1, Some(&exec.raw_beta), 0);
    encoder.set_buffer(2, Some(&exec.decay), 0);
    encoder.set_buffer(3, Some(&exec.time_bias), 0);
    encoder.set_buffer(4, Some(&exec.gate), 0);
    encoder.set_buffer(5, Some(&exec.beta), 0);
    set_bytes(encoder, 6, &exec.kda_control_params);
    encoder.dispatch_threads(thread_group(hidden), thread_group(256));
    Ok(())
}

fn encode_recurrence(
    exec: &KdaMoeMetalExecutor,
    encoder: &metal::ComputeCommandEncoderRef,
    seed: &MetalSeed,
    hidden: u64,
    recurrence_columns: u64,
) -> Result<(), String> {
    encoder.set_compute_pipeline_state(exec.pipeline("kda_step")?);
    encoder.set_buffer(0, Some(&exec.query), 0);
    encoder.set_buffer(1, Some(&exec.key), 0);
    encoder.set_buffer(2, Some(&exec.value), 0);
    encoder.set_buffer(3, Some(&exec.gate), 0);
    encoder.set_buffer(4, Some(&exec.beta), 0);
    encoder.set_buffer(5, Some(&exec.recurrence_state), 0);
    encoder.set_buffer(6, Some(&exec.post_kda), 0);
    set_bytes(encoder, 7, &exec.kda_params);
    encoder.dispatch_threads(thread_group(recurrence_columns), thread_group(256));

    encoder.set_compute_pipeline_state(exec.pipeline("kda_postprocess_16k")?);
    encoder.set_buffer(0, Some(&exec.post_kda), 0);
    encoder.set_buffer(1, Some(&exec.output_gate), 0);
    encoder.set_buffer(2, Some(&exec.output_norm), 0);
    encoder.set_buffer(3, Some(&exec.kda_gated), 0);
    set_bytes(encoder, 4, &exec.kda_control_params);
    encoder.dispatch_threads(thread_group(hidden), thread_group(256));

    exec.encode_projection(
        encoder,
        &exec.kda_gated,
        &exec.update,
        &exec.output_linear,
        seed,
        hidden,
    )?;

    encoder.set_compute_pipeline_state(exec.pipeline("decoder_residual")?);
    encoder.set_buffer(0, Some(&exec.hidden), 0);
    encoder.set_buffer(1, Some(&exec.update), 0);
    encoder.set_buffer(2, Some(&exec.hidden), 0);
    set_bytes(encoder, 3, &exec.params);
    encoder.dispatch_threads(thread_group(hidden), thread_group(256));
    Ok(())
}

fn encode_route(
    exec: &KdaMoeMetalExecutor,
    encoder: &metal::ComputeCommandEncoderRef,
    seed: &MetalSeed,
    tokens: u64,
    expert_activations: u64,
) -> Result<(), String> {
    encoder.set_compute_pipeline_state(exec.pipeline("moe_router_topk_simd")?);
    encoder.set_buffer(0, Some(&exec.normalized), 0);
    exec.set_packed(encoder, 1);
    encoder.set_buffer(2, Some(&exec.scales), 0);
    encoder.set_buffer(3, Some(&exec.biases), 0);
    encoder.set_buffer(4, Some(&exec.route_indices), 0);
    encoder.set_buffer(5, Some(&exec.route_weights), 0);
    encoder.set_buffer(6, Some(&exec.router), 0);
    set_bytes(encoder, 7, seed);
    set_bytes(encoder, 8, &exec.params);
    encoder.dispatch_thread_groups(thread_group(tokens), thread_group(512));

    encoder.set_compute_pipeline_state(exec.pipeline("moe_gate_up_simd")?);
    encoder.set_buffer(0, Some(&exec.normalized), 0);
    exec.set_packed(encoder, 1);
    encoder.set_buffer(2, Some(&exec.scales), 0);
    encoder.set_buffer(3, Some(&exec.biases), 0);
    encoder.set_buffer(4, Some(&exec.route_indices), 0);
    encoder.set_buffer(5, Some(&exec.expert_activation), 0);
    encoder.set_buffer(6, Some(&exec.expert_gate), 0);
    encoder.set_buffer(7, Some(&exec.expert_up), 0);
    set_bytes(encoder, 8, seed);
    set_bytes(encoder, 9, &exec.params);
    encoder.dispatch_threads(thread_group(expert_activations * 32), thread_group(256));
    Ok(())
}

fn encode_down(
    exec: &KdaMoeMetalExecutor,
    encoder: &metal::ComputeCommandEncoderRef,
    seed: &MetalSeed,
    hidden: u64,
) -> Result<(), String> {
    encoder.set_compute_pipeline_state(exec.pipeline("moe_down_simd")?);
    encoder.set_buffer(0, Some(&exec.expert_activation), 0);
    exec.set_packed(encoder, 1);
    encoder.set_buffer(2, Some(&exec.scales), 0);
    encoder.set_buffer(3, Some(&exec.biases), 0);
    encoder.set_buffer(4, Some(&exec.route_weights), 0);
    encoder.set_buffer(5, Some(&exec.expert_down), 0);
    encoder.set_buffer(6, Some(&exec.route_indices), 0);
    encoder.set_buffer(7, Some(&exec.update), 0);
    set_bytes(encoder, 8, seed);
    set_bytes(encoder, 9, &exec.params);
    encoder.dispatch_threads(thread_group(hidden * 32), thread_group(256));

    encoder.set_compute_pipeline_state(exec.pipeline("decoder_residual")?);
    encoder.set_buffer(0, Some(&exec.hidden), 0);
    encoder.set_buffer(1, Some(&exec.update), 0);
    encoder.set_buffer(2, Some(&exec.hidden), 0);
    set_bytes(encoder, 3, &exec.params);
    encoder.dispatch_threads(thread_group(hidden), thread_group(256));
    Ok(())
}

pub(super) fn encode_moe(
    exec: &KdaMoeMetalExecutor,
    encoder: &metal::ComputeCommandEncoderRef,
    seed: &MetalSeed,
    tokens: u64,
    expert_activations: u64,
    hidden: u64,
) -> Result<(), String> {
    encode_route(exec, encoder, seed, tokens, expert_activations)?;
    encode_down(exec, encoder, seed, hidden)
}

pub(super) fn encode_layer(
    exec: &KdaMoeMetalExecutor,
    encoder: &metal::ComputeCommandEncoderRef,
    seed: u64,
) -> Result<(), String> {
    if exec.params.batch != 1 || exec.params.length != 1 {
        return Err("resident layer decode currently requires shape [1, 1, hidden]".to_string());
    }
    let seed = exec.seed(seed);
    let tokens = u64::from(exec.params.batch) * u64::from(exec.params.length);
    let hidden = tokens * u64::from(exec.params.hidden_width);
    let qkv_width = u64::from(exec.kda_params.heads) * u64::from(exec.kda_params.key_width) * 3;
    let qkv_elements = qkv_width / 3;
    let control_width =
        2 * u64::from(exec.kda_control_params.gate_rank) + u64::from(exec.kda_params.heads);
    let recurrence_columns = u64::from(exec.kda_params.batch)
        * u64::from(exec.kda_params.heads)
        * u64::from(exec.kda_params.value_width);
    let expert_activations =
        tokens * u64::from(exec.params.top_k) * u64::from(exec.params.expert_width);

    encode_qkv(exec, encoder, &seed, tokens, qkv_width, qkv_elements)?;
    encode_control(exec, encoder, &seed, tokens, hidden, control_width)?;
    encode_recurrence(exec, encoder, &seed, hidden, recurrence_columns)?;

    encoder.set_compute_pipeline_state(exec.pipeline("decoder_rms_norm")?);
    encoder.set_buffer(0, Some(&exec.hidden), 0);
    encoder.set_buffer(1, Some(&exec.moe_norm), 0);
    encoder.set_buffer(2, Some(&exec.normalized), 0);
    set_bytes(encoder, 3, &exec.params);
    encoder.dispatch_threads(thread_group(tokens), thread_group(1));

    encode_moe(exec, encoder, &seed, tokens, expert_activations, hidden)
}

pub(super) fn encode_norm(
    exec: &mut KdaMoeMetalExecutor,
    pipeline_name: &str,
    norm: &[u16],
    norm_buffer: &Buffer,
    batch: usize,
    sequence_length: usize,
    epsilon: f32,
) -> Result<(), String> {
    let hidden_width = exec.memory.hidden_elements / (batch * sequence_length);
    if norm.len() != hidden_width {
        return Err(format!(
            "RMSNorm has {} elements, expected {hidden_width}",
            norm.len()
        ));
    }
    if batch == 0
        || sequence_length == 0
        || batch * sequence_length * hidden_width != exec.memory.hidden_elements
    {
        return Err("RMSNorm shape does not match resident hidden buffer".to_string());
    }
    unsafe {
        std::ptr::copy_nonoverlapping(
            norm.as_ptr(),
            norm_buffer.contents().cast::<u16>(),
            norm.len(),
        );
    }
    let params = MetalDecoderParams {
        batch: u32::try_from(batch).map_err(|_| "batch exceeds u32")?,
        length: u32::try_from(sequence_length).map_err(|_| "sequence length exceeds u32")?,
        hidden_width: u32::try_from(hidden_width).map_err(|_| "hidden width exceeds u32")?,
        rms_epsilon: epsilon,
        ..exec.params
    };
    let command = exec.runtime.queue.new_command_buffer();
    let encoder = command.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(exec.pipeline(pipeline_name)?);
    encoder.set_buffer(0, Some(&exec.hidden), 0);
    encoder.set_buffer(1, Some(norm_buffer), 0);
    encoder.set_buffer(2, Some(&exec.normalized), 0);
    encoder.set_bytes(
        3,
        std::mem::size_of::<MetalDecoderParams>() as u64,
        (&params as *const MetalDecoderParams).cast::<c_void>(),
    );
    encoder.dispatch_threads(
        thread_group((batch * sequence_length) as u64),
        thread_group(256),
    );
    encoder.end_encoding();
    command.commit();
    command.wait_until_completed();
    Ok(())
}

fn project_16k(
    exec: &KdaMoeMetalExecutor,
    encoder: &metal::ComputeCommandEncoderRef,
    input: &Buffer,
    output: &Buffer,
    linear: &Buffer,
    seed: &MetalSeed,
    threads: u64,
) -> Result<(), String> {
    encoder.set_compute_pipeline_state(exec.pipeline("kda_project_packed_16k")?);
    encoder.set_buffer(0, Some(input), 0);
    exec.set_packed(encoder, 1);
    encoder.set_buffer(2, Some(&exec.scales), 0);
    encoder.set_buffer(3, Some(&exec.biases), 0);
    encoder.set_buffer(4, Some(output), 0);
    encoder.set_buffer(5, Some(linear), 0);
    set_bytes(encoder, 6, seed);
    encoder.dispatch_threads(thread_group(threads), thread_group(256));
    Ok(())
}

pub(super) fn encode_gates(
    exec: &KdaMoeMetalExecutor,
    encoder: &metal::ComputeCommandEncoderRef,
    seed: &MetalSeed,
    tokens: u64,
    hidden: u64,
    qkv_threads: u64,
    qkv_elements: u64,
    control_width: u64,
) -> Result<(), String> {
    project_16k(
        exec,
        encoder,
        &exec.normalized,
        &exec.qkv,
        &exec.qkv_linear,
        seed,
        qkv_threads,
    )?;
    project_16k(
        exec,
        encoder,
        &exec.normalized,
        &exec.control,
        &exec.control_linear,
        seed,
        tokens * control_width,
    )?;

    encoder.set_compute_pipeline_state(exec.pipeline("kda_split_qkv_16k")?);
    encoder.set_buffer(0, Some(&exec.qkv), 0);
    encoder.set_buffer(1, Some(&exec.query), 0);
    encoder.set_buffer(2, Some(&exec.key), 0);
    encoder.set_buffer(3, Some(&exec.value), 0);
    set_bytes(encoder, 4, &exec.kda_params);
    encoder.dispatch_threads(thread_group(qkv_elements), thread_group(256));

    encoder.set_compute_pipeline_state(exec.pipeline("kda_split_control_16k")?);
    encoder.set_buffer(0, Some(&exec.control), 0);
    encoder.set_buffer(1, Some(&exec.forget_state), 0);
    encoder.set_buffer(2, Some(&exec.output_state), 0);
    encoder.set_buffer(3, Some(&exec.raw_beta), 0);
    set_bytes(encoder, 4, &exec.kda_control_params);
    encoder.dispatch_threads(thread_group(tokens * control_width), thread_group(256));

    project_16k(
        exec,
        encoder,
        &exec.forget_state,
        &exec.raw_gate,
        &exec.forget_linear,
        seed,
        hidden,
    )?;
    project_16k(
        exec,
        encoder,
        &exec.output_state,
        &exec.output_gate,
        &exec.output_gate_linear,
        seed,
        hidden,
    )?;

    encoder.set_compute_pipeline_state(exec.pipeline("kda_make_gate_beta_16k")?);
    encoder.set_buffer(0, Some(&exec.raw_gate), 0);
    encoder.set_buffer(1, Some(&exec.raw_beta), 0);
    encoder.set_buffer(2, Some(&exec.decay), 0);
    encoder.set_buffer(3, Some(&exec.time_bias), 0);
    encoder.set_buffer(4, Some(&exec.gate), 0);
    encoder.set_buffer(5, Some(&exec.beta), 0);
    set_bytes(encoder, 6, &exec.kda_control_params);
    encoder.dispatch_threads(thread_group(hidden), thread_group(256));
    Ok(())
}

pub(super) fn encode_state(
    exec: &KdaMoeMetalExecutor,
    encoder: &metal::ComputeCommandEncoderRef,
    seed: &MetalSeed,
    hidden: u64,
    recurrence_threads: u64,
) -> Result<(), String> {
    encoder.set_compute_pipeline_state(exec.pipeline("kda_recurrence_16k")?);
    encoder.set_buffer(0, Some(&exec.query), 0);
    encoder.set_buffer(1, Some(&exec.key), 0);
    encoder.set_buffer(2, Some(&exec.value), 0);
    encoder.set_buffer(3, Some(&exec.gate), 0);
    encoder.set_buffer(4, Some(&exec.beta), 0);
    encoder.set_buffer(5, Some(&exec.post_kda), 0);
    encoder.set_buffer(6, Some(&exec.recurrence_state), 0);
    set_bytes(encoder, 7, &exec.kda_params);
    encoder.dispatch_threads(thread_group(recurrence_threads), thread_group(256));

    encoder.set_compute_pipeline_state(exec.pipeline("kda_postprocess_16k")?);
    encoder.set_buffer(0, Some(&exec.post_kda), 0);
    encoder.set_buffer(1, Some(&exec.output_gate), 0);
    encoder.set_buffer(2, Some(&exec.output_norm), 0);
    encoder.set_buffer(3, Some(&exec.kda_gated), 0);
    set_bytes(encoder, 4, &exec.kda_control_params);
    encoder.dispatch_threads(thread_group(hidden), thread_group(256));

    project_16k(
        exec,
        encoder,
        &exec.kda_gated,
        &exec.update,
        &exec.output_linear,
        seed,
        hidden,
    )?;

    encoder.set_compute_pipeline_state(exec.pipeline("decoder_residual")?);
    encoder.set_buffer(0, Some(&exec.hidden), 0);
    encoder.set_buffer(1, Some(&exec.update), 0);
    encoder.set_buffer(2, Some(&exec.hidden), 0);
    set_bytes(encoder, 3, &exec.params);
    encoder.dispatch_threads(thread_group(hidden), thread_group(256));
    Ok(())
}
