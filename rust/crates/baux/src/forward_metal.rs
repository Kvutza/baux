//! Resident Metal resources for the experimental KDA-MoE forward path.
//!
//! This module deliberately owns no Python objects. Callers upload packed
//! weights once, then reuse the same buffers for every seeded BO round.

use std::collections::BTreeMap;
use std::sync::Arc;

use metal::{Buffer, ComputePipelineState};

use crate::apple_gpu::{Runtime, thread_group};
use crate::forward_program::kda::{KdaControlRequest, KdaMoeLayerRequest};
use crate::weights::ComputeDevice;

#[path = "forward_metal/encode.rs"]
mod encode;
#[path = "forward_metal/stack.rs"]
mod stack;
use encode::*;
pub use stack::KdaMoeMetalModel;
use stack::*;

const KERNELS: &[&str] = &[
    "decoder_rms_norm",
    "kda_project_packed_16k",
    "kda_split_qkv_16k",
    "kda_split_control_16k",
    "kda_make_gate_beta_16k",
    "kda_recurrence_16k",
    "kda_prefill_16k",
    "kda_step",
    "kda_postprocess_16k",
    "decoder_project_packed",
    "decoder_project_packed_simd",
    "packed_dequantize_row_half",
    "packed_dequantize_row_float",
    "kda_short_conv_decode",
    "kda_normalize_qk",
    "decoder_residual",
    "moe_router_topk",
    "moe_router_topk_simd",
    "moe_gate_up",
    "moe_gate_up_simd",
    "moe_down",
    "moe_down_simd",
    "packed_embedding_lookup",
    "decoder_argmax",
];

const KDA_SOURCE: &str = include_str!("kda_moe.metal");

/// Immutable packed model arenas uploaded once per resident evaluator.
pub struct KdaMoeMetalWeights<'a> {
    pub packed: &'a [u8],
    pub scales: &'a [f32],
    pub biases: &'a [f32],
}

/// Shared immutable model arenas. Cloning a Metal handle is cheap; all layer
/// executors refer to these three allocations instead of copying the model.
pub struct KdaMoeMetalArena {
    packed: Buffer,
    scales: Buffer,
    biases: Buffer,
    packed_bytes: usize,
}

impl KdaMoeMetalArena {
    pub fn new(weights: KdaMoeMetalWeights<'_>) -> Result<Self, String> {
        if weights.packed.is_empty() || weights.scales.is_empty() || weights.biases.is_empty() {
            return Err("KDA-MoE model arenas cannot be empty".to_string());
        }
        let runtime = Runtime::shared()?;
        let packed = runtime.buffer_with(weights.packed);
        let scales = runtime.buffer_with(weights.scales);
        let biases = runtime.buffer_with(weights.biases);
        if packed.contents().is_null() || scales.contents().is_null() || biases.contents().is_null()
        {
            return Err("Metal could not allocate KDA-MoE model arenas".to_string());
        }
        Ok(Self {
            packed,
            scales,
            biases,
            packed_bytes: weights.packed.len(),
        })
    }

    pub fn packed_buffer(&self) -> Buffer {
        self.packed.to_owned()
    }

    pub fn packed_bytes(&self) -> usize {
        self.packed_bytes
    }
}

/// Dequantized KDA vectors. They are tiny compared with model weights and are
/// uploaded once so the recurrent path can remain entirely on the GPU.
pub struct KdaMoeMetalKdaVectors<'a> {
    pub decay: &'a [f32],
    pub time_bias: &'a [f32],
    pub output_norm: &'a [f32],
}

/// Sizes of the device-resident forward buffers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KdaMoeMetalMemory {
    pub hidden_elements: usize,
    pub qkv_elements: usize,
    pub recurrence_state_elements: usize,
    pub route_elements: usize,
    pub expert_elems: usize,
}

/// Compiled fused pipeline set and all per-layer resident storage.
pub struct KdaMoeMetalExecutor {
    #[allow(dead_code)] // Retained for command-buffer encoding methods.
    runtime: Arc<Runtime>,
    pipelines: BTreeMap<&'static str, ComputePipelineState>,
    pub packed: Buffer,
    packed_bytes: usize,
    packed_offset: u64,
    device_view: bool,
    pub scales: Buffer,
    pub biases: Buffer,
    pub hidden: Buffer,
    pub normalized: Buffer,
    pub qkv: Buffer,
    pub query: Buffer,
    pub key: Buffer,
    pub value: Buffer,
    pub recurrence_state: Buffer,
    pub route_indices: Buffer,
    pub route_weights: Buffer,
    pub expert_activation: Buffer,
    pub update: Buffer,
    attention_norm: Buffer,
    moe_norm: Buffer,
    attention_norm_linear: Buffer,
    moe_norm_linear: Buffer,
    router: Buffer,
    expert_gate: Buffer,
    expert_up: Buffer,
    expert_down: Buffer,
    qkv_linear: Buffer,
    qkv_conv_linear: Buffer,
    qkv_conv_history: Buffer,
    control_linear: Buffer,
    forget_linear: Buffer,
    output_gate_linear: Buffer,
    output_linear: Buffer,
    decay_linear: Buffer,
    time_bias_linear: Buffer,
    output_norm_linear: Buffer,
    control: Buffer,
    forget_state: Buffer,
    output_state: Buffer,
    raw_beta: Buffer,
    raw_gate: Buffer,
    output_gate: Buffer,
    post_kda: Buffer,
    kda_gated: Buffer,
    gate: Buffer,
    beta: Buffer,
    decay: Buffer,
    time_bias: Buffer,
    output_norm: Buffer,
    params: MetalDecoderParams,
    kda_params: MetalKdaParams,
    kda_control_params: MetalKdaControlParams,
    memory: KdaMoeMetalMemory,
}

fn check_inputs(
    request: &KdaMoeLayerRequest,
    control: &KdaControlRequest,
    device: ComputeDevice,
    vectors: &KdaMoeMetalKdaVectors<'_>,
) -> Result<(), String> {
    request.validate()?;
    control.validate(request.kda.tensor, request.kda.qkv.input_width)?;
    if device != ComputeDevice::Metal {
        return Err("KDA-MoE Metal executor requires device 'metal'".to_string());
    }
    let tensor = request.kda.tensor;
    let hidden = request.kda.qkv.input_width;
    if vectors.decay.len() != tensor.heads
        || vectors.time_bias.len() != hidden
        || vectors.output_norm.len() != tensor.value_width
    {
        return Err("KDA vector dimensions do not match the tensor layout".to_string());
    }
    Ok(())
}

fn load_pipelines(
    runtime: &Runtime,
) -> Result<BTreeMap<&'static str, ComputePipelineState>, String> {
    let mut pipelines = BTreeMap::new();
    for &name in KERNELS {
        let pipeline = runtime.pipeline(KDA_SOURCE, "kda-moe", name)?;
        pipelines.insert(name, pipeline);
    }
    Ok(pipelines)
}

fn make_memory(request: &KdaMoeLayerRequest) -> Result<KdaMoeMetalMemory, String> {
    let tensor = request.kda.tensor;
    let hidden_elements = request.hidden_elements();
    let qkv_elements = hidden_elements
        .checked_mul(3)
        .ok_or("KDA-MoE QKV buffer size overflow")?;
    let recurrence_state_elements = tensor.state_elements();
    let route_elements = tensor
        .batch
        .checked_mul(tensor.sequence_length)
        .and_then(|n| n.checked_mul(request.top_k))
        .ok_or("KDA-MoE route buffer size overflow")?;
    let expert_elems = request.expert_elems();
    Ok(KdaMoeMetalMemory {
        hidden_elements,
        qkv_elements,
        recurrence_state_elements,
        route_elements,
        expert_elems,
    })
}

fn make_params(
    request: &KdaMoeLayerRequest,
    control: &KdaControlRequest,
) -> Result<(MetalDecoderParams, MetalKdaParams, MetalKdaControlParams), String> {
    let tensor = request.kda.tensor;
    let params = MetalDecoderParams {
        batch: u32::try_from(tensor.batch).map_err(|_| "batch exceeds u32")?,
        length: u32::try_from(tensor.sequence_length).map_err(|_| "sequence length exceeds u32")?,
        hidden_width: u32::try_from(request.kda.qkv.input_width)
            .map_err(|_| "hidden width exceeds u32")?,
        experts: u32::try_from(request.expert_gate.len())
            .map_err(|_| "expert count exceeds u32")?,
        top_k: u32::try_from(request.top_k).map_err(|_| "top_k exceeds u32")?,
        expert_width: u32::try_from(request.expert_gate[0].output_width)
            .map_err(|_| "expert width exceeds u32")?,
        residual_scale: request.residual_scale,
        rms_epsilon: request.rms_epsilon,
    };
    let kda_params = MetalKdaParams {
        batch: u32::try_from(tensor.batch).map_err(|_| "batch exceeds u32")?,
        length: u32::try_from(tensor.sequence_length).map_err(|_| "sequence length exceeds u32")?,
        heads: u32::try_from(tensor.heads).map_err(|_| "head count exceeds u32")?,
        key_width: u32::try_from(tensor.key_width).map_err(|_| "key width exceeds u32")?,
        value_width: u32::try_from(tensor.value_width).map_err(|_| "value width exceeds u32")?,
    };
    let kda_control_params = MetalKdaControlParams {
        batch: kda_params.batch,
        length: kda_params.length,
        heads: kda_params.heads,
        head_width: kda_params.value_width,
        gate_rank: u32::try_from(control.gate_rank).map_err(|_| "KDA gate rank exceeds u32")?,
        rms_epsilon: params.rms_epsilon,
    };
    Ok((params, kda_params, kda_control_params))
}

fn alloc_executor(
    runtime: Arc<Runtime>,
    request: &KdaMoeLayerRequest,
    control: &KdaControlRequest,
    vectors: KdaMoeMetalKdaVectors<'_>,
    arena: &KdaMoeMetalArena,
    memory: KdaMoeMetalMemory,
    params: MetalDecoderParams,
    kda_params: MetalKdaParams,
    kda_control_params: MetalKdaControlParams,
    pipelines: BTreeMap<&'static str, ComputePipelineState>,
) -> Result<KdaMoeMetalExecutor, String> {
    let tensor = request.kda.tensor;
    let tokens = tensor.batch * tensor.sequence_length;
    Ok(KdaMoeMetalExecutor {
        packed: arena.packed.to_owned(),
        packed_bytes: arena.packed_bytes,
        packed_offset: 0,
        device_view: false,
        scales: arena.scales.to_owned(),
        biases: arena.biases.to_owned(),
        hidden: runtime.buffer::<u16>(memory.hidden_elements),
        normalized: runtime.buffer::<u16>(memory.hidden_elements),
        qkv: runtime.buffer::<u16>(memory.qkv_elements),
        query: runtime.buffer::<u16>(memory.hidden_elements),
        key: runtime.buffer::<u16>(memory.hidden_elements),
        value: runtime.buffer::<u16>(memory.hidden_elements),
        recurrence_state: runtime.buffer::<f32>(memory.recurrence_state_elements),
        route_indices: runtime.buffer::<u32>(memory.route_elements),
        route_weights: runtime.buffer::<u16>(memory.route_elements),
        expert_activation: runtime.buffer::<u16>(memory.expert_elems),
        update: runtime.buffer::<u16>(memory.hidden_elements),
        attention_norm: runtime.buffer::<u16>(request.kda.qkv.input_width),
        moe_norm: runtime.buffer::<u16>(request.kda.qkv.input_width),
        attention_norm_linear: runtime
            .buffer_with(&[MetalPackedLinear::from(request.attention_norm)]),
        moe_norm_linear: runtime.buffer_with(&[MetalPackedLinear::from(request.moe_norm)]),
        router: runtime.buffer_with(&[MetalPackedLinear::from(request.router)]),
        expert_gate: runtime.buffer_with(&metal_linears(&request.expert_gate)),
        expert_up: runtime.buffer_with(&metal_linears(&request.expert_up)),
        expert_down: runtime.buffer_with(&metal_linears(&request.expert_down)),
        qkv_linear: runtime.buffer_with(&[MetalPackedLinear::from(request.kda.qkv)]),
        qkv_conv_linear: runtime.buffer_with(&[MetalPackedLinear::from(control.qkv_conv)]),
        qkv_conv_history: runtime.buffer::<u16>(tensor.batch * 3 * control.qkv_conv.output_width),
        control_linear: runtime.buffer_with(&[MetalPackedLinear::from(control.control)]),
        forget_linear: runtime.buffer_with(&[MetalPackedLinear::from(control.forget)]),
        output_gate_linear: runtime.buffer_with(&[MetalPackedLinear::from(control.output_gate)]),
        output_linear: runtime.buffer_with(&[MetalPackedLinear::from(control.output)]),
        decay_linear: runtime.buffer_with(&[MetalPackedLinear::from(control.decay)]),
        time_bias_linear: runtime.buffer_with(&[MetalPackedLinear::from(control.time_bias)]),
        output_norm_linear: runtime.buffer_with(&[MetalPackedLinear::from(control.output_norm)]),
        control: runtime.buffer::<u16>(tokens * (2 * control.gate_rank + tensor.heads)),
        forget_state: runtime.buffer::<u16>(tokens * control.gate_rank),
        output_state: runtime.buffer::<u16>(tokens * control.gate_rank),
        raw_beta: runtime.buffer::<u16>(tokens * tensor.heads),
        raw_gate: runtime.buffer::<u16>(memory.hidden_elements),
        output_gate: runtime.buffer::<u16>(memory.hidden_elements),
        post_kda: runtime.buffer::<u16>(memory.hidden_elements),
        kda_gated: runtime.buffer::<u16>(memory.hidden_elements),
        gate: runtime.buffer::<f32>(tensor.qkv_elements()),
        beta: runtime.buffer::<f32>(tensor.batch * tensor.sequence_length * tensor.heads),
        decay: runtime.buffer_with(vectors.decay),
        time_bias: runtime.buffer_with(vectors.time_bias),
        output_norm: runtime.buffer_with(vectors.output_norm),
        params,
        kda_params,
        kda_control_params,
        runtime,
        pipelines,
        memory,
    })
}

impl KdaMoeMetalExecutor {
    /// Compile or load every layer pipeline and allocate its resident buffers.
    pub fn new(
        request: &KdaMoeLayerRequest,
        control: KdaControlRequest,
        vectors: KdaMoeMetalKdaVectors<'_>,
        device: ComputeDevice,
        weights: KdaMoeMetalWeights<'_>,
    ) -> Result<Self, String> {
        let arena = KdaMoeMetalArena::new(weights)?;
        Self::new_arena(request, control, vectors, device, &arena)
    }

    pub fn new_arena(
        request: &KdaMoeLayerRequest,
        control: KdaControlRequest,
        vectors: KdaMoeMetalKdaVectors<'_>,
        device: ComputeDevice,
        arena: &KdaMoeMetalArena,
    ) -> Result<Self, String> {
        check_inputs(request, &control, device, &vectors)?;
        let runtime = Runtime::shared()?;
        let pipelines = load_pipelines(&runtime)?;
        let memory = make_memory(request)?;
        let (params, kda_params, kda_control_params) = make_params(request, &control)?;
        alloc_executor(
            runtime,
            request,
            &control,
            vectors,
            arena,
            memory,
            params,
            kda_params,
            kda_control_params,
            pipelines,
        )
    }

    pub fn memory(&self) -> KdaMoeMetalMemory {
        self.memory
    }

    pub fn pipeline(&self, name: &str) -> Result<&ComputePipelineState, String> {
        self.pipelines
            .get(name)
            .ok_or_else(|| format!("KDA-MoE pipeline {name:?} is not loaded"))
    }

    /// Bind the packed row selected by the resident ENN search.
    ///
    /// The search owns the allocation and materializes the candidate into a
    /// slot. This executor keeps another Metal handle to that allocation; it
    /// does not copy the model row. Kernels then read the row as-is.
    pub fn bind_row(
        &mut self,
        rows: &Buffer,
        offset: usize,
        row_bytes: usize,
    ) -> Result<(), String> {
        if row_bytes < self.packed_bytes {
            return Err(format!(
                "resident row has {row_bytes} bytes, needs at least {}",
                self.packed_bytes
            ));
        }
        if offset % 4 != 0 {
            return Err("resident row offset must be four-byte aligned".to_string());
        }
        let end = offset
            .checked_add(self.packed_bytes)
            .ok_or("resident row offset overflow")?;
        if end > rows.length() as usize {
            return Err("resident row exceeds the search model arena".to_string());
        }
        self.packed = rows.to_owned();
        self.packed_offset = offset as u64;
        self.device_view = true;
        Ok(())
    }

    fn set_packed(&self, encoder: &metal::ComputeCommandEncoderRef, index: u64) {
        encoder.set_buffer(index, Some(&self.packed), self.packed_offset);
    }

    fn seed(&self, value: u64) -> MetalSeed {
        if self.device_view {
            MetalSeed::disabled()
        } else {
            MetalSeed::from(value)
        }
    }

    pub fn upload_hidden(&mut self, hidden: &[u16]) -> Result<(), String> {
        if hidden.len() != self.memory.hidden_elements {
            return Err(format!(
                "hidden input has {} elements, expected {}",
                hidden.len(),
                self.memory.hidden_elements
            ));
        }
        unsafe {
            std::ptr::copy_nonoverlapping(
                hidden.as_ptr(),
                self.hidden.contents().cast::<u16>(),
                hidden.len(),
            );
        }
        Ok(())
    }

    pub fn upload_norms(&mut self, attention: &[u16], moe: &[u16]) -> Result<(), String> {
        let width = usize::try_from(self.params.hidden_width).unwrap();
        if attention.len() != width || moe.len() != width {
            return Err(format!("decoder norm vectors need {width} elements"));
        }
        unsafe {
            std::ptr::copy_nonoverlapping(
                attention.as_ptr(),
                self.attention_norm.contents().cast(),
                width,
            );
            std::ptr::copy_nonoverlapping(moe.as_ptr(), self.moe_norm.contents().cast(), width);
        }
        Ok(())
    }

    /// Clear the persistent KDA memory before prefill or a new decode stream.
    pub fn reset_memory(&mut self) {
        unsafe {
            std::ptr::write_bytes(
                self.recurrence_state.contents().cast::<f32>(),
                0,
                self.memory.recurrence_state_elements,
            );
        }
    }

    pub fn reset_state(&mut self) {
        self.reset_memory();
        let elements = usize::try_from(self.kda_params.batch).unwrap()
            * 3
            * usize::try_from(self.kda_params.heads).unwrap()
            * usize::try_from(self.kda_params.key_width).unwrap()
            * 3;
        unsafe {
            std::ptr::write_bytes(self.qkv_conv_history.contents().cast::<u16>(), 0, elements);
        }
    }

    /// Upload a single token's already-projected KDA inputs.
    ///
    /// This narrow entry point is also the numerical seam used to validate
    /// the persistent recurrence independently of projection and MoE kernels.
    pub fn kda_inputs(
        &mut self,
        query: &[u16],
        key: &[u16],
        value: &[u16],
        gate: &[f32],
        beta: &[f32],
    ) -> Result<(), String> {
        let qkv = usize::try_from(self.kda_params.batch).unwrap()
            * usize::try_from(self.kda_params.heads).unwrap()
            * usize::try_from(self.kda_params.key_width).unwrap();
        let values = usize::try_from(self.kda_params.batch).unwrap()
            * usize::try_from(self.kda_params.heads).unwrap()
            * usize::try_from(self.kda_params.value_width).unwrap();
        let rates = usize::try_from(self.kda_params.batch).unwrap()
            * usize::try_from(self.kda_params.heads).unwrap();
        if query.len() != qkv || key.len() != qkv || gate.len() != qkv {
            return Err(format!(
                "single-token query, key, and gate need {qkv} elements"
            ));
        }
        if value.len() != values {
            return Err(format!("single-token value needs {values} elements"));
        }
        if beta.len() != rates {
            return Err(format!("single-token beta needs {rates} elements"));
        }
        unsafe {
            std::ptr::copy_nonoverlapping(query.as_ptr(), self.query.contents().cast(), qkv);
            std::ptr::copy_nonoverlapping(key.as_ptr(), self.key.contents().cast(), qkv);
            std::ptr::copy_nonoverlapping(value.as_ptr(), self.value.contents().cast(), values);
            std::ptr::copy_nonoverlapping(gate.as_ptr(), self.gate.contents().cast(), qkv);
            std::ptr::copy_nonoverlapping(beta.as_ptr(), self.beta.contents().cast(), rates);
        }
        Ok(())
    }

    /// Advance persistent KDA memory by one token and leave the output on GPU.
    pub fn kda_step(&mut self) -> Result<(), String> {
        let params = MetalKdaParams {
            length: 1,
            ..self.kda_params
        };
        let columns =
            u64::from(params.batch) * u64::from(params.heads) * u64::from(params.value_width);
        let command = self.runtime.queue.new_command_buffer();
        let encoder = command.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(self.pipeline("kda_step")?);
        encoder.set_buffer(0, Some(&self.query), 0);
        encoder.set_buffer(1, Some(&self.key), 0);
        encoder.set_buffer(2, Some(&self.value), 0);
        encoder.set_buffer(3, Some(&self.gate), 0);
        encoder.set_buffer(4, Some(&self.beta), 0);
        encoder.set_buffer(5, Some(&self.recurrence_state), 0);
        encoder.set_buffer(6, Some(&self.post_kda), 0);
        set_bytes(encoder, 7, &params);
        encoder.dispatch_threads(thread_group(columns), thread_group(256));
        encoder.end_encoding();
        command.commit();
        command.wait_until_completed();
        Ok(())
    }

    fn encode_projection(
        &self,
        encoder: &metal::ComputeCommandEncoderRef,
        input: &Buffer,
        output: &Buffer,
        linear: &Buffer,
        seed: &MetalSeed,
        threads: u64,
    ) -> Result<(), String> {
        encoder.set_compute_pipeline_state(self.pipeline("decoder_project_packed_simd")?);
        encoder.set_buffer(0, Some(input), 0);
        self.set_packed(encoder, 1);
        encoder.set_buffer(2, Some(&self.scales), 0);
        encoder.set_buffer(3, Some(&self.biases), 0);
        encoder.set_buffer(4, Some(output), 0);
        encoder.set_buffer(5, Some(linear), 0);
        set_bytes(encoder, 6, seed);
        set_bytes(encoder, 7, &self.params);
        encoder.dispatch_threads(thread_group(threads * 32), thread_group(256));
        Ok(())
    }

    pub fn prepare_candidate(&mut self, seed: u64) -> Result<(), String> {
        let command = self.runtime.queue.new_command_buffer();
        let encoder = command.new_compute_command_encoder();
        encode_candidate(self, encoder, seed)?;
        encoder.end_encoding();
        command.commit();
        command.wait_until_completed();
        Ok(())
    }

    pub fn decode_layer(&mut self, seed: u64) -> Result<(), String> {
        let command = self.runtime.queue.new_command_buffer();
        let encoder = command.new_compute_command_encoder();
        encode_layer(self, encoder, seed)?;
        encoder.end_encoding();
        command.commit();
        command.wait_until_completed();
        Ok(())
    }

    pub fn recurrence_state(&self) -> Vec<f32> {
        unsafe {
            std::slice::from_raw_parts(
                self.recurrence_state.contents().cast::<f32>(),
                self.memory.recurrence_state_elements,
            )
            .to_vec()
        }
    }

    pub fn kda_output(&self) -> Vec<u16> {
        let elements = usize::try_from(self.kda_params.batch).unwrap()
            * usize::try_from(self.kda_params.heads).unwrap()
            * usize::try_from(self.kda_params.value_width).unwrap();
        unsafe {
            std::slice::from_raw_parts(self.post_kda.contents().cast::<u16>(), elements).to_vec()
        }
    }

    /// Run the first resident operation in a decoder layer. The normalization
    /// vector is uploaded once per layer; hidden state never leaves Metal.
    pub fn attention_norm(
        &mut self,
        norm: &[u16],
        batch: usize,
        sequence_length: usize,
        epsilon: f32,
    ) -> Result<(), String> {
        let norm_buffer = self.attention_norm.to_owned();
        encode_norm(
            self,
            "decoder_rms_norm",
            norm,
            &norm_buffer,
            batch,
            sequence_length,
            epsilon,
        )
    }

    pub fn moe_norm(
        &mut self,
        norm: &[u16],
        batch: usize,
        sequence_length: usize,
        epsilon: f32,
    ) -> Result<(), String> {
        let norm_buffer = self.moe_norm.to_owned();
        encode_norm(
            self,
            "decoder_rms_norm",
            norm,
            &norm_buffer,
            batch,
            sequence_length,
            epsilon,
        )
    }

    /// Route and evaluate only the selected MoE experts. All operations are
    /// encoded in one command buffer; neither indices nor activations return
    /// to the host.
    pub fn moe(&mut self, seed: u64) -> Result<(), String> {
        let seed = self.seed(seed);
        let tokens = u64::from(self.params.batch) * u64::from(self.params.length);
        let expert_activations =
            tokens * u64::from(self.params.top_k) * u64::from(self.params.expert_width);
        let hidden = tokens * u64::from(self.params.hidden_width);
        let command = self.runtime.queue.new_command_buffer();
        let encoder = command.new_compute_command_encoder();
        encode_moe(self, encoder, &seed, tokens, expert_activations, hidden)?;
        encoder.end_encoding();
        command.commit();
        command.wait_until_completed();
        Ok(())
    }

    /// Execute a complete KDA attention update without returning control,
    /// gates, state, or activations to the host.
    pub fn kda(&mut self, seed: u64) -> Result<(), String> {
        if self.kda_params.length != 16_384 {
            return Err(
                "the current fused KDA projection kernel requires sequence length 16384"
                    .to_string(),
            );
        }
        let seed = self.seed(seed);
        let tokens = u64::from(self.kda_params.batch) * u64::from(self.kda_params.length);
        let qkv_threads = u64::from(self.kda_params.batch)
            * u64::from(self.kda_params.length)
            * u64::from(self.kda_params.heads)
            * u64::from(self.kda_params.key_width)
            * 3;
        let qkv_elements = qkv_threads / 3;
        let recurrence_threads = u64::from(self.kda_params.batch)
            * u64::from(self.kda_params.heads)
            * u64::from(self.kda_params.value_width);
        let hidden = tokens * u64::from(self.params.hidden_width);
        let control_width =
            2 * u64::from(self.kda_control_params.gate_rank) + u64::from(self.kda_params.heads);
        let command = self.runtime.queue.new_command_buffer();
        let encoder = command.new_compute_command_encoder();

        encode_gates(
            self,
            encoder,
            &seed,
            tokens,
            hidden,
            qkv_threads,
            qkv_elements,
            control_width,
        )?;
        encode_state(self, encoder, &seed, hidden, recurrence_threads)?;

        encoder.end_encoding();
        command.commit();
        command.wait_until_completed();
        Ok(())
    }
}
