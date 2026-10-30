use super::*;
pub(super) use crate::apple_gpu::{gpu_interval, gpu_seconds, thread_group};
pub(super) use crate::fbt_mps::{Matmul, Matrix};
pub(super) use crate::resident_metal::{
    ControllerInfo, NoisyDecision, ParamBlock, ProgramAddress, Proposals, SearchState,
};
pub(super) use metal::MTLSize;
use rand::{Rng, SeedableRng};
use rand_distr::{Distribution, StandardNormal};

pub(super) fn filled(runtime: &Runtime, elements: usize, bits: u16) -> Buffer {
    let buffer = runtime.buffer::<u16>(elements);
    let values =
        unsafe { std::slice::from_raw_parts_mut(buffer.contents().cast::<u16>(), elements) };
    values.fill(bits);
    buffer
}

pub(super) fn patterned(runtime: &Runtime, elements: usize, seed: u32, exponent: u16) -> Buffer {
    let buffer = runtime.buffer::<u16>(elements);
    let values =
        unsafe { std::slice::from_raw_parts_mut(buffer.contents().cast::<u16>(), elements) };
    let mut state = seed;
    for value in values {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        let sign = ((state >> 31) as u16) << 15;
        *value = sign | exponent | ((state >> 6) as u16 & 0x03ff);
    }
    buffer
}

pub(super) fn half_bits(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exponent = ((bits >> 23) & 0xff) as i32;
    let fraction = bits & 0x7f_ffff;
    if exponent == 0xff {
        return sign | 0x7c00 | u16::from(fraction != 0);
    }
    let mut half_exponent = exponent - 127 + 15;
    if half_exponent >= 31 {
        return sign | 0x7c00;
    }
    if half_exponent <= 0 {
        if half_exponent < -10 {
            return sign;
        }
        let mantissa = fraction | 0x80_0000;
        let shift = (14 - half_exponent) as u32;
        let mut rounded = mantissa >> shift;
        let remainder = mantissa & ((1u32 << shift) - 1);
        let halfway = 1u32 << (shift - 1);
        if remainder > halfway || (remainder == halfway && rounded & 1 != 0) {
            rounded += 1;
        }
        return sign | rounded as u16;
    }
    let mut rounded = fraction >> 13;
    let remainder = fraction & 0x1fff;
    if remainder > 0x1000 || (remainder == 0x1000 && rounded & 1 != 0) {
        rounded += 1;
        if rounded == 0x400 {
            rounded = 0;
            half_exponent += 1;
            if half_exponent == 31 {
                return sign | 0x7c00;
            }
        }
    }
    sign | ((half_exponent as u16) << 10) | rounded as u16
}

pub(super) fn mhc_bias(runtime: &Runtime, layers: usize) -> Buffer {
    let mut values = vec![0u16; layers * MHC_COEFFICIENTS as usize];
    let input_bias = half_bits(-3.0f32.ln());
    for layer in 0..layers {
        let base = layer * MHC_COEFFICIENTS as usize;
        values[base..base + MHC_STREAMS as usize].fill(input_bias);
    }
    runtime.buffer_with(&values)
}

fn normal_buffer(runtime: &Runtime, elements: usize, seed: u64, scale: f32) -> Buffer {
    let buffer = runtime.buffer::<u16>(elements);
    let values =
        unsafe { std::slice::from_raw_parts_mut(buffer.contents().cast::<u16>(), elements) };
    let mut random = rand::rngs::StdRng::seed_from_u64(seed);
    for value in values {
        let sample: f32 = StandardNormal.sample(&mut random);
        *value = half_bits(sample * scale);
    }
    buffer
}

fn uniform_buffer(runtime: &Runtime, elements: usize, seed: u64, bound: f32) -> Buffer {
    let buffer = runtime.buffer::<u16>(elements);
    let values =
        unsafe { std::slice::from_raw_parts_mut(buffer.contents().cast::<u16>(), elements) };
    let mut random = rand::rngs::StdRng::seed_from_u64(seed);
    for value in values {
        *value = half_bits(random.gen_range(-bound..=bound));
    }
    buffer
}

pub(super) fn index_weights(runtime: &Runtime, _architecture: ResidualArchitecture) -> Buffer {
    let elements = 64 * 64;
    let mut values = vec![0u16; elements];
    for dim in 0..64 {
        values[dim * (elements / 64) + dim] = 0x3c00;
    }
    runtime.buffer_with(&values)
}

pub(super) fn factor(runtime: &Runtime, experts: u32, input: u32, output: u32) -> Buffer {
    let elements = (experts * input * output) as usize;
    let buffer = patterned(
        runtime,
        elements,
        0x9e37_79b9 ^ experts ^ (input << 8) ^ (output << 16),
        0x0800,
    );
    let values =
        unsafe { std::slice::from_raw_parts_mut(buffer.contents().cast::<u16>(), elements) };
    for expert in 0..experts as usize {
        for diagonal in 0..input.min(output) as usize {
            values[(expert * input as usize + diagonal) * output as usize + diagonal] = 0x2400;
        }
    }
    buffer
}

pub(super) fn rope_table(runtime: &Runtime, context: u32) -> Buffer {
    let base = if context > MAX_CONTEXT {
        let scale = f64::from(context) / f64::from(MAX_CONTEXT);
        ROPE_BASE * scale * scale
    } else {
        ROPE_BASE
    };
    let mut freqs = [0.0f64; ROPE_PAIRS as usize];
    for (pair, freq) in freqs.iter_mut().enumerate() {
        let exponent = -2.0 * (pair as f64) / f64::from(HEAD_DIM);
        *freq = base.powf(exponent);
    }
    let mut values = Vec::with_capacity((context * ROPE_PAIRS) as usize);
    for position in 0..context {
        let pos = f64::from(position);
        for &freq in &freqs {
            let angle = pos * freq;
            values.push([angle.cos() as f32, angle.sin() as f32]);
        }
    }
    runtime.buffer_with(&values)
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

pub(super) fn dispatch_on(
    encoder: &ComputeCommandEncoderRef,
    pipeline: &ComputePipelineState,
    buffers: &[&BufferRef],
    parameters: *const std::ffi::c_void,
    parameter_bytes: u64,
    threads: u64,
) {
    encoder.set_compute_pipeline_state(pipeline);
    for (index, buffer) in buffers.iter().enumerate() {
        encoder.set_buffer(index as u64, Some(buffer), 0);
    }
    encoder.set_bytes(buffers.len() as u64, parameter_bytes, parameters);
    let width = pipeline.max_total_threads_per_threadgroup().min(256);
    encoder.dispatch_threads(thread_group(threads), thread_group(width));
}

impl Pipelines {
    pub(super) fn moe_grad(&self) -> Result<&routing_grad::FineGrainedMoeGrad, String> {
        self.fine_grained_grad
            .get_or_init(|| {
                Runtime::shared()
                    .and_then(|runtime| routing_grad::FineGrainedMoeGrad::new(&runtime))
            })
            .as_ref()
            .map_err(Clone::clone)
    }

    pub(super) fn mhc_grad(&self) -> Result<&mhc_grad::Kernels, String> {
        self.mhc_grad
            .get_or_init(|| Runtime::shared().and_then(|runtime| mhc_grad::Kernels::new(&runtime)))
            .as_ref()
            .map_err(Clone::clone)
    }

    pub(super) fn train_grad(&self) -> Result<&train_grad::Kernels, String> {
        self.train_grad
            .get_or_init(|| {
                Runtime::shared().and_then(|runtime| train_grad::Kernels::new(&runtime))
            })
            .as_ref()
            .map_err(Clone::clone)
    }

    pub(super) fn for_trial(
        runtime: &Runtime,
        trial: Option<&crate::config::KernelTrial>,
    ) -> Result<Self, String> {
        let mut result = Self::new(runtime)?;
        if let Some(path) = trial.and_then(|trial| trial.moe.as_ref()) {
            let source = std::fs::read_to_string(path)
                .map_err(|error| format!("read MoE candidate {}: {error}", path.display()))?;
            result.fine_grained =
                routing::FineGrainedMoePipelines::with_source(runtime, Some(&source))?;
        }
        Ok(result)
    }

    pub(super) fn new(runtime: &Runtime) -> Result<Self, String> {
        let utility_source = include_str!("../fbt_moe.metal");
        let utility = |name| runtime.precise(utility_source, "grouped MoE layer probe", name);
        Ok(Self {
            fine_grained: routing::FineGrainedMoePipelines::new(runtime)?,
            fine_grained_grad: std::cell::OnceCell::new(),
            mhc_grad: std::cell::OnceCell::new(),
            train_grad: std::cell::OnceCell::new(),
            gate: utility("fbt_moe_balanced_gate")?,
            group: utility("fbt_moe_group_balanced")?,
            swiglu: utility("fbt_moe_swiglu")?,
            ungroup: utility("fbt_moe_ungroup_residual")?,
            feedback_fuse: utility("fbt_moe_feedback_fuse")?,
            cross_entropy: utility("fbt_moe_cross_entropy")?,
            readout_loss_reduce: utility("fbt_readout_loss_reduce")?,
            readout_proposal_reduce: utility("fbt_readout_proposal_reduce")?,
            sequence_loss: utility("fbt_moe_sequence_loss")?,
            embed: utility("fbt_moe_embed")?,
            denoise_embed: runtime.precise(
                include_str!("../fbt_denoise.metal"),
                "diffusion input",
                "denoise_embed",
            )?,
            denoise_reduce: runtime.precise(
                include_str!("../fbt_denoise.metal"),
                "diffusion sampling",
                "denoise_reduce",
            )?,
            rms: utility("fbt_moe_rms")?,
            residual: utility("fbt_moe_residual")?,
            residual_rms: utility("fbt_moe_residual_rms")?,
            feedback_fuse_rms: utility("fbt_moe_feedback_fuse_rms")?,
            mhc_replicate: utility("fbt_mhc_replicate")?,
            mhc_predict: utility("fbt_mhc_predict")?,
            mhc_predict_rows: utility("fbt_mhc_predict_rows")?,
            mhc_mix_rms: utility("fbt_mhc_mix_rms")?,
            mhc_update: utility("fbt_mhc_update_rows")?,
            combine_mhc: utility("combine_mhc")?,
            mhc_mean_rms: utility("fbt_mhc_mean_rms")?,
            rope: utility("fbt_moe_rope")?,
        })
    }
}

impl TensorOpsPipelines {
    pub(super) fn new(runtime: &Runtime) -> Result<Self, String> {
        let source = include_str!("../fbt_moe_tensorops.metal");
        Ok(Self {
            gate_up: runtime.precise_metal4(
                source,
                "Metal 4 TensorOps grouped MoE gate/up",
                "fbt_moe_tensorops_gate_up",
                &[],
            )?,
            gate_activation: runtime.precise_metal4(
                source,
                "Metal 4 gate/up activation",
                "gate_activation",
                &[],
            )?,
            down: runtime.precise_metal4(
                source,
                "Metal 4 TensorOps grouped MoE down",
                "fbt_moe_tensorops_down",
                &[],
            )?,
            materialize_gate_up: runtime.precise_metal4(
                source,
                "Metal 4 materialized Kronecker grouped MoE gate/up",
                "fbt_moe_materialize_kronecker_gate_up",
                &[],
            )?,
            materialize_down: runtime.precise_metal4(
                source,
                "Metal 4 materialized Kronecker grouped MoE down",
                "fbt_moe_materialize_kronecker_down",
                &[],
            )?,
            qkv: runtime.precise_metal4(
                source,
                "Metal 4 TensorOps fused QKV projection",
                "fbt_model_tensorops_qkv",
                &[],
            )?,
            output_projection: runtime.precise_metal4(
                source,
                "Metal 4 TensorOps attention output projection",
                "fbt_model_tensorops_output_projection",
                &[],
            )?,
            qkv_wide: runtime.precise_metal4(
                source,
                "Metal 4 TensorOps wide QKV projection",
                "fbt_model_tensorops_qkv_wide",
                &[],
            )?,
            qkv_rope: runtime.precise_metal4(source, "QKV with RoPE epilogue", "qkv_rope", &[])?,
            output_projection_wide: runtime.precise_metal4(
                source,
                "Metal 4 TensorOps wide attention output projection",
                "fbt_model_tensorops_output_projection_wide",
                &[],
            )?,
            readout: runtime.precise_metal4(
                source,
                "Metal 4 TensorOps vocabulary readout",
                "fbt_model_tensorops_readout",
                &[],
            )?,
            readout_loss_tiles: runtime.precise_metal4(
                include_str!("../fbt_loss_tensorops.metal"),
                "Metal 4 exact full-vocabulary loss tiles",
                "fbt_readout_loss_tiles",
                &[],
            )?,
            readout_proposal_tiles: runtime.precise_metal4(
                include_str!("../fbt_loss_tensorops.metal"),
                "Metal 4 full-vocabulary proposal tiles",
                "fbt_readout_proposal_tiles",
                &[],
            )?,
            denoise_tiles: runtime.precise_metal4(
                &format!(
                    "#define DENOISE\n{}",
                    include_str!("../fbt_loss_tensorops.metal")
                ),
                "diffusion readout",
                "fbt_readout_proposal_tiles",
                &[],
            )?,
        })
    }
}

pub(super) fn synthetic_buffer(runtime: &Runtime, shift: u32) -> Buffer {
    runtime.buffer_with(
        &(0..ROWS)
            .map(|row| (row.wrapping_mul(17) + row / CONTEXT * 97 + shift) % VOCAB)
            .collect::<Vec<_>>(),
    )
}

pub(super) fn pretrain_batch(
    buffers: &Buffers,
    dataset: &crate::pretrain_data::PretrainDataset,
    batch: u32,
) -> Result<(), String> {
    let tokens = dataset.batch(batch)?;
    let input = unsafe {
        std::slice::from_raw_parts_mut(buffers.tokens.contents().cast::<u32>(), ROWS as usize)
    };
    let labels = unsafe {
        std::slice::from_raw_parts_mut(buffers.labels.contents().cast::<u32>(), ROWS as usize)
    };
    let score_mask = unsafe {
        std::slice::from_raw_parts_mut(buffers.score_mask.contents().cast::<u8>(), ROWS as usize)
    };
    for sequence in 0..BATCH as usize {
        let start = sequence * CONTEXT as usize;
        let end = start + CONTEXT as usize;
        input[start..end]
            .iter_mut()
            .zip(&tokens[start..end])
            .for_each(|(output, &token)| *output = token);
        for position in 0..CONTEXT as usize - 1 {
            labels[start + position] = tokens[start + position + 1];
            score_mask[start + position] = 1;
        }
        labels[end - 1] = 0;
        score_mask[end - 1] = 0;
    }
    Ok(())
}

impl CandidateWeights {
    pub(super) fn flatten(&self, runtime: &Runtime) -> Result<Buffer, String> {
        let bytes = self
            .tensors()
            .iter()
            .try_fold(0u64, |sum, (_, tensor, _)| sum.checked_add(tensor.length()))
            .ok_or("model byte count overflow")?;
        let output = runtime
            .device
            .new_buffer(bytes, metal::MTLResourceOptions::StorageModeShared);
        let command = runtime.queue.new_command_buffer();
        let encoder = command.new_blit_command_encoder();
        let mut offset = 0;
        for (_, tensor, _) in self.tensors() {
            encoder.copy_from_buffer(tensor, 0, &output, offset, tensor.length());
            offset += tensor.length();
        }
        encoder.end_encoding();
        super::complete(command)?;
        Ok(output)
    }

    pub(super) fn new(runtime: &Runtime) -> Self {
        Self::seeded(runtime, None)
    }

    pub(super) fn seeded(runtime: &Runtime, seed: Option<u64>) -> Self {
        Self::seeded_for(runtime, seed, ResidualArchitecture::Legacy)
    }

    pub(super) fn seeded_for(
        runtime: &Runtime,
        seed: Option<u64>,
        architecture: ResidualArchitecture,
    ) -> Self {
        let family = |domain: u32| {
            seed.map_or(domain, |seed| {
                crate::hash::splitmix64(seed ^ u64::from(domain)) as u32
            })
        };
        let layers = MODEL_LAYERS as usize;
        let experts = routing::MOE_EXPERTS as usize;
        let width = WIDTH as usize;
        let expert_width = routing::ROUTED_WIDTH as usize;
        let gate_up = routing::MOE_GATEUP as usize;
        let qkv_width = QKV_WIDTH as usize;
        let vocab = VOCAB as usize;
        let router_elements = layers * width * routing::ROUTED_EXPERTS as usize;
        let qkv_elements = layers * width * qkv_width;
        let output_elements = layers * width * width;
        let gate_up_elements = layers * experts * width * gate_up;
        let down_elements = layers * experts * expert_width * width;
        let norm_elements = layers * width;
        let readout_elements = width * vocab;
        let feedback_elements = width * width;
        let mhc_predictor_elements = layers * MHC_INPUT as usize * MHC_COEFFICIENTS as usize;
        Self {
            architecture,
            initialization: crate::config::ModelInitialization::Patterned,
            router: patterned(runtime, router_elements, family(0x6a09_e667), 0x1800),
            qkv: patterned(runtime, qkv_elements, family(0xbb67_ae85), 0x2000),
            output: patterned(runtime, output_elements, family(0x3c6e_f372), 0x2000),
            gate_up: patterned(runtime, gate_up_elements, family(0xa54f_f53a), 0x2000),
            down: patterned(runtime, down_elements, family(0x510e_527f), 0x2000),
            attention_norm: filled(runtime, norm_elements, 0x3c00),
            ffn_norm: filled(runtime, norm_elements, 0x3c00),
            readout: patterned(runtime, readout_elements, family(0x9b05_688c), 0x1800),
            mask_embed: normal_buffer(runtime, width, u64::from(family(0x4d41_534b)), 0.02),
            index_query: index_weights(runtime, architecture),
            feedback_state: patterned(runtime, feedback_elements, family(0x1f83_d9ab), 0x2000),
            feedback_gate: patterned(runtime, feedback_elements, family(0x5be0_cd19), 0x2000),
            mhc_attention_predictor: patterned(
                runtime,
                mhc_predictor_elements,
                family(0x428a_2f98),
                0x1400,
            ),
            mhc_attention_bias: mhc_bias(runtime, layers),
            mhc_attention_control: filled(runtime, layers * 4, 0),
            mhc_moe_predictor: patterned(
                runtime,
                mhc_predictor_elements,
                family(0x7137_4491),
                0x1400,
            ),
            mhc_moe_bias: mhc_bias(runtime, layers),
            mhc_moe_control: filled(runtime, layers * 4, 0),
            final_norm: filled(runtime, width, 0x3c00),
        }
    }

    pub(super) fn initialized(
        runtime: &Runtime,
        seed: u64,
        initialization: crate::config::ModelInitialization,
        architecture: ResidualArchitecture,
    ) -> Result<Self, String> {
        use crate::config::ModelInitialization;
        Ok(match initialization {
            ModelInitialization::Patterned => Self::seeded_for(runtime, Some(seed), architecture),
            ModelInitialization::Gpt2 => Self::normal(runtime, seed, false, architecture),
            ModelInitialization::Megatron => Self::normal(runtime, seed, true, architecture),
            ModelInitialization::XavierUniform => Self::xavier(runtime, seed, architecture),
            ModelInitialization::FbtPisa1Mhc4 => Self::mhc4_weights(runtime, seed, architecture)?,
        })
    }

    fn normal(
        runtime: &Runtime,
        seed: u64,
        scaled: bool,
        architecture: ResidualArchitecture,
    ) -> Self {
        let layers = MODEL_LAYERS as usize;
        let experts = routing::MOE_EXPERTS as usize;
        let width = WIDTH as usize;
        let expert_width = routing::ROUTED_WIDTH as usize;
        let gate_up = routing::MOE_GATEUP as usize;
        let qkv_width = QKV_WIDTH as usize;
        let vocab = VOCAB as usize;
        let residual = if scaled {
            0.02 / (2.0 * MODEL_LAYERS as f32).sqrt()
        } else {
            0.02
        };
        let draw = |elements, domain, standard_deviation| {
            normal_buffer(
                runtime,
                elements,
                crate::hash::splitmix64(seed ^ domain),
                standard_deviation,
            )
        };
        Self {
            architecture,
            initialization: if scaled {
                crate::config::ModelInitialization::Megatron
            } else {
                crate::config::ModelInitialization::Gpt2
            },
            router: draw(
                layers * width * routing::ROUTED_EXPERTS as usize,
                0x6a09_e667,
                0.02,
            ),
            qkv: draw(layers * width * qkv_width, 0xbb67_ae85, 0.02),
            output: draw(layers * width * width, 0x3c6e_f372, residual),
            gate_up: draw(layers * experts * width * gate_up, 0xa54f_f53a, 0.02),
            down: draw(
                layers * experts * expert_width * width,
                0x510e_527f,
                residual,
            ),
            attention_norm: filled(runtime, layers * width, 0x3c00),
            ffn_norm: filled(runtime, layers * width, 0x3c00),
            readout: draw(width * vocab, 0x9b05_688c, 0.02),
            mask_embed: draw(width, 0x4d41_534b, 0.02),
            index_query: index_weights(runtime, architecture),
            feedback_state: draw(width * width, 0x1f83_d9ab, 0.02),
            feedback_gate: draw(width * width, 0x5be0_cd19, 0.02),
            mhc_attention_predictor: draw(
                layers * MHC_INPUT as usize * MHC_COEFFICIENTS as usize,
                0x428a_2f98,
                0.02,
            ),
            mhc_attention_bias: mhc_bias(runtime, layers),
            mhc_attention_control: filled(runtime, layers * 4, 0),
            mhc_moe_predictor: draw(
                layers * MHC_INPUT as usize * MHC_COEFFICIENTS as usize,
                0x7137_4491,
                0.02,
            ),
            mhc_moe_bias: mhc_bias(runtime, layers),
            mhc_moe_control: filled(runtime, layers * 4, 0),
            final_norm: filled(runtime, width, 0x3c00),
        }
    }

    fn xavier(runtime: &Runtime, seed: u64, architecture: ResidualArchitecture) -> Self {
        let layers = MODEL_LAYERS as usize;
        let experts = routing::MOE_EXPERTS as usize;
        let width = WIDTH as usize;
        let expert_width = routing::ROUTED_WIDTH as usize;
        let gate_up = routing::MOE_GATEUP as usize;
        let qkv_width = QKV_WIDTH as usize;
        let vocab = VOCAB as usize;
        let draw = |elements, domain, input, output| {
            let bound = (6.0f32 / (input + output) as f32).sqrt();
            uniform_buffer(
                runtime,
                elements,
                crate::hash::splitmix64(seed ^ domain),
                bound,
            )
        };
        Self {
            architecture,
            initialization: crate::config::ModelInitialization::XavierUniform,
            router: draw(
                layers * width * routing::ROUTED_EXPERTS as usize,
                0x6a09_e667,
                width,
                routing::ROUTED_EXPERTS as usize,
            ),
            qkv: draw(layers * width * qkv_width, 0xbb67_ae85, width, qkv_width),
            output: draw(layers * width * width, 0x3c6e_f372, width, width),
            gate_up: draw(
                layers * experts * width * gate_up,
                0xa54f_f53a,
                width,
                gate_up,
            ),
            down: draw(
                layers * experts * expert_width * width,
                0x510e_527f,
                expert_width,
                width,
            ),
            attention_norm: filled(runtime, layers * width, 0x3c00),
            ffn_norm: filled(runtime, layers * width, 0x3c00),
            readout: draw(width * vocab, 0x9b05_688c, width, vocab),
            mask_embed: draw(width, 0x4d41_534b, width, width),
            index_query: index_weights(runtime, architecture),
            feedback_state: draw(width * width, 0x1f83_d9ab, width, width),
            feedback_gate: draw(width * width, 0x5be0_cd19, width, width),
            mhc_attention_predictor: draw(
                layers * MHC_INPUT as usize * MHC_COEFFICIENTS as usize,
                0x428a_2f98,
                MHC_INPUT as usize,
                MHC_COEFFICIENTS as usize,
            ),
            mhc_attention_bias: mhc_bias(runtime, layers),
            mhc_attention_control: filled(runtime, layers * 4, 0),
            mhc_moe_predictor: draw(
                layers * MHC_INPUT as usize * MHC_COEFFICIENTS as usize,
                0x7137_4491,
                MHC_INPUT as usize,
                MHC_COEFFICIENTS as usize,
            ),
            mhc_moe_bias: mhc_bias(runtime, layers),
            mhc_moe_control: filled(runtime, layers * 4, 0),
            final_norm: filled(runtime, width, 0x3c00),
        }
    }

    pub(super) fn tensors(&self) -> Vec<(&'static str, &BufferRef, usize)> {
        let mut tensors: Vec<(&'static str, &BufferRef, usize)> = vec![
            (
                "router",
                &self.router,
                (WIDTH * routing::ROUTED_EXPERTS) as usize,
            ),
            ("qkv", &self.qkv, (WIDTH * QKV_WIDTH) as usize),
            ("attention_output", &self.output, (WIDTH * WIDTH) as usize),
            (
                "expert_gate_up",
                &self.gate_up,
                (WIDTH * routing::MOE_GATEUP) as usize,
            ),
            (
                "expert_down",
                &self.down,
                (routing::ROUTED_WIDTH * WIDTH) as usize,
            ),
            ("attention_norm", &self.attention_norm, WIDTH as usize),
            ("ffn_norm", &self.ffn_norm, WIDTH as usize),
            ("embedding_readout", &self.readout, (WIDTH * VOCAB) as usize),
        ];
        if self.architecture.is_multistream() {
            tensors.extend([
                (
                    "mhc_attention_predictor",
                    self.mhc_attention_predictor.as_ref(),
                    (MHC_INPUT * MHC_COEFFICIENTS) as usize,
                ),
                (
                    "mhc_attention_bias",
                    self.mhc_attention_bias.as_ref(),
                    MHC_COEFFICIENTS as usize,
                ),
                (
                    "mhc_attention_control",
                    self.mhc_attention_control.as_ref(),
                    4,
                ),
                (
                    "mhc_moe_predictor",
                    self.mhc_moe_predictor.as_ref(),
                    (MHC_INPUT * MHC_COEFFICIENTS) as usize,
                ),
                (
                    "mhc_moe_bias",
                    self.mhc_moe_bias.as_ref(),
                    MHC_COEFFICIENTS as usize,
                ),
                ("mhc_moe_control", self.mhc_moe_control.as_ref(), 4),
            ]);
        } else {
            tensors.extend([
                (
                    "feedback_state",
                    self.feedback_state.as_ref(),
                    (WIDTH * WIDTH) as usize,
                ),
                (
                    "feedback_gate",
                    self.feedback_gate.as_ref(),
                    (WIDTH * WIDTH) as usize,
                ),
            ]);
        }
        tensors.push(("final_norm", &self.final_norm, WIDTH as usize));
        if matches!(self.architecture, ResidualArchitecture::DiffusionMhc4) {
            tensors.push(("mask_embed", &self.mask_embed, WIDTH as usize));
        }
        if self.architecture == ResidualArchitecture::DiffusionMhc4 {
            tensors.push(("index_query", &self.index_query, 64 * 64));
        }
        tensors
    }
}

pub(super) fn canary(runtime: &Runtime, elements: usize) -> Buffer {
    runtime.buffer_with(&vec![0x7e00u16; elements + 64])
}
