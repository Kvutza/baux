use std::ffi::c_void;
use std::sync::Arc;

use metal::Buffer;

use super::{KdaMoeMetalExecutor, encode_candidate, encode_layer, encode_vector};
use crate::apple_gpu::{Runtime, thread_group};
use crate::forward_program::kda::KdaPackedLinear;
use crate::forward_program::{ResidentBoState, ResidentRound};
use crate::trials::{Trial, TrialEngine};

impl ResidentBoState {
    #[cfg(all(target_os = "macos", feature = "metal"))]
    pub fn bind_metal(
        &self,
        round: &ResidentRound,
        executor: &mut KdaMoeMetalExecutor,
    ) -> Result<(), String> {
        let (rows, offset, row_bytes) = self.metal_row(round)?;
        executor.bind_row(&rows, offset, row_bytes)
    }
}

impl KdaMoeMetalExecutor {
    pub fn bind_trial(&mut self, search: &TrialEngine, trial: Trial) -> Result<(), String> {
        let (rows, offset) = search.metal_row(trial)?;
        self.bind_row(&rows, offset, search.row_bytes())
    }
}

/// A complete decoder stack sharing one packed model arena and one hidden
/// buffer. All layers are encoded into one command buffer per generated token.
struct KdaMoeMetalHead {
    embedding_linear: Buffer,
    final_norm_linear: Buffer,
    final_norm: Buffer,
    logits: Buffer,
    token: Buffer,
    next_token: Buffer,
    params: MetalEmbeddingParams,
}

pub struct KdaMoeMetalModel {
    runtime: Arc<Runtime>,
    layers: Vec<KdaMoeMetalExecutor>,
    hidden: Buffer,
    hidden_elements: usize,
    head: Option<KdaMoeMetalHead>,
}

impl KdaMoeMetalModel {
    pub fn new(mut layers: Vec<KdaMoeMetalExecutor>) -> Result<Self, String> {
        let first = layers
            .first()
            .ok_or("resident KDA-MoE model has no layers")?;
        let hidden_elements = first.memory.hidden_elements;
        if first.params.batch != 1 || first.params.length != 1 {
            return Err("resident KDA-MoE model requires single-token layers".to_string());
        }
        if layers.iter().any(|layer| {
            layer.memory.hidden_elements != hidden_elements
                || layer.params.batch != 1
                || layer.params.length != 1
        }) {
            return Err("resident KDA-MoE layer hidden shapes do not match".to_string());
        }
        let runtime = Arc::clone(&first.runtime);
        let hidden = first.hidden.to_owned();
        for layer in &mut layers {
            layer.hidden = hidden.to_owned();
        }
        Ok(Self {
            runtime,
            layers,
            hidden,
            hidden_elements,
            head: None,
        })
    }

    pub fn attach_head(
        &mut self,
        embedding: KdaPackedLinear,
        final_norm: KdaPackedLinear,
        embedding_scale: f32,
    ) -> Result<(), String> {
        embedding.validate()?;
        final_norm.validate()?;
        if embedding.input_width != self.hidden_elements {
            return Err("embedding hidden width does not match decoder".to_string());
        }
        if final_norm.input_width != self.hidden_elements || final_norm.output_width != 1 {
            return Err("final norm must be a hidden-width vector".to_string());
        }
        if !embedding_scale.is_finite() {
            return Err("embedding scale must be finite".to_string());
        }
        self.head = Some(KdaMoeMetalHead {
            embedding_linear: self
                .runtime
                .buffer_with(&[MetalPackedLinear::from(embedding)]),
            final_norm_linear: self
                .runtime
                .buffer_with(&[MetalPackedLinear::from(final_norm)]),
            final_norm: self.runtime.buffer::<u16>(self.hidden_elements),
            logits: self.runtime.buffer::<u16>(embedding.output_width),
            token: self.runtime.buffer_with(&[0_u32]),
            next_token: self.runtime.buffer_with(&[0_u32]),
            params: MetalEmbeddingParams {
                vocab_size: u32::try_from(embedding.output_width)
                    .map_err(|_| "vocabulary exceeds u32")?,
                hidden_width: u32::try_from(self.hidden_elements)
                    .map_err(|_| "hidden width exceeds u32")?,
                embedding_scale,
            },
        });
        Ok(())
    }

    pub fn upload_hidden(&mut self, hidden: &[u16]) -> Result<(), String> {
        if hidden.len() != self.hidden_elements {
            return Err(format!(
                "model hidden input has {} elements, expected {}",
                hidden.len(),
                self.hidden_elements
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

    pub fn bind_row(
        &mut self,
        rows: &Buffer,
        offset: usize,
        row_bytes: usize,
    ) -> Result<(), String> {
        for layer in &mut self.layers {
            layer.bind_row(rows, offset, row_bytes)?;
        }
        Ok(())
    }

    pub fn bind_trial(&mut self, search: &TrialEngine, trial: Trial) -> Result<(), String> {
        let (rows, offset) = search.metal_row(trial)?;
        self.bind_row(&rows, offset, search.row_bytes())
    }

    pub fn reset_state(&mut self) {
        for layer in &mut self.layers {
            layer.reset_state();
        }
    }

    pub fn prepare_candidate(&mut self, seed: u64) -> Result<(), String> {
        let command = self.runtime.queue.new_command_buffer();
        let encoder = command.new_compute_command_encoder();
        for layer in &self.layers {
            encode_candidate(layer, encoder, seed)?;
        }
        if let Some(head) = &self.head {
            let first = &self.layers[0];
            let seed = first.seed(seed);
            encode_vector(
                first,
                encoder,
                "packed_dequantize_row_half",
                &head.final_norm,
                &head.final_norm_linear,
                &seed,
                u64::from(head.params.hidden_width),
            )?;
        }
        encoder.end_encoding();
        command.commit();
        command.wait_until_completed();
        Ok(())
    }

    pub fn decode(&mut self, seed: u64) -> Result<(), String> {
        let command = self.runtime.queue.new_command_buffer();
        let encoder = command.new_compute_command_encoder();
        for layer in &self.layers {
            encode_layer(layer, encoder, seed)?;
        }
        encoder.end_encoding();
        command.commit();
        command.wait_until_completed();
        Ok(())
    }

    pub fn decode_token(&mut self, token: u32, seed: u64) -> Result<u32, String> {
        let head = self
            .head
            .as_ref()
            .ok_or("resident KDA-MoE model has no causal head")?;
        if token >= head.params.vocab_size {
            return Err(format!("token {token} exceeds the model vocabulary"));
        }
        unsafe {
            head.token.contents().cast::<u32>().write(token);
        }
        let first = &self.layers[0];
        let perturbation = first.seed(seed);
        let command = self.runtime.queue.new_command_buffer();
        let encoder = command.new_compute_command_encoder();

        encoder.set_compute_pipeline_state(first.pipeline("packed_embedding_lookup")?);
        encoder.set_buffer(0, Some(&head.token), 0);
        first.set_packed(encoder, 1);
        encoder.set_buffer(2, Some(&first.scales), 0);
        encoder.set_buffer(3, Some(&first.biases), 0);
        encoder.set_buffer(4, Some(&self.hidden), 0);
        encoder.set_buffer(5, Some(&head.embedding_linear), 0);
        set_bytes(encoder, 6, &perturbation);
        set_bytes(encoder, 7, &head.params);
        encoder.dispatch_threads(
            thread_group(u64::from(head.params.hidden_width)),
            thread_group(256),
        );

        for layer in &self.layers {
            encode_layer(layer, encoder, seed)?;
        }

        encoder.set_compute_pipeline_state(first.pipeline("decoder_rms_norm")?);
        encoder.set_buffer(0, Some(&self.hidden), 0);
        encoder.set_buffer(1, Some(&head.final_norm), 0);
        encoder.set_buffer(2, Some(&first.normalized), 0);
        set_bytes(encoder, 3, &first.params);
        encoder.dispatch_threads(thread_group(1), thread_group(1));

        first.encode_projection(
            encoder,
            &first.normalized,
            &head.logits,
            &head.embedding_linear,
            &perturbation,
            u64::from(head.params.vocab_size),
        )?;
        encoder.set_compute_pipeline_state(first.pipeline("decoder_argmax")?);
        encoder.set_buffer(0, Some(&head.logits), 0);
        encoder.set_buffer(1, Some(&head.next_token), 0);
        set_bytes(encoder, 2, &head.params.vocab_size);
        encoder.dispatch_thread_groups(thread_group(1), thread_group(256));
        encoder.end_encoding();
        command.commit();
        command.wait_until_completed();
        Ok(unsafe { head.next_token.contents().cast::<u32>().read() })
    }

    pub fn logit_bits(&self) -> Result<Vec<u16>, String> {
        let head = self
            .head
            .as_ref()
            .ok_or("resident KDA-MoE model has no causal head")?;
        Ok(unsafe {
            std::slice::from_raw_parts(
                head.logits.contents().cast::<u16>(),
                usize::try_from(head.params.vocab_size).unwrap(),
            )
            .to_vec()
        })
    }

    pub fn generate(
        &mut self,
        prompt: &[u32],
        max_new_tokens: usize,
        seed: u64,
    ) -> Result<Vec<u32>, String> {
        let (&last, prefix) = prompt
            .split_last()
            .ok_or("generation prompt cannot be empty")?;
        for &token in prefix {
            self.decode_token(token, seed)?;
        }
        let mut next = self.decode_token(last, seed)?;
        let mut output = Vec::with_capacity(prompt.len() + max_new_tokens);
        output.extend_from_slice(prompt);
        for index in 0..max_new_tokens {
            output.push(next);
            if index + 1 < max_new_tokens {
                next = self.decode_token(next, seed)?;
            }
        }
        Ok(output)
    }

    pub fn hidden(&self) -> Vec<u16> {
        unsafe {
            std::slice::from_raw_parts(self.hidden.contents().cast::<u16>(), self.hidden_elements)
                .to_vec()
        }
    }

    pub fn layers_mut(&mut self) -> &mut [KdaMoeMetalExecutor] {
        &mut self.layers
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
pub(super) struct MetalDecoderParams {
    pub(super) batch: u32,
    pub(super) length: u32,
    pub(super) hidden_width: u32,
    pub(super) experts: u32,
    pub(super) top_k: u32,
    pub(super) expert_width: u32,
    pub(super) residual_scale: f32,
    pub(super) rms_epsilon: f32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub(super) struct MetalKdaParams {
    pub(super) batch: u32,
    pub(super) length: u32,
    pub(super) heads: u32,
    pub(super) key_width: u32,
    pub(super) value_width: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub(super) struct MetalKdaControlParams {
    pub(super) batch: u32,
    pub(super) length: u32,
    pub(super) heads: u32,
    pub(super) head_width: u32,
    pub(super) gate_rank: u32,
    pub(super) rms_epsilon: f32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct MetalEmbeddingParams {
    vocab_size: u32,
    hidden_width: u32,
    embedding_scale: f32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub(super) struct MetalPackedLinear {
    byte_offset: u32,
    scale_offset: u32,
    bias_offset: u32,
    input_width: u32,
    output_width: u32,
    bits: u32,
    group_size: u32,
    element_offset: u32,
    perturb_whole: u32,
    perturb_threshold: u32,
}

impl From<KdaPackedLinear> for MetalPackedLinear {
    fn from(value: KdaPackedLinear) -> Self {
        Self {
            byte_offset: u32::try_from(value.byte_offset).expect("packed byte offset exceeds u32"),
            scale_offset: u32::try_from(value.scale_offset)
                .expect("packed scale offset exceeds u32"),
            bias_offset: u32::try_from(value.bias_offset).expect("packed bias offset exceeds u32"),
            input_width: u32::try_from(value.input_width).expect("packed input width exceeds u32"),
            output_width: u32::try_from(value.output_width)
                .expect("packed output width exceeds u32"),
            bits: u32::from(value.bits),
            group_size: u32::try_from(value.group_size).expect("packed group size exceeds u32"),
            element_offset: u32::try_from(value.element_offset)
                .expect("packed element offset exceeds u32"),
            perturb_whole: value.perturb_whole,
            perturb_threshold: value.perturb_threshold,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
pub(super) struct MetalSeed {
    low: u32,
    high: u32,
    enabled: u32,
    _padding: u32,
}

impl From<u64> for MetalSeed {
    fn from(value: u64) -> Self {
        Self {
            low: value as u32,
            high: (value >> 32) as u32,
            enabled: 1,
            _padding: 0,
        }
    }
}

impl MetalSeed {
    pub(super) fn disabled() -> Self {
        Self {
            low: 0,
            high: 0,
            enabled: 0,
            _padding: 0,
        }
    }
}

pub(super) fn metal_linears(linears: &[KdaPackedLinear]) -> Vec<MetalPackedLinear> {
    linears
        .iter()
        .copied()
        .map(MetalPackedLinear::from)
        .collect()
}

pub(super) fn set_bytes<T>(encoder: &metal::ComputeCommandEncoderRef, slot: u64, value: &T) {
    encoder.set_bytes(
        slot,
        std::mem::size_of::<T>() as u64,
        (value as *const T).cast::<c_void>(),
    );
}
