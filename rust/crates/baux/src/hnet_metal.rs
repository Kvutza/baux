//! Low-level Apple Silicon Metal 4 dispatcher and hierarchical stage for H-Net.
//!
//! Baux's H-Net model consumes UTF-8 bytes directly, learns chunk boundaries
//! around its PISA/MHC core, and predicts a 256-way byte distribution after
//! EMA dechunking and the learned residual projection. It requires no BPE
//! tokenizer.
//!
//! The hierarchy uses FP32 arithmetic with FP16 stored weights and activations.
//! Component forward and gradient fixtures are pinned to upstream; the core is
//! Baux's, so this does not imply upstream model or checkpoint equivalence.
//!
//! # Academic & Open-Source Attribution
//!
//! This module implements the Apple GPU Metal execution and stage
//! wrapper for H-Net's dynamic chunking hierarchy and 256-way byte causal
//! language model:
//!
//! - **Paper**: Sukjun Hwang, Brandon Wang, and Albert Gu, *"Dynamic Chunking
//!   for End-to-End Hierarchical Sequence Modeling"*, arXiv:2507.07955 (2025).
//! - **Blog**: Cartesia AI, *"From Tokens to Bytes, Part 1"* (<https://www.cartesia.ai/blog/from-tokens-to-bytes-part-1>).
//! - **Upstream Implementation**: Goomba Lab / Cartesia AI, <https://github.com/goombalab/hnet>
//!   (MIT License), specifically:
//!   - `hnet/modules/dc.py` (`RoutingModule`, `ChunkLayer`, `DeChunkLayer`)
//!   - `hnet/models/hnet.py` (`STE`, `ste_func`, `HNet`, `residual_proj`,
//!     `pad_dimension`)
//!   - `hnet/models/mixer_seq.py` (`HNetForCausalLM`)
//!   - `hnet/utils/tokenizers.py` (`ByteTokenizer`)
//!   - `hnet/utils/train.py` (`load_balancing_loss`)

use std::sync::Arc;

use baux::semantics::hnet::{self, BYTE_VOCAB, EMA_EPS, HNetShape, RoutingModule, SteResidual};
#[cfg(test)]
use baux::semantics::hnet::{ByteTokenizer, ChunkLayer, DeChunkLayer, RoutingOutput, ratio_loss};
use metal::{Buffer, ComputePipelineState, MTLCommandBufferStatus, MTLDispatchType};

#[cfg(test)]
use crate::apple_gpu::slice_mut;
use crate::apple_gpu::{
    Runtime, barrier_buffers, bind_buffers, buffer_slice, encode_value, thread_group,
};

const SOURCE: &str = concat!(
    include_str!("hnet_wave.metal"),
    "\n",
    include_str!("hnet/bridge.metal"),
    "\n",
    include_str!("hnet/adjoint.metal"),
    "\n",
    include_str!("hnet/stage2.metal")
);

#[cfg(test)]
#[path = "hnet/bridge.rs"]
mod bridge;
#[cfg(test)]
#[path = "hnet/upstream.rs"]
mod upstream;

/// GPU output of a fused H-Net dynamic chunking stage on Apple Silicon Metal.
#[derive(Clone, Debug, PartialEq)]
pub struct GpuStageOut {
    pub boundary_prob: Vec<f32>,
    pub boundary_mask: Vec<bool>,
    pub selected_probs: Vec<f32>,
    pub chunks: Vec<f32>,
    pub plug_idx: Vec<u32>,
    pub num_chunks: Vec<usize>,
    pub dechunk_out: Vec<f32>,
    pub combined_out: Vec<f32>,
}

/// GPU output of the exact 256-byte readout and cross-entropy kernel.
#[derive(Clone, Debug, PartialEq)]
pub struct GpuReadoutOut {
    pub predictions: Vec<u32>,
    pub losses: Vec<f32>,
    pub mean_loss: f32,
}

struct StageBuffers {
    hidden: Buffer,
    q_proj: Buffer,
    k_proj: Buffer,
    mask: Buffer,
    prob: Buffer,
    bmask: Buffer,
    sel: Buffer,
    chunks: Buffer,
    plug: Buffer,
    nchunks: Buffer,
    res_w: Buffer,
    res_b: Buffer,
    dechunk: Buffer,
    combined: Buffer,
}

impl StageBuffers {
    fn allocate(
        runtime: &Runtime,
        router: &RoutingModule,
        ste: &SteResidual,
        hidden: &[f32],
        mask: &[bool],
        batch_size: usize,
        total: usize,
    ) -> Self {
        let d_model = router.d_model;
        let mask_u32: Vec<u32> = mask.iter().map(|&m| u32::from(m)).collect();
        Self {
            hidden: runtime.buffer_with(hidden),
            q_proj: runtime.buffer_with(&router.q_proj),
            k_proj: runtime.buffer_with(&router.k_proj),
            mask: runtime.buffer_with(&mask_u32),
            prob: runtime.buffer::<f32>(total),
            bmask: runtime.buffer::<u32>(total),
            sel: runtime.buffer::<f32>(total),
            chunks: runtime.buffer::<f32>(total * d_model),
            plug: runtime.buffer::<u32>(total),
            nchunks: runtime.buffer::<u32>(batch_size),
            res_w: runtime.buffer_with(&ste.weight),
            res_b: runtime.buffer_with(&ste.bias),
            dechunk: runtime.buffer::<f32>(total * d_model),
            combined: runtime.buffer::<f32>(total * d_model),
        }
    }

    fn read_output(
        &self,
        batch_size: usize,
        total: usize,
        d_model: usize,
    ) -> Result<GpuStageOut, String> {
        let boundary_prob = buffer_slice::<f32>(&self.prob, total)?.to_vec();
        let bmask_raw = buffer_slice::<u32>(&self.bmask, total)?;
        let boundary_mask = bmask_raw.iter().map(|&v| v != 0).collect();
        let selected_probs = buffer_slice::<f32>(&self.sel, total)?.to_vec();
        let chunks = buffer_slice::<f32>(&self.chunks, total * d_model)?.to_vec();
        let plug_idx = buffer_slice::<u32>(&self.plug, total)?.to_vec();
        let nchunks_raw = buffer_slice::<u32>(&self.nchunks, batch_size)?;
        let num_chunks = nchunks_raw.iter().map(|&v| v as usize).collect();
        let dechunk_out = buffer_slice::<f32>(&self.dechunk, total * d_model)?.to_vec();
        let combined_out = buffer_slice::<f32>(&self.combined, total * d_model)?.to_vec();
        Ok(GpuStageOut {
            boundary_prob,
            boundary_mask,
            selected_probs,
            chunks,
            plug_idx,
            num_chunks,
            dechunk_out,
            combined_out,
        })
    }
}

pub const EMA_PARTS: usize = 128;

pub struct HNet2Pipelines {
    pub prep_mean: ComputePipelineState,
    pub prep_byte: ComputePipelineState,
    pub prep_proj: ComputePipelineState,
    pub prep_readout: ComputePipelineState,
    pub s0_reduce: ComputePipelineState,
    pub s0_route: ComputePipelineState,
    pub s0_scan: ComputePipelineState,
    pub s0_pack: ComputePipelineState,
    pub s1_reduce: ComputePipelineState,
    pub s1_route: ComputePipelineState,
    pub s1_scan: ComputePipelineState,
    pub s1_pack: ComputePipelineState,
    pub tile_advance: ComputePipelineState,
    pub s2_load: ComputePipelineState,
    pub s2_store: ComputePipelineState,
    pub dec1_reduce: ComputePipelineState,
    pub dec1_scan: ComputePipelineState,
    pub dec0_reduce: ComputePipelineState,
    pub dec0_scan: ComputePipelineState,
    pub ema0_logits: ComputePipelineState,
    pub ro_reduce: ComputePipelineState,
    pub ro_scan: ComputePipelineState,
    pub ro_advance: ComputePipelineState,
}

impl HNet2Pipelines {
    fn compile(runtime: &Runtime, source: &str) -> Result<Self, String> {
        let c = |name| runtime.metal4_precise(source, "H-Net 2-stage waves", name, &[]);
        Ok(Self {
            prep_mean: c("hnet2_prep_mean")?,
            prep_byte: c("hnet2_prep_byte")?,
            prep_proj: c("hnet2_prep_proj")?,
            prep_readout: c("hnet2_prep_readout")?,
            s0_reduce: c("hnet2_stage0_reduce_tile")?,
            s0_route: c("hnet2_stage0_route_tile")?,
            s0_scan: c("hnet2_stage0_scan_tile")?,
            s0_pack: c("hnet2_stage0_pack_tile")?,
            s1_reduce: c("hnet2_stage1_reduce_tile")?,
            s1_route: c("hnet2_stage1_route_tile")?,
            s1_scan: c("hnet2_stage1_scan_tile")?,
            s1_pack: c("hnet2_stage1_pack_tile")?,
            tile_advance: c("hnet2_tile_advance")?,
            s2_load: c("hnet2_stage2_load")?,
            s2_store: c("hnet2_stage2_store")?,
            dec1_reduce: c("hnet2_dechunk1_reduce")?,
            dec1_scan: c("hnet2_dechunk1_scan")?,
            dec0_reduce: c("hnet2_dechunk0_reduce")?,
            dec0_scan: c("hnet2_dechunk0_scan")?,
            ema0_logits: c("hnet2_ema0_logits_tile")?,
            ro_reduce: c("hnet2_readout_reduce_tile")?,
            ro_scan: c("hnet2_readout_scan_tile")?,
            ro_advance: c("hnet2_readout_advance")?,
        })
    }
}

/// Compiled Metal 4 compute pipelines for H-Net dynamic chunking and byte
/// readout.
pub struct HNetWaves {
    runtime: Arc<Runtime>,
    route_pipe: ComputePipelineState,
    chunk_pipe: ComputePipelineState,
    dechunk_pipe: ComputePipelineState,
    readout_pipe: ComputePipelineState,
    bwd_pipe: ComputePipelineState,
    pub mhc_chunk: ComputePipelineState,
    pub mhc_chunk_scan: ComputePipelineState,
    pub mhc_chunk_pack: ComputePipelineState,
    pub mhc_project: ComputePipelineState,
    pub mhc_route: ComputePipelineState,
    pub mhc_route_state: ComputePipelineState,
    pub mhc_readout: ComputePipelineState,
    pub mhc_logits: ComputePipelineState,
    pub dechunk_grad: ComputePipelineState,
    pub route_grad: ComputePipelineState,
    pub input_grad: ComputePipelineState,
    pub weight_grad: ComputePipelineState,
    pub mhc_dechunk: ComputePipelineState,
    pub mhc_dechunk_reduce: ComputePipelineState,
    pub mhc_dechunk_scan: ComputePipelineState,
    pub mhc_dechunk_plug: ComputePipelineState,
    pub hnet2: HNet2Pipelines,
}

impl HNetWaves {
    pub fn new() -> Result<Self, String> {
        let runtime = Runtime::shared()?;
        Self::with_runtime(runtime)
    }

    pub fn with_runtime(runtime: Arc<Runtime>) -> Result<Self, String> {
        let source = hnet::metal_source(SOURCE);
        let compile =
            |name| runtime.metal4_precise(&source, "H-Net dynamic chunking waves", name, &[]);
        let hnet2 = HNet2Pipelines::compile(&runtime, &source)?;
        Ok(Self {
            route_pipe: compile("hnet_route")?,
            chunk_pipe: compile("hnet_chunk_scan")?,
            dechunk_pipe: compile("hnet_dechunk_ste")?,
            readout_pipe: compile("hnet_byte_readout")?,
            bwd_pipe: compile("hnet_ste_bwd")?,
            mhc_chunk: compile("hnet_mhc_chunk")?,
            mhc_chunk_scan: compile("hnet_mhc_chunk_scan")?,
            mhc_chunk_pack: compile("hnet_mhc_chunk_pack")?,
            mhc_project: compile("hnet_mhc_project")?,
            mhc_route: compile("hnet_mhc_route")?,
            mhc_route_state: compile("hnet_mhc_route_state")?,
            mhc_readout: compile("hnet_mhc_readout")?,
            mhc_logits: compile("hnet_mhc_logits")?,
            dechunk_grad: compile("hnet_dechunk_grad")?,
            route_grad: compile("hnet_route_grad")?,
            input_grad: compile("hnet_input_grad")?,
            weight_grad: compile("hnet_weight_grad")?,
            mhc_dechunk: compile("hnet_mhc_dechunk")?,
            mhc_dechunk_reduce: compile("hnet_mhc_dechunk_reduce")?,
            mhc_dechunk_scan: compile("hnet_mhc_dechunk_scan")?,
            mhc_dechunk_plug: compile("hnet_mhc_dechunk_plug")?,
            hnet2,
            runtime,
        })
    }

    pub fn run_stage(
        &self,
        router: &RoutingModule,
        ste: &SteResidual,
        hidden: &[f32],
        mask: &[bool],
        batch_size: usize,
        seq_len: usize,
    ) -> Result<GpuStageOut, String> {
        let d_model = router.d_model;
        let total = batch_size * seq_len;
        if batch_size == 0
            || seq_len == 0
            || d_model == 0
            || ste.d_model != d_model
            || hidden.len() != total * d_model
            || mask.len() != total
        {
            return Err("HNetWaves::run_stage dimension mismatch".into());
        }
        let shape = HNetShape {
            batch_size: batch_size as u32,
            seq_len: seq_len as u32,
            d_model: d_model as u32,
            max_chunks: seq_len as u32,
            vocab_size: BYTE_VOCAB as u32,
            eps: EMA_EPS,
            target_ratio: 2.0,
            score: 1,
        };
        let bufs =
            StageBuffers::allocate(&self.runtime, router, ste, hidden, mask, batch_size, total);
        let command = self.runtime.queue.new_command_buffer();
        let encoder = command.compute_command_encoder_with_dispatch_type(MTLDispatchType::Serial);

        encoder.set_compute_pipeline_state(&self.route_pipe);
        bind_buffers(
            encoder,
            &[
                (&bufs.hidden, 0),
                (&bufs.q_proj, 0),
                (&bufs.k_proj, 0),
                (&bufs.mask, 0),
                (&bufs.prob, 0),
                (&bufs.bmask, 0),
                (&bufs.sel, 0),
            ],
        );
        encode_value(encoder, 7, &shape);
        encoder.dispatch_thread_groups(thread_group(total as u64), thread_group(32));
        barrier_buffers(encoder);

        encoder.set_compute_pipeline_state(&self.chunk_pipe);
        bind_buffers(
            encoder,
            &[
                (&bufs.hidden, 0),
                (&bufs.bmask, 0),
                (&bufs.chunks, 0),
                (&bufs.plug, 0),
                (&bufs.nchunks, 0),
            ],
        );
        encode_value(encoder, 5, &shape);
        encoder.dispatch_thread_groups(thread_group(batch_size as u64), thread_group(32));
        barrier_buffers(encoder);

        encoder.set_compute_pipeline_state(&self.dechunk_pipe);
        bind_buffers(
            encoder,
            &[
                (&bufs.chunks, 0),
                (&bufs.bmask, 0),
                (&bufs.prob, 0),
                (&bufs.hidden, 0),
                (&bufs.res_w, 0),
                (&bufs.res_b, 0),
                (&bufs.sel, 0),
                (&bufs.dechunk, 0),
                (&bufs.combined, 0),
            ],
        );
        encode_value(encoder, 9, &shape);
        let channels = (batch_size * d_model) as u64;
        encoder.dispatch_threads(thread_group(channels), thread_group(64_u64.min(channels)));
        encoder.end_encoding();

        command.commit();
        command.wait_until_completed();
        if command.status() != MTLCommandBufferStatus::Completed {
            return Err("HNetWaves::run_stage command buffer failed".into());
        }
        bufs.read_output(batch_size, total, d_model)
    }

    pub fn run_readout(
        &self,
        hidden: &[f32],
        lm_head: &[f32],
        targets: &[u8],
        batch_size: usize,
        seq_len: usize,
        d_model: usize,
    ) -> Result<GpuReadoutOut, String> {
        let total = batch_size * seq_len;
        if total == 0
            || d_model == 0
            || hidden.len() != total * d_model
            || lm_head.len() != BYTE_VOCAB * d_model
            || targets.len() != total
        {
            return Err("HNetWaves::run_readout dimension mismatch".into());
        }
        let shape = HNetShape {
            batch_size: batch_size as u32,
            seq_len: seq_len as u32,
            d_model: d_model as u32,
            max_chunks: seq_len as u32,
            vocab_size: BYTE_VOCAB as u32,
            eps: EMA_EPS,
            target_ratio: 2.0,
            score: 1,
        };
        let target_u32: Vec<u32> = targets.iter().map(|&b| u32::from(b)).collect();
        let buf_hidden = self.runtime.buffer_with(hidden);
        let buf_head = self.runtime.buffer_with(lm_head);
        let buf_targets = self.runtime.buffer_with(&target_u32);
        let buf_preds = self.runtime.buffer::<u32>(total);
        let buf_losses = self.runtime.buffer::<f32>(total);

        let command = self.runtime.queue.new_command_buffer();
        let encoder = command.compute_command_encoder_with_dispatch_type(MTLDispatchType::Serial);
        encoder.set_compute_pipeline_state(&self.readout_pipe);
        bind_buffers(
            encoder,
            &[
                (&buf_hidden, 0),
                (&buf_head, 0),
                (&buf_targets, 0),
                (&buf_preds, 0),
                (&buf_losses, 0),
            ],
        );
        encode_value(encoder, 5, &shape);
        encoder.dispatch_thread_groups(thread_group(total as u64), thread_group(32));
        encoder.end_encoding();

        command.commit();
        command.wait_until_completed();
        if command.status() != MTLCommandBufferStatus::Completed {
            return Err("HNetWaves::run_readout command buffer failed".into());
        }

        let predictions = buffer_slice::<u32>(&buf_preds, total)?.to_vec();
        let losses = buffer_slice::<f32>(&buf_losses, total)?.to_vec();
        let mean_loss = losses.iter().sum::<f32>() / (total as f32);
        Ok(GpuReadoutOut {
            predictions,
            losses,
            mean_loss,
        })
    }

    pub fn backward_ste(
        &self,
        grad_combined: &[f32],
        dechunk_out: &[f32],
        boundary_prob: &[f32],
        batch_size: usize,
        seq_len: usize,
        d_model: usize,
    ) -> Result<Vec<f32>, String> {
        let total = batch_size * seq_len;
        if total == 0
            || d_model == 0
            || grad_combined.len() != total * d_model
            || dechunk_out.len() != total * d_model
            || boundary_prob.len() != total
        {
            return Err("HNetWaves::backward_ste dimension mismatch".into());
        }
        let shape = HNetShape {
            batch_size: batch_size as u32,
            seq_len: seq_len as u32,
            d_model: d_model as u32,
            max_chunks: seq_len as u32,
            vocab_size: BYTE_VOCAB as u32,
            eps: EMA_EPS,
            target_ratio: 2.0,
            score: 1,
        };
        let buf_grad = self.runtime.buffer_with(grad_combined);
        let buf_dechunk = self.runtime.buffer_with(dechunk_out);
        let buf_prob = self.runtime.buffer_with(boundary_prob);
        let buf_gprob = self.runtime.buffer::<f32>(total);

        let command = self.runtime.queue.new_command_buffer();
        let encoder = command.compute_command_encoder_with_dispatch_type(MTLDispatchType::Serial);
        encoder.set_compute_pipeline_state(&self.bwd_pipe);
        bind_buffers(
            encoder,
            &[
                (&buf_grad, 0),
                (&buf_dechunk, 0),
                (&buf_prob, 0),
                (&buf_gprob, 0),
            ],
        );
        encode_value(encoder, 4, &shape);
        encoder.dispatch_thread_groups(thread_group(total as u64), thread_group(32));
        encoder.end_encoding();

        command.commit();
        command.wait_until_completed();
        if command.status() != MTLCommandBufferStatus::Completed {
            return Err("HNetWaves::backward_ste command buffer failed".into());
        }
        Ok(buffer_slice::<f32>(&buf_gprob, total)?.to_vec())
    }
}

/// Dimension padding helper (`hnet/models/hnet.py:HNet.pad_dimension`).
pub fn pad_dimension(
    hidden: &[f32],
    tokens: usize,
    in_dim: usize,
    out_dim: usize,
    pad_vec: &[f32],
) -> Result<Vec<f32>, String> {
    if out_dim < in_dim
        || hidden.len() != tokens * in_dim
        || pad_vec.len() != out_dim.saturating_sub(in_dim)
    {
        return Err("pad_dimension shape mismatch".into());
    }
    if out_dim == in_dim {
        return Ok(hidden.to_vec());
    }
    let mut out = vec![0.0_f32; tokens * out_dim];
    for t in 0..tokens {
        let dst = &mut out[t * out_dim..(t + 1) * out_dim];
        dst[..in_dim].copy_from_slice(&hidden[t * in_dim..(t + 1) * in_dim]);
        dst[in_dim..].copy_from_slice(pad_vec);
    }
    Ok(out)
}

/// Dimension unpadding slice (`hidden_states[..., :D]` in
/// `hnet/models/hnet.py`).
pub fn unpad_dimension(
    hidden: &[f32],
    tokens: usize,
    padded_dim: usize,
    target_dim: usize,
) -> Result<Vec<f32>, String> {
    if target_dim > padded_dim || hidden.len() != tokens * padded_dim {
        return Err("unpad_dimension shape mismatch".into());
    }
    if target_dim == padded_dim {
        return Ok(hidden.to_vec());
    }
    let mut out = vec![0.0_f32; tokens * target_dim];
    for t in 0..tokens {
        let src = &hidden[t * padded_dim..t * padded_dim + target_dim];
        out[t * target_dim..(t + 1) * target_dim].copy_from_slice(src);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wave_compile() {
        let waves = HNetWaves::new().expect("HNet Metal 4 pipelines must compile with ABI");
        let router = RoutingModule::new(4);
        let ste = SteResidual::new(4);
        let hidden = vec![
            1.0, 0.0, 0.0, 0.0, // t=0: boundary (p=1.0)
            1.0, 0.0, 0.0, 0.0, // t=1: identical (p=0.0)
            -1.0, 0.0, 0.0, 0.0, // t=2: opposite (p=1.0)
            0.0, 1.0, 0.0, 0.0, // t=3: orthogonal (p=0.5)
        ];
        let mask = vec![true; 4];
        let gpu = waves
            .run_stage(&router, &ste, &hidden, &mask, 1, 4)
            .unwrap();

        let cpu_route = router.forward(&hidden, 1, 4, Some(&mask), None).unwrap();
        let cpu_chunk = ChunkLayer::new()
            .forward(&hidden, &cpu_route.boundary_mask, 1, 4, 4)
            .unwrap();
        let cpu_dechunk = DeChunkLayer::new(4)
            .forward(
                &cpu_chunk.chunks,
                cpu_chunk.max_chunks,
                &cpu_route.boundary_mask,
                &cpu_route.boundary_prob,
                1,
                4,
                None,
            )
            .unwrap();
        let cpu_combined = ste
            .forward(&cpu_dechunk, &hidden, &cpu_route.selected_probs)
            .unwrap();

        assert_eq!(gpu.boundary_mask, cpu_route.boundary_mask);
        assert_eq!(gpu.num_chunks, cpu_chunk.num_chunks);
        assert_eq!(gpu.plug_idx, vec![0, 0, 1, 1]);
        for i in 0..4 {
            assert!((gpu.boundary_prob[i] - cpu_route.boundary_prob[i]).abs() < 1e-6);
            assert!((gpu.selected_probs[i] - cpu_route.selected_probs[i]).abs() < 1e-6);
        }
        for i in 0..gpu.dechunk_out.len() {
            assert!((gpu.dechunk_out[i] - cpu_dechunk[i]).abs() < 1e-5);
            assert!((gpu.combined_out[i] - cpu_combined[i]).abs() < 1e-5);
        }
        let grad_comb = vec![1.0_f32; 16];
        let gpu_gprob = waves
            .backward_ste(&grad_comb, &gpu.dechunk_out, &gpu.boundary_prob, 1, 4, 4)
            .unwrap();
        let cpu_gprob = ste
            .backward_prob(&grad_comb, &cpu_dechunk, &cpu_route.boundary_prob)
            .unwrap();
        for i in 0..4 {
            assert!((gpu_gprob[i] - cpu_gprob[i]).abs() < 1e-5);
        }
        let r_loss = ratio_loss(
            &RoutingOutput {
                boundary_prob: gpu.boundary_prob,
                boundary_mask: gpu.boundary_mask,
                selected_probs: gpu.selected_probs,
                batch_size: 1,
                seq_len: 4,
            },
            Some(&mask),
            2.0,
        )
        .unwrap();
        assert!((r_loss - 1.0).abs() < 1e-6);
    }

    #[test]
    fn gpu_readout() {
        let waves = HNetWaves::new().unwrap();
        let tok = ByteTokenizer::new();
        let targets = tok.encode("ab", false, false);
        let d_model = 4;
        let hidden = vec![1.0, 2.0, -0.5, 0.25, -1.0, 0.5, 1.5, -0.25];
        let mut lm_head = vec![0.0_f32; BYTE_VOCAB * d_model];
        for v in 0..BYTE_VOCAB {
            for d in 0..d_model {
                lm_head[v * d_model + d] = ((v as f32) * 0.01 - 1.2) * ((d as f32) * 0.3 - 0.4);
            }
        }
        let gpu = waves
            .run_readout(&hidden, &lm_head, &targets, 1, 2, d_model)
            .unwrap();
        for t in 0..2 {
            let h = &hidden[t * d_model..(t + 1) * d_model];
            let mut logits = vec![0.0_f32; BYTE_VOCAB];
            let mut max_l = f32::NEG_INFINITY;
            let mut best_v = 0_usize;
            for v in 0..BYTE_VOCAB {
                let mut dot = 0.0_f32;
                for d in 0..d_model {
                    dot += lm_head[v * d_model + d] * h[d];
                }
                logits[v] = dot;
                if dot > max_l {
                    max_l = dot;
                    best_v = v;
                }
            }
            let sum_exp: f32 = logits.iter().map(|&l| (l - max_l).exp()).sum();
            let cpu_loss = (max_l + sum_exp.ln()) - logits[targets[t] as usize];
            assert_eq!(gpu.predictions[t], best_v as u32);
            assert!((gpu.losses[t] - cpu_loss).abs() < 1e-5);
        }
    }
}
