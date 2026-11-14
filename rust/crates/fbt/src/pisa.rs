//! Fixed-shape PISA attention for the proposed 4K subsecond scorer.
//!
//! # Primary Citation
//! - Bohao Tang, Zhen Qin, Yuqi Pan, Zheng Li, and Pengfei Liu, *"Block Sparse
//!   Attention with Log-Linear Complexity"*, `arXiv:2609.31093` (2026).
//!
//! Note: PISA is an external prior-art attention architecture used as a
//! sequence-modeling testbed, not a novel BAUX paper contribution.

use crate::apple_gpu::{Runtime, gpu_seconds, thread_group};
use crate::context::probe::half as decode_half;
use metal::{
    Buffer, BufferRef, CommandBufferRef, ComputeCommandEncoderRef, ComputePipelineState,
    MTLCommandBufferStatus, MTLSize,
};
use std::time::Instant;

mod cache;
mod certify;
mod dirty;
mod fallback;
mod grad;
mod hierarchical;
pub(crate) mod parity;

pub(super) const CONTEXT: u32 = 4096;
pub(crate) const BATCH: u32 = 2;
pub(super) const ROWS: u32 = BATCH * CONTEXT;
pub(crate) const QUERY_HEADS: u32 = 8;
pub(crate) const HEAD_DIM: u32 = 64;
pub(crate) const QUERY_WIDTH: u32 = QUERY_HEADS * HEAD_DIM;
pub(super) const QKV_WIDTH: u32 = QUERY_WIDTH + 2 * HEAD_DIM;
pub(super) const BLOCK: u32 = 64;
pub(super) const LEAVES: u32 = CONTEXT / BLOCK;
pub(super) const NODES: u32 = 2 * LEAVES - 1;
pub(super) const SELECTED: u32 = 8;

#[derive(Debug, Clone, Copy)]
pub(crate) struct PisaProbe {
    pub pyramid_gpu_seconds: f64,
    pub selection_gpu_seconds: f64,
    pub attention_gpu_seconds: f64,
    pub layer_gpu_seconds: f64,
    pub layer_wall_seconds: f64,
    pub max_abs_error: f64,
}

pub(crate) type Pisa1Probe = PisaProbe;

pub(crate) struct Pipelines {
    pub(super) leaf_means: ComputePipelineState,
    pub(super) leaf_range: ComputePipelineState,
    pub(super) upper_means: ComputePipelineState,
    pub(super) upper_range: ComputePipelineState,
    pub(super) select: ComputePipelineState,
    pub(super) attention: ComputePipelineState,
    pub(super) select_attention: ComputePipelineState,
    pub(super) select_attention_q4: ComputePipelineState,
    pub(super) exact_attention: ComputePipelineState,
    pub(super) leaf_kv: ComputePipelineState,
    pub(super) upper_kv: ComputePipelineState,
    pub(super) hierarchical: ComputePipelineState,
    pub(super) full_clear: ComputePipelineState,
    pub(super) full_intermediate: ComputePipelineState,
    pub(super) full_tiles: ComputePipelineState,
    pub(super) full_scatter: ComputePipelineState,
    pub(super) full_leaf_q4: ComputePipelineState,
    pub(super) full_final: ComputePipelineState,
    pub(super) full_attention_q4: ComputePipelineState,
    pub(crate) grad: ComputePipelineState,
    pub(super) query_tile: u32,
}

pub(crate) struct ProbeOutputs {
    pub(super) oracle_output: Buffer,
    pub(crate) exact_output: Buffer,
}

pub(crate) struct Pisa {
    pub(crate) context: u32,
    pub(super) query_rows: u32,
    pub(crate) pipelines: Pipelines,
    pub(super) pyramid: Buffer,
    pub(crate) blocks: Buffer,
    pub(super) leaf_candidates: Buffer,
    pub(super) leaf_scores: Buffer,
    pub(super) leaf_counts: Buffer,
    pub(super) leaf_offsets: Buffer,
    pub(super) leaf_cursors: Buffer,
    pub(super) leaf_refs: Buffer,
    pub(super) leaf_tiles: Buffer,
    pub(super) leaf_tile_count: Buffer,
    pub(super) output: Buffer,
    pub(super) probe: std::cell::OnceCell<ProbeOutputs>,
    pub(super) cached: std::cell::OnceCell<Result<crate::context_metal::ContextKernels, String>>,
    pub(super) indexed: std::cell::OnceCell<Result<crate::context_metal::ContextKernels, String>>,
}

pub(crate) type Pisa1 = Pisa;

impl std::ops::Deref for Pisa {
    type Target = ProbeOutputs;

    fn deref(&self) -> &Self::Target {
        self.probe.get_or_init(|| {
            let runtime = Runtime::shared().expect("shared Metal runtime");
            let len = (self.query_rows * QUERY_WIDTH) as usize + 64;
            ProbeOutputs {
                oracle_output: runtime.buffer_with(&vec![0x7e00u16; len]),
                exact_output: runtime.buffer_with(&vec![0x7e00u16; len]),
            }
        })
    }
}

#[derive(Clone, Copy)]
pub(super) enum Stage {
    Pyramid,
    Selection,
    Attention,
}

pub(super) fn dispatch(
    command: &CommandBufferRef,
    pipeline: &ComputePipelineState,
    buffers: &[&BufferRef],
    groups: MTLSize,
    threads: u64,
) {
    let encoder = command.new_compute_command_encoder();
    dispatch_on(&encoder, pipeline, buffers, groups, threads);
    encoder.end_encoding();
}

pub(super) fn dispatch_on(
    encoder: &ComputeCommandEncoderRef,
    pipeline: &ComputePipelineState,
    buffers: &[&BufferRef],
    groups: MTLSize,
    threads: u64,
) {
    encoder.set_compute_pipeline_state(pipeline);
    for (index, buffer) in buffers.iter().enumerate() {
        encoder.set_buffer(index as u64, Some(buffer), 0);
    }
    encoder.dispatch_thread_groups(groups, thread_group(threads));
}

pub(super) fn complete(command: &CommandBufferRef) -> Result<f64, String> {
    command.commit();
    command.wait_until_completed();
    if command.status() != MTLCommandBufferStatus::Completed {
        return Err(format!("PISA-1 command failed: {:?}", command.status()));
    }
    gpu_seconds(command).ok_or("Metal did not report PISA-1 GPU timing".into())
}

impl Pipelines {
    fn new(runtime: &Runtime, context: u32, rows: u32) -> Result<Self, String> {
        let source_text = format!(
            "#define PISA_CONTEXT {context}\n#define PISA_ROWS {rows}\n{}\n{}",
            include_str!("pisa.metal"),
            include_str!("pisa/full.metal")
        );
        let source = source_text.as_str();
        let compile = |name| runtime.pipeline(source, "PISA-1 attention", name);
        let production_source = format!(
            "#define PISA_SKIP_IDENTITY_RESCALE\n#define PISA_NO_STAGE_SHARED_KV\n#define \
             PISA_QUERY_TILE 4\n{source}"
        );
        let full_attention_source = format!(
            "#define PISA_SKIP_IDENTITY_RESCALE\n#define PISA_NO_STAGE_SHARED_KV\n#define \
             PISA_QUERY_TILE 4\n#define PISA_REUSE\n{source}"
        );
        let grad_source = include_str!("pisa1_grad.metal");
        let result = Self {
            leaf_means: compile("fbt_pisa1_leaf_means")?,
            leaf_range: compile("fbt_pisa1_leaf_range")?,
            upper_means: compile("fbt_pisa1_upper_means")?,
            upper_range: compile("fbt_pisa1_upper_range")?,
            select: compile("fbt_pisa1_select")?,
            attention: compile("fbt_pisa1_attention")?,
            select_attention: compile("fbt_pisa1_select_attention")?,
            exact_attention: compile("fbt_pisa1_exact_attention")?,
            leaf_kv: compile("fbt_pisa1_leaf_kv_means")?,
            upper_kv: compile("pisa1_upperkv")?,
            hierarchical: compile("fbt_pisa1_hierarchical_attention")?,
            full_clear: compile("fbt_pisa_full_clear")?,
            full_intermediate: compile("fbt_pisa_full_intermediate")?,
            full_tiles: compile("fbt_pisa_full_tiles")?,
            full_scatter: compile("fbt_pisa_full_scatter")?,
            full_leaf_q4: compile("fbt_pisa_full_leaf_q4")?,
            full_final: compile("fbt_pisa_full_final")?,
            full_attention_q4: runtime.pipeline(
                &full_attention_source,
                "full PISA attention",
                "fbt_pisa1_select_attention_q4",
            )?,
            grad: runtime.precise(grad_source, "PISA-1 gradient", "fbt_pisa1_grad")?,
            select_attention_q4: runtime.pipeline(
                &production_source,
                "PISA-1 attention",
                "fbt_pisa1_select_attention_q4",
            )?,
            query_tile: 4,
        };
        if result.select_attention_q4.thread_execution_width() != 32
            || result
                .select_attention_q4
                .max_total_threads_per_threadgroup()
                < 256
            || result
                .select_attention_q4
                .static_threadgroup_memory_length()
                > runtime.device.max_threadgroup_memory_length()
        {
            return Err("PISA-1 attention exceeds this GPU's SIMD or threadgroup limits".into());
        }
        Ok(result)
    }
}

impl Pisa {
    pub(crate) fn for_trial(
        runtime: &Runtime,
        trial: Option<&crate::config::KernelTrial>,
    ) -> Result<Self, String> {
        let mut result = Self::new(runtime)?;
        if let Some(path) = trial.and_then(|trial| trial.pisa.as_ref()) {
            let source = std::fs::read_to_string(path)
                .map_err(|error| format!("read PISA candidate {}: {error}", path.display()))?;
            let source = format!("#define PISA_SKIP_IDENTITY_RESCALE\n{source}");
            let pipeline = runtime.metal4_pipeline(
                &source,
                "PISA kernel trial",
                "fbt_pisa1_select_attention_q4",
                &[],
            )?;
            if pipeline.thread_execution_width() != 32
                || pipeline.max_total_threads_per_threadgroup() < 128
                || pipeline.static_threadgroup_memory_length()
                    > runtime.device.max_threadgroup_memory_length()
            {
                return Err("PISA candidate exceeds the production launch resource limits".into());
            }
            result.pipelines.select_attention_q4 = pipeline;
        }
        Ok(result)
    }

    pub(crate) fn new(runtime: &Runtime) -> Result<Self, String> {
        Self::with_context(runtime, CONTEXT)
    }

    pub(crate) fn with_context(runtime: &Runtime, context: u32) -> Result<Self, String> {
        Self::with_rows(runtime, context, context.max(ROWS))
    }

    pub(crate) fn with_rows(
        runtime: &Runtime,
        context: u32,
        query_rows: u32,
    ) -> Result<Self, String> {
        if !(4096..=crate::context::MAX_CONTEXT).contains(&context) || !context.is_power_of_two() {
            return Err("PISA context must be a power of two in 4096..2097152".into());
        }
        let query_rows = query_rows.max(ROWS);
        let leaves = context / BLOCK;
        let leaf_tile_capacity = query_rows * 4 + 2 * leaves;
        Ok(Self {
            context,
            query_rows,
            pipelines: Pipelines::new(runtime, context, query_rows)?,
            pyramid: runtime.buffer::<u16>((BATCH * (2 * context / BLOCK - 1) * HEAD_DIM) as usize),
            blocks: runtime.buffer_with(&vec![u32::MAX; (query_rows * SELECTED) as usize + 64]),
            leaf_candidates: runtime.buffer::<u32>((query_rows * 16) as usize),
            leaf_scores: runtime.buffer::<f32>((query_rows * 16) as usize),
            leaf_counts: runtime.buffer::<u32>((2 * leaves) as usize),
            leaf_offsets: runtime.buffer::<u32>((2 * leaves) as usize),
            leaf_cursors: runtime.buffer::<u32>((2 * leaves) as usize),
            leaf_refs: runtime.buffer::<u32>((query_rows * 16) as usize),
            leaf_tiles: runtime.buffer::<u32>((leaf_tile_capacity * 4) as usize),
            leaf_tile_count: runtime.buffer::<u32>(3),
            output: runtime.buffer_with(&vec![0x7e00u16; (query_rows * QUERY_WIDTH) as usize + 64]),
            probe: std::cell::OnceCell::new(),
            cached: std::cell::OnceCell::new(),
            indexed: std::cell::OnceCell::new(),
        })
    }

    pub(crate) fn context(&self) -> u32 {
        self.context
    }

    pub(super) fn encode_pyramid(&self, command: &CommandBufferRef, qkv: &BufferRef) {
        dispatch(
            command,
            &self.pipelines.leaf_means,
            &[qkv, &self.pyramid],
            MTLSize {
                width: u64::from(LEAVES),
                height: u64::from(BATCH),
                depth: 1,
            },
            64,
        );
        dispatch(
            command,
            &self.pipelines.upper_means,
            &[&self.pyramid],
            thread_group(u64::from(BATCH)),
            64,
        );
    }

    pub(crate) fn pyramid_output(
        &self,
        encoder: &ComputeCommandEncoderRef,
        qkv: &BufferRef,
        pyramid: &BufferRef,
        rows: u32,
    ) {
        debug_assert_eq!(rows % self.context, 0);
        let batch = rows / self.context;
        dispatch_on(
            encoder,
            &self.pipelines.leaf_means,
            &[qkv, pyramid],
            MTLSize {
                width: u64::from(self.context / BLOCK),
                height: u64::from(batch),
                depth: 1,
            },
            64,
        );
        encoder.memory_barrier_with_resources(&[pyramid]);
        dispatch_on(
            encoder,
            &self.pipelines.upper_means,
            &[pyramid],
            thread_group(u64::from(batch)),
            64,
        );
        encoder.memory_barrier_with_resources(&[pyramid]);
    }

    pub(super) fn encode_selection(&self, command: &CommandBufferRef, qkv: &BufferRef) {
        dispatch(
            command,
            &self.pipelines.select,
            &[qkv, &self.pyramid, &self.blocks],
            thread_group(u64::from(ROWS)),
            32,
        );
    }

    pub(super) fn encode_oracle(&self, command: &CommandBufferRef, qkv: &BufferRef) {
        dispatch(
            command,
            &self.pipelines.select_attention,
            &[qkv, &self.pyramid, &self.blocks, &self.oracle_output],
            thread_group(u64::from(ROWS)),
            128,
        );
    }

    pub(super) fn exact_range(
        &self,
        command: &CommandBufferRef,
        qkv: &BufferRef,
        start: u32,
        rows: u32,
    ) {
        let range = [start, rows];
        let encoder = command.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.pipelines.exact_attention);
        encoder.set_buffer(0, Some(qkv), 0);
        encoder.set_buffer(1, Some(&self.exact_output), 0);
        encoder.set_bytes(
            2,
            std::mem::size_of_val(&range) as u64,
            range.as_ptr().cast(),
        );
        encoder.dispatch_thread_groups(
            thread_group(u64::from(rows * QUERY_HEADS)),
            thread_group(64),
        );
        encoder.memory_barrier_with_resources(&[&self.exact_output]);
        encoder.end_encoding();
    }

    pub(crate) fn output(&self) -> &BufferRef {
        &self.output
    }
}

pub(super) fn half_values(buffer: &BufferRef, elements: usize) -> &[u16] {
    unsafe { std::slice::from_raw_parts(buffer.contents().cast::<u16>(), elements) }
}
