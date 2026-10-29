//! Unified PISA-1 attention sublayer across chunked scoring and single-token
//! decoding.
//!
//! Combines QKV linear projection, rotary position embeddings (RoPE), PISA-1
//! block-pyramid update and causal attention, and output linear projection on
//! typed [`View`] and linear [`Out`] handles.

use crate::nn::linear::{Linear, TileLinear};
use crate::nn::{MODEL_LAYERS, PISA_BLOCK, QKV_WIDTH, ROPE_HEADS, ROPE_PAIRS, Rows, WIDTH};
use crate::tensor::{Dispatch, F16, F32, Fence, Out, U32, View};
use metal::{BufferRef, ComputeCommandEncoderRef, ComputePipelineState};

/// Layer-stacked attention projection weights.
#[derive(Clone, Copy)]
pub(crate) struct Weights<'a> {
    /// Stacked QKV projection matrices `[MODEL_LAYERS * WIDTH, QKV_WIDTH]`.
    pub(crate) qkv: View<'a, F16>,
    /// Stacked output projection matrices `[MODEL_LAYERS * WIDTH, WIDTH]`.
    pub(crate) output: View<'a, F16>,
}

impl<'a> Weights<'a> {
    /// QKV weight matrix `[WIDTH, QKV_WIDTH]` for `layer`.
    pub(crate) fn qkv_layer(&self, layer: u32) -> View<'a, F16> {
        let _ = MODEL_LAYERS;
        self.qkv.window(layer * WIDTH, WIDTH)
    }

    /// Output projection weight matrix `[WIDTH, WIDTH]` for `layer`.
    pub(crate) fn output_layer(&self, layer: u32) -> View<'a, F16> {
        self.output.window(layer * WIDTH, WIDTH)
    }
}

/// Typed handle to the key-value cache and pyramid summary buffers.
///
/// Encapsulates the sequential key-value activation buffer and the
/// hierarchical multi-level block summary pyramid (from `Selection.summary`),
/// replacing raw `(&BufferRef, &BufferRef)` tuples.
#[derive(Clone, Copy)]
pub(crate) struct TreeKvCache<'a> {
    /// Contiguous QKV token history `[context, QKV_WIDTH]`.
    pub(crate) qkv: &'a BufferRef,
    /// Multi-scale pyramid tree summaries `[levels, nodes, WIDTH]`.
    pub(crate) pyramid: &'a BufferRef,
}

impl<'a> TreeKvCache<'a> {
    #[inline(always)]
    pub(crate) fn new(qkv: &'a BufferRef, pyramid: &'a BufferRef) -> Self {
        Self { qkv, pyramid }
    }

    #[inline(always)]
    pub(crate) fn qkv(&self) -> &'a BufferRef {
        self.qkv
    }

    #[inline(always)]
    pub(crate) fn pyramid(&self) -> &'a BufferRef {
        self.pyramid
    }

    /// Token row slice at `position` within the sequential QKV buffer.
    #[inline(always)]
    pub(crate) fn token_row(&self, position: u32) -> View<'a, F16> {
        View::slice(
            self.qkv,
            u64::from(position) * u64::from(QKV_WIDTH) * 2,
            1,
            QKV_WIDTH,
        )
    }
}

impl<'a> From<(&'a BufferRef, &'a BufferRef)> for TreeKvCache<'a> {
    #[inline(always)]
    fn from((qkv, pyramid): (&'a BufferRef, &'a BufferRef)) -> Self {
        Self::new(qkv, pyramid)
    }
}

impl<'a> From<(&'a metal::Buffer, &'a metal::Buffer)> for TreeKvCache<'a> {
    #[inline(always)]
    fn from((qkv, pyramid): (&'a metal::Buffer, &'a metal::Buffer)) -> Self {
        Self::new(qkv.as_ref(), pyramid.as_ref())
    }
}

/// Rotary position embedding (RoPE) operator.
#[derive(Clone, Copy)]
pub(crate) struct Rope<'a> {
    pub(crate) pipeline: &'a ComputePipelineState,
    pub(crate) table: View<'a, F32>,
    pub(crate) context: u32,
}

impl<'a> Rope<'a> {
    #[inline(always)]
    pub(crate) const fn new(
        pipeline: &'a ComputePipelineState,
        table: View<'a, F32>,
        context: u32,
    ) -> Self {
        Self {
            pipeline,
            table,
            context,
        }
    }

    /// Apply rotary position embedding (RoPE) in-place to `qkv` at `position`.
    pub(crate) fn rotate(
        &self,
        encoder: &ComputeCommandEncoderRef,
        position: u32,
        qkv: Out<'_, F16>,
    ) {
        let rope_shape = [1u32, position, self.context, ROPE_PAIRS];
        Dispatch::new(encoder, self.pipeline, Fence::Bound)
            .write(qkv)
            .read(self.table)
            .bytes(&rope_shape)
            .groups(ROPE_HEADS * ROPE_PAIRS, 256);
    }
}

/// Single-token autoregressive PISA-1 attention sublayer (`Rows::Step`).
#[derive(Clone, Copy)]
pub(crate) struct StepAttention<'a> {
    pub(crate) linear: Linear<'a>,
    pub(crate) rope: &'a ComputePipelineState,
    pub(crate) leaf: &'a ComputePipelineState,
    pub(crate) attention: &'a ComputePipelineState,
    pub(crate) weights: Weights<'a>,
    pub(crate) rope_table: View<'a, F32>,
    pub(crate) blocks: View<'a, U32>,
    pub(crate) attn_scratch: &'a BufferRef,
    pub(crate) context: u32,
}

impl<'a> StepAttention<'a> {
    /// Return the RoPE operator handle.
    #[inline(always)]
    pub(crate) fn rope(&self) -> Rope<'a> {
        Rope::new(self.rope, self.rope_table, self.context)
    }

    /// Apply rotary position embedding (RoPE) in-place to `qkv` at `position`.
    #[inline(always)]
    pub(crate) fn rotate(
        &self,
        encoder: &ComputeCommandEncoderRef,
        position: u32,
        qkv: Out<'_, F16>,
    ) {
        self.rope().rotate(encoder, position, qkv);
    }

    /// Encode single-step QKV projection, RoPE, PISA-1 leaf summary +
    /// attention, and output projection into `projected`.
    pub(crate) fn forward<'c>(
        &self,
        encoder: &ComputeCommandEncoderRef,
        (layer, position): (u32, u32),
        normalized: View<'_, F16>,
        cache: impl Into<TreeKvCache<'c>>,
        projected: Out<'_, F16>,
    ) {
        let cache = cache.into();
        let qkv_row = cache.token_row(position);
        self.linear.forward_vector(
            encoder,
            normalized,
            self.weights.qkv_layer(layer),
            qkv_row.out(),
        );
        self.rotate(encoder, position, qkv_row.out());
        if position % PISA_BLOCK == PISA_BLOCK - 1 {
            Dispatch::new(encoder, self.leaf, Fence::Bound)
                .read(View::<F16>::whole(cache.qkv))
                .write(View::<F16>::whole(cache.pyramid).out())
                .bytes(&position)
                .groups(1, 64);
        }
        Dispatch::new(encoder, self.attention, Fence::Bound)
            .read(View::<F16>::whole(cache.qkv))
            .read(View::<F16>::whole(cache.pyramid))
            .read(self.blocks)
            .write(View::<F16>::whole(self.attn_scratch).out())
            .bytes(&position)
            .groups(1, 32);
        let attn_row = View::slice(
            self.attn_scratch,
            u64::from(position) * u64::from(WIDTH) * 2,
            1,
            WIDTH,
        );
        self.linear.forward_vector(
            encoder,
            attn_row,
            self.weights.output_layer(layer),
            projected,
        );
    }
}

/// Chunked TensorOps attention projection sublayer (`Rows::Chunk`).
#[derive(Clone, Copy)]
pub(crate) struct ChunkAttention<'a> {
    pub(crate) output_projection: &'a ComputePipelineState,
    pub(crate) weights: Weights<'a>,
    pub(crate) rows: u32,
}

impl<'a> ChunkAttention<'a> {
    #[inline(always)]
    pub(crate) fn new(
        output_projection: &'a ComputePipelineState,
        weights: Weights<'a>,
        rows: u32,
    ) -> Self {
        Self {
            output_projection,
            weights,
            rows,
        }
    }

    /// Encode TensorOps output projection for `layer`.
    pub(crate) fn forward_output(
        &self,
        encoder: &ComputeCommandEncoderRef,
        layer: u32,
        attn_out: View<'_, F16>,
        projected: Out<'_, F16>,
    ) -> Result<(), String> {
        TileLinear::new(self.output_projection, 8)?.forward(
            encoder,
            attn_out,
            self.weights.output_layer(layer),
            projected,
        );
        Ok(())
    }

    /// Forward alias matching sublayer dispatch.
    #[inline(always)]
    pub(crate) fn forward(
        &self,
        encoder: &ComputeCommandEncoderRef,
        layer: u32,
        attn_out: View<'_, F16>,
        projected: Out<'_, F16>,
    ) -> Result<(), String> {
        self.forward_output(encoder, layer, attn_out, projected)
    }
}

/// Unified Attention Sublayer that bridges `Rows::Step` and `Rows::Chunk`
/// configurations.
#[derive(Clone, Copy)]
pub(crate) enum AttentionSublayer<'a> {
    Step(StepAttention<'a>),
    Chunk(ChunkAttention<'a>),
}

impl<'a> AttentionSublayer<'a> {
    #[inline(always)]
    pub(crate) fn weights(&self) -> &Weights<'a> {
        match self {
            Self::Step(step) => &step.weights,
            Self::Chunk(chunk) => &chunk.weights,
        }
    }

    #[inline(always)]
    pub(crate) fn rows(&self) -> Rows {
        match self {
            Self::Step(_) => Rows::Step,
            Self::Chunk(chunk) => Rows::Chunk(chunk.rows),
        }
    }

    #[inline(always)]
    pub(crate) fn as_step(&self) -> Option<&StepAttention<'a>> {
        match self {
            Self::Step(step) => Some(step),
            Self::Chunk(_) => None,
        }
    }

    #[inline(always)]
    pub(crate) fn as_chunk(&self) -> Option<&ChunkAttention<'a>> {
        match self {
            Self::Chunk(chunk) => Some(chunk),
            Self::Step(_) => None,
        }
    }

    #[inline(always)]
    pub(crate) fn is_step(&self) -> bool {
        matches!(self, Self::Step(_))
    }

    #[inline(always)]
    pub(crate) fn is_chunk(&self) -> bool {
        matches!(self, Self::Chunk(_))
    }

    #[inline(always)]
    pub(crate) fn step(self) -> Option<StepAttention<'a>> {
        match self {
            Self::Step(step) => Some(step),
            Self::Chunk(_) => None,
        }
    }

    #[inline(always)]
    pub(crate) fn chunk(self) -> Option<ChunkAttention<'a>> {
        match self {
            Self::Chunk(chunk) => Some(chunk),
            Self::Step(_) => None,
        }
    }

    #[inline(always)]
    pub(crate) fn rope(&self) -> Option<Rope<'a>> {
        match self {
            Self::Step(step) => Some(step.rope()),
            Self::Chunk(_) => None,
        }
    }

    /// Single-token forward pass dispatching to [`StepAttention::forward`].
    #[inline(always)]
    pub(crate) fn forward_step<'c>(
        &self,
        encoder: &ComputeCommandEncoderRef,
        (layer, position): (u32, u32),
        normalized: View<'_, F16>,
        cache: impl Into<TreeKvCache<'c>>,
        projected: Out<'_, F16>,
    ) -> Result<(), String> {
        match self {
            Self::Step(step) => {
                step.forward(encoder, (layer, position), normalized, cache, projected);
                Ok(())
            }
            Self::Chunk(_) => {
                Err("expected Rows::Step attention sublayer, found Rows::Chunk".to_string())
            }
        }
    }

    /// Chunked output projection dispatching to
    /// [`ChunkAttention::forward_output`].
    #[inline(always)]
    pub(crate) fn forward_chunk(
        &self,
        encoder: &ComputeCommandEncoderRef,
        layer: u32,
        attn_out: View<'_, F16>,
        projected: Out<'_, F16>,
    ) -> Result<(), String> {
        match self {
            Self::Chunk(chunk) => chunk.forward_output(encoder, layer, attn_out, projected),
            Self::Step(_) => {
                Err("expected Rows::Chunk attention sublayer, found Rows::Step".to_string())
            }
        }
    }

    /// Output projection dispatch for chunk mode.
    #[inline(always)]
    pub(crate) fn forward_output(
        &self,
        encoder: &ComputeCommandEncoderRef,
        layer: u32,
        attn_out: View<'_, F16>,
        projected: Out<'_, F16>,
    ) -> Result<(), String> {
        self.forward_chunk(encoder, layer, attn_out, projected)
    }

    /// Forward pass executing QKV, rotary RoPE table dispatch, causal
    /// attention, and output projection into `projected`.
    pub(crate) fn forward<'c>(
        &self,
        encoder: &ComputeCommandEncoderRef,
        (layer, position): (u32, u32),
        normalized: View<'_, F16>,
        cache: impl Into<TreeKvCache<'c>>,
        projected: Out<'_, F16>,
    ) -> Result<(), String> {
        let cache = cache.into();
        match self {
            Self::Step(step) => {
                step.forward(encoder, (layer, position), normalized, cache, projected);
                Ok(())
            }
            Self::Chunk(chunk) => chunk.forward_output(encoder, layer, normalized, projected),
        }
    }
}

impl<'a> From<StepAttention<'a>> for AttentionSublayer<'a> {
    #[inline(always)]
    fn from(step: StepAttention<'a>) -> Self {
        Self::Step(step)
    }
}

impl<'a> From<ChunkAttention<'a>> for AttentionSublayer<'a> {
    #[inline(always)]
    fn from(chunk: ChunkAttention<'a>) -> Self {
        Self::Chunk(chunk)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::apple_gpu::Runtime;

    #[test]
    fn tree_cache() -> Result<(), String> {
        let runtime = Runtime::shared()?;
        let b1 = runtime.buffer::<u16>(128);
        let b2 = runtime.buffer::<u16>(128);
        let cache = TreeKvCache::new(&b1, &b2);
        assert_eq!(cache.qkv().contents(), b1.contents());
        assert_eq!(cache.pyramid().contents(), b2.contents());

        let from_refs: TreeKvCache<'_> = (&b1, &b2).into();
        assert_eq!(from_refs.qkv().contents(), b1.contents());

        let from_raw: TreeKvCache<'_> = (b1.as_ref(), b2.as_ref()).into();
        assert_eq!(from_raw.pyramid().contents(), b2.contents());
        Ok(())
    }

    #[test]
    fn token_row() -> Result<(), String> {
        let runtime = Runtime::shared()?;
        let b1 = runtime.buffer::<u16>((QKV_WIDTH * 2) as usize);
        let b2 = runtime.buffer::<u16>(128);
        let cache = TreeKvCache::new(&b1, &b2);
        let row0 = cache.token_row(0);
        assert_eq!(row0.rows(), 1);
        assert_eq!(row0.cols(), QKV_WIDTH);
        assert_eq!(row0.offset(), 0);

        let row1 = cache.token_row(1);
        assert_eq!(row1.rows(), 1);
        assert_eq!(row1.cols(), QKV_WIDTH);
        assert_eq!(row1.offset(), u64::from(QKV_WIDTH) * 2);
        Ok(())
    }

    #[test]
    fn sublayer_accessors() -> Result<(), String> {
        let runtime = Runtime::shared()?;
        let pipeline = runtime.pipeline(include_str!("../fbt_moe.metal"), "test", "fbt_moe_rms")?;
        let qkv_weights = runtime.buffer::<u16>(1024);
        let out_weights = runtime.buffer::<u16>(1024);
        let weights = Weights {
            qkv: View::whole(&qkv_weights),
            output: View::whole(&out_weights),
        };

        let chunk = ChunkAttention {
            output_projection: &pipeline,
            weights,
            rows: 64,
        };
        let sublayer = AttentionSublayer::Chunk(chunk);
        assert!(sublayer.is_chunk());
        assert!(!sublayer.is_step());
        assert!(sublayer.as_chunk().is_some());
        assert!(sublayer.as_step().is_none());
        assert!(sublayer.rope().is_none());
        assert_eq!(sublayer.rows(), Rows::Chunk(64));
        assert_eq!(
            sublayer.weights().qkv.buffer().contents(),
            qkv_weights.contents()
        );
        Ok(())
    }

    #[test]
    fn sublayer_dispatch() -> Result<(), String> {
        let runtime = Runtime::shared()?;
        let pipeline = runtime.pipeline(include_str!("../fbt_moe.metal"), "test", "fbt_moe_rms")?;
        let qkv_weights = runtime.buffer::<u16>(1024);
        let out_weights = runtime.buffer::<u16>(1024);
        let weights = Weights {
            qkv: View::whole(&qkv_weights),
            output: View::whole(&out_weights),
        };

        let chunk = ChunkAttention {
            output_projection: &pipeline,
            weights,
            rows: 32,
        };
        let sublayer = AttentionSublayer::from(chunk);
        let input = runtime.buffer::<u16>(32 * WIDTH as usize);
        let out = runtime.buffer::<u16>(32 * WIDTH as usize);
        let b1 = runtime.buffer::<u16>(128);
        let b2 = runtime.buffer::<u16>(128);
        let cache = TreeKvCache::new(&b1, &b2);

        let command = runtime.queue.new_command_buffer();
        let encoder = command.new_compute_command_encoder();
        let step_err = sublayer.forward_step(
            &encoder,
            (0, 0),
            View::whole(&input),
            cache,
            View::<F16>::whole(&out).out(),
        );
        encoder.end_encoding();
        assert!(step_err.is_err());
        Ok(())
    }
}
