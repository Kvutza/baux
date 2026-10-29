//! Unified Transformer model container and forward execution pipeline.
//!
//! Orchestrates the full token-to-loss forward pass across:
//! - Token embedding lookup ([`View<'a, F16>`])
//! - Sequential layer blocks ([`TransformerBlock<'a>`]) combining attention and
//!   MoE
//! - Final normalization ([`FinalNorm<'a>`]: standalone RMSNorm or Hyper
//!   finalization)
//! - Vocabulary projection and cross-entropy chunk loss
//!   ([`ReadoutWeights<'a>`])
//!
//! Execution is parameterized by row geometry ([`Rows`]: chunked
//! prefill/scoring or single-step autoregressive decode) and temporary scratch
//! arenas ([`ScratchArena<'a>`]), emitting typed kernel dispatches into a
//! caller command encoder without owning resident buffers.

use metal::{ComputeCommandEncoderRef, ComputePipelineState};

use crate::nn::arena::MemoryArena;
use crate::nn::attention::TreeKvCache;
use crate::nn::block::TransformerBlock;
use crate::nn::hyper::Hyper;
use crate::nn::loss::{ChunkLoss, ReadoutWeights};
use crate::nn::norm::RmsNorm;
use crate::nn::{Rows, WIDTH};
use crate::tensor::{Dispatch, F16, F32, Out, U32, View};

const VOCAB: u32 = baux::semantics::mhc4::VOCABULARY as u32;

/// Final normalization layer: standalone RMSNorm or Hyper finalization.
#[derive(Clone, Copy)]
pub(crate) enum FinalNorm<'a> {
    Norm {
        norm: RmsNorm<'a>,
        weight: View<'a, F16>,
    },
    Hyper(&'a Hyper<'a>),
}

impl<'a> FinalNorm<'a> {
    /// Construct RMS normalization variant.
    #[inline(always)]
    pub(crate) fn norm(norm: RmsNorm<'a>, weight: View<'a, F16>) -> Self {
        Self::Norm { norm, weight }
    }

    /// Construct hyper-connection finalization variant.
    #[inline(always)]
    pub(crate) fn hyper(hyper: &'a Hyper<'a>) -> Self {
        Self::Hyper(hyper)
    }

    /// Execute final normalization into `output`.
    #[inline(always)]
    pub(crate) fn forward(
        &self,
        encoder: &ComputeCommandEncoderRef,
        input: View<'_, F16>,
        output: Out<'_, F16>,
    ) {
        match self {
            Self::Norm { norm, weight } => norm.forward(encoder, input, *weight, output),
            Self::Hyper(hyper) => hyper.finalize(encoder, output),
        }
    }
}

/// Scratch buffers and pipelines for chunk loss scoring and reduction.
#[derive(Clone, Copy)]
pub(crate) struct LossArena<'a> {
    pub(crate) loss_tiles: &'a ComputePipelineState,
    pub(crate) loss_reduce: &'a ComputePipelineState,
    pub(crate) targets: View<'a, U32>,
    pub(crate) partials: View<'a, F32>,
    pub(crate) stats: View<'a, F32>,
}

impl<'a> LossArena<'a> {
    pub(crate) fn new(
        loss_tiles: &'a ComputePipelineState,
        loss_reduce: &'a ComputePipelineState,
        targets: View<'a, U32>,
        partials: View<'a, F32>,
        stats: View<'a, F32>,
    ) -> Self {
        Self {
            loss_tiles,
            loss_reduce,
            targets,
            partials,
            stats,
        }
    }
}

/// Scratch arena for intermediate forward activations and caches.
#[derive(Clone, Copy)]
pub(crate) struct ScratchArena<'a> {
    pub(crate) hidden: View<'a, F16>,
    pub(crate) normalized: View<'a, F16>,
    pub(crate) projected: View<'a, F16>,
    pub(crate) residual: View<'a, F16>,
    pub(crate) cache: Option<TreeKvCache<'a>>,
    pub(crate) position: u32,
    pub(crate) embed_pipe: Option<&'a ComputePipelineState>,
    pub(crate) loss: Option<LossArena<'a>>,
}

impl<'a> ScratchArena<'a> {
    pub(crate) fn new(
        hidden: View<'a, F16>,
        normalized: View<'a, F16>,
        projected: View<'a, F16>,
        residual: View<'a, F16>,
    ) -> Self {
        Self {
            hidden,
            normalized,
            projected,
            residual,
            cache: None,
            position: 0,
            embed_pipe: None,
            loss: None,
        }
    }

    /// Construct scratch views from a resident memory arena.
    pub(crate) fn from_arena(arena: &'a MemoryArena, rows: u32) -> Self {
        let hidden = arena.alloc_view::<F16>(rows, WIDTH);
        let normalized = arena.alloc_view::<F16>(rows, WIDTH);
        let projected = arena.alloc_view::<F16>(rows, WIDTH);
        let residual = arena.alloc_view::<F16>(rows, WIDTH);
        Self::new(hidden, normalized, projected, residual)
    }

    pub(crate) fn with_cache(mut self, cache: TreeKvCache<'a>, position: u32) -> Self {
        self.cache = Some(cache);
        self.position = position;
        self
    }

    pub(crate) fn with_loss(mut self, loss: LossArena<'a>) -> Self {
        self.loss = Some(loss);
        self
    }

    pub(crate) fn with_embed(mut self, embed_pipe: &'a ComputePipelineState) -> Self {
        self.embed_pipe = Some(embed_pipe);
        self
    }
}

/// Result of the model forward pass: either final hidden states or chunk loss.
#[derive(Clone, Copy)]
pub(crate) enum ModelOutput<'b> {
    Hidden(View<'b, F16>),
    Loss(ChunkLoss<'b>),
}

impl<'b> ModelOutput<'b> {
    #[inline(always)]
    pub(crate) fn hidden(self) -> Option<View<'b, F16>> {
        match self {
            Self::Hidden(view) => Some(view),
            Self::Loss(_) => None,
        }
    }

    #[inline(always)]
    pub(crate) fn loss(self) -> Option<ChunkLoss<'b>> {
        match self {
            Self::Loss(loss) => Some(loss),
            Self::Hidden(_) => None,
        }
    }
}

/// Complete Transformer model pipeline container.
pub(crate) struct TransformerModel<'a> {
    pub(crate) embeddings: View<'a, F16>,
    pub(crate) blocks: Vec<TransformerBlock<'a>>,
    pub(crate) final_norm: FinalNorm<'a>,
    pub(crate) loss: ReadoutWeights<'a>,
}

impl<'a> TransformerModel<'a> {
    /// Create a new transformer model container.
    pub(crate) fn new(
        embeddings: View<'a, F16>,
        blocks: Vec<TransformerBlock<'a>>,
        final_norm: FinalNorm<'a>,
        loss: ReadoutWeights<'a>,
    ) -> Self {
        Self {
            embeddings,
            blocks,
            final_norm,
            loss,
        }
    }

    /// Construct from a borrowed slice of layer blocks.
    pub(crate) fn from_slice(
        embeddings: View<'a, F16>,
        blocks: &[TransformerBlock<'a>],
        final_norm: FinalNorm<'a>,
        loss: ReadoutWeights<'a>,
    ) -> Self {
        Self::new(embeddings, blocks.to_vec(), final_norm, loss)
    }

    /// Embedding table view `[VOCAB, WIDTH]`.
    #[inline(always)]
    pub(crate) fn embeddings(&self) -> View<'a, F16> {
        self.embeddings
    }

    /// Active transformer layer blocks.
    #[inline(always)]
    pub(crate) fn blocks(&self) -> &[TransformerBlock<'a>] {
        &self.blocks
    }

    /// Final normalization configuration.
    #[inline(always)]
    pub(crate) fn final_norm(&self) -> &FinalNorm<'a> {
        &self.final_norm
    }

    /// Readout loss weights.
    #[inline(always)]
    pub(crate) fn loss(&self) -> &ReadoutWeights<'a> {
        &self.loss
    }

    /// Execute the full forward pipeline across sequence tokens and layer
    /// blocks.
    ///
    /// Encodes embedding lookup, multi-layer attention and MoE evaluation,
    /// final normalization, and optional chunk cross-entropy loss computation
    /// into `encoder`.
    pub(crate) fn forward<'b>(
        &self,
        encoder: &ComputeCommandEncoderRef,
        tokens: View<'_, U32>,
        rows: Rows,
        scratch: &ScratchArena<'b>,
    ) -> Result<ModelOutput<'b>, String>
    where
        'a: 'b,
    {
        if let Some(pipe) = scratch.embed_pipe {
            match rows {
                Rows::Chunk(count) => {
                    Dispatch::new(encoder, pipe, rows.fence())
                        .read(self.embeddings)
                        .read(tokens)
                        .write(scratch.hidden.out())
                        .bytes(&VOCAB)
                        .threads(u64::from(count * WIDTH), 256);
                }
                Rows::Step => {
                    let pos = scratch.position;
                    Dispatch::new(encoder, pipe, rows.fence())
                        .read(self.embeddings)
                        .read(tokens)
                        .write(scratch.hidden.out())
                        .bytes(&pos)
                        .bytes(&VOCAB)
                        .groups(4, 128);
                }
            }
        }

        if let FinalNorm::Hyper(hyper) = self.final_norm {
            hyper.init_streams(encoder, scratch.hidden);
        }

        for (idx, block) in self.blocks.iter().enumerate() {
            let layer = idx as u32;
            match rows {
                Rows::Step => {
                    let cache = scratch
                        .cache
                        .ok_or_else(|| "step forward requires TreeKvCache".to_string())?;
                    block.forward_step(
                        encoder,
                        layer,
                        scratch.position,
                        scratch.normalized,
                        cache,
                        scratch.projected,
                        scratch.residual,
                    )?;
                }
                Rows::Chunk(_) => {
                    let attn_prep = block.stage_attention(encoder, layer, scratch.normalized.out());
                    let dummy =
                        TreeKvCache::new(scratch.projected.buffer(), scratch.projected.buffer());
                    block.attention.forward(
                        encoder,
                        (layer, 0),
                        scratch.normalized,
                        dummy,
                        scratch.projected.out(),
                    )?;
                    attn_prep.update(encoder, scratch.projected);

                    let ffn_prep = block.stage_moe(encoder, layer, scratch.normalized.out());
                    block.moe.forward_branch(
                        encoder,
                        layer,
                        scratch.normalized,
                        scratch.residual.out(),
                    )?;
                    ffn_prep.update(encoder, scratch.residual);
                }
            }
        }

        self.final_norm
            .forward(encoder, scratch.hidden, scratch.hidden.out());

        if let Some(loss_arena) = scratch.loss {
            let chunk_loss = ChunkLoss::new(
                loss_arena.loss_tiles,
                loss_arena.loss_reduce,
                self.loss,
                rows.count(),
            );
            chunk_loss.score_tiles(
                encoder,
                scratch.hidden,
                loss_arena.targets,
                loss_arena.partials.out(),
            );
            chunk_loss.reduce_loss(encoder, loss_arena.partials, loss_arena.stats.out());
            Ok(ModelOutput::Loss(chunk_loss))
        } else {
            Ok(ModelOutput::Hidden(scratch.hidden))
        }
    }

    /// Convenience forward pass returning the finalized hidden states.
    pub(crate) fn forward_hidden<'b>(
        &self,
        encoder: &ComputeCommandEncoderRef,
        tokens: View<'_, U32>,
        rows: Rows,
        scratch: &ScratchArena<'b>,
    ) -> Result<View<'b, F16>, String>
    where
        'a: 'b,
    {
        match self.forward(encoder, tokens, rows, scratch)? {
            ModelOutput::Hidden(hidden) => Ok(hidden),
            _ => Ok(scratch.hidden),
        }
    }

    /// Convenience forward pass returning the computed chunk loss.
    pub(crate) fn forward_loss<'b>(
        &self,
        encoder: &ComputeCommandEncoderRef,
        tokens: View<'_, U32>,
        rows: Rows,
        scratch: &ScratchArena<'b>,
    ) -> Result<ChunkLoss<'b>, String>
    where
        'a: 'b,
    {
        match self.forward(encoder, tokens, rows, scratch)? {
            ModelOutput::Loss(loss) => Ok(loss),
            _ => Err("forward did not produce chunk loss".to_string()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sizes() {
        assert!(std::mem::size_of::<FinalNorm<'_>>() > 0);
        assert!(std::mem::size_of::<ModelOutput<'_>>() > 0);
        assert!(std::mem::size_of::<ScratchArena<'_>>() > 0);
        assert!(std::mem::size_of::<LossArena<'_>>() > 0);
    }
}
