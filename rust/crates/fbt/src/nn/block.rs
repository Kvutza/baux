//! Unified Transformer block composing hyper-connections, attention, and MoE.
//!
//! Enforces linear typestate transitions across residual stream transport:
//!
//! ```text
//! prepare(Attention) -> attention -> update -> prepare(Ffn) -> moe -> combine/update
//! ```
//!
//! Following the OxCaml / DeepInfer (`HCLazyPostData`) design, residual streams
//! ping-pong between stream 0 and stream 1 across sublayer evaluations. The
//! move-only linear typestate [`Prepared`] from [`Hyper::prepare`] guarantees
//! at compile time that residual streams cannot leak or desynchronize.
//!
//! Supports both chunked pretrain/scoring ([`Rows::Chunk`]) and single-token
//! autoregressive decode ([`Rows::Step`]).

use metal::ComputeCommandEncoderRef;

use super::Rows;
use super::attention::{AttentionSublayer, TreeKvCache};
use super::hyper::{Hyper, Prepared, Routed, Site};
use super::moe::MoeSublayer;
use crate::tensor::{F16, Out, View};

/// Move-only linear token representing prepared attention coefficients.
#[must_use = "prepared attention coefficients must be transported back into residual streams or \
              discarded"]
pub(crate) struct PreparedAttention<'a> {
    hyper: &'a Hyper<'a>,
    prepared: Prepared,
}

impl<'a> PreparedAttention<'a> {
    /// Return the inner [`Prepared`] token, relinquishing wrapper control.
    #[inline(always)]
    pub(crate) fn into_inner(self) -> Prepared {
        self.prepared
    }

    /// Discard the prepared token when only the attention KV cache is needed.
    #[inline(always)]
    pub(crate) fn discard(self) {
        self.prepared.discard();
    }

    /// Transport the attention branch output into stream 1, consuming the
    /// token.
    #[inline(always)]
    pub(crate) fn update<'e>(self, encoder: &'e ComputeCommandEncoderRef, branch: View<'e, F16>) {
        self.hyper.update(encoder, self.prepared, branch);
    }
}

/// Move-only linear token representing prepared MoE coefficients.
#[must_use = "prepared MoE coefficients must be transported back into residual streams"]
pub(crate) struct PreparedMoe<'a> {
    hyper: &'a Hyper<'a>,
    prepared: Prepared,
}

impl<'a> PreparedMoe<'a> {
    /// Return the inner [`Prepared`] token, relinquishing wrapper control.
    #[inline(always)]
    pub(crate) fn into_inner(self) -> Prepared {
        let Self { prepared, .. } = self;
        prepared
    }

    /// Transport dense MoE branch output back into stream 0 (`Rows::Step`).
    #[inline(always)]
    pub(crate) fn update<'e>(self, encoder: &'e ComputeCommandEncoderRef, branch: View<'e, F16>) {
        self.hyper.update(encoder, self.prepared, branch);
    }

    /// Fused combine of routed experts into stream 0 (`Rows::Chunk`).
    #[inline(always)]
    pub(crate) fn combine<'e>(
        self,
        encoder: &'e ComputeCommandEncoderRef,
        branch: Out<'e, F16>,
        routed: Routed<'e>,
    ) -> Result<(), String> {
        self.hyper.combine(encoder, self.prepared, branch, routed)
    }
}

/// Unified Transformer block combining hyper-connection, attention, and MoE.
#[derive(Clone, Copy)]
pub(crate) struct TransformerBlock<'a> {
    pub(crate) hyper: Hyper<'a>,
    pub(crate) attention: AttentionSublayer<'a>,
    pub(crate) moe: MoeSublayer<'a>,
}

impl<'a> TransformerBlock<'a> {
    /// Construct a new TransformerBlock instance.
    #[inline(always)]
    pub(crate) const fn new(
        hyper: Hyper<'a>,
        attention: AttentionSublayer<'a>,
        moe: MoeSublayer<'a>,
    ) -> Self {
        Self {
            hyper,
            attention,
            moe,
        }
    }

    /// Active row geometry of this transformer block.
    #[inline(always)]
    pub(crate) fn rows(&self) -> Rows {
        self.hyper.rows
    }

    /// Verify that row geometries of all composed sublayers are compatible.
    pub(crate) fn validate(&self) -> Result<(), String> {
        match (self.hyper.rows, &self.attention, &self.moe) {
            (Rows::Step, AttentionSublayer::Step(_), MoeSublayer::Step(_)) => Ok(()),
            (Rows::Chunk(_), AttentionSublayer::Chunk(_), MoeSublayer::Chunk(_)) => Ok(()),
            (rows, attn, moe) => Err(format!(
                "incompatible block configuration: rows={rows:?}, attention={}, moe={}",
                match attn {
                    AttentionSublayer::Step(_) => "Step",
                    AttentionSublayer::Chunk(_) => "Chunk",
                },
                match moe {
                    MoeSublayer::Step(_) => "Step",
                    MoeSublayer::Chunk(_) => "Chunk",
                }
            )),
        }
    }

    /// Prepare hyper-connection coefficients and mixed normalized input for
    /// attention.
    #[inline(always)]
    pub(crate) fn prepare_attention<'e>(
        &'e self,
        encoder: &'e ComputeCommandEncoderRef,
        layer: u32,
        input: Out<'e, F16>,
    ) -> Prepared {
        self.hyper.prepare(encoder, layer, Site::Attention, input)
    }

    /// Transport the attention branch output into stream 1, consuming
    /// [`Prepared`].
    #[inline(always)]
    pub(crate) fn update_attention<'e>(
        &'e self,
        encoder: &'e ComputeCommandEncoderRef,
        prepared: Prepared,
        branch: View<'e, F16>,
    ) {
        debug_assert_eq!(prepared.site(), Site::Attention);
        self.hyper.update(encoder, prepared, branch);
    }

    /// Prepare hyper-connection coefficients and mixed normalized input for
    /// MoE.
    #[inline(always)]
    pub(crate) fn prepare_moe<'e>(
        &'e self,
        encoder: &'e ComputeCommandEncoderRef,
        layer: u32,
        input: Out<'e, F16>,
    ) -> Prepared {
        self.hyper.prepare(encoder, layer, Site::Ffn, input)
    }

    /// Transport dense MoE branch output into stream 0 (`Rows::Step`),
    /// consuming [`Prepared`].
    #[inline(always)]
    pub(crate) fn update_moe<'e>(
        &'e self,
        encoder: &'e ComputeCommandEncoderRef,
        prepared: Prepared,
        branch: View<'e, F16>,
    ) {
        debug_assert_eq!(prepared.site(), Site::Ffn);
        self.hyper.update(encoder, prepared, branch);
    }

    /// Fused combine of routed experts into stream 0 (`Rows::Chunk`), consuming
    /// [`Prepared`].
    #[inline(always)]
    pub(crate) fn combine_moe<'e>(
        &'e self,
        encoder: &'e ComputeCommandEncoderRef,
        prepared: Prepared,
        branch: Out<'e, F16>,
        routed: Routed<'e>,
    ) -> Result<(), String> {
        debug_assert_eq!(prepared.site(), Site::Ffn);
        self.hyper.combine(encoder, prepared, branch, routed)
    }

    /// Prepare hyper-connection coefficients for the attention sublayer.
    pub(crate) fn stage_attention<'e>(
        &'e self,
        encoder: &'e ComputeCommandEncoderRef,
        layer: u32,
        input: Out<'e, F16>,
    ) -> PreparedAttention<'e> {
        let prepared = self.prepare_attention(encoder, layer, input);
        PreparedAttention {
            hyper: &self.hyper,
            prepared,
        }
    }

    /// Prepare hyper-connection coefficients for the MoE sublayer.
    pub(crate) fn stage_moe<'e>(
        &'e self,
        encoder: &'e ComputeCommandEncoderRef,
        layer: u32,
        input: Out<'e, F16>,
    ) -> PreparedMoe<'e> {
        let prepared = self.prepare_moe(encoder, layer, input);
        PreparedMoe {
            hyper: &self.hyper,
            prepared,
        }
    }

    /// Execute forward pass for a single token across attention and MoE
    /// (`Rows::Step`).
    ///
    /// Compile-time enforced typestate lifecycle:
    ///   prepare(Attention) -> step_attention -> update ->
    ///   prepare(Ffn) -> step_moe -> update
    pub(crate) fn forward_step(
        &self,
        encoder: &ComputeCommandEncoderRef,
        layer: u32,
        position: u32,
        normalized: View<'_, F16>,
        cache: TreeKvCache<'_>,
        projected: View<'_, F16>,
        residual: View<'_, F16>,
    ) -> Result<(), String> {
        let attn_prep = self.stage_attention(encoder, layer, normalized.out());
        self.attention.forward(
            encoder,
            (layer, position),
            normalized,
            cache,
            projected.out(),
        )?;
        attn_prep.update(encoder, projected);

        let ffn_prep = self.stage_moe(encoder, layer, normalized.out());
        self.moe
            .forward_branch(encoder, layer, normalized, residual.out())?;
        ffn_prep.update(encoder, residual);

        Ok(())
    }

    /// Execute forward pass for a chunk of rows across attention and MoE
    /// (`Rows::Chunk`).
    ///
    /// Compile-time enforced typestate lifecycle:
    ///   prepare(Attention) -> chunk_attention -> update ->
    ///   prepare(Ffn) -> chunk_moe -> combine
    pub(crate) fn forward_chunk(
        &self,
        encoder: &ComputeCommandEncoderRef,
        layer: u32,
        attn_out: View<'_, F16>,
        routed: Routed<'_>,
        normalized: View<'_, F16>,
        projected: View<'_, F16>,
        output: Out<'_, F16>,
    ) -> Result<(), String> {
        let attn_prep = self.stage_attention(encoder, layer, normalized.out());
        let chunk_attn = self
            .attention
            .as_chunk()
            .ok_or_else(|| "forward_chunk requires chunk attention".to_string())?;
        chunk_attn.forward_output(encoder, layer, attn_out, projected.out())?;
        attn_prep.update(encoder, projected);

        let ffn_prep = self.stage_moe(encoder, layer, normalized.out());
        let chunk_moe = self
            .moe
            .as_chunk()
            .ok_or_else(|| "forward_chunk requires chunk moe".to_string())?;
        chunk_moe.route(encoder, layer, normalized)?;
        chunk_moe.gate(encoder, layer, normalized)?;
        chunk_moe.down(encoder, layer)?;
        ffn_prep.combine(encoder, output, routed)
    }

    /// Execute forward pass across sublayers matching the provided row
    /// geometry.
    pub(crate) fn forward(
        &self,
        encoder: &ComputeCommandEncoderRef,
        layer: u32,
        rows: Rows,
        position: u32,
        normalized: View<'_, F16>,
        cache: Option<TreeKvCache<'_>>,
        projected: View<'_, F16>,
        residual: View<'_, F16>,
    ) -> Result<(), String> {
        match rows {
            Rows::Step => {
                let cache = cache.ok_or_else(|| "step forward requires TreeKvCache".to_string())?;
                self.forward_step(
                    encoder, layer, position, normalized, cache, projected, residual,
                )
            }
            Rows::Chunk(_) => {
                let attn_prep = self.stage_attention(encoder, layer, normalized.out());
                let chunk_attn = self
                    .attention
                    .as_chunk()
                    .ok_or_else(|| "chunk forward requires chunk attention".to_string())?;
                chunk_attn.forward_output(encoder, layer, normalized, projected.out())?;
                attn_prep.update(encoder, projected);

                let ffn_prep = self.stage_moe(encoder, layer, normalized.out());
                self.moe
                    .forward_branch(encoder, layer, normalized, residual.out())?;
                ffn_prep.update(encoder, residual);
                Ok(())
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prepared_size() {
        assert_eq!(std::mem::size_of::<Prepared>(), 1);
    }
}
