//! RMS Normalization and Fused Residual + RMSNorm layers.
//!
//! Built on [`Dispatch`] and access-moded [`View`] / [`Out`] handles:
//! - [`RmsNorm`]: `output = rms_norm(input, weight)`
//! - [`ResidualRms`]: `residual_out = input + branch; normalized_out =
//!   rms_norm(residual_out, weight)` (also supports `feedback_fuse_rms` which
//!   fuses `feedback_state + sigmoid(feedback_gate)` with `rms_norm`).

use metal::{ComputeCommandEncoderRef, ComputePipelineState};

use crate::fbt_moe::MoeShape;
use crate::nn::{EXPERT_WIDTH, ROUTED_EXPERTS, Rows, WIDTH};
use crate::tensor::{Dispatch, Element, F16, Out, View};

fn shape_for(rows: Rows) -> MoeShape {
    match rows {
        Rows::Chunk(count) => MoeShape {
            rows: count,
            width: WIDTH,
            experts: ROUTED_EXPERTS,
            rows_per_expert: count / ROUTED_EXPERTS,
            expert_width: EXPERT_WIDTH,
        },
        Rows::Step => MoeShape {
            rows: 1,
            width: WIDTH,
            experts: 1,
            rows_per_expert: 1,
            expert_width: EXPERT_WIDTH,
        },
    }
}

/// Standalone RMS normalization operation across [`Rows`].
#[derive(Clone, Copy)]
pub(crate) struct RmsNorm<'a> {
    pipeline: &'a ComputePipelineState,
    rows: Rows,
}

impl<'a> RmsNorm<'a> {
    #[inline(always)]
    pub(crate) const fn new(pipeline: &'a ComputePipelineState, rows: Rows) -> Self {
        Self { pipeline, rows }
    }

    /// Forward pass: `output = rms_norm(input, weight)`.
    pub(crate) fn forward<E: Element>(
        self,
        encoder: &ComputeCommandEncoderRef,
        input: View<'_, E>,
        weight: View<'_, F16>,
        output: Out<'_, E>,
    ) {
        let shape = shape_for(self.rows);
        Dispatch::new(encoder, self.pipeline, self.rows.fence())
            .read(input)
            .read(weight)
            .write(output)
            .bytes(&shape)
            .groups(self.rows.count(), 128);
    }
}

/// Fused residual addition (or gated feedback fusion) + RMS normalization.
///
/// Writes both the updated residual (`residual_out`) and the normalized
/// sublayer input (`normalized_out`) in a single pass.
#[derive(Clone, Copy)]
pub(crate) struct ResidualRms<'a> {
    pipeline: &'a ComputePipelineState,
    rows: Rows,
}

impl<'a> ResidualRms<'a> {
    #[inline(always)]
    pub(crate) const fn new(pipeline: &'a ComputePipelineState, rows: Rows) -> Self {
        Self { pipeline, rows }
    }

    /// Forward pass: `residual_out = fuse(input, branch); normalized_out =
    /// rms_norm(residual_out, weight)`.
    pub(crate) fn forward(
        self,
        encoder: &ComputeCommandEncoderRef,
        [input, branch]: [View<'_, F16>; 2],
        weight: View<'_, F16>,
        [residual_out, normalized_out]: [Out<'_, F16>; 2],
    ) {
        let shape = shape_for(self.rows);
        Dispatch::new(encoder, self.pipeline, self.rows.fence())
            .read(input)
            .read(branch)
            .read(weight)
            .write(residual_out)
            .write(normalized_out)
            .bytes(&shape)
            .groups(self.rows.count(), 128);
    }
}
