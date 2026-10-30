//! Sparse Mixture-of-Experts (MoE) FFN sublayer.
//!
//! Unifies the routed + shared expert FFN across:
//! - Single-token autoregressive decode (`StepMoe`: top-3 routing + 4-expert
//!   GEMV + SwiGLU + combine)
//! - Chunked pretrain/scoring (`ChunkMoe`: histogram/prefix/pack + GPU-driven
//!   indirect TensorOps + combine)
//!
//! Bridges both configurations under the unified [`MoeSublayer`] enum and
//! typed state buffer handle [`MoeBuffers`].

use metal::{ComputeCommandEncoderRef, ComputePipelineState};

use crate::fbt_moe::MoeShape;
use crate::nn::linear::Linear;
use crate::nn::{EXPERT_WIDTH, ROUTED_EXPERTS, Rows, WIDTH};
use crate::tensor::{Dispatch, F16, F32, Fence, Out, U32, View};

const GATE_WIDTH: u32 = baux::semantics::shape::GATE_WIDTH as u32;

/// Layer-stacked MoE FFN parameters across `MODEL_LAYERS`.
#[derive(Clone, Copy)]
pub(crate) struct Weights<'a> {
    pub(crate) router: View<'a, F16>,
    pub(crate) gate_up: View<'a, F16>,
    pub(crate) down: View<'a, F16>,
}

/// Typed state buffers for chunked MoE execution (`Rows::Chunk`).
pub(crate) type ChunkBuffers<'a> = &'a crate::fbt_moe::routing::FineGrainedMoeBuffers;

/// Scratch buffers for single-token autoregressive MoE (`Rows::Step`).
#[derive(Clone, Copy)]
pub(crate) struct StepState<'a> {
    pub(crate) padded_router: View<'a, F16>,
    pub(crate) route_scores: View<'a, F16>,
    pub(crate) experts: View<'a, U32>,
    pub(crate) route_weights: View<'a, F16>,
    pub(crate) margin: View<'a, F32>,
    pub(crate) gate: View<'a, F16>,
    pub(crate) activation: View<'a, F16>,
    pub(crate) down: View<'a, F16>,
}

/// Typed state buffers for single-step MoE execution (`Rows::Step`).
pub(crate) type StepBuffers<'a> = StepState<'a>;

/// Unified typed state buffer handle across `Rows::Chunk` and `Rows::Step`.
#[derive(Clone, Copy)]
pub(crate) enum MoeBuffers<'a> {
    Chunk(ChunkBuffers<'a>),
    Step(StepBuffers<'a>),
}

/// Unified typed state buffer handle (alias for [`MoeBuffers`]).
pub(crate) type MoeState<'a> = MoeBuffers<'a>;

/// Chunked MoE FFN sublayer (`Rows::Chunk`) wrapping
/// [`crate::fbt_moe::routing::FineGrainedMoePipelines`].
#[derive(Clone, Copy)]
pub(crate) struct ChunkMoe<'a> {
    pub(crate) pipelines: &'a crate::fbt_moe::routing::FineGrainedMoePipelines,
    pub(crate) buffers: &'a crate::fbt_moe::routing::FineGrainedMoeBuffers,
    pub(crate) weights: Weights<'a>,
    pub(crate) rows: u32,
}

impl<'a> ChunkMoe<'a> {
    /// Stage 1: Router logits, top-3 selection, block histogram, prefix sum,
    /// and dropless pack.
    pub(crate) fn route(
        &self,
        encoder: &ComputeCommandEncoderRef,
        layer: u32,
        input: View<'_, F16>,
    ) -> Result<(), String> {
        let router = self.weights.router.row(layer);
        self.pipelines.route_rows(
            encoder,
            self.buffers,
            input.buffer(),
            input.offset(),
            router.buffer(),
            router.offset(),
            self.rows,
        )
    }

    /// Stage 2: Routed indirect TensorOps gate-up + shared expert TensorOps
    /// gate-up.
    pub(crate) fn gate(
        &self,
        encoder: &ComputeCommandEncoderRef,
        layer: u32,
        input: View<'_, F16>,
    ) -> Result<(), String> {
        let gate = self.weights.gate_up.row(layer);
        self.pipelines.gate_rows(
            encoder,
            self.buffers,
            input.buffer(),
            input.offset(),
            gate.buffer(),
            gate.offset(),
            self.rows,
        )
    }

    /// Stage 3: Routed indirect TensorOps down projection + shared expert
    /// TensorOps down projection.
    pub(crate) fn down(
        &self,
        encoder: &ComputeCommandEncoderRef,
        layer: u32,
    ) -> Result<(), String> {
        let down = self.weights.down.row(layer);
        self.pipelines.down_rows(
            encoder,
            self.buffers,
            down.buffer(),
            down.offset(),
            self.rows,
        )
    }

    /// Stages 2 & 3: Expert execution (gate-up + down projection).
    pub(crate) fn experts(
        &self,
        encoder: &ComputeCommandEncoderRef,
        layer: u32,
        input: View<'_, F16>,
    ) -> Result<(), String> {
        self.gate(encoder, layer, input)?;
        self.down(encoder, layer)
    }

    /// Stage 4 (Single-stream): Fused routed+shared combine + residual add +
    /// next-layer RMSNorm.
    pub(crate) fn combine_residual(
        &self,
        encoder: &ComputeCommandEncoderRef,
        residual: View<'_, F16>,
        norm: View<'_, F16>,
        [output, normalized]: [Out<'_, F16>; 2],
    ) {
        let out_view = output.into_view();
        let norm_view = normalized.into_view();
        self.pipelines.combine_offsets(
            encoder,
            self.buffers,
            (residual.buffer(), residual.offset()),
            (norm.buffer(), norm.offset()),
            (out_view.buffer(), out_view.offset()),
            (norm_view.buffer(), norm_view.offset()),
            self.rows,
        );
    }

    /// Row geometry of this chunked layer.
    #[inline(always)]
    pub(crate) fn rows(&self) -> Rows {
        Rows::Chunk(self.rows)
    }

    /// Typed state buffer handle for this chunked layer.
    #[inline(always)]
    pub(crate) fn state(&self) -> MoeBuffers<'a> {
        MoeBuffers::Chunk(self.buffers)
    }
}

/// Kernels for single-token autoregressive MoE (`Rows::Step`).
#[derive(Clone, Copy)]
pub(crate) struct StepKernels<'a> {
    pub(crate) linear: Linear<'a>,
    pub(crate) route: &'a ComputePipelineState,
    pub(crate) activation: &'a ComputePipelineState,
    pub(crate) combine_branch: &'a ComputePipelineState,
    pub(crate) combine_residual: &'a ComputePipelineState,
}

/// Single-token autoregressive MoE FFN layer (`Rows::Step`).
#[derive(Clone, Copy)]
pub(crate) struct StepMoe<'a> {
    pub(crate) kernels: StepKernels<'a>,
    pub(crate) weights: Weights<'a>,
    pub(crate) state: StepState<'a>,
}

impl<'a> StepMoe<'a> {
    /// Route 1 token across 31 routed experts (plus 1 shared expert = 4 active
    /// experts).
    pub(crate) fn route(
        &self,
        encoder: &ComputeCommandEncoderRef,
        layer: u32,
        input: View<'_, F16>,
    ) {
        self.kernels.linear.forward_vector(
            encoder,
            input,
            self.state.padded_router.row(layer),
            self.state.route_scores.out(),
        );
        let route_shape = [1u32, WIDTH, ROUTED_EXPERTS, 3, 64, 1];
        Dispatch::new(encoder, self.kernels.route, Fence::Bound)
            .read(self.state.route_scores)
            .write(self.state.experts.out())
            .write(self.state.route_weights.out())
            .write(self.state.margin.out())
            .bytes(&route_shape)
            .groups(1, 32);
    }

    /// Execute 4-expert Gate-Up GEMV and SwiGLU activation.
    pub(crate) fn gate(
        &self,
        encoder: &ComputeCommandEncoderRef,
        layer: u32,
        input: View<'_, F16>,
    ) {
        let gate = self.weights.gate_up.row(layer);
        let gate_weight = View::slice(gate.buffer(), gate.offset(), WIDTH, GATE_WIDTH);
        self.kernels
            .linear
            .forward_vector(encoder, input, gate_weight, self.state.gate.out());
        let swiglu_shape = MoeShape {
            rows: 4,
            width: WIDTH,
            experts: 4,
            rows_per_expert: 1,
            expert_width: EXPERT_WIDTH,
        };
        Dispatch::new(encoder, self.kernels.activation, Fence::Bound)
            .read(self.state.gate)
            .write(self.state.activation.out())
            .bytes(&swiglu_shape)
            .groups(7, 128);
    }

    /// Execute 4-expert Down GEMV.
    pub(crate) fn down(&self, encoder: &ComputeCommandEncoderRef, layer: u32) {
        let down_row = self.weights.down.row(layer);
        let down_weight = View::slice(down_row.buffer(), down_row.offset(), EXPERT_WIDTH, WIDTH);
        self.kernels.linear.forward_vector(
            encoder,
            self.state.activation,
            down_weight,
            self.state.down.out(),
        );
    }

    /// Execute 4-expert Gate-Up GEMV, SwiGLU activation, and 4-expert Down
    /// GEMV.
    pub(crate) fn experts(
        &self,
        encoder: &ComputeCommandEncoderRef,
        layer: u32,
        input: View<'_, F16>,
    ) {
        self.gate(encoder, layer, input);
        self.down(encoder, layer);
    }

    /// Combine 4 active experts into residual and compute next-layer RMSNorm.
    pub(crate) fn combine_residual(
        &self,
        encoder: &ComputeCommandEncoderRef,
        residual: View<'_, F16>,
        norm: View<'_, F16>,
        [state_out, normalized_out]: [Out<'_, F16>; 2],
    ) {
        Dispatch::new(encoder, self.kernels.combine_residual, Fence::Bound)
            .read(self.state.down)
            .read(self.state.route_weights)
            .read(residual)
            .read(norm)
            .write(state_out)
            .write(normalized_out)
            .groups(1, 128);
    }

    /// Forward pass producing a branch output (`projected`) for multistream
    /// mHC.
    pub(crate) fn forward_branch(
        &self,
        encoder: &ComputeCommandEncoderRef,
        layer: u32,
        input: View<'_, F16>,
        branch_out: Out<'_, F16>,
    ) {
        self.route(encoder, layer, input);
        self.experts(encoder, layer, input);
        Dispatch::new(encoder, self.kernels.combine_branch, Fence::Bound)
            .read(self.state.down)
            .read(self.state.route_weights)
            .write(branch_out)
            .groups(4, 128);
    }

    /// Forward pass fusing expert combine + residual addition + next-layer
    /// RMSNorm.
    pub(crate) fn forward_residual(
        &self,
        encoder: &ComputeCommandEncoderRef,
        (layer, input): (u32, View<'_, F16>),
        [residual, norm]: [View<'_, F16>; 2],
        [state_out, normalized_out]: [Out<'_, F16>; 2],
    ) {
        self.route(encoder, layer, input);
        self.experts(encoder, layer, input);
        self.combine_residual(encoder, residual, norm, [state_out, normalized_out]);
    }

    /// Row geometry of this single-token decode layer.
    #[inline(always)]
    pub(crate) fn rows(&self) -> Rows {
        Rows::Step
    }

    /// Typed state buffer handle for this single-step layer.
    #[inline(always)]
    pub(crate) fn state(&self) -> MoeBuffers<'a> {
        MoeBuffers::Step(self.state)
    }
}

/// Unified MoE FFN sublayer bridging `Rows::Step` and `Rows::Chunk`
/// configurations.
#[derive(Clone, Copy)]
pub(crate) enum MoeSublayer<'a> {
    Step(StepMoe<'a>),
    Chunk(ChunkMoe<'a>),
}

impl<'a> MoeSublayer<'a> {
    /// Construct a chunked MoE sublayer for `Rows::Chunk`.
    #[inline(always)]
    pub(crate) fn chunk(
        pipelines: &'a crate::fbt_moe::routing::FineGrainedMoePipelines,
        buffers: &'a crate::fbt_moe::routing::FineGrainedMoeBuffers,
        weights: Weights<'a>,
        rows: u32,
    ) -> Self {
        Self::Chunk(ChunkMoe {
            pipelines,
            buffers,
            weights,
            rows,
        })
    }

    /// Construct a single-token decode MoE sublayer for `Rows::Step`.
    #[inline(always)]
    pub(crate) fn step(
        kernels: StepKernels<'a>,
        weights: Weights<'a>,
        state: StepState<'a>,
    ) -> Self {
        Self::Step(StepMoe {
            kernels,
            weights,
            state,
        })
    }

    /// Parameter weights for this MoE sublayer.
    #[inline(always)]
    pub(crate) fn weights(&self) -> &Weights<'a> {
        match self {
            Self::Step(step) => &step.weights,
            Self::Chunk(chunk) => &chunk.weights,
        }
    }

    /// Row geometry of this MoE sublayer.
    #[inline(always)]
    pub(crate) fn rows(&self) -> Rows {
        match self {
            Self::Step(step) => step.rows(),
            Self::Chunk(chunk) => chunk.rows(),
        }
    }

    /// Typed state buffers for this MoE sublayer.
    #[inline(always)]
    pub(crate) fn state(&self) -> MoeBuffers<'a> {
        match self {
            Self::Step(step) => step.state(),
            Self::Chunk(chunk) => chunk.state(),
        }
    }

    /// Stage 1: Router logits and top-k selection.
    pub(crate) fn route(
        &self,
        encoder: &ComputeCommandEncoderRef,
        layer: u32,
        input: View<'_, F16>,
    ) -> Result<(), String> {
        match self {
            Self::Chunk(chunk) => chunk.route(encoder, layer, input),
            Self::Step(step) => {
                step.route(encoder, layer, input);
                Ok(())
            }
        }
    }

    /// Stage 2: Gate-up GEMV / indirect dispatch + SwiGLU activation.
    pub(crate) fn gate(
        &self,
        encoder: &ComputeCommandEncoderRef,
        layer: u32,
        input: View<'_, F16>,
    ) -> Result<(), String> {
        match self {
            Self::Chunk(chunk) => chunk.gate(encoder, layer, input),
            Self::Step(step) => {
                step.gate(encoder, layer, input);
                Ok(())
            }
        }
    }

    /// Stage 3: Down projection GEMV / indirect dispatch.
    pub(crate) fn down(
        &self,
        encoder: &ComputeCommandEncoderRef,
        layer: u32,
    ) -> Result<(), String> {
        match self {
            Self::Chunk(chunk) => chunk.down(encoder, layer),
            Self::Step(step) => {
                step.down(encoder, layer);
                Ok(())
            }
        }
    }

    /// Stages 2 & 3: Expert execution (gate-up, SwiGLU activation, down
    /// projection).
    pub(crate) fn experts(
        &self,
        encoder: &ComputeCommandEncoderRef,
        layer: u32,
        input: View<'_, F16>,
    ) -> Result<(), String> {
        match self {
            Self::Chunk(chunk) => chunk.experts(encoder, layer, input),
            Self::Step(step) => {
                step.experts(encoder, layer, input);
                Ok(())
            }
        }
    }

    /// Forward pass executing expert branch projection.
    pub(crate) fn forward_branch(
        &self,
        encoder: &ComputeCommandEncoderRef,
        layer: u32,
        input: View<'_, F16>,
        branch_out: Out<'_, F16>,
    ) -> Result<(), String> {
        match self {
            Self::Step(step) => {
                step.forward_branch(encoder, layer, input, branch_out);
                Ok(())
            }
            Self::Chunk(chunk) => {
                chunk.route(encoder, layer, input)?;
                chunk.gate(encoder, layer, input)?;
                chunk.down(encoder, layer)
            }
        }
    }

    #[inline(always)]
    pub(crate) fn as_step(&self) -> Option<&StepMoe<'a>> {
        match self {
            Self::Step(step) => Some(step),
            Self::Chunk(_) => None,
        }
    }

    #[inline(always)]
    pub(crate) fn as_chunk(&self) -> Option<&ChunkMoe<'a>> {
        match self {
            Self::Chunk(chunk) => Some(chunk),
            Self::Step(_) => None,
        }
    }
}

impl<'a> From<ChunkMoe<'a>> for MoeSublayer<'a> {
    #[inline(always)]
    fn from(chunk: ChunkMoe<'a>) -> Self {
        Self::Chunk(chunk)
    }
}

impl<'a> From<StepMoe<'a>> for MoeSublayer<'a> {
    #[inline(always)]
    fn from(step: StepMoe<'a>) -> Self {
        Self::Step(step)
    }
}
