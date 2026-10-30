use super::*;
use crate::nn::Rows;
use crate::nn::hyper::{Hyper, Kernels, Prepared, Site, State};
use crate::tensor::{F16, View};

impl Decoder {
    /// Hyper-connection layer over the decoder's single resident row.
    pub(super) fn hyper<'a>(&'a self, weights: CandidateRow<'a>) -> Hyper<'a> {
        Hyper {
            kernels: Kernels {
                replicate: &self.kernels.mhc_replicate,
                predict: &self.kernels.mhc_predict,
                mix_rms: &self.kernels.mhc_mix_rms,
                update: &self.kernels.mhc_update,
                mean_rms: &self.kernels.mhc_mean_rms,
            },
            weights: weights.hyper(),
            state: State {
                streams: [
                    View::new(&self.mhc_streams[0], 0, 1, MHC_STREAMS * WIDTH),
                    View::new(&self.mhc_streams[1], 0, 1, MHC_STREAMS * WIDTH),
                ],
                coefficients: View::new(&self.mhc_coefficients, 0, 1, MHC_COEFFICIENTS),
            },
            rows: Rows::Step,
        }
    }

    /// `WIDTH`-wide view of a single-row activation buffer.
    pub(super) fn row<'b>(buffer: &'b BufferRef) -> View<'b, F16> {
        View::new(buffer, 0, 1, WIDTH)
    }

    /// Single-token MoE FFN layer over the decoder's resident buffers.
    pub(super) fn moe<'a>(&'a self, weights: CandidateRow<'a>) -> crate::nn::moe::StepMoe<'a> {
        use crate::nn::linear::Linear;
        use crate::nn::moe::{StepKernels, StepMoe, StepState};
        StepMoe {
            kernels: StepKernels {
                linear: Linear::new(
                    &self.kernels.gemv,
                    &self.kernels.gemv_vector,
                    View::whole(&self.experts),
                ),
                route: &self.kernels.route,
                activation: &self.kernels.activation,
                combine_branch: &self.kernels.combine_branch,
                combine_residual: &self.kernels.combine_residual,
            },
            weights: weights.moe(),
            state: StepState {
                padded_router: View::new(&self.router, 0, MODEL_LAYERS, WIDTH * PADDED_EXPERTS),
                route_scores: View::new(&self.route_scores, 0, 1, PADDED_EXPERTS),
                experts: View::whole(&self.experts),
                route_weights: View::whole(&self.route_weights),
                margin: View::whole(&self.margin),
                gate: View::new(&self.gate, 0, 4, 432),
                activation: View::new(&self.activation, 0, 4, 216),
                down: View::new(&self.down, 0, 4, WIDTH),
            },
        }
    }

    /// Single-token PISA-1 attention layer over the decoder's resident buffers.
    pub(super) fn attention_layer<'a>(
        &'a self,
        weights: CandidateRow<'a>,
    ) -> crate::nn::attention::StepAttention<'a> {
        use crate::nn::attention::StepAttention;
        use crate::nn::linear::Linear;
        StepAttention {
            linear: Linear::new(
                &self.kernels.gemv,
                &self.kernels.gemv_vector,
                View::whole(&self.experts),
            ),
            rope: &self.kernels.rope,
            leaf: &self.kernels.leaf,
            attention: &self.kernels.attention,
            weights: weights.attention(),
            rope_table: View::whole(&self.rope),
            blocks: View::whole(&self.blocks),
            attn_scratch: &self.attention,
            context: self.context,
        }
    }

    /// Single-token attention sublayer over the decoder's resident buffers.
    pub(super) fn attention_sublayer<'a>(
        &'a self,
        weights: CandidateRow<'a>,
    ) -> crate::nn::attention::AttentionSublayer<'a> {
        crate::nn::attention::AttentionSublayer::Step(self.attention_layer(weights))
    }

    pub(super) fn finish_mhc(
        &self,
        encoder: &ComputeCommandEncoderRef,
        weights: CandidateRow<'_>,
        layer: u32,
        attn: Prepared,
    ) {
        let hyper = self.hyper(weights);
        hyper.update(encoder, attn, Self::row(&self.projected));
        let ffn = hyper.prepare(encoder, layer, Site::Ffn, Self::row(&self.normalized).out());
        self.moe(weights).forward_branch(
            encoder,
            layer,
            Self::row(&self.normalized),
            Self::row(&self.residual).out(),
        );
        hyper.update(encoder, ffn, Self::row(&self.residual));
    }
}
