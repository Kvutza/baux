//! Manifold-Constrained Hyper-Connections (`mHC-4`) decoding sublayer.
//!
//! # External Prior-Art Attribution
//! The manifold-constrained hyper-connection architecture and doubly-stochastic
//! residual stream transport represent external prior art used as a
//! sequence-modeling testbed, not a BAUX paper contribution:
//! - Xie et al. (DeepSeek-AI), *"mHC: Manifold-Constrained Hyper-Connections"*,
//!   `arXiv:2512.24880` (2025).
//! - Defa Zhu et al., *"Hyper-Connections"*, ICLR 2025, `arXiv:2409.19606`.
//! - Richard Sinkhorn, *"A relationship between arbitrary positive matrices and
//!   doubly stochastic matrices"*, Ann. Math. Statist. 35(2), 876–879 (1964);
//!   Richard Sinkhorn and Paul Knopp, Pacific J. Math. 21(2), 343–348 (1967).

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
            weights: weights.moe(None),
            state: StepState {
                padded_router: View::new(&self.router, 0, MODEL_LAYERS, WIDTH * PADDED_EXPERTS),
                route_scores: View::new(&self.route_scores, 0, 1, PADDED_EXPERTS),
                experts: View::whole(&self.experts),
                route_weights: View::whole(&self.route_weights),
                margin: View::whole(&self.margin),
                gate: View::new(&self.gate, 0, 4, 512),
                activation: View::new(&self.activation, 0, 4, 256),
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

    fn norm(&self, encoder: &ComputeCommandEncoderRef, weight: (&BufferRef, u64)) {
        crate::nn::norm::RmsNorm::new(&self.kernels.rms, Rows::Step).forward(
            encoder,
            Self::row(&self.state),
            View::slice(weight.0, weight.1, 1, WIDTH),
            Self::row(&self.normalized).out(),
        );
    }

    fn residual(
        &self,
        encoder: &ComputeCommandEncoderRef,
        input: &BufferRef,
        branch: &BufferRef,
        weight: (&BufferRef, u64),
        output: &BufferRef,
    ) {
        crate::nn::norm::ResidualRms::new(&self.kernels.residual, Rows::Step).forward(
            encoder,
            [Self::row(input), Self::row(branch)],
            View::slice(weight.0, weight.1, 1, WIDTH),
            [Self::row(output).out(), Self::row(&self.normalized).out()],
        );
    }

    fn layer(
        &self,
        encoder: &ComputeCommandEncoderRef,
        weights: CandidateRow<'_>,
        position: u32,
        layer: u32,
        pass: u32,
        execution: u32,
    ) {
        let attn = weights.architecture.is_multistream().then(|| {
            self.hyper(weights).prepare(
                encoder,
                layer,
                Site::Attention,
                Self::row(&self.normalized).out(),
            )
        });
        let cache = &self.cache[execution as usize];
        let tree_kv = cache.tree_kv();
        self.attention_layer(weights).forward(
            encoder,
            (layer, position),
            Self::row(&self.normalized),
            tree_kv,
            Self::row(&self.projected).out(),
        );
        if let Some(attn) = attn {
            self.finish_mhc(encoder, weights, layer, attn);
        } else {
            self.finish_layer(encoder, weights, layer, pass);
        }
    }

    fn finish_layer(
        &self,
        encoder: &ComputeCommandEncoderRef,
        weights: CandidateRow<'_>,
        layer: u32,
        pass: u32,
    ) {
        self.residual(
            encoder,
            &self.state,
            &self.projected,
            (
                weights.buffer,
                weights.ffn_norm + half_bytes(u64::from(layer * WIDTH)),
            ),
            &self.residual,
        );
        let norm = if layer + 1 < MODEL_LAYERS {
            (
                weights.buffer,
                weights.attention_norm + half_bytes(u64::from((layer + 1) * WIDTH)),
            )
        } else if pass + 1 < FEEDBACK_PASSES {
            if weights
                .architecture
                .feedback_transition(FeedbackTransition::ProjectedSigmoid)
                == FeedbackTransition::ProjectedSigmoid
            {
                (&*self.unit_norm, 0)
            } else {
                (weights.buffer, weights.attention_norm)
            }
        } else {
            (weights.buffer, weights.final_norm)
        };
        self.moe(weights).forward_residual(
            encoder,
            (layer, Self::row(&self.normalized)),
            [
                Self::row(&self.residual),
                View::slice(norm.0, norm.1, 1, WIDTH),
            ],
            [
                Self::row(&self.state).out(),
                Self::row(&self.normalized).out(),
            ],
        );
    }

    pub(crate) fn token(
        &self,
        encoder: &ComputeCommandEncoderRef,
        weights: CandidateRow<'_>,
        position: u32,
        feedback: FeedbackTransition,
    ) {
        let feedback = weights.architecture.feedback_transition(feedback);
        bytes(encoder, 3, &position);
        bytes(encoder, 4, &VOCAB);
        launch(
            encoder,
            &self.kernels.embed,
            &[
                (weights.buffer, weights.readout),
                (&self.tokens, 0),
                (&self.state, 0),
            ],
            thread_group(4),
            128,
        );
        if weights.architecture.is_multistream() {
            self.hyper(weights)
                .init_streams(encoder, Self::row(&self.state));
        } else {
            self.norm(encoder, (weights.buffer, weights.attention_norm));
        }
        for step in weights.architecture.layer_steps() {
            self.layer(
                encoder,
                weights,
                position,
                step.layer,
                step.pass,
                step.execution,
            );
            if !weights.architecture.is_multistream()
                && step.layer + 1 == MODEL_LAYERS
                && step.pass + 1 < FEEDBACK_PASSES
                && feedback == FeedbackTransition::ProjectedSigmoid
            {
                self.gemv(
                    encoder,
                    (&self.state, 0),
                    (weights.buffer, weights.feedback_state),
                    (&self.feedback_state, 0),
                    WIDTH,
                    WIDTH,
                    0,
                );
                self.gemv(
                    encoder,
                    (&self.normalized, 0),
                    (weights.buffer, weights.feedback_gate),
                    (&self.feedback_gate, 0),
                    WIDTH,
                    WIDTH,
                    0,
                );
                crate::nn::norm::ResidualRms::new(&self.kernels.feedback, Rows::Step).forward(
                    encoder,
                    [
                        Self::row(&self.feedback_state),
                        Self::row(&self.feedback_gate),
                    ],
                    View::slice(weights.buffer, weights.attention_norm, 1, WIDTH),
                    [
                        Self::row(&self.state).out(),
                        Self::row(&self.normalized).out(),
                    ],
                );
            }
        }
        if weights.architecture.is_multistream() {
            self.hyper(weights)
                .finalize(encoder, Self::row(&self.normalized).out());
        }
    }
}
