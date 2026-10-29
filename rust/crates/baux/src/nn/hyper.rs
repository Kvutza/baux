//! Hyper-connection residual (HC / mHC).
//!
//! The residual state is `STREAMS` parallel copies of the hidden vector. Around
//! each sublayer, per-row coefficients are predicted from the streams, the
//! streams are mixed and RMS-normalized into the sublayer input, and the
//! sublayer output (the branch) is transported back into the streams.
//!
//! Following the OxCaml / DeepInfer (`HCLazyPostData`) linear typestate design,
//! [`Hyper::prepare`] returns a move-only [`Prepared`] token that must be
//! consumed by [`Hyper::update`] or [`Hyper::combine`].
//!
//! Streams ping-pong between two buffers: the attention site reads stream 0
//! and writes stream 1; the FFN site reads stream 1 and writes stream 0. After
//! the last layer, stream 0 holds the state that [`Hyper::finalize`] reads.

use metal::ComputeCommandEncoderRef;
use metal::ComputePipelineState;

use super::Rows;
use crate::tensor::{Dispatch, F16, F32, Fence, Out, U32, View};

const WIDTH: u32 = baux::semantics::mhc4::WIDTH as u32;
const STREAMS: u32 = baux::semantics::shape::MHC4_SHAPE.streams as u32;

/// Which sublayer a hyper-connection wraps.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Site {
    Attention,
    Ffn,
}

impl Site {
    /// Stream buffer the site reads; it writes the other one.
    #[inline(always)]
    const fn source(self) -> usize {
        (self as usize) & 1
    }
}

/// Move-only linear proof that [`Hyper::prepare`] populated
/// `state.coefficients` for `site` and is awaiting stream transport via
/// [`Hyper::update`] or [`Hyper::combine`].
#[must_use = "prepared hyper-connection coefficients must be transported back into the streams via \
              update or combine"]
pub(crate) struct Prepared {
    site: Site,
}

impl Prepared {
    /// Explicitly discard when only the attention KV cache is needed on the
    /// final step.
    #[inline(always)]
    pub(crate) fn discard(self) {}

    /// Sublayer site this prepared token belongs to.
    #[inline(always)]
    pub(crate) fn site(&self) -> Site {
        self.site
    }
}

/// Layer-stacked parameters of one site: row `l` belongs to physical layer `l`.
#[derive(Clone, Copy)]
pub(crate) struct SiteWeights<'a> {
    pub(crate) predictor: View<'a, F16>,
    pub(crate) bias: View<'a, F16>,
    pub(crate) control: View<'a, F16>,
    pub(crate) norm: View<'a, F16>,
}

#[derive(Clone, Copy)]
pub(crate) struct Weights<'a> {
    pub(crate) attention: SiteWeights<'a>,
    pub(crate) ffn: SiteWeights<'a>,
    pub(crate) final_norm: View<'a, F16>,
    /// Residual family selector understood by the kernels (HC vs. mHC).
    pub(crate) code: u32,
}

impl<'a> Weights<'a> {
    fn site(&self, site: Site) -> SiteWeights<'a> {
        match site {
            Site::Attention => self.attention,
            Site::Ffn => self.ffn,
        }
    }
}

/// Pipelines for one row geometry. `predict` and `update` differ between
/// chunk (`*_rows`) and step kernels; the caller supplies the matching pair.
#[derive(Clone, Copy)]
pub(crate) struct Kernels<'a> {
    pub(crate) replicate: &'a ComputePipelineState,
    pub(crate) predict: &'a ComputePipelineState,
    pub(crate) mix_rms: &'a ComputePipelineState,
    pub(crate) update: &'a ComputePipelineState,
    pub(crate) mean_rms: &'a ComputePipelineState,
}

/// Residual state, already windowed to the active rows.
#[derive(Clone, Copy)]
pub(crate) struct State<'a> {
    pub(crate) streams: [View<'a, F16>; 2],
    pub(crate) coefficients: View<'a, F32>,
}

/// Routed-expert outputs folded into the streams by one fused kernel.
#[derive(Clone, Copy)]
pub(crate) struct Routed<'a> {
    pub(crate) combine: &'a ComputePipelineState,
    pub(crate) route_weights: View<'a, F16>,
    pub(crate) packed_rows: View<'a, U32>,
    pub(crate) routed: View<'a, F16>,
    pub(crate) shared: View<'a, F16>,
}

#[derive(Clone, Copy)]
pub(crate) struct Hyper<'a> {
    pub(crate) kernels: Kernels<'a>,
    pub(crate) weights: Weights<'a>,
    pub(crate) state: State<'a>,
    pub(crate) rows: Rows,
}

impl Hyper<'_> {
    fn shape(&self) -> [u32; 4] {
        [self.rows.count(), WIDTH, self.weights.code, 0]
    }

    /// Replicate initial hidden activations `input` across `STREAMS` into
    /// stream 0.
    pub(crate) fn init_streams<'e>(
        &'e self,
        encoder: &'e ComputeCommandEncoderRef,
        input: View<'e, F16>,
    ) {
        let replicate = Dispatch::new(encoder, self.kernels.replicate, self.rows.fence())
            .read(input)
            .write(self.state.streams[0].out())
            .bytes(&self.shape());
        match self.rows {
            Rows::Chunk(rows) => replicate.threads(
                u64::from(rows * WIDTH),
                self.kernels
                    .replicate
                    .max_total_threads_per_threadgroup()
                    .min(256),
            ),
            Rows::Step => replicate.groups(STREAMS * WIDTH / 512, 128),
        }
    }

    /// Predict coefficients for `site` of `layer`, then write the mixed,
    /// normalized sublayer input into `input`. Returns a linear [`Prepared`]
    /// token.
    pub(crate) fn prepare<'e>(
        &'e self,
        encoder: &'e ComputeCommandEncoderRef,
        layer: u32,
        site: Site,
        input: Out<'e, F16>,
    ) -> Prepared {
        let weights = self.weights.site(site);
        let streams = self.state.streams[site.source()];
        let coefficients = self.state.coefficients;
        let shape = self.shape();
        let fence = self.rows.fence();
        let predict = Dispatch::new(encoder, self.kernels.predict, fence)
            .read(streams)
            .read(weights.predictor.row(layer))
            .read(weights.bias.row(layer))
            .read(weights.control.row(layer))
            .write(coefficients.out())
            .bytes(&shape);
        match self.rows {
            Rows::Chunk(rows) => predict.groups(rows.div_ceil(64), 128),
            Rows::Step => predict.groups(1, 256),
        }
        Dispatch::new(encoder, self.kernels.mix_rms, fence)
            .read(streams)
            .read(coefficients)
            .read(weights.norm.row(layer))
            .write(input)
            .bytes(&shape)
            .groups(self.rows.count(), 128);
        Prepared { site }
    }

    /// Transport a dense sublayer output `branch` back into the streams,
    /// consuming the [`Prepared`] token.
    pub(crate) fn update<'e>(
        &'e self,
        encoder: &'e ComputeCommandEncoderRef,
        prepared: Prepared,
        branch: View<'e, F16>,
    ) {
        let source = prepared.site.source();
        let update = Dispatch::new(encoder, self.kernels.update, self.rows.fence())
            .read(self.state.streams[source])
            .read(branch)
            .read(self.state.coefficients)
            .write(self.state.streams[1 - source].out())
            .bytes(&self.shape());
        match self.rows {
            // The following sublayer overwrites the branch buffer.
            Rows::Chunk(rows) => update
                .fence(branch)
                .threads(u64::from(rows * WIDTH / 4), 256),
            Rows::Step => update.groups(STREAMS * WIDTH / 256, 256),
        }
    }

    /// Combine routed experts into `branch` and transport it into the
    /// streams in the same pass; the rounded branch is kept for backward.
    /// Chunk geometry only: the decoder combines experts separately.
    pub(crate) fn combine<'e>(
        &'e self,
        encoder: &'e ComputeCommandEncoderRef,
        prepared: Prepared,
        branch: Out<'e, F16>,
        routed: Routed<'e>,
    ) -> Result<(), String> {
        let Rows::Chunk(rows) = self.rows else {
            return Err("fused routed transport requires chunk rows".into());
        };
        let source = prepared.site.source();
        Dispatch::new(encoder, routed.combine, Fence::Writes)
            .read(self.state.streams[source])
            .write(branch)
            .read(self.state.coefficients)
            .write(self.state.streams[1 - source].out())
            .read(routed.route_weights)
            .read(routed.packed_rows)
            .read(routed.routed)
            .read(routed.shared)
            .bytes(&self.shape())
            .threads(u64::from(rows * WIDTH / 4), 256);
        Ok(())
    }

    /// Average the streams and RMS-normalize into `output` (readout input).
    pub(crate) fn finalize<'e>(
        &'e self,
        encoder: &'e ComputeCommandEncoderRef,
        output: Out<'e, F16>,
    ) {
        Dispatch::new(encoder, self.kernels.mean_rms, self.rows.fence())
            .read(self.state.streams[0])
            .read(self.weights.final_norm)
            .write(output)
            .bytes(&self.shape())
            .groups(self.rows.count(), 128);
    }
}
