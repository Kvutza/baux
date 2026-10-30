use super::*;
use crate::nn::Rows;
use crate::nn::hyper::{Hyper, Kernels, Routed, State};
use crate::tensor::{F16, View};

impl Scorer<'_> {
    #[inline(always)]
    pub(super) fn active_rows(&self) -> u32 {
        self.active_rows.get()
    }

    /// Hyper-connection layer over this scorer's active row window.
    pub(super) fn hyper(&self) -> Hyper<'_> {
        let start = self.scratch_start();
        let rows = self.active_rows();
        let streams = |stream: usize| {
            View::matrix(&self.buffers.mhc_streams[stream], MHC_STREAMS * WIDTH).window(start, rows)
        };
        Hyper {
            kernels: Kernels {
                replicate: &self.pipelines.mhc_replicate,
                predict: &self.pipelines.mhc_predict_rows,
                mix_rms: &self.pipelines.mhc_mix_rms,
                update: &self.pipelines.mhc_update,
                mean_rms: &self.pipelines.mhc_mean_rms,
            },
            weights: self.weights.hyper(),
            state: State {
                streams: [streams(0), streams(1)],
                coefficients: View::matrix(&self.buffers.mhc_coefficients, MHC_COEFFICIENTS)
                    .window(start, rows),
            },
            rows: Rows::Chunk(rows),
        }
    }

    /// `WIDTH`-wide activations of the active row window.
    pub(super) fn activations<'b>(&self, buffer: &'b BufferRef) -> View<'b, F16> {
        View::matrix(buffer, WIDTH).window(self.scratch_start(), self.active_rows())
    }

    /// Routed-expert outputs the FFN branch combines from.
    pub(super) fn routed(&self) -> Routed<'_> {
        let fine = &self.buffers.fine_grained;
        Routed {
            combine: &self.pipelines.combine_mhc,
            route_weights: View::whole(&fine.route_weights),
            packed_rows: View::whole(&fine.packed_rows),
            routed: View::whole(&fine.routed_output),
            shared: View::whole(&fine.shared_output),
        }
    }

    /// Chunked MoE FFN layer over this scorer's active row window.
    pub(super) fn moe(&self) -> Result<crate::nn::moe::ChunkMoe<'_>, String> {
        let fine = &self.pipelines.fine_grained;
        Ok(crate::nn::moe::ChunkMoe {
            pipelines: fine,
            buffers: &self.buffers.fine_grained,
            weights: self.weights.moe(),
            rows: self.active_rows(),
        })
    }

    /// Chunked attention projection layer over this scorer's active row window.
    pub(super) fn attention_layer(&self) -> crate::nn::attention::ChunkAttention<'_> {
        crate::nn::attention::ChunkAttention {
            output_projection: &self.tensorops.output_projection_wide,
            weights: self.weights.attention(),
            rows: self.active_rows(),
        }
    }

    /// Chunked attention projection sublayer over this scorer's active row
    /// window.
    pub(super) fn attention_sublayer(&self) -> crate::nn::attention::AttentionSublayer<'_> {
        crate::nn::attention::AttentionSublayer::Chunk(self.attention_layer())
    }

    /// Unified Transformer block composing hyper-connection, attention, and
    /// MoE.
    pub(super) fn block(&self) -> Result<crate::nn::block::TransformerBlock<'_>, String> {
        Ok(crate::nn::block::TransformerBlock::new(
            self.hyper(),
            crate::nn::attention::AttentionSublayer::Chunk(self.attention_layer()),
            crate::nn::moe::MoeSublayer::Chunk(self.moe()?),
        ))
    }

    /// Chunked readout and cross-entropy loss sublayer.
    pub(super) fn loss_layer(&self) -> crate::nn::loss::ChunkLoss<'_> {
        crate::nn::loss::ChunkLoss::new(
            &self.tensorops.readout_loss_tiles,
            &self.pipelines.readout_loss_reduce,
            self.weights.readout_weights(),
            self.rows,
        )
    }

    pub(super) fn feed_mhc(&self, layer: u32) -> Result<(), String> {
        let moe = self.moe()?;
        let normalized = self.activations(&self.buffers.normalized);
        self.start_stage("route")?;
        moe.route(self.active_encoder(), layer, normalized)?;
        self.start_stage("gate")?;
        moe.gate(self.active_encoder(), layer, normalized)?;
        self.start_stage("down")?;
        moe.down(self.active_encoder(), layer)
    }

    pub(super) fn mhc_attention(&self, layer: u32, execution: u32) -> Result<(), String> {
        let block = self.block()?;
        let normalized = self.activations(&self.buffers.normalized);
        self.start_stage("mhc_attention_input")?;
        let attn = block.stage_attention(self.active_encoder(), layer, normalized.out());
        self.attention(layer, execution)?;
        self.start_stage("mhc_attention_update")?;
        let projected = self.activations(&self.buffers.projected);
        attn.update(self.active_encoder(), projected);
        Ok(())
    }

    pub(super) fn mhc_moe(&self, layer: u32) -> Result<(), String> {
        let block = self.block()?;
        let normalized = self.activations(&self.buffers.normalized);
        self.start_stage("mhc_moe_input")?;
        let ffn = block.stage_moe(self.active_encoder(), layer, normalized.out());
        self.feed_mhc(layer)?;
        self.start_stage("mhc_moe_update")?;
        let output = self.activations(&self.buffers.output);
        ffn.combine(self.active_encoder(), output.out(), self.routed())
    }

    pub(super) fn hnet_chunk(&self) -> Result<(), String> {
        if !self.weights.architecture.is_hnet() || self.rows < 256 || self.scratch_start() != 0 {
            return Ok(());
        }
        let chunk_rows = ((self.rows / 2) & !127).max(128);
        let hnet = self.buffers.hnet()?;
        self.start_stage("hnet_chunk")?;
        let encoder = self.active_encoder();
        let normalized = self.activations(&self.buffers.normalized);
        self.hyper().finalize(encoder, normalized.out());
        let counts = [self.rows, chunk_rows];
        encoder.set_compute_pipeline_state(&hnet.waves.mhc_chunk);
        crate::apple_gpu::bind_buffers(
            encoder,
            &[
                (&self.buffers.normalized, 0),
                (&self.buffers.mhc_streams[0], 0),
                (&hnet.encoder_streams, 0),
                (&hnet.origins, 0),
                (&hnet.probs, 0),
                (&hnet.plug_idx, 0),
            ],
        );
        crate::apple_gpu::encode_value(encoder, 6, &counts);
        encoder.dispatch_thread_groups(thread_group(1), thread_group(256));
        encoder.memory_barrier_with_resources(&[
            &self.buffers.mhc_streams[0],
            &hnet.encoder_streams,
            &hnet.origins,
            &hnet.probs,
            &hnet.plug_idx,
        ]);
        self.active_rows.set(chunk_rows);
        Ok(())
    }

    pub(super) fn hnet_dechunk(&self) -> Result<(), String> {
        let chunk_rows = self.active_rows();
        if !self.weights.architecture.is_hnet() || chunk_rows == self.rows {
            return Ok(());
        }
        let hnet = self.buffers.hnet()?;
        self.start_stage("hnet_dechunk")?;
        let encoder = self.active_encoder();
        let counts = [self.rows, chunk_rows];
        encoder.set_compute_pipeline_state(&hnet.waves.mhc_dechunk);
        crate::apple_gpu::bind_buffers(
            encoder,
            &[
                (&self.buffers.mhc_streams[0], 0),
                (&hnet.encoder_streams, 0),
                (&hnet.chunk_scratch, 0),
                (&hnet.origins, 0),
                (&hnet.probs, 0),
                (&hnet.plug_idx, 0),
            ],
        );
        crate::apple_gpu::encode_value(encoder, 6, &counts);
        encoder.dispatch_thread_groups(thread_group(2), thread_group(256));
        encoder.memory_barrier_with_resources(&[&self.buffers.mhc_streams[0]]);
        self.active_rows.set(self.rows);
        Ok(())
    }
}
