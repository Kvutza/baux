use super::scorer::Scorer;
use super::*;
use crate::forward_program::mhc4::{self, WaveShape};

#[cfg(test)]
#[test]
fn wave_compile() {
    let runtime = Runtime::shared().expect("Metal runtime required");
    Mhc4Waves::new(&runtime).expect("production MHC kernels must compile with generated ABI");
}

pub(super) struct Mhc4Waves {
    centroids_pipe: ComputePipelineState,
    readout_pipe: ComputePipelineState,
    centroids: Buffer,
}

impl Mhc4Waves {
    pub(super) fn new(runtime: &Runtime) -> Result<Self, String> {
        let source = mhc4::metal_source(include_str!("fbt_wave.metal"));
        let pipeline =
            |name| runtime.precise_metal4(&source, "MHC batched causal waves", name, &[]);
        Ok(Self {
            centroids_pipe: pipeline("mhc4_centroids")?,
            readout_pipe: pipeline("mhc4_readout")?,
            centroids: runtime.buffer::<u16>(mhc4::HIERARCHY_NODES * mhc4::QUERY),
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn encode_mhc(
        &self,
        command: &CommandBufferRef,
        hidden: (&BufferRef, u64),
        qkv: (&BufferRef, u64),
        readout: (&BufferRef, u64),
        expected: (&BufferRef, u64),
        output: (&BufferRef, u64),
        losses: (&BufferRef, u64),
        tokens: u32,
        score: u32,
        update_centroids: bool,
    ) -> Result<(), String> {
        let shape = WaveShape {
            blocks: tokens,
            tokens,
            proposal_start: 0,
            intra: 0,
            vocabulary: VOCAB,
            shards: mhc4::SHARDS as u32,
            temperature: 0.0,
            score,
        };
        let encoder =
            command.compute_command_encoder_with_dispatch_type(metal::MTLDispatchType::Serial);
        if update_centroids {
            encoder.set_compute_pipeline_state(&self.centroids_pipe);
            encoder.set_buffer(0, Some(readout.0), readout.1);
            encoder.set_buffer(1, Some(&self.centroids), 0);
            encoder.set_bytes(
                2,
                size_of::<WaveShape>() as u64,
                std::ptr::from_ref(&shape).cast(),
            );
            encoder.dispatch_thread_groups(thread_group(584 * 64), thread_group(64));
            encoder.memory_barrier_with_resources(&[&self.centroids]);
        }
        encoder.set_compute_pipeline_state(&self.readout_pipe);
        for (index, (buffer, offset)) in [
            hidden,
            qkv,
            readout,
            (&*self.centroids, 0),
            expected,
            output,
            losses,
        ]
        .into_iter()
        .enumerate()
        {
            encoder.set_buffer(index as u64, Some(buffer), offset);
        }
        encoder.set_bytes(
            7,
            size_of::<WaveShape>() as u64,
            std::ptr::from_ref(&shape).cast(),
        );
        encoder.dispatch_thread_groups(thread_group(u64::from(tokens)), thread_group(64));
        encoder.end_encoding();
        Ok(())
    }
}

impl Scorer<'_> {
    pub(super) fn readout(&self) -> Result<(), String> {
        let encoder = self.active_encoder();
        let pipelines = self.pipelines;
        let buffers = self.buffers;
        let loss_partials = buffers.loss_partials()?;
        let shape = [self.rows, VOCAB];
        encoder.set_compute_pipeline_state(&self.tensorops.readout_loss_tiles);
        encoder.set_buffer(0, Some(&buffers.normalized), 0);
        encoder.set_buffer(1, Some(self.weights.buffer), self.weights.readout);
        encoder.set_buffer(2, Some(&buffers.labels), 0);
        encoder.set_buffer(3, Some(loss_partials), 0);
        encoder.set_bytes(
            4,
            std::mem::size_of_val(&shape) as u64,
            shape.as_ptr().cast(),
        );
        encoder.dispatch_thread_groups(
            MTLSize {
                width: u64::from(VOCAB / 64),
                height: u64::from(self.rows.div_ceil(128)),
                depth: 1,
            },
            thread_group(128),
        );
        encoder.memory_barrier_with_resources(&[loss_partials]);
        encoder.set_compute_pipeline_state(&pipelines.readout_loss_reduce);
        encoder.set_buffer(0, Some(loss_partials), 0);
        encoder.set_buffer(1, Some(&buffers.losses), 0);
        let tiles = VOCAB / 64;
        encoder.set_bytes(
            2,
            std::mem::size_of_val(&tiles) as u64,
            (&tiles as *const u32).cast(),
        );
        encoder.dispatch_thread_groups(thread_group(u64::from(self.rows)), thread_group(128));
        encoder.memory_barrier_with_resources(&[&buffers.losses]);
        encoder.set_compute_pipeline_state(&pipelines.sequence_loss);
        encoder.set_buffer(0, Some(&buffers.losses), 0);
        encoder.set_buffer(1, Some(&buffers.score_mask), 0);
        encoder.set_buffer(2, Some(&buffers.sequence_scores), 0);
        encoder.dispatch_thread_groups(
            thread_group(u64::from(self.rows / CONTEXT)),
            MTLSize {
                width: 256,
                height: 1,
                depth: 1,
            },
        );
        Ok(())
    }

    pub(super) fn readout_proposals(
        &self,
        output: &BufferRef,
        seeds: &BufferRef,
        temperature: f32,
        score_targets: bool,
    ) -> Result<(), String> {
        if !temperature.is_finite() || temperature < 0.0 {
            return Err("proposal readout temperature must be finite and nonnegative".into());
        }
        let loss_partials = self.buffers.loss_partials()?;
        let proposal_partials = self.buffers.proposal_partials()?;
        let encoder = self.active_encoder();
        let shape = [self.rows, VOCAB, self.row_start, self.pisa1.context()];
        let activation_offset = self.activation_offset();
        encoder.set_compute_pipeline_state(if self.diffusion.is_some() {
            &self.tensorops.denoise_tiles
        } else {
            &self.tensorops.readout_proposal_tiles
        });
        encoder.set_buffer(0, Some(&self.buffers.normalized), activation_offset);
        encoder.set_buffer(1, Some(self.weights.buffer), self.weights.readout);
        encoder.set_buffer(2, Some(loss_partials), 0);
        encoder.set_bytes(
            3,
            std::mem::size_of_val(&shape) as u64,
            shape.as_ptr().cast(),
        );
        encoder.set_buffer(4, Some(seeds), 0);
        encoder.set_bytes(
            5,
            std::mem::size_of_val(&temperature) as u64,
            std::ptr::from_ref(&temperature).cast(),
        );
        encoder.set_buffer(6, Some(&self.buffers.labels), 0);
        encoder.set_buffer(7, Some(proposal_partials), 0);
        encoder.set_bytes(
            8,
            std::mem::size_of_val(&score_targets) as u64,
            std::ptr::from_ref(&score_targets).cast(),
        );
        if let Some(input) = self.diffusion {
            encoder.set_buffer(9, Some(input.moments), 0);
        }
        encoder.dispatch_thread_groups(
            MTLSize {
                width: u64::from(VOCAB / 64),
                height: u64::from(self.rows.div_ceil(128)),
                depth: 1,
            },
            thread_group(128),
        );
        encoder.memory_barrier_with_resources(&[loss_partials, proposal_partials]);
        if let Some(input) = self.diffusion {
            encoder.memory_barrier_with_resources(&[input.moments]);
        }
        let tiles = VOCAB / 64;
        let proposal_source = if score_targets && temperature == 0.0 && self.diffusion.is_none() {
            proposal_partials
        } else {
            loss_partials
        };
        self.start_stage("proposal_reduce")?;
        let encoder = self.active_encoder();
        self.proposal_reduce(&encoder, proposal_source, output, temperature, tiles);
        if score_targets {
            self.start_stage("target_losses")?;
            self.target_losses(self.active_encoder(), proposal_partials, tiles);
        }
        Ok(())
    }

    fn target_losses(
        &self,
        encoder: &ComputeCommandEncoderRef,
        proposal_partials: &BufferRef,
        tiles: u32,
    ) {
        encoder.set_compute_pipeline_state(&self.pipelines.readout_loss_reduce);
        encoder.set_buffer(0, Some(proposal_partials), 0);
        encoder.set_buffer(
            1,
            Some(&self.buffers.losses),
            u64::from(self.row_start) * std::mem::size_of::<f32>() as u64,
        );
        encoder.set_bytes(
            2,
            std::mem::size_of_val(&tiles) as u64,
            std::ptr::from_ref(&tiles).cast(),
        );
        encoder.dispatch_thread_groups(thread_group(u64::from(self.rows)), thread_group(128));
        encoder.memory_barrier_with_resources(&[&self.buffers.losses]);
    }

    fn proposal_reduce(
        &self,
        encoder: &ComputeCommandEncoderRef,
        loss_partials: &BufferRef,
        output: &BufferRef,
        temperature: f32,
        tiles: u32,
    ) {
        encoder.set_compute_pipeline_state(if self.diffusion.is_some() {
            &self.pipelines.denoise_reduce
        } else {
            &self.pipelines.readout_proposal_reduce
        });
        encoder.set_buffer(0, Some(loss_partials), 0);
        encoder.set_buffer(
            1,
            Some(output),
            u64::from(self.row_start) * std::mem::size_of::<u32>() as u64,
        );
        if let Some(input) = self.diffusion {
            encoder.set_buffer(2, Some(input.confidence), u64::from(self.row_start) * 4);
            encoder.set_buffer(3, Some(input.moments), 0);
            encoder.set_bytes(
                4,
                std::mem::size_of_val(&temperature) as u64,
                std::ptr::from_ref(&temperature).cast(),
            );
        } else {
            encoder.set_bytes(
                2,
                std::mem::size_of_val(&tiles) as u64,
                std::ptr::from_ref(&tiles).cast(),
            );
        }
        encoder.dispatch_thread_groups(thread_group(u64::from(self.rows)), thread_group(128));
        encoder.memory_barrier_with_resources(&[output]);
    }
}
