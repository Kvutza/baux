use super::*;

impl Scorer<'_> {
    pub(super) fn embed(&self) -> Result<(), String> {
        let encoder = self.active_encoder();
        let pipelines = self.pipelines;
        let buffers = self.buffers;
        let weights = self.weights;
        let activation_offset = self.activation_offset();
        encoder.set_compute_pipeline_state(if self.diffusion.is_some() {
            &pipelines.denoise_embed
        } else {
            &pipelines.embed
        });
        encoder.set_buffer(0, Some(weights.buffer), weights.readout);
        encoder.set_buffer(
            1,
            Some(&buffers.tokens),
            u64::from(self.row_start) * std::mem::size_of::<u32>() as u64,
        );
        encoder.set_buffer(2, Some(&buffers.input), activation_offset);
        encoder.set_bytes(
            3,
            std::mem::size_of_val(&VOCAB) as u64,
            std::ptr::from_ref(&VOCAB).cast(),
        );
        if let Some(input) = self.diffusion {
            encoder.set_buffer(3, Some(weights.buffer), weights.mask_embed);
            encoder.set_buffer(4, Some(input.confidence), u64::from(self.row_start) * 4);
            encoder.dispatch_thread_groups(thread_group(u64::from(self.rows)), thread_group(128));
        } else {
            encoder.dispatch_threads(
                thread_group(u64::from(self.rows) * u64::from(WIDTH)),
                thread_group(pipelines.embed.max_total_threads_per_threadgroup().min(256)),
            );
        }
        encoder.memory_barrier_with_resources(&[&buffers.input]);

        if weights.architecture.is_multistream() {
            self.hyper()
                .init_streams(&encoder, self.activations(&buffers.input));
            return Ok(());
        }

        crate::nn::norm::RmsNorm::new(&pipelines.rms, crate::nn::Rows::Chunk(self.rows)).forward(
            &encoder,
            self.activations(&buffers.input),
            crate::tensor::View::slice(weights.buffer, weights.attention_norm, 1, WIDTH),
            self.activations(&buffers.normalized).out(),
        );
        Ok(())
    }

    #[inline]
    pub(super) fn capture_hidden(&self, _output: &BufferRef) -> Result<(), String> {
        Ok(())
    }

    pub(super) fn layer(
        &self,
        input: &BufferRef,
        layer: u32,
        pass: u32,
        execution: u32,
    ) -> Result<(), String> {
        if self.weights.architecture.is_multistream() {
            self.mhc_attention(layer, execution)?;
            return self.mhc_moe(layer);
        }
        self.attention(layer, execution)?;
        self.feed_forward(input, layer, pass)
    }
}
