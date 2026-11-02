use super::*;

impl Scorer<'_> {
    pub(super) fn embed(&self) -> Result<(), String> {
        let encoder = self.active_encoder();
        let pipelines = self.pipelines;
        let buffers = self.buffers;
        let weights = self.weights;
        let vocabulary = weights.architecture.vocabulary();
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
            std::mem::size_of_val(&vocabulary) as u64,
            std::ptr::from_ref(&vocabulary).cast(),
        );
        let active_rows = self.active_rows();
        if let Some(input) = self.diffusion {
            encoder.set_buffer(3, Some(weights.buffer), weights.mask_embed);
            encoder.set_buffer(4, Some(input.confidence), u64::from(self.row_start) * 4);
            encoder.dispatch_thread_groups(thread_group(u64::from(active_rows)), thread_group(128));
        } else {
            encoder.dispatch_threads(
                thread_group(u64::from(active_rows) * u64::from(WIDTH)),
                thread_group(pipelines.embed.max_total_threads_per_threadgroup().min(256)),
            );
        }
        encoder.memory_barrier_with_resources(&[&buffers.input]);

        if weights.architecture.is_multistream() {
            self.hyper()
                .init_streams(&encoder, self.activations(&buffers.input));
            return Ok(());
        }

        crate::nn::norm::RmsNorm::new(&pipelines.rms, crate::nn::Rows::Chunk(active_rows)).forward(
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

    pub(super) fn attention(&self, layer: u32, execution: u32) -> Result<(), String> {
        self.start_stage("qkv")?;
        let active_rows = self.active_rows();
        let encoder = self.active_encoder();
        let tensorops = self.tensorops;
        let pisa1 = self.pisa1;
        let buffers = self.buffers;
        let weights = self.weights;
        let qkv = self.qkv(execution);
        let activation_offset = self.activation_offset();
        let qkv_offset = self.qkv_offset();
        let (matrix_offset, _) = Self::attention_offsets(layer);
        let position = [self.row_start, self.pisa1.context()];
        let csa_reuse = self.diffusion.is_none()
            && ((self.weights.architecture == ResidualArchitecture::Mhc4
                && matches!(execution, 1 | 3 | 5 | 6))
                || (self.weights.architecture == ResidualArchitecture::HnetMhc4
                    && matches!(execution, 3 | 5)))
            && self
                .layer_cache(execution)
                .is_some_and(|entry| entry.kv.is_some());
        let qkv_tiles = if csa_reuse {
            WIDTH / 128
        } else {
            QKV_WIDTH / 128
        };
        encoder.set_bytes(
            4,
            std::mem::size_of_val(&position) as u64,
            position.as_ptr().cast(),
        );
        tensor_workspace(
            &encoder,
            &tensorops.qkv_rope,
            &[
                (&buffers.normalized, activation_offset),
                (weights.buffer, weights.qkv + matrix_offset),
                (qkv, qkv_offset),
                (&buffers.rope, 0),
            ],
            MTLSize {
                width: u64::from(qkv_tiles),
                height: u64::from(active_rows / 128),
                depth: 1,
            },
            8,
        )?;
        encoder.memory_barrier_with_resources(&[qkv]);
        if let Some(cache) = self
            .layer_cache(execution)
            .filter(|entry| entry.kv.is_some())
        {
            self.start_stage("pisa_attention")?;
            let encoder = self.active_encoder();
            if let Some(input) = self.diffusion {
                let step = crate::forward_program::RecurrentCore::selective_fbt()
                    .layer_visits(MODEL_LAYERS as usize, input.visits as usize)?
                    [execution as usize];
                let last_cache = self.cache_only && layer + 1 == MODEL_LAYERS;
                pisa1.denoise(
                    &encoder,
                    qkv,
                    cache.kv.as_ref().unwrap(),
                    &cache.pyramid,
                    self.row_start,
                    active_rows,
                    crate::context_metal::IndexPolicy {
                        weights: weights.buffer,
                        offset: weights.index_query,
                        block: input.config.block,
                        mode: input.config.index,
                        layer,
                        fresh: input.fresh && step.visit == 0,
                        reuse: input.config.reuse,
                    },
                    !last_cache,
                )?;
                if last_cache {
                    return Ok(());
                }
                return self.project_attention(layer);
            }
            let anchor = match execution {
                1 => 0,
                3 => 2,
                5 | 6 => 4,
                _ => execution,
            };
            let source = if csa_reuse {
                self.layer_cache(anchor).unwrap_or(cache)
            } else {
                cache
            };
            pisa1.cached_csa(
                &encoder,
                qkv,
                source.kv.as_ref().unwrap(),
                &source.pyramid,
                self.row_start,
                active_rows,
                csa_reuse,
            )?;
            return self.project_attention(layer);
        }
        self.start_stage("pisa_pyramid")?;
        let encoder = self.active_encoder();
        if let Some(cache) = self.layer_cache(execution) {
            if self.dirty || active_rows < pisa1.context() {
                pisa1.pyramid_range(&encoder, qkv, &cache.pyramid, self.row_start, active_rows);
            } else {
                pisa1.pyramid_output(&encoder, qkv, &cache.pyramid, pisa1.context());
            }
        } else {
            pisa1.pyramid_layer(&encoder, qkv, active_rows);
        }
        self.start_stage("pisa_attention")?;
        let encoder = self.active_encoder();
        if let Some(cache) = self.layer_cache(execution) {
            pisa1.attention_from(&encoder, qkv, &cache.pyramid, self.row_start, active_rows);
        } else {
            pisa1.attention_range(&encoder, qkv, self.row_start, active_rows);
        }
        self.project_attention(layer)
    }

    fn project_attention(&self, layer: u32) -> Result<(), String> {
        self.start_stage("projection")?;
        let encoder = self.active_encoder();
        self.attention_layer().forward_output(
            &encoder,
            layer,
            self.activations(self.pisa1.output()),
            self.activations(&self.buffers.projected).out(),
        )
    }

    fn feed_forward(&self, input: &BufferRef, layer: u32, pass: u32) -> Result<(), String> {
        self.start_stage("pre_ffn")?;
        let encoder = self.active_encoder();
        let pipelines = self.pipelines;
        let buffers = self.buffers;
        let weights = self.weights;
        crate::nn::norm::ResidualRms::new(
            &pipelines.residual_rms,
            crate::nn::Rows::Chunk(self.active_rows()),
        )
        .forward(
            &encoder,
            [
                self.activations(input),
                self.activations(&buffers.projected),
            ],
            crate::tensor::View::slice(
                weights.buffer,
                weights.ffn_norm + half_bytes(u64::from(layer) * u64::from(WIDTH)),
                1,
                WIDTH,
            ),
            [
                self.activations(&buffers.attention_state).out(),
                self.activations(&buffers.normalized).out(),
            ],
        );
        let next_norm = self.next_norm(layer, pass);
        self.feed_mhc(layer)?;
        self.start_stage("combine_post_ffn")?;
        self.moe()?.combine_residual(
            self.active_encoder(),
            self.activations(&buffers.attention_state),
            crate::tensor::View::slice(next_norm.0, next_norm.1, 1, WIDTH),
            [
                self.activations(&buffers.output).out(),
                self.activations(&buffers.normalized).out(),
            ],
        );
        Ok(())
    }

    pub(super) fn feedback(&self, input: &BufferRef) -> Result<(), String> {
        let encoder = self.active_encoder();
        let pipelines = self.pipelines;
        let buffers = self.buffers;
        let weights = self.weights;
        let wide = crate::nn::linear::TileLinear::new(&self.tensorops.output_projection_wide, 8)?;
        let mut wave = crate::tensor::Wave::new(&encoder);
        wide.forward_wave(
            &mut wave,
            self.activations(input),
            crate::tensor::View::slice(weights.buffer, weights.feedback_state, WIDTH, WIDTH),
            self.activations(&buffers.feedback_state).out(),
        );
        wide.forward_wave(
            &mut wave,
            self.activations(&buffers.normalized),
            crate::tensor::View::slice(weights.buffer, weights.feedback_gate, WIDTH, WIDTH),
            self.activations(&buffers.feedback_gate).out(),
        );
        wave.flush();
        crate::nn::norm::ResidualRms::new(
            &pipelines.feedback_fuse_rms,
            crate::nn::Rows::Chunk(self.active_rows()),
        )
        .forward(
            &encoder,
            [
                self.activations(&buffers.feedback_state),
                self.activations(&buffers.feedback_gate),
            ],
            crate::tensor::View::slice(weights.buffer, weights.attention_norm, 1, WIDTH),
            [
                self.activations(&buffers.feedback).out(),
                self.activations(&buffers.normalized).out(),
            ],
        );
        Ok(())
    }
}
