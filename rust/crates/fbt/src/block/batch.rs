use super::*;

fn subwindow_observation(
    _losses: &[f32],
    quality: &decode::TargetQuality,
    _ratio_penalty: f32,
) -> Result<crate::objective_observation::ObjectiveObservation, String> {
    use crate::objective_observation::ObjectiveObservation;
    // Replace 8-octile temporal multi-objective slicing with single scalar NLL
    // observation. Minimizing NLL corresponds to maximizing -NLL (with
    // ratio_penalty included in quality.mean_nll).
    ObjectiveObservation::scalar(-quality.mean_nll, quality.mean_nll_variance.max(0.0))
}

fn paired_gains(
    baseline: &[f32],
    losses: &[f32],
    patch_size: usize,
    quality: &mut decode::TargetQuality,
) -> Vec<f32> {
    let total_len = losses.len();
    let (mut diff_sum, mut diff_sq) = (0.0f64, 0.0f64);
    for i in 0..total_len {
        let diff = f64::from(baseline[i] - losses[i]);
        diff_sum += diff;
        diff_sq += diff * diff;
    }
    let n = total_len as f64;
    let mean_diff = diff_sum / n;
    let var_diff = if n > 1.0 {
        (diff_sq - diff_sum * mean_diff) / (n * (n - 1.0))
    } else {
        0.0
    };
    quality.paired_improvement = Some(mean_diff as f32);
    quality.paired_improvement_variance = Some(var_diff as f32);
    (0..total_len)
        .step_by(patch_size)
        .map(|chunk| {
            let end = (chunk + patch_size).min(total_len);
            let sum: f32 = (chunk..end).map(|idx| baseline[idx] - losses[idx]).sum();
            sum / (end - chunk) as f32
        })
        .collect()
}

fn encode_hnet(
    decoder: &BlockDecoder,
    command: &CommandBufferRef,
    weights: CandidateRow<'_>,
    byte_offset: u64,
    active_rows: u32,
    score: u32,
) -> Result<(), String> {
    let encoder = command.new_compute_command_encoder();
    let waves = &decoder.buffers.hnet()?.waves;
    encoder.set_compute_pipeline_state(&waves.mhc_readout);
    crate::apple_gpu::bind_buffers(
        &encoder,
        &[
            (&decoder.buffers.normalized, 0),
            (weights.buffer, weights.readout),
            (&decoder.buffers.labels, byte_offset),
            (&decoder.proposals, byte_offset),
            (&decoder.buffers.losses, byte_offset),
        ],
    );
    let shape = baux::semantics::hnet::HNetShape {
        batch_size: 1,
        seq_len: active_rows,
        d_model: WIDTH,
        max_chunks: active_rows,
        vocab_size: 256,
        eps: 1e-4,
        target_ratio: 4.0,
        score,
    };
    crate::apple_gpu::encode_value(&encoder, 5, &shape);
    encoder.dispatch_thread_groups(thread_group(u64::from(active_rows)), thread_group(32));
    encoder.end_encoding();
    Ok(())
}

impl BlockDecoder {
    pub(super) fn encode_waves(
        &self,
        command: &CommandBufferRef,
        weights: CandidateRow<'_>,
        byte_offset: u64,
        active_rows: u32,
        score: u32,
    ) -> Result<(), String> {
        if weights.architecture.is_hnet() {
            return encode_hnet(self, command, weights, byte_offset, active_rows, score);
        }
        let loss_partials = self.buffers.loss_partials()?;
        let proposal_partials = self.buffers.proposal_partials()?;
        let vocab = weights.architecture.vocabulary();
        let tiles = vocab / 64;
        let row_start = (byte_offset / std::mem::size_of::<u32>() as u64) as u32;
        let shape = [active_rows, vocab, row_start, self.context];
        let score_targets = score != 0;
        let temperature = 0.0f32;

        let encoder = command.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.tensorops.readout_proposal_tiles);
        encoder.set_buffer(0, Some(&self.buffers.normalized), 0);
        encoder.set_buffer(1, Some(weights.buffer), weights.readout);
        encoder.set_buffer(2, Some(loss_partials), 0);
        crate::apple_gpu::encode_value(&encoder, 3, &shape);
        encoder.set_buffer(4, Some(&self.seeds), 0);
        crate::apple_gpu::encode_value(&encoder, 5, &temperature);
        encoder.set_buffer(6, Some(&self.buffers.labels), 0);
        encoder.set_buffer(7, Some(proposal_partials), 0);
        crate::apple_gpu::encode_value(&encoder, 8, &score_targets);
        encoder.dispatch_thread_groups(
            MTLSize {
                width: u64::from(tiles),
                height: u64::from(active_rows.div_ceil(128)),
                depth: 1,
            },
            thread_group(128),
        );
        encoder.memory_barrier_with_resources(&[loss_partials, proposal_partials]);

        // Reduce proposal predictions into self.proposals at byte_offset
        let proposal_source = if score_targets {
            proposal_partials
        } else {
            loss_partials
        };
        encoder.set_compute_pipeline_state(&self.pipelines.readout_proposal_reduce);
        encoder.set_buffer(0, Some(proposal_source), 0);
        encoder.set_buffer(1, Some(&self.proposals), byte_offset);
        crate::apple_gpu::encode_value(&encoder, 2, &tiles);
        encoder.dispatch_thread_groups(thread_group(u64::from(active_rows)), thread_group(128));
        encoder.memory_barrier_with_resources(&[&self.proposals]);

        if score_targets {
            encoder.set_compute_pipeline_state(&self.pipelines.readout_loss_reduce);
            encoder.set_buffer(0, Some(proposal_partials), 0);
            encoder.set_buffer(1, Some(&self.buffers.losses), byte_offset);
            crate::apple_gpu::encode_value(&encoder, 2, &tiles);
            encoder.dispatch_thread_groups(thread_group(u64::from(active_rows)), thread_group(128));
            encoder.memory_barrier_with_resources(&[&self.buffers.losses]);
        }

        encoder.end_encoding();
        Ok(())
    }

    fn score_chunk(
        &self,
        runtime: &Runtime,
        decoder: &decode::Decoder,
        weights: CandidateRow<'_>,
        row_start: u32,
        active_rows: u32,
        score: u32,
    ) -> Result<f64, String> {
        let byte_offset = u64::from(row_start) * size_of::<u32>() as u64;
        let command = runtime.queue.new_command_buffer().to_owned();
        let traced = if decoder.cache()[0].kv.is_some() {
            self.profile_cached(
                &command,
                runtime,
                decoder,
                weights,
                0.0,
                crate::config::FeedbackTransition::Identity,
                true,
                row_start,
                active_rows,
                true,
            )?
        } else {
            self.profile_full(
                &command,
                runtime,
                decoder,
                weights,
                0.0,
                crate::config::FeedbackTransition::Identity,
                0,
                active_rows,
                true,
                true,
            )?
        };
        if let Some(traced_seconds) = traced {
            let wave_cmd = runtime.queue.new_command_buffer();
            self.encode_waves(wave_cmd, weights, byte_offset, active_rows, score)?;
            let wave_seconds = complete(wave_cmd)?;
            if weights.architecture.is_hnet() {
                let raw_chunks =
                    crate::apple_gpu::buffer_slice::<u32>(&self.buffers.hnet()?.count, 1)?[0];
                let aligned_rows = active_rows.max(1).div_ceil(128) * 128;
                let compact_rows = (raw_chunks.max(1).div_ceil(128) * 128).min(aligned_rows);
                self.buffers.hnet()?.record_stats(
                    active_rows,
                    self.buffers.rows,
                    compact_rows,
                    traced_seconds,
                    wave_seconds,
                )?;
            }
            return Ok(traced_seconds + wave_seconds);
        }
        if weights.architecture.is_hnet() && decoder.cache()[0].kv.is_none() && row_start == 0 {
            scorer::hnet_stage0(
                &command,
                &self.pipelines,
                &self.tensorops,
                &self.pisa,
                &self.buffers,
                weights,
                active_rows,
                Some(decoder.cache()),
            )?;
            command.commit();
            let stage_a_seconds = complete_committed(&command)?;
            let raw_chunks =
                crate::apple_gpu::buffer_slice::<u32>(&self.buffers.hnet()?.count, 1)?[0];
            let aligned_rows = active_rows.max(1).div_ceil(128) * 128;
            let compact_rows = (raw_chunks.max(1).div_ceil(128) * 128).min(aligned_rows);
            assert!(
                compact_rows <= self.buffers.rows,
                "compacted H-Net rows {compact_rows} exceed capacity {}",
                self.buffers.rows
            );
            let cmd_b = runtime.queue.new_command_buffer().to_owned();
            scorer::hnet_stage1(
                &cmd_b,
                &self.pipelines,
                &self.tensorops,
                &self.pisa,
                &self.buffers,
                weights,
                active_rows,
                compact_rows,
                Some(decoder.cache()),
            )?;
            self.encode_waves(&cmd_b, weights, byte_offset, active_rows, score)?;
            cmd_b.commit();
            let stage_b_seconds = complete_committed(&cmd_b)?;
            self.buffers.hnet()?.record_stats(
                active_rows,
                self.buffers.rows,
                compact_rows,
                stage_a_seconds,
                stage_b_seconds,
            )?;
            return Ok(stage_a_seconds + stage_b_seconds);
        }
        let readout = self.readout(0.0, crate::config::FeedbackTransition::Identity, true);
        if decoder.cache()[0].kv.is_some() {
            scorer::context_chunk(
                &command,
                &self.pipelines,
                &self.tensorops,
                &self.pisa,
                &self.buffers,
                weights,
                readout,
                row_start,
                active_rows,
                decoder.cache(),
                true,
                None,
            )?;
        } else {
            scorer::encode_hidden(
                &command,
                &self.pipelines,
                &self.tensorops,
                &self.pisa,
                &self.buffers,
                weights,
                readout,
                active_rows,
                Some(decoder.cache()),
                None,
            )?;
        }
        self.encode_waves(&command, weights, byte_offset, active_rows, score)?;
        command.commit();
        complete_committed(&command)
    }

    fn forward_batch(
        &self,
        decoder: &decode::Decoder,
        weights: CandidateRow<'_>,
        total_rows: u32,
        score: u32,
    ) -> Result<(f64, f32), String> {
        let use_hnet2 = weights.architecture.is_hnet()
            && (self.context > CONTEXT || self.buffers.hnet2.get().is_some());
        if use_hnet2 {
            let sec = self.score_hnet2(decoder, weights, total_rows, score)?;
            let pen = if score != 0 {
                self.buffers.hnet2()?.last_stats.get().total_penalty
            } else {
                0.0
            };
            return Ok((sec, pen));
        }
        let runtime = Runtime::shared()?;
        if weights.architecture.is_hnet() {
            self.buffers.hnet()?.reset_sequence()?;
        }
        let chunk_cap = ROWS.min(self.context);
        let (mut forward_seconds, mut row_start) = (0.0, 0u32);
        while row_start < total_rows {
            let active_rows = (total_rows - row_start).min(chunk_cap);
            forward_seconds +=
                self.score_chunk(&runtime, decoder, weights, row_start, active_rows, score)?;
            row_start += active_rows;
        }
        let pen = if weights.architecture.is_hnet() && score != 0 {
            self.buffers.hnet()?.last_stats.get().ratio_penalty
        } else {
            0.0
        };
        Ok((forward_seconds, pen))
    }

    /// Score one fixed corpus sequence under its causal next-token likelihood.
    ///
    /// No sampled token is fed into the scored state.
    pub(in crate::moe) fn score_batch(
        &self,
        runtime: &Runtime,
        decoder: &decode::Decoder,
        weights: CandidateRow<'_>,
        batch: &crate::pretrain_data::CausalBatch,
    ) -> Result<CausalScore, String> {
        self.score_objective(
            runtime,
            decoder,
            weights,
            batch,
            crate::config::ObjectiveFunction::CrossEntropy,
        )
    }

    /// Score one fixed corpus sequence under the configured token loss
    /// function.
    pub(in crate::moe) fn score_objective(
        &self,
        runtime: &Runtime,
        decoder: &decode::Decoder,
        weights: CandidateRow<'_>,
        batch: &crate::pretrain_data::CausalBatch,
        objective: crate::config::ObjectiveFunction,
    ) -> Result<CausalScore, String> {
        if !weights.architecture.is_mhc4() {
            return Err("long causal scoring requires an MHC4-family architecture".into());
        }
        if batch.labels.len() != batch.tokens.len() {
            return Err("causal batch token and label lengths do not match".into());
        }
        if batch
            .tokens
            .iter()
            .chain(&batch.labels)
            .any(|&value| value >= weights.architecture.vocabulary())
        {
            return Err("corpus IDs exceed the model vocabulary".into());
        }
        let total_rows = u32::try_from(batch.tokens.len()).map_err(|error| error.to_string())?;
        if total_rows == 0 || total_rows > self.context {
            return Err("causal batch length must be positive and within scorer context".into());
        }
        let _ = runtime;
        let total_len = total_rows as usize;
        crate::apple_gpu::slice_mut::<u32>(&self.buffers.tokens, total_len)?
            .copy_from_slice(&batch.tokens[..total_len]);
        crate::apple_gpu::slice_mut::<u32>(&self.buffers.labels, total_len)?
            .copy_from_slice(&batch.labels[..total_len]);
        crate::apple_gpu::slice_mut::<u64>(&self.seeds, 1)?[0] =
            crate::hash::splitmix64(batch.offset as u64);

        let score = objective.score_mode();
        let (forward_seconds, ratio_penalty) =
            self.forward_batch(decoder, weights, total_rows, score)?;
        let losses = crate::apple_gpu::buffer_slice::<f32>(&self.buffers.losses, total_len)?;
        let actual = crate::apple_gpu::buffer_slice::<u32>(&self.proposals, total_len)?;
        let mut stats = target::TargetStats::new(total_len);
        for (i, (&loss, (&pred, &exp))) in losses
            .iter()
            .zip(actual.iter().zip(batch.labels[..total_len].iter()))
            .enumerate()
        {
            stats.push(i, loss, pred == exp);
        }
        let mut quality = stats.finish()?;
        if ratio_penalty != 0.0 {
            quality.mean_nll += ratio_penalty;
            quality.codelength_bits = f64::from(quality.mean_nll)
                * (quality.scored_tokens as f64)
                * std::f64::consts::LOG2_E;
        }
        let planned = u64::from(total_rows);
        let prefilled = 0u64;
        let scored = quality.scored_tokens as u64;
        if !(planned > 0 && prefilled == 0 && scored == planned) {
            return Err("causal batch score failed full window coverage check".into());
        }
        let patch_gains = if self.paired_losses_ready.get() {
            let baseline = crate::apple_gpu::buffer_slice::<f32>(&self.paired_losses, total_len)?;
            paired_gains(
                baseline,
                losses,
                64.max(weights.architecture.patch_size()),
                &mut quality,
            )
        } else {
            Vec::new()
        };
        let observation = subwindow_observation(losses, &quality, ratio_penalty)?;
        Ok(CausalScore {
            quality,
            gpu_seconds: forward_seconds,
            patch_gains,
            observation,
        })
    }
}
