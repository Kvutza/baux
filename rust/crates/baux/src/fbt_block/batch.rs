use super::*;

fn subwindow_observation(
    losses: &[f32],
    quality: &decode::TargetQuality,
) -> Result<crate::objective_observation::ObjectiveObservation, String> {
    use crate::objective_observation::{MAX_OBJECTIVES, ObjectiveEstimate, ObjectiveObservation};
    let total_len = losses.len();
    let channels = MAX_OBJECTIVES.min(total_len.max(1));
    let mut estimates = Vec::with_capacity(channels);
    for b in 0..channels {
        let start = (b * total_len) / channels;
        let end = ((b + 1) * total_len) / channels;
        let slice = &losses[start..end];
        let (mean, var) = if slice.is_empty() {
            (
                f64::from(quality.mean_nll),
                f64::from(quality.mean_nll_variance),
            )
        } else {
            let n = slice.len() as f64;
            let sum: f64 = slice.iter().map(|&x| f64::from(x)).sum();
            let mean = sum / n;
            let var = if slice.len() > 1 {
                let sq: f64 = slice.iter().map(|&x| (f64::from(x) - mean).powi(2)).sum();
                (sq / (n * (n - 1.0))).max(0.0)
            } else {
                0.0
            };
            (mean, var)
        };
        estimates.push(ObjectiveEstimate {
            mean: -(mean as f32),
            variance: (var as f32).max(0.0),
        });
    }
    ObjectiveObservation::new(
        &estimates,
        ObjectiveEstimate {
            mean: -quality.mean_nll,
            variance: quality.mean_nll_variance.max(0.0),
        },
    )
}

fn paired_gains(
    baseline: &[f32],
    losses: &[f32],
    patch_size: usize,
    quality: &mut decode::TargetQuality,
) -> Vec<f32> {
    let total_len = losses.len();
    let mut diff_sum = 0.0f64;
    let mut diff_sq = 0.0f64;
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
    let patches = total_len.div_ceil(patch_size);
    let mut gains = Vec::with_capacity(patches);
    for chunk in (0..total_len).step_by(patch_size) {
        let end = (chunk + patch_size).min(total_len);
        let patch_sum: f32 = (chunk..end).map(|idx| baseline[idx] - losses[idx]).sum();
        gains.push(patch_sum / (end - chunk) as f32);
    }
    gains
}

impl BlockDecoder {
    fn encode_waves(
        &self,
        command: &CommandBufferRef,
        weights: CandidateRow<'_>,
        byte_offset: u64,
        active_rows: u32,
        score: u32,
        update_centroids: bool,
    ) -> Result<(), String> {
        self.waves.encode_mhc(
            command,
            (&self.buffers.normalized, 0),
            (weights.buffer, weights.qkv),
            (weights.buffer, weights.readout),
            (&self.buffers.labels, byte_offset),
            (&self.proposals, byte_offset),
            (&self.buffers.losses, byte_offset),
            active_rows,
            score,
            update_centroids,
        )
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
        let update_centroids = row_start == 0;
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
            self.encode_waves(
                wave_cmd,
                weights,
                byte_offset,
                active_rows,
                score,
                update_centroids,
            )?;
            return Ok(traced_seconds + complete(wave_cmd)?);
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
        self.encode_waves(
            &command,
            weights,
            byte_offset,
            active_rows,
            score,
            update_centroids,
        )?;
        command.commit();
        complete_committed(&command)
    }

    /// Score one fixed corpus sequence under its causal next-token likelihood.
    ///
    /// No sampled token is fed into the scored state.
    pub(in crate::fbt_moe) fn score_batch(
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
    pub(in crate::fbt_moe) fn score_objective(
        &self,
        runtime: &Runtime,
        decoder: &decode::Decoder,
        weights: CandidateRow<'_>,
        batch: &crate::pretrain_data::CausalBatch,
        objective: crate::config::ObjectiveFunction,
    ) -> Result<CausalScore, String> {
        if !weights.architecture.is_mhc4() {
            return Err("long causal scoring requires the MHC4 architecture".into());
        }
        if batch.labels.len() != batch.tokens.len() {
            return Err("causal batch token and label lengths do not match".into());
        }
        let total_rows = u32::try_from(batch.tokens.len()).map_err(|error| error.to_string())?;
        if total_rows == 0 || total_rows > self.context {
            return Err("causal batch length must be positive and within scorer context".into());
        }
        let total_len = total_rows as usize;
        let chunk_cap = ROWS.min(self.context);
        let tokens = crate::apple_gpu::slice_mut::<u32>(&self.buffers.tokens, total_len)?;
        tokens.copy_from_slice(&batch.tokens[..total_len]);
        let labels = crate::apple_gpu::slice_mut::<u32>(&self.buffers.labels, total_len)?;
        labels.copy_from_slice(&batch.labels[..total_len]);
        let seeds = crate::apple_gpu::slice_mut::<u64>(&self.seeds, 1)?;
        seeds[0] = crate::hash::splitmix64(batch.offset as u64);

        let mut forward_seconds = 0.0;
        let mut row_start = 0u32;
        let score = objective.score_mode();
        while row_start < total_rows {
            let active_rows = (total_rows - row_start).min(chunk_cap);
            forward_seconds +=
                self.score_chunk(runtime, decoder, weights, row_start, active_rows, score)?;
            row_start += active_rows;
        }
        let losses = crate::apple_gpu::buffer_slice::<f32>(&self.buffers.losses, total_len)?;
        let actual = crate::apple_gpu::buffer_slice::<u32>(&self.proposals, total_len)?;
        let target_labels = &batch.labels[..total_len];
        let mut stats = target::TargetStats::new(total_len);
        for (i, (&loss, (&pred, &exp))) in losses
            .iter()
            .zip(actual.iter().zip(target_labels.iter()))
            .enumerate()
        {
            stats.push(i, loss, pred == exp);
        }
        let mut quality = stats.finish()?;
        if !baux::semantics::optimizer::verify_full(
            u64::from(total_rows),
            0,
            quality.scored_tokens as u64,
        ) {
            return Err("causal batch score failed full window coverage check".into());
        }
        let patch_gains = if self.paired_losses_ready.get() {
            let baseline = crate::apple_gpu::buffer_slice::<f32>(&self.paired_losses, total_len)?;
            let patch_size = 64.max(weights.architecture.patch_size());
            paired_gains(baseline, losses, patch_size, &mut quality)
        } else {
            Vec::new()
        };
        let observation = subwindow_observation(losses, &quality)?;
        Ok(CausalScore {
            quality,
            gpu_seconds: forward_seconds,
            patch_gains,
            observation,
        })
    }
}
