use super::block_decode::BlockDecoder;
use super::generation::GenerationTask;
use super::stats::RepairStats;
use super::*;
use crate::config::GenerationConfig;

pub(super) fn score_targets(config: &GenerationConfig) -> bool {
    matches!(
        config.reward,
        crate::config::GenerationReward::ConditionalCrossEntropy
            | crate::config::GenerationReward::CausalObjectives
            | crate::config::GenerationReward::BoundedCrossEntropy
            | crate::config::GenerationReward::CodeObjectives { .. }
    )
}

pub(super) fn loss_window(config: &GenerationConfig) -> usize {
    match &config.reward {
        crate::config::GenerationReward::CodeObjectives { critical_window } => {
            *critical_window as usize
        }
        _ => config.max_tokens as usize,
    }
}

pub(super) fn route_sample(stats: routing::RouteStats) -> decode::RouteSample {
    decode::RouteSample {
        active_experts: stats.active_experts,
        routed_rows: stats.routed_rows,
        routed_tiles: stats.routed_tiles,
        expert_mask: stats.expert_mask,
    }
}

pub(super) fn readout_slice<'a>(weights: CandidateRow<'a>, patch: usize) -> Option<&'a [u16]> {
    if patch <= 1 || weights.buffer.contents().is_null() {
        return None;
    }
    let elements = (WIDTH * VOCAB) as usize;
    let offset = weights.readout as usize;
    if offset + elements * size_of::<u16>() > weights.buffer.length() as usize {
        return None;
    }
    unsafe {
        Some(std::slice::from_raw_parts(
            weights
                .buffer
                .contents()
                .cast::<u8>()
                .add(offset)
                .cast::<u16>(),
            elements,
        ))
    }
}

pub(super) fn patch_token(
    macro_token: u32,
    previous: u32,
    intra_patch: usize,
    readout: Option<&[u16]>,
) -> u32 {
    let weight = readout.map_or(0, |values| {
        let dimension = (intra_patch * 17) % WIDTH as usize;
        values[dimension * VOCAB as usize + previous as usize % VOCAB as usize]
    });
    let mixed = crate::hash::splitmix64(
        (u64::from(previous) << 32)
            ^ (u64::from(macro_token) << 16)
            ^ (u64::from(intra_patch as u32) << 8)
            ^ u64::from(weight),
    );
    (mixed % u64::from(VOCAB)) as u32
}

pub(super) fn repair_bounds(
    row: usize,
    context: u32,
    window: u32,
    prompt: usize,
    limit: usize,
) -> (u32, usize) {
    let start = ((row as u32 / 4) * 4).min(context - window);
    let end = (start as usize + window as usize)
        .saturating_sub(prompt - 1)
        .min(limit);
    (start, end)
}

pub(super) fn adapt_window(
    window: u32,
    config: &GenerationConfig,
    mismatch: bool,
    accepted: usize,
    span: usize,
) -> u32 {
    if !mismatch || (span > 0 && accepted * 2 >= span) {
        (window * 2).min(config.verify.max_window)
    } else if span > 0 && accepted * 4 < span {
        128
    } else {
        (window / 2).max(128)
    }
}

pub(super) fn next_window(
    window: u32,
    config: &GenerationConfig,
    mismatch: bool,
    accepted: usize,
    span: usize,
) -> (bool, u32) {
    (
        mismatch && accepted * 4 < span,
        adapt_window(window, config, mismatch, accepted, span),
    )
}

pub(super) fn readout_tile(position: u32, total: u32, prediction: u32) -> (u32, bool) {
    let before = prediction.saturating_sub(position);
    let rows = if before >= 128 {
        before.min(ROWS) / 128 * 128
    } else {
        (total - position).min(ROWS).div_ceil(128) * 128
    };
    (rows, position + rows <= prediction)
}

pub(super) fn expand_draft(
    draft: &decode::Rollout,
    maximum: usize,
    eos: Option<u32>,
) -> Result<Vec<u32>, String> {
    if draft.tokens.len() >= maximum {
        return Ok(draft.tokens[..maximum].to_vec());
    }
    if let Some(eos) = eos {
        if draft.tokens.last() != Some(&eos) {
            return Err("short incumbent rollout does not end in EOS".into());
        }
        let mut tokens = draft.tokens.clone();
        tokens.resize(maximum, eos);
        return Ok(tokens);
    }
    if draft.tokens.is_empty() {
        return Err("incumbent rollout has an invalid generated length".into());
    }
    // Full-context corpus windows have different prompt/output partitions.
    // The previous accepted trajectory is only the parallel solver's initial
    // state; tile it to the new extent instead of performing a second model
    // evaluation solely to manufacture a length-matched seed.
    let mut tokens = draft.tokens.clone();
    let seed = tokens.clone();
    tokens.extend((tokens.len()..maximum).map(|index| seed[index % seed.len()]));
    Ok(tokens)
}

pub(super) fn refresh_draft(
    tokens: &mut [u32],
    input: &[u32],
    cursor: usize,
    tile_end: usize,
    prompt: usize,
) {
    for position in cursor..tile_end.saturating_sub(1) {
        tokens[position] = input[prompt + position];
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn commit_tile(
    task: &GenerationTask,
    tokens: &mut [u32],
    input: &mut [u32],
    proposed: &[u32],
    exact_start: usize,
    tile_end: usize,
    context: usize,
    patch: usize,
    eos: Option<u32>,
    stats: &mut RepairStats,
) -> Result<(usize, bool), String> {
    let mut position = exact_start;
    let mut committed = None;
    let mut accepted = 0usize;
    while position < tile_end {
        let prediction = proposed[task.prompt.len() - 1 + position];
        if prediction >= VOCAB {
            return Err("block repair produced an invalid token".into());
        }
        let exact_prediction = committed.is_none();
        if exact_prediction {
            if tokens[position] == prediction {
                accepted += 1;
            } else {
                stats.first_mismatch.get_or_insert(position);
                let patch_end = if patch > 1 {
                    ((position / patch) + 1) * patch
                } else {
                    position + 1
                };
                committed = Some(patch_end.min(tile_end));
            }
        }
        tokens[position] = prediction;
        if task.prompt.len() + position < context {
            input[task.prompt.len() + position] = prediction;
        }
        position += 1;
        if exact_prediction && Some(prediction) == eos {
            tokens[position..].fill(prediction);
            stats.accepted_lengths.push(accepted);
            stats.committed_lengths.push(position - exact_start);
            return Ok((tokens.len(), true));
        }
    }
    let cursor = committed.unwrap_or(tile_end);
    stats.accepted_lengths.push(accepted);
    stats.committed_lengths.push(cursor - exact_start);
    Ok((cursor, false))
}
impl BlockDecoder {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn parallel_pass(
        &self,
        runtime: &Runtime,
        decoder: &decode::Decoder,
        weights: CandidateRow<'_>,
        temperature: f32,
        feedback: crate::config::FeedbackTransition,
        pass: usize,
        cursor: usize,
        total_rows: u32,
        score_targets: bool,
        prompt: usize,
    ) -> Result<(f64, u32, decode::RouteSample), String> {
        if decoder.cache()[0].kv.is_some() {
            return self.cached_pass(
                runtime,
                decoder,
                weights,
                temperature,
                feedback,
                pass,
                cursor,
                total_rows,
                score_targets,
                prompt,
            );
        }
        self.full_pass(
            runtime,
            decoder,
            weights,
            temperature,
            feedback,
            pass,
            cursor,
            total_rows,
            score_targets,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn cached_pass(
        &self,
        runtime: &Runtime,
        decoder: &decode::Decoder,
        weights: CandidateRow<'_>,
        temperature: f32,
        feedback: crate::config::FeedbackTransition,
        pass: usize,
        cursor: usize,
        total_rows: u32,
        score_targets: bool,
        prompt: usize,
    ) -> Result<(f64, u32, decode::RouteSample), String> {
        let patch = weights.architecture.patch_size() as u32;
        let start = if pass == 0 {
            0
        } else {
            (((prompt - 1 + cursor) as u32 / 4) * 4) / patch
        };
        let mut position = start;
        let mut seconds = 0.0;
        let mut evaluated = 0;
        let prediction_start = (prompt as u32 / patch).saturating_sub(1);
        while position < total_rows {
            let (rows, hidden_only) = readout_tile(position, total_rows, prediction_start);
            let row_start = position.min(self.context - rows);
            let command = runtime.queue.new_command_buffer().to_owned();
            if let Some(device_seconds) = self.profile_cached(
                &command,
                runtime,
                decoder,
                weights,
                temperature,
                feedback,
                score_targets,
                row_start,
                rows,
                hidden_only,
            )? {
                seconds += device_seconds;
            } else {
                scorer::context_chunk(
                    &command,
                    &self.pipelines,
                    &self.tensorops,
                    &self.pisa,
                    &self.buffers,
                    weights,
                    self.readout(temperature, feedback, score_targets),
                    row_start,
                    rows,
                    decoder.cache(),
                    hidden_only,
                    None,
                )?;
                command.commit();
                // Compact chunks reuse all activation/routing scratch at offset
                // zero. Complete each owner before submitting the next chunk.
                seconds += complete_committed(&command)?;
            }
            evaluated += rows;
            position += rows;
        }
        Ok((seconds, evaluated, self.route()))
    }

    #[allow(clippy::too_many_arguments)]
    fn full_pass(
        &self,
        runtime: &Runtime,
        decoder: &decode::Decoder,
        weights: CandidateRow<'_>,
        temperature: f32,
        feedback: crate::config::FeedbackTransition,
        pass: usize,
        cursor: usize,
        total_rows: u32,
        score_targets: bool,
    ) -> Result<(f64, u32, decode::RouteSample), String> {
        let command = runtime.queue.new_command_buffer();
        if let Some(device_seconds) = self.profile_full(
            command,
            runtime,
            decoder,
            weights,
            temperature,
            feedback,
            pass,
            total_rows,
            score_targets,
            false,
        )? {
            return Ok((device_seconds, total_rows, self.route()));
        }
        let readout = self.readout(temperature, feedback, score_targets);
        let active_rows = if pass == 0 {
            scorer::encode_proposals(
                command,
                &self.pipelines,
                &self.tensorops,
                &self.pisa,
                &self.buffers,
                weights,
                readout,
                total_rows,
                Some(decoder.cache()),
                None,
            )?;
            total_rows
        } else {
            let row_start = (cursor as u32 / 128) * 128;
            let rows = total_rows - row_start;
            scorer::suffix_proposals(
                command,
                &self.pipelines,
                &self.tensorops,
                &self.pisa,
                &self.buffers,
                weights,
                readout,
                row_start,
                rows,
                decoder.cache(),
            )?;
            rows
        };
        Ok((complete(command)?, active_rows, self.route()))
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn profile_cached(
        &self,
        command: &CommandBufferRef,
        runtime: &Runtime,
        decoder: &decode::Decoder,
        weights: CandidateRow<'_>,
        temperature: f32,
        feedback: crate::config::FeedbackTransition,
        score_targets: bool,
        row_start: u32,
        rows: u32,
        hidden_only: bool,
    ) -> Result<Option<f64>, String> {
        if !self.tracing() {
            return Ok(None);
        }
        let profile = self.profile.borrow();
        let trace = &profile.as_ref().unwrap().trace;
        let (seconds, stages) = scorer::profile_chunk(
            command,
            runtime,
            &self.pipelines,
            &self.tensorops,
            &self.pisa,
            &self.buffers,
            weights,
            self.readout(temperature, feedback, score_targets),
            decoder.cache(),
            trace,
            scorer::TracePlan {
                start: row_start,
                rows,
                hidden_only,
            },
        )?;
        drop(profile);
        self.finish_profile(row_start, rows, hidden_only, seconds, stages)?;
        Ok(Some(seconds))
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn profile_full(
        &self,
        command: &CommandBufferRef,
        runtime: &Runtime,
        decoder: &decode::Decoder,
        weights: CandidateRow<'_>,
        temperature: f32,
        feedback: crate::config::FeedbackTransition,
        pass: usize,
        rows: u32,
        score_targets: bool,
        hidden_only: bool,
    ) -> Result<Option<f64>, String> {
        if pass != 0 || !self.tracing() {
            return Ok(None);
        }
        let profile = self.profile.borrow();
        let trace = &profile.as_ref().unwrap().trace;
        let (seconds, stages) = scorer::profile_proposals(
            command,
            runtime,
            &self.pipelines,
            &self.tensorops,
            &self.pisa,
            &self.buffers,
            weights,
            self.readout(temperature, feedback, score_targets),
            rows,
            Some(decoder.cache()),
            hidden_only,
            trace,
        )?;
        drop(profile);
        self.finish_profile(0, rows, hidden_only, seconds, stages)?;
        Ok(Some(seconds))
    }

    pub(super) fn tracing(&self) -> bool {
        self.profile
            .borrow()
            .as_ref()
            .is_some_and(|profile| profile.remaining > 0)
    }

    pub(super) fn readout(
        &self,
        temperature: f32,
        feedback: crate::config::FeedbackTransition,
        score_targets: bool,
    ) -> scorer::ProposalReadout<'_> {
        scorer::ProposalReadout {
            output: &self.proposals,
            hidden: Some(&self.coarse),
            seeds: &self.seeds,
            temperature,
            score_targets,
            feedback,
        }
    }

    pub(super) fn route(&self) -> decode::RouteSample {
        route_sample(self.buffers.fine_grained.route_stats())
    }

    pub(super) fn prefill_prefix(
        &self,
        runtime: &Runtime,
        decoder: &decode::Decoder,
        weights: CandidateRow<'_>,
        batch_tokens: &[u32],
    ) -> Result<f64, String> {
        let _ = (runtime, decoder, weights);
        let total = u32::try_from(batch_tokens.len()).map_err(|error| error.to_string())?;
        if total > self.context {
            return Err("causal batch exceeds scorer context capacity".into());
        }
        Ok(0.0)
    }

    fn finish_profile(
        &self,
        row_start: u32,
        rows: u32,
        hidden_only: bool,
        seconds: f64,
        stages: std::collections::BTreeMap<&'static str, f64>,
    ) -> Result<(), String> {
        self.profile.borrow_mut().as_mut().unwrap().remaining -= 1;
        eprintln!(
            "scorer.stage {}",
            baux::artifact::json::to_string(&baux::artifact::json::json!({
                "row_start":row_start,"rows":rows,"hidden_only":hidden_only,
                "gpu_ms":seconds * 1000.0,"stage_ms":stages,
            }))
            .map_err(|error| error.to_string())?
        );
        Ok(())
    }
}
