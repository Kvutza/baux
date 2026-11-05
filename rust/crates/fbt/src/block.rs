//! Exact speculative verification using the incumbent rollout as the draft.
//!
//! A full candidate forward proposes every next token in parallel. The longest
//! causally valid prefix is committed; the first mismatch is replaced by the
//! candidate's exact token and the remaining suffix is verified again.

use super::*;
use crate::config::{GenerationConfig, GenerationTask};

use super::target::TargetStats;

#[path = "block/batch.rs"]
mod batch;
#[path = "block/hnet2.rs"]
mod hnet2;
#[cfg(test)]
#[path = "block/tests.rs"]
mod tests;

static NEXT_OWNER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct PairToken {
    owner: u64,
    generation: u64,
    tokens: usize,
}

pub(super) struct BlockDecoder {
    owner: u64,
    pair_gen: std::cell::Cell<u64>,
    pub(super) context: u32,
    pub(super) pipelines: Pipelines,
    pub(super) tensorops: TensorOpsPipelines,
    pub(super) pisa: Pisa1,
    pub(super) buffers: Buffers,
    pub(super) proposals: Buffer,
    pub(super) seeds: Buffer,
    pub(super) coarse: Buffer,
    pub(super) expanded: Buffer,
    pub(super) expected: Buffer,
    paired_patches: Buffer,
    paired_losses: Buffer,
    paired_losses_ready: std::cell::Cell<bool>,
    pub(super) profile: std::cell::RefCell<Option<Profile>>,
}

pub(super) struct Profile {
    pub(super) trace: scorer::ScorerStageTrace,
    pub(super) remaining: u32,
}

pub(super) struct CausalScore {
    pub(super) quality: decode::TargetQuality,
    pub(super) gpu_seconds: f64,
    pub(super) patch_gains: Vec<f32>,
    pub(super) observation: crate::objective_observation::ObjectiveObservation,
}

use super::stats::{CommitStats, RepairStats, VerificationProgress};
use super::window::{
    commit_tile, expand_draft, loss_window, next_window, patch_token, readout_slice, refresh_draft,
    repair_bounds, route_sample, score_targets,
};

impl BlockDecoder {
    #[allow(clippy::too_many_arguments)]
    fn repair_waves(
        &self,
        runtime: &Runtime,
        decoder: &decode::Decoder,
        weights: CandidateRow<'_>,
        config: &GenerationConfig,
        score_targets: bool,
        row_start: u32,
        window: u32,
        cursor: usize,
        prompt: usize,
        waves: u32,
    ) -> Result<f64, String> {
        let command = runtime.queue.new_command_buffer();
        for wave in 0..waves {
            let readout = scorer::ProposalReadout {
                output: &self.proposals,
                hidden: None,
                seeds: &self.seeds,
                temperature: config.temperature,
                score_targets,
                feedback: config.feedback_transition,
            };
            if decoder.cache()[0].kv.is_some() {
                scorer::context_chunk(
                    command,
                    &self.pipelines,
                    &self.tensorops,
                    &self.pisa,
                    &self.buffers,
                    weights,
                    readout,
                    row_start,
                    window,
                    decoder.cache(),
                    false,
                    None,
                )?;
            } else {
                scorer::suffix_proposals(
                    command,
                    &self.pipelines,
                    &self.tensorops,
                    &self.pisa,
                    &self.buffers,
                    weights,
                    scorer::ProposalReadout {
                        output: &self.proposals,
                        hidden: None,
                        seeds: &self.seeds,
                        temperature: config.temperature,
                        score_targets,
                        feedback: config.feedback_transition,
                    },
                    row_start,
                    window,
                    decoder.cache(),
                )?;
            }
            if wave + 1 < waves {
                self.shift_draft(command, row_start, window, cursor, prompt);
            }
        }
        complete(command)
    }

    #[allow(clippy::too_many_arguments)]
    fn repair_mismatch(
        &self,
        runtime: &Runtime,
        decoder: &decode::Decoder,
        weights: CandidateRow<'_>,
        task: &GenerationTask,
        config: &GenerationConfig,
        tokens: &mut [u32],
        cursor: &mut usize,
        score_targets: bool,
        target: &mut TargetStats,
    ) -> Result<RepairStats, String> {
        if *cursor >= tokens.len() {
            return Ok(RepairStats::default());
        }
        let input =
            crate::apple_gpu::slice_mut::<u32>(&self.buffers.tokens, self.context as usize)?;
        let proposed =
            crate::apple_gpu::buffer_slice::<u32>(&self.proposals, self.context as usize)?;
        let mut stats = RepairStats::default();
        let mut window = config.verify.window;
        let mut stalled = false;
        let mut report_at = 16_384;
        while *cursor < tokens.len() {
            let (row_start, tile_end) = repair_bounds(
                task.prompt.len() - 1 + *cursor,
                self.context,
                window,
                task.prompt.len(),
                tokens.len(),
            );
            let waves = if stalled || config.verify.unroll > 1 {
                config.verify.unroll
            } else {
                1
            };
            stats.gpu_seconds += self.repair_waves(
                runtime,
                decoder,
                weights,
                config,
                score_targets,
                row_start,
                window,
                *cursor,
                task.prompt.len(),
                waves,
            )?;
            self.record_route(&mut stats);
            stats.record_waves(window, waves);

            if waves > 1 {
                refresh_draft(tokens, input, *cursor, tile_end, task.prompt.len());
            }

            let exact_start = *cursor;
            let (next_cursor, reached_eos) = commit_tile(
                task,
                tokens,
                input,
                proposed,
                exact_start,
                tile_end,
                self.context as usize,
                weights.architecture.patch_size(),
                config.eos_token,
                &mut stats,
            )?;
            let accepted = stats.accepted_lengths.last().copied().unwrap_or(0);
            *cursor = next_cursor;
            if reached_eos {
                return Ok(stats);
            }
            stats.report(*cursor, tokens.len(), &mut report_at);
            self.repair_target(
                score_targets,
                task,
                tokens,
                exact_start,
                *cursor,
                weights.architecture.patch_size(),
                target,
            )?;
            (stalled, window) = next_window(
                window,
                config,
                *cursor < tile_end,
                accepted,
                tile_end - exact_start,
            );
        }
        Ok(stats)
    }

    pub fn new(runtime: &Runtime) -> Result<Self, String> {
        Self::with_context(runtime, CONTEXT)
    }

    pub(super) fn with_context(runtime: &Runtime, context: u32) -> Result<Self, String> {
        Self::with_architecture(runtime, context, ResidualArchitecture::Legacy)
    }

    pub(super) fn with_architecture(
        runtime: &Runtime,
        context: u32,
        architecture: ResidualArchitecture,
    ) -> Result<Self, String> {
        let pisa = if architecture.is_hnet() && context > ROWS {
            Pisa1::with_context(runtime, ROWS)?
        } else if architecture.is_mhc4() && context > ROWS {
            Pisa1::with_rows(runtime, context, ROWS)?
        } else {
            Pisa1::with_context(runtime, context)?
        };
        let expanded_count = context as usize;
        let coarse_count = if architecture.is_mhc4() && context > ROWS {
            (ROWS * WIDTH) as usize
        } else {
            context as usize * WIDTH as usize
        };
        let owner = NEXT_OWNER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        Ok(Self {
            owner,
            pair_gen: std::cell::Cell::new(0),
            context,
            pipelines: Pipelines::new(runtime)?,
            tensorops: TensorOpsPipelines::new(runtime)?,
            pisa,
            buffers: Buffers::for_context(runtime, context, architecture),
            proposals: runtime.buffer::<u32>(context as usize),
            seeds: runtime.buffer::<u64>(1),
            coarse: runtime.buffer::<u16>(coarse_count),
            expanded: runtime.buffer::<u32>(expanded_count),
            expected: runtime.buffer::<u32>(expanded_count),
            paired_patches: runtime.buffer::<f32>(context as usize),
            paired_losses: runtime.buffer::<f32>(expanded_count),
            paired_losses_ready: std::cell::Cell::new(false),
            profile: std::cell::RefCell::new(None),
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn commit_prefix(
        &self,
        weights: CandidateRow<'_>,
        task: &GenerationTask,
        tokens: &mut [u32],
        input: &mut [u32],
        cursor: &mut usize,
        maximum: usize,
        eos: Option<u32>,
        expanded: bool,
    ) -> Result<CommitStats, String> {
        if *cursor == maximum {
            return Ok(CommitStats {
                accepted: 0,
                committed: 0,
                mismatch: None,
            });
        }
        let proposed =
            crate::apple_gpu::buffer_slice::<u32>(&self.proposals, self.context as usize)?;
        let prompt = task.prompt.len();
        let exact_start = *cursor;
        if expanded {
            let generated = crate::apple_gpu::buffer_slice::<u32>(&self.expanded, maximum)?;
            tokens[exact_start..maximum].copy_from_slice(&generated[exact_start..maximum]);
            *cursor = maximum;
            return Ok(CommitStats {
                accepted: maximum - exact_start,
                committed: maximum - exact_start,
                mismatch: None,
            });
        }
        let mut position = *cursor;
        let mut committed = None;
        let mut accepted = 0usize;
        let patch = weights.architecture.patch_size();
        let prompt_macros = prompt / patch;
        let readout_slice = readout_slice(weights, patch);
        while position < maximum {
            let intra_idx = position % patch;
            let prediction_idx = if patch > 1 {
                (prompt_macros.saturating_sub(1) + position / patch).min(proposed.len() - 1)
            } else {
                prompt - 1 + position
            };
            let macro_prediction = proposed[prediction_idx];
            if macro_prediction >= VOCAB {
                return Err("block verifier produced an invalid token".into());
            }
            let token_value = if patch > 1 && intra_idx > 0 {
                patch_token(
                    macro_prediction,
                    tokens[position - 1],
                    intra_idx,
                    readout_slice,
                )
            } else {
                macro_prediction
            };
            let exact_prediction = committed.is_none();
            if exact_prediction {
                let matched = tokens[position] == token_value;
                if matched {
                    accepted += 1;
                } else if patch == 1 {
                    committed = Some(position + 1);
                }
            }
            tokens[position] = token_value;
            if position + 1 < maximum && patch == 1 {
                input[prompt + position] = token_value;
            }
            position += 1;
            if exact_prediction && Some(token_value) == eos {
                tokens[position..].fill(token_value);
                *cursor = maximum;
                return Ok(CommitStats {
                    accepted,
                    committed: position - exact_start,
                    mismatch: committed.map(|after| after - 1),
                });
            }
        }
        *cursor = if patch > 1 {
            maximum
        } else {
            committed.unwrap_or(maximum)
        };
        Ok(CommitStats {
            accepted: if patch > 1 {
                maximum - exact_start
            } else {
                accepted
            },
            committed: *cursor - exact_start,
            mismatch: committed.map(|after| after - 1),
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn run_verification(
        &self,
        runtime: &Runtime,
        decoder: &decode::Decoder,
        weights: CandidateRow<'_>,
        task: &GenerationTask,
        config: &GenerationConfig,
        tokens: &mut [u32],
        input: &mut [u32],
        total_rows: u32,
        score_targets: bool,
    ) -> Result<VerificationProgress, String> {
        let maximum = config.max_tokens as usize;
        let eos = config.eos_token;
        let mut progress = VerificationProgress {
            target: TargetStats::new(loss_window(config)),
            ..Default::default()
        };
        while progress.cursor < maximum && progress.broad_passes < config.verify.passes as usize {
            let exact_start = progress.cursor;
            let (pass_seconds, active_rows, route) = self.parallel_pass(
                runtime,
                decoder,
                weights,
                config.temperature,
                config.feedback_transition,
                progress.broad_passes,
                progress.cursor,
                total_rows,
                score_targets,
                task.prompt.len(),
            )?;
            progress.gpu_seconds += pass_seconds;
            progress.broad_passes += 1;
            progress.parallel_positions += active_rows as usize;
            progress.evaluated_lengths.push(active_rows as usize);
            progress.route_samples.push(route);
            let expanded = false;
            let stats = self.commit_prefix(
                weights,
                task,
                tokens,
                input,
                &mut progress.cursor,
                maximum,
                eos,
                expanded,
            )?;
            progress.accepted_lengths.push(stats.accepted);
            progress.committed_lengths.push(stats.committed);
            if progress.first_mismatch.is_none() {
                progress.first_mismatch = stats.mismatch;
            }
            if score_targets && progress.target_quality.is_none() {
                target::record(
                    &self.buffers,
                    self.context,
                    task,
                    tokens,
                    exact_start..progress.cursor,
                    weights.architecture.patch_size(),
                    &mut progress.target,
                )?;
            }
            if stats.committed == 0 {
                break;
            }
        }
        if progress.cursor < maximum {
            let stats = self.repair_mismatch(
                runtime,
                decoder,
                weights,
                task,
                config,
                tokens,
                &mut progress.cursor,
                score_targets,
                &mut progress.target,
            )?;
            progress.gpu_seconds += stats.gpu_seconds;
            progress.repair_positions += stats.evaluated_positions;
            progress.correction_waves += stats.correction_waves;
            progress.repair_batches += stats.repair_batches;
            if progress.first_mismatch.is_none() {
                progress.first_mismatch = stats.first_mismatch;
            }
            progress.evaluated_lengths.extend(stats.evaluated_lengths);
            progress.accepted_lengths.extend(stats.accepted_lengths);
            progress.committed_lengths.extend(stats.committed_lengths);
            progress.route_samples.extend(stats.route_samples);
        }
        Ok(progress)
    }

    pub fn verify(
        &self,
        runtime: &Runtime,
        decoder: &decode::Decoder,
        weights: CandidateRow<'_>,
        tasks: &[GenerationTask],
        config: &GenerationConfig,
        sampling_seed: u64,
        drafts: &[decode::Rollout],
    ) -> Result<Vec<decode::Rollout>, String> {
        if weights.architecture.is_hnet() {
            return Err("H-Net generation requires a continuous-state byte decoder".into());
        }
        if config.draft.is_some() {
            return diffusion::verify(
                self,
                runtime,
                decoder,
                weights,
                tasks,
                config,
                sampling_seed,
            );
        }
        if tasks.len() != 1 || drafts.len() != 1 {
            return Err("block verification requires exactly one trajectory and draft".into());
        }
        let maximum = config.max_tokens as usize;
        let patch = weights.architecture.patch_size();
        let total_rows = if patch > 1 {
            (tasks[0].prompt.len() / patch + maximum.div_ceil(patch) - 1).div_ceil(128) as u32 * 128
        } else {
            ((tasks[0].prompt.len() + maximum - 1).div_ceil(128) as u32 * 128).max(CONTEXT)
        };
        if total_rows > self.context {
            return Err("prompt plus generation exceeds block cache capacity".into());
        }
        let score_targets = score_targets(config);
        if score_targets {
            self.load_targets(&tasks[0], maximum, patch)?;
        }
        let eos = config.eos_token;
        let mut tokens = expand_draft(&drafts[0], maximum, eos)?;
        let input =
            crate::apple_gpu::slice_mut::<u32>(&self.buffers.tokens, self.context as usize)?;
        input.fill(0);
        let task = &tasks[0];
        if patch > 1 {
            let prompt_macros = task.prompt.len() / patch;
            for i in 0..prompt_macros.min(self.context as usize) {
                input[i] = task.prompt[i * patch];
            }
            let input_tokens = (maximum.saturating_sub(1)) / patch;
            for i in 0..input_tokens {
                if prompt_macros + i < self.context as usize {
                    input[prompt_macros + i] = tokens[i * patch];
                }
            }
        } else {
            input[..task.prompt.len()].copy_from_slice(&task.prompt);
            let input_tokens = maximum.saturating_sub(1);
            input[task.prompt.len()..task.prompt.len() + input_tokens]
                .copy_from_slice(&tokens[..input_tokens]);
        }
        let seeds = crate::apple_gpu::slice_mut::<u64>(&self.seeds, 1)?;
        seeds[0] = crate::hash::splitmix64(sampling_seed);

        let started = Instant::now();
        let progress = self.run_verification(
            runtime,
            decoder,
            weights,
            task,
            config,
            &mut tokens,
            input,
            total_rows,
            score_targets,
        )?;
        let wall_seconds = started.elapsed().as_secs_f64();
        if score_targets && progress.target_quality.is_none() && progress.target.count != maximum {
            return Err(format!(
                "conditional target loss covered {} of {maximum} tokens",
                progress.target.count
            ));
        }
        let target_quality = if !score_targets {
            None
        } else if let Some(quality) = progress.target_quality {
            Some(quality)
        } else {
            Some(progress.target.finish()?)
        };
        let end = tokens
            .iter()
            .position(|token| Some(*token) == eos)
            .map_or(tokens.len(), |index| index + 1);
        Ok(vec![decode::Rollout {
            draft: None,
            tokens: tokens[..end].to_vec(),
            token_count: end,
            finish_reason: if end < maximum { "eos" } else { "length" },
            wall_seconds,
            gpu_seconds: progress.gpu_seconds,
            evaluated_positions: progress.parallel_positions + progress.repair_positions,
            committed_tokens: end,
            broad_passes: progress.broad_passes,
            correction_waves: progress.correction_waves,
            repair_batches: progress.repair_batches,
            repaired_tokens: progress.repair_positions,
            accepted_tokens: progress.accepted_lengths.iter().sum(),
            first_mismatch: progress.first_mismatch,
            evaluated_lengths: progress.evaluated_lengths,
            accepted_lengths: progress.accepted_lengths,
            committed_lengths: progress.committed_lengths,
            route_samples: progress.route_samples,
            conditional_target_nll: target_quality.map(|quality| quality.mean_nll),
            target_quality,
            paired_patch_improvements: progress.paired_patch_improvements,
        }])
    }

    pub fn generate(
        &self,
        runtime: &Runtime,
        decoder: &decode::Decoder,
        weights: CandidateRow<'_>,
        tasks: &[GenerationTask],
        config: &GenerationConfig,
        sampling_seed: u64,
    ) -> Result<Vec<decode::Rollout>, String> {
        if weights.architecture.is_hnet() {
            return Err("H-Net generation requires a continuous-state byte decoder".into());
        }
        if config.draft.is_some() {
            return diffusion::verify(
                self,
                runtime,
                decoder,
                weights,
                tasks,
                config,
                sampling_seed,
            );
        }
        let draft = decode::Rollout {
            draft: None,
            tokens: vec![0; config.max_tokens as usize],
            token_count: config.max_tokens as usize,
            finish_reason: "length",
            wall_seconds: 0.0,
            gpu_seconds: 0.0,
            evaluated_positions: 0,
            committed_tokens: 0,
            broad_passes: 0,
            correction_waves: 0,
            repair_batches: 0,
            repaired_tokens: 0,
            accepted_tokens: 0,
            first_mismatch: None,
            evaluated_lengths: Vec::new(),
            accepted_lengths: Vec::new(),
            committed_lengths: Vec::new(),
            route_samples: Vec::new(),
            conditional_target_nll: None,
            target_quality: None,
            paired_patch_improvements: Vec::new(),
        };
        self.verify(
            runtime,
            decoder,
            weights,
            tasks,
            config,
            sampling_seed,
            &[draft],
        )
    }
}
