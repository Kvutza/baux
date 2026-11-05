use super::mhc_step::{Mhc4Profile, Mhc4Stage};
use super::*;
use baux::semantics::mhc4::{CausalPlan, LossEstimate, ObjectiveStats, PairShape, WaveShape};
use baux_cuda_device::{Mhc4EncodeShape, ScoreStats};

pub(super) struct Mhc4State {
    fine: DeviceBuffer<u32>,
    expected: DeviceBuffer<u32>,
    seeds: DeviceBuffer<u64>,
    pub(super) output: DeviceBuffer<u32>,
    pub(super) losses: DeviceBuffer<f32>,
    patch_nlls: DeviceBuffer<f32>,
    stats: DeviceBuffer<ObjectiveStats>,
    pub(super) baseline: DeviceBuffer<f32>,
    patch_gains: DeviceBuffer<f32>,
    pub(super) paired: bool,
    pub(super) index_incumbent: DeviceBuffer<u32>,
    pub(super) index_candidate: DeviceBuffer<u32>,
    pub(super) index_enabled: bool,
    pub(super) index_ready: bool,
    pub(super) index_shadow: bool,
    pub(super) index_force_full: bool,
    blocks: usize,
    pub(super) tokens: usize,
    proposal_start: usize,
    patch: usize,
    pub(super) score: bool,
    pub(super) generation_graph: Option<(u32, crate::graph::CudaGraph)>,
}

pub(super) struct Mhc4Evaluation {
    pub stats: Option<ScoreStats>,
    /// One mean target NLL per causal mHC-4 patch. Paired differences provide
    /// acceptance and reliability evidence; they do not train a proposal model.
    pub patch_nlls: Option<Vec<f32>>,
    pub estimate: Option<LossEstimate>,
    pub patch_gains: Option<Vec<f32>>,
    pub device_ms: f32,
    pub core_ms: f32,
    pub generation_ms: f32,
    pub objective_ms: f32,
    pub wall_ms: f32,
    pub visits: usize,
    pub core_profile: Option<WaveProfile>,
    pub generation_profile: Option<Mhc4Profile>,
}

impl FbtModel {
    pub(super) fn configure_mhc4(
        &mut self,
        prompt: &[u32],
        generated: usize,
        targets: Option<&[u32]>,
    ) -> CudaResult<()> {
        if self.weights.mhc4.is_none()
            || prompt.is_empty()
            || prompt.len() % 64 != 0
            || generated == 0
            || generated % 64 != 0
            || prompt
                .len()
                .checked_add(generated)
                .is_none_or(|tokens| tokens > self.rows * 64)
            || prompt.iter().any(|&token| token >= VOCAB as u32)
            || targets.is_some_and(|values| {
                values.len() != generated || values.iter().any(|&token| token >= VOCAB as u32)
            })
        {
            return Err(
                "CUDA mHC-4 configuration requires matching learned mHC-4 weights and aligned \
                 vocabulary-valid 64-token patches"
                    .into(),
            );
        }
        let shape = baux::semantics::mhc4::shape(prompt.len(), generated, 0.0, targets.is_some())?;
        self.configure_window(prompt, targets, shape, 64)
    }

    /// Score reference corpus tokens without feeding sampled tokens back into
    /// the scored state. The preceding complete patch determines label offset.
    pub(super) fn configure_causal(
        &mut self,
        batch: &baux::artifact::pretrain::CausalBatch,
    ) -> CudaResult<()> {
        let plan = CausalPlan::new(
            batch.tokens.len(),
            batch.labels.len(),
            &batch.loss_mask,
            self.rows,
        )?;
        self.configure_window(
            &batch.tokens,
            Some(&batch.labels[plan.start..]),
            plan.wave()?,
            64,
        )
    }

    pub(super) fn configure_linear(
        &mut self,
        batch: &baux::artifact::pretrain::CausalBatch,
    ) -> CudaResult<()> {
        let tokens = batch.tokens.len();
        if tokens == 0
            || tokens > self.rows
            || batch.labels.len() != tokens
            || batch.loss_mask.len() != tokens
            || batch.loss_mask.iter().any(|&v| v != 1)
        {
            return Err(
                "linear mHC-4 batch requires matching token/label lengths within active rows and \
                 an all-one loss mask"
                    .into(),
            );
        }
        let shape = WaveShape {
            blocks: u32::try_from(tokens.div_ceil(64)).map_err(cuda_error)?,
            tokens: u32::try_from(tokens).map_err(cuda_error)?,
            proposal_start: 0,
            intra: 0,
            vocabulary: VOCAB as u32,
            shards: (VOCAB / 128) as u32,
            temperature: 0.0,
            score: 1,
        };
        self.configure_window(
            &batch.tokens,
            Some(&batch.labels),
            shape,
            baux::semantics::shape::MHC4_PATCH as usize,
        )
    }

    fn configure_window(
        &mut self,
        input: &[u32],
        targets: Option<&[u32]>,
        shape: WaveShape,
        patch: usize,
    ) -> CudaResult<()> {
        let generated = shape.tokens as usize;
        if (patch != 1 && self.weights.mhc4.is_none())
            || !matches!(patch, 1 | 64)
            || input.len() > self.rows * patch
            || input.iter().any(|&t| t >= VOCAB as u32)
            || targets.is_some_and(|values| {
                values.len() != generated || values.iter().any(|&t| t >= VOCAB as u32)
            })
        {
            return Err("CUDA mHC-4 window has invalid capacity, weights or token IDs".into());
        }
        let blocks = shape.blocks as usize;
        let proposal_start = shape.proposal_start as usize;
        let fill = if patch == 1 { 0 } else { u32::MAX };
        let mut fine = vec![fill; self.rows * patch];
        fine[..input.len()].copy_from_slice(input);
        let expected = targets.map_or_else(|| vec![0; generated], <[u32]>::to_vec);
        if let Some(state) = self.mhc4.as_mut().filter(|state| {
            state.tokens == generated
                && state.proposal_start == proposal_start
                && state.patch == patch
                && state.score == targets.is_some()
                && state.fine.len() == fine.len()
        }) {
            copy_prefix(&state.fine, &fine, &self.stream)?;
            copy_prefix(&state.expected, &expected, &self.stream)?;
            state.paired = false;
            state.index_ready = false;
            state.index_shadow = false;
            state.index_force_full = false;
            // Kernel addresses and shapes are unchanged. Captured readout
            // graphs read the newly uploaded data and the current weights.
            return Ok(());
        }
        self.mhc4 = Some(Mhc4State {
            fine: DeviceBuffer::from_host(&self.stream, &fine).map_err(cuda_error)?,
            expected: DeviceBuffer::from_host(&self.stream, &expected).map_err(cuda_error)?,
            seeds: DeviceBuffer::zeroed(&self.stream, 1).map_err(cuda_error)?,
            output: DeviceBuffer::zeroed(&self.stream, generated).map_err(cuda_error)?,
            losses: DeviceBuffer::zeroed(&self.stream, generated).map_err(cuda_error)?,
            patch_nlls: DeviceBuffer::zeroed(&self.stream, blocks).map_err(cuda_error)?,
            stats: DeviceBuffer::zeroed(&self.stream, 1).map_err(cuda_error)?,
            baseline: DeviceBuffer::zeroed(&self.stream, generated).map_err(cuda_error)?,
            patch_gains: DeviceBuffer::zeroed(&self.stream, blocks).map_err(cuda_error)?,
            paired: false,
            index_incumbent: DeviceBuffer::zeroed(&self.stream, self.rows * 8 * 11)
                .map_err(cuda_error)?,
            index_candidate: DeviceBuffer::zeroed(&self.stream, self.rows * 8 * 11)
                .map_err(cuda_error)?,
            index_enabled: std::env::var_os("BAUX_CUDA_PISA_SHADOW").is_some(),
            index_ready: false,
            index_shadow: false,
            index_force_full: false,
            blocks,
            tokens: generated,
            proposal_start,
            patch,
            score: targets.is_some(),
            generation_graph: None,
        });
        Ok(())
    }

    pub(super) fn evaluate_mhc4(
        &mut self,
        visits: usize,
        temperature: f32,
        seed: u64,
        profile: bool,
    ) -> CudaResult<Mhc4Evaluation> {
        if !temperature.is_finite() || temperature < 0.0 {
            return Err("CUDA mHC-4 temperature must be finite and nonnegative".into());
        }
        let (blocks, tokens, proposal_start, patch, score, paired) = self
            .mhc4
            .as_ref()
            .map(|state| {
                (
                    state.blocks,
                    state.tokens,
                    state.proposal_start,
                    state.patch,
                    state.score,
                    state.paired,
                )
            })
            .ok_or("CUDA mHC-4 state is not configured")?;
        let plan = recurrence::RecurrentCore::selective_fbt().layer_visits(5, visits)?;
        if plan.len() > 11 {
            return Err("CUDA mHC-4 PISA state does not cover the recurrent plan".into());
        }
        let (wall, visits, start, core_end, core_events) =
            self.encode_mhc4(&plan, temperature, seed, patch, profile)?;
        let (generation_end, generation_events) =
            self.generate_mhc4(tokens, proposal_start, score, temperature, seed, profile)?;
        let (stats, patch_nlls, estimate, patch_gains, end) =
            self.finish_mhc4(blocks, tokens, score, paired)?;
        self.context.check_err().map_err(cuda_error)?;
        let core_profile = profile
            .then(|| WaveProfile::from_events(&start, &core_events))
            .transpose()?;
        let generation_profile = profile
            .then(|| Mhc4Profile::from_events(&core_end, &generation_events))
            .transpose()?;
        Ok(Mhc4Evaluation {
            stats,
            patch_nlls,
            estimate,
            patch_gains,
            device_ms: start.elapsed_ms(&end).map_err(cuda_error)?,
            core_ms: start.elapsed_ms(&core_end).map_err(cuda_error)?,
            generation_ms: core_end.elapsed_ms(&generation_end).map_err(cuda_error)?,
            objective_ms: generation_end.elapsed_ms(&end).map_err(cuda_error)?,
            wall_ms: wall.elapsed().as_secs_f32() * 1000.0,
            visits,
            core_profile,
            generation_profile,
        })
    }

    fn encode_mhc4(
        &mut self,
        plan: &[recurrence::LayerVisit],
        temperature: f32,
        seed: u64,
        patch: usize,
        profile: bool,
    ) -> CudaResult<(
        Instant,
        usize,
        CudaEvent,
        CudaEvent,
        Vec<(Stage, CudaEvent)>,
    )> {
        let index_shadow = {
            let state = self.mhc4.as_mut().ok_or("CUDA mHC-4 state disappeared")?;
            state.index_shadow =
                state.index_enabled && state.index_ready && !state.index_force_full;
            state.index_shadow
        };
        self.prepare_cache(plan.len())?;
        let fbt_shape =
            FbtShape::new(self.rows as u32, 512, VOCAB as u32, 1.0e-5).map_err(str::to_string)?;
        let encode_shape = Mhc4EncodeShape {
            rows: self.rows as u32,
            width: 512,
            vocabulary: VOCAB as u32,
            patch: patch as u32,
        };
        let embed_launch = self
            .module
            .prepare_embed_packed(LaunchConfig1D::new(encode_shape.rows, 256, 0))
            .map_err(cuda_error)?;
        let encode_launch = self
            .module
            .prepare_mhc4_encode(LaunchConfig1D::new(
                (encode_shape.rows * 512).div_ceil(256),
                256,
                0,
            ))
            .map_err(cuda_error)?;
        let replicate_launch = self
            .module
            .prepare_mhc_replicate(LaunchConfig1D::new(
                (encode_shape.rows * 512).div_ceil(256),
                256,
                0,
            ))
            .map_err(cuda_error)?;
        copy_prefix(
            &self
                .mhc4
                .as_ref()
                .ok_or("CUDA mHC-4 state disappeared")?
                .seeds,
            &[seed],
            &self.stream,
        )?;
        let wall = Instant::now();
        let start = timing_event(&self.stream)?;
        {
            let state = self.mhc4.as_ref().ok_or("CUDA mHC-4 state disappeared")?;
            if patch == 1 {
                self.module
                    .embed_packed(
                        &self.stream,
                        &embed_launch,
                        &self.weights.embedding,
                        &state.fine,
                        &mut self.scratch.embedded,
                        fbt_shape,
                    )
                    .map_err(cuda_error)?;
            } else {
                let weights = self
                    .weights
                    .mhc4
                    .as_ref()
                    .ok_or("checkpoint missing mHC-4 mask/index tensors")?;
                self.module
                    .mhc4_encode(
                        &self.stream,
                        &encode_launch,
                        &self.weights.readout,
                        &state.fine,
                        &weights.positions,
                        &weights.mask,
                        &mut self.scratch.embedded,
                        encode_shape,
                    )
                    .map_err(cuda_error)?;
            }
        }
        self.module
            .mhc_replicate(
                &self.stream,
                &replicate_launch,
                &self.scratch.embedded,
                &mut self.scratch.streams,
                self.rows as u32,
            )
            .map_err(cuda_error)?;
        let mut core_events = Vec::new();
        let visits = if profile {
            let mut events = Some(&mut core_events);
            mark(&mut events, &self.stream, Stage::Embed)?;
            let visits = self.enqueue_core(plan, temperature, seed, events, None, false)?;
            let mut events = Some(&mut core_events);
            mark(&mut events, &self.stream, Stage::Update)?;
            visits
        } else {
            self.enqueue_core(plan, temperature, seed, None, None, false)?
        };
        let core_end = timing_event(&self.stream)?;
        if let Some(state) = self.mhc4.as_mut() {
            if state.index_enabled && !index_shadow {
                state.index_ready = true;
            }
        }
        Ok((wall, visits, start, core_end, core_events))
    }

    fn generate_mhc4(
        &mut self,
        tokens: usize,
        proposal_start: usize,
        score: bool,
        temperature: f32,
        seed: u64,
        profile: bool,
    ) -> CudaResult<(CudaEvent, Vec<(Mhc4Stage, CudaEvent)>)> {
        let mut generation_events = Vec::new();
        let graph_key = temperature.to_bits();
        let cached = {
            let state = self.mhc4.as_mut().ok_or("CUDA mHC-4 state disappeared")?;
            (!profile)
                .then(|| state.generation_graph.take())
                .flatten()
                .filter(|(key, _)| *key == graph_key)
        };
        if let Some(graph) = cached {
            graph.1.launch()?;
            let state = self.mhc4.as_mut().ok_or("CUDA mHC-4 state disappeared")?;
            state.generation_graph = Some(graph);
        } else {
            self.sample_linear(
                tokens,
                proposal_start,
                score,
                temperature,
                seed,
                profile,
                graph_key,
                &mut generation_events,
            )?;
        }
        let generation_end = timing_event(&self.stream)?;
        Ok((generation_end, generation_events))
    }

    fn sample_linear(
        &mut self,
        tokens: usize,
        proposal_start: usize,
        score: bool,
        temperature: f32,
        seed: u64,
        profile: bool,
        graph_key: u32,
        generation_events: &mut Vec<(Mhc4Stage, CudaEvent)>,
    ) -> CudaResult<()> {
        let state = self.mhc4.as_mut().ok_or("CUDA mHC-4 state disappeared")?;
        let capacity = self.scratch.logits.len() / VOCAB;
        let mut record = || -> CudaResult<()> {
            for first in (0..tokens).step_by(capacity) {
                let count = capacity.min(tokens - first);
                matmul_at(
                    &self.module,
                    &self.stream,
                    &self.scratch.normalized,
                    &self.weights.readout,
                    &mut self.scratch.logits,
                    count,
                    VOCAB,
                    WIDTH,
                    (proposal_start + first).min(self.rows.saturating_sub(count)),
                )?;
                if profile {
                    generation_events.push((Mhc4Stage::Gemm, timing_event(&self.stream)?));
                }
                let sample_launch = self
                    .module
                    .prepare_sample(LaunchConfig1D::new(count as u32, 256, 0))
                    .map_err(cuda_error)?;
                self.module
                    .sample(
                        &self.stream,
                        &sample_launch,
                        &self.scratch.logits,
                        &mut state.output,
                        count as u32,
                        temperature,
                        seed,
                        first as u32,
                    )
                    .map_err(cuda_error)?;
                if score {
                    let score_launch = self
                        .module
                        .prepare_score(LaunchConfig1D::new(count as u32, 256, 0))
                        .map_err(cuda_error)?;
                    self.module
                        .score(
                            &self.stream,
                            &score_launch,
                            &self.scratch.logits,
                            &state.expected,
                            &mut state.losses,
                            count as u32,
                            first as u32,
                            first as u32,
                        )
                        .map_err(cuda_error)?;
                }
                if profile {
                    generation_events.push((Mhc4Stage::Reduce, timing_event(&self.stream)?));
                }
            }
            Ok(())
        };
        if profile {
            record()
        } else {
            let graph = crate::graph::CudaGraph::capture(
                self.context.clone(),
                self.stream.clone(),
                record,
            )?;
            graph.launch()?;
            state.generation_graph = Some((graph_key, graph));
            Ok(())
        }
    }

    fn finish_mhc4(
        &mut self,
        blocks: usize,
        tokens: usize,
        score: bool,
        paired: bool,
    ) -> CudaResult<(
        Option<ScoreStats>,
        Option<Vec<f32>>,
        Option<LossEstimate>,
        Option<Vec<f32>>,
        CudaEvent,
    )> {
        let blocks_u32 = u32::try_from(blocks).map_err(cuda_error)?;
        let tokens_u32 = u32::try_from(tokens).map_err(cuda_error)?;
        if score {
            let objective_launch = self
                .module
                .prepare_mhc4_objective(LaunchConfig1D::new(1, 256, 0))
                .map_err(cuda_error)?;
            let patch_launch = self
                .module
                .prepare_patch_mean_nll(LaunchConfig1D::new(blocks_u32, 64, 0))
                .map_err(cuda_error)?;
            let state = self.mhc4.as_mut().ok_or("CUDA mHC-4 state disappeared")?;
            self.module
                .mhc4_objective(
                    &self.stream,
                    &objective_launch,
                    &state.losses,
                    &state.output,
                    &state.expected,
                    &mut state.stats,
                    &state.baseline,
                    PairShape {
                        tokens: tokens_u32,
                        paired: u32::from(paired),
                        batch_tokens: 64,
                    },
                    &mut state.patch_gains,
                )
                .map_err(cuda_error)?;
            self.module
                .patch_mean_nll(
                    &self.stream,
                    &patch_launch,
                    &state.losses,
                    &mut state.patch_nlls,
                    tokens_u32,
                )
                .map_err(cuda_error)?;
        }
        let end = timing_event(&self.stream)?;
        let raw = score
            .then(|| {
                let state = self.mhc4.as_ref().ok_or("CUDA mHC-4 state disappeared")?;
                read_prefix(&state.stats, &self.stream, 1).map(|v| v[0])
            })
            .transpose()?;
        let estimate = raw
            .as_ref()
            .map(|raw| {
                raw.estimate(PairShape {
                    tokens: tokens_u32,
                    paired: u32::from(paired),
                    batch_tokens: 64,
                })
            })
            .transpose()
            .map_err(str::to_string)?;
        let stats = raw
            .as_ref()
            .zip(estimate.as_ref())
            .map(|(raw, estimate)| ScoreStats {
                reward: -estimate.mean,
                mean_nll: estimate.mean,
                maximum_nll: raw.maximum,
                positional_accuracy: raw.matches as f32 / tokens as f32,
                tokens: tokens_u32,
                mismatches: raw.mismatches,
                patch_nll_octiles: raw.patch_nll_octiles,
            });
        let patch_nlls = score
            .then(|| {
                let state = self.mhc4.as_ref().ok_or("CUDA mHC-4 state disappeared")?;
                read_prefix(&state.patch_nlls, &self.stream, blocks)
            })
            .transpose()?;
        let patch_gains = paired
            .then(|| {
                let state = self.mhc4.as_ref().ok_or("CUDA mHC-4 state disappeared")?;
                read_prefix(&state.patch_gains, &self.stream, blocks)
            })
            .transpose()?;
        Ok((stats, patch_nlls, estimate, patch_gains, end))
    }
}
