use super::*;
use baux::semantics::mhc4::{CausalPlan, LossEstimate, ObjectiveStats, PairShape, WaveShape};
use baux_cuda_device::{Mhc4EncodeShape, ObjectiveShape, ScoreStats};

pub(super) struct Mhc4State {
    fine: DeviceBuffer<u32>,
    expected: DeviceBuffer<u32>,
    seeds: DeviceBuffer<u64>,
    centroids: DeviceBuffer<u16>,
    pub(super) output: DeviceBuffer<u32>,
    pub(super) losses: DeviceBuffer<f32>,
    patch_nlls: DeviceBuffer<f32>,
    stats: DeviceBuffer<ObjectiveStats>,
    baseline: DeviceBuffer<f32>,
    patch_gains: DeviceBuffer<f32>,
    paired: bool,
    pub(super) index_incumbent: DeviceBuffer<u32>,
    pub(super) index_candidate: DeviceBuffer<u32>,
    pub(super) index_enabled: bool,
    index_ready: bool,
    pub(super) index_shadow: bool,
    index_force_full: bool,
    blocks: usize,
    tokens: usize,
    proposal_start: usize,
    patch: usize,
    score: bool,
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

pub(super) struct Mhc4Profile {
    pub init_ms: f32,
    pub gemm_ms: f32,
    pub project_ms: f32,
    pub reduce_ms: f32,
    pub total_ms: f32,
}

#[derive(Clone, Copy)]
enum Mhc4Stage {
    Init,
    Gemm,
    Project,
    Reduce,
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
            centroids: DeviceBuffer::zeroed(&self.stream, 37_448 * 64).map_err(cuda_error)?,
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

    pub(super) fn clear_pair(&mut self) -> CudaResult<()> {
        self.mhc4
            .as_mut()
            .ok_or("CUDA mHC-4 state is not configured")?
            .paired = false;
        Ok(())
    }

    /// Retain aligned incumbent token losses on the same ordered stream.
    pub(super) fn capture_pair(&mut self) -> CudaResult<()> {
        let state = self
            .mhc4
            .as_mut()
            .ok_or("CUDA mHC-4 state is not configured")?;
        if !state.score {
            return Err("CUDA paired capture requires reference targets".into());
        }
        unsafe {
            cuda_core::simt::memory::memcpy_dtod_async(
                state.baseline.cu_deviceptr(),
                state.losses.cu_deviceptr(),
                state.tokens * size_of::<f32>(),
                self.stream.cu_stream(),
            )
            .map_err(cuda_error)?;
        }
        state.paired = true;
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
        let (generation_end, generation_events) = self.generate_mhc4(
            blocks,
            tokens,
            proposal_start,
            patch,
            score,
            temperature,
            seed,
            profile,
        )?;
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
        blocks: usize,
        tokens: usize,
        proposal_start: usize,
        patch: usize,
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
        } else if patch == 1 {
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
        } else {
            self.sample_patch(
                blocks,
                tokens,
                proposal_start,
                score,
                temperature,
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
                    self.synth.as_ref(),
                    &self.module,
                    &self.stream,
                    &self.scratch.normalized,
                    &self.weights.readout,
                    &mut self.scratch.logits,
                    count,
                    VOCAB,
                    WIDTH,
                    proposal_start + first,
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

    fn sample_patch(
        &mut self,
        blocks: usize,
        tokens: usize,
        proposal_start: usize,
        score: bool,
        temperature: f32,
        profile: bool,
        graph_key: u32,
        generation_events: &mut Vec<(Mhc4Stage, CudaEvent)>,
    ) -> CudaResult<()> {
        let blocks_u32 = u32::try_from(blocks).map_err(cuda_error)?;
        let hierarchy_launch = self
            .module
            .prepare_mhc4_centroids(LaunchConfig1D::new((37_448_u32 * 64).div_ceil(256), 256, 0))
            .map_err(cuda_error)?;
        let expand_launch = self
            .module
            .prepare_mhc4_expand(LaunchConfig1D::new(blocks_u32, 64, 0))
            .map_err(cuda_error)?;
        let state = self.mhc4.as_mut().ok_or("CUDA mHC-4 state disappeared")?;
        let weights = self
            .weights
            .mhc4
            .as_ref()
            .ok_or("checkpoint missing mHC-4 mask/index tensors")?;
        let prompt_tokens = proposal_start
            .checked_add(1)
            .and_then(|patches| patches.checked_mul(baux::semantics::mhc4::PATCH))
            .ok_or("mHC-4 prompt size overflow")?;
        let shape = baux::semantics::mhc4::shape(prompt_tokens, tokens, temperature, score)?;
        let mut record = || -> CudaResult<()> {
            self.module
                .mhc4_centroids(
                    &self.stream,
                    &hierarchy_launch,
                    &self.weights.readout,
                    &mut state.centroids,
                    shape,
                )
                .map_err(cuda_error)?;
            if profile {
                generation_events.push((Mhc4Stage::Gemm, timing_event(&self.stream)?));
            }
            self.module
                .mhc4_expand(
                    &self.stream,
                    &expand_launch,
                    &self.scratch.normalized,
                    &self.weights.layers[0].qkv,
                    &weights.positions,
                    &self.weights.readout,
                    &state.centroids,
                    &state.seeds,
                    &state.expected,
                    &mut state.output,
                    &mut state.losses,
                    shape,
                )
                .map_err(cuda_error)?;
            if profile {
                generation_events.push((Mhc4Stage::Reduce, timing_event(&self.stream)?));
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

    pub(super) fn commit_index(&mut self, accepted: bool) {
        if let Some(state) = self.mhc4.as_mut() {
            if state.index_enabled && state.index_shadow && accepted {
                std::mem::swap(&mut state.index_incumbent, &mut state.index_candidate);
            }
            state.index_shadow = false;
        }
    }

    pub(super) fn force_index(&mut self) -> CudaResult<()> {
        let state = self
            .mhc4
            .as_mut()
            .ok_or("CUDA mHC-4 state is not configured")?;
        if !state.index_enabled || !state.index_ready {
            return Err("CUDA mHC-4 index audit requires initialized shadow state".into());
        }
        state.index_force_full = true;
        Ok(())
    }

    pub(super) fn mhc4_overlap(&self, visits: usize) -> CudaResult<(f32, u64, u64)> {
        let state = self
            .mhc4
            .as_ref()
            .ok_or("CUDA mHC-4 state is not configured")?;
        let values = visits
            .checked_mul(self.rows)
            .and_then(|rows| rows.checked_mul(8))
            .ok_or("CUDA mHC-4 index audit size overflow")?;
        let shadow = read_prefix(&state.index_candidate, &self.stream, values)?;
        let exact = read_prefix(&state.index_incumbent, &self.stream, values)?;
        let mut matches = 0_u64;
        let mut exact_rows = 0_u64;
        for (left, right) in shadow.chunks_exact(8).zip(exact.chunks_exact(8)) {
            let overlap = left.iter().filter(|leaf| right.contains(leaf)).count() as u64;
            matches += overlap;
            exact_rows += u64::from(overlap == 8);
        }
        let rows = (visits * self.rows) as u64;
        Ok((matches as f32 / (rows * 8) as f32, exact_rows, rows))
    }

    pub(super) fn mhc4_tokens(&self) -> CudaResult<Vec<u32>> {
        let state = self
            .mhc4
            .as_ref()
            .ok_or("CUDA mHC-4 state is not configured")?;
        read_prefix(&state.output, &self.stream, state.output.len())
    }

    pub(super) fn score_targets(&mut self, targets: &[u32]) -> CudaResult<()> {
        if targets.is_empty() || targets.iter().any(|&token| token >= VOCAB as u32) {
            return Err("CUDA mHC-4 scoring requires vocabulary-valid target macros".into());
        }
        self.scoring = Some(Scoring {
            targets: DeviceBuffer::from_host(&self.stream, targets).map_err(cuda_error)?,
            losses: DeviceBuffer::zeroed(&self.stream, targets.len()).map_err(cuda_error)?,
            count: targets.len(),
        });
        Ok(())
    }

    pub(super) fn expand_patches(&self, macros: &[u32]) -> CudaResult<(Vec<u32>, f32)> {
        let (tokens, elapsed) = self.expand_device(macros)?;
        Ok((read_prefix(&tokens, &self.stream, tokens.len())?, elapsed))
    }

    pub(super) fn expand_device(&self, macros: &[u32]) -> CudaResult<(DeviceBuffer<u32>, f32)> {
        if macros.is_empty() {
            return Err("mHC-4 expansion requires macro tokens".into());
        }
        let count = u32::try_from(macros.len()).map_err(|_| "too many mHC-4 patches")?;
        let macros = DeviceBuffer::from_host(&self.stream, macros).map_err(cuda_error)?;
        self.expand(&macros, 0, count)
    }

    pub(super) fn expand_generated(
        &self,
        first: usize,
        count: usize,
    ) -> CudaResult<(DeviceBuffer<u32>, f32)> {
        let end = first
            .checked_add(count)
            .ok_or("mHC-4 patch range overflow")?;
        if count == 0 || end > self.scratch.sampled.len() {
            return Err("mHC-4 generated patch range is invalid".into());
        }
        self.expand(
            &self.scratch.sampled,
            u32::try_from(first).map_err(|_| "mHC-4 patch offset is too large")?,
            u32::try_from(count).map_err(|_| "too many mHC-4 patches")?,
        )
    }

    fn expand(
        &self,
        macros: &DeviceBuffer<u32>,
        first: u32,
        count: u32,
    ) -> CudaResult<(DeviceBuffer<u32>, f32)> {
        let mut tokens =
            DeviceBuffer::zeroed(&self.stream, count as usize * 64).map_err(cuda_error)?;
        let launch = self
            .module
            .prepare_expand_patches(LaunchConfig1D::new(count, 256, 0))
            .map_err(cuda_error)?;
        let start = timing_event(&self.stream)?;
        self.module
            .expand_patches(
                &self.stream,
                &launch,
                &macros,
                &self.weights.readout,
                &mut tokens,
                count,
                first,
            )
            .map_err(cuda_error)?;
        let end = timing_event(&self.stream)?;
        Ok((tokens, start.elapsed_ms(&end).map_err(cuda_error)?))
    }

    pub(super) fn objective(
        &self,
        generated: &DeviceBuffer<u32>,
        targets: &DeviceBuffer<u32>,
    ) -> CudaResult<(ScoreStats, f32)> {
        let scoring = self
            .scoring
            .as_ref()
            .ok_or("CUDA mHC-4 target scoring was not configured")?;
        if generated.len() == 0 || generated.len() != targets.len() {
            return Err("CUDA mHC-4 objective requires aligned generated and target tokens".into());
        }
        let losses = u32::try_from(scoring.count).map_err(|_| "too many scored positions")?;
        let tokens = u32::try_from(generated.len()).map_err(|_| "too many generated tokens")?;
        let mut output = DeviceBuffer::zeroed(&self.stream, 1).map_err(cuda_error)?;
        let launch = self
            .module
            .prepare_objective(LaunchConfig1D::new(1, 256, 0))
            .map_err(cuda_error)?;
        let start = timing_event(&self.stream)?;
        self.module
            .objective(
                &self.stream,
                &launch,
                &scoring.losses,
                generated,
                targets,
                &mut output,
                ObjectiveShape {
                    losses,
                    tokens,
                    patch: 1,
                },
            )
            .map_err(cuda_error)?;
        let end = timing_event(&self.stream)?;
        let stats = read_prefix(&output, &self.stream, 1)?[0];
        Ok((stats, start.elapsed_ms(&end).map_err(cuda_error)?))
    }
}

impl Mhc4Profile {
    fn from_events(start: &CudaEvent, events: &[(Mhc4Stage, CudaEvent)]) -> CudaResult<Self> {
        let mut init_ms = 0.0;
        let mut gemm_ms = 0.0;
        let mut project_ms = 0.0;
        let mut reduce_ms = 0.0;
        let mut previous = start;
        for (stage, event) in events {
            let elapsed = previous.elapsed_ms(event).map_err(cuda_error)?;
            match stage {
                Mhc4Stage::Init => init_ms += elapsed,
                Mhc4Stage::Gemm => gemm_ms += elapsed,
                Mhc4Stage::Project => project_ms += elapsed,
                Mhc4Stage::Reduce => reduce_ms += elapsed,
            }
            previous = event;
        }
        let end = &events.last().ok_or("mHC-4 generation profile is empty")?.1;
        Ok(Self {
            init_ms,
            gemm_ms,
            project_ms,
            reduce_ms,
            total_ms: start.elapsed_ms(end).map_err(cuda_error)?,
        })
    }
}
