//! Production candidate scoring in one Metal compute pass, with optional
//! encoder-boundary timestamp instrumentation.

use super::architecture::LayerStep;
use super::*;
use metal::MTLDispatchType;
use std::cell::RefCell;

#[path = "fbt_scorer/input.rs"]
mod input;
#[path = "fbt_scorer/mhc.rs"]
mod mhc;

pub(super) use super::scorer_trace::ScorerStageTrace;

pub(super) struct Scorer<'a> {
    pub(super) command: &'a CommandBufferRef,
    pub(super) encoder: RefCell<Option<&'a ComputeCommandEncoderRef>>,
    pub(super) pipelines: &'a Pipelines,
    pub(super) tensorops: &'a TensorOpsPipelines,
    pub(super) pisa1: &'a Pisa1,
    pub(super) buffers: &'a Buffers,
    pub(super) weights: CandidateRow<'a>,
    pub(super) row_start: u32,
    pub(super) rows: u32,
    pub(super) active_rows: std::cell::Cell<u32>,
    pub(super) cache: Option<&'a [decode::Cache]>,
    pub(super) dirty: bool,
    pub(super) trace: Option<&'a ScorerStageTrace>,
    pub(super) feedback: crate::config::FeedbackTransition,
    pub(super) diffusion: Option<diffusion::Input<'a>>,
    pub(super) cache_only: bool,
}

#[derive(Clone, Copy)]
pub(super) struct ProposalReadout<'a> {
    pub output: &'a BufferRef,
    pub hidden: Option<&'a BufferRef>,
    pub seeds: &'a BufferRef,
    pub temperature: f32,
    pub score_targets: bool,
    pub feedback: crate::config::FeedbackTransition,
}

#[derive(Clone, Copy)]
pub(super) struct TracePlan {
    pub start: u32,
    pub rows: u32,
    pub hidden_only: bool,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum ReplaySite {
    Attention(u32),
    Moe(u32),
}

struct ProposalPlan<'a> {
    readout: ProposalReadout<'a>,
    row_start: u32,
    rows: u32,
    cache: Option<&'a [decode::Cache]>,
    dirty: bool,
    trace: Option<&'a ScorerStageTrace>,
    hidden_only: bool,
    diffusion: Option<diffusion::Input<'a>>,
}

pub(super) fn objective_fused(
    command: &CommandBufferRef,
    pipelines: &Pipelines,
    tensorops: &TensorOpsPipelines,
    pisa1: &Pisa1,
    buffers: &Buffers,
    weights: CandidateRow<'_>,
) -> Result<(), String> {
    objective_trace(command, pipelines, tensorops, pisa1, buffers, weights, None)
}

pub(super) fn encode_proposals(
    command: &CommandBufferRef,
    pipelines: &Pipelines,
    tensorops: &TensorOpsPipelines,
    pisa1: &Pisa1,
    buffers: &Buffers,
    weights: CandidateRow<'_>,
    readout: ProposalReadout<'_>,
    rows: u32,
    cache: Option<&[decode::Cache]>,
    trace: Option<&ScorerStageTrace>,
) -> Result<(), String> {
    if rows == 0 || rows > ROWS || (cache.is_none() && rows % CONTEXT != 0) || rows % 128 != 0 {
        return Err(format!("invalid active scorer row count {rows}"));
    }
    proposals_at(
        command,
        pipelines,
        tensorops,
        pisa1,
        buffers,
        weights,
        ProposalPlan {
            readout,
            row_start: 0,
            rows,
            cache,
            dirty: false,
            trace,
            hidden_only: false,
            diffusion: None,
        },
    )
}

pub(super) fn encode_hidden(
    command: &CommandBufferRef,
    pipelines: &Pipelines,
    tensorops: &TensorOpsPipelines,
    pisa1: &Pisa1,
    buffers: &Buffers,
    weights: CandidateRow<'_>,
    readout: ProposalReadout<'_>,
    rows: u32,
    cache: Option<&[decode::Cache]>,
    trace: Option<&ScorerStageTrace>,
) -> Result<(), String> {
    if rows == 0 || rows > ROWS || (cache.is_none() && rows % CONTEXT != 0) || rows % 128 != 0 {
        return Err(format!("invalid active scorer row count {rows}"));
    }
    proposals_at(
        command,
        pipelines,
        tensorops,
        pisa1,
        buffers,
        weights,
        ProposalPlan {
            readout,
            row_start: 0,
            rows,
            cache,
            dirty: false,
            trace,
            hidden_only: true,
            diffusion: None,
        },
    )
}

pub(super) fn suffix_proposals(
    command: &CommandBufferRef,
    pipelines: &Pipelines,
    tensorops: &TensorOpsPipelines,
    pisa1: &Pisa1,
    buffers: &Buffers,
    weights: CandidateRow<'_>,
    readout: ProposalReadout<'_>,
    row_start: u32,
    rows: u32,
    cache: &[decode::Cache],
) -> Result<(), String> {
    if row_start % 4 != 0 || rows == 0 || rows % 128 != 0 || row_start + rows > pisa1.context() {
        return Err(format!(
            "invalid candidate suffix rows {row_start}..{}",
            row_start + rows
        ));
    }
    proposals_at(
        command,
        pipelines,
        tensorops,
        pisa1,
        buffers,
        weights,
        ProposalPlan {
            readout,
            row_start,
            rows,
            cache: Some(cache),
            dirty: true,
            trace: None,
            hidden_only: false,
            diffusion: None,
        },
    )
}

/// Chunk-local activations with absolute token positions and persistent KV.
pub(super) fn context_chunk(
    command: &CommandBufferRef,
    pipelines: &Pipelines,
    tensorops: &TensorOpsPipelines,
    pisa: &Pisa1,
    buffers: &Buffers,
    weights: CandidateRow<'_>,
    readout: ProposalReadout<'_>,
    start: u32,
    rows: u32,
    cache: &[decode::Cache],
    hidden_only: bool,
    trace: Option<&ScorerStageTrace>,
) -> Result<(), String> {
    if rows == 0
        || rows > ROWS
        || rows % 128 != 0
        || start % 4 != 0
        || start
            .checked_add(rows)
            .is_none_or(|end| end > pisa.context())
        || cache.first().is_none_or(|entry| entry.kv.is_none())
    {
        return Err("invalid compact context chunk".into());
    }
    proposals_at(
        command,
        pipelines,
        tensorops,
        pisa,
        buffers,
        weights,
        ProposalPlan {
            readout,
            row_start: start,
            rows,
            cache: Some(cache),
            dirty: true,
            trace,
            hidden_only,
            diffusion: None,
        },
    )
}

pub(super) fn profile_chunk(
    command: &CommandBufferRef,
    runtime: &Runtime,
    pipelines: &Pipelines,
    tensorops: &TensorOpsPipelines,
    pisa: &Pisa1,
    buffers: &Buffers,
    weights: CandidateRow<'_>,
    readout: ProposalReadout<'_>,
    cache: &[decode::Cache],
    trace: &ScorerStageTrace,
    plan: TracePlan,
) -> Result<(f64, std::collections::BTreeMap<&'static str, f64>), String> {
    context_chunk(
        command,
        pipelines,
        tensorops,
        pisa,
        buffers,
        weights,
        readout,
        plan.start,
        plan.rows,
        cache,
        plan.hidden_only,
        Some(trace),
    )?;
    trace.resolve(command);
    let (mut cpu_start, mut gpu_start) = (0, 0);
    runtime
        .device
        .sample_timestamps(&mut cpu_start, &mut gpu_start);
    command.commit();
    let seconds = complete_committed(command)?;
    let (mut cpu_end, mut gpu_end) = (0, 0);
    runtime.device.sample_timestamps(&mut cpu_end, &mut gpu_end);
    let cpu_span = cpu_end
        .checked_sub(cpu_start)
        .ok_or("nonmonotonic generation CPU timestamp")?;
    let gpu_span = gpu_end
        .checked_sub(gpu_start)
        .filter(|span| *span > 0)
        .ok_or("invalid generation GPU timestamp calibration")?;
    Ok((
        seconds,
        trace.durations_ms(cpu_span as f64 / gpu_span as f64)?,
    ))
}

#[allow(clippy::too_many_arguments)]
pub(super) fn profile_proposals(
    command: &CommandBufferRef,
    runtime: &Runtime,
    pipelines: &Pipelines,
    tensorops: &TensorOpsPipelines,
    pisa: &Pisa1,
    buffers: &Buffers,
    weights: CandidateRow<'_>,
    readout: ProposalReadout<'_>,
    rows: u32,
    cache: Option<&[decode::Cache]>,
    hidden_only: bool,
    trace: &ScorerStageTrace,
) -> Result<(f64, std::collections::BTreeMap<&'static str, f64>), String> {
    if hidden_only {
        encode_hidden(
            command,
            pipelines,
            tensorops,
            pisa,
            buffers,
            weights,
            readout,
            rows,
            cache,
            Some(trace),
        )?;
    } else {
        encode_proposals(
            command,
            pipelines,
            tensorops,
            pisa,
            buffers,
            weights,
            readout,
            rows,
            cache,
            Some(trace),
        )?;
    }
    trace.resolve(command);
    let (mut cpu_start, mut gpu_start) = (0, 0);
    runtime
        .device
        .sample_timestamps(&mut cpu_start, &mut gpu_start);
    command.commit();
    let seconds = complete_committed(command)?;
    let (mut cpu_end, mut gpu_end) = (0, 0);
    runtime.device.sample_timestamps(&mut cpu_end, &mut gpu_end);
    let cpu_span = cpu_end
        .checked_sub(cpu_start)
        .ok_or("nonmonotonic generation CPU timestamp")?;
    let gpu_span = gpu_end
        .checked_sub(gpu_start)
        .filter(|span| *span > 0)
        .ok_or("invalid generation GPU timestamp calibration")?;
    Ok((
        seconds,
        trace.durations_ms(cpu_span as f64 / gpu_span as f64)?,
    ))
}

#[allow(clippy::too_many_arguments)]
pub(super) fn denoise_chunk(
    command: &CommandBufferRef,
    pipelines: &Pipelines,
    tensorops: &TensorOpsPipelines,
    pisa: &Pisa1,
    buffers: &Buffers,
    weights: CandidateRow<'_>,
    readout: ProposalReadout<'_>,
    start: u32,
    rows: u32,
    cache: &[decode::Cache],
    hidden_only: bool,
    input: diffusion::Input<'_>,
) -> Result<(), String> {
    if rows == 0
        || rows > 4096
        || rows % 128 != 0
        || start % 128 != 0
        || start
            .checked_add(rows)
            .is_none_or(|end| end > pisa.context())
        || cache.first().is_none_or(|entry| entry.kv.is_none())
    {
        return Err("invalid diffusion chunk".into());
    }
    proposals_at(
        command,
        pipelines,
        tensorops,
        pisa,
        buffers,
        weights,
        ProposalPlan {
            readout,
            row_start: start,
            rows,
            cache: Some(cache),
            dirty: true,
            trace: input.trace,
            hidden_only,
            diffusion: Some(input),
        },
    )
}

fn proposals_at(
    command: &CommandBufferRef,
    pipelines: &Pipelines,
    tensorops: &TensorOpsPipelines,
    pisa1: &Pisa1,
    buffers: &Buffers,
    weights: CandidateRow<'_>,
    plan: ProposalPlan<'_>,
) -> Result<(), String> {
    if let Some(trace) = plan.trace {
        trace.reset();
    }
    buffers.fine_grained.reset_routes(command);
    let scorer =
        Scorer {
            command,
            encoder: RefCell::new(plan.trace.is_none().then(|| {
                command.compute_command_encoder_with_dispatch_type(MTLDispatchType::Serial)
            })),
            pipelines,
            tensorops,
            pisa1,
            buffers,
            weights,
            row_start: plan.row_start,
            rows: plan.rows,
            active_rows: std::cell::Cell::new(plan.rows),
            cache: plan.cache,
            dirty: plan.dirty,
            trace: plan.trace,
            feedback: weights
                .architecture
                .feedback_transition(plan.readout.feedback),
            diffusion: plan.diffusion,
            cache_only: plan.hidden_only && plan.diffusion.is_some(),
        };
    let result = scorer.encode_hidden().and_then(|()| {
        if let Some(hidden) = plan.readout.hidden {
            scorer.capture_hidden(hidden)?;
        }
        if plan.hidden_only {
            return Ok(());
        }
        scorer.start_stage("proposal_readout")?;
        scorer.readout_proposals(
            plan.readout.output,
            plan.readout.seeds,
            plan.readout.temperature,
            plan.readout.score_targets,
        )
    });
    if let Some(encoder) = scorer.encoder.borrow_mut().take() {
        encoder.end_encoding();
    }
    result
}

pub(super) fn objective_trace(
    command: &CommandBufferRef,
    pipelines: &Pipelines,
    tensorops: &TensorOpsPipelines,
    pisa1: &Pisa1,
    buffers: &Buffers,
    weights: CandidateRow<'_>,
    trace: Option<&ScorerStageTrace>,
) -> Result<(), String> {
    if let Some(trace) = trace {
        trace.reset();
    }
    let scorer =
        Scorer {
            command,
            encoder: RefCell::new(trace.is_none().then(|| {
                command.compute_command_encoder_with_dispatch_type(MTLDispatchType::Serial)
            })),
            pipelines,
            tensorops,
            pisa1,
            buffers,
            weights,
            row_start: 0,
            rows: ROWS,
            active_rows: std::cell::Cell::new(ROWS),
            cache: None,
            dirty: false,
            trace,
            feedback: weights
                .architecture
                .feedback_transition(crate::config::FeedbackTransition::ProjectedSigmoid),
            diffusion: None,
            cache_only: false,
        };
    let result = scorer.encode();
    if let Some(encoder) = scorer.encoder.borrow_mut().take() {
        encoder.end_encoding();
    }
    result
}

impl Scorer<'_> {
    fn scratch_start(&self) -> u32 {
        if self.cache.is_some_and(|entries| entries[0].kv.is_some()) {
            0
        } else {
            self.row_start
        }
    }

    pub(super) fn activation_offset(&self) -> u64 {
        half_bytes(u64::from(self.scratch_start()) * u64::from(WIDTH))
    }

    fn qkv_offset(&self) -> u64 {
        half_bytes(u64::from(self.scratch_start()) * u64::from(QKV_WIDTH))
    }

    fn qkv(&self, execution: u32) -> &BufferRef {
        self.layer_cache(execution)
            .map_or(&self.buffers.qkv, |cache| &cache.qkv)
    }

    fn layer_cache(&self, execution: u32) -> Option<&decode::Cache> {
        self.cache.map(|cache| &cache[execution as usize])
    }

    fn next_norm(&self, layer: u32, pass: u32) -> (&BufferRef, u64) {
        if layer + 1 < MODEL_LAYERS {
            (
                self.weights.buffer,
                self.weights.attention_norm + half_bytes(u64::from(layer + 1) * u64::from(WIDTH)),
            )
        } else if pass + 1 < FEEDBACK_PASSES {
            if self.feedback == crate::config::FeedbackTransition::ProjectedSigmoid {
                (&self.buffers.unit_norm, 0)
            } else {
                (self.weights.buffer, self.weights.attention_norm)
            }
        } else {
            (self.weights.buffer, self.weights.final_norm)
        }
    }

    pub(super) fn attention_offsets(layer: u32) -> (u64, u64) {
        (
            half_bytes(u64::from(layer) * u64::from(WIDTH) * u64::from(QKV_WIDTH)),
            half_bytes(u64::from(layer) * u64::from(WIDTH) * u64::from(WIDTH)),
        )
    }

    pub(super) fn start_stage(&self, stage: &'static str) -> Result<(), String> {
        if let Some(trace) = self.trace {
            if let Some(encoder) = self.encoder.borrow_mut().take() {
                encoder.end_encoding();
            }
            self.encoder
                .replace(Some(trace.encoder(self.command, stage)?));
        } else {
            crate::apple_gpu::barrier_buffers(self.active_encoder());
        }
        Ok(())
    }

    pub(super) fn active_encoder(&self) -> &ComputeCommandEncoderRef {
        self.encoder
            .borrow()
            .expect("scorer stage encoder initialized")
    }

    fn encode(&self) -> Result<(), String> {
        self.encode_hidden()?;
        self.start_stage("readout")?;
        self.readout()
    }

    fn encode_hidden(&self) -> Result<(), String> {
        self.start_stage("embed")?;
        self.embed()?;
        let mut input: &BufferRef = &self.buffers.input;
        let diffusion_steps = self
            .diffusion
            .map(|input| {
                crate::forward_program::RecurrentCore::selective_fbt()
                    .layer_visits(MODEL_LAYERS as usize, input.visits as usize)
            })
            .transpose()?;
        let dynamic_steps = diffusion_steps.map(|steps| {
            steps
                .into_iter()
                .map(|step| LayerStep {
                    layer: step.layer as u32,
                    pass: step.visit as u32,
                    execution: step.execution as u32,
                })
                .collect::<Vec<_>>()
        });
        let steps = dynamic_steps
            .as_deref()
            .unwrap_or_else(|| self.weights.architecture.layer_steps());
        for step in steps {
            if step.execution as usize + 1 == steps.len() {
                self.hnet_dechunk()?;
            }
            // KV for the final execution depends only on its attention input.
            // Its residual update, FFN and final readout cannot affect any
            // cache.
            if self.cache_only && step.execution as usize + 1 == steps.len() {
                self.start_stage("mhc_attention_input")?;
                let normalized = self.activations(&self.buffers.normalized);
                self.block()?
                    .stage_attention(self.active_encoder(), step.layer, normalized.out())
                    .discard();
                self.attention(step.layer, step.execution)?;
                return Ok(());
            }
            self.layer(input, step.layer, step.pass, step.execution)?;
            if step.execution == 0 {
                self.hnet_chunk()?;
            }
            input = &self.buffers.output;
            if !self.weights.architecture.is_multistream()
                && step.layer + 1 == MODEL_LAYERS
                && step.pass + 1 < FEEDBACK_PASSES
            {
                if self.feedback == crate::config::FeedbackTransition::ProjectedSigmoid {
                    self.start_stage("feedback")?;
                    self.feedback(input)?;
                    input = &self.buffers.feedback;
                }
            }
        }
        if self.weights.architecture.is_multistream() {
            self.start_stage("mhc_readout")?;
            let normalized = self.activations(&self.buffers.normalized);
            self.hyper()
                .finalize(self.active_encoder(), normalized.out());
        }
        Ok(())
    }

    pub(super) fn encode_to(&self, target: ReplaySite) -> Result<(), String> {
        self.embed()?;
        for step in self.weights.architecture.layer_steps() {
            self.mhc_attention(step.layer, step.execution)?;
            if target == ReplaySite::Attention(step.execution) {
                return Ok(());
            }
            self.mhc_moe(step.layer)?;
            if target == ReplaySite::Moe(step.execution) {
                return Ok(());
            }
        }
        Err("training replay target is outside the recurrent core".into())
    }

    fn attention(&self, layer: u32, execution: u32) -> Result<(), String> {
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
            crate::nn::Rows::Chunk(self.rows),
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

    fn feedback(&self, input: &BufferRef) -> Result<(), String> {
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
            crate::nn::Rows::Chunk(self.rows),
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
