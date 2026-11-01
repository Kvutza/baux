use super::*;

pub(super) struct ProbeBuffers {
    pub(super) router: Buffer,
    pub(super) gates: Buffer,
    pub(super) grouped: Buffer,
    pub(super) gate_up_base: Buffer,
    pub(super) gate_up: Buffer,
    pub(super) gate_inner: Buffer,
    pub(super) gate_outer: Buffer,
    pub(super) activation: Buffer,
    pub(super) down_base: Buffer,
    pub(super) down_inner: Buffer,
    pub(super) down_outer: Buffer,
    pub(super) mps_gate_up: Buffer,
    pub(super) mps_down: Buffer,
    pub(super) materialized_gate_up: Buffer,
    pub(super) materialized_down: Buffer,
    pub(super) qkv_weights: Buffer,
    pub(super) output_projection_weights: Buffer,
    pub(super) mps_qkv: Buffer,
    pub(super) mps_projected: Buffer,
    pub(super) feedback_state_weights: Buffer,
    pub(super) feedback_gate_weights: Buffer,
    pub(super) readout_weights: Buffer,
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(crate) struct HnetHierarchyStats {
    pub(crate) total_bytes: u32,
    pub(crate) active_chunks: u32,
    pub(crate) aligned_rows: u32,
    pub(crate) capacity_rows: u32,
    pub(crate) true_ratio: f32,
    pub(crate) mean_prob: f32,
    pub(crate) target_ratio: f32,
    pub(crate) ratio_loss: f32,
    pub(crate) ratio_penalty: f32,
    pub(crate) degenerate_all_open: bool,
    pub(crate) degenerate_all_closed: bool,
    pub(crate) stage0_gpu_seconds: f64,
    pub(crate) stage1_gpu_seconds: f64,
}

pub(super) struct HnetScratch {
    pub(super) waves: crate::hnet_metal::HNetWaves,
    pub(super) encoder_streams: Buffer,
    pub(super) chunk_scratch: Buffer,
    pub(super) origins: Buffer,
    pub(super) probs: Buffer,
    pub(super) plug_idx: Buffer,
    pub(super) projected: Buffer,
    pub(super) residual: Buffer,
    pub(super) count: Buffer,
    pub(super) core: Buffer,
    pub(super) q_state: Buffer,
    pub(super) zero_seen: Buffer,
    pub(super) one_seen: Buffer,
    pub(super) ema_in: Buffer,
    pub(super) ema_out: Buffer,
    pub(super) part_a: Buffer,
    pub(super) part_b: Buffer,
    pub(super) tile_seen: std::cell::Cell<bool>,
    pub(super) ema_parity: std::cell::Cell<usize>,
    pub(super) last_stats: std::cell::Cell<HnetHierarchyStats>,
}

pub(super) struct Buffers {
    pub(super) rows: u32,
    pub(super) context_cap: u32,
    pub(super) fine_grained: routing::FineGrainedMoeBuffers,
    pub(super) fine_grained_grad: std::cell::OnceCell<Result<routing_grad::MoeGradBuffers, String>>,
    pub(super) hnet: std::cell::OnceCell<Result<HnetScratch, String>>,
    pub(super) hnet2: std::cell::OnceCell<Result<Hnet2Scratch, String>>,
    pub(super) quantized_gateup: std::cell::OnceCell<Result<Buffer, String>>,
    pub(super) probe: std::cell::OnceCell<ProbeBuffers>,
    pub(super) input: Buffer,
    pub(super) down: Buffer,
    pub(super) output: Buffer,
    pub(super) qkv: Buffer,
    pub(super) projected: Buffer,
    pub(super) feedback_state: Buffer,
    pub(super) feedback_gate: Buffer,
    pub(super) feedback: Buffer,
    pub(super) logits: std::cell::OnceCell<Result<Buffer, String>>,
    pub(super) loss_partials: std::cell::OnceCell<Result<Buffer, String>>,
    pub(super) proposal_partials: std::cell::OnceCell<Result<Buffer, String>>,
    pub(super) labels: Buffer,
    pub(super) losses: Buffer,
    pub(super) score_mask: Buffer,
    pub(super) sequence_scores: Buffer,
    pub(super) tokens: Buffer,
    pub(super) unit_norm: Buffer,
    pub(super) normalized: Buffer,
    pub(super) attention_state: Buffer,
    pub(super) mhc_streams: [Buffer; 2],
    pub(super) mhc_coefficients: Buffer,
    pub(super) rope: Buffer,
}

impl ProbeBuffers {
    fn new(runtime: &Runtime, rows: u32) -> Self {
        let half = |n: u32| n as usize;
        Self {
            router: patterned(runtime, half(WIDTH * EXPERTS), 0x85a3_08d3, 0x1800),
            gates: runtime.buffer::<u16>(half(rows)),
            grouped: runtime.buffer::<u16>(half(rows * WIDTH)),
            gate_up_base: patterned(
                runtime,
                half(EXPERTS * WIDTH * GATE_UP),
                0x1319_8a2e,
                0x2000,
            ),
            gate_up: canary(runtime, (rows * GATE_UP) as usize),
            gate_inner: factor(runtime, EXPERTS, 32, 32),
            gate_outer: factor(runtime, EXPERTS, 16, 64),
            activation: canary(runtime, (rows * EXPERT_WIDTH) as usize),
            down_base: patterned(
                runtime,
                half(EXPERTS * EXPERT_WIDTH * WIDTH),
                0x0370_7344,
                0x2000,
            ),
            down_inner: factor(runtime, EXPERTS, 32, 16),
            down_outer: factor(runtime, EXPERTS, 32, 32),
            mps_gate_up: canary(runtime, (rows * GATE_UP) as usize),
            mps_down: canary(runtime, (rows * WIDTH) as usize),
            materialized_gate_up: canary(runtime, (EXPERTS * WIDTH * GATE_UP) as usize),
            materialized_down: canary(runtime, (EXPERTS * EXPERT_WIDTH * WIDTH) as usize),
            qkv_weights: patterned(runtime, half(WIDTH * QKV_WIDTH), 0xa409_3822, 0x2000),
            output_projection_weights: patterned(runtime, half(WIDTH * WIDTH), 0x299f_31d0, 0x2000),
            mps_qkv: canary(runtime, (rows * QKV_WIDTH) as usize),
            mps_projected: canary(runtime, (rows * WIDTH) as usize),
            feedback_state_weights: patterned(runtime, half(WIDTH * WIDTH), 0x082e_fa98, 0x2000),
            feedback_gate_weights: patterned(runtime, half(WIDTH * WIDTH), 0xec4e_6c89, 0x2000),
            readout_weights: patterned(runtime, half(WIDTH * VOCAB), 0x4528_21e6, 0x1800),
        }
    }
}

impl std::ops::Deref for Buffers {
    type Target = ProbeBuffers;

    fn deref(&self) -> &Self::Target {
        self.probe.get_or_init(|| {
            let runtime = Runtime::shared().expect("shared Metal runtime");
            ProbeBuffers::new(&runtime, self.rows)
        })
    }
}

impl Buffers {
    pub(super) fn hnet(&self) -> Result<&HnetScratch, String> {
        self.hnet
            .get_or_init(|| {
                let runtime = Runtime::shared()?;
                let waves = crate::hnet_metal::HNetWaves::with_runtime(runtime.clone())?;
                let stream_elems = (self.rows * MHC_STREAMS * WIDTH) as usize;
                let row_elems = self.rows as usize;
                let width = WIDTH as usize;
                let parts = crate::hnet_metal::EMA_PARTS;
                Ok(HnetScratch {
                    waves,
                    encoder_streams: runtime.buffer::<u16>(stream_elems / 4),
                    chunk_scratch: runtime.buffer::<f32>(stream_elems / 4),
                    origins: runtime.buffer::<u32>(row_elems),
                    probs: runtime.buffer::<f32>(row_elems),
                    plug_idx: runtime.buffer::<u32>(row_elems),
                    projected: runtime.buffer::<f32>(stream_elems / 2),
                    residual: runtime.buffer::<f32>(stream_elems / 4),
                    count: runtime.buffer::<u32>(1),
                    core: runtime.buffer::<u16>(stream_elems / 4),
                    q_state: runtime.buffer_with(&vec![0.0f32; width]),
                    zero_seen: runtime.buffer_with(&[0u32]),
                    one_seen: runtime.buffer_with(&[1u32]),
                    ema_in: runtime.buffer_with(&vec![0.0f32; width]),
                    ema_out: runtime.buffer_with(&vec![0.0f32; width]),
                    part_a: runtime.buffer::<f32>(parts * width),
                    part_b: runtime.buffer::<f32>(parts * width),
                    tile_seen: std::cell::Cell::new(false),
                    ema_parity: std::cell::Cell::new(0),
                    last_stats: std::cell::Cell::new(HnetHierarchyStats::default()),
                })
            })
            .as_ref()
            .map_err(Clone::clone)
    }

    pub(super) fn hnet2(&self) -> Result<&Hnet2Scratch, String> {
        self.hnet2
            .get_or_init(|| {
                let runtime = Runtime::shared()?;
                let waves = crate::hnet_metal::HNetWaves::with_runtime(runtime.clone())?;
                Ok(Hnet2Scratch::new(&runtime, waves, self.context_cap))
            })
            .as_ref()
            .map_err(Clone::clone)
    }

    pub(super) fn moe_grad(&self) -> Result<&routing_grad::MoeGradBuffers, String> {
        self.fine_grained_grad
            .get_or_init(|| {
                Runtime::shared()
                    .map(|runtime| routing_grad::MoeGradBuffers::new(&runtime, self.rows))
            })
            .as_ref()
            .map_err(Clone::clone)
    }

    pub(super) fn quantized_gateup(&self) -> Result<&BufferRef, String> {
        self.quantized_gateup
            .get_or_init(|| {
                Runtime::shared().map(|runtime| {
                    runtime.buffer::<i8>(MODEL_LAYERS as usize * routing::GATE_PARAMETERS)
                })
            })
            .as_ref()
            .map(|buffer| buffer.as_ref())
            .map_err(Clone::clone)
    }

    pub(super) fn logits(&self) -> Result<&BufferRef, String> {
        self.logits
            .get_or_init(|| {
                Runtime::shared().map(|runtime| runtime.buffer::<u16>((self.rows * VOCAB) as usize))
            })
            .as_ref()
            .map(|buffer| buffer.as_ref())
            .map_err(Clone::clone)
    }

    pub(super) fn proposal_partials(&self) -> Result<&BufferRef, String> {
        self.proposal_partials
            .get_or_init(|| {
                Runtime::shared().map(|runtime| {
                    runtime.buffer::<[f32; 4]>(self.rows.max(ROWS) as usize * (VOCAB / 64) as usize)
                })
            })
            .as_ref()
            .map(|buffer| buffer.as_ref())
            .map_err(Clone::clone)
    }

    pub(super) fn loss_partials(&self) -> Result<&BufferRef, String> {
        self.loss_partials
            .get_or_init(|| {
                Runtime::shared().map(|runtime| {
                    runtime.buffer::<[f32; 4]>(self.rows.max(ROWS) as usize * (VOCAB / 64) as usize)
                })
            })
            .as_ref()
            .map(|buffer| buffer.as_ref())
            .map_err(Clone::clone)
    }

    pub(super) fn new(runtime: &Runtime) -> Self {
        Self::sized(runtime, ROWS)
    }

    fn sized(runtime: &Runtime, rows: u32) -> Self {
        Self::with_capacity(runtime, rows, None)
    }

    pub(super) fn with_capacity(runtime: &Runtime, rows: u32, context: Option<(u32, u32)>) -> Self {
        let half = |n: u32| n as usize;
        let residual = residual_buffers(runtime, rows, rows);
        let (tokens, labels, losses, score_mask, rope) = match context {
            Some((ctx, rope_ctx)) => {
                let cap = (ctx as usize).max(ROWS as usize);
                (
                    runtime.buffer_with(&vec![0u32; cap]),
                    runtime.buffer_with(&vec![0u32; cap]),
                    runtime.buffer_with(&vec![f32::NAN; cap + 64]),
                    runtime.buffer_with(&vec![1u8; cap]),
                    rope_table(runtime, rope_ctx.max(MAX_CONTEXT)),
                )
            }
            None => (
                synthetic_buffer(runtime, 0),
                synthetic_buffer(runtime, 1),
                runtime.buffer_with(&vec![f32::NAN; rows as usize + 64]),
                runtime.buffer_with(&vec![1u8; rows as usize]),
                rope_table(runtime, MAX_CONTEXT),
            ),
        };
        let context_cap = context.map_or(rows, |(ctx, _)| ctx.max(rows));
        Self {
            rows,
            context_cap,
            fine_grained: routing::FineGrainedMoeBuffers::for_rows(runtime, rows),
            fine_grained_grad: std::cell::OnceCell::new(),
            hnet: std::cell::OnceCell::new(),
            hnet2: std::cell::OnceCell::new(),
            quantized_gateup: std::cell::OnceCell::new(),
            probe: std::cell::OnceCell::new(),
            input: patterned(runtime, half(rows * WIDTH), 0x243f_6a88, 0x2c00),
            down: canary(runtime, (rows * WIDTH) as usize),
            output: canary(runtime, (rows * WIDTH) as usize),
            qkv: canary(runtime, (rows * QKV_WIDTH) as usize),
            projected: canary(runtime, (rows * WIDTH) as usize),
            feedback_state: residual.feedback_state,
            feedback_gate: residual.feedback_gate,
            feedback: residual.feedback,
            logits: std::cell::OnceCell::new(),
            loss_partials: std::cell::OnceCell::new(),
            proposal_partials: std::cell::OnceCell::new(),
            labels,
            losses,
            score_mask,
            sequence_scores: runtime.buffer_with(&[f32::NAN; BATCH as usize]),
            tokens,
            unit_norm: residual.unit_norm,
            normalized: residual.normalized,
            attention_state: residual.attention_state,
            mhc_streams: residual.mhc_streams,
            mhc_coefficients: residual.mhc_coefficients,
            rope,
        }
    }

    pub(super) fn for_context(
        runtime: &Runtime,
        context: u32,
        architecture: ResidualArchitecture,
    ) -> Self {
        let work_rows = if architecture.is_mhc4() && context > ROWS {
            ROWS
        } else {
            context.max(ROWS)
        };
        Self::with_capacity(runtime, work_rows, Some((context, context)))
    }
}

impl HnetScratch {
    pub(super) fn reset_sequence(&self) -> Result<(), String> {
        let d = WIDTH as usize;
        crate::apple_gpu::slice_mut::<f32>(&self.q_state, d)?.fill(0.0);
        crate::apple_gpu::slice_mut::<f32>(&self.ema_in, d)?.fill(0.0);
        crate::apple_gpu::slice_mut::<f32>(&self.ema_out, d)?.fill(0.0);
        self.tile_seen.set(false);
        self.ema_parity.set(0);
        Ok(())
    }

    pub(super) fn record_stats(
        &self,
        total_bytes: u32,
        capacity_rows: u32,
        aligned_rows: u32,
        stage0_gpu_seconds: f64,
        stage1_gpu_seconds: f64,
    ) -> Result<HnetHierarchyStats, String> {
        let active_chunks = crate::apple_gpu::buffer_slice::<u32>(&self.count, 1)?[0];
        let probs = crate::apple_gpu::buffer_slice::<f32>(&self.probs, total_bytes as usize)?;
        let prob_sum: f64 = probs.iter().map(|&p| f64::from(p)).sum();
        let denom = (total_bytes.max(1)) as f32;
        let true_ratio = (active_chunks as f32) / denom;
        let mean_prob = (prob_sum as f32) / denom;
        let target_ratio = 4.0f32;
        let rl = |f: f32, g: f32| {
            (target_ratio / (target_ratio - 1.0))
                * ((1.0 - f) * (1.0 - g) + (target_ratio - 1.0) * f * g)
        };
        let ratio_loss = rl(true_ratio, mean_prob);
        let dev_f = (rl(true_ratio, true_ratio) - 1.0).max(0.0);
        let dev_g = (rl(mean_prob, mean_prob) - 1.0).max(0.0);
        let ratio_penalty = 0.03 * 0.5 * (dev_f + dev_g);
        let stats = HnetHierarchyStats {
            total_bytes,
            active_chunks,
            aligned_rows,
            capacity_rows,
            true_ratio,
            mean_prob,
            target_ratio,
            ratio_loss,
            ratio_penalty,
            degenerate_all_open: active_chunks == total_bytes && total_bytes > 1,
            degenerate_all_closed: active_chunks == 1 && total_bytes > 1,
            stage0_gpu_seconds,
            stage1_gpu_seconds,
        };
        self.last_stats.set(stats);
        Ok(stats)
    }
}

struct ResidualBuffers {
    feedback_state: Buffer,
    feedback_gate: Buffer,
    feedback: Buffer,
    unit_norm: Buffer,
    normalized: Buffer,
    attention_state: Buffer,
    mhc_streams: [Buffer; 2],
    mhc_coefficients: Buffer,
}

fn residual_buffers(runtime: &Runtime, rows: u32, legacy_rows: u32) -> ResidualBuffers {
    ResidualBuffers {
        feedback_state: runtime.buffer::<u16>((legacy_rows * WIDTH) as usize),
        feedback_gate: runtime.buffer::<u16>((legacy_rows * WIDTH) as usize),
        feedback: runtime.buffer::<u16>((legacy_rows * WIDTH) as usize),
        unit_norm: filled(runtime, WIDTH as usize, 0x3c00),
        normalized: runtime.buffer::<u16>((rows * WIDTH) as usize),
        attention_state: runtime.buffer::<u16>((legacy_rows * WIDTH) as usize),
        mhc_streams: [
            runtime.buffer::<u16>((rows * MHC_STREAMS * WIDTH) as usize),
            runtime.buffer::<u16>((rows * MHC_STREAMS * WIDTH) as usize),
        ],
        mhc_coefficients: runtime.buffer::<f32>((rows * MHC_COEFFICIENTS) as usize),
    }
}
