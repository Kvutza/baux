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

pub(super) struct HnetScratch {
    pub(super) waves: crate::hnet_metal::HNetWaves,
    pub(super) encoder_streams: Buffer,
    pub(super) chunk_scratch: Buffer,
    pub(super) origins: Buffer,
    pub(super) probs: Buffer,
    pub(super) plug_idx: Buffer,
}

pub(super) struct Buffers {
    pub(super) rows: u32,
    pub(super) fine_grained: routing::FineGrainedMoeBuffers,
    pub(super) fine_grained_grad: std::cell::OnceCell<Result<routing_grad::MoeGradBuffers, String>>,
    pub(super) hnet: std::cell::OnceCell<Result<HnetScratch, String>>,
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
            gate_outer: factor(runtime, EXPERTS, 16, 54),
            activation: canary(runtime, (rows * EXPERT_WIDTH) as usize),
            down_base: patterned(
                runtime,
                half(EXPERTS * EXPERT_WIDTH * WIDTH),
                0x0370_7344,
                0x2000,
            ),
            down_inner: factor(runtime, EXPERTS, 27, 16),
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
                Ok(HnetScratch {
                    waves,
                    encoder_streams: runtime.buffer::<u16>(stream_elems),
                    chunk_scratch: runtime.buffer::<u16>(stream_elems),
                    origins: runtime.buffer::<u32>(row_elems),
                    probs: runtime.buffer::<f32>(row_elems),
                    plug_idx: runtime.buffer::<u32>(row_elems),
                })
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

    fn with_capacity(runtime: &Runtime, rows: u32, context: Option<u32>) -> Self {
        let half = |n: u32| n as usize;
        let residual = residual_buffers(runtime, rows);
        let (tokens, labels, losses, score_mask, rope) = match context {
            Some(ctx) => {
                let cap = (ctx as usize).max(ROWS as usize);
                (
                    runtime.buffer_with(&vec![0u32; cap]),
                    runtime.buffer_with(&vec![0u32; cap]),
                    runtime.buffer_with(&vec![f32::NAN; cap + 64]),
                    runtime.buffer_with(&vec![1u8; cap]),
                    rope_table(runtime, ctx.max(MAX_CONTEXT)),
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
        Self {
            rows,
            fine_grained: routing::FineGrainedMoeBuffers::for_rows(runtime, rows),
            fine_grained_grad: std::cell::OnceCell::new(),
            hnet: std::cell::OnceCell::new(),
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
        Self::with_capacity(runtime, work_rows, Some(context))
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

fn residual_buffers(runtime: &Runtime, rows: u32) -> ResidualBuffers {
    ResidualBuffers {
        feedback_state: runtime.buffer::<u16>((rows * WIDTH) as usize),
        feedback_gate: runtime.buffer::<u16>((rows * WIDTH) as usize),
        feedback: runtime.buffer::<u16>((rows * WIDTH) as usize),
        unit_norm: filled(runtime, WIDTH as usize, 0x3c00),
        normalized: runtime.buffer::<u16>((rows * WIDTH) as usize),
        attention_state: runtime.buffer::<u16>((rows * WIDTH) as usize),
        mhc_streams: [
            runtime.buffer::<u16>((rows * MHC_STREAMS * WIDTH) as usize),
            runtime.buffer::<u16>((rows * MHC_STREAMS * WIDTH) as usize),
        ],
        mhc_coefficients: runtime.buffer::<f32>((rows * MHC_COEFFICIENTS) as usize),
    }
}
