use std::sync::Arc;
use std::time::Instant;

use baux_cuda_device::{FbtShape, FeedbackShape, MatmulShape, PisaShape, RouteShape, fbt_model};
use cuda_core::{
    CudaContext, CudaEvent, CudaStream, DeviceBuffer, LaunchConfig1D, simt::LaunchConfig2D,
};

use crate::capture::{CudaResult, copy_prefix, cuda_error, read_prefix, timing_event};

const WIDTH: usize = baux::semantics::mhc4::WIDTH;
const VOCAB: usize = baux_cuda_device::VOCAB as usize;
const QKV: usize = 640;
const CONTEXT: usize = 4096;
const HEADS: usize = 8;
const SELECTED: usize = 8;
const PISA_NODES: usize = 127;

pub(crate) fn pack_embedding<T: Copy + Default>(
    embedding: &[T],
    width: usize,
    vocab: usize,
) -> Vec<T> {
    let mut packed = vec![T::default(); embedding.len()];
    for column in 0..width {
        for token in 0..vocab {
            packed[token * width + column] = embedding[column * vocab + token];
        }
    }
    packed
}

#[derive(Debug, Clone, Copy, Default)]
pub struct FbtPrefillProfile {
    pub embed_ms: f32,
    pub rms_ms: f32,
    pub projection_ms: f32,
    pub tree_ms: f32,
    pub selection_ms: f32,
    pub attention_ms: f32,
    pub residual_ms: f32,
    pub routing_ms: f32,
    pub device_ms: f32,
    /// Wall-clock interval covering token upload through copied-back QKV
    /// output.
    pub end_to_end_ms: f32,
}

/// Resident buffers for the embedding-through-PISA attention slice.
/// This is not yet a complete FBT layer or generation path.
pub struct FbtPrefill {
    context: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    module: fbt_model::LoadedModule,
    embedding: DeviceBuffer<f32>,
    rms_weight: DeviceBuffer<f32>,
    qkv_weight: DeviceBuffer<f32>,
    output_weight: DeviceBuffer<f32>,
    router_weight: DeviceBuffer<f32>,
    tokens: DeviceBuffer<u32>,
    embedded: DeviceBuffer<f32>,
    normalized: DeviceBuffer<f32>,
    qkv: DeviceBuffer<f32>,
    pyramid: DeviceBuffer<f32>,
    blocks: DeviceBuffer<u32>,
    attended: DeviceBuffer<f32>,
    projected: DeviceBuffer<f32>,
    residual: DeviceBuffer<f32>,
    route_norm: DeviceBuffer<f32>,
    route_scores: DeviceBuffer<f32>,
    route_experts: DeviceBuffer<u32>,
    route_weights: DeviceBuffer<f32>,
    route_margins: DeviceBuffer<f32>,
    capacity: usize,
}

fn widen(slice: &[u16]) -> Vec<f32> {
    slice
        .iter()
        .map(|&bits| half::f16::from_bits(bits).to_f32())
        .collect()
}

impl FbtPrefill {
    pub fn new(
        embedding: &[u16],
        rms_weight: &[u16],
        qkv_weight: &[u16],
        output_weight: &[u16],
        router_weight: &[u16],
    ) -> CudaResult<Self> {
        if embedding.len() != VOCAB * WIDTH
            || rms_weight.len() != WIDTH
            || qkv_weight.len() != WIDTH * QKV
            || output_weight.len() != WIDTH * WIDTH
            || router_weight.len() != WIDTH * 128
        {
            return Err("FBT attention weights have an invalid shape".into());
        }
        let context = CudaContext::new(0).map_err(cuda_error)?;
        let stream = context.default_stream();
        let module = crate::capture::load_model(&context)?;
        let capacity = 1;
        let packed_embedding = pack_embedding(&widen(embedding), WIDTH, VOCAB);
        let embedding = DeviceBuffer::from_host(&stream, &packed_embedding).map_err(cuda_error)?;
        let rms_weight =
            DeviceBuffer::from_host(&stream, &widen(rms_weight)).map_err(cuda_error)?;
        let qkv_weight =
            DeviceBuffer::from_host(&stream, &widen(qkv_weight)).map_err(cuda_error)?;
        let output_weight =
            DeviceBuffer::from_host(&stream, &widen(output_weight)).map_err(cuda_error)?;
        let router_weight =
            DeviceBuffer::from_host(&stream, &widen(router_weight)).map_err(cuda_error)?;
        let tokens = DeviceBuffer::zeroed(&stream, capacity).map_err(cuda_error)?;
        let embedded = DeviceBuffer::zeroed(&stream, capacity * WIDTH).map_err(cuda_error)?;
        let normalized = DeviceBuffer::zeroed(&stream, capacity * WIDTH).map_err(cuda_error)?;
        let qkv = DeviceBuffer::zeroed(&stream, capacity * QKV).map_err(cuda_error)?;
        let pyramid = DeviceBuffer::zeroed(&stream, PISA_NODES * 64).map_err(cuda_error)?;
        let blocks = DeviceBuffer::zeroed(&stream, CONTEXT * SELECTED).map_err(cuda_error)?;
        let attended = DeviceBuffer::zeroed(&stream, CONTEXT * WIDTH).map_err(cuda_error)?;
        let projected = DeviceBuffer::zeroed(&stream, CONTEXT * WIDTH).map_err(cuda_error)?;
        let residual = DeviceBuffer::zeroed(&stream, CONTEXT * WIDTH).map_err(cuda_error)?;
        let route_norm = DeviceBuffer::zeroed(&stream, CONTEXT * WIDTH).map_err(cuda_error)?;
        let route_scores = DeviceBuffer::zeroed(&stream, CONTEXT * 128).map_err(cuda_error)?;
        let route_experts = DeviceBuffer::zeroed(&stream, CONTEXT * 3).map_err(cuda_error)?;
        let route_weights = DeviceBuffer::zeroed(&stream, CONTEXT * 3).map_err(cuda_error)?;
        let route_margins = DeviceBuffer::zeroed(&stream, CONTEXT).map_err(cuda_error)?;
        Ok(Self {
            context,
            stream,
            module,
            embedding,
            rms_weight,
            qkv_weight,
            output_weight,
            router_weight,
            tokens,
            embedded,
            normalized,
            qkv,
            pyramid,
            blocks,
            attended,
            projected,
            residual,
            route_norm,
            route_scores,
            route_experts,
            route_weights,
            route_margins,
            capacity,
        })
    }

    fn reserve(&mut self, rows: usize) -> CudaResult<()> {
        if rows <= self.capacity {
            return Ok(());
        }
        let capacity = rows.next_power_of_two();
        self.tokens = DeviceBuffer::zeroed(&self.stream, capacity).map_err(cuda_error)?;
        self.embedded = DeviceBuffer::zeroed(&self.stream, capacity * WIDTH).map_err(cuda_error)?;
        self.normalized =
            DeviceBuffer::zeroed(&self.stream, capacity * WIDTH).map_err(cuda_error)?;
        self.qkv = DeviceBuffer::zeroed(&self.stream, capacity * QKV).map_err(cuda_error)?;
        self.capacity = capacity;
        Ok(())
    }

    pub fn run(&mut self, tokens: &[u32]) -> CudaResult<(Vec<u16>, Vec<u32>, FbtPrefillProfile)> {
        if tokens.len() != CONTEXT || tokens.iter().any(|&token| token as usize >= VOCAB) {
            return Err("FBT attention requires 4096 token IDs below vocab 262144".into());
        }
        let rows = tokens.len();
        self.reserve(rows)?;
        let start = Instant::now();
        copy_prefix(&self.tokens, tokens, &self.stream)?;
        let shape = FbtShape::new(rows as u32, WIDTH as u32, VOCAB as u32, 1.0e-5)
            .map_err(str::to_string)?;
        let attn = self.step_attention(rows, shape)?;
        let tail = self.step_routing(rows, shape)?;
        let output = read_prefix(&self.residual, &self.stream, rows * WIDTH)?
            .into_iter()
            .map(|v| half::f16::from_f32(v).to_bits())
            .collect();
        let experts = read_prefix(&self.route_experts, &self.stream, rows * 3)?;
        self.context.check_err().map_err(cuda_error)?;
        let profile = build_profile(&attn, &tail, start)?;
        Ok((output, experts, profile))
    }

    fn step_attention(&mut self, rows: usize, shape: FbtShape) -> CudaResult<[CudaEvent; 7]> {
        let embed_launch = self
            .module
            .prepare_embed_packed(LaunchConfig1D::new(rows as u32, 256, 0))
            .map_err(cuda_error)?;
        let rms_launch = self
            .module
            .prepare_rms(LaunchConfig1D::new(rows as u32, 256, 0))
            .map_err(cuda_error)?;
        let matmul_shape = MatmulShape {
            rows: rows as u32,
            columns: QKV as u32,
            inner: WIDTH as u32,
            input_stride: WIDTH as u32,
            weight_stride: QKV as u32,
            output_stride: QKV as u32,
            input_offset: 0,
            weight_offset: 0,
            output_offset: 0,
        };
        let mm_launch = self
            .module
            .prepare_matmul_turing(LaunchConfig2D::new(
                (QKV.div_ceil(16) as u32, rows.div_ceil(64) as u32),
                (256, 1),
                0,
            ))
            .map_err(cuda_error)?;
        let pisa_shape = PisaShape::fbt(rows as u32, CONTEXT as u32).map_err(str::to_string)?;
        let leaf_launch = self
            .module
            .prepare_pisaleaves(LaunchConfig2D::new((64, 1), (64, 1), 0))
            .map_err(cuda_error)?;
        let upper_launch = self
            .module
            .prepare_pisaupper(LaunchConfig1D::new(1, 64, 0))
            .map_err(cuda_error)?;
        let select_launch = self
            .module
            .prepare_pisa_select(LaunchConfig1D::new(rows as u32, 64, 0))
            .map_err(cuda_error)?;
        let attention_launch = self
            .module
            .prepare_pisa_attention(LaunchConfig1D::new((rows * HEADS) as u32, 256, 0))
            .map_err(cuda_error)?;
        let device_start = timing_event(&self.stream)?;
        // Prepared launches enforce each kernel's declared contract. Buffers
        // cover the validated row count and share one ordered stream.
        self.module
            .embed_packed(
                &self.stream,
                &embed_launch,
                &self.embedding,
                &self.tokens,
                &mut self.embedded,
                shape,
            )
            .map_err(cuda_error)?;
        let embed_end = timing_event(&self.stream)?;
        self.module
            .rms(
                &self.stream,
                &rms_launch,
                &self.embedded,
                &self.rms_weight,
                &mut self.normalized,
                shape,
            )
            .map_err(cuda_error)?;
        let rms_end = timing_event(&self.stream)?;
        self.module
            .matmul_turing(
                &self.stream,
                &mm_launch,
                &self.normalized,
                &self.qkv_weight,
                cuda_host::RowWidth::new(&mut self.qkv, QKV as u32),
                matmul_shape,
            )
            .map_err(cuda_error)?;
        let projection_end = timing_event(&self.stream)?;
        self.module
            .pisaleaves(
                &self.stream,
                &leaf_launch,
                &self.qkv,
                cuda_host::RowWidth::new(&mut self.pyramid, 64),
                pisa_shape,
            )
            .map_err(cuda_error)?;
        self.module
            .pisaupper(&self.stream, &upper_launch, &mut self.pyramid, pisa_shape)
            .map_err(cuda_error)?;
        let tree_end = timing_event(&self.stream)?;
        self.module
            .pisa_select(
                &self.stream,
                &select_launch,
                &self.qkv,
                &self.pyramid,
                &mut self.blocks,
                pisa_shape,
            )
            .map_err(cuda_error)?;
        let selection_end = timing_event(&self.stream)?;
        self.module
            .pisa_attention(
                &self.stream,
                &attention_launch,
                &self.qkv,
                &self.qkv,
                &self.blocks,
                &mut self.attended,
                pisa_shape,
            )
            .map_err(cuda_error)?;
        let attention_end = timing_event(&self.stream)?;
        Ok([
            device_start,
            embed_end,
            rms_end,
            projection_end,
            tree_end,
            selection_end,
            attention_end,
        ])
    }

    fn step_routing(&mut self, rows: usize, shape: FbtShape) -> CudaResult<[CudaEvent; 2]> {
        let rms_launch = self
            .module
            .prepare_rms(LaunchConfig1D::new(rows as u32, 256, 0))
            .map_err(cuda_error)?;
        let output_shape = MatmulShape {
            rows: rows as u32,
            columns: WIDTH as u32,
            inner: WIDTH as u32,
            input_stride: WIDTH as u32,
            weight_stride: WIDTH as u32,
            output_stride: WIDTH as u32,
            input_offset: 0,
            weight_offset: 0,
            output_offset: 0,
        };
        let output_launch = self
            .module
            .prepare_matmul_turing(LaunchConfig2D::new((32, 64), (256, 1), 0))
            .map_err(cuda_error)?;
        let residual_launch = self
            .module
            .prepare_residual(LaunchConfig1D::new(
                (rows * WIDTH).div_ceil(256) as u32,
                256,
                0,
            ))
            .map_err(cuda_error)?;
        let router_shape = MatmulShape {
            rows: rows as u32,
            columns: 128,
            inner: WIDTH as u32,
            input_stride: WIDTH as u32,
            weight_stride: 128,
            output_stride: 128,
            input_offset: 0,
            weight_offset: 0,
            output_offset: 0,
        };
        let router_launch = self
            .module
            .prepare_matmul_turing(LaunchConfig2D::new((8, 64), (256, 1), 0))
            .map_err(cuda_error)?;
        let route_shape = RouteShape::fbt(rows as u32).map_err(str::to_string)?;
        let topk_launch = self
            .module
            .prepare_routetopk(LaunchConfig1D::new(rows as u32, 256, 0))
            .map_err(cuda_error)?;
        self.module
            .matmul_turing(
                &self.stream,
                &output_launch,
                &self.attended,
                &self.output_weight,
                cuda_host::RowWidth::new(&mut self.projected, WIDTH as u32),
                output_shape,
            )
            .map_err(cuda_error)?;
        self.module
            .residual(
                &self.stream,
                &residual_launch,
                &self.embedded,
                &self.projected,
                &mut self.residual,
                (rows * WIDTH) as u32,
            )
            .map_err(cuda_error)?;
        let residual_end = timing_event(&self.stream)?;
        self.module
            .rms(
                &self.stream,
                &rms_launch,
                &self.residual,
                &self.rms_weight,
                &mut self.route_norm,
                shape,
            )
            .map_err(cuda_error)?;
        self.module
            .matmul_turing(
                &self.stream,
                &router_launch,
                &self.route_norm,
                &self.router_weight,
                cuda_host::RowWidth::new(&mut self.route_scores, 128),
                router_shape,
            )
            .map_err(cuda_error)?;
        self.module
            .routetopk(
                &self.stream,
                &topk_launch,
                &self.route_scores,
                &mut self.route_experts,
                &mut self.route_weights,
                &mut self.route_margins,
                route_shape,
            )
            .map_err(cuda_error)?;
        let device_end = timing_event(&self.stream)?;
        Ok([residual_end, device_end])
    }
}

fn build_profile(
    attn: &[CudaEvent; 7],
    tail: &[CudaEvent; 2],
    start: Instant,
) -> CudaResult<FbtPrefillProfile> {
    Ok(FbtPrefillProfile {
        embed_ms: attn[0].elapsed_ms(&attn[1]).map_err(cuda_error)?,
        rms_ms: attn[1].elapsed_ms(&attn[2]).map_err(cuda_error)?,
        projection_ms: attn[2].elapsed_ms(&attn[3]).map_err(cuda_error)?,
        tree_ms: attn[3].elapsed_ms(&attn[4]).map_err(cuda_error)?,
        selection_ms: attn[4].elapsed_ms(&attn[5]).map_err(cuda_error)?,
        attention_ms: attn[5].elapsed_ms(&attn[6]).map_err(cuda_error)?,
        residual_ms: attn[6].elapsed_ms(&tail[0]).map_err(cuda_error)?,
        routing_ms: tail[0].elapsed_ms(&tail[1]).map_err(cuda_error)?,
        device_ms: attn[0].elapsed_ms(&tail[1]).map_err(cuda_error)?,
        end_to_end_ms: start.elapsed().as_secs_f32() * 1000.0,
    })
}

/// Verify column-major checkpoint packing and token-major device lookup with
/// values that expose token/column transposition errors.
pub fn embed_parity() -> CudaResult<()> {
    const ROWS: usize = 3;
    const WIDTH: usize = 17;
    const VOCAB: usize = 19;
    let context = CudaContext::new(0).map_err(cuda_error)?;
    let stream = context.default_stream();
    let module = crate::capture::load_model(&context)?;
    let checkpoint = (0..WIDTH * VOCAB)
        .map(|index| index as f32)
        .collect::<Vec<_>>();
    let packed = pack_embedding(&checkpoint, WIDTH, VOCAB);
    let tokens = [3_u32, 7, 18];
    let weights = DeviceBuffer::from_host(&stream, &packed).map_err(cuda_error)?;
    let tokens_device = DeviceBuffer::from_host(&stream, &tokens).map_err(cuda_error)?;
    let mut output = DeviceBuffer::zeroed(&stream, ROWS * WIDTH).map_err(cuda_error)?;
    let shape =
        FbtShape::new(ROWS as u32, WIDTH as u32, VOCAB as u32, 1.0e-5).map_err(str::to_string)?;
    let launch = module
        .prepare_embed_packed(LaunchConfig1D::new(ROWS as u32, 256, 0))
        .map_err(cuda_error)?;
    module
        .embed_packed(
            &stream,
            &launch,
            &weights,
            &tokens_device,
            &mut output,
            shape,
        )
        .map_err(cuda_error)?;
    context.check_err().map_err(cuda_error)?;
    let actual = read_prefix(&output, &stream, ROWS * WIDTH)?;
    let mut expected = Vec::with_capacity(ROWS * WIDTH);
    for token in tokens {
        for column in 0..WIDTH {
            expected.push(checkpoint[column * VOCAB + token as usize]);
        }
    }
    if actual != expected {
        return Err("CUDA packed embedding parity mismatch".into());
    }
    Ok(())
}

/// Exercise the CUDA feedback primitives on-device with exact FP16 fixtures.
pub fn feedback_parity() -> CudaResult<()> {
    let context = CudaContext::new(0).map_err(cuda_error)?;
    let stream = context.default_stream();
    let module = crate::capture::load_model(&context)?;
    let mut shape = FeedbackShape::fbt(4, 2).map_err(str::to_string)?;
    shape.token_unit_rms = 0;
    shape.fused_unit_rms = 0;
    let elements = (shape.rows * shape.width) as usize;
    let history = vec![0x3c00_u16; elements];
    let tokens = vec![0x4000_u16; elements];
    let value = vec![0x4000_u16; elements];
    let logit = vec![0_u16; elements];
    let history = DeviceBuffer::from_host(&stream, &history).map_err(cuda_error)?;
    let tokens = DeviceBuffer::from_host(&stream, &tokens).map_err(cuda_error)?;
    let value = DeviceBuffer::from_host(&stream, &value).map_err(cuda_error)?;
    let logit = DeviceBuffer::from_host(&stream, &logit).map_err(cuda_error)?;
    let mut previous = DeviceBuffer::zeroed(&stream, elements).map_err(cuda_error)?;
    let mut normalized = DeviceBuffer::zeroed(&stream, elements).map_err(cuda_error)?;
    let mut fused = DeviceBuffer::zeroed(&stream, shape.rows as usize).map_err(cuda_error)?;
    let mut output = DeviceBuffer::zeroed(&stream, elements).map_err(cuda_error)?;
    let shift_launch = module
        .prepare_feedback_shift(LaunchConfig1D::new(shape.rows, 256, 0))
        .map_err(cuda_error)?;
    let norm_launch = module
        .prepare_feedback_norm(LaunchConfig1D::new(shape.rows, 256, 0))
        .map_err(cuda_error)?;
    let combine_launch = module
        .prepare_feedback_combine(LaunchConfig1D::new(shape.rows, 256, 0))
        .map_err(cuda_error)?;
    // Generated launch contracts validate the shapes and disjoint outputs.
    module
        .feedback_shift(
            &stream,
            &shift_launch,
            &history,
            &mut previous,
            &mut fused,
            shape,
        )
        .map_err(cuda_error)?;
    shape.token_unit_rms = 1;
    module
        .feedback_norm(&stream, &norm_launch, &history, &mut normalized, shape)
        .map_err(cuda_error)?;
    shape.token_unit_rms = 0;
    module
        .feedback_combine(
            &stream,
            &combine_launch,
            &value,
            &logit,
            &tokens,
            &fused,
            &mut output,
            shape,
        )
        .map_err(cuda_error)?;
    context.check_err().map_err(cuda_error)?;
    let previous = read_prefix(&previous, &stream, elements)?;
    let normalized = read_prefix(&normalized, &stream, elements)?;
    let fused = read_prefix(&fused, &stream, shape.rows as usize)?;
    let output = read_prefix(&output, &stream, elements)?;
    if fused != [0, 1, 0, 1]
        || previous[..shape.width as usize].iter().any(|&v| v != 0)
        || previous[shape.width as usize..2 * shape.width as usize]
            .iter()
            .any(|&v| v != 0x3c00)
        || normalized.iter().any(|&v| v != 0x3c00)
        || output[..shape.width as usize].iter().any(|&v| v != 0x4000)
        || output[shape.width as usize..2 * shape.width as usize]
            .iter()
            .any(|&v| v != 0x3c00)
    {
        return Err("CUDA feedback parity mismatch".into());
    }
    Ok(())
}
