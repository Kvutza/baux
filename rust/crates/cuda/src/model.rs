//! Resident checkpoint execution. Physical weights are shared across recurrent
//! visits.
use crate::capture::{CudaResult, copy_prefix, cuda_error, read_prefix, timing_event};
use baux_cuda_device::{FbtShape, MatmulShape, PisaShape, RoutedTile, fbt_model};
use cuda_core::{
    CudaContext, CudaEvent, CudaStream, DeviceBuffer, LaunchConfig1D, simt::LaunchConfig2D,
};
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;
mod bo_report;
mod cached;
mod gen_bench;
mod loader;
mod mhc_step;
mod readout;
pub use baux::semantics::recurrence;
mod bo_run;
mod bo_task;
mod proposal;
mod routing;
pub(crate) use crate::gemm::{matmul, matmul_at};
pub(crate) mod run;
pub(crate) mod weights;
use weights::Weights;
mod diffusion;
pub(crate) mod diffusion_run;
use diffusion::DiffusionState;

const WIDTH: usize = baux::semantics::mhc4::WIDTH;
const VOCAB: usize = baux_cuda_device::VOCAB as usize;
const EXPERTS: usize = baux::semantics::shape::MHC4_SHAPE.experts as usize;

#[derive(Clone, Copy)]
enum Stage {
    Embed,
    Prepare,
    Qkv,
    Index,
    Pisa,
    Output,
    Update,
    MoeRouter,
    MoeRoute,
    MoeExperts,
    MoeCombine,
    Readout,
}

fn mark(
    events: &mut Option<&mut Vec<(Stage, CudaEvent)>>,
    stream: &CudaStream,
    stage: Stage,
) -> CudaResult<()> {
    if let Some(events) = events.as_deref_mut() {
        events.push((stage, timing_event(stream)?));
    }
    Ok(())
}

pub struct ModelOutput {
    pub tokens: Vec<u32>,
    pub hidden: Vec<f32>,
    pub stream_probe: Vec<f32>,
    pub device_ms: f32,
    pub wall_ms: f32,
    pub visits: usize,
}

pub(super) struct EvaluationOutput {
    pub device_ms: f32,
    pub macro_score_ms: f32,
    pub wall_ms: f32,
    pub visits: usize,
}

pub struct GenerationOutput {
    pub tokens: Vec<u32>,
    pub frontiers: Vec<usize>,
    pub prompt_tokens: usize,
    pub passes: usize,
    pub visits: usize,
    pub device_ms: f32,
    pub wall_ms: f32,
    pub first_wave_ms: Option<f32>,
    pub chunk_profiles: Vec<(usize, WaveProfile)>,
}

pub struct WaveProfile {
    pub embed_ms: f32,
    pub mhc_ms: f32,
    pub qkv_ms: f32,
    pub index_ms: f32,
    pub pisa_ms: f32,
    pub output_ms: f32,
    pub attention_ms: f32,
    pub moe_router_ms: f32,
    pub moe_route_ms: f32,
    pub moe_experts_ms: f32,
    pub moe_combine_ms: f32,
    pub moe_ms: f32,
    pub readout_ms: f32,
    pub total_ms: f32,
}

pub struct FbtModel {
    pub(crate) context: Arc<CudaContext>,
    pub(crate) stream: Arc<CudaStream>,
    pub(crate) module: fbt_model::LoadedModule,
    pub(crate) weights: Weights,
    pub(crate) scratch: Scratch,
    pub(crate) rows: usize,
    pub(crate) sequence: usize,
    pub(crate) diffusion: Option<DiffusionState>,
    mhc4: Option<readout::Mhc4State>,
    synth: Option<crate::gemm::Gemm>,
    parallel_mhc: bool,
    cache: Option<cached::State>,
    first: usize,
    readout_from: usize,
    scoring: Option<Scoring>,
}

struct Scoring {
    targets: DeviceBuffer<u32>,
    losses: DeviceBuffer<f32>,
    count: usize,
}

struct Scratch {
    tokens: DeviceBuffer<u32>,
    embedded: DeviceBuffer<f32>,
    streams: DeviceBuffer<f32>,
    coefficients: DeviceBuffer<f32>,
    normalized: DeviceBuffer<f32>,
    qkv: DeviceBuffer<f32>,
    pyramid: DeviceBuffer<f32>,
    blocks: DeviceBuffer<u32>,
    attended: DeviceBuffer<f32>,
    branch: DeviceBuffer<f32>,
    route_scores: DeviceBuffer<f32>,
    experts: DeviceBuffer<u32>,
    route_weights: DeviceBuffer<f32>,
    margins: DeviceBuffer<f32>,
    counts: DeviceBuffer<u32>,
    offsets: DeviceBuffer<u32>,
    prefixes: DeviceBuffer<u32>,
    mapping: DeviceBuffer<u32>,
    packed: DeviceBuffer<f32>,
    tiles: DeviceBuffer<RoutedTile>,
    shared_tiles: DeviceBuffer<RoutedTile>,
    projection: DeviceBuffer<f32>,
    shared_projection: DeviceBuffer<f32>,
    activation: DeviceBuffer<f32>,
    routed: DeviceBuffer<f32>,
    shared_activation: DeviceBuffer<f32>,
    shared: DeviceBuffer<f32>,
    logits: DeviceBuffer<f32>,
    sampled: DeviceBuffer<u32>,
    frontier: DeviceBuffer<u32>,
    rope: DeviceBuffer<f32>,
    tile_capacity: usize,
}

impl Scratch {
    fn new(stream: &CudaStream, rows: usize, sequence: usize) -> CudaResult<Self> {
        fn buffer<T: cuda_core::DeviceCopy>(
            stream: &CudaStream,
            len: usize,
        ) -> CudaResult<DeviceBuffer<T>> {
            DeviceBuffer::zeroed(stream, len).map_err(cuda_error)
        }
        let tile_capacity = (rows * 3).div_ceil(64) + EXPERTS;
        let shared_tiles = (0..rows.div_ceil(64))
            .map(|tile| RoutedTile {
                expert: 0,
                first_row: (tile * 64) as u32,
                valid_rows: (rows - tile * 64).min(64) as u32,
                reserved: 1,
            })
            .collect::<Vec<_>>();
        let mut rope = Vec::with_capacity(sequence * 64);
        for position in 0..sequence {
            for pair in 0..32 {
                let angle = position as f32 * 10000.0_f32.powf(-2.0 * pair as f32 / 64.0);
                rope.extend([angle.cos(), angle.sin()]);
            }
        }
        Ok(Self {
            tokens: buffer(stream, rows)?,
            embedded: buffer(stream, rows * WIDTH)?,
            streams: buffer(stream, rows * WIDTH * 4)?,
            coefficients: buffer(stream, rows * 24)?,
            normalized: buffer(stream, rows * WIDTH)?,
            qkv: buffer(stream, rows * 640)?,
            pyramid: buffer(stream, (rows / sequence * (sequence / 32 - 1) * 64).max(1))?,
            blocks: buffer(stream, rows * 8)?,
            attended: buffer(stream, rows * WIDTH)?,
            branch: buffer(stream, rows * WIDTH)?,
            route_scores: buffer(stream, rows * EXPERTS)?,
            experts: buffer(stream, rows * 3)?,
            route_weights: buffer(stream, rows * 3)?,
            margins: buffer(stream, rows)?,
            counts: buffer(stream, EXPERTS)?,
            offsets: buffer(stream, EXPERTS)?,
            prefixes: buffer(stream, EXPERTS * (rows * 3).div_ceil(1024))?,
            mapping: buffer(stream, rows * 3)?,
            packed: buffer(stream, rows * 3 * WIDTH)?,
            tiles: buffer(stream, tile_capacity)?,
            shared_tiles: DeviceBuffer::from_host(stream, &shared_tiles).map_err(cuda_error)?,
            projection: buffer(stream, rows * 3 * 512)?,
            shared_projection: buffer(stream, rows * 512)?,
            activation: buffer(stream, rows * 3 * 256)?,
            routed: buffer(stream, rows * 3 * WIDTH)?,
            shared_activation: buffer(stream, rows * 256)?,
            shared: buffer(stream, rows * WIDTH)?,
            logits: buffer(stream, rows.min(512) * VOCAB)?,
            sampled: buffer(stream, rows)?,
            frontier: buffer(stream, 1)?,
            rope: DeviceBuffer::from_host(stream, &rope).map_err(cuda_error)?,
            tile_capacity,
        })
    }
}

impl FbtModel {
    pub fn fixture() -> CudaResult<Self> {
        let context = CudaContext::new(0).map_err(cuda_error)?;
        let stream = context.new_stream().map_err(cuda_error)?;
        let weights = Weights::fixture(&stream)?;
        Self::from_weights(context, stream, weights, 4096, 4096)
    }

    pub fn open(path: &Path, rows: usize, sequence: usize) -> CudaResult<Self> {
        if rows == 0
            || rows > 65536
            || sequence < 4096
            || sequence > 65536
            || !sequence.is_power_of_two()
            || rows % sequence != 0
        {
            return Err(
                "model requires 4096..65536 power-of-two context and complete sequences, at most \
                 65536 rows"
                    .into(),
            );
        }
        PisaShape::fbt(rows as u32, sequence as u32).map_err(str::to_string)?;
        Self::load(path, rows, sequence)
    }

    /// Allocate bounded activations directly, without a full-context workspace.
    pub fn open_streamed(path: &Path, sequence: usize, chunk: usize) -> CudaResult<Self> {
        if !(4096..=1_048_576).contains(&sequence)
            || !sequence.is_power_of_two()
            || !(64..=4096).contains(&chunk)
            || !chunk.is_power_of_two()
            || sequence % chunk != 0
        {
            return Err(
                "streamed model requires power-of-two context 4096..1048576 and chunk 64..4096"
                    .into(),
            );
        }
        PisaShape::cached(chunk as u32, sequence as u32, 0).map_err(str::to_string)?;
        Self::load(path, chunk, sequence)
    }

    pub fn initialize_mhc4(seed: u64, rows: usize, sequence: usize) -> CudaResult<Self> {
        if !(64..=65_536).contains(&rows)
            || !(4_096..=65_536).contains(&sequence)
            || !rows.is_power_of_two()
            || !sequence.is_power_of_two()
            || rows > sequence
            || sequence % rows != 0
        {
            return Err(
                "procedural mHC-4 requires power-of-two active rows 64..65536 dividing a \
                 4096..65536 context"
                    .into(),
            );
        }
        if rows == sequence {
            PisaShape::fbt(rows as u32, sequence as u32).map_err(str::to_string)?;
        } else {
            PisaShape::cached(rows as u32, sequence as u32, 0).map_err(str::to_string)?;
        }
        let context = CudaContext::new(0).map_err(cuda_error)?;
        let stream = context.new_stream().map_err(cuda_error)?;
        let weights = Weights::initialize_mhc4(seed, &stream)?;
        Self::from_weights(context, stream, weights, rows, sequence)
    }

    fn load(path: &Path, rows: usize, sequence: usize) -> CudaResult<Self> {
        let context = CudaContext::new(0).map_err(cuda_error)?;
        let stream = context.new_stream().map_err(cuda_error)?;
        let weights = Weights::open(path, &stream)?;
        Self::from_weights(context, stream, weights, rows, sequence)
    }

    fn from_weights(
        context: Arc<CudaContext>,
        stream: Arc<CudaStream>,
        weights: Weights,
        rows: usize,
        sequence: usize,
    ) -> CudaResult<Self> {
        let module = crate::capture::load_model(&context)?;
        let synth = crate::gemm::Gemm::from_env(&context)?;
        let scratch = Scratch::new(&stream, rows, sequence)?;
        Ok(Self {
            context,
            stream,
            module,
            weights,
            scratch,
            rows,
            sequence,
            diffusion: None,
            mhc4: None,
            synth,
            parallel_mhc: true,
            cache: None,
            first: 0,
            readout_from: 0,
            scoring: None,
        })
    }

    pub fn run(
        &mut self,
        tokens: &[u32],
        visits: usize,
        temperature: f32,
        seed: u64,
    ) -> CudaResult<ModelOutput> {
        let (wall, start, end, visits) = self.launch(tokens, visits, temperature, seed, None)?;
        let tokens = read_prefix(&self.scratch.sampled, &self.stream, self.rows)?;
        if tokens.iter().any(|&token| token >= VOCAB as u32) {
            return Err("readout contains no finite logit in at least one row".into());
        }
        let hidden = read_prefix(&self.scratch.normalized, &self.stream, self.rows * WIDTH)?;
        if hidden.iter().any(|&value| !value.is_finite()) {
            return Err("model produced non-finite hidden state".into());
        }
        let stream_probe = read_prefix(&self.scratch.streams, &self.stream, 2048)?;
        self.context.check_err().map_err(cuda_error)?;
        Ok(ModelOutput {
            tokens,
            hidden,
            stream_probe,
            device_ms: start.elapsed_ms(&end).map_err(cuda_error)?,
            wall_ms: wall.elapsed().as_secs_f32() * 1000.0,
            visits,
        })
    }

    pub(super) fn evaluate(
        &mut self,
        tokens: &[u32],
        visits: usize,
        temperature: f32,
        seed: u64,
    ) -> CudaResult<EvaluationOutput> {
        let mut score_events = Vec::new();
        let (wall, start, end, visits) =
            self.launch(tokens, visits, temperature, seed, Some(&mut score_events))?;
        self.context.check_err().map_err(cuda_error)?;
        let macro_score_ms = score_events
            .iter()
            .map(|(start, end)| start.elapsed_ms(end).map_err(cuda_error))
            .sum::<CudaResult<f32>>()?;
        Ok(EvaluationOutput {
            device_ms: start.elapsed_ms(&end).map_err(cuda_error)?,
            macro_score_ms,
            wall_ms: wall.elapsed().as_secs_f32() * 1000.0,
            visits,
        })
    }

    fn launch(
        &mut self,
        tokens: &[u32],
        visits: usize,
        temperature: f32,
        seed: u64,
        score_events: Option<&mut Vec<(CudaEvent, CudaEvent)>>,
    ) -> CudaResult<(Instant, CudaEvent, CudaEvent, usize)> {
        if self.weights.diffusion.is_some()
            || self.cache.is_some()
            || tokens.len() != self.rows
            || tokens.iter().any(|&token| token >= VOCAB as u32)
            || !temperature.is_finite()
            || temperature < 0.0
        {
            return Err("model token count, vocabulary, or temperature is invalid".into());
        }
        let plan = recurrence::RecurrentCore::selective_fbt().layer_visits(5, visits)?;
        let wall = Instant::now();
        copy_prefix(&self.scratch.tokens, tokens, &self.stream)?;
        let start = timing_event(&self.stream)?;
        let visits = self.enqueue(&plan, temperature, seed, score_events)?;
        let end = timing_event(&self.stream)?;
        Ok((wall, start, end, visits))
    }

    fn enqueue(
        &mut self,
        plan: &[recurrence::LayerVisit],
        temperature: f32,
        seed: u64,
        score_events: Option<&mut Vec<(CudaEvent, CudaEvent)>>,
    ) -> CudaResult<usize> {
        self.enqueue_events(plan, temperature, seed, None, score_events)
    }

    fn enqueue_events(
        &mut self,
        plan: &[recurrence::LayerVisit],
        temperature: f32,
        seed: u64,
        mut events: Option<&mut Vec<(Stage, CudaEvent)>>,
        score_events: Option<&mut Vec<(CudaEvent, CudaEvent)>>,
    ) -> CudaResult<usize> {
        let rows = self.rows as u32;
        let shape = FbtShape::new(rows, 512, VOCAB as u32, 1.0e-5).map_err(str::to_string)?;
        if let Some(state) = &mut self.diffusion {
            let weights = self
                .weights
                .diffusion
                .as_ref()
                .ok_or("diffusion checkpoint missing mask/index tensors")?;
            let launch = state
                .module
                .prepare_diffusion_embed(LaunchConfig1D::new(rows, 256, 0))
                .map_err(cuda_error)?;
            state
                .module
                .diffusion_embed(
                    &self.stream,
                    &launch,
                    &self.weights.embedding,
                    &self.scratch.tokens,
                    &weights.mask,
                    &state.confidence,
                    &mut self.scratch.embedded,
                    rows,
                )
                .map_err(cuda_error)?;
        } else {
            let launch = self
                .module
                .prepare_embed_packed(LaunchConfig1D::new(rows, 256, 0))
                .map_err(cuda_error)?;
            self.module
                .embed_packed(
                    &self.stream,
                    &launch,
                    &self.weights.embedding,
                    &self.scratch.tokens,
                    &mut self.scratch.embedded,
                    shape,
                )
                .map_err(cuda_error)?;
        }
        let launch = self
            .module
            .prepare_mhc_replicate(LaunchConfig1D::new((rows * 512).div_ceil(256), 256, 0))
            .map_err(cuda_error)?;
        self.module
            .mhc_replicate(
                &self.stream,
                &launch,
                &self.scratch.embedded,
                &mut self.scratch.streams,
                rows,
            )
            .map_err(cuda_error)?;
        mark(&mut events, &self.stream, Stage::Embed)?;
        self.enqueue_core(plan, temperature, seed, events, score_events, true)
    }

    pub(super) fn enqueue_core(
        &mut self,
        plan: &[recurrence::LayerVisit],
        temperature: f32,
        seed: u64,
        mut events: Option<&mut Vec<(Stage, CudaEvent)>>,
        score_events: Option<&mut Vec<(CudaEvent, CudaEvent)>>,
        readout: bool,
    ) -> CudaResult<usize> {
        let rows = self.rows as u32;
        for (ordinal, visit) in plan.iter().enumerate() {
            self.prepare(visit.layer, true)?;
            mark(&mut events, &self.stream, Stage::Prepare)?;
            if self.cache.is_some() {
                self.cached_attention(visit.layer, ordinal, &mut events)?;
            } else {
                self.attention(visit.layer, ordinal, &mut events)?;
            }
            self.update()?;
            mark(&mut events, &self.stream, Stage::Update)?;
            self.prepare(visit.layer, false)?;
            mark(&mut events, &self.stream, Stage::Prepare)?;
            self.moe(visit.layer, &mut events)?;
            mark(&mut events, &self.stream, Stage::Update)?;
        }
        let launch = self
            .module
            .prepare_mhc_mix(LaunchConfig1D::new(rows, 256, 0))
            .map_err(cuda_error)?;
        self.module
            .mhc_mix(
                &self.stream,
                &launch,
                &self.scratch.streams,
                &self.scratch.coefficients,
                &self.weights.final_norm,
                &mut self.scratch.normalized,
                rows,
                1,
            )
            .map_err(cuda_error)?;
        if readout {
            self.readout(temperature, seed, score_events)?;
            mark(&mut events, &self.stream, Stage::Readout)?;
        }
        Ok(plan.len())
    }

    pub fn profile_wave(
        &mut self,
        tokens: &[u32],
        visits: usize,
        temperature: f32,
        seed: u64,
    ) -> CudaResult<WaveProfile> {
        if self.cache.is_some()
            || tokens.len() != self.rows
            || tokens.iter().any(|&token| token >= VOCAB as u32)
            || !temperature.is_finite()
            || temperature < 0.0
        {
            return Err("wave profile token count, vocabulary, or temperature is invalid".into());
        }
        let plan = recurrence::RecurrentCore::selective_fbt().layer_visits(5, visits)?;
        copy_prefix(&self.scratch.tokens, tokens, &self.stream)?;
        let start = timing_event(&self.stream)?;
        let mut events = Vec::with_capacity(plan.len() * 12 + 2);
        self.enqueue_events(&plan, temperature, seed, Some(&mut events), None)?;
        if events.len() != plan.len() * 12 + 2 {
            return Err("wave profile event count is invalid".into());
        }
        let profile = WaveProfile::from_events(&start, &events)?;
        self.context.check_err().map_err(cuda_error)?;
        Ok(profile)
    }
}

impl WaveProfile {
    fn from_events(start: &CudaEvent, events: &[(Stage, CudaEvent)]) -> CudaResult<Self> {
        let mut embed_ms = 0.0;
        let mut mhc_ms = 0.0;
        let mut qkv_ms = 0.0;
        let mut index_ms = 0.0;
        let mut pisa_ms = 0.0;
        let mut output_ms = 0.0;
        let mut moe_router_ms = 0.0;
        let mut moe_route_ms = 0.0;
        let mut moe_experts_ms = 0.0;
        let mut moe_combine_ms = 0.0;
        let mut readout_ms = 0.0;
        let mut previous = start;
        for (stage, event) in events {
            let elapsed = previous.elapsed_ms(event).map_err(cuda_error)?;
            match stage {
                Stage::Embed => embed_ms += elapsed,
                Stage::Prepare | Stage::Update => mhc_ms += elapsed,
                Stage::Qkv => qkv_ms += elapsed,
                Stage::Index => index_ms += elapsed,
                Stage::Pisa => pisa_ms += elapsed,
                Stage::Output => output_ms += elapsed,
                Stage::MoeRouter => moe_router_ms += elapsed,
                Stage::MoeRoute => moe_route_ms += elapsed,
                Stage::MoeExperts => moe_experts_ms += elapsed,
                Stage::MoeCombine => moe_combine_ms += elapsed,
                Stage::Readout => readout_ms += elapsed,
            }
            previous = event;
        }
        let attention_ms = qkv_ms + index_ms + pisa_ms + output_ms;
        let moe_ms = moe_router_ms + moe_route_ms + moe_experts_ms + moe_combine_ms;
        let end = &events.last().ok_or("wave profile is empty")?.1;
        Ok(WaveProfile {
            embed_ms,
            mhc_ms,
            qkv_ms,
            index_ms,
            pisa_ms,
            output_ms,
            attention_ms,
            moe_router_ms,
            moe_route_ms,
            moe_experts_ms,
            moe_combine_ms,
            moe_ms,
            readout_ms,
            total_ms: start.elapsed_ms(end).map_err(cuda_error)?,
        })
    }
}

impl FbtModel {
    /// Verify and repair an accepted prefix on-device. Sampling is indexed by
    /// absolute position, so repairs reuse the same random draw at each row.
    pub fn generate(
        &mut self,
        prompt: &[u32],
        visits: usize,
        temperature: f32,
        seed: u64,
        unroll: usize,
    ) -> CudaResult<GenerationOutput> {
        if self.weights.diffusion.is_some()
            || self.cache.is_some()
            || self.rows != self.sequence
            || prompt.is_empty()
            || prompt.len() >= self.rows
            || prompt.iter().any(|&token| token >= VOCAB as u32)
            || !temperature.is_finite()
            || temperature < 0.0
            || !(1..=32).contains(&unroll)
        {
            return Err(
                "generation requires one complete context, a nonempty in-vocabulary prompt \
                 shorter than context, nonnegative finite temperature, and 1..32 unroll"
                    .into(),
            );
        }
        let plan = recurrence::RecurrentCore::selective_fbt().layer_visits(5, visits)?;
        let wall = Instant::now();
        let mut tokens = vec![0; self.rows];
        tokens[..prompt.len()].copy_from_slice(prompt);
        copy_prefix(&self.scratch.tokens, &tokens, &self.stream)?;
        copy_prefix(&self.scratch.frontier, &[prompt.len() as u32], &self.stream)?;
        let start = timing_event(&self.stream)?;
        let launch = self
            .module
            .prepare_verify_prefix(LaunchConfig1D::new(1, 256, 0))
            .map_err(cuda_error)?;
        let mut passes = 0;
        let mut frontier = prompt.len();
        let mut frontiers = vec![frontier];
        while frontier < self.rows {
            for _ in 0..unroll {
                self.enqueue(&plan, temperature, seed, None)?;
                self.module
                    .verify_prefix(
                        &self.stream,
                        &launch,
                        &self.scratch.sampled,
                        &mut self.scratch.tokens,
                        &mut self.scratch.frontier,
                        self.rows as u32,
                    )
                    .map_err(cuda_error)?;
                passes += 1;
            }
            let next = read_prefix(&self.scratch.frontier, &self.stream, 1)?[0] as usize;
            if next <= frontier || next > self.rows {
                return Err(
                    "accepted-prefix repair failed to advance or produced invalid logits".into(),
                );
            }
            frontier = next;
            frontiers.push(frontier);
        }
        let end = timing_event(&self.stream)?;
        let tokens = read_prefix(&self.scratch.tokens, &self.stream, self.rows)?;
        if tokens.iter().any(|&token| token >= VOCAB as u32) || tokens[..prompt.len()] != *prompt {
            return Err("generation produced invalid tokens or changed the prompt".into());
        }
        self.context.check_err().map_err(cuda_error)?;
        Ok(GenerationOutput {
            tokens,
            frontiers,
            prompt_tokens: prompt.len(),
            passes,
            visits: plan.len(),
            device_ms: start.elapsed_ms(&end).map_err(cuda_error)?,
            wall_ms: wall.elapsed().as_secs_f32() * 1000.0,
            first_wave_ms: None,
            chunk_profiles: Vec::new(),
        })
    }
}
