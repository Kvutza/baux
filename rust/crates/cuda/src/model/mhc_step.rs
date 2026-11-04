use super::readout::Mhc4State;
use super::*;
use baux_cuda_device::{ObjectiveShape, ScoreStats};

#[derive(Clone, Copy)]
pub(super) enum Mhc4Stage {
    Init,
    Gemm,
    Project,
    Reduce,
}

pub(super) struct Mhc4Profile {
    pub init_ms: f32,
    pub gemm_ms: f32,
    pub project_ms: f32,
    pub reduce_ms: f32,
    pub total_ms: f32,
}

impl FbtModel {
    pub(super) fn prepare(&mut self, layer: usize, attention: bool) -> CudaResult<()> {
        let weights = &self.weights.layers[layer];
        let site = if attention {
            &weights.attention
        } else {
            &weights.moe
        };
        let rows = self.rows as u32;
        let launch = self
            .module
            .prepare_mhc_predict(LaunchConfig1D::new(rows, 256, 0))
            .map_err(cuda_error)?;
        self.module
            .mhc_predict(
                &self.stream,
                &launch,
                &self.scratch.streams,
                &site.predictor,
                &site.bias,
                &site.control,
                &mut self.scratch.coefficients,
                rows,
                u32::from(self.parallel_mhc),
            )
            .map_err(cuda_error)?;
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
                &site.norm,
                &mut self.scratch.normalized,
                rows,
                0,
            )
            .map_err(cuda_error)
    }

    pub(super) fn update(&mut self) -> CudaResult<()> {
        let rows = self.rows as u32;
        let launch = self
            .module
            .prepare_mhc_update(LaunchConfig1D::new((rows * 512).div_ceil(256), 256, 0))
            .map_err(cuda_error)?;
        self.module
            .mhc_update(
                &self.stream,
                &launch,
                &mut self.scratch.streams,
                &self.scratch.branch,
                &self.scratch.coefficients,
                rows,
            )
            .map_err(cuda_error)
    }

    pub(super) fn attention(
        &mut self,
        layer: usize,
        ordinal: usize,
        events: &mut Option<&mut Vec<(Stage, CudaEvent)>>,
    ) -> CudaResult<()> {
        let rows = self.rows as u32;
        let mut shape = PisaShape::fbt(rows, self.sequence as u32).map_err(str::to_string)?;
        if let Some(state) = &self.diffusion {
            shape.visibility = state.block;
        }
        let weights = &self.weights.layers[layer];
        let s = &mut self.scratch;
        matmul(
            &self.module,
            &self.stream,
            &s.normalized,
            &weights.qkv,
            &mut s.qkv,
            self.rows,
            640,
            WIDTH,
        )?;
        mark(events, &self.stream, Stage::Qkv)?;
        let launch = self
            .module
            .prepare_rope(LaunchConfig1D::new((rows * 9 * 32).div_ceil(256), 256, 0))
            .map_err(cuda_error)?;
        self.module
            .rope(
                &self.stream,
                &launch,
                &mut s.qkv,
                &s.rope,
                rows,
                self.sequence as u32,
                0,
            )
            .map_err(cuda_error)?;
        let sequences = rows / self.sequence as u32;
        let launch = self
            .module
            .prepare_pisaleaves(LaunchConfig2D::new((shape.leaves(), sequences), (64, 1), 0))
            .map_err(cuda_error)?;
        self.module
            .pisaleaves(
                &self.stream,
                &launch,
                &s.qkv,
                cuda_host::RowWidth::new(&mut s.pyramid, 64),
                shape,
            )
            .map_err(cuda_error)?;
        let launch = self
            .module
            .prepare_pisaupper(LaunchConfig1D::new(sequences, 64, 0))
            .map_err(cuda_error)?;
        self.module
            .pisaupper(&self.stream, &launch, &mut s.pyramid, shape)
            .map_err(cuda_error)?;
        self.select_blocks(ordinal, rows, shape)?;
        mark(events, &self.stream, Stage::Index)?;
        let weights = &self.weights.layers[layer];
        let s = &mut self.scratch;
        let launch = self
            .module
            .prepare_pisa_multihead(LaunchConfig1D::new(rows * 2, 256, 0))
            .map_err(cuda_error)?;
        self.module
            .pisa_multihead(
                &self.stream,
                &launch,
                &s.qkv,
                &s.qkv,
                &s.blocks,
                &mut s.attended,
                shape,
            )
            .map_err(cuda_error)?;
        mark(events, &self.stream, Stage::Pisa)?;
        matmul(
            &self.module,
            &self.stream,
            &s.attended,
            &weights.output,
            &mut s.branch,
            self.rows,
            WIDTH,
            WIDTH,
        )?;
        mark(events, &self.stream, Stage::Output)
    }

    fn select_blocks(&mut self, ordinal: usize, rows: u32, shape: PisaShape) -> CudaResult<()> {
        let s = &mut self.scratch;
        if let Some(state) = &mut self.diffusion {
            let weights = self
                .weights
                .diffusion
                .as_ref()
                .ok_or("missing diffusion index weights")?;
            let launch = state
                .module
                .prepare_project_index(LaunchConfig1D::new(rows, 64, 0))
                .map_err(cuda_error)?;
            state
                .module
                .project_index(
                    &self.stream,
                    &launch,
                    &s.qkv,
                    &weights.index,
                    &mut state.index,
                    rows,
                )
                .map_err(cuda_error)?;
        }
        let index_query = self.diffusion.as_ref().map_or(&s.qkv, |state| &state.index);
        let shadow = self.mhc4.as_ref().is_some_and(|state| state.index_shadow);
        if shadow {
            let state = self.mhc4.as_ref().ok_or("missing mHC-4 PISA state")?;
            let visit_offset = u32::try_from(ordinal * self.rows * 8).map_err(cuda_error)?;
            let launch = self
                .module
                .prepare_pisa_shadow_refine(LaunchConfig1D::new(rows, 256, 0))
                .map_err(cuda_error)?;
            self.module
                .pisa_shadow_refine(
                    &self.stream,
                    &launch,
                    index_query,
                    &s.qkv,
                    &s.pyramid,
                    &state.index_incumbent,
                    &mut s.blocks,
                    visit_offset,
                    shape,
                )
                .map_err(cuda_error)?;
        } else {
            let launch = self
                .module
                .prepare_pisa_select(LaunchConfig1D::new(rows, 256, 0))
                .map_err(cuda_error)?;
            self.module
                .pisa_select(
                    &self.stream,
                    &launch,
                    index_query,
                    &s.qkv,
                    &s.pyramid,
                    &mut s.blocks,
                    shape,
                )
                .map_err(cuda_error)?;
        }
        if let Some(state) = self.mhc4.as_ref().filter(|state| state.index_enabled) {
            let destination = if shadow {
                &state.index_candidate
            } else {
                &state.index_incumbent
            };
            let bytes = self.rows * 8 * size_of::<u32>();
            let offset = ordinal * bytes;
            // SAFETY: the buffers reserve eleven complete recurrent plans and
            // ordinal was checked against that capacity by evaluate_mhc4.
            unsafe {
                cuda_core::simt::memory::memcpy_dtod_async(
                    destination.cu_deviceptr() + offset as u64,
                    s.blocks.cu_deviceptr(),
                    bytes,
                    self.stream.cu_stream(),
                )
                .map_err(cuda_error)?;
            }
        }
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
        let state: &mut Mhc4State = self
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
    pub(super) fn from_events(
        start: &CudaEvent,
        events: &[(Mhc4Stage, CudaEvent)],
    ) -> CudaResult<Self> {
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
