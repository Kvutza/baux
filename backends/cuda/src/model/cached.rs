//! Chunked causal execution. Each recurrent visit owns distinct persistent KV.
use super::*;

struct Visit {
    kv: DeviceBuffer<u16>,
    tree: DeviceBuffer<u16>,
}

pub(super) struct State {
    visits: Vec<Visit>,
    tokens: DeviceBuffer<u32>,
    queries: DeviceBuffer<u16>,
    pub(super) sample_from: usize,
}

impl FbtModel {
    pub(super) fn prepare_cache(&mut self, visits: usize) -> CudaResult<()> {
        if self.rows == self.sequence {
            return Ok(());
        }
        if self.cache.is_some() {
            return Ok(());
        }
        let mut states = Vec::with_capacity(visits);
        for _ in 0..visits {
            states.push(Visit {
                kv: DeviceBuffer::zeroed(&self.stream, self.sequence * 128).map_err(cuda_error)?,
                tree: DeviceBuffer::zeroed(&self.stream, (self.sequence / 32 - 1) * 64)
                    .map_err(cuda_error)?,
            });
        }
        self.cache = Some(State {
            visits: states,
            tokens: DeviceBuffer::zeroed(&self.stream, self.sequence).map_err(cuda_error)?,
            queries: DeviceBuffer::zeroed(&self.stream, self.rows * WIDTH).map_err(cuda_error)?,
            sample_from: 0,
        });
        self.first = 0;
        Ok(())
    }

    /// All KV is recomputed from position zero at the start of every
    /// invocation. Only accepted prefixes within that invocation may reuse
    /// cached states.
    pub fn generate_streamed(
        &mut self,
        prompt: &[u32],
        visits: usize,
        temperature: f32,
        seed: u64,
        unroll: usize,
        chunk: usize,
    ) -> CudaResult<(GenerationOutput, usize)> {
        let (plan, repair_chunk, wall) =
            self.init_streamed(prompt, visits, temperature, unroll, chunk)?;
        let begin = timing_event(&self.stream)?;
        let launch = self
            .module
            .prepare_verify_prefix(LaunchConfig1D::new(1, 256, 0))
            .map_err(cuda_error)?;
        let mut passes = 0;
        let mut evaluated = 0;
        let mut frontier = prompt.len();
        let mut frontiers = vec![frontier];
        let mut first_wave = None;
        let profile = std::env::var_os("BAUX_CUDA_PROFILE").is_some();
        let mut chunk_events = Vec::new();
        while frontier < self.sequence {
            // The token at frontier-1 may have just been corrected. Recompute
            // its chunk too; all preceding chunks depend only on unchanged IDs.
            for _ in 0..unroll {
                let rows = if passes == 0 { chunk } else { repair_chunk };
                self.rows = rows;
                let start = if passes == 0 {
                    0
                } else {
                    (frontier - 1) / rows * rows
                };
                for first in (start..self.sequence).step_by(rows) {
                    self.first = first;
                    // SAFETY: complete checked chunks in disjoint allocated
                    // buffers, ordered on the same stream
                    // as the forward and prefix update.
                    unsafe {
                        cuda_core::simt::memory::memcpy_dtod_async(
                            self.scratch.tokens.cu_deviceptr(),
                            self.cache.as_ref().unwrap().tokens.cu_deviceptr() + (first * 4) as u64,
                            rows * 4,
                            self.stream.cu_stream(),
                        )
                        .map_err(cuda_error)?;
                    }
                    if profile
                        && passes == 0
                        && [0, self.sequence / 2, self.sequence - chunk].contains(&first)
                    {
                        let start = timing_event(&self.stream)?;
                        let mut events = Vec::with_capacity(plan.len() * 12 + 2);
                        self.enqueue_events(&plan, temperature, seed, Some(&mut events), None)?;
                        chunk_events.push((first, start, events));
                    } else {
                        self.enqueue(&plan, temperature, seed, None)?;
                    }
                    evaluated += rows;
                }
                if passes == 0 {
                    first_wave = Some(timing_event(&self.stream)?);
                }
                self.module
                    .verify_prefix(
                        &self.stream,
                        &launch,
                        &self.scratch.sampled,
                        &mut self.cache.as_mut().unwrap().tokens,
                        &mut self.scratch.frontier,
                        self.sequence as u32,
                    )
                    .map_err(cuda_error)?;
                passes += 1;
            }
            let next = read_prefix(&self.scratch.frontier, &self.stream, 1)?[0] as usize;
            if next <= frontier || next > self.sequence {
                return Err("streamed accepted-prefix generation failed to advance".into());
            }
            frontier = next;
            frontiers.push(frontier);
            if self.sequence > 65536 {
                println!(
                    "GENERATION_PROGRESS context={} committed={} waves={} evaluated={}",
                    self.sequence,
                    frontier - prompt.len(),
                    passes,
                    evaluated
                );
            }
        }
        let end = timing_event(&self.stream)?;
        let tokens = read_prefix(
            &self.cache.as_ref().unwrap().tokens,
            &self.stream,
            self.sequence,
        )?;
        if tokens.iter().any(|&t| t >= VOCAB as u32) || tokens[..prompt.len()] != *prompt {
            return Err("streamed generation changed prompt or produced invalid tokens".into());
        }
        self.context.check_err().map_err(cuda_error)?;
        self.rows = chunk;
        let wall_ms = wall.elapsed().as_secs_f32() * 1000.0;
        let chunk_profiles = chunk_events
            .iter()
            .map(|(first, start, events)| Ok((*first, WaveProfile::from_events(start, events)?)))
            .collect::<CudaResult<Vec<_>>>()?;
        Ok((
            GenerationOutput {
                tokens,
                frontiers,
                prompt_tokens: prompt.len(),
                passes,
                visits: plan.len(),
                device_ms: begin.elapsed_ms(&end).map_err(cuda_error)?,
                wall_ms,
                first_wave_ms: first_wave
                    .as_ref()
                    .map(|event| begin.elapsed_ms(event).map_err(cuda_error))
                    .transpose()?,
                chunk_profiles,
            },
            evaluated,
        ))
    }

    fn init_streamed(
        &mut self,
        prompt: &[u32],
        visits: usize,
        temperature: f32,
        unroll: usize,
        chunk: usize,
    ) -> CudaResult<(Vec<usize>, usize, Instant)> {
        if self.weights.diffusion.is_some()
            || self.diffusion.is_some()
            || prompt.is_empty()
            || prompt.len() >= self.sequence
            || prompt.iter().any(|&t| t >= VOCAB as u32)
            || !temperature.is_finite()
            || temperature < 0.0
            || !(1..=32).contains(&unroll)
            || !(64..=4096).contains(&chunk)
            || !chunk.is_power_of_two()
            || self.sequence % chunk != 0
        {
            return Err(
                "streamed generation requires a causal checkpoint, nonempty prompt, finite \
                 temperature, 1..32 unroll and power-of-two chunk 64..4096 dividing context"
                    .into(),
            );
        }
        let plan = recurrence::RecurrentCore::selective_fbt().layer_visits(5, visits)?;
        let repair_chunk = repair_chunk(chunk)?;
        let wall = Instant::now();
        if self.rows != chunk {
            self.scratch = Scratch::new(&self.stream, chunk, self.sequence)?;
        }
        if self.scratch.sampled.len() != self.sequence {
            self.scratch.sampled =
                DeviceBuffer::zeroed(&self.stream, self.sequence).map_err(cuda_error)?;
        }
        let mut states = Vec::with_capacity(plan.len());
        for _ in &plan {
            states.push(Visit {
                kv: DeviceBuffer::zeroed(&self.stream, self.sequence * 128).map_err(cuda_error)?,
                tree: DeviceBuffer::zeroed(&self.stream, (self.sequence / 32 - 1) * 64)
                    .map_err(cuda_error)?,
            });
        }
        self.cache = Some(State {
            visits: states,
            tokens: DeviceBuffer::zeroed(&self.stream, self.sequence).map_err(cuda_error)?,
            queries: DeviceBuffer::zeroed(&self.stream, chunk * WIDTH).map_err(cuda_error)?,
            sample_from: prompt.len() - 1,
        });
        self.rows = chunk;
        self.first = 0;
        self.stream.synchronize().map_err(cuda_error)?;
        let mut tokens = vec![0; self.sequence];
        tokens[..prompt.len()].copy_from_slice(prompt);
        copy_prefix(&self.cache.as_ref().unwrap().tokens, &tokens, &self.stream)?;
        copy_prefix(&self.scratch.frontier, &[prompt.len() as u32], &self.stream)?;
        Ok((plan, repair_chunk, wall))
    }

    pub(super) fn cached_attention(
        &mut self,
        layer: usize,
        ordinal: usize,
        events: &mut Option<&mut Vec<(Stage, CudaEvent)>>,
    ) -> CudaResult<()> {
        let rows = self.rows as u32;
        let shape = PisaShape::cached(rows, self.sequence as u32, self.first as u32)
            .map_err(str::to_string)?;
        let cache = self.cache.as_mut().ok_or("missing visit cache")?;
        let state = cache
            .visits
            .get_mut(ordinal)
            .ok_or("missing recurrent visit cache")?;
        let s = &mut self.scratch;
        let weights = &self.weights.layers[layer];
        matmul(
            self.synth.as_ref(),
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
                self.first as u32,
            )
            .map_err(cuda_error)?;
        let launch = self
            .module
            .prepare_cache_split(LaunchConfig1D::new((rows * 640).div_ceil(256), 256, 0))
            .map_err(cuda_error)?;
        self.module
            .cache_split(
                &self.stream,
                &launch,
                &s.qkv,
                &mut cache.queries,
                &mut state.kv,
                rows,
                self.first as u32,
            )
            .map_err(cuda_error)?;
        let first = self.first as u32 / 64;
        let end = (self.first as u32 + rows) / 64;
        let launch = self
            .module
            .prepare_cache_leaves(LaunchConfig1D::new(end - first, 64, 0))
            .map_err(cuda_error)?;
        self.module
            .cache_leaves(
                &self.stream,
                &launch,
                &state.kv,
                &mut state.tree,
                first,
                end - first,
            )
            .map_err(cuda_error)?;
        let launch = self
            .module
            .prepare_cache_ancestors(LaunchConfig1D::new(1, 64, 0))
            .map_err(cuda_error)?;
        self.module
            .cache_ancestors(
                &self.stream,
                &launch,
                &mut state.tree,
                self.sequence as u32 / 64,
                first,
                end,
            )
            .map_err(cuda_error)?;
        self.select_cached(ordinal, rows, shape)?;
        mark(events, &self.stream, Stage::Index)?;
        let cache = self.cache.as_ref().ok_or("missing visit cache")?;
        let state = cache
            .visits
            .get(ordinal)
            .ok_or("missing recurrent visit cache")?;
        let s = &mut self.scratch;
        let launch = self
            .module
            .prepare_pisa_multihead(LaunchConfig1D::new(rows * 2, 256, 0))
            .map_err(cuda_error)?;
        self.module
            .pisa_multihead(
                &self.stream,
                &launch,
                &cache.queries,
                &state.kv,
                &s.blocks,
                &mut s.attended,
                shape,
            )
            .map_err(cuda_error)?;
        mark(events, &self.stream, Stage::Pisa)?;
        matmul(
            self.synth.as_ref(),
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

    fn select_cached(&mut self, ordinal: usize, rows: u32, shape: PisaShape) -> CudaResult<()> {
        let cache = self.cache.as_ref().ok_or("missing visit cache")?;
        let state = cache
            .visits
            .get(ordinal)
            .ok_or("missing recurrent visit cache")?;
        let s = &mut self.scratch;
        let shadow = self.mhc4.as_ref().is_some_and(|state| state.index_shadow);
        if shadow {
            let mhc4 = self.mhc4.as_ref().ok_or("missing mHC-4 PISA state")?;
            let visit_offset = u32::try_from(ordinal * self.rows * 8).map_err(cuda_error)?;
            let launch = self
                .module
                .prepare_pisa_shadow_refine(LaunchConfig1D::new(rows, 64, 0))
                .map_err(cuda_error)?;
            self.module
                .pisa_shadow_refine(
                    &self.stream,
                    &launch,
                    &cache.queries,
                    &state.tree,
                    &mhc4.index_incumbent,
                    &mut s.blocks,
                    visit_offset,
                    shape,
                )
                .map_err(cuda_error)?;
        } else {
            let launch = self
                .module
                .prepare_pisa_select(LaunchConfig1D::new(rows, 64, 0))
                .map_err(cuda_error)?;
            self.module
                .pisa_select(
                    &self.stream,
                    &launch,
                    &cache.queries,
                    &state.tree,
                    &mut s.blocks,
                    shape,
                )
                .map_err(cuda_error)?;
        }
        if let Some(mhc4) = self.mhc4.as_ref().filter(|state| state.index_enabled) {
            let destination = if shadow {
                &mhc4.index_candidate
            } else {
                &mhc4.index_incumbent
            };
            let bytes = self.rows * 8 * size_of::<u32>();
            let offset = ordinal * bytes;
            // SAFETY: evaluate_mhc4 bounds the recurrent ordinal and both
            // buffers reserve the complete eleven-visit support schedule.
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

    /// Reuse bounded vocabulary scratch. Sampling is keyed by absolute row,
    /// so changing chunk boundaries preserves the generated token sequence.
    pub(super) fn readout(
        &mut self,
        temperature: f32,
        seed: u64,
        mut score_events: Option<&mut Vec<(CudaEvent, CudaEvent)>>,
    ) -> CudaResult<()> {
        let capacity = self.scratch.logits.len() / VOCAB;
        // Prefix logits before the final prompt position cannot contribute to
        // generation. Keep absolute row seeds for the positions we do sample.
        let start = self.cache.as_ref().map_or(self.readout_from, |cache| {
            cache.sample_from.saturating_sub(self.first).min(self.rows)
        });
        for first in (start..self.rows).step_by(capacity) {
            let count = capacity.min(self.rows - first);
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
                first,
            )?;
            self.sample(count, temperature, seed, first)?;
            let score_start = score_events
                .as_ref()
                .map(|_| timing_event(&self.stream))
                .transpose()?;
            self.score(count, first.saturating_sub(start))?;
            if let (Some(score_start), Some(events)) = (score_start, score_events.as_deref_mut()) {
                events.push((score_start, timing_event(&self.stream)?));
            }
        }
        Ok(())
    }

    fn sample(
        &mut self,
        count: usize,
        temperature: f32,
        seed: u64,
        first: usize,
    ) -> CudaResult<()> {
        let position = (self.first + first) as u32;
        if let Some(state) = &mut self.diffusion {
            let launch = state
                .module
                .prepare_diffusion_sample(LaunchConfig1D::new(count as u32, 256, 0))
                .map_err(cuda_error)?;
            state
                .module
                .diffusion_sample(
                    &self.stream,
                    &launch,
                    &self.scratch.logits,
                    &mut self.scratch.sampled,
                    &mut state.confidence,
                    count as u32,
                    temperature,
                    seed,
                    position,
                )
                .map_err(cuda_error)
        } else {
            let launch = self
                .module
                .prepare_sample(LaunchConfig1D::new(count as u32, 256, 0))
                .map_err(cuda_error)?;
            self.module
                .sample(
                    &self.stream,
                    &launch,
                    &self.scratch.logits,
                    &mut self.scratch.sampled,
                    count as u32,
                    temperature,
                    seed,
                    position,
                )
                .map_err(cuda_error)
        }
    }

    fn score(&mut self, count: usize, offset: usize) -> CudaResult<()> {
        let Some(scoring) = &mut self.scoring else {
            return Ok(());
        };
        let count = count.min(scoring.count.saturating_sub(offset));
        if count == 0 {
            return Ok(());
        }
        let launch = self
            .module
            .prepare_score(LaunchConfig1D::new(count as u32, 256, 0))
            .map_err(cuda_error)?;
        self.module
            .score(
                &self.stream,
                &launch,
                &self.scratch.logits,
                &scoring.targets,
                &mut scoring.losses,
                count as u32,
                offset as u32,
                offset as u32,
            )
            .map_err(cuda_error)
    }
}

pub(super) fn repair_chunk(chunk: usize) -> CudaResult<usize> {
    let rows = std::env::var("BAUX_CUDA_REPAIR_CHUNK").map_or(Ok(chunk), |value| {
        value
            .parse::<usize>()
            .map_err(|_| "invalid repair chunk".to_string())
    })?;
    if !(64..=4096).contains(&rows) || !rows.is_power_of_two() {
        return Err("repair chunk must be a power of two from 64 through 4096".into());
    }
    Ok(rows.min(chunk))
}
