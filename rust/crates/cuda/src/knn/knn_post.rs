use baux_cuda_device::{
    BatchParams, BatchValue, DrawParams, KNN_K, PosteriorParams, THREADS, WeightedParams,
};
use cuda_core::{CudaStream, DeviceBuffer, LaunchConfig1D};

use super::*;

#[derive(Clone, Copy, Debug)]
pub struct PosteriorSpec {
    pub metrics: usize,
    pub input_k: usize,
    pub used_k: usize,
    pub skip: usize,
    pub epistemic_scale: f32,
    pub aleatoric_scale: f32,
    pub epsilon: f32,
}

#[derive(Debug)]
pub struct PosteriorOutput {
    pub means: Vec<f32>,
    pub errors: Vec<f32>,
    pub indices: Vec<u32>,
}

#[derive(Clone, Copy, Debug)]
pub struct WeightedSpec {
    pub metrics: usize,
    pub input_k: usize,
    pub used_k: usize,
    pub skip: usize,
    pub epistemic_scale: f32,
    pub aleatoric_scale: f32,
    pub epsilon: f32,
    pub observation_noise: bool,
}

#[derive(Debug)]
pub struct WeightedOutput {
    pub weights: Vec<f32>,
    pub l2: Vec<f32>,
    pub means: Vec<f32>,
    pub errors: Vec<f32>,
    pub epistemic: Vec<f32>,
    pub aleatoric: Vec<f32>,
    pub indices: Vec<u32>,
}

#[derive(Clone, Copy, Debug)]
pub struct BatchSpec {
    pub metrics: usize,
    pub input_k: usize,
    pub epsilon: f32,
    pub observation_noise: bool,
}

#[derive(Debug)]
pub struct BatchOutput {
    pub means: Vec<f32>,
    pub errors: Vec<f32>,
    pub epistemic: Vec<f32>,
    pub aleatoric: Vec<f32>,
}

#[derive(Debug)]
pub struct DrawOutput {
    pub draws: Vec<f64>,
    pub indices: Vec<u32>,
}

pub(crate) struct PosteriorScratch {
    outcome_capacity: usize,
    variance_capacity: usize,
    scale_capacity: usize,
    value_capacity: usize,
    weight_capacity: usize,
    param_capacity: usize,
    seed_capacity: usize,
    draw_capacity: usize,
    index_capacity: usize,
    pub(super) outcomes: DeviceBuffer<f32>,
    pub(super) variances: DeviceBuffer<f32>,
    pub(super) scales: DeviceBuffer<f32>,
    pub(super) weights: DeviceBuffer<f32>,
    pub(super) params: DeviceBuffer<BatchValue>,
    pub(super) seeds: DeviceBuffer<u64>,
    pub(super) draws: DeviceBuffer<f64>,
    pub(super) l2: DeviceBuffer<f32>,
    pub(super) means: DeviceBuffer<f32>,
    pub(super) errors: DeviceBuffer<f32>,
    pub(super) epistemic: DeviceBuffer<f32>,
    pub(super) aleatoric: DeviceBuffer<f32>,
    pub(super) indices: DeviceBuffer<u32>,
}

impl PosteriorScratch {
    pub(super) fn new(stream: &CudaStream) -> CudaResult<Self> {
        Ok(Self {
            outcome_capacity: 1,
            variance_capacity: 1,
            scale_capacity: 1,
            value_capacity: 1,
            weight_capacity: 1,
            param_capacity: 1,
            seed_capacity: 1,
            draw_capacity: 1,
            index_capacity: 1,
            outcomes: DeviceBuffer::zeroed(stream, 1).map_err(cuda_error)?,
            variances: DeviceBuffer::zeroed(stream, 1).map_err(cuda_error)?,
            scales: DeviceBuffer::zeroed(stream, 1).map_err(cuda_error)?,
            weights: DeviceBuffer::zeroed(stream, 1).map_err(cuda_error)?,
            params: DeviceBuffer::zeroed(stream, 1).map_err(cuda_error)?,
            seeds: DeviceBuffer::zeroed(stream, 1).map_err(cuda_error)?,
            draws: DeviceBuffer::zeroed(stream, 1).map_err(cuda_error)?,
            l2: DeviceBuffer::zeroed(stream, 1).map_err(cuda_error)?,
            means: DeviceBuffer::zeroed(stream, 1).map_err(cuda_error)?,
            errors: DeviceBuffer::zeroed(stream, 1).map_err(cuda_error)?,
            epistemic: DeviceBuffer::zeroed(stream, 1).map_err(cuda_error)?,
            aleatoric: DeviceBuffer::zeroed(stream, 1).map_err(cuda_error)?,
            indices: DeviceBuffer::zeroed(stream, 1).map_err(cuda_error)?,
        })
    }

    pub(super) fn ensure(
        &mut self,
        stream: &CudaStream,
        outcomes: usize,
        variances: usize,
        scales: usize,
        values: usize,
        weights: usize,
        indices: usize,
    ) -> CudaResult<()> {
        let outcome_capacity = next_capacity(outcomes)?;
        let variance_capacity = next_capacity(variances)?;
        let scale_capacity = next_capacity(scales)?;
        let value_capacity = next_capacity(values)?;
        let weight_capacity = next_capacity(weights)?;
        let index_capacity = next_capacity(indices)?;
        if outcome_capacity > self.outcome_capacity {
            self.outcomes = DeviceBuffer::zeroed(stream, outcome_capacity).map_err(cuda_error)?;
            self.outcome_capacity = outcome_capacity;
        }
        if variance_capacity > self.variance_capacity {
            self.variances = DeviceBuffer::zeroed(stream, variance_capacity).map_err(cuda_error)?;
            self.variance_capacity = variance_capacity;
        }
        if scale_capacity > self.scale_capacity {
            self.scales = DeviceBuffer::zeroed(stream, scale_capacity).map_err(cuda_error)?;
            self.scale_capacity = scale_capacity;
        }
        if value_capacity > self.value_capacity {
            self.l2 = DeviceBuffer::zeroed(stream, value_capacity).map_err(cuda_error)?;
            self.means = DeviceBuffer::zeroed(stream, value_capacity).map_err(cuda_error)?;
            self.errors = DeviceBuffer::zeroed(stream, value_capacity).map_err(cuda_error)?;
            self.epistemic = DeviceBuffer::zeroed(stream, value_capacity).map_err(cuda_error)?;
            self.aleatoric = DeviceBuffer::zeroed(stream, value_capacity).map_err(cuda_error)?;
            self.value_capacity = value_capacity;
        }
        if weight_capacity > self.weight_capacity {
            self.weights = DeviceBuffer::zeroed(stream, weight_capacity).map_err(cuda_error)?;
            self.weight_capacity = weight_capacity;
        }
        if index_capacity > self.index_capacity {
            self.indices = DeviceBuffer::zeroed(stream, index_capacity).map_err(cuda_error)?;
            self.index_capacity = index_capacity;
        }
        Ok(())
    }

    pub(super) fn sync(
        &mut self,
        stream: &CudaStream,
        outcomes: &[f32],
        scales: &[f32],
    ) -> CudaResult<()> {
        copy_prefix(&self.outcomes, outcomes, stream)?;
        copy_prefix(&self.scales, scales, stream)
    }

    pub(super) fn sync_variances(
        &mut self,
        stream: &CudaStream,
        variances: Option<&[f32]>,
    ) -> CudaResult<()> {
        if let Some(values) = variances {
            copy_prefix(&self.variances, values, stream)?;
        }
        Ok(())
    }

    fn ensure_params(&mut self, stream: &CudaStream, count: usize) -> CudaResult<()> {
        let capacity = next_capacity(count)?;
        if capacity > self.param_capacity {
            self.params = DeviceBuffer::zeroed(stream, capacity).map_err(cuda_error)?;
            self.param_capacity = capacity;
        }
        Ok(())
    }

    fn sync_params(&mut self, stream: &CudaStream, values: &[BatchValue]) -> CudaResult<()> {
        copy_prefix(&self.params, values, stream)
    }

    fn ensure_draws(
        &mut self,
        stream: &CudaStream,
        seed_count: usize,
        draw_count: usize,
    ) -> CudaResult<()> {
        let seed_capacity = next_capacity(seed_count)?;
        let draw_capacity = next_capacity(draw_count)?;
        if seed_capacity > self.seed_capacity {
            self.seeds = DeviceBuffer::zeroed(stream, seed_capacity).map_err(cuda_error)?;
            self.seed_capacity = seed_capacity;
        }
        if draw_capacity > self.draw_capacity {
            self.draws = DeviceBuffer::zeroed(stream, draw_capacity).map_err(cuda_error)?;
            self.draw_capacity = draw_capacity;
        }
        Ok(())
    }

    fn sync_seeds(&mut self, stream: &CudaStream, seeds: &[u64]) -> CudaResult<()> {
        copy_prefix(&self.seeds, seeds, stream)
    }
}

impl CudaIndex {
    pub fn posterior(
        &self,
        scratch: &mut CudaScratch,
        queries: &[f32],
        query_count: usize,
        outcomes: &[f32],
        scales: &[f32],
        spec: PosteriorSpec,
    ) -> CudaResult<PosteriorOutput> {
        self.check_posterior(outcomes, scales, spec)?;
        let value_count = query_count
            .checked_mul(spec.metrics)
            .ok_or("CUDA posterior output size overflow")?;
        let index_count = query_count
            .checked_mul(spec.used_k)
            .ok_or("CUDA posterior index size overflow")?;
        scratch.posterior.ensure(
            &self.runtime.stream,
            outcomes.len(),
            1,
            scales.len(),
            value_count,
            1,
            index_count,
        )?;
        scratch
            .posterior
            .sync(&self.runtime.stream, outcomes, scales)?;
        let stage = self.stage_search(scratch, queries, query_count, spec.input_k)?;
        self.launch_posterior(scratch, &stage, spec)?;
        let end = timing_event(&self.runtime.stream)?;
        self.runtime.context.check_err().map_err(cuda_error)?;
        let output = PosteriorOutput {
            means: read_prefix(&scratch.posterior.means, &self.runtime.stream, value_count)?,
            errors: read_prefix(&scratch.posterior.errors, &self.runtime.stream, value_count)?,
            indices: read_prefix(
                &scratch.posterior.indices,
                &self.runtime.stream,
                index_count,
            )?,
        };
        let profile = self.make_profile(&stage, &end)?;
        scratch.profile = Some(profile);
        Ok(output)
    }

    pub fn weighted(
        &self,
        scratch: &mut CudaScratch,
        queries: &[f32],
        query_count: usize,
        outcomes: &[f32],
        variances: Option<&[f32]>,
        scales: &[f32],
        spec: WeightedSpec,
    ) -> CudaResult<WeightedOutput> {
        self.check_weighted(outcomes, variances, scales, spec)?;
        let value_count = query_count
            .checked_mul(spec.metrics)
            .ok_or("CUDA weighted output size overflow")?;
        let weight_count = value_count
            .checked_mul(spec.used_k)
            .ok_or("CUDA weighted weight size overflow")?;
        let index_count = query_count
            .checked_mul(spec.used_k)
            .ok_or("CUDA weighted index size overflow")?;
        scratch.posterior.ensure(
            &self.runtime.stream,
            outcomes.len(),
            variances.map_or(1, <[f32]>::len),
            scales.len(),
            value_count,
            weight_count,
            index_count,
        )?;
        scratch
            .posterior
            .sync(&self.runtime.stream, outcomes, scales)?;
        scratch
            .posterior
            .sync_variances(&self.runtime.stream, variances)?;
        let stage = self.stage_search(scratch, queries, query_count, spec.input_k)?;
        self.launch_weighted(scratch, &stage, spec, variances.is_some())?;
        let end = timing_event(&self.runtime.stream)?;
        self.runtime.context.check_err().map_err(cuda_error)?;
        let output = WeightedOutput {
            weights: read_prefix(
                &scratch.posterior.weights,
                &self.runtime.stream,
                weight_count,
            )?,
            l2: read_prefix(&scratch.posterior.l2, &self.runtime.stream, value_count)?,
            means: read_prefix(&scratch.posterior.means, &self.runtime.stream, value_count)?,
            errors: read_prefix(&scratch.posterior.errors, &self.runtime.stream, value_count)?,
            epistemic: read_prefix(
                &scratch.posterior.epistemic,
                &self.runtime.stream,
                value_count,
            )?,
            aleatoric: read_prefix(
                &scratch.posterior.aleatoric,
                &self.runtime.stream,
                value_count,
            )?,
            indices: read_prefix(
                &scratch.posterior.indices,
                &self.runtime.stream,
                index_count,
            )?,
        };
        let profile = self.make_profile(&stage, &end)?;
        scratch.profile = Some(profile);
        Ok(output)
    }

    pub fn batch(
        &self,
        scratch: &mut CudaScratch,
        queries: &[f32],
        query_count: usize,
        outcomes: &[f32],
        variances: Option<&[f32]>,
        scales: &[f32],
        values: &[BatchValue],
        spec: BatchSpec,
    ) -> CudaResult<BatchOutput> {
        self.check_batch(outcomes, variances, scales, values, spec)?;
        let output_count = values
            .len()
            .checked_mul(query_count)
            .and_then(|count| count.checked_mul(spec.metrics))
            .ok_or("CUDA batch output size overflow")?;
        scratch.posterior.ensure(
            &self.runtime.stream,
            outcomes.len(),
            variances.map_or(1, <[f32]>::len),
            scales.len(),
            output_count,
            1,
            1,
        )?;
        scratch
            .posterior
            .ensure_params(&self.runtime.stream, values.len())?;
        scratch
            .posterior
            .sync(&self.runtime.stream, outcomes, scales)?;
        scratch
            .posterior
            .sync_variances(&self.runtime.stream, variances)?;
        scratch
            .posterior
            .sync_params(&self.runtime.stream, values)?;
        let stage = self.stage_search(scratch, queries, query_count, spec.input_k)?;
        self.launch_batch(scratch, &stage, spec, values.len(), variances.is_some())?;
        let end = timing_event(&self.runtime.stream)?;
        self.runtime.context.check_err().map_err(cuda_error)?;
        let output = BatchOutput {
            means: read_prefix(&scratch.posterior.means, &self.runtime.stream, output_count)?,
            errors: read_prefix(
                &scratch.posterior.errors,
                &self.runtime.stream,
                output_count,
            )?,
            epistemic: read_prefix(
                &scratch.posterior.epistemic,
                &self.runtime.stream,
                output_count,
            )?,
            aleatoric: read_prefix(
                &scratch.posterior.aleatoric,
                &self.runtime.stream,
                output_count,
            )?,
        };
        let profile = self.make_profile(&stage, &end)?;
        scratch.profile = Some(profile);
        Ok(output)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn draws(
        &self,
        scratch: &mut CudaScratch,
        queries: &[f32],
        query_count: usize,
        outcomes: &[f32],
        variances: Option<&[f32]>,
        scales: &[f32],
        seeds: &[u64],
        spec: WeightedSpec,
    ) -> CudaResult<DrawOutput> {
        if seeds.is_empty() {
            return Err("CUDA draws require at least one seed".to_string());
        }
        self.check_weighted(outcomes, variances, scales, spec)?;
        let value_count = query_count
            .checked_mul(spec.metrics)
            .ok_or("CUDA draw value size overflow")?;
        let weight_count = value_count
            .checked_mul(spec.used_k)
            .ok_or("CUDA draw weight size overflow")?;
        let index_count = query_count
            .checked_mul(spec.used_k)
            .ok_or("CUDA draw index size overflow")?;
        let draw_count = seeds
            .len()
            .checked_mul(value_count)
            .ok_or("CUDA draw output size overflow")?;
        scratch.posterior.ensure(
            &self.runtime.stream,
            outcomes.len(),
            variances.map_or(1, <[f32]>::len),
            scales.len(),
            value_count,
            weight_count,
            index_count,
        )?;
        scratch
            .posterior
            .ensure_draws(&self.runtime.stream, seeds.len(), draw_count)?;
        scratch
            .posterior
            .sync(&self.runtime.stream, outcomes, scales)?;
        scratch
            .posterior
            .sync_variances(&self.runtime.stream, variances)?;
        scratch.posterior.sync_seeds(&self.runtime.stream, seeds)?;
        let stage = self.stage_search(scratch, queries, query_count, spec.input_k)?;
        self.launch_weighted(scratch, &stage, spec, variances.is_some())?;
        self.launch_draw(scratch, &stage, spec, seeds.len())?;
        let end = timing_event(&self.runtime.stream)?;
        self.runtime.context.check_err().map_err(cuda_error)?;
        let output = DrawOutput {
            draws: read_prefix(&scratch.posterior.draws, &self.runtime.stream, draw_count)?,
            indices: read_prefix(
                &scratch.posterior.indices,
                &self.runtime.stream,
                index_count,
            )?,
        };
        let profile = self.make_profile(&stage, &end)?;
        scratch.profile = Some(profile);
        Ok(output)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn conditional(
        &self,
        scratch: &mut CudaScratch,
        queries: &[f32],
        query_count: usize,
        extra_rows: &[f32],
        outcomes: &[f32],
        variances: Option<&[f32]>,
        scales: &[f32],
        spec: WeightedSpec,
    ) -> CudaResult<WeightedOutput> {
        let extended = self.clone().add(scratch, extra_rows)?;
        extended.weighted(
            scratch,
            queries,
            query_count,
            outcomes,
            variances,
            scales,
            spec,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn conditional_draws(
        &self,
        scratch: &mut CudaScratch,
        queries: &[f32],
        query_count: usize,
        extra_rows: &[f32],
        outcomes: &[f32],
        variances: Option<&[f32]>,
        scales: &[f32],
        seeds: &[u64],
        spec: WeightedSpec,
    ) -> CudaResult<DrawOutput> {
        let extended = self.clone().add(scratch, extra_rows)?;
        extended.draws(
            scratch,
            queries,
            query_count,
            outcomes,
            variances,
            scales,
            seeds,
            spec,
        )
    }

    fn launch_posterior(
        &self,
        scratch: &mut CudaScratch,
        stage: &SearchStage,
        spec: PosteriorSpec,
    ) -> CudaResult<()> {
        let params = PosteriorParams {
            queries: to_u32(stage.queries, "posterior query count")?,
            input_k: to_u32(spec.input_k, "posterior input neighbors")?,
            used_k: to_u32(spec.used_k, "posterior used neighbors")?,
            skip: to_u32(spec.skip, "posterior skip")?,
            metrics: to_u32(spec.metrics, "posterior metric count")?,
            epistemic_scale: spec.epistemic_scale,
            aleatoric_scale: spec.aleatoric_scale,
            epsilon: spec.epsilon,
        };
        let launch = self
            .runtime
            .module
            .prepare_posterior_light(LaunchConfig1D::new(params.queries, THREADS, 0))
            .map_err(cuda_error)?;
        if stage.input_a {
            self.runtime
                .module
                .posterior_light(
                    &self.runtime.stream,
                    &launch,
                    &scratch.knn.distances_a,
                    &scratch.knn.indices_a,
                    &scratch.posterior.outcomes,
                    &scratch.posterior.scales,
                    &mut scratch.posterior.means,
                    &mut scratch.posterior.errors,
                    &mut scratch.posterior.indices,
                    params,
                )
                .map_err(cuda_error)
        } else {
            self.runtime
                .module
                .posterior_light(
                    &self.runtime.stream,
                    &launch,
                    &scratch.knn.distances_b,
                    &scratch.knn.indices_b,
                    &scratch.posterior.outcomes,
                    &scratch.posterior.scales,
                    &mut scratch.posterior.means,
                    &mut scratch.posterior.errors,
                    &mut scratch.posterior.indices,
                    params,
                )
                .map_err(cuda_error)
        }
    }

    pub(super) fn launch_weighted(
        &self,
        scratch: &mut CudaScratch,
        stage: &SearchStage,
        spec: WeightedSpec,
        has_yvar: bool,
    ) -> CudaResult<()> {
        let params = WeightedParams {
            queries: to_u32(stage.queries, "weighted query count")?,
            input_k: to_u32(spec.input_k, "weighted input neighbors")?,
            used_k: to_u32(spec.used_k, "weighted used neighbors")?,
            skip: to_u32(spec.skip, "weighted skip")?,
            metrics: to_u32(spec.metrics, "weighted metric count")?,
            has_yvar: u32::from(has_yvar),
            observation_noise: u32::from(spec.observation_noise),
            pad: 0,
            epistemic_scale: spec.epistemic_scale,
            aleatoric_scale: spec.aleatoric_scale,
            epsilon: spec.epsilon,
            padf: 0.0,
        };
        let blocks = params
            .queries
            .checked_mul(params.metrics)
            .ok_or("CUDA weighted grid size overflow")?;
        let launch = self
            .runtime
            .module
            .prepare_posterior_full(LaunchConfig1D::new(blocks, THREADS, 0))
            .map_err(cuda_error)?;
        if stage.input_a {
            self.runtime.module.posterior_full(
                &self.runtime.stream,
                &launch,
                &scratch.knn.distances_a,
                &scratch.knn.indices_a,
                &scratch.posterior.outcomes,
                &scratch.posterior.variances,
                &scratch.posterior.scales,
                &mut scratch.posterior.weights,
                &mut scratch.posterior.l2,
                &mut scratch.posterior.means,
                &mut scratch.posterior.errors,
                &mut scratch.posterior.epistemic,
                &mut scratch.posterior.aleatoric,
                &mut scratch.posterior.indices,
                params,
            )
        } else {
            self.runtime.module.posterior_full(
                &self.runtime.stream,
                &launch,
                &scratch.knn.distances_b,
                &scratch.knn.indices_b,
                &scratch.posterior.outcomes,
                &scratch.posterior.variances,
                &scratch.posterior.scales,
                &mut scratch.posterior.weights,
                &mut scratch.posterior.l2,
                &mut scratch.posterior.means,
                &mut scratch.posterior.errors,
                &mut scratch.posterior.epistemic,
                &mut scratch.posterior.aleatoric,
                &mut scratch.posterior.indices,
                params,
            )
        }
        .map_err(cuda_error)
    }

    fn launch_batch(
        &self,
        scratch: &mut CudaScratch,
        stage: &SearchStage,
        spec: BatchSpec,
        param_count: usize,
        has_yvar: bool,
    ) -> CudaResult<()> {
        let params = BatchParams {
            queries: to_u32(stage.queries, "batch query count")?,
            input_k: to_u32(spec.input_k, "batch input neighbors")?,
            metrics: to_u32(spec.metrics, "batch metric count")?,
            param_count: to_u32(param_count, "batch parameter count")?,
            has_yvar: u32::from(has_yvar),
            observation_noise: u32::from(spec.observation_noise),
            pad0: 0,
            pad1: 0,
            epsilon: spec.epsilon,
            padf0: 0.0,
            padf1: 0.0,
            padf2: 0.0,
        };
        let blocks = params
            .param_count
            .checked_mul(params.queries)
            .and_then(|count| count.checked_mul(params.metrics))
            .ok_or("CUDA batch grid size overflow")?;
        let launch = self
            .runtime
            .module
            .prepare_posterior_batch(LaunchConfig1D::new(blocks, THREADS, 0))
            .map_err(cuda_error)?;
        if stage.input_a {
            self.runtime.module.posterior_batch(
                &self.runtime.stream,
                &launch,
                &scratch.knn.distances_a,
                &scratch.knn.indices_a,
                &scratch.posterior.outcomes,
                &scratch.posterior.variances,
                &scratch.posterior.scales,
                &scratch.posterior.params,
                &mut scratch.posterior.means,
                &mut scratch.posterior.errors,
                &mut scratch.posterior.epistemic,
                &mut scratch.posterior.aleatoric,
                params,
            )
        } else {
            self.runtime.module.posterior_batch(
                &self.runtime.stream,
                &launch,
                &scratch.knn.distances_b,
                &scratch.knn.indices_b,
                &scratch.posterior.outcomes,
                &scratch.posterior.variances,
                &scratch.posterior.scales,
                &scratch.posterior.params,
                &mut scratch.posterior.means,
                &mut scratch.posterior.errors,
                &mut scratch.posterior.epistemic,
                &mut scratch.posterior.aleatoric,
                params,
            )
        }
        .map_err(cuda_error)
    }

    fn launch_draw(
        &self,
        scratch: &mut CudaScratch,
        stage: &SearchStage,
        spec: WeightedSpec,
        seed_count: usize,
    ) -> CudaResult<()> {
        let params = DrawParams {
            queries: to_u32(stage.queries, "draw query count")?,
            neighbors: to_u32(spec.used_k, "draw neighbor count")?,
            metrics: to_u32(spec.metrics, "draw metric count")?,
            seed_count: to_u32(seed_count, "draw seed count")?,
        };
        let outputs = params
            .seed_count
            .checked_mul(params.queries)
            .and_then(|count| count.checked_mul(params.metrics))
            .ok_or("CUDA draw grid size overflow")?;
        let blocks = outputs.div_ceil(THREADS);
        let launch = self
            .runtime
            .module
            .prepare_posterior_draw(LaunchConfig1D::new(blocks, THREADS, 0))
            .map_err(cuda_error)?;
        self.runtime
            .module
            .posterior_draw(
                &self.runtime.stream,
                &launch,
                &scratch.posterior.weights,
                &scratch.posterior.l2,
                &scratch.posterior.means,
                &scratch.posterior.errors,
                &scratch.posterior.indices,
                &scratch.posterior.seeds,
                &mut scratch.posterior.draws,
                params,
            )
            .map_err(cuda_error)
    }

    fn check_posterior(
        &self,
        outcomes: &[f32],
        scales: &[f32],
        spec: PosteriorSpec,
    ) -> CudaResult<()> {
        let expected = self
            .row_count
            .checked_mul(spec.metrics)
            .ok_or("CUDA posterior outcome size overflow")?;
        let used_end = spec
            .skip
            .checked_add(spec.used_k)
            .ok_or("CUDA posterior neighbor range overflow")?;
        if spec.metrics == 0
            || spec.input_k == 0
            || spec.used_k == 0
            || spec.input_k > KNN_K
            || used_end > spec.input_k
        {
            return Err("CUDA posterior shape is unsupported".to_string());
        }
        if outcomes.len() != expected || scales.len() != spec.metrics {
            return Err("CUDA posterior outcomes or scales have the wrong shape".to_string());
        }
        if outcomes.iter().any(|value| !value.is_finite())
            || scales.iter().any(|value| !value.is_finite())
            || !spec.epistemic_scale.is_finite()
            || !spec.aleatoric_scale.is_finite()
            || !spec.epsilon.is_finite()
            || spec.epsilon <= 0.0
        {
            return Err("CUDA posterior values must be finite".to_string());
        }
        Ok(())
    }

    pub(super) fn check_weighted(
        &self,
        outcomes: &[f32],
        variances: Option<&[f32]>,
        scales: &[f32],
        spec: WeightedSpec,
    ) -> CudaResult<()> {
        self.check_posterior(
            outcomes,
            scales,
            PosteriorSpec {
                metrics: spec.metrics,
                input_k: spec.input_k,
                used_k: spec.used_k,
                skip: spec.skip,
                epistemic_scale: spec.epistemic_scale,
                aleatoric_scale: spec.aleatoric_scale,
                epsilon: spec.epsilon,
            },
        )?;
        let expected = self
            .row_count
            .checked_mul(spec.metrics)
            .ok_or("CUDA weighted variance size overflow")?;
        if let Some(values) = variances {
            if values.len() != expected || values.iter().any(|value| !value.is_finite()) {
                return Err("CUDA weighted variances have the wrong shape or values".to_string());
            }
        }
        if scales.iter().any(|value| *value <= 0.0) {
            return Err("CUDA weighted scales must be positive".to_string());
        }
        Ok(())
    }

    fn check_batch(
        &self,
        outcomes: &[f32],
        variances: Option<&[f32]>,
        scales: &[f32],
        values: &[BatchValue],
        spec: BatchSpec,
    ) -> CudaResult<()> {
        let Some(first) = values.first() else {
            return Err("CUDA batch requires parameters".to_string());
        };
        self.check_weighted(
            outcomes,
            variances,
            scales,
            WeightedSpec {
                metrics: spec.metrics,
                input_k: spec.input_k,
                used_k: first.used_k as usize,
                skip: first.skip as usize,
                epistemic_scale: first.epistemic_scale,
                aleatoric_scale: first.aleatoric_scale,
                epsilon: spec.epsilon,
                observation_noise: spec.observation_noise,
            },
        )?;
        for value in values {
            let end = (value.skip as usize)
                .checked_add(value.used_k as usize)
                .ok_or("CUDA batch neighbor range overflow")?;
            if value.used_k == 0
                || end > spec.input_k
                || !value.epistemic_scale.is_finite()
                || !value.aleatoric_scale.is_finite()
            {
                return Err("CUDA batch parameters are unsupported".to_string());
            }
        }
        Ok(())
    }
}
