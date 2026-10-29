//! General ENN moments and joint draws after the existing device KNN scan.
use super::*;
extern crate metal as mtl;
use mtl::foreign_types::{ForeignType, ForeignTypeRef};
use mtl::{Buffer, BufferRef, MTLCommandBufferStatus};

const POSTERIOR: &str = include_str!("posterior.metal");

#[repr(C)]
#[derive(Clone, Copy)]
struct Shape {
    queries: u32,
    metrics: u32,
    input_k: u32,
    k: u32,
    skip: u32,
    has_var: u32,
    noise: u32,
    seeds: u32,
    epistemic: f32,
    aleatoric: f32,
}

pub(crate) struct MetalPosterior {
    runtime: Arc<Runtime>,
    pub(crate) means: Buffer,
    pub(crate) errors: Buffer,
    weights: Buffer,
    indices: Buffer,
    l2: Buffer,
    shape: Shape,
}

fn completed(command: &metal::CommandBufferRef) -> Result<(), String> {
    command.commit();
    command.wait_until_completed();
    if command.status() == MTLCommandBufferStatus::Error {
        return Err("Metal posterior command failed".into());
    }
    Ok(())
}

fn validate_posterior(
    index: &MetalIndex,
    pointer: u64,
    _byte_offset: usize,
    queries: usize,
    outcomes: &[f32],
    variances: Option<&[f32]>,
    scales: &[f32],
    k: usize,
    skip: usize,
) -> Result<(usize, usize, usize, usize, usize), String> {
    let input_k = k.checked_add(skip).ok_or("neighbor count overflow")?;
    if pointer == 0
        || queries == 0
        || !MetalIndex::supports_neighbors(input_k)
        || input_k > index.len()
        || scales.is_empty()
    {
        return Err("unsupported Metal posterior shape".into());
    }
    let metrics = scales.len();
    let count = queries
        .checked_mul(metrics)
        .ok_or("posterior size overflow")?;
    let neighbors = queries.checked_mul(k).ok_or("neighbor size overflow")?;
    let weight_count = neighbors
        .checked_mul(metrics)
        .ok_or("weight size overflow")?;
    for size in [count, neighbors, weight_count, outcomes.len()] {
        to_u32(size, "posterior extent").map_err(|e| e.to_string())?;
    }
    if outcomes.len() != index.len() * metrics
        || variances.is_some_and(|v| v.len() != outcomes.len())
    {
        return Err("posterior observation shape mismatch".into());
    }
    let bytes = queries
        .checked_mul(index.num_dim)
        .and_then(|v| v.checked_mul(4))
        .ok_or("query size overflow")?;
    Ok((input_k, count, neighbors, weight_count, bytes))
}

impl MetalIndex {
    /// # Safety
    /// pointer is a live MTLBuffer with queries*dims contiguous FP32 values at
    /// byte_offset, synchronized by the producer before calling this method.
    pub(crate) unsafe fn device_posterior(
        &mut self,
        pointer: u64,
        byte_offset: usize,
        queries: usize,
        outcomes: &[f32],
        variances: Option<&[f32]>,
        scales: &[f32],
        k: usize,
        skip: usize,
        epistemic: f32,
        aleatoric: f32,
        noise: bool,
    ) -> Result<MetalPosterior, String> {
        let (input_k, count, neighbors, weight_count, bytes) = validate_posterior(
            self,
            pointer,
            byte_offset,
            queries,
            outcomes,
            variances,
            scales,
            k,
            skip,
        )?;
        let metrics = scales.len();
        // SAFETY: DLPack supplies the retained Objective-C MTLBuffer, not a CPU
        // data pointer. It remains alive through command completion below.
        let source = unsafe { BufferRef::from_ptr(pointer as *mut _) };
        if byte_offset % 4 != 0
            || byte_offset
                .checked_add(bytes)
                .is_none_or(|v| v as u64 > source.length())
            || source.device().as_ptr() != self.runtime.device.as_ptr()
        {
            return Err("Metal query extent or device mismatch".into());
        }
        self.scratch
            .ensure(&self.runtime, self.num_dim, queries, input_k);
        self.scratch.ensure_tree(
            &self.runtime,
            queries,
            input_k,
            plan_lists(Plan::Wide, self.len()),
            2,
        );
        let shape = Shape {
            queries: to_u32(queries, "queries").map_err(|e| e.to_string())?,
            metrics: to_u32(metrics, "metrics").map_err(|e| e.to_string())?,
            input_k: input_k as u32,
            k: k as u32,
            skip: skip as u32,
            has_var: u32::from(variances.is_some()),
            noise: u32::from(noise),
            seeds: 0,
            epistemic,
            aleatoric,
        };
        let output = MetalPosterior {
            runtime: Arc::clone(&self.runtime),
            means: self.runtime.buffer::<f32>(count),
            errors: self.runtime.buffer::<f32>(count),
            weights: self.runtime.buffer::<f32>(weight_count),
            indices: self.runtime.buffer::<u32>(neighbors),
            l2: self.runtime.buffer::<f32>(count),
            shape,
        };
        let y = self.runtime.buffer_with(outcomes);
        let var = self.runtime.buffer_with(variances.unwrap_or(&[0.0]));
        let scale = self.runtime.buffer_with(scales);
        let kernel = self
            .runtime
            .precise(POSTERIOR, "ENN posterior", "enn_moments")?;
        let command = self.runtime.queue.new_command_buffer();
        let blit = command.new_blit_command_encoder();
        blit.copy_from_buffer(
            source,
            byte_offset as u64,
            &self.scratch.query,
            0,
            bytes as u64,
        );
        blit.end_encoding();
        self.encode_tree(Plan::Wide, command, queries, input_k)?;
        let encoder = command.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&kernel);
        for (slot, buffer) in [
            &self.scratch.result_distances,
            &self.scratch.result_indices,
            &y,
            &var,
            &scale,
            &output.means,
            &output.errors,
            &output.weights,
            &output.l2,
            &output.indices,
        ]
        .iter()
        .enumerate()
        {
            encoder.set_buffer(slot as u64, Some(buffer), 0);
        }
        encoder.set_bytes(
            10,
            std::mem::size_of::<Shape>() as u64,
            (&shape as *const Shape).cast(),
        );
        encoder.dispatch_threads(MTLSize::new(count as u64, 1, 1), MTLSize::new(256, 1, 1));
        encoder.end_encoding();
        completed(command)?;
        Ok(output)
    }
}

impl MetalPosterior {
    pub(crate) fn draw(&self, seeds: &[u64]) -> Result<Buffer, String> {
        if seeds.is_empty() {
            return Err("draw seeds must be nonempty".into());
        }
        let count = seeds
            .len()
            .checked_mul(self.shape.queries as usize)
            .and_then(|v| v.checked_mul(self.shape.metrics as usize))
            .ok_or("draw size overflow")?;
        to_u32(count, "draw extent").map_err(|e| e.to_string())?;
        let mut shape = self.shape;
        shape.seeds = to_u32(seeds.len(), "seeds").map_err(|e| e.to_string())?;
        let output = self.runtime.buffer::<f32>(count);
        let seeds = self.runtime.buffer_with(seeds);
        let kernel = self.runtime.precise(POSTERIOR, "ENN draw", "enn_draw")?;
        let command = self.runtime.queue.new_command_buffer();
        let encoder = command.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&kernel);
        for (slot, buffer) in [
            &self.weights,
            &self.l2,
            &self.means,
            &self.errors,
            &self.indices,
            &seeds,
            &output,
        ]
        .iter()
        .enumerate()
        {
            encoder.set_buffer(slot as u64, Some(buffer), 0);
        }
        encoder.set_bytes(
            7,
            std::mem::size_of::<Shape>() as u64,
            (&shape as *const Shape).cast(),
        );
        encoder.dispatch_threads(MTLSize::new(count as u64, 1, 1), MTLSize::new(256, 1, 1));
        encoder.end_encoding();
        completed(command)?;
        Ok(output)
    }
}
