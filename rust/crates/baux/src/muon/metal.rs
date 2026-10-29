//! Encodable optimizer updates. Submission and gradient production belong to
//! the training loop. All scratch and optimizer state remain on the device.
extern crate metal as metal_crate;

use super::{POLAR, Polynomial, Shape, Step};
use crate::apple_gpu::{Runtime, thread_group};
use crate::fbt_mps::{Matmul, Matrix};
use metal_crate::foreign_types::{ForeignType, ForeignTypeRef};
use metal_crate::{
    Buffer, BufferRef, CommandBufferRef, ComputePipelineState, MTLCommandBufferStatus,
};
use std::sync::Arc;

const SOURCE: &str = include_str!("update.metal");

struct Kernels {
    init: ComputePipelineState,
    momentum: ComputePipelineState,
    norm: ComputePipelineState,
    normalize: ComputePipelineState,
    gram: ComputePipelineState,
    polynomial: ComputePipelineState,
    axpy: ComputePipelineState,
    update: ComputePipelineState,
    adam: ComputePipelineState,
}

impl Kernels {
    fn new(runtime: &Runtime) -> Result<Self, String> {
        let kernel = |name| runtime.precise(SOURCE, "Muon/AdamW", name);
        let result = Self {
            init: kernel("muon_init")?,
            momentum: kernel("muon_momentum")?,
            norm: kernel("muon_norm")?,
            normalize: kernel("muon_normalize")?,
            gram: kernel("muon_gram")?,
            polynomial: kernel("muon_polynomial")?,
            axpy: kernel("muon_axpy")?,
            update: kernel("muon_update")?,
            adam: kernel("adam_update")?,
        };
        if result.momentum.max_total_threads_per_threadgroup() < 256
            || result.momentum.thread_execution_width() != 32
        {
            return Err("Muon norm reduction requires 256 threads and 32-lane SIMD groups".into());
        }
        Ok(result)
    }
}

fn extent(runtime: &Runtime, buffer: &BufferRef, offset: u64, bytes: u64) -> Result<(), String> {
    if offset % 4 != 0
        || offset
            .checked_add(bytes)
            .is_none_or(|n| n > buffer.length())
        || buffer.device().as_ptr() != runtime.device.as_ptr()
    {
        return Err("optimizer buffer extent, alignment, or device mismatch".into());
    }
    Ok(())
}

fn bytes<T: Copy>(encoder: &metal_crate::ComputeCommandEncoderRef, slot: u64, value: &T) {
    encoder.set_bytes(slot, size_of::<T>() as u64, (value as *const T).cast());
}

fn dispatch(
    command: &CommandBufferRef,
    kernel: &ComputePipelineState,
    buffers: &[(&BufferRef, u64)],
    count: u32,
    shape: Option<&Shape>,
    step: Option<&Step>,
    polynomial: Option<&Polynomial>,
    groups: bool,
) {
    let encoder = command.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(kernel);
    for (slot, (buffer, offset)) in buffers.iter().enumerate() {
        encoder.set_buffer(slot as u64, Some(buffer), *offset);
    }
    let mut slot = buffers.len() as u64;
    if let Some(shape) = shape {
        bytes(encoder, slot, shape);
    } else {
        bytes(encoder, slot, &count);
    }
    slot += 1;
    if let Some(step) = step {
        bytes(encoder, slot, step);
        slot += 1;
    }
    if let Some(polynomial) = polynomial {
        bytes(encoder, slot, polynomial);
    }
    if groups {
        encoder.dispatch_thread_groups(thread_group(u64::from(count)), thread_group(256));
    } else {
        encoder.dispatch_threads(thread_group(u64::from(count)), thread_group(256));
    }
    encoder.end_encoding();
}

struct State {
    runtime: Arc<Runtime>,
    kernels: Kernels,
    master: Buffer,
    momentum: Buffer,
    count: u32,
    weights: Buffer,
    offset: u64,
}

impl State {
    fn new(weights: &BufferRef, offset: u64, count: u32) -> Result<Self, String> {
        let runtime = Runtime::shared()?;
        if count == 0 || u64::from(count) * 4 > runtime.device.max_buffer_length() {
            return Err("optimizer state exceeds the device buffer limit".into());
        }
        extent(&runtime, weights, offset, u64::from(count) * 2)?;
        let kernels = Kernels::new(&runtime)?;
        let master = runtime.buffer::<f32>(count as usize);
        let momentum = runtime.buffer::<f32>(count as usize);
        let command = runtime.queue.new_command_buffer();
        dispatch(
            command,
            &kernels.init,
            &[(weights, offset), (&master, 0), (&momentum, 0)],
            count,
            None,
            None,
            None,
            false,
        );
        command.commit();
        command.wait_until_completed();
        if command.status() != MTLCommandBufferStatus::Completed {
            return Err("optimizer state initialization failed".into());
        }
        Ok(Self {
            runtime,
            kernels,
            master,
            momentum,
            count,
            weights: weights.to_owned(),
            offset,
        })
    }

    fn validate(
        &self,
        command: &CommandBufferRef,
        weights: &BufferRef,
        offset: u64,
        gradient: &BufferRef,
        grad_offset: u64,
        step: Step,
    ) -> Result<(), String> {
        step.validate()?;
        extent(&self.runtime, weights, offset, u64::from(self.count) * 2)?;
        extent(
            &self.runtime,
            gradient,
            grad_offset,
            u64::from(self.count) * 4,
        )?;
        if weights.as_ptr() != self.weights.as_ptr()
            || offset != self.offset
            || weights.as_ptr() == gradient.as_ptr()
            || command.status() != MTLCommandBufferStatus::NotEnqueued
        {
            return Err("optimizer requires separate gradients and an unsubmitted command".into());
        }
        Ok(())
    }
}

/// One matrix bank, with independent momentum and polar normalization for each
/// matrix. Experts are a batch; never flatten their expert axis into a matrix.
pub struct Muon {
    state: State,
    shape: Shape,
}

/// One reusable scratch allocation for sequentially updated matrix banks.
/// Choose capacity from the largest bank, not from the total model parameters.
pub struct Workspace {
    capacity: Shape,
    matmul: Matmul,
    x: [Buffer; 2],
    gram: Buffer,
    polynomial: Buffer,
    product: Buffer,
    partials: Buffer,
    norms: Buffer,
}

impl Muon {
    pub fn new(shape: Shape, weights: &BufferRef, offset: u64) -> Result<Self, String> {
        let shape = Shape::new(shape.rows, shape.cols, shape.batch)?;
        let state = State::new(weights, offset, shape.elements() as u32)?;
        Ok(Self { shape, state })
    }

    /// Consumes FP32 gradients in place and encodes all five polar iterations.
    /// No submission or readback. MPS descriptors are cached on first use.
    /// `scale` multiplies the RMS matching factor 0.2 * sqrt(max(rows, cols)).
    pub fn encode(
        &mut self,
        workspace: &mut Workspace,
        command: &CommandBufferRef,
        weights: &BufferRef,
        offset: u64,
        gradient: &BufferRef,
        grad_offset: u64,
        mut step: Step,
    ) -> Result<(), String> {
        self.state
            .validate(command, weights, offset, gradient, grad_offset, step)?;
        step.scale *= 0.2 * (self.shape.rows.max(self.shape.cols) as f32).sqrt();
        let matrix = Shape::new(self.shape.rows, self.shape.cols, 1)?;
        let weight_stride = matrix.elements() as u64 * 2;
        let gradient_stride = matrix.elements() as u64 * 4;
        let mut batch = 0;
        while batch < self.shape.batch {
            let count = (self.shape.batch - batch).min(workspace.capacity.batch);
            let chunk = Shape::new(self.shape.rows, self.shape.cols, count)?;
            workspace.validate(chunk)?;
            self.encode_matrix(
                workspace,
                command,
                weights,
                offset + u64::from(batch) * weight_stride,
                gradient,
                grad_offset + u64::from(batch) * gradient_stride,
                u64::from(batch) * gradient_stride,
                chunk,
                step,
            )?;
            batch += count;
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn encode_matrix(
        &self,
        workspace: &mut Workspace,
        command: &CommandBufferRef,
        weights: &BufferRef,
        weight_offset: u64,
        gradient: &BufferRef,
        gradient_offset: u64,
        state_offset: u64,
        shape: Shape,
        step: Step,
    ) -> Result<(), String> {
        let state = &self.state;
        let k = &state.kernels;
        let count = shape.elements() as u32;
        dispatch(
            command,
            &k.momentum,
            &[
                (gradient, gradient_offset),
                (&state.momentum, state_offset),
                (&workspace.partials, 0),
            ],
            shape.chunks(),
            Some(&shape),
            Some(&step),
            None,
            true,
        );
        dispatch(
            command,
            &k.norm,
            &[(&workspace.partials, 0), (&workspace.norms, 0)],
            1,
            Some(&shape),
            Some(&step),
            None,
            true,
        );
        dispatch(
            command,
            &k.normalize,
            &[
                (gradient, gradient_offset),
                (&workspace.norms, 0),
                (&workspace.x[0], 0),
            ],
            count,
            Some(&shape),
            None,
            None,
            false,
        );
        let wide = shape.wide();
        let matrix = |buffer, rows, cols, element_bytes| {
            let matrix = if element_bytes == 2 {
                Matrix::half(buffer, rows, cols)
            } else {
                Matrix::new(buffer, rows, cols)
            };
            matrix.layout(wide.batch, u64::from(rows) * u64::from(cols), 0)
        };
        let gram_count = wide.rows * wide.rows * wide.batch;
        for (index, polynomial) in POLAR.iter().enumerate() {
            let x = &workspace.x[index % 2];
            let next = &workspace.x[1 - index % 2];
            workspace.matmul.encode(
                &state.runtime.device,
                command,
                matrix(x, wide.rows, wide.cols, 2),
                matrix(x, wide.rows, wide.cols, 2),
                matrix(&workspace.product, wide.rows, wide.rows, 4),
                true,
                1.0,
            )?;
            dispatch(
                command,
                &k.gram,
                &[(&workspace.product, 0), (&workspace.gram, 0)],
                gram_count,
                None,
                None,
                None,
                false,
            );
            workspace.matmul.encode(
                &state.runtime.device,
                command,
                matrix(&workspace.gram, wide.rows, wide.rows, 2),
                matrix(&workspace.gram, wide.rows, wide.rows, 2),
                matrix(&workspace.product, wide.rows, wide.rows, 4),
                false,
                1.0,
            )?;
            dispatch(
                command,
                &k.polynomial,
                &[
                    (&workspace.gram, 0),
                    (&workspace.product, 0),
                    (&workspace.polynomial, 0),
                ],
                gram_count,
                None,
                None,
                Some(polynomial),
                false,
            );
            workspace.matmul.encode(
                &state.runtime.device,
                command,
                matrix(&workspace.polynomial, wide.rows, wide.rows, 2),
                matrix(x, wide.rows, wide.cols, 2),
                matrix(&workspace.product, wide.rows, wide.cols, 4),
                false,
                1.0,
            )?;
            dispatch(
                command,
                &k.axpy,
                &[(x, 0), (&workspace.product, 0), (next, 0)],
                count,
                None,
                None,
                Some(polynomial),
                false,
            );
        }
        dispatch(
            command,
            &k.update,
            &[
                (&workspace.x[POLAR.len() % 2], 0),
                (&state.master, state_offset),
                (weights, weight_offset),
            ],
            count,
            Some(&shape),
            Some(&step),
            None,
            false,
        );
        Ok(())
    }

    pub fn state_bytes(&self) -> u64 {
        self.state.master.length() + self.state.momentum.length()
    }
}

impl Workspace {
    pub fn new(capacity: Shape) -> Result<Self, String> {
        let capacity = Shape::new(capacity.rows, capacity.cols, capacity.batch)?;
        let runtime = Runtime::shared()?;
        let wide = capacity.wide();
        let gram = wide.rows as usize * wide.rows as usize * wide.batch as usize;
        if capacity.elements().max(gram) as u64 * 4 > runtime.device.max_buffer_length() {
            return Err("Muon workspace exceeds device buffer limit".into());
        }
        Ok(Self {
            capacity,
            matmul: Matmul::default(),
            x: [
                runtime.buffer::<u16>(capacity.elements()),
                runtime.buffer::<u16>(capacity.elements()),
            ],
            gram: runtime.buffer::<u16>(gram),
            polynomial: runtime.buffer::<u16>(gram),
            product: runtime.buffer::<f32>(capacity.elements().max(gram)),
            partials: runtime.buffer::<f32>((capacity.chunks() * capacity.batch) as usize),
            norms: runtime.buffer::<f32>(capacity.batch as usize),
        })
    }

    fn validate(&self, shape: Shape) -> Result<(), String> {
        let wide = shape.wide();
        let gram = u64::from(wide.rows) * u64::from(wide.rows) * u64::from(wide.batch) * 2;
        if shape.elements() > self.capacity.elements()
            || gram > self.gram.length()
            || shape.chunks() * shape.batch > self.capacity.chunks() * self.capacity.batch
            || shape.batch > self.capacity.batch
        {
            return Err("Muon bank exceeds workspace capacity".into());
        }
        Ok(())
    }

    pub fn bytes(&self) -> u64 {
        [
            &self.x[0],
            &self.x[1],
            &self.gram,
            &self.polynomial,
            &self.product,
            &self.partials,
            &self.norms,
        ]
        .iter()
        .map(|b| b.length())
        .sum()
    }
}

/// Embeddings/readout, normalization gains, biases, and scalar controls use
/// AdamW. Their layout is not treated as a projection matrix by Muon.
pub struct Adam {
    state: State,
    second: Buffer,
    updates: u32,
}

impl Adam {
    pub fn new(weights: &BufferRef, offset: u64, count: u32) -> Result<Self, String> {
        let state = State::new(weights, offset, count)?;
        let second = state.runtime.buffer::<f32>(count as usize);
        let command = state.runtime.queue.new_command_buffer();
        let encoder = command.new_blit_command_encoder();
        encoder.fill_buffer(&second, metal_crate::NSRange::new(0, second.length()), 0);
        encoder.end_encoding();
        command.commit();
        command.wait_until_completed();
        if command.status() != MTLCommandBufferStatus::Completed {
            return Err("AdamW state initialization failed".into());
        }
        Ok(Self {
            state,
            second,
            updates: 0,
        })
    }

    pub fn encode(
        &mut self,
        command: &CommandBufferRef,
        weights: &BufferRef,
        offset: u64,
        gradient: &BufferRef,
        grad_offset: u64,
        step: Step,
    ) -> Result<(), String> {
        let next = self
            .updates
            .checked_add(1)
            .ok_or("AdamW update count overflow")?;
        let step = step.adam(next)?;
        self.state
            .validate(command, weights, offset, gradient, grad_offset, step)?;
        dispatch(
            command,
            &self.state.kernels.adam,
            &[
                (gradient, grad_offset),
                (&self.state.momentum, 0),
                (&self.second, 0),
                (&self.state.master, 0),
                (weights, offset),
            ],
            self.state.count,
            None,
            Some(&step),
            None,
            false,
        );
        self.updates = next;
        Ok(())
    }
}
