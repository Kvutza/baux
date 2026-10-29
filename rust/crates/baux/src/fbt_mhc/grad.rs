//! Reverse kernels for the four-stream manifold hyperconnection.

use crate::apple_gpu::{Runtime, bind_buffers, thread_group};
use metal::{BufferRef, ComputeCommandEncoderRef, ComputePipelineState};

const SOURCE: &str = include_str!("../fbt_mhc_grad.metal");

#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct Shape {
    pub rows: u32,
    pub width: u32,
    pub architecture: u32,
    pub reserved: u32,
}

pub(crate) struct Kernels {
    mean: ComputePipelineState,
    mix: ComputePipelineState,
    update: ComputePipelineState,
    predict: ComputePipelineState,
}

pub(crate) struct Mean<'a> {
    pub streams: (&'a BufferRef, u64),
    pub weight: (&'a BufferRef, u64),
    pub output_grad: (&'a BufferRef, u64),
    pub stream_grad: (&'a BufferRef, u64),
    pub weight_grad: (&'a BufferRef, u64),
}

pub(crate) struct Mix<'a> {
    pub streams: (&'a BufferRef, u64),
    pub coefficients: (&'a BufferRef, u64),
    pub weight: (&'a BufferRef, u64),
    pub output_grad: (&'a BufferRef, u64),
    pub stream_grad: (&'a BufferRef, u64),
    pub coefficient_grad: (&'a BufferRef, u64),
    pub weight_grad: (&'a BufferRef, u64),
}

pub(crate) struct Update<'a> {
    pub streams: (&'a BufferRef, u64),
    pub branch: (&'a BufferRef, u64),
    pub coefficients: (&'a BufferRef, u64),
    pub output_grad: (&'a BufferRef, u64),
    pub stream_grad: (&'a BufferRef, u64),
    pub branch_grad: (&'a BufferRef, u64),
    pub coefficient_grad: (&'a BufferRef, u64),
}

pub(crate) struct Predict<'a> {
    pub streams: (&'a BufferRef, u64),
    pub predictor: (&'a BufferRef, u64),
    pub bias: (&'a BufferRef, u64),
    pub control: (&'a BufferRef, u64),
    pub coefficient_grad: (&'a BufferRef, u64),
    pub stream_grad: (&'a BufferRef, u64),
    pub predictor_grad: (&'a BufferRef, u64),
    pub bias_grad: (&'a BufferRef, u64),
    pub control_grad: (&'a BufferRef, u64),
}

fn shape(encoder: &ComputeCommandEncoderRef, slot: u64, value: &Shape) {
    encoder.set_bytes(
        slot,
        size_of::<Shape>() as u64,
        std::ptr::from_ref(value).cast(),
    );
}

impl Kernels {
    pub(crate) fn new(runtime: &Runtime) -> Result<Self, String> {
        let kernel = |name| runtime.precise(SOURCE, "mHC gradient", name);
        Ok(Self {
            mean: kernel("mhc_mean_rms_grad")?,
            mix: kernel("mhc_mix_rms_grad")?,
            update: kernel("mhc_update_grad")?,
            predict: kernel("mhc_predict_grad")?,
        })
    }

    pub(crate) fn mean(&self, encoder: &ComputeCommandEncoderRef, p: Shape, x: Mean<'_>) {
        encoder.set_compute_pipeline_state(&self.mean);
        bind_buffers(
            encoder,
            &[
                x.streams,
                x.weight,
                x.output_grad,
                x.stream_grad,
                x.weight_grad,
            ],
        );
        shape(encoder, 5, &p);
        encoder.dispatch_thread_groups(thread_group(u64::from(p.rows)), thread_group(128));
    }

    pub(crate) fn mix(&self, encoder: &ComputeCommandEncoderRef, p: Shape, x: Mix<'_>) {
        encoder.set_compute_pipeline_state(&self.mix);
        bind_buffers(
            encoder,
            &[
                x.streams,
                x.coefficients,
                x.weight,
                x.output_grad,
                x.stream_grad,
                x.coefficient_grad,
                x.weight_grad,
            ],
        );
        shape(encoder, 7, &p);
        encoder.dispatch_thread_groups(thread_group(u64::from(p.rows)), thread_group(128));
    }

    pub(crate) fn update(&self, encoder: &ComputeCommandEncoderRef, p: Shape, x: Update<'_>) {
        encoder.set_compute_pipeline_state(&self.update);
        bind_buffers(
            encoder,
            &[
                x.streams,
                x.branch,
                x.coefficients,
                x.output_grad,
                x.stream_grad,
                x.branch_grad,
                x.coefficient_grad,
            ],
        );
        shape(encoder, 7, &p);
        encoder.dispatch_threads(
            thread_group(u64::from(p.rows) * u64::from(p.width)),
            thread_group(self.update.max_total_threads_per_threadgroup().min(256)),
        );
    }

    pub(crate) fn predict(&self, encoder: &ComputeCommandEncoderRef, p: Shape, x: Predict<'_>) {
        encoder.set_compute_pipeline_state(&self.predict);
        bind_buffers(
            encoder,
            &[
                x.streams,
                x.predictor,
                x.bias,
                x.control,
                x.coefficient_grad,
                x.stream_grad,
                x.predictor_grad,
                x.bias_grad,
                x.control_grad,
            ],
        );
        shape(encoder, 9, &p);
        encoder.dispatch_threads(
            thread_group(u64::from(p.rows)),
            thread_group(self.predict.max_total_threads_per_threadgroup().min(256)),
        );
    }
}
