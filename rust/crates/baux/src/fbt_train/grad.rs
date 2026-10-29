use crate::apple_gpu::{Runtime, bind_buffers, encode_value, thread_group};
use metal::{BufferRef, ComputeCommandEncoderRef, ComputePipelineState};

const SOURCE: &str = include_str!("../fbt_train_grad.metal");

#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct DenseShape {
    pub rows: u32,
    pub input: u32,
    pub output: u32,
    pub reserved: u32,
}

pub(crate) struct DenseGrad<'a> {
    pub input: (&'a BufferRef, u64),
    pub weights: (&'a BufferRef, u64),
    pub output_grad: (&'a BufferRef, u64),
    pub input_grad: (&'a BufferRef, u64),
    pub weight_grad: (&'a BufferRef, u64),
    pub half_grad: &'a BufferRef,
}

pub(crate) struct EmbedGrad<'a> {
    pub readout: (&'a BufferRef, u64),
    pub tokens: (&'a BufferRef, u64),
    pub positions: (&'a BufferRef, u64),
    pub output_grad: (&'a BufferRef, u64),
    pub readout_grad: (&'a BufferRef, u64),
    pub position_grad: (&'a BufferRef, u64),
    pub mask_grad: (&'a BufferRef, u64),
}

#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct ReadoutShape {
    pub rows: u32,
    pub width: u32,
    pub vocab: u32,
    pub start: u32,
    pub scale: f32,
}

pub(crate) struct ReadoutGrad<'a> {
    pub hidden: (&'a BufferRef, u64),
    pub readout: (&'a BufferRef, u64),
    pub expected: (&'a BufferRef, u64),
    pub mask: (&'a BufferRef, u64),
    pub hidden_grad: (&'a BufferRef, u64),
    pub readout_grad: (&'a BufferRef, u64),
}

pub(crate) struct Kernels {
    dense_input: ComputePipelineState,
    dense_weight: ComputePipelineState,
    half: ComputePipelineState,
    rope: ComputePipelineState,
    embed: ComputePipelineState,
    readout: ComputePipelineState,
    replicate: ComputePipelineState,
}

fn dispatch(encoder: &ComputeCommandEncoderRef, pipeline: &ComputePipelineState, count: u64) {
    encoder.set_compute_pipeline_state(pipeline);
    encoder.dispatch_threads(
        thread_group(count),
        thread_group(pipeline.max_total_threads_per_threadgroup().min(256)),
    );
}

impl Kernels {
    pub(crate) fn new(runtime: &Runtime) -> Result<Self, String> {
        let kernel = |name| runtime.precise_metal4(SOURCE, "training gradient", name, &[]);
        Ok(Self {
            dense_input: kernel("dense_input_tensor")?,
            dense_weight: kernel("dense_weight_tensor")?,
            half: kernel("gradient_half")?,
            rope: kernel("rope_grad")?,
            embed: kernel("embed_grad")?,
            readout: kernel("readout_grad")?,
            replicate: kernel("replicate_grad")?,
        })
    }

    pub(crate) fn dense(
        &self,
        encoder: &ComputeCommandEncoderRef,
        p: DenseShape,
        x: DenseGrad<'_>,
    ) {
        let count = p.rows * p.output;
        bind_buffers(encoder, &[x.output_grad, (x.half_grad, 0)]);
        encode_value(encoder, 2, &count);
        dispatch(encoder, &self.half, u64::from(count));
        encoder.memory_barrier_with_resources(&[x.half_grad]);
        bind_buffers(encoder, &[x.weights, (x.half_grad, 0), x.input_grad]);
        encode_value(encoder, 3, &p);
        encoder.set_compute_pipeline_state(&self.dense_input);
        encoder.dispatch_thread_groups(
            metal::MTLSize {
                width: u64::from(p.input.div_ceil(64)),
                height: u64::from(p.rows.div_ceil(128)),
                depth: 1,
            },
            thread_group(128),
        );
        encoder.memory_barrier_with_resources(&[x.input_grad.0]);
        bind_buffers(encoder, &[x.input, (x.half_grad, 0), x.weight_grad]);
        encode_value(encoder, 3, &p);
        encoder.set_compute_pipeline_state(&self.dense_weight);
        encoder.dispatch_thread_groups(
            metal::MTLSize {
                width: u64::from(p.output.div_ceil(64)),
                height: u64::from(p.input.div_ceil(64)),
                depth: 1,
            },
            thread_group(128),
        );
    }

    pub(crate) fn rope(
        &self,
        encoder: &ComputeCommandEncoderRef,
        grad: (&BufferRef, u64),
        table: &BufferRef,
        rows: u32,
        start: u32,
        context: u32,
    ) {
        let p = [rows, start, context, 32u32];
        bind_buffers(encoder, &[grad, (table, 0)]);
        encode_value(encoder, 2, &p);
        dispatch(encoder, &self.rope, u64::from(rows) * 9 * 32);
    }

    pub(crate) fn embed(
        &self,
        encoder: &ComputeCommandEncoderRef,
        p: DenseShape,
        x: EmbedGrad<'_>,
    ) {
        bind_buffers(
            encoder,
            &[
                x.readout,
                x.tokens,
                x.positions,
                x.output_grad,
                x.readout_grad,
                x.position_grad,
                x.mask_grad,
            ],
        );
        encode_value(encoder, 7, &p);
        dispatch(encoder, &self.embed, u64::from(p.rows) * u64::from(p.input));
    }

    pub(crate) fn readout(
        &self,
        encoder: &ComputeCommandEncoderRef,
        p: ReadoutShape,
        x: ReadoutGrad<'_>,
    ) {
        bind_buffers(
            encoder,
            &[
                x.hidden,
                x.readout,
                x.expected,
                x.mask,
                x.hidden_grad,
                x.readout_grad,
            ],
        );
        encode_value(encoder, 6, &p);
        encoder.set_compute_pipeline_state(&self.readout);
        encoder.dispatch_thread_groups(thread_group(u64::from(p.rows)), thread_group(256));
    }

    pub(crate) fn replicate(
        &self,
        encoder: &ComputeCommandEncoderRef,
        p: DenseShape,
        stream_grad: (&BufferRef, u64),
        input_grad: (&BufferRef, u64),
    ) {
        bind_buffers(encoder, &[stream_grad, input_grad]);
        encode_value(encoder, 2, &p);
        dispatch(
            encoder,
            &self.replicate,
            u64::from(p.rows) * u64::from(p.input),
        );
    }
}
