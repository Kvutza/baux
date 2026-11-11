//! Run the production causal scorer directly or as a Relax native call.
//! This preserves its Metal kernels; it is not StableHLO code generation.
use super::*;
use baux::portable::{Tensor, TvmExecutable, TvmTensor};
use metal::foreign_types::{ForeignType, ForeignTypeRef};
use std::sync::{Arc, Mutex};

pub(super) struct State {
    pub runtime: Arc<Runtime>,
    pub decoder: decode::Decoder,
    pub scorer: block_decode::BlockDecoder,
    pub layout: Mhc4Layout,
    pub batch: crate::pretrain_data::CausalBatch,
}

pub(super) struct Sample {
    pub mean_nll: f64,
    pub gpu_seconds: f64,
    pub wall_seconds: f64,
    pub scored_tokens: usize,
    pub losses: Vec<u8>,
}

impl State {
    fn score(&self, weights: &BufferRef) -> Result<block_decode::CausalScore, String> {
        self.scorer.score_batch(
            &self.runtime,
            &self.decoder,
            self.layout.view(weights),
            &self.batch,
        )
    }

    fn direct(&self, weights: &BufferRef) -> Result<Sample, String> {
        let started = Instant::now();
        let score = self.score(weights)?;
        let wall_seconds = started.elapsed().as_secs_f64();
        let losses = crate::apple_gpu::buffer_slice::<f32>(
            &self.scorer.buffers.losses,
            self.batch.tokens.len(),
        )?
        .iter()
        .flat_map(|value| value.to_ne_bytes())
        .collect();
        Ok(Sample {
            mean_nll: f64::from(score.quality.mean_nll),
            gpu_seconds: score.gpu_seconds,
            scored_tokens: score.quality.scored_tokens,
            wall_seconds,
            losses,
        })
    }

    fn tensors(&self, weights: &BufferRef) -> Result<Vec<TvmTensor>, String> {
        let score = self.score(weights)?;
        let metrics = [
            f64::from(score.quality.mean_nll),
            score.gpu_seconds,
            score.quality.scored_tokens as f64,
        ];
        let bytes = metrics
            .into_iter()
            .flat_map(f64::to_ne_bytes)
            .collect::<Vec<_>>();
        let losses = self.runtime.buffer::<f32>(self.batch.tokens.len());
        let command = self.runtime.queue.new_command_buffer();
        let blit = command.new_blit_command_encoder();
        blit.copy_from_buffer(&self.scorer.buffers.losses, 0, &losses, 0, losses.length());
        blit.end_encoding();
        complete(command)?;
        let pointer = losses.as_ptr().cast();
        Ok(vec![
            TvmTensor::put(Tensor::new(vec![3], "float64")?, &bytes)?,
            unsafe {
                // Completed copy, uniquely published allocation retained by
                // DLPack.
                TvmTensor::import_metal(
                    Tensor::new(vec![self.batch.tokens.len()], "float32")?,
                    pointer,
                    0,
                    Arc::new(losses),
                )?
            },
        ])
    }
}

pub(super) enum Forward {
    Native(Box<State>, Buffer),
    Tvm(TvmExecutable, TvmTensor),
}

impl Forward {
    pub fn new(
        state: State,
        weights: Buffer,
        backend: program::Backend,
    ) -> Result<(Self, Option<Sample>), String> {
        if backend == program::Backend::Native {
            return Ok((Self::Native(Box::new(state), weights), None));
        }
        // Independent direct invocation, before compilation and timed samples.
        // Retain every reference token loss, not just the aggregate NLL.
        let reference = state.direct(&weights)?;
        let spec = Tensor::new(vec![MHC_FULLPARAMS], "float16")?;
        let outputs = vec![
            Tensor::new(vec![3], "float64")?,
            Tensor::new(vec![state.batch.tokens.len()], "float32")?,
        ];
        let state = Mutex::new(state);
        let compiled =
            TvmExecutable::native(vec![spec.clone()], outputs, "metal", move |inputs| {
                autoreleasepool(|| {
                    let (pointer, offset) = unsafe { inputs[0].metal()? };
                    if offset != 0 {
                        return Err("causal scorer requires whole weights allocation".into());
                    }
                    // The input DLPack lease stays live until this synchronous
                    // call completes.
                    let weights = unsafe { BufferRef::from_ptr(pointer.cast()) };
                    state
                        .lock()
                        .map_err(|_| "causal scorer mutex poisoned")?
                        .tensors(weights)
                })
            })?;
        let pointer = weights.as_ptr().cast();
        let weights = unsafe { TvmTensor::import_metal(spec, pointer, 0, Arc::new(weights))? };
        Ok((Self::Tvm(compiled, weights), Some(reference)))
    }

    pub fn run(&self) -> Result<Sample, String> {
        match self {
            Self::Native(state, weights) => state.direct(weights),
            Self::Tvm(program, weights) => {
                let start = Instant::now();
                let outputs = program.run(&[weights])?;
                let wall_seconds = start.elapsed().as_secs_f64();
                let bytes = outputs[0].read()?;
                let metrics = bytes
                    .chunks_exact(8)
                    .map(|chunk| f64::from_ne_bytes(chunk.try_into().expect("eight-byte chunk")))
                    .collect::<Vec<_>>();
                Ok(Sample {
                    mean_nll: metrics[0],
                    gpu_seconds: metrics[1],
                    scored_tokens: metrics[2] as usize,
                    wall_seconds,
                    losses: outputs[1].read()?,
                })
            }
        }
    }
}
