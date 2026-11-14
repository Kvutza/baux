//! Existing FBT/PISA device kernels behind the shared resident program
//! contract. Parameter rows stay on Metal; each call publishes fresh tensor
//! observations.
use super::*;
use baux::portable::{Buffer as TensorBuffer, Executable, Tensor};
use metal::foreign_types::{ForeignType, ForeignTypeRef};
use std::sync::{Arc, Mutex};
pub mod experiment;
mod exports;
mod inspect;
mod layout;
pub use layout::layout;
mod tvm;

/// Execution orchestration; both paths use the production Metal kernels.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, deser::Deserialize)]
#[deser(rename_all = "kebab-case")]
pub enum Backend {
    #[default]
    Native,
    Tvm,
}

/// A full initialized model and its existing resident search controller.
/// The forward signature is (FP16 parameters, u32 tokens, u32 labels, bool
/// mask) -> (f32 token losses, f32 sequence NLLs). Search mutation leases are
/// enforced by `program_base`/`program_candidate` on the search state. Use
/// `batch` to validate vocabulary IDs and masks before invoking `forward`.
pub struct Resident {
    pub forward: Executable,
    pub search: SearchState,
    pub parameters: usize,
    pub backend: Backend,
    state: Arc<Mutex<State>>,
}

struct State {
    runtime: Arc<Runtime>,
    pipelines: Pipelines,
    tensorops: TensorOpsPipelines,
    pisa: Pisa,
    buffers: Buffers,
    layout: Mhc4Layout,
    gpu_seconds: f64,
}

fn tensor(shape: usize, dtype: &str) -> Result<Tensor, String> {
    Tensor::new(vec![shape], dtype)
}

// The borrowed input keeps the allocation and its external mutation lease live
// until the command completes. This kernel reads inputs; it never donates them.
fn metal(value: &TensorBuffer) -> Result<metal::Buffer, String> {
    let (pointer, offset) = unsafe { value.metal()? };
    if offset != 0 {
        return Err("FBT inputs require whole contiguous allocations".into());
    }
    // BufferRef borrows the retained MTLBuffer. to_owned adds a native retain;
    // the portable input remains live through completion, preserving its lease.
    Ok(unsafe { metal::BufferRef::from_ptr(pointer.cast()) }.to_owned())
}

impl State {
    fn execute(&mut self, inputs: &[TensorBuffer]) -> Result<f64, String> {
        self.execute_on(&inputs.iter().map(metal).collect::<Result<Vec<_>, _>>()?)
    }

    fn execute_on(&mut self, inputs: &[metal::Buffer]) -> Result<f64, String> {
        let weights = &inputs[0];
        self.buffers.tokens = inputs[1].clone();
        self.buffers.labels = inputs[2].clone();
        self.buffers.score_mask = inputs[3].clone();
        let command = self.runtime.queue.new_command_buffer();
        scorer::objective_fused(
            command,
            &self.pipelines,
            &self.tensorops,
            &self.pisa,
            &self.buffers,
            self.layout.view(weights),
        )?;
        self.gpu_seconds = complete(command)?;
        Ok(self.gpu_seconds)
    }

    fn outputs(&self) -> Result<Vec<(metal::Buffer, usize)>, String> {
        // Scorer scratch is reused on the next call. Publish independent device
        // allocations rather than allowing a later round to overwrite results.
        let sources = [
            (&self.buffers.losses, ROWS as usize),
            (&self.buffers.sequence_scores, BATCH as usize),
        ];
        let command = self.runtime.queue.new_command_buffer();
        let blit = command.new_blit_command_encoder();
        let mut copied = Vec::new();
        for (source, count) in sources {
            let buffer = self.runtime.buffer::<f32>(count);
            blit.copy_from_buffer(source, 0, &buffer, 0, (count * 4) as u64);
            copied.push((buffer, count));
        }
        blit.end_encoding();
        complete(command)?;
        Ok(copied)
    }

    fn publish(&self, program: &Executable) -> Result<Vec<TensorBuffer>, String> {
        self.outputs()?
            .into_iter()
            .map(|(buffer, count)| {
                let handle = buffer.as_ptr().cast();
                // No mutable alias survives publication; the lease owns the
                // sole source handle and IREE also retains the
                // underlying allocation.
                unsafe {
                    program.import_metal(tensor(count, "float32")?, handle, 0, Arc::new(buffer))
                }
            })
            .collect()
    }
}

impl Resident {
    /// Explicit host snapshot for export or independent compiler parity checks.
    /// The read holds the same immutable search lease used by a forward call.
    ///
    /// # Returns
    /// Owned native-endian FP16 bytes in canonical packed order. Reading blocks
    /// until the device copy completes; editing the result does not mutate the
    /// resident parameters. [`layout`] describes the MHC4 parameter views.
    ///
    /// # Errors
    /// Returns an error if the optimizer cannot grant an immutable lease or
    /// the native readback fails.
    pub fn weights(&self) -> Result<Vec<u8>, String> {
        self.search.program_base(&self.forward)?.read()
    }

    pub fn new(
        model: crate::config::ModelArchitecture,
        initialization: crate::config::ModelInitialization,
        seed: u64,
        length: crate::trust_region::TRLengthConfig,
        perturbation: crate::ProposalDistribution,
    ) -> Result<Self, String> {
        Self::with_backend(
            model,
            initialization,
            seed,
            length,
            perturbation,
            Backend::Native,
        )
    }

    pub fn with_backend(
        model: crate::config::ModelArchitecture,
        initialization: crate::config::ModelInitialization,
        seed: u64,
        length: crate::trust_region::TRLengthConfig,
        perturbation: crate::ProposalDistribution,
        backend: Backend,
    ) -> Result<Self, String> {
        metal::objc::rc::autoreleasepool(|| {
            let runtime = Runtime::shared()?;
            let architecture = ResidualArchitecture::from_model(model);
            let weights =
                CandidateWeights::initialized(&runtime, seed, initialization, architecture)?;
            let parameters = architecture.parameter_count();
            let layout = Mhc4Layout::from_weights(&weights, 0);
            let (search, _) = weights.search(perturbation, length)?;
            let state = Arc::new(Mutex::new(State {
                pipelines: Pipelines::new(&runtime)?,
                tensorops: TensorOpsPipelines::new(&runtime)?,
                pisa: Pisa::new(&runtime)?,
                buffers: Buffers::new(&runtime),
                runtime,
                layout,
                gpu_seconds: 0.0,
            }));
            let inputs = vec![
                tensor(parameters, "float16")?,
                tensor(ROWS as usize, "uint32")?,
                tensor(ROWS as usize, "uint32")?,
                tensor(ROWS as usize, "bool")?,
            ];
            let outputs = vec![
                tensor(ROWS as usize, "float32")?,
                tensor(BATCH as usize, "float32")?,
            ];
            let forward = match backend {
                Backend::Tvm => tvm::forward(state.clone(), inputs, outputs)?,
                Backend::Native => {
                    let captured = state.clone();
                    Executable::native("metal", inputs, outputs, move |program, inputs| {
                        metal::objc::rc::autoreleasepool(|| {
                            let mut state =
                                captured.lock().map_err(|_| "FBT scorer mutex poisoned")?;
                            state.execute(inputs)?;
                            state.publish(program)
                        })
                    })?
                }
            };
            Ok(Self {
                forward,
                search,
                parameters,
                backend,
                state,
            })
        })
    }

    /// Upload one complete fixed-shape causal batch. This copies input tokens,
    /// labels and masks only; model parameters are leased directly from search.
    pub fn batch(
        &self,
        tokens: &[u32],
        labels: &[u32],
        mask: &[u8],
    ) -> Result<Vec<TensorBuffer>, String> {
        if tokens.len() != ROWS as usize
            || labels.len() != tokens.len()
            || mask.len() != tokens.len()
            || tokens.iter().chain(labels).any(|&token| token >= VOCAB)
            || mask.iter().any(|&value| value > 1)
        {
            return Err(
                "FBT batch requires valid vocabulary IDs and a binary mask at the model row count"
                    .into(),
            );
        }
        let words = |values: &[u32]| {
            values
                .iter()
                .flat_map(|v| v.to_ne_bytes())
                .collect::<Vec<_>>()
        };
        Ok(vec![
            self.forward
                .put(tensor(tokens.len(), "uint32")?, &words(tokens))?,
            self.forward
                .put(tensor(labels.len(), "uint32")?, &words(labels))?,
            self.forward.put(tensor(mask.len(), "bool")?, mask)?,
        ])
    }

    pub fn rows(&self) -> usize {
        ROWS as usize
    }

    /// Retain tensor observations until explicitly read. Releasing the returned
    /// completion does not cancel queued work or remove its parameter lease.
    pub fn submit(
        &self,
        weights: TensorBuffer,
        batch: &[TensorBuffer],
    ) -> Result<baux::portable::Pending, String> {
        let mut inputs = vec![weights];
        inputs.extend_from_slice(batch);
        self.forward.submit(inputs)
    }
}

fn floats(buffer: &TensorBuffer) -> Result<Vec<f32>, String> {
    Ok(buffer
        .read()?
        .chunks_exact(4)
        .map(|bytes| f32::from_ne_bytes(bytes.try_into().expect("four bytes")))
        .collect())
}

/// CPU control summary derived from explicitly read tensor observations.
#[derive(Clone, Debug)]
pub struct Observation {
    pub reward: f32,
    pub variance: f32,
    pub scored_tokens: usize,
    pub sequence_nlls: Vec<f32>,
}

impl Resident {
    pub fn observe(&self, outputs: &[TensorBuffer], mask: &[u8]) -> Result<Observation, String> {
        if outputs.len() != 2 {
            return Err("FBT requires token and sequence observations".into());
        }
        for (output, count) in outputs.iter().zip([ROWS as usize, BATCH as usize]) {
            if output.spec() != &tensor(count, "float32")? {
                return Err("FBT observation tensor differs from the scorer signature".into());
            }
        }
        if mask.iter().any(|&value| value > 1) {
            return Err("FBT observation mask must be binary".into());
        }
        let stats = tensor_objective(&floats(&outputs[1])?, &floats(&outputs[0])?, mask)?;
        Ok(Observation {
            reward: stats.reward,
            variance: stats.variance,
            scored_tokens: stats.scored_tokens,
            sequence_nlls: stats.sequence_nlls.to_vec(),
        })
    }

    /// The last complete forward's device interval, excluding result
    /// publication and optimizer kernels. Call after waiting for
    /// completion.
    pub fn gpu_seconds(&self) -> Result<f64, String> {
        Ok(self
            .state
            .lock()
            .map_err(|_| "FBT scorer mutex poisoned")?
            .gpu_seconds)
    }
}
