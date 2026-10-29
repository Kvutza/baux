use super::generation::{GenerationTask, Serialize};
use super::*;

const GROUPS: usize = 1024;
const THREADS: usize = 256;
const AUDIT_CHUNK: usize = 1 << 20;
const SOURCE: &str = include_str!("alignment.metal");

#[repr(C)]
#[derive(Clone, Copy)]
struct Params {
    elements: u64,
    stride: u64,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct AuditSpan {
    offset: u64,
    elements: u32,
    family: u32,
}

#[derive(Clone, Copy, Serialize)]
pub(super) struct Alignment {
    pub(super) schema: &'static str,
    pub(super) reference: &'static str,
    pub(super) cosine: f64,
    pub(super) candidate_norm: f64,
    pub(super) muon_norm: f64,
    pub(super) incumbent_nll: f32,
    pub(super) candidate_nll: f32,
    pub(super) muon_nll: Option<f32>,
    pub(super) candidate_nll_change: f32,
    pub(super) muon_nll_change: Option<f32>,
    pub(super) backward_ms: f64,
    pub(super) update_ms: f64,
    pub(super) reduction_ms: f64,
    pub(super) diagnostic_ms: f64,
}

pub(super) struct Probe {
    trainer: train::Trainer,
    incumbent: Buffer,
    partials: Buffer,
    pipeline: ComputePipelineState,
    audit_spans: Buffer,
    audit_output: Buffer,
    audit_gradient: ComputePipelineState,
    audit_weight: ComputePipelineState,
    chunk_families: Vec<usize>,
    families: Vec<&'static str>,
    elements: usize,
}

#[derive(Serialize)]
pub(super) struct Preflight {
    schema: &'static str,
    context_tokens: usize,
    generated_tokens: u32,
    coarse_rows: u32,
    conditional_nll: f32,
    backward_ms: f64,
    update_ms: f64,
}

impl Probe {
    pub(super) fn new(
        runtime: &Runtime,
        weights: &CandidateWeights,
        rows: u32,
    ) -> Result<Self, String> {
        let trainer = train::Trainer::new(runtime, weights, rows)?;
        let elements = weights.architecture.parameter_count();
        let mut spans = Vec::new();
        let mut chunk_families = Vec::new();
        let mut families = Vec::new();
        let mut offset = 0usize;
        for (family_index, (family, tensor, _)) in weights.tensors().into_iter().enumerate() {
            let family_elements = tensor.length() as usize / size_of::<u16>();
            families.push(family);
            for chunk in (0..family_elements).step_by(AUDIT_CHUNK) {
                spans.push(AuditSpan {
                    offset: (offset + chunk) as u64,
                    elements: (family_elements - chunk).min(AUDIT_CHUNK) as u32,
                    family: family_index as u32,
                });
                chunk_families.push(family_index);
            }
            offset += family_elements;
        }
        if offset != elements {
            return Err("alignment audit does not cover the complete model".into());
        }
        Ok(Self {
            incumbent: runtime.buffer::<u16>(elements),
            partials: runtime.buffer::<[f32; 4]>(2 * GROUPS),
            pipeline: runtime.precise(SOURCE, "Optimizer alignment", "update_alignment")?,
            audit_spans: runtime.buffer_with(&spans),
            audit_output: runtime.buffer::<[f32; 2]>(spans.len()),
            audit_gradient: runtime.precise(SOURCE, "Optimizer audit", "audit_gradients")?,
            audit_weight: runtime.precise(SOURCE, "Optimizer audit", "audit_weights")?,
            chunk_families,
            families,
            trainer,
            elements,
        })
    }

    pub(super) fn preserve(&self, runtime: &Runtime, source: &BufferRef) -> Result<(), String> {
        let bytes = (self.elements * size_of::<u16>()) as u64;
        if source.length() != bytes {
            return Err("alignment incumbent has the wrong weight extent".into());
        }
        let command = runtime.queue.new_command_buffer();
        let blit = command.new_blit_command_encoder();
        blit.copy_from_buffer(source, 0, &self.incumbent, 0, bytes);
        blit.copy_from_buffer(source, 0, self.trainer.weights(), 0, bytes);
        blit.end_encoding();
        command.commit();
        complete_committed(command)?;
        Ok(())
    }

    pub(super) fn incumbent(&self) -> &BufferRef {
        &self.incumbent
    }

    pub(super) fn muon(&self) -> &BufferRef {
        self.trainer.weights()
    }

    pub(super) fn update(
        &mut self,
        runtime: &Runtime,
        verifier: &block_decode::BlockDecoder,
        task: &GenerationTask,
        tokens: u32,
        candidate: &BufferRef,
        incumbent_nll: f32,
        candidate_nll: f32,
    ) -> Result<Alignment, String> {
        let active_rows = task.prompt.len() + tokens as usize - 1;
        let hidden_max = check_half(
            "normalized state",
            &verifier.buffers.normalized,
            active_rows * WIDTH as usize,
        )?;
        let expected = unsafe {
            std::slice::from_raw_parts(verifier.expected.contents().cast::<u32>(), tokens as usize)
        };
        if let Some((index, value)) = expected
            .iter()
            .copied()
            .enumerate()
            .find(|(_, value)| *value >= VOCAB)
        {
            return Err(format!(
                "target {value} at generated position {index} exceeds vocabulary {VOCAB}"
            ));
        }
        let backward = self.trainer.backward(runtime, verifier, task, tokens)?;
        self.audit(runtime, self.trainer.gradients(), false)
            .map_err(|error| format!("{error}; forward maxima: normalized={hidden_max:.6}"))?;
        let mut optimizer = train::Optimizer::muon(&self.trainer)?;
        let update = optimizer.encode(runtime, &self.trainer)?;
        self.audit(runtime, self.trainer.weights(), true)?;
        let reduction = self.reduce(runtime, candidate)?;
        Ok(Alignment {
            incumbent_nll,
            candidate_nll,
            candidate_nll_change: candidate_nll - incumbent_nll,
            backward_ms: backward * 1000.0,
            update_ms: update * 1000.0,
            ..reduction
        })
    }

    fn audit(&self, runtime: &Runtime, values: &BufferRef, weights: bool) -> Result<f64, String> {
        let command = runtime.queue.new_command_buffer();
        let encoder = command.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(if weights {
            &self.audit_weight
        } else {
            &self.audit_gradient
        });
        encoder.set_buffer(0, Some(values), 0);
        encoder.set_buffer(1, Some(&self.audit_spans), 0);
        encoder.set_buffer(2, Some(&self.audit_output), 0);
        encoder.dispatch_thread_groups(
            crate::apple_gpu::thread_group(self.chunk_families.len() as u64),
            crate::apple_gpu::thread_group(THREADS as u64),
        );
        encoder.end_encoding();
        command.commit();
        let seconds = complete_committed(command)?;
        let values = unsafe {
            std::slice::from_raw_parts(
                self.audit_output.contents().cast::<[f32; 2]>(),
                self.chunk_families.len(),
            )
        };
        let mut invalid = vec![0u64; self.families.len()];
        let mut maximum = vec![0.0f32; self.families.len()];
        for ((value, &family), span) in values.iter().zip(&self.chunk_families).zip(unsafe {
            std::slice::from_raw_parts(
                self.audit_spans.contents().cast::<AuditSpan>(),
                self.chunk_families.len(),
            )
        }) {
            if span.family as usize != family {
                return Err("optimizer audit family map is corrupt".into());
            }
            invalid[family] += value[0] as u64;
            maximum[family] = maximum[family].max(value[1]);
        }
        let failures = self
            .families
            .iter()
            .enumerate()
            .filter(|(index, _)| invalid[*index] != 0)
            .map(|(index, family)| {
                format!(
                    "{family}: invalid={} max={}",
                    invalid[index], maximum[index]
                )
            })
            .collect::<Vec<_>>();
        if !failures.is_empty() {
            return Err(format!(
                "optimizer {} audit failed: {}",
                if weights { "weight" } else { "gradient" },
                failures.join(", ")
            ));
        }
        Ok(seconds)
    }

    pub(super) fn preflight(
        &mut self,
        runtime: &Runtime,
        verifier: &block_decode::BlockDecoder,
        task: &GenerationTask,
        tokens: u32,
        nll: f32,
    ) -> Result<Preflight, String> {
        let backward = self.trainer.backward(runtime, verifier, task, tokens)?;
        let mut optimizer = train::Optimizer::muon(&self.trainer)?;
        let update = optimizer.encode(runtime, &self.trainer)?;
        Ok(Preflight {
            schema: "baux.update-alignment-preflight.v1",
            context_tokens: task.prompt.len() + tokens as usize,
            generated_tokens: tokens,
            coarse_rows: verifier.context,
            conditional_nll: nll,
            backward_ms: backward * 1000.0,
            update_ms: update * 1000.0,
        })
    }

    fn reduce(&self, runtime: &Runtime, candidate: &BufferRef) -> Result<Alignment, String> {
        let bytes = (self.elements * size_of::<u16>()) as u64;
        if candidate.length() != bytes || self.trainer.weights().length() != bytes {
            return Err("alignment candidate has the wrong weight extent".into());
        }
        let params = Params {
            elements: self.elements as u64,
            stride: (GROUPS * THREADS) as u64,
        };
        let command = runtime.queue.new_command_buffer();
        let encoder = command.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.pipeline);
        for (index, buffer) in [
            &self.incumbent,
            candidate,
            self.trainer.weights(),
            &self.partials,
        ]
        .into_iter()
        .enumerate()
        {
            encoder.set_buffer(index as u64, Some(buffer), 0);
        }
        encoder.set_bytes(
            4,
            size_of::<Params>() as u64,
            (&params as *const Params).cast(),
        );
        encoder.dispatch_thread_groups(
            crate::apple_gpu::thread_group(GROUPS as u64),
            crate::apple_gpu::thread_group(THREADS as u64),
        );
        encoder.end_encoding();
        command.commit();
        let seconds = complete_committed(command)?;
        let partials = unsafe {
            std::slice::from_raw_parts(self.partials.contents().cast::<[f32; 4]>(), 2 * GROUPS)
        };
        let totals = partials
            .iter()
            .step_by(2)
            .fold([0.0f64; 4], |mut total, value| {
                for index in 0..4 {
                    total[index] += f64::from(value[index]);
                }
                total
            });
        let invalid = partials
            .iter()
            .skip(1)
            .step_by(2)
            .fold([0.0f64; 3], |mut total, value| {
                for index in 0..3 {
                    total[index] += f64::from(value[index]);
                }
                total
            });
        if totals.iter().any(|value| !value.is_finite())
            || invalid.iter().any(|value| !value.is_finite())
            || totals[1] < 0.0
            || totals[2] < 0.0
        {
            return Err(format!(
                "optimizer alignment reduction is invalid: moments={totals:?} invalid={invalid:?}"
            ));
        }
        if invalid.iter().any(|count| *count != 0.0) {
            return Err(format!(
                "optimizer comparison non-finite coordinates: incumbent={} candidate={} muon={}",
                invalid[0] as u64, invalid[1] as u64, invalid[2] as u64
            ));
        }
        let candidate_norm = totals[1].sqrt();
        let muon_norm = totals[2].sqrt();
        if candidate_norm == 0.0 || muon_norm == 0.0 {
            return Err("optimizer alignment requires two nonzero updates".into());
        }
        Ok(Alignment {
            schema: "baux.update-alignment.v1",
            reference: "instantaneous-muon-without-momentum-history",
            cosine: (totals[0] / (candidate_norm * muon_norm)).clamp(-1.0, 1.0),
            candidate_norm,
            muon_norm,
            incumbent_nll: 0.0,
            candidate_nll: 0.0,
            muon_nll: None,
            candidate_nll_change: 0.0,
            muon_nll_change: None,
            backward_ms: 0.0,
            update_ms: 0.0,
            reduction_ms: seconds * 1000.0,
            diagnostic_ms: 0.0,
        })
    }
}

fn check_half(name: &str, buffer: &BufferRef, elements: usize) -> Result<f64, String> {
    if elements * size_of::<u16>() > buffer.length() as usize {
        return Err(format!("{name} diagnostic extent exceeds its buffer"));
    }
    let values = unsafe { std::slice::from_raw_parts(buffer.contents().cast::<u16>(), elements) };
    let invalid = values
        .iter()
        .filter(|value| **value & 0x7c00 == 0x7c00)
        .count();
    if invalid != 0 {
        return Err(format!("{name} contains {invalid} non-finite FP16 values"));
    }
    Ok(values
        .iter()
        .map(|value| validation::decode_half(*value).abs())
        .fold(0.0f64, f64::max))
}
