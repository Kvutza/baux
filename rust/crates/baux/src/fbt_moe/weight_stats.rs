use super::*;
use deser::Serialize;
use std::io::{BufWriter, Write};
use std::path::Path;

#[derive(Default)]
struct Moments {
    left_sum: f64,
    right_sum: f64,
    left_sq: f64,
    right_sq: f64,
    delta_sum: f64,
    delta_sq: f64,
    dot: f64,
    changed: usize,
    left_min: f32,
    left_max: f32,
    right_min: f32,
    right_max: f32,
}

impl Moments {
    fn new() -> Self {
        Self {
            left_min: f32::INFINITY,
            left_max: f32::NEG_INFINITY,
            right_min: f32::INFINITY,
            right_max: f32::NEG_INFINITY,
            ..Self::default()
        }
    }

    fn push(&mut self, left: f32, right: f32) {
        let delta = right - left;
        let left = f64::from(left);
        let right = f64::from(right);
        let delta = f64::from(delta);
        self.left_sum += left;
        self.right_sum += right;
        self.left_sq += left * left;
        self.right_sq += right * right;
        self.delta_sum += delta;
        self.delta_sq += delta * delta;
        self.dot += left * right;
        self.changed += usize::from(left != right);
        self.left_min = self.left_min.min(left as f32);
        self.left_max = self.left_max.max(left as f32);
        self.right_min = self.right_min.min(right as f32);
        self.right_max = self.right_max.max(right as f32);
    }
}

#[derive(Serialize)]
struct Stats<'a> {
    schema: &'static str,
    tensor: String,
    family: &'a str,
    layer: Option<usize>,
    expert: Option<usize>,
    elements: usize,
    initial_mean: f64,
    final_mean: f64,
    initial_rms: f64,
    final_rms: f64,
    delta_mean: f64,
    delta_rms: f64,
    relative_delta_rms: Option<f64>,
    changed_fraction: f64,
    cosine: Option<f64>,
    initial_min: f32,
    initial_max: f32,
    final_min: f32,
    final_max: f32,
}

fn identity(family: &str, index: usize) -> (String, Option<usize>, Option<usize>) {
    let (layer, expert) = match family {
        "expert_gate_up" | "expert_down" => (
            Some(index / routing::MOE_EXPERTS as usize),
            Some(index % routing::MOE_EXPERTS as usize),
        ),
        "embedding_readout" | "feedback_state" | "feedback_gate" | "final_norm" | "mask_embed"
        | "index_query" => (None, None),
        _ => (Some(index), None),
    };
    let name = match (layer, expert) {
        (Some(layer), Some(expert)) => format!("layer.{layer}.expert.{expert}.{family}"),
        (Some(layer), None) => format!("layer.{layer}.{family}"),
        _ => family.to_string(),
    };
    (name, layer, expert)
}

fn measure<'a>(
    family: &'a str,
    index: usize,
    before: &[u16],
    after: &[u16],
) -> Result<Stats<'a>, String> {
    let mut moments = Moments::new();
    for (&left, &right) in before.iter().zip(after) {
        let left = decode_half(left) as f32;
        let right = decode_half(right) as f32;
        if !left.is_finite() || !right.is_finite() {
            return Err(format!("nonfinite weight in {family} tensor {index}"));
        }
        moments.push(left, right);
    }
    let count = before.len() as f64;
    let initial_rms = (moments.left_sq / count).sqrt();
    let final_rms = (moments.right_sq / count).sqrt();
    let delta_rms = (moments.delta_sq / count).sqrt();
    let (tensor, layer, expert) = identity(family, index);
    Ok(Stats {
        schema: "baux.tensor-state.v1",
        tensor,
        family,
        layer,
        expert,
        elements: before.len(),
        initial_mean: moments.left_sum / count,
        final_mean: moments.right_sum / count,
        initial_rms,
        final_rms,
        delta_mean: moments.delta_sum / count,
        delta_rms,
        relative_delta_rms: (initial_rms > 0.0).then_some(delta_rms / initial_rms),
        changed_fraction: moments.changed as f64 / count,
        cosine: (moments.left_sq > 0.0 && moments.right_sq > 0.0)
            .then_some(moments.dot / (moments.left_sq * moments.right_sq).sqrt()),
        initial_min: moments.left_min,
        initial_max: moments.left_max,
        final_min: moments.right_min,
        final_max: moments.right_max,
    })
}

impl CandidateWeights {
    /// Compare a trained state with its exact initialization. This diagnostic
    /// runs after optimization and is excluded from round latency.
    pub(super) fn write_stats(&self, trained: &BufferRef, path: &Path) -> Result<(), String> {
        let expected = self.architecture.parameter_count();
        if trained.length() < (expected * size_of::<u16>()) as u64 {
            return Err("trained weight buffer has the wrong size for tensor statistics".into());
        }
        let current =
            unsafe { std::slice::from_raw_parts(trained.contents().cast::<u16>(), expected) };
        let mut writer =
            BufWriter::new(std::fs::File::create(path).map_err(|error| error.to_string())?);
        let mut flat = 0usize;
        for (family, buffer, block_len) in self.tensors() {
            let initial = unsafe {
                std::slice::from_raw_parts(
                    buffer.contents().cast::<u16>(),
                    buffer.length() as usize / size_of::<u16>(),
                )
            };
            for (index, before) in initial.chunks_exact(block_len).enumerate() {
                let after = &current[flat..flat + before.len()];
                baux::artifact::json::to_writer(
                    &mut writer,
                    &measure(family, index, before, after)?,
                )
                .map_err(|error| error.to_string())?;
                writer.write_all(b"\n").map_err(|error| error.to_string())?;
                flat += before.len();
            }
        }
        if flat != expected {
            return Err("tensor statistics did not cover the complete model".into());
        }
        writer.flush().map_err(|error| error.to_string())
    }

    /// Populate final RMS and net squared displacement (`||theta_T -
    /// theta_0||_2^2`) in-place on [`UpdateLog`] after optimization
    /// completes.
    pub(super) fn finalize_updates(
        &self,
        trained: &BufferRef,
        updates: &mut UpdateLog,
    ) -> Result<(), String> {
        let expected = self.architecture.parameter_count();
        if trained.length() < (expected * size_of::<u16>()) as u64 {
            return Err("trained weight buffer has the wrong size for tensor statistics".into());
        }
        let current =
            unsafe { std::slice::from_raw_parts(trained.contents().cast::<u16>(), expected) };
        let mut flat = 0usize;
        let mut block_idx = 0usize;
        for (family, buffer, block_len) in self.tensors() {
            let initial = unsafe {
                std::slice::from_raw_parts(
                    buffer.contents().cast::<u16>(),
                    buffer.length() as usize / size_of::<u16>(),
                )
            };
            for (index, before) in initial.chunks_exact(block_len).enumerate() {
                let after = &current[flat..flat + before.len()];
                let mut right_sq = 0.0f64;
                let mut delta_sq = 0.0f64;
                for (&left, &right) in before.iter().zip(after) {
                    let left = f64::from(decode_half(left));
                    let right = f64::from(decode_half(right));
                    if !left.is_finite() || !right.is_finite() {
                        return Err(format!("nonfinite weight in {family} tensor {index}"));
                    }
                    let delta = right - left;
                    right_sq += right * right;
                    delta_sq += delta * delta;
                }
                if let Some(block) = updates.blocks.get_mut(block_idx) {
                    block.final_rms = (right_sq / before.len() as f64).sqrt() as f32;
                    block.net_sq_change = delta_sq;
                }
                flat += before.len();
                block_idx += 1;
            }
        }
        if flat != expected || block_idx != updates.blocks.len() {
            return Err("tensor statistics did not cover the complete model".into());
        }
        Ok(())
    }
}
