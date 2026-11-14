use super::*;
use deser::Serialize;
use std::io::{BufWriter, Write};
use std::path::Path;

pub(super) fn checkpoint_layout(
    architecture: ResidualArchitecture,
) -> Vec<(&'static str, Vec<usize>)> {
    let layers = MODEL_LAYERS as usize;
    let width = WIDTH as usize;
    let experts = routing::MOE_EXPERTS as usize;
    let mut layout = vec![
        (
            "router",
            vec![layers, width, routing::ROUTED_EXPERTS as usize],
        ),
        ("qkv", vec![layers, width, QKV_WIDTH as usize]),
        ("attention_output", vec![layers, width, width]),
        (
            "expert_gate_up",
            vec![layers, experts, width, routing::MOE_GATEUP as usize],
        ),
        (
            "expert_down",
            vec![layers, experts, routing::ROUTED_WIDTH as usize, width],
        ),
        ("attention_norm", vec![layers, width]),
        ("ffn_norm", vec![layers, width]),
        (
            "embedding_readout",
            vec![width, architecture.vocabulary() as usize],
        ),
    ];
    if architecture.is_multistream() {
        layout.extend([
            (
                "mhc_attention_predictor",
                vec![layers, MHC_INPUT as usize, MHC_COEFFICIENTS as usize],
            ),
            (
                "mhc_attention_bias",
                vec![layers, MHC_COEFFICIENTS as usize],
            ),
            ("mhc_attention_control", vec![layers, 4]),
            (
                "mhc_moe_predictor",
                vec![layers, MHC_INPUT as usize, MHC_COEFFICIENTS as usize],
            ),
            ("mhc_moe_bias", vec![layers, MHC_COEFFICIENTS as usize]),
            ("mhc_moe_control", vec![layers, 4]),
        ]);
    } else {
        layout.extend([
            ("feedback_state", vec![width, width]),
            ("feedback_gate", vec![width, width]),
        ]);
    }
    layout.push(("final_norm", vec![width]));
    if architecture == ResidualArchitecture::DiffusionMhc4 {
        layout.push(("mask_embed", vec![width]));
        layout.push(("index_query", vec![64, 64]));
    }
    if architecture.is_hnet() {
        layout.push(("hnet", vec![3 * width * width + width]));
    }
    layout
}

pub(super) fn load_checkpoint(weights: &CandidateWeights, path: &Path) -> Result<(), String> {
    let file = std::fs::File::open(path).map_err(|e| e.to_string())?;
    if file.metadata().map_err(|e| e.to_string())?.len()
        == weights.architecture.parameter_count() as u64 * 2
    {
        return Err(
            "raw v1 checkpoint has no positional-encoding metadata; use a RoPE v2 Safetensors \
             checkpoint"
                .into(),
        );
    }
    drop(file);
    let checkpoint = crate::tensor_store::SafeTensors::open(path)?;
    let metadata = checkpoint.metadata();
    let (legacy_format, canonical_format) = match weights.architecture {
        ResidualArchitecture::Legacy
        | ResidualArchitecture::Residual1
        | ResidualArchitecture::ProjectedBoundary => {
            ("baux.fbt-pisa1-rope.v2", "baux.pisa-rope.v2")
        }
        ResidualArchitecture::Hc4 => ("baux.fbt-pisa1-hc4-rope.v1", "baux.pisa-hc4-rope.v1"),
        ResidualArchitecture::Mhc4 => ("baux.fbt-pisa1-mhc4-rope.v1", "baux.pisa-mhc4-rope.v1"),
        ResidualArchitecture::HnetMhc4 => (
            "baux.fbt-pisa1-hnet-bytes-rope.v2",
            "baux.pisa-hnet-bytes-rope.v2",
        ),
        ResidualArchitecture::LoopedMhc4 => (
            "baux.fbt-pisa1-looped-mhc4-rope.v1",
            "baux.pisa-looped-mhc4-rope.v1",
        ),
        ResidualArchitecture::DiffusionMhc4 => (
            "baux.fbt-pisa1-diffusion-mhc4-rope.v1",
            "baux.pisa-diffusion-mhc4-rope.v1",
        ),
    };
    let meta_format = metadata.get("format").map(String::as_str);
    if (meta_format != Some(legacy_format) && meta_format != Some(canonical_format))
        || metadata.get("position_encoding").map(String::as_str) != Some("rope")
        || metadata.get("rope_base").map(String::as_str) != Some("10000")
        || metadata.get("rope_dimensions").map(String::as_str) != Some("64")
    {
        return Err("checkpoint architecture is not BAUX PISA RoPE".into());
    }
    let layout = checkpoint_layout(weights.architecture);
    for (((name, buffer, _), (expected_name, expected_shape)), index) in
        weights.tensors().into_iter().zip(&layout).zip(0usize..)
    {
        if name != *expected_name {
            return Err(format!("checkpoint layout mismatch at tensor {index}"));
        }
        let (tensor, bytes) = checkpoint
            .tensor(name)
            .ok_or_else(|| format!("checkpoint is missing tensor {name:?}"))?;
        if tensor.dtype != crate::tensor_store::DType::F16
            || tensor.shape != *expected_shape
            || bytes.len() != buffer.length() as usize
        {
            return Err(format!(
                "checkpoint tensor {name:?} has the wrong dtype or shape"
            ));
        }
        unsafe {
            std::ptr::copy_nonoverlapping(
                bytes.as_ptr(),
                buffer.contents().cast::<u8>(),
                bytes.len(),
            );
        }
    }
    Ok(())
}

pub(super) fn save_checkpoint(
    row: &BufferRef,
    path: &Path,
    architecture: ResidualArchitecture,
) -> Result<(), String> {
    let bytes =
        unsafe { std::slice::from_raw_parts(row.contents().cast::<u8>(), row.length() as usize) };
    let layout = checkpoint_layout(architecture);
    let mut offset = 0usize;
    let mut tensors = Vec::with_capacity(layout.len());
    for (name, shape) in &layout {
        let length = shape
            .iter()
            .try_fold(2usize, |bytes, &dimension| bytes.checked_mul(dimension))
            .ok_or("checkpoint tensor length overflow")?;
        let end = offset
            .checked_add(length)
            .ok_or("checkpoint tensor range overflow")?;
        if end > bytes.len() {
            return Err(format!(
                "checkpoint tensor {name:?} exceeds the candidate row"
            ));
        }
        tensors.push(crate::tensor_store::TensorData {
            name,
            dtype: crate::tensor_store::DType::F16,
            shape,
            bytes: &bytes[offset..end],
        });
        offset = end;
    }
    if offset != bytes.len() {
        return Err("checkpoint layout does not cover the candidate row".into());
    }
    let format = match architecture {
        ResidualArchitecture::Legacy
        | ResidualArchitecture::Residual1
        | ResidualArchitecture::ProjectedBoundary => "baux.pisa-rope.v2",
        ResidualArchitecture::Hc4 => "baux.pisa-hc4-rope.v1",
        ResidualArchitecture::Mhc4 => "baux.pisa-mhc4-rope.v1",
        ResidualArchitecture::HnetMhc4 => "baux.pisa-hnet-bytes-rope.v2",
        ResidualArchitecture::LoopedMhc4 => "baux.pisa-looped-mhc4-rope.v1",
        ResidualArchitecture::DiffusionMhc4 => "baux.pisa-diffusion-mhc4-rope.v1",
    };
    crate::tensor_store::write_safetensors(
        path,
        std::collections::BTreeMap::from([
            ("format".into(), format.into()),
            (
                "parameters".into(),
                architecture.parameter_count().to_string(),
            ),
            ("position_encoding".into(), "rope".into()),
            ("rope_base".into(), "10000".into()),
            ("rope_dimensions".into(), HEAD_DIM.to_string()),
        ]),
        &tensors,
    )
}

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
        | "index_query" | "hnet" => (None, None),
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
