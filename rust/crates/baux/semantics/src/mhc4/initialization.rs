//! Host-side mHC-4 initialization and dimensionless perturbation scales.
//! These are construction rules, not statistical guarantees about a network.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScaleShape {
    Scalar,
    TensorFamilyStatic,
}

impl ScaleShape {
    pub fn parse(name: &str) -> Result<Self, String> {
        match name {
            "scalar" => Ok(Self::Scalar),
            "tensor-family-static" => Ok(Self::TensorFamilyStatic),
            _ => Err(format!("unsupported mHC-4 trust-region shape {name}")),
        }
    }
    pub fn multiplier(self, family: &str) -> f32 {
        match self {
            Self::Scalar => 1.0,
            Self::TensorFamilyStatic => family_multiplier(family),
        }
    }
}

pub fn expert_multiplier(family: &str, expert: Option<usize>) -> f32 {
    if matches!(
        family,
        "expert_gate_up"
            | "expert_down"
            | "qkv"
            | "attention_output"
            | "embedding_readout"
            | "mask_embed"
            | "index_query"
            | "router"
            | "feedback_state"
            | "feedback_gate"
            | "mhc_attention_predictor"
            | "mhc_moe_predictor"
            | "mhc_attention_bias"
            | "mhc_attention_control"
            | "mhc_moe_bias"
            | "mhc_moe_control"
            | "attention_norm"
            | "ffn_norm"
            | "final_norm"
    ) {
        let tree = baux::semantics::optimizer::ParamTree::<usize>::active_multipliers();
        let mapped = baux::semantics::optimizer::ParamFamily::from_expert(family, expert);
        *tree.get(mapped)
    } else {
        1.0
    }
}

pub fn family_multiplier(family: &str) -> f32 {
    expert_multiplier(family, None)
}

/// Fill complete row-major matrices. Draw order and FP64 column normalization
/// precede one caller-supplied FP16 rounding step. Keeping allocation and RNG
/// ownership with the caller permits both packed Metal and per-layer CUDA
/// views.
pub fn columns(
    values: &mut [u16],
    input: usize,
    output: usize,
    column_norm: f32,
    mut normal: impl FnMut() -> f32,
    round: impl Fn(f32) -> u16,
) -> Result<(), String> {
    let size = input
        .checked_mul(output)
        .filter(|&v| v != 0)
        .ok_or("invalid column shape")?;
    if values.is_empty()
        || values.len() % size != 0
        || !column_norm.is_finite()
        || column_norm < 0.0
    {
        return Err("invalid column initialization".into());
    }
    let mut matrix = vec![0.0_f32; size];
    let mut norms = vec![0.0_f64; output];
    for values in values.chunks_exact_mut(size) {
        norms.fill(0.0);
        for row in 0..input {
            for column in 0..output {
                let sample = normal();
                if !sample.is_finite() {
                    return Err("nonfinite initialization draw".into());
                }
                matrix[row * output + column] = sample;
                norms[column] += f64::from(sample) * f64::from(sample);
            }
        }
        for norm in &mut norms {
            if *norm == 0.0 {
                return Err("zero initialization column".into());
            }
            *norm = f64::from(column_norm) / norm.sqrt();
        }
        for row in 0..input {
            let base = row * output;
            for column in 0..output {
                values[base + column] =
                    round((f64::from(matrix[base + column]) * norms[column]) as f32);
            }
        }
    }
    Ok(())
}

#[derive(Debug, Clone, Copy)]
pub struct Initialization {
    pub residual_depth: f32,
    pub residual_gain: f32,
    pub router_column_norm: f32,
    pub predictor_control: f32,
    pub transport_lambda: f32,
    pub sites: usize,
}

impl Initialization {
    pub fn new() -> Self {
        let core = baux::semantics::recurrence::RecurrentCore::selective_fbt();
        let plan = core
            .layer_visits(5, core.training_visits)
            .expect("fixed mHC-4 recurrence");
        let mut visits = [0u32; 5];
        for step in &plan {
            visits[step.layer] += 1;
        }
        let residual_depth = (2 * visits.iter().map(|&n| n * n).sum::<u32>()) as f32;
        let sites = 2 * plan.len();
        Self {
            residual_depth,
            residual_gain: residual_depth.sqrt().recip(),
            router_column_norm: (2.0 * 625.0_f32.ln()).sqrt().recip(),
            predictor_control: 2048.0_f32.sqrt().recip(),
            transport_lambda: 1.0 / (8 * sites) as f32,
            sites,
        }
    }

    pub fn down_norm(self) -> f32 {
        self.residual_gain / 0.355_775_5_f32.sqrt()
    }

    pub fn proposal_scale(self, family: &str) -> Result<f32, String> {
        let width = 512.0_f32.sqrt();
        Ok(match family {
            "router" => self.router_column_norm / width,
            "qkv" | "expert_gate_up" | "embedding_readout" | "feedback_state" | "feedback_gate"
            | "mask_embed" => width.recip(),
            "attention_output" => self.residual_gain / width,
            "expert_down" => self.down_norm() / 216.0_f32.sqrt(),
            "mhc_attention_predictor" | "mhc_moe_predictor" => 2048.0_f32.sqrt().recip(),
            "mhc_attention_bias" | "mhc_attention_control" | "mhc_moe_bias" | "mhc_moe_control" => {
                self.predictor_control
            }
            "attention_norm" | "ffn_norm" | "final_norm" => width.recip(),
            "index_query" => 64.0_f32.sqrt().recip(),
            _ => return Err(format!("unmapped mHC-4 initialization family {family}")),
        })
    }
}

impl Default for Initialization {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn column_order() {
        let mut whole = [0; 16];
        let mut split = [0; 16];
        let mut draw = 0;
        columns(
            &mut whole,
            2,
            2,
            1.0,
            || {
                draw += 1;
                draw as f32
            },
            |v| (v * 1000.0) as u16,
        )
        .unwrap();
        assert_eq!(&whole[..4], &[316, 447, 948, 894]);
        assert_eq!(draw, 16);
        draw = 0;
        for matrix in split.chunks_exact_mut(4) {
            columns(
                matrix,
                2,
                2,
                1.0,
                || {
                    draw += 1;
                    draw as f32
                },
                |v| (v * 1000.0) as u16,
            )
            .unwrap();
        }
        assert_eq!(whole, split);
        assert!(columns(&mut split, 3, 2, 1.0, || 1.0, |_| 0).is_err());
        assert!(columns(&mut split, 2, 2, 1.0, || 0.0, |_| 0).is_err());
        assert!(columns(&mut split, 2, 2, 1.0, || f32::NAN, |_| 0).is_err());
    }
    #[test]
    fn scale_inventory() {
        let policy = Initialization::new();
        assert_eq!(policy.residual_depth, 22.0);
        assert_eq!(policy.sites, 14);
        assert_eq!(policy.transport_lambda, 1.0 / 112.0);
        for name in ["attention_norm", "ffn_norm", "final_norm"] {
            assert_eq!(
                policy.proposal_scale(name).unwrap().to_bits(),
                512.0_f32.sqrt().recip().to_bits()
            );
        }
        assert!(policy.proposal_scale("unknown").is_err());
    }
}
