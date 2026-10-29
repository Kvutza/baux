//! One full-size initialized mHC-4 forward case, not a pretraining experiment.
use super::super::{VOCABULARY, WIDTH};

pub const SEED: u64 = 17;
pub const PATCH: usize = 1;
pub const ROWS: usize = 128;
pub const TOKENS: usize = 128;
pub const WORDS: usize = ROWS * WIDTH;

pub fn sequence() -> Vec<u32> {
    (0..=TOKENS)
        .map(|i| (baux::semantics::random::splitmix64(73 + i as u64) % VOCABULARY as u64) as u32)
        .collect()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ParameterFamily {
    pub name: &'static str,
    pub elements: u64,
}

/// Logical families in Metal's packed parameter order. CUDA's tied embedding
/// is a second physical view of embedding_readout, not another logical family.
pub const FAMILIES: [ParameterFamily; 15] = {
    use baux::semantics::shape as d;
    [
        ParameterFamily {
            name: "router",
            elements: d::MHC4_LAYERS * d::MHC4_WIDTH * d::ROUTED_COUNT,
        },
        ParameterFamily {
            name: "qkv",
            elements: d::MHC4_LAYERS * d::MHC4_WIDTH * d::QKV_WIDTH,
        },
        ParameterFamily {
            name: "attention_output",
            elements: d::MHC4_LAYERS * d::MHC4_WIDTH * d::MHC4_WIDTH,
        },
        ParameterFamily {
            name: "expert_gate_up",
            elements: d::MHC4_LAYERS * d::EXPERT_COUNT * d::MHC4_WIDTH * d::GATE_WIDTH,
        },
        ParameterFamily {
            name: "expert_down",
            elements: d::MHC4_LAYERS * d::EXPERT_COUNT * d::EXPERT_WIDTH * d::MHC4_WIDTH,
        },
        ParameterFamily {
            name: "attention_norm",
            elements: d::MHC4_LAYERS * d::MHC4_WIDTH,
        },
        ParameterFamily {
            name: "ffn_norm",
            elements: d::MHC4_LAYERS * d::MHC4_WIDTH,
        },
        ParameterFamily {
            name: "embedding_readout",
            elements: d::MHC4_WIDTH * VOCABULARY as u64,
        },
        ParameterFamily {
            name: "mhc_attention_predictor",
            elements: d::MHC4_LAYERS * d::MHC_INPUT * d::MHC_COEFS,
        },
        ParameterFamily {
            name: "mhc_attention_bias",
            elements: d::MHC4_LAYERS * d::MHC_COEFS,
        },
        ParameterFamily {
            name: "mhc_attention_control",
            elements: d::MHC4_LAYERS * 4,
        },
        ParameterFamily {
            name: "mhc_moe_predictor",
            elements: d::MHC4_LAYERS * d::MHC_INPUT * d::MHC_COEFS,
        },
        ParameterFamily {
            name: "mhc_moe_bias",
            elements: d::MHC4_LAYERS * d::MHC_COEFS,
        },
        ParameterFamily {
            name: "mhc_moe_control",
            elements: d::MHC4_LAYERS * 4,
        },
        ParameterFamily {
            name: "final_norm",
            elements: d::MHC4_WIDTH,
        },
    ]
};

pub const TARGETS: usize = TOKENS;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn families_params() {
        let total: u64 = FAMILIES.iter().map(|family| family.elements).sum();
        assert_eq!(total, baux::semantics::shape::MHC4_PARAMS);
        assert_eq!(TARGETS, 128);
    }
}
