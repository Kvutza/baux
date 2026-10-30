//! Shared mHC-4 ABI. Imported by the CUDA host and Metal
//! dispatcher; shaders use the same repr(C) layouts and canonical row-major F16
//! weights. Fine encoding rounds after its ordered F32 FMA sum. Decoder base
//! and prefix remain F32; queries round to F16 before projection; logits and
//! full-vocabulary log-sum-exp remain F32. Each block has 64 causal
//! dependencies.

use baux::semantics::abi::{metal_field, metal_kind, record};
#[path = "mhc4/objective.rs"]
mod objective;
pub use objective::{LossEstimate, ObjectiveStats, PairShape};
#[path = "mhc4/causal.rs"]
mod causal;
#[path = "mhc4/initialization.rs"]
mod initialization;
#[path = "mhc4/layout.rs"]
pub mod layout;
#[path = "mhc4/probe.rs"]
pub mod probe;
pub use causal::CausalPlan;
pub use initialization::{
    Initialization, ScaleShape, columns, expert_multiplier, family_multiplier,
};
pub use layout::{LAYOUT, LayerVisit, MHC4_VISITS, Mhc4Layout, Slot};

pub const PATCH: usize = baux::semantics::shape::MHC4_SHAPE.patch as usize;
pub const QUERY: usize = baux::semantics::shape::MHC4_SHAPE.query as usize;
pub const VOCABULARY: usize = baux::semantics::shape::MHC4_VOCABULARY as usize;
pub const SHARD: usize = 128;
pub const SHARDS: usize = VOCABULARY / SHARD;
pub const RADIX: usize = 8;
pub const RADIX_LEVELS: usize = 6;
pub const HIERARCHY_NODES: usize = 8 + 64 + 512 + 4_096 + 32_768;
pub const WIDTH: usize = baux::semantics::shape::MHC4_SHAPE.width as usize;
pub const TARGET_PROMPT: usize = 524_288;
pub const TARGET_GENERATED: usize = 524_288;

record! {
    EncodeShape {
        rows: u32,
        width: u32,
        vocabulary: u32,
        patch: u32,
    }
}

record! {
    WaveShape {
        blocks: u32,
        tokens: u32,
        proposal_start: u32,
        intra: u32,
        vocabulary: u32,
        shards: u32,
        temperature: f32,
        score: u32,
    }
}

// One bounded vocabulary shard; the token is an integer, never a float cast.
record! {
    ShardStats {
        maximum: f32,
        sum: f32,
        target: f32,
        best: f32,
        token: u32,
        reserved: u32,
    }
}

/// Preamble compiled with the consuming Metal kernel, including layout checks.
pub fn metal_source(body: &str) -> String {
    let mut source = String::from("#include <metal_stdlib>\nusing namespace metal;\n");
    source.push_str(&EncodeShape::metal_decl());
    source.push_str(&WaveShape::metal_decl());
    source.push_str(&ShardStats::metal_decl());
    source.push_str(&PairShape::metal_decl());
    source.push_str(&ObjectiveStats::metal_decl());
    source.push_str(body);
    source
}

pub use baux::semantics::random::splitmix64 as splitmix;

pub fn sampling_bits(seed: u64, generated_position: u32, token: u32) -> u64 {
    splitmix(seed ^ (u64::from(generated_position) << 32) ^ u64::from(token))
}

pub fn uniform(bits: u64) -> f32 {
    ((bits >> 41) as u32 as f32 + 0.5) / 8_388_608.0
}

pub fn shape(
    prompt: usize,
    generated: usize,
    temperature: f32,
    score: bool,
) -> Result<WaveShape, String> {
    if prompt < PATCH
        || prompt % PATCH != 0
        || generated == 0
        || !temperature.is_finite()
        || temperature < 0.0
    {
        return Err(
            "mHC-4 requires an aligned nonempty prompt, nonempty output, and finite nonnegative \
             temperature"
                .into(),
        );
    }
    Ok(WaveShape {
        blocks: u32::try_from(generated.div_ceil(PATCH)).map_err(|error| error.to_string())?,
        tokens: u32::try_from(generated).map_err(|error| error.to_string())?,
        proposal_start: u32::try_from(prompt / PATCH - 1).map_err(|error| error.to_string())?,
        intra: 0,
        vocabulary: VOCABULARY as u32,
        shards: SHARDS as u32,
        temperature,
        score: u32::from(score),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wave_layout() {
        assert_eq!(size_of::<WaveShape>(), 32);
        assert_eq!(align_of::<WaveShape>(), 4);
        assert_eq!(core::mem::offset_of!(WaveShape, vocabulary), 16);
        assert_eq!(core::mem::offset_of!(WaveShape, temperature), 24);
        assert_eq!(core::mem::offset_of!(WaveShape, score), 28);
        assert_eq!(size_of::<ShardStats>(), 24);
        assert_eq!(core::mem::offset_of!(ShardStats, token), 16);
        let source = metal_source("");
        assert_eq!(source.matches("struct WaveShape").count(), 1);
        assert_eq!(source.matches("struct ShardStats").count(), 1);
        assert!(source.contains("__builtin_offsetof(WaveShape, temperature) == 24"));
    }

    #[test]
    fn invalid_shape() {
        assert!(shape(0, 64, 1.0, true).is_err());
        assert!(shape(65, 64, 1.0, true).is_err());
        assert!(shape(64, 0, 1.0, true).is_err());
        assert!(shape(64, 64, f32::NAN, true).is_err());
        assert!(shape(64, 64, -1.0, true).is_err());
        assert!(shape(64, u32::MAX as usize + 1, 1.0, true).is_err());
        let wave = shape(128, 65, 0.0, true).unwrap();
        assert_eq!(wave.blocks, 2);
        assert_eq!(wave.tokens, 65);
        assert_eq!(wave.proposal_start, 1);
        assert_eq!(wave.vocabulary, VOCABULARY as u32);
    }
}
