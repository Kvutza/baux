//! Declare scalar Rust/CUDA records and their Metal source from one field list.
//!
//! Shared across host, resident Apple Metal, and `cuda-oxide` device kernels
//! for tiled arbitrary-`num_candidates` optimization without fixed 4-slot caps.

#[doc(hidden)]
#[macro_export]
macro_rules! metal_kind {
    (u64) => {
        "ulong"
    };
    (i64) => {
        "long"
    };
    (u32) => {
        "uint"
    };
    (i32) => {
        "int"
    };
    (u16) => {
        "ushort"
    };
    (f32) => {
        "float"
    };
}

#[doc(hidden)]
#[macro_export]
macro_rules! metal_field {
    ($field:ident, $kind:ident) => {
        concat!(
            "    ",
            $crate::semantics::abi::metal_kind!($kind),
            " ",
            stringify!($field),
            ";\n"
        )
    };
    ($field:ident,[$kind:ident; $count:literal]) => {
        concat!(
            "    ",
            $crate::semantics::abi::metal_kind!($kind),
            " ",
            stringify!($field),
            "[",
            stringify!($count),
            "];\n"
        )
    };
}

#[doc(hidden)]
#[macro_export]
macro_rules! abi_record {
    ($name:ident { $($field:ident: $kind:tt),* $(,)? }) => {
        #[repr(C)]
        #[derive(Clone, Copy, Debug, Default, PartialEq)]
        pub struct $name { $(pub $field: $kind),* }

        // SAFETY: only repr(C) DeviceCopy scalars and fixed arrays are accepted.
        #[cfg(feature = "cuda")]
        unsafe impl cuda_core::DeviceCopy for $name {}

        impl $name {
            /// Metal checks its actual layout against the compiling Rust target.
            #[cfg(not(target_os = "cuda"))]
            pub fn metal_decl() -> String {
                let mut source = String::from(concat!(
                    "struct ", stringify!($name), " {\n",
                    $($crate::semantics::abi::metal_field!($field, $kind),)*
                    "};\n"
                ));
                source.push_str(&format!(
                    "static_assert(sizeof({}) == {}, \"{} size\");\n",
                    stringify!($name), core::mem::size_of::<Self>(), stringify!($name)
                ));
                $(source.push_str(&format!(
                    "static_assert(__builtin_offsetof({}, {}) == {}, \"{} offset\");\n",
                    stringify!($name), stringify!($field),
                    core::mem::offset_of!(Self, $field), stringify!($field)
                ));)*
                source
            }
        }
    };
}

pub use crate::{abi_record as record, metal_field, metal_kind};

/// Sentinel indicating normal acquisition ranking (no forced candidate index).
pub const NO_FORCED: u32 = u32::MAX;

/// Canonical UCB exploration weight (`beta = 1.0`).
pub const DEFAULT_BETA: f32 = 1.0;

/// Perturbation / pool generation mode discriminants shared with GPU shaders.
pub const MODE_GAUSSIAN: u32 = 0;
pub const MODE_RADEMACHER: u32 = 1;
pub const MODE_UNIFORM: u32 = 2;
pub const MODE_RAASP: u32 = 3;
pub const MODE_SOBOL: u32 = 4;
pub const MODE_LHD: u32 = 5;
pub const MODE_MASKED: u32 = 6;

/// Acquisition rule discriminants shared with GPU shaders.
pub const ACQ_UCB: u32 = 0;
pub const ACQ_THOMPSON: u32 = 1;
pub const ACQ_PARETO: u32 = 2;
pub const ACQ_RANDOM: u32 = 3;

/// Pairwise dissimilarity discriminants shared with GPU shaders.
pub const METRIC_EUCLIDEAN: u32 = 0;
pub const METRIC_CONFORMAL: u32 = 1;
pub const METRIC_HYPERBOLIC: u32 = 2;
pub const METRIC_WASSERSTEIN: u32 = 3;

/// Posterior uncertainty law discriminants shared with GPU shaders.
pub const LAW_PRECISION: u32 = 0;
pub const LAW_BIASVAR: u32 = 1;
pub const LAW_GEOMETRIC: u32 = 2;

record! {
    TileSpan {
        leaf: u32,
        start: u32,
        length: u32,
        pad: u32,
    }
}

record! {
    PartialWord {
        anchor: f32,
        rejected: f32,
        squared: f32,
        changed: u32,
        invalid: u32,
    }
}

record! {
    TilePartial {
        anchor: f32,
        rejected: f32,
        squared: f32,
        changed: u32,
        invalid: u32,
        pad: u32,
    }
}

record! {
    PoolDispatch {
        key_hi: u64,
        key_lo: u64,
        root_seed: u64,
        stream_seed: u64,
        radius: f32,
        prob_pert: f32,
        num_candidates: u32,
        num_arms: u32,
        num_dim: u32,
        num_pert: u32,
        base_idx: u32,
        tiles: u32,
        history: u32,
        mode: u32,
        base_slot: u32,
        blocks: u32,
        initialize: u32,
        scramble: u32,
    }
}

record! {
    SelectHeader {
        root_seed: u64,
        acq_seed: u64,
        epistemic_scale: f32,
        aleatoric_scale: f32,
        y_scale: f32,
        beta: f32,
        radius: f32,
        axis_base: f32,
        axis_weight: f32,
        morbo_alpha: f32,
        num_candidates: u32,
        num_arms: u32,
        neighbors: u32,
        history: u32,
        acquisition: u32,
        tiles: u32,
        mode: u32,
        resident_history: u32,
        implicit_history: u32,
        forced_candidate: u32,
        distance_metric: u32,
        scale_neighbors: u32,
        incumbent_index: u32,
        candidate_floor: u32,
        adaptive_k: u32,
        uncertainty_law: u32,
        axis_enabled: u32,
        num_metrics: u32,
        has_yvar: u32,
        has_ybounds: u32,
    }
}

record! {
    ArmDecision {
        root_seed: u64,
        seed: u64,
        index: u32,
        arm: u32,
        valid: u32,
        mode: u32,
        radius: f32,
        score: f32,
        pred_mean: f32,
        pred_se: f32,
        pred_epi: f32,
        pred_ale: f32,
        incumbent_mean: f32,
        incumbent_se: f32,
        neighbors: u32,
        incumbent_k: u32,
    }
}

/// Emit shared Metal constants, discriminants, and `static_assert`-checked
/// tiled selection records (with zero fixed `[f32; 4]` arrays).
#[cfg(not(target_os = "cuda"))]
pub fn metal_abi() -> String {
    let mut out = String::new();
    for (name, val) in [
        ("NO_FORCED", NO_FORCED),
        ("MODE_GAUSSIAN", MODE_GAUSSIAN),
        ("MODE_RADEMACHER", MODE_RADEMACHER),
        ("MODE_UNIFORM", MODE_UNIFORM),
        ("MODE_RAASP", MODE_RAASP),
        ("MODE_SOBOL", MODE_SOBOL),
        ("MODE_LHD", MODE_LHD),
        ("MODE_MASKED", MODE_MASKED),
        ("ACQ_UCB", ACQ_UCB),
        ("ACQ_THOMPSON", ACQ_THOMPSON),
        ("ACQ_PARETO", ACQ_PARETO),
        ("ACQ_RANDOM", ACQ_RANDOM),
        ("METRIC_EUCLIDEAN", METRIC_EUCLIDEAN),
        ("METRIC_CONFORMAL", METRIC_CONFORMAL),
        ("METRIC_HYPERBOLIC", METRIC_HYPERBOLIC),
        ("METRIC_WASSERSTEIN", METRIC_WASSERSTEIN),
        ("LAW_PRECISION", LAW_PRECISION),
        ("LAW_BIASVAR", LAW_BIASVAR),
        ("LAW_GEOMETRIC", LAW_GEOMETRIC),
    ] {
        out.push_str(&format!("constant uint {name} = {val}u;\n"));
    }
    out.push_str(&format!(
        "constant float DEFAULT_BETA = {DEFAULT_BETA:?}f;\n"
    ));
    out.push_str(&TileSpan::metal_decl());
    out.push_str(&PartialWord::metal_decl());
    out.push_str(&TilePartial::metal_decl());
    out.push_str(&PoolDispatch::metal_decl());
    out.push_str(&SelectHeader::metal_decl());
    out.push_str(&ArmDecision::metal_decl());
    out
}

/// Prepend `<metal_stdlib>` and [`metal_abi`] to a Metal kernel source body.
#[cfg(not(target_os = "cuda"))]
pub fn metal_source(body: &str) -> String {
    let mut source = String::from("#include <metal_stdlib>\nusing namespace metal;\n");
    source.push_str(&metal_abi());
    source.push_str(body);
    source
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn abi_layouts() {
        assert_eq!(core::mem::size_of::<TileSpan>(), 16);
        assert_eq!(core::mem::size_of::<PartialWord>(), 20);
        assert_eq!(core::mem::size_of::<TilePartial>(), 24);
        assert_eq!(core::mem::size_of::<PoolDispatch>(), 88);
        assert_eq!(core::mem::align_of::<PoolDispatch>(), 8);
        assert_eq!(core::mem::size_of::<SelectHeader>(), 128);
        assert_eq!(core::mem::align_of::<SelectHeader>(), 8);
        assert_eq!(core::mem::size_of::<ArmDecision>(), 72);
        assert_eq!(core::mem::align_of::<ArmDecision>(), 8);
        let msl = metal_abi();
        assert!(msl.contains("struct PoolDispatch"));
        assert!(msl.contains("struct SelectHeader"));
        assert!(msl.contains("struct ArmDecision"));
        assert!(!msl.contains("[4]"));
    }
}
