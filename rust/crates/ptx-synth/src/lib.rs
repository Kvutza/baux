//! `ptx-synth`: Type-driven symbolic PTX code synthesizer for NVIDIA GPUs.
//!
//! Emits PTX recipes for hardware validation. Timing, occupancy, and physical
//! register use must be measured after assembly on the target device.

pub mod emit;
pub mod gemm;
pub mod hierarchy;
pub mod ir;
pub mod probe;
pub mod reg;

pub use emit::PtxEmitter;
pub use gemm::{GemmConfig, emit_gemm16, emit_gemm32};
pub use hierarchy::{
    HierarchicalAttentionConfig, HierarchicalAttentionOutput, build_leaf_summaries,
    fine_attention_reference, mass_preserving_hierarchical_attention,
};
pub use ir::{PtxKernelBuilder, RegAlloc, TargetArch};
pub use probe::{ProbeOp, ProbeShape, emit_probe, emit_rademacher, emit_scale};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scale_emit() {
        for target in [TargetArch::Sm75, TargetArch::Sm80, TargetArch::Sm90a] {
            let ptx = emit_scale("vector_scale_test", target);
            assert!(ptx.contains(&format!(".target {}", target.ptx_target())));
            assert!(ptx.contains("ld.global.nc.v4.f32"));
            assert!(ptx.contains("mul.f32"));
            assert!(ptx.contains("st.global.v4.f32"));
            assert!(ptx.contains(".maxnreg 32"));
        }
    }

    #[test]
    fn gemm_emit() {
        for target in [TargetArch::Sm75, TargetArch::Sm80, TargetArch::Sm90a] {
            let config = GemmConfig {
                target,
                ..GemmConfig::default()
            };
            let ptx = emit_gemm16(&config);
            assert!(ptx.contains(&format!(".target {}", target.ptx_target())));
            assert!(ptx.contains("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32"));
            assert!(ptx.contains(".reqntid 256, 1, 1"));
        }
    }

    #[test]
    fn exact_targets() {
        assert_eq!(TargetArch::from_compute(7, 5), Some(TargetArch::Sm75));
        assert_eq!(TargetArch::from_compute(8, 0), Some(TargetArch::Sm80));
        assert_eq!(TargetArch::from_compute(9, 0), Some(TargetArch::Sm90a));
        assert_eq!(TargetArch::from_compute(8, 9), None);
        assert_eq!(TargetArch::Sm90a.ptx_target(), "sm_90a");
    }

    #[test]
    fn rademacher_emit() {
        for target in [TargetArch::Sm75, TargetArch::Sm80, TargetArch::Sm90a] {
            let ptx = emit_rademacher("rademacher_test", target);
            assert!(ptx.contains(&format!(".target {}", target.ptx_target())));
            assert!(ptx.contains("ld.global.nc.v4.f32"));
            assert!(ptx.contains("st.global.v4.f32"));
            assert!(ptx.contains("selp.f32"));
        }
    }
}
