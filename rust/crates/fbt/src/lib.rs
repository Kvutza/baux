//! Forward-backward transformer (FBT), grouped-MoE, PISA-1 attention, and
//! subsecond generation/pretraining execution on Apple Silicon Metal.

#![allow(clippy::pedantic, clippy::nursery, clippy::cargo)]

#[cfg(test)]
pub(crate) use baux::TRLengthConfig;
pub use baux::semantics::context;
pub(crate) use baux::{
    BasePolicyManifest, ProposalKernel, config, forward_program, hash, muon, objective_observation,
    pretrain_data, procedural_pool, response_observation, search, tensor_store, text, traits,
    trust_region, weights,
};
#[cfg(all(any(target_os = "macos", target_os = "ios"), feature = "metal"))]
pub(crate) use baux::{apple_gpu, hnet_metal, mps as fbt_mps, reconstruction, resident_metal};

#[cfg(any(
    test,
    all(any(target_os = "macos", target_os = "ios"), feature = "metal")
))]
mod code_overlap;
#[cfg(all(any(target_os = "macos", target_os = "ios"), feature = "metal"))]
pub mod context_metal;
#[cfg(all(any(target_os = "macos", target_os = "ios"), feature = "metal"))]
pub mod micro;
#[cfg(all(any(target_os = "macos", target_os = "ios"), feature = "metal"))]
pub use micro as fbt_micro;
#[cfg(all(any(target_os = "macos", target_os = "ios"), feature = "metal"))]
#[allow(dead_code)]
pub mod moe;
#[cfg(all(any(target_os = "macos", target_os = "ios"), feature = "metal"))]
pub use moe as fbt_moe;
#[cfg(all(any(target_os = "macos", target_os = "ios"), feature = "metal"))]
#[allow(dead_code)]
pub mod nn;
#[cfg(all(any(target_os = "macos", target_os = "ios"), feature = "metal"))]
#[allow(dead_code)]
pub mod pisa1;
#[cfg(all(any(target_os = "macos", target_os = "ios"), feature = "metal"))]
pub use pisa1 as fbt_pisa1;
#[cfg(all(any(target_os = "macos", target_os = "ios"), feature = "metal"))]
#[allow(dead_code)]
pub mod tensor;

#[cfg(all(any(target_os = "macos", target_os = "ios"), feature = "metal"))]
pub use context_metal::ContextCache;
#[cfg(all(any(target_os = "macos", target_os = "ios"), feature = "metal"))]
pub use micro::{MicroMetrics, benchmark_micro};
#[cfg(all(any(target_os = "macos", target_os = "ios"), feature = "metal"))]
pub use moe::experiment::{moe_dataset, moe_probe, run_pretrain};
#[cfg(all(any(target_os = "macos", target_os = "ios"), feature = "metal"))]
pub use moe::generation::{context_loop, run_generated, run_generation};
#[cfg(all(any(target_os = "macos", target_os = "ios"), feature = "metal"))]
pub use moe::{ActualBoResult, GroupedMoeProbe};

/// Default embedded Metal shader sources for full-loop kernel experiments.
pub fn kernel_source(operator: &str) -> Result<&'static str, String> {
    match operator {
        "pisa" => Ok(include_str!("pisa1.metal")),
        "moe" => Ok(include_str!("routing_tensorops.metal")),
        "decode" => Ok(include_str!("decode.metal")),
        "readout" => Ok(include_str!("moe.metal")),
        "perturb" => Ok(include_str!("denoise.metal")),
        "mhc" => Ok(include_str!("moe.metal")),
        _ => Err(format!(
            "unsupported kernel operator {operator:?}; use pisa, moe, decode, readout, \
             perturb, or mhc"
        )),
    }
}

#[cfg(all(test, any(target_os = "macos", target_os = "ios"), feature = "metal"))]
mod shader_archive_tests {
    use baux::apple_gpu::{Runtime, archive_metrics};

    #[test]
    fn shader_archives() {
        let Ok(runtime) = Runtime::shared() else {
            return;
        };
        let directory = tempfile::tempdir().unwrap();
        let pisa_source = include_str!("pisa1.metal");
        runtime
            .compile_at(
                pisa_source,
                "pisa1",
                "fbt_pisa1_select_attention_q4",
                false,
                None,
                &[],
                Some(directory.path()),
            )
            .unwrap();
        let moe_source = include_str!("moe.metal");
        for entry in [
            "fbt_mhc_mix_rms",
            "fbt_mhc_mean_rms",
            "fbt_moe_residual_rms",
            "fbt_moe_ungroup_residual_rms",
            "fbt_moe_feedback_fuse_rms",
        ] {
            runtime
                .compile_at(
                    moe_source,
                    "moe",
                    entry,
                    false,
                    None,
                    &[],
                    Some(directory.path()),
                )
                .unwrap();
        }
        let records = baux::artifact::gpu::records(directory.path()).unwrap();
        assert_eq!(records.len(), 6);
        for (_, record) in records {
            let blob = record.output.expect("native binary output present");
            let bytes = std::fs::read(
                directory
                    .path()
                    .join("blobs")
                    .join(format!("{}.bin", blob.sha256)),
            )
            .unwrap();
            let metrics = archive_metrics(&bytes).expect("valid Apple GPU archive metrics");
            assert!(
                metrics.text_bytes > 0,
                "{} has empty machine code",
                record.recipe.entry
            );
            assert_eq!(
                metrics.spill_bytes, 0,
                "kernel {} spilled {} bytes/thread to stack",
                record.recipe.entry, metrics.spill_bytes
            );
        }
    }
}
