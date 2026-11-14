use baux::semantics::schedule::Backend;
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};

pub use baux::semantics::shape::*;

/// Source files whose behavior contributes directly to a complete generation
/// and optimization round on the selected accelerator.
pub fn perf_sources(root: &Path, backend: Backend) -> Result<Vec<PathBuf>, String> {
    let source = root.join("rust/crates/baux/src");
    let fbt_source = root.join("rust/crates/fbt/src");
    let mut files = [
        "apple_gpu.rs",
        "config.rs",
        "forward_program.rs",
        "hash.rs",
        "objective_observation.rs",
        "pretrain_data.rs",
        "procedural_pool.rs",
        "resident_metal.rs",
        "search.rs",
        "trust_region.rs",
    ]
    .map(|path| source.join(path))
    .to_vec();
    files.extend(
        [
            "ablation.rs",
            "block.rs",
            "code_environment.rs",
            "context_metal.rs",
            "contrast.rs",
            "decode.rs",
            "generation.rs",
            "heldout.rs",
            "loss_tests.rs",
            "micro.rs",
            "moe.rs",
            "pisa.rs",
            "reconstruction.rs",
            "reward.rs",
            "routing.rs",
            "scorer.rs",
            "train.rs",
            "updates.rs",
            "bin/baux-tune-worker/main.rs",
        ]
        .map(|path| fbt_source.join(path)),
    );
    files.push(source.join("bin/baux-tune/main.rs"));
    for directory in [
        root.join("rust/crates/baux/semantics/src"),
        root.join("rust/crates/baux/artifact/src"),
        source.join("config"),
        fbt_source.join("context_metal"),
        fbt_source.join("block"),
        fbt_source.join("decode"),
        fbt_source.join("generation"),
        fbt_source.join("heldout"),
        fbt_source.join("mhc"),
        fbt_source.join("moe"),
        fbt_source.join("nn"),
        fbt_source.join("pisa"),
        fbt_source.join("routing"),
        fbt_source.join("scorer"),
        fbt_source.join("tensor"),
        fbt_source.join("train"),
        source.join("forward_program"),
        source.join("resident_metal"),
    ] {
        collect(&directory, &mut files)?;
    }
    match backend {
        Backend::Metal => {
            for dir in [&source, &fbt_source] {
                for entry in std::fs::read_dir(dir).map_err(|error| error.to_string())? {
                    let path = entry.map_err(|error| error.to_string())?.path();
                    if path
                        .extension()
                        .is_some_and(|extension| extension == "metal")
                    {
                        files.push(path);
                    }
                }
            }
        }
        Backend::CudaOxide => {
            collect(&root.join("rust/crates/cuda/src"), &mut files)?;
            collect(&root.join("rust/crates/cuda/device/src"), &mut files)?;
            collect(&root.join("rust/crates/modal-runner/src"), &mut files)?;
            files.push(root.join("rust/crates/runner/src/cuda_tune.rs"));
        }
    }
    // The manifest implementation is measurement infrastructure, not part of
    // the model execution path, and including it would make its digest
    // definition self-referential.
    files.retain(|path| path != &source.join("forward_program/contract.rs"));
    files.sort();
    files.dedup();
    if files.iter().any(|path| !path.is_file()) {
        return Err("performance source manifest contains a missing file".into());
    }
    Ok(files)
}

/// Content identity for the complete hot path. A source change invalidates a
/// measured baseline even when the semantic schedule remains unchanged.
pub fn perf_digest(root: &Path, backend: Backend) -> Result<String, String> {
    let mut digest = Sha256::new();
    for path in perf_sources(root, backend)? {
        let relative = path.strip_prefix(root).unwrap_or(&path);
        let bytes = std::fs::read(&path).map_err(|error| error.to_string())?;
        digest.update(relative.to_string_lossy().as_bytes());
        digest.update([0]);
        digest.update((bytes.len() as u64).to_le_bytes());
        digest.update(&bytes);
    }
    Ok(format!("{:x}", digest.finalize()))
}

fn collect(directory: &Path, files: &mut Vec<PathBuf>) -> Result<(), String> {
    for entry in std::fs::read_dir(directory).map_err(|error| error.to_string())? {
        let path = entry.map_err(|error| error.to_string())?.path();
        if path.is_dir() {
            collect(&path, files)?;
        } else if path.extension().is_some_and(|extension| extension == "rs")
            && !path
                .file_name()
                .is_some_and(|name| name.to_string_lossy().starts_with("._"))
        {
            files.push(path);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn contract_constants() {
        assert_eq!(MHC4_PARAMS, 265_623_320);
        assert_eq!(MHC4_PATCH, 1);
        assert_eq!(MHC4_COEFFS, 16_515_072);
        assert_eq!(PISA_BYTES, 2 * 64 * 2 + 2 * 64 / 64 * 2);
    }
}
