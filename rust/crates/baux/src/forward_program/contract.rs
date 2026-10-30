use baux::semantics::schedule::ExecutionTarget;
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};

pub use baux::semantics::shape::*;

/// Source files whose behavior contributes directly to a complete generation
/// and optimization round on the selected accelerator.
pub fn perf_sources(root: &Path, target: ExecutionTarget) -> Result<Vec<PathBuf>, String> {
    let source = root.join("rust/crates/baux/src");
    let mut files = [
        "apple_gpu.rs",
        "code_environment.rs",
        "config.rs",
        "context_metal.rs",
        "forward_program.rs",
        "hash.rs",
        "objective_observation.rs",
        "procedural_pool.rs",
        "pretrain_data.rs",
        "reliability_region.rs",
        "resident_cuda.rs",
        "resident_metal.rs",
        "search.rs",
        "trust_region.rs",
    ]
    .map(|path| source.join(path))
    .to_vec();
    files.extend([
        source.join("bin/baux-tune/main.rs"),
        source.join("bin/baux-tune-worker/main.rs"),
    ]);
    for directory in [
        root.join("rust/crates/baux/semantics/src"),
        root.join("rust/crates/baux/artifact/src"),
        source.join("config"),
        source.join("context_metal"),
        source.join("fbt_block"),
        source.join("fbt_decode"),
        source.join("fbt_generation"),
        source.join("fbt_moe"),
        source.join("fbt_pisa1"),
        source.join("fbt_scorer"),
        source.join("forward_program"),
        source.join("resident_metal"),
        source.join("search"),
    ] {
        collect(&directory, &mut files)?;
    }
    for entry in std::fs::read_dir(&source).map_err(|error| error.to_string())? {
        let path = entry.map_err(|error| error.to_string())?.path();
        if path.is_file()
            && path
                .file_name()
                .is_some_and(|name| name.to_string_lossy().starts_with("fbt"))
            && path.extension().is_some_and(|extension| extension == "rs")
        {
            files.push(path);
        }
    }
    match target {
        ExecutionTarget::Metal => {
            for entry in std::fs::read_dir(&source).map_err(|error| error.to_string())? {
                let path = entry.map_err(|error| error.to_string())?.path();
                if path
                    .extension()
                    .is_some_and(|extension| extension == "metal")
                {
                    files.push(path);
                }
            }
        }
        ExecutionTarget::CudaOxide => {
            collect(&root.join("backends/cuda/src"), &mut files)?;
            collect(&root.join("backends/cuda/device/src"), &mut files)?;
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
pub fn perf_digest(root: &Path, target: ExecutionTarget) -> Result<String, String> {
    let mut digest = Sha256::new();
    for path in perf_sources(root, target)? {
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
        assert_eq!(MHC4_PARAMS, 1_177_723_160);
        assert_eq!(MHC4_PATCH, 1);
        assert_eq!(MHC4_COEFFS, 16_805_376);
        assert_eq!(PISA_BYTES, 2 * 64 * 2 + 2 * 64 / 64 * 2);
    }
}
