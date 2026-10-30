//! Conservative source inventory for the GPU proof workstream.
//!
//! This records source bytes, not a reachable call graph or compiled device
//! image. Neither a successful inventory nor its digest establishes GPU
//! correctness.

use baux::artifact::{
    cbor,
    json::{Value, json},
    runs::digest,
};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

const TREES: [&str; 6] = [
    "rust/crates/baux/src",
    "rust/crates/baux/semantics/src",
    "backends/cuda/src",
    "backends/cuda/device/src",
    "backends/cuda/interop",
    "rust/crates/ptx-synth/src",
];
const FILES: [&str; 15] = [
    "Cargo.toml",
    "Cargo.lock",
    "rust-toolchain.toml",
    "backends/cuda/Cargo.toml",
    "backends/cuda/device/Cargo.toml",
    "backends/cuda/rust-toolchain.toml",
    "build/buck2/cuda/defs.bzl",
    "rust/crates/baux/BUCK",
    "backends/cuda/build.rs",
    "backends/cuda/BUCK",
    "rust/crates/ptx-synth/Cargo.toml",
    "rust/crates/ptx-synth/BUCK",
    "rust/crates/baux/Cargo.toml",
    "rust/crates/baux/defs.bzl",
    "build/buck2/baux_targets.bzl",
];
const OUTPUT: &str = ".cache/baux/verification/gpu-inventory.cbor";
const GAPS: [&str; 6] = [
    "complete optimizer and objective semantics",
    "numerical error and decision preservation",
    "parallel memory, synchronization and termination",
    "source-to-native-device compilation correspondence",
    "host/device launch and buffer protocol",
    "composed theorem bound to the executed artifact",
];
static NEXT_FILE: AtomicU64 = AtomicU64::new(0);

#[derive(Debug)]
struct Inventory {
    record: Value,
    digest: String,
    sources: usize,
    metal: usize,
    cuda: usize,
}

impl Inventory {
    fn summary(&self) -> String {
        format!(
            "GPU verification: NOT ESTABLISHED\nSource inventory: {} files, {} Metal sources, {} \
             CUDA device Rust sources\nSource inventory SHA256: {}\nScope: source inventory only; \
             reachable closure and executed device bytes are unknown.\nRequired proof stages (not \
             established by this check):\n  - {}\n",
            self.sources,
            self.metal,
            self.cuda,
            self.digest,
            GAPS.join("\n  - ")
        )
    }
}

pub fn report(root: &Path) -> Result<String, String> {
    Ok(inventory(root)?.summary())
}

/// Save source evidence. Success means only that the inventory was recorded.
pub fn check(root: &Path) -> Result<String, String> {
    let evidence = inventory(root)?;
    let output = root.join(OUTPUT);
    let parent = output
        .parent()
        .ok_or("GPU inventory has no parent directory")?;
    fs::create_dir_all(parent).map_err(|error| format!("create {}: {error}", parent.display()))?;
    let bytes = cbor::to_vec(&evidence.record).map_err(|error| error.to_string())?;
    let temporary = parent.join(format!(
        ".gpu-inventory-{}-{}.tmp",
        std::process::id(),
        NEXT_FILE.fetch_add(1, Ordering::Relaxed)
    ));
    fs::write(&temporary, bytes)
        .map_err(|error| format!("write {}: {error}", temporary.display()))?;
    if let Err(error) = fs::rename(&temporary, &output) {
        let _ = fs::remove_file(&temporary);
        return Err(format!("publish {}: {error}", output.display()));
    }
    Ok(format!("{}Evidence: {OUTPUT}\n", evidence.summary()))
}

fn inventory(root: &Path) -> Result<Inventory, String> {
    let mut paths = Vec::new();
    for relative in TREES {
        collect(&root.join(relative), &mut paths)?;
    }
    paths.extend(FILES.map(|relative| root.join(relative)));
    paths.sort();
    paths.dedup();
    let mut sources = Vec::with_capacity(paths.len());
    let mut identity = Vec::new();
    let mut metal = 0;
    let mut cuda = 0;
    let mut missing = Vec::new();
    for path in paths {
        // Cargo.lock is ignored in this repository and may not exist in a new
        // JJ workspace. Its absence is evidence, not a reason to hide status.
        if path == root.join("Cargo.lock")
            && fs::symlink_metadata(&path)
                .is_err_and(|error| error.kind() == std::io::ErrorKind::NotFound)
        {
            missing.push("Cargo.lock");
            identity.extend_from_slice(b"missing:Cargo.lock\0");
            continue;
        }
        let metadata = fs::symlink_metadata(&path)
            .map_err(|error| format!("inspect {}: {error}", path.display()))?;
        if !metadata.is_file() {
            return Err(format!(
                "GPU inventory requires a regular file: {}",
                path.display()
            ));
        }
        let relative = path.strip_prefix(root).map_err(|error| error.to_string())?;
        let relative = relative.to_str().ok_or("GPU source path is not UTF-8")?;
        let bytes = fs::read(&path).map_err(|error| format!("read {}: {error}", path.display()))?;
        let sha256 = digest(&bytes);
        let kind = if path
            .extension()
            .is_some_and(|extension| extension == "metal")
        {
            metal += 1;
            "metal-source"
        } else if relative.starts_with("backends/cuda/device/src/") {
            cuda += 1;
            "cuda-device-source"
        } else {
            "supporting-source"
        };
        identity.extend_from_slice(&(relative.len() as u64).to_le_bytes());
        identity.extend_from_slice(relative.as_bytes());
        identity.extend_from_slice(&(bytes.len() as u64).to_le_bytes());
        identity.extend_from_slice(sha256.as_bytes());
        sources.push(json!({
            "path": relative,
            "kind": kind,
            "bytes": bytes.len() as u64,
            "sha256": sha256,
        }));
    }
    if metal == 0 || cuda == 0 {
        return Err("GPU source inventory is empty for a required execution target".into());
    }
    let count = sources.len();
    let digest = digest(&identity);
    let record = json!({
        "schema": "baux.gpu-source-inventory.v1",
        "scope": "source-inventory",
        "source_digest": digest.clone(),
        "fully_verified": false,
        "reachable_closure": "not-established",
        "executed_device_artifact": "not-captured",
        "atomic_source_snapshot": false,
        "missing_optional_inputs": missing,
        "proof_chain": GAPS.iter().map(|stage| json!({
            "stage": *stage,
            "status": "not-established-by-inventory",
        })).collect::<Vec<_>>(),
        "sources": sources,
    });
    Ok(Inventory {
        record,
        digest,
        sources: count,
        metal,
        cuda,
    })
}

fn collect(directory: &Path, paths: &mut Vec<PathBuf>) -> Result<(), String> {
    let metadata = fs::symlink_metadata(directory)
        .map_err(|error| format!("inspect {}: {error}", directory.display()))?;
    if !metadata.is_dir() {
        return Err(format!(
            "GPU inventory requires a real directory: {}",
            directory.display()
        ));
    }
    for entry in
        fs::read_dir(directory).map_err(|error| format!("read {}: {error}", directory.display()))?
    {
        let entry = entry.map_err(|error| error.to_string())?;
        let path = entry.path();
        let kind = entry.file_type().map_err(|error| error.to_string())?;
        if kind.is_symlink() {
            return Err(format!(
                "GPU inventory does not follow symlinks: {}",
                path.display()
            ));
        }
        if kind.is_dir() {
            collect(&path, paths)?;
        } else if kind.is_file()
            && path.extension().is_some_and(|extension| {
                matches!(
                    extension.to_str(),
                    Some("rs" | "metal" | "cu" | "h" | "cpp" | "cl")
                )
            })
        {
            paths.push(path);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Fixture(PathBuf);

    impl Fixture {
        fn new() -> Self {
            let root = std::env::temp_dir().join(format!(
                "baux-gpu-{}-{}",
                std::process::id(),
                NEXT_FILE.fetch_add(1, Ordering::Relaxed)
            ));
            for tree in TREES {
                fs::create_dir_all(root.join(tree)).unwrap();
            }
            for file in FILES {
                let path = root.join(file);
                fs::create_dir_all(path.parent().unwrap()).unwrap();
                fs::write(path, "fixture").unwrap();
            }
            fs::write(
                root.join("rust/crates/baux/src/op.metal"),
                "kernel void op() {}",
            )
            .unwrap();
            fs::write(root.join("backends/cuda/device/src/op.rs"), "fn op() {}").unwrap();
            Self(root)
        }
    }

    impl Drop for Fixture {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn changes() {
        let fixture = Fixture::new();
        let original = inventory(&fixture.0).unwrap();
        assert_eq!(original.digest, inventory(&fixture.0).unwrap().digest);
        fs::write(
            fixture.0.join("rust/crates/baux/src/op.metal"),
            "kernel void changed() {}",
        )
        .unwrap();
        let changed = inventory(&fixture.0).unwrap();
        assert_ne!(original.digest, changed.digest);
        fs::rename(
            fixture.0.join("backends/cuda/device/src/op.rs"),
            fixture.0.join("backends/cuda/device/src/renamed.rs"),
        )
        .unwrap();
        assert_ne!(changed.digest, inventory(&fixture.0).unwrap().digest);
        assert!(changed.summary().contains("NOT ESTABLISHED"));
    }

    #[test]
    fn empty() {
        let fixture = Fixture::new();
        fs::remove_file(fixture.0.join("backends/cuda/device/src/op.rs")).unwrap();
        assert!(inventory(&fixture.0).unwrap_err().contains("empty"));
    }

    #[test]
    fn missing_lock() {
        let fixture = Fixture::new();
        let before = inventory(&fixture.0).unwrap();
        fs::remove_file(fixture.0.join("Cargo.lock")).unwrap();
        let after = inventory(&fixture.0).unwrap();
        assert_ne!(before.digest, after.digest);
        assert_eq!(
            after.record["missing_optional_inputs"][0].as_str(),
            Some("Cargo.lock")
        );
        assert!(after.summary().contains("NOT ESTABLISHED"));
    }

    #[test]
    fn store() {
        let fixture = Fixture::new();
        check(&fixture.0).unwrap();
        let bytes = fs::read(fixture.0.join(OUTPUT)).unwrap();
        let record: Value = cbor::from_slice(&bytes).unwrap();
        assert_eq!(record["fully_verified"].as_bool(), Some(false));
        assert_eq!(
            record["reachable_closure"].as_str(),
            Some("not-established")
        );
    }

    #[cfg(unix)]
    #[test]
    fn symlinks() {
        let fixture = Fixture::new();
        std::os::unix::fs::symlink(
            "op.metal",
            fixture.0.join("rust/crates/baux/src/link.metal"),
        )
        .unwrap();
        assert!(inventory(&fixture.0).unwrap_err().contains("symlinks"));
    }
}

pub(super) fn lint_names(root: &Path) -> Result<(), String> {
    let source = root.join("research/lean");
    let mut issues = Vec::new();
    scan_path(&source, root, &mut issues)?;
    if issues.is_empty() {
        return Ok(());
    }
    Err(format!(
        "Lean naming policy failed (two words, one underscore, 24 characters):\n{}",
        issues.join("\n")
    ))
}

fn scan_path(path: &Path, root: &Path, issues: &mut Vec<String>) -> Result<(), String> {
    for entry in fs::read_dir(path).map_err(|error| error.to_string())? {
        let entry = entry.map_err(|error| error.to_string())?;
        let path = entry.path();
        if path.is_dir() {
            if path.file_name().is_some_and(|name| name == ".lake") {
                continue;
            }
            scan_path(&path, root, issues)?;
            continue;
        }
        if path.extension().is_none_or(|extension| extension != "lean") {
            continue;
        }
        let relative = path.strip_prefix(root).unwrap_or(&path).display();
        if let Some(stem) = path.file_stem().and_then(|name| name.to_str())
            && !name_ok(stem)
        {
            issues.push(format!("{relative}:1: file `{stem}`"));
        }
        let text = fs::read_to_string(&path).map_err(|error| error.to_string())?;
        for (line_index, line) in text.lines().enumerate() {
            let mut words = line.split_whitespace();
            let mut word = words.next().unwrap_or("");
            while matches!(word, "private" | "protected" | "nonrec" | "noncomputable") {
                word = words.next().unwrap_or("");
            }
            let kind = match word {
                "abbrev" | "class" | "def" | "inductive" | "lemma" | "namespace" | "opaque"
                | "structure" | "theorem" => word,
                _ => continue,
            };
            let Some(raw_name) = words.next() else {
                continue;
            };
            let name = raw_name.trim_end_matches(|character: char| {
                !character.is_ascii_alphanumeric() && character != '_' && character != '.'
            });
            // Extracted identifiers preserve source/trait names, including
            // upstream names. Both generated files are compared
            // with fresh extraction before this lint.
            if (path.ends_with("BauxResearch/Generated/Floating/Funs.lean")
                || path.ends_with("BauxResearch/Generated/Floating/Types.lean")
                || path.ends_with("BauxResearch/Generated/LocalCoord/Funs.lean")
                || path.ends_with("BauxResearch/Generated/LocalCoord/Types.lean")
                || path.ends_with("BauxResearch/Generated/Optimizer/FunsExternal.lean")
                || path.ends_with("BauxResearch/Generated/Optimizer/TypesExternal.lean")
                || path.ends_with("BauxResearch/Generated/Enn/Funs.lean")
                || path.ends_with("BauxResearch/Generated/Enn/FunsExternal.lean")
                || path.ends_with("BauxResearch/Generated/Enn/Types.lean")
                || path.ends_with("BauxResearch/Generated/Enn/TypesExternal.lean")
                || path.ends_with("BauxResearch/Generated/Pure/Funs.lean")
                || path.ends_with("BauxResearch/Generated/Pure/FunsExternal.lean")
                || path.ends_with("BauxResearch/Generated/Pure/Types.lean")
                || path.ends_with("BauxResearch/Generated/Pure/TypesExternal.lean"))
                && text.starts_with("-- THIS FILE WAS AUTOMATICALLY GENERATED BY AENEAS\n")
            {
                continue;
            }
            if name.is_empty() || !name.chars().next().is_some_and(|c| c.is_ascii_alphabetic()) {
                continue;
            }
            if name.split('.').any(|segment| !name_ok(segment)) {
                issues.push(format!("{relative}:{}: {kind} `{name}`", line_index + 1));
            }
        }
    }
    Ok(())
}

fn name_ok(name: &str) -> bool {
    name.len() <= 24 && name.matches('_').count() <= 1 && word_count(name) <= 2
}

fn part_words(part: &str) -> usize {
    let bytes = part.as_bytes();
    if bytes.is_empty() {
        return 0;
    }
    1 + (1..bytes.len())
        .filter(|&index| {
            let previous = bytes[index - 1];
            let current = bytes[index];
            let next = bytes.get(index + 1).copied();
            (previous.is_ascii_lowercase() && current.is_ascii_uppercase())
                || (previous.is_ascii_alphabetic() && current.is_ascii_digit())
                || (previous.is_ascii_digit() && current.is_ascii_alphabetic())
                || (previous.is_ascii_uppercase()
                    && current.is_ascii_uppercase()
                    && next.is_some_and(|value| value.is_ascii_lowercase()))
        })
        .count()
}

fn word_count(name: &str) -> usize {
    let mut normalized = name.to_owned();
    for atom in ["BFloat16", "BF16", "BAUX"] {
        normalized = normalized.replace(atom, "x");
    }
    normalized.split('_').map(part_words).sum()
}
