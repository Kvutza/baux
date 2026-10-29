mod audits;
mod cargo;
mod harness;
mod tpl_local;
mod tpl_up;

use audits::{audit_axioms, verify_llbc};
pub use cargo::snapshot_upstream;
use cargo::{
    CargoClosure, cargo_build, ensure_slice, ensure_upstream, file_sha, inspect_header, tree_digest,
};
use harness::{build_rustc, run_binaries, stage_deps, verify_outputs};
use tpl_local::LOCAL_SOURCE;
use tpl_up::UPSTREAM_SOURCE;

use baux::artifact::json::{self, Value, json};
use baux::artifact::runs::digest;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

const CABI_SYMBOLS: &[&str] = &[
    "baux_accept",
    "baux_action",
    "baux_world",
    "baux_anchored",
    "baux_routed",
    "baux_total",
    "baux_window",
    "baux_cell",
    "baux_slot",
    "baux_sign",
    "baux_numerical_init",
    "baux_numerical",
    "baux_numerical_fini",
];

fn canonical_json(val: &Value) -> Value {
    if let Some(map) = val.as_map() {
        let mut keys = map.keys().filter_map(|k| k.as_str()).collect::<Vec<_>>();
        keys.sort_unstable();
        let mut sorted = json::Map::new();
        for key in keys {
            if let Some(child) = map.get(key) {
                sorted.insert(key, canonical_json(child));
            }
        }
        return Value::from(sorted);
    }
    if let Some(seq) = val.as_seq() {
        return Value::from(seq.iter().map(canonical_json).collect::<Vec<_>>());
    }
    val.clone()
}

fn compile_harnesses(
    root: &Path,
    up: &CargoClosure,
    loc: &CargoClosure,
    ext: &CargoClosure,
    lean_archive: &Path,
    numerical_so: &Path,
) -> Result<(Value, PathBuf, PathBuf), String> {
    let verify_dir = root.join(".cache/baux/aeneas/artifact-verify");
    fs::create_dir_all(&verify_dir).map_err(|e| e.to_string())?;
    let up_deps = verify_dir.join("upstream-deps");
    let loc_deps = verify_dir.join("local-deps");
    stage_deps(&up.dep_files, &up.artifacts, &up.extern_names, &up_deps)?;
    let mut merged_art = ext.artifacts.clone();
    merged_art.extend(loc.artifacts.clone());
    let mut merged_ext = ext.extern_names.clone();
    merged_ext.extend(loc.extern_names.clone());
    let mut all_deps = ext.dep_files.clone();
    all_deps.extend(loc.dep_files.clone());
    stage_deps(&all_deps, &merged_art, &merged_ext, &loc_deps)?;

    let up_src = verify_dir.join("upstream_artifact_harness.rs");
    let up_bin = verify_dir.join("upstream_artifact_harness");
    fs::write(&up_src, UPSTREAM_SOURCE).map_err(|e| e.to_string())?;
    build_rustc(
        root,
        &up_src,
        &up_bin,
        &up_deps,
        &up.artifacts,
        &up.extern_names,
        &["ennbo", "ennbo_bpann", "ndarray", "rand"],
        &[],
        &up.link_search,
        &up.link_libs,
    )?;

    let loc_src = verify_dir.join("local_artifact_harness.rs");
    let loc_bin = verify_dir.join("local_artifact_harness");
    fs::write(&loc_src, LOCAL_SOURCE).map_err(|e| e.to_string())?;
    let extra = vec![
        format!("-Clink-arg={}", lean_archive.display()),
        "-C".into(),
        "link-arg=-Wl,-dead_strip".into(),
    ];
    build_rustc(
        root,
        &loc_src,
        &loc_bin,
        &loc_deps,
        &merged_art,
        &merged_ext,
        &["baux", "baux_optimizer_extraction", "ndarray", "rand"],
        &extra,
        &loc.link_search,
        &loc.link_libs,
    )?;

    let (up_json, loc_json) = run_binaries(root, &verify_dir, &up_bin, &loc_bin, numerical_so)?;
    let suite = verify_outputs(&up_json, &loc_json)?;
    Ok((suite, up_bin, loc_bin))
}

fn olean_records(lean_lib: &Path) -> Result<Value, String> {
    let mut records = json::Map::new();
    for (name, rel) in [
        (
            "Registry.olean",
            "lean/BauxResearch/Semantics/Registry.olean",
        ),
        (
            "Optimizer.olean",
            "lean/BauxResearch/Extraction/Optimizer.olean",
        ),
        ("Hash.olean", "lean/BauxResearch/Extraction/Hash.olean"),
        ("Bpann.olean", "lean/BauxResearch/Semantics/Bpann.olean"),
        ("Box.olean", "lean/BauxResearch/Extraction/Box.olean"),
        (
            "Numerical.olean",
            "lean/BauxResearch/Semantics/Numerical.olean",
        ),
    ] {
        let p = lean_lib.join(rel);
        let meta = fs::metadata(&p)
            .map_err(|_| format!("missing compiled Lean olean: {}", p.display()))?;
        records.insert(name, json!({"bytes": meta.len(), "sha256": file_sha(&p)?}));
    }
    Ok(Value::from(records))
}

fn build_assumptions(root: &Path) -> Result<Value, String> {
    let rustc_out = Command::new("rustc")
        .args(["--version", "--verbose"])
        .output()
        .map_err(|e| e.to_string())?;
    let lean_out = Command::new("lake")
        .current_dir(root.join("research/lean"))
        .args(["env", "lean", "--version"])
        .output()
        .map_err(|e| e.to_string())?;
    Ok(json!({
        "isa": "aarch64-apple-darwin (ARM64 LP64 little-endian, Mach-O 64-bit CPU_TYPE_ARM64)",
        "floating_point": "IEEE-754 binary64 (roundTiesToEven), hardware fsqrt/fadd/fsub/fmul/fdiv, deterministic system libm (ln, cos, exp, powf, ln_1p) on host darwin-arm64; no -ffast-math or FMA contraction across statement boundaries",
        "runtime_and_resources": "Deterministic single-seeded StdRng (ChaCha12) and NumpyPcg64 / SplitMix64 streams; bounded heap allocation without OOM; POSIX mmap/pwrite for BpAnnDiskBackend; 16 MiB stack owner thread for libbaux_numerical.so C-ABI checker",
        "toolchain": {
            "rustc": String::from_utf8_lossy(&rustc_out.stdout).trim(),
            "lean": String::from_utf8_lossy(&lean_out.stdout).trim(),
            "charon_rev": "de8736f7c995ac54dc490345cef9ea5143c06c62",
            "aeneas_rev": "e62dd83d133d9043e7d3628096838b7430f36565",
        },
    }))
}

pub fn build_manifest(root: &Path) -> Result<Value, String> {
    let (upstream_dir, upstream_meta) = ensure_upstream(root)?;
    let (slice_dir, local_digest) = ensure_slice(root)?;
    let cache = root.join(".cache/baux/aeneas");
    let up = cargo_build(
        root,
        &upstream_dir.join("rust/Cargo.toml"),
        &cache.join("upstream-target"),
        &["ennbo", "ennbo-bpann"],
    )?;
    let loc = cargo_build(
        root,
        &root.join("Cargo.toml"),
        &cache.join("local-production-target"),
        &["baux"],
    )?;
    let ext = cargo_build(
        root,
        &slice_dir.join("Cargo.toml"),
        &cache.join("local-default-target"),
        &["baux-optimizer-extraction"],
    )?;

    let lean_lib = root.join("research/lean/.lake/build/lib");
    let lean_archive = lean_lib.join("libbaux__research_BauxResearch.a");
    let numerical_so = root.join("research/lean/.lake/build/bin/libbaux_numerical.so");
    if !lean_archive.is_file() || !numerical_so.is_file() {
        return Err("Lean static archive or numerical shared library is not built".into());
    }
    let (suite, up_bin, loc_bin) =
        compile_harnesses(root, &up, &loc, &ext, &lean_archive, &numerical_so)?;
    let suite_bytes = json::to_string(&canonical_json(&suite)).map_err(|e| e.to_string())?;
    let binaries = json!({
        "upstream_ennbo_rlib": inspect_header(&up.artifacts["ennbo"])?,
        "upstream_bpann_rlib": inspect_header(&up.artifacts["ennbo_bpann"])?,
        "local_baux_rlib": inspect_header(&loc.artifacts["baux"])?,
        "local_optimizer_extraction_rlib": inspect_header(&ext.artifacts["baux_optimizer_extraction"])?,
        "lean_cabi_static_archive": inspect_header(&lean_archive)?,
        "lean_numerical_shared_object": inspect_header(&numerical_so)?,
        "upstream_artifact_harness_bin": inspect_header(&up_bin)?,
        "local_artifact_harness_bin": inspect_header(&loc_bin)?,
    });
    Ok(json!({
        "schema": "baux-compiled-optimizer-artifact-correspondence-v1",
        "task_id": "baux-x0b.2.9.49.3",
        "assumptions": build_assumptions(root)?,
        "device_scope": {
            "cpu_optimizer_equivalence": "ESTABLISHED",
            "gpu_device_backend": "NOT_ESTABLISHED",
            "statement": "Equivalence and compiled-artifact correspondence are established ONLY for the CPU reference optimizer and BPANN backends (ennbo 0.4.1, ennbo-bpann 0.4.1, baux, baux-optimizer-extraction, libbaux__research_BauxResearch.a, libbaux_numerical.so). Accelerator device kernels (Metal/MPS and CUDA/PTX/SASS) are explicitly excluded and NOT established by this CPU proof.",
        },
        "source_bindings": {
            "upstream": upstream_meta,
            "coverage_manifest_sha256": file_sha(&root.join("research/upstream/coverage_manifest.json"))?,
            "charon_callsite_patch_sha256": file_sha(&root.join("research/extraction/charon-callsite.patch"))?,
            "aeneas_float_patch_sha256": file_sha(&root.join("research/extraction/aeneas-float.patch"))?,
            "aeneas_dyn_patch_sha256": file_sha(&root.join("research/extraction/aeneas-dyn.patch"))?,
            "aeneas_dyn_dispatch_patch_sha256": file_sha(&root.join("research/extraction/aeneas-dyn-dispatch.patch"))?,
            "box_pair_metadata_sha256": file_sha(&root.join("research/extraction/box.json"))?,
            "local_baux_src_tree_sha256": local_digest,
            "local_semantics_src_tree_sha256": tree_digest(&root.join("rust/crates/baux/semantics/src"))?,
            "local_neighbor_index_src_tree_sha256": tree_digest(&root.join("rust/crates/baux/neighbor-index/src"))?,
        },
        "compiled_artifacts": {
            "binaries": binaries,
            "lean_oleans": olean_records(&lean_lib)?,
            "charon_llbc": verify_llbc(root)?,
        },
        "axiom_and_trust_audit": audit_axioms(root)?,
        "trajectory_verification": {
            "trajectory_suite_sha256": digest(suite_bytes.as_bytes()),
            "bit_exact_f64_parity": true,
            "cabi_symbols_verified": CABI_SYMBOLS,
            "suite": suite,
        },
    }))
}

pub fn check_manifest(root: &Path, current: &Value) -> Result<(), String> {
    let path = root.join("research/extraction/artifact_manifest.json");
    let text = fs::read_to_string(&path).map_err(|_| format!("missing {}", path.display()))?;
    let recorded: Value = json::from_str(&text).map_err(|e| e.to_string())?;
    for key in [
        "schema",
        "task_id",
        "assumptions",
        "device_scope",
        "source_bindings",
        "axiom_and_trust_audit",
        "trajectory_verification",
    ] {
        if current[key] != recorded[key] {
            return Err(format!("artifact manifest drift in `{key}`"));
        }
    }
    if current["compiled_artifacts"]["lean_oleans"] != recorded["compiled_artifacts"]["lean_oleans"]
        || current["compiled_artifacts"]["charon_llbc"]
            != recorded["compiled_artifacts"]["charon_llbc"]
        || current["compiled_artifacts"]["binaries"] != recorded["compiled_artifacts"]["binaries"]
    {
        return Err("compiled artifact digests drifted from \
                    research/extraction/artifact_manifest.json"
            .into());
    }
    Ok(())
}

pub fn run(root: &Path, check: bool) -> Result<(), String> {
    let current = build_manifest(root)?;
    let sha = current["trajectory_verification"]["trajectory_suite_sha256"]
        .as_str()
        .unwrap_or("");
    if check {
        check_manifest(root, &current)?;
        println!(
            "baux: compiled optimizer artifacts, LLBC coverage, C-ABI symbols, Lean axiom audit, \
             and 5 bit-exact trajectories verified (suite_sha256={}...)",
            &sha[..sha.len().min(16)]
        );
    } else {
        let path = root.join("research/extraction/artifact_manifest.json");
        let rendered = json::pretty_string(&current).map_err(|e| e.to_string())?;
        fs::write(&path, format!("{rendered}\n")).map_err(|e| e.to_string())?;
        println!("baux: wrote research/extraction/artifact_manifest.json (suite_sha256={sha})");
    }
    Ok(())
}
