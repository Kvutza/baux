use baux::artifact::json::{Map, Value};
use baux::artifact::wire_json;
use std::collections::BTreeSet;
use std::fs;
use std::path::Path;
use std::process::Command;
use std::time::Duration;

const EXCLUSIONS: &[(&str, &str, &str, &str)] = &[
    (
        "tests/*",
        "directory",
        "tests",
        "Automated pytest and json regression suites; outside library wheel runtime.",
    ),
    (
        "_reports/*",
        "directory",
        "reports",
        "Data processing and LaTeX generation scripts for reports; not imported by runtime.",
    ),
    (
        "ops/*",
        "directory",
        "ops",
        "Modal and cluster operational scripts for job launching; not part of optimizer core.",
    ),
    (
        "docs/*",
        "directory",
        "documentation",
        "Documentation sources and PDF figures; non-executable.",
    ),
    (
        "papers/*",
        "directory",
        "documentation",
        "Academic research paper manuscript source files; non-executable.",
    ),
    (
        "admin/*",
        "directory",
        "admin",
        "Administrative and CI workflow scripts; not packaged into the library.",
    ),
    (
        "scripts/*",
        "directory",
        "scripts",
        "Development environment bootstrap scripts; not called by library code.",
    ),
    (
        "rust/crates/bpann/tests/*",
        "directory",
        "tests",
        "BPANN integration test targets executed by cargo test; excluded from artifact.",
    ),
    (
        "rust/crates/ennbo/tests/*",
        "directory",
        "tests",
        "ENNBO integration test targets executed by cargo test; excluded from artifact.",
    ),
    (
        "rust/crates/ennbo/examples/*",
        "directory",
        "examples",
        "Standalone Cargo example binaries; excluded from artifact.",
    ),
    (
        "rust/crates/bpann/src/index/*_tests.rs",
        "source_pattern",
        "tests",
        "Unit test assertions compiled strictly under #[cfg(test)].",
    ),
    (
        "rust/crates/ennbo/src/optimizer/tests*.rs",
        "source_pattern",
        "tests",
        "Unit test assertions compiled strictly under #[cfg(test)].",
    ),
    (
        "rust/crates/ennbo/src/posterior/tests*.rs",
        "source_pattern",
        "tests",
        "Unit test assertions compiled strictly under #[cfg(test)].",
    ),
    (
        "rust/crates/ennbo/src/strategy/tests*.rs",
        "source_pattern",
        "tests",
        "Unit test assertions compiled strictly under #[cfg(test)].",
    ),
    (
        "rust/crates/ennbo/src/test_helpers.rs",
        "source_file",
        "tests",
        "Test harness helpers compiled strictly under #[cfg(test)].",
    ),
    (
        "rust/crates/ennbo/src/golden_parity.rs",
        "source_file",
        "tests",
        "Parity assertion harness used strictly in tests/weak_mathy_golden.rs; never called in \
         release.",
    ),
    (
        "rust/crates/ennbo/src/benchmarks.rs",
        "source_file",
        "benchmarks",
        "Synthetic test objective functions for offline benchmark runners; unreached from \
         Ask/Tell.",
    ),
    (
        "rust/crates/enn-py/src/py_fit.rs",
        "source_file",
        "unreached_standalone_binding",
        "Standalone PyO3 fit bindings; unreached from PyOptimizer ask/tell.",
    ),
    (
        "rust/crates/enn-py/src/py_fitter.rs",
        "source_file",
        "unreached_standalone_binding",
        "Standalone PyO3 fitter bindings; unreached from PyOptimizer ask/tell.",
    ),
    (
        "rust/crates/enn-py/src/py_hash.rs",
        "source_file",
        "unreached_standalone_binding",
        "Standalone PyO3 hash bindings; unreached from PyOptimizer ask/tell.",
    ),
    (
        "rust/crates/enn-py/src/py_hypervolume.rs",
        "source_file",
        "unreached_standalone_binding",
        "Standalone PyO3 hypervolume bindings; unreached from PyOptimizer ask/tell.",
    ),
    (
        "rust/crates/enn-py/src/py_metric.rs",
        "source_file",
        "unreached_standalone_binding",
        "Standalone PyO3 metric bindings; unreached from PyOptimizer ask/tell.",
    ),
    (
        "rust/crates/enn-py/src/py_model.rs",
        "source_file",
        "unreached_standalone_binding",
        "Standalone PyO3 model bindings; unreached from PyOptimizer ask/tell.",
    ),
    (
        "rust/crates/enn-py/src/py_numpy_normal.rs",
        "source_file",
        "unreached_standalone_binding",
        "Standalone PyO3 normal sampling bindings; unreached from PyOptimizer ask/tell.",
    ),
    (
        "rust/crates/enn-py/src/py_params.rs",
        "source_file",
        "unreached_standalone_binding",
        "Standalone PyO3 params bindings; unreached from PyOptimizer ask/tell.",
    ),
    (
        "rust/crates/enn-py/src/py_ports.rs",
        "source_file",
        "unreached_standalone_binding",
        "Standalone PyO3 ports bindings; unreached from PyOptimizer ask/tell.",
    ),
    (
        "rust/crates/enn-py/src/py_util.rs",
        "source_file",
        "unreached_standalone_binding",
        "Standalone PyO3 utility bindings; unreached from PyOptimizer ask/tell.",
    ),
    (
        "rust/crates/enn-py/src/pymodule_wrappers.rs",
        "source_file",
        "unreached_standalone_binding",
        "PyO3 module registration wrappers for auxiliary structs; unreached from Ask/Tell.",
    ),
    (
        "rust/crates/ennbo/src/hypervolume.rs",
        "source_file",
        "unreached_module",
        "Standalone 2D hypervolume utility; Pareto acquisition uses non-dominated sort, not \
         hypervolume.",
    ),
    (
        "rust/crates/ennbo/src/incremental_fit.rs",
        "source_file",
        "unreached_module",
        "Experimental incremental fitting interface; not wired into ENNSurrogate.",
    ),
    (
        "rust/crates/ennbo/src/ndtri.rs",
        "source_file",
        "unreached_module",
        "Cephes ndtri reference implementation; unreached by optimizer runtime.",
    ),
    (
        "rust/crates/ennbo/src/normal_sample.rs",
        "source_file",
        "unreached_module",
        "Standalone normal sampling utility used only by golden parity tests.",
    ),
    (
        "rust/crates/ennbo/src/numpy_ziggurat_tables.rs",
        "source_file",
        "unreached_module",
        "NumPy ziggurat tables for standard_normal; only integers_high is called by reservoir.",
    ),
    (
        "rust/crates/ennbo/src/philox.rs",
        "source_file",
        "unreached_module",
        "Philox reference PRNG; SplitMix64 is used for runtime hash sampling.",
    ),
    (
        "rust/crates/ennbo/src/philox_hash.rs",
        "source_file",
        "unreached_module",
        "Philox reference hash wrapper; unreached by optimizer runtime.",
    ),
    (
        "rust/crates/ennbo/cmake/*",
        "directory",
        "build_config",
        "CMake toolchains and build helper scripts for Faiss C++ interop.",
    ),
    (
        "rust/crates/enn-py/scripts/*",
        "directory",
        "build_config",
        "ELF patchelf and wheel packaging scripts.",
    ),
    (
        "rust/crates/enn-py/TESTING.md",
        "source_file",
        "documentation",
        "Python test documentation; non-executable.",
    ),
    (
        "rust/crates/ennbo/README.md",
        "source_file",
        "documentation",
        "ENNBO crate documentation; non-executable.",
    ),
    (
        "rust/crates/*/Cargo.toml",
        "config_pattern",
        "build_config",
        "Crate level Cargo manifest specifications.",
    ),
    (
        "rust/Cargo.*",
        "config_pattern",
        "build_config",
        "Rust workspace Cargo manifest and lockfile.",
    ),
    (
        "rust/.cargo/*",
        "config_pattern",
        "build_config",
        "Cargo config settings.",
    ),
    (
        ".cargo/*",
        "config_pattern",
        "build_config",
        "Root Cargo config settings.",
    ),
    (
        "src/enn/benchmarks/*",
        "module_tree",
        "benchmarks",
        "Synthetic test objective functions for evaluation runners; not imported by optimization \
         machine.",
    ),
    (
        "src/enn/enn/*",
        "module_tree",
        "pure_python_reference",
        "Pure Python prototype implementation; optimizer runtime delegates to compiled Rust ennbo.",
    ),
    (
        "src/enn/turbo/types/gp_*.py",
        "source_pattern",
        "pure_python_reference",
        "Legacy pure Python GP types; unreached by Rust-backed optimizer.",
    ),
    (
        "src/enn/turbo/types/obs_lists.py",
        "source_file",
        "pure_python_reference",
        "Legacy pure Python observation container; unreached by Rust-backed optimizer.",
    ),
    (
        "src/enn/turbo/types/prepare_ask_result.py",
        "source_file",
        "pure_python_reference",
        "Legacy pure Python ask result container; unreached by Rust-backed optimizer.",
    ),
    (
        "src/enn/turbo/types/tell_inputs.py",
        "source_file",
        "pure_python_reference",
        "Legacy pure Python tell input container; unreached by Rust-backed optimizer.",
    ),
    (
        "src/enn/turbo/types/__init__.py",
        "source_file",
        "pure_python_reference",
        "Legacy pure Python types package re-export; unreached by Rust-backed optimizer.",
    ),
    (
        "src/enn/turbo/tell_inputs.py",
        "source_file",
        "pure_python_reference",
        "Legacy pure Python tell input validator; unreached by Rust-backed optimizer.",
    ),
    (
        "src/enn/turbo/hypervolume.py",
        "source_file",
        "unreached_module",
        "Standalone Python wrapper for hypervolume_2d_max; unreached by Rust-backed optimizer.",
    ),
    (
        "src/enn/turbo/proposal.py",
        "source_file",
        "pure_python_reference",
        "Legacy pure Python proposal generator; unreached by Rust-backed optimizer.",
    ),
    (
        "src/enn/turbo/components/__init__.py",
        "source_file",
        "empty_package",
        "Empty package marker file.",
    ),
    (
        "src/enn/turbo/strategies/__init__.py",
        "source_file",
        "empty_package",
        "Empty package marker file.",
    ),
    (
        "*",
        "root_config",
        "build_config",
        "Root repository build configuration, metadata, licenses, or lockfiles.",
    ),
];

const EXT_CALLS: &[(&str, &str, &[&str])] = &[
    ("faiss", "C++ Faiss in-memory exact search", &["ennbo"]),
    (
        "memmap2",
        "Memory-mapped disk index backing",
        &["bpann", "ennbo"],
    ),
    (
        "ndarray",
        "Tensor views and multidimensional indexing",
        &["bpann", "ennbo", "enn-py"],
    ),
    (
        "ndarray-linalg",
        "Linear algebra solver routines",
        &["ennbo"],
    ),
    ("numpy", "Zero-copy NumPy to ndarray bridging", &["enn-py"]),
    ("pyo3", "Python C-extension FFI runtime", &["enn-py"]),
    (
        "rand",
        "StdRng candidate generation and sampling",
        &["bpann", "ennbo", "enn-py"],
    ),
    (
        "rayon",
        "Parallel neighbor search and parallel evaluations",
        &["bpann", "ennbo"],
    ),
    ("serde", "Serialization primitives", &["bpann", "ennbo"]),
    (
        "sobol",
        "Quasi-random Sobol sequence generation",
        &["ennbo"],
    ),
    ("thiserror", "Error enum derive macros", &["bpann", "ennbo"]),
    ("toml", "Config parser for offline presets", &["ennbo"]),
];

pub(super) fn exclusion_defs() -> Vec<Value> {
    EXCLUSIONS
        .iter()
        .map(|&(scope, kind, category, proof)| {
            wire_json!({
                "category": category,
                "proof": proof,
                "scope": scope,
                "type": kind,
            })
        })
        .collect()
}

pub(super) fn manifest_roots() -> Value {
    wire_json!({
        "ffi": {
            "ask": "rust/crates/enn-py/src/py_optimizer.rs:PyOptimizer::ask",
            "factory": "rust/crates/enn-py/src/py_optimizer.rs:py_create_optimizer_enn",
            "tell": "rust/crates/enn-py/src/py_optimizer.rs:PyOptimizer::tell",
        },
        "python": {
            "ask": "src/enn/turbo/rust_optimizer.py:RustOptimizer.ask",
            "factory": "src/enn/turbo/rust_optimizer.py:create_optimizer",
            "tell": "src/enn/turbo/rust_optimizer.py:RustOptimizer.tell",
        },
        "rust": {
            "ask": "rust/crates/ennbo/src/optimizer/mod.rs:Optimizer.ask",
            "factory": "rust/crates/ennbo/src/optimizer_factory.rs:create_optimizer_enn",
            "factory_with_overrides": "rust/crates/ennbo/src/optimizer_factory.rs:create_optimizer_enn_with_overrides",
            "tell": "rust/crates/ennbo/src/optimizer/mod.rs:Optimizer.tell",
        },
    })
}

pub(super) fn cfg_features() -> Value {
    wire_json!({
        "feature_flags": [
            {"crate": "ennbo-py", "feature": "extension-module", "package": "pyo3"},
            {"crate": "ennbo", "feature": "static", "package": "faiss", "target": "linux"},
        ],
        "target_os_branches": [
            {"cfg": "cfg(target_os = 'linux')", "file": "rust/crates/bpann/src/index/tree_store.rs:85", "role": "Linux memory layout and pre-allocation"},
            {"cfg": "cfg(not(target_os = 'linux'))", "file": "rust/crates/bpann/src/index/tree_store.rs:89", "role": "Portable POSIX memory layout"},
            {"cfg": "cfg(target_os = 'linux')", "file": "rust/crates/ennbo/src/metric_auto.rs:47", "role": "Linux hardware performance counter query"},
        ],
        "test_isolation": "All #[cfg(test)] modules strictly partitioned into verified exclusions.",
    })
}

pub(super) fn external_calls() -> Value {
    let mut map = Map::new();
    for &(name, role, crates) in EXT_CALLS {
        map.insert(
            name,
            wire_json!({
                "crates": crates,
                "role": role,
            }),
        );
    }
    Value::from(map)
}

pub(super) fn generated_records(enn_root: &Path, file_hash: fn(&Path) -> String) -> Vec<Value> {
    let f1 = "rust/crates/ennbo/src/ennbo_build_api.inc.rs";
    let f2 = "rust/crates/enn-py/src/enn_py_build_api.inc.rs";
    vec![
        wire_json!({
            "file": f1,
            "generator": "rust/crates/ennbo/src/ennbo_build.rs",
            "include_site": "rust/crates/ennbo/src/lib.rs:12",
            "role": "Generated C-ABI entrypoints and API descriptor tables for native optimizer interop",
            "sha256": file_hash(&enn_root.join(f1)),
        }),
        wire_json!({
            "file": f2,
            "generator": "rust/crates/enn-py/src/enn_py_build.rs",
            "include_site": "rust/crates/enn-py/src/lib.rs:14",
            "role": "Generated PyO3 module initialization tables and function registration wrappers",
            "sha256": file_hash(&enn_root.join(f2)),
        }),
    ]
}

fn walk_repo(dir: &Path, enn_root: &Path, files: &mut Vec<String>) {
    let Ok(entries) = fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let p = entry.path();
        let name = p.file_name().and_then(|s| s.to_str()).unwrap_or("");
        if matches!(name, ".git" | "target") {
            continue;
        }
        if p.is_dir() {
            walk_repo(&p, enn_root, files);
        } else if let Ok(rel) = p.strip_prefix(enn_root) {
            files.push(rel.to_string_lossy().replace('\\', "/"));
        }
    }
}

pub(super) fn list_files(enn_root: &Path) -> Vec<String> {
    if let Ok(out) = Command::new("git")
        .arg("-C")
        .arg(enn_root)
        .arg("ls-files")
        .output()
        && out.status.success()
    {
        let mut files: Vec<String> = String::from_utf8_lossy(&out.stdout)
            .lines()
            .map(str::to_owned)
            .collect();
        files.sort();
        return files;
    }
    let mut files = Vec::new();
    walk_repo(enn_root, enn_root, &mut files);
    files.sort();
    files
}

fn glob_match(text: &str, pat: &str) -> bool {
    let Some((prefix, suffix)) = pat.split_once('*') else {
        return text == pat;
    };
    let Some(rest) = text.strip_prefix(prefix) else {
        return false;
    };
    if !suffix.contains('*') {
        return rest.ends_with(suffix);
    }
    (0..=rest.len()).any(|i| rest.is_char_boundary(i) && glob_match(&rest[i..], suffix))
}

pub(super) fn matches_scope(rel_path: &str, scope: &str) -> bool {
    if scope == "*" {
        return !rel_path.contains('/');
    }
    if let Some(prefix) = scope.strip_suffix("/*") {
        return rel_path
            .strip_prefix(prefix)
            .is_some_and(|r| r.starts_with('/'));
    }
    glob_match(rel_path, scope)
}

pub(super) fn check_partition(
    repo_files: &[String],
    reached: &BTreeSet<String>,
    defs: &[Value],
) -> (Vec<String>, usize) {
    let mut errors = Vec::new();
    let mut excl_count = 0;
    for f in repo_files {
        if reached.contains(f) {
            continue;
        }
        let matched = defs.iter().any(|entry| {
            entry
                .get("scope")
                .and_then(|s| s.as_str())
                .is_some_and(|sc| matches_scope(f, sc))
        });
        if matched {
            excl_count += 1;
        } else {
            errors.push(format!(
                "Unclassified repository file (neither reachable nor excluded): {f}"
            ));
        }
    }
    (errors, excl_count)
}

pub(super) fn check_excluded(defs: &[Value], reached: &BTreeSet<String>) -> Vec<String> {
    let mut errors = Vec::new();
    for entry in defs {
        let scope = entry.get("scope").and_then(|s| s.as_str()).unwrap_or("");
        if scope == "*" {
            continue;
        }
        for src in reached {
            if matches_scope(src, scope) {
                errors.push(format!(
                    "Exclusion violation: {src} is reachable but matches scope {scope}"
                ));
            }
        }
    }
    errors
}

fn next_quoted(s: &str) -> Option<(&str, &str)> {
    let start = s.find('"')? + 1;
    let rest = &s[start..];
    let end = rest.find('"')?;
    Some((&rest[..end], &rest[end + 1..]))
}

type RegTuple = (usize, String, String, usize, String);

fn parse_entry(line: &str, errors: &mut Vec<String>) -> Option<RegTuple> {
    let rest = line.trim().strip_prefix("mkEntry")?;
    let q1 = rest.find('"')?;
    let idx: usize = rest[..q1].trim().parse().ok()?;
    let (fid, r1) = next_quoted(&rest[q1..])?;
    let (ffile, r2) = next_quoted(r1)?;
    let q3 = r2.find('"')?;
    let fline: usize = r2[..q3].trim().parse().ok()?;
    let (fname, r3) = next_quoted(&r2[q3..])?;
    let (ldef, r4) = next_quoted(r3)?;
    let (lthm, r5) = next_quoted(r4)?;
    let mut syms = r5
        .trim()
        .trim_end_matches(',')
        .split_whitespace()
        .filter_map(|t| t.strip_prefix('@'));
    let (ldef_sym, lthm_sym) = (syms.next()?, syms.next()?);
    if ldef != ldef_sym || lthm != lthm_sym {
        errors.push(format!(
            "Lean proof registry symbol mismatch for {fid}: ({ldef}, {lthm}) vs (@{ldef_sym}, \
             @{lthm_sym})"
        ));
    }
    if matches!(lthm, "True" | "True.intro" | "trivial") {
        errors.push(format!(
            "Lean proof registry uses trivial theorem {lthm} for {fid}"
        ));
    }
    Some((
        idx,
        fid.to_owned(),
        ffile.to_owned(),
        fline,
        fname.to_owned(),
    ))
}

fn check_olean(reg_path: &Path, olean_path: &Path, errors: &mut Vec<String>) {
    if !olean_path.is_file() {
        errors.push(format!(
            "Compiled Lean proof registry artifact missing: {}",
            olean_path.display()
        ));
        return;
    }
    let Ok(reg_m) = fs::metadata(reg_path).and_then(|m| m.modified()) else {
        return;
    };
    let Ok(ol_m) = fs::metadata(olean_path).and_then(|m| m.modified()) else {
        return;
    };
    if ol_m + Duration::from_millis(1) < reg_m {
        errors.push(
            "Compiled Lean proof registry artifact is stale relative to Registry.lean".to_owned(),
        );
    }
}

fn func_tuple(i: usize, f: &Value) -> RegTuple {
    let get_s = |k: &str| match f.get(k) {
        Some(v) => v.as_str().unwrap_or("").to_owned(),
        None => String::new(),
    };
    let ln = match f.get("line") {
        Some(v) => v.as_u64().unwrap_or(0) as usize,
        None => 0,
    };
    (i, get_s("id"), get_s("file"), ln, get_s("name"))
}

pub(super) fn check_registry(root: &Path, manifest_funcs: &[Value]) -> Vec<String> {
    let reg_path = root.join("research/lean/BauxResearch/Semantics/Registry.lean");
    let olean_path =
        root.join("research/lean/.lake/build/lib/lean/BauxResearch/Semantics/Registry.olean");
    let Ok(text) = fs::read_to_string(&reg_path) else {
        return vec![format!(
            "Lean proof registry missing on disk: {}",
            reg_path.display()
        )];
    };
    let mut errors = Vec::new();
    if text
        .split(|c: char| !c.is_alphanumeric() && c != '_')
        .any(|w| matches!(w, "sorry" | "axiom"))
    {
        errors.push("Lean proof registry contains forbidden sorry or axiom".to_owned());
    }
    let mut recorded = Vec::new();
    for line in text.lines() {
        if let Some(row) = parse_entry(line, &mut errors) {
            recorded.push(row);
        }
    }
    if recorded.len() != manifest_funcs.len() {
        errors.push(format!(
            "Lean proof registry count ({}) differs from reachable functions ({})",
            recorded.len(),
            manifest_funcs.len()
        ));
    }
    let expected: Vec<RegTuple> = manifest_funcs
        .iter()
        .enumerate()
        .map(|(i, f)| func_tuple(i, f))
        .collect();
    if recorded != expected {
        errors.push("Lean proof registry entries differ from manifest reachable_functions".into());
    }
    check_olean(&reg_path, &olean_path, &mut errors);
    errors
}
