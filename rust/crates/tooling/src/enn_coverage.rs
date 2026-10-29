#[path = "enn_coverage/adversary.rs"]
mod adversary;
#[path = "enn_coverage/exclude.rs"]
mod exclude;
#[path = "enn_coverage/python.rs"]
mod python;
#[path = "enn_coverage/scip.rs"]
mod scip;

use baux::artifact::json::{self, Map, Value};
use baux::artifact::runs::digest;
use baux::artifact::toml;
use baux::artifact::wire_json;
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::Path;
use std::process::Command;

struct PinInfo {
    revision: String,
    tree: String,
    version: String,
    root_lock: String,
    rust_lock: String,
    pyproject: String,
}

struct DerivedState {
    pin: PinInfo,
    checkout_errs: Vec<String>,
    repo_files: Vec<String>,
    reached_funcs: Vec<Value>,
    reached_sources: BTreeSet<String>,
    bridges: Vec<Value>,
    sources_map: Value,
    total_lines: usize,
    rust_count: usize,
    python_count: usize,
    boundary: Value,
}

fn file_hash(path: &Path) -> String {
    fs::read(path)
        .map(|b| digest(&b))
        .unwrap_or_else(|_| String::new())
}

fn count_lines(path: &Path) -> usize {
    fs::read(path)
        .map(|b| String::from_utf8_lossy(&b).lines().count())
        .unwrap_or(0)
}

fn canon_val(v: &Value) -> Value {
    if let Some(map) = v.as_map() {
        let mut sorted = BTreeMap::new();
        for (k, val) in map {
            if let Some(ks) = k.as_str() {
                sorted.insert(ks.to_owned(), canon_val(val));
            }
        }
        let mut out = Map::new();
        for (k, val) in sorted {
            out.insert(k, val);
        }
        return Value::from(out);
    }
    if let Some(seq) = v.as_seq() {
        let out: Vec<Value> = seq.iter().map(canon_val).collect();
        return Value::from(out);
    }
    v.clone()
}

fn read_pin(root: &Path) -> Result<PinInfo, String> {
    let p = root.join("research/upstream/enn.lock.toml");
    let text = fs::read_to_string(&p).map_err(|e| format!("read {}: {e}", p.display()))?;
    let val: Value = toml::from_str(&text).map_err(|e| format!("parse {}: {e}", p.display()))?;
    let get_s = |sec: &str, k: &str| -> String {
        val.get(sec)
            .and_then(|s| s.get(k))
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_owned()
    };
    Ok(PinInfo {
        revision: get_s("repository", "revision"),
        tree: get_s("repository", "tree"),
        version: get_s("repository", "version"),
        root_lock: get_s("inputs", "root_cargo_lock_sha256"),
        rust_lock: get_s("inputs", "rust_cargo_lock_sha256"),
        pyproject: get_s("inputs", "pyproject_sha256"),
    })
}

fn check_inputs(enn_root: &Path, pin: &PinInfo) -> Vec<String> {
    let mut errs = Vec::new();
    for (rel, exp) in [
        ("Cargo.lock", &pin.root_lock),
        ("rust/Cargo.lock", &pin.rust_lock),
        ("pyproject.toml", &pin.pyproject),
    ] {
        if &file_hash(&enn_root.join(rel)) != exp {
            errs.push(format!("Pinned input SHA256 mismatch: {rel}"));
        }
    }
    let cargo_txt = fs::read_to_string(enn_root.join("Cargo.toml")).unwrap_or_default();
    let ws: Value = toml::from_str(&cargo_txt).unwrap_or_else(|_| Value::null());
    let ver = ws
        .get("workspace")
        .and_then(|w| w.get("package"))
        .and_then(|p| p.get("version"))
        .and_then(|v| v.as_str())
        .unwrap_or("");
    if ver != pin.version {
        errs.push("Pinned ENN version differs from the upstream workspace".to_owned());
    }
    errs
}

fn check_git(enn_root: &Path, pin: &PinInfo) -> Vec<String> {
    let mut errs = check_inputs(enn_root, pin);
    let rev = Command::new("git")
        .arg("-C")
        .arg(enn_root)
        .args(["rev-parse", "HEAD", "HEAD^{tree}"])
        .output();
    let rev_ok = rev.is_ok_and(|o| {
        let lines: Vec<&str> = std::str::from_utf8(&o.stdout)
            .unwrap_or("")
            .lines()
            .collect();
        o.status.success() && lines == [pin.revision.as_str(), pin.tree.as_str()]
    });
    if !rev_ok {
        errs.push("Upstream checkout identity differs from the pinned revision and tree".into());
    }
    let clean = Command::new("git")
        .arg("-C")
        .arg(enn_root)
        .args(["diff", "--quiet", "HEAD", "--"])
        .status();
    if !clean.is_ok_and(|s| s.success()) {
        errs.push("Upstream checkout has tracked source changes or cannot be inspected".into());
    }
    errs
}

fn locked_pkgs(enn_root: &Path) -> Result<Vec<Value>, String> {
    let text = fs::read_to_string(enn_root.join("Cargo.lock"))
        .map_err(|e| format!("read Cargo.lock: {e}"))?;
    let val: Value = toml::from_str(&text).map_err(|e| format!("parse Cargo.lock: {e}"))?;
    let Some(pkgs) = val.get("package").and_then(|p| p.as_seq()) else {
        return Ok(Vec::new());
    };
    let mut rows = Vec::with_capacity(pkgs.len());
    for pkg in pkgs {
        let name = pkg.get("name").and_then(|v| v.as_str()).unwrap_or("");
        let version = pkg.get("version").and_then(|v| v.as_str()).unwrap_or("");
        let source = pkg
            .get("source")
            .and_then(|v| v.as_str())
            .unwrap_or("upstream_workspace");
        let checksum = pkg.get("checksum").cloned().unwrap_or_else(Value::null);
        let mut deps: Vec<String> = match pkg.get("dependencies").and_then(|v| v.as_seq()) {
            Some(seq) => seq
                .iter()
                .filter_map(|d| d.as_str().map(str::to_owned))
                .collect(),
            None => Vec::new(),
        };
        deps.sort();
        rows.push((
            name.to_owned(),
            version.to_owned(),
            source.to_owned(),
            wire_json!({
                "checksum": checksum,
                "dependencies": deps,
                "name": name,
                "source": source,
                "version": version,
            }),
        ));
    }
    rows.sort_by(|a, b| (&a.0, &a.1, &a.2).cmp(&(&b.0, &b.1, &b.2)));
    Ok(rows.into_iter().map(|(_, _, _, r)| r).collect())
}

fn py_reqs(enn_root: &Path) -> Result<Value, String> {
    let text = fs::read_to_string(enn_root.join("pyproject.toml"))
        .map_err(|e| format!("read pyproject.toml: {e}"))?;
    let val: Value = toml::from_str(&text).map_err(|e| format!("parse pyproject.toml: {e}"))?;
    let proj = val.get("project");
    let req_py = proj
        .and_then(|p| p.get("requires-python"))
        .cloned()
        .unwrap_or_else(Value::null);
    let deps = proj
        .and_then(|p| p.get("dependencies"))
        .cloned()
        .unwrap_or_else(|| Value::from(Vec::<Value>::new()));
    let opt_deps = proj
        .and_then(|p| p.get("optional-dependencies"))
        .map(canon_val)
        .unwrap_or_else(|| Value::from(Map::new()));
    let build_sys = val
        .get("build-system")
        .map(canon_val)
        .unwrap_or_else(|| Value::from(Map::new()));
    Ok(wire_json!({
        "build_system": build_sys,
        "dependencies": deps,
        "optional_dependencies": opt_deps,
        "requires_python": req_py,
        "resolved_environment": "not_captured",
    }))
}

fn build_boundary(
    enn_root: &Path,
    docs: &[scip::ScipDoc],
    funcs: &[Value],
    sources: &BTreeSet<String>,
) -> Result<Value, String> {
    let rust_pkgs = locked_pkgs(enn_root)?;
    let rust_refs = scip::rust_refs(enn_root, docs, funcs, sources);
    let py_imps = python::python_imports(enn_root, sources);
    let reqs = py_reqs(enn_root)?;
    let mut inputs = Map::new();
    for rel in ["Cargo.lock", "pyproject.toml", "rust/Cargo.lock"] {
        inputs.insert(rel, Value::from(file_hash(&enn_root.join(rel))));
    }
    Ok(canon_val(&wire_json!({
        "inputs": Value::from(inputs),
        "meaning": "Dependency inventory, not implementation correspondence or executable binding",
        "proof_status": "unproved",
        "python_imports": py_imps,
        "python_requirements": reqs,
        "remaining_obligations": [
            "Prove the semantics of each dependency operation used by the optimizer",
            "Resolve generic dispatch, implicit operators, macros and missing SCIP references",
            "Bind enabled features, target, compiler, standard library and native libraries",
            "Bind Python, NumPy, SciPy and native extension/runtime versions",
            "Account for FFI, libm/BLAS/Faiss, mmap, threads and OS effects",
            "Prove correspondence to the compiled artifact and complete optimizer executions",
        ],
        "rust_lock_scope": "All locked packages; includes build/dev/target-specific dependencies",
        "rust_packages": rust_pkgs,
        "rust_references": rust_refs,
    })))
}

fn build_sources(
    enn_root: &Path,
    reached_sources: &BTreeSet<String>,
    reached_funcs: &[Value],
) -> (Value, usize, usize, usize) {
    let mut counts: BTreeMap<String, usize> = BTreeMap::new();
    for f in reached_funcs {
        let fp = f.get("file").and_then(|v| v.as_str()).unwrap_or("");
        *counts.entry(fp.to_owned()).or_insert(0) += 1;
    }
    let mut map = Map::new();
    let (mut tot_lines, mut r_cnt, mut py_cnt) = (0, 0, 0);
    for rel in reached_sources {
        let full = enn_root.join(rel);
        let lines = count_lines(&full);
        tot_lines += lines;
        let lang = if rel.ends_with(".rs") {
            r_cnt += 1;
            "rust"
        } else {
            py_cnt += 1;
            "python"
        };
        map.insert(
            rel.clone(),
            wire_json!({
                "functions": counts.get(rel).copied().unwrap_or(0),
                "language": lang,
                "lines": lines,
                "sha256": file_hash(&full),
            }),
        );
    }
    (Value::from(map), tot_lines, r_cnt, py_cnt)
}

fn derive_state(root: &Path, enn_root: &Path) -> Result<DerivedState, String> {
    let pin = read_pin(root)?;
    let checkout_errs = check_git(enn_root, &pin);
    let repo_files = exclude::list_files(enn_root);
    let rust_reach = scip::trace_rust(root, enn_root)?;
    let (py_funcs, py_files) = python::trace_python(enn_root);
    let mut reached_funcs = rust_reach.funcs;
    reached_funcs.extend(py_funcs);
    let mut reached_sources = rust_reach.files;
    reached_sources.extend(py_files);
    let (sources_map, total_lines, rust_count, python_count) =
        build_sources(enn_root, &reached_sources, &reached_funcs);
    let boundary = build_boundary(enn_root, &rust_reach.docs, &reached_funcs, &reached_sources)?;
    Ok(DerivedState {
        pin,
        checkout_errs,
        repo_files,
        reached_funcs,
        reached_sources,
        bridges: rust_reach.bridges,
        sources_map,
        total_lines,
        rust_count,
        python_count,
        boundary,
    })
}

fn multiset(items: &[Value]) -> BTreeMap<String, usize> {
    let mut counts = BTreeMap::new();
    for item in items {
        let s = json::to_string(&canon_val(item)).unwrap_or_default();
        *counts.entry(s).or_insert(0) += 1;
    }
    counts
}

fn check_sources(
    enn_root: &Path,
    manifest_sources: Option<&Map>,
    reached: &BTreeSet<String>,
) -> Vec<String> {
    let mut errs = Vec::new();
    let Some(ms) = manifest_sources else {
        return vec!["Missing reachable_sources object".into()];
    };
    for s in reached {
        if !ms.contains_key(s.as_str()) {
            errs.push(format!(
                "Reachable source missing from reachable_sources: {s}"
            ));
        }
    }
    for (k, meta) in ms {
        let Some(rel) = k.as_str() else {
            continue;
        };
        let full = enn_root.join(rel);
        if !full.is_file() {
            errs.push(format!("Source file not found on disk: {rel}"));
            continue;
        }
        let actual = file_hash(&full);
        let expected = meta.get("sha256").and_then(|v| v.as_str()).unwrap_or("");
        if actual != expected {
            errs.push(format!(
                "SHA256 mismatch for {rel}: expected {expected}, got {actual}"
            ));
        }
    }
    errs
}

fn check_funcs(manifest_funcs: &[Value], reached_funcs: &[Value]) -> Vec<String> {
    let mut errs = Vec::new();
    let m_ids: BTreeSet<&str> = manifest_funcs
        .iter()
        .filter_map(|f| f.get("id").and_then(|v| v.as_str()))
        .collect();
    let r_ids: BTreeSet<&str> = reached_funcs
        .iter()
        .filter_map(|f| f.get("id").and_then(|v| v.as_str()))
        .collect();
    let missing: Vec<&&str> = r_ids.difference(&m_ids).collect();
    if !missing.is_empty() {
        let sample: Vec<&str> = missing.iter().take(5).map(|&&s| s).collect();
        errs.push(format!(
            "Missing reachable functions in manifest ({} omitted, e.g. {sample:?})",
            missing.len()
        ));
    }
    if multiset(manifest_funcs) != multiset(reached_funcs) {
        errs.push(
            "Reachable function records differ from derived source locations and reachability"
                .into(),
        );
    }
    errs
}

fn check_head(manifest: &Value, state: &DerivedState) -> Vec<String> {
    let mut errs = state.checkout_errs.clone();
    if manifest.get("manifest_version").and_then(|v| v.as_u64()) != Some(3) {
        errs.push("Unsupported manifest version; expected 3".into());
    }
    let exp_id = wire_json!({
        "component": "upstream_turbo_enn",
        "repository": "https://github.com/yubo-research/enn",
        "revision": state.pin.revision,
        "tree": state.pin.tree,
    });
    let act_id = manifest.get("identity").map(canon_val);
    if act_id.as_ref() != Some(&exp_id) {
        errs.push("Upstream identity differs from the pinned revision and tree".into());
    }
    let act_db = manifest.get("dependency_boundary").map(canon_val);
    if act_db.as_ref() != Some(&state.boundary) {
        errs.push(
            "Dependency boundary differs from locked packages and observed source references"
                .into(),
        );
    }
    errs
}

fn check_manifest(
    root: &Path,
    enn_root: &Path,
    manifest: &Value,
    state: &DerivedState,
) -> Vec<String> {
    let mut errs = check_head(manifest, state);
    let ms_val = manifest.get("reachable_sources").map(canon_val);
    let mf_seq = manifest
        .get("reachable_functions")
        .and_then(|v| v.as_seq())
        .map(|s| s.as_slice())
        .unwrap_or(&[]);
    errs.extend(check_sources(
        enn_root,
        ms_val.as_ref().and_then(|v| v.as_map()),
        &state.reached_sources,
    ));
    errs.extend(check_funcs(mf_seq, &state.reached_funcs));
    let dd_seq = manifest
        .get("dynamic_dispatch")
        .and_then(|v| v.as_seq())
        .map(|s| s.as_slice())
        .unwrap_or(&[]);
    if multiset(dd_seq) != multiset(&state.bridges) {
        errs.push("Dynamic dispatch table differs from derived concrete targets".into());
    }
    if ms_val.as_ref() != Some(&state.sources_map) {
        errs.push(
            "Reachable source records differ from derived hashes, line counts, or function counts"
                .into(),
        );
    }
    let ex_seq = manifest
        .get("exclusions")
        .and_then(|v| v.as_seq())
        .map(|s| s.as_slice())
        .unwrap_or(&[]);
    let (_, std_excl_cnt) = exclude::check_partition(
        &state.repo_files,
        &state.reached_sources,
        &exclude::exclusion_defs(),
    );
    let exp_sum = wire_json!({
        "reachable_python_files": state.python_count,
        "reachable_rust_files": state.rust_count,
        "total_excluded_files": std_excl_cnt,
        "total_reachable_files": state.reached_sources.len(),
        "total_reachable_functions": state.reached_funcs.len(),
        "total_reachable_lines": state.total_lines,
    });
    if manifest.get("summary").map(canon_val).as_ref() != Some(&exp_sum) {
        errs.push("Summary counts differ from the derived source inventory".into());
    }
    let (part_errs, _) =
        exclude::check_partition(&state.repo_files, &state.reached_sources, ex_seq);
    errs.extend(part_errs);
    errs.extend(exclude::check_excluded(ex_seq, &state.reached_sources));
    errs.extend(exclude::check_registry(root, mf_seq));
    errs
}

fn format_manifest(mut manifest: Value) -> Result<String, String> {
    let mut refs_rows = Vec::new();
    let marker = "__BAUX_EXTERNAL_REFERENCES__";
    if let Some(db) = manifest.get_mut("dependency_boundary")
        && let Some(map) = db.as_map_mut()
        && let Some(refs) = map.remove("rust_references")
    {
        if let Some(seq) = refs.as_seq() {
            for r in seq {
                let line = json::to_string(&canon_val(r)).map_err(|e| e.to_string())?;
                refs_rows.push(format!("      {line}"));
            }
        }
        map.insert("rust_references", Value::from(marker));
    }
    let rendered = json::pretty_string(&canon_val(&manifest)).map_err(|e| e.to_string())?;
    let replacement = format!("[\n{}\n    ]", refs_rows.join(",\n"));
    let target = format!("\"{marker}\"");
    Ok(format!("{}\n", rendered.replacen(&target, &replacement, 1)))
}

fn do_generate(root: &Path, manifest_path: &Path, enn_root: &Path) -> Result<(), String> {
    let state = derive_state(root, enn_root)?;
    let excl_defs = exclude::exclusion_defs();
    let (_, excl_cnt) =
        exclude::check_partition(&state.repo_files, &state.reached_sources, &excl_defs);
    let manifest = canon_val(&wire_json!({
        "cfg_and_features": exclude::cfg_features(),
        "dependency_boundary": state.boundary.clone(),
        "dynamic_dispatch": state.bridges.clone(),
        "exclusions": excl_defs,
        "external_calls": exclude::external_calls(),
        "generated_code": exclude::generated_records(enn_root, file_hash),
        "identity": {
            "component": "upstream_turbo_enn",
            "repository": "https://github.com/yubo-research/enn",
            "revision": state.pin.revision.clone(),
            "tree": state.pin.tree.clone(),
        },
        "manifest_version": 3,
        "reachable_functions": state.reached_funcs.clone(),
        "reachable_sources": state.sources_map.clone(),
        "roots": exclude::manifest_roots(),
        "summary": {
            "reachable_python_files": state.python_count,
            "reachable_rust_files": state.rust_count,
            "total_excluded_files": excl_cnt,
            "total_reachable_files": state.reached_sources.len(),
            "total_reachable_functions": state.reached_funcs.len(),
            "total_reachable_lines": state.total_lines,
        },
    }));
    let errs = check_manifest(root, enn_root, &manifest, &state);
    if !errs.is_empty() {
        return Err(format!(
            "Refusing to write invalid coverage evidence:\n{}",
            errs.join("\n")
        ));
    }
    if let Some(parent) = manifest_path.parent() {
        fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    }
    let text = format_manifest(manifest)?;
    let staging = manifest_path.with_extension("json.tmp");
    fs::write(&staging, text).map_err(|e| e.to_string())?;
    fs::rename(&staging, manifest_path).map_err(|e| e.to_string())?;
    println!(
        "baux: generated coverage manifest at {}",
        manifest_path.display()
    );
    println!(
        "Reachable files: {} ({} lines), Functions: {}",
        state.reached_sources.len(),
        state.total_lines,
        state.reached_funcs.len()
    );
    Ok(())
}

fn do_check(root: &Path, manifest_path: &Path, enn_root: &Path) -> Result<(), String> {
    let bytes =
        fs::read(manifest_path).map_err(|e| format!("read {}: {e}", manifest_path.display()))?;
    let raw: Value =
        json::from_slice(&bytes).map_err(|e| format!("parse {}: {e}", manifest_path.display()))?;
    let manifest = canon_val(&raw);
    let state = derive_state(root, enn_root)?;
    let errs = check_manifest(root, enn_root, &manifest, &state);
    if !errs.is_empty() {
        return Err(format!(
            "Drift or omission detected in upstream coverage manifest:\n  {}",
            errs.join("\n  ")
        ));
    }
    let (passed, results) =
        adversary::run_adversary(&manifest, |m| check_manifest(root, enn_root, m, &state));
    if !passed {
        return Err(format!(
            "Adversarial evasion testing failed:\n  {}",
            results.join("\n  ")
        ));
    }
    let n_files = state.reached_sources.len();
    let n_lines = state.total_lines;
    let n_funcs = state.reached_funcs.len();
    println!(
        "baux: coverage manifest is current ({n_files} files, {n_lines} lines, {n_funcs} \
         reachable functions)"
    );
    println!(
        "baux: Lean proof registry verified ({n_funcs}/{n_funcs} functions bound in Registry.lean)"
    );
    println!(
        "baux: adversarial tests passed ({}/{} evasion attempts detected and rejected)",
        results.len(),
        results.len()
    );
    Ok(())
}

pub fn run(root: &Path, check: bool) -> Result<(), String> {
    let manifest_path = root.join("research/upstream/coverage_manifest.json");
    let enn_root = root.join(".cache/baux/upstream/enn");
    if !enn_root.is_dir() {
        return Err(format!(
            "ENN root directory does not exist at {}",
            enn_root.display()
        ));
    }
    if check {
        do_check(root, &manifest_path, &enn_root)
    } else {
        do_generate(root, &manifest_path, &enn_root)
    }
}
