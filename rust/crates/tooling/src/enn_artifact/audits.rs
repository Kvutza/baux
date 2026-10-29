use baux::artifact::json::{self, Value, json};
use baux::artifact::runs::digest;
use std::collections::BTreeMap;
use std::fs;
use std::io::Write;
use std::path::Path;
use std::process::{Command, Stdio};

const AUDIT_DECLS: &[&str] = &[
    "BauxResearch.Semantics.Registry.fullCount",
    "BauxResearch.Extraction.Optimizer.askTurbo",
    "BauxResearch.Extraction.Optimizer.askPhase",
    "BauxResearch.Extraction.Optimizer.tellWrap",
    "BauxResearch.Extraction.Optimizer.tellPhase",
    "BauxResearch.Extraction.Optimizer.drawFast",
    "BauxResearch.Extraction.Optimizer.muPredict",
    "BauxResearch.Extraction.Hash.pureSplitmix",
    "BauxResearch.Extraction.Hash.pureBox",
    "BauxResearch.Extraction.Hash.pureIndex",
    "BauxResearch.Extraction.Hash.pureQuery",
    "BauxResearch.Semantics.Bpann.bpannCount",
    "BauxResearch.Extraction.Box.equivalent",
    "BauxResearch.Semantics.Numerical.rustAccept",
    "BauxResearch.Semantics.Numerical.sourceMargin",
];

fn normalize_path(fname: &str) -> String {
    for pfx in ["rust/crates/", "crates/"] {
        if let Some(idx) = fname.find(pfx) {
            return format!("rust/crates/{}", &fname[idx + pfx.len()..]);
        }
    }
    fname.to_owned()
}

fn item_span(meta: &Value, files: &[String]) -> (String, (u64, u64)) {
    let span_raw = &meta["span"];
    let span = if !span_raw["Untagged"].is_null() {
        &span_raw["Untagged"]["data"]
    } else {
        &span_raw["data"]
    };
    let fid = span["file_id"].as_u64().unwrap_or(0) as usize;
    let fname = files.get(fid).map_or("", String::as_str);
    let beg = span["beg"]["line"].as_u64().unwrap_or(0);
    let end = span["end"]["line"].as_u64().unwrap_or(0);
    (normalize_path(fname), (beg, end))
}

fn record_decl(
    fn_decl: &Value,
    files: &[String],
    prefix: &str,
    spans: &mut BTreeMap<String, Vec<(u64, u64)>>,
) -> (u64, u64, u64) {
    let meta = &fn_decl["item_meta"];
    let (rel, line_span) = item_span(meta, files);
    let body = &fn_decl["body"];
    if meta["is_local"].as_bool() == Some(true) {
        if !body["Structured"].is_null() {
            spans.entry(rel).or_default().push(line_span);
            return (1, 0, 0);
        }
        if !body["Error"].is_null() {
            return (0, 1, 0);
        }
        return (0, 0, 0);
    }
    if rel.starts_with(prefix) {
        spans.entry(rel).or_default().push(line_span);
        return (0, 0, 1);
    }
    (0, 0, 0)
}

fn file_names(tr: &Value) -> Vec<String> {
    let Some(seq) = tr["files"].as_seq() else {
        return Vec::new();
    };
    seq.iter()
        .map(|f| f["name"]["Local"].as_str().unwrap_or("").to_owned())
        .collect()
}

fn covers_fn(mf: &Value, spans: &BTreeMap<String, Vec<(u64, u64)>>) -> bool {
    let file = mf["file"].as_str().unwrap_or("");
    let line = mf["line"].as_u64().unwrap_or(0);
    let Some(list) = spans.get(file) else {
        return false;
    };
    list.iter().any(|&(b, e)| b <= line && line <= e)
}

fn crate_llbc(
    probe_dir: &Path,
    cov: &Value,
    crate_name: &str,
    filename: &str,
) -> Result<Value, String> {
    let path = probe_dir.join(filename);
    let raw = fs::read(&path)
        .map_err(|_| format!("missing full-crate LLBC artifact: {}", path.display()))?;
    let data: Value = json::from_slice(&raw).map_err(|e| e.to_string())?;
    if data["has_errors"].as_bool() != Some(false) {
        return Err(format!("{filename} reports has_errors != false"));
    }
    let tr = &data["translated"];
    let files = file_names(tr);
    let prefix = format!("rust/crates/{crate_name}/");
    let mut spans = BTreeMap::<String, Vec<(u64, u64)>>::new();
    let mut counts = (0_u64, 0_u64, 0_u64);
    if let Some(decls) = tr["fun_decls"].as_seq() {
        for decl in decls.iter().filter(|d| !d.is_null()) {
            let (s, e, c) = record_decl(decl, &files, &prefix, &mut spans);
            counts = (counts.0 + s, counts.1 + e, counts.2 + c);
        }
    }
    if let Some(traits) = tr["trait_decls"].as_seq() {
        for td in traits.iter().filter(|t| !t.is_null()) {
            if td["item_meta"]["is_local"].as_bool() == Some(true) {
                let (rel, span) = item_span(&td["item_meta"], &files);
                spans.entry(rel).or_default().push(span);
            }
        }
    }
    if counts.1 != 0 {
        return Err(format!(
            "{filename} contains {} function body errors",
            counts.1
        ));
    }
    let Some(all_fns) = cov["reachable_functions"].as_seq() else {
        return Err("missing reachable_functions".into());
    };
    let manifest_fns = all_fns
        .iter()
        .filter(|f| {
            f["language"].as_str() == Some("rust")
                && f["file"].as_str().unwrap_or("").starts_with(&prefix)
        })
        .collect::<Vec<_>>();
    let covered = manifest_fns
        .iter()
        .filter(|mf| covers_fn(mf, &spans))
        .count();
    if covered != manifest_fns.len() {
        return Err(format!(
            "{filename} covers only {covered}/{} manifest functions",
            manifest_fns.len()
        ));
    }
    Ok(json!({
        "path": format!(".cache/baux/aeneas/probe-full-upstream/{filename}"),
        "sha256": digest(&raw),
        "bytes": raw.len() as u64,
        "has_errors": false,
        "local_structured_functions": counts.0,
        "local_error_functions": counts.1,
        "extern_c_declarations": counts.2,
        "manifest_functions_covered": covered as u64,
        "manifest_functions_total": manifest_fns.len() as u64,
        "target_information": tr["target_information"].clone(),
    }))
}

pub fn verify_llbc(root: &Path) -> Result<Value, String> {
    let probe_dir = root.join(".cache/baux/aeneas/probe-full-upstream");
    let text = fs::read_to_string(root.join("research/upstream/coverage_manifest.json"))
        .map_err(|e| e.to_string())?;
    let cov: Value = json::from_str(&text).map_err(|e| e.to_string())?;
    let mut summary = json::Map::new();
    for (crate_name, filename) in [
        ("bpann", "bpann.llbc"),
        ("ennbo", "ennbo.llbc"),
        ("enn-py", "enn-py.llbc"),
    ] {
        summary.insert(
            crate_name,
            crate_llbc(&probe_dir, &cov, crate_name, filename)?,
        );
    }
    Ok(Value::from(summary))
}

fn parse_axiom(line: &str, per_decl: &mut json::Map, custom: &mut Vec<String>) -> usize {
    if line.contains("does not depend on any axioms") {
        if let Some(decl) = line.split('\'').nth(1) {
            per_decl.insert(decl, json!([]));
        }
        return 0;
    }
    if !line.contains("depends on axioms:") {
        return 0;
    }
    let Some(decl) = line.split('\'').nth(1) else {
        return 0;
    };
    let bracket = line
        .split_once('[')
        .and_then(|(_, rest)| rest.rsplit_once(']'))
        .map_or("", |(inner, _)| inner);
    let axs = bracket
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(str::to_owned)
        .collect::<Vec<_>>();
    for ax in &axs {
        let allowed = matches!(ax.as_str(), "propext" | "Classical.choice" | "Quot.sound");
        if ax != "sorryAx" && !allowed && !custom.contains(ax) {
            custom.push(ax.clone());
        }
    }
    let sorry = axs.iter().filter(|a| a.as_str() == "sorryAx").count();
    per_decl.insert(decl, json!(axs));
    sorry
}

pub fn audit_axioms(root: &Path) -> Result<Value, String> {
    let mut stdin_text = String::from("import BauxResearch\n");
    for decl in AUDIT_DECLS {
        stdin_text.push_str(&format!("#print axioms {decl}\n"));
    }
    let mut child = Command::new("lake")
        .current_dir(root.join("research/lean"))
        .args(["env", "lean", "--stdin"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| e.to_string())?;
    if let Some(mut stdin) = child.stdin.take() {
        stdin
            .write_all(stdin_text.as_bytes())
            .map_err(|e| e.to_string())?;
    }
    let out = child.wait_with_output().map_err(|e| e.to_string())?;
    if !out.status.success() {
        return Err(format!(
            "lean axiom audit failed: {}",
            String::from_utf8_lossy(&out.stderr)
        ));
    }
    let mut per_decl = json::Map::new();
    let mut custom = Vec::new();
    let mut sorry_count = 0_usize;
    for raw in String::from_utf8_lossy(&out.stdout).lines() {
        let line = raw.trim();
        if !line.is_empty() {
            sorry_count += parse_axiom(line, &mut per_decl, &mut custom);
        }
    }
    if sorry_count != 0 || !custom.is_empty() {
        return Err(format!(
            "Lean axiom audit failed: sorry_count={sorry_count}, custom_axioms={custom:?}"
        ));
    }
    Ok(json!({
        "checked_declarations": AUDIT_DECLS,
        "declaration_axioms": Value::from(per_decl),
        "allowed_standard_axioms": ["Classical.choice", "Quot.sound", "propext"],
        "custom_axioms": custom,
        "sorry_count": sorry_count as u64,
    }))
}
