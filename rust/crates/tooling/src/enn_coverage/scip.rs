use crate::radar_scip::scip_graph::body_end;
use crate::radar_scip::scip_pb::{DEFINITION, Document, Occurrence, load_index};
use baux::artifact::json::Value;
use baux::artifact::toml;
use baux::artifact::wire_json;
use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::fs;
use std::path::{Path, PathBuf};

pub(super) type ScipOcc = Occurrence;
pub(super) type ScipDoc = Document;

pub(super) struct RustReach {
    pub(super) funcs: Vec<Value>,
    pub(super) files: BTreeSet<String>,
    pub(super) bridges: Vec<Value>,
    pub(super) docs: Vec<ScipDoc>,
}

type DefMap = BTreeMap<String, Vec<(String, i32)>>;
type EdgeMap = BTreeMap<String, BTreeSet<String>>;

fn load_scip(path: &Path) -> Result<Vec<ScipDoc>, String> {
    Ok(load_index(path)?.documents)
}

fn find_scip(root: &Path, enn_root: &Path) -> Result<PathBuf, String> {
    for d in [
        root.join(".cache/baux/radar"),
        enn_root.join(".cache/baux/radar"),
    ] {
        let mut matches: Vec<_> = fs::read_dir(&d)
            .into_iter()
            .flatten()
            .flatten()
            .map(|e| e.path())
            .filter(|p| {
                p.file_name()
                    .and_then(|s| s.to_str())
                    .is_some_and(|n| n.starts_with("enn-") && n.ends_with(".scip"))
            })
            .collect();
        matches.sort();
        if let Some(first) = matches.into_iter().next().or_else(|| {
            let local = d.join("rust.scip");
            local.is_file().then_some(local)
        }) {
            return Ok(first);
        }
    }
    Err(format!(
        "No SCIP index found in {}",
        root.join(".cache/baux/radar").display()
    ))
}

fn doc_spans(raw_defs: &[(i32, String)], src_lines: &[&str]) -> Vec<(i32, i32, String)> {
    raw_defs
        .iter()
        .enumerate()
        .map(|(idx, (st, sym))| {
            let st_u = (*st).max(0) as usize;
            let ed = if st_u < src_lines.len() {
                body_end(src_lines, st_u) as i32
            } else {
                raw_defs.get(idx + 1).map_or(*st + 1000, |(n, _)| n - 1)
            };
            (*st, (*st).max(ed), sym.clone())
        })
        .collect()
}

fn find_enclosing<'a>(spans: &'a [(i32, i32, String)], line: i32) -> Option<&'a str> {
    spans
        .iter()
        .rev()
        .find(|(st, ed, _)| *st <= line && line <= *ed)
        .map(|(_, _, sym)| sym.as_str())
}

fn collect_defs(docs: &[ScipDoc]) -> (BTreeMap<String, String>, DefMap) {
    let mut sym_names = BTreeMap::new();
    let mut sym_defs: DefMap = BTreeMap::new();
    for doc in docs {
        if !doc.path.contains("/src/") {
            continue;
        }
        for info in &doc.symbols {
            if !info.symbol.starts_with("local ") {
                sym_names.insert(info.symbol.clone(), info.display.clone());
            }
        }
        for occ in &doc.occurrences {
            if (occ.roles & DEFINITION) != 0 && !occ.symbol.starts_with("local ") {
                let line = occ.range.first().copied().unwrap_or(0);
                sym_defs
                    .entry(occ.symbol.clone())
                    .or_default()
                    .push((doc.path.clone(), line));
            }
        }
    }
    (sym_names, sym_defs)
}

fn doc_edges(
    occs: &[ScipOcc],
    spans: &[(i32, i32, String)],
    known: &BTreeSet<String>,
    callees: &mut EdgeMap,
) {
    for occ in occs {
        if (occ.roles & DEFINITION) != 0
            || occ.symbol.starts_with("local ")
            || !known.contains(&occ.symbol)
        {
            continue;
        }
        let line = occ.range.first().copied().unwrap_or(0);
        if let Some(enc) = find_enclosing(spans, line)
            && enc != occ.symbol
        {
            callees
                .entry(enc.to_owned())
                .or_default()
                .insert(occ.symbol.clone());
        }
    }
}

fn rust_graph(
    enn_root: &Path,
    docs: &[ScipDoc],
) -> (BTreeMap<String, String>, DefMap, EdgeMap, BTreeSet<String>) {
    let mut callees = EdgeMap::new();
    let (sym_names, sym_defs) = collect_defs(docs);
    let known: BTreeSet<String> = sym_defs.keys().chain(sym_names.keys()).cloned().collect();
    for doc in docs {
        if !doc.path.contains("/src/") {
            continue;
        }
        let Ok(text) = fs::read_to_string(enn_root.join(&doc.path)) else {
            continue;
        };
        let src_lines: Vec<&str> = text.lines().collect();
        let mut raw_defs: Vec<(i32, String)> = doc
            .occurrences
            .iter()
            .filter(|o| (o.roles & DEFINITION) != 0 && !o.symbol.starts_with("local "))
            .map(|o| (o.range.first().copied().unwrap_or(0), o.symbol.clone()))
            .collect();
        raw_defs.sort();
        let spans = doc_spans(&raw_defs, &src_lines);
        doc_edges(&doc.occurrences, &spans, &known, &mut callees);
    }
    (sym_names, sym_defs, callees, known)
}

fn bridge_dyn(known: &BTreeSet<String>, callees: &mut EdgeMap) -> Vec<Value> {
    let mut bridges = Vec::new();
    for sym in known {
        let Some((_, method)) = sym.split_once("surrogate/Surrogate#") else {
            continue;
        };
        if !sym.ends_with("().") {
            continue;
        }
        let pattern = format!("[ENNSurrogate][Surrogate]{method}");
        let Some(target) = known.iter().find(|c| c.contains(&pattern)).cloned() else {
            continue;
        };
        callees
            .entry(sym.clone())
            .or_default()
            .insert(target.clone());
        bridges.push(wire_json!({
            "caller_trait": "Surrogate",
            "method": method.trim_end_matches("()."),
            "target_impl": "ENNSurrogate",
            "target_symbol": target,
            "trait_symbol": sym,
        }));
    }
    bridges
}

fn opt_roots(enn_root: &Path) -> Result<Vec<String>, String> {
    let text = fs::read_to_string(enn_root.join("Cargo.toml"))
        .map_err(|e| format!("read Cargo.toml: {e}"))?;
    let val: Value = toml::from_str(&text).map_err(|e| format!("parse Cargo.toml: {e}"))?;
    let ver = val
        .get("workspace")
        .and_then(|w| w.get("package"))
        .and_then(|p| p.get("version"))
        .and_then(|v| v.as_str())
        .ok_or("missing workspace.package.version in Cargo.toml")?;
    Ok(vec![
        format!("rust-analyzer cargo ennbo {ver} optimizer_factory/create_optimizer_enn()."),
        format!(
            "rust-analyzer cargo ennbo {ver} \
             optimizer_factory/create_optimizer_enn_with_overrides()."
        ),
        format!("rust-analyzer cargo ennbo {ver} optimizer/impl#[Optimizer]ask()."),
        format!("rust-analyzer cargo ennbo {ver} optimizer/impl#[Optimizer]tell()."),
        format!("rust-analyzer cargo ennbo-py {ver} py_optimizer/create_optimizer_enn_py()."),
        format!("rust-analyzer cargo ennbo-py {ver} py_optimizer/impl#[PyOptimizer]ask()."),
        format!("rust-analyzer cargo ennbo-py {ver} py_optimizer/impl#[PyOptimizer]tell()."),
    ])
}

fn run_bfs(
    roots: &[String],
    known: &BTreeSet<String>,
    callees: &EdgeMap,
) -> (
    BTreeSet<String>,
    BTreeMap<String, usize>,
    BTreeMap<String, String>,
) {
    let mut visited = BTreeSet::new();
    let mut depth_map = BTreeMap::new();
    let mut root_map = BTreeMap::new();
    let mut queue = VecDeque::new();
    for r in roots {
        if known.contains(r) {
            visited.insert(r.clone());
            depth_map.insert(r.clone(), 0);
            let short = r.split_whitespace().last().unwrap_or("root").to_owned();
            root_map.insert(r.clone(), short);
            queue.push_back(r.clone());
        }
    }
    while let Some(curr) = queue.pop_front() {
        let d = depth_map[&curr];
        let rt = root_map[&curr].clone();
        let Some(nexts) = callees.get(&curr) else {
            continue;
        };
        for nxt in nexts {
            if visited.insert(nxt.clone()) {
                depth_map.insert(nxt.clone(), d + 1);
                root_map.insert(nxt.clone(), rt.clone());
                queue.push_back(nxt.clone());
            }
        }
    }
    (visited, depth_map, root_map)
}

pub(super) fn trace_rust(root: &Path, enn_root: &Path) -> Result<RustReach, String> {
    let scip_path = find_scip(root, enn_root)?;
    let docs = load_scip(&scip_path)?;
    let (sym_names, sym_defs, mut callees, known) = rust_graph(enn_root, &docs);
    let bridges = bridge_dyn(&known, &mut callees);
    let roots = opt_roots(enn_root)?;
    let missing: Vec<&String> = roots
        .iter()
        .filter(|r| !sym_defs.contains_key(*r))
        .collect();
    if !missing.is_empty() {
        return Err(format!(
            "Required optimizer roots missing from SCIP index: {missing:?}"
        ));
    }
    let (visited, depth_map, root_map) = run_bfs(&roots, &known, &callees);
    let mut files = BTreeSet::new();
    let mut funcs = Vec::new();
    for s in &visited {
        let Some(defs) = sym_defs.get(s) else {
            continue;
        };
        for (dpath, line) in defs {
            files.insert(dpath.clone());
            if s.ends_with("().") {
                let name = sym_names
                    .get(s)
                    .cloned()
                    .unwrap_or_else(|| s.split_whitespace().last().unwrap_or("").to_owned());
                funcs.push(wire_json!({
                    "depth": depth_map.get(s).copied().unwrap_or(0),
                    "file": dpath,
                    "id": s,
                    "language": "rust",
                    "line": line + 1,
                    "name": name,
                    "root": root_map.get(s).map(String::as_str).unwrap_or("root"),
                }));
            }
        }
    }
    for inc in [
        "rust/crates/ennbo/src/ennbo_build_api.inc.rs",
        "rust/crates/enn-py/src/enn_py_build_api.inc.rs",
    ] {
        if enn_root.join(inc).is_file() {
            files.insert(inc.to_owned());
        }
    }
    Ok(RustReach {
        funcs,
        files,
        bridges,
        docs,
    })
}

fn doc_refs(
    enn_root: &Path,
    doc: &ScipDoc,
    reached: &BTreeSet<String>,
    refs: &mut BTreeMap<(String, Vec<i32>, String), (String, i32, i32, String, Value)>,
) {
    let Ok(text) = fs::read_to_string(enn_root.join(&doc.path)) else {
        return;
    };
    let lines: Vec<&str> = text.lines().collect();
    let mut defs: Vec<(i32, String)> = doc
        .occurrences
        .iter()
        .filter(|o| {
            !o.range.is_empty() && (o.roles & DEFINITION) != 0 && !o.symbol.starts_with("local ")
        })
        .map(|o| (o.range[0], o.symbol.clone()))
        .collect();
    defs.sort();
    let spans = doc_spans(&defs, &lines);
    for occ in &doc.occurrences {
        if (occ.roles & DEFINITION) != 0 || occ.range.len() < 2 || occ.symbol.starts_with("local ")
        {
            continue;
        }
        let parts: Vec<&str> = occ.symbol.splitn(5, ' ').collect();
        if parts.len() != 5 || parts[0] != "rust-analyzer" || parts[1] != "cargo" {
            continue;
        }
        let (pkg, ver) = (parts[2], parts[3]);
        if matches!(pkg, "ennbo" | "ennbo-py" | "ennbo-bpann") {
            continue;
        }
        let owner = find_enclosing(&spans, occ.range[0]);
        let scope = if owner.is_some_and(|o| reached.contains(o)) {
            "reachable_function"
        } else {
            "other_source_scope"
        };
        let kind = if occ.symbol.ends_with("().") {
            "function_reference"
        } else {
            "symbol_reference"
        };
        let (ln, col) = (occ.range[0] + 1, occ.range[1] + 1);
        let rec = wire_json!({
            "column": col,
            "enclosing_symbol": owner.map(Value::from).unwrap_or_else(Value::null),
            "file": doc.path,
            "kind": kind,
            "line": ln,
            "package": pkg,
            "scope": scope,
            "symbol": occ.symbol,
            "version": ver,
        });
        refs.insert(
            (doc.path.clone(), occ.range.clone(), occ.symbol.clone()),
            (doc.path.clone(), ln, col, occ.symbol.clone(), rec),
        );
    }
}

pub(super) fn rust_refs(
    enn_root: &Path,
    docs: &[ScipDoc],
    funcs: &[Value],
    sources: &BTreeSet<String>,
) -> Vec<Value> {
    let reached: BTreeSet<String> = funcs
        .iter()
        .filter(|f| f.get("language").and_then(|v| v.as_str()) == Some("rust"))
        .filter_map(|f| f.get("id").and_then(|v| v.as_str()).map(str::to_owned))
        .collect();
    let mut refs = BTreeMap::new();
    for doc in docs {
        if sources.contains(&doc.path) && doc.path.ends_with(".rs") {
            doc_refs(enn_root, doc, &reached, &mut refs);
        }
    }
    let mut sorted: Vec<_> = refs.into_values().collect();
    sorted.sort_by(|a, b| (&a.0, a.1, a.2, &a.3).cmp(&(&b.0, b.1, b.2, &b.3)));
    sorted.into_iter().map(|(_, _, _, _, r)| r).collect()
}
