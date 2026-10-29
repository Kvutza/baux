use baux::artifact::json::Value;
use baux::artifact::toml;
use baux::artifact::wire_json;
use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Clone, Default)]
pub(super) struct ScipOcc {
    pub(super) range: Vec<i32>,
    pub(super) symbol: String,
    pub(super) roles: i32,
}

#[derive(Clone, Default)]
pub(super) struct ScipDoc {
    pub(super) path: String,
    pub(super) occs: Vec<ScipOcc>,
    pub(super) syms: Vec<(String, String)>,
}

pub(super) struct RustReach {
    pub(super) funcs: Vec<Value>,
    pub(super) files: BTreeSet<String>,
    pub(super) bridges: Vec<Value>,
    pub(super) docs: Vec<ScipDoc>,
}

type DefMap = BTreeMap<String, Vec<(String, i32)>>;
type EdgeMap = BTreeMap<String, BTreeSet<String>>;

fn read_varint(data: &[u8], pos: &mut usize) -> Result<u64, String> {
    let (mut res, mut shift) = (0_u64, 0_u32);
    loop {
        let b = *data.get(*pos).ok_or("truncated SCIP varint")?;
        *pos += 1;
        res |= u64::from(b & 0x7f) << shift;
        if (b & 0x80) == 0 {
            return Ok(res);
        }
        shift += 7;
    }
}

fn skip_field(data: &[u8], pos: &mut usize, wire: u64) -> Result<(), String> {
    match wire {
        0 => {
            read_varint(data, pos)?;
        }
        1 => *pos += 8,
        5 => *pos += 4,
        2 => {
            let len = read_varint(data, pos)? as usize;
            *pos += len;
        }
        _ => return Err(format!("unsupported SCIP wire {wire}")),
    }
    Ok(())
}

fn unpack_range(data: &[u8], out: &mut Vec<i32>) -> Result<(), String> {
    let mut rpos = 0;
    while rpos < data.len() {
        out.push(read_varint(data, &mut rpos)? as i32);
    }
    Ok(())
}

fn parse_occ(sub: &[u8]) -> Result<ScipOcc, String> {
    let mut opos = 0;
    let mut occ = ScipOcc::default();
    while opos < sub.len() {
        let key = read_varint(sub, &mut opos)?;
        match (key & 7, key >> 3) {
            (2, 1) => {
                let len = read_varint(sub, &mut opos)? as usize;
                unpack_range(&sub[opos..opos + len], &mut occ.range)?;
                opos += len;
            }
            (2, 2) => {
                let len = read_varint(sub, &mut opos)? as usize;
                occ.symbol = String::from_utf8_lossy(&sub[opos..opos + len]).into_owned();
                opos += len;
            }
            (0, 1) => occ.range.push(read_varint(sub, &mut opos)? as i32),
            (0, 3) => occ.roles = read_varint(sub, &mut opos)? as i32,
            (w, _) => skip_field(sub, &mut opos, w)?,
        }
    }
    Ok(occ)
}

fn parse_sym(sub: &[u8]) -> Result<(String, String), String> {
    let mut spos = 0;
    let (mut sym, mut disp) = (String::new(), String::new());
    while spos < sub.len() {
        let key = read_varint(sub, &mut spos)?;
        match (key & 7, key >> 3) {
            (2, 1) => {
                let len = read_varint(sub, &mut spos)? as usize;
                sym = String::from_utf8_lossy(&sub[spos..spos + len]).into_owned();
                spos += len;
            }
            (2, 6) => {
                let len = read_varint(sub, &mut spos)? as usize;
                disp = String::from_utf8_lossy(&sub[spos..spos + len]).into_owned();
                spos += len;
            }
            (w, _) => skip_field(sub, &mut spos, w)?,
        }
    }
    Ok((sym, disp))
}

fn parse_doc(val: &[u8]) -> Result<ScipDoc, String> {
    let mut dpos = 0;
    let mut doc = ScipDoc::default();
    while dpos < val.len() {
        let key = read_varint(val, &mut dpos)?;
        match (key & 7, key >> 3) {
            (2, 1) => {
                let len = read_varint(val, &mut dpos)? as usize;
                doc.path = String::from_utf8_lossy(&val[dpos..dpos + len]).into_owned();
                dpos += len;
            }
            (2, 2) => {
                let len = read_varint(val, &mut dpos)? as usize;
                doc.occs.push(parse_occ(&val[dpos..dpos + len])?);
                dpos += len;
            }
            (2, 3) => {
                let len = read_varint(val, &mut dpos)? as usize;
                doc.syms.push(parse_sym(&val[dpos..dpos + len])?);
                dpos += len;
            }
            (w, _) => skip_field(val, &mut dpos, w)?,
        }
    }
    Ok(doc)
}

fn load_scip(path: &Path) -> Result<Vec<ScipDoc>, String> {
    let data = fs::read(path).map_err(|e| format!("read {}: {e}", path.display()))?;
    let mut pos = 0;
    let mut docs = Vec::new();
    while pos < data.len() {
        let key = read_varint(&data, &mut pos)?;
        if (key & 7) == 2 && (key >> 3) == 2 {
            let len = read_varint(&data, &mut pos)? as usize;
            docs.push(parse_doc(&data[pos..pos + len])?);
            pos += len;
        } else {
            skip_field(&data, &mut pos, key & 7)?;
        }
    }
    Ok(docs)
}

fn scan_dir(dir: &Path) -> Option<PathBuf> {
    let entries = fs::read_dir(dir).ok()?;
    let mut matches = Vec::new();
    for entry in entries.flatten() {
        let p = entry.path();
        let name = p.file_name().and_then(|s| s.to_str()).unwrap_or("");
        if name.starts_with("enn-") && name.ends_with(".scip") {
            matches.push(p);
        }
    }
    matches.sort();
    if let Some(first) = matches.into_iter().next() {
        return Some(first);
    }
    let local = dir.join("rust.scip");
    local.is_file().then_some(local)
}

fn find_scip(root: &Path, enn_root: &Path) -> Result<PathBuf, String> {
    let dirs = [
        root.join(".cache/baux/radar"),
        enn_root.join(".cache/baux/radar"),
    ];
    for d in &dirs {
        if let Some(p) = scan_dir(d) {
            return Ok(p);
        }
    }
    Err(format!("No SCIP index found in {}", dirs[0].display()))
}

fn line_depth(line: &str, mut depth: i32, mut opened: bool) -> (i32, bool) {
    let code = line.split("//").next().unwrap_or(line);
    for ch in code.chars() {
        if ch == '{' {
            depth += 1;
            opened = true;
        } else if ch == '}' {
            depth -= 1;
        }
    }
    (depth, opened)
}

fn body_end(lines: &[&str], start: usize) -> usize {
    let (mut depth, mut opened) = (0, false);
    for (offset, line) in lines[start..].iter().enumerate() {
        (depth, opened) = line_depth(line, depth, opened);
        if opened && depth <= 0 {
            return start + offset;
        }
    }
    lines.len()
}

fn doc_spans(raw_defs: &[(i32, String)], src_lines: &[&str]) -> Vec<(i32, i32, String)> {
    let mut spans = Vec::with_capacity(raw_defs.len());
    for (idx, (st, sym)) in raw_defs.iter().enumerate() {
        let st_u = (*st).max(0) as usize;
        let ed = if st_u < src_lines.len() {
            body_end(src_lines, st_u) as i32
        } else if idx + 1 < raw_defs.len() {
            raw_defs[idx + 1].0 - 1
        } else {
            *st + 1000
        };
        spans.push((*st, (*st).max(ed), sym.clone()));
    }
    spans
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
        for (s, d) in &doc.syms {
            if !s.starts_with("local ") {
                sym_names.insert(s.clone(), d.clone());
            }
        }
        for occ in &doc.occs {
            if (occ.roles & 1) != 0 && !occ.symbol.starts_with("local ") {
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
        if (occ.roles & 1) != 0 || occ.symbol.starts_with("local ") || !known.contains(&occ.symbol)
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
            .occs
            .iter()
            .filter(|o| (o.roles & 1) != 0 && !o.symbol.starts_with("local "))
            .map(|o| (o.range.first().copied().unwrap_or(0), o.symbol.clone()))
            .collect();
        raw_defs.sort();
        let spans = doc_spans(&raw_defs, &src_lines);
        doc_edges(&doc.occs, &spans, &known, &mut callees);
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
        .occs
        .iter()
        .filter(|o| !o.range.is_empty() && (o.roles & 1) != 0 && !o.symbol.starts_with("local "))
        .map(|o| (o.range[0], o.symbol.clone()))
        .collect();
    defs.sort();
    let spans = doc_spans(&defs, &lines);
    for occ in &doc.occs {
        if (occ.roles & 1) != 0 || occ.range.len() < 2 || occ.symbol.starts_with("local ") {
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
