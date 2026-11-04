mod scip_build;
pub(crate) mod scip_graph;
pub(crate) mod scip_pb;
mod scip_split;

use scip_build::build_graph;
use scip_graph::{
    annotate_line, bfs_walk, doc_spans, doctor as graph_doctor, print_symbol, select_symbols,
};
use scip_pb::{DEFINITION, cache_path, load_index, local_symbol, require_index, resolve_target};
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::Path;
use std::process::Command;

pub fn index(root: &Path, target: Option<&Path>, force: bool) -> Result<(), String> {
    scip_pb::index(root, target, force)
}

pub fn map(
    root: &Path,
    target: Option<&Path>,
    filter: Option<&str>,
    limit: usize,
) -> Result<(), String> {
    let dir = resolve_target(root, target)?;
    let value = require_index(root, &dir)?;
    let graph = build_graph(&dir, &value);
    let filter_low = filter.map(str::to_ascii_lowercase);
    let mut docs = Vec::new();
    let mut ranked = Vec::new();
    for doc in &value.documents {
        if let Some(sub) = filter_low.as_deref() {
            if !doc.path.to_ascii_lowercase().contains(sub) {
                continue;
            }
        }
        let lines = fs::read_to_string(dir.join(&doc.path))
            .map(|text| text.lines().count())
            .unwrap_or(0);
        let defs = doc
            .occurrences
            .iter()
            .filter(|occ| {
                occ.roles & DEFINITION != 0
                    && !local_symbol(&occ.symbol)
                    && !occ.symbol.ends_with('/')
            })
            .map(|occ| occ.symbol.as_str())
            .collect::<BTreeSet<_>>();
        docs.push((doc.path.clone(), lines, defs.len()));
        for info in &doc.symbols {
            if !defs.contains(info.symbol.as_str())
                || info.display.is_empty()
                || info.symbol.ends_with('/')
            {
                continue;
            }
            let sig = info
                .signature
                .as_deref()
                .map(|text| text.replace('\n', " "))
                .unwrap_or_else(|| info.display.clone());
            if sig.starts_with("extern crate") || sig.starts_with("mod ") {
                continue;
            }
            let in_deg = graph.callers.get(&info.symbol).map_or(0, Vec::len);
            let out_deg = graph.callees.get(&info.symbol).map_or(0, Vec::len);
            let score = in_deg * 3 + out_deg;
            let site = graph
                .defs
                .get(&info.symbol)
                .and_then(|list| list.first())
                .cloned()
                .unwrap_or_else(|| doc.path.clone());
            ranked.push((score, in_deg, out_deg, site, info.display.clone(), sig));
        }
    }
    docs.sort();
    ranked.sort_by(|left, right| {
        right
            .0
            .cmp(&left.0)
            .then_with(|| right.1.cmp(&left.1))
            .then_with(|| left.3.cmp(&right.3))
    });
    let total_lines = docs.iter().map(|(_, lines, _)| *lines).sum::<usize>();
    println!(
        "REPOMAP target={} files={} lines={} symbols={}",
        dir.display(),
        docs.len(),
        total_lines,
        ranked.len()
    );
    for (path, lines, count) in &docs {
        println!("file: {path} lines={lines} defs={count}");
    }
    for (_, in_deg, out_deg, site, name, sig) in ranked.into_iter().take(limit) {
        println!("rank: callers={in_deg} callees={out_deg} {name} @ {site} :: {sig}");
    }
    Ok(())
}

pub fn symbol(
    root: &Path,
    target: Option<&Path>,
    query: &str,
    references: bool,
    limit: usize,
) -> Result<(), String> {
    let dir = resolve_target(root, target)?;
    let value = require_index(root, &dir)?;
    let symbols = select_symbols(&value, query);
    if symbols.is_empty() {
        return Err(format!(
            "no compiler-resolved Rust symbol matches `{query}`"
        ));
    }
    let mut remaining = limit;
    for (position, symbol) in symbols.iter().enumerate() {
        if position > 0 {
            println!();
        }
        print_symbol(&value, symbol, references, &mut remaining);
        if remaining == 0 {
            break;
        }
    }
    Ok(())
}

pub fn find(
    root: &Path,
    target: Option<&Path>,
    pattern: &str,
    regex: bool,
    context: usize,
) -> Result<(), String> {
    let dir = resolve_target(root, target)?;
    let mut command = Command::new("rg");
    command
        .current_dir(&dir)
        .args(["--line-number", "--column", "--no-heading", "--color=never"]);
    if !regex {
        command.arg("--fixed-strings");
    }
    if context > 0 {
        command.arg(format!("--context={context}"));
    }
    command.args(["--", pattern]);
    let output = command
        .output()
        .map_err(|error| format!("start ripgrep search: {error}"))?;
    if !output.status.success() && output.status.code() != Some(1) {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(format!("ripgrep search failed: {}", stderr.trim()));
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    if stdout.trim().is_empty() {
        println!("no matches for `{pattern}`");
        return Ok(());
    }
    let path = cache_path(root, &dir);
    let spans = if path.is_file() {
        load_index(&path)
            .ok()
            .map(|idx| {
                idx.documents
                    .iter()
                    .map(|doc| (doc.path.clone(), doc_spans(&dir, doc)))
                    .collect::<BTreeMap<_, _>>()
            })
            .unwrap_or_default()
    } else {
        BTreeMap::new()
    };
    for line in stdout.lines() {
        println!("{}", annotate_line(line, &spans));
    }
    Ok(())
}

pub fn callers(root: &Path, target: Option<&Path>, query: &str) -> Result<(), String> {
    walk_cmd(root, target, query, true)
}

pub fn callees(root: &Path, target: Option<&Path>, query: &str) -> Result<(), String> {
    walk_cmd(root, target, query, false)
}

fn walk_cmd(root: &Path, target: Option<&Path>, query: &str, reverse: bool) -> Result<(), String> {
    let dir = resolve_target(root, target)?;
    let value = require_index(root, &dir)?;
    let targets = select_symbols(&value, query);
    if targets.is_empty() {
        return Err(format!(
            "no compiler-resolved Rust symbol matches `{query}`"
        ));
    }
    let graph = build_graph(&dir, &value);
    let (header, empty, role, max_depth, adj) = if reverse {
        ("CALLERS", "callers: 0", "caller", 5, &graph.callers)
    } else {
        ("CALLEES", "callees: 0", "callee", 3, &graph.callees)
    };
    for (idx, target) in targets.iter().enumerate() {
        if idx > 0 {
            println!();
        }
        let sym = &target.info.symbol;
        println!("{header} {} ({sym})", target.info.display);
        if let Some(defs) = graph.defs.get(sym) {
            for def in defs {
                println!("definition: {def}");
            }
        }
        let layers = bfs_walk(adj, sym, reverse, max_depth);
        if layers.is_empty() {
            println!("{empty}");
            continue;
        }
        for (depth, edges) in layers {
            for edge in edges {
                let peer = if reverse { &edge.caller } else { &edge.callee };
                let name = graph.names.get(peer).map_or(peer.as_str(), String::as_str);
                println!("depth={depth} {role}: {name} [{peer}] @ {}", edge.site);
            }
        }
    }
    Ok(())
}

pub fn impact(root: &Path, target: Option<&Path>, query: &str) -> Result<(), String> {
    let dir = resolve_target(root, target)?;
    let value = require_index(root, &dir)?;
    let targets = select_symbols(&value, query);
    if targets.is_empty() {
        return Err(format!(
            "no compiler-resolved Rust symbol matches `{query}`"
        ));
    }
    let graph = build_graph(&dir, &value);
    for (idx, target) in targets.iter().enumerate() {
        if idx > 0 {
            println!();
        }
        let sym = &target.info.symbol;
        println!("IMPACT {} ({sym})", target.info.display);
        if let Some(defs) = graph.defs.get(sym) {
            for def in defs {
                println!("definition: {def}");
            }
        }
        let layers = bfs_walk(&graph.callers, sym, true, 8);
        let direct = layers.first().map_or(0, |(_, edges)| edges.len());
        let mut files = BTreeSet::new();
        let mut tests = BTreeSet::new();
        let mut total = 0_usize;
        for (_, edges) in &layers {
            for edge in edges {
                total += 1;
                if let Some((file, _)) = edge.site.split_once(':') {
                    files.insert(file.to_owned());
                }
                if edge.caller.contains("test") || edge.site.contains("test") {
                    let name = graph
                        .names
                        .get(&edge.caller)
                        .map(String::as_str)
                        .unwrap_or(&edge.caller);
                    tests.insert(format!("{name} @ {}", edge.site));
                }
            }
        }
        println!(
            "summary: direct_callers={direct} transitive_edges={total} affected_files={} \
             affected_tests={}",
            files.len(),
            tests.len()
        );
        for file in &files {
            println!("affected_file: {file}");
        }
        for test in &tests {
            println!("affected_test: {test}");
        }
        for (depth, edges) in layers {
            for edge in edges {
                let name = graph
                    .names
                    .get(&edge.caller)
                    .map(String::as_str)
                    .unwrap_or(&edge.caller);
                println!("depth={depth} reach: {name} @ {}", edge.site);
            }
        }
    }
    Ok(())
}

pub fn doctor(root: &Path) -> Result<(), String> {
    graph_doctor(root)
}

pub fn split(
    root: &Path,
    target: Option<&Path>,
    granularity: usize,
    exclude_root: bool,
) -> Result<(), String> {
    scip_split::split(root, target, granularity, exclude_root)
}

#[cfg(test)]
mod tests {
    use super::scip_graph::body_end;
    use super::scip_pb::{Document, Occurrence, ScipIndex, SymbolInfo, format_hit, parse_index};
    use super::*;

    fn varint(mut value: u64) -> Vec<u8> {
        let mut bytes = Vec::new();
        loop {
            let mut byte = (value & 0x7f) as u8;
            value >>= 7;
            if value != 0 {
                byte |= 0x80;
            }
            bytes.push(byte);
            if value == 0 {
                return bytes;
            }
        }
    }

    fn bytes(field: u8, value: &[u8]) -> Vec<u8> {
        let mut output = varint(u64::from(field) << 3 | 2);
        output.extend(varint(value.len() as u64));
        output.extend(value);
        output
    }

    #[test]
    fn decode_graph() {
        let occ = Occurrence {
            range: vec![4, 2, 9],
            ..Occurrence::default()
        };
        assert_eq!(format_hit("source.rs", &occ), "source.rs:5:3");

        let mut occurrence = bytes(1, &[4, 2, 9]);
        occurrence.extend(bytes(2, b"rust test symbol"));
        occurrence.extend([24, 1]);
        let signature = bytes(5, b"fn test()");
        let mut info = bytes(1, b"rust test symbol");
        info.extend(bytes(6, b"test"));
        info.extend(bytes(7, &signature));
        let mut document = bytes(1, b"source.rs");
        document.extend(bytes(2, &occurrence));
        document.extend(bytes(3, &info));
        let mut decoded = parse_index(&bytes(2, &document)).expect("valid synthetic SCIP");
        assert_eq!(decoded.documents[0].path, "source.rs");
        assert_eq!(select_symbols(&decoded, "source.rs:test").len(), 1);
        decoded.documents[0].occurrences[0].roles = 0;
        assert!(select_symbols(&decoded, "test").is_empty());

        let index = ScipIndex {
            documents: vec![Document {
                path: "synthetic.rs".to_owned(),
                occurrences: vec![
                    Occurrence {
                        range: vec![0, 3, 9],
                        symbol: "crate/caller().".to_owned(),
                        roles: DEFINITION,
                    },
                    Occurrence {
                        range: vec![2, 4, 10],
                        symbol: "crate/callee().".to_owned(),
                        roles: 0,
                    },
                    Occurrence {
                        range: vec![6, 3, 9],
                        symbol: "crate/callee().".to_owned(),
                        roles: DEFINITION,
                    },
                ],
                symbols: vec![
                    SymbolInfo {
                        symbol: "crate/caller().".to_owned(),
                        display: "caller".to_owned(),
                        signature: Some("fn caller()".to_owned()),
                        ..SymbolInfo::default()
                    },
                    SymbolInfo {
                        symbol: "crate/callee().".to_owned(),
                        display: "callee".to_owned(),
                        signature: Some("fn callee()".to_owned()),
                        ..SymbolInfo::default()
                    },
                ],
            }],
        };
        let graph = build_graph(Path::new("/nonexistent"), &index);
        assert_eq!(
            graph.callers["crate/callee()."][0].caller,
            "crate/caller()."
        );
        assert_eq!(
            graph.callees["crate/caller()."][0].callee,
            "crate/callee()."
        );
        assert_eq!(body_end(&["fn a() {", "  b();", "}"], 0), 2);
        assert_eq!(
            body_end(&["fn a() {", "  let s = \"{\"; // }", "  b();", "}"], 0),
            3
        );
        assert_eq!(
            scip_pb::canonical_symbol(
                "rust-analyzer cargo baux-cuda 0.0.1 semantics/optimizer/surrogate/impl#[NeighborValueEstimator]select_pool()."
                    .to_owned()
            ),
            "rust-analyzer cargo baux 0.0.1 semantics/optimizer/surrogate/impl#[NeighborValueEstimator]select_pool()."
        );
    }
}
