mod scip_pb;
mod scip_split;

use scip_pb::{
    DEFINITION, Document, ScipIndex, cache_path, format_hit, load_index, local_symbol,
    require_index, resolve_target, select_symbols,
};
use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::fs;
use std::path::Path;
use std::process::Command;

#[derive(Clone, Debug, PartialEq, Eq)]
struct Span {
    symbol: String,
    display: String,
    callable: bool,
    start: usize,
    end: usize,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct Edge {
    caller: String,
    callee: String,
    site: String,
}

#[derive(Default)]
struct CallGraph {
    names: BTreeMap<String, String>,
    defs: BTreeMap<String, Vec<String>>,
    callers: BTreeMap<String, Vec<Edge>>,
    callees: BTreeMap<String, Vec<Edge>>,
}

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
            .filter(|occ| occ.roles & DEFINITION != 0 && !local_symbol(&occ.symbol))
            .map(|occ| occ.symbol.as_str())
            .collect::<BTreeSet<_>>();
        docs.push((doc.path.clone(), lines, defs.len()));
        for info in &doc.symbols {
            if !defs.contains(info.symbol.as_str()) {
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
            let sig = info
                .signature
                .as_deref()
                .map(|text| text.replace('\n', " "))
                .unwrap_or_else(|| info.display.clone());
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
        scip_pb::print_symbol(&value, symbol, references, &mut remaining);
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
    scip_pb::doctor(root)
}

pub fn split(
    root: &Path,
    target: Option<&Path>,
    granularity: usize,
    exclude_root: bool,
) -> Result<(), String> {
    scip_split::split(root, target, granularity, exclude_root)
}

fn build_graph(root: &Path, index: &ScipIndex) -> CallGraph {
    let mut graph = CallGraph::default();
    let mut known = BTreeSet::new();
    for doc in &index.documents {
        for info in &doc.symbols {
            if !local_symbol(&info.symbol) {
                graph
                    .names
                    .entry(info.symbol.clone())
                    .or_insert_with(|| info.display.clone());
                known.insert(info.symbol.clone());
            }
        }
        for occ in &doc.occurrences {
            if occ.roles & DEFINITION != 0 && !local_symbol(&occ.symbol) {
                graph
                    .defs
                    .entry(occ.symbol.clone())
                    .or_default()
                    .push(format_hit(&doc.path, occ));
                known.insert(occ.symbol.clone());
            }
        }
    }
    let mut seen_edges = BTreeSet::new();
    for doc in &index.documents {
        let spans = doc_spans(root, doc);
        for occ in &doc.occurrences {
            if occ.roles & DEFINITION != 0 || local_symbol(&occ.symbol) {
                continue;
            }
            if !known.contains(&occ.symbol) {
                continue;
            }
            let Some(&line_raw) = occ.range.first() else {
                continue;
            };
            let line = usize::try_from(line_raw).unwrap_or(0);
            let Some(enclosing) = enclosing_span(&spans, line) else {
                continue;
            };
            if enclosing.symbol == occ.symbol {
                continue;
            }
            let edge = Edge {
                caller: enclosing.symbol.clone(),
                callee: occ.symbol.clone(),
                site: format_hit(&doc.path, occ),
            };
            if seen_edges.insert(edge.clone()) {
                graph
                    .callers
                    .entry(edge.callee.clone())
                    .or_default()
                    .push(edge.clone());
                graph
                    .callees
                    .entry(edge.caller.clone())
                    .or_default()
                    .push(edge);
            }
        }
    }
    for defs in graph.defs.values_mut() {
        defs.sort();
        defs.dedup();
    }
    for edges in graph.callers.values_mut() {
        edges.sort();
    }
    for edges in graph.callees.values_mut() {
        edges.sort();
    }
    graph
}

fn doc_spans(root: &Path, doc: &Document) -> Vec<Span> {
    let info_map = doc
        .symbols
        .iter()
        .filter(|info| !local_symbol(&info.symbol))
        .map(|info| (info.symbol.as_str(), info))
        .collect::<BTreeMap<_, _>>();
    let mut raw_defs = Vec::new();
    for occ in &doc.occurrences {
        if occ.roles & DEFINITION == 0 || local_symbol(&occ.symbol) {
            continue;
        }
        let Some(&line_raw) = occ.range.first() else {
            continue;
        };
        let start = usize::try_from(line_raw).unwrap_or(0);
        let (display, callable) = match info_map.get(occ.symbol.as_str()) {
            Some(info) => {
                let is_fn = info.symbol.ends_with("().")
                    || info
                        .signature
                        .as_deref()
                        .is_some_and(|sig| sig.contains("fn "));
                (info.display.clone(), is_fn)
            }
            None => (occ.symbol.clone(), occ.symbol.ends_with("().")),
        };
        raw_defs.push((start, occ.symbol.clone(), display, callable));
    }
    raw_defs.sort_by_key(|(start, _, _, _)| *start);
    let source = fs::read_to_string(root.join(&doc.path)).ok();
    let lines = source
        .as_deref()
        .map(|text| text.lines().collect::<Vec<_>>());
    let mut spans = Vec::with_capacity(raw_defs.len());
    for (idx, (start, symbol, display, callable)) in raw_defs.iter().enumerate() {
        let end = match lines.as_deref() {
            Some(src_lines) if *start < src_lines.len() => body_end(src_lines, *start),
            _ => raw_defs
                .get(idx + 1)
                .map_or(usize::MAX / 2, |(next, _, _, _)| next.saturating_sub(1)),
        };
        spans.push(Span {
            symbol: symbol.clone(),
            display: display.clone(),
            callable: *callable,
            start: *start,
            end: end.max(*start),
        });
    }
    spans
}

fn body_end(lines: &[&str], start: usize) -> usize {
    let mut depth = 0_isize;
    let mut opened = false;
    for (offset, line) in lines[start..].iter().enumerate() {
        let code = line.split("//").next().unwrap_or(line);
        for ch in code.chars() {
            if ch == '{' {
                depth += 1;
                opened = true;
            } else if ch == '}' {
                depth -= 1;
                if opened && depth <= 0 {
                    return start + offset;
                }
            } else if ch == ';' && !opened {
                return start + offset;
            }
        }
    }
    start
}

fn enclosing_span(spans: &[Span], line: usize) -> Option<&Span> {
    let containing = spans
        .iter()
        .filter(|span| span.start <= line && line <= span.end)
        .collect::<Vec<_>>();
    containing
        .iter()
        .copied()
        .filter(|span| span.callable)
        .min_by_key(|span| (span.end.saturating_sub(span.start), usize::MAX - span.start))
        .or_else(|| {
            containing
                .into_iter()
                .min_by_key(|span| (span.end.saturating_sub(span.start), usize::MAX - span.start))
        })
}

fn annotate_line(line: &str, spans: &BTreeMap<String, Vec<Span>>) -> String {
    let mut parts = line.splitn(4, ':');
    let (Some(path), Some(line_str), Some(col), Some(rest)) =
        (parts.next(), parts.next(), parts.next(), parts.next())
    else {
        return line.to_owned();
    };
    let Ok(line_one) = line_str.parse::<usize>() else {
        return line.to_owned();
    };
    let Some(doc_spans) = spans.get(path) else {
        return line.to_owned();
    };
    let Some(span) = enclosing_span(doc_spans, line_one.saturating_sub(1)) else {
        return line.to_owned();
    };
    format!("{path}:{line_str}:{col} [{}] {rest}", span.display)
}

fn bfs_walk(
    adj: &BTreeMap<String, Vec<Edge>>,
    root_sym: &str,
    reverse: bool,
    max_depth: usize,
) -> Vec<(usize, Vec<Edge>)> {
    let mut visited = BTreeSet::new();
    visited.insert(root_sym.to_owned());
    let mut frontier = VecDeque::new();
    frontier.push_back((root_sym.to_owned(), 1_usize));
    let mut by_depth = BTreeMap::<usize, Vec<Edge>>::new();
    while let Some((current, depth)) = frontier.pop_front() {
        if depth > max_depth {
            continue;
        }
        let Some(edges) = adj.get(&current) else {
            continue;
        };
        for edge in edges {
            by_depth.entry(depth).or_default().push(edge.clone());
            let next = if reverse { &edge.caller } else { &edge.callee };
            if visited.insert(next.clone()) {
                frontier.push_back((next.clone(), depth + 1));
            }
        }
    }
    by_depth.into_iter().collect()
}

#[cfg(test)]
mod tests {
    use super::scip_pb::{Occurrence, SymbolInfo};
    use super::*;

    #[test]
    fn graph_edges() {
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
                    },
                    SymbolInfo {
                        symbol: "crate/callee().".to_owned(),
                        display: "callee".to_owned(),
                        signature: Some("fn callee()".to_owned()),
                    },
                ],
            }],
        };
        let graph = build_graph(Path::new("/nonexistent"), &index);
        let callers = graph.callers.get("crate/callee().").expect("caller edge");
        assert_eq!(callers.len(), 1);
        assert_eq!(callers[0].caller, "crate/caller().");
        assert_eq!(callers[0].site, "synthetic.rs:3:5");
        let callees = graph.callees.get("crate/caller().").expect("callee edge");
        assert_eq!(callees.len(), 1);
        assert_eq!(callees[0].callee, "crate/callee().");
        assert_eq!(body_end(&["fn a() {", "  b();", "}"], 0), 2);
    }
}
