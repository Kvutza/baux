use super::scip_pb::{
    DEFINITION, Document, ScipIndex, SymbolInfo, SymbolMatch, cache_path, format_hit, fresh,
    local_symbol, resolve_analyzer, rust_scip,
};
use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::fs;
use std::path::Path;
use std::process::Command;

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct Span {
    pub(super) symbol: String,
    pub(super) display: String,
    pub(super) callable: bool,
    pub(super) start: usize,
    pub(super) end: usize,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(super) struct Edge {
    pub(super) caller: String,
    pub(super) callee: String,
    pub(super) site: String,
}

#[derive(Default)]
pub(super) struct CallGraph {
    pub(super) names: BTreeMap<String, String>,
    pub(super) defs: BTreeMap<String, Vec<String>>,
    pub(super) callers: BTreeMap<String, Vec<Edge>>,
    pub(super) callees: BTreeMap<String, Vec<Edge>>,
}

pub(super) fn select_symbols<'a>(index: &'a ScipIndex, query: &str) -> Vec<SymbolMatch<'a>> {
    let (file_hint, sym_raw) = if !query.contains("::") {
        query.split_once(':').map_or((None, query), |(file, sym)| {
            (Some(file.to_ascii_lowercase()), sym)
        })
    } else {
        (None, query)
    };
    let query = sym_raw.to_ascii_lowercase();
    let mut matches = BTreeMap::<String, (u8, SymbolMatch<'a>)>::new();
    for document in &index.documents {
        if let Some(hint) = file_hint.as_deref() {
            let doc_lower = document.path.to_ascii_lowercase();
            if !doc_lower.ends_with(hint) && !doc_lower.contains(hint) {
                continue;
            }
        }
        let definitions = document
            .occurrences
            .iter()
            .filter(|occurrence| occurrence.roles & DEFINITION != 0)
            .map(|occurrence| occurrence.symbol.as_str())
            .collect::<BTreeSet<_>>();
        for info in &document.symbols {
            if !definitions.contains(info.symbol.as_str()) {
                continue;
            }
            let Some(score) = match_score(info, &query) else {
                continue;
            };
            let key = if local_symbol(&info.symbol) {
                format!("{}\0{}", document.path, info.symbol)
            } else {
                info.symbol.clone()
            };
            matches
                .entry(key)
                .or_insert((score, SymbolMatch { document, info }));
        }
    }
    let Some(best) = matches.values().map(|(score, _)| *score).min() else {
        return Vec::new();
    };
    matches
        .into_values()
        .filter(|(score, _)| *score == best)
        .map(|(_, info)| info)
        .collect()
}

fn match_score(info: &SymbolInfo, query: &str) -> Option<u8> {
    let display = info.display.to_ascii_lowercase();
    let symbol = info.symbol.to_ascii_lowercase();
    if display == query {
        return Some(0);
    }
    if symbol == query {
        return Some(1);
    }
    if query.contains("::") {
        let parts = query
            .split("::")
            .filter(|part| !part.is_empty())
            .collect::<Vec<_>>();
        if let Some((last, prefix)) = parts.split_last() {
            if display == *last && prefix.iter().all(|part| symbol.contains(part)) {
                return Some(1);
            }
        }
    }
    if display.contains(query) {
        Some(2)
    } else if symbol.contains(query) {
        Some(3)
    } else {
        None
    }
}

pub(super) fn print_symbol(
    index: &ScipIndex,
    found: &SymbolMatch<'_>,
    references: bool,
    limit: &mut usize,
) {
    let info = found.info;
    println!("SYMBOL {}", info.display);
    if local_symbol(&info.symbol) {
        println!("canonical: {} @ {}", info.symbol, found.document.path);
    } else {
        println!("canonical: {}", info.symbol);
    }
    if let Some(signature) = info.signature.as_ref() {
        println!("signature: {}", signature.replace('\n', " "));
    }
    let mut definitions = Vec::new();
    let mut refs = Vec::new();
    let documents: &[Document] = if local_symbol(&info.symbol) {
        std::slice::from_ref(found.document)
    } else {
        &index.documents
    };
    for document in documents {
        for occurrence in &document.occurrences {
            if occurrence.symbol != info.symbol {
                continue;
            }
            let hit = format_hit(&document.path, occurrence);
            if occurrence.roles & DEFINITION != 0 {
                definitions.push(hit);
            } else {
                refs.push(hit);
            }
        }
    }
    definitions.sort();
    refs.sort();
    for hit in definitions {
        if *limit == 0 {
            break;
        }
        println!("definition: {hit}");
        *limit -= 1;
    }
    println!("references: {}", refs.len());
    if references {
        for hit in refs {
            if *limit == 0 {
                break;
            }
            println!("reference: {hit}");
            *limit -= 1;
        }
    }
}

fn has_command(program: &str, argument: &str) -> bool {
    Command::new(program)
        .arg(argument)
        .output()
        .is_ok_and(|output| output.status.success())
}

fn report(name: &str, ready: bool, detail: &str) {
    let state = if ready { "ready" } else { "absent" };
    println!("{name:<12} {state:<6} {detail}");
}

pub(super) fn doctor(root: &Path) -> Result<(), String> {
    let path = cache_path(root, root);
    report(
        "lexical",
        has_command("rg", "--version"),
        "ripgrep lexical search",
    );
    let analyzer = resolve_analyzer(root);
    report("rust", rust_scip(&analyzer), "rust-analyzer SCIP");
    report(
        "rust-index",
        path.is_file() && fresh(root, &path)?,
        path.to_string_lossy().as_ref(),
    );
    report(
        "lean",
        has_command("lake", "--version"),
        "Lean kernel and Lake",
    );
    report(
        "theorem",
        root.join("research/lean/.lake/packages/LeanSearchClient")
            .is_dir(),
        "LeanSearch and Loogle clients",
    );
    let metal = Command::new("xcrun")
        .args(["-sdk", "macosx", "--find", "metal"])
        .output()
        .is_ok_and(|output| output.status.success());
    report("metal", metal, "Apple Metal compiler");
    report("cuda", has_command("nvcc", "--version"), "NVIDIA compiler");
    report(
        "codeql",
        has_command("codeql", "version"),
        "optional batch data-flow analysis",
    );
    report(
        "glean",
        has_command("glean", "--version"),
        "optional cross-language fact store",
    );
    Ok(())
}

pub(super) fn is_callable(symbol: &str, signature: Option<&str>) -> bool {
    symbol.ends_with("().")
        || symbol.ends_with('!')
        || signature.is_some_and(|sig| sig.contains("fn ") || sig.contains("macro_rules!"))
}

pub(super) fn build_graph(root: &Path, index: &ScipIndex) -> CallGraph {
    let mut graph = CallGraph::default();
    let mut known = BTreeSet::new();
    let mut callable_syms = BTreeSet::new();
    let mut impl_bridges = Vec::<(String, String)>::new();
    for doc in &index.documents {
        for info in &doc.symbols {
            if !local_symbol(&info.symbol) && !info.symbol.ends_with('/') {
                graph
                    .names
                    .entry(info.symbol.clone())
                    .or_insert_with(|| info.display.clone());
                known.insert(info.symbol.clone());
                if is_callable(&info.symbol, info.signature.as_deref()) {
                    callable_syms.insert(info.symbol.clone());
                    for trait_sym in &info.implements {
                        if is_callable(trait_sym, None) {
                            impl_bridges.push((trait_sym.clone(), info.symbol.clone()));
                        }
                    }
                }
            }
        }
        for occ in &doc.occurrences {
            if occ.roles & DEFINITION != 0
                && !local_symbol(&occ.symbol)
                && !occ.symbol.ends_with('/')
            {
                graph
                    .defs
                    .entry(occ.symbol.clone())
                    .or_default()
                    .push(format_hit(&doc.path, occ));
                known.insert(occ.symbol.clone());
                if is_callable(&occ.symbol, None) {
                    callable_syms.insert(occ.symbol.clone());
                }
            }
        }
    }
    let mut seen_callers = BTreeSet::new();
    let mut seen_callees = BTreeSet::new();
    for doc in &index.documents {
        let spans = doc_spans(root, doc);
        for occ in &doc.occurrences {
            if occ.roles & DEFINITION != 0 || local_symbol(&occ.symbol) || occ.symbol.ends_with('/')
            {
                continue;
            }
            if !known.contains(&occ.symbol) {
                continue;
            }
            let Some(&line_raw) = occ.range.first() else {
                continue;
            };
            let line = usize::try_from(line_raw).unwrap_or(0);
            let Some(enclosing) = callable_span(&spans, line) else {
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
            if seen_callers.insert(edge.clone()) {
                graph
                    .callers
                    .entry(edge.callee.clone())
                    .or_default()
                    .push(edge.clone());
            }
            if callable_syms.contains(&occ.symbol) && seen_callees.insert(edge.clone()) {
                graph
                    .callees
                    .entry(edge.caller.clone())
                    .or_default()
                    .push(edge);
            }
        }
    }
    for (trait_sym, impl_sym) in impl_bridges {
        if !known.contains(&trait_sym) || trait_sym == impl_sym {
            continue;
        }
        let site = graph
            .defs
            .get(&impl_sym)
            .and_then(|defs| defs.first())
            .cloned()
            .unwrap_or_default();
        let edge = Edge {
            caller: trait_sym,
            callee: impl_sym,
            site,
        };
        if seen_callers.insert(edge.clone()) {
            graph
                .callers
                .entry(edge.callee.clone())
                .or_default()
                .push(edge.clone());
        }
        if seen_callees.insert(edge.clone()) {
            graph
                .callees
                .entry(edge.caller.clone())
                .or_default()
                .push(edge);
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

pub(super) fn base_dir(dir: &Path) -> &Path {
    if dir.is_file() {
        dir.parent().unwrap_or(dir)
    } else {
        dir
    }
}

pub(super) fn doc_spans(root: &Path, doc: &Document) -> Vec<Span> {
    let root = base_dir(root);
    let info_map = doc
        .symbols
        .iter()
        .filter(|info| !local_symbol(&info.symbol))
        .map(|info| (info.symbol.as_str(), info))
        .collect::<BTreeMap<_, _>>();
    let mut raw_defs = Vec::new();
    for occ in &doc.occurrences {
        if occ.roles & DEFINITION == 0 || local_symbol(&occ.symbol) || occ.symbol.ends_with('/') {
            continue;
        }
        let Some(&line_raw) = occ.range.first() else {
            continue;
        };
        let start = usize::try_from(line_raw).unwrap_or(0);
        let (display, callable, is_extern) = match info_map.get(occ.symbol.as_str()) {
            Some(info) => {
                let is_fn = is_callable(&info.symbol, info.signature.as_deref());
                let ext = info
                    .signature
                    .as_deref()
                    .is_some_and(|sig| sig.starts_with("extern crate"));
                (info.display.clone(), is_fn, ext)
            }
            None => (occ.symbol.clone(), is_callable(&occ.symbol, None), false),
        };
        if is_extern || (!callable && occ.symbol.contains('#') && occ.symbol.ends_with('.')) {
            continue;
        }
        raw_defs.push((start, occ.symbol.clone(), display, callable));
    }
    raw_defs.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
    raw_defs.dedup_by(|a, b| a.0 == b.0 && a.1 == b.1);
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

pub(crate) fn body_end(lines: &[&str], start: usize) -> usize {
    let mut depth = 0_isize;
    let mut opened = false;
    let mut in_block_comment = 0_usize;
    let mut in_string = false;
    for (offset, line) in lines[start..].iter().enumerate() {
        let bytes = line.as_bytes();
        let mut i = 0;
        while i < bytes.len() {
            let b = bytes[i];
            let next = bytes.get(i + 1).copied();
            if in_block_comment > 0 {
                if b == b'/' && next == Some(b'*') {
                    in_block_comment += 1;
                    i += 2;
                    continue;
                } else if b == b'*' && next == Some(b'/') {
                    in_block_comment -= 1;
                    i += 2;
                    continue;
                }
                i += 1;
                continue;
            }
            if in_string {
                if b == b'\\' {
                    i += 2;
                    continue;
                } else if b == b'"' {
                    in_string = false;
                }
                i += 1;
                continue;
            }
            if b == b'/' && next == Some(b'/') {
                break;
            }
            if b == b'/' && next == Some(b'*') {
                in_block_comment += 1;
                i += 2;
                continue;
            }
            if b == b'"' {
                in_string = true;
                i += 1;
                continue;
            }
            if b == b'\'' {
                if next == Some(b'\\') && bytes.get(i + 3) == Some(&b'\'') {
                    i += 4;
                    continue;
                } else if bytes.get(i + 2) == Some(&b'\'') {
                    i += 3;
                    continue;
                }
            }
            if b == b'{' {
                depth += 1;
                opened = true;
            } else if b == b'}' {
                depth -= 1;
                if opened && depth <= 0 {
                    return start + offset;
                }
            } else if b == b';' && !opened {
                return start + offset;
            }
            i += 1;
        }
    }
    start
}

pub(super) fn callable_span(spans: &[Span], line: usize) -> Option<&Span> {
    spans
        .iter()
        .filter(|span| span.callable && span.start <= line && line <= span.end)
        .min_by_key(|span| (span.end.saturating_sub(span.start), usize::MAX - span.start))
}

pub(super) fn enclosing_span(spans: &[Span], line: usize) -> Option<&Span> {
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

pub(super) fn annotate_line(line: &str, spans: &BTreeMap<String, Vec<Span>>) -> String {
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

pub(super) fn bfs_walk(
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
