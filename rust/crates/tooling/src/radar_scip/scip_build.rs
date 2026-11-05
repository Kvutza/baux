use super::scip_graph::{CallGraph, Edge, callable_span, doc_spans, is_callable};
use super::scip_pb::{DEFINITION, ScipIndex, format_hit, local_symbol};
use std::collections::BTreeSet;
use std::path::Path;

type Bridge = (String, String);

fn collect_symbols(
    index: &ScipIndex,
    graph: &mut CallGraph,
) -> (BTreeSet<String>, BTreeSet<String>, Vec<Bridge>) {
    let mut known = BTreeSet::new();
    let mut callable = BTreeSet::new();
    let mut bridges = Vec::new();
    for doc in &index.documents {
        for info in &doc.symbols {
            if local_symbol(&info.symbol) || info.symbol.ends_with('/') {
                continue;
            }
            graph
                .names
                .entry(info.symbol.clone())
                .or_insert_with(|| info.display.clone());
            known.insert(info.symbol.clone());
            if is_callable(&info.symbol, info.signature.as_deref()) {
                callable.insert(info.symbol.clone());
                bridges.extend(
                    info.implements
                        .iter()
                        .filter(|symbol| is_callable(symbol, None))
                        .map(|symbol| (symbol.clone(), info.symbol.clone())),
                );
            }
        }
        for occ in &doc.occurrences {
            if occ.roles & DEFINITION == 0 || local_symbol(&occ.symbol) || occ.symbol.ends_with('/')
            {
                continue;
            }
            graph
                .defs
                .entry(occ.symbol.clone())
                .or_default()
                .push(format_hit(&doc.path, occ));
            known.insert(occ.symbol.clone());
            if is_callable(&occ.symbol, None) {
                callable.insert(occ.symbol.clone());
            }
        }
    }
    (known, callable, bridges)
}

fn insert_edge(
    graph: &mut CallGraph,
    edge: Edge,
    callable: bool,
    seen_callers: &mut BTreeSet<Edge>,
    seen_callees: &mut BTreeSet<Edge>,
) {
    if seen_callers.insert(edge.clone()) {
        graph
            .callers
            .entry(edge.callee.clone())
            .or_default()
            .push(edge.clone());
    }
    if callable && seen_callees.insert(edge.clone()) {
        graph
            .callees
            .entry(edge.caller.clone())
            .or_default()
            .push(edge);
    }
}

fn collect_edges(
    root: &Path,
    index: &ScipIndex,
    graph: &mut CallGraph,
    known: &BTreeSet<String>,
    callable: &BTreeSet<String>,
    seen_callers: &mut BTreeSet<Edge>,
    seen_callees: &mut BTreeSet<Edge>,
) {
    for doc in &index.documents {
        let spans = doc_spans(root, doc);
        for occ in &doc.occurrences {
            if occ.roles & DEFINITION != 0
                || local_symbol(&occ.symbol)
                || occ.symbol.ends_with('/')
                || !known.contains(&occ.symbol)
            {
                continue;
            }
            let Some(&line_raw) = occ.range.first() else {
                continue;
            };
            let Some(enclosing) = callable_span(&spans, usize::try_from(line_raw).unwrap_or(0))
            else {
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
            insert_edge(
                graph,
                edge,
                callable.contains(&occ.symbol),
                seen_callers,
                seen_callees,
            );
        }
    }
}

fn connect_bridges(
    graph: &mut CallGraph,
    bridges: Vec<Bridge>,
    known: &BTreeSet<String>,
    seen_callers: &mut BTreeSet<Edge>,
    seen_callees: &mut BTreeSet<Edge>,
) {
    for (trait_sym, impl_sym) in bridges {
        if !known.contains(&trait_sym) || trait_sym == impl_sym {
            continue;
        }
        let site = graph
            .defs
            .get(&impl_sym)
            .and_then(|defs| defs.first())
            .cloned()
            .unwrap_or_default();
        insert_edge(
            graph,
            Edge {
                caller: trait_sym,
                callee: impl_sym,
                site,
            },
            true,
            seen_callers,
            seen_callees,
        );
    }
}

fn sort_graph(graph: &mut CallGraph) {
    for defs in graph.defs.values_mut() {
        defs.sort();
        defs.dedup();
    }
    graph.callers.values_mut().for_each(|edges| edges.sort());
    graph.callees.values_mut().for_each(|edges| edges.sort());
}

pub(super) fn build_graph(root: &Path, index: &ScipIndex) -> CallGraph {
    let mut graph = CallGraph::default();
    let (known, callable, bridges) = collect_symbols(index, &mut graph);
    let mut seen_callers = BTreeSet::new();
    let mut seen_callees = BTreeSet::new();
    collect_edges(
        root,
        index,
        &mut graph,
        &known,
        &callable,
        &mut seen_callers,
        &mut seen_callees,
    );
    connect_bridges(
        &mut graph,
        bridges,
        &known,
        &mut seen_callers,
        &mut seen_callees,
    );
    sort_graph(&mut graph);
    graph
}
