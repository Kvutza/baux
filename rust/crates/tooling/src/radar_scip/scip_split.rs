use super::scip_graph::{Span, base_dir, doc_spans, enclosing_span};
use super::scip_pb::{
    DEFINITION, Document, ScipIndex, local_symbol, require_index, resolve_target,
};
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::Path;

const SYMBOL_GRANULARITY: usize = 10;

#[derive(Clone, Debug)]
struct SplitRefSite {
    file: String,
    line: usize,
    symbol: String,
    display: String,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct WeightedEdge {
    from: usize,
    to: usize,
    weight: u64,
}

struct FasState {
    succ: BTreeMap<usize, Vec<(usize, u64)>>,
    pred: BTreeMap<usize, Vec<(usize, u64)>>,
    removed: BTreeSet<usize>,
    live_out: BTreeMap<usize, i64>,
    live_in: BTreeMap<usize, i64>,
}

impl FasState {
    fn remove_vertex(&mut self, v: usize) {
        for &(target, weight) in &self.succ[&v] {
            if !self.removed.contains(&target) {
                *self.live_in.get_mut(&target).unwrap() -= weight as i64;
            }
        }
        for &(source, weight) in &self.pred[&v] {
            if !self.removed.contains(&source) {
                *self.live_out.get_mut(&source).unwrap() -= weight as i64;
            }
        }
    }

    fn drain_zero(&mut self, nodes: &[usize], check_out: bool, out: &mut Vec<usize>) {
        let mut progress = true;
        while progress {
            progress = false;
            for &v in nodes {
                let deg = if check_out {
                    self.live_out[&v]
                } else {
                    self.live_in[&v]
                };
                if !self.removed.contains(&v) && deg == 0 {
                    out.push(v);
                    self.removed.insert(v);
                    self.remove_vertex(v);
                    progress = true;
                }
            }
        }
    }
}

fn module_label(doc_path: &str, granularity: usize) -> String {
    let trimmed = doc_path
        .strip_prefix("rust/crates/")
        .unwrap_or(doc_path)
        .strip_prefix("src/")
        .unwrap_or(doc_path.strip_prefix("rust/crates/").unwrap_or(doc_path));
    let trimmed = trimmed.strip_suffix(".rs").unwrap_or(trimmed);
    if trimmed == "lib" || trimmed == "main" {
        return "root".to_owned();
    }
    let parts = trimmed
        .split('/')
        .filter(|&part| part != "mod" && part != "src")
        .collect::<Vec<_>>();
    if parts.is_empty() || (parts.len() == 1 && (parts[0] == "lib" || parts[0] == "main")) {
        return "root".to_owned();
    }
    let take = parts.len().min(granularity.max(1));
    parts[..take].join("::")
}

fn unit_spans(dir: &Path, doc: &Document) -> Vec<Span> {
    doc_spans(dir, doc)
        .into_iter()
        .filter(|span| {
            if span.callable || span.symbol.ends_with('#') || span.symbol.ends_with("().") {
                return true;
            }
            let tail = span.symbol.rsplit('/').next().unwrap_or(&span.symbol);
            !tail.contains('#')
        })
        .collect()
}

fn span_label(file_mod: &str, span: &Span) -> String {
    let tail = span.symbol.rsplit('/').next().unwrap_or(&span.symbol);
    let item = if let Some((ty, _)) = tail.strip_prefix("impl#[").and_then(|s| s.split_once(']')) {
        let clean_ty = ty
            .split(',')
            .next()
            .unwrap_or(ty)
            .trim()
            .trim_matches('`')
            .split('<')
            .next()
            .unwrap_or(ty);
        format!("{clean_ty}::{}", span.display)
    } else if let Some((ty, rest)) = tail.split_once('#')
        && !ty.is_empty()
        && !rest.is_empty()
        && ty != span.display
    {
        format!("{ty}::{}", span.display)
    } else if !span.display.is_empty() {
        span.display.clone()
    } else {
        tail.to_owned()
    };
    if file_mod == "root" {
        item
    } else {
        format!("{file_mod}::{item}")
    }
}

/// Weighted Eades-Lin-Smyth (ELS) minimum feedback-arc-set vertex ordering.
fn fas_order(nodes: &[usize], edges: &[WeightedEdge]) -> Vec<usize> {
    let node_set = nodes.iter().copied().collect::<BTreeSet<_>>();
    let mut state = FasState {
        succ: nodes.iter().map(|&n| (n, Vec::new())).collect(),
        pred: nodes.iter().map(|&n| (n, Vec::new())).collect(),
        removed: BTreeSet::new(),
        live_out: nodes.iter().map(|&n| (n, 0_i64)).collect(),
        live_in: nodes.iter().map(|&n| (n, 0_i64)).collect(),
    };

    for edge in edges {
        if edge.from == edge.to || !node_set.contains(&edge.from) || !node_set.contains(&edge.to) {
            continue;
        }
        *state.live_out.get_mut(&edge.from).unwrap() += edge.weight as i64;
        *state.live_in.get_mut(&edge.to).unwrap() += edge.weight as i64;
        state
            .succ
            .get_mut(&edge.from)
            .unwrap()
            .push((edge.to, edge.weight));
        state
            .pred
            .get_mut(&edge.to)
            .unwrap()
            .push((edge.from, edge.weight));
    }

    let mut head = Vec::new();
    let mut tail = Vec::new();
    while state.removed.len() < nodes.len() {
        state.drain_zero(nodes, true, &mut tail);
        state.drain_zero(nodes, false, &mut head);
        if state.removed.len() < nodes.len() {
            let best = nodes
                .iter()
                .copied()
                .filter(|v| !state.removed.contains(v))
                .max_by_key(|v| state.live_out[v] - state.live_in[v]);
            if let Some(v) = best {
                head.push(v);
                state.removed.insert(v);
                state.remove_vertex(v);
            } else {
                break;
            }
        }
    }
    tail.reverse();
    head.extend(tail);
    head
}

/// Tarjan's Strongly Connected Components algorithm.
fn tarjan_scc(n: usize, adj: &[Vec<usize>]) -> Vec<Vec<usize>> {
    struct State {
        index: usize,
        indices: Vec<Option<usize>>,
        lowlink: Vec<usize>,
        on_stack: Vec<bool>,
        stack: Vec<usize>,
        sccs: Vec<Vec<usize>>,
    }
    fn strongconnect(v: usize, adj: &[Vec<usize>], st: &mut State) {
        st.indices[v] = Some(st.index);
        st.lowlink[v] = st.index;
        st.index += 1;
        st.stack.push(v);
        st.on_stack[v] = true;
        for &w in &adj[v] {
            if st.indices[w].is_none() {
                strongconnect(w, adj, st);
                st.lowlink[v] = st.lowlink[v].min(st.lowlink[w]);
            } else if st.on_stack[w] {
                st.lowlink[v] = st.lowlink[v].min(st.indices[w].unwrap());
            }
        }
        if st.lowlink[v] == st.indices[v].unwrap() {
            let mut comp = Vec::new();
            while let Some(w) = st.stack.pop() {
                st.on_stack[w] = false;
                comp.push(w);
                if w == v {
                    break;
                }
            }
            st.sccs.push(comp);
        }
    }
    let mut st = State {
        index: 0,
        indices: vec![None; n],
        lowlink: vec![0; n],
        on_stack: vec![false; n],
        stack: Vec::new(),
        sccs: Vec::new(),
    };
    for v in 0..n {
        if st.indices[v].is_none() {
            strongconnect(v, adj, &mut st);
        }
    }
    st.sccs
}

fn classify_symbols(sites: &[SplitRefSite]) -> (&'static str, Vec<String>) {
    let mut syms = BTreeSet::new();
    let mut has_type = false;
    let mut has_val = false;
    for site in sites {
        let label = if !site.display.is_empty() {
            site.display.clone()
        } else {
            site.symbol
                .rsplit('/')
                .next()
                .unwrap_or(&site.symbol)
                .to_owned()
        };
        if site.symbol.ends_with('#') {
            has_type = true;
            syms.insert(format!("{label} (type/trait)"));
        } else if site.symbol.ends_with("().") {
            has_val = true;
            syms.insert(format!("{label}() (fn/method)"));
        } else {
            has_val = true;
            syms.insert(format!("{label} (const/val)"));
        }
    }
    let kind = match (has_type, has_val) {
        (true, false) => "move-type-to-foundation",
        (false, true) => "move-const-or-invert-fn",
        _ => "extract-shared",
    };
    (kind, syms.into_iter().collect())
}

type ModuleData = (
    BTreeMap<String, String>,
    BTreeMap<String, String>,
    BTreeMap<String, usize>,
    BTreeMap<String, BTreeSet<String>>,
    BTreeMap<String, Vec<Span>>,
);

fn collect_modules(
    dir: &Path,
    index: &ScipIndex,
    granularity: usize,
    exclude_root: bool,
) -> ModuleData {
    let base = base_dir(dir);
    let sym_mode = granularity >= SYMBOL_GRANULARITY;
    let mut sym_defs = BTreeMap::new();
    let mut sym_display = BTreeMap::new();
    let mut mod_loc = BTreeMap::<String, usize>::new();
    let mut mod_files = BTreeMap::<String, BTreeSet<String>>::new();
    let mut spans_by_doc = BTreeMap::<String, Vec<Span>>::new();

    for doc in &index.documents {
        let file_mod = module_label(&doc.path, granularity);
        if exclude_root && (file_mod == "root" || file_mod.ends_with("::root")) {
            continue;
        }
        for info in &doc.symbols {
            if !local_symbol(&info.symbol) {
                sym_display.insert(info.symbol.clone(), info.display.clone());
            }
        }
        if sym_mode {
            let spans = unit_spans(base, doc);
            for span in &spans {
                let node = span_label(&file_mod, span);
                let loc = span.end.saturating_sub(span.start) + 1;
                *mod_loc.entry(node.clone()).or_insert(0) += loc;
                mod_files
                    .entry(node.clone())
                    .or_default()
                    .insert(doc.path.clone());
                sym_defs.entry(span.symbol.clone()).or_insert(node);
            }
            for occ in &doc.occurrences {
                if occ.roles & DEFINITION != 0 && !local_symbol(&occ.symbol) {
                    let line = occ.range.first().copied().unwrap_or(0) as usize;
                    if let Some(span) = enclosing_span(&spans, line) {
                        sym_defs
                            .entry(occ.symbol.clone())
                            .or_insert_with(|| span_label(&file_mod, span));
                    }
                }
            }
            spans_by_doc.insert(doc.path.clone(), spans);
        } else {
            let loc = fs::read_to_string(base.join(&doc.path))
                .map(|text| text.lines().count())
                .unwrap_or(0);
            *mod_loc.entry(file_mod.clone()).or_insert(0) += loc;
            mod_files
                .entry(file_mod.clone())
                .or_default()
                .insert(doc.path.clone());
            for occ in &doc.occurrences {
                if occ.roles & DEFINITION != 0 && !local_symbol(&occ.symbol) {
                    sym_defs
                        .entry(occ.symbol.clone())
                        .or_insert_with(|| file_mod.clone());
                }
            }
        }
    }
    (sym_defs, sym_display, mod_loc, mod_files, spans_by_doc)
}

fn collect_edges(
    index: &ScipIndex,
    granularity: usize,
    exclude_root: bool,
    mod_idx: &BTreeMap<String, usize>,
    sym_defs: &BTreeMap<String, String>,
    sym_display: &BTreeMap<String, String>,
    spans_by_doc: &BTreeMap<String, Vec<Span>>,
) -> BTreeMap<(usize, usize), Vec<SplitRefSite>> {
    let sym_mode = granularity >= SYMBOL_GRANULARITY;
    let mut edge_sites = BTreeMap::<(usize, usize), Vec<SplitRefSite>>::new();
    for doc in &index.documents {
        let file_mod = module_label(&doc.path, granularity);
        if exclude_root && (file_mod == "root" || file_mod.ends_with("::root")) {
            continue;
        }
        let doc_spans = spans_by_doc.get(&doc.path);
        for occ in &doc.occurrences {
            if occ.roles & DEFINITION != 0 || local_symbol(&occ.symbol) || occ.symbol.ends_with('/')
            {
                continue;
            }
            let raw_line = occ.range.first().copied().unwrap_or(0) as usize;
            let src_node = if sym_mode {
                let Some(spans) = doc_spans else { continue };
                let Some(span) = enclosing_span(spans, raw_line) else {
                    continue;
                };
                span_label(&file_mod, span)
            } else {
                file_mod.clone()
            };
            let Some(&u) = mod_idx.get(&src_node) else {
                continue;
            };
            let Some(dst_node) = sym_defs.get(&occ.symbol) else {
                continue;
            };
            let Some(&v) = mod_idx.get(dst_node) else {
                continue;
            };
            if u == v {
                continue;
            }
            let display = sym_display.get(&occ.symbol).cloned().unwrap_or_default();
            edge_sites.entry((u, v)).or_default().push(SplitRefSite {
                file: doc.path.clone(),
                line: raw_line + 1,
                symbol: occ.symbol.clone(),
                display,
            });
        }
    }
    edge_sites
}

type BackEdge = (usize, usize, Vec<SplitRefSite>);

fn find_cuts(
    multi_sccs: &[&Vec<usize>],
    edge_sites: &BTreeMap<(usize, usize), Vec<SplitRefSite>>,
) -> (Vec<BackEdge>, BTreeSet<usize>) {
    let mut back_edges = Vec::new();
    let mut locked_raw = BTreeSet::new();

    for comp in multi_sccs {
        for &node in *comp {
            locked_raw.insert(node);
        }
        let comp_set = comp.iter().copied().collect::<BTreeSet<_>>();
        let mut els_edges = Vec::new();
        for (&(u, v), sites) in edge_sites {
            if comp_set.contains(&u) && comp_set.contains(&v) {
                els_edges.push(WeightedEdge {
                    from: v,
                    to: u,
                    weight: sites.len() as u64,
                });
            }
        }
        let order = fas_order(comp, &els_edges);
        let pos = order
            .iter()
            .enumerate()
            .map(|(idx, &node)| (node, idx))
            .collect::<BTreeMap<_, _>>();
        for (&(u, v), sites) in edge_sites {
            if comp_set.contains(&u) && comp_set.contains(&v) && pos[&v] > pos[&u] {
                back_edges.push((u, v, sites.clone()));
            }
        }
    }
    back_edges.sort_by_key(|(_, _, sites)| sites.len());
    (back_edges, locked_raw)
}

fn print_frontier(
    modules: &[String],
    mod_loc: &BTreeMap<String, usize>,
    edge_sites: &BTreeMap<(usize, usize), Vec<SplitRefSite>>,
    back_edges: &[BackEdge],
    locked_raw: &BTreeSet<usize>,
) {
    let n = modules.len();
    let total = back_edges.len();
    let step = (total / 32).max(1);
    let mut already_free = BTreeSet::new();
    for k in (0..=total).step_by(step).chain(std::iter::once(total)) {
        let cut_set = back_edges[..k]
            .iter()
            .map(|(u, v, _)| (*u, *v))
            .collect::<BTreeSet<_>>();
        let refs_cut = back_edges[..k]
            .iter()
            .map(|(_, _, s)| s.len())
            .sum::<usize>();
        let mut cur_adj = vec![Vec::new(); n];
        for &(u, v) in edge_sites.keys() {
            if !cut_set.contains(&(u, v)) {
                cur_adj[u].push(v);
            }
        }
        let sccs = tarjan_scc(n, &cur_adj);
        let largest_loc = sccs
            .iter()
            .map(|c| c.iter().map(|&u| mod_loc[&modules[u]]).sum::<usize>())
            .max()
            .unwrap_or(0);
        let mut newly_freed = Vec::new();
        for c in &sccs {
            if c.len() == 1 {
                let u = c[0];
                if locked_raw.contains(&u) && !already_free.contains(&u) {
                    already_free.insert(u);
                    newly_freed.push(modules[u].clone());
                }
            }
        }
        newly_freed.sort();
        println!(
            "frontier: cuts={k} refs_cut={refs_cut} sccs={} largest_loc={largest_loc} \
             freed={newly_freed:?}",
            sccs.len()
        );
    }
}

pub(super) fn split(
    root: &Path,
    target: Option<&Path>,
    granularity: usize,
    exclude_root: bool,
) -> Result<(), String> {
    let dir = resolve_target(root, target)?;
    let index = require_index(root, &dir)?;
    let (sym_defs, sym_display, mod_loc, mod_files, spans_by_doc) =
        collect_modules(&dir, &index, granularity, exclude_root);

    let modules = mod_loc.keys().cloned().collect::<Vec<_>>();
    let mod_idx = modules
        .iter()
        .enumerate()
        .map(|(idx, name)| (name.clone(), idx))
        .collect::<BTreeMap<_, _>>();
    let edge_sites = collect_edges(
        &index,
        granularity,
        exclude_root,
        &mod_idx,
        &sym_defs,
        &sym_display,
        &spans_by_doc,
    );

    let n = modules.len();
    let mut adj = vec![Vec::new(); n];
    edge_sites.keys().for_each(|&(u, v)| adj[u].push(v));

    let raw_sccs = tarjan_scc(n, &adj);
    let multi_sccs: Vec<_> = raw_sccs.iter().filter(|c| c.len() > 1).collect();
    println!(
        "SPLIT target={} granularity={} modules={} total_loc={} raw_sccs={} cyclic_sccs={}",
        dir.display(),
        granularity,
        n,
        mod_loc.values().sum::<usize>(),
        raw_sccs.len(),
        multi_sccs.len()
    );

    let (back_edges, locked_raw) = find_cuts(&multi_sccs, &edge_sites);
    for comp in &raw_sccs {
        if let [u] = comp.as_slice() {
            let m = &modules[*u];
            let deps: Vec<_> = adj[*u].iter().map(|&v| modules[v].as_str()).collect();
            println!(
                "dag_ready: {m} loc={} files={} depends_on={deps:?}",
                mod_loc[m],
                mod_files[m].len()
            );
        }
    }
    for (idx, comp) in multi_sccs.iter().enumerate() {
        let names: Vec<_> = comp.iter().map(|&u| modules[u].as_str()).collect();
        let loc: usize = comp.iter().map(|&u| mod_loc[&modules[u]]).sum();
        println!(
            "cyclic_scc: id={} modules={} loc={loc} members={names:?}",
            idx + 1,
            comp.len()
        );
    }
    for (k, (u, v, sites)) in back_edges.iter().enumerate() {
        let (kind, syms) = classify_symbols(sites);
        println!(
            "cut: id={} {} -> {} refs={} kind={kind} symbols={}",
            k + 1,
            modules[*u],
            modules[*v],
            sites.len(),
            syms.join(", ")
        );
        sites
            .iter()
            .take(6)
            .for_each(|s| println!("cut_site: {}:{} -> {}", s.file, s.line, s.display));
    }
    print_frontier(&modules, &mod_loc, &edge_sites, &back_edges, &locked_raw);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{WeightedEdge, fas_order, module_label, tarjan_scc};

    #[test]
    fn scc_cuts() {
        assert_eq!(
            module_label("src/optimizer/trust_region/controller.rs", 2),
            "optimizer::trust_region"
        );
        assert_eq!(
            module_label("rust/crates/baux/src/fbt_moe/candidate.rs", 2),
            "baux::fbt_moe"
        );
        assert_eq!(tarjan_scc(3, &[vec![1], vec![2], vec![0]])[0].len(), 3);
        let edges = [
            WeightedEdge {
                from: 0,
                to: 1,
                weight: 20,
            },
            WeightedEdge {
                from: 1,
                to: 2,
                weight: 20,
            },
            WeightedEdge {
                from: 2,
                to: 0,
                weight: 1,
            },
        ];
        let order = fas_order(&[0, 1, 2], &edges);
        let pos = |node: usize| order.iter().position(|&x| x == node).unwrap();
        let back: Vec<_> = edges.iter().filter(|e| pos(e.to) < pos(e.from)).collect();
        assert_eq!((back.len(), back[0].from, back[0].to), (1, 2, 0));
    }
}
