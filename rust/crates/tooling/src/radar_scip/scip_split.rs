use super::{DEFINITION, ScipIndex, local_symbol, require_index, resolve_target};
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::Path;

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
        .strip_prefix("src/")
        .unwrap_or(doc_path)
        .strip_suffix(".rs")
        .unwrap_or(doc_path);
    if trimmed == "lib" || trimmed == "main" {
        return "root".to_owned();
    }
    let parts = trimmed
        .split('/')
        .filter(|&part| part != "mod")
        .collect::<Vec<_>>();
    if parts.is_empty() {
        return "root".to_owned();
    }
    let take = parts.len().min(granularity.max(1));
    parts[..take].join("::")
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
);

fn collect_modules(
    dir: &Path,
    index: &ScipIndex,
    granularity: usize,
    exclude_root: bool,
) -> ModuleData {
    let mut sym_defs = BTreeMap::new();
    let mut sym_display = BTreeMap::new();
    let mut mod_loc = BTreeMap::<String, usize>::new();
    let mut mod_files = BTreeMap::<String, BTreeSet<String>>::new();

    for doc in &index.documents {
        let label = module_label(&doc.path, granularity);
        if exclude_root && label == "root" {
            continue;
        }
        let loc = fs::read_to_string(dir.join(&doc.path))
            .map(|text| text.lines().count())
            .unwrap_or(0);
        *mod_loc.entry(label.clone()).or_insert(0) += loc;
        mod_files.entry(label).or_default().insert(doc.path.clone());

        for info in &doc.symbols {
            if !local_symbol(&info.symbol) {
                sym_display.insert(info.symbol.clone(), info.display.clone());
            }
        }
        for occ in &doc.occurrences {
            if occ.roles & DEFINITION != 0 && !local_symbol(&occ.symbol) {
                sym_defs
                    .entry(occ.symbol.clone())
                    .or_insert_with(|| doc.path.clone());
            }
        }
    }
    (sym_defs, sym_display, mod_loc, mod_files)
}

fn collect_edges(
    index: &ScipIndex,
    granularity: usize,
    exclude_root: bool,
    mod_idx: &BTreeMap<String, usize>,
    sym_defs: &BTreeMap<String, String>,
    sym_display: &BTreeMap<String, String>,
) -> BTreeMap<(usize, usize), Vec<SplitRefSite>> {
    let mut edge_sites = BTreeMap::<(usize, usize), Vec<SplitRefSite>>::new();
    for doc in &index.documents {
        let src_mod = module_label(&doc.path, granularity);
        if exclude_root && src_mod == "root" {
            continue;
        }
        let Some(&u) = mod_idx.get(&src_mod) else {
            continue;
        };
        for occ in &doc.occurrences {
            if occ.roles & DEFINITION != 0 || local_symbol(&occ.symbol) || occ.symbol.ends_with('/')
            {
                continue;
            }
            let Some(def_path) = sym_defs.get(&occ.symbol) else {
                continue;
            };
            let dst_mod = module_label(def_path, granularity);
            if exclude_root && dst_mod == "root" {
                continue;
            }
            let Some(&v) = mod_idx.get(&dst_mod) else {
                continue;
            };
            if u == v {
                continue;
            }
            let line = occ.range.first().copied().unwrap_or(0) as usize + 1;
            let display = sym_display.get(&occ.symbol).cloned().unwrap_or_default();
            edge_sites.entry((u, v)).or_default().push(SplitRefSite {
                file: doc.path.clone(),
                line,
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
    let mut already_free = BTreeSet::new();
    for k in 0..=back_edges.len() {
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
    let (sym_defs, sym_display, mod_loc, mod_files) =
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
    );

    let n = modules.len();
    let mut adj = vec![Vec::new(); n];
    for &(u, v) in edge_sites.keys() {
        adj[u].push(v);
    }

    let raw_sccs = tarjan_scc(n, &adj);
    let multi_sccs = raw_sccs.iter().filter(|c| c.len() > 1).collect::<Vec<_>>();
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
        if comp.len() == 1 {
            let u = comp[0];
            let m = &modules[u];
            let deps = adj[u]
                .iter()
                .map(|&v| modules[v].as_str())
                .collect::<Vec<_>>();
            println!(
                "dag_ready: {m} loc={} files={} depends_on={deps:?}",
                mod_loc[m],
                mod_files[m].len()
            );
        }
    }
    for (idx, comp) in multi_sccs.iter().enumerate() {
        let names = comp
            .iter()
            .map(|&u| modules[u].as_str())
            .collect::<Vec<_>>();
        let loc = comp.iter().map(|&u| mod_loc[&modules[u]]).sum::<usize>();
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
        for site in sites.iter().take(6) {
            println!("cut_site: {}:{} -> {}", site.file, site.line, site.display);
        }
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
        let adj = vec![vec![1], vec![2], vec![0]];
        let sccs = tarjan_scc(3, &adj);
        assert_eq!(sccs.len(), 1);
        assert_eq!(sccs[0].len(), 3);

        let edges = vec![
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
        let back = edges
            .iter()
            .filter(|e| pos(e.to) < pos(e.from))
            .collect::<Vec<_>>();
        assert_eq!(back.len(), 1);
        assert_eq!(back[0].from, 2);
        assert_eq!(back[0].to, 0);
        assert_eq!(back[0].weight, 1);
    }
}
