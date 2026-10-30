use crate::research_site::{DeclRecord, LeanModuleRecord, lean_files};
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::Path;

fn opt_layer(short: &str) -> (&'static str, &'static str) {
    if matches!(
        short,
        "Envelope"
            | "EnvelopeMemory"
            | "SlopeEnvelope"
            | "LocalMemory"
            | "RecentMemory"
            | "Retention"
            | "AdaptiveKnn"
            | "KnnLearning"
            | "KnnDecision"
            | "Localized"
    ) {
        ("pointwise_metric", "Pointwise Metric & Slope Envelopes")
    } else if matches!(
        short,
        "LieOperator"
            | "PairRotation"
            | "Perturbation"
            | "Smooth"
            | "Guide"
            | "EvaluationSpace"
            | "ActiveEnergy"
            | "Moments"
            | "Field"
            | "Operator"
            | "Metric"
            | "SeedSupport"
            | "Reachability"
    ) {
        ("operator_bo", "Neural Operator BO & Matrix Manifolds")
    } else if matches!(
        short,
        "PretrainHorizon"
            | "PretrainRegret"
            | "PretrainWorkload"
            | "Process"
            | "Trajectory"
            | "Reliability"
            | "Statistic"
            | "Bayes"
            | "Information"
    ) {
        ("stochastic_tail", "Stochastic Objectives & Tail Bounds")
    } else {
        ("trust_region", "Trust-Region & Surrogate Geometry")
    }
}

fn classify_layer(module_name: &str) -> (&'static str, &'static str) {
    if module_name == "BauxResearch" || module_name == "Main" {
        return ("root", "Root & Entrypoint");
    }
    if module_name.starts_with("BauxResearch.Generated.")
        || module_name.starts_with("BauxResearch.Extraction.")
    {
        return ("semantics_core", "Aeneas & Core Semantics");
    }
    if let Some(short) = module_name.strip_prefix("BauxResearch.Semantics.") {
        return if matches!(
            short,
            "Float" | "Sampling" | "TrustRegion" | "Score" | "Numerical" | "Window"
        ) {
            ("semantics_core", "Aeneas & Core Semantics")
        } else {
            ("semantics_enn", "ENN Semantics & Parity")
        };
    }
    if module_name.starts_with("BauxResearch.Forward.") {
        return ("operator_bo", "Neural Operator BO & Matrix Manifolds");
    }
    if module_name.starts_with("BauxResearch.Objective.") {
        return ("stochastic_tail", "Stochastic Objectives & Tail Bounds");
    }
    if let Some(short) = module_name.strip_prefix("BauxResearch.Optimization.") {
        return opt_layer(short);
    }
    ("foundations", "Measure & Functional Foundations")
}

fn layer_key(layer: &str) -> usize {
    match layer {
        "foundations" => 0,
        "semantics_core" => 1,
        "semantics_enn" => 2,
        "operator_bo" => 3,
        "pointwise_metric" => 4,
        "trust_region" => 5,
        "stochastic_tail" => 6,
        "root" => 7,
        _ => 8,
    }
}

pub(crate) fn parse_decl(line: &str) -> Option<(&'static str, String)> {
    let mut rest = line;
    loop {
        let mut stripped = false;
        for kw in [
            "noncomputable ",
            "private ",
            "protected ",
            "scoped ",
            "partial ",
            "unsafe ",
        ] {
            if let Some(after) = rest.strip_prefix(kw) {
                rest = after.trim_start();
                stripped = true;
            }
        }
        if !stripped {
            break;
        }
    }
    let kinds = [
        ("theorem", "theorem"),
        ("lemma", "theorem"),
        ("def", "def"),
        ("abbrev", "abbrev"),
        ("structure", "structure"),
        ("inductive", "inductive"),
        ("class", "class"),
        ("opaque", "opaque"),
        ("axiom", "axiom"),
    ];
    for (kw, norm_kind) in kinds {
        if let Some(after) = rest.strip_prefix(kw) {
            if after.starts_with(char::is_whitespace) {
                let name: String = after
                    .trim_start()
                    .chars()
                    .take_while(|c| c.is_alphanumeric() || matches!(*c, '_' | '\'' | '.'))
                    .collect();
                if !name.is_empty() {
                    return Some((norm_kind, name));
                }
            }
        }
    }
    None
}

fn module_doc(text: &str) -> String {
    let Some(start) = text.find("/-!") else {
        return String::new();
    };
    let after = &text[start + 3..];
    let Some(end) = after.find("-/") else {
        return String::new();
    };
    after[..end]
        .trim()
        .lines()
        .map(|l| l.trim().trim_start_matches('#').trim())
        .filter(|l| !l.is_empty())
        .collect::<Vec<&str>>()
        .join(" ")
}

fn extract_imports(text: &str) -> Vec<String> {
    let mut imports = BTreeSet::new();
    for line in text.lines() {
        if let Some(rest) = line.strip_prefix("import ") {
            let imp: String = rest
                .trim()
                .chars()
                .take_while(|c| c.is_alphanumeric() || matches!(*c, '_' | '.'))
                .collect();
            if imp.starts_with("BauxResearch") {
                imports.insert(imp);
            }
        }
    }
    imports.into_iter().collect()
}

fn preceding_doc(lines: &[&str], idx: usize) -> String {
    if idx == 0 || !lines[idx - 1].trim().ends_with("-/") {
        return String::new();
    }
    let mut doc_lines = Vec::new();
    let mut j = idx - 1;
    loop {
        doc_lines.push(lines[j]);
        if lines[j].trim().starts_with("/--") || j == 0 {
            break;
        }
        j -= 1;
    }
    doc_lines.reverse();
    let joined = doc_lines.join("\n");
    let trimmed = joined
        .trim()
        .strip_prefix("/--")
        .unwrap_or(&joined)
        .strip_suffix("-/")
        .unwrap_or(&joined)
        .trim();
    trimmed.lines().map(str::trim).collect::<Vec<_>>().join(" ")
}

fn decl_end(lines: &[&str], start: usize) -> usize {
    let mut end_i = start + 1;
    while end_i < lines.len() && end_i < start + 65 {
        let l = lines[end_i];
        if parse_decl(l).is_some()
            || l.starts_with("namespace ")
            || l.starts_with("end ")
            || l.starts_with("section ")
            || l.starts_with("/-!")
            || l.starts_with("/--")
        {
            break;
        }
        end_i += 1;
    }
    while end_i > start + 1 && lines[end_i - 1].trim().is_empty() {
        end_i -= 1;
    }
    end_i
}

fn parse_decls(
    lines: &[&str],
    mod_info: (&str, &str, &str),
    sym_lookup: &mut BTreeMap<String, DeclRecord>,
) -> (Vec<DeclRecord>, BTreeMap<String, usize>) {
    let (mod_name, short_name, rel_file) = mod_info;
    let max_code_lines = if short_name == "JoeKuo" || short_name == "Registry" {
        14
    } else {
        40
    };
    let mut decls = Vec::new();
    let mut counts: BTreeMap<String, usize> = [
        "theorem",
        "def",
        "structure",
        "inductive",
        "abbrev",
        "class",
        "opaque",
        "axiom",
    ]
    .into_iter()
    .map(|k| (k.to_owned(), 0))
    .collect();

    let mut i = 0;
    while i < lines.len() {
        let Some((kind, name)) = parse_decl(lines[i]) else {
            i += 1;
            continue;
        };
        *counts.entry(kind.to_owned()).or_insert(0) += 1;
        let doc_str = preceding_doc(lines, i);
        let end_i = decl_end(lines, i);
        let code_end = end_i.min(i + max_code_lines);
        let code_str = lines[i..code_end].join("\n").trim_end().to_owned();
        let rec = DeclRecord {
            kind: kind.to_owned(),
            name: name.clone(),
            module: mod_name.to_owned(),
            file: rel_file.to_owned(),
            line: i + 1,
            end_line: end_i,
            doc: doc_str,
            code: code_str,
        };
        sym_lookup.insert(format!("{mod_name}.{name}"), rec.clone());
        sym_lookup.insert(format!("{short_name}.{name}"), rec.clone());
        sym_lookup.entry(name).or_insert_with(|| rec.clone());
        decls.push(rec);
        i = end_i;
    }
    (decls, counts)
}

fn parse_file(
    root: &Path,
    lean_dir: &Path,
    path: &Path,
    sym_lookup: &mut BTreeMap<String, DeclRecord>,
) -> Result<LeanModuleRecord, String> {
    let rel_to_lean = path
        .strip_prefix(lean_dir)
        .map_err(|e| e.to_string())?
        .to_string_lossy()
        .replace('\\', "/");
    let mod_name = rel_to_lean
        .strip_suffix(".lean")
        .unwrap_or(&rel_to_lean)
        .replace('/', ".");
    let short_name = mod_name.rsplit('.').next().unwrap_or(&mod_name).to_owned();
    let rel_file = path
        .strip_prefix(root)
        .map_err(|e| e.to_string())?
        .to_string_lossy()
        .replace('\\', "/");
    let text =
        fs::read_to_string(path).map_err(|e| format!("failed to read {}: {e}", path.display()))?;
    let lines_vec: Vec<&str> = text.lines().collect();
    let (layer_id, layer_label) = classify_layer(&mod_name);
    let (decls, counts) = parse_decls(&lines_vec, (&mod_name, &short_name, &rel_file), sym_lookup);
    let total_decls: usize = counts.values().sum();
    Ok(LeanModuleRecord {
        module: mod_name,
        short: short_name,
        file: rel_file,
        layer: layer_id.to_owned(),
        layer_label: layer_label.to_owned(),
        lines: lines_vec.len(),
        doc: module_doc(&text),
        imports: extract_imports(&text),
        depends_on: Vec::new(),
        depended_by: Vec::new(),
        dag_rank: 0,
        dag_order: 0,
        dag_layer_size: 1,
        dag_x: 0.5,
        dag_y: 0.0,
        counts,
        total_decls,
        proved_registry_count: 0,
        declarations: decls,
    })
}

pub(crate) fn extract_modules(
    root: &Path,
) -> Result<(Vec<LeanModuleRecord>, BTreeMap<String, DeclRecord>), String> {
    let lean_dir = root.join("research/lean");
    let mut files = Vec::new();
    lean_files(&lean_dir.join("BauxResearch"), &mut files)?;
    for top in ["BauxResearch.lean", "Main.lean"] {
        let p = lean_dir.join(top);
        if p.exists() {
            files.push(p);
        }
    }
    files.sort();
    let mut modules = Vec::with_capacity(files.len());
    let mut sym_lookup: BTreeMap<String, DeclRecord> = BTreeMap::new();
    for path in files {
        modules.push(parse_file(root, &lean_dir, &path, &mut sym_lookup)?);
    }
    compute_dag(&mut modules);
    Ok((modules, sym_lookup))
}

fn dag_edges(modules: &mut [LeanModuleRecord]) -> (Vec<Vec<usize>>, Vec<Vec<usize>>) {
    let n = modules.len();
    let name_to_idx: BTreeMap<String, usize> = modules
        .iter()
        .enumerate()
        .map(|(idx, m)| (m.module.clone(), idx))
        .collect();
    let mut deps_idx: Vec<Vec<usize>> = vec![Vec::new(); n];
    let mut rev_idx: Vec<Vec<usize>> = vec![Vec::new(); n];

    for i in 0..n {
        let mut d_names = Vec::new();
        for imp in &modules[i].imports {
            if imp != &modules[i].module {
                if let Some(&j) = name_to_idx.get(imp) {
                    deps_idx[i].push(j);
                    rev_idx[j].push(i);
                    d_names.push(imp.clone());
                }
            }
        }
        d_names.sort();
        d_names.dedup();
        modules[i].depends_on = d_names;
    }
    for i in 0..n {
        let mut r_names: Vec<String> = rev_idx[i]
            .iter()
            .map(|&j| modules[j].module.clone())
            .collect();
        r_names.sort();
        r_names.dedup();
        modules[i].depended_by = r_names;
    }
    (deps_idx, rev_idx)
}

fn dfs_rank(
    u: usize,
    deps: &[Vec<usize>],
    ranks: &mut [Option<usize>],
    visiting: &mut [bool],
) -> usize {
    if let Some(r) = ranks[u] {
        return r;
    }
    if visiting[u] {
        return 0;
    }
    visiting[u] = true;
    let mut max_dep = 0usize;
    let mut has_dep = false;
    for &v in &deps[u] {
        has_dep = true;
        let r = dfs_rank(v, deps, ranks, visiting);
        if r > max_dep {
            max_dep = r;
        }
    }
    visiting[u] = false;
    let my_rank = if has_dep { max_dep + 1 } else { 0 };
    ranks[u] = Some(my_rank);
    my_rank
}

fn update_pos(layer: &[usize], pos: &mut [f64]) {
    let len = layer.len().max(1) as f64;
    for (ord, &u) in layer.iter().enumerate() {
        pos[u] = (ord as f64 + 0.5) / len;
    }
}

fn sweep_layer(
    layer: &mut Vec<usize>,
    adj: &[Vec<usize>],
    modules: &[LeanModuleRecord],
    pos: &mut [f64],
) {
    let mut scored: Vec<(f64, usize, String, usize)> = layer
        .iter()
        .map(|&u| {
            let bary = if adj[u].is_empty() {
                pos[u]
            } else {
                let sum: f64 = adj[u].iter().map(|&v| pos[v]).sum();
                sum / (adj[u].len() as f64)
            };
            (
                bary,
                layer_key(&modules[u].layer),
                modules[u].module.clone(),
                u,
            )
        })
        .collect();
    scored.sort_by(|a, b| {
        a.0.partial_cmp(&b.0)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.1.cmp(&b.1))
            .then_with(|| a.2.cmp(&b.2))
    });
    *layer = scored.into_iter().map(|t| t.3).collect();
    update_pos(layer, pos);
}

fn assign_coords(modules: &mut [LeanModuleRecord], layers: &[Vec<usize>], max_rank: usize) {
    let max_rank_f = max_rank.max(1) as f64;
    for (r, layer) in layers.iter().enumerate() {
        let lsize = layer.len();
        let lsize_f = lsize.max(1) as f64;
        for (ord, &u) in layer.iter().enumerate() {
            modules[u].dag_rank = r;
            modules[u].dag_order = ord;
            modules[u].dag_layer_size = lsize;
            modules[u].dag_x = (((ord as f64 + 0.5) / lsize_f) * 10000.0).round() / 10000.0;
            modules[u].dag_y = (((r as f64) / max_rank_f) * 10000.0).round() / 10000.0;
        }
    }
}

fn compute_dag(modules: &mut [LeanModuleRecord]) {
    let n = modules.len();
    if n == 0 {
        return;
    }
    let (deps_idx, rev_idx) = dag_edges(modules);
    let mut ranks: Vec<Option<usize>> = vec![None; n];
    let mut visiting: Vec<bool> = vec![false; n];
    let mut max_rank = 0usize;
    for i in 0..n {
        let r = dfs_rank(i, &deps_idx, &mut ranks, &mut visiting);
        modules[i].dag_rank = r;
        if r > max_rank {
            max_rank = r;
        }
    }

    let mut layers: Vec<Vec<usize>> = vec![Vec::new(); max_rank + 1];
    for i in 0..n {
        layers[modules[i].dag_rank].push(i);
    }
    for layer in &mut layers {
        layer.sort_by(|&a, &b| {
            layer_key(&modules[a].layer)
                .cmp(&layer_key(&modules[b].layer))
                .then_with(|| modules[a].module.cmp(&modules[b].module))
        });
    }

    let mut pos: Vec<f64> = vec![0.5; n];
    for layer in &layers {
        update_pos(layer, &mut pos);
    }
    for _pass in 0..4 {
        for r in 1..=max_rank {
            sweep_layer(&mut layers[r], &deps_idx, modules, &mut pos);
        }
        if max_rank > 0 {
            for r in (0..max_rank).rev() {
                sweep_layer(&mut layers[r], &rev_idx, modules, &mut pos);
            }
        }
    }
    assign_coords(modules, &layers, max_rank);
}
