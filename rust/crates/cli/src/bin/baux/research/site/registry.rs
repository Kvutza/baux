use crate::research_site::{AeneasRecord, DeclRecord, LeanModuleRecord, LlbcRecord, shared_path};
use crate::site_aeneas::normalize_key;
use baux::artifact::json::{self, Map, Value};
use baux::artifact::wire_json;
use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

fn source_snippet(
    root: &Path,
    rel_file: &str,
    line_num: usize,
    max_lines: usize,
) -> (usize, String) {
    let enn_root = shared_path(root, ".cache/baux/upstream/enn");
    let path = enn_root.join(rel_file);
    let Ok(text) = fs::read_to_string(&path) else {
        return (1, format!("// Source file {rel_file} (line {line_num})"));
    };
    let lines: Vec<&str> = text.lines().collect();
    let start = line_num.saturating_sub(1).min(lines.len());
    let end = (start + max_lines).min(lines.len());
    (
        start + 1,
        lines[start..end].join("\n").trim_end().to_owned(),
    )
}

fn decl_tail(
    modules: &[LeanModuleRecord],
    default_module: &str,
    last_part: &str,
) -> Option<String> {
    let dotted = format!(".{last_part}");
    for m in modules {
        if m.module != default_module
            && m.declarations
                .iter()
                .any(|d| d.name == last_part || d.name.ends_with(&dotted))
        {
            return Some(m.module.clone());
        }
    }
    None
}

fn symbol_module(rhs_expr: &str, modules: &[LeanModuleRecord], default_module: &str) -> String {
    let rhs = rhs_expr.trim();
    if rhs.is_empty() || rhs == "rfl" {
        return default_module.to_owned();
    }
    let head: String = rhs
        .chars()
        .take_while(|c| c.is_alphanumeric() || matches!(*c, '_' | '.' | '\''))
        .collect();
    if head.is_empty() {
        return default_module.to_owned();
    }
    let parts: Vec<&str> = head.split('.').collect();
    for m in modules {
        if parts.contains(&m.short.as_str()) {
            return m.module.clone();
        }
    }
    if let Some(&last_part) = parts.last() {
        if let Some(found) = decl_tail(modules, default_module, last_part) {
            return found;
        }
    }
    default_module.to_owned()
}

fn next_quoted(s: &str) -> Option<(String, &str)> {
    let s = s.trim_start();
    if !s.starts_with('"') {
        return None;
    }
    let mut out = String::new();
    let mut chars = s[1..].char_indices();
    while let Some((i, ch)) = chars.next() {
        if ch == '\\' {
            if let Some((_, esc)) = chars.next() {
                out.push(esc);
            }
        } else if ch == '"' {
            return Some((out, &s[1 + i + 1..]));
        } else {
            out.push(ch);
        }
    }
    None
}

struct RegEntry {
    idx: usize,
    func_id: String,
    file: String,
    line_num: usize,
    func_name: String,
    lean_def: String,
    lean_thm: String,
}

fn parse_entry(line: &str) -> Option<RegEntry> {
    let after = line.trim().strip_prefix("mkEntry ")?;
    let idx_str: String = after.chars().take_while(|c| c.is_ascii_digit()).collect();
    let idx = idx_str.parse::<usize>().ok()?;
    let rest = after[idx_str.len()..].trim_start();
    let (func_id, r1) = next_quoted(rest)?;
    let (file, r2) = next_quoted(r1)?;
    let rest = r2.trim_start();
    let line_str: String = rest.chars().take_while(|c| c.is_ascii_digit()).collect();
    let line_num = line_str.parse::<usize>().unwrap_or(1);
    let rest = rest[line_str.len()..].trim_start();
    let (func_name, r3) = next_quoted(rest)?;
    let (lean_def, r4) = next_quoted(r3)?;
    let (lean_thm, _) = next_quoted(r4)?;
    Some(RegEntry {
        idx,
        func_id,
        file,
        line_num,
        func_name,
        lean_def,
        lean_thm,
    })
}

fn insert_symbol(registry_symbols: &mut Map, sym: &str, sym_lookup: &BTreeMap<String, DeclRecord>) {
    if registry_symbols.contains_key(sym) {
        return;
    }
    if let Some(d) = sym_lookup.get(sym) {
        registry_symbols.insert(
            sym.to_owned(),
            wire_json!({
                "kind": d.kind,
                "name": d.name,
                "module": d.module,
                "file": d.file,
                "line": d.line,
                "end_line": d.end_line,
                "doc": d.doc,
                "code": d.code,
            }),
        );
    }
}

fn load_entries(
    root: &Path,
    sym_lookup: &BTreeMap<String, DeclRecord>,
) -> Result<(BTreeMap<usize, RegEntry>, Map), String> {
    let reg_path = root.join("research/lean/BauxResearch/Semantics/Registry.lean");
    let reg_text = fs::read_to_string(&reg_path)
        .map_err(|e| format!("failed to read {}: {e}", reg_path.display()))?;
    let mut reg_entries: BTreeMap<usize, RegEntry> = BTreeMap::new();
    let mut registry_symbols = Map::new();
    for line in reg_text.lines() {
        if let Some(entry) = parse_entry(line) {
            insert_symbol(&mut registry_symbols, &entry.lean_def, sym_lookup);
            insert_symbol(&mut registry_symbols, &entry.lean_thm, sym_lookup);
            reg_entries.insert(entry.idx, entry);
        }
    }
    Ok((reg_entries, registry_symbols))
}

fn match_llbc(
    file: &str,
    line: usize,
    short_func: &str,
    last_two: &str,
    llbc_by_key: &BTreeMap<String, LlbcRecord>,
) -> Value {
    let candidates = [
        format!("{file}:{line}"),
        format!("{file}:{}", line.saturating_sub(1)),
        format!("{file}:{}", line + 1),
        format!("{file}::{short_func}"),
        last_two.to_owned(),
        short_func.to_owned(),
    ];
    for ck in &candidates {
        if let Some(rec) = llbc_by_key.get(ck) {
            return wire_json!({
                "crate": rec.crate_name,
                "llbc_file": format!("{}.llbc", rec.crate_name),
                "llbc_name": rec.full_ident,
                "def_id": rec.def_id,
                "opacity": rec.opacity,
                "file": rec.file,
                "span": rec.span,
                "beg_line": rec.beg_line,
                "end_line": rec.end_line,
                "inputs": rec.arg_count,
                "locals": rec.locals_count,
                "statements": rec.statements_count,
                "named_locals": rec.named_locals,
                "sig": rec.sig,
                "snippet": rec.snippet,
            });
        }
    }
    Value::null()
}

fn impl_key(func_id: &str, short_func: &str) -> Option<String> {
    let ipos = func_id.find("impl#[")?;
    let after = &func_id[ipos + 6..];
    let cpos = after.find(']')?;
    let impl_ty = after[..cpos]
        .rsplit(',')
        .next()
        .unwrap_or(&after[..cpos])
        .trim();
    (!impl_ty.is_empty()).then(|| format!("impl::{impl_ty}::{short_func}"))
}

fn match_aeneas(
    file: &str,
    line: usize,
    func_id: &str,
    short_func: &str,
    last_two: &str,
    aeneas_by_key: &BTreeMap<String, AeneasRecord>,
) -> Value {
    let file_stem = Path::new(file)
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("");
    let mut candidates = vec![
        format!("span::{file}:{line}::{short_func}"),
        format!("span::{file}:{}::{short_func}", line.saturating_sub(1)),
        format!("span::{file}:{}::{short_func}", line + 1),
    ];
    if let Some(key) = impl_key(func_id, short_func) {
        candidates.push(key);
    } else {
        candidates.push(format!("free::{file_stem}::{short_func}"));
        candidates.push(format!("free::{last_two}"));
    }
    for ak in &candidates {
        if let Some(arec) = aeneas_by_key.get(ak) {
            return wire_json!({
                "lean_name": arec.lean_name,
                "file": arec.file,
                "rust_item": arec.rust_item,
                "span": arec.span,
                "code": arec.code,
            });
        }
    }
    Value::null()
}

fn classify_crate(file: &str) -> &'static str {
    if file.starts_with("rust/crates/bpann") || file.starts_with("crates/enn-core") {
        "bpann"
    } else if file.starts_with("rust/crates/ennbo-py") || file.starts_with("crates/enn-py") {
        "ennbo-py"
    } else if file.starts_with("rust/crates/ennbo") || file.starts_with("crates/enn-optimize") {
        "ennbo"
    } else if file.starts_with("src/enn/turbo") {
        "enn.turbo"
    } else if file.starts_with("src/enn/enn") {
        "enn.enn"
    } else {
        "enn"
    }
}

fn match_ir(
    language: &str,
    file: &str,
    line: usize,
    func_id: &str,
    name: &str,
    aeneas_by_key: &BTreeMap<String, AeneasRecord>,
    llbc_by_key: &BTreeMap<String, LlbcRecord>,
) -> (Value, Value) {
    if language != "rust" {
        return (Value::null(), Value::null());
    }
    let short_func = name.rsplit("::").next().unwrap_or(name);
    let norm_name = normalize_key(name);
    let norm_segs: Vec<&str> = norm_name.split("::").filter(|s| !s.is_empty()).collect();
    let last_two = if norm_segs.len() >= 2 {
        format!(
            "{}::{}",
            norm_segs[norm_segs.len() - 2],
            norm_segs[norm_segs.len() - 1]
        )
    } else {
        norm_name
    };
    let llbc_val = match_llbc(file, line, short_func, &last_two, llbc_by_key);
    let aeneas_val = match_aeneas(file, line, func_id, short_func, &last_two, aeneas_by_key);
    (llbc_val, aeneas_val)
}

struct RegistryCtx<'a> {
    root: &'a Path,
    modules: &'a [LeanModuleRecord],
    aeneas_by_key: &'a BTreeMap<String, AeneasRecord>,
    llbc_by_key: &'a BTreeMap<String, LlbcRecord>,
    sym_lookup: &'a BTreeMap<String, DeclRecord>,
}

fn resolve_syms(
    idx: usize,
    reg_entry: Option<&RegEntry>,
    ctx: &RegistryCtx<'_>,
) -> (String, String, String, String) {
    let (def_sym, thm_sym) = reg_entry
        .map(|r| (r.lean_def.clone(), r.lean_thm.clone()))
        .unwrap_or_else(|| {
            (
                format!("ENN.Registry.f{idx:03}"),
                format!("ENN.Registry.t{idx:03}"),
            )
        });
    let thm_mod = ctx
        .sym_lookup
        .get(&thm_sym)
        .map(|d| d.module.clone())
        .unwrap_or_else(|| symbol_module(&thm_sym, ctx.modules, "BauxResearch.Semantics.Registry"));
    let def_mod = ctx
        .sym_lookup
        .get(&def_sym)
        .map(|d| d.module.clone())
        .unwrap_or_else(|| symbol_module(&def_sym, ctx.modules, &thm_mod));
    (def_sym, thm_sym, thm_mod, def_mod)
}

fn build_func(
    idx: usize,
    func_val: Option<&Value>,
    reg_entry: Option<&RegEntry>,
    ctx: &RegistryCtx<'_>,
) -> (String, Value) {
    let file = func_val
        .and_then(|v| v.get("file"))
        .and_then(|v| v.as_str())
        .map(str::to_owned)
        .or_else(|| reg_entry.map(|r| r.file.clone()))
        .unwrap_or_default();
    let line = func_val
        .and_then(|v| v.get("line"))
        .and_then(|v| v.as_u64())
        .map(|u| u as usize)
        .or_else(|| reg_entry.map(|r| r.line_num))
        .unwrap_or(1);
    let name = func_val
        .and_then(|v| v.get("name"))
        .and_then(|v| v.as_str())
        .map(str::to_owned)
        .or_else(|| reg_entry.map(|r| r.func_name.clone()))
        .unwrap_or_default();
    let func_id = func_val
        .and_then(|v| v.get("id"))
        .and_then(|v| v.as_str())
        .map(str::to_owned)
        .or_else(|| reg_entry.map(|r| r.func_id.clone()))
        .unwrap_or_else(|| format!("f{idx:03}"));
    let language = func_val
        .and_then(|v| v.get("language"))
        .and_then(|v| v.as_str())
        .unwrap_or_else(|| {
            if file.ends_with(".py") {
                "python"
            } else {
                "rust"
            }
        })
        .to_owned();
    let depth = func_val
        .and_then(|v| v.get("depth"))
        .and_then(|v| v.as_u64())
        .unwrap_or(0) as usize;
    let root_val = func_val
        .and_then(|v| v.get("root"))
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_owned();

    let (def_sym, thm_sym, thm_mod, def_mod) = resolve_syms(idx, reg_entry, ctx);
    let (src_start, src_code) = source_snippet(ctx.root, &file, line, 22);
    let (llbc_val, aeneas_val) = match_ir(
        &language,
        &file,
        line,
        &func_id,
        &name,
        ctx.aeneas_by_key,
        ctx.llbc_by_key,
    );
    let val = wire_json!({
        "idx": idx,
        "id": func_id,
        "name": name,
        "file": file,
        "line": line,
        "language": language,
        "crate": classify_crate(&file),
        "depth": depth,
        "root": root_val,
        "src_start": src_start,
        "src_code": src_code,
        "llbc": llbc_val,
        "aeneas": aeneas_val,
        "lean_def": def_sym,
        "lean_thm": thm_sym,
        "lean_thm_module": thm_mod,
        "lean_def_module": def_mod,
    });
    (thm_mod, val)
}

pub(crate) fn extract_registry(
    root: &Path,
    modules: &mut [LeanModuleRecord],
    aeneas_by_key: &BTreeMap<String, AeneasRecord>,
    llbc_by_key: &BTreeMap<String, LlbcRecord>,
    sym_lookup: &BTreeMap<String, DeclRecord>,
) -> Result<(Vec<Value>, Map), String> {
    let cov_path = root.join("research/upstream/coverage_manifest.json");
    let cov_bytes =
        fs::read(&cov_path).map_err(|e| format!("failed to read {}: {e}", cov_path.display()))?;
    let cov_json: Value = json::from_slice(&cov_bytes)
        .map_err(|e| format!("invalid JSON in {}: {e}", cov_path.display()))?;
    let cov_functions: Vec<Value> = cov_json
        .get("reachable_functions")
        .or_else(|| cov_json.get("functions"))
        .and_then(|v| v.as_seq())
        .map(|s| s.to_vec())
        .unwrap_or_default();

    let (reg_entries, registry_symbols) = load_entries(root, sym_lookup)?;
    let mut mod_reg_counts: BTreeMap<String, usize> = BTreeMap::new();
    let total_count = cov_functions.len().max(reg_entries.len());
    let mut functions_out: Vec<Value> = Vec::with_capacity(total_count);
    let ctx = RegistryCtx {
        root,
        modules,
        aeneas_by_key,
        llbc_by_key,
        sym_lookup,
    };

    for idx in 0..total_count {
        let (thm_mod, func_json) =
            build_func(idx, cov_functions.get(idx), reg_entries.get(&idx), &ctx);
        *mod_reg_counts.entry(thm_mod).or_insert(0) += 1;
        functions_out.push(func_json);
    }

    for m in modules.iter_mut() {
        m.proved_registry_count = *mod_reg_counts.get(&m.module).unwrap_or(&0);
        if m.module == "BauxResearch.Semantics.Registry" {
            m.proved_registry_count = functions_out.len();
        }
    }
    Ok((functions_out, registry_symbols))
}
