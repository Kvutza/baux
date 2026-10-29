use crate::research::site::{LlbcRecord, shared_path};
use baux::artifact::json::{self, Value};
use baux::artifact::wire_json;
use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

fn span_line(span: Option<&Value>, span_table: &mut BTreeMap<u64, usize>) -> usize {
    let Some(s) = span else {
        return 0;
    };
    if let Some(val_arr) = s.get("Value").and_then(|v| v.as_seq()) {
        let sid = val_arr.first().and_then(|v| v.as_u64()).unwrap_or(0);
        let line = val_arr
            .get(1)
            .and_then(|v| v.get("data"))
            .and_then(|d| d.get("beg"))
            .and_then(|b| b.get("line"))
            .and_then(|v| v.as_u64())
            .unwrap_or(0) as usize;
        if sid > 0 && line > 0 {
            span_table.insert(sid, line);
        }
        return line;
    }
    if let Some(sid) = s.get("Deduplicated").and_then(|v| v.as_u64()) {
        return span_table.get(&sid).copied().unwrap_or(0);
    }
    s.get("data")
        .or_else(|| s.get("Untagged").and_then(|u| u.get("data")))
        .and_then(|d| d.get("beg"))
        .and_then(|b| b.get("line"))
        .and_then(|v| v.as_u64())
        .unwrap_or(0) as usize
}

fn place_local(place: &Value) -> u64 {
    let Some(k) = place.get("kind") else {
        return 0;
    };
    if let Some(loc) = k.get("Local").and_then(|v| v.as_u64()) {
        return loc;
    }
    if let Some(base) = k
        .get("Projection")
        .and_then(|v| v.as_seq())
        .and_then(|a| a.first())
    {
        return place_local(base);
    }
    0
}

fn format_operand(op: &Value) -> String {
    if let Some(p) = op.get("Copy") {
        return format!("Copy(_{})", place_local(p));
    }
    if let Some(p) = op.get("Move") {
        return format!("Move(_{})", place_local(p));
    }
    if op.get("Const").is_some() {
        return "Const".to_owned();
    }
    "Op".to_owned()
}

fn first_elem(v: &Value) -> Option<&Value> {
    v.as_seq().and_then(|a| a.first())
}

fn op_name(o: &Value, fallback: &str) -> String {
    if let Some(s) = o.as_str() {
        return s.to_owned();
    }
    o.as_map()
        .and_then(|om| om.keys().next())
        .and_then(|k| k.as_str())
        .unwrap_or(fallback)
        .to_owned()
}

fn format_rvalue(rv: &Value) -> String {
    let Some(m) = rv.as_map() else {
        return rv.as_str().unwrap_or("Rvalue").to_owned();
    };
    let Some((k_val, v_val)) = m.iter().next() else {
        return "Rvalue".to_owned();
    };
    let kind = k_val.as_str().unwrap_or("Rvalue");
    match kind {
        "Use" => format!(
            "Use({})",
            format_operand(first_elem(v_val).unwrap_or(v_val))
        ),
        "Ref" => first_elem(v_val)
            .map(|p| format!("Ref(_{})", place_local(p)))
            .unwrap_or_else(|| "Ref".to_owned()),
        "UnaryOp" => first_elem(v_val)
            .map(|o| format!("UnaryOp({})", op_name(o, "UnOp")))
            .unwrap_or_else(|| "UnaryOp".to_owned()),
        "BinaryOp" => first_elem(v_val)
            .map(|o| format!("BinaryOp({})", op_name(o, "BinOp")))
            .unwrap_or_else(|| "BinaryOp".to_owned()),
        "Discriminant" => format!(
            "Discriminant(_{})",
            place_local(first_elem(v_val).unwrap_or(v_val))
        ),
        other => other.to_owned(),
    }
}

fn format_callee(func: &Value, callee_names: &BTreeMap<u64, String>) -> String {
    if let Some(k) = func.get("Regular").and_then(|r| r.get("kind")) {
        if let Some(fid) = k.get("Fun").and_then(|v| v.as_u64()) {
            return callee_names
                .get(&fid)
                .cloned()
                .unwrap_or_else(|| format!("Fun#{fid}"));
        }
        if let Some(bk) = k
            .get("Builtin")
            .and_then(|v| v.as_map())
            .and_then(|bm| bm.keys().next())
            .and_then(|k| k.as_str())
        {
            return format!("Builtin::{bk}");
        }
        if let Some(mname) = k
            .get("TraitMethod")
            .and_then(|v| v.as_seq())
            .and_then(|tm| tm.get(1))
            .and_then(|v| v.as_str())
        {
            return format!("TraitMethod::{mname}");
        }
    }
    if let Some(p) = func.get("FnPtr") {
        return format!("FnPtr(_{})", place_local(p));
    }
    "Fn".to_owned()
}

fn switch_discr(v_val: &Value) -> String {
    let scrut = v_val
        .get("data")
        .and_then(|d| d.get("scrutinee"))
        .or_else(|| v_val.get("scrutinee"));
    let Some(sc) = scrut else {
        return "Discr".to_owned();
    };
    if let Some(op) = sc.get("Value") {
        return format_operand(op);
    }
    sc.as_map()
        .and_then(|sm| sm.keys().next())
        .and_then(|k| k.as_str())
        .unwrap_or("Discr")
        .to_owned()
}

fn assign_call(k_str: &str, v_val: &Value, callee_names: &BTreeMap<u64, String>) -> String {
    if k_str == "Assign" {
        let Some(arr) = v_val.as_seq() else {
            return "Assign".to_owned();
        };
        let loc = arr.first().map(place_local).unwrap_or(0);
        let rv = arr
            .get(1)
            .map(format_rvalue)
            .unwrap_or_else(|| "Rvalue".to_owned());
        return format!("_{loc} := {rv}");
    }
    let call_inner = v_val.get("call").unwrap_or(v_val);
    let dest = call_inner.get("dest").map(place_local).unwrap_or(0);
    let callee = call_inner
        .get("func")
        .map(|f| format_callee(f, callee_names))
        .unwrap_or_else(|| "Fn".to_owned());
    format!("Call(_{dest} := {callee})")
}

fn stmt_kind<'a>(
    kind_val: &'a Value,
    callee_names: &BTreeMap<u64, String>,
) -> (String, Option<&'a [Value]>, Option<&'a Value>) {
    if let Some(s) = kind_val.as_str() {
        return (s.to_owned(), None, None);
    }
    let Some((k_key, v_val)) = kind_val.as_map().and_then(|km| km.iter().next()) else {
        return ("Stmt".to_owned(), None, None);
    };
    let k_str = k_key.as_str().unwrap_or("Stmt");
    match k_str {
        "StorageLive" | "StorageDead" => (
            format!("{k_str}(_{})", v_val.as_u64().unwrap_or(0)),
            None,
            None,
        ),
        "Assign" | "Call" => (assign_call(k_str, v_val, callee_names), None, None),
        "Switch" => (
            format!("Switch({})", switch_discr(v_val)),
            v_val
                .get("branches")
                .and_then(|b| b.as_seq())
                .map(|v| v.as_slice()),
            None,
        ),
        "Loop" => ("Loop".to_owned(), None, Some(v_val)),
        "Drop" => {
            let loc = first_elem(v_val)
                .map(place_local)
                .unwrap_or_else(|| place_local(v_val));
            (format!("Drop(_{loc})"), None, None)
        }
        "Assert" => ("Assert(...)".to_owned(), None, None),
        "Break" | "Continue" => (
            format!("{k_str}({})", v_val.as_u64().unwrap_or(0)),
            None,
            None,
        ),
        other => (other.to_owned(), None, None),
    }
}

struct LlbcWalker<'a> {
    span_table: &'a mut BTreeMap<u64, usize>,
    callee_names: &'a BTreeMap<u64, String>,
    statements_count: usize,
    formatted_stmts: Vec<String>,
}

fn walk_block(block: &Value, fallback_line: usize, walker: &mut LlbcWalker<'_>) {
    let Some(stmts) = block.get("statements").and_then(|v| v.as_seq()) else {
        return;
    };
    for stmt in stmts {
        walker.statements_count += 1;
        let l = span_line(stmt.get("span"), walker.span_table);
        let line_no = if l > 0 { l } else { fallback_line };
        let Some(kind_val) = stmt.get("kind") else {
            continue;
        };
        let (desc, switch_branches, loop_body) = stmt_kind(kind_val, walker.callee_names);
        if walker.formatted_stmts.len() < 36 {
            walker
                .formatted_stmts
                .push(format!("  [L{line_no}] {desc}"));
        }
        if let Some(branches) = switch_branches {
            for br in branches {
                walk_block(br, line_no, walker);
            }
        }
        if let Some(lb) = loop_body {
            walk_block(lb, line_no, walker);
        }
    }
}

fn local_files(translated: &Value) -> BTreeMap<usize, String> {
    let mut files_map = BTreeMap::new();
    let Some(files_arr) = translated.get("files").and_then(|v| v.as_seq()) else {
        return files_map;
    };
    for (idx, f) in files_arr.iter().enumerate() {
        if let Some(local_name) = f
            .get("name")
            .and_then(|n| n.get("Local"))
            .and_then(|v| v.as_str())
        {
            if !local_name.starts_with('/') {
                files_map.insert(idx, local_name.to_owned());
            }
        }
    }
    files_map
}

fn ident_segs(meta: &Value, skip_opaque: bool) -> Vec<String> {
    let mut segs = Vec::new();
    let Some(name_arr) = meta.get("name").and_then(|v| v.as_seq()) else {
        return segs;
    };
    for pe in name_arr {
        if let Some(s) = pe
            .get("Ident")
            .and_then(|v| v.as_seq())
            .and_then(|a| a.first())
            .and_then(|v| v.as_str())
        {
            if !skip_opaque || !s.starts_with("opaque_external_impl") {
                segs.push(s.to_owned());
            }
        }
    }
    segs
}

fn func_map(fun_decls: &[Value], span_table: &mut BTreeMap<u64, usize>) -> BTreeMap<u64, String> {
    let mut callee_names = BTreeMap::new();
    for (idx_fd, fd) in fun_decls.iter().enumerate() {
        let Some(fd_obj) = fd.as_map() else {
            continue;
        };
        let def_id = fd_obj
            .get("def_id")
            .and_then(|v| v.as_u64())
            .unwrap_or(idx_fd as u64);
        if let Some(meta) = fd_obj.get("item_meta") {
            let _ = span_line(meta.get("span"), span_table);
            let segs = ident_segs(meta, true);
            if !segs.is_empty() {
                callee_names.insert(def_id, segs.join("::"));
            }
        }
    }
    callee_names
}

fn span_data<'a>(meta: &'a Value, fallback: &'a Value) -> &'a Value {
    let Some(s) = meta.get("span") else {
        return fallback;
    };
    if let Some(d) = s.get("data") {
        return d;
    }
    if let Some(d) = s.get("Untagged").and_then(|u| u.get("data")) {
        return d;
    }
    s.get("Value")
        .and_then(|v| v.as_seq())
        .and_then(|a| a.get(1))
        .and_then(|v| v.get("data"))
        .unwrap_or(fallback)
}

fn named_locals(
    locals_obj: &Value,
    span_table: &mut BTreeMap<u64, usize>,
    default_args: usize,
) -> (usize, usize, Vec<String>) {
    let arg_count = locals_obj
        .get("arg_count")
        .and_then(|v| v.as_u64())
        .map(|u| u as usize)
        .unwrap_or(default_args);
    let Some(loc_arr) = locals_obj.get("locals").and_then(|v| v.as_seq()) else {
        return (arg_count, 0, Vec::new());
    };
    let mut locals_out = Vec::new();
    for loc in loc_arr {
        let l_idx = loc.get("index").and_then(|v| v.as_u64()).unwrap_or(0);
        let l_line = span_line(loc.get("span"), span_table);
        if let Some(nm) = loc.get("name").and_then(|v| v.as_str()) {
            let entry = if l_line > 0 {
                format!("_{l_idx}: {nm} [L{l_line}]")
            } else {
                format!("_{l_idx}: {nm}")
            };
            locals_out.push(entry);
        }
    }
    (arg_count, loc_arr.len(), locals_out)
}

fn index_llbc(
    llbc_by_key: &mut BTreeMap<String, LlbcRecord>,
    fname: &str,
    beg_line: usize,
    segs: &[String],
    rec: LlbcRecord,
) {
    llbc_by_key.insert(format!("{fname}:{beg_line}"), rec.clone());
    llbc_by_key.insert(format!("rust/{fname}:{beg_line}"), rec.clone());
    if segs.len() >= 2 {
        let tail_key = format!("{}::{}", segs[segs.len() - 2], segs[segs.len() - 1]);
        llbc_by_key.entry(tail_key).or_insert_with(|| rec.clone());
    }
    if let Some(last) = segs.last() {
        llbc_by_key
            .entry(format!("{fname}::{last}"))
            .or_insert_with(|| rec.clone());
        llbc_by_key
            .entry(format!("rust/{fname}::{last}"))
            .or_insert_with(|| rec.clone());
        llbc_by_key.entry(last.clone()).or_insert(rec);
    }
}

fn parse_decl(
    idx_fd: usize,
    fd: &Value,
    crate_name: &str,
    files_map: &BTreeMap<usize, String>,
    callee_names: &BTreeMap<u64, String>,
    span_table: &mut BTreeMap<u64, usize>,
) -> (bool, Option<(Vec<String>, LlbcRecord)>) {
    let (Some(fd_obj), null_val) = (fd.as_map(), Value::null()) else {
        return (false, None);
    };
    let Some(meta) = fd_obj.get("item_meta") else {
        return (false, None);
    };
    let is_local_item = meta
        .get("is_local")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    let structured_opt = fd_obj.get("body").and_then(|b| b.get("Structured"));
    let is_local_structured = is_local_item && structured_opt.is_some();
    let sp_data = span_data(meta, &null_val);
    let Some(fid) = sp_data.get("file_id").and_then(|v| v.as_u64()) else {
        return (is_local_structured, None);
    };
    let Some(fname) = files_map.get(&(fid as usize)) else {
        return (is_local_structured, None);
    };

    let def_id = fd_obj
        .get("def_id")
        .and_then(|v| v.as_u64())
        .unwrap_or(idx_fd as u64);
    let opacity = meta
        .get("opacity")
        .and_then(|v| v.as_str())
        .unwrap_or("Transparent")
        .to_owned();
    let beg_line = sp_data
        .get("beg")
        .and_then(|b| b.get("line"))
        .and_then(|v| v.as_u64())
        .unwrap_or(0) as usize;
    let end_line = sp_data
        .get("end")
        .and_then(|e| e.get("line"))
        .and_then(|v| v.as_u64())
        .unwrap_or(0) as usize;

    let segs = ident_segs(meta, false);
    let full_ident = segs.join("::");
    let sig = fd_obj.get("signature").unwrap_or(&null_val);
    let n_inputs = sig
        .get("inputs")
        .and_then(|v| v.as_seq())
        .map(|a| a.len())
        .unwrap_or(0);
    let out_ty = sig
        .get("output")
        .map(|o| op_name(o, "Adt"))
        .unwrap_or_else(|| "Adt".to_owned());

    let (arg_count, locals_count, locals_vec) = structured_opt
        .and_then(|s| s.get("locals"))
        .map(|l| named_locals(l, span_table, n_inputs))
        .unwrap_or((n_inputs, 0, Vec::new()));
    let mut walker = LlbcWalker {
        span_table,
        callee_names,
        statements_count: 0,
        formatted_stmts: Vec::new(),
    };
    if let Some(body_block) = structured_opt.and_then(|s| s.get("body")) {
        walk_block(body_block, beg_line, &mut walker);
    }

    let rec = build_rec(
        (crate_name, &full_ident, def_id, &opacity),
        (fname, beg_line, end_line, &out_ty),
        (arg_count, locals_count, walker.statements_count),
        locals_vec,
        walker.formatted_stmts,
    );
    (is_local_structured, Some((segs, rec)))
}

fn build_rec(
    ident: (&str, &str, u64, &str),
    loc: (&str, usize, usize, &str),
    counts: (usize, usize, usize),
    locals_vec: Vec<String>,
    formatted_stmts: Vec<String>,
) -> LlbcRecord {
    let (crate_name, full_ident, def_id, opacity) = ident;
    let (fname, beg_line, end_line, out_ty) = loc;
    let (arg_count, locals_count, statements_count) = counts;
    let locals_header = if locals_vec.is_empty() {
        format!("({locals_count} anonymous temporaries)")
    } else {
        locals_vec.join(", ")
    };
    let stmts_body = if formatted_stmts.is_empty() {
        format!("  [L{beg_line}] // Opaque / external declaration ({opacity})")
    } else {
        formatted_stmts.join("\n")
    };
    let sig_str = format!(
        "fn({arg_count} args, {locals_count} locals, {statements_count} stmts) -> {out_ty}"
    );
    let snippet = format!(
        "// Charon 0.1.178 LLBC MIR [{crate_name}.llbc :: def_id={def_id} :: {opacity}]\n// \
             Item      : {full_ident}\n// Span      : {fname}:{beg_line}-{end_line}\n// Signature \
             : {sig_str}\n// Locals    : {locals_header}\n{stmts_body}"
    );
    LlbcRecord {
        crate_name: crate_name.to_owned(),
        full_ident: full_ident.to_owned(),
        def_id,
        opacity: opacity.to_owned(),
        file: fname.to_owned(),
        span: format!("{fname}:{beg_line}-{end_line}"),
        beg_line,
        end_line,
        arg_count,
        locals_count,
        statements_count,
        named_locals: locals_vec,
        sig: sig_str,
        snippet,
    }
}

fn process_crate(
    crate_name: &str,
    llbc_path: &Path,
    llbc_by_key: &mut BTreeMap<String, LlbcRecord>,
) -> Option<Value> {
    let bytes = fs::read(llbc_path).ok()?;
    let data = json::from_slice::<Value>(&bytes).ok()?;
    let translated = data.get("translated").unwrap_or(&data);
    let files_map = local_files(translated);
    let fun_decls: Vec<Value> = translated
        .get("fun_decls")
        .and_then(|v| v.as_seq())
        .map(|s| s.to_vec())
        .unwrap_or_default();

    let mut span_table: BTreeMap<u64, usize> = BTreeMap::new();
    let callee_names = func_map(&fun_decls, &mut span_table);
    let mut local_func_count = 0usize;
    let mut local_structured_count = 0usize;

    for (idx_fd, fd) in fun_decls.iter().enumerate() {
        let (is_struct, parsed) = parse_decl(
            idx_fd,
            fd,
            crate_name,
            &files_map,
            &callee_names,
            &mut span_table,
        );
        if is_struct {
            local_structured_count += 1;
        }
        if let Some((segs, rec)) = parsed {
            local_func_count += 1;
            let fname = rec.file.clone();
            let beg_line = rec.beg_line;
            index_llbc(llbc_by_key, &fname, beg_line, &segs, rec);
        }
    }

    Some(wire_json!({
        "crate": crate_name,
        "llbc_file": format!(".cache/baux/aeneas/probe-full-upstream/{crate_name}.llbc"),
        "total_fun_decls": fun_decls.len(),
        "local_fun_decls": local_func_count,
        "local_structured_functions": local_structured_count,
        "local_source_files": files_map.len(),
    }))
}

pub(crate) fn extract_llbc(root: &Path) -> (BTreeMap<String, LlbcRecord>, Vec<Value>) {
    let llbc_dir = shared_path(root, ".cache/baux/aeneas/probe-full-upstream");
    let mut llbc_by_key: BTreeMap<String, LlbcRecord> = BTreeMap::new();
    let mut crate_summaries: Vec<Value> = Vec::new();
    for crate_name in ["bpann", "ennbo", "enn-py"] {
        let llbc_path = llbc_dir.join(format!("{crate_name}.llbc"));
        if llbc_path.exists() {
            if let Some(summary) = process_crate(crate_name, &llbc_path, &mut llbc_by_key) {
                crate_summaries.push(summary);
            }
        }
    }
    (llbc_by_key, crate_summaries)
}
