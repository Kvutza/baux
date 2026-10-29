use baux::artifact::json::Value;
use baux::artifact::wire_json;
use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::fs;
use std::path::Path;

fn is_triple(bytes: &[u8], q: u8) -> bool {
    bytes.len() >= 3 && bytes[0] == q && bytes[1] == q && bytes[2] == q
}

fn step_code(bytes: &[u8], i: &mut usize, state: &mut u8, out: &mut String) {
    let b = bytes[*i];
    if b == b'#' {
        out.push(' ');
        *i += 1;
        *state = 1;
    } else if is_triple(&bytes[*i..], b'"') {
        out.push_str("   ");
        *i += 3;
        *state = 4;
    } else if is_triple(&bytes[*i..], b'\'') {
        out.push_str("   ");
        *i += 3;
        *state = 5;
    } else if b == b'"' || b == b'\'' {
        out.push(' ');
        *i += 1;
        *state = if b == b'"' { 2 } else { 3 };
    } else {
        out.push(b as char);
        *i += 1;
    }
}

fn step_str(bytes: &[u8], i: &mut usize, state: &mut u8, out: &mut String) {
    let b = bytes[*i];
    let q = if *state == 2 || *state == 4 {
        b'"'
    } else {
        b'\''
    };
    if b == b'\\' {
        out.push(' ');
        if bytes.get(*i + 1) == Some(&b'\n') {
            out.push('\n');
        } else {
            out.push(' ');
        }
        *i = (*i + 2).min(bytes.len());
    } else if *state >= 4 && is_triple(&bytes[*i..], q) {
        out.push_str("   ");
        *i += 3;
        *state = 0;
    } else if *state < 4 && (b == q || b == b'\n') {
        out.push(if b == b'\n' { '\n' } else { ' ' });
        *i += 1;
        *state = 0;
    } else {
        out.push(if b == b'\n' { '\n' } else { ' ' });
        *i += 1;
    }
}

fn blank_src(raw: &str) -> String {
    let bytes = raw.as_bytes();
    let mut out = String::with_capacity(bytes.len());
    let (mut i, mut state) = (0_usize, 0_u8);
    while i < bytes.len() {
        match state {
            0 => step_code(bytes, &mut i, &mut state, &mut out),
            1 => {
                if bytes[i] == b'\n' {
                    out.push('\n');
                    state = 0;
                } else {
                    out.push(' ');
                }
                i += 1;
            }
            _ => step_str(bytes, &mut i, &mut state, &mut out),
        }
    }
    out
}

fn collect_stmts(blanked: &str) -> Vec<(usize, String)> {
    let lines: Vec<&str> = blanked.lines().collect();
    let mut out = Vec::new();
    let mut idx = 0;
    while idx < lines.len() {
        let t = lines[idx].trim();
        if t.starts_with("import ") || t.starts_with("from ") {
            let line_no = idx + 1;
            let mut stmt = t.trim_end_matches('\\').to_owned();
            while (stmt.contains('(') && !stmt.contains(')'))
                || lines[idx].trim_end().ends_with('\\')
            {
                if idx + 1 >= lines.len() {
                    break;
                }
                idx += 1;
                stmt.push(' ');
                stmt.push_str(lines[idx].trim().trim_end_matches('\\'));
            }
            out.push((line_no, stmt));
        }
        idx += 1;
    }
    out
}

fn resolve_rel(mod_parts: &[&str], level: usize, module: &str) -> String {
    let base = if mod_parts.len() >= level {
        &mod_parts[..mod_parts.len() - level]
    } else {
        &[]
    };
    let mut parts: Vec<&str> = base.to_vec();
    if !module.is_empty() {
        parts.extend(module.split('.'));
    }
    parts.join(".")
}

fn parse_from(stmt: &str, mod_parts: &[&str], imported: &mut BTreeSet<String>) {
    let Some((from_p, imp_p)) = stmt
        .strip_prefix("from ")
        .and_then(|s| s.split_once(" import "))
    else {
        return;
    };
    let spec = from_p.trim();
    let level = spec.chars().take_while(|&c| c == '.').count();
    let module = spec[level..].trim();
    if level == 0 {
        if !module.is_empty() {
            imported.insert(module.to_owned());
        }
        return;
    }
    let full_mod = resolve_rel(mod_parts, level, module);
    if !module.is_empty() {
        imported.insert(full_mod);
        return;
    }
    for part in imp_p.replace(['(', ')'], " ").split(',') {
        let Some(name) = part.split_whitespace().next() else {
            continue;
        };
        let target = if full_mod.is_empty() {
            name.to_owned()
        } else {
            format!("{full_mod}.{name}")
        };
        imported.insert(target);
    }
}

fn scan_imports(blanked: &str, rel_path: &str) -> BTreeSet<String> {
    let mut imported = BTreeSet::new();
    let stem = rel_path.strip_suffix(".py").unwrap_or(rel_path);
    let mod_parts: Vec<&str> = stem.split('/').skip(1).collect();
    for (_, stmt) in collect_stmts(blanked) {
        if let Some(rest) = stmt.strip_prefix("import ") {
            for part in rest.split(',') {
                if let Some(name) = part.split_whitespace().next() {
                    imported.insert(name.to_owned());
                }
            }
        } else if stmt.starts_with("from ") {
            parse_from(&stmt, &mod_parts, &mut imported);
        }
    }
    imported
}

fn mod_path(enn_root: &Path, mod_name: &str) -> Option<String> {
    let base = format!("src/{}", mod_name.replace('.', "/"));
    let rel_py = format!("{base}.py");
    if enn_root.join(&rel_py).is_file() {
        return Some(rel_py);
    }
    let rel_init = format!("{base}/__init__.py");
    enn_root.join(&rel_init).is_file().then_some(rel_init)
}

fn walk_py(dir: &Path, enn_root: &Path, out: &mut BTreeMap<String, String>) {
    let Ok(entries) = fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let p = entry.path();
        if p.is_dir() {
            walk_py(&p, enn_root, out);
        } else if p.extension().and_then(|s| s.to_str()) == Some("py")
            && let Ok(rel) = p.strip_prefix(enn_root)
            && let Ok(raw) = fs::read_to_string(&p)
        {
            let rel_s = rel.to_string_lossy().replace('\\', "/");
            out.insert(rel_s, blank_src(&raw));
        }
    }
}

fn def_name(trimmed: &str) -> Option<&str> {
    let rest = trimmed
        .strip_prefix("def ")
        .or_else(|| trimmed.strip_prefix("async def "))?;
    let name = rest.split('(').next()?.trim();
    (!name.is_empty()).then_some(name)
}

fn tree_funcs(blanked: &str, rel_path: &str, depth: usize) -> Vec<Value> {
    let mut funcs = Vec::new();
    let mut curr_cls: Option<String> = None;
    for (idx, line) in blanked.lines().enumerate() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('@') {
            continue;
        }
        let indent = line.len() - line.trim_start().len();
        if indent == 0 {
            if let Some(rest) = trimmed.strip_prefix("class ") {
                let cname = rest
                    .split(|c: char| !c.is_alphanumeric() && c != '_')
                    .next()
                    .unwrap_or("");
                curr_cls = (!cname.is_empty()).then(|| cname.to_owned());
            } else if let Some(fname) = def_name(trimmed) {
                curr_cls = None;
                funcs.push(wire_json!({
                    "depth": depth,
                    "file": rel_path,
                    "id": format!("{rel_path}:{fname}"),
                    "language": "python",
                    "line": idx + 1,
                    "name": fname,
                    "root": "create_optimizer",
                }));
            } else if trimmed.starts_with(|c: char| c.is_alphabetic() || c == '_') {
                curr_cls = None;
            }
        } else if indent == 4
            && let Some(cls) = &curr_cls
            && let Some(mname) = def_name(trimmed)
        {
            let full = format!("{cls}.{mname}");
            funcs.push(wire_json!({
                "depth": depth,
                "file": rel_path,
                "id": format!("{rel_path}:{full}"),
                "language": "python",
                "line": idx + 1,
                "name": full,
                "root": "create_optimizer",
            }));
        }
    }
    funcs
}

pub(super) fn trace_python(enn_root: &Path) -> (Vec<Value>, BTreeSet<String>) {
    let mut trees = BTreeMap::new();
    walk_py(&enn_root.join("src"), enn_root, &mut trees);
    let mut adj: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    for (rel, blanked) in &trees {
        let targets: BTreeSet<String> = scan_imports(blanked, rel)
            .into_iter()
            .filter_map(|spec| mod_path(enn_root, &spec))
            .collect();
        adj.insert(rel.clone(), targets);
    }
    let starts = [
        "src/enn/turbo/rust_optimizer.py",
        "src/enn/__init__.py",
        "src/enn/turbo/optimizer_config.py",
    ];
    let mut visited = BTreeSet::new();
    let mut depth_map = BTreeMap::new();
    let mut queue = VecDeque::new();
    for s in starts {
        if trees.contains_key(s) {
            visited.insert(s.to_owned());
            depth_map.insert(s.to_owned(), 0_usize);
            queue.push_back(s.to_owned());
        }
    }
    while let Some(curr) = queue.pop_front() {
        let d = depth_map[&curr];
        if let Some(nexts) = adj.get(&curr) {
            for nxt in nexts {
                if visited.insert(nxt.clone()) {
                    depth_map.insert(nxt.clone(), d + 1);
                    queue.push_back(nxt.clone());
                }
            }
        }
    }
    let mut funcs = Vec::new();
    for rel in &visited {
        if let Some(blanked) = trees.get(rel) {
            funcs.extend(tree_funcs(blanked, rel, depth_map[rel]));
        }
    }
    (funcs, visited)
}

fn file_imports(enn_root: &Path, rel_path: &str) -> Vec<(String, usize, String)> {
    let Ok(raw) = fs::read_to_string(enn_root.join(rel_path)) else {
        return Vec::new();
    };
    let blanked = blank_src(&raw);
    let mut out = Vec::new();
    for (line_no, stmt) in collect_stmts(&blanked) {
        if let Some(rest) = stmt.strip_prefix("import ") {
            for part in rest.split(',') {
                if let Some(m) = part.split_whitespace().next()
                    && m.split('.').next() != Some("enn")
                {
                    out.push((rel_path.to_owned(), line_no, m.to_owned()));
                }
            }
        } else if let Some((from_p, _)) = stmt
            .strip_prefix("from ")
            .and_then(|s| s.split_once(" import "))
        {
            let spec = from_p.trim();
            if !spec.starts_with('.') && !spec.is_empty() && spec.split('.').next() != Some("enn") {
                out.push((rel_path.to_owned(), line_no, spec.to_owned()));
            }
        }
    }
    out
}

pub(super) fn python_imports(enn_root: &Path, sources: &BTreeSet<String>) -> Vec<Value> {
    let mut rows = Vec::new();
    for path in sources {
        if path.ends_with(".py") {
            rows.extend(file_imports(enn_root, path));
        }
    }
    rows.sort();
    rows.into_iter()
        .map(|(file, line, module)| {
            wire_json!({
                "file": file,
                "line": line,
                "module": module,
            })
        })
        .collect()
}
