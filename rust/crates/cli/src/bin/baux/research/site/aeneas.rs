use crate::research::site::{AeneasRecord, lean_files};
use crate::site_lean::parse_decl;
use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

fn brace_type(chars: &mut std::iter::Peekable<std::str::Chars<'_>>) -> String {
    let mut inner = String::new();
    let mut d = 1usize;
    for ic in chars.by_ref() {
        if ic == '{' {
            d += 1;
        } else if ic == '}' {
            d -= 1;
            if d == 0 {
                break;
            }
        } else if d == 1 {
            inner.push(ic);
        }
    }
    let Some(last) = inner.split("::").last() else {
        return String::new();
    };
    last.split('<').next().unwrap_or(last).trim().to_owned()
}

pub(crate) fn normalize_key(s: &str) -> String {
    let mut out = String::new();
    let mut chars = s.chars().peekable();
    let mut depth = 0usize;
    while let Some(c) = chars.next() {
        match c {
            '{' => {
                let base = brace_type(&mut chars);
                if !base.is_empty() {
                    out.push_str(&base);
                }
            }
            '<' => depth += 1,
            '>' => depth = depth.saturating_sub(1),
            _ if depth == 0 => out.push(c),
            _ => {}
        }
    }
    let parts: Vec<&str> = out
        .split("::")
        .filter(|p| !p.is_empty() && !p.starts_with("Impl"))
        .collect();
    if parts.len() >= 2 {
        format!("{}::{}", parts[parts.len() - 2], parts[parts.len() - 1])
    } else {
        out
    }
}

fn external_item(rec: &AeneasRecord) -> bool {
    if rec.span.contains("'/rustc/") || rec.span.contains("/.cargo/") {
        return true;
    }
    let trimmed = rec.rust_item.trim_start_matches('&');
    [
        "core::",
        "alloc::",
        "std::",
        "ndarray::",
        "rand::",
        "rand_core::",
        "rand_chacha::",
        "rand_pcg::",
        "serde::",
        "serde_json::",
        "pyo3::",
        "numpy::",
        "rayon::",
        "faiss::",
        "memmap2::",
        "thiserror::",
    ]
    .iter()
    .any(|prefix| trimmed.starts_with(prefix))
}

fn impl_type(rust_item: &str) -> Option<&str> {
    let open = rust_item.find('{')?;
    let close = rust_item[open + 1..].find('}')?;
    let inside = &rust_item[open + 1..open + 1 + close];
    let impl_ty = inside.rsplit("::").next()?;
    let trimmed = impl_ty.split('<').next().unwrap_or(impl_ty).trim();
    (!trimmed.is_empty()).then_some(trimmed)
}

fn index_lines(
    aeneas_by_key: &mut BTreeMap<String, AeneasRecord>,
    rec: &AeneasRecord,
    src_file: &str,
    after_q2: &str,
    short_func: &str,
) {
    let Some(lpos) = after_q2.find("lines ") else {
        return;
    };
    let line_part = &after_q2[lpos + 6..];
    let beg_str: String = line_part
        .chars()
        .take_while(|c| c.is_ascii_digit())
        .collect();
    let Ok(beg_line) = beg_str.parse::<usize>() else {
        return;
    };
    for delta in [0isize, -1, 1, -2, 2] {
        let ln = (beg_line as isize + delta).max(1) as usize;
        aeneas_by_key
            .entry(format!("span::{src_file}:{ln}::{short_func}"))
            .or_insert_with(|| rec.clone());
        aeneas_by_key
            .entry(format!("span::rust/{src_file}:{ln}::{short_func}"))
            .or_insert_with(|| rec.clone());
    }
}

fn index_span(
    aeneas_by_key: &mut BTreeMap<String, AeneasRecord>,
    rec: &AeneasRecord,
    short_func: &str,
    has_impl: bool,
) {
    let Some(q1) = rec.span.find('\'') else {
        return;
    };
    let after_q1 = &rec.span[q1 + 1..];
    let Some(q2) = after_q1.find('\'') else {
        return;
    };
    let src_file = &after_q1[..q2];
    let after_q2 = &after_q1[q2 + 1..];
    if !has_impl {
        if let Some(stem) = Path::new(src_file).file_stem().and_then(|s| s.to_str()) {
            aeneas_by_key
                .entry(format!("free::{stem}::{short_func}"))
                .or_insert_with(|| rec.clone());
        }
    }
    index_lines(aeneas_by_key, rec, src_file, after_q2, short_func);
}

fn index_record(aeneas_by_key: &mut BTreeMap<String, AeneasRecord>, rec: &AeneasRecord) {
    if external_item(rec) {
        return;
    }
    let norm = normalize_key(&rec.rust_item);
    let segs: Vec<&str> = norm.split("::").filter(|s| !s.is_empty()).collect();
    let Some(&last) = segs.last() else {
        return;
    };
    let short_func = last.trim_end_matches('}');
    let has_impl_block = rec.rust_item.contains('{');

    if let Some(impl_ty) = impl_type(&rec.rust_item) {
        aeneas_by_key
            .entry(format!("impl::{impl_ty}::{short_func}"))
            .or_insert_with(|| rec.clone());
    } else if !has_impl_block && segs.len() >= 2 {
        let prev = segs[segs.len() - 2].trim_matches(|c| c == '{' || c == '}');
        aeneas_by_key
            .entry(format!("free::{prev}::{short_func}"))
            .or_insert_with(|| rec.clone());
    }
    index_span(aeneas_by_key, rec, short_func, has_impl_block);
}

fn body_boundary(line: &str, extracted_mode: bool) -> bool {
    if extracted_mode {
        line.starts_with("/--") || line.starts_with("end ")
    } else {
        line.starts_with("/-- [") || line.starts_with("/- [")
    }
}

fn body_snippet(lines: &[&str], start: usize, extracted_mode: bool) -> (String, String) {
    let mut j = start;
    while j < lines.len() && lines[j].trim().is_empty() {
        j += 1;
    }
    let mut body_lines = Vec::new();
    let mut lean_name = String::new();
    while j < lines.len() && !body_boundary(lines[j], extracted_mode) && body_lines.len() < 32 {
        if lean_name.is_empty() {
            if let Some((_, nm)) = parse_decl(lines[j]) {
                lean_name = nm;
            }
        }
        body_lines.push(lines[j]);
        j += 1;
    }
    (lean_name, body_lines.join("\n").trim().to_owned())
}

fn bracket_item(
    lines: &[&str],
    idx: &mut usize,
    after: &str,
    rel_path: &str,
) -> Option<AeneasRecord> {
    let mut comment_lines = vec![lines[*idx]];
    if !lines[*idx].contains("-/") {
        *idx += 1;
        while *idx < lines.len() {
            comment_lines.push(lines[*idx]);
            if lines[*idx].contains("-/") {
                break;
            }
            *idx += 1;
        }
    }
    let comment = comment_lines.join("\n");
    let rust_item = after.split(']').next().unwrap_or("").trim().to_owned();
    let span = comment
        .lines()
        .find_map(|l| {
            l.trim()
                .strip_prefix("Source:")
                .map(|s| s.trim().to_owned())
        })
        .unwrap_or_default();
    let (lean_name, code) = body_snippet(lines, *idx + 1, false);
    if rust_item.is_empty() || code.is_empty() {
        return None;
    }
    Some(AeneasRecord {
        lean_name,
        file: rel_path.to_owned(),
        rust_item,
        span,
        code,
    })
}

fn extracted_item(lines: &[&str], idx: usize, rel_path: &str) -> Option<AeneasRecord> {
    let b1 = lines[idx].find("Extracted `")?;
    let after_b1 = &lines[idx][b1 + "Extracted `".len()..];
    let b2 = after_b1.find('`')?;
    let rust_item = after_b1[..b2].trim().to_owned();
    let span = after_b1[b2 + 1..]
        .trim_start_matches(',')
        .trim_end_matches("-/")
        .trim()
        .to_owned();
    let (lean_name, code) = body_snippet(lines, idx + 1, true);
    if rust_item.is_empty() || code.is_empty() {
        return None;
    }
    Some(AeneasRecord {
        lean_name,
        file: rel_path.to_owned(),
        rust_item,
        span,
        code,
    })
}

fn scan_file(root: &Path, path: &Path, aeneas_by_key: &mut BTreeMap<String, AeneasRecord>) {
    let rel_path = path
        .strip_prefix(root)
        .map(|p| p.to_string_lossy().replace('\\', "/"))
        .unwrap_or_default();
    let Ok(text) = fs::read_to_string(path) else {
        return;
    };
    let lines: Vec<&str> = text.lines().collect();
    let mut i = 0;
    while i < lines.len() {
        let rec_opt = if let Some(after) = lines[i]
            .strip_prefix("/-- [")
            .or_else(|| lines[i].strip_prefix("/- ["))
        {
            bracket_item(&lines, &mut i, after, &rel_path)
        } else if lines[i].contains("Extracted `") {
            extracted_item(&lines, i, &rel_path)
        } else {
            None
        };
        if let Some(rec) = rec_opt {
            index_record(aeneas_by_key, &rec);
        }
        i += 1;
    }
}

pub(crate) fn extract_aeneas(root: &Path) -> BTreeMap<String, AeneasRecord> {
    let gen_dir = root.join("research/lean/BauxResearch/Generated");
    let ext_dir = root.join("research/lean/BauxResearch/Extraction");
    let mut files = Vec::new();
    let _ = lean_files(&gen_dir, &mut files);
    let _ = lean_files(&ext_dir, &mut files);
    files.sort();
    let mut aeneas_by_key: BTreeMap<String, AeneasRecord> = BTreeMap::new();
    for path in files {
        scan_file(root, &path, &mut aeneas_by_key);
    }
    aeneas_by_key
}
