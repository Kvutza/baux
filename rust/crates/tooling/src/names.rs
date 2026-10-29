use std::path::{Path, PathBuf};

const SKIP_PARTS: &[&str] = &[
    ".cache",
    ".git",
    ".home",
    ".jj",
    ".kiss",
    ".lake",
    ".malvin",
    ".pixi",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
    "archive",
    "buck-out",
    "cuda",
    "dist",
    "docs",
    "examples",
    "ptx-synth",
    "target",
    "vendor",
];

const FILE_SUFFIXES: &[&str] = &[
    "bzl", "cl", "ipynb", "json", "md", "metal", "py", "rs", "toml", "yaml", "yml",
];

const EXACT_FILES: &[&str] = &["BUCK", "Cargo.toml", "MODULE.bazel", "README.md"];

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Finding {
    pub path: String,
    pub line: usize,
    pub language: String,
    pub kind: String,
    pub name: String,
    pub underscores: usize,
    pub reason: String,
}

#[derive(Debug, Clone)]
pub struct Config {
    pub max_underscores: usize,
    pub include_public: bool,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            max_underscores: 1,
            include_public: true,
        }
    }
}

fn is_notation(token: &str) -> bool {
    let lower = token.to_ascii_lowercase();
    if lower == "bf16"
        || lower == "splitmix64"
        || lower == "mix64"
        || lower == "mix32"
        || lower == "dist2"
        || lower == "l2"
    {
        return true;
    }
    if lower.ends_with("bf16") || lower.ends_with("dist2") || lower.ends_with("l2") {
        return true;
    }
    let bytes = lower.as_bytes();
    let mut i = bytes.len();
    while i > 0 && bytes[i - 1].is_ascii_digit() {
        i -= 1;
    }
    if i < bytes.len() && i > 0 {
        let ch = bytes[i - 1];
        if ch == b'f' || ch == b'i' || ch == b'u' {
            return true;
        }
    }
    if lower.starts_with("fp") && lower.len() >= 3 && lower[2..].chars().all(|c| c.is_ascii_digit())
    {
        return true;
    }
    if lower.starts_with("int")
        && lower.len() >= 4
        && lower[3..].chars().all(|c| c.is_ascii_digit())
    {
        return true;
    }
    if lower.starts_with("uint")
        && lower.len() >= 5
        && lower[4..].chars().all(|c| c.is_ascii_digit())
    {
        return true;
    }
    if lower.starts_with('e') {
        if let Some(m_pos) = lower.find('m') {
            if m_pos > 1
                && lower[1..m_pos].chars().all(|c| c.is_ascii_digit())
                && lower[m_pos + 1..].chars().all(|c| c.is_ascii_digit())
                && !lower[m_pos + 1..].is_empty()
            {
                return true;
            }
        }
    }
    false
}

fn is_overload(name: &str) -> bool {
    let Some(last) = name.chars().last() else {
        return false;
    };
    if !last.is_ascii_digit() {
        return false;
    }
    if is_notation(name) {
        return false;
    }
    if let Some(pos) = name.rfind('_') {
        let suffix = &name[pos + 1..];
        if is_notation(suffix) {
            return false;
        }
    }
    true
}

fn bad_name(name: &str, max_underscores: usize, check_number: bool) -> bool {
    if name.starts_with("__") && name.ends_with("__") {
        return false;
    }
    let count = name.chars().filter(|&c| c == '_').count();
    if count > max_underscores || name.len() > 24 {
        return true;
    }
    if check_number && is_overload(name) {
        return true;
    }
    false
}

fn make_reason(name: &str, kind: &str, max_underscores: usize, check_number: bool) -> String {
    let count = name.chars().filter(|&c| c == '_').count();
    let mut parts = Vec::new();
    if count > max_underscores {
        parts.push(format!("more than {max_underscores} underscore"));
    }
    if name.len() > 24 {
        parts.push("more than 24 characters".to_string());
    }
    if check_number && is_overload(name) {
        parts.push("trailing overload number".to_string());
    }
    format!("local {kind} name has {}", parts.join(" and "))
}

fn check_stem(path: &Path, root: &Path, config: &Config, findings: &mut Vec<Finding>) {
    let Some(stem) = path.file_stem().and_then(|s| s.to_str()) else {
        return;
    };
    if stem == "__init__" || path.extension().map_or(false, |e| e == "metal") {
        return;
    }
    if bad_name(stem, config.max_underscores, false) {
        let rel = path.strip_prefix(root).unwrap_or(path);
        findings.push(Finding {
            path: rel.to_string_lossy().to_string(),
            line: 1,
            language: "file".to_string(),
            kind: "file".to_string(),
            name: stem.to_string(),
            underscores: stem.chars().filter(|&c| c == '_').count(),
            reason: make_reason(stem, "file", config.max_underscores, false),
        });
    }
}

fn is_trait(line: &str) -> bool {
    if !line.starts_with("impl") {
        return false;
    }
    for token in line.split_whitespace() {
        if token == "for" {
            return true;
        }
    }
    false
}

fn parse_def(mut line: &str) -> Option<(&'static str, String, bool, bool)> {
    let mut is_pub = false;
    let mut is_ext = false;

    if line.starts_with("pub(") {
        if let Some(idx) = line.find(')') {
            is_pub = true;
            is_ext = false;
            line = line[idx + 1..].trim_start();
        }
    } else if let Some(rest) = line.strip_prefix("pub ") {
        is_pub = true;
        is_ext = true;
        line = rest.trim_start();
    }

    while let Some(rest) = qualifier(line) {
        line = rest;
    }

    let kinds = [
        ("fn", "fn"),
        ("struct", "struct"),
        ("enum", "enum"),
        ("trait", "trait"),
        ("type", "type"),
        ("const", "const"),
        ("static", "static"),
        ("mod", "mod"),
        ("macro_rules!", "macro_rules!"),
    ];

    for &(prefix, kind) in &kinds {
        if let Some(rest) = line.strip_prefix(prefix) {
            let mut rest = rest.trim_start();
            if kind == "macro_rules!" && rest.starts_with('!') {
                rest = rest[1..].trim_start();
            }
            let len = rest
                .chars()
                .take_while(|c| c.is_ascii_alphanumeric() || *c == '_')
                .count();
            if len > 0 {
                let name = &rest[..len];
                return Some((kind, name.to_string(), is_pub, is_ext));
            }
        }
    }

    None
}

/// Consume one Rust declaration qualifier, preserving constants and malformed
/// ABIs.
fn qualifier(line: &str) -> Option<&str> {
    if let Some(rest) = line
        .strip_prefix("async ")
        .or_else(|| line.strip_prefix("unsafe "))
    {
        return Some(rest.trim_start());
    }
    if let Some(rest) = line.strip_prefix("const ") {
        return rest
            .trim_start()
            .starts_with("fn ")
            .then(|| rest.trim_start());
    }
    let rest = line.strip_prefix("extern ")?.trim_start();
    match rest.strip_prefix('"') {
        Some(abi) => Some(abi[abi.find('"')? + 1..].trim_start()),
        None => Some(rest),
    }
}

fn check_rust(path: &Path, root: &Path, config: &Config, findings: &mut Vec<Finding>) {
    let Ok(content) = std::fs::read_to_string(path) else {
        return;
    };
    let mut depth: isize = 0;
    let mut traits: Vec<isize> = Vec::new();
    let rel = path
        .strip_prefix(root)
        .unwrap_or(path)
        .to_string_lossy()
        .to_string();

    for (idx, raw) in content.lines().enumerate() {
        let line_num = idx + 1;
        let trimmed = raw.trim_start();

        if is_trait(trimmed) {
            let open = raw.chars().filter(|&c| c == '{').count() as isize;
            let close = raw.chars().filter(|&c| c == '}').count() as isize;
            let next_depth = depth + open - close;
            if next_depth > depth {
                traits.push(next_depth);
            }
        }

        let in_trait = traits.iter().any(|&d| depth >= d);

        if let Some((kind, name, is_pub, is_ext)) = parse_def(trimmed) {
            if kind != "mod" && !(in_trait && kind == "fn") {
                if (config.include_public || !is_pub)
                    && bad_name(&name, config.max_underscores, is_ext)
                {
                    findings.push(Finding {
                        path: rel.clone(),
                        line: line_num,
                        language: "rust".to_string(),
                        kind: kind.to_string(),
                        name: name.clone(),
                        underscores: name.chars().filter(|&c| c == '_').count(),
                        reason: make_reason(&name, kind, config.max_underscores, is_ext),
                    });
                }
            }
        }

        let open = raw.chars().filter(|&c| c == '{').count() as isize;
        let close = raw.chars().filter(|&c| c == '}').count() as isize;
        depth += open - close;
        traits.retain(|&d| depth >= d);
    }
}

fn check_python(path: &Path, root: &Path, config: &Config, findings: &mut Vec<Finding>) {
    let Ok(content) = std::fs::read_to_string(path) else {
        return;
    };
    let rel = path.strip_prefix(root).unwrap_or(path);
    let rel_str = rel.to_string_lossy().to_string();
    let is_src = rel
        .components()
        .next()
        .map_or(false, |c| c.as_os_str() == "src");

    let mut prev_override = false;
    let mut in_quote = false;

    for (idx, raw) in content.lines().enumerate() {
        let line_num = idx + 1;
        let trimmed = raw.trim();

        let count_triples = raw.matches("\"\"\"").count() + raw.matches("'''").count();
        if in_quote {
            if count_triples % 2 != 0 {
                in_quote = false;
            }
            continue;
        } else if count_triples % 2 != 0 {
            in_quote = true;
            continue;
        }

        if trimmed == "@override" || trimmed.ends_with(".override") {
            prev_override = true;
            continue;
        }

        let mut is_def = false;
        let mut kind = "fn";
        let mut rest = "";

        if let Some(r) = trimmed.strip_prefix("def ") {
            is_def = true;
            rest = r;
        } else if let Some(r) = trimmed.strip_prefix("async def ") {
            is_def = true;
            rest = r;
        } else if let Some(r) = trimmed.strip_prefix("class ") {
            is_def = true;
            kind = "class";
            rest = r;
        }

        if is_def {
            if prev_override {
                prev_override = false;
                continue;
            }
            let len = rest
                .chars()
                .take_while(|c| c.is_ascii_alphanumeric() || *c == '_')
                .count();
            if len > 0 {
                let name = &rest[..len];
                let is_pub = !name.starts_with('_');
                let check_num = is_pub && is_src;
                if (config.include_public || !is_pub)
                    && bad_name(name, config.max_underscores, check_num)
                {
                    findings.push(Finding {
                        path: rel_str.clone(),
                        line: line_num,
                        language: "python".to_string(),
                        kind: kind.to_string(),
                        name: name.to_string(),
                        underscores: name.chars().filter(|&c| c == '_').count(),
                        reason: make_reason(name, kind, config.max_underscores, check_num),
                    });
                }
            }
        }

        if !trimmed.is_empty() && !trimmed.starts_with('@') {
            prev_override = false;
        }
    }
}

fn walk_path(path: &Path, root: &Path, config: &Config, findings: &mut Vec<Finding>) {
    let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
    if SKIP_PARTS.contains(&name) {
        return;
    }
    if path.is_dir() {
        let Ok(entries) = std::fs::read_dir(path) else {
            return;
        };
        for entry in entries.flatten() {
            walk_path(&entry.path(), root, config, findings);
        }
    } else if path.is_file() {
        let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
        let is_target = FILE_SUFFIXES.contains(&ext) || EXACT_FILES.contains(&name);
        if is_target {
            check_stem(path, root, config, findings);
        }
        if ext == "rs" {
            check_rust(path, root, config, findings);
        } else if ext == "py" {
            check_python(path, root, config, findings);
        }
    }
}

pub fn scan(root: &Path, paths: &[PathBuf], config: &Config) -> Vec<Finding> {
    let mut findings = Vec::new();
    if paths.is_empty() {
        walk_path(root, root, config, &mut findings);
    } else {
        for path in paths {
            let full = if path.is_absolute() {
                path.clone()
            } else {
                root.join(path)
            };
            walk_path(&full, root, config, &mut findings);
        }
    }
    findings.sort_by(|a, b| (&a.path, a.line, &a.name).cmp(&(&b.path, b.line, &b.name)));
    findings
}

pub fn cli(root: &Path, paths: &[PathBuf], json: bool) -> Result<(), String> {
    let start = std::time::Instant::now();
    let findings = scan(root, paths, &Config::default());
    let elapsed = start.elapsed();

    if json {
        let mut out = String::from("[\n");
        for (idx, item) in findings.iter().enumerate() {
            let comma = if idx + 1 < findings.len() { "," } else { "" };
            out.push_str(&format!(
                "  {{\"path\": \"{}\", \"line\": {}, \"language\": \"{}\", \"kind\": \"{}\", \
                 \"name\": \"{}\", \"underscores\": {}, \"reason\": \"{}\"}}{}\n",
                item.path,
                item.line,
                item.language,
                item.kind,
                item.name,
                item.underscores,
                item.reason,
                comma
            ));
        }
        out.push(']');
        println!("{out}");
    } else {
        for item in &findings {
            println!(
                "{}:{}: {} {} {} ({} underscores): {}",
                item.path,
                item.line,
                item.language,
                item.kind,
                item.name,
                item.underscores,
                item.reason
            );
        }
    }

    if findings.is_empty() {
        eprintln!("names: ok ({} ms)", elapsed.as_millis());
        Ok(())
    } else {
        Err(format!(
            "found {} naming violations in {} ms",
            findings.len(),
            elapsed.as_millis()
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_notations() {
        assert!(is_notation("bf16"));
        assert!(is_notation("f32"));
        assert!(is_notation("u64"));
        assert!(is_notation("dist2"));
        assert!(is_notation("l2"));
        assert!(is_notation("fp16"));
        assert!(is_notation("int8"));
        assert!(is_notation("e4m3"));
        assert!(!is_notation("test2"));
    }

    #[test]
    fn test_names() {
        assert!(!bad_name("foo", 1, false));
        assert!(!bad_name("foo_bar", 1, false));
        assert!(bad_name("foo_bar_baz", 1, false));
        assert!(bad_name("a_very_long_name_exceeding_twenty_four", 1, false));
        assert!(bad_name("pub_test2", 1, true));
        assert!(!bad_name("pub_dist2", 1, true));
    }

    #[test]
    fn test_parser() {
        let (kind, name, is_pub, is_ext) = parse_def("pub fn compute_loss() {").unwrap();
        assert_eq!(kind, "fn");
        assert_eq!(name, "compute_loss");
        assert!(is_pub);
        assert!(is_ext);

        let (kind, name, is_pub, is_ext) =
            parse_def("pub(crate) struct OptimizerConfig {").unwrap();
        assert_eq!(kind, "struct");
        assert_eq!(name, "OptimizerConfig");
        assert!(is_pub);
        assert!(!is_ext);

        for declaration in [
            "pub async unsafe fn compute_loss() {",
            "pub const fn compute_loss() {",
            "pub unsafe extern \"C\" fn compute_loss() {",
            "pub extern fn compute_loss() {",
            "pub extern \"C\" unsafe fn compute_loss() {",
        ] {
            assert_eq!(
                parse_def(declaration),
                Some(("fn", "compute_loss".to_owned(), true, true)),
                "{declaration}",
            );
        }
        assert_eq!(
            parse_def("const WIDTH: usize = 4;"),
            Some(("const", "WIDTH".to_owned(), false, false)),
        );
        assert!(parse_def("extern \"unterminated fn hidden() {").is_none());
    }

    #[test]
    fn test_python() {
        let mut findings = Vec::new();
        let content = "\
from typing import override

class Sampler:
    @override
    def infer_relative_search_space(self):
        pass

    def locally_named_long_method(self):
        pass
";
        let temp_dir = std::env::temp_dir();
        let file_path = temp_dir.join("test_sample.py");
        std::fs::write(&file_path, content).unwrap();
        let config = Config::default();
        check_python(&file_path, &temp_dir, &config, &mut findings);
        let _ = std::fs::remove_file(&file_path);

        assert_eq!(findings.len(), 1);
        assert_eq!(findings[0].name, "locally_named_long_method");
    }
}
