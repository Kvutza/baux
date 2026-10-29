use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::SystemTime;

pub(super) const DEFINITION: i32 = 1;

#[derive(Default)]
pub(super) struct ScipIndex {
    pub(super) documents: Vec<Document>,
}

#[derive(Default)]
pub(super) struct Document {
    pub(super) path: String,
    pub(super) occurrences: Vec<Occurrence>,
    pub(super) symbols: Vec<SymbolInfo>,
}

#[derive(Default)]
pub(super) struct Occurrence {
    pub(super) range: Vec<i32>,
    pub(super) symbol: String,
    pub(super) roles: i32,
}

#[derive(Default)]
pub(super) struct SymbolInfo {
    pub(super) symbol: String,
    pub(super) display: String,
    pub(super) signature: Option<String>,
}

pub(super) struct SymbolMatch<'a> {
    pub(super) document: &'a Document,
    pub(super) info: &'a SymbolInfo,
}

enum Wire<'a> {
    Var(u64),
    Bytes(&'a [u8]),
    Fixed,
}

struct Proto<'a> {
    bytes: &'a [u8],
    cursor: usize,
}

impl<'a> Proto<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, cursor: 0 }
    }

    fn next(&mut self) -> Result<Option<(u32, Wire<'a>)>, String> {
        if self.cursor == self.bytes.len() {
            return Ok(None);
        }
        let key = self.varint()?;
        let field = u32::try_from(key >> 3).map_err(|_| "SCIP field number overflow")?;
        if field == 0 {
            return Err("SCIP contains field zero".to_owned());
        }
        let wire = match key & 7 {
            0 => Wire::Var(self.varint()?),
            1 => {
                self.take(8)?;
                Wire::Fixed
            }
            2 => {
                let size = usize::try_from(self.varint()?)
                    .map_err(|_| "SCIP field length overflow".to_owned())?;
                Wire::Bytes(self.take(size)?)
            }
            5 => {
                self.take(4)?;
                Wire::Fixed
            }
            wire => return Err(format!("unsupported SCIP wire type {wire}")),
        };
        Ok(Some((field, wire)))
    }

    fn varint(&mut self) -> Result<u64, String> {
        let mut value = 0_u64;
        for shift in (0..70).step_by(7) {
            let byte = *self.bytes.get(self.cursor).ok_or("truncated SCIP varint")?;
            self.cursor += 1;
            if shift == 63 && byte > 1 {
                return Err("SCIP varint overflow".to_owned());
            }
            value |= u64::from(byte & 0x7f) << shift;
            if byte & 0x80 == 0 {
                return Ok(value);
            }
        }
        Err("SCIP varint overflow".to_owned())
    }

    fn take(&mut self, size: usize) -> Result<&'a [u8], String> {
        let end = self
            .cursor
            .checked_add(size)
            .ok_or("SCIP field length overflow")?;
        let bytes = self
            .bytes
            .get(self.cursor..end)
            .ok_or("truncated SCIP field")?;
        self.cursor = end;
        Ok(bytes)
    }
}

pub(super) fn resolve_target(root: &Path, target: Option<&Path>) -> Result<PathBuf, String> {
    let raw = match target {
        Some(path) if path.is_absolute() => path.to_path_buf(),
        Some(path) => root.join(path),
        None => return Ok(root.to_path_buf()),
    };
    fs::canonicalize(&raw).map_err(|error| format!("resolve target {}: {error}", raw.display()))
}

pub(super) fn cache_path(root: &Path, target: &Path) -> PathBuf {
    let canon_root = fs::canonicalize(root).unwrap_or_else(|_| root.to_path_buf());
    if target == root || target == canon_root {
        return root.join(".cache/baux/radar/rust.scip");
    }
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    for byte in target.to_string_lossy().bytes() {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x0100_0000_01b3);
    }
    let stem = target
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("external");
    root.join(format!(".cache/baux/radar/{stem}-{hash:016x}.scip"))
}

pub(super) fn require_index(root: &Path, target: &Path) -> Result<ScipIndex, String> {
    let path = cache_path(root, target);
    if !path.is_file() || !fresh(target, &path)? {
        println!(
            "o|radar: SCIP index missing or stale for {}; rebuilding automatically...",
            target.display()
        );
        super::index(root, Some(target), true)?;
    }
    load_index(&path)
}

pub(super) fn fresh(root: &Path, index: &Path) -> Result<bool, String> {
    let Ok(index_time) = fs::metadata(index).and_then(|metadata| metadata.modified()) else {
        return Ok(false);
    };
    let candidates = [
        root.join("Cargo.toml"),
        root.join("src"),
        root.join("rust"),
        root.join("backends/cuda"),
    ];
    let mut checked = false;
    for path in candidates {
        if path.exists() {
            checked = true;
            if newer(&path, index_time)? {
                return Ok(false);
            }
        }
    }
    if !checked && newer(root, index_time)? {
        return Ok(false);
    }
    Ok(true)
}

fn newer(path: &Path, cutoff: SystemTime) -> Result<bool, String> {
    let metadata =
        fs::metadata(path).map_err(|error| format!("inspect {}: {error}", path.display()))?;
    if metadata.is_file() {
        let relevant = path.file_name().is_some_and(|name| name == "Cargo.toml")
            || path.extension().is_some_and(|extension| extension == "rs");
        return Ok(relevant && metadata.modified().is_ok_and(|time| time > cutoff));
    }
    for entry in fs::read_dir(path).map_err(|error| format!("read {}: {error}", path.display()))? {
        let entry = entry.map_err(|error| format!("read {}: {error}", path.display()))?;
        let child = entry.path();
        if child.file_name().is_some_and(|name| {
            matches!(
                name.to_str(),
                Some(".cache" | ".cargo" | ".git" | ".jj" | "buck-out" | "target" | "vendor")
            )
        }) {
            continue;
        }
        if newer(&child, cutoff)? {
            return Ok(true);
        }
    }
    Ok(false)
}

pub(super) fn load_index(path: &Path) -> Result<ScipIndex, String> {
    let bytes = fs::read(path).map_err(|error| format!("read {}: {error}", path.display()))?;
    parse_index(&bytes).map_err(|error| format!("decode {}: {error}", path.display()))
}

fn parse_index(bytes: &[u8]) -> Result<ScipIndex, String> {
    let mut index = ScipIndex::default();
    let mut proto = Proto::new(bytes);
    while let Some((field, wire)) = proto.next()? {
        if let (2, Wire::Bytes(bytes)) = (field, wire) {
            index.documents.push(parse_doc(bytes)?);
        }
    }
    Ok(index)
}

fn parse_doc(bytes: &[u8]) -> Result<Document, String> {
    let mut document = Document::default();
    let mut proto = Proto::new(bytes);
    while let Some((field, wire)) = proto.next()? {
        match (field, wire) {
            (1, Wire::Bytes(bytes)) => document.path = parse_text(bytes)?,
            (2, Wire::Bytes(bytes)) => document.occurrences.push(parse_occurrence(bytes)?),
            (3, Wire::Bytes(bytes)) => document.symbols.push(parse_info(bytes)?),
            _ => {}
        }
    }
    Ok(document)
}

fn parse_occurrence(bytes: &[u8]) -> Result<Occurrence, String> {
    let mut occurrence = Occurrence::default();
    let mut proto = Proto::new(bytes);
    while let Some((field, wire)) = proto.next()? {
        match (field, wire) {
            (1, Wire::Bytes(bytes)) => {
                let mut packed = Proto::new(bytes);
                while packed.cursor < packed.bytes.len() {
                    occurrence.range.push(packed.varint()? as i32);
                }
            }
            (1, Wire::Var(value)) => occurrence.range.push(value as i32),
            (2, Wire::Bytes(bytes)) => occurrence.symbol = parse_text(bytes)?,
            (3, Wire::Var(value)) => occurrence.roles = value as i32,
            _ => {}
        }
    }
    Ok(occurrence)
}

fn parse_info(bytes: &[u8]) -> Result<SymbolInfo, String> {
    let mut info = SymbolInfo::default();
    let mut proto = Proto::new(bytes);
    while let Some((field, wire)) = proto.next()? {
        match (field, wire) {
            (1, Wire::Bytes(bytes)) => info.symbol = parse_text(bytes)?,
            (6, Wire::Bytes(bytes)) => info.display = parse_text(bytes)?,
            (7, Wire::Bytes(bytes)) => info.signature = parse_signature(bytes)?,
            _ => {}
        }
    }
    Ok(info)
}

fn parse_signature(bytes: &[u8]) -> Result<Option<String>, String> {
    let mut proto = Proto::new(bytes);
    while let Some((field, wire)) = proto.next()? {
        if let (5, Wire::Bytes(bytes)) = (field, wire) {
            return parse_text(bytes).map(Some);
        }
    }
    Ok(None)
}

fn parse_text(bytes: &[u8]) -> Result<String, String> {
    std::str::from_utf8(bytes)
        .map(str::to_owned)
        .map_err(|error| format!("SCIP contains invalid UTF-8: {error}"))
}

pub(super) fn print_stats(path: &Path, index: &ScipIndex) {
    let definitions = index
        .documents
        .iter()
        .flat_map(|document| &document.occurrences)
        .filter(|occurrence| occurrence.roles & DEFINITION != 0)
        .count();
    let occurrences = index
        .documents
        .iter()
        .map(|document| document.occurrences.len())
        .sum::<usize>();
    println!(
        "SCIP {} documents={} definitions={} occurrences={}",
        path.display(),
        index.documents.len(),
        definitions,
        occurrences
    );
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

/// Rank case-insensitive exact names ahead of substring and canonical-symbol
/// matches.
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

pub(super) fn local_symbol(symbol: &str) -> bool {
    symbol.starts_with("local ")
}

pub(super) fn format_hit(path: &str, occurrence: &Occurrence) -> String {
    let (line, column) = match occurrence.range.as_slice() {
        [line, column, _] | [line, column, _, _] => (*line + 1, *column + 1),
        _ => (0, 0),
    };
    format!("{path}:{line}:{column}")
}

pub(super) fn resolve_analyzer(root: &Path) -> PathBuf {
    if let Ok(output) = Command::new("rustup")
        .current_dir(root)
        .args(["which", "rust-analyzer"])
        .output()
    {
        if output.status.success() {
            let text = String::from_utf8_lossy(&output.stdout).trim().to_owned();
            if !text.is_empty() {
                let candidate = PathBuf::from(text);
                if rust_scip(&candidate) {
                    return candidate;
                }
            }
        }
    }
    PathBuf::from("rust-analyzer")
}

pub(super) fn rust_scip(binary: &Path) -> bool {
    Command::new(binary)
        .arg("--help")
        .output()
        .is_ok_and(|output| {
            output.status.success() && String::from_utf8_lossy(&output.stdout).contains("scip")
        })
}

pub(super) fn has_command(program: &str, argument: &str) -> bool {
    Command::new(program)
        .arg(argument)
        .output()
        .is_ok_and(|output| output.status.success())
}

pub(super) fn metal_ready() -> bool {
    Command::new("xcrun")
        .args(["-sdk", "macosx", "--find", "metal"])
        .output()
        .is_ok_and(|output| output.status.success())
}

pub(super) fn report(name: &str, ready: bool, detail: &str) {
    let state = if ready { "ready" } else { "absent" };
    println!("{name:<12} {state:<6} {detail}");
}

pub(super) fn index(root: &Path, target: Option<&Path>, force: bool) -> Result<(), String> {
    use std::io::Write;
    let dir = resolve_target(root, target)?;
    let path = cache_path(root, &dir);
    if !force && fresh(&dir, &path)? {
        let value = load_index(&path)?;
        print_stats(&path, &value);
        return Ok(());
    }
    let parent = path.parent().ok_or("SCIP index has no parent directory")?;
    fs::create_dir_all(parent).map_err(|error| format!("create SCIP cache: {error}"))?;
    let temporary = path.with_extension("scip.tmp");
    let analyzer = resolve_analyzer(root);
    println!(
        "o|radar: indexing {} with `{} scip .` -> {}",
        dir.display(),
        analyzer.display(),
        path.display()
    );
    let _ = std::io::stdout().flush();
    let mut command = Command::new(&analyzer);
    command
        .current_dir(&dir)
        .args(["scip", ".", "--output"])
        .arg(&temporary);
    crate::execute(command)?;
    let value = load_index(&temporary)?;
    fs::rename(&temporary, &path).map_err(|error| format!("publish SCIP index: {error}"))?;
    print_stats(&path, &value);
    Ok(())
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
    report("metal", metal_ready(), "Apple Metal compiler");
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

#[cfg(test)]
mod tests {
    use super::*;

    fn varint(mut value: u64) -> Vec<u8> {
        let mut bytes = Vec::new();
        loop {
            let mut byte = (value & 0x7f) as u8;
            value >>= 7;
            if value != 0 {
                byte |= 0x80;
            }
            bytes.push(byte);
            if value == 0 {
                return bytes;
            }
        }
    }

    fn bytes(field: u8, value: &[u8]) -> Vec<u8> {
        let mut output = varint(u64::from(field) << 3 | 2);
        output.extend(varint(value.len() as u64));
        output.extend(value);
        output
    }

    #[test]
    fn location() {
        let occurrence = Occurrence {
            range: vec![4, 2, 9],
            ..Occurrence::default()
        };
        assert_eq!(format_hit("source.rs", &occurrence), "source.rs:5:3");
    }

    #[test]
    fn decode_index() {
        let mut occurrence = bytes(1, &[4, 2, 9]);
        occurrence.extend(bytes(2, b"rust test symbol"));
        occurrence.extend([24, 1]);

        let signature = bytes(5, b"fn test()");
        let mut info = bytes(1, b"rust test symbol");
        info.extend(bytes(6, b"test"));
        info.extend(bytes(7, &signature));

        let mut document = bytes(1, b"source.rs");
        document.extend(bytes(2, &occurrence));
        document.extend(bytes(3, &info));
        let mut index = parse_index(&bytes(2, &document)).expect("valid synthetic SCIP");

        assert_eq!(index.documents[0].path, "source.rs");
        assert_eq!(index.documents[0].occurrences[0].roles, DEFINITION);
        assert_eq!(index.documents[0].symbols[0].display, "test");
        assert_eq!(
            index.documents[0].symbols[0].signature.as_deref(),
            Some("fn test()")
        );
        for query in [
            "TEST",
            "rust test symbol",
            "tes",
            "rust test",
            "source.rs:test",
        ] {
            let selected = select_symbols(&index, query);
            assert_eq!(selected.len(), 1, "{query}");
            assert_eq!(selected[0].info.symbol, "rust test symbol");
        }
        assert!(select_symbols(&index, "other.rs:test").is_empty());
        assert!(select_symbols(&index, "absent").is_empty());
        index.documents[0].occurrences[0].roles = 0;
        assert!(select_symbols(&index, "test").is_empty());
    }
}
