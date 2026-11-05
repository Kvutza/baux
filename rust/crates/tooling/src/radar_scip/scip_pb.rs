use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::SystemTime;

pub(crate) const DEFINITION: i32 = 1;

#[derive(Default)]
pub(crate) struct ScipIndex {
    pub(crate) documents: Vec<Document>,
}

#[derive(Default)]
pub(crate) struct Document {
    pub(crate) path: String,
    pub(crate) occurrences: Vec<Occurrence>,
    pub(crate) symbols: Vec<SymbolInfo>,
}

#[derive(Clone, Default, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct Occurrence {
    pub(crate) range: Vec<i32>,
    pub(crate) symbol: String,
    pub(crate) roles: i32,
}

#[derive(Clone, Default)]
pub(crate) struct SymbolInfo {
    pub(crate) symbol: String,
    pub(crate) display: String,
    pub(crate) signature: Option<String>,
    pub(crate) implements: Vec<String>,
}

pub(crate) fn canonical_symbol(symbol: String) -> String {
    let Some(rest) = symbol.strip_prefix("rust-analyzer cargo ") else {
        return symbol;
    };
    let Some((pkg, after_pkg)) = rest.split_once(' ') else {
        return symbol;
    };
    if !matches!(pkg, "baux-cuda" | "baux-cuda-device") {
        return symbol;
    }
    let Some((ver, descriptor)) = after_pkg.split_once(' ') else {
        return symbol;
    };
    if descriptor.starts_with("semantics/")
        || descriptor.starts_with("artifact/")
        || descriptor.starts_with("neighbor-index/")
    {
        format!("rust-analyzer cargo baux {ver} {descriptor}")
    } else {
        symbol
    }
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
    let canon_root = fs::canonicalize(root).unwrap_or_else(|_| root.to_path_buf());
    if target != root && target != canon_root {
        if let Ok(rel) = target.strip_prefix(&canon_root) {
            let root_path = cache_path(root, &canon_root);
            if root_path.is_file() && fresh(target, &root_path)? {
                return slice_index(&root_path, target, rel);
            }
        }
    }
    let path = cache_path(root, target);
    if !path.is_file() {
        println!(
            "o|radar: SCIP index missing for {}; building automatically...",
            target.display()
        );
        index(root, Some(target), true)?;
    } else if !fresh(target, &path)? {
        if std::env::var_os("BAUX_RADAR_REINDEX").is_some() {
            println!(
                "o|radar: SCIP index stale for {}; rebuilding automatically...",
                target.display()
            );
            index(root, Some(target), true)?;
        } else {
            eprintln!(
                "warning: SCIP index {} is older than modified source files \
                 (run `./baux radar index --force` or set BAUX_RADAR_REINDEX=1 to refresh)",
                path.display()
            );
        }
    }
    load_index(&path)
}

fn slice_index(root_path: &Path, target: &Path, rel: &Path) -> Result<ScipIndex, String> {
    let mut index = load_index(root_path)?;
    let rel_str = rel.to_string_lossy();
    if target.is_file() {
        let name = target
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or(&rel_str)
            .to_owned();
        index.documents.retain_mut(|doc| {
            let keep = doc.path == rel_str;
            if keep {
                doc.path = name.clone();
            }
            keep
        });
    } else {
        let prefix = format!("{rel_str}/");
        index.documents.retain_mut(|doc| {
            if let Some(rest) = doc.path.strip_prefix(&prefix) {
                doc.path = rest.to_owned();
                true
            } else {
                false
            }
        });
    }
    Ok(index)
}

pub(super) fn fresh(root: &Path, index: &Path) -> Result<bool, String> {
    let Ok(index_time) = fs::metadata(index).and_then(|metadata| metadata.modified()) else {
        return Ok(false);
    };
    let candidates = [root.join("Cargo.toml"), root.join("src"), root.join("rust")];
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

pub(crate) fn load_index(path: &Path) -> Result<ScipIndex, String> {
    let bytes = fs::read(path).map_err(|error| format!("read {}: {error}", path.display()))?;
    parse_index(&bytes).map_err(|error| format!("decode {}: {error}", path.display()))
}

pub(super) fn parse_index(bytes: &[u8]) -> Result<ScipIndex, String> {
    let mut by_path = std::collections::BTreeMap::<String, Document>::new();
    let mut order = Vec::<String>::new();
    let mut proto = Proto::new(bytes);
    while let Some((field, wire)) = proto.next()? {
        if let (2, Wire::Bytes(bytes)) = (field, wire) {
            let doc = parse_doc(bytes)?;
            if let Some(existing) = by_path.get_mut(&doc.path) {
                existing.occurrences.extend(doc.occurrences);
                existing.symbols.extend(doc.symbols);
            } else {
                order.push(doc.path.clone());
                by_path.insert(doc.path.clone(), doc);
            }
        }
    }
    let mut documents = Vec::with_capacity(order.len());
    for path in order {
        if let Some(mut doc) = by_path.remove(&path) {
            doc.occurrences.sort();
            doc.occurrences.dedup();
            let mut merged_syms = std::collections::BTreeMap::<String, SymbolInfo>::new();
            for sym in doc.symbols {
                merged_syms
                    .entry(sym.symbol.clone())
                    .and_modify(|existing| {
                        if existing.display.is_empty() && !sym.display.is_empty() {
                            existing.display = sym.display.clone();
                        }
                        if existing.signature.is_none() && sym.signature.is_some() {
                            existing.signature = sym.signature.clone();
                        }
                        for imp in &sym.implements {
                            if !existing.implements.contains(imp) {
                                existing.implements.push(imp.clone());
                            }
                        }
                    })
                    .or_insert(sym);
            }
            doc.symbols = merged_syms.into_values().collect();
            documents.push(doc);
        }
    }
    Ok(ScipIndex { documents })
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
            (2, Wire::Bytes(bytes)) => occurrence.symbol = canonical_symbol(parse_text(bytes)?),
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
            (1, Wire::Bytes(bytes)) => info.symbol = canonical_symbol(parse_text(bytes)?),
            (4, Wire::Bytes(bytes)) => {
                if let Some(target) = parse_relationship(bytes)? {
                    info.implements.push(target);
                }
            }
            (6, Wire::Bytes(bytes)) => info.display = parse_text(bytes)?,
            (7, Wire::Bytes(bytes)) => info.signature = parse_signature(bytes)?,
            _ => {}
        }
    }
    Ok(info)
}

fn parse_relationship(bytes: &[u8]) -> Result<Option<String>, String> {
    let mut proto = Proto::new(bytes);
    let mut symbol = None;
    let mut is_impl = false;
    while let Some((field, wire)) = proto.next()? {
        match (field, wire) {
            (1, Wire::Bytes(bytes)) => symbol = Some(canonical_symbol(parse_text(bytes)?)),
            (3, Wire::Var(value)) => is_impl = value != 0,
            _ => {}
        }
    }
    Ok(if is_impl { symbol } else { None })
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
