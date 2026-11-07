//! Full-directory RFC-6962 binary Merkle tree (SHA-256 + SHA-512) and
//! archive/source lock verifier for `research/upstream/*.lock.toml`.

use crate::artifact::{json::Value, toml};
use sha2::{Digest, Sha256, Sha512};
use std::fs::{self, File};
use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::Command;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TreeStats {
    pub root_sha256: String,
    pub root_sha512: String,
    pub file_count: u64,
    pub total_bytes: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ArchiveStats {
    pub revision: String,
    pub tree: String,
    pub archive_sha256: String,
    pub archive_sha512: String,
    pub archive_bytes: u64,
}

pub fn hash_bytes(data: &[u8]) -> (String, String) {
    let mut s256 = Sha256::new();
    let mut s512 = Sha512::new();
    s256.update(data);
    s512.update(data);
    (
        format!("{:x}", s256.finalize()),
        format!("{:x}", s512.finalize()),
    )
}

fn raw_file(path: &Path) -> Result<([u8; 32], [u8; 64], u64), String> {
    let mut file = File::open(path).map_err(|e| format!("open {}: {e}", path.display()))?;
    let mut s256 = Sha256::new();
    let mut s512 = Sha512::new();
    let mut size = 0u64;
    let mut buf = [0u8; 65536];
    loop {
        let n = file.read(&mut buf).map_err(|e| e.to_string())?;
        if n == 0 {
            break;
        }
        s256.update(&buf[..n]);
        s512.update(&buf[..n]);
        size += n as u64;
    }
    let d256: [u8; 32] = s256.finalize().into();
    let d512: [u8; 64] = s512.finalize().into();
    Ok((d256, d512, size))
}

fn hex_str(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        use std::fmt::Write;
        let _ = write!(&mut out, "{b:02x}");
    }
    out
}

pub fn hash_file(path: &Path) -> Result<(String, String, u64), String> {
    let (d256, d512, sz) = raw_file(path)?;
    Ok((hex_str(&d256), hex_str(&d512), sz))
}

fn keep_dir(name: &str) -> bool {
    !matches!(
        name,
        ".git" | "__pycache__" | ".pytest_cache" | ".ruff_cache"
    )
}

fn keep_file(name: &str) -> bool {
    name != ".DS_Store" && !name.ends_with(".pyc")
}

fn scan_dir(
    root: &Path,
    cur: &Path,
    stack: &mut Vec<PathBuf>,
    rels: &mut Vec<String>,
) -> Result<(), String> {
    let entries = fs::read_dir(cur).map_err(|e| format!("read_dir {}: {e}", cur.display()))?;
    for item in entries {
        let entry = item.map_err(|e| e.to_string())?;
        let ft = entry.file_type().map_err(|e| e.to_string())?;
        if ft.is_symlink() {
            continue;
        }
        let path = entry.path();
        let name = entry.file_name();
        let name_str = name.to_string_lossy();
        if ft.is_dir() && keep_dir(&name_str) {
            stack.push(path);
        } else if ft.is_file() && keep_file(&name_str) {
            let rel = path.strip_prefix(root).map_err(|e| e.to_string())?;
            rels.push(rel.to_string_lossy().replace('\\', "/"));
        }
    }
    Ok(())
}

fn walk_files(root: &Path) -> Result<Vec<String>, String> {
    if !root.is_dir() {
        return Err(format!("missing upstream directory: {}", root.display()));
    }
    let mut rels = Vec::new();
    let mut stack = vec![root.to_path_buf()];
    while let Some(cur) = stack.pop() {
        scan_dir(root, &cur, &mut stack, &mut rels)?;
    }
    rels.sort();
    Ok(rels)
}

fn leaf_pair(rel: &str, d256: &[u8; 32], d512: &[u8; 64], sz: u64) -> ([u8; 32], [u8; 64]) {
    let rb = rel.as_bytes();
    let rlen = (rb.len() as u32).to_le_bytes();
    let slen = sz.to_le_bytes();
    let mut h256 = Sha256::new();
    h256.update([0x00]);
    h256.update(rlen);
    h256.update(rb);
    h256.update(slen);
    h256.update(d256);
    let mut h512 = Sha512::new();
    h512.update([0x00]);
    h512.update(rlen);
    h512.update(rb);
    h512.update(slen);
    h512.update(d512);
    (h256.finalize().into(), h512.finalize().into())
}

fn fold_sha256(leaves: &[[u8; 32]]) -> String {
    if leaves.is_empty() {
        let mut h = Sha256::new();
        h.update([0x00]);
        return format!("{:x}", h.finalize());
    }
    let mut level = leaves.to_vec();
    while level.len() > 1 {
        let mut nxt = Vec::with_capacity(level.len().div_ceil(2));
        for pair in level.chunks(2) {
            if pair.len() == 2 {
                let mut h = Sha256::new();
                h.update([0x01]);
                h.update(pair[0]);
                h.update(pair[1]);
                nxt.push(h.finalize().into());
            } else {
                nxt.push(pair[0]);
            }
        }
        level = nxt;
    }
    hex_str(&level[0])
}

fn fold_sha512(leaves: &[[u8; 64]]) -> String {
    if leaves.is_empty() {
        let mut h = Sha512::new();
        h.update([0x00]);
        return format!("{:x}", h.finalize());
    }
    let mut level = leaves.to_vec();
    while level.len() > 1 {
        let mut nxt = Vec::with_capacity(level.len().div_ceil(2));
        for pair in level.chunks(2) {
            if pair.len() == 2 {
                let mut h = Sha512::new();
                h.update([0x01]);
                h.update(pair[0]);
                h.update(pair[1]);
                nxt.push(h.finalize().into());
            } else {
                nxt.push(pair[0]);
            }
        }
        level = nxt;
    }
    hex_str(&level[0])
}

pub fn compute_tree(root: &Path) -> Result<TreeStats, String> {
    let rels = walk_files(root)?;
    let mut l256 = Vec::with_capacity(rels.len());
    let mut l512 = Vec::with_capacity(rels.len());
    let mut total_bytes = 0u64;
    for rel in &rels {
        let (d256, d512, sz) = raw_file(&root.join(rel))?;
        let (a, b) = leaf_pair(rel, &d256, &d512, sz);
        l256.push(a);
        l512.push(b);
        total_bytes += sz;
    }
    Ok(TreeStats {
        root_sha256: fold_sha256(&l256),
        root_sha512: fold_sha512(&l512),
        file_count: rels.len() as u64,
        total_bytes,
    })
}

pub fn git_archive(repo_dir: &Path, revision: &str) -> Result<ArchiveStats, String> {
    let rev_out = Command::new("git")
        .arg("-C")
        .arg(repo_dir)
        .args(["rev-parse", "HEAD", &format!("{revision}^{{tree}}")])
        .output()
        .map_err(|e| e.to_string())?;
    if !rev_out.status.success() {
        return Err(format!("git rev-parse failed in {}", repo_dir.display()));
    }
    let lines: Vec<&str> = std::str::from_utf8(&rev_out.stdout)
        .unwrap_or("")
        .lines()
        .collect();
    if lines.first().copied() != Some(revision) {
        return Err(format!("git HEAD mismatch in {}", repo_dir.display()));
    }
    let tree = lines.get(1).copied().unwrap_or("").to_owned();
    let arch = Command::new("git")
        .arg("-C")
        .arg(repo_dir)
        .args(["archive", "--format=tar", revision])
        .output()
        .map_err(|e| e.to_string())?;
    if !arch.status.success() {
        return Err(format!("git archive failed in {}", repo_dir.display()));
    }
    let (s256, s512) = hash_bytes(&arch.stdout);
    Ok(ArchiveStats {
        revision: revision.to_owned(),
        tree,
        archive_sha256: s256,
        archive_sha512: s512,
        archive_bytes: arch.stdout.len() as u64,
    })
}

fn check_merkle(target: &Path, sec: &Value) -> Result<(), String> {
    let actual = compute_tree(target)?;
    if let Some(exp) = sec["merkle_root_sha256"].as_str()
        && actual.root_sha256 != exp
    {
        return Err(format!(
            "merkle_root_sha256 mismatch for {}",
            target.display()
        ));
    }
    if let Some(exp) = sec["merkle_root_sha512"].as_str()
        && actual.root_sha512 != exp
    {
        return Err(format!(
            "merkle_root_sha512 mismatch for {}",
            target.display()
        ));
    }
    if let Some(exp) = sec["file_count"].as_u64()
        && actual.file_count != exp
    {
        return Err(format!(
            "file_count mismatch for {}: {} != {exp}",
            target.display(),
            actual.file_count
        ));
    }
    if let Some(exp) = sec["total_bytes"].as_u64()
        && actual.total_bytes != exp
    {
        return Err(format!(
            "total_bytes mismatch for {}: {} != {exp}",
            target.display(),
            actual.total_bytes
        ));
    }
    Ok(())
}

fn check_git(target: &Path, repo_sec: &Value, arch_sec: &Value) -> Result<(), String> {
    if !target.join(".git").exists() {
        return Ok(());
    }
    let rev = repo_sec["revision"]
        .as_str()
        .ok_or("missing repository.revision")?;
    let info = git_archive(target, rev)?;
    if let Some(t) = repo_sec["tree"].as_str()
        && info.tree != t
    {
        return Err(format!("git tree mismatch for {}", target.display()));
    }
    if let Some(s) = arch_sec["archive_sha256"].as_str()
        && info.archive_sha256 != s
    {
        return Err(format!("archive_sha256 mismatch for {}", target.display()));
    }
    if let Some(s) = arch_sec["archive_sha512"].as_str()
        && info.archive_sha512 != s
    {
        return Err(format!("archive_sha512 mismatch for {}", target.display()));
    }
    if let Some(b) = arch_sec["archive_bytes"].as_u64()
        && info.archive_bytes != b
    {
        return Err(format!("archive_bytes mismatch for {}", target.display()));
    }
    Ok(())
}

fn check_inputs(target: &Path, lock: &Value) -> Result<(), String> {
    let Some(paths) = lock["input_paths"].as_map() else {
        return Ok(());
    };
    let sha256_sec = &lock["inputs"];
    let sha512_sec = &lock["inputs_sha512"];
    for (k, v) in paths {
        let Some(key) = k.as_str() else { continue };
        let Some(rel) = v.as_str() else { continue };
        let (s256, s512, _) = hash_file(&target.join(rel))?;
        let k256 = format!("{key}_sha256");
        if let Some(exp) = sha256_sec[k256.as_str()]
            .as_str()
            .or_else(|| sha256_sec[key].as_str())
            && s256 != exp
        {
            return Err(format!("SHA-256 mismatch for {rel}: {s256} != {exp}"));
        }
        let k512 = format!("{key}_sha512");
        if let Some(exp) = sha512_sec[k512.as_str()]
            .as_str()
            .or_else(|| sha512_sec[key].as_str())
            && s512 != exp
        {
            return Err(format!("SHA-512 mismatch for {rel}: {s512} != {exp}"));
        }
    }
    Ok(())
}

pub fn verify_lock(root: &Path, name: &str, override_dir: Option<&Path>) -> Result<Value, String> {
    let stem = name.trim_end_matches(".lock.toml");
    let path = root
        .join("research/upstream")
        .join(format!("{stem}.lock.toml"));
    let text = fs::read_to_string(&path).map_err(|e| format!("read {}: {e}", path.display()))?;
    let lock: Value =
        toml::from_str(&text).map_err(|e| format!("parse {}: {e}", path.display()))?;
    let merkle_sec = &lock["merkle"];
    let target = match override_dir {
        Some(d) => d.to_path_buf(),
        None => {
            let rel = merkle_sec["cache_relpath"]
                .as_str()
                .ok_or_else(|| format!("missing merkle.cache_relpath in {}", path.display()))?;
            root.join(rel)
        }
    };
    check_merkle(&target, merkle_sec)?;
    if lock.get("repository").is_some() {
        check_git(&target, &lock["repository"], &lock["archive"])?;
    }
    check_inputs(&target, &lock)?;
    Ok(lock)
}
