use baux::artifact::json::{self, Value, json};
use baux::artifact::runs::digest;
use baux::artifact::toml;
use std::collections::BTreeMap;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

const MACHO_MAGIC: u32 = 0xFEED_FACF;
const CPU_ARM64: u32 = 0x0100_000C;
const AR_MAGIC: &[u8] = b"!<arch>\n";

#[derive(Default)]
pub struct CargoClosure {
    pub artifacts: BTreeMap<String, PathBuf>,
    pub extern_names: BTreeMap<String, String>,
    pub dep_files: Vec<String>,
    pub link_search: Vec<String>,
    pub link_libs: Vec<String>,
}

pub fn file_sha(path: &Path) -> Result<String, String> {
    let bytes = fs::read(path).map_err(|e| format!("read {}: {e}", path.display()))?;
    Ok(digest(&bytes))
}

fn walk_files(root: &Path, dir: &Path, out: &mut Vec<String>) -> Result<(), String> {
    let entries = fs::read_dir(dir).map_err(|e| format!("read_dir {}: {e}", dir.display()))?;
    for entry in entries {
        let path = entry.map_err(|e| e.to_string())?.path();
        if path.is_dir() {
            walk_files(root, &path, out)?;
        } else if path.is_file() {
            let rel = path.strip_prefix(root).map_err(|e| e.to_string())?;
            out.push(rel.to_string_lossy().replace('\\', "/"));
        }
    }
    Ok(())
}

pub fn tree_digest(root: &Path) -> Result<String, String> {
    let mut files = Vec::new();
    walk_files(root, root, &mut files)?;
    files.sort();
    let mut framed = Vec::new();
    for rel in files {
        let name = rel.as_bytes();
        let data = fs::read(root.join(&rel)).map_err(|e| e.to_string())?;
        framed.extend_from_slice(&(name.len() as u64).to_le_bytes());
        framed.extend_from_slice(name);
        framed.extend_from_slice(&(data.len() as u64).to_le_bytes());
        framed.extend_from_slice(&data);
    }
    Ok(digest(&framed))
}

pub fn inspect_header(path: &Path) -> Result<Value, String> {
    let data = fs::read(path).map_err(|e| format!("read {}: {e}", path.display()))?;
    let size = data.len() as u64;
    let sha = digest(&data);
    let magic_bytes = MACHO_MAGIC.to_le_bytes();
    let (fmt, cputype) = if data.starts_with(AR_MAGIC) {
        let pos = data
            .windows(4)
            .position(|w| w == magic_bytes)
            .ok_or_else(|| {
                format!(
                    "archive {} contains no Mach-O 64-bit object",
                    path.display()
                )
            })?;
        if pos + 8 > data.len() {
            return Err(format!("truncated Mach-O header in {}", path.display()));
        }
        let cpu = u32::from_le_bytes(data[pos + 4..pos + 8].try_into().unwrap());
        ("ar+mach-o-64", cpu)
    } else if data.len() >= 8 {
        let magic = u32::from_le_bytes(data[0..4].try_into().unwrap());
        if magic != MACHO_MAGIC {
            return Err(format!(
                "binary {} is not Mach-O 64-bit (magic={magic:#x})",
                path.display()
            ));
        }
        let cpu = u32::from_le_bytes(data[4..8].try_into().unwrap());
        ("mach-o-64", cpu)
    } else {
        return Err(format!(
            "binary {} is too small ({size} bytes)",
            path.display()
        ));
    };
    if cputype != CPU_ARM64 {
        return Err(format!(
            "binary {} has unexpected cputype {cputype:#x}",
            path.display()
        ));
    }
    Ok(json!({
        "format": fmt,
        "isa": "arm64",
        "cputype": format!("{cputype:#010x}"),
        "bytes": size,
        "sha256": sha,
    }))
}

fn verify_lockfiles(dir: &Path, pin: &Value) -> Result<(), String> {
    for (rel, key) in [
        ("Cargo.lock", "root_cargo_lock_sha256"),
        ("rust/Cargo.lock", "rust_cargo_lock_sha256"),
        ("pyproject.toml", "pyproject_sha256"),
    ] {
        let actual = file_sha(&dir.join(rel))?;
        if Some(actual.as_str()) != pin["inputs"][key].as_str() {
            return Err(format!("upstream pinned input mismatch for {rel}"));
        }
    }
    Ok(())
}

fn extract_archive(source: &Path, pin: &Value, destination: &Path) -> Result<Vec<u8>, String> {
    let revision = pin["repository"]["revision"]
        .as_str()
        .ok_or("missing upstream revision")?;
    let tree_out = Command::new("git")
        .args([
            "-C",
            &source.to_string_lossy(),
            "rev-parse",
            &format!("{revision}^{{tree}}"),
        ])
        .output()
        .map_err(|e| e.to_string())?;
    if !tree_out.status.success() {
        return Err("git rev-parse failed on upstream cache".into());
    }
    let tree = String::from_utf8_lossy(&tree_out.stdout).trim().to_owned();
    if Some(tree.as_str()) != pin["repository"]["tree"].as_str() {
        return Err("upstream revision does not match the pinned tree".into());
    }
    let archive_out = Command::new("git")
        .args(["-C", &source.to_string_lossy(), "archive", revision])
        .output()
        .map_err(|e| e.to_string())?;
    if !archive_out.status.success() {
        return Err("git archive failed on upstream cache".into());
    }
    fs::create_dir_all(destination).map_err(|e| e.to_string())?;
    let mut tar = Command::new("tar")
        .args(["-xf", "-", "-C"])
        .arg(destination)
        .stdin(Stdio::piped())
        .spawn()
        .map_err(|e| e.to_string())?;
    if let Some(mut stdin) = tar.stdin.take() {
        stdin
            .write_all(&archive_out.stdout)
            .map_err(|e| e.to_string())?;
    }
    if !tar.wait().map_err(|e| e.to_string())?.success() {
        return Err("tar extraction failed for upstream archive".into());
    }
    verify_lockfiles(destination, pin)?;
    Ok(archive_out.stdout)
}

pub fn snapshot_upstream(root: &Path, evidence: &Path) -> Result<PathBuf, String> {
    let pin_bytes =
        fs::read(root.join("research/upstream/enn.lock.toml")).map_err(|e| e.to_string())?;
    let pin_text = String::from_utf8(pin_bytes.clone()).map_err(|e| e.to_string())?;
    let pin: Value = toml::from_str(&pin_text).map_err(|e| e.to_string())?;
    let destination = evidence.join("upstream");
    let archive = extract_archive(&root.join(".cache/baux/upstream/enn"), &pin, &destination)?;
    let harness_bytes =
        fs::read(root.join("research/extraction/enn_harness.rs")).map_err(|e| e.to_string())?;
    let ennbo = destination.join("rust/crates/ennbo");
    fs::write(ennbo.join("src/baux_extract.rs"), &harness_bytes).map_err(|e| e.to_string())?;
    let lib_path = ennbo.join("src/lib.rs");
    let mut lib_bytes = fs::read(&lib_path).map_err(|e| e.to_string())?;
    let declaration = b"\npub mod baux_extract;\n";
    if String::from_utf8_lossy(&lib_bytes).contains("pub mod baux_extract;") {
        return Err("pinned upstream unexpectedly defines baux_extract".into());
    }
    lib_bytes.extend_from_slice(declaration);
    fs::write(&lib_path, &lib_bytes).map_err(|e| e.to_string())?;
    let metadata = json!({
        "repository": pin["repository"].clone(),
        "pin_sha256": digest(&pin_bytes),
        "archive_sha256": digest(&archive),
        "harness_sha256": digest(&harness_bytes),
        "harness_module_declaration_sha256": digest(declaration),
        "source": "git archive of pinned revision; working tree edits excluded; tracked single-call extraction harness injected",
    });
    let rendered = json::pretty_string(&metadata).map_err(|e| e.to_string())?;
    fs::write(evidence.join("upstream.json"), format!("{rendered}\n"))
        .map_err(|e| e.to_string())?;
    Ok(ennbo)
}

pub fn ensure_upstream(root: &Path) -> Result<(PathBuf, Value), String> {
    let probe_dir = root.join(".cache/baux/aeneas/probe-full-upstream");
    let upstream_dir = probe_dir.join("upstream");
    let upstream_json = probe_dir.join("upstream.json");
    let pin_bytes =
        fs::read(root.join("research/upstream/enn.lock.toml")).map_err(|e| e.to_string())?;
    let pin_text = String::from_utf8(pin_bytes.clone()).map_err(|e| e.to_string())?;
    let pin: Value = toml::from_str(&pin_text).map_err(|e| e.to_string())?;
    let pin_sha = digest(&pin_bytes);
    let harness_sha = file_sha(&root.join("research/extraction/enn_harness.rs"))?;

    let mut need_snapshot = true;
    if upstream_dir.is_dir() && upstream_json.is_file() {
        let text = fs::read_to_string(&upstream_json).map_err(|e| e.to_string())?;
        let meta: Value = json::from_str(&text).map_err(|e| e.to_string())?;
        if meta["pin_sha256"].as_str() == Some(pin_sha.as_str())
            && meta["harness_sha256"].as_str() == Some(harness_sha.as_str())
        {
            need_snapshot = false;
        }
    }
    if need_snapshot {
        if upstream_dir.exists() {
            fs::remove_dir_all(&upstream_dir).map_err(|e| e.to_string())?;
        }
        fs::create_dir_all(&probe_dir).map_err(|e| e.to_string())?;
        snapshot_upstream(root, &probe_dir)?;
    }
    verify_lockfiles(&upstream_dir, &pin)?;
    let text = fs::read_to_string(&upstream_json).map_err(|e| e.to_string())?;
    let meta: Value = json::from_str(&text).map_err(|e| e.to_string())?;
    Ok((upstream_dir, meta))
}

fn copy_dir(src: &Path, dst: &Path) -> Result<(), String> {
    fs::create_dir_all(dst).map_err(|e| e.to_string())?;
    for entry in fs::read_dir(src).map_err(|e| e.to_string())? {
        let entry = entry.map_err(|e| e.to_string())?;
        let path = entry.path();
        let target = dst.join(entry.file_name());
        if path.is_dir() {
            copy_dir(&path, &target)?;
        } else if path.is_file() {
            fs::copy(&path, &target).map_err(|e| e.to_string())?;
        }
    }
    Ok(())
}

pub fn ensure_slice(root: &Path) -> Result<(PathBuf, String), String> {
    let cache = root.join(".cache/baux/aeneas");
    fs::create_dir_all(&cache).map_err(|e| e.to_string())?;
    let source = root.join("rust/crates/baux/src");
    let dest = cache.join("local-slice");
    let src_digest = tree_digest(&source)?;
    let lib_root = root.join("research/extraction/local/lib.rs");
    let manifest_tpl = root.join("research/extraction/local/Cargo.toml.in");
    let expected = fs::read_to_string(&manifest_tpl)
        .map_err(|e| e.to_string())?
        .replace("@ROOT@", &root.to_string_lossy());

    let mut need_copy = true;
    if dest.join("src").is_dir() && dest.join("Cargo.toml").is_file() {
        let cur_manifest = fs::read_to_string(dest.join("Cargo.toml")).unwrap_or_default();
        let cur_lib = fs::read(dest.join("src/lib.rs")).unwrap_or_default();
        let exp_lib = fs::read(&lib_root).unwrap_or_default();
        if cur_manifest == expected && cur_lib == exp_lib {
            need_copy = false;
        }
    }
    if need_copy {
        if dest.exists() {
            fs::remove_dir_all(&dest).map_err(|e| e.to_string())?;
        }
        copy_dir(&source, &dest.join("src"))?;
        if tree_digest(&dest.join("src"))? != src_digest {
            return Err("local optimizer slice does not match production source tree".into());
        }
        fs::copy(&lib_root, dest.join("src/lib.rs")).map_err(|e| e.to_string())?;
        fs::write(dest.join("Cargo.toml"), &expected).map_err(|e| e.to_string())?;
    }
    Ok((dest, src_digest))
}

fn record_artifact(msg: &Value, closure: &mut CargoClosure) {
    let name = msg["target"]["name"]
        .as_str()
        .unwrap_or("")
        .replace('-', "_");
    let Some(seq) = msg["filenames"].as_seq() else {
        return;
    };
    let filenames = seq
        .iter()
        .filter_map(|v| v.as_str().map(PathBuf::from))
        .collect::<Vec<_>>();
    let rlibs = filenames
        .iter()
        .filter(|p| p.extension().and_then(|s| s.to_str()) == Some("rlib"))
        .cloned()
        .collect::<Vec<_>>();
    let rmetas = filenames
        .iter()
        .filter(|p| p.extension().and_then(|s| s.to_str()) == Some("rmeta"))
        .cloned()
        .collect::<Vec<_>>();
    for p in &filenames {
        if matches!(
            p.extension().and_then(|s| s.to_str()),
            Some("rlib" | "rmeta" | "dylib" | "so")
        ) {
            closure.dep_files.push(p.to_string_lossy().into_owned());
        }
    }
    if rlibs.is_empty() {
        return;
    }
    let pkg_id = msg["package_id"].as_str().unwrap_or("");
    if closure.artifacts.contains_key(&name) && name == "rand" && !pkg_id.contains("0.8.") {
        return;
    }
    closure.artifacts.insert(name.clone(), rlibs[0].clone());
    let ext_name = if let Some(rmeta) = rmetas.first() {
        let stem = rmeta.file_stem().and_then(|s| s.to_str()).unwrap_or("");
        format!("{stem}.rlib")
    } else {
        rlibs[0]
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("")
            .to_owned()
    };
    closure.extern_names.insert(name, ext_name);
}

fn record_script(msg: &Value, closure: &mut CargoClosure) {
    if let Some(paths) = msg["linked_paths"].as_seq() {
        for lp in paths.iter().filter_map(|v| v.as_str()) {
            if !closure.link_search.iter().any(|s| s == lp) {
                closure.link_search.push(lp.to_owned());
            }
        }
    }
    if let Some(libs) = msg["linked_libs"].as_seq() {
        for ll in libs.iter().filter_map(|v| v.as_str()) {
            if !closure.link_libs.iter().any(|s| s == ll) {
                closure.link_libs.push(ll.to_owned());
            }
        }
    }
}

pub fn cargo_build(
    root: &Path,
    manifest: &Path,
    target: &Path,
    packages: &[&str],
) -> Result<CargoClosure, String> {
    let mut cmd = Command::new("cargo");
    cmd.current_dir(root).args([
        "build",
        "--locked",
        "--offline",
        "--manifest-path",
        &manifest.to_string_lossy(),
        "--target-dir",
        &target.to_string_lossy(),
        "--lib",
        "--message-format=json",
    ]);
    for pkg in packages {
        cmd.args(["-p", pkg]);
    }
    let out = cmd.output().map_err(|e| e.to_string())?;
    if !out.status.success() {
        return Err(format!("cargo build failed for {}", manifest.display()));
    }
    let mut closure = CargoClosure::default();
    for raw in String::from_utf8_lossy(&out.stdout).lines() {
        let line = raw.trim();
        if !line.starts_with('{') {
            continue;
        }
        let msg: Value = json::from_str(line).map_err(|e| e.to_string())?;
        match msg["reason"].as_str() {
            Some("compiler-artifact") => record_artifact(&msg, &mut closure),
            Some("build-script-executed") => record_script(&msg, &mut closure),
            _ => {}
        }
    }
    Ok(closure)
}
