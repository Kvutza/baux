//! Verified Pyodide WASM staging, deterministic `baux_py.zip` builder,
//! and unified Rust security audit runner replacing legacy Python scripts.

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Component, Path, PathBuf};
use std::process::Command;

use baux::artifact::blake3::sha256_hex;
use baux::artifact::json::{self, Map, Value, json};

use crate::pages_verify;

const LOCAL_MARKERS: [&str; 2] = ["/Users/", "/home/runner/"];
const PYODIDE_PREFIX: &str = "https://cdn.jsdelivr.net/pyodide/";
const WASM_FILENAME: &str = "baux_rust.cpython-312-wasm32-emscripten.so";
const WASM_ARCNAME: &str = "baux/baux_rust.cpython-312-wasm32-emscripten.so";

const PYODIDE_VERSION: &str = "0.27.5";
const PYODIDE_URL: &str = "https://cdn.jsdelivr.net/pyodide/v0.27.5/full/";

struct ManifestEntry {
    name: &'static str,
    size: u64,
    sha256: &'static str,
}

const PYODIDE_FILES: [ManifestEntry; 6] = [
    ManifestEntry {
        name: "pyodide.js",
        size: 14_928,
        sha256: "7fdbe66e53f68f6a4e93c295a667371759be093d2bd402bb44545514584039b6",
    },
    ManifestEntry {
        name: "pyodide.asm.js",
        size: 1_253_804,
        sha256: "3a889f073e628c2196c705b42fa0e955ba2e25c034b1e3dd589c35be675bc01b",
    },
    ManifestEntry {
        name: "pyodide.asm.wasm",
        size: 10_103_326,
        sha256: "f7fefe563134714a17abd65516d94960e8dbd96fe6778a7a842947fc9686b3a1",
    },
    ManifestEntry {
        name: "python_stdlib.zip",
        size: 2_358_894,
        sha256: "6030964967e447c887abc46c5f0967c55688644d759496de82a3ef09f49f5cba",
    },
    ManifestEntry {
        name: "pyodide-lock.json",
        size: 112_205,
        sha256: "be1807745da93daa09d360b109c17a0e526e74d664d1f1b9870aafcce98ce426",
    },
    ManifestEntry {
        name: "numpy-2.0.2-cp312-cp312-pyodide_2024_0_wasm32.whl",
        size: 3_106_119,
        sha256: "d9cc75a959bbfb14efe05e26ca04cb2c85acbdad8b0a07a1c0140c4b820b4eec",
    },
];

const THEME_SCRIPTS: [(&str, &str); 3] = [
    ("repl_worker_js", "repl_worker.js"),
    ("webgpu_bridge_js", "webgpu_bridge.js"),
    ("webgpu_worker_js", "webgpu_worker.js"),
];

const NET_MARKERS: [&str; 6] = [
    "ConnectionError",
    "NameResolutionError",
    "Temporary failure in name resolution",
    "Failed to establish a new connection",
    "Max retries exceeded",
    "SSLError",
];

fn check_noleak(label: &str, payload: &[u8]) -> Result<(), String> {
    for marker in LOCAL_MARKERS {
        if payload
            .windows(marker.len())
            .any(|window| window == marker.as_bytes())
        {
            return Err(format!("Host path marker '{marker}' leaked into {label}"));
        }
    }
    Ok(())
}

fn crc32(data: &[u8]) -> u32 {
    let mut crc = !0_u32;
    for &byte in data {
        crc ^= u32::from(byte);
        for _ in 0..8 {
            crc = (crc >> 1) ^ (0xedb8_8320 & (0_u32.wrapping_sub(crc & 1)));
        }
    }
    !crc
}

fn u16_le(out: &mut Vec<u8>, value: u16) {
    out.extend_from_slice(&value.to_le_bytes());
}

fn u32_le(out: &mut Vec<u8>, value: u32) {
    out.extend_from_slice(&value.to_le_bytes());
}

fn raw_deflate(data: &[u8]) -> Option<Vec<u8>> {
    if data.len() < 32 || data.len() > u32::MAX as usize {
        return None;
    }
    let out = miniz_oxide::deflate::compress_to_vec(data, 9);
    if out.len() >= data.len() {
        return None;
    }
    Some(out)
}

fn deterministic_zip(entries: &BTreeMap<String, Vec<u8>>) -> Result<Vec<u8>, String> {
    let mut out = Vec::new();
    let mut central = Vec::new();
    for (name, data) in entries {
        let name_bytes = name.as_bytes();
        let name_len = u16::try_from(name_bytes.len())
            .map_err(|_| format!("ZIP entry path too long: {name}"))?;
        let offset = u32::try_from(out.len())
            .map_err(|_| "ZIP payload exceeded 4 GiB offset limit".to_string())?;
        let uncomp_len =
            u32::try_from(data.len()).map_err(|_| format!("ZIP entry too large: {name}"))?;
        let crc = crc32(data);
        let deflated = raw_deflate(data);
        let (method, payload): (u16, &[u8]) = match deflated.as_deref() {
            Some(compressed) => (8, compressed),
            None => (0, data.as_slice()),
        };
        let comp_len = u32::try_from(payload.len())
            .map_err(|_| format!("Compressed ZIP entry too large: {name}"))?;

        u32_le(&mut out, 0x0403_4b50);
        u16_le(&mut out, 20);
        u16_le(&mut out, 0);
        u16_le(&mut out, method);
        u16_le(&mut out, 0);
        u16_le(&mut out, 0x21);
        u32_le(&mut out, crc);
        u32_le(&mut out, comp_len);
        u32_le(&mut out, uncomp_len);
        u16_le(&mut out, name_len);
        u16_le(&mut out, 0);
        out.extend_from_slice(name_bytes);
        out.extend_from_slice(payload);

        u32_le(&mut central, 0x0201_4b50);
        u16_le(&mut central, 0x0314);
        u16_le(&mut central, 20);
        u16_le(&mut central, 0);
        u16_le(&mut central, method);
        u16_le(&mut central, 0);
        u16_le(&mut central, 0x21);
        u32_le(&mut central, crc);
        u32_le(&mut central, comp_len);
        u32_le(&mut central, uncomp_len);
        u16_le(&mut central, name_len);
        u16_le(&mut central, 0);
        u16_le(&mut central, 0);
        u16_le(&mut central, 0);
        u16_le(&mut central, 0);
        u32_le(&mut central, 0o100644_u32 << 16);
        u32_le(&mut central, offset);
        central.extend_from_slice(name_bytes);
    }

    let central_offset =
        u32::try_from(out.len()).map_err(|_| "ZIP central directory exceeded 4 GiB".to_string())?;
    let central_len =
        u32::try_from(central.len()).map_err(|_| "ZIP central directory too large".to_string())?;
    let count = u16::try_from(entries.len()).map_err(|_| "Too many ZIP entries".to_string())?;
    out.extend_from_slice(&central);
    u32_le(&mut out, 0x0605_4b50);
    u16_le(&mut out, 0);
    u16_le(&mut out, 0);
    u16_le(&mut out, count);
    u16_le(&mut out, count);
    u32_le(&mut out, central_len);
    u32_le(&mut out, central_offset);
    u16_le(&mut out, 0);
    Ok(out)
}

fn walk_files(root: &Path, out: &mut Vec<PathBuf>) -> Result<(), String> {
    if !root.is_dir() {
        return Ok(());
    }
    for entry in fs::read_dir(root).map_err(|e| format!("read_dir {}: {e}", root.display()))? {
        let entry = entry.map_err(|e| e.to_string())?;
        let path = entry.path();
        if path.is_dir() {
            walk_files(&path, out)?;
        } else {
            out.push(path);
        }
    }
    Ok(())
}

fn find_wasmso(root: &Path, explicit: Option<&Path>) -> Result<PathBuf, String> {
    let verify = |cand: &Path, origin: &str| -> Result<PathBuf, String> {
        let resolved = if cand.is_absolute() {
            cand.to_path_buf()
        } else {
            root.join(cand)
        };
        if resolved.is_file() {
            Ok(resolved)
        } else {
            Err(format!(
                "WASM extension specified via {origin} not found: {}",
                cand.display()
            ))
        }
    };

    if let Some(p) = explicit {
        return verify(p, "--wasm-so");
    }
    if let Ok(env_val) = std::env::var("BAUX_WASM_SO")
        && !env_val.trim().is_empty()
    {
        return verify(Path::new(env_val.trim()), "BAUX_WASM_SO");
    }

    let direct_candidates = [
        root.join(".cache/baux/wasm").join(WASM_FILENAME),
        root.join("rust/target/wasm32-unknown-emscripten/release")
            .join(WASM_FILENAME),
        root.join("rust/target/wasm32-unknown-emscripten/release/libbaux_rust.so"),
    ];
    for cand in direct_candidates {
        if cand.is_file() {
            return Ok(cand);
        }
    }

    let buck_out = root.join("buck-out");
    if buck_out.is_dir() {
        let mut all = Vec::new();
        walk_files(&buck_out, &mut all)?;
        all.sort();
        if let Some(exact) = all
            .iter()
            .find(|p| p.file_name().and_then(|n| n.to_str()) == Some(WASM_FILENAME))
        {
            return Ok(exact.clone());
        }
        if let Some(target_so) = all.iter().find(|p| {
            p.extension().and_then(|e| e.to_str()) == Some("so")
                && p.to_string_lossy().contains("baux-py-wasm")
        }) {
            return Ok(target_so.clone());
        }
    }

    Err(format!(
        "Missing compiled {WASM_FILENAME}. Pass --wasm-so PATH, set BAUX_WASM_SO, \
         or build //:baux-py-wasm into buck-out or .cache/baux/wasm/."
    ))
}

fn skip_pyfile(src: &Path) -> bool {
    for comp in src.components() {
        if let Component::Normal(name) = comp {
            let s = name.to_string_lossy();
            if matches!(
                s.as_ref(),
                "__pycache__" | ".ruff_cache" | ".pytest_cache" | ".mypy_cache"
            ) {
                return true;
            }
        }
    }
    matches!(
        src.extension().and_then(|e| e.to_str()),
        Some("pyc" | "pyo" | "so" | "pyd" | "dylib")
    )
}

fn pyzip_bytes(root: &Path, so_payload: &[u8]) -> Result<Vec<u8>, String> {
    let py_root = root.join("python/baux");
    if !py_root.is_dir() {
        return Err(format!(
            "Required browser REPL source not found: {}",
            py_root.display()
        ));
    }

    let mut py_files = Vec::new();
    walk_files(&py_root, &mut py_files)?;
    py_files.sort();

    let mut entries = BTreeMap::new();
    for src in py_files {
        if skip_pyfile(&src) {
            continue;
        }
        let rel = src
            .strip_prefix(&py_root)
            .map_err(|e| e.to_string())?
            .to_string_lossy()
            .replace('\\', "/");
        let arcname = format!("baux/{rel}");
        let payload = fs::read(&src).map_err(|e| format!("read {}: {e}", src.display()))?;
        check_noleak(&arcname, &payload)?;
        entries.insert(arcname, payload);
    }

    check_noleak(WASM_ARCNAME, so_payload)?;
    entries.insert(WASM_ARCNAME.to_string(), so_payload.to_vec());
    deterministic_zip(&entries)
}

fn download_verified(
    url: &str,
    dst: &Path,
    expected_size: u64,
    expected_sha256: &str,
) -> Result<(), String> {
    if !url.starts_with(PYODIDE_PREFIX) {
        return Err(format!("Refusing non-allowlisted Pyodide URL: {url}"));
    }
    if let Some(parent) = dst.parent() {
        fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    }
    let tmp = dst.with_extension(format!(
        "{}.tmp",
        dst.extension().and_then(|e| e.to_str()).unwrap_or("bin")
    ));
    let status = Command::new("curl")
        .args([
            "--fail",
            "--location",
            "--silent",
            "--show-error",
            "--proto",
            "=https",
            "--tlsv1.2",
            "--max-time",
            "120",
            "-A",
            "baux-pages-builder/1.0",
            "-o",
            &tmp.display().to_string(),
            url,
        ])
        .status()
        .map_err(|e| format!("run curl {url}: {e}"))?;
    if !status.success() {
        let _ = fs::remove_file(&tmp);
        return Err(format!("curl failed fetching {url} (exit {status})"));
    }
    let bytes = fs::read(&tmp).map_err(|e| format!("read {}: {e}", tmp.display()))?;
    let actual_size = bytes.len() as u64;
    let actual_sha = sha256_hex(&bytes);
    if actual_size != expected_size || actual_sha != expected_sha256 {
        let _ = fs::remove_file(&tmp);
        return Err(format!(
            "Verification failed for {}: size {} != {} or sha256 {} != {}",
            dst.display(),
            actual_size,
            expected_size,
            actual_sha,
            expected_sha256
        ));
    }
    fs::rename(&tmp, dst).map_err(|e| format!("rename {} -> {}: {e}", tmp.display(), dst.display()))
}

fn stage_pyodide(cache_dir: &Path, offline: bool) -> Result<Vec<(String, PathBuf)>, String> {
    let mut resolved = Vec::with_capacity(PYODIDE_FILES.len());
    for entry in PYODIDE_FILES {
        if entry.name.contains('/') || entry.name.contains('\\') || entry.name.contains("..") {
            return Err(format!(
                "Unsafe filename in Pyodide manifest: {}",
                entry.name
            ));
        }
        let target = cache_dir.join(entry.name);
        let mut valid_cached = false;
        if target.is_file() {
            let bytes = fs::read(&target).map_err(|e| format!("read {}: {e}", target.display()))?;
            if bytes.len() as u64 == entry.size && sha256_hex(&bytes) == entry.sha256 {
                valid_cached = true;
            }
        }
        if !valid_cached {
            if offline {
                return Err(format!(
                    "Missing or invalid cached Pyodide file {} in --offline mode",
                    target.display()
                ));
            }
            let url = format!("{PYODIDE_URL}{}", entry.name);
            download_verified(&url, &target, entry.size, entry.sha256)?;
        }
        resolved.push((entry.name.to_string(), target));
    }
    Ok(resolved)
}

fn trim_lockfile(lock_path: &Path, allowed_roots: &[&str]) -> Result<Vec<u8>, String> {
    let raw =
        fs::read_to_string(lock_path).map_err(|e| format!("read {}: {e}", lock_path.display()))?;
    let data: Value =
        json::from_str(&raw).map_err(|e| format!("parse {}: {e}", lock_path.display()))?;
    let all_pkgs = data["packages"]
        .as_map()
        .ok_or_else(|| format!("{} missing 'packages' map", lock_path.display()))?;

    let mut keep = BTreeSet::new();
    let mut stack: Vec<String> = allowed_roots.iter().map(|s| (*s).to_string()).collect();
    while let Some(pkg) = stack.pop() {
        if keep.contains(&pkg) {
            continue;
        }
        let Some(pkg_val) = all_pkgs.get(pkg.as_str()) else {
            continue;
        };
        keep.insert(pkg);
        if let Some(deps) = pkg_val["depends"].as_seq() {
            for dep in deps {
                if let Some(dep_name) = dep.as_str() {
                    stack.push(dep_name.to_string());
                }
            }
        }
    }

    let mut trimmed_pkgs = Map::new();
    for pkg_name in keep {
        if let Some(pkg_val) = all_pkgs.get(pkg_name.as_str()) {
            trimmed_pkgs.insert(pkg_name, pkg_val.clone());
        }
    }
    let mut root_obj = Map::new();
    root_obj.insert("info", data["info"].clone());
    root_obj.insert("packages", Value::from(trimmed_pkgs));
    json::to_vec(&Value::from(root_obj)).map_err(|e| e.to_string())
}

fn stage_scripts(root: &Path, assets_dirs: &[PathBuf]) -> Result<Value, String> {
    let mut companion_entries = Map::new();
    let theme_dir = root.join("forest/theme");

    for (manifest_key, filename) in THEME_SCRIPTS {
        let src = theme_dir.join(filename);
        if !src.is_file() {
            continue;
        }
        let blob = fs::read(&src).map_err(|e| format!("read {}: {e}", src.display()))?;
        check_noleak(filename, &blob)?;
        for assets_dir in assets_dirs {
            let dst = assets_dir.join(filename);
            let _ = fs::remove_file(&dst);
            fs::write(&dst, &blob).map_err(|e| format!("write {}: {e}", dst.display()))?;
        }
        companion_entries.insert(
            manifest_key,
            json!({
                "path": filename,
                "sha256": sha256_hex(&blob),
                "size": blob.len() as u64,
            }),
        );
    }

    let oxcaml_candidates = [
        root.join("admin/ui/_build/default/oxcaml_eval.js"),
        theme_dir.join("oxcaml_eval.js"),
    ];
    let oxcaml_src = oxcaml_candidates.into_iter().find(|p| p.is_file());
    if let Some(src) = oxcaml_src {
        let blob = fs::read(&src).map_err(|e| format!("read {}: {e}", src.display()))?;
        check_noleak("oxcaml_eval.js", &blob)?;
        for assets_dir in assets_dirs {
            let dst = assets_dir.join("oxcaml_eval.js");
            let _ = fs::remove_file(&dst);
            fs::write(&dst, &blob).map_err(|e| format!("write {}: {e}", dst.display()))?;
        }
        companion_entries.insert(
            "oxcaml_eval_js",
            json!({
                "path": "oxcaml_eval.js",
                "sha256": sha256_hex(&blob),
                "size": blob.len() as u64,
            }),
        );
    } else if !assets_dirs.is_empty()
        && assets_dirs
            .iter()
            .all(|d| d.join("oxcaml_eval.js").is_file())
    {
        let blob = fs::read(assets_dirs[0].join("oxcaml_eval.js")).map_err(|e| e.to_string())?;
        check_noleak("oxcaml_eval.js", &blob)?;
        companion_entries.insert(
            "oxcaml_eval_js",
            json!({
                "path": "oxcaml_eval.js",
                "sha256": sha256_hex(&blob),
                "size": blob.len() as u64,
            }),
        );
    }

    Ok(Value::from(companion_entries))
}

pub struct ReplStageConfig<'a> {
    pub wasm_so: Option<&'a Path>,
    pub assets_dirs: &'a [PathBuf],
    pub cache_dir: Option<&'a Path>,
    pub offline: bool,
    pub zip_only: bool,
}

/// Stage verified Pyodide WASM assets, `baux_py.zip`, companion JS scripts, and
/// `repl_manifest.json`.
pub fn stage_repl(root: &Path, cfg: &ReplStageConfig<'_>) -> Result<Value, String> {
    if cfg.assets_dirs.is_empty() {
        return Err("at least one --assets-dir is required".into());
    }
    let so_path = find_wasmso(root, cfg.wasm_so)?;
    let so_payload = fs::read(&so_path).map_err(|e| format!("read {}: {e}", so_path.display()))?;
    if so_payload.is_empty() {
        return Err(format!(
            "WASM extension file is empty: {}",
            so_path.display()
        ));
    }
    check_noleak(WASM_ARCNAME, &so_payload)?;

    let zip_bytes = pyzip_bytes(root, &so_payload)?;
    let zip_sha = sha256_hex(&zip_bytes);

    let default_cache = root
        .join(".cache/baux/pyodide")
        .join(format!("v{PYODIDE_VERSION}"));
    let cache_dir = cfg.cache_dir.unwrap_or(&default_cache);

    let mut pyodide_entries = Map::new();
    let mut staged_pyodide_blobs = BTreeMap::new();
    if !cfg.zip_only {
        for (name, src_path) in stage_pyodide(cache_dir, cfg.offline)? {
            let blob = if name == "pyodide-lock.json" {
                trim_lockfile(&src_path, &["numpy"])?
            } else {
                fs::read(&src_path).map_err(|e| format!("read {}: {e}", src_path.display()))?
            };
            if name.ends_with(".js") || name.ends_with(".json") {
                check_noleak(&name, &blob)?;
            }
            pyodide_entries.insert(
                name.clone(),
                json!({
                    "sha256": sha256_hex(&blob),
                    "size": blob.len() as u64,
                }),
            );
            staged_pyodide_blobs.insert(name, blob);
        }
    }

    for assets_dir in cfg.assets_dirs {
        fs::create_dir_all(assets_dir)
            .map_err(|e| format!("mkdir {}: {e}", assets_dir.display()))?;
        let zip_dst = assets_dir.join("baux_py.zip");
        let _ = fs::remove_file(&zip_dst);
        fs::write(&zip_dst, &zip_bytes).map_err(|e| format!("write {}: {e}", zip_dst.display()))?;
        if !cfg.zip_only {
            let pyo_dir = assets_dir.join("pyodide");
            fs::create_dir_all(&pyo_dir)
                .map_err(|e| format!("mkdir {}: {e}", pyo_dir.display()))?;
            for (name, blob) in &staged_pyodide_blobs {
                let dst = pyo_dir.join(name);
                let _ = fs::remove_file(&dst);
                fs::write(&dst, blob).map_err(|e| format!("write {}: {e}", dst.display()))?;
            }
        }
    }

    let companion_entries = stage_scripts(root, cfg.assets_dirs)?;
    let repl_manifest = json!({
        "baux_py_zip": {
            "path": "baux_py.zip",
            "sha256": zip_sha,
            "size": zip_bytes.len() as u64,
        },
        "companion_files": companion_entries,
        "pyodide_files": Value::from(pyodide_entries),
        "pyodide_version": PYODIDE_VERSION,
        "schema_version": 1_u64,
        "wasm_extension": {
            "path": WASM_ARCNAME,
            "sha256": sha256_hex(&so_payload),
            "size": so_payload.len() as u64,
        },
    });
    let mut manifest_text = json::pretty_string(&repl_manifest).map_err(|e| e.to_string())?;
    manifest_text.push('\n');

    for assets_dir in cfg.assets_dirs {
        let dst = assets_dir.join("repl_manifest.json");
        let _ = fs::remove_file(&dst);
        fs::write(&dst, manifest_text.as_bytes())
            .map_err(|e| format!("write {}: {e}", dst.display()))?;
    }

    Ok(repl_manifest)
}

#[derive(Clone)]
pub struct AuditConfig {
    pub bundle_dir: Option<PathBuf>,
    pub strict: bool,
    pub offline: bool,
    pub graceful_offline: bool,
    pub ignore_vulns: Vec<String>,
    pub skip_zizmor: bool,
    pub skip_semgrep: bool,
    pub skip_pip_audit: bool,
}

struct GateResult {
    name: &'static str,
    status: &'static str,
    detail: String,
}

fn capture_cmd(
    root: &Path,
    program: &str,
    args: &[&str],
    envs: &[(&str, &str)],
) -> Result<(i32, String), String> {
    let mut cmd = Command::new(program);
    cmd.current_dir(root).args(args);
    for (k, v) in envs {
        cmd.env(k, v);
    }
    let output = cmd.output().map_err(|e| format!("spawn {program}: {e}"))?;
    let mut combined = String::new();
    let out = String::from_utf8_lossy(&output.stdout);
    let err = String::from_utf8_lossy(&output.stderr);
    if !out.trim().is_empty() {
        combined.push_str(out.trim());
    }
    if !err.trim().is_empty() {
        if !combined.is_empty() {
            combined.push('\n');
        }
        combined.push_str(err.trim());
    }
    Ok((output.status.code().unwrap_or(1), combined))
}

fn gate_tree(root: &Path) -> GateResult {
    println!("[1/5] Public repository hygiene & exclude guard");
    match pages_verify::verify_publictree(root) {
        Ok(()) => GateResult {
            name: "tree_guard",
            status: "PASS",
            detail: "Tracked files obey public allowlist and exclude gates".into(),
        },
        Err(e) => GateResult {
            name: "tree_guard",
            status: "FAIL",
            detail: e,
        },
    }
}

fn gate_zizmor(root: &Path, skip: bool) -> GateResult {
    println!("[2/5] GitHub Actions audit (zizmor --pedantic)");
    if skip {
        return GateResult {
            name: "zizmor",
            status: "SKIP",
            detail: "Skipped via --skip-zizmor".into(),
        };
    }
    let wf_dir = root.join(".github/workflows");
    if !wf_dir.is_dir() {
        return GateResult {
            name: "zizmor",
            status: "SKIP",
            detail: "No .github/workflows directory".into(),
        };
    }
    match capture_cmd(
        root,
        "uvx",
        &[
            "zizmor",
            "--pedantic",
            "-qq",
            "--no-progress",
            "--color=never",
            ".github/workflows/",
        ],
        &[],
    ) {
        Ok((0, _)) => GateResult {
            name: "zizmor",
            status: "PASS",
            detail: "0 workflow security findings".into(),
        },
        Ok((code, out)) => {
            if !out.is_empty() {
                println!("{out}");
            }
            GateResult {
                name: "zizmor",
                status: "FAIL",
                detail: format!("zizmor exited with {code}"),
            }
        }
        Err(e) => GateResult {
            name: "zizmor",
            status: "FAIL",
            detail: e,
        },
    }
}

fn gate_semgrep(root: &Path, skip: bool) -> GateResult {
    println!("[3/5] Semgrep static analysis (.semgrep/security.yml)");
    if skip {
        return GateResult {
            name: "semgrep",
            status: "SKIP",
            detail: "Skipped via --skip-semgrep".into(),
        };
    }
    let cfg_path = root.join(".semgrep/security.yml");
    if !cfg_path.is_file() {
        return GateResult {
            name: "semgrep",
            status: "FAIL",
            detail: "Missing .semgrep/security.yml".into(),
        };
    }
    let envs = [("SEMGREP_SEND_METRICS", "off"), ("NO_COLOR", "1")];
    match capture_cmd(
        root,
        "uvx",
        &[
            "semgrep",
            "scan",
            "--quiet",
            "--metrics=off",
            "--validate",
            "--config",
            ".semgrep/security.yml",
        ],
        &envs,
    ) {
        Ok((0, _)) => {}
        Ok((code, out)) => {
            if !out.is_empty() {
                println!("{out}");
            }
            return GateResult {
                name: "semgrep",
                status: "FAIL",
                detail: format!("Semgrep rule validation exited with {code}"),
            };
        }
        Err(e) => {
            return GateResult {
                name: "semgrep",
                status: "FAIL",
                detail: e,
            };
        }
    }

    let mut targets = Vec::new();
    for rel in ["admin/ops", "admin/ui", "forest/theme"] {
        if root.join(rel).exists() {
            targets.push(rel);
        }
    }
    let mut args = vec![
        "semgrep",
        "scan",
        "--quiet",
        "--metrics=off",
        "--no-git-ignore",
        "--error",
        "--config",
        ".semgrep/security.yml",
        "--exclude",
        "node_modules",
        "--exclude",
        "forester.js",
        "--exclude",
        "__pycache__",
        "--exclude",
        "_build",
        "--exclude",
        "fonts",
    ];
    args.extend(targets.iter().copied());
    match capture_cmd(root, "uvx", &args, &envs) {
        Ok((0, _)) => GateResult {
            name: "semgrep",
            status: "PASS",
            detail: format!("0 findings across {} target path(s)", targets.len()),
        },
        Ok((code, out)) => {
            if !out.is_empty() {
                println!("{out}");
            }
            GateResult {
                name: "semgrep",
                status: "FAIL",
                detail: format!("Semgrep reported findings (exit {code})"),
            }
        }
        Err(e) => GateResult {
            name: "semgrep",
            status: "FAIL",
            detail: e,
        },
    }
}

fn gate_pipaudit(root: &Path, cfg: &AuditConfig) -> GateResult {
    println!("[4/5] Dependency vulnerability audit (pip-audit)");
    if cfg.skip_pip_audit {
        return GateResult {
            name: "pip-audit",
            status: "SKIP",
            detail: "Skipped via --skip-pip-audit".into(),
        };
    }
    if cfg.offline {
        return GateResult {
            name: "pip-audit",
            status: "SKIP",
            detail: "Offline mode requested (--offline); skipping OSV query".into(),
        };
    }

    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let tmp_req = std::env::temp_dir().join(format!("baux-req-{}-{nanos}.txt", std::process::id()));
    let tmp_req_str = tmp_req.display().to_string();

    if root.join("uv.lock").is_file() {
        match capture_cmd(
            root,
            "uv",
            &[
                "export",
                "--frozen",
                "--no-emit-project",
                "--no-hashes",
                "--output-file",
                &tmp_req_str,
            ],
            &[],
        ) {
            Ok((0, _)) => {}
            Ok((code, out)) => {
                let _ = fs::remove_file(&tmp_req);
                if !out.is_empty() {
                    println!("{out}");
                }
                return GateResult {
                    name: "pip-audit",
                    status: "FAIL",
                    detail: format!("uv export exited with {code}"),
                };
            }
            Err(e) => {
                let _ = fs::remove_file(&tmp_req);
                return GateResult {
                    name: "pip-audit",
                    status: "FAIL",
                    detail: e,
                };
            }
        }
    }

    let mut args: Vec<&str> = vec!["pip-audit"];
    if cfg.strict {
        args.push("--strict");
    }
    if root.join("uv.lock").is_file() {
        args.extend(["--no-deps", "--disable-pip", "-r", &tmp_req_str]);
    }
    for vuln in &cfg.ignore_vulns {
        args.push("--ignore-vuln");
        args.push(vuln.as_str());
    }

    let res = capture_cmd(root, "uvx", &args, &[]);
    let _ = fs::remove_file(&tmp_req);
    match res {
        Ok((0, out)) => {
            if !out.is_empty() {
                println!("{out}");
            }
            GateResult {
                name: "pip-audit",
                status: "PASS",
                detail: "0 known vulnerabilities in uv.lock".into(),
            }
        }
        Ok((code, out)) => {
            if !out.is_empty() {
                println!("{out}");
            }
            if cfg.graceful_offline && NET_MARKERS.iter().any(|m| out.contains(m)) {
                GateResult {
                    name: "pip-audit",
                    status: "SKIP",
                    detail: "Network unreachable during OSV lookup (--graceful-offline)".into(),
                }
            } else {
                GateResult {
                    name: "pip-audit",
                    status: "FAIL",
                    detail: format!("pip-audit exited with {code}"),
                }
            }
        }
        Err(e) => GateResult {
            name: "pip-audit",
            status: "FAIL",
            detail: e,
        },
    }
}

fn gate_bundle(bundle_dir: Option<&Path>) -> GateResult {
    println!("[5/5] Public Pages bundle verification");
    let Some(dir) = bundle_dir else {
        return GateResult {
            name: "verify_pages",
            status: "SKIP",
            detail: "No --bundle-dir provided; bundle check skipped".into(),
        };
    };
    match pages_verify::verify_bundle(dir) {
        Ok(()) => GateResult {
            name: "verify_pages",
            status: "PASS",
            detail: format!("Verified bundle at {}", dir.display()),
        },
        Err(e) => GateResult {
            name: "verify_pages",
            status: "FAIL",
            detail: e,
        },
    }
}

/// Run the full repository security audit suite without Python orchestration
/// scripts.
pub fn security_audit(root: &Path, cfg: &AuditConfig) -> Result<(), String> {
    let results = [
        gate_tree(root),
        gate_zizmor(root, cfg.skip_zizmor),
        gate_semgrep(root, cfg.skip_semgrep),
        gate_pipaudit(root, cfg),
        gate_bundle(cfg.bundle_dir.as_deref()),
    ];

    println!("Security audit summary:");
    let mut failed = Vec::new();
    for res in &results {
        println!("  [{:4}] {:14} : {}", res.status, res.name, res.detail);
        if res.status == "FAIL" {
            failed.push(format!("{}: {}", res.name, res.detail));
        }
    }
    if !failed.is_empty() {
        return Err(failed.join("; "));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn zip_deterministic() {
        let mut entries = BTreeMap::new();
        entries.insert(
            "baux/hello.py".to_string(),
            b"print('hello baux')".repeat(40),
        );
        let z1 = deterministic_zip(&entries).expect("zip 1");
        let z2 = deterministic_zip(&entries).expect("zip 2");
        assert_eq!(z1, z2);
        assert!(z1.len() < 400);
        assert_eq!(&z1[0..4], &[0x50, 0x4b, 0x03, 0x04]);
    }

    #[test]
    fn lockfile_trimmed() {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock")
            .as_nanos();
        let lock_path = std::env::temp_dir().join(format!("baux-lock-{nanos}.json"));
        fs::write(
            &lock_path,
            r#"{"info":{"arch":"wasm32"},"packages":{"numpy":{"depends":["openblas"]},"openblas":{"depends":[]},"pandas":{"depends":["numpy"]}}}"#,
        )
        .unwrap();
        let out = trim_lockfile(&lock_path, &["numpy"]).expect("trim");
        let _ = fs::remove_file(&lock_path);
        let s = String::from_utf8(out).unwrap();
        assert!(s.contains("\"numpy\""));
        assert!(s.contains("\"openblas\""));
        assert!(!s.contains("\"pandas\""));
    }
}
