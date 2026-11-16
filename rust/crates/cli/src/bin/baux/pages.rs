use std::fs;
use std::io::{BufRead, BufReader, BufWriter, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::{Component, Path, PathBuf};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Mutex, OnceLock};
use std::time::Instant;

use baux::artifact::blake3::hash_hex;
use baux::artifact::json::{self, Map, Value, json};
use baux::artifact::runs;
use clap::Subcommand;

use crate::{pages_repl, pages_verify};

#[derive(Subcommand)]
pub enum PagesAction {
    /// Build OCaml 3D CAD UI, export FUSE.jl environments,
    /// run Forester, stamp SRI, and verify.
    Build {
        #[arg(long, default_value = ".cache/pages-bundle")]
        bundle_dir: PathBuf,
    },
    /// Verify CSP, SRI, zero machine-path leaks, and bundle hygiene.
    Check {
        #[arg(long, default_value = ".cache/pages-bundle")]
        bundle_dir: PathBuf,
        /// Check only HTML/CSP/SRI/leak/suffix security rules (skip full bundle
        /// inventory).
        #[arg(long)]
        security_only: bool,
    },
    /// Stamp content-hash cache busters (?v=...) and SHA-384 SRI attributes
    /// onto local assets.
    StampSri {
        #[arg(long, default_value = ".cache/pages-bundle")]
        bundle_dir: PathBuf,
    },
    /// Stage verified Pyodide assets, baux_py.zip, companion scripts, and
    /// repl_manifest.json.
    ReplAssets {
        #[arg(long)]
        wasm_so: Option<PathBuf>,
        #[arg(long = "assets-dir", required = true)]
        assets_dirs: Vec<PathBuf>,
        #[arg(long)]
        cache_dir: Option<PathBuf>,
        #[arg(long)]
        offline: bool,
        #[arg(long)]
        zip_only: bool,
        #[arg(long)]
        json: bool,
    },
    /// Audit public-tree hygiene, GitHub Actions, Semgrep rules, Python deps,
    /// and Pages bundle.
    Audit {
        #[arg(long)]
        bundle_dir: Option<PathBuf>,
        #[arg(long)]
        no_strict: bool,
        #[arg(long)]
        offline: bool,
        #[arg(long)]
        graceful_offline: bool,
        #[arg(long = "ignore-vuln")]
        ignore_vulns: Vec<String>,
        #[arg(long)]
        skip_zizmor: bool,
        #[arg(long)]
        skip_semgrep: bool,
        #[arg(long)]
        skip_pip_audit: bool,
    },
    /// Serve the static bundle with same-origin /api/env and /api/runs
    /// endpoints.
    Serve {
        #[arg(long, default_value = ".cache/pages-bundle")]
        bundle_dir: PathBuf,
        #[arg(long, default_value_t = 8787)]
        port: u16,
    },
    /// Build, verify, and deploy the clean static bundle to origin/gh-pages via
    /// a transient JJ workspace.
    Deploy,
}

pub fn run(root: &Path, command: PagesAction) -> Result<(), String> {
    match command {
        PagesAction::Build { bundle_dir } => build(root, &resolve_bundle(root, &bundle_dir)),
        PagesAction::Check {
            bundle_dir,
            security_only,
        } => {
            let dir = resolve_bundle(root, &bundle_dir);
            if security_only {
                pages_verify::check_security(&dir)
            } else {
                pages_verify::verify_bundle(&dir)
            }
        }
        PagesAction::StampSri { bundle_dir } => {
            pages_verify::stamp_sri(&resolve_bundle(root, &bundle_dir))
        }
        PagesAction::ReplAssets {
            wasm_so,
            assets_dirs,
            cache_dir,
            offline,
            zip_only,
            json: emit_json,
        } => {
            let resolved_dirs: Vec<PathBuf> = assets_dirs
                .iter()
                .map(|dir| resolve_bundle(root, dir))
                .collect();
            let resolved_cache = cache_dir.as_deref().map(|p| resolve_bundle(root, p));
            let manifest = pages_repl::stage_repl(
                root,
                &pages_repl::ReplStageConfig {
                    wasm_so: wasm_so.as_deref(),
                    assets_dirs: &resolved_dirs,
                    cache_dir: resolved_cache.as_deref(),
                    offline,
                    zip_only,
                },
            )?;
            if emit_json {
                println!(
                    "{}",
                    json::pretty_string(&manifest).map_err(|e| e.to_string())?
                );
            }
            Ok(())
        }
        PagesAction::Audit {
            bundle_dir,
            no_strict,
            offline,
            graceful_offline,
            ignore_vulns,
            skip_zizmor,
            skip_semgrep,
            skip_pip_audit,
        } => {
            let resolved_bundle = bundle_dir.as_deref().map(|p| resolve_bundle(root, p));
            pages_repl::security_audit(
                root,
                &pages_repl::AuditConfig {
                    bundle_dir: resolved_bundle,
                    strict: !no_strict,
                    offline,
                    graceful_offline,
                    ignore_vulns,
                    skip_zizmor,
                    skip_semgrep,
                    skip_pip_audit,
                },
            )
        }
        PagesAction::Serve { bundle_dir, port } => {
            serve(root, &resolve_bundle(root, &bundle_dir), port)
        }
        PagesAction::Deploy => deploy(root),
    }
}

fn resolve_bundle(root: &Path, bundle_dir: &Path) -> PathBuf {
    if bundle_dir.is_absolute() {
        bundle_dir.to_path_buf()
    } else {
        root.join(bundle_dir)
    }
}

fn augmented_path() -> String {
    let mut parts: Vec<String> = Vec::new();
    if let Ok(home) = std::env::var("HOME") {
        for rel in [".opam/5.2.0+ox/bin", ".opam/forester/bin", ".local/bin"] {
            let p = PathBuf::from(&home).join(rel);
            if p.is_dir() {
                parts.push(p.display().to_string());
            }
        }
    }
    if let Ok(cur) = std::env::var("PATH") {
        parts.push(cur);
    }
    parts.join(":")
}

fn run_checked(cmd: &mut Command, label: &str) -> Result<(), String> {
    let status = cmd
        .env("PATH", augmented_path())
        .status()
        .map_err(|e| format!("{label}: {e}"))?;
    if !status.success() {
        return Err(format!("{label} exited with {status}"));
    }
    Ok(())
}

fn julia_bin() -> String {
    if let Ok(bin) = std::env::var("JULIA_BIN")
        && !bin.trim().is_empty()
    {
        return bin;
    }
    if let Ok(home) = std::env::var("HOME") {
        let up = PathBuf::from(home).join(".juliaup/bin/julia");
        if up.exists() {
            return up.to_string_lossy().into_owned();
        }
    }
    "julia".into()
}

fn daemon_sock(root: &Path) -> PathBuf {
    let tag = &hash_hex(root.as_os_str().as_encoded_bytes())[..10];
    std::env::temp_dir().join(format!("baux_jl_{tag}_fuse.sock"))
}

#[cfg(unix)]
fn daemon_req(sock: &Path, msg: &str) -> Result<String, String> {
    use std::os::unix::net::UnixStream;
    let mut stream =
        UnixStream::connect(sock).map_err(|e| format!("connect {}: {e}", sock.display()))?;
    stream
        .write_all(msg.as_bytes())
        .map_err(|e| e.to_string())?;
    stream.flush().map_err(|e| e.to_string())?;
    let mut reader = std::io::BufReader::new(stream);
    let mut line = String::new();
    reader.read_line(&mut line).map_err(|e| e.to_string())?;
    Ok(line)
}

#[cfg(not(unix))]
fn daemon_req(_sock: &Path, _msg: &str) -> Result<String, String> {
    Err("unix socket required".into())
}

fn spawn_daemon(root: &Path) -> Result<PathBuf, String> {
    let sock = daemon_sock(root);
    let proj = root.join("evals/fuse");
    let log_dir = root.join(".cache/baux/shm");
    fs::create_dir_all(&log_dir).map_err(|e| e.to_string())?;
    let log_path = log_dir.join("julia_fuse_daemon.log");
    let log_file = fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&log_path)
        .map_err(|e| format!("open daemon log {}: {e}", log_path.display()))?;

    let mut cmd = Command::new(julia_bin());
    cmd.args(["--startup-file=no", "--threads=auto", "-O1"]);
    cmd.arg(format!("--project={}", proj.display()))
        .arg(proj.join("worker.jl"))
        .arg("--daemon")
        .arg(&sock)
        .current_dir(root)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::from(log_file));
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        cmd.process_group(0);
    }
    let mut child = cmd
        .spawn()
        .map_err(|e| format!("spawn julia daemon {}: {e}", proj.display()))?;

    let deadline = Instant::now() + std::time::Duration::from_secs(240);
    while Instant::now() < deadline {
        if let Ok(Some(status)) = child.try_wait() {
            let tail = fs::read_to_string(&log_path).unwrap_or_default();
            return Err(format!("julia daemon exited with {status}: {tail}"));
        }
        if let Ok(pong) = daemon_req(&sock, "PING\n")
            && pong.trim() == "PONG"
        {
            return Ok(sock);
        }
        std::thread::sleep(std::time::Duration::from_millis(80));
    }
    Err(format!(
        "timed out waiting for julia daemon {}",
        sock.display()
    ))
}

fn is_fresh(target: &Path, deps: &[PathBuf]) -> bool {
    let Ok(meta) = fs::metadata(target) else {
        return false;
    };
    if meta.len() < 256 {
        return false;
    }
    let Ok(t_mod) = meta.modified() else {
        return false;
    };
    for d in deps {
        if let Ok(dm) = fs::metadata(d).and_then(|m| m.modified())
            && dm > t_mod
        {
            return false;
        }
    }
    true
}

fn export_env(root: &Path) -> Result<String, String> {
    let cache_dir = root.join(".cache/baux");
    fs::create_dir_all(&cache_dir).map_err(|e| e.to_string())?;
    let out_path = cache_dir.join("fuse_env.json");
    let proj = root.join("evals/fuse");
    let deps = [proj.join("worker.jl"), root.join("evals/julia/shm_ipc.jl")];
    if is_fresh(&out_path, &deps) {
        return fs::read_to_string(&out_path).map_err(|e| e.to_string());
    }

    let label = "FUSE.jl 1.2.1";
    println!("o|pages: exporting 3D CAD & PDE environment via {label} daemon...");

    let sock = daemon_sock(root);
    let mut exported = false;
    if let Ok(pong) = daemon_req(&sock, "PING\n")
        && pong.trim() == "PONG"
    {
        let req = format!("ENV\t{}\n", out_path.display());
        if let Ok(rep) = daemon_req(&sock, &req) {
            if rep.trim() == "OK" && out_path.is_file() {
                exported = true;
            } else {
                let _ = daemon_req(&sock, "SHUTDOWN\n");
                std::thread::sleep(std::time::Duration::from_millis(120));
            }
        }
    }

    if !exported {
        let live_sock = spawn_daemon(root)?;
        let req = format!("ENV\t{}\n", out_path.display());
        let rep = daemon_req(&live_sock, &req)?;
        if rep.trim() != "OK" {
            return Err(format!("{label} ENV export failed: {rep}"));
        }
    }

    let raw = fs::read_to_string(&out_path).map_err(|e| e.to_string())?;
    if raw.contains("/Users/") || raw.contains("/home/runner/") {
        return Err(format!("{label} environment JSON leaked host path"));
    }
    Ok(raw)
}

fn env_json(root: &Path) -> Result<String, String> {
    let fuse = export_env(root)?;

    let records = runs::history(
        root,
        &runs::Query {
            limit: 128,
            ..Default::default()
        },
    )
    .unwrap_or_default();

    let mut run_items = Vec::new();
    for r in records {
        if r.task.to_ascii_lowercase().contains("ocean")
            || r.model.to_ascii_lowercase().contains("ocean")
        {
            continue;
        }
        let meas_json =
            baux::artifact::json::to_string(&r.measurements).unwrap_or_else(|_| "{}".to_string());
        if meas_json.to_ascii_lowercase().contains("ocean") {
            continue;
        }
        let meas_clean = if meas_json.contains("/Users/") || meas_json.contains("/home/runner/") {
            "{}".to_string()
        } else {
            meas_json
        };
        run_items.push(format!(
            "{{\"id\":{},\"task\":{},\"model\":{},\"optimizer\":{},\"started\":{},\"status\":{},\"metrics\":{}}}",
            json_str(&r.id),
            json_str(&r.task),
            json_str(&r.model),
            json_str(&r.optimizer),
            r.started,
            json_str(&format!("{:?}", r.status)),
            meas_clean
        ));
        if run_items.len() >= 32 {
            break;
        }
    }

    Ok(format!(
        "{{\"fuse\":{},\"runs\":[{}]}}",
        fuse.trim(),
        run_items.join(",")
    ))
}

const ROOT_FILES: [&str; 11] = [
    "style.css",
    "katex.min.css",
    "forester.js",
    "coi-serviceworker.js",
    "forest.json",
    "favicon.ico",
    "favicon-16x16.png",
    "favicon-32x32.png",
    "apple-touch-icon.png",
    "android-chrome-192x192.png",
    "android-chrome-512x512.png",
];

pub fn build(root: &Path, bundle_dir: &Path) -> Result<(), String> {
    println!("o|pages [1/4]: building OCaml WebGL CAD UI & OxCaml JS bundles (admin/ui)...");
    let ui_dir = root.join("admin/ui");
    let ui_switch = if std::env::var("HOME")
        .map(|h| PathBuf::from(h).join(".opam/5.2.0+ox/bin/dune").is_file())
        .unwrap_or(false)
    {
        "--switch=5.2.0+ox"
    } else {
        "--switch=forester"
    };
    run_checked(
        Command::new("opam")
            .args([
                "exec",
                ui_switch,
                "--",
                "dune",
                "build",
                "./baux_ui.js",
                "./oxcaml_eval.js",
            ])
            .current_dir(&ui_dir),
        "dune build",
    )?;
    let built_js = ui_dir.join("_build/default/baux_ui.js");
    if !built_js.is_file() {
        return Err(format!(
            "missing compiled UI bundle: {}",
            built_js.display()
        ));
    }
    let built_oxcaml_js = ui_dir.join("_build/default/oxcaml_eval.js");
    if !built_oxcaml_js.is_file() {
        return Err(format!(
            "missing compiled OxCaml evaluator bundle: {}",
            built_oxcaml_js.display()
        ));
    }

    println!("o|pages [2/4]: assembling FUSE.jl 3D CAD environments...");
    let cad_env = env_json(root)?;

    println!("o|pages [3/4]: building Forester site...");
    let forest_dir = root.join("forest");
    let theme_dir = forest_dir.join("theme");
    let output_dir = forest_dir.join("output");
    if output_dir.exists() {
        fs::remove_dir_all(&output_dir).map_err(|e| e.to_string())?;
    }
    run_checked(
        Command::new("opam")
            .args([
                "exec",
                "--switch=forester",
                "--",
                "forester",
                "build",
                "forest.toml",
            ])
            .current_dir(&forest_dir),
        "forester build",
    )?;

    for entry in fs::read_dir(&theme_dir).map_err(|e| e.to_string())? {
        let entry = entry.map_err(|e| e.to_string())?;
        let p = entry.path();
        if p.is_file() {
            let ext = p.extension().and_then(|e| e.to_str()).unwrap_or("");
            let fname = entry.file_name();
            let fname_str = fname.to_string_lossy();
            if matches!(ext, "xsl" | "css")
                || (ext == "js"
                    && matches!(fname_str.as_ref(), "forester.js" | "coi-serviceworker.js"))
            {
                copy_rw(&p, &output_dir.join(&fname))?;
            }
        }
    }

    let mut xml_files = Vec::new();
    for entry in fs::read_dir(&output_dir).map_err(|e| e.to_string())? {
        let entry = entry.map_err(|e| e.to_string())?;
        let p = entry.path();
        if p.extension().and_then(|e| e.to_str()) == Some("xml") {
            let fname = entry.file_name().to_string_lossy().to_string();
            if !fname.starts_with("unstable-") {
                xml_files.push(fname);
            }
        }
    }
    xml_files.sort();
    for xml_name in &xml_files {
        let html_name = xml_name.trim_end_matches(".xml").to_string() + ".html";
        run_checked(
            Command::new("xsltproc")
                .args(["--nonet", "-o", &html_name, "default.xsl", xml_name])
                .current_dir(&output_dir),
            "xsltproc",
        )?;
    }

    for (src_name, dst_name) in [
        ("baux-0001.html", "index.html"),
        ("baux-0007.html", "docs.html"),
        ("baux-0004.html", "explorer.html"),
    ] {
        let src_html = output_dir.join(src_name);
        if !src_html.is_file() {
            return Err(format!(
                "expected compiled tree missing: {}",
                src_html.display()
            ));
        }
        copy_rw(&src_html, &output_dir.join(dst_name))?;
    }

    let out_assets = output_dir.join("assets");
    fs::create_dir_all(&out_assets).map_err(|e| e.to_string())?;
    copy_rw(&built_js, &out_assets.join("baux_ui.js"))?;
    copy_rw(&built_oxcaml_js, &out_assets.join("oxcaml_eval.js"))?;
    for asset_name in ["webgpu_bridge.js", "webgpu_worker.js"] {
        let src = theme_dir.join(asset_name);
        if !src.is_file() {
            return Err(format!("missing theme runtime asset: {}", src.display()));
        }
        copy_rw(&src, &out_assets.join(asset_name))?;
    }
    fs::write(out_assets.join("baux_env.json"), &cad_env).map_err(|e| e.to_string())?;

    clean_bundle(bundle_dir)?;
    let bundle_assets = bundle_dir.join("assets");
    fs::create_dir_all(&bundle_assets).map_err(|e| e.to_string())?;

    for entry in fs::read_dir(&output_dir).map_err(|e| e.to_string())? {
        let entry = entry.map_err(|e| e.to_string())?;
        let p = entry.path();
        let fname = entry.file_name().to_string_lossy().to_string();
        if p.is_file() && fname.ends_with(".html") && !fname.starts_with("unstable-") {
            copy_rw(&p, &bundle_dir.join(&fname))?;
        }
    }

    for rel_name in ROOT_FILES {
        let from_out = output_dir.join(rel_name);
        let from_theme = theme_dir.join(rel_name);
        let src = if from_out.is_file() {
            from_out
        } else if from_theme.is_file() {
            from_theme
        } else {
            return Err(format!("missing runtime asset: {rel_name}"));
        };
        copy_rw(&src, &bundle_dir.join(rel_name))?;
    }

    for dir_name in ["fonts", "LICENSES"] {
        let from_out = output_dir.join(dir_name);
        let from_theme = theme_dir.join(dir_name);
        if from_out.is_dir() {
            copy_filtered(&from_out, &bundle_dir.join(dir_name))?;
        } else if from_theme.is_dir() {
            copy_filtered(&from_theme, &bundle_dir.join(dir_name))?;
        }
    }

    copy_rw(&built_js, &bundle_assets.join("baux_ui.js"))?;
    copy_rw(&built_oxcaml_js, &bundle_assets.join("oxcaml_eval.js"))?;
    for asset_name in ["webgpu_bridge.js", "webgpu_worker.js"] {
        copy_rw(&theme_dir.join(asset_name), &bundle_assets.join(asset_name))?;
    }
    fs::write(bundle_assets.join("baux_env.json"), &cad_env).map_err(|e| e.to_string())?;
    fs::write(bundle_dir.join(".nojekyll"), b"").map_err(|e| e.to_string())?;

    println!("o|pages [3.5/4]: building WASM extension & staging Pyodide assets...");
    let buck2_out = Command::new(root.join("admin/tools/buck2w"))
        .args(["build", "//:baux-py-wasm", "--show-output"])
        .env("PATH", augmented_path())
        .current_dir(root)
        .stderr(Stdio::inherit())
        .output()
        .map_err(|e| format!("buck2w build //:baux-py-wasm: {e}"))?;
    if !buck2_out.status.success() {
        return Err(format!(
            "buck2w build //:baux-py-wasm exited with {}",
            buck2_out.status
        ));
    }
    let buck2_stdout = String::from_utf8(buck2_out.stdout)
        .map_err(|e| format!("buck2w output was not UTF-8: {e}"))?;
    let wasm_rel = buck2_stdout
        .lines()
        .find_map(|line| {
            let trimmed = line.trim();
            trimmed
                .strip_prefix("root//:baux-py-wasm ")
                .or_else(|| trimmed.strip_prefix("//:baux-py-wasm "))
        })
        .map(str::trim)
        .map(PathBuf::from)
        .ok_or_else(|| {
            format!("buck2w did not report //:baux-py-wasm output path: {buck2_stdout}")
        })?;
    let wasm_so = if wasm_rel.is_absolute() {
        wasm_rel
    } else {
        root.join(wasm_rel)
    };
    if !wasm_so.is_file() {
        return Err(format!(
            "missing compiled wasm extension: {}",
            wasm_so.display()
        ));
    }

    let stage_dirs = [out_assets.clone(), bundle_assets.clone()];
    pages_repl::stage_repl(
        root,
        &pages_repl::ReplStageConfig {
            wasm_so: Some(&wasm_so),
            assets_dirs: &stage_dirs,
            cache_dir: None,
            offline: false,
            zip_only: false,
        },
    )?;

    println!("o|pages [4/4]: stamping SRI hashes and running security check...");
    pages_verify::stamp_sri(bundle_dir)?;
    pages_verify::stamp_assets(&output_dir, bundle_dir)?;
    pages_verify::verify_bundle(bundle_dir)?;
    println!("o|pages: built & verified {}", bundle_dir.display());
    Ok(())
}

fn copy_rw(src: &Path, dst: &Path) -> Result<(), String> {
    let _ = fs::remove_file(dst);
    fs::copy(src, dst).map_err(|e| format!("copy {} -> {}: {e}", src.display(), dst.display()))?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let _ = fs::set_permissions(dst, fs::Permissions::from_mode(0o644));
    }
    Ok(())
}

fn clean_bundle(bundle_dir: &Path) -> Result<(), String> {
    if !bundle_dir.exists() {
        fs::create_dir_all(bundle_dir).map_err(|e| e.to_string())?;
        return Ok(());
    }
    for entry in fs::read_dir(bundle_dir).map_err(|e| e.to_string())? {
        let entry = entry.map_err(|e| e.to_string())?;
        let name = entry.file_name();
        if name == ".jj" || name == ".git" {
            continue;
        }
        let p = entry.path();
        if p.is_dir() {
            fs::remove_dir_all(&p).map_err(|e| e.to_string())?;
        } else {
            fs::remove_file(&p).map_err(|e| e.to_string())?;
        }
    }
    Ok(())
}

fn copy_filtered(src: &Path, dst: &Path) -> Result<(), String> {
    fs::create_dir_all(dst).map_err(|e| e.to_string())?;
    for entry in fs::read_dir(src).map_err(|e| e.to_string())? {
        let entry = entry.map_err(|e| e.to_string())?;
        let path = entry.path();
        let name = entry.file_name();
        let name_str = name.to_string_lossy();
        if name_str.starts_with('.') || name_str == "forest" || name_str == "node_modules" {
            continue;
        }
        let target = dst.join(&name);
        if path.is_dir() {
            copy_filtered(&path, &target)?;
        } else {
            let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
            if matches!(
                ext,
                "xml" | "xsl" | "map" | "env" | "sh" | "ml" | "mli" | "toml"
            ) || name_str == "site_data.js"
            {
                continue;
            }
            let _ = fs::remove_file(&target);
            fs::copy(&path, &target).map_err(|e| e.to_string())?;
            #[cfg(unix)]
            {
                use std::os::unix::fs::PermissionsExt;
                let _ = fs::set_permissions(&target, fs::Permissions::from_mode(0o644));
            }
        }
    }
    Ok(())
}

fn json_str(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for ch in s.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push(' '),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

const MAX_HDRBYTES: usize = 64 * 1024;
const MAX_BODYBYTES: usize = 2 * 1024 * 1024;

#[cfg(unix)]
const SIGINT_NUM: i32 = 2;
#[cfg(unix)]
const SIGKILL_NUM: i32 = 9;

#[cfg(unix)]
fn send_signal(pid: u32, sig: i32) -> bool {
    unsafe extern "C" {
        fn kill(pid: i32, sig: i32) -> i32;
    }
    if pid == 0 || pid > i32::MAX as u32 {
        return false;
    }
    unsafe { kill(pid as i32, sig) == 0 }
}

#[cfg(not(unix))]
fn send_signal(_pid: u32, _sig: i32) -> bool {
    false
}

struct NativeKernelProc {
    child: Child,
    stdin: BufWriter<ChildStdin>,
    stdout: BufReader<ChildStdout>,
    python_label: String,
}

struct NativeKernelHub {
    io: Mutex<Option<NativeKernelProc>>,
    active_pid: AtomicU32,
    cached_meta: Mutex<Option<(String, Value)>>,
}

static KERNEL_HUB: OnceLock<NativeKernelHub> = OnceLock::new();

fn kernel_hub() -> &'static NativeKernelHub {
    KERNEL_HUB.get_or_init(|| NativeKernelHub {
        io: Mutex::new(None),
        active_pid: AtomicU32::new(0),
        cached_meta: Mutex::new(None),
    })
}

const KERNEL_WORKER: &str = r####"
import ast
import code
import importlib.util
import io
import json
import math
import os
import rlcompleter
import sys
import time
import traceback

_proto_fd = os.dup(1)
os.dup2(2, 1)
_PROTO_OUT = os.fdopen(_proto_fd, "w", encoding="utf-8", buffering=1)

try:
    import numpy as np
except Exception:
    np = None

try:
    import baux
except Exception:
    baux = None


def _make_ns():
    ns = {"__name__": "__main__"}
    if np is not None:
        ns["np"] = np
        ns["numpy"] = np
    if baux is not None:
        ns["baux"] = baux
    return ns


_REPL_NS = _make_ns()
_CONSOLE = code.InteractiveConsole(locals=_REPL_NS)


def _reset_namespace():
    global _REPL_NS, _CONSOLE
    _REPL_NS = _make_ns()
    _CONSOLE = code.InteractiveConsole(locals=_REPL_NS)


def _probe_capabilities():
    native_baux = baux is not None and importlib.util.find_spec("baux.baux_rust") is not None
    caps = {
        "native_baux": bool(native_baux),
        "cpu": False,
        "metal": False,
        "cuda": False,
        "opencl": False,
        "metal_compiled": False,
        "cuda_compiled": False,
        "torch": importlib.util.find_spec("torch") is not None,
        "jax": importlib.util.find_spec("jax") is not None,
        "flame": importlib.util.find_spec("flame") is not None,
        "baux_tune": os.environ.get("BAUX_HAS_TUNE") == "1",
    }
    if baux is not None and hasattr(baux, "capabilities"):
        try:
            try:
                rows = baux.capabilities(probe=True)
            except Exception:
                rows = baux.capabilities(probe=False)
            for row in rows:
                name = getattr(row, "name", "")
                compiled = bool(getattr(row, "compiled", False))
                avail = getattr(row, "available", None)
                is_active = bool(avail) if avail is not None else compiled
                if name == "cpu":
                    caps["cpu"] = is_active
                elif name == "metal":
                    caps["metal_compiled"] = compiled
                    caps["metal"] = bool(compiled and is_active)
                elif name == "cuda":
                    caps["cuda_compiled"] = compiled
                    caps["cuda"] = bool(compiled and is_active)
                elif name == "opencl":
                    caps["opencl"] = bool(compiled and is_active)
        except Exception:
            pass
    return caps


_CAPS_CACHE = _probe_capabilities()


def _json_scalar(val):
    if isinstance(val, (bool, int)):
        return int(val) if not isinstance(val, bool) else val
    try:
        f = float(val)
        return f if math.isfinite(f) else None
    except Exception:
        return None


def _clean_float_list(seq, limit=128):
    if np is None:
        return []
    out = []
    try:
        arr = np.asarray(seq, dtype=float).ravel().tolist()
    except Exception:
        return out
    for v in arr[:limit]:
        fv = _json_scalar(v)
        if fv is not None:
            out.append(round(fv, 6))
    return out


def _clean_pairs(seq, limit=128):
    if np is None:
        return []
    out = []
    try:
        arr = np.asarray(seq, dtype=float)
        if arr.ndim != 2 or arr.shape[1] < 2:
            return out
        for row in arr[:limit]:
            rx = _json_scalar(row[0])
            ry = _json_scalar(row[1])
            if rx is not None and ry is not None:
                out.append([round(rx, 5), round(ry, 5)])
    except Exception:
        return out
    return out


def _sanitize_curves(raw):
    if not isinstance(raw, dict):
        return None
    raw_series = raw.get("series")
    if not isinstance(raw_series, (list, tuple)) or not raw_series:
        return None
    series_out = []
    for item in raw_series[:6]:
        if not isinstance(item, dict):
            continue
        ys = _clean_float_list(item.get("y", []), limit=160)
        if not ys:
            continue
        xs = _clean_float_list(item.get("x", []), limit=160)
        if len(xs) != len(ys):
            xs = list(range(1, len(ys) + 1))
        series_out.append({
            "label": str(item.get("label", f"series {len(series_out) + 1}"))[:40],
            "x": xs,
            "y": ys,
            "dashed": bool(item.get("dashed", False)),
        })
    if not series_out:
        return None
    return {
        "kind": "curves",
        "title": str(raw.get("title", "Comparison"))[:80],
        "subtitle": str(raw.get("subtitle", ""))[:120],
        "x_label": str(raw.get("x_label", "Evaluations"))[:40],
        "y_label": str(raw.get("y_label", "Objective"))[:40],
        "log_y": bool(raw.get("log_y", False)),
        "series": series_out,
    }


def _extract_fig(raw):
    if not isinstance(raw, dict):
        return None
    kind = str(raw.get("kind", ""))
    if kind == "curves":
        return _sanitize_curves(raw)

    if kind == "tokamak":
        fg = raw.get("flux_grid") or {}
        nr = int(fg.get("nr", 0)) if isinstance(fg, dict) else 0
        nz = int(fg.get("nz", 0)) if isinstance(fg, dict) else 0
        if nr <= 1 or nz <= 1 or nr * nz > 2500:
            return None
        coils_out = []
        for c in (raw.get("coils") or [])[:40]:
            try:
                coils_out.append([round(float(x), 4) for x in c[:5]])
            except Exception:
                pass
        curves_sub = _sanitize_curves(raw.get("curves")) if isinstance(raw.get("curves"), dict) else None
        out = {
            "kind": "tokamak",
            "title": str(raw.get("title", "Tokamak Equilibrium"))[:80],
            "subtitle": str(raw.get("subtitle", ""))[:120],
            "r_range": _clean_float_list(raw.get("r_range", [0.4, 3.0]), limit=2),
            "z_range": _clean_float_list(raw.get("z_range", [-1.6, 1.6]), limit=2),
            "first_wall": _clean_pairs(raw.get("first_wall", []), limit=128),
            "lcfs": _clean_pairs(raw.get("lcfs", []), limit=128),
            "x_point": _clean_float_list(raw.get("x_point", []), limit=2),
            "coils": coils_out,
            "flux_grid": {
                "nr": nr,
                "nz": nz,
                "psi_init": _clean_float_list(fg.get("psi_init", []), limit=nr * nz),
                "psi_opt": _clean_float_list(fg.get("psi_opt", []), limit=nr * nz),
                "psi_init_target": _json_scalar(fg.get("psi_init_target")),
                "psi_opt_target": _json_scalar(fg.get("psi_opt_target")),
            },
        }
        if curves_sub is not None:
            out["curves"] = curves_sub
        return out

    if kind == "field2d":
        panels_out = []
        for p in (raw.get("panels") or [])[:2]:
            if not isinstance(p, dict):
                continue
            nx = int(p.get("nx", 0))
            nz = int(p.get("nz", 0))
            vals = _clean_float_list(p.get("vals", []), limit=1024)
            if nx >= 2 and nz >= 2 and len(vals) == nx * nz:
                panels_out.append({
                    "label": str(p.get("label", "field"))[:40],
                    "nx": nx,
                    "nz": nz,
                    "vals": vals,
                    "vmin": _json_scalar(p.get("vmin")),
                    "vmax": _json_scalar(p.get("vmax")),
                })
        if not panels_out:
            return None
        out = {
            "kind": "field2d",
            "title": str(raw.get("title", "2D Field"))[:80],
            "subtitle": str(raw.get("subtitle", ""))[:120],
            "panels": panels_out,
        }
        curves_sub = _sanitize_curves(raw.get("curves")) if isinstance(raw.get("curves"), dict) else None
        if curves_sub is not None:
            out["curves"] = curves_sub
        return out

    if kind == "band1d":
        xg = _clean_float_list(raw.get("x_grid", []), limit=160)
        mu = _clean_float_list(raw.get("mu", []), limit=160)
        if not xg or len(xg) != len(mu):
            return None
        out = {
            "kind": "band1d",
            "title": str(raw.get("title", "Surrogate Fit & Uncertainty"))[:80],
            "subtitle": str(raw.get("subtitle", ""))[:120],
            "x_grid": xg,
            "y_true": _clean_float_list(raw.get("y_true", []), limit=160),
            "mu": mu,
            "lo_2se": _clean_float_list(raw.get("lo_2se", []), limit=160),
            "hi_2se": _clean_float_list(raw.get("hi_2se", []), limit=160),
            "mw_lo": _clean_float_list(raw.get("mw_lo", []), limit=160),
            "mw_hi": _clean_float_list(raw.get("mw_hi", []), limit=160),
            "train_x": _clean_float_list(raw.get("train_x", []), limit=128),
            "train_y": _clean_float_list(raw.get("train_y", []), limit=128),
        }
        curves_sub = _sanitize_curves(raw.get("curves")) if isinstance(raw.get("curves"), dict) else None
        if curves_sub is not None:
            out["curves"] = curves_sub
        return out

    return None


def _summarize_state(ns):
    opt = None
    opt_name = None
    all_opts = []
    seen_ids = set()
    OptimizerState = getattr(baux, "OptimizerState", None) if baux is not None else None
    if OptimizerState is not None:
        for k, v in ns.items():
            if k.startswith("_") or not isinstance(v, OptimizerState):
                continue
            if int(getattr(v, "num_observations", 0)) <= 0 and k != "state":
                continue
            oid = id(v)
            if oid in seen_ids:
                continue
            seen_ids.add(oid)
            all_opts.append((k, v))
        if isinstance(ns.get("state"), OptimizerState):
            opt = ns["state"]
            opt_name = "state"
        elif all_opts:
            opt_name, opt = all_opts[-1]
        else:
            for k in reversed(list(ns.keys())):
                if not k.startswith("_") and isinstance(ns[k], OptimizerState):
                    opt = ns[k]
                    opt_name = k
                    break

    fig_out = _extract_fig(ns.get("fig"))
    if opt is None and fig_out is None:
        return None

    summary = {"fig": fig_out}
    if opt is not None and np is not None:
        y_raw = getattr(opt, "y_obs", None)
        x_raw = getattr(opt, "x_obs", None)
        y_list = []
        best_trace = []
        recent_rows = []
        if y_raw is not None:
            ya = np.asarray(y_raw, dtype=float)
            if ya.ndim == 1:
                ya = ya[:, None]
            if ya.shape[0] > 0:
                prim = ya[:, 0] if ya.shape[1] == 1 else np.sum(ya, axis=1)
                running = -math.inf
                for v in prim.tolist():
                    fv = float(v)
                    if fv > running:
                        running = fv
                    y_list.append(fv if math.isfinite(fv) else None)
                    best_trace.append(running if math.isfinite(running) else None)
                if x_raw is not None:
                    xa = np.asarray(x_raw, dtype=float)
                    start = max(0, xa.shape[0] - 8)
                    for idx in range(start, xa.shape[0]):
                        row_x = [round(float(z), 4) for z in xa[idx, :6].tolist()]
                        recent_rows.append({
                            "obs": idx + 1,
                            "y": round(float(prim[idx]), 6),
                            "x": row_x,
                            "best_so_far": round(float(best_trace[idx]), 6) if best_trace[idx] is not None else None,
                        })

        best_x_val = getattr(opt, "best_x", None)
        best_x_list = None
        if best_x_val is not None:
            arr = np.asarray(best_x_val, dtype=float).ravel()
            best_x_list = [round(float(z), 5) for z in arr[:8].tolist()]

        best_params_repr = None
        bp = getattr(opt, "best_params", None)
        if bp is not None:
            try:
                best_params_repr = repr(bp)
                if len(best_params_repr) > 180:
                    best_params_repr = best_params_repr[:177] + "..."
            except Exception:
                best_params_repr = None

        summary.update({
            "var": opt_name,
            "dim": int(opt.dim),
            "epoch": int(opt.epoch),
            "num_observations": int(opt.num_observations),
            "num_regions": int(opt.num_regions),
            "tr_length": _json_scalar(opt.tr_length),
            "best_y": _json_scalar(opt.best_y),
            "best_x": best_x_list,
            "best_params": best_params_repr,
            "pending_count": 0 if opt.pending_ids is None else len(opt.pending_ids),
            "y_history": y_list[-128:],
            "best_history": best_trace[-128:],
            "recent": recent_rows,
        })

    if fig_out is None and np is not None and len(all_opts) >= 2:
        auto_series = []
        for name, item in all_opts[:5]:
            ya = np.asarray(getattr(item, "y_obs", None), dtype=float)
            if ya.ndim == 1:
                ya = ya[:, None]
            if ya.shape[0] <= 0:
                continue
            prim = ya[:, 0] if ya.shape[1] == 1 else np.sum(ya, axis=1)
            running = -math.inf
            bt = []
            for v in prim.tolist():
                fv = float(v)
                if fv > running:
                    running = fv
                if math.isfinite(running):
                    bt.append(round(running, 6))
            if bt:
                auto_series.append({
                    "label": str(name)[:40],
                    "x": list(range(1, len(bt) + 1))[-128:],
                    "y": bt[-128:],
                    "dashed": False,
                })
        if len(auto_series) >= 2:
            summary["fig"] = {
                "kind": "curves",
                "title": "Cumulative Best Objective Comparison",
                "subtitle": "Auto-detected OptimizerState trajectories in workspace",
                "x_label": "Evaluations",
                "y_label": "Best Objective",
                "log_y": False,
                "series": auto_series,
            }

    return summary


def _baux_exec_script(src, reset_ns=False):
    if reset_ns:
        _reset_namespace()
    out_buf = io.StringIO()
    err_buf = io.StringIO()
    old_out, old_err = sys.stdout, sys.stderr
    sys.stdout = out_buf
    sys.stderr = err_buf
    t0 = time.perf_counter()
    ok = True
    try:
        tree = ast.parse(src, filename="<local_native>", mode="exec")
        if tree.body and isinstance(tree.body[-1], ast.Expr):
            last_expr = ast.Expression(body=tree.body.pop().value)
            if tree.body:
                exec(compile(tree, "<local_native>", "exec"), _REPL_NS)
            val = eval(compile(last_expr, "<local_native>", "eval"), _REPL_NS)
            if val is not None:
                print(repr(val))
        else:
            exec(compile(tree, "<local_native>", "exec"), _REPL_NS)
    except BaseException:
        ok = False
        traceback.print_exc(file=err_buf)
    finally:
        dt_ms = (time.perf_counter() - t0) * 1000.0
        sys.stdout = old_out
        sys.stderr = old_err

    return {
        "ok": ok,
        "stdout": out_buf.getvalue(),
        "stderr": err_buf.getvalue(),
        "incomplete": False,
        "elapsed_ms": round(dt_ms, 2),
        "state": _summarize_state(_REPL_NS),
        "mode": "local_native",
    }


def _baux_repl_push(line):
    out_buf = io.StringIO()
    err_buf = io.StringIO()
    old_out, old_err = sys.stdout, sys.stderr
    sys.stdout = out_buf
    sys.stderr = err_buf
    t0 = time.perf_counter()
    ok = True
    incomplete = False
    try:
        incomplete = bool(_CONSOLE.push(line))
    except BaseException:
        ok = False
        traceback.print_exc(file=err_buf)
    finally:
        dt_ms = (time.perf_counter() - t0) * 1000.0
        sys.stdout = old_out
        sys.stderr = old_err

    err_txt = err_buf.getvalue()
    if err_txt and not incomplete:
        ok = False
    return {
        "ok": ok,
        "stdout": out_buf.getvalue(),
        "stderr": err_txt,
        "incomplete": incomplete,
        "elapsed_ms": round(dt_ms, 2),
        "state": _summarize_state(_REPL_NS),
        "mode": "local_native",
    }


def _baux_complete(prefix):
    comp = rlcompleter.Completer(_REPL_NS)
    matches = []
    seen = set()
    i = 0
    while i < 64:
        m = comp.complete(prefix, i)
        if m is None:
            break
        if m not in seen:
            seen.add(m)
            matches.append(m)
        i += 1
    return matches


def _emit(payload):
    _PROTO_OUT.write(json.dumps(payload) + "\n")
    _PROTO_OUT.flush()


while True:
    try:
        raw = sys.stdin.readline()
    except KeyboardInterrupt:
        continue
    if not raw:
        break
    raw = raw.strip()
    if not raw:
        continue
    try:
        req = json.loads(raw)
        op = req.get("op", "")
        if op == "status":
            _emit({
                "ok": True,
                "alive": True,
                "pid": os.getpid(),
                "capabilities": _CAPS_CACHE,
            })
        elif op == "exec":
            code_src = str(req.get("code", ""))
            reset_ns = bool(req.get("resetNamespace", False))
            _emit(_baux_exec_script(code_src, reset_ns))
        elif op == "repl":
            line_src = str(req.get("code", ""))
            _emit(_baux_repl_push(line_src))
        elif op == "complete":
            prefix = str(req.get("prefix", ""))
            _emit({"ok": True, "matches": _baux_complete(prefix)})
        elif op == "reset":
            _reset_namespace()
            _emit({"ok": True, "state": None, "mode": "local_native"})
        else:
            _emit({"ok": False, "error": f"unknown kernel op: {op}"})
    except KeyboardInterrupt:
        _emit({
            "ok": False,
            "stdout": "",
            "stderr": "KeyboardInterrupt",
            "incomplete": False,
            "elapsed_ms": 0.0,
            "state": _summarize_state(_REPL_NS),
            "mode": "local_native",
        })
    except Exception:
        _emit({
            "ok": False,
            "stdout": "",
            "stderr": traceback.format_exc(),
            "incomplete": False,
            "elapsed_ms": 0.0,
            "state": None,
            "mode": "local_native",
        })
"####;

fn native_python(root: &Path) -> (PathBuf, String, Vec<PathBuf>) {
    let candidates = [
        (
            root.join(".pixi/envs/baux/bin/python"),
            ".pixi/envs/baux/bin/python",
        ),
        (root.join(".venv/bin/python"), ".venv/bin/python"),
    ];
    let mut exe = PathBuf::from("python3");
    let mut label = "python3".to_string();
    for (cand, cand_lbl) in candidates {
        if cand.is_file() {
            exe = cand;
            label = cand_lbl.to_string();
            break;
        }
    }

    let mut py_paths = Vec::new();
    let wheel_site = root.join(".cache/baux/wheel-site");
    if wheel_site.join("baux").is_dir() {
        py_paths.push(wheel_site);
    }
    let py_src = root.join("python");
    if py_src.is_dir() {
        py_paths.push(py_src);
    }
    (exe, label, py_paths)
}

fn spawn_kernel(root: &Path) -> Result<NativeKernelProc, String> {
    let (exe, python_label, py_paths) = native_python(root);
    let mut cmd = Command::new(&exe);
    cmd.args(["-u", "-c", KERNEL_WORKER])
        .current_dir(root)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .env(
            "BAUX_HAS_TUNE",
            if root.join("baux").is_file() {
                "1"
            } else {
                "0"
            },
        );

    let mut full_py_paths = py_paths;
    if let Some(existing) = std::env::var_os("PYTHONPATH") {
        full_py_paths.extend(std::env::split_paths(&existing));
    }
    if let Ok(joined) = std::env::join_paths(&full_py_paths) {
        cmd.env("PYTHONPATH", joined);
    }

    let mut child = cmd
        .spawn()
        .map_err(|e| format!("spawn {python_label}: {e}"))?;
    let stdin = child
        .stdin
        .take()
        .ok_or_else(|| "native kernel missing stdin pipe".to_string())?;
    let stdout = child
        .stdout
        .take()
        .ok_or_else(|| "native kernel missing stdout pipe".to_string())?;
    Ok(NativeKernelProc {
        child,
        stdin: BufWriter::new(stdin),
        stdout: BufReader::new(stdout),
        python_label,
    })
}

fn kernel_roundtrip(root: &Path, req: &Value) -> Result<(String, Value), String> {
    let hub = kernel_hub();
    let mut guard = hub
        .io
        .lock()
        .map_err(|_| "native kernel lock poisoned".to_string())?;

    if let Some(proc_ref) = guard.as_mut() {
        let dead = matches!(proc_ref.child.try_wait(), Ok(Some(_)) | Err(_));
        if dead {
            hub.active_pid.store(0, Ordering::Release);
            *guard = None;
        }
    }

    if guard.is_none() {
        let spawned = spawn_kernel(root)?;
        hub.active_pid.store(spawned.child.id(), Ordering::Release);
        *guard = Some(spawned);
    }

    let line = json::to_string(req).map_err(|e| e.to_string())?;
    let mut resp_line = String::new();
    let io_result: Result<Option<String>, String> = {
        let proc_ref = guard
            .as_mut()
            .ok_or_else(|| "failed to acquire native kernel".to_string())?;
        if let Err(e) = proc_ref
            .stdin
            .write_all(line.as_bytes())
            .and_then(|_| proc_ref.stdin.write_all(b"\n"))
            .and_then(|_| proc_ref.stdin.flush())
        {
            Err(format!("kernel write error: {e}"))
        } else {
            match proc_ref.stdout.read_line(&mut resp_line) {
                Err(e) => Err(format!("kernel read error: {e}")),
                Ok(0) => {
                    let _ = proc_ref.child.wait();
                    Ok(None)
                }
                Ok(_) => Ok(Some(proc_ref.python_label.clone())),
            }
        }
    };
    let py_label = match io_result {
        Ok(Some(label)) => label,
        Ok(None) => {
            hub.active_pid.store(0, Ordering::Release);
            *guard = None;
            return Err("native kernel exited cleanly or was force-terminated".to_string());
        }
        Err(msg) => {
            hub.active_pid.store(0, Ordering::Release);
            *guard = None;
            return Err(msg);
        }
    };
    let parsed: Value = json::from_str(resp_line.trim())
        .map_err(|e| format!("invalid JSON from native kernel: {e}"))?;
    Ok((py_label, parsed))
}

fn kernel_status(root: &Path) -> String {
    let hub = kernel_hub();
    // Do not block status polling behind a long-running exec/repl invocation.
    match hub.io.try_lock() {
        Ok(guard) => {
            drop(guard);
            match kernel_roundtrip(root, &json!({ "op": "status" })) {
                Ok((py_label, val)) => {
                    let caps = val
                        .get("capabilities")
                        .cloned()
                        .unwrap_or_else(|| Value::from(Map::new()));
                    if let Ok(mut meta) = hub.cached_meta.lock() {
                        *meta = Some((py_label.clone(), caps.clone()));
                    }
                    let pid = hub.active_pid.load(Ordering::Acquire);
                    json::to_string(&json!({
                        "ok": true,
                        "available": true,
                        "mode": "local_native",
                        "alive": pid != 0,
                        "running": false,
                        "pid": pid,
                        "python": py_label,
                        "capabilities": caps,
                    }))
                    .unwrap_or_else(|_| "{}".to_string())
                }
                Err(err) => json::to_string(&json!({
                    "ok": false,
                    "available": false,
                    "mode": "local_native",
                    "alive": false,
                    "running": false,
                    "pid": 0,
                    "error": err,
                }))
                .unwrap_or_else(|_| "{\"ok\":false,\"available\":false}".to_string()),
            }
        }
        Err(_) => {
            let pid = hub.active_pid.load(Ordering::Acquire);
            let (py_label, caps) = hub
                .cached_meta
                .lock()
                .ok()
                .and_then(|m| m.clone())
                .unwrap_or_else(|| ("python3".to_string(), Value::from(Map::new())));
            json::to_string(&json!({
                "ok": true,
                "available": true,
                "mode": "local_native",
                "alive": pid != 0,
                "running": true,
                "pid": pid,
                "python": py_label,
                "capabilities": caps,
            }))
            .unwrap_or_else(|_| "{}".to_string())
        }
    }
}

fn kernel_interrupt(force: bool) -> String {
    let hub = kernel_hub();
    let pid = hub.active_pid.load(Ordering::Acquire);
    let sig = if force {
        #[cfg(unix)]
        {
            SIGKILL_NUM
        }
        #[cfg(not(unix))]
        {
            0
        }
    } else {
        #[cfg(unix)]
        {
            SIGINT_NUM
        }
        #[cfg(not(unix))]
        {
            0
        }
    };
    let signaled = if pid != 0 {
        send_signal(pid, sig)
    } else {
        false
    };
    if force && signaled {
        hub.active_pid.store(0, Ordering::Release);
    }
    json::to_string(&json!({
        "ok": true,
        "signaled": signaled,
        "force": force,
        "pid": pid,
        "mode": "local_native",
    }))
    .unwrap_or_else(|_| "{\"ok\":true}".to_string())
}

fn kernel_reset(root: &Path, restart: bool) -> String {
    if restart {
        let hub = kernel_hub();
        let pid = hub.active_pid.load(Ordering::Acquire);
        if pid != 0 {
            #[cfg(unix)]
            let _ = send_signal(pid, SIGKILL_NUM);
            hub.active_pid.store(0, Ordering::Release);
        }
        if let Ok(mut guard) = hub.io.lock() {
            if let Some(mut proc_ref) = guard.take() {
                let _ = proc_ref.child.kill();
                let _ = proc_ref.child.wait();
            }
        }
        return match kernel_roundtrip(root, &json!({ "op": "reset" })) {
            Ok((_, val)) => json::to_string(&val).unwrap_or_else(|_| "{\"ok\":true}".to_string()),
            Err(err) => json::to_string(&json!({
                "ok": false,
                "stderr": err,
                "mode": "local_native",
            }))
            .unwrap_or_else(|_| "{\"ok\":false}".to_string()),
        };
    }

    match kernel_roundtrip(root, &json!({ "op": "reset" })) {
        Ok((_, val)) => json::to_string(&val).unwrap_or_else(|_| "{\"ok\":true}".to_string()),
        Err(err) => json::to_string(&json!({
            "ok": false,
            "stderr": err,
            "mode": "local_native",
        }))
        .unwrap_or_else(|_| "{\"ok\":false}".to_string()),
    }
}

fn serve(root: &Path, bundle_dir: &Path, port: u16) -> Result<(), String> {
    let addr = format!("127.0.0.1:{port}");
    let listener = TcpListener::bind(&addr).map_err(|e| format!("bind {addr}: {e}"))?;
    println!(
        "o|pages: serving {} at http://{addr}/explorer.html",
        bundle_dir.display()
    );
    for stream in listener.incoming() {
        if let Ok(mut s) = stream {
            let root_own = root.to_path_buf();
            let bundle_own = bundle_dir.to_path_buf();
            std::thread::spawn(move || {
                let _ = s.set_read_timeout(Some(std::time::Duration::from_secs(5)));
                let _ = handle_http(&root_own, &bundle_own, &mut s);
            });
        }
    }
    Ok(())
}

struct ParsedHttpRequest {
    method: String,
    path_only: String,
    origin: Option<String>,
    body: Vec<u8>,
}

enum ReadHttpResult {
    Eof,
    Req(ParsedHttpRequest),
    Reject(&'static str, &'static str),
}

fn header_end(buf: &[u8]) -> Option<(usize, usize)> {
    for i in 0..buf.len() {
        if i + 4 <= buf.len() && &buf[i..i + 4] == b"\r\n\r\n" {
            return Some((i, 4));
        }
        if i + 2 <= buf.len() && &buf[i..i + 2] == b"\n\n" {
            return Some((i, 2));
        }
    }
    None
}

fn read_http(stream: &mut TcpStream) -> Result<ReadHttpResult, String> {
    let mut raw = Vec::with_capacity(4096);
    let mut chunk = [0u8; 4096];
    let (hdr_end, sep_len) = loop {
        if let Some(found) = header_end(&raw) {
            break found;
        }
        if raw.len() >= MAX_HDRBYTES {
            return Ok(ReadHttpResult::Reject(
                "431 Request Header Fields Too Large",
                "headers too large",
            ));
        }
        match stream.read(&mut chunk) {
            Ok(0) => {
                if raw.is_empty() {
                    return Ok(ReadHttpResult::Eof);
                }
                return Ok(ReadHttpResult::Reject(
                    "400 Bad Request",
                    "malformed HTTP request",
                ));
            }
            Ok(n) => raw.extend_from_slice(&chunk[..n]),
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
            Err(e) => return Err(e.to_string()),
        }
    };

    let hdr_text = String::from_utf8_lossy(&raw[..hdr_end]);
    let mut lines = hdr_text.lines();
    let first_line = lines.next().unwrap_or("");
    let mut parts = first_line.split_whitespace();
    let method = parts.next().unwrap_or("GET").to_ascii_uppercase();
    let raw_path = parts.next().unwrap_or("/");
    let path_only = raw_path.split('?').next().unwrap_or("/").to_string();

    let mut content_length: usize = 0;
    let mut origin: Option<String> = None;
    for line in lines {
        if let Some((name, val)) = line.split_once(':') {
            let name_trim = name.trim();
            if name_trim.eq_ignore_ascii_case("content-length") {
                match val.trim().parse::<usize>() {
                    Ok(len) => content_length = len,
                    Err(_) => {
                        return Ok(ReadHttpResult::Reject(
                            "400 Bad Request",
                            "invalid content-length",
                        ));
                    }
                }
            } else if name_trim.eq_ignore_ascii_case("origin") {
                origin = Some(val.trim().to_string());
            }
        }
    }

    if content_length > MAX_BODYBYTES {
        return Ok(ReadHttpResult::Reject(
            "413 Payload Too Large",
            "request body exceeds 2 MiB limit",
        ));
    }

    let body_start = hdr_end + sep_len;
    let mut body = Vec::with_capacity(content_length);
    if raw.len() > body_start {
        body.extend_from_slice(&raw[body_start..]);
    }
    while body.len() < content_length {
        match stream.read(&mut chunk) {
            Ok(0) => {
                return Ok(ReadHttpResult::Reject(
                    "400 Bad Request",
                    "unexpected EOF in body",
                ));
            }
            Ok(n) => body.extend_from_slice(&chunk[..n]),
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
            Err(e) => return Err(e.to_string()),
        }
    }
    body.truncate(content_length);

    Ok(ReadHttpResult::Req(ParsedHttpRequest {
        method,
        path_only,
        origin,
        body,
    }))
}

fn allowed_origin(origin: &str) -> bool {
    let trimmed = origin.trim();
    let Some(host_port) = trimmed
        .strip_prefix("http://")
        .or_else(|| trimmed.strip_prefix("https://"))
    else {
        return false;
    };
    host_port == "127.0.0.1"
        || host_port.starts_with("127.0.0.1:")
        || host_port == "localhost"
        || host_port.starts_with("localhost:")
}

fn kernel_api(root: &Path, stream: &mut TcpStream, req: &ParsedHttpRequest) -> Result<(), String> {
    if let Some(orig) = req.origin.as_deref()
        && !allowed_origin(orig)
    {
        return write_api(
            stream,
            "403 Forbidden",
            r#"{"ok":false,"error":"cross-origin access forbidden"}"#.as_bytes(),
        );
    }

    let parse_json_body = || -> Result<Value, String> {
        if req.body.is_empty() {
            return Ok(Value::from(Map::new()));
        }
        let s =
            std::str::from_utf8(&req.body).map_err(|_| "body must be valid UTF-8".to_string())?;
        json::from_str::<Value>(s).map_err(|e| format!("invalid JSON body: {e}"))
    };

    match (req.method.as_str(), req.path_only.as_str()) {
        ("GET", "/api/kernel/status") => {
            let body = kernel_status(root);
            write_api(stream, "200 OK", body.as_bytes())
        }
        ("POST", "/api/kernel/exec") => {
            let payload = match parse_json_body() {
                Ok(v) => v,
                Err(msg) => {
                    let err = json::to_string(&json!({ "ok": false, "stderr": msg }))
                        .unwrap_or_else(|_| "{\"ok\":false}".to_string());
                    return write_api(stream, "400 Bad Request", err.as_bytes());
                }
            };
            let code = payload
                .get("code")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            let reset_ns = payload
                .get("resetNamespace")
                .or_else(|| payload.get("reset_ns"))
                .and_then(|v| v.as_bool())
                .unwrap_or(false);
            let resp = match kernel_roundtrip(
                root,
                &json!({
                    "op": "exec",
                    "code": code,
                    "resetNamespace": reset_ns,
                }),
            ) {
                Ok((_, val)) => {
                    json::to_string(&val).unwrap_or_else(|_| "{\"ok\":false}".to_string())
                }
                Err(msg) => json::to_string(&json!({
                    "ok": false,
                    "stdout": "",
                    "stderr": msg,
                    "incomplete": false,
                    "elapsed_ms": 0.0,
                    "state": null,
                    "mode": "local_native",
                }))
                .unwrap_or_else(|_| "{\"ok\":false}".to_string()),
            };
            write_api(stream, "200 OK", resp.as_bytes())
        }
        ("POST", "/api/kernel/repl") => {
            let payload = match parse_json_body() {
                Ok(v) => v,
                Err(msg) => {
                    let err = json::to_string(&json!({ "ok": false, "stderr": msg }))
                        .unwrap_or_else(|_| "{\"ok\":false}".to_string());
                    return write_api(stream, "400 Bad Request", err.as_bytes());
                }
            };
            let code = payload
                .get("code")
                .or_else(|| payload.get("line"))
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            let resp = match kernel_roundtrip(
                root,
                &json!({
                    "op": "repl",
                    "code": code,
                }),
            ) {
                Ok((_, val)) => {
                    json::to_string(&val).unwrap_or_else(|_| "{\"ok\":false}".to_string())
                }
                Err(msg) => json::to_string(&json!({
                    "ok": false,
                    "stdout": "",
                    "stderr": msg,
                    "incomplete": false,
                    "elapsed_ms": 0.0,
                    "state": null,
                    "mode": "local_native",
                }))
                .unwrap_or_else(|_| "{\"ok\":false}".to_string()),
            };
            write_api(stream, "200 OK", resp.as_bytes())
        }
        ("POST", "/api/kernel/complete") => {
            let payload = match parse_json_body() {
                Ok(v) => v,
                Err(msg) => {
                    let err = json::to_string(&json!({ "ok": false, "matches": [], "error": msg }))
                        .unwrap_or_else(|_| "{\"ok\":false,\"matches\":[]}".to_string());
                    return write_api(stream, "400 Bad Request", err.as_bytes());
                }
            };
            let prefix = payload
                .get("prefix")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            let resp = match kernel_roundtrip(
                root,
                &json!({
                    "op": "complete",
                    "prefix": prefix,
                }),
            ) {
                Ok((_, val)) => json::to_string(&val)
                    .unwrap_or_else(|_| "{\"ok\":true,\"matches\":[]}".to_string()),
                Err(_) => "{\"ok\":false,\"matches\":[]}".to_string(),
            };
            write_api(stream, "200 OK", resp.as_bytes())
        }
        ("POST", "/api/kernel/interrupt") => {
            let force = parse_json_body()
                .ok()
                .and_then(|v| v.get("force").and_then(|f| f.as_bool()))
                .unwrap_or(false);
            let resp = kernel_interrupt(force);
            write_api(stream, "200 OK", resp.as_bytes())
        }
        ("POST", "/api/kernel/reset") => {
            let restart = parse_json_body()
                .ok()
                .and_then(|v| v.get("restart").and_then(|f| f.as_bool()))
                .unwrap_or(false);
            let resp = kernel_reset(root, restart);
            write_api(stream, "200 OK", resp.as_bytes())
        }
        (
            _,
            "/api/kernel/status"
            | "/api/kernel/exec"
            | "/api/kernel/repl"
            | "/api/kernel/complete"
            | "/api/kernel/interrupt"
            | "/api/kernel/reset",
        ) => write_api(
            stream,
            "405 Method Not Allowed",
            r#"{"ok":false,"error":"method not allowed"}"#.as_bytes(),
        ),
        _ => write_api(
            stream,
            "404 Not Found",
            r#"{"ok":false,"error":"unknown kernel endpoint"}"#.as_bytes(),
        ),
    }
}

fn handle_http(root: &Path, bundle_dir: &Path, stream: &mut TcpStream) -> Result<(), String> {
    let req = match read_http(stream)? {
        ReadHttpResult::Eof => return Ok(()),
        ReadHttpResult::Reject(status, msg) => {
            return write_http(stream, status, "text/plain; charset=utf-8", msg.as_bytes());
        }
        ReadHttpResult::Req(r) => r,
    };

    if req.method == "GET" && (req.path_only == "/api/env" || req.path_only == "/api/runs") {
        let body = env_json(root)?;
        return write_api(stream, "200 OK", body.as_bytes());
    }

    if req.path_only.starts_with("/api/kernel/") {
        return kernel_api(root, stream, &req);
    }

    let rel = if req.path_only == "/" {
        "index.html"
    } else {
        req.path_only.trim_start_matches('/')
    };
    let rel_path = Path::new(rel);
    if rel_path
        .components()
        .any(|c| matches!(c, Component::ParentDir | Component::RootDir))
    {
        return write_http(stream, "403 Forbidden", "text/plain", b"forbidden");
    }
    let target = bundle_dir.join(rel_path);
    if !target.is_file() {
        return write_http(stream, "404 Not Found", "text/plain", b"not found");
    }
    let bytes = fs::read(&target).map_err(|e| e.to_string())?;
    let ctype = match target.extension().and_then(|e| e.to_str()).unwrap_or("") {
        "html" => "text/html; charset=utf-8",
        "js" | "mjs" => "application/javascript; charset=utf-8",
        "css" => "text/css; charset=utf-8",
        "json" => "application/json; charset=utf-8",
        "svg" => "image/svg+xml",
        "wasm" => "application/wasm",
        "zip" => "application/zip",
        "woff2" => "font/woff2",
        "png" => "image/png",
        _ => "application/octet-stream",
    };
    write_http(stream, "200 OK", ctype, &bytes)
}

fn write_headers(
    stream: &mut TcpStream,
    status: &str,
    ctype: &str,
    body: &[u8],
    no_store: bool,
) -> Result<(), String> {
    let cache_line = if no_store {
        "Cache-Control: no-store\r\n"
    } else {
        ""
    };
    let hdr = format!(
        "HTTP/1.1 {status}\r\n\
         Content-Type: {ctype}\r\n\
         Content-Length: {}\r\n\
         Cross-Origin-Opener-Policy: same-origin\r\n\
         Cross-Origin-Embedder-Policy: require-corp\r\n\
         Cross-Origin-Resource-Policy: same-origin\r\n\
         X-Content-Type-Options: nosniff\r\n\
         {cache_line}\
         Connection: close\r\n\r\n",
        body.len()
    );
    stream
        .write_all(hdr.as_bytes())
        .map_err(|e| e.to_string())?;
    stream.write_all(body).map_err(|e| e.to_string())
}

fn write_api(stream: &mut TcpStream, status: &str, body: &[u8]) -> Result<(), String> {
    write_headers(
        stream,
        status,
        "application/json; charset=utf-8",
        body,
        true,
    )
}

fn write_http(
    stream: &mut TcpStream,
    status: &str,
    ctype: &str,
    body: &[u8],
) -> Result<(), String> {
    write_headers(stream, status, ctype, body, false)
}

struct DeployGuard<'a> {
    root: &'a Path,
    tmp_dir: PathBuf,
}

impl Drop for DeployGuard<'_> {
    fn drop(&mut self) {
        let _ = Command::new("jj")
            .args(["workspace", "forget", "gh-pages-deploy"])
            .current_dir(self.root)
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status();
        let _ = fs::remove_dir_all(&self.tmp_dir);
    }
}

fn deploy(root: &Path) -> Result<(), String> {
    let _ = Command::new("jj")
        .args(["workspace", "forget", "gh-pages-deploy"])
        .current_dir(root)
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status();

    let nonce = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_err(|e| e.to_string())?
        .as_nanos();
    let tmp_dir =
        std::env::temp_dir().join(format!("baux-gh-pages-{}-{nonce}", std::process::id()));
    fs::create_dir_all(&tmp_dir).map_err(|e| e.to_string())?;
    let _guard = DeployGuard {
        root,
        tmp_dir: tmp_dir.clone(),
    };
    let ws_dir = tmp_dir.join("ws");

    run_checked(
        Command::new("jj")
            .args([
                "workspace",
                "add",
                &ws_dir.display().to_string(),
                "--name",
                "gh-pages-deploy",
                "-r",
                "gh-pages",
            ])
            .current_dir(root),
        "jj workspace add gh-pages-deploy",
    )?;
    run_checked(
        Command::new("jj")
            .args(["new", "gh-pages"])
            .current_dir(&ws_dir),
        "jj new gh-pages",
    )?;

    build(root, &ws_dir)?;

    println!("o|pages: deploying verified bundle to origin/gh-pages...");
    run_checked(
        Command::new("jj")
            .args(["describe", "-m", "kvutza/pages"])
            .current_dir(&ws_dir),
        "jj describe",
    )?;
    run_checked(
        Command::new("jj")
            .args(["bookmark", "set", "gh-pages", "-r", "@"])
            .current_dir(&ws_dir),
        "jj bookmark set gh-pages",
    )?;
    run_checked(
        Command::new("jj")
            .args(["git", "push", "--bookmark", "gh-pages"])
            .current_dir(&ws_dir),
        "jj git push gh-pages",
    )?;
    println!("o|pages: deployed to origin/gh-pages");
    Ok(())
}
