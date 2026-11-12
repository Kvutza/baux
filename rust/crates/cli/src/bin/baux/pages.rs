use std::collections::BTreeMap;
use std::fs;
use std::io::{BufRead, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::{Component, Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::Instant;

use baux::artifact::blake3::{hash_hex, sha256_hex, sha384_sri};
use baux::artifact::runs;
use clap::Subcommand;

#[derive(Subcommand)]
pub enum PagesAction {
    /// Build OCaml 3D CAD UI, export FUSE.jl & Oceananigans.jl environments,
    /// run Forester, stamp SRI, and verify.
    Build {
        #[arg(long, default_value = ".cache/pages-bundle")]
        bundle_dir: PathBuf,
    },
    /// Verify CSP, SRI, zero machine-path leaks, and bundle hygiene.
    Check {
        #[arg(long, default_value = ".cache/pages-bundle")]
        bundle_dir: PathBuf,
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
        PagesAction::Check { bundle_dir } => check_bundle(&resolve_bundle(root, &bundle_dir)),
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

fn daemon_sock(root: &Path, ocean: bool) -> PathBuf {
    let tag = &hash_hex(root.as_os_str().as_encoded_bytes())[..10];
    let kind = if ocean { "ocean" } else { "fuse" };
    std::env::temp_dir().join(format!("baux_jl_{tag}_{kind}.sock"))
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

fn spawn_daemon(root: &Path, ocean: bool) -> Result<PathBuf, String> {
    let sock = daemon_sock(root, ocean);
    let proj = if ocean {
        root.join("evals/oceananigans")
    } else {
        root.join("evals/fuse")
    };
    let log_dir = root.join(".cache/baux/shm");
    fs::create_dir_all(&log_dir).map_err(|e| e.to_string())?;
    let log_path = log_dir.join(if ocean {
        "julia_ocean_daemon.log"
    } else {
        "julia_fuse_daemon.log"
    });
    let log_file = fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&log_path)
        .map_err(|e| format!("open daemon log {}: {e}", log_path.display()))?;

    let mut cmd = Command::new(julia_bin());
    cmd.args(["--startup-file=no", "--threads=auto"]);
    if !ocean {
        cmd.arg("-O1");
    }
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

fn export_env(root: &Path, ocean: bool) -> Result<String, String> {
    let cache_dir = root.join(".cache/baux");
    fs::create_dir_all(&cache_dir).map_err(|e| e.to_string())?;
    let out_path = cache_dir.join(if ocean {
        "ocean_env.json"
    } else {
        "fuse_env.json"
    });
    let proj = if ocean {
        root.join("evals/oceananigans")
    } else {
        root.join("evals/fuse")
    };
    let deps = [proj.join("worker.jl"), root.join("evals/julia/shm_ipc.jl")];
    if is_fresh(&out_path, &deps) {
        return fs::read_to_string(&out_path).map_err(|e| e.to_string());
    }

    let label = if ocean {
        "Oceananigans.jl 0.117.2"
    } else {
        "FUSE.jl 1.2.1"
    };
    println!("o|pages: exporting 3D CAD & PDE environment via {label} daemon...");

    let sock = daemon_sock(root, ocean);
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
        let live_sock = spawn_daemon(root, ocean)?;
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
    let ocean = export_env(root, true)?;
    let fuse = export_env(root, false)?;

    let records = runs::history(
        root,
        &runs::Query {
            limit: 32,
            ..Default::default()
        },
    )
    .unwrap_or_default();

    let mut run_items = Vec::new();
    for r in records {
        let meas_json =
            baux::artifact::json::to_string(&r.measurements).unwrap_or_else(|_| "{}".to_string());
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
    }

    Ok(format!(
        "{{\"fuse\":{},\"ocean\":{},\"runs\":[{}]}}",
        fuse.trim(),
        ocean.trim(),
        run_items.join(",")
    ))
}

const SRI_ASSETS: [&str; 4] = [
    "style.css",
    "katex.min.css",
    "forester.js",
    "assets/baux_ui.js",
];

const ROOT_FILES: [&str; 10] = [
    "style.css",
    "katex.min.css",
    "forester.js",
    "forest.json",
    "favicon.ico",
    "favicon-16x16.png",
    "favicon-32x32.png",
    "apple-touch-icon.png",
    "android-chrome-192x192.png",
    "android-chrome-512x512.png",
];

pub fn build(root: &Path, bundle_dir: &Path) -> Result<(), String> {
    println!("o|pages [1/4]: building OCaml 3D WebGL CAD UI (admin/ui)...");
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
            .args(["exec", ui_switch, "--", "dune", "build", "./baux_ui.js"])
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

    println!("o|pages [2/4]: assembling FUSE.jl & Oceananigans.jl 3D CAD environments...");
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
            if matches!(ext, "xsl" | "css" | "js") {
                copy_rw(&p, &output_dir.join(entry.file_name()))?;
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
        ("baux-0004.html", "simulation.html"),
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
    fs::write(bundle_assets.join("baux_env.json"), &cad_env).map_err(|e| e.to_string())?;
    fs::write(bundle_dir.join(".nojekyll"), b"").map_err(|e| e.to_string())?;

    println!("o|pages [4/4]: stamping SRI hashes and running security check...");
    stamp_sri(bundle_dir)?;
    stamp_assets(&output_dir, bundle_dir)?;
    check_bundle(bundle_dir)?;
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

fn stamp_sri(bundle_dir: &Path) -> Result<(), String> {
    stamp_assets(bundle_dir, bundle_dir)
}

fn stamp_assets(html_dir: &Path, asset_root: &Path) -> Result<(), String> {
    let mut hashes: BTreeMap<String, (String, String)> = BTreeMap::new();
    for rel in SRI_ASSETS {
        let p = asset_root.join(rel);
        if p.is_file() {
            let bytes = fs::read(&p).map_err(|e| e.to_string())?;
            let ver = sha256_hex(&bytes)[..12].to_string();
            let sri = format!("sha384-{}", sha384_sri(&bytes));
            hashes.insert(rel.to_string(), (ver, sri));
        }
    }

    let mut html_files = Vec::new();
    collect_ext(html_dir, "html", &mut html_files)?;
    for html_path in html_files {
        let mut content = fs::read_to_string(&html_path).map_err(|e| e.to_string())?;
        for (rel, (ver, sri)) in &hashes {
            for prefix in ["", "./"] {
                let raw_ref = format!("{prefix}{rel}");
                let needle_plain = format!("\"{raw_ref}\"");
                let stamped = format!("\"{raw_ref}?v={ver}\" integrity=\"{sri}\"");
                if content.contains(&needle_plain)
                    && !content.contains(&format!("\"{raw_ref}?v={ver}\""))
                {
                    content = content.replace(&needle_plain, &stamped);
                }
            }
        }
        fs::write(&html_path, content).map_err(|e| e.to_string())?;
    }
    Ok(())
}

fn collect_ext(dir: &Path, ext: &str, out: &mut Vec<PathBuf>) -> Result<(), String> {
    for entry in fs::read_dir(dir).map_err(|e| e.to_string())? {
        let entry = entry.map_err(|e| e.to_string())?;
        let name = entry.file_name();
        if name == ".jj" || name == ".git" || name == "node_modules" {
            continue;
        }
        let p = entry.path();
        if p.is_dir() {
            collect_ext(&p, ext, out)?;
        } else if p.extension().and_then(|e| e.to_str()) == Some(ext) {
            out.push(p);
        }
    }
    Ok(())
}

pub fn check_bundle(bundle_dir: &Path) -> Result<(), String> {
    if !bundle_dir.is_dir() {
        return Err(format!(
            "bundle directory missing: {}",
            bundle_dir.display()
        ));
    }
    for req in [
        "index.html",
        "docs.html",
        "simulation.html",
        "style.css",
        "katex.min.css",
        "forester.js",
        "forest.json",
        ".nojekyll",
        "assets/baux_ui.js",
        "assets/baux_env.json",
    ] {
        if !bundle_dir.join(req).is_file() {
            return Err(format!("required bundle asset missing: {req}"));
        }
    }
    for forbidden in ["forest", "node_modules", "assets/site_data.js"] {
        if bundle_dir.join(forbidden).exists() {
            return Err(format!("forbidden bundle path present: {forbidden}"));
        }
    }

    let mut all_files = Vec::new();
    walk_all(bundle_dir, &mut all_files)?;
    for path in &all_files {
        let rel = path
            .strip_prefix(bundle_dir)
            .unwrap_or(path)
            .to_string_lossy()
            .to_string();
        let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
        if matches!(
            ext,
            "map" | "env" | "xml" | "xsl" | "sh" | "ml" | "mli" | "toml"
        ) {
            return Err(format!("forbidden file suffix in bundle: {rel}"));
        }
        if matches!(ext, "html" | "js" | "css" | "json") {
            let text = fs::read_to_string(path).map_err(|e| e.to_string())?;
            if text.contains("/Users/") || text.contains("/home/runner/") {
                return Err(format!("machine path leak in {rel}"));
            }
            if ext == "html" {
                check_html(bundle_dir, &rel, &text)?;
            }
            if ext == "js" {
                run_checked(
                    Command::new("node").arg("-c").arg(path),
                    &format!("node -c {rel}"),
                )?;
            }
        }
    }
    Ok(())
}

fn extract_attr<'a>(tag: &'a str, attr: &str) -> Option<&'a str> {
    let pat = format!("{attr}=\"");
    let start = tag.find(&pat)? + pat.len();
    let rest = &tag[start..];
    let end = rest.find('"')?;
    Some(&rest[..end])
}

fn verify_sri(bundle_dir: &Path, rel: &str, tag_head: &str, url_attr: &str) -> Result<(), String> {
    let raw_ref = extract_attr(tag_head, url_attr)
        .ok_or_else(|| format!("{rel}: missing {url_attr} on asset tag"))?;
    let sri = extract_attr(tag_head, "integrity")
        .ok_or_else(|| format!("{rel}: missing SRI integrity on {raw_ref}"))?;
    let asset_rel = raw_ref
        .split('?')
        .next()
        .unwrap_or(raw_ref)
        .trim_start_matches("./")
        .trim_start_matches('/');
    let asset_bytes = fs::read(bundle_dir.join(asset_rel))
        .map_err(|e| format!("{rel}: missing asset {asset_rel}: {e}"))?;
    let expected = format!("sha384-{}", sha384_sri(&asset_bytes));
    if sri != expected {
        return Err(format!(
            "{rel}: SRI sha384 mismatch for {asset_rel} (got {sri}, expected {expected})"
        ));
    }
    Ok(())
}

fn check_html(bundle_dir: &Path, rel: &str, html: &str) -> Result<(), String> {
    let lower = html.to_ascii_lowercase();
    if !lower.contains("content-security-policy")
        || !lower.contains("default-src 'none'")
        || !lower.contains("script-src 'self'")
    {
        return Err(format!("{rel}: missing strict CSP meta tag"));
    }
    if lower.contains("'unsafe-eval'") || lower.contains("script-src 'self' 'unsafe-inline'") {
        return Err(format!(
            "{rel}: forbidden unsafe script-src directive in CSP"
        ));
    }
    if lower.contains("javascript:") {
        return Err(format!("{rel}: forbidden javascript: URI"));
    }
    if lower.contains("href=\"http://") || lower.contains("src=\"http://") {
        return Err(format!("{rel}: forbidden cleartext http:// reference"));
    }
    let mut rest = html;
    while let Some(pos) = rest.find("<script") {
        let after = &rest[pos..];
        let end_tag = after
            .find('>')
            .ok_or_else(|| format!("{rel}: unclosed <script"))?;
        let tag_head = &after[..end_tag];
        verify_sri(bundle_dir, rel, tag_head, "src")?;
        rest = &after[end_tag + 1..];
    }
    let mut l_rest = html;
    while let Some(pos) = l_rest.find("<link") {
        let after = &l_rest[pos..];
        let end_tag = after
            .find('>')
            .ok_or_else(|| format!("{rel}: unclosed <link"))?;
        let tag_head = &after[..end_tag];
        if let Some(rel_val) = extract_attr(tag_head, "rel")
            && rel_val.eq_ignore_ascii_case("stylesheet")
        {
            verify_sri(bundle_dir, rel, tag_head, "href")?;
        }
        l_rest = &after[end_tag + 1..];
    }
    let mut a_rest = lower.as_str();
    while let Some(pos) = a_rest.find("<a ") {
        let after = &a_rest[pos..];
        let Some(end_tag) = after.find('>') else {
            break;
        };
        let tag_head = &after[..end_tag];
        if tag_head.contains("target=\"_blank\"")
            && (!tag_head.contains("noopener") || !tag_head.contains("noreferrer"))
        {
            return Err(format!("{rel}: target=_blank missing noopener noreferrer"));
        }
        a_rest = &after[end_tag + 1..];
    }
    Ok(())
}

fn walk_all(dir: &Path, out: &mut Vec<PathBuf>) -> Result<(), String> {
    for entry in fs::read_dir(dir).map_err(|e| e.to_string())? {
        let entry = entry.map_err(|e| e.to_string())?;
        let name = entry.file_name();
        if name == ".jj" || name == ".git" {
            continue;
        }
        let p = entry.path();
        if p.is_dir() {
            walk_all(&p, out)?;
        } else {
            out.push(p);
        }
    }
    Ok(())
}

fn serve(root: &Path, bundle_dir: &Path, port: u16) -> Result<(), String> {
    let addr = format!("127.0.0.1:{port}");
    let listener = TcpListener::bind(&addr).map_err(|e| format!("bind {addr}: {e}"))?;
    println!(
        "o|pages: serving {} at http://{addr}/simulation.html",
        bundle_dir.display()
    );
    for stream in listener.incoming() {
        if let Ok(mut s) = stream {
            let _ = handle_http(root, bundle_dir, &mut s);
        }
    }
    Ok(())
}

fn handle_http(root: &Path, bundle_dir: &Path, stream: &mut TcpStream) -> Result<(), String> {
    let mut buf = [0u8; 8192];
    let n = stream.read(&mut buf).map_err(|e| e.to_string())?;
    if n == 0 {
        return Ok(());
    }
    let req = String::from_utf8_lossy(&buf[..n]);
    let first_line = req.lines().next().unwrap_or("");
    let mut parts = first_line.split_whitespace();
    let method = parts.next().unwrap_or("GET");
    let raw_path = parts.next().unwrap_or("/");
    let path_only = raw_path.split('?').next().unwrap_or("/");

    if method == "GET" && (path_only == "/api/env" || path_only == "/api/runs") {
        let body = env_json(root)?;
        return write_http(stream, "200 OK", "application/json", body.as_bytes());
    }

    let rel = if path_only == "/" {
        "index.html"
    } else {
        path_only.trim_start_matches('/')
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
        "js" => "application/javascript; charset=utf-8",
        "css" => "text/css; charset=utf-8",
        "json" => "application/json; charset=utf-8",
        "svg" => "image/svg+xml",
        _ => "application/octet-stream",
    };
    write_http(stream, "200 OK", ctype, &bytes)
}

fn write_http(
    stream: &mut TcpStream,
    status: &str,
    ctype: &str,
    body: &[u8],
) -> Result<(), String> {
    let hdr = format!(
        "HTTP/1.1 {status}\r\nContent-Type: {ctype}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        body.len()
    );
    stream
        .write_all(hdr.as_bytes())
        .map_err(|e| e.to_string())?;
    stream.write_all(body).map_err(|e| e.to_string())
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
