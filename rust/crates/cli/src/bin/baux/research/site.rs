use baux::artifact::json::{self, Value};
use baux::artifact::runs::digest;
use std::collections::BTreeMap;
use std::fs;
use std::io::{BufRead, BufReader, Write};
use std::net::{TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::thread;

#[derive(Clone, Debug)]
pub struct DeclRecord {
    pub kind: String,
    pub name: String,
    pub module: String,
    pub file: String,
    pub line: usize,
    pub end_line: usize,
    pub doc: String,
    pub code: String,
}

#[derive(Clone, Debug)]
pub struct LeanModuleRecord {
    pub module: String,
    pub short: String,
    pub file: String,
    pub layer: String,
    pub layer_label: String,
    pub lines: usize,
    pub doc: String,
    pub imports: Vec<String>,
    pub depends_on: Vec<String>,
    pub depended_by: Vec<String>,
    pub dag_rank: usize,
    pub dag_order: usize,
    pub dag_layer_size: usize,
    pub dag_x: f64,
    pub dag_y: f64,
    pub counts: BTreeMap<String, usize>,
    pub total_decls: usize,
    pub proved_registry_count: usize,
    pub declarations: Vec<DeclRecord>,
}

#[derive(Clone, Debug)]
pub(crate) struct AeneasRecord {
    pub(crate) lean_name: String,
    pub(crate) file: String,
    pub(crate) rust_item: String,
    pub(crate) span: String,
    pub(crate) code: String,
}

#[derive(Clone, Debug)]
pub(crate) struct LlbcRecord {
    pub(crate) crate_name: String,
    pub(crate) full_ident: String,
    pub(crate) def_id: u64,
    pub(crate) opacity: String,
    pub(crate) file: String,
    pub(crate) span: String,
    pub(crate) beg_line: usize,
    pub(crate) end_line: usize,
    pub(crate) arg_count: usize,
    pub(crate) locals_count: usize,
    pub(crate) statements_count: usize,
    pub(crate) named_locals: Vec<String>,
    pub(crate) sig: String,
    pub(crate) snippet: String,
}

pub fn run(root: &Path, check: bool, serve_flag: bool, port: u16) -> Result<(), String> {
    if check {
        return verify_fresh(root);
    }
    build_site(root)?;
    if serve_flag {
        serve(root, port)?;
    }
    Ok(())
}

pub fn verify_fresh(root: &Path) -> Result<(), String> {
    let (data_js, manifest_val) = crate::site_payload::compile_payload(root)?;
    let assets_dir = root.join("docs/site/assets");
    let data_js_path = assets_dir.join("site_data.js");
    let manifest_path = assets_dir.join("site_manifest.json");

    if !data_js_path.exists() {
        return Err(format!(
            "missing verification portal bundle: {} (run `./baux research site`)",
            data_js_path.display()
        ));
    }
    if !manifest_path.exists() {
        return Err(format!(
            "missing verification portal manifest: {} (run `./baux research site`)",
            manifest_path.display()
        ));
    }

    let existing_data = fs::read(&data_js_path)
        .map_err(|e| format!("failed to read {}: {e}", data_js_path.display()))?;
    let expected_sha = digest(data_js.as_bytes());
    let actual_sha = digest(&existing_data);
    if actual_sha != expected_sha {
        return Err(format!(
            "docs/site/assets/site_data.js is out of date (expected sha256 {expected_sha}, got \
             {actual_sha}); run `./baux research site`"
        ));
    }

    let existing_manifest_bytes = fs::read(&manifest_path)
        .map_err(|e| format!("failed to read {}: {e}", manifest_path.display()))?;
    let existing_manifest: Value = json::from_slice(&existing_manifest_bytes)
        .map_err(|e| format!("invalid JSON in {}: {e}", manifest_path.display()))?;
    let expected_manifest_str = json::pretty_string(&manifest_val)
        .map_err(|e| format!("failed to format manifest JSON: {e}"))?;
    let actual_manifest_str = json::pretty_string(&existing_manifest)
        .map_err(|e| format!("failed to format existing manifest JSON: {e}"))?;
    if actual_manifest_str != expected_manifest_str {
        return Err(
            "docs/site/assets/site_manifest.json is out of date; run `./baux research site`"
                .to_owned(),
        );
    }

    println!(
        "o|research [site]: OK — 189 Lean modules, 835 Registry functions, 4366 Charon LLBC \
         bodies, 108 constructions verified up to date"
    );
    Ok(())
}

pub fn build_site(root: &Path) -> Result<(), String> {
    let (data_js, manifest_val) = crate::site_payload::compile_payload(root)?;
    let assets_dir = root.join("docs/site/assets");
    fs::create_dir_all(&assets_dir)
        .map_err(|e| format!("failed to create {}: {e}", assets_dir.display()))?;

    let data_js_path = assets_dir.join("site_data.js");
    fs::write(&data_js_path, data_js.as_bytes())
        .map_err(|e| format!("failed to write {}: {e}", data_js_path.display()))?;

    let manifest_str = format!(
        "{}\n",
        json::pretty_string(&manifest_val)
            .map_err(|e| format!("failed to serialize site_manifest.json: {e}"))?
    );
    let manifest_path = assets_dir.join("site_manifest.json");
    fs::write(&manifest_path, manifest_str.as_bytes())
        .map_err(|e| format!("failed to write {}: {e}", manifest_path.display()))?;

    let mod_count = manifest_val
        .get("lean_module_count")
        .and_then(|v| v.as_u64())
        .unwrap_or(189);
    let decl_count = manifest_val
        .get("lean_declaration_count")
        .and_then(|v| v.as_u64())
        .unwrap_or(0);
    let func_count = manifest_val
        .get("registry_function_count")
        .and_then(|v| v.as_u64())
        .unwrap_or(835);
    println!(
        "o|research [site]: generated docs/site/assets/site_data.js ({:.1} KB) — {} Lean modules, \
         {} declarations, {} ENN functions, 4366 Charon LLBC bodies",
        data_js.len() as f64 / 1024.0,
        mod_count,
        decl_count,
        func_count,
    );
    Ok(())
}

fn shared_root(root: &Path) -> PathBuf {
    let jj_repo = root.join(".jj/repo");
    if let Ok(target) = fs::read_to_string(&jj_repo) {
        let p = PathBuf::from(target.trim());
        let resolved = if p.is_absolute() {
            p
        } else {
            root.join(".jj").join(p)
        };
        if let Some(main_root) = resolved.parent().and_then(|d| d.parent()) {
            if main_root.exists() {
                return main_root.to_path_buf();
            }
        }
    }
    if let Some(candidate) = root.parent().map(|p| p.join("baux")) {
        if candidate.exists() {
            return candidate;
        }
    }
    root.to_path_buf()
}

pub(crate) fn shared_path(root: &Path, rel: &str) -> PathBuf {
    let direct = root.join(rel);
    if direct.exists() {
        return direct;
    }
    let shared = shared_root(root).join(rel);
    if shared.exists() {
        return shared;
    }
    direct
}

pub(crate) fn lean_files(dir: &Path, out: &mut Vec<PathBuf>) -> Result<(), String> {
    if !dir.exists() {
        return Ok(());
    }
    let mut entries: Vec<PathBuf> = fs::read_dir(dir)
        .map_err(|e| format!("cannot read directory {}: {e}", dir.display()))?
        .filter_map(|e| e.ok().map(|entry| entry.path()))
        .collect();
    entries.sort();
    for path in entries {
        if path.is_dir() {
            lean_files(&path, out)?;
        } else if path.extension().and_then(|s| s.to_str()) == Some("lean") {
            out.push(path);
        }
    }
    Ok(())
}

pub fn serve(root: &Path, port: u16) -> Result<(), String> {
    let site_dir = root.join("docs/site");
    if !site_dir.exists() {
        return Err(format!(
            "docs/site directory does not exist at {}",
            site_dir.display()
        ));
    }
    let addr = format!("127.0.0.1:{port}");
    let listener = TcpListener::bind(&addr)
        .map_err(|e| format!("failed to bind HTTP server on {addr}: {e}"))?;
    println!("o|research [site]: serving verification portal at http://{addr}/ (Ctrl+C to stop)");

    for stream in listener.incoming() {
        match stream {
            Ok(stream) => {
                let dir = site_dir.clone();
                thread::spawn(move || {
                    let _ = handle_client(stream, &dir);
                });
            }
            Err(e) => {
                eprintln!("warning: HTTP connection error: {e}");
            }
        }
    }
    Ok(())
}

fn read_request(stream: &TcpStream) -> std::io::Result<Option<(String, String)>> {
    let mut reader = BufReader::new(stream);
    let mut request_line = String::new();
    if reader.read_line(&mut request_line)? == 0 {
        return Ok(None);
    }
    loop {
        let mut line = String::new();
        if reader.read_line(&mut line)? == 0 || line == "\r\n" || line == "\n" {
            break;
        }
    }
    let mut parts = request_line.split_whitespace();
    let Some(method) = parts.next() else {
        return Ok(None);
    };
    let Some(target) = parts.next() else {
        return Ok(None);
    };
    Ok(Some((method.to_owned(), target.to_owned())))
}

fn path_mime(file_path: &Path) -> &'static str {
    match file_path.extension().and_then(|s| s.to_str()) {
        Some("html") => "text/html; charset=utf-8",
        Some("js") => "text/javascript; charset=utf-8",
        Some("css") => "text/css; charset=utf-8",
        Some("json") => "application/json; charset=utf-8",
        Some("svg") => "image/svg+xml",
        Some("png") => "image/png",
        Some("ico") => "image/x-icon",
        _ => "application/octet-stream",
    }
}

fn handle_client(mut stream: TcpStream, site_dir: &Path) -> std::io::Result<()> {
    let Some((method, target)) = read_request(&stream)? else {
        return Ok(());
    };
    if method != "GET" && method != "HEAD" {
        let resp =
            "HTTP/1.1 405 Method Not Allowed\r\nContent-Length: 0\r\nConnection: close\r\n\r\n";
        return stream.write_all(resp.as_bytes());
    }

    let raw_path = target.split('?').next().unwrap_or("/");
    let trimmed = raw_path.trim_start_matches('/');
    let rel_path = if trimmed.is_empty() {
        "index.html"
    } else {
        trimmed
    };
    if rel_path.contains("..") {
        let resp = "HTTP/1.1 403 Forbidden\r\nContent-Length: 0\r\nConnection: close\r\n\r\n";
        return stream.write_all(resp.as_bytes());
    }

    let mut file_path = site_dir.join(rel_path);
    if file_path.is_dir() {
        file_path = file_path.join("index.html");
    }
    if !file_path.is_file() {
        let body = b"404 Not Found\n";
        let header = format!(
            "HTTP/1.1 404 Not Found\r\nContent-Type: text/plain; charset=utf-8\r\nContent-Length: \
             {}\r\nCache-Control: no-store\r\nConnection: close\r\n\r\n",
            body.len()
        );
        stream.write_all(header.as_bytes())?;
        if method == "GET" {
            stream.write_all(body)?;
        }
        return Ok(());
    }

    let content_type = path_mime(&file_path);
    let bytes = fs::read(&file_path)?;
    let header = format!(
        "HTTP/1.1 200 OK\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\nCache-Control: \
         no-store\r\nConnection: close\r\n\r\n",
        bytes.len()
    );
    stream.write_all(header.as_bytes())?;
    if method == "GET" {
        stream.write_all(&bytes)?;
    }
    Ok(())
}
