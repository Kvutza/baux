//! Standalone HTML/CSP/SRI/hygiene verifier and public-tree guard for `baux`.
//!
//! Zero external crate dependencies: compiles both as `mod pages_verify;`
//! inside `baux` and standalone via `rustc --edition 2021 -O --cfg
//! pages_verify_cli`.

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::io::Write;
use std::path::{Component, Path, PathBuf};
use std::process::{Command, Stdio};

const SRI_ASSETS: [&str; 4] = [
    "style.css",
    "katex.min.css",
    "forester.js",
    "assets/baux_ui.js",
];

const SENSITIVE_PATTERNS: [&str; 10] = [
    "tab-btn-monograph",
    "view-monograph",
    "renderMonographView",
    "renderFallbackMonographView",
    "monograph-shell",
    "monograph-switcher",
    "fb-mono-switcher",
    "BAUX FORMAL VERIFICATION CONTRACT",
    "VERIFICATION CONTRACT & TRUST BOUNDARY",
    "GATEWAY_GROUPS",
];

const REQUIRED_FILES: [&str; 25] = [
    "index.html",
    "docs.html",
    "explorer.html",
    "style.css",
    "katex.min.css",
    "forester.js",
    "coi-serviceworker.js",
    "forest.json",
    ".nojekyll",
    "assets/baux_ui.js",
    "assets/oxcaml_eval.js",
    "assets/webgpu_bridge.js",
    "assets/webgpu_worker.js",
    "assets/baux_env.json",
    "assets/repl_worker.js",
    "assets/baux_py.zip",
    "assets/repl_manifest.json",
    "assets/pyodide/pyodide.js",
    "assets/pyodide/pyodide.asm.js",
    "assets/pyodide/pyodide.asm.wasm",
    "assets/pyodide/python_stdlib.zip",
    "assets/pyodide/pyodide-lock.json",
    "assets/pyodide/numpy-2.0.2-cp312-cp312-pyodide_2024_0_wasm32.whl",
    "LICENSES/PYODIDE",
    "LICENSES/NUMPY",
];

const FORBIDDEN_PATHS: [&str; 12] = [
    "forest",
    "node_modules",
    "javascript-source",
    "package.json",
    "package-lock.json",
    "bundle-js.sh",
    "baux_api_data.json",
    "assets/site_style.css",
    "assets/site_data.js",
    "assets/site_manifest.json",
    "assets/blueprint_engine.js",
    "assets/explorer_engine.js",
];

const RETIRED_SCRIPTS: [&str; 5] = [
    "admin/ops/publish_pages.py",
    "admin/ops/verify_pages.py",
    "admin/ops/build_repl.py",
    "admin/ops/security_audit.py",
    "admin/ops/pyodide_manifest.json",
];

const TEXT_ALLOWLIST: [&str; 14] = [
    "README.md",
    "admin/buck2/iree/CMakeLists.txt",
    "admin/ops/flame/requirements-coding.txt",
    "admin/ops/flame/requirements-cuda.txt",
    "admin/ops/flame/requirements-metal.txt",
    "admin/ops/flame/requirements-native.txt",
    "admin/ops/qwen/requirements-metal.txt",
    "admin/ops/qwen/requirements.txt",
    "forest/theme/LICENSES/CC0-1.0.txt",
    "forest/theme/LICENSES/MIT.txt",
    "forest/theme/LICENSES/OFL-1.1.txt",
    "ocaml/extraction/charon/charon-ml/README.md",
    "tests/cli/drills_test.md",
    "tests/cli/toolchain_test.md",
];

const USER_MARKERS: [&str; 2] = ["/Users/", "/home/runner/"];

#[rustfmt::skip]
const SHA256_K: [u32; 64] = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
];

#[rustfmt::skip]
const SHA512_K: [u64; 80] = [
    0x428a2f98d728ae22, 0x7137449123ef65cd, 0xb5c0fbcfec4d3b2f, 0xe9b5dba58189dbbc,
    0x3956c25bf348b538, 0x59f111f1b605d019, 0x923f82a4af194f9b, 0xab1c5ed5da6d8118,
    0xd807aa98a3030242, 0x12835b0145706fbe, 0x243185be4ee4b28c, 0x550c7dc3d5ffb4e2,
    0x72be5d74f27b896f, 0x80deb1fe3b1696b1, 0x9bdc06a725c71235, 0xc19bf174cf692694,
    0xe49b69c19ef14ad2, 0xefbe4786384f25e3, 0x0fc19dc68b8cd5b5, 0x240ca1cc77ac9c65,
    0x2de92c6f592b0275, 0x4a7484aa6ea6e483, 0x5cb0a9dcbd41fbd4, 0x76f988da831153b5,
    0x983e5152ee66dfab, 0xa831c66d2db43210, 0xb00327c898fb213f, 0xbf597fc7beef0ee4,
    0xc6e00bf33da88fc2, 0xd5a79147930aa725, 0x06ca6351e003826f, 0x142929670a0e6e70,
    0x27b70a8546d22ffc, 0x2e1b21385c26c926, 0x4d2c6dfc5ac42aed, 0x53380d139d95b3df,
    0x650a73548baf63de, 0x766a0abb3c77b2a8, 0x81c2c92e47edaee6, 0x92722c851482353b,
    0xa2bfe8a14cf10364, 0xa81a664bbc423001, 0xc24b8b70d0f89791, 0xc76c51a30654be30,
    0xd192e819d6ef5218, 0xd69906245565a910, 0xf40e35855771202a, 0x106aa07032bbd1b8,
    0x19a4c116b8d2d0c8, 0x1e376c085141ab53, 0x2748774cdf8eeb99, 0x34b0bcb5e19b48a8,
    0x391c0cb3c5c95a63, 0x4ed8aa4ae3418acb, 0x5b9cca4f7763e373, 0x682e6ff3d6b2b8a3,
    0x748f82ee5defb2fc, 0x78a5636f43172f60, 0x84c87814a1f0ab72, 0x8cc702081a6439ec,
    0x90befffa23631e28, 0xa4506cebde82bde9, 0xbef9a3f7b2c67915, 0xc67178f2e372532b,
    0xca273eceea26619c, 0xd186b8c721c0c207, 0xeada7dd6cde0eb1e, 0xf57d4f7fee6ed178,
    0x06f067aa72176fba, 0x0a637dc5a2c898a6, 0x113f9804bef90dae, 0x1b710b35131c471b,
    0x28db77f523047d84, 0x32caab7b40c72493, 0x3c9ebe0a15c9bebc, 0x431d67c49c100d4c,
    0x4cc5d4becb3e42b6, 0x597f299cfc657e2a, 0x5fcb6fab3ad6faec, 0x6c44198c4a475817,
];

/// Compute lowercase hex-encoded SHA-256 digest using FIPS 180-4.
pub fn sha256_hex(data: &[u8]) -> String {
    let mut h: [u32; 8] = [
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
        0x5be0cd19,
    ];
    let bit_len = (data.len() as u64).wrapping_mul(8);
    let mut padded = Vec::with_capacity(data.len() + 72);
    padded.extend_from_slice(data);
    padded.push(0x80);
    while padded.len() % 64 != 56 {
        padded.push(0);
    }
    padded.extend_from_slice(&bit_len.to_be_bytes());
    for chunk in padded.chunks_exact(64) {
        let mut w = [0_u32; 64];
        for (i, word) in chunk.chunks_exact(4).enumerate() {
            w[i] = u32::from_be_bytes([word[0], word[1], word[2], word[3]]);
        }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16]
                .wrapping_add(s0)
                .wrapping_add(w[i - 7])
                .wrapping_add(s1);
        }
        let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut hh] = h;
        for i in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ ((!e) & g);
            let t1 = hh
                .wrapping_add(s1)
                .wrapping_add(ch)
                .wrapping_add(SHA256_K[i])
                .wrapping_add(w[i]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let maj = (a & b) ^ (a & c) ^ (b & c);
            let t2 = s0.wrapping_add(maj);
            hh = g;
            g = f;
            f = e;
            e = d.wrapping_add(t1);
            d = c;
            c = b;
            b = a;
            a = t1.wrapping_add(t2);
        }
        let delta = [a, b, c, d, e, f, g, hh];
        for (slot, val) in h.iter_mut().zip(delta.iter()) {
            *slot = slot.wrapping_add(*val);
        }
    }
    let mut hex = String::with_capacity(64);
    for word in &h {
        hex.push_str(&format!("{word:08x}"));
    }
    hex
}

fn sha384(data: &[u8]) -> [u8; 48] {
    let mut h: [u64; 8] = [
        0xcbbb9d5dc1059ed8,
        0x629a292a367cd507,
        0x9159015a3070dd17,
        0x152fecd8f70e5939,
        0x67332667ffc00b31,
        0x8eb44a8768581511,
        0xdb0c2e0d64f98fa7,
        0x47b5481dbefa4fa4,
    ];
    let bit_len = (data.len() as u128).wrapping_mul(8);
    let mut padded = Vec::with_capacity(data.len() + 144);
    padded.extend_from_slice(data);
    padded.push(0x80);
    while padded.len() % 128 != 112 {
        padded.push(0);
    }
    padded.extend_from_slice(&bit_len.to_be_bytes());
    for chunk in padded.chunks_exact(128) {
        let mut w = [0_u64; 80];
        for (i, word) in chunk.chunks_exact(8).enumerate() {
            w[i] = u64::from_be_bytes([
                word[0], word[1], word[2], word[3], word[4], word[5], word[6], word[7],
            ]);
        }
        for i in 16..80 {
            let s0 = w[i - 15].rotate_right(1) ^ w[i - 15].rotate_right(8) ^ (w[i - 15] >> 7);
            let s1 = w[i - 2].rotate_right(19) ^ w[i - 2].rotate_right(61) ^ (w[i - 2] >> 6);
            w[i] = w[i - 16]
                .wrapping_add(s0)
                .wrapping_add(w[i - 7])
                .wrapping_add(s1);
        }
        let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut hh] = h;
        for i in 0..80 {
            let s1 = e.rotate_right(14) ^ e.rotate_right(18) ^ e.rotate_right(41);
            let ch = (e & f) ^ ((!e) & g);
            let t1 = hh
                .wrapping_add(s1)
                .wrapping_add(ch)
                .wrapping_add(SHA512_K[i])
                .wrapping_add(w[i]);
            let s0 = a.rotate_right(28) ^ a.rotate_right(34) ^ a.rotate_right(39);
            let maj = (a & b) ^ (a & c) ^ (b & c);
            let t2 = s0.wrapping_add(maj);
            hh = g;
            g = f;
            f = e;
            e = d.wrapping_add(t1);
            d = c;
            c = b;
            b = a;
            a = t1.wrapping_add(t2);
        }
        let delta = [a, b, c, d, e, f, g, hh];
        for (slot, val) in h.iter_mut().zip(delta.iter()) {
            *slot = slot.wrapping_add(*val);
        }
    }
    let mut digest = [0_u8; 48];
    for (i, word) in h[..6].iter().enumerate() {
        digest[i * 8..(i + 1) * 8].copy_from_slice(&word.to_be_bytes());
    }
    digest
}

fn base64_std(data: &[u8]) -> String {
    const TABLE: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(((data.len() + 2) / 3) * 4);
    for chunk in data.chunks(3) {
        let b0 = u32::from(chunk[0]);
        let b1 = u32::from(*chunk.get(1).unwrap_or(&0));
        let b2 = u32::from(*chunk.get(2).unwrap_or(&0));
        let n = (b0 << 16) | (b1 << 8) | b2;
        out.push(TABLE[((n >> 18) & 63) as usize] as char);
        out.push(TABLE[((n >> 12) & 63) as usize] as char);
        out.push(if chunk.len() > 1 {
            TABLE[((n >> 6) & 63) as usize] as char
        } else {
            '='
        });
        out.push(if chunk.len() > 2 {
            TABLE[(n & 63) as usize] as char
        } else {
            '='
        });
    }
    out
}

/// Format Subresource Integrity token (`sha384-<base64>`).
pub fn sri_digest(data: &[u8]) -> String {
    format!("sha384-{}", base64_std(&sha384(data)))
}

#[cfg_attr(pages_verify_cli, allow(dead_code))]
pub fn stamp_sri(bundle_dir: &Path) -> Result<(), String> {
    stamp_assets(bundle_dir, bundle_dir)
}

#[cfg_attr(pages_verify_cli, allow(dead_code))]
pub fn stamp_assets(html_dir: &Path, asset_root: &Path) -> Result<(), String> {
    let mut hashes: BTreeMap<String, (String, String)> = BTreeMap::new();
    for rel in SRI_ASSETS {
        let path = asset_root.join(rel);
        if path.is_file() {
            let bytes = fs::read(&path).map_err(|e| format!("read {}: {e}", path.display()))?;
            let ver = sha256_hex(&bytes)[..12].to_string();
            let sri = sri_digest(&bytes);
            hashes.insert(rel.to_string(), (ver, sri));
        }
    }

    let mut files = Vec::new();
    collect_files(html_dir, &mut files)?;
    for path in files {
        if path.extension().and_then(|e| e.to_str()) != Some("html") {
            continue;
        }
        let mut content =
            fs::read_to_string(&path).map_err(|e| format!("read {}: {e}", path.display()))?;
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
        fs::write(&path, content).map_err(|e| format!("write {}: {e}", path.display()))?;
    }
    Ok(())
}

fn collect_files(dir: &Path, out: &mut Vec<PathBuf>) -> Result<(), String> {
    let entries = fs::read_dir(dir).map_err(|e| format!("read_dir {}: {e}", dir.display()))?;
    for entry in entries {
        let entry = entry.map_err(|e| e.to_string())?;
        let name = entry.file_name();
        if name == ".jj" || name == ".git" || name == "node_modules" {
            continue;
        }
        let path = entry.path();
        if path.is_dir() {
            collect_files(&path, out)?;
        } else {
            out.push(path);
        }
    }
    out.sort();
    Ok(())
}

fn rel_posix(root: &Path, path: &Path) -> String {
    path.strip_prefix(root)
        .unwrap_or(path)
        .to_string_lossy()
        .replace('\\', "/")
}

fn check_views(target_dir: &Path) -> Result<(), String> {
    let index_html = target_dir.join("index.html");
    let content = fs::read_to_string(&index_html)
        .map_err(|e| format!("missing index.html at {}: {e}", index_html.display()))?;
    let lower = content.to_ascii_lowercase();
    if !lower.contains("<title>baux") && !lower.contains("baux") {
        return Err("index.html missing title 'baux'".into());
    }
    if !content.contains("id=\"quickstart\"") {
        return Err("index.html missing quickstart section".into());
    }
    if content.contains("Functions <span") || content.contains("Module Graph <span") {
        return Err("Found obsolete Functions or Module Graph tabs in index.html".into());
    }
    println!("[ok] Page layout: Clean baux black box optimization documentation.");
    Ok(())
}

fn check_sensitive(target_dir: &Path) -> Result<(), String> {
    let mut files = Vec::new();
    collect_files(target_dir, &mut files)?;
    let mut violations = Vec::new();
    for path in files {
        let ext = path
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("")
            .to_ascii_lowercase();
        if !matches!(ext.as_str(), "html" | "js" | "css" | "json" | "md") {
            continue;
        }
        let raw = fs::read(&path).map_err(|e| format!("read {}: {e}", path.display()))?;
        let text = String::from_utf8_lossy(&raw);
        let rel = rel_posix(target_dir, &path);
        for pat in SENSITIVE_PATTERNS {
            let count = text.matches(pat).count();
            if count > 0 {
                violations.push(format!("{rel}: matches '{pat}' ({count} occurrences)"));
            }
        }
    }
    if !violations.is_empty() {
        return Err(format!(
            "Sensitive internal text found in public bundle:\n  {}",
            violations.join("\n  ")
        ));
    }
    println!("[ok] Sensitive content check: Zero internal monograph text present.");
    Ok(())
}

fn check_assets(target_dir: &Path) -> Result<(), String> {
    for rel in REQUIRED_FILES {
        let p = target_dir.join(rel);
        let meta = fs::metadata(&p).map_err(|_| format!("Missing required bundle file: {rel}"))?;
        if !meta.is_file() {
            return Err(format!("Missing required bundle file: {rel}"));
        }
        if meta.len() == 0 && rel != ".nojekyll" {
            return Err(format!("Bundle file is empty: {rel}"));
        }
    }
    for rel in FORBIDDEN_PATHS {
        if target_dir.join(rel).exists() {
            return Err(format!(
                "Obsolete or build-only path present in bundle: {rel}"
            ));
        }
    }
    println!(
        "[ok] Link and asset integrity: All required runtime assets exist and build/legacy bloat excluded."
    );
    Ok(())
}

fn validate_csp(rel_html: &str, csp_contents: &[String]) -> Vec<String> {
    if csp_contents.is_empty() {
        return vec![format!(
            "{rel_html}: missing <meta http-equiv=\"Content-Security-Policy\">"
        )];
    }
    let mut errors = Vec::new();
    for csp in csp_contents {
        let mut directives: BTreeMap<String, Vec<String>> = BTreeMap::new();
        for raw_dir in csp.split(';') {
            let parts: Vec<&str> = raw_dir.split_whitespace().collect();
            if parts.is_empty() {
                continue;
            }
            let name = parts[0].to_ascii_lowercase();
            let entry = directives.entry(name).or_default();
            for val in &parts[1..] {
                entry.push(val.to_ascii_lowercase());
            }
        }

        let default_vals = directives.get("default-src").cloned().unwrap_or_default();
        if default_vals != ["'none'"] {
            errors.push(format!(
                "{rel_html}: CSP must set default-src 'none' (got {default_vals:?})"
            ));
        }

        let script_vals = directives.get("script-src").cloned().unwrap_or_default();
        if !script_vals.iter().any(|v| v == "'self'") {
            errors.push(format!(
                "{rel_html}: CSP script-src must include 'self' (got {script_vals:?})"
            ));
        }
        if script_vals.iter().any(|v| v == "'unsafe-inline'") {
            errors.push(format!(
                "{rel_html}: CSP script-src forbids 'unsafe-inline'"
            ));
        }
        if script_vals.iter().any(|v| v == "'unsafe-eval'") {
            errors.push(format!("{rel_html}: CSP script-src forbids 'unsafe-eval'"));
        }
        let unexpected_script: Vec<&str> = script_vals
            .iter()
            .map(String::as_str)
            .filter(|v| !matches!(*v, "'self'" | "'wasm-unsafe-eval'"))
            .collect();
        if !unexpected_script.is_empty() {
            errors.push(format!(
                "{rel_html}: CSP script-src forbids unexpected sources {unexpected_script:?}"
            ));
        }

        if let Some(worker_vals) = directives.get("worker-src") {
            if worker_vals != &["'self'".to_string()] {
                errors.push(format!(
                    "{rel_html}: CSP worker-src must equal ['self'] (got {worker_vals:?})"
                ));
            }
        }

        if let Some(connect_vals) = directives.get("connect-src") {
            if connect_vals.is_empty()
                || !connect_vals
                    .iter()
                    .all(|v| matches!(v.as_str(), "'self'" | "'none'"))
            {
                errors.push(format!(
                    "{rel_html}: CSP connect-src forbids external destinations (got {connect_vals:?})"
                ));
            }
        }

        let base_vals = directives.get("base-uri").cloned().unwrap_or_default();
        if base_vals.is_empty()
            || !base_vals
                .iter()
                .all(|v| matches!(v.as_str(), "'self'" | "'none'"))
        {
            errors.push(format!(
                "{rel_html}: CSP must include base-uri ('self' or 'none', got {base_vals:?})"
            ));
        }

        let form_vals = directives.get("form-action").cloned().unwrap_or_default();
        if form_vals != ["'none'"] {
            errors.push(format!(
                "{rel_html}: CSP must set form-action 'none' (got {form_vals:?})"
            ));
        }
    }
    errors
}

fn unescape_entities(raw: &str) -> String {
    if !raw.contains('&') {
        return raw.to_string();
    }
    let bytes = raw.as_bytes();
    let mut out = String::with_capacity(raw.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'&' {
            if let Some(rel_semi) = raw[i + 1..].find(';') {
                if rel_semi <= 10 {
                    let entity = &raw[i + 1..i + 1 + rel_semi];
                    let decoded = match entity {
                        "amp" => Some('&'),
                        "lt" => Some('<'),
                        "gt" => Some('>'),
                        "quot" => Some('"'),
                        "apos" => Some('\''),
                        _ if entity.starts_with("#x") || entity.starts_with("#X") => {
                            u32::from_str_radix(&entity[2..], 16)
                                .ok()
                                .and_then(char::from_u32)
                        }
                        _ if entity.starts_with('#') => {
                            entity[1..].parse::<u32>().ok().and_then(char::from_u32)
                        }
                        _ => None,
                    };
                    if let Some(ch) = decoded {
                        out.push(ch);
                        i += rel_semi + 2;
                        continue;
                    }
                }
            }
        }
        let ch = raw[i..].chars().next().unwrap();
        out.push(ch);
        i += ch.len_utf8();
    }
    out
}

fn percent_decode(raw: &str) -> String {
    let bytes = raw.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%' && i + 2 < bytes.len() {
            let hex = std::str::from_utf8(&bytes[i + 1..i + 3]).ok();
            if let Some(byte) = hex.and_then(|s| u8::from_str_radix(s, 16).ok()) {
                out.push(byte);
                i += 3;
                continue;
            }
        }
        out.push(bytes[i]);
        i += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}

fn has_netloc(raw_ref: &str) -> bool {
    if raw_ref.starts_with("//") {
        return true;
    }
    for ch in raw_ref.chars() {
        match ch {
            '/' | '?' | '#' => return false,
            ':' => return true,
            _ => {}
        }
    }
    false
}

fn normalize_lexical(rel: &Path) -> Option<PathBuf> {
    let mut clean = PathBuf::new();
    for comp in rel.components() {
        match comp {
            Component::CurDir | Component::RootDir | Component::Prefix(_) => {}
            Component::ParentDir => {
                if !clean.pop() {
                    return None;
                }
            }
            Component::Normal(seg) => clean.push(seg),
        }
    }
    Some(clean)
}

fn check_tagsri(
    rel_html: &str,
    html_dir_rel: &Path,
    target_root: &Path,
    tag_name: &str,
    raw_ref: &str,
    integrity: &str,
    line: usize,
    violations: &mut Vec<String>,
) {
    let sha384_tokens: Vec<&str> = integrity
        .split_whitespace()
        .filter(|t| t.starts_with("sha384-") && t.len() > "sha384-".len())
        .collect();
    if sha384_tokens.is_empty() {
        violations.push(format!(
            "{rel_html}:{line}: <{tag_name}> referencing '{raw_ref}' missing non-empty sha384-... integrity attribute"
        ));
        return;
    }
    if has_netloc(raw_ref) {
        violations.push(format!(
            "{rel_html}:{line}: <{tag_name}> references non-local asset '{raw_ref}'"
        ));
        return;
    }
    let path_only = raw_ref.split(['?', '#']).next().unwrap_or("");
    let clean_path = percent_decode(path_only);
    if clean_path.is_empty() {
        violations.push(format!(
            "{rel_html}:{line}: <{tag_name}> has empty asset path in '{raw_ref}'"
        ));
        return;
    }
    let joined = if let Some(stripped) = clean_path.strip_prefix('/') {
        PathBuf::from(stripped)
    } else {
        html_dir_rel.join(&clean_path)
    };
    let Some(rel_asset) = normalize_lexical(&joined) else {
        violations.push(format!(
            "{rel_html}:{line}: <{tag_name}> asset path escapes bundle root: '{raw_ref}'"
        ));
        return;
    };
    let asset_path = target_root.join(&rel_asset);
    let Ok(bytes) = fs::read(&asset_path) else {
        violations.push(format!(
            "{rel_html}:{line}: <{tag_name}> local asset not found: '{raw_ref}'"
        ));
        return;
    };
    let expected = sri_digest(&bytes);
    if sha384_tokens.iter().any(|tok| *tok != expected) {
        violations.push(format!(
            "{rel_html}:{line}: SRI sha384 mismatch for '{raw_ref}' (expected {expected}, got '{integrity}')"
        ));
    }
}

fn scan_html(
    rel_html: &str,
    html_dir_rel: &Path,
    target_root: &Path,
    html: &str,
    violations: &mut Vec<String>,
) -> Vec<String> {
    let mut csp_contents = Vec::new();
    let bytes = html.as_bytes();
    let mut i = 0;
    let mut line = 1usize;
    let mut in_script = false;
    let mut script_line = 0usize;
    let mut script_has_code = false;

    while i < bytes.len() {
        if in_script {
            if html[i..].len() >= 8 && html[i..i + 8].eq_ignore_ascii_case("</script") {
                let mut j = i + 8;
                while j < bytes.len() && bytes[j].is_ascii_whitespace() {
                    if bytes[j] == b'\n' {
                        line += 1;
                    }
                    j += 1;
                }
                if j < bytes.len() && bytes[j] == b'>' {
                    if script_has_code {
                        violations.push(format!(
                            "{rel_html}:{script_line}: <script> contains inline executable script content"
                        ));
                    }
                    in_script = false;
                    script_has_code = false;
                    i = j + 1;
                    continue;
                }
            }
            if bytes[i] == b'\n' {
                line += 1;
            } else if !bytes[i].is_ascii_whitespace() {
                script_has_code = true;
            }
            i += 1;
            continue;
        }

        if bytes[i] != b'<' {
            if bytes[i] == b'\n' {
                line += 1;
            }
            i += 1;
            continue;
        }

        if html[i..].starts_with("<!--") {
            let end = html[i + 4..]
                .find("-->")
                .map(|p| i + 4 + p + 3)
                .unwrap_or(bytes.len());
            line += bytes[i..end].iter().filter(|&&b| b == b'\n').count();
            i = end;
            continue;
        }

        let tag_line = line;
        let next_b = bytes.get(i + 1).copied().unwrap_or(0);
        if next_b == b'!' || next_b == b'?' {
            let end = html[i + 1..]
                .find('>')
                .map(|p| i + 1 + p + 1)
                .unwrap_or(bytes.len());
            line += bytes[i..end].iter().filter(|&&b| b == b'\n').count();
            i = end;
            continue;
        }

        let mut j = i + 1;
        let is_end_tag = bytes.get(j).copied() == Some(b'/');
        if is_end_tag {
            j += 1;
        }
        while j < bytes.len() && bytes[j].is_ascii_whitespace() {
            if bytes[j] == b'\n' {
                line += 1;
            }
            j += 1;
        }
        let name_start = j;
        while j < bytes.len()
            && !bytes[j].is_ascii_whitespace()
            && bytes[j] != b'>'
            && bytes[j] != b'/'
        {
            j += 1;
        }
        let tag_lower = html[name_start..j].to_ascii_lowercase();

        let mut attrs: Vec<(String, String)> = Vec::new();
        while j < bytes.len() {
            while j < bytes.len() && bytes[j].is_ascii_whitespace() {
                if bytes[j] == b'\n' {
                    line += 1;
                }
                j += 1;
            }
            if j >= bytes.len() || bytes[j] == b'>' {
                break;
            }
            if bytes[j] == b'/' {
                j += 1;
                continue;
            }
            let k_start = j;
            while j < bytes.len()
                && !bytes[j].is_ascii_whitespace()
                && bytes[j] != b'='
                && bytes[j] != b'>'
                && bytes[j] != b'/'
            {
                j += 1;
            }
            let raw_key = &html[k_start..j];
            while j < bytes.len() && bytes[j].is_ascii_whitespace() {
                if bytes[j] == b'\n' {
                    line += 1;
                }
                j += 1;
            }
            let val = if j < bytes.len() && bytes[j] == b'=' {
                j += 1;
                while j < bytes.len() && bytes[j].is_ascii_whitespace() {
                    if bytes[j] == b'\n' {
                        line += 1;
                    }
                    j += 1;
                }
                if j < bytes.len() && (bytes[j] == b'"' || bytes[j] == b'\'') {
                    let quote = bytes[j];
                    j += 1;
                    let v_start = j;
                    while j < bytes.len() && bytes[j] != quote {
                        if bytes[j] == b'\n' {
                            line += 1;
                        }
                        j += 1;
                    }
                    let slice = &html[v_start..j];
                    if j < bytes.len() {
                        j += 1;
                    }
                    unescape_entities(slice)
                } else {
                    let v_start = j;
                    while j < bytes.len() && !bytes[j].is_ascii_whitespace() && bytes[j] != b'>' {
                        j += 1;
                    }
                    unescape_entities(&html[v_start..j])
                }
            } else {
                String::new()
            };
            if !raw_key.is_empty() {
                attrs.push((raw_key.to_string(), val));
            }
        }
        if j < bytes.len() && bytes[j] == b'>' {
            j += 1;
        }
        i = j;

        if is_end_tag || tag_lower.is_empty() {
            continue;
        }

        let mut attr_map: BTreeMap<String, String> = BTreeMap::new();
        for (raw_k, v) in &attrs {
            let key = raw_k.to_ascii_lowercase();
            attr_map.entry(key.clone()).or_insert_with(|| v.clone());
            if key.starts_with("on") {
                violations.push(format!(
                    "{rel_html}:{tag_line}: forbidden inline event handler attribute '{raw_k}' on <{tag_lower}>"
                ));
            }
            if matches!(key.as_str(), "href" | "src") {
                let norm: String = v
                    .chars()
                    .filter(|c| (*c as u32) > 0x20)
                    .flat_map(char::to_lowercase)
                    .collect();
                if norm.starts_with("javascript:")
                    || norm.starts_with("vbscript:")
                    || norm.starts_with("file:")
                    || norm.starts_with("blob:")
                {
                    violations.push(format!(
                        "{rel_html}:{tag_line}: forbidden URL scheme in {raw_k}='{v}'"
                    ));
                } else if norm.starts_with("data:") {
                    let allowed_raster = tag_lower == "img"
                        && key == "src"
                        && (norm.starts_with("data:image/png;")
                            || norm.starts_with("data:image/jpeg;")
                            || norm.starts_with("data:image/webp;")
                            || norm.starts_with("data:image/gif;"));
                    if !allowed_raster {
                        violations.push(format!(
                            "{rel_html}:{tag_line}: forbidden data: URI in {raw_k}='{v}'"
                        ));
                    }
                } else if norm.starts_with("http://") {
                    violations.push(format!(
                        "{rel_html}:{tag_line}: cleartext http:// URL in {raw_k}='{v}'"
                    ));
                }
            }
        }

        if tag_lower == "a"
            && attr_map
                .get("target")
                .map(|v| v.trim().eq_ignore_ascii_case("_blank"))
                .unwrap_or(false)
        {
            let rel_attr = attr_map.get("rel").map(String::as_str).unwrap_or("");
            let tokens: BTreeSet<String> = rel_attr
                .split_whitespace()
                .map(|t| t.to_ascii_lowercase())
                .collect();
            if !tokens.contains("noopener") || !tokens.contains("noreferrer") {
                violations.push(format!(
                    "{rel_html}:{tag_line}: <a target=\"_blank\"> must include both noopener and noreferrer in rel (got '{rel_attr}')"
                ));
            }
        }

        if tag_lower == "meta"
            && attr_map
                .get("http-equiv")
                .map(|v| v.trim().eq_ignore_ascii_case("content-security-policy"))
                .unwrap_or(false)
        {
            csp_contents.push(attr_map.get("content").cloned().unwrap_or_default());
        }

        if tag_lower == "script" {
            in_script = true;
            script_line = tag_line;
            script_has_code = false;
            let src = attr_map.get("src").map(|s| s.trim()).unwrap_or("");
            if src.is_empty() {
                violations.push(format!(
                    "{rel_html}:{tag_line}: <script> tag lacks non-empty src attribute"
                ));
            } else {
                let integrity = attr_map.get("integrity").map(|s| s.trim()).unwrap_or("");
                check_tagsri(
                    rel_html,
                    html_dir_rel,
                    target_root,
                    "script",
                    src,
                    integrity,
                    tag_line,
                    violations,
                );
            }
        }

        if tag_lower == "link" {
            let is_stylesheet = attr_map
                .get("rel")
                .map(|v| {
                    v.split_whitespace()
                        .any(|t| t.eq_ignore_ascii_case("stylesheet"))
                })
                .unwrap_or(false);
            if is_stylesheet {
                let href = attr_map.get("href").map(|s| s.trim()).unwrap_or("");
                if href.is_empty() {
                    violations.push(format!(
                        "{rel_html}:{tag_line}: <link rel=\"stylesheet\"> missing href"
                    ));
                } else {
                    let integrity = attr_map.get("integrity").map(|s| s.trim()).unwrap_or("");
                    check_tagsri(
                        rel_html,
                        html_dir_rel,
                        target_root,
                        "link",
                        href,
                        integrity,
                        tag_line,
                        violations,
                    );
                }
            }
        }
    }

    if in_script && script_has_code {
        violations.push(format!(
            "{rel_html}:{script_line}: unclosed <script> contains inline executable script content"
        ));
    }
    csp_contents
}

/// Verify HTML security gates (CSP, SRI, URI schemes, event handlers, forbidden
/// files, leaks).
pub fn check_security(target_dir: &Path) -> Result<(), String> {
    if !target_dir.is_dir() {
        return Err(format!(
            "bundle directory missing: {}",
            target_dir.display()
        ));
    }
    let mut files = Vec::new();
    collect_files(target_dir, &mut files)?;

    let mut violations = Vec::new();
    let mut html_files = Vec::new();

    for path in files {
        let rel = rel_posix(target_dir, &path);
        let name_lower = path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("")
            .to_ascii_lowercase();
        let ext = path
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("")
            .to_ascii_lowercase();

        if matches!(
            ext.as_str(),
            "map" | "env" | "pem" | "key" | "sqlite" | "xml" | "xsl" | "sh" | "ml" | "mli" | "toml"
        ) || name_lower == ".env"
            || name_lower.starts_with(".env.")
        {
            violations.push(format!("{rel}: forbidden file type in public bundle"));
        }

        let raw = fs::read(&path).map_err(|e| format!("read {}: {e}", path.display()))?;
        let is_known_text = matches!(
            ext.as_str(),
            "html"
                | "htm"
                | "js"
                | "mjs"
                | "cjs"
                | "css"
                | "json"
                | "md"
                | "xml"
                | "xsl"
                | "svg"
                | "txt"
                | "sh"
                | "toml"
                | "yaml"
                | "yml"
        );
        let decoded: Option<String> = if is_known_text {
            Some(String::from_utf8_lossy(&raw).into_owned())
        } else if !raw.contains(&0) {
            String::from_utf8(raw).ok()
        } else {
            None
        };
        if let Some(text) = &decoded {
            for marker in USER_MARKERS {
                if text.contains(marker) {
                    violations.push(format!(
                        "{rel}: leaks local build-machine path ({marker}...)"
                    ));
                }
            }
        }
        if ext == "html" {
            html_files.push(path);
        }
    }

    if html_files.is_empty() {
        return Err(format!(
            "No HTML files found in bundle directory: {}",
            target_dir.display()
        ));
    }

    for html_path in html_files {
        let rel_html = rel_posix(target_dir, &html_path);
        let rel_dir = html_path
            .strip_prefix(target_dir)
            .unwrap_or(&html_path)
            .parent()
            .unwrap_or_else(|| Path::new(""));
        let html_bytes =
            fs::read(&html_path).map_err(|e| format!("read {}: {e}", html_path.display()))?;
        let html = String::from_utf8_lossy(&html_bytes);
        let csps = scan_html(&rel_html, rel_dir, target_dir, &html, &mut violations);
        violations.extend(validate_csp(&rel_html, &csps));
    }

    if !violations.is_empty() {
        return Err(format!(
            "Security verification failed for public bundle:\n  {}",
            violations.join("\n  ")
        ));
    }
    println!(
        "[ok] Security gates: CSP, zero inline scripts, SHA-384 SRI, link/scheme hygiene, and secret hygiene verified."
    );
    Ok(())
}

fn check_smoke(target_dir: &Path) -> Result<(), String> {
    let mut js_files = Vec::new();
    let assets_dir = target_dir.join("assets");
    if assets_dir.is_dir() {
        for entry in fs::read_dir(&assets_dir).map_err(|e| e.to_string())? {
            let entry = entry.map_err(|e| e.to_string())?;
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) == Some("js") {
                js_files.push(path);
            }
        }
        js_files.sort();
    }
    for root_js in ["forester.js", "coi-serviceworker.js"] {
        let p = target_dir.join(root_js);
        if p.is_file() {
            js_files.push(p);
        }
    }
    for js in js_files {
        let status = Command::new("node")
            .arg("-c")
            .arg(&js)
            .status()
            .map_err(|e| format!("run node -c {}: {e}", js.display()))?;
        if !status.success() {
            return Err(format!("node -c {} exited with {status}", js.display()));
        }
    }
    println!("[ok] JavaScript syntax: All assets pass node syntax verification.");
    Ok(())
}

/// Full public Pages bundle verification (views, sensitive patterns, assets,
/// security, JS syntax).
pub fn verify_bundle(target_dir: &Path) -> Result<(), String> {
    if !target_dir.is_dir() {
        return Err(format!(
            "bundle directory missing: {}",
            target_dir.display()
        ));
    }
    println!("Verifying public bundle: {}", target_dir.display());
    check_views(target_dir)?;
    check_sensitive(target_dir)?;
    check_assets(target_dir)?;
    check_security(target_dir)?;
    check_smoke(target_dir)?;
    println!("Public bundle verification passed.");
    Ok(())
}

fn tracked_paths(root: &Path) -> Result<Vec<String>, String> {
    let output = if root.join(".jj").exists() {
        Command::new("jj")
            .current_dir(root)
            .args(["file", "list", "-T", "path ++ \"\\0\""])
            .output()
            .map_err(|e| format!("start jj file list: {e}"))?
    } else {
        Command::new("git")
            .current_dir(root)
            .args(["ls-files", "-z"])
            .output()
            .map_err(|e| format!("start git ls-files: {e}"))?
    };
    if !output.status.success() {
        return Err(format!(
            "list tracked files failed: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        ));
    }
    Ok(output
        .stdout
        .split(|&b| b == 0)
        .filter(|s| !s.is_empty())
        .map(|s| String::from_utf8_lossy(s).replace('\\', "/"))
        .collect())
}

fn check_gitbypass(root: &Path, paths: &[String]) -> Result<(), String> {
    if !root.join(".git/info/exclude").is_file() {
        return Ok(());
    }
    let mut child = Command::new("git")
        .current_dir(root)
        .args(["check-ignore", "--no-index", "-v", "-z", "--stdin"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| format!("start git check-ignore: {e}"))?;

    if let Some(mut stdin) = child.stdin.take() {
        for p in paths {
            stdin
                .write_all(p.as_bytes())
                .and_then(|()| stdin.write_all(&[0]))
                .map_err(|e| format!("write git check-ignore stdin: {e}"))?;
        }
    }
    let output = child
        .wait_with_output()
        .map_err(|e| format!("wait git check-ignore: {e}"))?;
    // Exit status 0 means one or more paths matched an ignore/exclude pattern;
    // 1 means none matched.
    if output.status.code() == Some(1) {
        return Ok(());
    }
    if !output.status.success() {
        return Err(format!(
            "git check-ignore failed: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        ));
    }
    let fields: Vec<String> = output
        .stdout
        .split(|&b| b == 0)
        .map(|s| String::from_utf8_lossy(s).into_owned())
        .collect();
    let mut matches = Vec::new();
    for tuple in fields.chunks(4) {
        if tuple.len() == 4 && !tuple[3].is_empty() {
            matches.push(format!(
                "{} (matched {}:{}: {})",
                tuple[3], tuple[0], tuple[1], tuple[2]
            ));
        }
    }
    if matches.is_empty() && !output.stdout.is_empty() {
        matches.push(String::from_utf8_lossy(&output.stdout).trim().to_string());
    }
    if !matches.is_empty() {
        return Err(format!(
            "Tracked file(s) match ignore/exclude rules (bypass via force-track):\n  {}",
            matches.join("\n  ")
        ));
    }
    Ok(())
}

/// Core pure validator over a list of repo-relative tracked POSIX paths.
pub fn check_publictree(root: &Path, paths: &[String]) -> Result<(), String> {
    let allowed_text: BTreeSet<&str> = TEXT_ALLOWLIST.iter().copied().collect();
    let retired: BTreeSet<&str> = RETIRED_SCRIPTS.iter().copied().collect();
    let mut violations = Vec::new();

    for rel in paths {
        if rel == "docs" || rel.starts_with("docs/") {
            violations.push(format!("{rel}: forbidden tracked documentation path"));
        }
        if retired.contains(rel.as_str()) {
            violations.push(format!(
                "{rel}: retired Python/manifest website script must remain absent"
            ));
        }
        let file_name = rel.rsplit('/').next().unwrap_or(rel);
        let lower_name = file_name.to_ascii_lowercase();
        let ext = file_name
            .rsplit_once('.')
            .map(|(_, e)| e.to_ascii_lowercase())
            .unwrap_or_default();

        if ext == "py"
            && (rel.starts_with(".github/")
                || rel.starts_with("forest/")
                || rel.starts_with("admin/ui/"))
        {
            violations.push(format!(
                "{rel}: unexpected Python script inside website/CI path"
            ));
        }

        if lower_name == ".env"
            || lower_name.starts_with(".env.")
            || (lower_name.starts_with("agent-") && ext == "log")
            || matches!(
                ext.as_str(),
                "pdf"
                    | "tex"
                    | "sty"
                    | "bib"
                    | "bbl"
                    | "aux"
                    | "toc"
                    | "lof"
                    | "lot"
                    | "fls"
                    | "fdb_latexmk"
                    | "log"
                    | "sqlite"
                    | "pem"
                    | "key"
            )
        {
            violations.push(format!(
                "{rel}: forbidden non-public artifact extension/name"
            ));
        }

        if matches!(ext.as_str(), "md" | "txt") && !allowed_text.contains(rel.as_str()) {
            violations.push(format!(
                "{rel}: unapproved .md/.txt file in tracked public tree"
            ));
        }

        let scan_public_text = rel == "README.md"
            || rel.starts_with("forest/trees/")
            || rel.starts_with("forest/theme/")
            || rel.starts_with("admin/ui/")
            || rel.starts_with(".github/");
        if scan_public_text {
            let full = root.join(rel);
            if full.is_file() {
                if let Ok(raw) = fs::read(&full) {
                    let text = String::from_utf8_lossy(&raw);
                    for marker in USER_MARKERS {
                        if text.contains(marker) {
                            violations.push(format!(
                                "{rel}: tracked public source contains local path marker ({marker}...)"
                            ));
                        }
                    }
                }
            }
        }
    }

    if !violations.is_empty() {
        return Err(format!(
            "Public repository hygiene violation(s):\n  {}",
            violations.join("\n  ")
        ));
    }
    Ok(())
}

/// Enforce strict public-tree allowlist and verify no ignored/excluded files
/// were force-tracked.
pub fn verify_publictree(root: &Path) -> Result<(), String> {
    let paths = tracked_paths(root)?;
    check_publictree(root, &paths)?;
    check_gitbypass(root, &paths)?;
    println!(
        "[ok] Public tree hygiene: {} tracked files pass allowlist and exclude gates.",
        paths.len()
    );
    Ok(())
}

#[cfg(pages_verify_cli)]
fn main() -> std::process::ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() {
        eprintln!(
            "Usage: verify-pages [--tree <REPO>] [--bundle <DIR>] [--security-only] [--stamp-sri <DIR>]"
        );
        return std::process::ExitCode::FAILURE;
    }

    let mut tree_dir: Option<PathBuf> = None;
    let mut bundle_dir: Option<PathBuf> = None;
    let mut stamp_dir: Option<PathBuf> = None;
    let mut security_only = false;
    let mut i = 0;

    while i < args.len() {
        match args[i].as_str() {
            "--tree" => {
                i += 1;
                let Some(val) = args.get(i) else {
                    eprintln!("--tree requires a path");
                    return std::process::ExitCode::FAILURE;
                };
                tree_dir = Some(PathBuf::from(val));
            }
            "--bundle" => {
                i += 1;
                let Some(val) = args.get(i) else {
                    eprintln!("--bundle requires a path");
                    return std::process::ExitCode::FAILURE;
                };
                bundle_dir = Some(PathBuf::from(val));
            }
            "--stamp-sri" => {
                i += 1;
                let Some(val) = args.get(i) else {
                    eprintln!("--stamp-sri requires a path");
                    return std::process::ExitCode::FAILURE;
                };
                stamp_dir = Some(PathBuf::from(val));
            }
            "--security-only" => security_only = true,
            flag if flag.starts_with('-') => {
                eprintln!("unknown option: {flag}");
                return std::process::ExitCode::FAILURE;
            }
            positional => {
                bundle_dir = Some(PathBuf::from(positional));
            }
        }
        i += 1;
    }

    let res = (|| -> Result<(), String> {
        if let Some(root) = &tree_dir {
            verify_publictree(root)?;
        }
        if let Some(dir) = &stamp_dir {
            stamp_sri(dir)?;
        }
        if let Some(dir) = &bundle_dir {
            if security_only {
                check_security(dir)?;
            } else {
                verify_bundle(dir)?;
            }
        }
        Ok(())
    })();

    match res {
        Ok(()) => std::process::ExitCode::SUCCESS,
        Err(err) => {
            eprintln!("error: {err}");
            std::process::ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    const VALID_CSP: &str = "default-src 'none'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; font-src 'self'; connect-src 'self'; base-uri 'self'; form-action 'none'";

    struct TempDir(PathBuf);

    impl TempDir {
        fn new(label: &str) -> Self {
            let nanos = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("clock")
                .as_nanos();
            let dir = std::env::temp_dir()
                .join(format!("baux-pv-{label}-{}-{nanos}", std::process::id()));
            fs::create_dir_all(&dir).expect("mkdir temp");
            Self(dir)
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    fn write_synthetic(dest: &Path) {
        fs::create_dir_all(dest.join("assets")).expect("mkdir assets");
        fs::write(dest.join("style.css"), "body { margin: 0; }\n").expect("style");
        fs::write(dest.join("katex.min.css"), ".katex { font-size: 1em; }\n").expect("katex");
        fs::write(dest.join("forester.js"), "export const ok = true;\n").expect("forester");
        fs::write(dest.join("assets/baux_ui.js"), "console.log('baux');\n").expect("ui");
        fs::write(
            dest.join("index.html"),
            format!(
                "<!DOCTYPE html>\n<html xmlns=\"http://www.w3.org/1999/xhtml\">\n<head>\n  <meta http-equiv=\"Content-Security-Policy\" content=\"{VALID_CSP}\" />\n  <link rel=\"stylesheet\" href=\"style.css\"></link>\n  <link rel=\"stylesheet\" href=\"katex.min.css\" />\n  <script type=\"module\" src=\"forester.js\"></script>\n  <script type=\"text/javascript\" src=\"assets/baux_ui.js\" defer=\"defer\"></script>\n  <title>Baux</title>\n</head>\n<body>\n  <a href=\"https://github.com/Kvutza/baux\" target=\"_blank\" rel=\"noopener noreferrer\">GitHub</a>\n</body>\n</html>\n"
            ),
        )
        .expect("index");
        stamp_sri(dest).expect("stamp_sri");
    }

    #[test]
    fn digest_vectors() {
        assert_eq!(
            sha256_hex(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
        assert_eq!(
            sri_digest(b"abc"),
            "sha384-ywB1P0WjXou1oD1pmsZQBycsMqsO3tFjGotgWkP/W+2AhgcroefMI1i67KE0yCWn"
        );
    }

    #[test]
    fn security_regressions() {
        let tmp = TempDir::new("sec");
        write_synthetic(&tmp.0);
        check_security(&tmp.0).expect("valid synthetic bundle");

        let index = fs::read_to_string(tmp.0.join("index.html")).expect("read index");
        stamp_sri(&tmp.0).expect("idempotent stamp");
        assert_eq!(
            fs::read_to_string(tmp.0.join("index.html")).expect("read stamped"),
            index
        );

        let tamper_dir = TempDir::new("tamper");
        write_synthetic(&tamper_dir.0);
        fs::write(
            tamper_dir.0.join("assets/baux_ui.js"),
            "console.log('x');\n",
        )
        .unwrap();
        assert!(
            check_security(&tamper_dir.0)
                .unwrap_err()
                .contains("sha384")
        );

        let onclick_dir = TempDir::new("onclick");
        write_synthetic(&onclick_dir.0);
        let h = fs::read_to_string(onclick_dir.0.join("index.html")).unwrap();
        fs::write(
            onclick_dir.0.join("index.html"),
            h.replace("<body>", "<body onclick=\"alert(1)\">"),
        )
        .unwrap();
        assert!(
            check_security(&onclick_dir.0)
                .unwrap_err()
                .contains("onclick")
        );

        let svg_dir = TempDir::new("svgdata");
        write_synthetic(&svg_dir.0);
        let h = fs::read_to_string(svg_dir.0.join("index.html")).unwrap();
        fs::write(
            svg_dir.0.join("index.html"),
            h.replace(
                "</body>",
                "<img src=\"data:image/svg+xml;base64,QQ==\" /></body>",
            ),
        )
        .unwrap();
        assert!(check_security(&svg_dir.0).unwrap_err().contains("data:"));

        let wasm_csp = TempDir::new("wasmcsp");
        write_synthetic(&wasm_csp.0);
        let h = fs::read_to_string(wasm_csp.0.join("index.html")).unwrap();
        fs::write(
            wasm_csp.0.join("index.html"),
            h.replace(
                "script-src 'self';",
                "script-src 'self' 'wasm-unsafe-eval'; worker-src 'self';",
            ),
        )
        .unwrap();
        check_security(&wasm_csp.0).expect("wasm-unsafe-eval with self worker allowed");
    }

    #[test]
    fn publictree_allowlist() {
        let tmp = TempDir::new("tree");
        let good = vec![
            "README.md".to_string(),
            "forest/theme/LICENSES/MIT.txt".to_string(),
            "rust/crates/cli/src/bin/baux/pages.rs".to_string(),
        ];
        check_publictree(&tmp.0, &good).expect("allowlisted files pass");

        let mut bad_doc = good.clone();
        bad_doc.push("docs/internal-notes.md".to_string());
        assert!(check_publictree(&tmp.0, &bad_doc).is_err());

        let mut bad_md = good.clone();
        bad_md.push("admin/notes.md".to_string());
        assert!(check_publictree(&tmp.0, &bad_md).is_err());

        let mut bad_py = good;
        bad_py.push("admin/ops/verify_pages.py".to_string());
        assert!(check_publictree(&tmp.0, &bad_py).is_err());
    }
}
