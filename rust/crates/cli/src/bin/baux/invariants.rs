//! Zero-dependency cross-boundary invariants verifier for `baux`.
//!
//! Compiles both as `mod invariants;` inside `./baux check` and standalone in
//! fast CI via `rustc --edition 2024 -O -D warnings --cfg invariants_cli`.

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::Path;
#[cfg(invariants_cli)]
use std::path::PathBuf;

#[derive(Clone, Copy, Debug, Default)]
pub struct Options {
    pub strict_extraction: bool,
}

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

fn sha256_hex(data: &[u8]) -> String {
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
        for (slot, delta) in h.iter_mut().zip([a, b, c, d, e, f, g, hh]) {
            *slot = slot.wrapping_add(delta);
        }
    }
    let mut out = String::with_capacity(64);
    for word in h {
        for b in word.to_be_bytes() {
            out.push_str(&format!("{b:02x}"));
        }
    }
    out
}

fn file_sha256(path: &Path) -> Result<String, String> {
    let data = fs::read(path).map_err(|e| format!("read {}: {e}", path.display()))?;
    Ok(sha256_hex(&data))
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

/// Canonical tree SHA-256 matching
/// `baux_tooling::enn_artifact::cargo::tree_digest`.
fn tree_sha256(dir: &Path) -> Result<String, String> {
    let mut files = Vec::new();
    walk_files(dir, dir, &mut files)?;
    files.sort();
    let mut framed = Vec::new();
    for rel in files {
        let name = rel.as_bytes();
        let data = fs::read(dir.join(&rel)).map_err(|e| format!("read {rel}: {e}"))?;
        framed.extend_from_slice(&(name.len() as u64).to_le_bytes());
        framed.extend_from_slice(name);
        framed.extend_from_slice(&(data.len() as u64).to_le_bytes());
        framed.extend_from_slice(&data);
    }
    Ok(sha256_hex(&framed))
}

fn em_sig(ty: &str) -> char {
    let t = ty.trim();
    if t == "()" {
        'v'
    } else if t == "f32" {
        'f'
    } else if t == "f64" {
        'd'
    } else if t == "i64" || t == "u64" {
        'j'
    } else {
        // i32, u32, usize, isize, bool, and raw pointers map to i32 under
        // wasm32.
        'i'
    }
}

fn strip_comments(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    for line in text.lines() {
        let clean = line.split("//").next().unwrap_or("");
        out.push_str(clean);
        out.push(' ');
    }
    out
}

/// Extract `(symbol_name, expected_emscripten_sig)` from `extern "C" { ... }`
/// in `webgpu.rs`.
pub fn scan_rust(rust_src: &str) -> Result<BTreeMap<String, String>, String> {
    let clean = strip_comments(rust_src);
    let mut rest = clean.as_str();
    let mut symbols = BTreeMap::new();
    while let Some(pos) = rest.find("pub fn bauxgpu_") {
        let after_fn = &rest[pos + "pub fn ".len()..];
        let Some(lparen) = after_fn.find('(') else {
            return Err("malformed pub fn bauxgpu_* missing '('".into());
        };
        let name = after_fn[..lparen].trim().to_string();
        let after_lp = &after_fn[lparen + 1..];
        let Some(rparen) = after_lp.find(')') else {
            return Err(format!("malformed parameter list for {name}"));
        };
        let params_str = &after_lp[..rparen];
        let after_rp = &after_lp[rparen + 1..];
        let Some(semi) = after_rp.find(';') else {
            return Err(format!("missing ';' after signature for {name}"));
        };
        let ret_part = after_rp[..semi].trim();
        let ret_char = match ret_part.strip_prefix("->") {
            Some(ret_ty) => em_sig(ret_ty),
            None if ret_part.is_empty() => 'v',
            None => return Err(format!("unexpected return clause for {name}: {ret_part}")),
        };
        let mut sig = String::new();
        sig.push(ret_char);
        for param in params_str.split(',') {
            let p = param.trim();
            if p.is_empty() {
                continue;
            }
            let Some((_, ty)) = p.split_once(':') else {
                return Err(format!("malformed param '{p}' in {name}"));
            };
            sig.push(em_sig(ty));
        }
        symbols.insert(name, sig);
        rest = &after_rp[semi + 1..];
    }
    if symbols.is_empty() {
        return Err("no `pub fn bauxgpu_*` externs parsed from webgpu.rs".into());
    }
    Ok(symbols)
}

fn quoted_after<'a>(text: &'a str, needle: &str) -> Option<(String, &'a str)> {
    let pos = text.find(needle)?;
    let tail = &text[pos + needle.len()..];
    let quote = tail.chars().next()?;
    if quote != '"' && quote != '\'' {
        return None;
    }
    let inner = &tail[1..];
    let end = inner.find(quote)?;
    Some((inner[..end].to_string(), &inner[end + 1..]))
}

/// Extract `.sig = "..."` bindings for `bauxgpu_*` functions from JS worker
/// code.
pub fn scan_js(js_src: &str) -> Result<BTreeMap<String, String>, String> {
    let mut sigs = BTreeMap::new();
    let mut cursor = js_src;
    while let Some(pos) = cursor.find("bauxgpu_") {
        let tail = &cursor[pos..];
        let Some(dot_sig) = tail.find(".sig") else {
            break;
        };
        // Only inspect direct `<ident>.sig = "..."` assignments.
        let ident = tail[..dot_sig].trim();
        if ident.starts_with("bauxgpu_")
            && ident.chars().all(|c| c.is_ascii_alphanumeric() || c == '_')
        {
            let after_sig = tail[dot_sig + ".sig".len()..].trim_start();
            if let Some(after_eq) = after_sig.strip_prefix('=') {
                let trimmed = after_eq.trim_start();
                if let Some((sig, _)) = quoted_after(trimmed, "") {
                    sigs.insert(ident.to_string(), sig);
                }
            }
        }
        cursor = &tail["bauxgpu_".len()..];
    }
    Ok(sigs)
}

/// Extract `bauxgpu_<name>: makeStub("<sig>", ...)` stubs from
/// `tests/test_repl.py`.
pub fn scan_stubs(py_src: &str) -> Result<BTreeMap<String, String>, String> {
    let mut stubs = BTreeMap::new();
    for line in py_src.lines() {
        let t = line.trim();
        let Some((lhs, rhs)) = t.split_once(':') else {
            continue;
        };
        let name = lhs.trim();
        if !name.starts_with("bauxgpu_") {
            continue;
        }
        if let Some((sig, _)) = quoted_after(rhs.trim_start(), "makeStub(") {
            stubs.insert(name.to_string(), sig);
        }
    }
    Ok(stubs)
}

/// Enforce 3-way ABI alignment across Rust FFI (`webgpu.rs`), JS worker
/// (`repl_worker.js`), and browser REPL test stubs (`tests/test_repl.py`).
pub fn check_webgpu(root: &Path) -> Result<usize, String> {
    let rust_path = root.join("rust/crates/baux/src/webgpu.rs");
    let js_path = root.join("forest/theme/repl_worker.js");
    let test_path = root.join("tests/test_repl.py");

    let rust_src =
        fs::read_to_string(&rust_path).map_err(|e| format!("read {}: {e}", rust_path.display()))?;
    let js_src =
        fs::read_to_string(&js_path).map_err(|e| format!("read {}: {e}", js_path.display()))?;
    let test_src =
        fs::read_to_string(&test_path).map_err(|e| format!("read {}: {e}", test_path.display()))?;

    let rust_syms = scan_rust(&rust_src)?;
    let js_sigs = scan_js(&js_src)?;
    let test_stubs = scan_stubs(&test_src)?;

    let mut errors = Vec::new();
    for (name, rust_sig) in &rust_syms {
        if !js_src.contains(&format!("function {name}(")) {
            errors.push(format!("repl_worker.js: missing `function {name}(...)`"));
        }
        if !js_src.contains(&format!("mod.{name} = {name}")) {
            errors.push(format!(
                "repl_worker.js: missing `mod.{name} = {name}` registration"
            ));
        }
        match js_sigs.get(name) {
            Some(js_sig) if js_sig == rust_sig => {}
            Some(js_sig) => errors.push(format!(
                "ABI mismatch for {name}: webgpu.rs={rust_sig:?} vs repl_worker.js={js_sig:?}"
            )),
            None => errors.push(format!("repl_worker.js: missing `{name}.sig`")),
        }
        match test_stubs.get(name) {
            Some(stub_sig) if stub_sig == rust_sig => {}
            Some(stub_sig) => errors.push(format!(
                "ABI mismatch for {name}: webgpu.rs={rust_sig:?} vs test_repl.py={stub_sig:?}"
            )),
            None => errors.push(format!("tests/test_repl.py: missing stub for `{name}`")),
        }
    }

    for extra in js_sigs.keys() {
        if !rust_syms.contains_key(extra) {
            errors.push(format!(
                "repl_worker.js defines `{extra}.sig` absent from webgpu.rs"
            ));
        }
    }
    for extra in test_stubs.keys() {
        if !rust_syms.contains_key(extra) {
            errors.push(format!(
                "tests/test_repl.py stubs `{extra}` absent from webgpu.rs"
            ));
        }
    }

    if !errors.is_empty() {
        return Err(format!(
            "WebGPU 3-way ABI verification failed:\n  {}",
            errors.join("\n  ")
        ));
    }
    Ok(rust_syms.len())
}

/// Tokenize Lean code outside `--` line comments, nested `/- ... -/` comments,
/// and string literals, returning any forbidden proof-hole keyword matches.
pub fn lean_holes(src: &str) -> Vec<(usize, String)> {
    let bytes = src.as_bytes();
    let len = bytes.len();
    let mut i = 0;
    let mut line = 1_usize;
    let mut block_depth = 0_usize;
    let mut in_line_comment = false;
    let mut in_string = false;
    let mut holes = Vec::new();

    while i < len {
        let b = bytes[i];
        if b == b'\n' {
            line += 1;
            in_line_comment = false;
            i += 1;
            continue;
        }
        if in_line_comment {
            i += 1;
            continue;
        }
        if in_string {
            if b == b'\\' && i + 1 < len {
                if bytes[i + 1] == b'\n' {
                    line += 1;
                }
                i += 2;
                continue;
            }
            if b == b'"' {
                in_string = false;
            }
            i += 1;
            continue;
        }
        if i + 1 < len && b == b'/' && bytes[i + 1] == b'-' {
            block_depth += 1;
            i += 2;
            continue;
        }
        if block_depth > 0 {
            if i + 1 < len && b == b'-' && bytes[i + 1] == b'/' {
                block_depth -= 1;
                i += 2;
            } else {
                i += 1;
            }
            continue;
        }
        if i + 1 < len && b == b'-' && bytes[i + 1] == b'-' {
            in_line_comment = true;
            i += 2;
            continue;
        }
        if b == b'"' {
            in_string = true;
            i += 1;
            continue;
        }
        if b.is_ascii_alphabetic() || b == b'_' {
            let start = i;
            while i < len
                && (bytes[i].is_ascii_alphanumeric() || bytes[i] == b'_' || bytes[i] == b'.')
            {
                i += 1;
            }
            let ident = &src[start..i];
            if matches!(
                ident,
                "sorry" | "admit" | "axiom" | "implemented_by" | "unsafe"
            ) {
                holes.push((line, ident.to_string()));
            }
            continue;
        }
        i += 1;
    }
    holes
}

fn toml_str(text: &str, section: Option<&str>, key: &str) -> Option<String> {
    let mut active_section = "";
    for line in text.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        if trimmed.starts_with('[') && !trimmed.starts_with("[[") && trimmed.ends_with(']') {
            active_section = &trimmed[1..trimmed.len() - 1];
            continue;
        }
        if let Some(wanted) = section {
            if active_section != wanted {
                continue;
            }
        } else if !active_section.is_empty() {
            continue;
        }
        let Some((lhs, rhs)) = trimmed.split_once('=') else {
            continue;
        };
        if lhs.trim() != key {
            continue;
        }
        let rhs = rhs.trim();
        if rhs.starts_with('"') && rhs.len() >= 2 && rhs.ends_with('"') {
            return Some(rhs[1..rhs.len() - 1].to_string());
        }
    }
    None
}

/// Check Lean files for axioms/sorry/unsafe and verify cryptographic digests in
/// `research/extraction/{box.toml,artifact_manifest.toml}`.
pub fn check_lean(root: &Path, strict_source_bindings: bool) -> Result<(usize, usize), String> {
    let lean_dir = root.join("research/lean");
    if !lean_dir.is_dir() {
        return Err("missing research/lean directory".into());
    }
    let mut lean_files = Vec::new();
    walk_files(&lean_dir, &lean_dir, &mut lean_files)?;
    lean_files.sort();
    let mut checked_lean = 0_usize;
    let mut proof_errors = Vec::new();
    for rel in lean_files {
        if !rel.ends_with(".lean") || rel.starts_with(".lake/") {
            continue;
        }
        checked_lean += 1;
        let full = lean_dir.join(&rel);
        let text = fs::read_to_string(&full).map_err(|e| format!("read {rel}: {e}"))?;
        for (line, kw) in lean_holes(&text) {
            proof_errors.push(format!(
                "research/lean/{rel}:{line}: forbidden `{kw}` token"
            ));
        }
    }
    if !proof_errors.is_empty() {
        return Err(format!(
            "Lean proof hygiene violation(s):\n  {}",
            proof_errors.join("\n  ")
        ));
    }

    let box_path = root.join("research/extraction/box.toml");
    let box_text =
        fs::read_to_string(&box_path).map_err(|e| format!("read {}: {e}", box_path.display()))?;
    let man_path = root.join("research/extraction/artifact_manifest.toml");
    let man_text =
        fs::read_to_string(&man_path).map_err(|e| format!("read {}: {e}", man_path.display()))?;

    let mut hard_errors = Vec::new();
    let mut stale_warnings = Vec::new();
    let pin_checks = [
        (
            "box.toml pin_sha256",
            toml_str(&box_text, None, "pin_sha256"),
            "research/upstream/enn.lock.toml",
        ),
        (
            "box.toml adapter_sha256",
            toml_str(&box_text, None, "adapter_sha256"),
            "rust/crates/tooling/src/box_pair.rs",
        ),
        (
            "artifact_manifest coverage_manifest_sha256",
            toml_str(
                &man_text,
                Some("source_bindings"),
                "coverage_manifest_sha256",
            ),
            "research/upstream/coverage_manifest.toml",
        ),
        (
            "artifact_manifest charon_callsite_patch_sha256",
            toml_str(
                &man_text,
                Some("source_bindings"),
                "charon_callsite_patch_sha256",
            ),
            "research/extraction/charon-callsite.patch",
        ),
        (
            "artifact_manifest aeneas_float_patch_sha256",
            toml_str(
                &man_text,
                Some("source_bindings"),
                "aeneas_float_patch_sha256",
            ),
            "research/extraction/aeneas-float.patch",
        ),
        (
            "artifact_manifest aeneas_dyn_patch_sha256",
            toml_str(
                &man_text,
                Some("source_bindings"),
                "aeneas_dyn_patch_sha256",
            ),
            "research/extraction/aeneas-dyn.patch",
        ),
        (
            "artifact_manifest aeneas_dyn_dispatch_patch_sha256",
            toml_str(
                &man_text,
                Some("source_bindings"),
                "aeneas_dyn_dispatch_patch_sha256",
            ),
            "research/extraction/aeneas-dyn-dispatch.patch",
        ),
        (
            "artifact_manifest box_pair_metadata_sha256",
            toml_str(
                &man_text,
                Some("source_bindings"),
                "box_pair_metadata_sha256",
            ),
            "research/extraction/box.toml",
        ),
    ];
    for (label, expected, rel_file) in &pin_checks {
        let Some(exp) = expected else {
            hard_errors.push(format!("{label}: missing field in manifest"));
            continue;
        };
        let actual = file_sha256(&root.join(rel_file))?;
        if actual != *exp {
            hard_errors.push(format!(
                "{label} mismatch vs {rel_file} (expected {exp}, actual {actual})"
            ));
        }
    }

    if let Some(exp) = toml_str(&box_text, Some("local"), "source_sha256") {
        let actual = file_sha256(&root.join("rust/crates/baux/semantics/src/random.rs"))?;
        if actual != *exp {
            stale_warnings.push(format!("box.toml local.source_sha256 differs from rust/crates/baux/semantics/src/random.rs ({} vs {})", &exp[..12], &actual[..12]));
        }
    }
    let tree_checks = [
        ("local_baux_src_tree_sha256", "rust/crates/baux/src"),
        (
            "local_semantics_src_tree_sha256",
            "rust/crates/baux/semantics/src",
        ),
        (
            "local_neighbor_index_src_tree_sha256",
            "rust/crates/baux/neighbor-index/src",
        ),
    ];
    for (key, rel_dir) in tree_checks {
        if let Some(exp) = toml_str(&man_text, Some("source_bindings"), key) {
            let actual = tree_sha256(&root.join(rel_dir))?;
            if actual != *exp {
                stale_warnings.push(format!(
                    "artifact_manifest.source_bindings.{key} differs from {rel_dir} ({} vs {})",
                    &exp[..12],
                    &actual[..12]
                ));
            }
        }
    }

    if !stale_warnings.is_empty() {
        if strict_source_bindings {
            hard_errors.extend(stale_warnings);
        } else {
            for w in &stale_warnings {
                println!(
                    "[warn] extraction binding stale (run `./baux research check` to re-seal): {w}"
                );
            }
        }
    }
    if !hard_errors.is_empty() {
        return Err(format!(
            "Extraction integrity violation(s):\n  {}",
            hard_errors.join("\n  ")
        ));
    }
    Ok((checked_lean, pin_checks.len()))
}

fn cargo_members(cargo_toml: &str) -> Vec<String> {
    let mut members = Vec::new();
    let mut in_members = false;
    for line in cargo_toml.lines() {
        let t = line.trim();
        if t.starts_with("members = [") {
            in_members = true;
            continue;
        }
        if in_members {
            if t.starts_with(']') {
                break;
            }
            if let Some((path, _)) = quoted_after(t, "") {
                members.push(path);
            }
        }
    }
    members
}

/// Verify workspace `Cargo.toml`, `admin/targets.bzl` projections, and
/// `admin/tools/*` script refs.
pub fn check_parity(root: &Path) -> Result<usize, String> {
    let cargo_path = root.join("Cargo.toml");
    let targets_path = root.join("admin/targets.bzl");
    let cargo_text =
        fs::read_to_string(&cargo_path).map_err(|e| format!("read Cargo.toml: {e}"))?;
    let targets_text =
        fs::read_to_string(&targets_path).map_err(|e| format!("read admin/targets.bzl: {e}"))?;

    let mut errors = Vec::new();
    let cargo_edition = toml_str(&cargo_text, Some("workspace.package"), "edition")
        .ok_or("Cargo.toml missing [workspace.package].edition")?;
    if !targets_text.contains(&format!("RUST_EDITION = \"{cargo_edition}\"")) {
        errors.push(format!(
            "admin/targets.bzl RUST_EDITION disagrees with Cargo.toml ({cargo_edition}); run admin/tools/build-parity --sync"
        ));
    }
    if !cargo_text.contains("[workspace.lints.rust]")
        || !cargo_text.contains("[workspace.lints.clippy]")
    {
        errors.push(
            "Cargo.toml must define [workspace.lints.rust] and [workspace.lints.clippy]".into(),
        );
    }

    let members = cargo_members(&cargo_text);
    for member in &members {
        if !root.join(member).join("Cargo.toml").is_file() {
            errors.push(format!("workspace member `{member}` missing Cargo.toml"));
            continue;
        }
        if member.starts_with("rust/crates/") && member.matches('/').count() == 2 {
            let dir_name = member.rsplit('/').next().unwrap_or("");
            let stem = dir_name.replace('-', "_").to_ascii_uppercase();
            if !targets_text.contains(&format!("{stem}_PROGRAMS = [")) {
                errors.push(format!(
                    "admin/targets.bzl missing `{stem}_PROGRAMS` for `{member}`; run admin/tools/build-parity --sync"
                ));
            }
        }
    }

    // Verify all referenced `admin/tools/<name>` script paths resolve cleanly
    // on disk.
    let mut checked_tools = BTreeSet::new();
    let ref_files = [
        "admin/targets.bzl",
        "admin/tools/baux-daemon",
        "rust/crates/tooling/src/bin/build-graph.rs",
    ];
    for rel in ref_files {
        let p = root.join(rel);
        let Ok(txt) = fs::read_to_string(&p) else {
            continue;
        };
        let mut cursor = txt.as_str();
        while let Some(pos) = cursor.find("admin/tools/") {
            let tail = &cursor[pos + "admin/tools/".len()..];
            let tool: String = tail
                .chars()
                .take_while(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.'))
                .collect();
            if !tool.is_empty() && checked_tools.insert(tool.clone()) {
                let tool_path = root.join("admin/tools").join(&tool);
                if !tool_path.exists() {
                    errors.push(format!(
                        "{rel} references missing tool `admin/tools/{tool}`"
                    ));
                }
            }
            cursor = tail;
        }
    }

    if !errors.is_empty() {
        return Err(format!(
            "Build-parity / workspace lint verification failed:\n  {}",
            errors.join("\n  ")
        ));
    }
    Ok(members.len())
}

/// Reject crate-wide `#[allow(clippy::all)]` / `#[allow(warnings)]` escapes in
/// Rust workspace code.
pub fn check_hygiene(root: &Path) -> Result<usize, String> {
    let crates_dir = root.join("rust/crates");
    let mut files = Vec::new();
    walk_files(&crates_dir, &crates_dir, &mut files)?;
    files.sort();
    let mut count = 0_usize;
    let mut violations = Vec::new();
    for rel in files {
        if !rel.ends_with(".rs") {
            continue;
        }
        count += 1;
        let text = fs::read_to_string(crates_dir.join(&rel))
            .map_err(|e| format!("read rust/crates/{rel}: {e}"))?;
        let lines: Vec<&str> = text.lines().collect();
        let mut idx = 0_usize;
        while idx < lines.len() {
            let trimmed = lines[idx].trim_start();
            if trimmed.starts_with("#![allow(") {
                let mut attr = String::from(trimmed);
                while !attr.contains(']') && idx + 1 < lines.len() {
                    idx += 1;
                    attr.push_str(lines[idx].trim());
                }
                let compact: String = attr.chars().filter(|c| !c.is_whitespace()).collect();
                for forbidden in ["warnings", "clippy::all"] {
                    if compact.contains(&format!("allow({forbidden}"))
                        || compact.contains(&format!(",{forbidden}"))
                    {
                        violations.push(format!(
                            "rust/crates/{rel}:{}: forbidden crate-wide `#![allow({forbidden})]`",
                            idx + 1
                        ));
                    }
                }
            }
            idx += 1;
        }
    }
    if !violations.is_empty() {
        return Err(format!(
            "Rust lint hygiene violation(s):\n  {}",
            violations.join("\n  ")
        ));
    }
    Ok(count)
}

pub fn verify_all(root: &Path, options: Options) -> Result<(), String> {
    let webgpu_count = check_webgpu(root)?;
    let (lean_files, pin_count) = check_lean(root, options.strict_extraction)?;
    let crate_count = check_parity(root)?;
    let rust_count = check_hygiene(root)?;
    println!(
        "[ok] Invariants: webgpu_abi={webgpu_count} syms, lean={lean_files} files ({pin_count} seals), crates={crate_count}, rust_files={rust_count}."
    );
    Ok(())
}

#[cfg(invariants_cli)]
fn main() -> std::process::ExitCode {
    let mut root = PathBuf::from(".");
    let mut opts = Options::default();
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--root" => {
                let Some(val) = args.next() else {
                    eprintln!("--root requires a path");
                    return std::process::ExitCode::FAILURE;
                };
                root = PathBuf::from(val);
            }
            "--strict-extraction" => opts.strict_extraction = true,
            flag => {
                eprintln!("unknown option: {flag}");
                return std::process::ExitCode::FAILURE;
            }
        }
    }
    match verify_all(&root, opts) {
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

    #[test]
    fn test_webgpu() {
        let src = concat!(
            "unsafe extern \"C\" {\n",
            "    pub ",
            "fn bauxgpu_available() -> i32;\n",
            "    pub ",
            "fn bauxgpu_device_posterior(\n",
            "        queries: usize,\n",
            "        scale: f32,\n",
            "        flag: i32,\n",
            "        ptr: *mut f32,\n",
            "    ) -> i32;\n",
            "    pub ",
            "fn bauxgpu_void(x: f64);\n",
            "}\n",
        );
        let map = scan_rust(src).expect("parsed");
        assert_eq!(map["bauxgpu_available"], "i");
        assert_eq!(map["bauxgpu_device_posterior"], "iifii");
        assert_eq!(map["bauxgpu_void"], "vd");
    }

    #[test]
    fn test_lean() {
        let clean = r#"
            -- contains sorry and axiom inside line comment
            /- nested /- axiom -/ sorry inside block comment -/
            def msg : String := "axiom sorry admit"
            theorem ok : True := trivial
        "#;
        assert!(lean_holes(clean).is_empty());

        let dirty = "theorem broken : False := by sorry\n";
        let hits = lean_holes(dirty);
        assert_eq!(hits, vec![(1, "sorry".to_string())]);
    }
}
