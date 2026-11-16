use clap::Args;
use std::collections::HashMap;
use std::ffi::{OsStr, OsString};
use std::fs;
use std::io::ErrorKind;
use std::path::{Component, Path, PathBuf};
use std::process::Command;
use std::time::UNIX_EPOCH;

use crate::execute;

#[derive(Args, Default)]
pub(super) struct CheckArgs {
    /// Bypass the incremental stamp cache and verify every tracked file.
    #[arg(long)]
    pub(super) all: bool,
    /// Verify formatting and checks without modifying tracked files.
    #[arg(long)]
    pub(super) check: bool,
    /// Run compiled analyzers (Buck2 Clippy, build parity, Dune UI, coverage,
    /// workflows).
    #[arg(long)]
    pub(super) deep: bool,
    /// Fail on stale extraction source-tree SHA-256 bindings.
    #[arg(long)]
    pub(super) strict_extraction: bool,
}

fn tracked_files(root: &Path) -> Result<Vec<PathBuf>, String> {
    let output = Command::new("jj")
        .current_dir(root)
        .args(["file", "list", "-T", "path ++ \"\\0\""])
        .output()
        .map_err(|error| format!("start jj file list: {error}"))?;
    if !output.status.success() {
        return Err(format!(
            "jj file list failed: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        ));
    }
    output
        .stdout
        .split(|byte| *byte == 0)
        .filter(|path| !path.is_empty())
        .map(path_bytes)
        .collect()
}

#[cfg(unix)]
fn path_bytes(bytes: &[u8]) -> Result<PathBuf, String> {
    use std::os::unix::ffi::OsStringExt;
    Ok(PathBuf::from(OsString::from_vec(bytes.to_vec())))
}

#[cfg(windows)]
fn path_bytes(bytes: &[u8]) -> Result<PathBuf, String> {
    String::from_utf8(bytes.to_vec())
        .map(PathBuf::from)
        .map_err(|error| format!("jj returned a non-UTF-8 Windows path: {error}"))
}

fn file_sig(root: &Path, rel: &Path) -> Option<(u128, u64)> {
    let meta = fs::metadata(root.join(rel)).ok()?;
    let mtime = meta.modified().ok()?.duration_since(UNIX_EPOCH).ok()?;
    Some((mtime.as_nanos(), meta.len()))
}

fn changed_files(root: &Path, files: &[PathBuf]) -> Vec<PathBuf> {
    let stamp = root.join(".cache/baux/check.stamp");
    let raw = fs::read_to_string(stamp).unwrap_or_default();
    let mut cached = HashMap::new();
    for line in raw.lines() {
        let mut parts = line.splitn(3, '\t');
        if let (Some(ns), Some(len), Some(path)) = (parts.next(), parts.next(), parts.next()) {
            if let (Ok(ns), Ok(len)) = (ns.parse::<u128>(), len.parse::<u64>()) {
                cached.insert(PathBuf::from(path), (ns, len));
            }
        }
    }
    files
        .iter()
        .filter(|path| file_sig(root, path) != cached.get(*path).copied())
        .cloned()
        .collect()
}

fn write_stamp(root: &Path, files: &[PathBuf]) {
    let stamp = root.join(".cache/baux/check.stamp");
    if let Some(parent) = stamp.parent() {
        let _ = fs::create_dir_all(parent);
    }
    let mut out = String::new();
    for path in files {
        if let (Some((ns, len)), Some(rel)) = (file_sig(root, path), path.to_str()) {
            out.push_str(&format!("{ns}\t{len}\t{rel}\n"));
        }
    }
    let _ = fs::write(stamp, out);
}

fn typed_files(files: &[PathBuf], extension: &str) -> Vec<OsString> {
    files
        .iter()
        .filter(|path| path.extension() == Some(OsStr::new(extension)))
        .map(|path| path.as_os_str().to_owned())
        .collect()
}

fn rust_files(files: &[PathBuf]) -> Vec<OsString> {
    files
        .iter()
        .filter(|path| path.extension() == Some(OsStr::new("rs")))
        .filter(|path| {
            let mut components = path.components();
            !matches!(
                (components.next(), components.next()),
                (Some(Component::Normal(a)), Some(Component::Normal(b)))
                    if a == "research" && b == "extraction"
            )
        })
        .map(|path| path.as_os_str().to_owned())
        .collect()
}

fn run_files(
    root: &Path,
    program: &str,
    arguments: &[&str],
    files: &[OsString],
) -> Result<(), String> {
    if files.is_empty() {
        return Ok(());
    }
    println!("o|check: {program} files={}", files.len());
    let status = Command::new(program)
        .current_dir(root)
        .args(arguments)
        .args(files)
        .status()
        .map_err(|error| error.to_string())?;
    status
        .success()
        .then_some(())
        .ok_or_else(|| format!("{program} exited with {status}"))
}

fn run_all(root: &Path, program: &str, arguments: &[&str]) -> Result<(), String> {
    let mut command = Command::new(program);
    command.current_dir(root).args(arguments);
    execute(command)
}

fn shell_files(root: &Path, files: &[PathBuf]) -> Vec<OsString> {
    files
        .iter()
        .filter(|path| {
            let ext = path.extension().and_then(|e| e.to_str());
            if matches!(ext, Some("sh" | "bash")) {
                return true;
            }
            if ext.is_some() {
                return false;
            }
            let in_tools = path.starts_with("admin/tools/") || path.as_path() == Path::new("baux");
            if !in_tools {
                return false;
            }
            let Ok(head) = fs::read(root.join(path)) else {
                return false;
            };
            let first_line = head.split(|b| *b == b'\n').next().unwrap_or(&[]);
            let first = std::str::from_utf8(first_line).unwrap_or("");
            first.starts_with("#!/bin/sh")
                || first.starts_with("#!/bin/bash")
                || first.starts_with("#!/usr/bin/env sh")
                || first.starts_with("#!/usr/bin/env bash")
        })
        .map(|path| path.as_os_str().to_owned())
        .collect()
}

fn js_files(files: &[PathBuf]) -> Vec<PathBuf> {
    files
        .iter()
        .filter(|path| {
            matches!(
                path.extension().and_then(|e| e.to_str()),
                Some("js" | "mjs")
            )
        })
        .filter(|path| {
            !matches!(
                path.components().next(),
                Some(Component::Normal(c)) if c == "docs" || c == "assets"
            )
        })
        .cloned()
        .collect()
}

fn docs_dirty(files: &[PathBuf]) -> bool {
    files.iter().any(|path| {
        path.starts_with("forest/trees/")
            || path.starts_with("python/baux/")
            || path.starts_with("rust/crates/baux-py/")
            || path.as_path() == Path::new("admin/tools/verify_docs.py")
    })
}

fn try_files(
    root: &Path,
    program: &str,
    arguments: &[&str],
    files: &[OsString],
) -> Result<(), String> {
    if files.is_empty() {
        return Ok(());
    }
    println!("o|check: {program} files={}", files.len());
    match Command::new(program)
        .current_dir(root)
        .args(arguments)
        .args(files)
        .status()
    {
        Ok(status) => status
            .success()
            .then_some(())
            .ok_or_else(|| format!("{program} exited with {status}")),
        Err(error) if error.kind() == ErrorKind::NotFound => {
            println!("[warn] skipping {program}: binary not found on PATH");
            Ok(())
        }
        Err(error) => Err(format!("start {program}: {error}")),
    }
}

fn check_js(root: &Path, files: &[PathBuf]) -> Result<(), String> {
    if files.is_empty() {
        return Ok(());
    }
    println!("o|check: node --check files={}", files.len());
    for file in files {
        match Command::new("node")
            .current_dir(root)
            .arg("--check")
            .arg(file)
            .status()
        {
            Ok(status) if status.success() => {}
            Ok(status) => {
                return Err(format!(
                    "node --check {} exited with {status}",
                    file.display()
                ));
            }
            Err(error) if error.kind() == ErrorKind::NotFound => {
                println!("[warn] skipping node --check: node not found on PATH");
                return Ok(());
            }
            Err(error) => return Err(format!("start node --check: {error}")),
        }
    }
    Ok(())
}

pub(super) fn run(root: &Path, args: CheckArgs) -> Result<(), String> {
    let files = tracked_files(root)?;
    let scope_files = if args.all {
        files.clone()
    } else {
        changed_files(root, &files)
    };
    let rust = rust_files(&scope_files);
    let python = typed_files(&scope_files, "py");
    let toml = typed_files(&scope_files, "toml");
    let shell = shell_files(root, &scope_files);
    let js = js_files(&scope_files);
    let workflows_dirty = args.all
        || scope_files
            .iter()
            .any(|path| path.starts_with(".github/workflows/"));
    let docs_dirty = args.all || docs_dirty(&scope_files);
    let uv_dirty = args.all
        || !python.is_empty()
        || scope_files.iter().any(|path| {
            matches!(
                path.file_name().and_then(|n| n.to_str()),
                Some("pyproject.toml" | "uv.lock")
            )
        });

    std::thread::scope(|scope| -> Result<(), String> {
        // Rustfmt recursively follows modules. Separate shards can write the
        // same child file concurrently, even when their input lists differ.
        let rust_job = scope.spawn(|| {
            if args.check {
                run_files(root, "rustfmt", &["--edition", "2024", "--check"], &rust)
            } else {
                run_files(root, "rustfmt", &["--edition", "2024"], &rust)
            }
        });
        let py_job = scope.spawn(|| -> Result<(), String> {
            if args.check {
                run_files(root, "uvx", &["ruff", "format", "--check"], &python)?;
            } else {
                run_files(root, "uvx", &["ruff", "format"], &python)?;
            }
            run_files(root, "uvx", &["ruff", "check"], &python)?;
            if uv_dirty {
                run_all(
                    root,
                    "uv",
                    &["check", "--preview-features", "check-command"],
                )?;
                run_all(root, "uvx", &["ty", "check"])?;
            }
            Ok(())
        });
        let toml_job = scope.spawn(|| -> Result<(), String> {
            if args.check {
                run_files(root, "uvx", &["taplo", "fmt", "--check"], &toml)?;
            } else {
                run_files(root, "uvx", &["taplo", "fmt"], &toml)?;
            }
            run_files(root, "uvx", &["taplo", "check"], &toml)
        });
        let static_job = scope.spawn(|| -> Result<(), String> {
            try_files(root, "shellcheck", &[], &shell)?;
            check_js(root, &js)?;
            if docs_dirty {
                run_all(root, "python3", &["admin/tools/verify_docs.py"])?;
            }
            if workflows_dirty && root.join(".github/workflows").is_dir() {
                run_all(
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
                )?;
            }
            Ok(())
        });
        rust_job.join().map_err(|_| "rustfmt thread panicked")??;
        py_job
            .join()
            .map_err(|_| "python check thread panicked")??;
        toml_job
            .join()
            .map_err(|_| "toml check thread panicked")??;
        static_job
            .join()
            .map_err(|_| "static check thread panicked")??;
        Ok(())
    })?;

    crate::invariants::verify_all(
        root,
        crate::invariants::Options {
            strict_extraction: args.strict_extraction,
        },
    )?;
    crate::pages_verify::verify_publictree(root)?;

    if args.deep {
        let clippy_out = Command::new("./admin/tools/buck2w")
            .current_dir(root)
            .args([
                "bxl",
                "--console",
                "simple",
                "//admin/buck2:baux.bxl:clippy_all",
            ])
            .output()
            .map_err(|e| format!("run buck2w clippy: {e}"))?;
        if !clippy_out.status.success() {
            return Err(format!(
                "buck2w clippy exited with {}:\n{}",
                clippy_out.status,
                String::from_utf8_lossy(&clippy_out.stderr)
            ));
        }
        let mut clippy_diags = String::new();
        for line in String::from_utf8_lossy(&clippy_out.stdout).lines() {
            if let Some((label, rel)) = line.split_once(' ') {
                let diag_path = root.join(rel.trim());
                if diag_path.is_file() {
                    let body = fs::read_to_string(&diag_path)
                        .map_err(|e| format!("read {label} {}: {e}", rel.trim()))?;
                    if !body.trim().is_empty() {
                        clippy_diags.push_str(&format!("== {label} ==\n{body}\n"));
                    }
                }
            }
        }
        if !clippy_diags.is_empty() {
            return Err(format!("Clippy diagnostics found:\n{clippy_diags}"));
        }
        run_all(root, "./admin/tools/build-parity", &["--check"])?;
        let exe = std::env::current_exe().map_err(|e| format!("current_exe: {e}"))?;
        let status = Command::new(exe)
            .current_dir(root)
            .args(["research", "coverage", "--check"])
            .status()
            .map_err(|e| format!("run research coverage --check: {e}"))?;
        if !status.success() {
            return Err(format!("research coverage --check exited with {status}"));
        }
        match Command::new("opam")
            .current_dir(root)
            .args([
                "exec",
                "--switch=5.2.0+ox",
                "--",
                "dune",
                "build",
                "--root=admin/ui",
                "@all",
            ])
            .status()
        {
            Ok(status) if status.success() => {}
            Ok(status) => return Err(format!("dune build --root=admin/ui exited with {status}")),
            Err(error) if error.kind() == ErrorKind::NotFound => {
                println!("[warn] skipping Dune UI build: opam not found on PATH");
            }
            Err(error) => return Err(format!("start opam dune build: {error}")),
        }
    }
    baux_tooling::names::cli(root, &[], false)?;

    if !args.check {
        write_stamp(root, &files);
    }
    Ok(())
}
