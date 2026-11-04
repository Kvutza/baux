use std::collections::HashMap;
use std::ffi::{OsStr, OsString};
use std::fs;
use std::path::{Component, Path, PathBuf};
use std::process::Command;
use std::time::UNIX_EPOCH;

use crate::execute;

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

pub(super) fn run(root: &Path) -> Result<(), String> {
    let files = tracked_files(root)?;
    let changed = changed_files(root, &files);
    let rust = rust_files(&changed);
    let python = typed_files(&changed, "py");
    let toml = typed_files(&changed, "toml");
    let uv_dirty = !python.is_empty()
        || changed.iter().any(|path| {
            matches!(
                path.file_name().and_then(|n| n.to_str()),
                Some("pyproject.toml" | "uv.lock")
            )
        });

    std::thread::scope(|scope| -> Result<(), String> {
        // Rustfmt recursively follows modules. Separate shards can write the
        // same child file concurrently, even when their input lists differ.
        let rust_job = scope.spawn(|| run_files(root, "rustfmt", &["--edition", "2024"], &rust));
        let py_job = scope.spawn(|| -> Result<(), String> {
            run_files(root, "uvx", &["ruff", "format"], &python)?;
            run_files(root, "uvx", &["ruff", "check"], &python)?;
            if uv_dirty {
                run_all(
                    root,
                    "uv",
                    &["check", "--preview-features", "check-command"],
                )?;
            }
            Ok(())
        });
        let toml_job = scope.spawn(|| -> Result<(), String> {
            run_files(root, "uvx", &["taplo", "fmt"], &toml)?;
            run_files(root, "uvx", &["taplo", "check"], &toml)
        });
        rust_job.join().map_err(|_| "rustfmt thread panicked")??;
        py_job
            .join()
            .map_err(|_| "python check thread panicked")??;
        toml_job
            .join()
            .map_err(|_| "toml check thread panicked")??;
        Ok(())
    })?;
    std::thread::scope(|scope| -> Result<(), String> {
        let names_job = scope.spawn(|| baux_tooling::names::cli(root, &[], false));
        let kiss_job = scope.spawn(|| run_all(root, "kiss", &["check", "."]));
        names_job.join().map_err(|_| "names thread panicked")??;
        kiss_job.join().map_err(|_| "kiss thread panicked")??;
        Ok(())
    })?;
    write_stamp(root, &files);
    Ok(())
}
