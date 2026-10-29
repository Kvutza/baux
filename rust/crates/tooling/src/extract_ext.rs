use std::collections::BTreeMap;
use std::fs;
use std::os::unix::fs::symlink;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use baux::artifact::json::{self, Value, json};
use baux::artifact::runs::digest;

const AENEAS_REV: &str = "e62dd83d133d9043e7d3628096838b7430f36565";
const CHARON_REV: &str = "de8736f7c995ac54dc490345cef9ea5143c06c62";
const MINIRUST_REV: &str = "c0b555e5b15accc6c1366b1d911a1bfd49c3d9d0";

const FMT_PRELUDE: &str = "abbrev core.fmt.Formatter : Type := Unit
abbrev core.fmt.Error : Type := Unit
abbrev core.fmt.rt.Argument : Type := Unit
abbrev core.fmt.Arguments : Type := Unit

structure core.fmt.Display (T : Type _) where
  fmt : T → core.fmt.Formatter → Result ((core.result.Result Unit core.fmt.Error) × \
                           core.fmt.Formatter)
";

pub fn build_float(root: &Path) -> Result<PathBuf, String> {
    let patches = vec![root.join("research/extraction/aeneas-float.patch")];
    build_aeneas(root, "float", "Float extension", &patches)
}

pub fn build_dyn(root: &Path) -> Result<PathBuf, String> {
    let patches = vec![
        root.join("research/extraction/aeneas-float.patch"),
        root.join("research/extraction/aeneas-dyn.patch"),
        root.join("research/extraction/aeneas-dyn-dispatch.patch"),
        root.join("research/extraction/aeneas-dyn-reborrow.patch"),
    ];
    build_aeneas(root, "dyn", "Dynamic trait extension", &patches)
}

pub fn build_callsite(root: &Path) -> Result<PathBuf, String> {
    let base = root.join(".cache/baux/tools/aeneas/charon");
    verify_repo(&base, CHARON_REV, "Charon")?;
    verify_repo(&base.join("crates/minirust"), MINIRUST_REV, "minirust")?;
    let patch = root.join("research/extraction/charon-callsite.patch");
    let identity = patch_identity(CHARON_REV, std::slice::from_ref(&patch))?;
    let parent = base.parent().ok_or("missing charon parent")?;
    let dest = parent.join(format!("charon-callsite-{}", &identity[..16]));
    prepare_charon(&base, &dest, &patch, &identity)?;
    compile_charon(&base, &dest, &identity, root)
}

pub fn sanitize_lean(dir: &Path) -> Result<(), String> {
    for name in ["Types.lean", "TypesExternal.lean"] {
        let target = dir.join(name);
        if !target.exists() {
            continue;
        }
        let mut text = fs::read_to_string(&target).map_err(|e| e.to_string())?;
        text = text.replace(
            "axiom core.fmt.Formatter : Type",
            "def core.fmt.Formatter : Type := Unit",
        );
        if name == "TypesExternal.lean"
            && !text.contains("abbrev core.fmt.Formatter : Type := Unit")
        {
            let marker = "namespace BauxResearch.Generated.Optimizer\n";
            if text.contains(marker) {
                let replacement = format!("{marker}\n{FMT_PRELUDE}\n");
                text = text.replacen(marker, &replacement, 1);
            }
        }
        fs::write(&target, text).map_err(|e| e.to_string())?;
    }
    Ok(())
}

fn build_aeneas(
    root: &Path,
    tag: &str,
    label: &str,
    patches: &[PathBuf],
) -> Result<PathBuf, String> {
    let base = root.join(".cache/baux/tools/aeneas");
    verify_repo(&base, AENEAS_REV, "Aeneas")?;
    verify_repo(&base.join("charon"), CHARON_REV, "Charon")?;
    let identity = patch_identity(AENEAS_REV, patches)?;
    let parent = base.parent().ok_or("missing aeneas parent")?;
    let dest = parent.join(format!("aeneas-{tag}-{}", &identity[..16]));
    if !dest.join("patch.sha256").exists() {
        populate_aeneas(&base, &dest, patches, &identity)?;
    }
    verify_cache(&dest, &identity, label, "_build")?;
    compile_aeneas(&base, &dest, tag, &identity)
}

fn patch_identity(revision: &str, patches: &[PathBuf]) -> Result<String, String> {
    let mut payload = revision.as_bytes().to_vec();
    for patch in patches {
        let bytes = fs::read(patch).map_err(|e| format!("{}: {e}", patch.display()))?;
        payload.extend_from_slice(&bytes);
    }
    Ok(digest(&payload))
}

fn verify_repo(path: &Path, revision: &str, label: &str) -> Result<(), String> {
    let out = Command::new("git")
        .args(["-C", &path.to_string_lossy(), "rev-parse", "HEAD"])
        .output()
        .map_err(|e| e.to_string())?;
    if !out.status.success() {
        return Err(format!("git rev-parse failed at {}", path.display()));
    }
    let actual = String::from_utf8_lossy(&out.stdout).trim().to_string();
    if actual != revision {
        return Err(format!(
            "Expected {label} {revision}, found {actual} at {}",
            path.display()
        ));
    }
    let diff = Command::new("git")
        .args([
            "-C",
            &path.to_string_lossy(),
            "diff",
            "--quiet",
            "HEAD",
            "--",
        ])
        .status()
        .map_err(|e| e.to_string())?;
    if !diff.success() {
        return Err(format!("Dirty working tree at {}", path.display()));
    }
    Ok(())
}

fn extract_archive(base: &Path, dest: &Path, rev: &str) -> Result<(), String> {
    fs::create_dir(dest).map_err(|e| format!("{}: {e}", dest.display()))?;
    let tar_path = dest.join(".archive.tar");
    let status = Command::new("git")
        .args([
            "-C",
            &base.to_string_lossy(),
            "archive",
            "--format=tar",
            "-o",
            &tar_path.to_string_lossy(),
            rev,
        ])
        .status()
        .map_err(|e| e.to_string())?;
    if !status.success() {
        return Err(format!("git archive {rev} failed"));
    }
    let untar = Command::new("tar")
        .args([
            "-xf",
            &tar_path.to_string_lossy(),
            "-C",
            &dest.to_string_lossy(),
        ])
        .status()
        .map_err(|e| e.to_string())?;
    let _ = fs::remove_file(&tar_path);
    if !untar.success() {
        return Err("tar extract failed".to_string());
    }
    Ok(())
}

fn apply_patches(
    dest: &Path,
    patches: &[PathBuf],
    identity: &str,
    skip_dir: &str,
) -> Result<(), String> {
    for patch in patches {
        let status = Command::new("git")
            .args(["apply", "--no-index", &patch.to_string_lossy()])
            .current_dir(dest)
            .status()
            .map_err(|e| e.to_string())?;
        if !status.success() {
            return Err(format!("git apply failed for {}", patch.display()));
        }
    }
    fs::write(dest.join("patch.sha256"), format!("{identity}\n")).map_err(|e| e.to_string())?;
    let hashes = source_hashes(dest, skip_dir)?;
    let encoded = json::to_string(&hashes).map_err(|e| e.to_string())?;
    fs::write(dest.join("sources.json"), encoded).map_err(|e| e.to_string())
}

fn populate_aeneas(
    base: &Path,
    dest: &Path,
    patches: &[PathBuf],
    identity: &str,
) -> Result<(), String> {
    extract_archive(base, dest, AENEAS_REV)?;
    let charon = dest.join("charon");
    if charon.exists() {
        fs::remove_dir(&charon).map_err(|e| e.to_string())?;
    }
    symlink(base.join("charon"), &charon).map_err(|e| e.to_string())?;
    apply_patches(dest, patches, identity, "_build")
}

fn verify_cache(dest: &Path, identity: &str, label: &str, skip_dir: &str) -> Result<(), String> {
    let stamp = fs::read_to_string(dest.join("patch.sha256")).map_err(|e| e.to_string())?;
    if stamp.trim() != identity {
        return Err(format!("{label} identity mismatch"));
    }
    let sources_path = dest.join("sources.json");
    if !sources_path.exists() {
        return Err(format!(
            "Missing source manifest at {}",
            sources_path.display()
        ));
    }
    let recorded: BTreeMap<String, String> =
        json::from_str(&fs::read_to_string(&sources_path).map_err(|e| e.to_string())?)
            .map_err(|e| e.to_string())?;
    let current = source_hashes(dest, skip_dir)?;
    if recorded != current {
        return Err(format!(
            "{label} source cache was modified after patch application"
        ));
    }
    Ok(())
}

fn source_hashes(path: &Path, skip_dir: &str) -> Result<BTreeMap<String, String>, String> {
    let mut out = BTreeMap::new();
    walk_hashes(path, path, skip_dir, &mut out)?;
    Ok(out)
}

fn walk_hashes(
    root: &Path,
    dir: &Path,
    skip_dir: &str,
    out: &mut BTreeMap<String, String>,
) -> Result<(), String> {
    for entry in fs::read_dir(dir).map_err(|e| e.to_string())? {
        let entry = entry.map_err(|e| e.to_string())?;
        let file_type = entry.file_type().map_err(|e| e.to_string())?;
        if file_type.is_symlink() {
            continue;
        }
        let path = entry.path();
        let rel = path.strip_prefix(root).map_err(|e| e.to_string())?;
        if rel.components().any(|c| c.as_os_str() == skip_dir) {
            continue;
        }
        let rel_str = rel.to_string_lossy().to_string();
        if matches!(
            rel_str.as_str(),
            "patch.sha256" | "sources.json" | "build.json"
        ) {
            continue;
        }
        if file_type.is_dir() {
            walk_hashes(root, &path, skip_dir, out)?;
        } else if file_type.is_file() {
            let bytes = fs::read(&path).map_err(|e| e.to_string())?;
            out.insert(rel_str, digest(&bytes));
        }
    }
    Ok(())
}

fn compile_aeneas(base: &Path, dest: &Path, tag: &str, identity: &str) -> Result<PathBuf, String> {
    let exe = dest.join("src/_build/default/main.exe");
    let receipt = dest.join("build.json");
    if let (Ok(raw), Some(sha)) = (fs::read_to_string(&receipt), file_digest(&exe)?) {
        let val: Value = json::from_str(&raw).unwrap_or_else(|_| Value::null());
        if val == json!({ "binary_sha256": sha, "identity": identity }) {
            return Ok(exe);
        }
    }
    let mut cmd = Command::new("opam");
    cmd.arg("exec");
    let switch = std::env::var("BAUX_OCAML_SWITCH").ok().or_else(|| {
        let candidate = base.canonicalize().ok()?.join(".opam");
        candidate
            .exists()
            .then(|| candidate.to_string_lossy().to_string())
    });
    if let Some(sw) = switch {
        cmd.args(["--switch", &sw]);
    }
    let version = format!("{AENEAS_REV}+baux-{tag}-{}", &identity[..16]);
    let status = cmd
        .args(["--", "dune", "build", "main.exe"])
        .current_dir(dest.join("src"))
        .env("AENEAS_VERSION", version)
        .stdout(Stdio::from(std::io::stderr()))
        .status()
        .map_err(|e| e.to_string())?;
    if !status.success() {
        return Err("dune build main.exe failed".to_string());
    }
    let sha = file_digest(&exe)?.ok_or("missing compiled main.exe")?;
    let payload = json!({ "binary_sha256": sha, "identity": identity });
    fs::write(
        &receipt,
        json::to_string(&payload).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    Ok(exe)
}

fn prepare_charon(base: &Path, dest: &Path, patch: &Path, identity: &str) -> Result<(), String> {
    if !dest.join("patch.sha256").exists() {
        extract_archive(base, dest, CHARON_REV)?;
        let sub = dest.join("crates/minirust");
        if sub.exists() {
            fs::remove_dir(&sub).map_err(|e| e.to_string())?;
        }
        symlink(base.join("crates/minirust"), &sub).map_err(|e| e.to_string())?;
        apply_patches(dest, &[patch.to_path_buf()], identity, "target")?;
    }
    verify_cache(dest, identity, "Charon", "target")
}

fn seed_target(base: &Path, target: &Path, root: &Path) -> Result<(), String> {
    if target.exists() {
        return Ok(());
    }
    let parent = base.parent().ok_or("missing base parent")?;
    let mut candidates = Vec::new();
    for entry in fs::read_dir(parent).map_err(|e| e.to_string())? {
        let path = entry.map_err(|e| e.to_string())?.path();
        let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
        let cand = path.join("charon/target");
        if name.starts_with("charon-callsite-") && cand.exists() && cand != *target {
            let mtime = cand.metadata().and_then(|m| m.modified()).ok();
            candidates.push((mtime, cand));
        }
    }
    candidates.sort_by(|a, b| b.0.cmp(&a.0));
    let fallback = root.join(".cache/baux/tools/aeneas/charon-staging/target");
    let bootstrap = candidates
        .into_iter()
        .map(|(_, p)| p)
        .next()
        .unwrap_or(fallback);
    if bootstrap.exists() {
        let _ = Command::new("cp")
            .args([
                "-c",
                "-R",
                &bootstrap.to_string_lossy(),
                &target.to_string_lossy(),
            ])
            .status();
    }
    Ok(())
}

fn compile_charon(
    base: &Path,
    dest: &Path,
    identity: &str,
    root: &Path,
) -> Result<PathBuf, String> {
    let project = dest.join("charon");
    let target = project.join("target");
    seed_target(base, &target, root)?;
    let exe = target.join("release/charon");
    let driver = target.join("release/charon-driver");
    let receipt = dest.join("build.json");
    let expected = json!({
        "charon_sha256": file_digest(&exe)?,
        "driver_sha256": file_digest(&driver)?,
        "identity": identity,
    });
    let current = fs::read_to_string(&receipt)
        .ok()
        .and_then(|s| json::from_str::<Value>(&s).ok());
    if current.as_ref() != Some(&expected) || !exe.exists() {
        let status = Command::new("cargo")
            .args([
                "build",
                "--locked",
                "--release",
                "--bin",
                "charon",
                "--bin",
                "charon-driver",
            ])
            .current_dir(&project)
            .stdout(Stdio::from(std::io::stderr()))
            .status()
            .map_err(|e| e.to_string())?;
        if !status.success() {
            return Err("charon build failed".to_string());
        }
        let fresh = json!({
            "charon_sha256": file_digest(&exe)?,
            "driver_sha256": file_digest(&driver)?,
            "identity": identity,
        });
        fs::write(
            &receipt,
            json::to_string(&fresh).map_err(|e| e.to_string())?,
        )
        .map_err(|e| e.to_string())?;
    }
    Ok(exe)
}

fn file_digest(path: &Path) -> Result<Option<String>, String> {
    if !path.exists() {
        return Ok(None);
    }
    let bytes = fs::read(path).map_err(|e| e.to_string())?;
    Ok(Some(digest(&bytes)))
}
