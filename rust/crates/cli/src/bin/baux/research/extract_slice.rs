use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use super::{AENEAS_REV, CHARON_REV, ExtractOpts};

fn copy_tree(source: &Path, destination: &Path) -> Result<(), String> {
    fs::create_dir_all(destination)
        .map_err(|error| format!("cannot create {}: {error}", destination.display()))?;
    for entry in fs::read_dir(source)
        .map_err(|error| format!("cannot read {}: {error}", source.display()))?
    {
        let entry = entry.map_err(|error| error.to_string())?;
        let source_path = entry.path();
        let destination_path = destination.join(entry.file_name());
        if entry
            .file_type()
            .map_err(|error| error.to_string())?
            .is_dir()
        {
            copy_tree(&source_path, &destination_path)?;
        } else {
            fs::copy(&source_path, &destination_path).map_err(|error| {
                format!(
                    "cannot copy {} to {}: {error}",
                    source_path.display(),
                    destination_path.display()
                )
            })?;
        }
    }
    Ok(())
}

fn collect_files(root: &Path, path: &Path, files: &mut Vec<PathBuf>) -> Result<(), String> {
    for entry in
        fs::read_dir(path).map_err(|error| format!("cannot read {}: {error}", path.display()))?
    {
        let entry = entry.map_err(|error| error.to_string())?;
        let entry_path = entry.path();
        if entry
            .file_type()
            .map_err(|error| error.to_string())?
            .is_dir()
        {
            collect_files(root, &entry_path, files)?;
        } else {
            files.push(
                entry_path
                    .strip_prefix(root)
                    .map_err(|error| error.to_string())?
                    .to_owned(),
            );
        }
    }
    Ok(())
}

pub(super) fn tree_digest(root: &Path) -> Result<String, String> {
    let mut files = Vec::new();
    collect_files(root, root, &mut files)?;
    files.sort();
    let mut framed = Vec::new();
    for relative in files {
        let name = relative.to_string_lossy().as_bytes().to_vec();
        let bytes = fs::read(root.join(&relative)).map_err(|error| error.to_string())?;
        framed.extend_from_slice(&(name.len() as u64).to_le_bytes());
        framed.extend_from_slice(&name);
        framed.extend_from_slice(&(bytes.len() as u64).to_le_bytes());
        framed.extend_from_slice(&bytes);
    }
    Ok(baux::artifact::runs::digest(&framed))
}

fn copy_slice(source: &Path, destination: &Path) -> Result<String, String> {
    if destination.exists() {
        fs::remove_dir_all(destination)
            .map_err(|error| format!("cannot clear {}: {error}", destination.display()))?;
    }
    let destination_src = destination.join("src");
    copy_tree(source, &destination_src)?;
    let source_digest = tree_digest(source)?;
    if source_digest != tree_digest(&destination_src)? {
        return Err("extraction slice does not match the production source tree".into());
    }
    Ok(source_digest)
}

pub(super) fn local_slice(root: &Path, cache: &Path) -> Result<(PathBuf, String), String> {
    let destination = cache.join("local-slice");
    let source_digest = copy_slice(&root.join("rust/crates/baux/src"), &destination)?;
    fs::copy(
        root.join("research/extraction/local/lib.rs"),
        destination.join("src/lib.rs"),
    )
    .map_err(|error| format!("cannot install local extraction root: {error}"))?;
    let manifest = fs::read_to_string(root.join("research/extraction/local/Cargo.toml.in"))
        .map_err(|error| error.to_string())?
        .replace("@ROOT@", &root.to_string_lossy());
    fs::write(destination.join("Cargo.toml"), manifest).map_err(|error| error.to_string())?;
    Ok((destination, source_digest))
}

pub(super) fn semantics_slice(root: &Path, cache: &Path) -> Result<PathBuf, String> {
    let destination = cache.join("semantics-slice");
    copy_slice(&root.join("rust/crates/baux/semantics/src"), &destination)?;
    let lib_path = destination.join("src/lib.rs");
    let original = fs::read_to_string(&lib_path).map_err(|error| error.to_string())?;
    fs::write(
        &lib_path,
        format!(
            "#![allow(incomplete_features, internal_features)]\n\
             #![feature(min_adt_const_params, adt_const_params, contracts, diagnostic_on_move, negative_impls, portable_simd)]\n\
             {original}\n\
             extern crate self as baux;\n\
             pub use crate as semantics;\n"
        ),
    )
    .map_err(|error| error.to_string())?;
    let _ = fs::copy(root.join("Cargo.lock"), destination.join("Cargo.lock"));
    fs::write(
        destination.join("Cargo.toml"),
        "[workspace]\n\n[package]\nname = \"baux-semantics\"\nversion = \"0.0.1\"\nedition = \"2024\"\nrust-version = \"1.88\"\nlicense = \"MIT\"\npublish = false\n\n[dependencies]\ndeser = { version = \"=0.9.1\", features = [\"derive\"] }\nrand = \"0.8\"\n",
    )
    .map_err(|error| error.to_string())?;
    Ok(destination)
}

pub(super) fn resolve_tools(root: &Path, opts: &ExtractOpts) -> Result<(PathBuf, PathBuf), String> {
    let tools = root.join(".cache/baux/tools/aeneas");
    let mut aeneas = if let Ok(bin) = std::env::var("BAUX_AENEAS_BIN") {
        PathBuf::from(bin)
    } else {
        let native_bin = root.join("ocaml/extraction/bin/aeneas");
        if native_bin.exists() {
            native_bin
        } else {
            tools.join("bin/aeneas")
        }
    };
    let charon_root = tools.join("charon");
    let mut charon = if let Ok(bin) = std::env::var("BAUX_CHARON_BIN") {
        PathBuf::from(bin)
    } else {
        charon_root.join("bin/charon")
    };
    require_tool(&aeneas, "Aeneas", AENEAS_REV)?;
    require_tool(&charon, "Charon", CHARON_REV)?;
    if tools.join(".git").exists() {
        require_revision(&tools, "Aeneas", AENEAS_REV)?;
    }
    if charon_root.join(".git").exists() {
        require_revision(&charon_root, "Charon", CHARON_REV)?;
    }
    if opts.charon_callsite {
        charon = baux_tooling::extract_ext::build_callsite(root)?;
        require_tool(&charon, "Charon callsite extension", CHARON_REV)?;
    }
    if opts.float_backend() {
        aeneas = if opts.dyn_traits {
            baux_tooling::extract_ext::build_dyn(root)?
        } else {
            baux_tooling::extract_ext::build_float(root)?
        };
        require_tool(&aeneas, "Aeneas float extension", AENEAS_REV)?;
    }
    Ok((aeneas, charon))
}

fn probe_ext(root: &Path, opts: &ExtractOpts) -> Result<String, String> {
    let builder = root.join("rust/crates/tooling/src/extract_ext.rs");
    let base = if opts.dyn_traits {
        format!(
            "float_patch_sha256={}\ndyn_patch_sha256={}\ndyn_dispatch_patch_sha256={}\ndyn_reborrow_patch_sha256={}\ndyn_builder_sha256={}\n",
            baux::artifact::runs::file_digest(
                &root.join("research/extraction/aeneas-float.patch")
            )?,
            baux::artifact::runs::file_digest(&root.join("research/extraction/aeneas-dyn.patch"))?,
            baux::artifact::runs::file_digest(
                &root.join("research/extraction/aeneas-dyn-dispatch.patch")
            )?,
            baux::artifact::runs::file_digest(
                &root.join("research/extraction/aeneas-dyn-reborrow.patch")
            )?,
            baux::artifact::runs::file_digest(&builder)?,
        )
    } else if opts.floats {
        format!(
            "float_patch_sha256={}\nfloat_builder_sha256={}\n",
            baux::artifact::runs::file_digest(
                &root.join("research/extraction/aeneas-float.patch")
            )?,
            baux::artifact::runs::file_digest(&builder)?,
        )
    } else {
        String::new()
    };
    if opts.charon_callsite {
        Ok(format!(
            "{base}charon_callsite_patch_sha256={}\ncharon_callsite_builder_sha256={}\n",
            baux::artifact::runs::file_digest(
                &root.join("research/extraction/charon-callsite.patch")
            )?,
            baux::artifact::runs::file_digest(&builder)?,
        ))
    } else {
        Ok(base)
    }
}

fn jj_rev(root: &Path) -> Result<String, String> {
    let output = Command::new("jj")
        .current_dir(root)
        .args([
            "log",
            "-r",
            "@",
            "--no-graph",
            "--no-pager",
            "-T",
            "commit_id",
        ])
        .output()
        .map_err(|error| format!("cannot read local JJ revision: {error}"))?;
    if !output.status.success() {
        return Err(format!("cannot read local JJ revision: {}", output.status));
    }
    Ok(String::from_utf8(output.stdout)
        .map_err(|error| error.to_string())?
        .trim()
        .to_owned())
}

fn probe_source(
    root: &Path,
    opts: &ExtractOpts,
    slice_info: Option<&(PathBuf, String)>,
) -> Result<String, String> {
    if opts.upstream {
        return Ok(format!(
            "upstream_harness_sha256={}\n",
            baux::artifact::runs::file_digest(&root.join("research/extraction/enn_harness.rs"))?,
        ));
    }
    if opts.production {
        return Ok(format!(
            "local_revision={}\nproduction_source_tree_sha256={}\nproduction_manifest_sha256={}\nproduction_extraction_root_sha256={}\n",
            jj_rev(root)?,
            tree_digest(&root.join("rust/crates/baux/src"))?,
            baux::artifact::runs::file_digest(&root.join("rust/crates/baux/Cargo.toml"))?,
            baux::artifact::runs::file_digest(&root.join("rust/crates/baux/src/extraction.rs"))?,
        ));
    }
    if opts.local {
        let (_, prod_digest) = slice_info.ok_or("local extraction slice was not prepared")?;
        return Ok(format!(
            "local_revision={}\nlocal_slice_manifest_sha256={}\nlocal_slice_root_sha256={}\nproduction_source_tree_sha256={prod_digest}\nproduction_extraction_root_sha256={}\n",
            jj_rev(root)?,
            baux::artifact::runs::file_digest(
                &root.join("research/extraction/local/Cargo.toml.in")
            )?,
            baux::artifact::runs::file_digest(&root.join("research/extraction/local/lib.rs"))?,
            baux::artifact::runs::file_digest(&root.join("rust/crates/baux/src/extraction.rs"))?,
        ));
    }
    Ok(String::new())
}

pub(super) fn init_probe(
    root: &Path,
    cache: &Path,
    opts: &ExtractOpts,
    tools: (&Path, &Path),
    slice_info: Option<&(PathBuf, String)>,
) -> Result<Option<PathBuf>, String> {
    if opts.probe.is_empty() {
        return Ok(None);
    }
    let stamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_err(|error| error.to_string())?
        .as_nanos();
    let path = cache.join(format!("probe-{}-{stamp}", std::process::id()));
    fs::create_dir(&path).map_err(|error| error.to_string())?;
    let source_mode = opts.source_mode();
    fs::write(path.join("source.txt"), format!("{source_mode}\n"))
        .map_err(|error| error.to_string())?;
    for (name, items) in [
        ("roots.txt", opts.probe),
        ("opaque.txt", opts.opaque),
        ("include.txt", opts.include),
    ] {
        let body = if items.is_empty() {
            String::new()
        } else {
            format!("{}\n", items.join("\n"))
        };
        fs::write(path.join(name), body).map_err(|error| error.to_string())?;
    }
    let extension = probe_ext(root, opts)?;
    let source_binding = probe_source(root, opts, slice_info)?;
    fs::write(
        path.join("toolchain.txt"),
        format!(
            "aeneas_base={AENEAS_REV}\ncharon={CHARON_REV}\naeneas_binary_sha256={}\ncharon_binary_sha256={}\nsource={source_mode}\n{extension}{source_binding}",
            baux::artifact::runs::file_digest(tools.0)?,
            baux::artifact::runs::file_digest(tools.1)?,
        ),
    )
    .map_err(|error| error.to_string())?;
    println!("baux: extraction probe evidence: {}", path.display());
    Ok(Some(path))
}

pub(super) fn resolve_source(
    root: &Path,
    opts: &ExtractOpts,
    probe_root: Option<&Path>,
    slice_info: Option<&(PathBuf, String)>,
) -> Result<PathBuf, String> {
    if opts.upstream {
        let evidence = probe_root.ok_or("upstream extraction requires a probe")?;
        return baux_tooling::enn_artifact::snapshot_upstream(root, evidence);
    }
    if opts.production {
        return Ok(root.join("rust/crates/baux"));
    }
    if opts.local {
        return Ok(slice_info
            .ok_or("local extraction slice was not prepared")?
            .0
            .clone());
    }
    semantics_slice(root, &root.join(".cache/baux/aeneas"))
}

fn require_tool(path: &Path, name: &str, revision: &str) -> Result<(), String> {
    if path.is_file() {
        Ok(())
    } else {
        Err(format!(
            "{name} {} is not built at {}; install the pinned research toolchain before extraction",
            short_revision(revision),
            path.display()
        ))
    }
}

fn require_revision(path: &Path, name: &str, expected: &str) -> Result<(), String> {
    let output = Command::new("git")
        .current_dir(path)
        .args(["rev-parse", "HEAD"])
        .output()
        .map_err(|error| format!("cannot inspect cached {name}: {error}"))?;
    let actual = String::from_utf8_lossy(&output.stdout).trim().to_owned();
    if output.status.success() && actual == expected {
        Ok(())
    } else {
        Err(format!(
            "cached {name} is {actual}, but BAUX requires {expected}"
        ))
    }
}

pub(super) fn short_revision(revision: &str) -> &str {
    &revision[..7]
}

pub(crate) fn compare_translation(lean_root: &Path, check_root: &Path) -> Result<(), String> {
    compare_family(lean_root, check_root, "BauxResearch/Generated/Optimizer")
}

pub(crate) fn compare_family(
    lean_root: &Path,
    check_root: &Path,
    relative: &str,
) -> Result<(), String> {
    let relative = Path::new(relative);
    for file in ["Types.lean", "Funs.lean"] {
        let tracked = lean_root.join(relative).join(file);
        let generated = check_root.join(relative).join(file);
        let tracked_bytes = fs::read(&tracked)
            .map_err(|error| format!("cannot read {}: {error}", tracked.display()))?;
        let generated_bytes = fs::read(&generated)
            .map_err(|error| format!("cannot read {}: {error}", generated.display()))?;
        if tracked_bytes != generated_bytes {
            return Err(format!(
                "{} is stale; run `./baux research extract`",
                tracked.display()
            ));
        }
    }
    Ok(())
}

pub(crate) fn validate_llbc(bytes: &[u8]) -> Result<(), String> {
    let value: baux::artifact::json::Value = baux::artifact::json::from_slice(bytes)
        .map_err(|error| format!("cannot parse Charon LLBC: {error}"))?;
    if value.get("has_errors").and_then(|flag| flag.as_bool()) != Some(false) {
        return Err("Charon LLBC reports errors or lacks an explicit error-free status".into());
    }
    Ok(())
}
