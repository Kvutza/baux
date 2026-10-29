use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

const AENEAS_REV: &str = "e62dd83d133d9043e7d3628096838b7430f36565";
const CHARON_REV: &str = "de8736f7c995ac54dc490345cef9ea5143c06c62";
const FLOAT_ROOTS: [&str; 8] = [
    "crate::random::u64_f53",
    "crate::random::box_muller",
    "crate::random::from_unit",
    "crate::random::to_unit",
    "crate::float::single_from",
    "crate::float::single_bits",
    "crate::float::double_from",
    "crate::float::double_bits",
];
const LOCAL_ROOTS: [&str; 5] = [
    "crate::coordinate_flat::from_unit",
    "crate::coordinate_flat::to_unit",
    "crate::optimizer_dispatch::ask_gate",
    "crate::optimizer_dispatch::tell_gate",
    "crate::optimizer_dispatch::variance_gate",
];
const ACTION_ROOTS: [&str; 28] = [
    "crate::gaussian::normal_bounds",
    "crate::gaussian::mix32",
    "crate::gaussian::bits",
    "crate::hazard::conflict",
    "crate::hazard::flush",
    "crate::hazard::capacity",
    "crate::memory::view_bytes",
    "crate::memory::span_end",
    "crate::memory::overlaps",
    "crate::dispatch::tiles",
    "crate::dispatch::tile_start",
    "crate::dispatch::tile_len",
    "crate::optimizer::trust_region::controller::accept_bits",
    "crate::optimizer::trust_region::controller::accept_decision",
    "crate::optimizer::trust_region::controller::decide_action",
    "crate::optimizer::provenance::same_world",
    "crate::optimizer::provenance::exact_anchored",
    "crate::optimizer::param_tree::routed_count",
    "crate::optimizer::param_tree::total_count",
    "crate::optimizer::window_plan::verify_full",
    "crate::optimizer::reuse::conditioned_slot",
    "crate::optimizer::reuse::conditioned_sign",
    "crate::optimizer::transaction::commit_round",
    "crate::float::single_lt",
    "crate::float::double_lt",
    "crate::float::to_double",
    // These are proof roots, not incidental dependencies of candidate draws.
    "crate::random::sobol_base",
    "crate::random::sobol_seed",
];

#[derive(Default)]
pub struct ExtractOpts<'a> {
    pub check: bool,
    pub probe: &'a [String],
    pub floats: bool,
    pub dyn_traits: bool,
    pub charon_callsite: bool,
    pub upstream: bool,
    pub local: bool,
    pub production: bool,
    pub opaque: &'a [String],
    pub include: &'a [String],
}

impl ExtractOpts<'_> {
    fn tracked_local(&self) -> bool {
        self.local && self.probe.is_empty()
    }

    fn float_backend(&self) -> bool {
        self.floats || self.dyn_traits || self.tracked_local() || self.production
    }

    fn family(&self) -> &'static str {
        if self.tracked_local() {
            "LocalCoord"
        } else if self.floats {
            "Floating"
        } else {
            "Optimizer"
        }
    }

    fn source_mode(&self) -> &'static str {
        if self.upstream {
            "upstream"
        } else if self.production {
            "local-production"
        } else if self.local {
            "local"
        } else {
            "semantics"
        }
    }
}

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

fn tree_digest(root: &Path) -> Result<String, String> {
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

fn local_slice(root: &Path, cache: &Path) -> Result<(PathBuf, String), String> {
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

fn semantics_slice(root: &Path, cache: &Path) -> Result<PathBuf, String> {
    let destination = cache.join("semantics-slice");
    copy_slice(&root.join("rust/crates/baux/semantics/src"), &destination)?;
    let lib_path = destination.join("src/lib.rs");
    fs::write(
        &lib_path,
        format!(
            "{}\nextern crate self as baux;\npub use crate as semantics;\n",
            fs::read_to_string(&lib_path).map_err(|error| error.to_string())?
        ),
    )
    .map_err(|error| error.to_string())?;
    let _ = fs::copy(root.join("Cargo.lock"), destination.join("Cargo.lock"));
    fs::write(
        destination.join("Cargo.toml"),
        "[workspace]\n\
            \n\
            [package]\n\
            name = \"baux-semantics\"\n\
            version = \"0.0.1\"\n\
            edition = \"2024\"\n\
            rust-version = \"1.88\"\n\
            license = \"MIT\"\n\
            publish = false\n\
            \n\
            [dependencies]\n\
            deser = { version = \"=0.9.1\", features = [\"derive\"] }\n\
            rand = \"0.8\"\n",
    )
    .map_err(|error| error.to_string())?;
    Ok(destination)
}

fn resolve_tools(root: &Path, opts: &ExtractOpts) -> Result<(PathBuf, PathBuf), String> {
    let tools = root.join(".cache/baux/tools/aeneas");
    let mut aeneas = tools.join("bin/aeneas");
    let charon_root = tools.join("charon");
    let mut charon = charon_root.join("bin/charon");
    require_tool(&aeneas, "Aeneas", AENEAS_REV)?;
    require_tool(&charon, "Charon", CHARON_REV)?;
    require_revision(&tools, "Aeneas", AENEAS_REV)?;
    require_revision(&charon_root, "Charon", CHARON_REV)?;
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
            "float_patch_sha256={}\n\
                dyn_patch_sha256={}\n\
                dyn_dispatch_patch_sha256={}\n\
                dyn_reborrow_patch_sha256={}\n\
                dyn_builder_sha256={}\n",
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
            "local_revision={}\n\
                production_source_tree_sha256={}\n\
                production_manifest_sha256={}\n\
                production_extraction_root_sha256={}\n",
            jj_rev(root)?,
            tree_digest(&root.join("rust/crates/baux/src"))?,
            baux::artifact::runs::file_digest(&root.join("rust/crates/baux/Cargo.toml"))?,
            baux::artifact::runs::file_digest(&root.join("rust/crates/baux/src/extraction.rs"))?,
        ));
    }
    if opts.local {
        let (_, prod_digest) = slice_info.ok_or("local extraction slice was not prepared")?;
        return Ok(format!(
            "local_revision={}\n\
                local_slice_manifest_sha256={}\n\
                local_slice_root_sha256={}\n\
                production_source_tree_sha256={prod_digest}\n\
                production_extraction_root_sha256={}\n",
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

fn init_probe(
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
            "aeneas_base={AENEAS_REV}\n\
                charon={CHARON_REV}\n\
                aeneas_binary_sha256={}\n\
                charon_binary_sha256={}\n\
                source={source_mode}\n\
                {extension}{source_binding}",
            baux::artifact::runs::file_digest(tools.0)?,
            baux::artifact::runs::file_digest(tools.1)?,
        ),
    )
    .map_err(|error| error.to_string())?;
    println!("baux: extraction probe evidence: {}", path.display());
    Ok(Some(path))
}

fn resolve_source(
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

fn charon_cmd(
    charon: &Path,
    source: &Path,
    cache: &Path,
    llbc: &Path,
    opts: &ExtractOpts,
) -> Command {
    let mut lower = Command::new(charon);
    lower.current_dir(source).args(["cargo", "--preset=aeneas"]);
    if opts.local && !opts.production {
        lower.args(["--sysroot", "default"]);
    }
    let dir = if opts.upstream {
        "upstream-target"
    } else if opts.production {
        "local-production-target"
    } else if opts.local {
        "local-default-target"
    } else {
        "semantics-target"
    };
    lower.env("CARGO_TARGET_DIR", cache.join(dir));
    let actions: Vec<&str> = if opts.probe.is_empty() {
        if opts.tracked_local() {
            LOCAL_ROOTS.to_vec()
        } else if opts.floats {
            FLOAT_ROOTS.to_vec()
        } else {
            ACTION_ROOTS.to_vec()
        }
    } else {
        opts.probe.iter().map(String::as_str).collect()
    };
    for action in actions {
        lower.args(["--start-from", action]);
    }
    for pattern in opts.opaque {
        lower.args(["--opaque", pattern]);
    }
    for pattern in opts.include {
        lower.args(["--include", pattern]);
    }
    lower
        .arg("--dest-file")
        .arg(llbc)
        .args(["--", "--no-default-features"]);
    if opts.local && !opts.production {
        lower.args(["--offline", "--features", "extraction"]);
    } else if opts.production {
        lower.args(["--locked", "--lib", "--features", "extraction"]);
    } else if opts.upstream {
        lower.arg("--locked");
    } else {
        lower.arg("--offline");
    }
    lower
}

fn run_charon(
    lower: Command,
    source: &Path,
    llbc: &Path,
    probe_root: Option<&Path>,
    prod_digest: Option<&str>,
) -> Result<(), String> {
    let lowered = extraction_step(lower, probe_root, "charon").and_then(|()| {
        if let Some(initial) = prod_digest {
            if tree_digest(&source.join("src"))? != initial {
                return Err("production optimizer source changed during extraction".into());
            }
        }
        let bytes = fs::read(llbc).map_err(|error| error.to_string())?;
        if let Some(path) = probe_root {
            fs::write(
                path.join("llbc.sha256"),
                baux::artifact::runs::digest(&bytes),
            )
            .map_err(|error| error.to_string())?;
        }
        crate::research::validate_llbc(&bytes)
    });
    if let Err(error) = &lowered {
        if let Some(path) = probe_root {
            fs::write(
                path.join("result.txt"),
                format!("Lowering rejected: {error}\n"),
            )
            .map_err(|e| e.to_string())?;
        }
    }
    lowered
}

pub fn extract(root: &Path, opts: &ExtractOpts) -> Result<(), String> {
    let family = opts.family();
    let namespace = format!("BauxResearch.Generated.{family}");
    let relative = format!("BauxResearch/Generated/{family}");
    let (aeneas, charon) = resolve_tools(root, opts)?;

    let cache = root.join(".cache/baux/aeneas");
    fs::create_dir_all(&cache)
        .map_err(|error| format!("cannot create {}: {error}", cache.display()))?;
    let slice_info = if opts.local && !opts.production {
        Some(local_slice(root, &cache)?)
    } else {
        None
    };
    let probe_root = init_probe(root, &cache, opts, (&aeneas, &charon), slice_info.as_ref())?;
    let llbc = probe_root
        .as_deref()
        .unwrap_or(&cache)
        .join(format!("{family}.llbc"));
    let source = resolve_source(root, opts, probe_root.as_deref(), slice_info.as_ref())?;
    let prod_digest = if opts.production {
        Some(tree_digest(&source.join("src"))?)
    } else {
        None
    };
    let lower = charon_cmd(&charon, &source, &cache, &llbc, opts);
    run_charon(
        lower,
        &source,
        &llbc,
        probe_root.as_deref(),
        prod_digest.as_deref(),
    )?;

    let lean_root = root.join("research/lean");
    let check_root = cache.join(format!("check-{}", std::process::id()));
    let destination = prepare_dest(&lean_root, &check_root, probe_root.as_deref(), opts.check)?;
    let translated = run_aeneas(
        root,
        &aeneas,
        &llbc,
        (&namespace, &relative),
        destination,
        probe_root.as_deref(),
    );
    finish_extract(
        opts,
        family,
        &relative,
        (&lean_root, &check_root),
        probe_root.as_deref(),
        translated,
    )
}

fn prepare_dest<'a>(
    lean_root: &'a Path,
    check_root: &'a Path,
    probe_root: Option<&'a Path>,
    check: bool,
) -> Result<&'a Path, String> {
    if let Some(path) = probe_root {
        return Ok(path);
    }
    if check {
        if check_root.exists() {
            fs::remove_dir_all(check_root)
                .map_err(|error| format!("cannot clear {}: {error}", check_root.display()))?;
        }
        fs::create_dir_all(check_root)
            .map_err(|error| format!("cannot create {}: {error}", check_root.display()))?;
        return Ok(check_root);
    }
    sync_aeneas(lean_root)?;
    Ok(lean_root)
}

fn run_aeneas(
    root: &Path,
    aeneas: &Path,
    llbc: &Path,
    target: (&str, &str),
    destination: &Path,
    probe_root: Option<&Path>,
) -> Result<(), String> {
    let (namespace, relative) = target;
    let mut translate = Command::new(aeneas);
    translate
        .current_dir(root)
        .args(["-backend", "lean", "-namespace", namespace, "-dest"])
        .arg(destination)
        .args(["-subdir", relative, "-split-files"])
        .arg(llbc);
    extraction_step(translate, probe_root, "aeneas").and_then(|()| {
        let generated = destination.join(relative).join("Funs.lean");
        let text = fs::read_to_string(&generated).map_err(|error| error.to_string())?;
        if text.contains(&format!("{namespace}.FunsExternal")) {
            return Err(
                "Aeneas extraction requires unsupported external functions; their semantics and correspondence must be checked before accepting this translation"
                    .into(),
            );
        }
        Ok(())
    })
}

fn finish_extract(
    opts: &ExtractOpts,
    family: &str,
    relative: &str,
    roots: (&Path, &Path),
    probe_root: Option<&Path>,
    translated: Result<(), String>,
) -> Result<(), String> {
    let (lean_root, check_root) = roots;
    if let Some(path) = probe_root {
        let result = match &translated {
            Ok(()) => {
                "Translation produced; Lean elaboration and correspondence are not checked.\n"
                    .to_owned()
            }
            Err(error) => format!("Translation rejected: {error}\n"),
        };
        fs::write(path.join("result.txt"), result).map_err(|error| error.to_string())?;
        return translated;
    }
    let float_backend = opts.float_backend();
    if opts.check {
        let checked = translated.and_then(|()| {
            if float_backend {
                crate::research::compare_family(lean_root, check_root, relative)
            } else {
                crate::research::compare_translation(lean_root, check_root)
            }
        });
        let cleaned = fs::remove_dir_all(check_root)
            .map_err(|error| format!("cannot clear {}: {error}", check_root.display()));
        checked?;
        cleaned?;
        println!("baux: generated {family} semantics are current");
        return Ok(());
    }
    translated?;
    println!(
        "baux: extracted {family} with Aeneas {}{} and Charon {}",
        short_revision(AENEAS_REV),
        if float_backend {
            " + tracked float extension"
        } else {
            ""
        },
        short_revision(CHARON_REV)
    );
    Ok(())
}

fn extraction_step(
    mut command: Command,
    evidence: Option<&Path>,
    stage: &str,
) -> Result<(), String> {
    let Some(evidence) = evidence else {
        return crate::execute(command);
    };
    fs::write(
        evidence.join(format!("{stage}.command.txt")),
        format!("{command:?}\n"),
    )
    .map_err(|error| error.to_string())?;
    let output = command.output().map_err(|error| {
        format!(
            "cannot run {stage}; probe evidence at {}: {error}",
            evidence.display()
        )
    })?;
    fs::write(evidence.join(format!("{stage}.stdout.log")), &output.stdout)
        .map_err(|error| error.to_string())?;
    fs::write(evidence.join(format!("{stage}.stderr.log")), &output.stderr)
        .map_err(|error| error.to_string())?;
    fs::write(
        evidence.join(format!("{stage}.status.txt")),
        format!("{}\n", output.status),
    )
    .map_err(|error| error.to_string())?;
    if output.status.success() {
        Ok(())
    } else {
        Err(format!(
            "{stage} failed ({}); inspect {}",
            output.status,
            evidence.display()
        ))
    }
}

fn sync_aeneas(lean_root: &Path) -> Result<(), String> {
    let manifest = lean_root.join("lake-manifest.json");
    let current = fs::read_to_string(&manifest).unwrap_or_default();
    if current.contains("\"name\": \"aeneas\"") && current.contains(AENEAS_REV) {
        return Ok(());
    }
    let mut update = Command::new("lake");
    update
        .current_dir(lean_root)
        .env("MATHLIB_NO_CACHE_ON_UPDATE", "1")
        .args(["update", "aeneas"]);
    crate::execute(update)
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

fn short_revision(revision: &str) -> &str {
    &revision[..7]
}
