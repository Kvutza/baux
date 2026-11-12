use std::fs;
use std::path::Path;
use std::process::Command;

#[path = "extract_slice.rs"]
pub(crate) mod extract_slice;

use extract_slice::{
    compare_family, compare_translation, init_probe, local_slice, resolve_source, resolve_tools,
    short_revision, tree_digest, validate_llbc,
};

pub(super) const AENEAS_REV: &str = "e62dd83d133d9043e7d3628096838b7430f36565";
pub(super) const CHARON_REV: &str = "de8736f7c995ac54dc490345cef9ea5143c06c62";
const FLOAT_ROOTS: [&str; 8] = [
    "crate::random::u64_f53",
    "crate::random::box_muller",
    "crate::random::from_unit",
    "crate::random::to_unit",
    "crate::float::bits::single_from",
    "crate::float::bits::single_bits",
    "crate::float::bits::double_from",
    "crate::float::bits::double_bits",
];
const LOCAL_ROOTS: [&str; 5] = [
    "crate::coordinate_flat::from_unit",
    "crate::coordinate_flat::to_unit",
    "crate::optimizer_dispatch::ask_gate",
    "crate::optimizer_dispatch::tell_gate",
    "crate::optimizer_dispatch::variance_gate",
];
const ACTION_ROOTS: [&str; 24] = [
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
    "crate::optimizer::provenance::same_world",
    "crate::optimizer::provenance::exact_anchored",
    "crate::optimizer::param_tree::routed_count",
    "crate::optimizer::param_tree::total_count",
    "crate::optimizer::transaction::commit_round",
    "crate::float::single_lt",
    "crate::float::double_lt",
    "crate::float::to_double",
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
    pub(super) fn tracked_local(&self) -> bool {
        self.local && self.probe.is_empty()
    }

    pub(super) fn float_backend(&self) -> bool {
        self.floats || self.dyn_traits || self.tracked_local() || self.production
    }

    pub(super) fn family(&self) -> &'static str {
        if self.tracked_local() {
            "LocalCoord"
        } else if self.floats {
            "Floating"
        } else {
            "Optimizer"
        }
    }

    pub(super) fn source_mode(&self) -> &'static str {
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
                baux::artifact::blake3::sha256_hex(&bytes),
            )
            .map_err(|error| error.to_string())?;
        }
        validate_llbc(&bytes)
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
                compare_family(lean_root, check_root, relative)
            } else {
                compare_translation(lean_root, check_root)
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
