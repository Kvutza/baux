use clap::Subcommand;
use std::fs;
use std::io::{self, Read, Write};
use std::path::Path;
use std::process::{Command, Stdio};

use crate::cli_cfg::BuildSystem;
use crate::ext_ops as extract;

const RUNTIME_TARGET: &str = "//rust/crates/runner:baux-runner";

#[derive(Subcommand)]
pub enum ResearchAction {
    /// Check all Lean proofs and the typed experiment contract.
    #[command(
        long_about = "Check all Lean proofs and the typed experiment contract.\n\nThe default \
                      reuses Lake's local build cache. Use --via-build to route the Lean build \
                      through the selected Buck2 or Bazel target. Both paths currently use the \
                      host's pinned Lake toolchain."
    )]
    Check {
        /// Route the Lean build through Buck2 or Bazel instead of local Lake.
        #[arg(long = "via-build", visible_alias = "isolated", alias = "clean")]
        via_build: bool,
    },
    /// Recheck accepted proofs with every kernel shipped by Lean (requires
    /// bubblewrap).
    Audit,
    /// Show the checked research frontier.
    #[command(visible_alias = "st")]
    Status,
    /// List formalized construction topics.
    #[command(
        visible_alias = "ls",
        long_about = "List formalized construction topics.\n\nAn optional query performs a \
                      case-insensitive substring filter. Use `show TOPIC` to read one complete \
                      construction record."
    )]
    List {
        /// Keep only topics containing this text.
        query: Option<String>,
    },
    /// Show a construction or Lean declaration.
    #[command(
        alias = "inspect",
        long_about = "Show a construction or Lean declaration.\n\nA construction topic is read \
                      immediately from the research catalogue. A fully qualified Lean declaration \
                      is elaborated in the pinned BAUX environment and reports its type, axioms, \
                      home module, and minimal import."
    )]
    Show {
        /// Exact construction topic or fully qualified Lean declaration.
        name: String,
        /// Print the complete proof or definition body.
        #[arg(short, long)]
        body: bool,
        /// Load all of Mathlib instead of the BAUX import closure.
        #[arg(long, visible_alias = "all-mathlib")]
        mathlib: bool,
    },
    /// Search for an exact proof of a Lean proposition.
    #[command(
        alias = "suggest",
        long_about = "Search for an exact proof of a Lean proposition.\n\nThis runs Lean's \
                      indexed `exact?` search. The result is a candidate, not an accepted \
                      theorem; place it in a declaration and run `./baux research check` for \
                      kernel replay. Pass '-' to read the proposition from standard input."
    )]
    Prove {
        /// Closed proposition, or '-' to read it from standard input.
        proposition: String,
        /// Load all of Mathlib instead of the BAUX import closure.
        #[arg(long, visible_alias = "all-mathlib")]
        mathlib: bool,
    },
    /// Search Mathlib by mathematical meaning or Lean type structure.
    #[command(long_about = "Search Mathlib by mathematical meaning or Lean type \
                            structure.\n\nThe default sends a natural-language query through \
                            Mathlib's LeanSearch client. --structural sends Lean syntax through \
                            its Loogle client. Search results are discovery evidence; inspect a \
                            result with `show`, then kernel-check any theorem that uses it.")]
    Search {
        /// Natural-language question or Loogle structural pattern.
        query: String,
        /// Interpret the query as a Loogle type pattern.
        #[arg(long)]
        structural: bool,
    },
    /// Start a persistent Lean proof session.
    Repl,
    /// Extract the production optimizer decision law from Rust into Lean.
    #[command(
        long_about = "Extract the production optimizer decision law from Rust into \
                      Lean.\n\nCharon reads the selected Rust call graph and emits LLBC; Aeneas \
                      translates that LLBC into the checked Lean model under \
                      BauxResearch/Generated. The locally cached tools must match the \
                      repository's pinned commits. Use --check to reject drift without modifying \
                      tracked files."
    )]
    Extract {
        /// Verify that the checked-in Lean translation matches the current Rust
        /// source.
        #[arg(long)]
        check: bool,
        /// Probe selected Rust roots into retained cache artifacts, without
        /// changing checked-in Lean.
        #[arg(long, value_name = "RUST_PATH", conflicts_with = "check")]
        probe: Vec<String>,
        /// Build the tracked experimental FloatLib extension over the pinned
        /// Aeneas revision.
        #[arg(long)]
        floats: bool,
        /// Probe with the tracked dynamic-trait Aeneas extension.
        #[arg(long, requires = "probe")]
        dyn_traits: bool,
        /// Build the tracked callsite-external Charon extension over the pinned
        /// revision.
        #[arg(long, requires = "probe")]
        charon_callsite: bool,
        /// Probe the pinned upstream ENN crate in an isolated source snapshot.
        #[arg(long, requires = "probe", conflicts_with = "local")]
        upstream: bool,
        /// Probe the local production optimizer crate at the current JJ
        /// revision. Without --probe, regenerate or check the tracked
        /// scalar coordinate kernel.
        #[arg(long, conflicts_with = "upstream")]
        local: bool,
        /// Probe the full production crate so public ask/tell roots remain
        /// reachable.
        #[arg(long, requires_all = ["local", "probe"])]
        production: bool,
        /// Keep matching Rust items opaque during a probe while retaining their
        /// signatures.
        #[arg(long, value_name = "RUST_PATH", requires = "probe")]
        opaque: Vec<String>,
        /// Make matching Rust items transparent, overriding a broader opaque
        /// boundary.
        #[arg(long, value_name = "RUST_PATH", requires = "probe")]
        include: Vec<String>,
    },
    /// Import an Anti-Gravity conversation as compact research context.
    #[command(
        long_about = "Import an Anti-Gravity conversation from its local SQLite store.\n\nBy \
                      default this prints the newest context summary followed by accepted user \
                      and assistant turns after that summary. Use --full for the complete human \
                      transcript and --tools to include tool calls. The database is opened \
                      read-only and step payloads are decoded in-process."
    )]
    Import {
        /// Anti-Gravity conversation UUID.
        conversation: String,
        /// Print the complete human transcript instead of resuming from the
        /// newest summary.
        #[arg(long)]
        full: bool,
        /// Include tool names and arguments; tool output remains omitted.
        #[arg(long)]
        tools: bool,
        /// Write the import to this file instead of standard output.
        #[arg(short, long)]
        output: Option<std::path::PathBuf>,
    },
    /// Check or regenerate the exact upstream optimizer coverage manifest.
    Coverage {
        /// Verify that the checked-in manifest matches the upstream tree
        /// without writing changes.
        #[arg(long)]
        check: bool,
    },
    /// Extract both Box-Muller implementations and generate a
    /// dependency-parametric comparison.
    Pair {
        /// Reject stale generated definitions instead of writing them.
        #[arg(long)]
        check: bool,
    },
    /// Check or regenerate the compiled optimizer artifact correspondence
    /// manifest.
    Artifact {
        /// Verify that the checked-in manifest matches compiled
        /// upstream/local/Lean artifacts.
        #[arg(long)]
        check: bool,
    },
    /// Build, verify, or serve the static interactive verification portal
    /// (`docs/site/`)
    Site {
        /// Check that generated portal data is up to date without rewriting
        /// files
        #[arg(long)]
        check: bool,
        /// Serve `docs/site/` locally over an embedded native Rust HTTP server
        #[arg(long)]
        serve: bool,
        /// Port for `--serve` (default: 8410)
        #[arg(long, default_value_t = 8410)]
        port: u16,
    },
    /// Compatibility spelling for the old catalogue command.
    #[command(hide = true)]
    Catalogue { topic: Option<String> },
    /// Produce a multi-target kernel scorecard from captured GPU binaries, PTX,
    /// SASS, or Metal archives.
    KernelScorecard {
        /// Captured GPU artifact file or directory.
        path: std::path::PathBuf,
        /// Optional substring filter on kernel name.
        #[arg(long)]
        kernel: Option<String>,
    },
    /// Diff two captured GPU artifact directories and report only changed
    /// kernels and aggregate deltas.
    KernelDiff {
        /// Baseline capture file or directory.
        before: std::path::PathBuf,
        /// Candidate capture file or directory.
        after: std::path::PathBuf,
    },
    /// Extract a bounded basic-block or hazard instruction window (max 60
    /// lines) for a kernel.
    KernelSlice {
        /// Captured GPU artifact file or directory.
        path: std::path::PathBuf,
        /// Exact kernel entry name.
        #[arg(long)]
        kernel: String,
        /// Hazard class (`spill`, `fp64`, `rcp`, `trap`, `bsync`, `mma`) or
        /// basic-block/mnemonic substring.
        #[arg(long)]
        filter: Option<String>,
        /// Context lines around each matched instruction (capped at 25).
        #[arg(long, default_value_t = 8)]
        context: usize,
    },
}

pub fn run(
    root: &std::path::Path,
    action: ResearchAction,
    build_system: BuildSystem,
) -> Result<(), String> {
    match action {
        ResearchAction::Coverage { check } => coverage(root, check),
        ResearchAction::Pair { check } => pair(root, check),
        ResearchAction::Artifact { check } => artifact(root, check),
        ResearchAction::Site { check, serve, port } => site(root, check, serve, port),
        ResearchAction::Extract {
            check,
            probe,
            floats,
            dyn_traits,
            charon_callsite,
            upstream,
            local,
            production,
            opaque,
            include,
        } => extract::extract(
            root,
            &extract::ExtractOpts {
                check,
                probe: &probe,
                floats,
                dyn_traits,
                charon_callsite,
                upstream,
                local,
                production,
                opaque: &opaque,
                include: &include,
            },
        ),
        other => run_research(root, other, build_system),
    }
}

fn run_research(
    root: &std::path::Path,
    action: ResearchAction,
    build_system: BuildSystem,
) -> Result<(), String> {
    match action {
        ResearchAction::Check { via_build } => check(root, build_system, via_build),
        ResearchAction::Audit => {
            require_sandbox()?;
            check(root, build_system, false)?;
            let mut command = Command::new("lake");
            command
                .current_dir(root.join("research/lean"))
                .args(["check", "--paranoid"]);
            crate::execute(command)?;
            println!("baux: Lean proofs independently rechecked");
            Ok(())
        }
        ResearchAction::Status => {
            print!("{}", crate::research_gpu::report(root)?);
            catalogue(root, Some("frontier"))
        }
        ResearchAction::List { query } => list(root, query.as_deref()),
        ResearchAction::Show {
            name,
            body,
            mathlib,
        } => show(root, &name, body, mathlib),
        ResearchAction::Prove {
            proposition,
            mathlib,
        } => prove(root, &proposition, mathlib),
        ResearchAction::Search { query, structural } => search(root, &query, structural),
        ResearchAction::Repl => {
            let mut command = Command::new("lake");
            command
                .current_dir(root.join("research/lean"))
                .args(["exe", "repl"]);
            crate::execute(command)
        }
        ResearchAction::Import {
            conversation,
            full,
            tools,
            output,
        } => crate::research_agy::import(&conversation, full, tools, output.as_deref()),
        ResearchAction::Catalogue { topic } => catalogue(root, topic.as_deref()),
        other => run_kernel(other),
    }
}

fn run_kernel(action: ResearchAction) -> Result<(), String> {
    let val = match action {
        ResearchAction::KernelScorecard { path, kernel } => {
            baux::artifact::gpu::inspect::inspect_filtered(&path, kernel.as_deref())?
        }
        ResearchAction::KernelDiff { before, after } => {
            baux::artifact::gpu::inspect::diff_directories(&before, &after)?
        }
        ResearchAction::KernelSlice {
            path,
            kernel,
            filter,
            context,
        } => {
            baux::artifact::gpu::inspect::slice_kernel(&path, &kernel, filter.as_deref(), context)?
        }
        _ => unreachable!("handled by run"),
    };
    println!(
        "{}",
        baux::artifact::json::pretty_string(&val).map_err(|e| e.to_string())?
    );
    Ok(())
}

pub(super) fn validate_llbc(bytes: &[u8]) -> Result<(), String> {
    let value: baux::artifact::json::Value = baux::artifact::json::from_slice(bytes)
        .map_err(|error| format!("cannot parse Charon LLBC: {error}"))?;
    if value.get("has_errors").and_then(|flag| flag.as_bool()) != Some(false) {
        return Err("Charon LLBC reports errors or lacks an explicit error-free status".into());
    }
    Ok(())
}

fn pair(root: &Path, check: bool) -> Result<(), String> {
    baux_tooling::box_pair::run(root, check)?;
    let mut build = Command::new("lake");
    build
        .current_dir(root.join("research/lean"))
        .args(["build", "BauxResearch.Extraction.Box"]);
    crate::execute(build)?;
    lean_stdin(
        root,
        "import BauxResearch.Extraction.Box\n#check \
         @BauxResearch.Extraction.Box.equivalent\n#print axioms \
         BauxResearch.Extraction.Box.equivalent\n",
    )
}

pub(super) fn compare_translation(lean_root: &Path, check_root: &Path) -> Result<(), String> {
    compare_family(lean_root, check_root, "BauxResearch/Generated/Optimizer")
}

pub(super) fn compare_family(
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

fn require_sandbox() -> Result<(), String> {
    let sandbox = std::env::var_os("COMPARATOR_BWRAP").unwrap_or_else(|| "bwrap".into());
    let available = Command::new(&sandbox)
        .arg("--version")
        .output()
        .is_ok_and(|output| output.status.success());
    if available {
        Ok(())
    } else {
        Err(
            "Lean's sandboxed audit requires bubblewrap; run `./baux research audit` on a Linux \
             host with bwrap installed"
                .to_owned(),
        )
    }
}

fn check(root: &Path, build_system: BuildSystem, via_build: bool) -> Result<(), String> {
    println!("o|research [1/7]: verifying upstream optimizer coverage manifest...");
    coverage(root, true)?;
    println!("o|research [2/7]: verifying Charon/Aeneas Rust -> Lean extraction...");
    extract::extract(
        root,
        &extract::ExtractOpts {
            check: true,
            ..extract::ExtractOpts::default()
        },
    )?;
    extract::extract(
        root,
        &extract::ExtractOpts {
            check: true,
            floats: true,
            ..extract::ExtractOpts::default()
        },
    )?;
    extract::extract(
        root,
        &extract::ExtractOpts {
            check: true,
            local: true,
            ..extract::ExtractOpts::default()
        },
    )?;
    pair(root, true)?;
    println!("o|research [3/7]: checking Lean naming policy...");
    crate::research_gpu::lint_names(root)?;
    let target = if via_build {
        println!("o|research [4/7]: building //research/lean:check via build system...");
        let artifact = build_system.build_artifact(root, "//research/lean:check")?;
        fs::read_to_string(&artifact.path).map_err(|error| {
            format!(
                "cannot read checked Lean artifact {}: {error}",
                artifact.path.display()
            )
        })?
    } else {
        println!("o|research [4/7]: building BauxResearch via Lake...");
        ensure_cache(root)?;
        let mut build = Command::new("lake");
        build
            .current_dir(root.join("research/lean"))
            .args(["build", "BauxResearch"]);
        crate::execute(build)?;
        println!("o|research [5/7]: verifying Rust <-> Lean 4 C-ABI parity...");
        verify_cabi(root)?;
        println!("o|research [6/7]: running `lake env lean --run Main.lean`...");
        let output = Command::new("lake")
            .current_dir(root.join("research/lean"))
            .args(["env", "lean", "--run", "Main.lean"])
            .stderr(Stdio::inherit())
            .output()
            .map_err(|error| format!("cannot run checked Lean target: {error}"))?;
        if !output.status.success() {
            return Err("checked Lean target executable failed".into());
        }
        String::from_utf8(output.stdout)
            .map_err(|error| format!("Lean target output is not UTF-8: {error}"))?
    };
    println!("o|research [7/7]: verifying numerical C-ABI & research contract...");
    baux_tooling::numerical::verify(root, build_system)?;
    if !via_build {
        artifact(root, true)?;
        check_site(root)?;
    }
    check_contract(root, build_system, &target)?;
    print!("{}", crate::research_gpu::check(root)?);
    println!("baux: Lean optimizer formalization checked");
    Ok(())
}

fn artifact(root: &Path, check: bool) -> Result<(), String> {
    baux_tooling::enn_artifact::run(root, check)
}

fn site(root: &Path, check: bool, serve: bool, port: u16) -> Result<(), String> {
    crate::research_site::run(root, check, serve, port)
}

fn check_site(root: &Path) -> Result<(), String> {
    crate::research_site::verify_fresh(root)
}

fn coverage(root: &Path, check: bool) -> Result<(), String> {
    baux_tooling::enn_coverage::run(root, check)
}

fn ensure_cache(root: &Path) -> Result<(), String> {
    let lean_root = root.join("research/lean");
    let mathlib =
        lean_root.join(".lake/packages/mathlib/.lake/build/lib/lean/Mathlib/Data/Real/Basic.olean");
    if mathlib.is_file() {
        return Ok(());
    }
    let mut cache = Command::new("lake");
    cache.current_dir(lean_root).args(["exe", "cache", "get"]);
    crate::execute(cache)
}

fn verify_cabi(root: &Path) -> Result<(), String> {
    let archive = root.join("research/lean/.lake/build/lib/libbaux__research_BauxResearch.a");
    if !archive.is_file() {
        return Err(format!(
            "missing compiled Lean 4 C-ABI static archive {}",
            archive.display()
        ));
    }
    let bytes = fs::read(&archive)
        .map_err(|error| format!("cannot read {}: {error}", archive.display()))?;
    for symbol in [
        "baux_accept",
        "baux_action",
        "baux_world",
        "baux_anchored",
        "baux_routed",
        "baux_total",
        "baux_window",
        "baux_cell",
        "baux_slot",
        "baux_sign",
    ] {
        if !bytes
            .windows(symbol.len())
            .any(|window| window == symbol.as_bytes())
        {
            return Err(format!(
                "compiled Lean 4 static archive lacks C-ABI symbol `{symbol}`"
            ));
        }
    }
    let cache = root.join(".cache/baux/cabi");
    fs::create_dir_all(&cache)
        .map_err(|error| format!("cannot create {}: {error}", cache.display()))?;
    let sem_dir = root.join("rust/crates/baux/semantics/src");
    let harness_src = cache.join("cabi_parity.rs");
    let harness_bin = cache.join("cabi_parity");
    fs::write(
        &harness_src,
        format!(
            r#"#![allow(dead_code)]
extern crate self as baux;
pub mod semantics {{
    pub use crate::*;
    pub mod optimizer {{
        pub use crate::*;
        pub mod surrogate {{
            pub use crate::radius::*;
        }}
    }}
}}
#[path = "{sem}/optimizer/calibration.rs"]
pub mod calibration;
#[path = "{sem}/numerical.rs"]
pub mod numerical;
#[path = "{sem}/optimizer/trust_region/controller.rs"]
pub mod radius;
#[path = "{sem}/optimizer/provenance.rs"]
pub mod provenance;
#[path = "{sem}/optimizer/param_tree.rs"]
pub mod param_tree;
#[path = "{sem}/optimizer/window_plan.rs"]
pub mod window_plan;
#[path = "{sem}/optimizer/reuse.rs"]
pub mod reuse;
#[path = "{sem}/neighbor.rs"]
pub mod neighbor;

use param_tree::{{ParamTree, routed_count, total_count}};
use provenance::{{exact_anchored, same_world}};
use radius::{{RadiusAction, accept_bits, decide_action}};
use reuse::{{conditioned_sign, conditioned_slot}};
use window_plan::verify_full;

unsafe extern "C" {{
    fn baux_accept(improvement: u64, margin: u64) -> u8;
    fn baux_action(
        adequate: u8,
        inadequate: u8,
        sharp: u8,
        accepted: u8,
        positive: u8,
        boundary: u8,
        failure: u8,
    ) -> u8;
    fn baux_world(left: u64, right: u64) -> u8;
    fn baux_anchored(is_fixed: u8, from_step: u64, to_step: u64) -> u8;
    fn baux_routed(patch: u64) -> u64;
    fn baux_total(patch: u64) -> u64;
    fn baux_window(planned: u64, prefilled: u64, scored: u64) -> u8;
    fn baux_cell(level: u32, grid: u32) -> u64;
    fn baux_slot(slot: u64, accepted: u8) -> u64;
    fn baux_sign(reversed: u8, accepted: u8) -> u8;
}}

fn main() {{
    let samples: [u64; 12] = [
        0x0000_0000_0000_0000,
        0x8000_0000_0000_0000,
        0x0000_0000_0000_0001,
        0x8000_0000_0000_0001,
        0x3ff0_0000_0000_0000,
        0xbff0_0000_0000_0000,
        0x4000_0000_0000_0000,
        0xc000_0000_0000_0000,
        0x7ff0_0000_0000_0000,
        0xfff0_0000_0000_0000,
        0x7ff8_0000_0000_0001,
        0xfff8_0000_0000_0001,
    ];
    for &imp in &samples {{
        for &mar in &samples {{
            let rust_acc = accept_bits(imp, mar);
            let lean_acc = unsafe {{ baux_accept(imp, mar) != 0 }};
            assert_eq!(rust_acc, lean_acc, "accept_bits mismatch for ({{imp:#x}}, {{mar:#x}})");
        }}
    }}
    for mask in 0u8..128 {{
        let b = |bit: u8| (mask >> bit) & 1 != 0;
        let rust_act = match decide_action(b(0), b(1), b(2), b(3), b(4), b(5), b(6)) {{
            RadiusAction::Hold => 0u8,
            RadiusAction::Expand => 1u8,
            RadiusAction::Shrink => 2u8,
        }};
        let lean_act = unsafe {{
            baux_action(
                u8::from(b(0)),
                u8::from(b(1)),
                u8::from(b(2)),
                u8::from(b(3)),
                u8::from(b(4)),
                u8::from(b(5)),
                u8::from(b(6)),
            )
        }};
        assert_eq!(rust_act, lean_act, "decide_action mismatch at mask {{mask:#09b}}");
    }}
    for &(l, r) in &[(0u64, 0u64), (42, 42), (42, 43), (u64::MAX, u64::MAX - 1)] {{
        assert_eq!(same_world(l, r), unsafe {{ baux_world(l, r) != 0 }});
        for fixed in [false, true] {{
            assert_eq!(
                exact_anchored(fixed, l, r),
                unsafe {{ baux_anchored(u8::from(fixed), l, r) != 0 }}
            );
        }}
    }}
    assert_eq!(routed_count(), unsafe {{ baux_routed(1) }});
    assert_eq!(total_count(), unsafe {{ baux_total(1) }});
    assert_eq!(
        ParamTree::<usize>::total_parameters(1) as u64,
        unsafe {{ baux_total(1) }}
    );
    for &(planned, prefilled, scored) in &[
        (0u64, 0u64, 0u64),
        (4096, 0, 4096),
        (1_048_576, 0, 1_048_576),
        (1_048_576, 2048, 1024),
        (4096, 0, 2048),
    ] {{
        assert_eq!(
            verify_full(planned, prefilled, scored),
            unsafe {{ baux_window(planned, prefilled, scored) != 0 }}
        );
    }}
    assert_eq!(
        neighbor::grid_cell(7.0, 8.0),
        unsafe {{ baux_cell(3, 0) }}
    );
    for slot in 0u64..64 {{
        for acc in [false, true] {{
            assert_eq!(
                conditioned_slot(slot, acc),
                unsafe {{ baux_slot(slot, u8::from(acc)) }}
            );
        }}
    }}
    for rev in [false, true] {{
        for acc in [false, true] {{
            assert_eq!(
                conditioned_sign(rev, acc),
                unsafe {{ baux_sign(u8::from(rev), u8::from(acc)) != 0 }}
            );
        }}
    }}
}}
"#,
            sem = sem_dir.display()
        ),
    )
    .map_err(|error| format!("cannot write {}: {error}", harness_src.display()))?;
    let strip_flag = if cfg!(target_os = "macos") {
        "-C\nlink-arg=-Wl,-dead_strip"
    } else {
        "-C\nlink-arg=-Wl,--gc-sections"
    };
    let mut rustc_bin = Command::new("rustc");
    rustc_bin
        .arg("--edition=2024")
        .arg("--cfg=cabi_parity")
        .arg(&harness_src)
        .arg(format!("-Clink-arg={}", archive.display()));
    for arg in strip_flag.lines() {
        rustc_bin.arg(arg);
    }
    rustc_bin.arg("-o").arg(&harness_bin);
    crate::execute(rustc_bin)?;
    crate::execute(Command::new(&harness_bin))?;
    Ok(())
}

fn check_contract(root: &Path, build_system: BuildSystem, target: &str) -> Result<(), String> {
    let value = |key: &str| -> Result<&str, String> {
        target
            .lines()
            .find_map(|line| line.strip_prefix(key))
            .map(str::trim)
            .ok_or_else(|| format!("Lean target output lacks `{key}`"))
    };
    let number = |key: &str| -> Result<u64, String> {
        value(key)?
            .parse::<u64>()
            .map_err(|error| format!("invalid Lean target value for `{key}`: {error}"))
    };

    let args = vec![
        "research-contract".to_owned(),
        "performance/configs/mhc4-1m-causal-1000.toml".to_owned(),
        "--parameters".to_owned(),
        number("target.parameters: ")?.to_string(),
        "--context-tokens".to_owned(),
        number("target.context_tokens: ")?.to_string(),
        "--candidates".to_owned(),
        number("target.candidates: ")?.to_string(),
        "--neighbors".to_owned(),
        number("target.neighbors: ")?.to_string(),
        "--target-ms".to_owned(),
        number("target.transaction_target_ms: ")?.to_string(),
        "--ceiling-ms".to_owned(),
        number("target.transaction_ceiling_ms: ")?.to_string(),
        "--dense-flops".to_owned(),
        number("target.dense_flops: ")?.to_string(),
        "--objective".to_owned(),
        value("target.objective: ")?.to_owned(),
        "--sampling".to_owned(),
        value("target.sampling: ")?.to_owned(),
    ];
    build_system.run(root, RUNTIME_TARGET, &args)
}

fn show(root: &Path, name: &str, body: bool, mathlib: bool) -> Result<(), String> {
    let contents = read_catalogue(root)?;
    if let Some(card) = find_card(&contents, name) {
        if body || mathlib {
            return Err("--body and --mathlib apply only to Lean declarations".to_owned());
        }
        println!("CONSTRUCTION {card}");
        return Ok(());
    }
    if !name.contains('.') {
        return Err(format!(
            "unknown construction `{}`; run `./baux research list`",
            name.trim()
        ));
    }
    inspect(root, name, body, mathlib)
}

fn inspect(root: &Path, declaration: &str, body: bool, all_mathlib: bool) -> Result<(), String> {
    let declaration = declaration.trim();
    if !declaration_ok(declaration) {
        return Err(format!(
            "invalid Lean declaration `{declaration}`; use a fully qualified name"
        ));
    }
    let body = if body {
        format!("#print {declaration}\n")
    } else {
        String::new()
    };
    let imports = if all_mathlib {
        "import Mathlib\nimport BauxResearch"
    } else {
        "import BauxResearch"
    };
    let input = format!(
        "{imports}\n\n#check @{declaration}\n{body}#print axioms {declaration}\n#find_home \
         {declaration}\n#min_imports in\n#check @{declaration}\n"
    );
    lean_stdin(root, &input)
}

fn prove(root: &Path, proposition: &str, all_mathlib: bool) -> Result<(), String> {
    let mut standard_input = String::new();
    let proposition = if proposition == "-" {
        io::stdin()
            .read_to_string(&mut standard_input)
            .map_err(|error| format!("could not read proposition: {error}"))?;
        standard_input.trim()
    } else {
        proposition.trim()
    };
    if proposition.is_empty() || proposition.contains('\n') || proposition.contains('\r') {
        return Err("Lean proposition must be one nonempty line".to_owned());
    }
    let imports = if all_mathlib {
        "import Mathlib\nimport BauxResearch"
    } else {
        "import BauxResearch"
    };
    let input = format!("{imports}\n\nexample : {proposition} := by\n  exact?\n");
    lean_stdin(root, &input)
}

fn search(root: &Path, query: &str, structural: bool) -> Result<(), String> {
    let query = query.trim();
    if query.is_empty() || query.contains('\n') || query.contains('\r') {
        return Err("theorem query must be one nonempty line".to_owned());
    }
    let command = if structural {
        format!("#loogle {query}")
    } else {
        let mut sentence = query.to_owned();
        if !sentence.ends_with(['.', '?']) {
            sentence.push('?');
        }
        format!("#leansearch {}", lean_string(&sentence))
    };
    lean_stdin(root, &format!("import Mathlib\n\n{command}\n"))
}

fn lean_string(value: &str) -> String {
    let mut output = String::with_capacity(value.len() + 2);
    output.push('"');
    for character in value.chars() {
        match character {
            '"' => output.push_str("\\\""),
            '\\' => output.push_str("\\\\"),
            character => output.push(character),
        }
    }
    output.push('"');
    output
}

fn lean_stdin(root: &Path, input: &str) -> Result<(), String> {
    let mut child = Command::new("lake")
        .current_dir(root.join("research/lean"))
        .args(["env", "lean", "--stdin"])
        .stdin(Stdio::piped())
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit())
        .spawn()
        .map_err(|error| format!("could not start Lean: {error}"))?;
    child
        .stdin
        .take()
        .ok_or_else(|| "could not open Lean input".to_owned())?
        .write_all(input.as_bytes())
        .map_err(|error| format!("could not send input to Lean: {error}"))?;
    let status = child
        .wait()
        .map_err(|error| format!("could not wait for Lean: {error}"))?;
    if status.success() {
        Ok(())
    } else {
        Err(format!("Lean feedback failed with {status}"))
    }
}

fn declaration_ok(declaration: &str) -> bool {
    !declaration.is_empty()
        && !declaration.starts_with('.')
        && !declaration.ends_with('.')
        && !declaration.contains("..")
        && declaration.split('.').all(|part| {
            part.chars()
                .next()
                .is_some_and(|character| character.is_ascii_alphabetic() || character == '_')
                && part.chars().all(|character| {
                    character.is_ascii_alphanumeric() || character == '_' || character == '\''
                })
        })
}

fn catalogue(root: &Path, topic: Option<&str>) -> Result<(), String> {
    let contents = read_catalogue(root)?;
    let Some(topic) = topic else {
        println!("{contents}");
        return Ok(());
    };
    if let Some(card) = find_card(&contents, topic) {
        println!("CONSTRUCTION {card}");
        return Ok(());
    }
    let topics = contents
        .split("\nCONSTRUCTION ")
        .skip(1)
        .filter_map(|card| card.lines().next())
        .collect::<Vec<_>>()
        .join(", ");
    Err(format!(
        "unknown construction `{topic}`; available topics: {topics}"
    ))
}

fn list(root: &Path, query: Option<&str>) -> Result<(), String> {
    let contents = read_catalogue(root)?;
    let query = query.map(str::trim).filter(|query| !query.is_empty());
    let matches = topics(&contents)
        .filter(|topic| {
            query.is_none_or(|query| {
                topic
                    .to_ascii_lowercase()
                    .contains(&query.to_ascii_lowercase())
            })
        })
        .collect::<Vec<_>>();
    if matches.is_empty() {
        return Err(format!(
            "no construction topic matches `{}`; run `./baux research list`",
            query.unwrap_or("")
        ));
    }
    for topic in matches {
        println!("{topic}");
    }
    Ok(())
}

fn read_catalogue(root: &Path) -> Result<String, String> {
    fs::read_to_string(root.join("docs/mathlib-constructions.txt"))
        .map_err(|error| format!("could not read construction catalogue: {error}"))
}

fn topics(contents: &str) -> impl Iterator<Item = &str> {
    contents
        .split("\nCONSTRUCTION ")
        .skip(1)
        .filter_map(|card| card.lines().next())
}

fn find_card<'a>(contents: &'a str, topic: &str) -> Option<&'a str> {
    contents
        .split("\nCONSTRUCTION ")
        .skip(1)
        .find(|card| card.lines().next().is_some_and(|name| name == topic.trim()))
}

#[cfg(test)]
mod tests {
    #[test]
    fn incomplete_llbc() {
        assert!(super::validate_llbc(br#"{"has_errors": false}"#).is_ok());
        for bytes in [
            br#"{"has_errors": true}"#.as_slice(),
            br#"{"has_errors": "false"}"#.as_slice(),
            br#"{}"#.as_slice(),
            b"partial JSON".as_slice(),
        ] {
            assert!(super::validate_llbc(bytes).is_err());
        }
    }

    use super::{declaration_ok, find_card, lean_string, topics};

    #[test]
    fn missing_extractor() {
        let root =
            std::env::temp_dir().join(format!("baux-missing-extractor-{}", std::process::id()));
        std::fs::create_dir(&root).unwrap();
        for check in [false, true] {
            let error = super::extract::extract(
                &root,
                &super::extract::ExtractOpts {
                    check,
                    ..super::extract::ExtractOpts::default()
                },
            )
            .unwrap_err();
            assert!(error.contains("Aeneas"), "{error}");
            assert!(error.contains("is not built"), "{error}");
        }
        std::fs::remove_dir(root).unwrap();
    }

    #[test]
    fn stale_translation() {
        let root = std::env::temp_dir().join(format!("baux-extract-test-{}", std::process::id()));
        let lean = root.join("lean");
        let fresh = root.join("fresh");
        let relative = "BauxResearch/Generated/Optimizer";
        for base in [&lean, &fresh] {
            std::fs::create_dir_all(base.join(relative)).unwrap();
            for name in ["Types.lean", "Funs.lean"] {
                std::fs::write(base.join(relative).join(name), "original").unwrap();
            }
        }
        assert!(super::compare_translation(&lean, &fresh).is_ok());
        std::fs::write(fresh.join(relative).join("Funs.lean"), "changed decision").unwrap();
        assert!(
            super::compare_translation(&lean, &fresh)
                .unwrap_err()
                .contains("is stale")
        );
        std::fs::remove_file(fresh.join(relative).join("Funs.lean")).unwrap();
        assert!(super::compare_translation(&lean, &fresh).is_err());
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn exact_topic() {
        let text = "Heading\nCONSTRUCTION parameter\nfirst\nCONSTRUCTION parameters\nsecond\n";
        assert_eq!(find_card(text, "parameter"), Some("parameter\nfirst"));
        assert_eq!(
            find_card(text, " parameters "),
            Some("parameters\nsecond\n")
        );
        assert!(find_card(text, "param").is_none());
        assert!(find_card(text, "missing").is_none());
    }

    #[test]
    fn topic_names() {
        let text = "Heading\nCONSTRUCTION parameter\nfirst\nCONSTRUCTION regret\nsecond\n";
        assert_eq!(topics(text).collect::<Vec<_>>(), ["parameter", "regret"]);
    }

    #[test]
    fn declaration_names() {
        assert!(declaration_ok("BauxResearch.Optimization.Rate.expected"));
        assert!(declaration_ok("Metric.mem_closedBall'"));
        assert!(!declaration_ok(""));
        assert!(!declaration_ok("Rate.expected\n#eval 1"));
        assert!(!declaration_ok("Rate..expected"));
        assert!(!declaration_ok("Rate.expected x"));
    }

    #[test]
    fn query_string() {
        assert_eq!(
            lean_string("say \"hello\" \\"),
            "\"say \\\"hello\\\" \\\\\""
        );
    }
}
