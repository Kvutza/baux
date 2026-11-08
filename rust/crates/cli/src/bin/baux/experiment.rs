use std::path::{Path, PathBuf};

use crate::cli_cfg::BuildSystem;

const RUNTIME_TARGET: &str = "//rust/crates/runner:baux-runner";

#[derive(clap::Args)]
pub struct Tune {
    #[arg(required_unless_present = "history", conflicts_with_all = ["history", "since", "until", "task", "optimizer", "limit", "reindex", "diff", "serve", "json"])]
    pub config: Option<PathBuf>,
    #[arg(long, conflicts_with_all = ["history"])]
    pub rounds: Option<u32>,
    #[arg(long, conflicts_with_all = ["history"])]
    pub standalone: bool,
    /// Browse saved runs; never launches training.
    #[arg(long)]
    pub history: bool,
    /// Inclusive local date (today/yesterday/YYYY-MM-DD) or RFC3339 instant.
    #[arg(long, requires = "history")]
    pub since: Option<String>,
    /// Exclusive date or instant; use with --since for a calendar day.
    #[arg(long, requires = "history")]
    pub until: Option<String>,
    /// Filter by Beads task, or 'current' for this workspace's active task.
    #[arg(long, requires = "history")]
    pub task: Option<String>,
    #[arg(long, requires = "history")]
    pub optimizer: Option<String>,
    #[arg(long, requires = "history", default_value_t = 20, value_parser = clap::value_parser!(u32).range(1..=100))]
    pub limit: u32,
    /// Rebuild the shared catalogue from saved run manifests.
    #[arg(long, requires = "history")]
    pub reindex: bool,
    /// Include JJ optimizer source diffs between compared runs.
    #[arg(long, requires = "history")]
    pub diff: bool,
    /// Serve the interactive HTML comparison dashboard on localhost (default
    /// port 8420).
    #[arg(long, requires = "history", num_args = 0..=1, default_missing_value = "8420")]
    pub serve: Option<u16>,
    #[arg(long, requires = "history")]
    pub json: bool,
}

pub(crate) fn tune(root: &Path, args: Tune, engine: BuildSystem) -> Result<(), String> {
    if !args.history {
        return execute_tune(root, args, engine);
    }
    use baux::artifact::{json, runs};
    let task = match args.task.as_deref() {
        Some("current") => Some(crate::work::active_task(root)?.ok_or("no active Beads task")?),
        _ => args.task,
    };
    let since = args.since.as_deref().map(runs::since).transpose()?;
    let until = args.until.as_deref().map(runs::since).transpose()?;
    if since.zip(until).is_some_and(|(a, b)| a >= b) {
        return Err("--since must precede --until".into());
    }
    if args.reindex {
        runs::reindex(root)?;
    }
    let records = runs::history(
        root,
        &runs::Query {
            since,
            until,
            task,
            optimizer: args.optimizer,
            limit: args.limit as usize,
            ..Default::default()
        },
    )?;
    let comparisons = runs::compare(root, &records).unwrap_or_else(|_| json::Value::null());
    let diffs = runs::source_diffs(root, &records, args.diff || args.serve.is_some());
    let dashboard = runs::write_html(root, &records, &comparisons, &diffs).ok();
    if args.json {
        println!(
            "{}",
            json::pretty_string(&json::json!({
                "runs": records,
                "graph": runs::graph(&records),
                "comparisons": comparisons,
                "source_diffs": diffs,
                "dashboard": dashboard.as_ref().map(|p| p.to_string_lossy()),
                "limit": args.limit,
                "catalogue": runs::directory(root)?.to_string_lossy()
            }))
            .map_err(|e| e.to_string())?
        );
    } else {
        if records.is_empty() {
            println!(
                "No matching recorded runs. Older runs without a run manifest are not indexed."
            );
        }
        let diff_seq = diffs.as_seq();
        for (idx, record) in records.iter().enumerate() {
            println!(
                "{}  {} / {} ({})  [{}]  task {}\n  source {} ({})\n  artifacts {}\n  report {}{}",
                runs::local_time(record.started)?,
                record.model,
                record.optimizer,
                record.recipe,
                record.status,
                if record.task.is_empty() {
                    "unassigned"
                } else {
                    &record.task
                },
                record.commit,
                record.change,
                record.output,
                record.report_status,
                trace_summary(&record.measurements),
            );
            if let Some(entry) = diff_seq.and_then(|s| s.get(idx)) {
                print_diff(entry, args.diff);
            }
            println!("  {}", gpu_summary(record));
        }
        if let Some(plots) = comparisons.as_seq()
            && !plots.is_empty()
        {
            println!("comparison plots:");
            for p in plots {
                println!(
                    "  {} ({} runs): {}",
                    runs::field(p, "metric"),
                    p["runs_plotted"].as_u64().unwrap_or(0),
                    runs::field(p, "path")
                );
            }
            if let Some(html_path) = &dashboard {
                println!("  dashboard: {}", html_path.display());
            }
        }
    }
    if let Some(port) = args.serve {
        runs::serve(root, port)?;
    }
    Ok(())
}

fn execute_tune(root: &Path, args: Tune, engine: BuildSystem) -> Result<(), String> {
    if !args.standalone {
        if let Err(err) = crate::work::check_claim(root) {
            eprintln!("warning: continuing standalone without task lease: {err}");
        }
    }
    run(
        root,
        args.config.as_deref().ok_or("missing experiment config")?,
        args.rounds,
        engine,
    )
}

fn gpu_summary(record: &baux::artifact::runs::Record) -> String {
    use baux::artifact::gpu::Kind;
    match baux::artifact::runs::gpu_records(record) {
        Ok(evidence) => {
            let records = evidence.captures;
            let scope = if evidence.sealed {
                "set fixed at run completion"
            } else {
                "unsealed directory contents"
            };
            let native = records
                .iter()
                .filter(|(_, record)| {
                    matches!(record.recipe.input_kind, Kind::Cubin | Kind::NativeImage)
                        || matches!(
                            record.recipe.output_kind,
                            Some(Kind::Cubin | Kind::NativeImage)
                        )
                })
                .count();
            format!(
                "GPU capture: {} records, {native} with native payloads ({scope}); execution \
                 coverage and correctness not established",
                records.len()
            )
        }
        Err(error) => format!("GPU capture invalid: {error}"),
    }
}

fn trace_summary(measurements: &baux::artifact::json::Value) -> String {
    let Some(seq) = measurements.as_seq() else {
        return String::new();
    };
    let mut parts = Vec::new();
    for item in seq {
        let y = baux::artifact::runs::field(item, "y");
        let pts = item["points"].as_u64().unwrap_or(0);
        let first = item["first"]
            .as_seq()
            .and_then(|a| a.get(1))
            .and_then(|v| v.as_f64());
        let last = item["last"]
            .as_seq()
            .and_then(|a| a.get(1))
            .and_then(|v| v.as_f64());
        if let (Some(f), Some(l)) = (first, last) {
            parts.push(format!("{y}: {f:.6}->{l:.6} ({pts} pts)"));
        }
    }
    if parts.is_empty() {
        String::new()
    } else {
        format!("\n  traces {}", parts.join(" | "))
    }
}

fn print_diff(entry: &baux::artifact::json::Value, show_patch: bool) {
    let from = baux::artifact::runs::field(entry, "compared_from_commit");
    if let Some(files) = entry["optimizer_files_changed"].as_seq()
        && !files.is_empty()
    {
        let list: Vec<&str> = files.iter().filter_map(|v| v.as_str()).collect();
        println!(
            "  optimizer diff from {}: {}",
            from.get(..8).unwrap_or(from),
            list.join(", ")
        );
    }
    if show_patch {
        let patch = baux::artifact::runs::field(entry, "optimizer_patch");
        if !patch.is_empty() {
            for line in patch.lines() {
                println!("    {line}");
            }
        }
    }
}

pub(crate) fn run(
    root: &Path,
    path: &Path,
    rounds: Option<u32>,
    engine: BuildSystem,
) -> Result<(), String> {
    let text = std::fs::read_to_string(path).map_err(|e| e.to_string())?;
    let value: baux::artifact::json::Value =
        baux::artifact::toml::from_str(&text).map_err(|e| e.to_string())?;
    if value["experiment"].as_str() == Some("oxcaml-catalogue") {
        return run_ocaml(root, &text, rounds, engine, OcamlExperiment::Catalogue);
    }
    if value["experiment"].as_str() == Some("hardcaml-distance") {
        return run_ocaml(root, &text, rounds, engine, OcamlExperiment::Hardware);
    }
    baux_tooling::terminal::stderr(
        baux_tooling::terminal::Tag::Orchestration,
        "runtime · load experiment engine",
    );
    let mut args = vec!["tune".to_owned(), path.display().to_string()];
    if let Some(rounds) = rounds {
        args.extend(["--rounds".to_owned(), rounds.to_string()]);
    }
    engine.run(root, RUNTIME_TARGET, &args)
}

pub(crate) fn resolve(
    root: &Path,
    path: &Path,
    rounds: Option<u32>,
    engine: BuildSystem,
) -> Result<PathBuf, String> {
    let mut args = vec!["resolve".to_owned(), path.display().to_string()];
    if let Some(rounds) = rounds {
        args.extend(["--rounds".to_owned(), rounds.to_string()]);
    }
    let output = engine.run_output(root, RUNTIME_TARGET, &args)?;
    output
        .lines()
        .last()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .map(PathBuf::from)
        .ok_or("runtime helper did not return a resolved configuration".into())
}

#[cfg(test)]
mod tests {
    use clap::Parser;

    #[test]
    fn history_arguments() {
        for args in [
            vec!["baux", "tune", "config.toml"],
            vec!["baux", "tune", "--history"],
            vec!["baux", "tune", "--history", "--serve"],
            vec!["baux", "tune", "--history", "--serve", "8420"],
            vec![
                "baux",
                "tune",
                "--history",
                "--since",
                "yesterday",
                "--task",
                "current",
                "--json",
            ],
        ] {
            assert!(crate::Cli::try_parse_from(args).is_ok());
        }
        for args in [
            vec!["baux", "tune"],
            vec!["baux", "tune", "config.toml", "--history"],
            vec!["baux", "tune", "--history", "--rounds", "10"],
            vec!["baux", "tune", "config.toml", "--since", "today"],
            vec!["baux", "tune", "--history", "--limit", "0"],
        ] {
            assert!(
                crate::Cli::try_parse_from(args.clone()).is_err(),
                "accepted conflicting arguments: {args:?}"
            );
        }
    }
}

#[derive(Clone, Copy)]
enum OcamlExperiment {
    Catalogue,
    Hardware,
}

fn run_ocaml(
    root: &Path,
    text: &str,
    rounds: Option<u32>,
    engine: BuildSystem,
    kind: OcamlExperiment,
) -> Result<(), String> {
    use baux::artifact::{json, runs};
    use std::{fs, process::Command};
    let (binary, model, optimizer, recipe) = match kind {
        OcamlExperiment::Catalogue => (
            "benchmarks.exe",
            "sfu-catalogue",
            "oxcaml-catalogue",
            "core-bench-packed-scheduling",
        ),
        OcamlExperiment::Hardware => {
            if rounds.is_some() {
                return Err("Hardcaml runs enumerate the configured design grid; set samples in TOML instead of --rounds".into());
            }
            (
                "hardware_bench.exe",
                "squared-distance-circuit",
                "hardcaml-distance",
                "exhaustive-circuit-grid",
            )
        }
    };
    if matches!(kind, OcamlExperiment::Catalogue) {
        engine.run(root, "//ocaml:tool", &["deps".into()])?;
    }
    let build = match kind {
        OcamlExperiment::Catalogue => "build",
        OcamlExperiment::Hardware => "hardware-build",
    };
    engine.run(root, "//ocaml:tool", &[build.into()])?;
    let worker = root
        .join(".cache/baux/ocaml-build/default/bin")
        .join(binary);
    let commit = runs::jj(root, &["log", "-r", "@", "--no-graph", "-T", "commit_id"])?;
    let output = root.join(".cache/baux/runs/ocaml").join(
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_err(|e| e.to_string())?
            .as_nanos()
            .to_string(),
    );
    fs::create_dir_all(&output).map_err(|e| e.to_string())?;
    let requested = output.join("requested.toml");
    let resolved_path = output.join("config.toml");
    fs::write(&requested, text).map_err(|e| e.to_string())?;
    let mut resolve = Command::new(&worker);
    resolve.arg("resolve").arg(&requested).arg(&resolved_path);
    if let Some(rounds) = rounds {
        resolve.arg(rounds.to_string());
    }
    let status = resolve.status().map_err(|e| e.to_string())?;
    if !status.success() {
        return Err("OCaml rejected the experiment configuration".into());
    }
    let resolved = fs::read_to_string(&resolved_path).map_err(|e| e.to_string())?;
    let mut record = runs::start_worker(root, &output, &resolved, commit.trim(), &worker)?;
    record.model = model.into();
    record.optimizer = optimizer.into();
    record.recipe = recipe.into();
    record.protocol = runs::digest(resolved.as_bytes());
    let result = (|| {
        if matches!(kind, OcamlExperiment::Catalogue) {
            let sleef = root.join(".cache/baux/sleef-3.9.0/install/lib/libsleef.a");
            fs::write(output.join("sleef.sha256"), runs::file_digest(&sleef)?)
                .map_err(|e| e.to_string())?;
        }
        let switch = std::env::var("BAUX_OCAML_SWITCH").unwrap_or_else(|_| "5.2.0+ox".into());
        let compiler = Command::new("opam")
            .args(["exec", "--switch", &switch, "--", "ocamlopt", "-config"])
            .output()
            .map_err(|e| e.to_string())?;
        if !compiler.status.success() {
            return Err("Cannot record OxCaml configuration".into());
        }
        fs::write(output.join("compiler.txt"), compiler.stdout).map_err(|e| e.to_string())?;
        let packages = Command::new("opam")
            .args([
                "list",
                "--switch",
                &switch,
                "--installed",
                "--columns=name,version",
            ])
            .output()
            .map_err(|e| e.to_string())?;
        if !packages.status.success() {
            return Err("Cannot record installed toolchain packages".into());
        }
        fs::write(output.join("packages.txt"), packages.stdout).map_err(|e| e.to_string())?;
        if matches!(kind, OcamlExperiment::Catalogue) {
            fs::copy(
                root.join("ocaml/coverage.json"),
                output.join("coverage.json"),
            )
            .map_err(|e| e.to_string())?;
        }
        let hardware = Command::new("sysctl")
            .args(["-n", "machdep.cpu.brand_string"])
            .output()
            .map_err(|e| e.to_string())?;
        if !hardware.status.success() {
            return Err("Cannot record hardware configuration".into());
        }
        fs::write(output.join("hardware.txt"), hardware.stdout).map_err(|e| e.to_string())?;
        let status = Command::new(&worker)
            .current_dir(root)
            .arg("experiment")
            .arg(&resolved_path)
            .arg(&output)
            .status()
            .map_err(|e| e.to_string())?;
        if !status.success() {
            return Err(format!("OxCaml experiment failed: {status}"));
        }
        if matches!(kind, OcamlExperiment::Hardware) {
            fs::write(output.join("notes.json"), json::pretty_string(&json::json!({
                "scope": "Hardcaml cycle simulation and RTL generation; no FPGA execution or synthesis",
                "search": "exhaustive configured grid; single-design evaluator available for future Rust ask/tell integration",
                "objective": "maximize reciprocal cycles per vector subject to corpus error and signed-multiplier-output-bit budgets; infeasible reward is negative normalized constraint violation",
                "resources": "pre-synthesis graph statistics, not LUTs, DSPs, area, power or maximum clock frequency",
                "timing": "reset excluded; pipeline fill and drain included; unlimited input bandwidth; no host-transfer cost",
                "precision": "signed fixed point with bits-2 fractional bits; nearest rounding with halfway cases away from zero; full-width difference, square and accumulation"
            })).map_err(|e| e.to_string())?).map_err(|e| e.to_string())?;
            return Ok(());
        }
        let status = Command::new("python3")
            .current_dir(root)
            .arg(root.join("ocaml/audit.py"))
            .arg(&output)
            .status()
            .map_err(|e| e.to_string())?;
        if !status.success() {
            return Err("OxCaml generated-code audit failed".into());
        }
        fs::write(output.join("notes.json"), json::pretty_string(&json::json!({
            "timing": "monotonic samples collected inside the scheduler; geometric iteration counts; Core_bench regression with 100 bootstrap trials; raw samples and analysis saved",
            "allocation": "GC counters retain the runtime domain/global semantics; not aggregate worker allocation. Cycle counter is not collected; only wall-time regressions are interpreted",
            "comparison": "same formula definitions and inputs; float64; SLEEF 3.9.0 vector elementary functions",
            "phases": "calibration searches backend/workers/block size; independent confirmation uses changed inputs; packed_pipeline includes refill/run/fold; direct_pipeline generates points/evaluates/reduces",
            "hardware_caveat": "uncontrolled local machine load; measurements are exploratory",
            "scope": "local microbenchmarks; no Rust comparison or optimizer convergence claim"
        })).map_err(|e| e.to_string())?).map_err(|e| e.to_string())?;
        Ok(())
    })();
    runs::finish(root, &mut record, &result)?;
    result?;
    println!("{optimizer} evidence: {}", output.display());
    Ok(())
}
