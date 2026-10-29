use std::path::{Path, PathBuf};

use crate::cli_cfg::BuildSystem;

const RUNTIME_TARGET: &str = "//rust/crates/runner:baux-runner";

#[derive(clap::Args)]
pub struct Tune {
    #[arg(required_unless_present = "history", conflicts_with_all = ["history", "since", "until", "task", "optimizer", "limit", "reindex", "diff", "serve", "json"])]
    pub config: Option<PathBuf>,
    #[arg(long, conflicts_with_all = ["rounds", "history"])]
    pub prepare: bool,
    #[arg(long, conflicts_with_all = ["prepare", "history"])]
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
    if !args.prepare && !args.standalone {
        if let Err(err) = crate::work::check_claim(root) {
            eprintln!("warning: continuing standalone without task lease: {err}");
        }
    }
    run(
        root,
        args.config.as_deref().ok_or("missing experiment config")?,
        args.prepare,
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
    prepare: bool,
    rounds: Option<u32>,
    engine: BuildSystem,
) -> Result<(), String> {
    baux_tooling::terminal::stderr(
        baux_tooling::terminal::Tag::Orchestration,
        "runtime · load experiment engine",
    );
    let mut args = vec!["tune".to_owned(), path.display().to_string()];
    if prepare {
        args.push("--prepare".to_owned());
    }
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
