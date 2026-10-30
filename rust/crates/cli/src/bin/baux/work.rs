//! Workspace-local task selection backed by the shared Beads store.

pub(crate) use baux::artifact::json::{self, Value, json};
use clap::Args;
use std::path::{Path, PathBuf};

use crate::work_store as store;

mod coordination;
mod handoff;

pub(crate) fn active_task(root: &Path) -> Result<Option<String>, String> {
    crate::work_store::load(root)?
        .map(|binding| text(&binding, "id").map(str::to_owned))
        .transpose()
}

/// History can inspect an old binding; starting new work requires its live
/// claim.
pub(crate) fn check_claim(root: &Path) -> Result<(), String> {
    let Some(binding) = crate::work_store::load(root)? else {
        return Ok(());
    };
    let beads = crate::work_store::Beads::bound(&binding)?;
    let id = text(&binding, "id")?;
    validate_owner(&beads.issue(id)?, &beads.actor)?;
    beads.write(&["heartbeat", id])?;
    Ok(())
}

fn run_evidence(root: &Path, binding: &Value, revision: &Value) -> Value {
    match baux::artifact::runs::brief(root, field(binding, "id"), field(revision, "commit")) {
        Ok(value) => value,
        Err(error) => json!({"runs": [], "error": error, "status": "unavailable"}),
    }
}

fn print_evidence(evidence: &Value) {
    println!("\nExperiment evidence (up to 6 task runs; no checks executed):");
    if let Some(error) = evidence.get("error") {
        println!(
            "  Unavailable: {}",
            error.as_str().unwrap_or("unknown error")
        );
    }
    if let Some(rows) = evidence["runs"].as_seq() {
        if rows.is_empty() {
            println!("  No recorded runs for this task.");
        }
        for row in rows {
            let run = &row["run"];
            println!(
                "  {} / {} [{}] — {}\n    source {}\n    {}",
                field(run, "model"),
                field(run, "optimizer"),
                field(run, "status"),
                run["started"]
                    .as_i64()
                    .and_then(|t| baux::artifact::runs::local_time(t).ok())
                    .unwrap_or_default(),
                field(run, "commit"),
                field(run, "output")
            );
            if let Some(measurements) = run["measurements"].as_seq() {
                for metric in measurements.iter().take(4) {
                    println!(
                        "    {}: {} → {} ({})",
                        field(metric, "y"),
                        json::to_string(&metric["first"]).unwrap_or_default(),
                        json::to_string(&metric["last"]).unwrap_or_default(),
                        field(metric, "source")
                    );
                }
            }
            if !field(run, "error").is_empty() {
                println!("    Failure: {}", field(run, "error"));
            }
            let diff = field(row, "source_diff");
            if !diff.is_empty() {
                println!("    Changes since run:\n{diff}");
            }
            if row["source_diff_truncated"].as_bool() == Some(true) {
                println!(
                    "    Showing the first 30 changed files; use the source revision for the full \
                     diff."
                );
            }
            if !field(row, "source_diff_error").is_empty() {
                println!(
                    "    Source comparison unavailable: {}",
                    field(row, "source_diff_error")
                );
            }
        }
    }
    let baseline = &evidence["baseline_candidate"];
    if !baseline.is_null() {
        println!(
            "  Muon baseline candidate: {}\n    {}\n    {}",
            field(&baseline["run"], "output"),
            field(baseline, "selection_reason"),
            field(baseline, "limitation")
        );
    }
    println!("  Recorded outcomes do not establish task completion or causal improvement.");
}

#[derive(Args)]
#[command(group(clap::ArgGroup::new("selection").args(["query", "task", "new", "claim"])))]
#[command(group(clap::ArgGroup::new("queue").args(["claim", "scope"]).multiple(true)))]
pub struct Context {
    /// Search existing task titles (does not claim a search result).
    pub query: Option<String>,
    /// Claim and remember this existing task in the current workspace.
    #[arg(long)]
    pub task: Option<String>,
    /// Atomically claim ready work within --scope (or the remembered scope).
    #[arg(long)]
    pub claim: bool,
    /// Beads epic/task whose descendants define this worker's work queue.
    #[arg(long, conflicts_with_all = ["query", "new", "parent"])]
    pub scope: Option<String>,
    /// Require these labels when listing or claiming ready work (repeatable).
    #[arg(long, requires = "queue", conflicts_with_all = ["query", "new", "task"])]
    pub label: Vec<String>,
    /// Create and remember a task with this title.
    #[arg(long, requires = "outcome")]
    pub new: Option<String>,
    /// Describe the intended result and completion criteria for a new task.
    #[arg(long, requires = "new")]
    pub outcome: Option<String>,
    /// Parent Beads epic or task for newly created work.
    #[arg(long, requires = "new")]
    pub parent: Option<String>,
    /// Repository containing the shared .beads store (remembered on selection).
    #[arg(long)]
    pub store: Option<PathBuf>,
    /// Beads actor for a claim; defaults to BEADS_ACTOR or workspace identity.
    #[arg(long)]
    pub actor: Option<String>,
    /// Emit machine-readable task context.
    #[arg(long)]
    pub json: bool,
}

#[derive(Args)]
pub struct Feedback {
    /// Append an agent-reported progress checkpoint; does not run verification.
    #[arg(long, requires = "next")]
    pub progress: Option<String>,
    /// Next action or remaining uncertainty for this checkpoint.
    #[arg(long, requires = "progress")]
    pub next: Option<String>,
    /// Evidence references to record, such as a report path or exact check
    /// command.
    #[arg(long, requires = "progress")]
    pub evidence: Vec<String>,
    /// Close the task with Beads' blocker/child checks; completion is
    /// agent-reported.
    #[arg(long, requires = "progress", conflicts_with_all = ["release", "discover", "blocked_by"])]
    pub done: bool,
    /// Release this claim after recording the handoff, preserving its
    /// dependencies.
    #[arg(long, requires = "progress")]
    pub release: bool,
    /// Record newly discovered work without changing this workspace's
    /// assignment.
    #[arg(long, requires_all = ["progress", "outcome"])]
    pub discover: Option<String>,
    /// Completion criteria for the discovered task.
    #[arg(long, requires = "discover")]
    pub outcome: Option<String>,
    /// The discovered work blocks the active task.
    #[arg(long, requires = "discover")]
    pub blocking: bool,
    /// Add an existing prerequisite to the active task (repeatable).
    #[arg(long, requires = "progress")]
    pub blocked_by: Vec<String>,
    /// Emit machine-readable progress and revision information.
    #[arg(long)]
    pub json: bool,
}

pub fn context(root: &Path, args: Context) -> Result<(), String> {
    let root = root.canonicalize().map_err(|error| error.to_string())?;
    let saved = crate::work_store::load(&root)?;
    let database = crate::work_store::discover(&root, args.store.as_deref(), saved.as_ref())?;
    let actor = args
        .actor
        .clone()
        .or_else(|| std::env::var("BEADS_ACTOR").ok())
        .or_else(|| saved.as_ref().map(|v| field(v, "actor").to_owned()))
        .unwrap_or_else(|| format!("baux:{}", root.display()));
    let beads = crate::work_store::Beads {
        root: database,
        actor,
    };
    if let Some(query) = args.query {
        let rows = beads.read(&["search", &query, "--limit", "20"])?;
        return print_results(
            "Search results (select with context --task ID)",
            rows,
            args.json,
        );
    }
    if args.scope.is_some() && !args.claim && args.task.is_none() {
        let scope = args.scope.as_deref().unwrap_or_default();
        return coordination::print_scope(&beads, scope, &args.label, args.json);
    }
    if args.task.is_some() || args.new.is_some() || args.claim {
        let _lock = crate::work_store::Lock::acquire(&root)?;
        let current = crate::work_store::load(&root)?;
        coordination::check_switch(&beads, current.as_ref(), args.task.as_deref())?;
        let Some(binding) = select(&root, &beads, &args, current.as_ref())? else {
            return print_results(
                "No ready work in this scope; nothing claimed",
                json!([]),
                args.json,
            );
        };
        crate::work_store::save(&root, &binding)?;
        return print_context(&beads, &binding, args.json);
    }
    if let Some(binding) = saved {
        // A persisted binding always identifies its own store and actor.
        if beads.root != Path::new(text(&binding, "store")?) {
            return Err(
                "active task belongs to another Beads store; select a task explicitly".into(),
            );
        }
        print_context(
            &crate::work_store::Beads::bound(&binding)?,
            &binding,
            args.json,
        )
    } else {
        print_results(
            "No active task. Ready work (or use context --new TITLE --outcome RESULT)",
            beads.read(&["ready", "--limit", "20"])?,
            args.json,
        )
    }
}

fn select(
    root: &Path,
    beads: &crate::work_store::Beads,
    args: &Context,
    saved: Option<&Value>,
) -> Result<Option<Value>, String> {
    let revision = crate::work_store::revision(root)?;
    let mut scope = args.scope.clone().or_else(|| {
        saved
            .filter(|b| args.claim && Path::new(field(b, "store")) == beads.root)
            .and_then(|b| b.get("scope"))
            .and_then(|v| v.as_str())
            .filter(|value| !value.is_empty())
            .map(str::to_owned)
    });
    if args.claim
        && scope.is_none()
        && let Some(previous) = saved.filter(|b| Path::new(field(b, "store")) == beads.root)
    {
        let issue = beads.issue(text(previous, "id")?)?;
        let parent = coordination::parent(&issue);
        if !parent.is_empty() {
            scope = Some(parent.to_owned());
        }
    }
    let labels = if args.label.is_empty() {
        coordination::labels(
            saved.filter(|b| args.claim && Path::new(field(b, "store")) == beads.root),
        )
    } else {
        args.label.clone()
    };
    let resume = match (args.task.as_deref(), saved) {
        (Some(id), Some(previous))
            if field(previous, "id") == id
                && Path::new(field(previous, "store")) == beads.root
                && field(previous, "actor") == beads.actor =>
        {
            validate_owner(&beads.issue(id)?, &beads.actor).is_ok()
        }
        _ => false,
    };
    let Some(id) = acquire_task(beads, args, scope.as_deref(), &labels)? else {
        return Ok(None);
    };
    if let Some(previous) = saved
        && resume
        && field(previous, "id") == id
        && Path::new(field(previous, "store")) == beads.root
    {
        let mut resumed = previous.clone();
        resumed["actor"] = Value::from(beads.actor.clone());
        return Ok(Some(resumed));
    }
    let issue = beads.issue(&id)?;
    let scope = scope.unwrap_or_else(|| coordination::parent(&issue).to_owned());
    Ok(Some(
        json!({"version": 1, "id": id, "store": beads.root.to_string_lossy(),
        "workspace": root.to_string_lossy(), "actor": beads.actor, "scope": scope, "labels": labels,
        "base": text(&revision, "commit")?, "change": text(&revision, "change")?}),
    ))
}

fn acquire_task(
    beads: &crate::work_store::Beads,
    args: &Context,
    scope: Option<&str>,
    labels: &[String],
) -> Result<Option<String>, String> {
    if args.claim {
        let scope =
            scope.ok_or("first claim requires --scope EPIC; subsequent claims remember it")?;
        let rows = coordination::ready(beads, scope, labels, true)?;
        let rows = rows
            .as_seq()
            .ok_or("Beads ready --claim returned a non-list")?;
        if rows.is_empty() {
            return Ok(None);
        }
        if rows.len() != 1 {
            return Err("Beads claimed an unexpected number of tasks".into());
        }
        Ok(Some(text(&rows[0], "id")?.to_owned()))
    } else if let Some(id) = &args.task {
        let issue = beads.issue(id)?;
        if field(&issue, "status") == "closed" {
            return Err(format!("task {id} is closed; select open work"));
        }
        beads.write(&["update", id, "--claim"])?;
        Ok(Some(id.clone()))
    } else {
        let title = args.new.as_deref().ok_or("missing task title")?;
        let outcome = args.outcome.as_deref().ok_or("missing task outcome")?;
        if title.trim().is_empty() || outcome.trim().is_empty() {
            return Err("task title and outcome must not be empty".into());
        }
        let mut command = vec![
            "create",
            "--title",
            title,
            "--description",
            outcome,
            "--acceptance",
            outcome,
        ];
        if let Some(parent) = &args.parent {
            command.extend(["--parent", parent]);
        }
        let issue = beads.write(&command)?;
        let id = text(&issue, "id")?;
        beads
            .write(&["update", id, "--claim"])
            .map_err(|e| format!("created {id}, but claiming failed: {e}"))?;
        Ok(Some(id.to_owned()))
    }
}

pub fn feedback(root: &Path, args: Feedback) -> Result<(), String> {
    let root = root.canonicalize().map_err(|error| error.to_string())?;
    let _lock = crate::work_store::Lock::acquire(&root)?;
    let binding =
        crate::work_store::load(&root)?.ok_or("no active task; use ./baux context first")?;
    let beads = crate::work_store::Beads::bound(&binding)?;
    let mut issue = beads.issue(text(&binding, "id")?)?;
    let revision = crate::work_store::revision(&root)?;
    let diff = crate::work_store::diff(&root, text(&binding, "base")?, text(&revision, "commit")?)?;
    let actions = handoff::apply(&beads, &binding, &revision, &issue, &args)?;
    issue = beads.issue(text(&binding, "id")?)?;
    let coordination = coordination::brief(&beads, &binding, &issue);
    let evidence = run_evidence(&root, &binding, &revision);
    let report = json!({"task": issue, "binding": binding, "revision": revision, "evidence": evidence,
        "changed": diff, "coordination": coordination, "actions": actions,
        "verification": "not_run", "checkpoint_source": "agent_reported"});
    if args.json {
        return print_json(&report);
    }
    print_issue(&report["task"]);
    println!("\nRevision: {}", text(&report["revision"], "commit")?);
    println!(
        "Changes since task start:\n{}",
        if diff.is_empty() { "  none" } else { &diff }
    );
    println!("Verification: not run by this command. Checkpoints are agent-reported.");
    coordination::print(&report["coordination"]);
    handoff::print(&report["actions"]);
    print_evidence(&evidence);
    Ok(())
}

fn validate_owner(issue: &Value, actor: &str) -> Result<(), String> {
    if field(issue, "assignee") != actor || field(issue, "status") != "in_progress" {
        return Err(
            "task ownership/status changed; resume it with context --task ID before updating"
                .into(),
        );
    }
    Ok(())
}

fn checkpoint(
    binding: &Value,
    revision: &Value,
    progress: &str,
    next: &str,
    evidence: &[String],
) -> Result<String, String> {
    if progress.trim().is_empty() || next.trim().is_empty() {
        return Err("progress and next action must not be empty".into());
    }
    Ok(format!(
        "\nCheckpoint (agent-reported; verification not run)\nWorkspace: {}\nRevision: \
         {}\nProgress: {}\nNext: {}\nEvidence references: {}",
        text(binding, "workspace")?,
        text(revision, "commit")?,
        progress,
        next,
        if evidence.is_empty() {
            "none".to_owned()
        } else {
            evidence.join("; ")
        }
    ))
}

fn print_context(
    beads: &crate::work_store::Beads,
    binding: &Value,
    machine: bool,
) -> Result<(), String> {
    let issue = beads.issue(text(binding, "id")?)?;
    let lease = coordination::heartbeat(beads, &issue);
    let coordination = coordination::brief(beads, binding, &issue);
    let root = Path::new(text(binding, "workspace")?);
    let revision = crate::work_store::revision(root)?;
    let evidence = run_evidence(root, binding, &revision);
    if machine {
        return print_json(
            &json!({"task": issue, "binding": binding, "revision": revision, "evidence": evidence, "coordination": coordination, "lease": lease}),
        );
    }
    print_issue(&issue);
    println!("\nWorkspace: {}", text(binding, "workspace")?);
    println!("Base revision: {}", text(binding, "base")?);
    println!("Use ./baux feedback to see progress; no task ID needed.");
    if let Some(error) = lease.get("error") {
        println!(
            "Lease refresh failed: {}",
            error.as_str().unwrap_or("unknown")
        );
    }
    coordination::print(&coordination);
    print_evidence(&evidence);
    Ok(())
}

fn print_results(label: &str, rows: Value, machine: bool) -> Result<(), String> {
    if machine {
        return print_json(&rows);
    }
    println!("{label}");
    let items = rows.as_seq().ok_or("Beads returned a non-list result")?;
    for row in items {
        println!(
            "{}  [{}] {}",
            field(row, "id"),
            field(row, "status"),
            field(row, "title")
        );
    }
    if items.is_empty() {
        println!("  No matching tasks.");
    }
    Ok(())
}

fn print_issue(issue: &Value) {
    println!(
        "{}  [{}] {}",
        field(issue, "id"),
        field(issue, "status"),
        field(issue, "title")
    );
    for (key, label) in [
        ("assignee", "Owner"),
        ("description", "Objective"),
        ("acceptance_criteria", "Completion criteria"),
        ("notes", "Progress"),
    ] {
        let value = field(issue, key);
        if !value.is_empty() {
            let display = if key == "notes" {
                value.rsplit("\nCheckpoint ").next().unwrap_or(value)
            } else {
                value
            };
            println!("\n{label}:\n{display}");
        }
    }
}

fn print_json(value: &Value) -> Result<(), String> {
    println!(
        "{}",
        json::pretty_string(value).map_err(|error| error.to_string())?
    );
    Ok(())
}

fn field<'a>(value: &'a Value, key: &str) -> &'a str {
    value.get(key).and_then(|item| item.as_str()).unwrap_or("")
}

pub(crate) fn text<'a>(value: &'a Value, key: &str) -> Result<&'a str, String> {
    let result = field(value, key);
    if result.is_empty() {
        Err(format!("missing or invalid task field: {key}"))
    } else {
        Ok(result)
    }
}

#[cfg(test)]
mod tests;
