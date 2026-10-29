use super::*;
use clap::Parser;
use std::fs;
use std::sync::atomic::{AtomicU64, Ordering};

mod native;

fn feedback_args(args: &[&str]) -> Feedback {
    let mut command = vec!["baux", "feedback"];
    command.extend_from_slice(args);
    match crate::Cli::try_parse_from(command).unwrap().action {
        crate::Action::Feedback(args) => args,
        _ => unreachable!(),
    }
}

#[test]
fn swarm_arguments() {
    for args in [
        vec!["context", "--claim", "--scope", "epic", "--label", "math"],
        vec!["context", "--scope", "epic"],
        vec![
            "feedback",
            "--done",
            "--progress",
            "done",
            "--next",
            "integration",
            "--evidence",
            "proof.lean",
        ],
        vec![
            "feedback",
            "--discover",
            "missing lemma",
            "--outcome",
            "proof",
            "--blocking",
            "--release",
            "--progress",
            "blocked",
            "--next",
            "prove lemma",
        ],
    ] {
        assert!(crate::Cli::try_parse_from(std::iter::once("baux").chain(args)).is_ok());
    }
    for args in [
        vec!["context", "--label", "math"],
        vec!["context", "--claim", "--task", "x"],
        vec!["feedback", "--done"],
        vec!["feedback", "--blocking"],
        vec![
            "feedback",
            "--discover",
            "missing",
            "--progress",
            "found",
            "--next",
            "fix",
        ],
        vec![
            "feedback",
            "--done",
            "--release",
            "--progress",
            "done",
            "--next",
            "next",
        ],
    ] {
        assert!(crate::Cli::try_parse_from(std::iter::once("baux").chain(args)).is_err());
    }
}

#[test]
fn completion_evidence() {
    assert!(
        handoff::validate(&feedback_args(&[
            "--done",
            "--progress",
            "done",
            "--next",
            "next"
        ]))
        .is_err()
    );
    assert!(
        handoff::validate(&feedback_args(&[
            "--progress",
            "done",
            "--next",
            "next",
            "--evidence",
            " "
        ]))
        .is_err()
    );
    assert!(
        handoff::validate(&feedback_args(&[
            "--done",
            "--progress",
            "done",
            "--next",
            "next",
            "--evidence",
            "report"
        ]))
        .is_ok()
    );
}

static SERIAL: AtomicU64 = AtomicU64::new(0);

struct Directory(PathBuf);

impl Directory {
    fn new() -> Self {
        let path = std::env::temp_dir().join(format!(
            "baux-tasks-{}-{}",
            std::process::id(),
            SERIAL.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir_all(&path).expect("create test directory");
        Self(path.canonicalize().expect("canonical test directory"))
    }
}

impl Drop for Directory {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn binding(root: &Path) -> Value {
    json!({"version": 1, "id": "test-task", "workspace": root.to_string_lossy(),
        "store": root.to_string_lossy(), "actor": "worker", "base": "base", "change": "change"})
}

#[test]
fn shared_store() {
    let directory = Directory::new();
    let owner = directory.0.join("owner");
    let workspace = directory.0.join("workspace");
    fs::create_dir_all(owner.join(".jj/repo")).unwrap();
    fs::create_dir_all(owner.join(".beads")).unwrap();
    fs::write(owner.join(".beads/metadata.json"), "{}").unwrap();
    fs::create_dir_all(workspace.join(".jj")).unwrap();
    fs::write(workspace.join(".jj/repo"), "../../owner/.jj/repo\n").unwrap();
    // Copied metadata in an added JJ workspace must not shadow the shared
    // store.
    fs::create_dir_all(workspace.join(".beads")).unwrap();
    fs::write(workspace.join(".beads/metadata.json"), "{}").unwrap();
    assert_eq!(store::discover(&workspace, None, None).unwrap(), owner);
}

#[test]
fn isolated_binding() {
    let first = Directory::new();
    let second = Directory::new();
    let _lock = store::Lock::acquire(&first.0).unwrap();
    store::save(&first.0, &binding(&first.0)).unwrap();
    assert_eq!(
        field(&store::load(&first.0).unwrap().unwrap(), "id"),
        "test-task"
    );
    assert!(store::load(&second.0).unwrap().is_none());
    let _other = store::Lock::acquire(&second.0).unwrap();
    fs::copy(
        first.0.join(".cache/baux/task.json"),
        second.0.join(".cache/baux/task.json"),
    )
    .unwrap();
    assert!(
        store::load(&second.0)
            .unwrap_err()
            .contains("different workspace")
    );
}

#[test]
fn corrupt_state() {
    let directory = Directory::new();
    let _lock = store::Lock::acquire(&directory.0).unwrap();
    fs::write(directory.0.join(".cache/baux/task.json"), "{broken").unwrap();
    assert!(store::load(&directory.0).is_err());
    assert!(store::discover(&directory.0, None, None).is_err());
    assert!(!directory.0.join(".beads").exists());
}

#[test]
fn exclusive_update() {
    let directory = Directory::new();
    let lock = store::Lock::acquire(&directory.0).unwrap();
    assert!(store::Lock::acquire(&directory.0).is_err());
    drop(lock);
    assert!(store::Lock::acquire(&directory.0).is_ok());
}

#[test]
fn changed_ownership() {
    let mut issue = json!({"assignee": "worker", "status": "in_progress"});
    assert!(validate_owner(&issue, "worker").is_ok());
    assert!(validate_owner(&issue, "another-agent").is_err());
    issue["status"] = Value::from("closed");
    assert!(validate_owner(&issue, "worker").is_err());
}

#[test]
fn reported_evidence() {
    let directory = Directory::new();
    let revision = json!({"commit": "exact-revision"});
    let report = checkpoint(
        &binding(&directory.0),
        &revision,
        "Found the cause",
        "Run the reproduction",
        &["report.json".into()],
    )
    .unwrap();
    assert!(report.contains("agent-reported; verification not run"));
    assert!(report.contains("exact-revision"));
    assert!(report.contains("report.json"));
    assert!(checkpoint(&binding(&directory.0), &revision, " ", "next", &[]).is_err());
}

#[test]
fn failed_command() {
    let result = store::output(
        std::process::Command::new("sh").args(["-c", "printf 'discovery failed' >&2; exit 7"]),
    );
    assert!(result.unwrap_err().contains("discovery failed"));
}

#[test]
fn cli_contract() {
    assert!(crate::Cli::try_parse_from(["baux", "context"]).is_ok());
    assert!(crate::Cli::try_parse_from(["baux", "feedback"]).is_ok());
    assert!(crate::Cli::try_parse_from(["baux", "context", "--new", "task"]).is_err());
    assert!(crate::Cli::try_parse_from(["baux", "context", "search", "--task", "id"]).is_err());
    assert!(crate::Cli::try_parse_from(["baux", "feedback", "--progress", "done"]).is_err());
    assert!(
        crate::Cli::try_parse_from(["baux", "context", "--new", "task", "--outcome", "result"])
            .is_ok()
    );
}
