//! Opt-in contract test against the installed Beads binary, never the real
//! store.
use super::*;

fn create(beads: &store::Beads, title: &str, extra: &[&str]) -> String {
    let mut args = vec![
        "create",
        "--title",
        title,
        "--description",
        "Disposable native integration test",
    ];
    args.extend_from_slice(extra);
    field(&beads.write(&args).unwrap(), "id").to_owned()
}

fn apply(beads: &store::Beads, id: &str, scope: &str, args: &[&str]) -> Result<Value, String> {
    let binding = json!({"id": id, "actor": beads.actor, "workspace": beads.root.to_string_lossy(), "scope": scope});
    handoff::apply(
        beads,
        &binding,
        &json!({"commit": "native-test-revision", "change": "test"}),
        &beads.issue(id)?,
        &feedback_args(args),
    )
}

#[test]
#[ignore = "requires BAUX_BEADS_TEST_STORE pointing at a marked disposable Beads store"]
fn swarm_lifecycle() {
    let root =
        PathBuf::from(std::env::var("BAUX_BEADS_TEST_STORE").expect("explicit disposable store"));
    assert!(
        root.canonicalize()
            .unwrap()
            .starts_with(std::env::temp_dir().canonicalize().unwrap())
    );
    assert_eq!(
        fs::read_to_string(root.join(".baux-disposable-store"))
            .unwrap()
            .trim(),
        "beads-workflow-tests"
    );
    let beads = store::Beads {
        root: root.clone(),
        actor: "native-test".into(),
    };
    let epic = create(&beads, "Native swarm lifecycle", &["--type", "epic"]);
    let first = create(&beads, "Prerequisite", &["--parent", &epic]);
    let blocked = create(
        &beads,
        "Dependent",
        &["--parent", &epic, "--deps", &format!("blocked-by:{first}")],
    );
    let other = create(&beads, "Independent", &["--parent", &epic]);
    let workers: Vec<_> = ["native-a", "native-b"]
        .into_iter()
        .map(|actor| {
            let root = root.clone();
            let epic = epic.clone();
            std::thread::spawn(move || {
                let beads = store::Beads {
                    root,
                    actor: actor.into(),
                };
                let result = coordination::ready(&beads, &epic, &[], true).unwrap();
                (beads, field(&result[0], "id").to_owned())
            })
        })
        .collect();
    let mut claims: Vec<_> = workers.into_iter().map(|w| w.join().unwrap()).collect();
    assert_ne!(claims[0].1, claims[1].1);
    assert!(claims.iter().all(|(_, id)| id == &first || id == &other));
    let first_index = claims.iter().position(|(_, id)| id == &first).unwrap();
    let (first_worker, _) = claims.remove(first_index);
    let (other_worker, _) = claims.remove(0);
    let binding = json!({"id": first, "store": root.to_string_lossy(), "actor": first_worker.actor, "scope": epic});
    assert!(coordination::check_switch(&first_worker, Some(&binding), Some(&blocked)).is_err());
    let done = [
        "--done",
        "--progress",
        "Verified test result",
        "--next",
        "dependent work",
        "--evidence",
        "native-test-report",
    ];
    apply(&first_worker, &first, &epic, &done).unwrap();
    let workspace = Directory::new();
    let _lock = store::Lock::acquire(&workspace.0).unwrap();
    let mut saved = super::binding(&workspace.0);
    saved["id"] = Value::from(first.clone());
    saved["store"] = Value::from(root.to_string_lossy().into_owned());
    saved["actor"] = Value::from(first_worker.actor.clone());
    store::save(&workspace.0, &saved).unwrap();
    assert!(check_claim(&workspace.0).is_err());
    let ready = coordination::ready(&beads, &epic, &[], false).unwrap();
    assert!(
        ready
            .as_seq()
            .unwrap()
            .iter()
            .any(|r| field(r, "id") == blocked)
    );
    assert!(apply(&first_worker, &first, &epic, &done).is_err());
    let found = apply(
        &other_worker,
        &other,
        &epic,
        &[
            "--discover",
            "Missing prerequisite",
            "--outcome",
            "Supply missing result",
            "--blocking",
            "--progress",
            "Found prerequisite",
            "--next",
            "resolve it",
        ],
    )
    .unwrap();
    let discovered = field(&found["discovered"], "id");
    assert!(!discovered.is_empty());
    assert!(apply(&other_worker, &other, &epic, &done).is_err());
    assert_eq!(
        field(&beads.issue(&other).unwrap(), "status"),
        "in_progress"
    );
    let checkpoint = [
        "--progress",
        "Waiting on prerequisite",
        "--next",
        "handoff",
        "--evidence",
        "same-report",
    ];
    apply(&other_worker, &other, &epic, &checkpoint).unwrap();
    let before = beads
        .read(&["provenance", "log", &other])
        .unwrap()
        .as_seq()
        .unwrap()
        .len();
    apply(&other_worker, &other, &epic, &checkpoint).unwrap();
    assert_eq!(
        beads
            .read(&["provenance", "log", &other])
            .unwrap()
            .as_seq()
            .unwrap()
            .len(),
        before
    );
    let brief = coordination::brief(
        &other_worker,
        &json!({"scope": epic}),
        &beads.issue(&other).unwrap(),
    );
    assert!(
        brief["prerequisites"]["items"]
            .as_seq()
            .unwrap()
            .iter()
            .any(|r| field(r, "id") == discovered && field(r, "dependency_type") == "blocks")
    );
    assert!(brief["swarm"].get("total_issues").is_some());
    apply(
        &other_worker,
        &other,
        &epic,
        &[
            "--release",
            "--progress",
            "Blocked",
            "--next",
            "Resolve prerequisite",
        ],
    )
    .unwrap();
    assert!(apply(&other_worker, &other, &epic, &checkpoint).is_err());
    let ready = coordination::ready(&beads, &epic, &[], false).unwrap();
    assert!(
        ready
            .as_seq()
            .unwrap()
            .iter()
            .any(|r| field(r, "id") == discovered)
    );
    assert!(
        !ready
            .as_seq()
            .unwrap()
            .iter()
            .any(|r| field(r, "id") == other)
    );
    beads.write(&["update", discovered, "--claim"]).unwrap();
    apply(&beads, discovered, &epic, &done).unwrap();
    assert!(
        coordination::ready(&beads, &epic, &[], false)
            .unwrap()
            .as_seq()
            .unwrap()
            .iter()
            .any(|r| field(r, "id") == other)
    );
    println!(
        "PASS: concurrent distinct claims, dependency readiness, guarded closure, linked \
         discovery, release, provenance replay, scoped context ({epic})"
    );
}
