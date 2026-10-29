//! Explicit Beads mutations; never infer completion from a successful process.
use super::*;

fn record(
    beads: &store::Beads,
    id: &str,
    kind: &str,
    reference: &str,
    payload: &Value,
) -> Result<Value, String> {
    let payload = json::to_string(payload).map_err(|e| e.to_string())?;
    beads.write(&[
        "provenance",
        "record",
        "--issue",
        id,
        "--kind",
        kind,
        "--source",
        "baux",
        "--ref",
        reference,
        "--ref-kind",
        "work-id",
        "--payload",
        &payload,
    ])
}

fn evidence(
    beads: &store::Beads,
    binding: &Value,
    revision: &Value,
    args: &Feedback,
) -> Result<(), String> {
    let id = text(binding, "id")?;
    let commit = text(revision, "commit")?;
    let payload = json!({"workspace": text(binding, "workspace")?, "revision": revision,
        "verification": "not_run", "source": "agent_reported"});
    record(beads, id, "commit", &format!("jj:{commit}"), &payload)?;
    for reference in &args.evidence {
        // Keep the supplied reference intact in payload. Qualify its event key
        // by workspace and revision so another run cannot silently reuse it.
        let key = json::to_string(&json!([text(binding, "workspace")?, commit, reference]))
            .map_err(|e| e.to_string())?;
        record(
            beads,
            id,
            "used",
            &key,
            &json!({"reference": reference, "context": payload}),
        )?;
    }
    Ok(())
}

fn discover(
    beads: &store::Beads,
    binding: &Value,
    issue: &Value,
    args: &Feedback,
) -> Result<Value, String> {
    let Some(title) = args.discover.as_deref() else {
        return Ok(Value::null());
    };
    let outcome = args
        .outcome
        .as_deref()
        .ok_or("discovery requires --outcome")?;
    let id = text(binding, "id")?;
    let deps = if args.blocking {
        format!("discovered-from:{id},blocks:{id}")
    } else {
        format!("discovered-from:{id}")
    };
    let scope = if field(binding, "scope").is_empty() {
        coordination::parent(issue)
    } else {
        field(binding, "scope")
    };
    let mut command = vec![
        "create",
        "--title",
        title,
        "--description",
        outcome,
        "--acceptance",
        outcome,
        "--deps",
        &deps,
    ];
    if !scope.is_empty() {
        command.extend(["--parent", scope]);
    }
    beads.write(&command)
}

pub(super) fn validate(args: &Feedback) -> Result<(), String> {
    for value in args
        .evidence
        .iter()
        .chain(args.blocked_by.iter())
        .map(String::as_str)
        .chain(args.discover.as_deref())
        .chain(args.outcome.as_deref())
    {
        if value.trim().is_empty() {
            return Err("evidence, prerequisites and discovery fields must not be empty".into());
        }
    }
    if args.done && args.evidence.is_empty() {
        return Err(
            "completion requires --evidence; it is recorded as agent-reported, not independently \
             verified"
                .into(),
        );
    }
    Ok(())
}

pub(super) fn apply(
    beads: &store::Beads,
    binding: &Value,
    revision: &Value,
    issue: &Value,
    args: &Feedback,
) -> Result<Value, String> {
    validate(args)?;
    let Some(progress) = args.progress.as_deref() else {
        return Ok(json!({"lease": coordination::heartbeat(beads, issue)}));
    };
    validate_owner(issue, &beads.actor)?;
    let id = text(binding, "id")?;
    let note = checkpoint(
        binding,
        revision,
        progress,
        args.next.as_deref().ok_or("checkpoint requires --next")?,
        &args.evidence,
    )?;
    for prerequisite in &args.blocked_by {
        beads.issue(prerequisite)?;
    }
    // Native guarded update is the authority; the earlier read is only
    // feedback.
    beads.write(&[
        "update",
        id,
        "--if-assignee",
        &beads.actor,
        "--if-status",
        "in_progress",
        "--append-notes",
        &note,
    ])?;
    let mut actions = json!({"checkpoint": "recorded", "verification": "not_run", "discovered": null, "dependencies_added": []});
    mutate(beads, binding, revision, issue, args, &mut actions).map_err(|e| {
        format!(
            "checkpoint saved; some following actions may have completed: {}. Inspect the task \
             before retrying. Error: {e}",
            json::to_string(&actions).unwrap_or_default()
        )
    })?;
    Ok(actions)
}

fn mutate(
    beads: &store::Beads,
    binding: &Value,
    revision: &Value,
    issue: &Value,
    args: &Feedback,
    actions: &mut Value,
) -> Result<(), String> {
    let id = text(binding, "id")?;
    let mut added = Vec::new();
    for prerequisite in &args.blocked_by {
        validate_owner(&beads.issue(id)?, &beads.actor)?;
        beads.write(&["dep", "add", id, prerequisite, "--type", "blocks"])?;
        added.push(prerequisite.clone());
        actions["dependencies_added"] = json!(added);
    }
    if args.discover.is_some() {
        validate_owner(&beads.issue(id)?, &beads.actor)?;
    }
    actions["discovered"] = discover(beads, binding, issue, args)?;
    evidence(beads, binding, revision, args)?;
    actions["provenance"] = Value::from("recorded");
    if args.done {
        beads.write(&[
            "update",
            id,
            "--if-assignee",
            &beads.actor,
            "--if-status",
            "in_progress",
            "--status",
            "closed",
        ])?;
        actions["lifecycle"] = Value::from("closed");
    } else if args.release {
        beads.write(&[
            "unclaim",
            id,
            "--if-assignee",
            &beads.actor,
            "--reason",
            args.next.as_deref().unwrap_or_default(),
        ])?;
        actions["lifecycle"] = Value::from("released");
    } else {
        actions["lease"] = coordination::heartbeat(beads, &beads.issue(id)?);
    }
    Ok(())
}

pub(super) fn print(actions: &Value) {
    if !field(actions, "lifecycle").is_empty() {
        println!(
            "Task {}. Use context --claim to take ready work in the remembered scope.",
            field(actions, "lifecycle")
        );
    }
    if !actions["discovered"].is_null() && actions.get("discovered").is_some() {
        println!(
            "Discovered task: {} — {}",
            field(&actions["discovered"], "id"),
            field(&actions["discovered"], "title")
        );
    }
    if let Some(error) = actions["lease"].get("error") {
        println!(
            "Lease refresh failed: {}",
            error.as_str().unwrap_or("unknown")
        );
    }
}
