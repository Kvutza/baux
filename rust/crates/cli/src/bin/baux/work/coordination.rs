//! Beads owns readiness, claims and dependencies. This module curates its
//! output.
use super::*;

const LIMIT: usize = 12;

pub(super) fn parent(issue: &Value) -> &str {
    if !field(issue, "parent").is_empty() {
        return field(issue, "parent");
    }
    issue
        .get("dependencies")
        .and_then(|v| v.as_seq())
        .and_then(|rows| {
            rows.iter()
                .find(|r| field(r, "dependency_type") == "parent-child")
        })
        .map(|r| field(r, "id"))
        .unwrap_or("")
}

pub(super) fn labels(binding: Option<&Value>) -> Vec<String> {
    binding
        .and_then(|b| b.get("labels"))
        .and_then(|v| v.as_seq())
        .map(|rows| {
            rows.iter()
                .filter_map(|v| v.as_str().map(str::to_owned))
                .collect()
        })
        .unwrap_or_default()
}

pub(super) fn check_switch(
    beads: &store::Beads,
    saved: Option<&Value>,
    target: Option<&str>,
) -> Result<(), String> {
    let Some(saved) = saved else {
        return Ok(());
    };
    let previous = store::Beads::bound(saved)?;
    let id = text(saved, "id")?;
    if beads.root == previous.root && target == Some(id) && beads.actor == previous.actor {
        return Ok(());
    }
    let issue = previous.issue(id)?;
    if field(&issue, "status") == "in_progress" && field(&issue, "assignee") == previous.actor {
        return Err(format!(
            "{id} is still assigned to this workspace; use feedback --release or --done with a \
             progress checkpoint before selecting more work"
        ));
    }
    Ok(())
}

pub(super) fn ready(
    beads: &store::Beads,
    scope: &str,
    labels: &[String],
    claim: bool,
) -> Result<Value, String> {
    if scope.is_empty() {
        return Err("ready work requires a scope".into());
    }
    beads.issue(scope)?;
    let mut args = vec![
        "ready",
        "--parent",
        scope,
        "--limit",
        "12",
        "--exclude-type",
        "epic",
        "--exclude-type",
        "gate",
    ];
    for label in labels {
        args.extend(["--label", label]);
    }
    if claim {
        args.push("--claim");
        beads.write(&args)
    } else {
        beads.read(&args)
    }
}

pub(super) fn heartbeat(beads: &store::Beads, issue: &Value) -> Value {
    if validate_owner(issue, &beads.actor).is_err() {
        return json!({"status": "not_owned"});
    }
    observed(beads.write(&["heartbeat", field(issue, "id")]))
}

fn observed(result: Result<Value, String>) -> Value {
    result.unwrap_or_else(|error| json!({"status": "unavailable", "error": error}))
}

fn clip(value: &str, limit: usize) -> String {
    let mut chars = value.chars();
    let mut result: String = chars.by_ref().take(limit).collect();
    if chars.next().is_some() {
        result.push_str("… [truncated]");
    }
    result
}

fn summary(row: &Value) -> Value {
    let notes = field(row, "notes");
    let latest = notes.rsplit("\nCheckpoint ").next().unwrap_or(notes);
    json!({"id": field(row, "id"), "title": clip(field(row, "title"), 200),
        "status": field(row, "status"), "assignee": field(row, "assignee"),
        "dependency_type": field(row, "dependency_type"),
        "outcome": clip(field(row, "acceptance_criteria"), 1200),
        "latest_finding": clip(latest, 2000), "blocked_by": row.get("blocked_by").cloned().unwrap_or(Value::null())})
}

fn bounded(rows: Value) -> Value {
    let Some(items) = rows.as_seq() else {
        return rows;
    };
    let kept: Vec<Value> = items.iter().take(LIMIT).map(summary).collect();
    json!({"items": kept, "total": items.len(), "truncated": items.len() > LIMIT})
}

fn available(rows: Value) -> Value {
    let mut result = bounded(rows);
    let total = result["total"].as_u64().unwrap_or(0);
    result["total"] = Value::null();
    result["limit"] = json!(LIMIT);
    result["may_have_more"] = json!(total >= LIMIT as u64);
    result
}

fn provenance(beads: &store::Beads, id: &str) -> Result<Value, String> {
    let events = beads.read(&["provenance", "log", id])?;
    let events = events
        .as_seq()
        .ok_or("Beads provenance returned a non-list")?;
    let items: Vec<Value> = events
        .iter()
        .rev()
        .take(LIMIT)
        .map(|event| {
            let mut event = event.clone();
            if let Some(payload) = event.get("payload").and_then(|v| v.as_str()) {
                event["payload"] = Value::from(clip(payload, 2000));
            }
            event
        })
        .collect();
    Ok(json!({"items": items, "total": events.len(), "truncated": events.len() > LIMIT}))
}

fn dependencies(beads: &store::Beads, issue: &Value) -> Result<Value, String> {
    let rows = issue
        .get("dependencies")
        .and_then(|v| v.as_seq())
        .map(|r| r.as_slice())
        .unwrap_or(&[]);
    // Bound the number of full prerequisite records fetched. Parents are kept
    // separately from blockers by preserving Beads' dependency_type field.
    let mut ordered: Vec<&Value> = rows.iter().collect();
    ordered.sort_by_key(|r| (field(r, "dependency_type") != "blocks", field(r, "id")));
    let selected: Vec<&Value> = ordered.into_iter().take(LIMIT).collect();
    let ids: Vec<&str> = selected.iter().map(|r| field(r, "id")).collect();
    let full = beads.issues(&ids)?;
    let mut details = Vec::new();
    for row in selected {
        let mut detail = full
            .iter()
            .find(|r| field(r, "id") == field(row, "id"))
            .ok_or("Beads omitted a prerequisite")?
            .clone();
        detail["dependency_type"] = Value::from(field(row, "dependency_type"));
        details.push(summary(&detail));
    }
    Ok(json!({"items": details, "total": rows.len(), "truncated": rows.len() > LIMIT}))
}

fn swarm(beads: &store::Beads, scope: &str) -> Result<Value, String> {
    let issue = beads.issue(scope)?;
    if field(&issue, "issue_type") != "epic" {
        return Ok(json!({"status": "not_an_epic", "scope": scope}));
    }
    let mut status = beads.read(&["swarm", "status", scope])?;
    for key in ["active", "blocked", "ready", "completed"] {
        if let Some(rows) = status.get(key).cloned() {
            status[key] = bounded(rows);
        }
    }
    // Counts are Beads' task states, not proof of research progress.
    status["meaning"] = Value::from("task states; closure is not independent verification");
    Ok(status)
}

pub(super) fn brief(beads: &store::Beads, binding: &Value, issue: &Value) -> Value {
    let scope = if field(binding, "scope").is_empty() {
        parent(issue)
    } else {
        field(binding, "scope")
    };
    let labels = labels(Some(binding));
    let available = if scope.is_empty() {
        json!({"status": "no_scope"})
    } else {
        observed(ready(beads, scope, &labels, false).map(available))
    };
    let status = if scope.is_empty() {
        json!({"status": "no_scope"})
    } else {
        observed(swarm(beads, scope))
    };
    json!({"scope": scope, "labels": labels,
        "prerequisites": observed(dependencies(beads, issue)),
        "dependents": observed(beads.read(&["dep", "list", field(issue, "id"), "--direction", "up"]).map(bounded)),
        "ready": available, "swarm": status,
        "provenance": observed(provenance(beads, field(issue, "id"))),
        "ownership": if validate_owner(issue, &beads.actor).is_ok() { "owned" } else { "not_owned" },
        "lease_policy": "context/feedback refresh owned claims; worker hosts must heartbeat during long work; no automatic reclaim"})
}

pub(super) fn print_scope(
    beads: &store::Beads,
    scope: &str,
    labels: &[String],
    machine: bool,
) -> Result<(), String> {
    let ready = ready(beads, scope, labels, false)?;
    let report =
        json!({"scope": scope, "ready": available(ready), "swarm": observed(swarm(beads, scope))});
    if machine {
        print_json(&report)
    } else {
        print(&report);
        Ok(())
    }
}

pub(super) fn print(report: &Value) {
    println!("\nBeads coordination — scope {}", field(report, "scope"));
    if field(report, "ownership") == "not_owned" {
        println!("  This binding is not an active claim. Select ready work with context --claim.");
    }
    for (key, label) in [
        ("prerequisites", "Prerequisites and findings"),
        ("dependents", "Dependent work"),
        ("ready", "Ready work"),
    ] {
        let section = &report[key];
        if let Some(error) = section.get("error") {
            println!(
                "  {label}: unavailable: {}",
                error.as_str().unwrap_or("unknown")
            );
        }
        if let Some(rows) = section.get("items").and_then(|v| v.as_seq()) {
            println!("  {label}:");
            for row in rows.iter().take(LIMIT) {
                println!(
                    "    {} [{}; {}] {}",
                    field(row, "id"),
                    field(row, "status"),
                    field(row, "dependency_type"),
                    clip(field(row, "title"), 200)
                );
                if !field(row, "latest_finding").is_empty() {
                    println!("      {}", field(row, "latest_finding"));
                }
            }
            if rows.is_empty() {
                println!("    none");
            }
            if section["truncated"].as_bool() == Some(true) || rows.len() == LIMIT {
                println!("    Bounded view; inspect Beads for additional work.");
            }
        }
    }
    if let Some(error) = report["provenance"].get("error") {
        println!(
            "  Provenance unavailable: {}",
            error.as_str().unwrap_or("unknown")
        );
    }
    if let Some(rows) = report["provenance"]["items"].as_seq() {
        println!("  Recent provenance (at most {LIMIT}):");
        for row in rows {
            println!(
                "    {} {} — {}",
                field(row, "kind"),
                clip(field(row, "ref"), 300),
                field(row, "actor")
            );
        }
    }
    let swarm = &report["swarm"];
    if let Some(error) = swarm.get("error") {
        println!(
            "  Swarm status unavailable: {}",
            error.as_str().unwrap_or("unknown")
        );
    }
    if swarm.get("total_issues").is_some() {
        println!(
            "  Swarm task counts: active {}, ready {}, blocked {} (not verified outcomes)",
            swarm["active_count"].as_u64().unwrap_or(0),
            swarm["ready_count"].as_u64().unwrap_or(0),
            swarm["blocked_count"].as_u64().unwrap_or(0)
        );
        if let Some(rows) = swarm["active"]["items"].as_seq() {
            for row in rows {
                println!(
                    "    {} — {} — {}",
                    field(row, "id"),
                    field(row, "assignee"),
                    field(row, "title")
                );
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bounded_findings() {
        let rows: Vec<Value> = (0..20)
            .map(|i| {
                json!({"id": format!("task-{i}"),
            "notes": format!("old findings\nCheckpoint {}", "λ".repeat(2500))})
            })
            .collect();
        let brief = bounded(json!(rows));
        assert_eq!(brief["total"].as_u64(), Some(20));
        assert_eq!(brief["items"].as_seq().unwrap().len(), LIMIT);
        assert_eq!(brief["truncated"].as_bool(), Some(true));
        let finding = field(&brief["items"][0], "latest_finding");
        assert!(!finding.contains("old findings"));
        assert!(finding.ends_with("[truncated]"));
        assert_eq!(clip("λβ", 2), "λβ");
    }

    #[test]
    fn visible_errors() {
        let value = observed(Err("database unavailable".into()));
        assert_eq!(field(&value, "status"), "unavailable");
        assert!(value.get("items").is_none());
        let empty = available(json!([]));
        assert_eq!(empty["may_have_more"].as_bool(), Some(false));
        assert!(empty["total"].is_null());
    }

    #[test]
    fn parent_link() {
        let issue = json!({"dependencies": [
            {"id": "prerequisite", "dependency_type": "blocks"},
            {"id": "epic", "dependency_type": "parent-child"}
        ]});
        assert_eq!(parent(&issue), "epic");
    }
}
