use baux::artifact::json::Value;
use baux::artifact::wire_json;

type Mutator = fn(&mut Value);

fn drop_source(m: &mut Value) {
    if let Some(sub) = m.get_mut("reachable_sources")
        && let Some(map) = sub.as_map_mut()
    {
        map.remove("rust/crates/ennbo/src/acquisition.rs");
    }
}

fn drop_func(m: &mut Value) {
    if let Some(sub) = m.get_mut("reachable_functions")
        && let Some(seq) = sub.as_seq_mut()
    {
        seq.retain(|f| {
            let name = match f.get("name") {
                Some(v) => v.as_str().unwrap_or(""),
                None => "",
            };
            !name.contains("ask")
        });
    }
}

fn inject_false(m: &mut Value) {
    if let Some(sub) = m.get_mut("exclusions")
        && let Some(seq) = sub.as_seq_mut()
    {
        seq.push(wire_json!({
            "category": "falsified_exclusion",
            "exclusion_proof": "Adversarial false exclusion injection",
            "scope": "rust/crates/ennbo/src/candidates.rs",
            "type": "source_file",
        }));
    }
}

fn tamper_hash(m: &mut Value) {
    if let Some(srcs) = m.get_mut("reachable_sources")
        && let Some(entry) = srcs.get_mut("rust/crates/ennbo/src/optimizer_factory.rs")
        && let Some(map) = entry.as_map_mut()
    {
        map.insert("sha256", Value::from("0".repeat(64)));
    }
}

fn drop_dispatch(m: &mut Value) {
    if let Some(map) = m.as_map_mut() {
        map.insert("dynamic_dispatch", Value::from(Vec::<Value>::new()));
    }
}

fn drop_scope(m: &mut Value) {
    if let Some(sub) = m.get_mut("exclusions")
        && let Some(seq) = sub.as_seq_mut()
    {
        seq.retain(|e| {
            let scope = match e.get("scope") {
                Some(v) => v.as_str().unwrap_or(""),
                None => "",
            };
            scope != "tests/*"
        });
    }
}

fn forge_revision(m: &mut Value) {
    if let Some(id) = m.get_mut("identity")
        && let Some(map) = id.as_map_mut()
    {
        map.insert("revision", Value::from("0".repeat(40)));
    }
}

fn forge_dispatch(m: &mut Value) {
    if let Some(dd) = m.get_mut("dynamic_dispatch")
        && let Some(seq) = dd.as_seq_mut()
        && let Some(first) = seq.first_mut()
        && let Some(map) = first.as_map_mut()
    {
        map.insert("target_symbol", Value::from("fabricated"));
    }
}

fn forge_location(m: &mut Value) {
    if let Some(rf) = m.get_mut("reachable_functions")
        && let Some(seq) = rf.as_seq_mut()
        && let Some(first) = seq.first_mut()
        && let Some(map) = first.as_map_mut()
    {
        map.insert("line", Value::from(999_999_u64));
    }
}

fn forge_summary(m: &mut Value) {
    if let Some(sum) = m.get_mut("summary")
        && let Some(map) = sum.as_map_mut()
    {
        map.insert("total_reachable_functions", Value::from(999_999_u64));
    }
}

fn duplicate_func(m: &mut Value) {
    if let Some(rf) = m.get_mut("reachable_functions")
        && let Some(seq) = rf.as_seq_mut()
        && let Some(first) = seq.first().cloned()
    {
        seq.push(first);
    }
}

fn drop_dep(m: &mut Value) {
    if let Some(db) = m.get_mut("dependency_boundary")
        && let Some(pkgs) = db.get_mut("rust_packages")
        && let Some(seq) = pkgs.as_seq_mut()
    {
        seq.pop();
    }
}

fn forge_dep(m: &mut Value) {
    if let Some(db) = m.get_mut("dependency_boundary")
        && let Some(pkgs) = db.get_mut("rust_packages")
        && let Some(seq) = pkgs.as_seq_mut()
        && let Some(first) = seq.first_mut()
        && let Some(map) = first.as_map_mut()
    {
        map.insert("checksum", Value::from("0".repeat(64)));
    }
}

fn drop_ref(m: &mut Value) {
    if let Some(db) = m.get_mut("dependency_boundary")
        && let Some(refs) = db.get_mut("rust_references")
        && let Some(seq) = refs.as_seq_mut()
    {
        seq.pop();
    }
}

fn claim_verified(m: &mut Value) {
    if let Some(db) = m.get_mut("dependency_boundary")
        && let Some(map) = db.as_map_mut()
    {
        map.insert("proof_status", Value::from("proved"));
    }
}

fn forge_version(m: &mut Value) {
    if let Some(map) = m.as_map_mut() {
        map.insert("manifest_version", Value::from(2_u64));
    }
}

fn forge_registry(m: &mut Value) {
    if let Some(rf) = m.get_mut("reachable_functions")
        && let Some(seq) = rf.as_seq_mut()
        && let Some(first) = seq.first_mut()
        && let Some(map) = first.as_map_mut()
    {
        map.insert("name", Value::from("forged_unregistered_function"));
    }
}

const CASES: &[(&str, Mutator, &str)] = &[
    (
        "omitted_reachable_source",
        drop_source,
        "missing from reachable_sources",
    ),
    (
        "omitted_reachable_function",
        drop_func,
        "missing reachable functions",
    ),
    (
        "false_exclusion_of_reachable_file",
        inject_false,
        "exclusion violation",
    ),
    ("tampered_source_hash", tamper_hash, "sha256 mismatch"),
    (
        "dropped_dynamic_dispatch",
        drop_dispatch,
        "dynamic dispatch",
    ),
    ("unclassified_repo_file", drop_scope, "unclassified"),
    ("forged_revision", forge_revision, "identity"),
    ("forged_dispatch_target", forge_dispatch, "dynamic dispatch"),
    (
        "forged_function_location",
        forge_location,
        "function records",
    ),
    ("forged_summary_count", forge_summary, "summary counts"),
    ("duplicated_function", duplicate_func, "function records"),
    ("omitted_locked_dependency", drop_dep, "dependency boundary"),
    (
        "forged_dependency_checksum",
        forge_dep,
        "dependency boundary",
    ),
    (
        "omitted_external_reference",
        drop_ref,
        "dependency boundary",
    ),
    (
        "unsupported_proof_claim",
        claim_verified,
        "dependency boundary",
    ),
    (
        "unsupported_manifest_schema",
        forge_version,
        "manifest version",
    ),
    (
        "unregistered_lean_proof_function",
        forge_registry,
        "lean proof registry",
    ),
];

pub(super) fn run_adversary(
    base_manifest: &Value,
    checker: impl Fn(&Value) -> Vec<String>,
) -> (bool, Vec<String>) {
    let mut results = Vec::with_capacity(CASES.len());
    let mut all_passed = true;
    for &(name, mutator, expected_sub) in CASES {
        let mut mutated = base_manifest.clone();
        mutator(&mut mutated);
        let errors = checker(&mutated);
        if errors.is_empty() {
            all_passed = false;
            results.push(format!(
                "Adversarial evasion succeeded for {name}: checker failed to detect omission"
            ));
            continue;
        }
        let joined = errors.join("\n");
        if !joined
            .to_ascii_lowercase()
            .contains(&expected_sub.to_ascii_lowercase())
        {
            all_passed = false;
            results.push(format!(
                "Adversary {name} caught with unexpected error:\nExpected substring: \
                 {expected_sub}\nActual errors: {joined}"
            ));
        } else {
            results.push(format!("Passed: {name} detected ({})", errors[0]));
        }
    }
    (all_passed, results)
}
