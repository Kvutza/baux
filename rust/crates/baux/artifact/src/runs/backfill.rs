use super::*;
use std::collections::BTreeMap;

/// Backfill pre-manifest runs from `.cache/baux/runs/pretrain`,
/// `performance/runs`, `results/cuda`, and tracked `performance/*.json`
/// summaries into the shared manifest store.
pub(super) fn discover(root: &Path, evidence_dir: &Path) -> Result<usize, String> {
    let manifests_dir = evidence_dir.join("manifests");
    fs::create_dir_all(&manifests_dir).map_err(|e| e.to_string())?;
    let mut change_cache = BTreeMap::new();
    let mut added = 0;

    let pretrain_root = root.join(".cache/baux/runs/pretrain");
    if pretrain_root.is_dir() {
        for group in read_subdirs(&pretrain_root)? {
            for run_dir in read_subdirs(&group)? {
                if backfill_dir(root, &manifests_dir, &run_dir, &mut change_cache)? {
                    added += 1;
                }
            }
        }
    }

    let perf_runs = root.join("performance/runs");
    if perf_runs.is_dir() {
        for run_dir in read_subdirs(&perf_runs)? {
            if backfill_dir(root, &manifests_dir, &run_dir, &mut change_cache)? {
                added += 1;
            }
        }
    }

    let cuda_runs = root.join("results/cuda");
    if cuda_runs.is_dir() {
        for run_dir in read_subdirs(&cuda_runs)? {
            if run_dir
                .file_name()
                .and_then(|s| s.to_str())
                .is_some_and(|name| name.contains("hnet"))
                && backfill_dir(root, &manifests_dir, &run_dir, &mut change_cache)?
            {
                added += 1;
            }
        }
    }

    added += backfill_summaries(root, &manifests_dir, &mut change_cache)?;
    Ok(added)
}

fn read_subdirs(dir: &Path) -> Result<Vec<PathBuf>, String> {
    let mut dirs = Vec::new();
    for entry in fs::read_dir(dir).map_err(|e| e.to_string())? {
        let path = entry.map_err(|e| e.to_string())?.path();
        if path.is_dir() {
            dirs.push(path);
        }
    }
    dirs.sort();
    Ok(dirs)
}

fn render_plot(
    canonical: &Path,
    manifests_dir: &Path,
    id: &str,
    is_cache: bool,
) -> (String, Option<String>, Value) {
    let plot_target = if is_cache {
        canonical.to_path_buf()
    } else {
        manifests_dir
            .parent()
            .unwrap_or(manifests_dir)
            .join("plots")
            .join(id)
    };
    match plot::render_to(canonical, &plot_target) {
        Ok((count, measurements)) => (
            if count == 0 {
                "no_measurements"
            } else {
                "complete"
            }
            .to_owned(),
            None,
            measurements,
        ),
        Err(err) => ("failed".to_owned(), Some(err), Value::null()),
    }
}

fn read_meta(canonical: &Path) -> (String, Value, Value, Value, Value, Value) {
    let config_text = ["experiment.toml", "config.toml", "schema-snapshot.toml"]
        .iter()
        .find_map(|name| fs::read_to_string(canonical.join(name)).ok())
        .unwrap_or_default();
    let config_val: Value = if config_text.is_empty() {
        Value::null()
    } else {
        baux::artifact::toml::from_str(&config_text).unwrap_or_else(|_| Value::null())
    };
    let result_toml: Value = fs::read_to_string(canonical.join("result.toml"))
        .ok()
        .and_then(|t| baux::artifact::toml::from_str(&t).ok())
        .unwrap_or_else(Value::null);
    let manifest_json = read_json(&canonical.join("manifest.json"));
    let result_json = read_json(&canonical.join("result.json"));
    let validation_json = read_json(&canonical.join("validation.json"));
    (
        config_text,
        config_val,
        result_toml,
        manifest_json,
        result_json,
        validation_json,
    )
}

fn backfill_dir(
    root: &Path,
    manifests_dir: &Path,
    run_dir: &Path,
    change_cache: &mut BTreeMap<String, String>,
) -> Result<bool, String> {
    let canonical = run_dir
        .canonicalize()
        .unwrap_or_else(|_| run_dir.to_path_buf());
    let id = digest(canonical.to_string_lossy().as_bytes());
    if manifests_dir.join(format!("{id}.cbor")).exists()
        || manifests_dir.join(format!("{id}.json")).exists()
    {
        return Ok(false);
    }
    if ![
        "experiment.toml",
        "config.toml",
        "result.toml",
        "result.json",
        "ucb-result.json",
    ]
    .iter()
    .any(|f| canonical.join(f).is_file())
    {
        return Ok(false);
    }

    let (config_text, config_val, result_toml, manifest_json, result_json, validation_json) =
        read_meta(&canonical);

    let (commit, fallback_change) = extract_revision(&canonical, &manifest_json, &result_json);
    if commit.is_empty() {
        return Ok(false);
    }
    let change = resolve_change(root, &commit, &fallback_change, change_cache);
    let started = extract_started(&canonical, &manifest_json);

    let model = first_nonempty(&[
        field(&config_val, "model"),
        field(&manifest_json, "model"),
        field(&result_json, "execution_architecture"),
        "fbt-pisa1-hnet-mhc4-v1",
    ]);
    let optimizer = first_nonempty(&[
        field(&config_val["optimizer"], "algorithm"),
        field(&config_val["run"], "optimizer"),
        field(&manifest_json, "acquisition"),
        "turbo-enn",
    ]);
    let ctx_tokens = context_tokens(&config_val, &manifest_json, &result_json, &validation_json);
    let base_recipe = first_nonempty(&[
        field(&config_val["data"], "recipe"),
        field(&config_val, "corpus"),
        "corpus",
    ]);
    let recipe = if ctx_tokens > 0 {
        format!("{base_recipe}@{ctx_tokens}")
    } else {
        base_recipe
    };
    let status = if field(&result_toml, "status") == "failure" {
        "failed"
    } else {
        "completed"
    };

    let is_cache = canonical.to_string_lossy().contains("/.cache/");
    let (report_status, report_error, measurements) =
        render_plot(&canonical, manifests_dir, &id, is_cache);

    let record = Record {
        schema: 1,
        id: id.clone(),
        started,
        task: String::new(),
        task_store: root.to_string_lossy().into_owned(),
        workspace: root.to_string_lossy().into_owned(),
        output: canonical.to_string_lossy().into_owned(),
        commit,
        change,
        runner_sha256: "historical".into(),
        worker_sha256: "historical".into(),
        config_sha256: digest(config_text.as_bytes()),
        data_sha256: String::new(),
        model,
        optimizer,
        recipe: recipe.clone(),
        protocol: digest(format!("{}:{}", field(&config_val, "model"), ctx_tokens).as_bytes()),
        status: status.into(),
        ended: Some(started),
        report_status,
        report_error,
        error: None,
        measurements,
    };
    let bytes = codec::encode(&record)?;
    if is_cache {
        let _ = atomic(&canonical.join("run.cbor"), &bytes);
    }
    atomic(&manifests_dir.join(format!("{id}.cbor")), &bytes)?;
    Ok(true)
}

fn backfill_summaries(
    root: &Path,
    manifests_dir: &Path,
    change_cache: &mut BTreeMap<String, String>,
) -> Result<usize, String> {
    let mut added = 0;
    let state_path = root.join("performance/bo-muon-state.json");
    if state_path.is_file() {
        let v = read_json(&state_path);
        let commit = field(&v, "revision").to_owned();
        let change = resolve_change(root, &commit, "", change_cache);
        let init_nll = plot::number(&v["initial_mean_nll"]).unwrap_or(12.485831);
        for (key, opt_name, offset_ms) in [
            ("bayesian_optimization", "turbo-enn", 0_i64),
            ("muon", "muon", 1000_i64),
        ] {
            let sub = &v[key];
            if sub.is_null() {
                continue;
            }
            let rounds = plot::number(&sub["rounds"]).unwrap_or(1.0);
            let fin_nll = plot::number(&sub["final_mean_nll"]).unwrap_or(init_nll);
            let ms = plot::number(&sub["mean_round_ms"]).unwrap_or(0.0);
            let id = digest(format!("performance/bo-muon-state.json:{key}").as_bytes());
            if !manifests_dir.join(format!("{id}.cbor")).exists() {
                let series = vec![
                    plot::series_summary(
                        "summary",
                        "round",
                        "nll",
                        &[(0.0, init_nll), (rounds, fin_nll)],
                    ),
                    plot::series_summary(
                        "summary",
                        "round",
                        "incumbent_nll",
                        &[(0.0, init_nll), (rounds, fin_nll)],
                    ),
                    plot::series_summary(
                        "summary",
                        "round",
                        "round_wall_ms",
                        &[(1.0, ms), (rounds, ms)],
                    ),
                ];
                let record = summary_record(
                    root,
                    &id,
                    1791597958472 + offset_ms,
                    &state_path,
                    &commit,
                    &change,
                    "fbt-pisa1-hnet-mhc4-v1",
                    opt_name,
                    "stack-v3-rust-go-v1@4096",
                    json!(series),
                );
                atomic(
                    &manifests_dir.join(format!("{id}.cbor")),
                    &codec::encode(&record)?,
                )?;
                added += 1;
            }
        }
    }

    for (rel_path, started_ms, ctx, nll_init_key, nll_fin_key) in [
        (
            "performance/hnet-64k-bo-20261010.json",
            1791629340002_i64,
            65536_u64,
            "initial_validation_nll",
            "final_validation_nll",
        ),
        (
            "performance/hnet-1m-learn.json",
            1791660939589_i64,
            1048576_u64,
            "initial_nll",
            "final_nll",
        ),
    ] {
        let path = root.join(rel_path);
        if !path.is_file() {
            continue;
        }
        let id = digest(rel_path.as_bytes());
        if manifests_dir.join(format!("{id}.cbor")).exists() {
            continue;
        }
        let v = read_json(&path);
        let commit = first_nonempty(&[
            field(&v, "source_revision"),
            field(&v, "revision"),
            "a1667f04b9418395df60a6b5c71393ca3ac9ab9b",
        ]);
        let change = resolve_change(root, &commit, "", change_cache);
        let rounds = plot::number(&v["experiment"]["rounds"]).unwrap_or(128.0);
        let init_nll = plot::number(&v["learning"][nll_init_key]).unwrap_or(12.484);
        let fin_nll = plot::number(&v["learning"][nll_fin_key]).unwrap_or(init_nll);
        let ms = plot::number(&v["systems"]["mean_round_ms"]).unwrap_or(0.0);
        let series = vec![
            plot::series_summary(
                "summary",
                "round",
                "nll",
                &[(0.0, init_nll), (rounds, fin_nll)],
            ),
            plot::series_summary(
                "summary",
                "round",
                "incumbent_nll",
                &[(0.0, init_nll), (rounds, fin_nll)],
            ),
            plot::series_summary(
                "summary",
                "round",
                "round_wall_ms",
                &[(1.0, ms), (rounds, ms)],
            ),
        ];
        let record = summary_record(
            root,
            &id,
            started_ms,
            &path,
            &commit,
            &change,
            "fbt-pisa1-hnet-mhc4-v1",
            "turbo-enn",
            &format!("stack-v3-rust-go-v1@{ctx}"),
            json!(series),
        );
        atomic(
            &manifests_dir.join(format!("{id}.cbor")),
            &codec::encode(&record)?,
        )?;
        added += 1;
    }
    Ok(added)
}

#[allow(clippy::too_many_arguments)]
fn summary_record(
    root: &Path,
    id: &str,
    started: i64,
    path: &Path,
    commit: &str,
    change: &str,
    model: &str,
    optimizer: &str,
    recipe: &str,
    measurements: Value,
) -> Record {
    Record {
        schema: 1,
        id: id.into(),
        started,
        task: String::new(),
        task_store: root.to_string_lossy().into_owned(),
        workspace: root.to_string_lossy().into_owned(),
        output: path.to_string_lossy().into_owned(),
        commit: commit.into(),
        change: change.into(),
        runner_sha256: "historical".into(),
        worker_sha256: "historical".into(),
        config_sha256: digest(recipe.as_bytes()),
        data_sha256: digest(recipe.as_bytes()),
        model: model.into(),
        optimizer: optimizer.into(),
        recipe: recipe.into(),
        protocol: digest(format!("{model}:{recipe}").as_bytes()),
        status: "completed".into(),
        ended: Some(started),
        report_status: "complete".into(),
        report_error: None,
        error: None,
        measurements,
    }
}

fn read_json(path: &Path) -> Value {
    fs::read(path)
        .ok()
        .and_then(|b| json::from_slice(&b).ok())
        .unwrap_or_else(Value::null)
}

fn first_nonempty(candidates: &[&str]) -> String {
    candidates
        .iter()
        .copied()
        .find(|s| !s.is_empty())
        .unwrap_or("")
        .to_owned()
}

fn extract_revision(dir: &Path, manifest: &Value, result: &Value) -> (String, String) {
    if let Ok(rev) = fs::read_to_string(dir.join("source-revision.txt")) {
        let trimmed = rev.trim();
        if !trimmed.is_empty() {
            return (trimmed.to_owned(), String::new());
        }
    }
    if let Ok(rev) = fs::read_to_string(dir.join("revision.txt")) {
        let parts: Vec<&str> = rev.split_whitespace().collect();
        if parts.len() >= 2 {
            return (parts[1].to_owned(), parts[0].to_owned());
        } else if parts.len() == 1 {
            return (parts[0].to_owned(), String::new());
        }
    }
    let mut commit = first_nonempty(&[
        field(manifest, "commit"),
        field(manifest, "base_revision"),
        field(manifest, "revision"),
        field(result, "commit"),
    ]);
    if commit.is_empty() {
        let path_str = dir.to_string_lossy();
        if path_str.contains("20261012-") {
            commit = "f15c1ec3a1806eceabd437c273920554dfcce39a".into();
        } else if path_str.contains("results/cuda/") {
            commit = "a1667f04b9418395df60a6b5c71393ca3ac9ab9b".into();
        }
    }
    let change = field(manifest, "change_id").to_owned();
    (commit, change)
}

fn resolve_change(
    root: &Path,
    commit: &str,
    fallback: &str,
    cache: &mut BTreeMap<String, String>,
) -> String {
    if let Some(cached) = cache.get(commit) {
        return cached.clone();
    }
    let resolved = jj(
        root,
        &["log", "-r", commit, "--no-graph", "-T", "change_id"],
    )
    .ok()
    .map(|s| s.trim().to_owned())
    .filter(|s| !s.is_empty())
    .unwrap_or_else(|| {
        if !fallback.is_empty() {
            fallback.to_owned()
        } else {
            commit.get(..12).unwrap_or(commit).to_owned()
        }
    });
    cache.insert(commit.to_owned(), resolved.clone());
    resolved
}

fn extract_started(dir: &Path, manifest: &Value) -> i64 {
    if let Some(name) = dir.file_name().and_then(|s| s.to_str()) {
        if let Some(rest) = name.strip_prefix("run-")
            && let Some(ts_str) = rest.split('-').next()
            && let Ok(ms) = ts_str.parse::<i64>()
        {
            return ms;
        }
        if name.len() >= 8
            && name.as_bytes()[..8].iter().all(u8::is_ascii_digit)
            && let Ok(ms) =
                catalog::since(&format!("{}-{}-{}", &name[..4], &name[4..6], &name[6..8]))
        {
            return ms;
        }
    }
    let started_at = field(manifest, "started_at");
    if !started_at.is_empty()
        && let Ok(ms) = catalog::since(started_at)
    {
        return ms;
    }
    fs::metadata(dir)
        .and_then(|m| m.modified())
        .ok()
        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
        .map(|d| d.as_millis() as i64)
        .unwrap_or(1791500000000)
}

fn context_tokens(config: &Value, manifest: &Value, result: &Value, validation: &Value) -> u64 {
    for candidate in [
        &manifest["context_tokens"],
        &result["context_window_tokens"],
        &result["context_tokens"],
        &config["data"]["context_tokens"],
        &config["evaluation"]["context_tokens"],
    ] {
        if let Some(v) = candidate.as_u64() {
            return v;
        }
    }
    if let Some(seq) = validation.get("measurements").and_then(|v| v.as_seq())
        && let Some(first) = seq.first()
        && let Some(v) = first["scored_tokens"]
            .as_u64()
            .or_else(|| first["heldout_causal_targets"].as_u64())
    {
        return v;
    }
    let prompt = config["generation"]["corpus-prompt-tokens"]
        .as_u64()
        .or_else(|| config["generation"]["corpus_prompt_tokens"].as_u64())
        .unwrap_or(0);
    let max_tok = config["generation"]["max-tokens"]
        .as_u64()
        .or_else(|| config["generation"]["max_tokens"].as_u64())
        .unwrap_or(0);
    prompt + max_tok
}
