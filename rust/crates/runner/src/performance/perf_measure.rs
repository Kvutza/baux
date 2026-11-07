use super::perf_stats::{distribution, experiment_regression, pool_rank, vec_path};
use baux::artifact::json::{Value, json};
use std::collections::BTreeMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

pub(super) fn measured(
    root: &Path,
    run: &Path,
    current_digest: &str,
    target_ms: f64,
) -> Result<Value, String> {
    let run = root.join(run);
    let source = std::fs::read_to_string(run.join("performance-source.sha256"))
        .ok()
        .map(|text| text.trim().to_owned());
    if run.join("optimization.jsonl").is_file() {
        let first = BufReader::new(
            File::open(run.join("optimization.jsonl")).map_err(|error| error.to_string())?,
        )
        .lines()
        .next()
        .transpose()
        .map_err(|error| error.to_string())?
        .ok_or("optimization trace is empty")?;
        let first: Value =
            baux::artifact::json::from_str(&first).map_err(|error| error.to_string())?;
        if first["objective"] == "fixed-corpus-causal-likelihood" {
            return causal_measure(root, &run, source, current_digest, target_ms);
        }
    }
    if run.join("run.log").is_file()
        && !run.join("result.cbor").is_file()
        && !run.join("result.json").is_file()
    {
        return causal_log(root, &run, source, current_digest, target_ms);
    }
    let result: Value = if run.join("result.cbor").is_file() {
        let bytes = std::fs::read(run.join("result.cbor")).map_err(|error| error.to_string())?;
        baux::artifact::cbor::from_slice(&bytes).map_err(|error| error.to_string())?
    } else {
        baux::artifact::json::from_reader(
            File::open(run.join("result.json")).map_err(|error| error.to_string())?,
        )
        .map_err(|error| error.to_string())?
    };
    if run.join("optimization.jsonl").is_file() {
        return metal_measure(root, &run, result, source, current_digest, target_ms);
    }
    cuda_measure(root, &run, result, source, current_digest, target_ms)
}

fn causal_log(
    root: &Path,
    run: &Path,
    source: Option<String>,
    current_digest: &str,
    target_ms: f64,
) -> Result<Value, String> {
    let file = File::open(run.join("run.log")).map_err(|error| error.to_string())?;
    let mut wall = Vec::new();
    let mut gpu = Vec::new();
    let mut window_tokens = None;
    let mut scored_tokens = None;
    let mut expected_rounds = None;
    for line in BufReader::new(file).lines() {
        let line = line.map_err(|error| error.to_string())?;
        if let Some((_, fields)) = line.split_once("pretrain.initial ") {
            window_tokens = log_field(fields, "window_tokens")
                .or_else(|| log_field(fields, "sequence_tokens"))
                .map(str::parse::<u64>)
                .transpose()
                .map_err(|error| error.to_string())?;
            continue;
        }
        let Some((_, fields)) = line.split_once("pretrain.round ") else {
            continue;
        };
        let (w_ms, g_ms, scored, rounds) = parse_round(fields)?;
        wall.push(w_ms);
        gpu.push(g_ms);
        if scored_tokens
            .replace(scored)
            .is_some_and(|old| old != scored)
            || expected_rounds
                .replace(rounds)
                .is_some_and(|old| old != rounds)
        {
            return Err("causal round log changes geometry between rounds".into());
        }
    }
    if wall.is_empty() {
        return Err("causal round log has no completed rounds".into());
    }
    let window_tokens = window_tokens.ok_or("causal log lacks window_tokens")?;
    let scored_tokens = scored_tokens.unwrap();
    let samples = wall.len();
    let wall_report = distribution(wall)?;
    let gpu_report = distribution(gpu)?;
    let wall_p95 = wall_report["p95"]
        .as_f64()
        .ok_or("causal wall p95 is unavailable")?;
    let wall_mean = wall_report["mean"]
        .as_f64()
        .ok_or("causal wall mean is unavailable")?;
    let stage_profile = stage_profile(run)?;
    Ok(json!({
        "run": run.strip_prefix(root).unwrap_or(run),
        "source-digest": source,
        "matches-current-source": source.as_deref() == Some(current_digest),
        "hardware": "apple-metal",
        "samples": samples,
        "diagnostic-samples": 0,
        "provisional": true,
        "completed-rounds": samples,
        "expected-rounds": expected_rounds,
        "truth": {
            "transaction-kind": "causal-mhc4-bo",
            "window-tokens": window_tokens,
            "scored-tokens": scored_tokens,
            "evaluations-per-round": 1,
            "complete-candidate-transaction": true,
        },
        "latency": {
            "wall-ms": wall_report,
            "gpu-ms": gpu_report,
        },
        "target-round-ms": target_ms,
        "p95-target-met": wall_p95 <= target_ms,
        "strict-target-met": false,
        "required-p95-speedup": wall_p95 / target_ms,
        "throughput": {
            "mean-scored-tokens-per-second": scored_tokens as f64 * 1000.0 / wall_mean,
            "p95-scored-tokens-per-second": scored_tokens as f64 * 1000.0 / wall_p95,
        },
        "stage-profile": stage_profile,
        "stage-boundary": "provisional completed-round log rows; unavailable as a recordable baseline",
    }))
}

fn parse_round(fields: &str) -> Result<(f64, f64, u64, u64), String> {
    let wall = log_field(fields, "wall_ms")
        .ok_or("causal round log lacks wall_ms")?
        .parse::<f64>()
        .map_err(|error| error.to_string())?;
    let gpu = log_field(fields, "gpu_ms")
        .ok_or("causal round log lacks gpu_ms")?
        .parse::<f64>()
        .map_err(|error| error.to_string())?;
    let scored = log_field(fields, "scored_tokens")
        .ok_or("causal round log lacks scored_tokens")?
        .parse::<u64>()
        .map_err(|error| error.to_string())?;
    let rounds = log_field(fields, "total")
        .ok_or("causal round log lacks total")?
        .parse::<u64>()
        .map_err(|error| error.to_string())?;
    Ok((wall, gpu, scored, rounds))
}

fn log_field<'a>(fields: &'a str, name: &str) -> Option<&'a str> {
    fields.split_whitespace().find_map(|entry| {
        entry
            .strip_prefix(name)
            .and_then(|value| value.strip_prefix('='))
    })
}

fn causal_measure(
    root: &Path,
    run: &Path,
    source: Option<String>,
    current_digest: &str,
    target_ms: f64,
) -> Result<Value, String> {
    let result = std::fs::read_to_string(run.join("result.toml"))
        .map_err(|error| format!("causal result is unavailable: {error}"))?;
    if !result
        .lines()
        .any(|line| line.trim() == "status = \"completed\"")
    {
        return Err("performance evidence requires a completed causal run".into());
    }
    let file = File::open(run.join("optimization.jsonl")).map_err(|error| error.to_string())?;
    let rows = BufReader::new(file)
        .lines()
        .map(|line| {
            baux::artifact::json::from_str::<Value>(&line.map_err(|error| error.to_string())?)
                .map_err(|error| error.to_string())
        })
        .collect::<Result<Vec<_>, _>>()?;
    if rows.is_empty() {
        return Err("causal performance evidence has no rounds".into());
    }
    let metrics = [
        ("wall-ms", vec_path(&rows, &["round_wall_ms"], 1.0)),
        ("proposal-ms", vec_path(&rows, &["ask_ms"], 1.0)),
        ("evaluation-ms", vec_path(&rows, &["candidate_gpu_ms"], 1.0)),
        ("gpu-ms", vec_path(&rows, &["round_gpu_ms"], 1.0)),
        ("tell-ms", vec_path(&rows, &["tell_ms"], 1.0)),
    ];
    let latency = metrics
        .into_iter()
        .map(|(name, values)| Ok((name, distribution(values?)?)))
        .collect::<Result<baux::artifact::json::Map, String>>()?;
    let wall = latency
        .get("wall-ms")
        .ok_or("causal wall latency is unavailable")?;
    let wall_p95 = wall["p95"]
        .as_f64()
        .ok_or("causal wall p95 is unavailable")?;
    let wall_mean = wall["mean"]
        .as_f64()
        .ok_or("causal wall mean is unavailable")?;
    let window_tokens = rows[0]["window_tokens"]
        .as_u64()
        .or_else(|| rows[0]["sequence_tokens"].as_u64())
        .ok_or("causal trace lacks window_tokens")?;
    let scored_tokens = rows[0]["scored_tokens"]
        .as_u64()
        .ok_or("causal trace lacks scored_tokens")?;
    if rows.iter().any(|row| {
        row["window_tokens"]
            .as_u64()
            .or_else(|| row["sequence_tokens"].as_u64())
            != Some(window_tokens)
            || row["scored_tokens"].as_u64() != Some(scored_tokens)
    }) {
        return Err("causal trace changes token geometry between rounds".into());
    }
    let stage_profile = stage_profile(run)?;
    Ok(json!({
        "run": run.strip_prefix(root).unwrap_or(run),
        "source-digest": source,
        "matches-current-source": source.as_deref() == Some(current_digest),
        "hardware": "apple-metal",
        "samples": rows.len(),
        "diagnostic-samples": 0,
        "truth": {
            "transaction-kind": "causal-mhc4-bo",
            "window-tokens": window_tokens,
            "scored-tokens": scored_tokens,
            "evaluations-per-round": 1,
            "complete-candidate-transaction": true,
        },
        "latency": Value::from(latency),
        "initialization-samples": rows.iter().filter(|row| row["initializing"] == true).count(),
        "guided-samples": rows.iter().filter(|row| row["initializing"] == false).count(),
        "target-round-ms": target_ms,
        "p95-target-met": wall_p95 <= target_ms,
        "strict-target-met": rows.iter().all(|row| row["round_wall_ms"].as_f64().is_some_and(|wall| wall <= target_ms)),
        "required-p95-speedup": wall_p95 / target_ms,
        "throughput": {
            "mean-scored-tokens-per-second": scored_tokens as f64 * 1000.0 / wall_mean,
            "p95-scored-tokens-per-second": scored_tokens as f64 * 1000.0 / wall_p95,
        },
        "stage-profile": stage_profile,
        "stage-boundary": "one proposal, one causal likelihood evaluation, posterior update, and commit or rejection",
    }))
}

fn stage_profile(run: &Path) -> Result<Value, String> {
    let path = run.join("run.log");
    if !path.is_file() {
        return Ok(Value::null());
    }
    let text = std::fs::read_to_string(path).map_err(|error| error.to_string())?;
    stage_text(&text)
}

fn stage_text(text: &str) -> Result<Value, String> {
    let mut gpu = Vec::new();
    let mut rows = Vec::new();
    let mut stages = BTreeMap::<String, Vec<f64>>::new();
    for line in text.lines() {
        let Some((_, encoded)) = line.split_once("scorer.stage ") else {
            continue;
        };
        let sample: Value =
            baux::artifact::json::from_str(encoded).map_err(|error| error.to_string())?;
        gpu.push(
            sample["gpu_ms"]
                .as_f64()
                .ok_or("scorer stage sample lacks gpu_ms")?,
        );
        rows.push(
            sample["rows"]
                .as_u64()
                .ok_or("scorer stage sample lacks rows")?,
        );
        let fields = sample["stage_ms"]
            .as_map()
            .ok_or("scorer stage sample lacks stage_ms")?;
        for (name, value) in fields {
            let name = name
                .kind()
                .as_str()
                .ok_or("scorer stage name is not a string")?;
            stages.entry(name.to_owned()).or_default().push(
                value
                    .as_f64()
                    .ok_or("scorer stage duration is not a number")?,
            );
        }
    }
    if gpu.is_empty() {
        return Ok(Value::null());
    }
    let rows_per_sample = rows
        .iter()
        .all(|sample_rows| *sample_rows == rows[0])
        .then_some(rows[0]);
    let mut stage_reports = baux::artifact::json::Map::new();
    for (name, values) in stages {
        stage_reports.insert(Value::from(name), distribution(values)?);
    }
    Ok(json!({
        "samples": gpu.len(),
        "rows-per-sample": rows_per_sample,
        "profiled-gpu-ms": distribution(gpu)?,
        "stage-ms": Value::from(stage_reports),
        "boundary": "one profiled scorer chunk; stage means are not complete-round sums",
    }))
}

fn metal_measure(
    root: &Path,
    run: &Path,
    result: Value,
    source: Option<String>,
    current_digest: &str,
    target_ms: f64,
) -> Result<Value, String> {
    if result["status"] != "completed" {
        return Err("performance evidence requires a completed run".into());
    }
    let file = File::open(run.join("optimization.jsonl")).map_err(|error| error.to_string())?;
    let rows = BufReader::new(file)
        .lines()
        .map(|line| {
            baux::artifact::json::from_str::<Value>(&line.map_err(|error| error.to_string())?)
                .map_err(|error| error.to_string())
        })
        .collect::<Result<Vec<_>, _>>()?;
    let operational = rows
        .iter()
        .filter(|row| row["pool_rank_audit"].is_null())
        .cloned()
        .collect::<Vec<_>>();
    if operational.is_empty() {
        return Err("performance evidence has no non-diagnostic rounds".into());
    }
    let last = rows.last().ok_or("optimization trace is empty")?;
    let generation = &last["generation"];
    let tokens = &last["response"]["tokens"];
    let repaired = generation["repaired_tokens"]
        .as_u64()
        .or_else(|| (generation["repair_batches"].as_u64() == Some(0)).then_some(0));
    let drafted = generation["draft"]
        .as_seq()
        .map(|drafts| {
            drafts
                .iter()
                .filter_map(|draft| draft["generated_tokens"].as_u64())
                .sum::<u64>()
        })
        .unwrap_or(0);
    let metrics = [
        ("wall-ms", vec_path(&operational, &["wall_seconds"], 1000.0)),
        (
            "proposal-ms",
            vec_path(&operational, &["generation", "proposal_seconds"], 1000.0),
        ),
        (
            "evaluation-ms",
            vec_path(&operational, &["generation", "evaluation_seconds"], 1000.0),
        ),
        (
            "gpu-ms",
            vec_path(&operational, &["generation", "rollout_gpu_seconds"], 1000.0),
        ),
        (
            "reward-ms",
            vec_path(&operational, &["generation", "reward_seconds"], 1000.0),
        ),
        (
            "tell-ms",
            vec_path(&operational, &["generation", "tell_seconds"], 1000.0),
        ),
    ];
    let distributions = metrics
        .into_iter()
        .map(|(name, values)| Ok((name, distribution(values?)?)))
        .collect::<Result<baux::artifact::json::Map, String>>()?;
    let wall_p95 = distributions
        .get("wall-ms")
        .and_then(|value| value["p95"].as_f64())
        .ok_or("wall latency lacks p95")?;
    let experiment_regression = experiment_regression(run, &result, &rows)?;
    Ok(json!({
        "run": run.strip_prefix(root).unwrap_or(run),
        "source-digest": source,
        "matches-current-source": source.as_deref() == Some(current_digest),
        "hardware": "apple-metal",
        "samples": operational.len(),
        "diagnostic-samples": rows.len() - operational.len(),
        "truth": {
            "context-tokens": tokens["context"].clone(),
            "prompt-tokens": tokens["prompt"].clone(),
            "drafted-tokens": drafted,
            "evaluated-tokens": generation["evaluated_tokens"].clone(),
            "evaluated-core-positions": generation["evaluated_positions"].clone(),
            "scored-tokens": generation["scored_tokens"].clone(),
            "repaired-tokens": repaired,
            "generated-tokens": tokens["generated"].clone(),
            "committed-tokens": tokens["committed"].clone(),
            "repair-batches": generation["repair_batches"].clone(),
            "exact-per-token-generation": result["exact_per_token_generation"].clone(),
            "direct-token-decisions": result["direct_token_decisions_per_candidate"].clone(),
            "hash-expanded-tokens": result["hash_expanded_tokens_per_candidate"].clone(),
        },
        "experiment-regression": experiment_regression,
        "latency": Value::from(distributions),
        "initialization-samples": operational.iter().filter(|row| row["phase"] == "initialization").count(),
        "guided-samples": operational.iter().filter(|row| row["phase"] == "guided").count(),
        "pool-rank-audit": pool_rank(&rows),
        "target-round-ms": target_ms,
        "p95-target-met": wall_p95 <= target_ms,
        "strict-target-met": operational.iter().all(|row| row["wall_seconds"].as_f64().is_some_and(|seconds| seconds * 1000.0 <= target_ms)),
        "required-p95-speedup": wall_p95 / target_ms,
    }))
}

fn cuda_measure(
    root: &Path,
    run: &Path,
    result: Value,
    source: Option<String>,
    current_digest: &str,
    target_ms: f64,
) -> Result<Value, String> {
    if result["schema"] == "baux.cuda.mhc4-bo-round.v7" {
        return cuda_transaction(root, run, result, source, current_digest, target_ms);
    }
    if result["schema"] != "baux.cuda.mhc4-bo-round.v2" || result["full_bo_loop"] != true {
        return Err("CUDA performance evidence requires a complete mHC-4 BO round".into());
    }
    let evaluation_ms = result["candidate"]["total_device_ms"]
        .as_f64()
        .ok_or("CUDA result lacks candidate stage timing")?;
    let metrics = [
        ("wall-ms", result["wall_ms"].as_f64()),
        ("proposal-ms", result["proposal"]["device_ms"].as_f64()),
        ("evaluation-ms", Some(evaluation_ms)),
        ("gpu-ms", result["device_ms"].as_f64()),
        ("tell-ms", result["tell"]["wall_ms"].as_f64()),
    ];
    let latency = metrics
        .into_iter()
        .map(|(name, value)| {
            let value = value.ok_or_else(|| format!("CUDA result lacks {name}"))?;
            Ok((name, distribution(vec![value])?))
        })
        .collect::<Result<baux::artifact::json::Map, String>>()?;
    let wall = result["wall_ms"].as_f64().unwrap();
    Ok(json!({
        "run": run.strip_prefix(root).unwrap_or(run),
        "source-digest": source,
        "matches-current-source": source.as_deref() == Some(current_digest),
        "hardware": result["hardware"].clone(),
        "samples": 1,
        "truth": {
            "context-tokens": result["context_window_tokens"].clone(),
            "prompt-tokens": result["prompt_tokens"].clone(),
            "drafted-tokens": result["drafted_tokens"].clone(),
            "evaluated-tokens": result["evaluated_positions"].clone(),
            "repaired-tokens": result["repaired_tokens"].clone(),
            "generated-tokens": result["generated_tokens"].clone(),
            "committed-tokens": result["committed_tokens"].clone(),
            "repair-batches": result["repair_waves"].clone(),
            "exact-per-token-generation": result["exact_per_token_generation"].clone(),
            "direct-token-decisions": result["direct_token_decisions"].clone(),
            "hash-expanded-tokens": result["hash_expanded_tokens"].clone(),
        },
        "latency": Value::from(latency),
        "target-round-ms": target_ms,
        "p95-target-met": wall <= target_ms,
        "strict-target-met": wall <= target_ms,
        "required-p95-speedup": wall / target_ms,
        "stage-boundary": "CUDA scoring is fused into candidate evaluation; one Modal invocation contributes one sample",
    }))
}

fn cuda_metric(rows: &[Value], key: &str) -> Result<Vec<f64>, String> {
    rows.iter()
        .map(|row| {
            row[key]
                .as_f64()
                .ok_or_else(|| format!("CUDA round lacks {key}"))
        })
        .collect()
}

fn cuda_transaction(
    root: &Path,
    run: &Path,
    result: Value,
    source: Option<String>,
    current_digest: &str,
    target_ms: f64,
) -> Result<Value, String> {
    if result["full_bo_loop"] != true
        || result["full_generation_loop"] != true
        || result["full_scoring_loop"] != true
        || result["complete_candidate_transaction"] != true
    {
        return Err(
            "CUDA performance evidence requires a complete paired mHC-4 BO transaction".into(),
        );
    }
    let file = File::open(run.join("rounds.jsonl")).map_err(|error| error.to_string())?;
    let rows = BufReader::new(file)
        .lines()
        .map(|line| {
            baux::artifact::json::from_str::<Value>(&line.map_err(|error| error.to_string())?)
                .map_err(|error| error.to_string())
        })
        .collect::<Result<Vec<_>, _>>()?;
    let expected = result["bo"]["rounds"]
        .as_u64()
        .ok_or("CUDA result lacks bo.rounds")?;
    if rows.len() as u64 != expected || rows.is_empty() {
        return Err(format!(
            "CUDA round trace contains {} rows, expected {expected}",
            rows.len()
        ));
    }
    let wall = cuda_metric(&rows, "wall_ms")?;
    let gpu = cuda_metric(&rows, "device_ms")?;
    let host_overhead = wall
        .iter()
        .zip(&gpu)
        .map(|(wall, gpu)| (wall - gpu).max(0.0))
        .collect::<Vec<_>>();
    let metrics = [
        ("wall-ms", wall),
        ("proposal-ms", cuda_metric(&rows, "proposal_device_ms")?),
        (
            "incumbent-evaluation-ms",
            cuda_metric(&rows, "incumbent_evaluation_device_ms")?,
        ),
        (
            "candidate-evaluation-ms",
            cuda_metric(&rows, "candidate_evaluation_device_ms")?,
        ),
        ("evaluation-ms", cuda_metric(&rows, "evaluation_device_ms")?),
        ("gpu-ms", gpu),
        ("restore-ms", cuda_metric(&rows, "restore_device_ms")?),
        ("host-overhead-ms", host_overhead),
    ];
    let latency = metrics
        .into_iter()
        .map(|(name, values)| Ok((name, distribution(values)?)))
        .collect::<Result<baux::artifact::json::Map, String>>()?;
    let wall_p95 = latency
        .get("wall-ms")
        .and_then(|value| value["p95"].as_f64())
        .ok_or("CUDA wall latency lacks p95")?;
    let context = result["context_window_tokens"]
        .as_u64()
        .ok_or("CUDA result lacks context_window_tokens")?;
    let prompt = result["prompt_tokens"]
        .as_u64()
        .ok_or("CUDA result lacks prompt_tokens")?;
    let generated = context
        .checked_sub(prompt)
        .ok_or("CUDA prompt exceeds context")?;
    Ok(json!({
        "run": run.strip_prefix(root).unwrap_or(run),
        "source-digest": source,
        "matches-current-source": source.as_deref() == Some(current_digest),
        "hardware": result["hardware"].clone(),
        "samples": rows.len(),
        "truth": {
            "transaction-kind": "paired-mhc4-bo",
            "context-tokens": context,
            "prompt-tokens": prompt,
            "generated-tokens": generated,
            "evaluations-per-round": 2,
            "evaluated-tokens": result["evaluated_tokens"].clone(),
            "committed-tokens": result["committed_tokens"].clone(),
            "complete-candidate-transaction": true,
            "same-condition-pair": result["training_conditions"]["paired_candidate_incumbent"].clone(),
            "incumbent-remeasured": result["training_conditions"]["incumbent_remeasured_each_round"].clone(),
        },
        "latency": Value::from(latency),
        "initialization-samples": rows.iter().filter(|row| row["phase"] == "initialization").count(),
        "guided-samples": rows.iter().filter(|row| row["phase"] == "turbo_enn").count(),
        "target-round-ms": target_ms,
        "p95-target-met": wall_p95 <= target_ms,
        "strict-target-met": rows.iter().all(|row| row["wall_ms"].as_f64().is_some_and(|wall| wall <= target_ms)),
        "required-p95-speedup": wall_p95 / target_ms,
        "stage-boundary": "each sample includes proposal materialization, same-condition incumbent evaluation, candidate evaluation, posterior/reliability update, and commit or restore",
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn causal_fields() {
        let fields = "round=721 total=1000 scored_tokens=1048513 gpu_ms=1107.184 wall_ms=1267.110";
        assert_eq!(log_field(fields, "round"), Some("721"));
        assert_eq!(log_field(fields, "scored_tokens"), Some("1048513"));
        assert_eq!(log_field(fields, "wall_ms"), Some("1267.110"));
        assert_eq!(log_field(fields, "missing"), None);
    }

    #[test]
    fn scorer_profile() {
        let text = concat!(
            "time scorer.stage \
             {\"rows\":8192,\"gpu_ms\":500.0,\"stage_ms\":{\"gate\":120.0,\"pisa_attention\":140.\
             0}}\n",
            "time scorer.stage \
             {\"rows\":8192,\"gpu_ms\":480.0,\"stage_ms\":{\"gate\":110.0,\"pisa_attention\":130.\
             0}}\n",
        );
        let profile = stage_text(text).unwrap();
        assert_eq!(profile["samples"], 2);
        assert_eq!(profile["rows-per-sample"], 8192);
        assert_eq!(profile["profiled-gpu-ms"]["mean"], 490.0);
        assert_eq!(profile["stage-ms"]["gate"]["mean"], 115.0);
        assert_eq!(profile["stage-ms"]["pisa_attention"]["mean"], 135.0);
    }
}
