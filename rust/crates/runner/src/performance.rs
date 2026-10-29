mod perf_measure;

use baux::artifact::json::{Value, json, pretty_string};
use baux::experimental::{
    Backend, BackendSchedule, MHC4_COEFFS, MHC4_PARAMS, MHC4_PATCH, ModelProgram, PISA_BYTES,
    ProgramKind, perf_digest, perf_sources,
};
use baux_tooling::radar::{PerformanceAction, PerformanceBackend};
use perf_measure::measured;
use std::fs::File;
use std::path::Path;

const SCHEMA: &str = "baux.performance-certificate.v1";
const REGRESSION_SCHEMA: &str = "baux.experiment-regression.v1";
const REGRESSION_BASELINE: &str = "performance/mhc4-1m-round-parity.json";
const CONTEXT_GOAL: u64 = 1_048_576;
const NOISE_MS: f64 = 2.0;

pub(super) fn run(root: &Path, action: PerformanceAction) -> Result<(), String> {
    match action {
        PerformanceAction::Show {
            config,
            backend,
            run,
            json,
        } => {
            let certificate = certificate(root, &config, backend, run.as_deref())?;
            if json {
                print(&certificate)
            } else {
                summary(&certificate)
            }
        }
        PerformanceAction::Record {
            config,
            backend,
            run,
            output,
        } => record(root, &config, backend, &run, &output),
        PerformanceAction::Gate {
            config,
            backend,
            run,
            baseline,
            tolerance_percent,
        } => gate(
            root,
            &config,
            backend,
            run.as_deref(),
            &baseline,
            tolerance_percent,
        ),
    }
}

fn workload_spec(
    config: &baux::config::RunConfig,
) -> Result<(&'static str, Option<u64>, Option<u64>, u64, u64), String> {
    if let Some(generation) = config.generation.as_ref() {
        let prompt = prompt_tokens(generation)?;
        let generated = u64::from(generation.max_tokens);
        let retained = prompt
            .checked_add(generated)
            .ok_or("retained context length overflow")?
            .min(CONTEXT_GOAL);
        Ok((
            "generation",
            Some(prompt),
            Some(generated),
            retained,
            retained - u64::from(generation.draft.is_none()),
        ))
    } else {
        let tokens = config
            .window_tokens
            .map(u64::from)
            .ok_or("performance certificate requires objective.window-tokens")?;
        Ok(("causal-likelihood", None, None, tokens, tokens))
    }
}

fn certificate(
    root: &Path,
    config_path: &Path,
    backend: PerformanceBackend,
    run: Option<&Path>,
) -> Result<Value, String> {
    let config_path = root.join(config_path);
    let (config, _) = baux::config::load_tune(&config_path)?;
    let model = config
        .model
        .ok_or("performance certificate requires a model")?;
    if model.id() != "fbt-pisa1-mhc4-v1" {
        return Err("performance certificate currently covers fbt-pisa1-mhc4-v1".into());
    }
    let (mode, prompt, generated, retained, input_positions) = workload_spec(&config)?;
    let core_positions = input_positions.div_ceil(MHC4_PATCH);
    let broad_positions = core_positions.div_ceil(128) * 128;
    let cache_capacity = core_positions
        .checked_next_power_of_two()
        .ok_or("cache capacity overflow")?
        .max(4096);

    let program = ModelProgram::mhc4_program();
    let schedule = schedule(backend);
    let missing = schedule.coverage(&program, ProgramKind::ExperimentRound)?;
    let recurrent = program
        .recurrent_core
        .ok_or("model program has no recurrent core")?;
    let executions = program.layer_visits(recurrent.training_visits)?.len() as u64;
    let contraction_flops = MHC4_COEFFS
        .checked_mul(2)
        .and_then(|flops| flops.checked_mul(broad_positions))
        .ok_or("contraction work overflow")?;
    let cache_bytes = cache_capacity
        .checked_mul(PISA_BYTES)
        .and_then(|bytes| bytes.checked_mul(executions))
        .ok_or("cache byte count overflow")?;
    let digest = perf_digest(root, backend_kind(backend))?;
    let sources = perf_sources(root, backend_kind(backend))?
        .into_iter()
        .map(|path| {
            path.strip_prefix(root)
                .unwrap_or(&path)
                .display()
                .to_string()
        })
        .collect::<Vec<_>>();
    let active_kernels = schedule.kernels_for(&program, ProgramKind::ExperimentRound)?;
    let static_contract = json!({
        "model": model.id(),
        "program-version": program.version,
        "backend": backend.name(),
        "compiler": compiler_name(schedule.compiler),
        "source-digest": digest,
        "source-files": sources,
        "workload": {
            "mode": mode,
            "prompt-token-cap": prompt,
            "generated-token-cap": generated,
            "window-tokens": (mode == "causal-likelihood").then_some(retained),
            "retained-context-tokens": retained,
            "target-input-positions": input_positions,
            "mhc4-patch-tokens": MHC4_PATCH,
            "core-positions": core_positions,
            "broad-pass-positions": broad_positions,
            "rounds": config.rounds(),
            "target-round-ms": config.target_ms(),
            "target-tokens-per-second": retained as f64 * 1000.0 / f64::from(config.target_ms()),
        },
        "architecture": {
            "parameters": MHC4_PARAMS,
            "physical-layers": program.physical_layers,
            "executed-layer-visits": executions,
            "contraction-coefficients-per-position": MHC4_COEFFS,
            "contraction-flops": contraction_flops,
            "pisa-state-bytes": cache_bytes,
            "cache-capacity": cache_capacity,
        },
        "lowering": {
            "kernel-count": active_kernels.len(),
            "kernels": kernel_rows(&active_kernels),
            "missing-count": missing.len(),
            "missing": missing.iter().map(|op| format!("{op:?}")).collect::<Vec<_>>(),
        },
        "goal": {
            "maximum-context-tokens": CONTEXT_GOAL,
            "context-met": retained == CONTEXT_GOAL,
            "lowering-complete": missing.is_empty(),
            "static-goal-met": retained == CONTEXT_GOAL && missing.is_empty(),
        },
        "counting-boundary": "mHC-4 core dense contractions and resident PISA state; excludes routing logistics, normalization, sampling, BO proposal, reward, and tell",
    });
    let mut measurement = run
        .map(|path| {
            measured(
                root,
                path,
                static_contract["source-digest"].as_str().unwrap(),
                f64::from(config.target_ms()),
            )
        })
        .transpose()?
        .unwrap_or(Value::null());
    if !measurement.is_null() {
        measurement["matches-workload"] = json!(truth_matches(&measurement, retained));
    }
    Ok(json!({
        "schema": SCHEMA,
        "config": config_path.strip_prefix(root).unwrap_or(&config_path),
        "static-contract": static_contract,
        "measurement": measurement,
    }))
}

fn prompt_tokens(generation: &baux::config::GenerationConfig) -> Result<u64, String> {
    generation
        .corpus_prompt_tokens
        .map(u64::from)
        .or_else(|| {
            (!generation.corpus_prompt.is_empty()).then_some(generation.corpus_prompt.len() as u64)
        })
        .or_else(|| {
            generation
                .tasks
                .iter()
                .map(|task| task.prompt.len() as u64)
                .max()
        })
        .ok_or_else(|| "generation workload has no prompt length".into())
}

fn truth_matches(measurement: &Value, context: u64) -> bool {
    let truth = &measurement["truth"];
    if truth["transaction-kind"] == "causal-mhc4-bo" {
        return truth["window-tokens"]
            .as_u64()
            .or_else(|| truth["sequence-tokens"].as_u64())
            == Some(context)
            && truth["evaluations-per-round"].as_u64() == Some(1)
            && truth["complete-candidate-transaction"].as_bool() == Some(true);
    }
    let prompt = truth["prompt-tokens"].as_u64();
    let generated = truth["generated-tokens"].as_u64();
    let geometry_matches = truth["context-tokens"].as_u64() == Some(context)
        && prompt
            .zip(generated)
            .is_some_and(|(prompt, generated)| prompt.checked_add(generated) == Some(context));
    if truth["transaction-kind"] == "paired-mhc4-bo" {
        return geometry_matches
            && truth["evaluations-per-round"].as_u64() == Some(2)
            && truth["complete-candidate-transaction"].as_bool() == Some(true)
            && truth["same-condition-pair"].as_bool() == Some(true);
    }
    geometry_matches
        && truth["committed-tokens"].as_u64() == generated
        && truth["exact-per-token-generation"].as_bool() == Some(true)
}

fn kernel_rows(kernels: &[&baux::experimental::ScheduledKernel]) -> Vec<Value> {
    kernels
        .iter()
        .map(|kernel| {
            json!({
                "name": kernel.kernel,
                "nodes": kernel.nodes,
                "tile": kernel.tile,
                "threads": kernel.threads,
            })
        })
        .collect()
}

fn record(
    root: &Path,
    config: &Path,
    backend: PerformanceBackend,
    run: &Path,
    output: &Path,
) -> Result<(), String> {
    let certificate = certificate(root, config, backend, Some(run))?;
    if certificate["measurement"]["provisional"] == true {
        return Err("an incomplete run cannot be recorded as a performance baseline".into());
    }
    if certificate["measurement"]["matches-current-source"] != true {
        return Err(
            "run predates the current hot-path source; execute a short matching replay".into(),
        );
    }
    if certificate["measurement"]["matches-workload"] != true {
        return Err("run truth labels do not match the certified workload".into());
    }
    let output = root.join(output);
    if let Some(parent) = output.parent() {
        std::fs::create_dir_all(parent).map_err(|error| error.to_string())?;
    }
    std::fs::write(
        &output,
        format!(
            "{}\n",
            pretty_string(&certificate).map_err(|error| error.to_string())?
        ),
    )
    .map_err(|error| error.to_string())?;
    println!("recorded {}", output.display());
    Ok(())
}

fn check_arch(baseline: &Value, current: &Value, failures: &mut Vec<String>) -> Result<(), String> {
    for field in [
        "contraction-flops",
        "pisa-state-bytes",
        "executed-layer-visits",
    ] {
        let old = baseline["static-contract"]["architecture"][field]
            .as_u64()
            .ok_or_else(|| format!("baseline lacks architecture.{field}"))?;
        let new = current["static-contract"]["architecture"][field]
            .as_u64()
            .ok_or_else(|| format!("certificate lacks architecture.{field}"))?;
        if new > old {
            failures.push(format!("{field} increased {old} -> {new}"));
        }
    }
    let old_missing = baseline["static-contract"]["lowering"]["missing-count"]
        .as_u64()
        .ok_or("baseline lacks lowering.missing-count")?;
    let new_missing = current["static-contract"]["lowering"]["missing-count"]
        .as_u64()
        .ok_or("certificate lacks lowering.missing-count")?;
    if new_missing > old_missing {
        failures.push(format!(
            "missing backend lowerings increased {old_missing} -> {new_missing}"
        ));
    }
    Ok(())
}

fn gate(
    root: &Path,
    config: &Path,
    backend: PerformanceBackend,
    run: Option<&Path>,
    baseline: &Path,
    tolerance_percent: u32,
) -> Result<(), String> {
    if tolerance_percent > 100 {
        return Err("performance tolerance percent must be in 0..=100".into());
    }
    let tolerance = f64::from(tolerance_percent) / 100.0;
    let baseline_path = root.join(baseline);
    let baseline: Value = baux::artifact::json::from_reader(
        File::open(&baseline_path).map_err(|error| error.to_string())?,
    )
    .map_err(|error| error.to_string())?;
    if baseline["schema"] != SCHEMA {
        return Err("unsupported performance baseline schema".into());
    }
    let experiment_baseline = if backend == PerformanceBackend::Metal {
        let path = root.join(REGRESSION_BASELINE);
        let value: Value = baux::artifact::json::from_reader(
            File::open(&path).map_err(|error| error.to_string())?,
        )
        .map_err(|error| error.to_string())?;
        if value["schema"] != REGRESSION_SCHEMA {
            return Err("unsupported experiment regression baseline schema".into());
        }
        Some(value)
    } else {
        None
    };
    let current = certificate(root, config, backend, run)?;
    let hot_path_changed =
        baseline["static-contract"]["source-digest"] != current["static-contract"]["source-digest"];
    let schedule_changed = baseline["static-contract"]["lowering"]["kernels"]
        != current["static-contract"]["lowering"]["kernels"];
    let workload_changed =
        baseline["static-contract"]["workload"] != current["static-contract"]["workload"];
    if workload_changed {
        return Err(
            "performance workload differs from the baseline; record a separate baseline".into(),
        );
    }
    let mut failures = Vec::new();
    check_arch(&baseline, &current, &mut failures)?;
    let replay_required = hot_path_changed || schedule_changed || !failures.is_empty();
    if replay_required {
        if current["measurement"].is_null() {
            return Err(format!(
                "hot path changed; a matching short replay is required{}",
                if failures.is_empty() {
                    String::new()
                } else {
                    format!(": {}", failures.join("; "))
                }
            ));
        }
        if current["measurement"]["matches-current-source"] != true {
            return Err("measured replay does not match the current hot-path source".into());
        }
        if current["measurement"]["provisional"] == true {
            return Err("performance gates require a completed run".into());
        }
        if current["measurement"]["matches-workload"] != true {
            return Err("measured replay truth labels do not match the baseline workload".into());
        }
        if let Some(experiment_baseline) = &experiment_baseline {
            compare_experiment(experiment_baseline, &current, &mut failures)?;
        }
        compare_latency(&baseline, &current, tolerance, &mut failures)?;
    }
    let passed = failures.is_empty();
    let report = json!({
        "schema": "baux.performance-gate.v1",
        "baseline": baseline_path.strip_prefix(root).unwrap_or(&baseline_path),
        "passed": passed,
        "hot-path-changed": hot_path_changed,
        "schedule-changed": schedule_changed,
        "measured-replay-required": replay_required,
        "tolerance": tolerance,
        "failures": failures,
        "p95-target-met": current["measurement"]["p95-target-met"].clone(),
        "strict-target-met": current["measurement"]["strict-target-met"].clone(),
    });
    print(&report)?;
    if passed {
        Ok(())
    } else {
        Err("performance regression gate failed".into())
    }
}

fn compare_experiment(
    expected: &Value,
    current: &Value,
    failures: &mut Vec<String>,
) -> Result<(), String> {
    if expected["config"] != current["config"] {
        failures.push("experiment config changed".into());
    }
    if expected["model"] != current["static-contract"]["model"] {
        failures.push("experiment model changed".into());
    }
    if expected["truth"] != current["measurement"]["truth"] {
        failures.push("truth labels changed".into());
    }
    let actual = &current["measurement"]["experiment-regression"];
    if actual.is_null() {
        return Err("measured replay lacks experiment-regression evidence".into());
    }
    for field in [
        "parameters",
        "round",
        "evaluated-tokens",
        "candidate-objective",
        "incumbent-objective",
        "candidate-nll",
        "incumbent-nll",
        "accepted",
    ] {
        if expected[field] != actual[field] {
            failures.push(format!(
                "experiment {field} changed {:?} -> {:?}",
                expected[field], actual[field]
            ));
        }
    }
    for role in ["identity", "candidate", "incumbent"] {
        if expected["decoded-output"][role] != actual["decoded-output"][role] {
            failures.push(format!("experiment decoded output {role} changed"));
        }
    }
    Ok(())
}

fn compare_latency(
    baseline: &Value,
    current: &Value,
    tolerance: f64,
    failures: &mut Vec<String>,
) -> Result<(), String> {
    for metric in ["wall-ms", "proposal-ms", "evaluation-ms", "gpu-ms"] {
        let Some(old) = baseline["measurement"]["latency"][metric]["p95"].as_f64() else {
            continue;
        };
        let new = current["measurement"]["latency"][metric]["p95"]
            .as_f64()
            .ok_or_else(|| format!("replay lacks measurement.latency.{metric}.p95"))?;
        let limit = old * (1.0 + tolerance) + NOISE_MS;
        if new > limit {
            failures.push(format!("{metric} p95 {new:.3} exceeds {limit:.3}"));
        }
    }
    Ok(())
}

fn schedule(backend: PerformanceBackend) -> BackendSchedule {
    match backend {
        PerformanceBackend::Metal => BackendSchedule::mhc4_metal(),
        PerformanceBackend::CudaOxide => BackendSchedule::mhc4_cuda(),
    }
}

fn backend_kind(value: PerformanceBackend) -> Backend {
    match value {
        PerformanceBackend::Metal => Backend::Metal,
        PerformanceBackend::CudaOxide => Backend::CudaOxide,
    }
}

fn compiler_name(compiler: baux::experimental::Compiler) -> &'static str {
    match compiler {
        baux::experimental::Compiler::Native => "native",
        baux::experimental::Compiler::CudaOxide => "cuda-oxide",
        baux::experimental::Compiler::Tvm => "tvm",
    }
}

fn print(value: &Value) -> Result<(), String> {
    println!(
        "{}",
        pretty_string(value).map_err(|error| error.to_string())?
    );
    Ok(())
}

fn summary(value: &Value) -> Result<(), String> {
    let contract = &value["static-contract"];
    let workload = &contract["workload"];
    let lowering = &contract["lowering"];
    let goal = &contract["goal"];
    let measurement = &value["measurement"];
    println!(
        "performance {} · {} · {} context · {:.0}ms target · {} token/s target",
        contract["backend"].as_str().unwrap_or("unknown-backend"),
        contract["model"].as_str().unwrap_or("unknown-model"),
        workload["retained-context-tokens"].as_u64().unwrap_or(0),
        workload["target-round-ms"].as_f64().unwrap_or(0.0),
        decimal(&workload["target-tokens-per-second"]),
    );
    println!(
        "static lowerings: {} kernels · {} missing · complete={}",
        lowering["kernel-count"].as_u64().unwrap_or(0),
        lowering["missing-count"].as_u64().unwrap_or(0),
        goal["lowering-complete"].as_bool().unwrap_or(false),
    );
    let mut blockers = Vec::new();
    if goal["context-met"] != true {
        blockers.push(format!(
            "workload has {} tokens; {} required",
            workload["retained-context-tokens"].as_u64().unwrap_or(0),
            goal["maximum-context-tokens"].as_u64().unwrap_or(0),
        ));
    }
    if goal["lowering-complete"] != true {
        blockers.push(format!(
            "{} semantic operations have no backend lowering",
            lowering["missing-count"].as_u64().unwrap_or(0)
        ));
    }
    if let Some(boundary) = contract["counting-boundary"].as_str() {
        blockers.push(format!("static work account is partial: {boundary}"));
    }
    if measurement.is_null() {
        println!("measurement: missing");
        blockers.push("no completed matching-source run was supplied".to_owned());
    } else {
        println!(
            "measurement: {} · {} operational + {} diagnostic samples · source={} · \
             max-workload={} · provisional={}",
            measurement["hardware"]
                .as_str()
                .unwrap_or("unknown-hardware"),
            measurement["samples"].as_u64().unwrap_or(0),
            measurement["diagnostic-samples"].as_u64().unwrap_or(0),
            verdict(&measurement["matches-current-source"]),
            verdict(&measurement["matches-workload"]),
            if measurement["provisional"] == true {
                "yes"
            } else {
                "no"
            },
        );
        let wall = &measurement["latency"]["wall-ms"];
        println!(
            "wall ms: mean={} · p95={} · max={} · target={} · required speedup={}x",
            decimal(&wall["mean"]),
            decimal(&wall["p95"]),
            decimal(&wall["max"]),
            verdict(&measurement["p95-target-met"]),
            decimal(&measurement["required-p95-speedup"]),
        );
        if !measurement["throughput"].is_null() {
            println!(
                "throughput token/s: mean={} · p95={}",
                decimal(&measurement["throughput"]["mean-scored-tokens-per-second"]),
                decimal(&measurement["throughput"]["p95-scored-tokens-per-second"]),
            );
        }
        let profile = &measurement["stage-profile"];
        if let Some(stage_map) = profile["stage-ms"].as_map() {
            let mut stages = stage_map
                .iter()
                .filter_map(|(name, report)| {
                    Some((name.kind().as_str()?, report["mean"].as_f64()?))
                })
                .collect::<Vec<_>>();
            stages.sort_by(|left, right| right.1.total_cmp(&left.1));
            println!(
                "profiled scorer chunk: {} rows · gpu mean={}ms · hot stages={}",
                profile["rows-per-sample"].as_u64().unwrap_or(0),
                decimal(&profile["profiled-gpu-ms"]["mean"]),
                stages
                    .into_iter()
                    .take(5)
                    .map(|(name, mean)| format!("{name}={mean:.3}ms"))
                    .collect::<Vec<_>>()
                    .join(","),
            );
        }
        let audit = &measurement["pool-rank-audit"];
        if !audit.is_null() {
            println!(
                "pool audit: {} points · top1={}/{} ({}) · mean rank={} · corr \
                 score/response-mean/response-se={}/{}/{} · positive={}/{} · missed={} · mean \
                 regret={} · selected={} · oracle={}",
                audit["points"].as_u64().unwrap_or(0),
                audit["selected-top1"].as_u64().unwrap_or(0),
                audit["points"].as_u64().unwrap_or(0),
                decimal(&audit["selected-top1-rate"]),
                decimal(&audit["mean-selected-rank"]),
                decimal(&audit["score-improvement-correlation"]),
                decimal(&audit["response-mean-improvement-correlation"]),
                decimal(&audit["response-uncertainty-improvement-correlation"]),
                audit["selected-positive"].as_u64().unwrap_or(0),
                audit["oracle-positive"].as_u64().unwrap_or(0),
                audit["missed-positive"].as_u64().unwrap_or(0),
                scientific(&audit["mean-selection-regret"]),
                scientific(&audit["mean-selected-improvement"]),
                scientific(&audit["mean-best-improvement"]),
            );
            println!(
                "policy replay mean/ucb1/lcb1: top1={}/{}/{} · improvement={}/{}/{}",
                audit["mean-policy"]["top1"].as_u64().unwrap_or(0),
                audit["ucb-one-policy"]["top1"].as_u64().unwrap_or(0),
                audit["lcb-one-policy"]["top1"].as_u64().unwrap_or(0),
                scientific(&audit["mean-policy"]["mean-improvement"]),
                scientific(&audit["ucb-one-policy"]["mean-improvement"]),
                scientific(&audit["lcb-one-policy"]["mean-improvement"]),
            );
        }
        if measurement["matches-current-source"] != true {
            blockers.push("measurement predates the current hot-path source".to_owned());
        }
        if measurement["provisional"] == true {
            blockers.push(format!(
                "run stopped after {} of {} rounds; evidence cannot be recorded as a baseline",
                measurement["completed-rounds"].as_u64().unwrap_or(0),
                measurement["expected-rounds"].as_u64().unwrap_or(0),
            ));
        }
        if measurement["matches-workload"] != true {
            blockers
                .push("measurement does not exercise the configured maximum geometry".to_owned());
        }
        if measurement["p95-target-met"] != true {
            blockers.push("measured p95 misses the transaction target".to_owned());
        }
    }
    blockers.push("no matched BAUX-versus-Muon outcome is represented".to_owned());
    println!("blockers:");
    for blocker in blockers {
        println!("- {blocker}");
    }
    println!("full evidence: rerun with --json");
    Ok(())
}

fn decimal(value: &Value) -> String {
    value
        .as_f64()
        .map(|number| format!("{number:.3}"))
        .unwrap_or_else(|| "unknown".to_owned())
}

fn scientific(value: &Value) -> String {
    value
        .as_f64()
        .map(|number| format!("{number:.6e}"))
        .unwrap_or_else(|| "unknown".to_owned())
}

fn verdict(value: &Value) -> &'static str {
    match value.as_bool() {
        Some(true) => "yes",
        Some(false) => "no",
        None => "unknown",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cuda_geometry() {
        let measurement = json!({
            "truth": {
                "transaction-kind": "paired-mhc4-bo",
                "context-tokens": 1_048_576,
                "prompt-tokens": 524_288,
                "generated-tokens": 524_288,
                "evaluations-per-round": 2,
                "evaluated-tokens": 134_217_728,
                "committed-tokens": 18_874_368,
                "complete-candidate-transaction": true,
                "same-condition-pair": true,
            }
        });
        assert!(truth_matches(&measurement, 1_048_576));
    }

    #[test]
    fn window_geometry() {
        let measurement = json!({
            "truth": {
                "transaction-kind": "causal-mhc4-bo",
                "window-tokens": 1_048_576,
                "scored-tokens": 1_048_513,
                "evaluations-per-round": 1,
                "complete-candidate-transaction": true,
            }
        });
        assert!(truth_matches(&measurement, 1_048_576));
        assert!(!truth_matches(&measurement, 65_536));
    }

    fn semantic_fixture(candidate_nll: f64, candidate_digest: &str) -> Value {
        json!({
            "measurement": {
                "truth": {"context-tokens": 1_048_576, "evaluated-tokens": 871_296},
                "experiment-regression": {
                    "parameters": 1_177_756_440,
                    "round": 1,
                    "evaluated-tokens": 871_296,
                    "candidate-objective": -12.495894,
                    "incumbent-objective": -12.48408,
                    "candidate-nll": candidate_nll,
                    "incumbent-nll": 12.483136,
                    "accepted": false,
                    "decoded-output": {
                        "identity": "sha256-decoded-utf8",
                        "candidate": candidate_digest,
                        "incumbent": "incumbent-stable",
                    },
                },
            },
        })
    }

    #[test]
    fn gate_drift() {
        let replay = semantic_fixture(12.494953, "candidate-stable");
        let baseline = json!({
            "truth": replay["measurement"]["truth"].clone(),
            "parameters": 1_177_756_440,
            "round": 1,
            "evaluated-tokens": 871_296,
            "candidate-objective": -12.495894,
            "incumbent-objective": -12.48408,
            "candidate-nll": 12.494953,
            "incumbent-nll": 12.483136,
            "accepted": false,
            "decoded-output": {
                "identity": "sha256-decoded-utf8",
                "candidate": "candidate-stable",
                "incumbent": "incumbent-stable",
            },
        });
        let current = semantic_fixture(12.498162, "candidate-changed");
        let mut failures = Vec::new();
        compare_experiment(&baseline, &current, &mut failures).unwrap();
        assert!(
            failures
                .iter()
                .any(|failure| failure.contains("candidate-nll changed"))
        );
        assert!(
            failures
                .iter()
                .any(|failure| failure.contains("decoded output candidate changed"))
        );
    }

    #[test]
    fn gate_replay() {
        let current = semantic_fixture(12.494953, "candidate-stable");
        let mut baseline = current["measurement"]["experiment-regression"].clone();
        baseline["truth"] = current["measurement"]["truth"].clone();
        let mut failures = Vec::new();
        compare_experiment(&baseline, &current, &mut failures).unwrap();
        assert!(failures.is_empty(), "{failures:?}");
    }
}
