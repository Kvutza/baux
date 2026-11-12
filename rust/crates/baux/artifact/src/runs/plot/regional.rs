//! Keep each regional experiment arm's observed reward trajectory separate.
use super::*;

pub(super) fn render(source: &Path, output: &Path) -> Result<Vec<Value>, String> {
    let mut summaries = Vec::new();
    let (file, path) = if source.join("regional.cbor").exists() {
        ("regional.cbor", source.join("regional.cbor"))
    } else if source.join("regional.json").exists() {
        ("regional.json", source.join("regional.json"))
    } else {
        ("", PathBuf::new())
    };
    if !file.is_empty() {
        let bytes = fs::read(path).map_err(|e| e.to_string())?;
        let report: Value = if file.ends_with(".cbor") {
            baux::artifact::cbor::from_slice(&bytes).map_err(|e| e.to_string())?
        } else {
            json::from_slice(&bytes).map_err(|e| e.to_string())?
        };
        let results = report["results"]
            .as_seq()
            .ok_or("missing regional results")?;
        if report["experiment"].as_str() == Some("host-regions") {
            summaries.extend(host_results(output, file, results)?);
        } else {
            summaries.extend(resident_results(output, file, results)?);
        }
    }
    summaries.extend(tokamak_results(source, output)?);
    for (cbor_name, json_name) in [
        ("oceananigans.cbor", "oceananigans.json"),
        ("fuse.cbor", "fuse.json"),
    ] {
        summaries.extend(trace_results(source, output, cbor_name, json_name)?);
    }
    Ok(summaries)
}

fn tokamak_results(source: &Path, output: &Path) -> Result<Vec<Value>, String> {
    trace_results(source, output, "tokamak.cbor", "tokamak.json")
}

fn trace_results(
    source: &Path,
    output: &Path,
    cbor_name: &str,
    json_name: &str,
) -> Result<Vec<Value>, String> {
    let (file, path) = if source.join(cbor_name).exists() {
        (cbor_name, source.join(cbor_name))
    } else if source.join(json_name).exists() {
        (json_name, source.join(json_name))
    } else {
        return Ok(Vec::new());
    };
    let bytes = fs::read(path).map_err(|e| e.to_string())?;
    let report: Value = if file.ends_with(".cbor") {
        baux::artifact::cbor::from_slice(&bytes).map_err(|e| e.to_string())?
    } else {
        json::from_slice(&bytes).map_err(|e| e.to_string())?
    };
    let Some(traces) = report["traces"].as_seq() else {
        return Ok(Vec::new());
    };
    let mut summaries = Vec::new();
    for (y_key, alt_key, sem_key) in [
        (
            "mean_tracking_error",
            "tracking_error",
            "sem_tracking_error",
        ),
        ("mean_best_score", "best_score", "sem_best_score"),
    ] {
        let mut data = Vec::with_capacity(traces.len());
        let mut sems = Vec::with_capacity(traces.len());
        for entry in traces {
            let y_val = number(&entry[y_key]).or_else(|| number(&entry[alt_key]));
            if let (Some(x), Some(y)) = (number(&entry["round"]), y_val)
                && x.is_finite()
                && y.is_finite()
            {
                data.push((x, y));
                let sem = number(&entry[sem_key]).unwrap_or(0.0);
                sems.push(if sem.is_finite() && sem > 0.0 {
                    sem
                } else {
                    0.0
                });
            }
        }
        if !data.is_empty() {
            write_band(output, file, "round", y_key, &data, &sems)?;
            summaries.push(series_band(file, "round", y_key, &data, &sems));
        }
    }
    Ok(summaries)
}

fn resident_results(output: &Path, file: &str, results: &[Value]) -> Result<Vec<Value>, String> {
    let mut summaries = Vec::new();
    for result in results {
        let seed = result["seed"].as_u64().ok_or("missing regional seed")?;
        let regions = result["regions"].as_u64().ok_or("missing region count")?;
        let allocation = result["allocation"].as_str().ok_or("missing allocation")?;
        let sharing = result["sharing"].as_str().unwrap_or("origin_only");
        let starts = result["starts"].as_str().unwrap_or("separated");
        if !matches!(
            allocation,
            "least_trials" | "incumbent_ucb1" | "joint_acquisition" | "balanced" | "ucb"
        ) {
            return Err("unknown regional allocation".into());
        }
        let trace = result["trace"].as_seq().ok_or("missing regional trace")?;
        let data = trace
            .iter()
            .map(|point| {
                let values = point.as_seq().ok_or("invalid regional point")?;
                if values.len() != 2 {
                    return Err("regional point needs evaluation count and reward");
                }
                let x = values[0].as_f64().ok_or("invalid evaluation count")?;
                let y = values[1].as_f64().ok_or("invalid regional reward")?;
                if !x.is_finite() || !y.is_finite() {
                    return Err("nonfinite regional point");
                }
                Ok((x, y))
            })
            .collect::<Result<Vec<_>, &str>>()?;
        if data.is_empty() {
            continue;
        }
        let label = format!("reward-seed{seed}-regions{regions}-{starts}-{sharing}-{allocation}");
        write_svg(output, file, "evaluations", &label, &data)?;
        summaries.push(series_summary(file, "evaluations", &label, &data));
    }
    Ok(summaries)
}

fn host_results(output: &Path, file: &str, results: &[Value]) -> Result<Vec<Value>, String> {
    let mut summaries = Vec::new();
    for (index, result) in results.iter().enumerate() {
        let objective = result["objective"]
            .as_str()
            .ok_or("missing host objective")?;
        let sharing = result["sharing"].as_str().ok_or("missing host sharing")?;
        let roles = result["roles"].as_str().ok_or("missing host roles")?;
        let centers = result["restart_centers"]
            .as_str()
            .ok_or("missing host restart centers")?;
        for memory in ["fresh", "retain_relevant"] {
            let values = result[memory]["observation_values"]
                .as_seq()
                .ok_or("missing host observations")?;
            let mut best = f64::NEG_INFINITY;
            let mut data = Vec::with_capacity(values.len());
            for (step, value) in values.iter().enumerate() {
                let reward = value.as_f64().ok_or("invalid host reward")?;
                if !reward.is_finite() {
                    return Err("nonfinite host reward".into());
                }
                best = best.max(reward);
                data.push(((step + 1) as f64, best));
            }
            if data.is_empty() {
                return Err("empty host observations".into());
            }
            let label = format!(
                "best_reward-host{index:04}-{objective}-{sharing}-{roles}-{centers}-{memory}"
            );
            write_svg(output, file, "evaluations", &label, &data)?;
            summaries.push(series_summary(file, "evaluations", &label, &data));
        }
    }
    Ok(summaries)
}
