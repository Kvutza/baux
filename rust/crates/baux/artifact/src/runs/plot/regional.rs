//! Keep each regional experiment arm's observed reward trajectory separate.
use super::*;

pub(super) fn render(source: &Path, output: &Path) -> Result<Vec<Value>, String> {
    let path = source.join("regional.json");
    if !path.exists() {
        return Ok(Vec::new());
    }
    let text = fs::read_to_string(path).map_err(|e| e.to_string())?;
    let report: Value = json::from_str(&text).map_err(|e| e.to_string())?;
    let results = report["results"]
        .as_seq()
        .ok_or("missing regional results")?;
    if report["experiment"].as_str() == Some("host-regions") {
        return host_results(output, results);
    }
    resident_results(output, results)
}

fn resident_results(output: &Path, results: &[Value]) -> Result<Vec<Value>, String> {
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
        write_svg(output, "regional.json", "evaluations", &label, &data)?;
        summaries.push(series_summary(
            "regional.json",
            "evaluations",
            &label,
            &data,
        ));
    }
    Ok(summaries)
}

fn host_results(output: &Path, results: &[Value]) -> Result<Vec<Value>, String> {
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
            write_svg(output, "regional.json", "evaluations", &label, &data)?;
            summaries.push(series_summary(
                "regional.json",
                "evaluations",
                &label,
                &data,
            ));
        }
    }
    Ok(summaries)
}
