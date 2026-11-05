use baux::artifact::json::{Value, json};
use sha2::{Digest, Sha256};
use std::fs::File;
use std::io::Read;
use std::path::Path;

pub(super) fn pool_rank(rows: &[Value]) -> Value {
    let audits = rows
        .iter()
        .map(|row| &row["pool_rank_audit"])
        .filter(|audit| {
            matches!(
                audit["schema"].as_str(),
                Some("baux.pool_rank_audit.v1" | "baux.pool_rank_audit.v2")
            )
        })
        .collect::<Vec<_>>();
    if audits.is_empty() {
        return Value::null();
    }
    let mean = |field: &str| {
        let values = audits
            .iter()
            .filter_map(|audit| audit[field].as_f64())
            .collect::<Vec<_>>();
        (!values.is_empty()).then(|| values.iter().sum::<f64>() / values.len() as f64)
    };
    let selected_top1 = audits
        .iter()
        .filter(|audit| audit["selected_rank"].as_u64() == Some(1))
        .count();
    let oracle_positive = audits
        .iter()
        .filter(|audit| {
            audit["best_improvement"]
                .as_f64()
                .is_some_and(|value| value > 0.0)
        })
        .count();
    let selected_positive = audits
        .iter()
        .filter(|audit| {
            audit["selected_improvement"]
                .as_f64()
                .is_some_and(|value| value > 0.0)
        })
        .count();
    let missed_positive = audits
        .iter()
        .filter(|audit| {
            audit["best_improvement"]
                .as_f64()
                .is_some_and(|value| value > 0.0)
                && audit["selected_improvement"]
                    .as_f64()
                    .is_some_and(|value| value <= 0.0)
        })
        .count();
    let correlation = |field: &str| {
        let pairs = audits
            .iter()
            .filter_map(|audit| audit["candidates"].as_seq())
            .flatten()
            .filter_map(|candidate| {
                Some((
                    candidate[field].as_f64()?,
                    candidate["improvement"].as_f64()?,
                ))
            })
            .collect::<Vec<_>>();
        pearson(&pairs)
    };
    let response_audit = audits
        .iter()
        .any(|audit| audit["schema"] == "baux.pool_rank_audit.v2");
    let mean_field = if response_audit {
        "response_mean"
    } else {
        "predicted_mean"
    };
    let error_field = if response_audit {
        "response_standard_error"
    } else {
        "predicted_standard_error"
    };
    json!({
        "points": audits.len(),
        "pool-size": 4,
        "selected-top1": selected_top1,
        "selected-top1-rate": selected_top1 as f64 / audits.len() as f64,
        "random-top1-rate": 0.25,
        "mean-selected-rank": mean("selected_rank"),
        "mean-selection-regret": mean("selection_regret"),
        "mean-selected-improvement": mean("selected_improvement"),
        "mean-best-improvement": mean("best_improvement"),
        "oracle-positive": oracle_positive,
        "selected-positive": selected_positive,
        "missed-positive": missed_positive,
        "score-improvement-correlation": correlation("acquisition_score"),
        "ranking-model": response_audit.then_some("historical-linear-operator-response-v1"),
        "response-mean-improvement-correlation": correlation(mean_field),
        "response-uncertainty-improvement-correlation": correlation(error_field),
        "mean-policy": counterfactual(&audits, 0.0),
        "ucb-one-policy": counterfactual(&audits, 1.0),
        "lcb-one-policy": counterfactual(&audits, -1.0),
    })
}

fn counterfactual(audits: &[&Value], beta: f64) -> Value {
    let choices = audits
        .iter()
        .filter_map(|audit| {
            let candidates = audit["candidates"].as_seq()?;
            let response_audit = audit["schema"] == "baux.pool_rank_audit.v2";
            let mean_field = if response_audit {
                "response_mean"
            } else {
                "predicted_mean"
            };
            let error_field = if response_audit {
                "response_standard_error"
            } else {
                "predicted_standard_error"
            };
            let selected = candidates
                .iter()
                .enumerate()
                .filter_map(|(index, candidate)| {
                    Some((
                        index,
                        candidate[mean_field].as_f64()? + beta * candidate[error_field].as_f64()?,
                    ))
                })
                .max_by(|left, right| left.1.total_cmp(&right.1))?
                .0;
            Some((
                selected,
                candidates[selected]["improvement"].as_f64()?,
                audit["best"].as_u64()? as usize,
            ))
        })
        .collect::<Vec<_>>();
    if choices.is_empty() {
        return Value::null();
    }
    json!({
        "points": choices.len(),
        "top1": choices.iter().filter(|choice| choice.0 == choice.2).count(),
        "positive": choices.iter().filter(|choice| choice.1 > 0.0).count(),
        "mean-improvement": choices.iter().map(|choice| choice.1).sum::<f64>() / choices.len() as f64,
    })
}

fn pearson(values: &[(f64, f64)]) -> Option<f64> {
    if values.len() < 2 {
        return None;
    }
    let count = values.len() as f64;
    let mean_x = values.iter().map(|pair| pair.0).sum::<f64>() / count;
    let mean_y = values.iter().map(|pair| pair.1).sum::<f64>() / count;
    let covariance = values
        .iter()
        .map(|pair| (pair.0 - mean_x) * (pair.1 - mean_y))
        .sum::<f64>();
    let variance_x = values
        .iter()
        .map(|pair| (pair.0 - mean_x).powi(2))
        .sum::<f64>();
    let variance_y = values
        .iter()
        .map(|pair| (pair.1 - mean_y).powi(2))
        .sum::<f64>();
    let scale = (variance_x * variance_y).sqrt();
    (scale > 0.0).then_some(covariance / scale)
}

pub(super) fn vec_path(rows: &[Value], path: &[&str], scale: f64) -> Result<Vec<f64>, String> {
    rows.iter()
        .map(|row| {
            path.iter()
                .fold(row, |value, key| &value[*key])
                .as_f64()
                .map(|value| value * scale)
                .ok_or_else(|| format!("controller trace lacks {}", path.join(".")))
        })
        .collect()
}

pub(super) fn distribution(mut values: Vec<f64>) -> Result<Value, String> {
    if values.is_empty()
        || values
            .iter()
            .any(|value| !value.is_finite() || *value < 0.0)
    {
        return Err("latency distribution requires finite nonnegative samples".into());
    }
    values.sort_by(f64::total_cmp);
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let p95 = values[((values.len() * 95).div_ceil(100)).saturating_sub(1)];
    Ok(json!({
        "count": values.len(),
        "mean": mean,
        "p95": p95,
        "max": values[values.len() - 1],
    }))
}

pub(super) fn experiment_regression(
    run: &Path,
    result: &Value,
    rows: &[Value],
) -> Result<Value, String> {
    let first = rows.first().ok_or("optimization trace is empty")?;
    let response = &first["response"];
    let paired = &first["generation"]["paired_generation"];
    let journal_accepted = first["acceptance"]["accepted"]
        .as_bool()
        .ok_or("controller trace lacks acceptance.accepted")?;
    let response_accepted = response["accepted"]
        .as_bool()
        .ok_or("controller trace lacks response.accepted")?;
    if journal_accepted != response_accepted {
        return Err("controller trace disagrees on the accepted decision".into());
    }
    let round = first["round"]
        .as_u64()
        .ok_or("controller trace lacks round")?;
    let candidate_path = run.join(format!("round-{round:04}/candidate/generated.txt"));
    let incumbent_path = run.join(format!("round-{round:04}/incumbent/generated.txt"));
    let candidate_nll = -paired["candidate_reward"]
        .as_f64()
        .ok_or("controller trace lacks paired candidate NLL")?;
    let incumbent_nll = -paired["incumbent_reward"]
        .as_f64()
        .ok_or("controller trace lacks paired incumbent NLL")?;
    Ok(json!({
        "parameters": result["parameters"].clone(),
        "round": round,
        "evaluated-tokens": response["tokens"]["evaluated"].clone(),
        "candidate-objective": response["candidate"]["mean"].clone(),
        "incumbent-objective": response["baseline"]["mean"].clone(),
        "candidate-nll": candidate_nll,
        "incumbent-nll": incumbent_nll,
        "accepted": response_accepted,
        "decoded-output": {
            "identity": "sha256-decoded-utf8",
            "candidate": sha256_file(&candidate_path)?,
            "incumbent": sha256_file(&incumbent_path)?,
        },
    }))
}

fn sha256_file(path: &Path) -> Result<String, String> {
    let mut file = File::open(path)
        .map_err(|error| format!("decoded output {} is unavailable: {error}", path.display()))?;
    let mut hash = Sha256::new();
    let mut buffer = [0u8; 64 * 1024];
    loop {
        let read = file.read(&mut buffer).map_err(|error| error.to_string())?;
        if read == 0 {
            break;
        }
        hash.update(&buffer[..read]);
    }
    Ok(format!("{:x}", hash.finalize()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn p95_tail() {
        let values = (1..=100).map(f64::from).collect();
        let report = distribution(values).unwrap();
        assert_eq!(report["p95"], 95.0);
        assert_eq!(report["max"], 100.0);
    }

    #[test]
    fn pool_quality() {
        let rows = [json!({
            "pool_rank_audit": {
                "schema": "baux.pool_rank_audit.v2",
                "selected_rank": 1,
                "selected_improvement": 4.0,
                "best_improvement": 4.0,
                "selection_regret": 0.0,
                "best": 3,
                "candidates": [
                    {"predicted_mean": 11.0, "predicted_standard_error": 5.0, "response_mean": 1.0, "response_standard_error": 4.0, "acquisition_score": 1.0, "improvement": 1.0},
                    {"predicted_mean": 12.0, "predicted_standard_error": 4.0, "response_mean": 2.0, "response_standard_error": 3.0, "acquisition_score": 2.0, "improvement": 2.0},
                    {"predicted_mean": 13.0, "predicted_standard_error": 3.0, "response_mean": 3.0, "response_standard_error": 2.0, "acquisition_score": 3.0, "improvement": 3.0},
                    {"predicted_mean": 14.0, "predicted_standard_error": 2.0, "response_mean": 4.0, "response_standard_error": 1.0, "acquisition_score": 4.0, "improvement": 4.0},
                ],
            }
        })];
        let report = pool_rank(&rows);
        assert_eq!(report["selected-top1"], 1);
        assert_eq!(report["mean-policy"]["top1"], 1);
        assert_eq!(report["score-improvement-correlation"], 1.0);
        assert_eq!(report["response-mean-improvement-correlation"], 1.0);
    }
}
