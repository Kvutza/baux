use super::*;
use kuva::prelude::*;

#[path = "plot/regional.rs"]
mod regional;
#[path = "plot/series.rs"]
mod series;
pub(super) use series::{
    candidate_series, cohort_tag, comparison_band, comparison_curve, context_tag, extract_series,
    incumbent_series, infer_recipe, number, run_label, summary_only,
};
use series::{observation_points, points, update_points, validation_points};

const SOURCES: &[(&str, &str, &str)] = &[
    ("validation.jsonl", "round", "reward"),
    ("validation.jsonl", "round", "mean_nll"),
    ("learning.jsonl", "round", "reward"),
    ("learning.jsonl", "round", "optimizer_incumbent"),
    ("optimization.jsonl", "round", "nll_after"),
    ("optimization.jsonl", "round", "incumbent_nll"),
    ("optimization.jsonl", "round", "candidate_nll"),
    ("optimization.jsonl", "round", "round_wall_ms"),
    ("rounds.jsonl", "round", "best_nll"),
    ("rounds.jsonl", "round", "mean_nll"),
    ("rounds.jsonl", "round", "wall_ms"),
];

const PALETTE: &[&str] = &[
    "#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#332288", "#882255",
];

/// Render only observed values. Reward is not relabelled as loss; missing
/// measurements are not fabricated. All figures are derived from saved streams
/// and compact series are embedded into the CBOR manifest for cross-run
/// comparison.
pub(super) fn render(output: &Path) -> Result<(usize, Value), String> {
    render_to(output, output)
}

pub(super) fn render_to(source_dir: &Path, plot_dir: &Path) -> Result<(usize, Value), String> {
    let mut count = 0;
    let mut summaries = Vec::new();
    for &(file, x, y) in SOURCES {
        let path = source_dir.join(file);
        if !path.exists() {
            continue;
        }
        let data = points(&path, x, y)?;
        if data.is_empty() {
            continue;
        }
        write_svg(plot_dir, file, x, y, &data)?;
        summaries.push(series_summary(file, x, y, &data));
        count += 1;
    }
    let val_file = if source_dir.join("validation.cbor").exists() {
        "validation.cbor"
    } else {
        "validation.json"
    };
    if let Some(data) = validation_points(&source_dir.join(val_file))?
        && !data.is_empty()
    {
        write_svg(plot_dir, val_file, "round", "nll", &data)?;
        summaries.push(series_summary(val_file, "round", "nll", &data));
        count += 1;
    }
    if let Some((rms, coh)) = update_points(&source_dir.join("tensor-updates.bin"))? {
        if !rms.is_empty() {
            write_svg(plot_dir, "tensor-updates.bin", "family", "rms", &rms)?;
            summaries.push(series_summary("tensor-updates.bin", "family", "rms", &rms));
            count += 1;
        }
        if !coh.is_empty() {
            write_svg(plot_dir, "tensor-updates.bin", "family", "coherence", &coh)?;
            summaries.push(series_summary(
                "tensor-updates.bin",
                "family",
                "coherence",
                &coh,
            ));
            count += 1;
        }
    }
    if summaries.is_empty() {
        for file in [
            "result.cbor",
            "result.json",
            "ucb-result.json",
            "thompson-result.json",
        ] {
            let path = source_dir.join(file);
            if !path.exists() {
                continue;
            }
            for y in ["best_nll", "mean_nll", "wall_ms"] {
                if let Some(data) = observation_points(&path, y)?
                    && !data.is_empty()
                {
                    write_svg(plot_dir, file, "round", y, &data)?;
                    summaries.push(series_summary(file, "round", y, &data));
                    count += 1;
                }
            }
        }
    }
    let regional = regional::render(source_dir, plot_dir)?;
    count += regional.len();
    summaries.extend(regional);
    Ok((count, json!(summaries)))
}

fn write_svg(
    output: &Path,
    file: &str,
    x: &str,
    y: &str,
    data: &[(f64, f64)],
) -> Result<(), String> {
    write_band(output, file, x, y, data, &[])
}

pub(super) fn write_band(
    output: &Path,
    file: &str,
    x: &str,
    y: &str,
    data: &[(f64, f64)],
    sems: &[f64],
) -> Result<(), String> {
    let mut line = LinePlot::new()
        .with_data(data.to_vec())
        .with_color(PALETTE[0])
        .with_stroke_width(2.2);
    if sems.len() == data.len() && sems.iter().any(|&s| s > 0.0) {
        let lower: Vec<f64> = data.iter().zip(sems).map(|(&(_, py), &s)| py - s).collect();
        let upper: Vec<f64> = data.iter().zip(sems).map(|(&(_, py), &s)| py + s).collect();
        line = line.with_band(lower, upper);
    }
    let plots: Vec<Plot> = vec![line.into()];
    let mut layout = Layout::auto_from_plots(&plots)
        .with_theme(Theme::minimal())
        .with_title(file)
        .with_x_label(x)
        .with_y_label(y)
        .with_interactive();
    if y.contains("nll") || y.contains("reward") || y.contains("score") || y.contains("error") {
        layout = layout.with_y_tick_format(TickFormat::Fixed(3));
    }
    let svg = render_to_svg(plots, layout);
    atomic(&output.join(format!("{file}.{y}.svg")), svg.as_bytes())
}

pub(super) fn series_summary(file: &str, x: &str, y: &str, data: &[(f64, f64)]) -> Value {
    series_band(file, x, y, data, &[])
}

pub(super) fn series_band(
    file: &str,
    x: &str,
    y: &str,
    data: &[(f64, f64)],
    sems: &[f64],
) -> Value {
    let indices = sample_indices(data.len(), 512);
    let compact: Vec<[f64; 2]> = indices
        .iter()
        .map(|&idx| [data[idx].0, data[idx].1])
        .collect();
    if sems.len() == data.len() && sems.iter().any(|&s| s > 0.0) {
        let compact_sem: Vec<f64> = indices.iter().map(|&idx| sems[idx]).collect();
        json!({
            "source": file,
            "x": x,
            "y": y,
            "points": data.len(),
            "first": data.first(),
            "last": data.last(),
            "series": compact,
            "sem": compact_sem,
        })
    } else {
        json!({
            "source": file,
            "x": x,
            "y": y,
            "points": data.len(),
            "first": data.first(),
            "last": data.last(),
            "series": compact,
        })
    }
}

fn sample_indices(len: usize, max_points: usize) -> Vec<usize> {
    if len <= max_points || max_points < 4 {
        return (0..len).collect();
    }
    let mut out = Vec::with_capacity(max_points);
    let step = (len - 1) as f64 / (max_points - 1) as f64;
    for i in 0..max_points {
        let idx = ((i as f64 * step).round() as usize).min(len - 1);
        out.push(idx);
    }
    out
}

fn is_deepbo(r: &Record, nll_keys: &[&str]) -> bool {
    let Some(s) = extract_series(r, false, nll_keys) else {
        return false;
    };
    s.len() >= 10
        && s.first().is_some_and(|(_, y)| *y > 12.1)
        && s.last().is_some_and(|(_, y)| *y > 12.1)
}

/// Render overlaid multi-run comparison SVGs and terminal braille previews from
/// CBOR-stored traces.
pub fn compare(root: &Path, records: &[Record]) -> Result<Value, String> {
    let out_dir = directory(root)?.join("comparisons");
    fs::create_dir_all(&out_dir).map_err(|e| e.to_string())?;
    let nll_keys: &[&str] = &["incumbent_nll", "nll_after", "best_nll", "mean_nll"];
    let val_keys: &[&str] = &["nll", "mean_nll"];
    let wall_keys: &[&str] = &["round_wall_ms", "wall_ms"];
    let has_full_vocab = records.iter().any(|r| {
        extract_series(r, false, nll_keys)
            .and_then(|s| s.first().copied())
            .is_some_and(|(_, y)| y > 11.0)
    });
    let has_deep_bo = records.iter().filter(|r| is_deepbo(r, nll_keys)).count() >= 2;
    let specs: &[(&str, &str, &str, &str, bool)] = &[
        ("nll", "NLL", "step", "NLL", false),
        ("delta_nll", "ΔNLL", "step", "ΔNLL", false),
        ("nll_by_time", "ΔNLL vs. Time", "time (s)", "ΔNLL", false),
        ("validation_nll", "Val ΔNLL", "step", "val ΔNLL", false),
        ("wall_ms", "ms/round", "step", "ms/round", true),
        (
            "mean_tracking_error",
            "Mean Tracking Error",
            "round",
            "tracking error",
            false,
        ),
        (
            "mean_best_score",
            "Mean Best Score",
            "round",
            "best score",
            false,
        ),
        (
            "regional_reward",
            "Regional Best Reward",
            "evaluations",
            "best reward",
            false,
        ),
    ];
    let mut generated = Vec::new();
    for &(name, title, x_label, y_label, log_y) in specs {
        let mut plots: Vec<Plot> = Vec::new();
        let mut included = Vec::new();
        let mut seen_signatures = Vec::new();
        for record in records {
            let Some(pts) = comparison_curve(
                record,
                name,
                has_full_vocab,
                has_deep_bo,
                nll_keys,
                val_keys,
                wall_keys,
            ) else {
                continue;
            };
            let sig = format!(
                "{}:{}:{:.5}:{:.5}:{}",
                record.optimizer,
                record.recipe,
                pts.first().map_or(0.0, |p| p.1),
                pts.last().map_or(0.0, |p| p.1),
                pts.len()
            );
            if seen_signatures.contains(&sig) {
                continue;
            }
            seen_signatures.push(sig);
            let color = PALETTE[plots.len() % PALETTE.len()];
            let short_commit = record.commit.get(..8).unwrap_or(&record.commit);
            let label = run_label(record);
            let mut lp = LinePlot::new()
                .with_data(pts.clone())
                .with_color(color)
                .with_stroke_width(2.2)
                .with_legend(label.clone());
            if let Some(sems) = comparison_band(record, name)
                && sems.len() == pts.len()
                && sems.iter().any(|&s| s > 0.0)
            {
                let lower: Vec<f64> = pts.iter().zip(&sems).map(|(&(_, y), &s)| y - s).collect();
                let upper: Vec<f64> = pts.iter().zip(&sems).map(|(&(_, y), &s)| y + s).collect();
                lp = lp.with_band(lower, upper);
            }
            if matches!(name, "nll" | "delta_nll" | "nll_by_time") {
                lp = lp.with_step();
            }
            plots.push(lp.into());
            included.push(json!({"id": record.id, "commit": short_commit, "optimizer": record.optimizer, "label": label}));
            if plots.len() >= 10 {
                break;
            }
        }
        if plots.is_empty() {
            continue;
        }
        generated.push(emit_comparison(
            &out_dir, name, title, x_label, y_label, log_y, plots, included,
        )?);
    }
    Ok(json!(generated))
}

fn emit_comparison(
    out_dir: &Path,
    name: &str,
    title: &str,
    x_label: &str,
    y_label: &str,
    log_y: bool,
    plots: Vec<Plot>,
    included: Vec<Value>,
) -> Result<Value, String> {
    let mut layout = Layout::auto_from_plots(&plots)
        .with_theme(Theme::minimal())
        .with_title(title)
        .with_x_label(x_label)
        .with_y_label(y_label)
        .with_legend_position(LegendPosition::OutsideRightTop)
        .with_interactive();
    if !matches!(
        name,
        "mean_tracking_error" | "mean_best_score" | "regional_reward"
    ) {
        layout = layout.with_log_x();
    }
    layout = match (
        log_y,
        matches!(name, "delta_nll" | "nll_by_time" | "validation_nll"),
    ) {
        (true, _) => layout.with_log_y(),
        (false, true) => layout.with_y_tick_format(TickFormat::Fixed(4)),
        (false, false) => layout.with_y_tick_format(TickFormat::Fixed(3)),
    };
    let svg = render_to_svg(plots, layout);
    let path = out_dir.join(format!("comparison.{name}.svg"));
    atomic(&path, svg.as_bytes())?;
    Ok(json!({
        "metric": name,
        "path": path.to_string_lossy(),
        "runs_plotted": included.len(),
        "series": included,
    }))
}
