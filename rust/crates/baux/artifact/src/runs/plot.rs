use super::*;
use kuva::prelude::*;
use std::io::{BufRead, BufReader};

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
    if let Some(data) = validation_points(&source_dir.join("validation.json"))?
        && !data.is_empty()
    {
        write_svg(plot_dir, "validation.json", "round", "nll", &data)?;
        summaries.push(series_summary("validation.json", "round", "nll", &data));
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
        for file in ["result.json", "ucb-result.json", "thompson-result.json"] {
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
    Ok((count, json!(summaries)))
}

fn write_svg(
    output: &Path,
    file: &str,
    x: &str,
    y: &str,
    data: &[(f64, f64)],
) -> Result<(), String> {
    let plots: Vec<Plot> = vec![
        LinePlot::new()
            .with_data(data.to_vec())
            .with_color("steelblue")
            .into(),
    ];
    let mut layout = Layout::auto_from_plots(&plots)
        .with_title(file)
        .with_x_label(x)
        .with_y_label(y)
        .with_interactive();
    if y.contains("nll") || y.contains("reward") {
        layout = layout.with_y_tick_format(TickFormat::Fixed(3));
    }
    let svg = render_to_svg(plots, layout);
    atomic(&output.join(format!("{file}.{y}.svg")), svg.as_bytes())
}

pub(super) fn series_summary(file: &str, x: &str, y: &str, data: &[(f64, f64)]) -> Value {
    let compact: Vec<[f64; 2]> = downsample(data, 512)
        .into_iter()
        .map(|(px, py)| [px, py])
        .collect();
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

fn downsample(data: &[(f64, f64)], max_points: usize) -> Vec<(f64, f64)> {
    if data.len() <= max_points || max_points < 4 {
        return data.to_vec();
    }
    let mut out = Vec::with_capacity(max_points);
    let step = (data.len() - 1) as f64 / (max_points - 1) as f64;
    for i in 0..max_points {
        let idx = ((i as f64 * step).round() as usize).min(data.len() - 1);
        out.push(data[idx]);
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
                "{}:{:.5}:{:.5}:{}",
                record.optimizer,
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
                .with_data(pts)
                .with_color(color)
                .with_stroke_width(2.2)
                .with_legend(label.clone());
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

fn filtered_incumbent(
    record: &Record,
    has_full_vocab: bool,
    has_deep_bo: bool,
    nll_keys: &[&str],
) -> Option<Vec<(f64, f64)>> {
    let s = incumbent_series(record, nll_keys)?;
    if (has_full_vocab && s.first()?.1 < 11.0)
        || (has_deep_bo && (s.len() < 10 || s.last()?.1 < 12.1))
    {
        None
    } else {
        Some(s)
    }
}

pub(super) fn comparison_curve(
    record: &Record,
    name: &str,
    has_full_vocab: bool,
    has_deep_bo: bool,
    nll_keys: &[&str],
    val_keys: &[&str],
    wall_keys: &[&str],
) -> Option<Vec<(f64, f64)>> {
    if summary_only(record) {
        return None;
    }
    match name {
        "nll" => Some(
            filtered_incumbent(record, has_full_vocab, has_deep_bo, nll_keys)?
                .into_iter()
                .map(|(x, y)| (x.max(0.0) + 1.0, y))
                .collect(),
        ),
        "delta_nll" => {
            let s = filtered_incumbent(record, has_full_vocab, has_deep_bo, nll_keys)?;
            let y0 = s.first()?.1;
            Some(
                s.into_iter()
                    .map(|(x, y)| (x.max(0.0) + 1.0, y - y0))
                    .collect(),
            )
        }
        "nll_by_time" => {
            let nll = filtered_incumbent(record, has_full_vocab, has_deep_bo, nll_keys)?;
            let wall = extract_series(record, false, wall_keys)?;
            let mean_sec =
                (wall.iter().map(|(_, w)| *w).sum::<f64>() / wall.len().max(1) as f64 / 1000.0)
                    .max(0.01);
            let y0 = nll.first()?.1;
            Some(
                nll.into_iter()
                    .map(|(x, y)| (((x.max(0.0) + 1.0) * mean_sec).max(0.05), y - y0))
                    .collect(),
            )
        }
        "validation_nll" => {
            let s = extract_series(record, true, val_keys).filter(|s| {
                (!has_full_vocab || s.first().is_some_and(|p| p.1 >= 11.0))
                    && (!has_deep_bo || s.len() >= 4)
            })?;
            let y0 = s.first()?.1;
            Some(
                s.into_iter()
                    .map(|(x, y)| (x.max(0.0) + 1.0, y - y0))
                    .collect(),
            )
        }
        "wall_ms" => Some(
            extract_series(record, false, wall_keys)
                .filter(|s| !has_deep_bo || s.len() >= 10)?
                .into_iter()
                .map(|(x, y)| (x.max(0.0) + 1.0, y.max(1.0)))
                .collect(),
        ),
        _ => None,
    }
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
        .with_log_x();
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

pub(super) fn run_label(record: &Record) -> String {
    let ctx = context_tag(&record.recipe);
    let out = &record.output;
    let short_commit = record.commit.get(..8).unwrap_or(&record.commit);
    if out.contains("bo-muon-state") {
        let m = if record.optimizer == "muon" {
            "muon"
        } else {
            "warm-start"
        };
        return format!("{ctx} {m} ({short_commit})");
    }
    let known = [
        ("t4-stage-profile-128", "paired-reliability"),
        ("t4-toml-group4-128", "group4"),
        ("t4-hnet-1m-gaussian-enn", "gaussian-t4"),
        ("t4-hnet-1m-identifiable", "identifiable"),
        ("64k-bo-20261010", "gaussian"),
        ("4k-stable-transition", "stable"),
        ("4k-patch-profile", "patch"),
        ("1m-learn", "probe"),
    ];
    if let Some((_, mech)) = known.iter().find(|&&(pat, _)| out.contains(pat)) {
        return format!("{ctx} {mech} ({short_commit})");
    }
    let slug = Path::new(out)
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("run")
        .trim_end_matches(".json");
    format!("{ctx} {} ({short_commit})", slug.get(..14).unwrap_or(slug))
}

pub(super) fn cohort_tag(record: &Record) -> &'static str {
    let ctx = context_tag(&record.recipe);
    match (
        record.recipe.starts_with("stack-v3-rust-go"),
        record.recipe.starts_with("stack-v3-python"),
        ctx,
    ) {
        (true, _, "1M") => "1M Rust+Go",
        (true, _, "577K") => "577K Rust+Go",
        (true, _, _) => "Rust+Go",
        (_, true, "1M") => "1M Python",
        (_, true, "64K") => "64K Python",
        (_, true, "4K") => "4K Python",
        (_, true, _) => "Python",
        _ => ctx,
    }
}

pub(super) fn context_tag(recipe: &str) -> &'static str {
    [
        ("@1048576", "1M"),
        ("@1056510", "1M"),
        ("@577088", "577K"),
        ("@65536", "64K"),
        ("@4096", "4K"),
        ("@4224", "4K"),
    ]
    .iter()
    .find_map(|&(suf, tag)| recipe.ends_with(suf).then_some(tag))
    .unwrap_or("ctx")
}

pub(super) fn summary_only(record: &Record) -> bool {
    let Some(seq) = record.measurements.as_seq() else {
        return false;
    };
    !seq.is_empty()
        && seq.iter().all(|item| {
            field(item, "source") == "summary" && item["points"].as_u64().unwrap_or(0) <= 4
        })
}

pub(super) fn incumbent_series(record: &Record, nll_keys: &[&str]) -> Option<Vec<(f64, f64)>> {
    let pts = extract_series(record, false, nll_keys)?;
    let mut running_min = f64::INFINITY;
    Some(
        pts.into_iter()
            .map(|(x, y)| {
                if y < running_min {
                    running_min = y;
                }
                (x, running_min)
            })
            .collect(),
    )
}

pub(super) fn candidate_series(record: &Record) -> Option<Vec<(f64, f64)>> {
    if summary_only(record) {
        return None;
    }
    extract_series(record, false, &["candidate_nll", "mean_nll"])
        .map(|s| s.into_iter().map(|(x, y)| (x.max(0.0) + 1.0, y)).collect())
}

pub(super) fn extract_series(
    record: &Record,
    validation_only: bool,
    y_keys: &[&str],
) -> Option<Vec<(f64, f64)>> {
    let seq = record.measurements.as_seq()?;
    for y_key in y_keys {
        for item in seq {
            let source = field(item, "source");
            if validation_only && !source.starts_with("validation") {
                continue;
            }
            if !validation_only && source.starts_with("validation") {
                continue;
            }
            if field(item, "y") != *y_key {
                continue;
            }
            if let Some(raw) = item.get("series").and_then(|v| v.as_seq()) {
                let pts: Vec<(f64, f64)> = raw
                    .iter()
                    .filter_map(|pair| {
                        let arr = pair.as_seq()?;
                        Some((number(arr.first()?)?, number(arr.get(1)?)?))
                    })
                    .collect();
                if !pts.is_empty() {
                    return Some(pts);
                }
            }
        }
    }
    None
}

fn validation_points(path: &Path) -> Result<Option<Vec<(f64, f64)>>, String> {
    if !path.exists() {
        return Ok(None);
    }
    let bytes = fs::read(path).map_err(|e| e.to_string())?;
    let Ok(value) = json::from_slice::<Value>(&bytes) else {
        return Ok(None);
    };
    let Some(measurements) = value.get("measurements").and_then(|v| v.as_seq()) else {
        return Ok(None);
    };
    let mut pts = Vec::new();
    for entry in measurements {
        if let (Some(x), Some(y)) = (number(&entry["round"]), number(&entry["nll"]))
            && x.is_finite()
            && y.is_finite()
        {
            pts.push((x, y));
        }
    }
    Ok(Some(pts))
}

fn update_points(path: &Path) -> Result<Option<(Vec<(f64, f64)>, Vec<(f64, f64)>)>, String> {
    if !path.exists() {
        return Ok(None);
    }
    let bytes = fs::read(path).map_err(|e| e.to_string())?;
    if bytes.len() < 24 || &bytes[0..8] != b"BAUXUPD1" {
        return Err("invalid BAUXUPD1 header".into());
    }
    let u32_le = |s: &[u8], off: usize| u32::from_le_bytes(s[off..off + 4].try_into().unwrap());
    let f32_le = |s: &[u8], off: usize| f32::from_le_bytes(s[off..off + 4].try_into().unwrap());
    let f64_le = |s: &[u8], off: usize| f64::from_le_bytes(s[off..off + 8].try_into().unwrap());
    if u32_le(&bytes, 8) != 1 {
        return Err("unsupported BAUXUPD1 version".into());
    }
    let count = u32_le(&bytes, 20) as usize;
    if bytes.len() != 24 + count * 80 {
        return Err("truncated BAUXUPD1 payload".into());
    }
    let mut init_sq = [0.0f64; 17];
    let mut acc_sq = [0.0f64; 17];
    let mut net_sq = [0.0f64; 17];
    for rec in bytes[24..].chunks_exact(80) {
        let fam = usize::from(rec[8]);
        if fam < 17 {
            let elems = f64::from(u32_le(rec, 4));
            let irms = f64::from(f32_le(rec, 20));
            init_sq[fam] += elems * irms * irms;
            acc_sq[fam] += f64_le(rec, 64);
            net_sq[fam] += f64_le(rec, 72);
        }
    }
    let mut rms = Vec::new();
    let mut coh = Vec::new();
    for fam in 0..17 {
        if init_sq[fam] > 0.0 {
            let ratio = (net_sq[fam] / init_sq[fam]).sqrt();
            if ratio.is_finite() {
                rms.push((fam as f64, ratio));
            }
        }
        if acc_sq[fam] > 0.0 {
            let drift = net_sq[fam] / acc_sq[fam];
            if drift.is_finite() {
                coh.push((fam as f64, drift));
            }
        }
    }
    Ok(Some((rms, coh)))
}

fn observation_points(path: &Path, y: &str) -> Result<Option<Vec<(f64, f64)>>, String> {
    let bytes = fs::read(path).map_err(|e| e.to_string())?;
    let Ok(value) = json::from_slice::<Value>(&bytes) else {
        return Ok(None);
    };
    let Some(obs) = value["bo"].get("observations").and_then(|v| v.as_seq()) else {
        return Ok(None);
    };
    let mut pts = Vec::new();
    for entry in obs {
        if let (Some(px), Some(py)) = (number(&entry["round"]), number(&entry[y]))
            && px.is_finite()
            && py.is_finite()
        {
            pts.push((px, py));
        }
    }
    Ok(Some(pts))
}

fn points(path: &Path, x: &str, y: &str) -> Result<Vec<(f64, f64)>, String> {
    let mut points = Vec::new();
    let reader = BufReader::new(File::open(path).map_err(|e| e.to_string())?);
    for (index, line) in reader.lines().enumerate() {
        let line = line.map_err(|e| e.to_string())?;
        if line.trim().is_empty() {
            continue;
        }
        let value: Value = json::from_str(&line)
            .map_err(|e| format!("{} line {}: {e}", path.display(), index + 1))?;
        if let (Some(x), Some(y)) = (number(&value[x]), number(&value[y])) {
            if !x.is_finite() || !y.is_finite() {
                return Err("non-finite plotted measurement".into());
            }
            points.push((x, y));
            if points.len() > 100_000 {
                return Err("plot exceeds 100000 points; raw measurements retained".into());
            }
        }
    }
    Ok(points)
}

pub(super) fn number(value: &Value) -> Option<f64> {
    value
        .as_f64()
        .or_else(|| value.as_u64().map(|v| v as f64))
        .or_else(|| value.as_i64().map(|v| v as f64))
}
