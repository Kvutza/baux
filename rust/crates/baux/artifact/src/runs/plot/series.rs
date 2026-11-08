use std::fs::{self, File};
use std::io::{BufRead, BufReader};
use std::path::Path;

use baux::artifact::json::{self, Value};
use baux::artifact::runs::{Record, field};

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

pub(in super::super) fn comparison_curve(
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

pub(in super::super) fn run_label(record: &Record) -> String {
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
        .trim_end_matches(".cbor")
        .trim_end_matches(".json");
    format!("{ctx} {} ({short_commit})", slug.get(..14).unwrap_or(slug))
}

pub(in super::super) fn cohort_tag(record: &Record) -> &'static str {
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

pub(in super::super) fn context_tag(recipe: &str) -> &'static str {
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

pub(in super::super) fn summary_only(record: &Record) -> bool {
    let Some(seq) = record.measurements.as_seq() else {
        return false;
    };
    !seq.is_empty()
        && seq.iter().all(|item| {
            field(item, "source") == "summary" && item["points"].as_u64().unwrap_or(0) <= 4
        })
}

pub(in super::super) fn incumbent_series(
    record: &Record,
    nll_keys: &[&str],
) -> Option<Vec<(f64, f64)>> {
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

pub(in super::super) fn candidate_series(record: &Record) -> Option<Vec<(f64, f64)>> {
    if summary_only(record) {
        return None;
    }
    extract_series(record, false, &["candidate_nll", "mean_nll"])
        .map(|s| s.into_iter().map(|(x, y)| (x.max(0.0) + 1.0, y)).collect())
}

pub(in super::super) fn extract_series(
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

fn decode_value(bytes: &[u8]) -> Option<Value> {
    let first = bytes.iter().find(|b| !b.is_ascii_whitespace());
    if first == Some(&b'{') {
        json::from_slice(bytes).ok()
    } else {
        baux::artifact::cbor::from_slice(bytes).ok()
    }
}

pub(super) fn validation_points(path: &Path) -> Result<Option<Vec<(f64, f64)>>, String> {
    if !path.exists() {
        return Ok(None);
    }
    let bytes = fs::read(path).map_err(|e| e.to_string())?;
    let Some(value) = decode_value(&bytes) else {
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

pub(super) fn update_points(
    path: &Path,
) -> Result<Option<(Vec<(f64, f64)>, Vec<(f64, f64)>)>, String> {
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

pub(super) fn observation_points(path: &Path, y: &str) -> Result<Option<Vec<(f64, f64)>>, String> {
    let bytes = fs::read(path).map_err(|e| e.to_string())?;
    let Some(value) = decode_value(&bytes) else {
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

pub(super) fn points(path: &Path, x: &str, y: &str) -> Result<Vec<(f64, f64)>, String> {
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

pub(in super::super) fn number(value: &Value) -> Option<f64> {
    value
        .as_f64()
        .or_else(|| value.as_u64().map(|v| v as f64))
        .or_else(|| value.as_i64().map(|v| v as f64))
}
