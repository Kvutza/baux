//! Parameter space and domain resolution for Julia physics experiments.

use baux::artifact::json::Value;
use baux::optimizer::DomainFrame;
use ndarray::Array2;
use std::path::Path;

use crate::experiment::julia::config::{Config, fuse_key, is_ocean};
use crate::experiment::julia::worker::{daemon_request, ensure_daemon};

pub fn ensure_upstream(root: &Path, lock_name: &str) -> Result<Value, String> {
    if let Ok(v) = baux::artifact::merkle::verify_lock(root, lock_name, None) {
        return Ok(v);
    }
    let lock_path = root
        .join("research/locks")
        .join(format!("{lock_name}.lock"));
    let text = std::fs::read_to_string(&lock_path)
        .map_err(|e| format!("read {}: {e}", lock_path.display()))?;
    baux::artifact::json::from_str(&text).map_err(|e| format!("parse lock: {e}"))
}

fn ocean_space(cfg: &Config) -> Result<(DomainFrame, Vec<f64>, Vec<f64>, usize), String> {
    let oc = &cfg.ocean;
    if oc.nx == 0 || oc.nz == 0 || oc.steps == 0 || oc.bound <= 0.0 {
        return Err("oceananigans config requires positive grid, steps, and bound".into());
    }
    let ny_eff = oc.ny.max(1);
    let (dim, state_len) = if oc.task == "kelvin-helmholtz" {
        if oc.actuators == 0 {
            return Err("oceananigans actuators must be >= 1".into());
        }
        (oc.nx * ny_eff * oc.nz * oc.actuators, 2 * oc.nx * oc.nz)
    } else if oc.task == "wind-mixing" {
        (2 * oc.nx * ny_eff, 2 * oc.nx * oc.nz)
    } else {
        return Err(format!("unknown oceananigans task '{}'", oc.task));
    };
    let lo = if oc.task == "wind-mixing" {
        -2.5
    } else {
        -oc.bound
    };
    let hi = if oc.task == "wind-mixing" {
        2.5
    } else {
        oc.bound
    };
    let mut bounds = Array2::zeros((dim, 2));
    for j in 0..dim {
        bounds[[j, 0]] = lo;
        bounds[[j, 1]] = hi;
    }
    let low = vec![lo; dim];
    let span = vec![hi - lo; dim];
    let domain = DomainFrame::new(bounds).map_err(|e| e.to_string())?;
    Ok((domain, low, span, state_len))
}

fn parse_floats(csv: &str, label: &str) -> Result<Vec<f64>, String> {
    csv.split(',')
        .filter(|s| !s.is_empty())
        .map(|s| {
            s.trim()
                .parse::<f64>()
                .map_err(|e| format!("parse {label}: {e}"))
        })
        .collect()
}

fn fuse_space(
    root: &Path,
    cfg: &Config,
) -> Result<(DomainFrame, Vec<f64>, Vec<f64>, usize), String> {
    let sock_path = ensure_daemon(root, false)?;
    let key = fuse_key(cfg);
    let reply = daemon_request(&sock_path, &format!("BOUNDS\t{key}\n"))?;
    let rest = reply
        .trim()
        .strip_prefix("BOUNDS\t")
        .ok_or_else(|| format!("unexpected FUSE daemon reply: {reply}"))?;
    let (lo_csv, hi_csv) = rest
        .split_once('\t')
        .ok_or_else(|| format!("malformed FUSE daemon reply: {reply}"))?;
    let low = parse_floats(lo_csv, "lo")?;
    let high = parse_floats(hi_csv, "hi")?;
    if low.is_empty() || low.len() != high.len() {
        return Err(format!(
            "invalid FUSE bounds lengths: {} vs {}",
            low.len(),
            high.len()
        ));
    }
    let dim = low.len();
    let mut bounds = Array2::zeros((dim, 2));
    let mut span = Vec::with_capacity(dim);
    for j in 0..dim {
        let (lo, hi) = (low[j], high[j]);
        if !(hi > lo) {
            return Err(format!("invalid FUSE interval [{lo}, {hi}] at dim {j}"));
        }
        bounds[[j, 0]] = lo;
        bounds[[j, 1]] = hi;
        span.push(hi - lo);
    }
    let domain = DomainFrame::new(bounds).map_err(|e| e.to_string())?;
    Ok((domain, low, span, 4096))
}

pub fn resolve_space(
    root: &Path,
    cfg: &Config,
) -> Result<(DomainFrame, Vec<f64>, Vec<f64>, usize), String> {
    if is_ocean(cfg) {
        ocean_space(cfg)
    } else {
        fuse_space(root, cfg)
    }
}
