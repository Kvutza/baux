//! Resident Apple Silicon Metal GPU worker helpers for TokaGym and TORAX 2.0.0.

use std::path::Path;

use baux::config::OptimizerSpec;
use baux::optimizer::DomainFrame;
use ndarray::Array2;

use super::Config;
use super::ckpt::SeedRecord;
use super::worker::TokamakWorker;
use crate::experiment::julia::config::Backend;

pub(super) fn check_metal(spec: &OptimizerSpec) -> Result<(), String> {
    if !cfg!(all(target_os = "macos", target_arch = "aarch64")) {
        return Err("resident-metal backend requires macOS Apple Silicon".into());
    }
    baux::UnifiedEngine::check_metal(spec).map_err(|e| e.to_string())
}

pub(super) fn low_span(domain: &DomainFrame) -> (Vec<f64>, Vec<f64>) {
    let dim = domain.num_dim();
    let z = Array2::zeros((1, dim));
    let o = Array2::ones((1, dim));
    let lo_arr = domain.from_unit(&z.view());
    let hi_arr = domain.from_unit(&o.view());
    let low: Vec<f64> = (0..dim).map(|j| lo_arr[[0, j]]).collect();
    let span: Vec<f64> = (0..dim).map(|j| hi_arr[[0, j]] - lo_arr[[0, j]]).collect();
    (low, span)
}

pub(super) fn eval_rows(
    worker: &mut Option<TokamakWorker>,
    rows: &[Vec<u16>],
    low: &[f64],
    span: &[f64],
    scratch: &mut Vec<f64>,
) -> Result<(Vec<f64>, Vec<f64>, u64), String> {
    let count = rows.len();
    if count == 0 {
        return Ok((Vec::new(), Vec::new(), 0));
    }
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    if let Some(w) = worker.as_mut() {
        if count == 1 {
            let (sc, dt_us) = w.eval_half(rows[0].as_ptr(), low, span)?;
            return Ok((vec![sc], vec![0.0], dt_us));
        }
        let refs: Vec<&[u16]> = rows.iter().map(Vec::as_slice).collect();
        let (scores, cost_us) = w.eval_halves(&refs, low, span)?;
        return Ok((scores, vec![0.0; count], cost_us * (count as u64)));
    }
    #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
    let _ = worker;
    let dim = low.len();
    scratch.resize(count * dim, 0.0);
    for (i, row) in rows.iter().enumerate() {
        for j in 0..dim {
            let u = f64::from(baux::unified_mem::decode_half(row[j])).clamp(0.0, 1.0);
            scratch[i * dim + j] = low[j] + u * span[j];
        }
    }
    let mut out = vec![0.0f64; count];
    baux_tokamak::evaluate_batch(scratch, count, &mut out);
    Ok((out, vec![0.0; count], 0))
}

pub(super) fn metal_seed(
    cfg: &Config,
    _domain: &DomainFrame,
    rep: u32,
    ckpt_dir: Option<&Path>,
    worker: &mut Option<TokamakWorker>,
) -> Result<SeedRecord, String> {
    let mut m_cfg = cfg.clone();
    m_cfg.backend = Backend::ResidentMetal;
    super::run_seed(&m_cfg, rep, ckpt_dir, worker)
}
