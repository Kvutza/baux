//! Resident Apple Silicon Metal GPU execution for TokaGym and TORAX 2.0.0.

use std::path::Path;

use baux::config::OptimizerSpec;
use baux::optimizer::DomainFrame;
use ndarray::Array2;

use super::Config;
use super::ckpt::{SeedRecord, load_done, save_done};
use super::worker::TokamakWorker;

pub(super) fn check_metal(spec: &OptimizerSpec) -> Result<(), String> {
    if !cfg!(all(target_os = "macos", target_arch = "aarch64")) {
        return Err("resident-metal backend requires macOS Apple Silicon".into());
    }
    match spec {
        OptimizerSpec::SingleStateBo {
            perturbation: Some(_),
            ..
        } => Err("resident-metal single-state-bo does not accept [optimizer.perturbation]".into()),
        OptimizerSpec::SingleStateBo {
            regional_search, ..
        } => {
            if let Some(rs) = regional_search {
                rs.validate()?;
            }
            Ok(())
        }
        OptimizerSpec::BayesianSearch {
            candidates,
            arms,
            perturbation,
            regional_search,
            ..
        } => {
            if let Some(pert) = perturbation {
                pert.validate()?;
            }
            if *arms == 0 || *candidates == 0 || !candidates.is_multiple_of(*arms) {
                return Err(
                    "bayesian-search requires positive candidates divisible by arms".into(),
                );
            }
            if let Some(rs) = regional_search {
                rs.validate()?;
            }
            Ok(())
        }
        _ => Err("resident-metal backend requires single-state-bo or bayesian-search".into()),
    }
}

fn low_span(domain: &DomainFrame) -> (Vec<f64>, Vec<f64>) {
    let dim = domain.num_dim();
    let z = Array2::zeros((1, dim));
    let o = Array2::ones((1, dim));
    let lo_arr = domain.from_unit(&z.view());
    let hi_arr = domain.from_unit(&o.view());
    let low: Vec<f64> = (0..dim).map(|j| lo_arr[[0, j]]).collect();
    let span: Vec<f64> = (0..dim).map(|j| hi_arr[[0, j]] - lo_arr[[0, j]]).collect();
    (low, span)
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
fn eval_rows(
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
    if let Some(w) = worker.as_mut() {
        if count == 1 {
            let (sc, dt_us) = w.eval_half(rows[0].as_ptr(), low, span)?;
            return Ok((vec![sc], vec![0.0], dt_us));
        }
        let refs: Vec<&[u16]> = rows.iter().map(Vec::as_slice).collect();
        let (scores, cost_us) = w.eval_halves(&refs, low, span)?;
        return Ok((scores, vec![0.0; count], cost_us * (count as u64)));
    }
    let dim = low.len();
    scratch.resize(count * dim, 0.0);
    for (i, row) in rows.iter().enumerate() {
        for j in 0..dim {
            let u = f64::from(baux::shm_slab::decode_half(row[j])).clamp(0.0, 1.0);
            scratch[i * dim + j] = low[j] + u * span[j];
        }
    }
    let mut out = vec![0.0f64; count];
    baux_tokamak::evaluate_batch(scratch, count, &mut out);
    Ok((out, vec![0.0; count], 0))
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
pub(super) fn metal_seed(
    cfg: &Config,
    domain: &DomainFrame,
    rep: u32,
    ckpt_dir: Option<&Path>,
    worker: &mut Option<TokamakWorker>,
) -> Result<SeedRecord, String> {
    use crate::experiment::julia::metal::metal_job;
    use crate::experiment::julia::metal_step::MetalJobSpec;
    check_metal(&cfg.optimizer)?;
    if let Some(dir) = ckpt_dir
        && let Some(done) = load_done(dir, rep)
        && done.best_scores.len() == cfg.rounds
    {
        return Ok(done);
    }
    let seed = baux::tune_seed(&cfg.experiment, "proposal", rep, 0);
    let cfg_opt = baux::optimizer_config::OptimizerConfig::from_spec(&cfg.optimizer)
        .map_err(|e| e.to_string())?;
    let spec = MetalJobSpec::from_parts(
        rep,
        seed,
        domain.num_dim(),
        1,
        cfg.rounds,
        &cfg.optimizer,
        cfg_opt,
        None,
    )?;
    let (low, span) = low_span(domain);
    let mut scratch = Vec::new();
    let is_gym = cfg.experiment == "tokamak-tokagym";
    let loss_fn = |sc: f64| {
        if is_gym {
            (-sc).max(0.0)
        } else {
            (100.0 - sc).max(0.0)
        }
    };
    let (st, _) = metal_job(&spec, ckpt_dir, loss_fn, |rows| {
        eval_rows(worker, rows, &low, &span, &mut scratch)
    })?;
    let final_error = loss_fn(st.best_score);
    let reduction_pct = if st.initial_error > 0.0 {
        ((st.initial_error - final_error) / st.initial_error * 100.0).max(0.0)
    } else {
        0.0
    };
    eprintln!(
        "tokamak.metal.seed rep={}/{} seed={} final_score={:.6} final_error={:.6} reduction={:.2}% accepts={}/{} elapsed_s={:.2}",
        rep + 1,
        cfg.repetitions.max(1),
        seed,
        st.best_score,
        final_error,
        reduction_pct,
        st.total_accepts,
        cfg.rounds,
        st.elapsed_seconds
    );
    let rec = SeedRecord {
        rep,
        seed,
        num_dim: domain.num_dim(),
        initial_error: st.initial_error,
        final_score: st.best_score,
        final_error,
        reduction_pct,
        total_accepts: st.total_accepts,
        elapsed_seconds: st.elapsed_seconds,
        best_scores: st.best_scores,
    };
    if let Some(dir) = ckpt_dir {
        save_done(dir, &rec)?;
    }
    Ok(rec)
}

#[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
pub(super) fn metal_seed(
    _cfg: &Config,
    _domain: &DomainFrame,
    _rep: u32,
    _ckpt_dir: Option<&Path>,
    _worker: &mut Option<TokamakWorker>,
) -> Result<SeedRecord, String> {
    Err("resident-metal backend requires macOS Apple Silicon".into())
}
