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
        OptimizerSpec::OperatorBo {
            perturbation: Some(_),
            ..
        } => Err("resident-metal does not accept host [optimizer.perturbation] overrides".into()),
        OptimizerSpec::OperatorBo { .. } => Ok(()),
        OptimizerSpec::BayesianSearch {
            candidates,
            arms,
            perturbation,
            ..
        } => {
            if perturbation.is_some() {
                return Err(
                    "resident-metal does not accept host [optimizer.perturbation] overrides".into(),
                );
            }
            if *arms != 1 || *candidates == 0 || *candidates > 4 {
                return Err("resident-metal tokamak requires arms = 1 and 1..=4 candidates".into());
            }
            Ok(())
        }
        _ => Err("resident-metal backend requires operator-bo or bayesian-search".into()),
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
fn eval_half(
    worker: &mut Option<TokamakWorker>,
    half_ptr: *const u16,
    low: &[f64],
    span: &[f64],
    scratch: &mut [f64],
) -> Result<(Vec<f64>, f64, u64), String> {
    if let Some(w) = worker.as_mut() {
        let (sc, dt_us) = w.eval_half(half_ptr, low, span)?;
        return Ok((vec![sc], 0.0, dt_us));
    }
    let raw = unsafe { std::slice::from_raw_parts(half_ptr, low.len()) };
    for j in 0..low.len() {
        let u = f64::from(baux::shm_slab::decode_half(raw[j])).clamp(0.0, 1.0);
        scratch[j] = low[j] + u * span[j];
    }
    let mut out = vec![0.0f64; 1];
    baux::tokamak::evaluate_batch(scratch, 1, &mut out);
    Ok((out, 0.0, 0))
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
pub(super) fn metal_seed(
    cfg: &Config,
    domain: &DomainFrame,
    rep: u32,
    ckpt_dir: Option<&Path>,
    worker: &mut Option<TokamakWorker>,
) -> Result<SeedRecord, String> {
    use crate::experiment::julia::metal::{
        MetalJob, make_enn, metal_job, regional_spec, resolve_kernel,
    };
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
    let enn_res = make_enn(&cfg.optimizer, &cfg_opt, seed)?;
    let tr_cfg = match &cfg_opt.trust_region {
        baux::trust_region::TrustRegionConfig::Turbo(l) => *l,
        baux::trust_region::TrustRegionConfig::Morbo(m) => m.length,
    };
    let has_surrogate = matches!(
        cfg_opt.surrogate,
        baux::optimizer_config::SurrogateConfig::ENN(_)
    );
    let (low, span) = low_span(domain);
    let mut scratch = vec![0.0f64; domain.num_dim()];
    let is_gym = cfg.experiment == "tokamak-tokagym";
    let loss_fn = |sc: f64| {
        if is_gym {
            (-sc).max(0.0)
        } else {
            (100.0 - sc).max(0.0)
        }
    };
    let job = MetalJob {
        rep,
        seed,
        num_dim: domain.num_dim(),
        n_obj: 1,
        rounds: cfg.rounds,
        has_surrogate,
        tr_cfg,
        kernel: resolve_kernel(&cfg.optimizer),
        reg_spec: regional_spec(&cfg.optimizer),
        enn_res,
        policy: None,
        ckpt_dir,
    };
    let (st, _) = metal_job(&job, loss_fn, |ptr| {
        eval_half(worker, ptr, &low, &span, &mut scratch)
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
