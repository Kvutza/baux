//! Optimization driver, host execution, and run reporting for Julia
//! experiments.

use std::path::Path;
use std::time::Instant;

use baux::artifact::json::{Value, json};
use baux::optimizer::DomainFrame;
use baux::{BayesOptimizer, prng};
use ndarray::Array2;

use crate::experiment::julia::config::{Config, ExecMode, is_ocean};
use crate::experiment::julia::label::{model_label, opt_label};
use crate::experiment::julia::metal::{SeedRun, run_metal};
use crate::experiment::julia::space::{ensure_upstream, resolve_space};
use crate::experiment::julia::worker::JuliaWorker;

pub fn loss_metric(cfg: &Config, score: f64) -> f64 {
    if is_ocean(cfg) {
        (-score).max(0.0)
    } else {
        (cfg.fuse.penalty - score).max(0.0)
    }
}

pub fn mean_sem(vals: &[f64]) -> (f64, f64) {
    let n = vals.len();
    if n == 0 {
        return (0.0, 0.0);
    }
    let m = vals.iter().sum::<f64>() / (n as f64);
    if n == 1 {
        return (m, 0.0);
    }
    let var = vals.iter().map(|v| (v - m).powi(2)).sum::<f64>() / ((n - 1) as f64);
    (m, (var / (n as f64)).sqrt())
}

pub fn scalar_score(scores: &[f64], row: usize, n_obj: usize) -> f64 {
    let base = row * n_obj;
    scores[base..base + n_obj].iter().sum()
}

fn configure_manifold(
    cfg_opt: &mut baux::optimizer_config::OptimizerConfig,
    manifold: Option<&crate::experiment::julia::config::ManifoldCfg>,
) {
    let Some(mf) = manifold else {
        return;
    };
    if mf.noise_aware {
        cfg_opt.noise_aware = true;
    }
    let baux::optimizer_config::SurrogateConfig::ENN(ref mut enn) = cfg_opt.surrogate else {
        return;
    };
    if mf.scale_x {
        enn.scale_x = true;
    }
    if mf.metric_learning {
        enn.metric_learning = baux::layout::MetricLearning::Auto;
    }
    if let Some(rank) = mf.stiefel_rank {
        enn.stiefel_rank = Some(rank);
    }
    if let Some(alpha) = mf.calibrate_alpha {
        enn.calibrate_alpha = Some(alpha);
    }
}

fn configure_morbo(
    cfg_opt: &mut baux::optimizer_config::OptimizerConfig,
    n_obj: usize,
    morbo: Option<&crate::experiment::julia::config::MorboCfg>,
    noise: bool,
) {
    if n_obj < 2 {
        if noise {
            if let baux::trust_region::TrustRegionConfig::Morbo(ref mut ms) = cfg_opt.trust_region {
                ms.noise_aware = true;
            }
        }
        return;
    }
    let len = match cfg_opt.trust_region.clone() {
        baux::trust_region::TrustRegionConfig::Turbo(l) => l,
        baux::trust_region::TrustRegionConfig::Morbo(ms) => ms.length,
    };
    let (alpha, rescal) = match morbo {
        Some(m) => (m.alpha, m.rescalarize),
        None => (0.05, baux::Rescalarize::OnRestart),
    };
    cfg_opt.trust_region =
        baux::trust_region::TrustRegionConfig::Morbo(baux::trust_region::MorboTRSettings {
            num_metrics: n_obj,
            alpha,
            length: len,
            rescalarize: rescal,
            noise_aware: noise || cfg_opt.noise_aware,
        });
}

pub fn opt_cfg(
    cfg: &Config,
    n_obj: usize,
) -> Result<baux::optimizer_config::OptimizerConfig, String> {
    let mut cfg_opt = baux::optimizer_config::OptimizerConfig::from_spec(&cfg.optimizer)
        .map_err(|e| e.to_string())?;
    let noise = cfg.manifold.as_ref().is_some_and(|m| m.noise_aware);
    configure_manifold(&mut cfg_opt, cfg.manifold.as_ref());
    configure_morbo(&mut cfg_opt, n_obj, cfg.morbo.as_ref(), noise);
    Ok(cfg_opt)
}

fn step_round(
    unit: &Array2<f64>,
    num_dim: usize,
    _n_obj: usize,
    best_unit: &mut [f64],
    worker: &mut JuliaWorker,
) -> Result<(Vec<f64>, Vec<f64>, u64), String> {
    let (scores, viols, dt_us) = worker.eval_unit(&unit.view())?;
    for (i, _) in viols.iter().enumerate() {
        if let Some(sl) = unit.as_slice() {
            best_unit.copy_from_slice(&sl[i * num_dim..(i + 1) * num_dim]);
        }
    }
    Ok((scores, viols, dt_us))
}

fn init_optimizer(
    cfg: &Config,
    mode: ExecMode,
    domain: &DomainFrame,
    out_dir: &Path,
    rep: u32,
    _n_obj: usize,
    rng: &mut baux::Prng,
) -> Result<BayesOptimizer, String> {
    let cfg_opt = opt_cfg(cfg, _n_obj)?;
    if mode == ExecMode::HostDisk {
        let work = out_dir.join(format!("bpann_rep{rep}"));
        BayesOptimizer::with_disk(domain.clone(), cfg_opt, &cfg.optimizer, 0, work, rng)
            .map_err(|e| e.to_string())
    } else {
        BayesOptimizer::with_domain(domain.clone(), cfg_opt, &cfg.optimizer, 0, rng)
            .map_err(|e| e.to_string())
    }
}

pub fn run_host(
    cfg: &Config,
    mode: ExecMode,
    domain: &DomainFrame,
    out_dir: &Path,
    rep: u32,
    worker: &mut JuliaWorker,
) -> Result<SeedRun, String> {
    let seed = baux::tune_seed(&cfg.experiment, "proposal", rep, 0);
    let n_obj = worker.num_obj;
    let mut opt = init_optimizer(cfg, mode, domain, out_dir, rep, n_obj, &mut prng(seed))?;

    let num_dim = opt.num_dim();
    let arms = cfg
        .optimizer
        .candidate_generation()
        .map(|c| c.arms as usize)
        .unwrap_or(1)
        .max(1);
    let mut best_score = f64::NEG_INFINITY;
    let mut best_viol = f64::INFINITY;
    let mut best_unit = vec![0.5f64; num_dim];
    let mut init_err = 0.0;
    let mut accepts = 0usize;
    let mut worker_us = 0u64;
    let mut best_scores = Vec::with_capacity(cfg.rounds);
    let mut best_viols = Vec::with_capacity(cfg.rounds);
    let t0 = Instant::now();

    for round in 1..=cfg.rounds {
        let mut step_rng = crate::experiment::tokamak::label::round_rng(seed, round);
        let unit = opt.ask(arms, &mut step_rng).map_err(|e| e.to_string())?;
        let (scores, viols, dt_us) = step_round(&unit, num_dim, n_obj, &mut best_unit, worker)?;
        worker_us += dt_us;
        for (i, &vl) in viols.iter().enumerate() {
            let sc = scalar_score(&scores, i, n_obj);
            if sc > best_score {
                best_score = sc;
                best_viol = vl;
                accepts += 1;
            }
        }
        if round == 1 {
            init_err = loss_metric(cfg, best_score);
        }
        let y = Array2::from_shape_fn((arms, n_obj), |(i, m)| scores[i * n_obj + m]);
        opt.tell(&unit.view(), &y.view(), &mut step_rng)
            .map_err(|e| e.to_string())?;
        best_scores.push(best_score);
        best_viols.push(best_viol);
    }

    let (_, _, state_snap) = worker.eval_snap(&best_unit)?;
    let final_err = loss_metric(cfg, best_score);
    let reduction_pct = if init_err > 0.0 {
        ((init_err - final_err) / init_err * 100.0).max(0.0)
    } else {
        0.0
    };

    Ok(SeedRun {
        rep,
        seed,
        num_dim,
        init_err,
        final_score: best_score,
        final_err,
        final_viol: best_viol,
        reduction_pct,
        accepts,
        elapsed_s: t0.elapsed().as_secs_f64(),
        worker_s: (worker_us as f64) * 1.0e-6,
        best_scores,
        best_viols,
        state_snap,
    })
}

pub fn run_ppo(
    cfg: &Config,
    num_dim: usize,
    rep: u32,
    worker: &mut JuliaWorker,
) -> Result<SeedRun, String> {
    let seed = baux::tune_seed(&cfg.experiment, "ppo", rep, 0);
    let mut rng = prng(seed);
    let n_obj = worker.num_obj;
    let batch_cap = (cfg.rounds / 2).clamp(2, 16);
    let mut ppo = baux::tokamak::ppo::FlatPpo::new(num_dim, batch_cap)?;
    if !ppo.is_metal() {
        return Err("FlatPpo requires Apple Metal GPU device".into());
    }

    let mut row = Array2::<f64>::zeros((1, num_dim));
    let mut best_score = f64::NEG_INFINITY;
    let mut best_viol = f64::INFINITY;
    let mut best_unit = vec![0.5f64; num_dim];
    let mut init_err = 0.0;
    let mut accepts = 0usize;
    let mut worker_us = 0u64;
    let mut best_scores = Vec::with_capacity(cfg.rounds);
    let mut best_viols = Vec::with_capacity(cfg.rounds);
    let t0 = Instant::now();

    for round in 1..=cfg.rounds {
        let slice = row
            .as_slice_mut()
            .ok_or("ppo row buffer must be contiguous")?;
        let raw_u = ppo.sample_unit(false, &mut rng, slice)?;
        let (scores, viols, dt_us) = worker.eval_unit(&row.view())?;
        worker_us += dt_us;
        let sc = scalar_score(&scores, 0, n_obj);
        if sc > best_score {
            best_score = sc;
            best_viol = viols[0];
            accepts += 1;
            best_unit.copy_from_slice(row.as_slice().unwrap());
        }
        if round == 1 {
            init_err = loss_metric(cfg, best_score);
        }
        let _ = ppo.observe(&raw_u, sc)?;
        best_scores.push(best_score);
        best_viols.push(best_viol);
    }

    let (_, _, state_snap) = worker.eval_snap(&best_unit)?;
    let final_err = loss_metric(cfg, best_score);
    let reduction_pct = if init_err > 0.0 {
        ((init_err - final_err) / init_err * 100.0).max(0.0)
    } else {
        0.0
    };

    Ok(SeedRun {
        rep,
        seed,
        num_dim,
        init_err,
        final_score: best_score,
        final_err,
        final_viol: best_viol,
        reduction_pct,
        accepts,
        elapsed_s: t0.elapsed().as_secs_f64(),
        worker_s: (worker_us as f64) * 1.0e-6,
        best_scores,
        best_viols,
        state_snap,
    })
}

fn aggregate_traces(cfg: &Config, runs: &[SeedRun]) -> Vec<Value> {
    (0..cfg.rounds)
        .map(|idx| {
            let scores: Vec<f64> = runs.iter().map(|r| r.best_scores[idx]).collect();
            let errs: Vec<f64> = scores.iter().map(|&s| loss_metric(cfg, s)).collect();
            let viols: Vec<f64> = runs.iter().map(|r| r.best_viols[idx]).collect();
            let (ms, ss) = mean_sem(&scores);
            let (me, se) = mean_sem(&errs);
            let (mv, sv) = mean_sem(&viols);
            json!({
                "round": idx + 1,
                "best_score": ms,
                "mean_best_score": ms,
                "sem_best_score": ss,
                "tracking_error": me,
                "mean_tracking_error": me,
                "sem_tracking_error": se,
                "mean_violation": mv,
                "sem_violation": sv,
            })
        })
        .collect()
}

fn build_summary(
    cfg: &Config,
    mode: ExecMode,
    num_dim: usize,
    runs: &[SeedRun],
    traces: Vec<Value>,
    merkle_root: &str,
    upstream_rev: &str,
) -> Value {
    let final_scores: Vec<f64> = runs.iter().map(|r| r.final_score).collect();
    let final_errs: Vec<f64> = runs.iter().map(|r| r.final_err).collect();
    let final_viols: Vec<f64> = runs.iter().map(|r| r.final_viol).collect();
    let reds: Vec<f64> = runs.iter().map(|r| r.reduction_pct).collect();
    let (sc_m, sc_s) = mean_sem(&final_scores);
    let (err_m, err_s) = mean_sem(&final_errs);
    let (vl_m, vl_s) = mean_sem(&final_viols);
    let (red_m, red_s) = mean_sem(&reds);
    let total_s: f64 = runs.iter().map(|r| r.elapsed_s).sum();
    let worker_s: f64 = runs.iter().map(|r| r.worker_s).sum();
    let seed_runs: Vec<Value> = runs
        .iter()
        .map(|r| {
            json!({
                "rep": r.rep,
                "seed": r.seed,
                "initial_tracking_error": r.init_err,
                "final_score": r.final_score,
                "final_tracking_error": r.final_err,
                "tracking_error_reduction_pct": r.reduction_pct,
                "total_accepts": r.accepts,
                "elapsed_seconds": r.elapsed_s,
                "best_scores": r.best_scores,
            })
        })
        .collect();

    json!({
        "model": model_label(cfg),
        "optimizer": opt_label(cfg, mode),
        "optimizer_label": crate::experiment::tokamak::label::opt_label(&cfg.optimizer),
        "backend_mode": opt_label(cfg, mode),
        "mode": format!("{mode:?}"),
        "experiment": cfg.experiment,
        "objectives": crate::experiment::julia::config::obj_count(cfg),
        "parameters": num_dim,
        "disk_backed": mode == ExecMode::HostDisk,
        "rounds": cfg.rounds,
        "repetitions": runs.len(),
        "final_score_mean": sc_m,
        "final_score_sem": sc_s,
        "final_tracking_error_mean": err_m,
        "final_tracking_error_sem": err_s,
        "final_violation_mean": vl_m,
        "final_violation_sem": vl_s,
        "tracking_error_reduction_pct_mean": red_m,
        "tracking_error_reduction_pct_sem": red_s,
        "elapsed_seconds_total": total_s,
        "worker_seconds_total": worker_s,
        "upstream_lock_merkle_root": merkle_root,
        "upstream_revision": upstream_rev,
        "traces": traces,
        "seed_runs": seed_runs,
    })
}

pub fn run_opt(cfg: &Config, mode: ExecMode, root: &Path, out_dir: &Path) -> Result<Value, String> {
    let lock_name = if is_ocean(cfg) {
        "oceananigans"
    } else {
        "fuse"
    };
    let lock_val = ensure_upstream(root, lock_name)?;
    let merkle_root = lock_val["merkle"]["merkle_root_blake3"]
        .as_str()
        .unwrap_or("")
        .to_owned();
    let upstream_rev = lock_val["repository"]["revision"]
        .as_str()
        .unwrap_or("")
        .to_owned();

    let (domain, low, span, state_len) = resolve_space(root, cfg)?;
    let num_dim = domain.num_dim();
    let arms = if mode == ExecMode::HostRam || mode == ExecMode::HostDisk {
        cfg.optimizer
            .candidate_generation()
            .map(|c| c.arms as usize)
            .unwrap_or(1)
            .max(1)
    } else {
        1
    };
    let ckpt_dir = crate::experiment::tokamak::ckpt::ckpt_dir(out_dir);
    let reps = crate::experiment::tokamak::ckpt::rep_indices(cfg.repetitions)?;
    let mut worker = JuliaWorker::spawn(root, cfg, arms, low, span, state_len)?;

    let mut runs = Vec::with_capacity(reps.len());
    for rep in reps {
        let sr = match mode {
            ExecMode::HostRam | ExecMode::HostDisk => {
                run_host(cfg, mode, &domain, out_dir, rep, &mut worker)?
            }
            ExecMode::ResidentMetal => run_metal(cfg, num_dim, rep, Some(&ckpt_dir), &mut worker)?,
            ExecMode::MetalPpo => run_ppo(cfg, num_dim, rep, &mut worker)?,
        };
        runs.push(sr);
    }
    let traces = aggregate_traces(cfg, &runs);
    Ok(build_summary(
        cfg,
        mode,
        num_dim,
        &runs,
        traces,
        &merkle_root,
        &upstream_rev,
    ))
}
