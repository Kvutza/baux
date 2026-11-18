//! Unified half-precision (`eval_half` / `eval_halves`) worker job runner and
//! objective policy helpers for Julia physics experiments.

use std::path::Path;
use std::time::Instant;

use baux::config::{AcquisitionSpec, ObjectivePolicy, OptimizerSpec, ResidentEnnConfig};
pub use baux::unified_opt::encode_half;

use crate::experiment::julia::config::{Config, is_ocean};
use crate::experiment::julia::metal_step::{MetalFiber, MetalJobSpec};
use crate::experiment::julia::worker::JuliaWorker;
use crate::experiment::tokamak::ckpt;

pub struct SeedRun {
    pub rep: u32,
    pub seed: u64,
    pub num_dim: usize,
    pub init_err: f64,
    pub final_score: f64,
    pub final_err: f64,
    pub final_viol: f64,
    pub reduction_pct: f64,
    pub accepts: usize,
    pub elapsed_s: f64,
    pub worker_s: f64,
    pub best_scores: Vec<f64>,
    pub best_viols: Vec<f64>,
    pub state_snap: Vec<f32>,
}

#[inline]
pub(crate) fn round_seed(seed: u64, round: usize) -> u64 {
    let tag = (round as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15);
    baux::semantics::random::splitmix64(seed ^ tag)
}

/// Generate `count` stratified Latin Hypercube Design (`LHD`) rows in `[0,
/// 1]^D` encoded as IEEE FP16 words via `DelayedPool::Lhd`.
pub fn lhd_batch(seed: u64, count: usize, num_dim: usize) -> Vec<Vec<u16>> {
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    {
        return baux::resident_metal::engine::lhd_rows(seed, count, num_dim);
    }
    #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
    {
        let n = count.max(1);
        let d = num_dim.max(1);
        let lo = vec![0.0f32; d];
        let hi = vec![1.0f32; d];
        let pool = baux::semantics::pool::DelayedPool::Lhd {
            key: baux::semantics::key::Key::new(seed),
            lower: &lo,
            upper: &hi,
            num_candidates: n,
        };
        let mut coords = vec![0.0f32; d];
        let mut out = Vec::with_capacity(n);
        for cand in 0..n {
            pool.write_row(cand, &mut coords);
            out.push(
                coords[..num_dim]
                    .iter()
                    .map(|&u| encode_half(u.clamp(0.0, 1.0)))
                    .collect(),
            );
        }
        out
    }
}

pub fn res_enn(
    cfg: &Config,
    cfg_opt: &baux::optimizer_config::OptimizerConfig,
    seed: u64,
) -> Result<ResidentEnnConfig, String> {
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    {
        let dim = if is_ocean(cfg) {
            let ny = cfg.ocean.ny.max(1);
            if cfg.ocean.task == "wind-mixing" {
                2 * cfg.ocean.nx * ny
            } else {
                cfg.ocean.nx * ny * cfg.ocean.nz * cfg.ocean.actuators
            }
        } else {
            1
        };
        let arms = cfg
            .optimizer
            .candidate_generation()
            .map_or(1, |cg| cg.arms.max(1));
        let single = matches!(cfg.optimizer, OptimizerSpec::SingleStateBo { .. });
        return baux::resident_metal::engine::make_enn(cfg_opt, dim.max(1), arms, single, seed)
            .map_err(|e| e.to_string());
    }
    #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
    {
        let _ = (cfg, cfg_opt, seed);
        Err("resident Metal backend requires macOS + metal feature".into())
    }
}

pub(crate) fn objective_policy(cfg: &Config, n_obj: usize, seed: u64) -> Option<ObjectivePolicy> {
    if n_obj < 2 {
        return None;
    }
    if let Some(ref m) = cfg.morbo {
        let scales = if m.scales.len() == n_obj {
            m.scales.clone()
        } else {
            vec![1.0f32; n_obj]
        };
        let weights = if m.preferences.len() == n_obj {
            m.preferences.clone()
        } else {
            vec![1.0f32; n_obj]
        };
        return Some(ObjectivePolicy {
            acquisition: baux::config::ObjectiveAcquisition::Morbo {
                weights,
                regions: m.regions.max(1),
                alpha: m.alpha as f32,
                seed,
                rescalarize: m.rescalarize,
                clip: true,
            },
            scales,
        });
    }
    let scales = match &cfg.optimizer {
        OptimizerSpec::BayesianSearch {
            acquisition: AcquisitionSpec::Pareto { scales },
            ..
        } if scales.len() == n_obj => scales.clone(),
        _ => vec![1.0f32; n_obj],
    };
    Some(ObjectivePolicy {
        acquisition: baux::config::ObjectiveAcquisition::Pareto,
        scales,
    })
}

pub(crate) fn metal_job<F, L>(
    spec: &MetalJobSpec,
    ckpt_dir: Option<&Path>,
    loss_fn: L,
    mut eval_fn: F,
) -> Result<(ckpt::LiveCkpt, Vec<u16>), String>
where
    F: FnMut(&[Vec<u16>]) -> Result<(Vec<f64>, Vec<f64>, u64), String>,
    L: Fn(f64) -> f64,
{
    let saved = match ckpt_dir {
        Some(dir) => ckpt::load_meta(dir, spec.rep)?,
        None => None,
    };
    let saved_n = saved
        .as_ref()
        .map_or(0, |s| s.completed_round.min(spec.rounds));
    let mut st = ckpt::LiveCkpt::empty(spec.rounds);
    st.elapsed_seconds = saved.as_ref().map_or(0.0, |s| s.elapsed_seconds);
    st.worker_seconds = saved.as_ref().map_or(0.0, |s| s.worker_seconds);
    let mut fiber = MetalFiber::new(spec.clone());
    let every = ckpt::ckpt_interval();
    let mut chunk_t0 = Instant::now();

    for round in 1..=spec.rounds {
        if round == saved_n + 1 {
            chunk_t0 = Instant::now();
        }
        let rows = fiber.prepare_batch()?;
        let (scores, viols, dt_us) = if let Some(ref sv) = saved
            && round <= saved_n
        {
            let vl = sv.eval_viols.get(round - 1).copied().unwrap_or(0.0);
            (sv.eval_scores[round - 1].clone(), vec![vl; rows.len()], 0)
        } else {
            eval_fn(&rows)?
        };
        let rec = fiber.commit_batch(&scores, &viols, dt_us, &loss_fn)?;
        st.worker_seconds += (dt_us as f64) * 1.0e-6;
        st.best_score = rec.best_score;
        st.initial_error = fiber.initial_error;
        st.total_accepts = fiber.total_accepts as usize;
        st.best_scores.push(rec.best_score);
        st.best_viols.push(rec.best_viol);
        st.eval_scores.push(scores);
        st.eval_viols.push(rec.eval_viol);
        st.completed_round = round;
        if let Some(dir) = ckpt_dir
            && round > saved_n
            && round % every == 0
            && round < spec.rounds
        {
            st.elapsed_seconds += chunk_t0.elapsed().as_secs_f64();
            chunk_t0 = Instant::now();
            ckpt::save_meta(dir, spec.rep, &st)?;
        }
    }
    if spec.rounds > saved_n {
        st.elapsed_seconds += chunk_t0.elapsed().as_secs_f64();
    }
    Ok((st, fiber.best_row))
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
pub fn run_metal(
    cfg: &Config,
    num_dim: usize,
    rep: u32,
    ckpt_dir: Option<&Path>,
    worker: &mut JuliaWorker,
) -> Result<SeedRun, String> {
    let domain = crate::experiment::julia::metal_step::unit_domain(num_dim)?;
    crate::experiment::julia::driver::run_bo(
        cfg,
        crate::experiment::julia::config::ExecMode::ResidentMetal,
        &domain,
        None,
        ckpt_dir,
        rep,
        worker,
    )
}

#[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
pub fn run_metal(
    _cfg: &Config,
    _num_dim: usize,
    _rep: u32,
    _ckpt_dir: Option<&Path>,
    _worker: &mut JuliaWorker,
) -> Result<SeedRun, String> {
    Err("resident Metal backend requires macOS + metal feature".into())
}
