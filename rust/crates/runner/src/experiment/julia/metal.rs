//! Resident Apple Silicon Metal GPU execution for closed-loop physics & policy
//! search.

use std::path::Path;
use std::time::Instant;

use baux::config::{
    AcquisitionSpec, ObjectivePolicy, ObservationSharing, OptimizerSpec, RegionAllocation,
    RegionController, RegionMemory, RegionalSearchSpec, ResidentEnnConfig, RestartCenter,
};
use baux::proposal_distribution::ProposalDistribution;

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
use baux::resident_metal::{
    ParamBlock, RegionalProposal, RegionalSearch, ResidentManifold, ResidentSampler, SearchState,
};

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

pub(crate) fn make_enn(
    spec: &OptimizerSpec,
    cfg_opt: &baux::optimizer_config::OptimizerConfig,
    num_dim: usize,
    seed: u64,
) -> Result<ResidentEnnConfig, String> {
    let arms = spec.candidate_generation().map_or(1, |cg| cg.arms.max(1));
    let total = if matches!(spec, OptimizerSpec::SingleStateBo { .. }) {
        1usize
    } else {
        cfg_opt
            .candidates
            .num_candidates(num_dim.max(1), arms as usize)
            .max(arms as usize)
    };
    let slots = total.div_ceil(arms as usize).max(1) as u32;
    let pool = baux::procedural_pool::ProceduralPoolConfig {
        arms,
        candidates_per_arm: slots,
    }
    .resolve_resident()?;
    let (kind, beta) = match cfg_opt.acquisition {
        baux::optimizer_config::AcquisitionConfig::Random => {
            (baux::weights::AcquisitionKind::Random, 0.0)
        }
        baux::optimizer_config::AcquisitionConfig::UCB { beta } => {
            (baux::weights::AcquisitionKind::Ucb, beta as f32)
        }
        baux::optimizer_config::AcquisitionConfig::Thompson => {
            (baux::weights::AcquisitionKind::Thompson, 1.0)
        }
        baux::optimizer_config::AcquisitionConfig::Pareto => {
            (baux::weights::AcquisitionKind::Pareto, 1.0)
        }
    };
    let mut ask = baux::trials::RoundConfig::default();
    let tr_init = match &cfg_opt.trust_region {
        baux::trust_region::TrustRegionConfig::Turbo(l) => l.length_init,
        baux::trust_region::TrustRegionConfig::Morbo(m) => m.length.length_init,
    };
    ask.proposal.radius = tr_init as f32;
    ask.acquisition = baux::trials::AcquisitionParams { kind, beta, seed };
    let baux::optimizer_config::SurrogateConfig::ENN(ref enn) = cfg_opt.surrogate else {
        ask.surrogate.neighbors = 1;
        return Ok(ResidentEnnConfig {
            pool,
            ask,
            num_candidates: 1,
            num_samples: 1,
            fit_params: false,
            adapt_neighbors: false,
            distance_metric: baux::config::Dissimilarity::Euclidean,
            local_scale_neighbors: 1,
        });
    };
    ask.surrogate.neighbors = usize::try_from(enn.k.max(1)).unwrap_or(10);
    ask.surrogate.epistemic_scale = enn.epistemic_scale as f32;
    ask.surrogate.aleatoric_scale = enn.aleatoric_scale as f32;
    ask.surrogate.output_scale = enn.y_scale.unwrap_or(1.0) as f32;
    let (distance_metric, local_scale_neighbors) = match enn.geometry {
        baux::semantics::neighbor::DistanceMetric::Euclidean => {
            (baux::config::Dissimilarity::Euclidean, 8)
        }
        baux::semantics::neighbor::DistanceMetric::Conformal { local_k } => {
            (baux::config::Dissimilarity::Conformal, local_k)
        }
        baux::semantics::neighbor::DistanceMetric::Hyperbolic { local_k } => {
            (baux::config::Dissimilarity::Hyperbolic, local_k)
        }
        baux::semantics::neighbor::DistanceMetric::Wasserstein { local_k } => {
            (baux::config::Dissimilarity::Wasserstein, local_k)
        }
    };
    Ok(ResidentEnnConfig {
        pool,
        ask,
        num_candidates: enn.num_candidates.max(1),
        num_samples: enn.num_samples.max(1),
        fit_params: enn.num_candidates > 0 && enn.num_samples > 0,
        adapt_neighbors: enn.adapt_neighbors,
        distance_metric,
        local_scale_neighbors,
    })
}

pub fn res_enn(
    cfg: &Config,
    cfg_opt: &baux::optimizer_config::OptimizerConfig,
    seed: u64,
) -> Result<ResidentEnnConfig, String> {
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
    make_enn(&cfg.optimizer, cfg_opt, dim.max(1), seed)
}

#[inline]
pub fn encode_half(val: f32) -> u16 {
    let bits = val.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xff) as i32 - 127 + 15;
    let mant = (bits >> 13) & 0x03ff;
    if exp <= 0 {
        sign
    } else if exp >= 31 {
        sign | 0x7c00
    } else {
        sign | ((exp as u16) << 10) | (mant as u16)
    }
}

#[inline]
pub(crate) fn round_seed(seed: u64, round: usize) -> u64 {
    let tag = (round as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15);
    baux::semantics::random::splitmix64(seed ^ tag)
}

#[inline]
pub(crate) fn restart_seed(seed: u64, slot: usize, gen_seq: u64) -> u64 {
    baux::semantics::random::splitmix64(
        seed ^ ((slot as u64 + 1).wrapping_mul(0x9e37_79b9_7f4a_7c15))
            ^ gen_seq.wrapping_mul(0xbf58_476d_1ce4_e5b9),
    )
}

/// Generate `count` stratified Latin Hypercube Design (`LHD`) rows in `[0,
/// 1]^D` encoded as IEEE FP16 words via
/// `baux::semantics::pool::DelayedPool::Lhd`.
pub fn lhd_batch(seed: u64, count: usize, num_dim: usize) -> Vec<Vec<u16>> {
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

pub(crate) fn regional_spec(
    spec: &OptimizerSpec,
    num_dim: usize,
    arms: usize,
) -> RegionalSearchSpec {
    if let Some(rs) = spec.regional_search() {
        return rs.clone();
    }
    let tol = i32::try_from(baux::semantics::optimizer::turbo_tolerance(
        num_dim.max(1) as f64,
        arms.max(1),
    ))
    .unwrap_or(i32::MAX);
    RegionalSearchSpec {
        count: 1,
        sharing: ObservationSharing::OriginOnly,
        allocation: RegionAllocation::LeastTrials,
        controller: RegionController::Turbo,
        acquisitions: Vec::new(),
        restart_memory: RegionMemory::Fresh,
        restart_candidates: 1,
        restart_center: RestartCenter::First,
        success_tolerance: 3,
        failure_tolerance: tol,
    }
}

pub(crate) fn resolve_kernel(spec: &OptimizerSpec) -> ProposalDistribution {
    match spec {
        OptimizerSpec::SingleStateBo { distribution, .. }
        | OptimizerSpec::BayesianSearch {
            distribution: Some(distribution),
            ..
        } => *distribution,
        _ => ProposalDistribution::Gaussian,
    }
}

pub(crate) fn explicit_dist(spec: &OptimizerSpec) -> bool {
    matches!(
        spec,
        OptimizerSpec::BayesianSearch {
            distribution: Some(_),
            perturbation: None,
            ..
        }
    )
}

pub(crate) fn is_single(spec: &OptimizerSpec) -> bool {
    matches!(spec, OptimizerSpec::SingleStateBo { .. })
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

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
pub fn obj_obs(
    scores: &[f64],
    n_obj: usize,
) -> Result<baux::objective_observation::ObjectiveObservation, String> {
    let items: Vec<baux::objective_observation::ObjectiveEstimate> = scores[..n_obj]
        .iter()
        .map(|&v| baux::objective_observation::ObjectiveEstimate {
            mean: v as f32,
            variance: 0.0,
        })
        .collect();
    let mean = items.iter().map(|e| e.mean).sum::<f32>() / (items.len() as f32);
    let ctrl = baux::objective_observation::ObjectiveEstimate {
        mean,
        variance: 0.0,
    };
    baux::objective_observation::ObjectiveObservation::new(&items, ctrl)
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
fn build_manifold(cfg_opt: &baux::optimizer_config::OptimizerConfig) -> Option<ResidentManifold> {
    let coord_system = cfg_opt.candidates.coord_system;
    let noise_aware = cfg_opt.noise_aware;
    let (
        scale_x,
        metric_learning,
        tied_dims,
        unscaled_dims,
        stiefel_rank,
        calibrate_alpha,
        y_bounds,
    ) = match &cfg_opt.surrogate {
        baux::optimizer_config::SurrogateConfig::ENN(enn) => {
            let tied = if !enn.tied_dims.is_empty() {
                enn.tied_dims.clone()
            } else {
                cfg_opt.candidates.tied_dims.clone()
            };
            let yb = enn.y_bounds.as_ref().map(|arr| {
                (0..arr.nrows())
                    .map(|r| (arr[[r, 0]], arr[[r, 1]]))
                    .collect()
            });
            (
                enn.scale_x,
                enn.metric_learning,
                tied,
                enn.unscaled_dims.clone(),
                enn.stiefel_rank,
                enn.calibrate_alpha,
                yb,
            )
        }
        baux::optimizer_config::SurrogateConfig::None => (
            false,
            baux::layout::MetricLearning::None,
            cfg_opt.candidates.tied_dims.clone(),
            Vec::new(),
            None,
            None,
            None,
        ),
    };
    let active = scale_x
        || metric_learning != baux::layout::MetricLearning::None
        || !tied_dims.is_empty()
        || !unscaled_dims.is_empty()
        || coord_system != baux::proposal_distribution::CoordSystem::Cartesian
        || stiefel_rank.is_some()
        || calibrate_alpha.is_some()
        || y_bounds.is_some()
        || noise_aware;
    if !active {
        return None;
    }
    Some(ResidentManifold {
        scale_x,
        metric_learning,
        tied_dims,
        unscaled_dims,
        categories: Vec::new(),
        coord_system,
        stiefel_rank,
        calibrate_alpha,
        y_bounds,
        noise_aware,
        no_trust_region: false,
    })
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
pub(crate) fn make_region(
    spec: &MetalJobSpec,
    base: &[u16],
    slot: usize,
    gen_seq: u64,
    scores: &[f64],
) -> Result<SearchState, String> {
    let r_seed = restart_seed(spec.seed, slot, gen_seq);
    let weight = 1.0f32 / (spec.num_dim.max(1) as f32);
    let block = ParamBlock::new(r_seed, 0, spec.num_dim, 0.5, weight)?;
    let mut st = if spec.has_surrogate {
        let s = SearchState::new_implicit(base, vec![block], 128, spec.tr_cfg, spec.kernel)?;
        s.configure_enn(spec.enn_res)?
    } else {
        let s = SearchState::new_fp16(base, vec![block], 0, spec.tr_cfg, spec.kernel)?;
        s.configure_pool(spec.enn_res.pool)?
    };
    if !spec.single_state {
        if spec.explicit_dist {
            st =
                st.configure_sampler(ResidentSampler::from_distribution(spec.kernel), None, true)?;
        } else {
            let is_morbo = matches!(
                spec.cfg_opt.trust_region,
                baux::trust_region::TrustRegionConfig::Morbo(_)
            ) || matches!(
                spec.policy,
                Some(ObjectivePolicy {
                    acquisition: baux::config::ObjectiveAcquisition::Morbo { .. },
                    ..
                })
            );
            let sampler = match spec.cfg_opt.candidates.candidate_rv {
                baux::candidates::CandidateSampler::RAASP => ResidentSampler::Raasp,
                baux::candidates::CandidateSampler::Uniform => ResidentSampler::Uniform,
                baux::candidates::CandidateSampler::Sobol if is_morbo => {
                    ResidentSampler::SobolMasked
                }
                baux::candidates::CandidateSampler::Sobol => ResidentSampler::Sobol,
            };
            st = st.configure_sampler(sampler, Some(spec.cfg_opt.candidates.num_pert), true)?;
        }
        st = st.configure_lhd(spec.enn_res.pool.count() as usize)?;
    }
    if let Some(mf) = build_manifold(&spec.cfg_opt) {
        st = st.configure_manifold(mf)?;
    }
    st = if spec.n_obj >= 2 {
        st.observe_objectives(obj_obs(scores, spec.n_obj)?)?
    } else {
        st.observe_initial(scores[0] as f32, 0.0)?
    };
    if let Some(ref p) = spec.policy {
        st = st.configure_objectives(Some(p.clone()))?;
    }
    let successes = usize::try_from(spec.reg_spec.success_tolerance)
        .ok()
        .filter(|&v| v > 0)
        .ok_or("regional success tolerance must be positive")?;
    let failures = match usize::try_from(spec.reg_spec.failure_tolerance).ok() {
        Some(f) if f > 0 => f,
        _ => SearchState::failure_tolerance(spec.num_dim, spec.arms()),
    };
    let fixed = spec.reg_spec.controller == RegionController::FixedRadius;
    st.set_tolerance(failures)?
        .set_successes(successes)?
        .fix_radius(fixed)
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
pub(crate) fn ask_retry(
    search: RegionalSearch,
    r_seed: u64,
    mut r_ask: baux::trials::RoundConfig,
) -> Result<(RegionalSearch, RegionalProposal), String> {
    let mut last_err = String::new();
    let current = search;
    for attempt in 0..4u64 {
        let s = r_seed.wrapping_add(attempt.wrapping_mul(0x9e37_79b9_7f4a_7c15));
        r_ask.acquisition.seed = s;
        match current.clone().ask(s, r_ask) {
            Ok((next_s, p)) => return Ok((next_s, p)),
            Err(e) => last_err = e,
        }
    }
    Err(last_err)
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
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
    let seed = baux::tune_seed(&cfg.experiment, "proposal", rep, 0);
    let n_obj = worker.num_obj;
    let cfg_opt = crate::experiment::julia::driver::opt_cfg(cfg, n_obj)?;
    let policy = objective_policy(cfg, n_obj, seed);
    let spec = MetalJobSpec::from_parts(
        rep,
        seed,
        num_dim,
        n_obj,
        cfg.rounds,
        &cfg.optimizer,
        cfg_opt,
        policy,
    )?;
    let (st, best_row) = metal_job(
        &spec,
        ckpt_dir,
        |sc| crate::experiment::julia::driver::loss_metric(cfg, sc),
        |rows| worker.eval_halves(rows),
    )?;
    let state_snap = worker.half_snap(best_row.as_ptr())?;
    let final_err = crate::experiment::julia::driver::loss_metric(cfg, st.best_score);
    let final_viol = st.best_viols.last().copied().unwrap_or(0.0);
    let reduction_pct = if st.initial_error > 0.0 {
        ((st.initial_error - final_err) / st.initial_error * 100.0).max(0.0)
    } else {
        0.0
    };
    if let Some(dir) = ckpt_dir {
        let done = ckpt::SeedRecord {
            rep,
            seed,
            num_dim,
            initial_error: st.initial_error,
            final_score: st.best_score,
            final_error: final_err,
            reduction_pct,
            total_accepts: st.total_accepts,
            elapsed_seconds: st.elapsed_seconds,
            best_scores: st.best_scores.clone(),
        };
        ckpt::save_done(dir, &done)?;
    }
    Ok(SeedRun {
        rep,
        seed,
        num_dim,
        init_err: st.initial_error,
        final_score: st.best_score,
        final_err,
        final_viol,
        reduction_pct,
        accepts: st.total_accepts,
        elapsed_s: st.elapsed_seconds,
        worker_s: st.worker_seconds,
        best_scores: st.best_scores,
        best_viols: st.best_viols,
        state_snap,
    })
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
