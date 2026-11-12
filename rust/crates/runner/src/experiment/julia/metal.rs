//! Resident Apple Silicon Metal GPU execution for closed-loop physics & policy
//! search.

use std::collections::VecDeque;
use std::path::Path;
use std::time::Instant;

use baux::config::{ObjectivePolicy, OptimizerSpec, RegionalSearchSpec, ResidentEnnConfig};
use baux::proposal_kernel::ProposalKernel;
use baux::resident_metal::{ParamBlock, RegionalProposal, RegionalSearch, SearchState};

use crate::experiment::julia::config::Config;
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
    seed: u64,
) -> Result<ResidentEnnConfig, String> {
    let (arms, cands) = spec
        .candidate_generation()
        .map_or((1, 4), |cg| (cg.arms, cg.candidates));
    let pool = baux::procedural_pool::ProceduralPoolConfig {
        arms,
        candidates_per_arm: cands,
    }
    .resolve_resident()?;
    let (kind, beta) = match cfg_opt.acquisition {
        baux::optimizer_config::AcquisitionConfig::Random => {
            (baux::weights::AcquisitionKind::Random, 0.0)
        }
        baux::optimizer_config::AcquisitionConfig::UCB { beta } => {
            (baux::weights::AcquisitionKind::Ucb, beta as f32)
        }
        baux::optimizer_config::AcquisitionConfig::Thompson
        | baux::optimizer_config::AcquisitionConfig::Pareto => {
            (baux::weights::AcquisitionKind::Thompson, 0.0)
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
            distance_metric: baux::config::DistanceMetric::Euclidean,
            local_scale_neighbors: 1,
        });
    };
    ask.surrogate.neighbors = usize::try_from(enn.k.max(1)).unwrap_or(10);
    ask.surrogate.epistemic_scale = enn.epistemic_scale as f32;
    ask.surrogate.aleatoric_scale = enn.aleatoric_scale as f32;
    ask.surrogate.output_scale = enn.y_scale.unwrap_or(1.0) as f32;
    let (distance_metric, local_scale_neighbors) = match enn.geometry {
        baux::semantics::neighbor::DistanceMetric::Euclidean => {
            (baux::config::DistanceMetric::Euclidean, 10)
        }
        baux::semantics::neighbor::DistanceMetric::Conformal { local_k } => {
            (baux::config::DistanceMetric::Conformal, local_k)
        }
        baux::semantics::neighbor::DistanceMetric::Hyperbolic { local_k } => {
            (baux::config::DistanceMetric::Hyperbolic, local_k)
        }
        baux::semantics::neighbor::DistanceMetric::Wasserstein { local_k } => {
            (baux::config::DistanceMetric::Wasserstein, local_k)
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
    make_enn(&cfg.optimizer, cfg_opt, seed)
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
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

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
pub fn unit_center(seed: u64, slot: usize, gen_seq: u64, num_dim: usize) -> Vec<u16> {
    let root = if slot == 0 && gen_seq == 0 {
        seed ^ 0x5eed_0000_ba00_0001
    } else {
        seed ^ ((slot as u64 + 1).wrapping_mul(0x9e37_79b9_7f4a_7c15))
            ^ (gen_seq.wrapping_mul(0xbf58_476d_1ce4_e5b9))
    };
    (0..num_dim)
        .map(|i| {
            let h = baux::hash::splitmix64(
                root.wrapping_add((i as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15)),
            );
            let u = ((h >> 40) as f32) * (1.0 / ((1u32 << 24) as f32));
            encode_half(u)
        })
        .collect()
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
pub(crate) fn regional_spec(spec: &OptimizerSpec) -> RegionalSearchSpec {
    spec.regional_search()
        .cloned()
        .unwrap_or_else(|| RegionalSearchSpec {
            count: 1,
            sharing: baux::config::ObservationSharing::OriginOnly,
            allocation: baux::config::RegionAllocation::LeastTrials,
            controller: baux::config::RegionController::Turbo,
            acquisitions: Vec::new(),
            restart_memory: baux::config::RegionMemory::Fresh,
            restart_candidates: 1,
            restart_center: baux::config::RestartCenter::First,
            success_tolerance: 3,
            failure_tolerance: 3,
        })
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
pub(crate) fn resolve_kernel(spec: &OptimizerSpec) -> ProposalKernel {
    match spec {
        OptimizerSpec::OperatorBo { distribution, .. }
        | OptimizerSpec::BayesianSearch {
            distribution: Some(distribution),
            ..
        } => *distribution,
        _ => ProposalKernel::Gaussian,
    }
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
pub(crate) struct MetalJob<'a> {
    pub rep: u32,
    pub seed: u64,
    pub num_dim: usize,
    pub n_obj: usize,
    pub rounds: usize,
    pub has_surrogate: bool,
    pub tr_cfg: baux::trust_region::TRLengthConfig,
    pub kernel: ProposalKernel,
    pub reg_spec: RegionalSearchSpec,
    pub enn_res: ResidentEnnConfig,
    pub policy: Option<ObjectivePolicy>,
    pub ckpt_dir: Option<&'a Path>,
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
pub(crate) struct MetalEngine {
    pub(crate) init_states: Vec<SearchState>,
    pub(crate) search: Option<RegionalSearch>,
    pub(crate) slot_gen: Vec<u64>,
    pub(crate) pending_restarts: VecDeque<(usize, u64)>,
    pub(crate) best_row: Vec<u16>,
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
pub(crate) fn unscored_state(
    job: &MetalJob<'_>,
    slot: usize,
    gen_seq: u64,
) -> Result<SearchState, String> {
    let base = unit_center(job.seed, slot, gen_seq, job.num_dim);
    let r_seed = job
        .seed
        .wrapping_add((slot as u64 + 1).wrapping_mul(0x9e37_79b9_7f4a_7c15))
        .wrapping_add(gen_seq.wrapping_mul(0xbf58_476d_1ce4_e5b9));
    let weight = 1.0f32 / (job.num_dim.max(1) as f32);
    let block = ParamBlock::new(r_seed, 0, job.num_dim, 0.5, weight)?;
    if job.has_surrogate {
        let mut st = SearchState::new_implicit(&base, vec![block], 128, job.tr_cfg, job.kernel)?;
        st.configure_enn(job.enn_res.clone())?;
        Ok(st)
    } else {
        let mut st = SearchState::new_fp16(&base, vec![block], 0, job.tr_cfg, job.kernel)?;
        st.configure_pool(job.enn_res.pool)?;
        Ok(st)
    }
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
pub(crate) fn observe_init(
    job: &MetalJob<'_>,
    st: &mut SearchState,
    scores: &[f64],
) -> Result<(), String> {
    if job.n_obj >= 2 {
        st.observe_objectives(obj_obs(scores, job.n_obj)?)?;
    } else {
        st.observe_initial(scores[0] as f32, 0.0)?;
    }
    if let Some(ref p) = job.policy {
        st.configure_objectives(Some(p.clone()))?;
    }
    Ok(())
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
pub(crate) fn ask_retry(
    search: &mut RegionalSearch,
    r_seed: u64,
    mut r_ask: baux::trials::RoundConfig,
) -> Result<RegionalProposal, String> {
    let mut last_err = String::new();
    for attempt in 0..4u64 {
        let s = r_seed.wrapping_add(attempt.wrapping_mul(0x9e37_79b9_7f4a_7c15));
        r_ask.acquisition.seed = s;
        match search.ask(s, r_ask) {
            Ok(p) => return Ok(p),
            Err(e) => last_err = e,
        }
    }
    Err(last_err)
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
pub(crate) fn copy_best(ptr: *const u16, sc: f64, best_score: f64, dst: &mut [u16]) {
    if sc > best_score {
        let src = unsafe { std::slice::from_raw_parts(ptr, dst.len()) };
        dst.copy_from_slice(src);
    }
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
impl MetalEngine {
    pub(crate) fn new(job: &MetalJob<'_>) -> Self {
        let n_reg = job.reg_spec.count.max(1);
        Self {
            init_states: Vec::with_capacity(n_reg),
            search: None,
            slot_gen: vec![0; n_reg],
            pending_restarts: VecDeque::new(),
            best_row: unit_center(job.seed, 0, 0, job.num_dim),
        }
    }

    fn step_init<F>(
        &mut self,
        job: &MetalJob<'_>,
        best_score: f64,
        mut eval: F,
    ) -> Result<(Vec<f64>, f64, u64), String>
    where
        F: FnMut(*const u16) -> Result<(Vec<f64>, f64, u64), String>,
    {
        let slot = self.init_states.len();
        let mut st = unscored_state(job, slot, 0)?;
        let buf = st.base_buffer();
        let ptr = buf.contents().cast::<u16>();
        let (scores, vl, dt_us) = eval(ptr)?;
        let sc = crate::experiment::julia::driver::scalar_score(&scores, 0, job.n_obj);
        copy_best(ptr, sc, best_score, &mut self.best_row);
        drop(buf);
        observe_init(job, &mut st, &scores)?;
        self.init_states.push(st);
        if self.init_states.len() == job.reg_spec.count.max(1) {
            let states = std::mem::take(&mut self.init_states);
            self.search = Some(RegionalSearch::from_spec(states, &job.reg_spec)?);
        }
        Ok((scores, vl, dt_us))
    }

    fn step_restart<F>(
        &mut self,
        job: &MetalJob<'_>,
        slot: usize,
        gen_seq: u64,
        best_score: f64,
        mut eval: F,
    ) -> Result<(Vec<f64>, f64, u64), String>
    where
        F: FnMut(*const u16) -> Result<(Vec<f64>, f64, u64), String>,
    {
        let mut st = unscored_state(job, slot, gen_seq)?;
        let buf = st.base_buffer();
        let ptr = buf.contents().cast::<u16>();
        let (scores, vl, dt_us) = eval(ptr)?;
        let sc = crate::experiment::julia::driver::scalar_score(&scores, 0, job.n_obj);
        copy_best(ptr, sc, best_score, &mut self.best_row);
        drop(buf);
        observe_init(job, &mut st, &scores)?;
        self.search.as_mut().ok_or("no search")?.restart(slot, st)?;
        Ok((scores, vl, dt_us))
    }

    fn step_ask<F>(
        &mut self,
        job: &MetalJob<'_>,
        round: usize,
        best_score: f64,
        mut eval: F,
    ) -> Result<(Vec<f64>, f64, u64), String>
    where
        F: FnMut(*const u16) -> Result<(Vec<f64>, f64, u64), String>,
    {
        let s = self.search.as_mut().ok_or("no search")?;
        let r_seed = job.seed.wrapping_add(round as u64);
        let prop = ask_retry(s, r_seed, job.enn_res.ask)?;
        let slot = prop.ticket().region.slot;
        let buf = s.buffer(&prop)?;
        let ptr = buf.contents().cast::<u16>();
        let (scores, vl, dt_us) = eval(ptr)?;
        let sc = crate::experiment::julia::driver::scalar_score(&scores, 0, job.n_obj);
        copy_best(ptr, sc, best_score, &mut self.best_row);
        drop(buf);
        if job.n_obj >= 2 {
            let _ = s.tell_objectives(&prop, obj_obs(&scores, job.n_obj)?)?;
        } else {
            let _ = s.tell(&prop, scores[0] as f32, 0.0)?;
        }
        if baux::resident_metal::regional_restarts(s, slot)? > 0 {
            self.slot_gen[slot] += 1;
            self.pending_restarts.push_back((slot, self.slot_gen[slot]));
        }
        Ok((scores, vl, dt_us))
    }
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
pub(crate) fn metal_job<F, L>(
    job: &MetalJob<'_>,
    loss_fn: L,
    mut eval_fn: F,
) -> Result<(ckpt::LiveCkpt, Vec<u16>), String>
where
    F: FnMut(*const u16) -> Result<(Vec<f64>, f64, u64), String>,
    L: Fn(f64) -> f64,
{
    let saved = match job.ckpt_dir {
        Some(dir) => ckpt::load_meta(dir, job.rep)?,
        None => None,
    };
    let saved_n = saved
        .as_ref()
        .map_or(0, |s| s.completed_round.min(job.rounds));
    let mut st = ckpt::LiveCkpt::empty(job.rounds);
    st.elapsed_seconds = saved.as_ref().map_or(0.0, |s| s.elapsed_seconds);
    st.worker_seconds = saved.as_ref().map_or(0.0, |s| s.worker_seconds);
    let mut best_viol = f64::INFINITY;
    let mut eng = MetalEngine::new(job);
    let every = ckpt::ckpt_interval();
    let mut chunk_t0 = Instant::now();

    for round in 1..=job.rounds {
        if round == saved_n + 1 {
            chunk_t0 = Instant::now();
        }
        let step_eval = |ptr: *const u16| -> Result<(Vec<f64>, f64, u64), String> {
            if let Some(ref sv) = saved
                && round <= saved_n
            {
                let vl = sv.eval_viols.get(round - 1).copied().unwrap_or(0.0);
                return Ok((sv.eval_scores[round - 1].clone(), vl, 0));
            }
            eval_fn(ptr)
        };
        let (scores, vl, dt_us) = if eng.search.is_none() {
            eng.step_init(job, st.best_score, step_eval)?
        } else if let Some((slot, gseq)) = eng.pending_restarts.pop_front() {
            eng.step_restart(job, slot, gseq, st.best_score, step_eval)?
        } else {
            eng.step_ask(job, round, st.best_score, step_eval)?
        };
        st.worker_seconds += (dt_us as f64) * 1.0e-6;
        let sc = crate::experiment::julia::driver::scalar_score(&scores, 0, job.n_obj);
        if sc > st.best_score {
            st.best_score = sc;
            best_viol = vl;
            st.total_accepts += 1;
        }
        if round == 1 {
            st.initial_error = loss_fn(st.best_score);
        }
        st.best_scores.push(st.best_score);
        st.best_viols.push(best_viol);
        st.eval_scores.push(scores);
        st.eval_viols.push(vl);
        st.completed_round = round;
        if let Some(dir) = job.ckpt_dir
            && round > saved_n
            && round % every == 0
            && round < job.rounds
        {
            st.elapsed_seconds += chunk_t0.elapsed().as_secs_f64();
            chunk_t0 = Instant::now();
            ckpt::save_meta(dir, job.rep, &st)?;
        }
    }
    if job.rounds > saved_n {
        st.elapsed_seconds += chunk_t0.elapsed().as_secs_f64();
    }
    Ok((st, eng.best_row))
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
pub(crate) fn objective_policy(cfg: &Config, n_obj: usize, seed: u64) -> Option<ObjectivePolicy> {
    if n_obj < 2 {
        return None;
    }
    let scales = match cfg.morbo {
        Some(ref m) if m.scales.len() == n_obj => m.scales.clone(),
        _ => vec![1.0f32; n_obj],
    };
    let acq = if let Some(ref m) = cfg.morbo {
        let weights = if m.preferences.len() == n_obj {
            m.preferences.clone()
        } else {
            vec![1.0f32; n_obj]
        };
        baux::config::ObjectiveAcquisition::Morbo {
            weights,
            regions: m.regions,
            alpha: m.alpha as f32,
            seed,
            rescalarize: m.rescalarize,
            clip: true,
        }
    } else {
        baux::config::ObjectiveAcquisition::Pareto
    };
    Some(ObjectivePolicy {
        acquisition: acq,
        scales,
    })
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
    let enn_res = res_enn(cfg, &cfg_opt, seed)?;
    let tr_cfg = match &cfg_opt.trust_region {
        baux::trust_region::TrustRegionConfig::Turbo(l) => *l,
        baux::trust_region::TrustRegionConfig::Morbo(m) => m.length,
    };
    let has_surrogate = matches!(
        cfg_opt.surrogate,
        baux::optimizer_config::SurrogateConfig::ENN(_)
    );
    let job = MetalJob {
        rep,
        seed,
        num_dim,
        n_obj,
        rounds: cfg.rounds,
        has_surrogate,
        tr_cfg,
        kernel: resolve_kernel(&cfg.optimizer),
        reg_spec: regional_spec(&cfg.optimizer),
        enn_res,
        policy: objective_policy(cfg, n_obj, seed),
        ckpt_dir,
    };
    let (st, best_row) = metal_job(
        &job,
        |sc| crate::experiment::julia::driver::loss_metric(cfg, sc),
        |ptr| worker.eval_half(ptr),
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
