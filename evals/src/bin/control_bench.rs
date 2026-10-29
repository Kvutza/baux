//! Entrypoint for continuous control evaluation benchmarks.

use baux::BayesOptimizer;
use baux::semantics::neighbor::MAX_HISTORY;
use baux::semantics::optimizer::surrogate::Surrogate;
use baux::semantics::optimizer::trust_region::TrustRegionController;
use baux_evals as bench;
use baux_evals::BayesOptimizerExt;
use baux_evals::bench_chain::{ChainPolicy, PerturbMode};
use baux_evals::bench_oracle::ControlOracle;
use baux_evals::bench_schedule::{ActiveSampling, RolloutSchedule, control_seed};
use baux_evals::bench_task::{ControlTask, HalfCheetahTask, HopperTask};
use ndarray::{Array1, Array2, ArrayView1};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use std::env;
use std::fs;
use std::path::Path;

struct CandidatePool {
    weights: Vec<Vec<f64>>,
    distances: Vec<Vec<f64>>,
}

struct OperatorState {
    history: Vec<Vec<f64>>,
    surrogate: Surrogate,
    tr: TrustRegionController,
}

fn build_policy(
    arch: bench::ControlArch,
    obs_dim: usize,
    act_dim: usize,
    seed: u64,
) -> ChainPolicy {
    let mut policy = match arch {
        bench::ControlArch::Linear => ChainPolicy::new(&[obs_dim, act_dim]),
        bench::ControlArch::Mlp => ChainPolicy::new(&[obs_dim, 32, 16, act_dim]),
        bench::ControlArch::Simba => ChainPolicy::simba(obs_dim, 128, act_dim),
    };
    let zeros = policy.weights.clone();
    policy.apply_perturbation(&zeros, seed, 0.25, PerturbMode::Uncoupled);
    policy
}

fn perturb_mode(generator: bench::ProposalGenerator) -> PerturbMode {
    match generator {
        bench::ProposalGenerator::AdjointChain => PerturbMode::AdjointChain,
        bench::ProposalGenerator::Independent => PerturbMode::Uncoupled,
        bench::ProposalGenerator::PairRotation | bench::ProposalGenerator::Compass => {
            PerturbMode::AdjointChain
        }
    }
}

fn candidate_distances(cand: &[f64], history: &[Vec<f64>], surrogate_len: usize) -> Vec<f64> {
    let dim = (cand.len() as f64).max(1.0);
    history
        .iter()
        .take(surrogate_len)
        .map(|hist| {
            let sum_sq: f64 = cand
                .iter()
                .zip(hist.iter())
                .map(|(&c, &h)| (c - h).powi(2))
                .sum();
            sum_sq / dim
        })
        .collect()
}

fn propose_pool(
    incumbent: &ChainPolicy,
    history: &[Vec<f64>],
    surrogate_len: usize,
    prop_seed: u64,
    radius: f32,
    sign: f32,
    cfg: &bench::ControlConfig,
) -> CandidatePool {
    let num_candidates = cfg.experiment.num_arms.max(4);
    let mode = perturb_mode(cfg.optimizer.proposal.generator);
    let mut pool = CandidatePool {
        weights: Vec::with_capacity(num_candidates),
        distances: Vec::with_capacity(num_candidates),
    };
    let mut cand = incumbent.clone();

    for arm in 0..num_candidates {
        let idx = prop_seed.wrapping_mul(8).wrapping_add(arm as u64);
        let arm_seed = control_seed(0, "control-arm", idx);
        cand.apply_signed(&incumbent.weights, arm_seed, radius, sign, mode);
        let dists = candidate_distances(&cand.weights, history, surrogate_len);
        pool.weights.push(cand.weights.clone());
        pool.distances.push(dists);
    }
    pool
}

fn step_operator<T: ControlTask>(
    round: usize,
    oracle: &mut ControlOracle<T>,
    incumbent: &mut ChainPolicy,
    state: &mut OperatorState,
    cfg: &bench::ControlConfig,
) -> Result<(), String> {
    let radius = state.tr.radius();
    let surrogate_len = state.surrogate.identities().len();
    let prop_root = control_seed(0, "control-proposal", 0);
    let prop_seed = prop_root
        .checked_add(round as u64)
        .ok_or_else(|| "proposal seed overflow".to_string())?;
    let pool = propose_pool(
        incumbent,
        &state.history,
        surrogate_len,
        prop_seed,
        radius,
        1.0,
        cfg,
    );

    let acq_seed = control_seed(0, "control-acquisition", round as u64);
    let rule = cfg.optimizer.acquisition_rule(acq_seed, round);
    let sel = state
        .surrogate
        .select_pool(&pool.distances, round as u32, rule, None)?;
    let slot = sel.selected.min(pool.weights.len() - 1);

    let mut cand = incumbent.clone();
    cand.weights = pool.weights[slot].clone();
    let (obs, evidence) = oracle.evaluate_candidate(&mut cand)?;

    let chosen_mean = obs.estimates()[0].mean;
    let chosen_var = obs.estimates()[0].variance;
    let inc_base = oracle.incumbent_return().unwrap_or(sel.baseline_value);
    let mut sel_eval = sel;
    sel_eval.baseline_value = inc_base;
    let decision = state
        .tr
        .evaluate_selection(chosen_mean, chosen_var, radius, &sel_eval)?;

    let written = state.surrogate.append(
        &pool.distances[slot],
        chosen_mean,
        chosen_var,
        decision.accepted,
    )?;

    if decision.accepted {
        *incumbent = cand;
        oracle.set_incumbent(chosen_mean);
    }
    if written < state.history.len() {
        state.history[written] = pool.weights[slot].clone();
    } else if state.history.len() < MAX_HISTORY {
        state.history.push(pool.weights[slot].clone());
    }

    log_progress(
        round,
        chosen_mean,
        inc_base,
        radius,
        decision.accepted,
        evidence.x_displacement,
        oracle,
        incumbent,
        cfg,
    )?;

    oracle.advance_round();
    Ok(())
}

fn decode_unit(unit_row: &ArrayView1<f64>, bounds: &Array2<f64>, weights: &mut [f64]) {
    for (j, w) in weights.iter_mut().enumerate() {
        let low = bounds[[j, 0]];
        let high = bounds[[j, 1]];
        *w = low + unit_row[j] * (high - low);
    }
}

fn eval_batch<T: ControlTask>(
    asked: &Array2<f64>,
    prev_inc: Option<&Array1<f64>>,
    passive_count: usize,
    bounds: &Array2<f64>,
    oracle: &mut ControlOracle<T>,
    incumbent: &mut ChainPolicy,
) -> Result<(Array2<f64>, Array2<f64>, Array2<f64>, f32, f32, ChainPolicy), String> {
    let total_rows = asked.nrows() + passive_count;
    let mut x = Array2::zeros((total_rows, incumbent.weights.len()));
    let mut y = Array2::zeros((total_rows, 1));
    let mut yvar = Array2::zeros((total_rows, 1));
    let mut cand = incumbent.clone();
    let mut best_mean = f32::NEG_INFINITY;
    let mut best_disp = 0.0;
    let mut best_cand = incumbent.clone();

    for arm in 0..asked.nrows() {
        let row = asked.row(arm);
        x.row_mut(arm).assign(&row);
        decode_unit(&row, bounds, &mut cand.weights);
        let (obs, ev) = oracle.evaluate_candidate(&mut cand)?;
        let mean = obs.estimates()[0].mean;
        y[[arm, 0]] = f64::from(mean);
        yvar[[arm, 0]] = f64::from(obs.estimates()[0].variance);
        if mean > best_mean {
            best_mean = mean;
            best_disp = ev.x_displacement;
            best_cand = cand.clone();
        }
    }

    if let Some(inc_x) = prev_inc {
        decode_unit(&inc_x.view(), bounds, &mut incumbent.weights);
        for p in 0..passive_count {
            let row_idx = asked.nrows() + p;
            x.row_mut(row_idx).assign(inc_x);
            let (p_obs, _) = oracle.evaluate_candidate(incumbent)?;
            let p_mean = p_obs.estimates()[0].mean;
            oracle.set_incumbent(p_mean);
            y[[row_idx, 0]] = f64::from(p_mean);
            yvar[[row_idx, 0]] = f64::from(p_obs.estimates()[0].variance);
        }
    }

    Ok((x, y, yvar, best_mean, best_disp, best_cand))
}

fn step_turbo<T: ControlTask>(
    round: usize,
    oracle: &mut ControlOracle<T>,
    incumbent: &mut ChainPolicy,
    opt: &mut BayesOptimizer,
    rng: &mut ChaCha8Rng,
    cfg: &bench::ControlConfig,
) -> Result<(), String> {
    let num_arms = cfg.experiment.num_arms.max(1);
    let asked = opt.ask(num_arms, rng).map_err(|e| e.to_string())?;
    let prev_inc = opt.incumbent_x();
    let passive = if prev_inc.is_some() {
        cfg.experiment.num_denoise_passive
    } else {
        0
    };
    let bounds = opt.bounds().clone();
    let (x, y, yvar, best_mean, best_disp, best_cand) = eval_batch(
        &asked,
        prev_inc.as_ref(),
        passive,
        &bounds,
        oracle,
        incumbent,
    )?;

    opt.tell_variance(&x.view(), &y.view(), Some(&yvar.view()), rng)
        .map_err(|e| e.to_string())?;

    let new_inc = opt.incumbent_x();
    let accepted = new_inc != prev_inc;
    if accepted {
        *incumbent = best_cand;
        oracle.set_incumbent(best_mean);
    }
    if let Some(ref inc_x) = new_inc {
        decode_unit(&inc_x.view(), &bounds, &mut incumbent.weights);
    }
    let base = opt.y_scalar().map_or(best_mean, |yv| yv[0] as f32);
    let radius = opt.tr_length() as f32;

    log_progress(
        round, best_mean, base, radius, accepted, best_disp, oracle, incumbent, cfg,
    )?;

    oracle.advance_round();
    Ok(())
}

fn log_progress<T: ControlTask>(
    round: usize,
    chosen: f32,
    base: f32,
    radius: f32,
    accepted: bool,
    disp: f32,
    oracle: &mut ControlOracle<T>,
    incumbent: &mut ChainPolicy,
    _cfg: &bench::ControlConfig,
) -> Result<(), String> {
    let h = oracle.heldout_cached(incumbent)?;
    println!(
        "o|round {:05}: chosen_ret={:7.2} base={:7.2} radius={:.4} acc={} heldout_ret={:7.2} \
         gap={:7.2} disp={:.2}",
        round, chosen, base, radius, accepted, h.mean_return, h.active_minus_heldout_gap, disp
    );
    Ok(())
}

fn run_turbo<T: ControlTask>(
    oracle: &mut ControlOracle<T>,
    incumbent: &mut ChainPolicy,
    cfg: &bench::ControlConfig,
) -> Result<(), String> {
    let turbo_seed = control_seed(0, "control-turbo", 0);
    let mut rng = ChaCha8Rng::seed_from_u64(turbo_seed);
    let (base_obs, _) = oracle.evaluate_candidate(incumbent)?;
    let base_mean = base_obs.estimates()[0].mean;
    oracle.set_incumbent(base_mean);
    let mut opt = cfg
        .optimizer
        .build_turbo(incumbent.weights.len(), &mut rng)?;
    let inc_f32: Vec<f32> = incumbent.weights.iter().map(|&x| x as f32).collect();
    opt.seed_incumbent(&inc_f32, f64::from(base_mean), &mut rng)?;
    for round in 0..cfg.experiment.num_rounds {
        step_turbo(round, oracle, incumbent, &mut opt, &mut rng, cfg)?;
    }
    Ok(())
}

fn run_operator<T: ControlTask>(
    oracle: &mut ControlOracle<T>,
    incumbent: &mut ChainPolicy,
    cfg: &bench::ControlConfig,
) -> Result<(), String> {
    let fit_seed = control_seed(0, "control-fit", 0);
    let (base_obs, _) = oracle.evaluate_candidate(incumbent)?;
    let base_mean = base_obs.estimates()[0].mean;
    let base_var = base_obs.estimates()[0].variance;
    oracle.set_incumbent(base_mean);
    let mut state = OperatorState {
        history: vec![incumbent.weights.clone()],
        surrogate: cfg
            .optimizer
            .build_surrogate(base_mean, base_var, fit_seed)?,
        tr: cfg.optimizer.build_controller()?,
    };
    for round in 0..cfg.experiment.num_rounds {
        step_operator(round, oracle, incumbent, &mut state, cfg)?;
    }
    Ok(())
}

fn run_experiment<T: ControlTask>(task: T, cfg: &bench::ControlConfig) -> Result<(), String> {
    let obs_dim = task.obs_dim();
    let act_dim = task.model().nu;
    let model_seed = control_seed(0, "control-model", 0);
    let schedule = RolloutSchedule::new(
        0,
        ActiveSampling::Frozen {
            episodes: cfg.experiment.num_denoise_passive.max(1) as u32,
        },
        cfg.experiment.heldout_episodes.max(1) as u32,
    );
    let mut oracle = ControlOracle::new(task, schedule);
    let mut incumbent = build_policy(cfg.experiment.policy, obs_dim, act_dim, model_seed);
    println!(
        "o|control-bench: env={:?} policy={:?} params={} arms={} rounds={} passive={} reps={}",
        cfg.experiment.env,
        cfg.experiment.policy,
        incumbent.weights.len(),
        cfg.experiment.num_arms,
        cfg.experiment.num_rounds,
        cfg.experiment.num_denoise_passive,
        cfg.experiment.repetitions
    );
    println!(
        "o|control-bench: algo={} surrogate={:?} index={} k={} tr_policy={}",
        cfg.optimizer.algorithm,
        cfg.optimizer.surrogate.estimator,
        cfg.optimizer.surrogate.index,
        cfg.optimizer.surrogate.neighbors,
        cfg.optimizer.trust_region.policy
    );
    if cfg.optimizer.is_turbo() {
        run_turbo(&mut oracle, &mut incumbent, cfg)?;
    } else if cfg.optimizer.algorithm == "operator-bo" {
        run_operator(&mut oracle, &mut incumbent, cfg)?;
    } else {
        return Err(format!(
            "unsupported optimizer algorithm: {}",
            cfg.optimizer.algorithm
        ));
    }
    let final_heldout = oracle.evaluate_heldout(&mut incumbent)?;
    println!(
        "o|control-bench: completed. Final heldout return: {:.2} +/- {:.2} (survival: {:.1}%)",
        final_heldout.mean_return,
        final_heldout.se_return,
        final_heldout.survival_fraction * 100.0
    );
    Ok(())
}

fn main() -> Result<(), String> {
    let default_path = "evals/experiments/cheetah-turbo.toml";
    let path = env::args()
        .nth(1)
        .unwrap_or_else(|| default_path.to_string());
    if !Path::new(&path).exists() {
        return Err(format!("config file not found: {path}"));
    }
    let content = fs::read_to_string(&path).map_err(|e| format!("read {path}: {e}"))?;
    let mut cfg = bench::ControlConfig::from_toml(&content)?;
    if let Some(r_str) = env::args().nth(2) {
        cfg.experiment.num_rounds = r_str
            .parse::<usize>()
            .map_err(|e| format!("parse rounds {r_str}: {e}"))?;
    }

    match cfg.experiment.env {
        bench::ControlEnv::Cheetah => run_experiment(HalfCheetahTask::new(), &cfg),
        bench::ControlEnv::Hopper => run_experiment(HopperTask::new(), &cfg),
    }
}
