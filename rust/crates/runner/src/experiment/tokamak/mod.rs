//! Tokamak reactor controller optimization experiments (TORAX 2.0.0 and
//! TokaGym).

pub mod ckpt;
pub mod label;
pub mod metal;
pub mod worker;

#[cfg(test)]
mod tests;

use std::fs;
use std::path::Path;
use std::time::Instant;

use baux::artifact::json::{Value, json};
use baux::artifact::runs;
use baux::config::OptimizerSpec;
use baux::optimizer::DomainFrame;
use baux::{BayesOptimizer, Prng, prng};
use deser::{Deserialize, Serialize};
use ndarray::Array2;

use self::ckpt::{
    LiveCkpt, SeedRecord, ckpt_dir, ckpt_interval, load_done, load_live, rep_indices, save_done,
    save_live,
};
use self::label::{opt_label, round_rng};
use self::worker::{TokamakWorker, WorkerSpec};
use crate::experiment::julia::config::Backend;
use crate::experiment::julia::driver::mean_sem;

#[derive(Clone, Deserialize, Serialize)]
#[deser(deny_unknown_fields, rename_all = "kebab-case")]
pub(super) struct Config {
    pub version: u32,
    pub experiment: String,
    pub rounds: usize,
    #[deser(default = default_reps())]
    pub repetitions: u32,
    #[deser(default)]
    pub backend: Backend,
    pub optimizer: OptimizerSpec,
}

const fn default_reps() -> u32 {
    1
}

fn is_tokagym(cfg: &Config) -> bool {
    cfg.experiment == "tokamak-tokagym"
}

fn exp_domain(cfg: &Config) -> DomainFrame {
    if is_tokagym(cfg) {
        baux::tokamak::tokagym_domain()
    } else {
        baux::tokamak::domain_frame()
    }
}

fn tracking_error(cfg: &Config, score: f64) -> f64 {
    if is_tokagym(cfg) {
        (-score).max(0.0)
    } else {
        (100.0 - score).max(0.0)
    }
}

fn eval_batch(
    worker: &mut Option<TokamakWorker>,
    batch_weights: &[f64],
    count: usize,
) -> Result<Vec<f64>, String> {
    if let Some(w) = worker.as_mut() {
        return w.eval(batch_weights, count);
    }
    let mut out = vec![0.0f64; count];
    baux::tokamak::evaluate_batch(batch_weights, count, &mut out);
    Ok(out)
}

fn make_opt(
    config: &Config,
    ckpt_dir: Option<&Path>,
    rep: u32,
    rng: &mut Prng,
) -> Result<BayesOptimizer, String> {
    let use_disk = std::env::var("BAUX_ENN_DISK").ok().as_deref() == Some("1");
    if use_disk && let Some(dir) = ckpt_dir {
        let work = dir.join(format!("bpann_rep{rep}"));
        return BayesOptimizer::from_disk(exp_domain(config), &config.optimizer, 0, work, rng)
            .map_err(|e| e.to_string());
    }
    BayesOptimizer::from_domain(exp_domain(config), &config.optimizer, 0, rng)
        .map_err(|e| e.to_string())
}

fn seed_weights(seed: u64, config: &Config) -> Result<Vec<f64>, String> {
    let (n_pert, cands, fast) = if is_tokagym(config) {
        (13671, 4, true)
    } else {
        (16053, 4, false)
    };
    let ref_toml = format!(
        "algorithm = \"bayesian-search\"\ncandidates = {cands}\narms = 1\nacquisition = \"random\"\n\
         [perturbation]\nsampler = \"raasp\"\nnum-pert = {n_pert}\nraasp-fast = {fast}\n\
         [trust-region]\ninitial = 0.8\nmin = 0.00048828125\nmax = 1.6\n\
         [regional-search]\ncount = 1\nsharing = \"origin-only\"\nallocation = \"incumbent-ucb1\"\n\
         controller = \"turbo\"\nsuccess-tolerance = 2\nfailure-tolerance = 3\n"
    );
    let spec: OptimizerSpec =
        baux::artifact::toml::from_str(&ref_toml).map_err(|e| e.to_string())?;
    let mut init_rng = prng(seed);
    let domain = exp_domain(config);
    let mut opt = BayesOptimizer::from_domain(domain.clone(), &spec, 0, &mut init_rng)
        .map_err(|e| e.to_string())?;
    let mut step_rng = round_rng(seed, 1);
    let unit = opt.ask(1, &mut step_rng).map_err(|e| e.to_string())?;
    let points = domain.from_unit(&unit.view());
    Ok(points
        .as_slice_memory_order()
        .ok_or("point non-contiguous")?
        .to_vec())
}

fn finalize_seed(config: &Config, rep: u32, seed: u64, num_dim: usize, st: LiveCkpt) -> SeedRecord {
    let final_error = tracking_error(config, st.best_score);
    let reduction_pct = if st.initial_error > 0.0 {
        ((st.initial_error - final_error) / st.initial_error * 100.0).max(0.0)
    } else {
        0.0
    };
    eprintln!(
        "tokamak.seed rep={}/{} seed={} final_score={:.6} final_error={:.6} reduction={:.2}% accepts={}/{} elapsed_s={:.2}",
        rep + 1,
        config.repetitions.max(1),
        seed,
        st.best_score,
        final_error,
        reduction_pct,
        st.total_accepts,
        config.rounds,
        st.elapsed_seconds
    );
    SeedRecord {
        rep,
        seed,
        num_dim,
        initial_error: st.initial_error,
        final_score: st.best_score,
        final_error,
        reduction_pct,
        total_accepts: st.total_accepts,
        elapsed_seconds: st.elapsed_seconds,
        best_scores: st.best_scores,
    }
}

fn run_ppo(
    config: &Config,
    rep: u32,
    worker: &mut Option<TokamakWorker>,
) -> Result<SeedRecord, String> {
    let seed = baux::tune_seed(&config.experiment, "proposal", rep, 0);
    let init_weights = seed_weights(seed, config)?;
    let mut rng = prng(seed ^ 0x5050_4f00);
    let mut agent = baux::tokamak::ppo::CandlePpo::new(&init_weights, &mut rng)?;
    let steps = if is_tokagym(config) { 50usize } else { 20usize };
    let mut st = LiveCkpt::empty(config.rounds);
    let start_time = Instant::now();
    let mut weights = agent.actor_weights()?;
    for round in 1..=config.rounds {
        let det = round == 1 || round % 8 == 1;
        let eps = agent.sample_noise(steps, det, &mut rng)?;
        let (score, obs, u, rewards) = match worker.as_mut() {
            Some(w) => w.eval_traj(&weights, &eps, steps)?,
            None => baux::tokamak::rollout_traj(&weights, &eps, steps),
        };
        if score > st.best_score {
            st.best_score = score;
            st.total_accepts += 1;
        }
        if round == 1 {
            st.initial_error = tracking_error(config, score);
        }
        if agent.record_step(obs, u, rewards)? {
            weights = agent.actor_weights()?;
        }
        st.best_scores.push(st.best_score);
        st.completed_round = round;
    }
    st.elapsed_seconds = start_time.elapsed().as_secs_f64();
    let num_dim = if is_tokagym(config) {
        baux::tokamak::TOKAGYM_PARAMS
    } else {
        baux::tokamak::TOTAL_PARAMS
    };
    Ok(finalize_seed(config, rep, seed, num_dim, st))
}

fn run_seed(
    config: &Config,
    rep: u32,
    ckpt_dir: Option<&Path>,
    worker: &mut Option<TokamakWorker>,
) -> Result<SeedRecord, String> {
    if let Some(dir) = ckpt_dir
        && let Some(done) = load_done(dir, rep)
        && done.best_scores.len() == config.rounds
    {
        eprintln!(
            "tokamak.seed rep={} restored completed seed from checkpoint",
            rep + 1
        );
        return Ok(done);
    }
    let seed = baux::tune_seed(&config.experiment, "proposal", rep, 0);
    let mut init_rng = prng(seed);
    let template = make_opt(config, ckpt_dir, rep, &mut init_rng)?;
    let num_dim = template.num_dim();
    let empty_ckpt = || LiveCkpt::empty(config.rounds);
    let (mut optimizer, mut st) = match ckpt_dir {
        Some(dir) => match load_live(dir, rep, seed, template)? {
            Some(pair) => pair,
            None => {
                let mut r2 = prng(seed);
                (make_opt(config, ckpt_dir, rep, &mut r2)?, empty_ckpt())
            }
        },
        None => (template, empty_ckpt()),
    };
    let arms = config
        .optimizer
        .candidate_generation()
        .map_or(1, |c| c.arms as usize)
        .max(1);
    let every = ckpt_interval();
    let domain = exp_domain(config);
    let start_round = st.completed_round + 1;
    let mut chunk_start = Instant::now();
    for round in start_round..=config.rounds {
        let mut step_rng = round_rng(seed, round);
        let unit = optimizer
            .ask(arms, &mut step_rng)
            .map_err(|e| e.to_string())?;
        let points = domain.from_unit(&unit.view());
        let slice = points
            .as_slice_memory_order()
            .ok_or("point non-contiguous")?;
        let scores = eval_batch(worker, slice, arms)?;
        let score = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        if score > st.best_score {
            st.best_score = score;
            st.total_accepts += 1;
        }
        if round == 1 {
            st.initial_error = tracking_error(config, score);
        }
        let y = Array2::from_shape_fn((arms, 1), |(i, _)| scores[i]);
        optimizer
            .tell(&unit.view(), &y.view(), &mut step_rng)
            .map_err(|e| e.to_string())?;
        st.best_scores.push(st.best_score);
        st.eval_scores.push(scores);
        st.completed_round = round;
        if let Some(dir) = ckpt_dir
            && round % every == 0
            && round < config.rounds
        {
            st.elapsed_seconds += chunk_start.elapsed().as_secs_f64();
            chunk_start = Instant::now();
            save_live(dir, rep, &optimizer, &st)?;
        }
    }
    st.elapsed_seconds += chunk_start.elapsed().as_secs_f64();
    let rec = finalize_seed(config, rep, seed, num_dim, st);
    if let Some(dir) = ckpt_dir {
        save_done(dir, &rec)?;
    }
    Ok(rec)
}

fn aggregate_traces(cfg: &Config, runs: &[SeedRecord], total_rounds: usize) -> Vec<Value> {
    (0..total_rounds)
        .map(|idx| {
            let scores: Vec<f64> = runs.iter().map(|r| r.best_scores[idx]).collect();
            let errors: Vec<f64> = scores.iter().map(|&s| tracking_error(cfg, s)).collect();
            let (mean_score, sem_score) = mean_sem(&scores);
            let (mean_err, sem_err) = mean_sem(&errors);
            json!({
                "round": idx + 1,
                "best_score": mean_score,
                "mean_best_score": mean_score,
                "sem_best_score": sem_score,
                "tracking_error": mean_err,
                "mean_tracking_error": mean_err,
                "sem_tracking_error": sem_err,
            })
        })
        .collect()
}

fn spawn_worker(
    config: &Config,
    root: Option<&Path>,
    arms: usize,
) -> Result<Option<TokamakWorker>, String> {
    let Some(r) = root else {
        return Ok(None);
    };
    let spec = if is_tokagym(config) {
        WorkerSpec {
            kind: "tokagym",
            lock: "tokagym",
            arms,
            num_dim: baux::tokamak::TOKAGYM_PARAMS,
            obs: baux::tokamak::TOKAGYM_OBS,
            act: baux::tokamak::TOKAGYM_ACT,
            steps: 50,
        }
    } else {
        WorkerSpec {
            kind: "torax",
            lock: "torax",
            arms,
            num_dim: baux::tokamak::TOTAL_PARAMS,
            obs: baux::tokamak::OBS_DIM,
            act: baux::tokamak::ACTION_DIM,
            steps: 20,
        }
    };
    TokamakWorker::spawn(r, spec).map(Some)
}

fn build_report(config: &Config, regions: usize, runs: &[SeedRecord]) -> Value {
    let num_dim = runs.first().map_or(0, |r| r.num_dim);
    let init_errs: Vec<f64> = runs.iter().map(|r| r.initial_error).collect();
    let final_scores: Vec<f64> = runs.iter().map(|r| r.final_score).collect();
    let final_errs: Vec<f64> = runs.iter().map(|r| r.final_error).collect();
    let reductions: Vec<f64> = runs.iter().map(|r| r.reduction_pct).collect();
    let (init_mean, init_sem) = mean_sem(&init_errs);
    let (score_mean, score_sem) = mean_sem(&final_scores);
    let (err_mean, err_sem) = mean_sem(&final_errs);
    let (red_mean, red_sem) = mean_sem(&reductions);
    let elapsed: f64 = runs.iter().map(|r| r.elapsed_seconds).sum();
    let traces = aggregate_traces(config, runs, config.rounds);
    let seed_runs: Vec<Value> = runs
        .iter()
        .map(|r| {
            json!({
                "rep": r.rep,
                "seed": r.seed,
                "initial_tracking_error": r.initial_error,
                "final_score": r.final_score,
                "final_tracking_error": r.final_error,
                "tracking_error_reduction_pct": r.reduction_pct,
                "total_accepts": r.total_accepts,
                "elapsed_seconds": r.elapsed_seconds,
                "best_scores": r.best_scores,
            })
        })
        .collect();
    let (exp_name, task_name) = if is_tokagym(config) {
        ("tokamak-tokagym", "TCV-11-to-8-Policy-Optimization")
    } else {
        ("tokamak-tcv", "TCV-93-to-19-Policy-Optimization")
    };
    json!({
        "experiment": exp_name,
        "task": task_name,
        "optimizer_label": opt_label(&config.optimizer),
        "parameters": num_dim,
        "regions": regions,
        "rounds": config.rounds,
        "repetitions": runs.len(),
        "initial_tracking_error": init_mean,
        "initial_tracking_error_sem": init_sem,
        "final_score_mean": score_mean,
        "final_score_sem": score_sem,
        "final_tracking_error": err_mean,
        "final_tracking_error_sem": err_sem,
        "tracking_error_reduction_pct": red_mean,
        "tracking_error_reduction_pct_sem": red_sem,
        "elapsed_seconds": elapsed,
        "traces": traces,
        "seed_runs": seed_runs,
    })
}

fn run_opt(config: &Config, root: Option<&Path>, ckpt_dir: Option<&Path>) -> Result<Value, String> {
    let reps = rep_indices(config.repetitions)?;
    let regions = config.optimizer.regional_search().map_or(1, |r| r.count);
    let arms = config
        .optimizer
        .candidate_generation()
        .map_or(1, |c| c.arms as usize)
        .max(1);
    let mut worker = spawn_worker(config, root, arms)?;
    let is_ppo = matches!(config.optimizer, OptimizerSpec::Ppo);
    let is_metal = config.backend == Backend::ResidentMetal;
    let domain = exp_domain(config);
    eprintln!(
        "[tune] starting {} closed-loop {} ({:?}) | rounds={} regions={} seeds={}",
        if is_tokagym(config) {
            "TokaGym"
        } else {
            "TORAX 2.0.0"
        },
        opt_label(&config.optimizer),
        config.backend,
        config.rounds,
        regions,
        reps.len()
    );
    let runs: Vec<SeedRecord> = reps
        .into_iter()
        .map(|rep| {
            if is_ppo {
                run_ppo(config, rep, &mut worker)
            } else if is_metal {
                metal::metal_seed(config, &domain, rep, ckpt_dir, &mut worker)
            } else {
                run_seed(config, rep, ckpt_dir, &mut worker)
            }
        })
        .collect::<Result<_, _>>()?;
    Ok(build_report(config, regions, &runs))
}

fn prepare_dir(root: &Path, config: &Config) -> Result<(std::path::PathBuf, String), String> {
    let nonce = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_err(|e| e.to_string())?
        .as_nanos();
    let output = root
        .join(".cache/baux/runs/tokamak")
        .join(nonce.to_string());
    fs::create_dir_all(&output).map_err(|e| e.to_string())?;
    let resolved = baux::artifact::toml::pretty_string(config).map_err(|e| e.to_string())?;
    fs::write(output.join("config.toml"), &resolved).map_err(|e| e.to_string())?;
    Ok((output, resolved))
}

fn write_artifacts(output: &Path, report: &Value) -> Result<(), String> {
    let bytes = baux::artifact::cbor::to_vec(report).map_err(|e| e.to_string())?;
    fs::write(output.join("tokamak.cbor"), &bytes).map_err(|e| e.to_string())?;
    let summary = baux::artifact::json::to_string(report).map_err(|e| e.to_string())?;
    fs::write(output.join("summary.json"), summary).map_err(|e| e.to_string())?;
    println!(
        "Tokamak closed-loop rollout experiment completed. Evidence: {}",
        output.display()
    );
    Ok(())
}

pub(super) fn execute(root: &Path, path: &Path, rounds: Option<u32>) -> Result<bool, String> {
    let text = fs::read_to_string(path).map_err(|e| e.to_string())?;
    let val: Value = baux::artifact::toml::from_str(&text).map_err(|e| e.to_string())?;
    let exp = val["experiment"].as_str().unwrap_or_default();
    if exp != "tokamak-tcv" && exp != "tokamak-tokagym" {
        return Ok(false);
    }

    let mut config: Config = baux::artifact::toml::from_str(&text).map_err(|e| e.to_string())?;
    if config.version != 4 {
        return Err(format!(
            "{exp} experiment requires version 4 tune-spec schema"
        ));
    }
    if let Some(r) = rounds {
        config.rounds = r as usize;
    }
    if config.rounds == 0 {
        return Err("rounds must be positive".into());
    }

    let commit = runs::jj(root, &["log", "-r", "@", "--no-graph", "-T", "commit_id"])?;
    let exe = std::env::current_exe().map_err(|e| e.to_string())?;
    let (output, resolved) = prepare_dir(root, &config)?;
    let dir = ckpt_dir(&output);

    let mut record = runs::start_worker(root, &output, &resolved, commit.trim(), &exe)?;
    record.model = if is_tokagym(&config) {
        "tokamak-tokagym-mlp-64x10".into()
    } else {
        "tokamak-tcv-mlp-93x19".into()
    };
    record.optimizer = opt_label(&config.optimizer);
    record.protocol = runs::digest(resolved.as_bytes());

    let result =
        run_opt(&config, Some(root), Some(&dir)).and_then(|rep| write_artifacts(&output, &rep));
    runs::finish(root, &mut record, &result)?;
    result?;
    Ok(true)
}
