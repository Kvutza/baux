//! Tokamak TCV Policy Multi-Trust-Region experiment runner for `baux tune`.

use baux::artifact::{
    json::{self, Value, json},
    runs,
};
use baux::config::OptimizerSpec;
use baux::{BayesOptimizer, Prng, prng};
use deser::{Deserialize, Serialize};
use ndarray::Array2;
use std::{
    fs,
    io::{Read, Write},
    path::Path,
    process::{Child, ChildStdin, ChildStdout, Command, Stdio},
    time::Instant,
};

#[derive(Clone, Deserialize, Serialize)]
#[deser(deny_unknown_fields, rename_all = "kebab-case")]
struct Config {
    version: u32,
    experiment: String,
    rounds: usize,
    #[deser(default = default_reps())]
    repetitions: u32,
    optimizer: OptimizerSpec,
}

const fn default_reps() -> u32 {
    1
}

pub(super) fn execute(root: &Path, path: &Path, rounds: Option<u32>) -> Result<bool, String> {
    let text = fs::read_to_string(path).map_err(|e| e.to_string())?;
    let value: Value = baux::artifact::toml::from_str(&text).map_err(|e| e.to_string())?;
    let exp = value["experiment"].as_str().unwrap_or_default();
    if exp != "tokamak-tcv" && exp != "tokamak-tokagym" {
        return Ok(false);
    }

    let mut config: Config = baux::artifact::toml::from_str(&text).map_err(|e| e.to_string())?;
    if config.version != 4 {
        return Err("tokamak experiment requires version 4 tune-spec schema".into());
    }
    if let Some(rounds) = rounds {
        config.rounds = rounds as usize;
    }
    if config.rounds == 0 {
        return Err("rounds must be positive".into());
    }

    let commit = runs::jj(root, &["log", "-r", "@", "--no-graph", "-T", "commit_id"])?;
    let executable = std::env::current_exe().map_err(|e| e.to_string())?;
    let nonce = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_err(|e| e.to_string())?
        .as_nanos();
    let output = root
        .join(".cache/baux/runs/tokamak")
        .join(nonce.to_string());
    fs::create_dir_all(&output).map_err(|e| e.to_string())?;
    let resolved = baux::artifact::toml::pretty_string(&config).map_err(|e| e.to_string())?;
    fs::write(output.join("config.toml"), &resolved).map_err(|e| e.to_string())?;

    let mut record = runs::start_worker(root, &output, &resolved, commit.trim(), &executable)?;
    record.model = if is_tokagym(&config) {
        "tokagym-11-to-8".into()
    } else {
        "torax-2.0.0-tcv-93-to-19".into()
    };
    record.optimizer = if matches!(config.optimizer, OptimizerSpec::Ppo) {
        "candle-ppo-metal".into()
    } else {
        "multi-trust-region-bo".into()
    };
    record.protocol = runs::digest(resolved.as_bytes());

    let result = run_opt(&config, Some(root)).and_then(|report| {
        let report_str = json::pretty_string(&report).map_err(|e| e.to_string())?;
        fs::write(output.join("tokamak.json"), &report_str).map_err(|e| e.to_string())?;
        println!(
            "Tokamak Multi-Trust-Region BO experiment completed. Evidence: {}",
            output.display()
        );
        Ok(())
    });
    runs::finish(root, &mut record, &result)?;
    result?;
    Ok(true)
}

struct ToraxWorker {
    child: Child,
    stdin: ChildStdin,
    stdout: ChildStdout,
    obs: usize,
    act: usize,
}

impl ToraxWorker {
    fn spawn(
        root: &Path,
        module: &str,
        lock: &str,
        obs: usize,
        act: usize,
    ) -> Result<Self, String> {
        baux::artifact::merkle::verify_lock(root, lock, None)
            .map_err(|e| format!("{lock} lock: {e}"))?;
        let mut child = Command::new("uv")
            .args(["run", "python3", "-m", module, "--worker"])
            .current_dir(root)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .map_err(|e| format!("spawn {module}: {e}"))?;
        let stdin = child.stdin.take().ok_or("missing worker stdin")?;
        let stdout = child.stdout.take().ok_or("missing worker stdout")?;
        Ok(Self {
            child,
            stdin,
            stdout,
            obs,
            act,
        })
    }

    fn eval(&mut self, batch_weights: &[f64], count: usize) -> Result<Vec<f64>, String> {
        self.stdin
            .write_all(&(count as u32).to_le_bytes())
            .map_err(|e| e.to_string())?;
        let mut payload = Vec::with_capacity(batch_weights.len() * 4);
        for &w in batch_weights {
            payload.extend_from_slice(&(w as f32).to_le_bytes());
        }
        self.stdin.write_all(&payload).map_err(|e| e.to_string())?;
        self.stdin.flush().map_err(|e| e.to_string())?;
        let mut resp = vec![0u8; count * 8];
        self.stdout
            .read_exact(&mut resp)
            .map_err(|e| format!("worker read: {e}"))?;
        Ok(resp
            .chunks_exact(8)
            .map(|c| f64::from_le_bytes(c.try_into().unwrap()))
            .collect())
    }

    fn eval_traj(
        &mut self,
        weights: &[f64],
        eps: &[f32],
        steps: usize,
    ) -> Result<(f64, Vec<f32>, Vec<f32>, Vec<f32>), String> {
        let cmd = 0x8000_0001u32;
        self.stdin
            .write_all(&cmd.to_le_bytes())
            .map_err(|e| e.to_string())?;
        let mut payload = Vec::with_capacity((weights.len() + eps.len()) * 4);
        for &w in weights {
            payload.extend_from_slice(&(w as f32).to_le_bytes());
        }
        for &e in eps {
            payload.extend_from_slice(&e.to_le_bytes());
        }
        self.stdin.write_all(&payload).map_err(|e| e.to_string())?;
        self.stdin.flush().map_err(|e| e.to_string())?;

        let obs_len = steps * self.obs;
        let u_len = steps * self.act;
        let mut resp = vec![0u8; 8 + (obs_len + u_len + steps) * 4];
        self.stdout
            .read_exact(&mut resp)
            .map_err(|e| format!("worker traj read: {e}"))?;
        let score = f64::from_le_bytes(resp[..8].try_into().unwrap());
        let floats: Vec<f32> = resp[8..]
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        let obs = floats[..obs_len].to_vec();
        let u = floats[obs_len..obs_len + u_len].to_vec();
        let r = floats[obs_len + u_len..].to_vec();
        Ok((score, obs, u, r))
    }
}

impl Drop for ToraxWorker {
    fn drop(&mut self) {
        let _ = self.stdin.write_all(&0u32.to_le_bytes());
        let _ = self.stdin.flush();
        let _ = self.child.wait();
    }
}

fn eval_batch(
    worker: &mut Option<ToraxWorker>,
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

fn is_tokagym(cfg: &Config) -> bool {
    cfg.experiment == "tokamak-tokagym"
}

fn tracking_error(cfg: &Config, score: f64) -> f64 {
    if is_tokagym(cfg) {
        (-score).max(0.0)
    } else {
        (100.0 - score).max(0.0)
    }
}

fn exp_domain(cfg: &Config) -> baux::optimizer::DomainFrame {
    if is_tokagym(cfg) {
        baux::tokamak::tokagym_domain()
    } else {
        baux::tokamak::domain_frame()
    }
}

fn make_opt(config: &Config, rng: &mut Prng) -> Result<BayesOptimizer, String> {
    BayesOptimizer::from_domain(exp_domain(config), &config.optimizer, 0, rng)
        .map_err(|e| e.to_string())
}

fn seed_weights(seed: u64, config: &Config) -> Result<Vec<f64>, String> {
    let n_pert = if is_tokagym(config) { 13671 } else { 16053 };
    let ref_toml = format!(
        r#"
algorithm = "bayesian-search"
candidates = 2
arms = 1
acquisition = "random"
[perturbation]
sampler = "raasp"
num-pert = {n_pert}
[trust-region]
initial = 0.8
min = 0.00048828125
max = 1.6
[regional-search]
count = 10
sharing = "ellipsoid-ball"
allocation = "incumbent-ucb1"
controller = "turbo"
success-tolerance = 2
failure-tolerance = 3
restart-center = "maximin"
restart-candidates = 8
"#
    );
    let spec: OptimizerSpec =
        baux::artifact::toml::from_str(&ref_toml).map_err(|e| e.to_string())?;
    let mut rng = prng(seed);
    let domain = exp_domain(config);
    let mut opt = BayesOptimizer::from_domain(domain.clone(), &spec, 0, &mut rng)
        .map_err(|e| e.to_string())?;
    let unit = opt.ask(1, &mut rng).map_err(|e| e.to_string())?;
    let points = domain.from_unit(&unit.view());
    Ok(points
        .as_slice_memory_order()
        .ok_or("point non-contiguous")?
        .to_vec())
}

fn run_ppo(config: &Config, rep: u32, worker: &mut Option<ToraxWorker>) -> Result<SeedRun, String> {
    let seed = baux::tune_seed(&config.experiment, "proposal", rep, 0);
    let init_weights = seed_weights(seed, config)?;
    let mut rng = prng(seed ^ 0x5050_4f00);
    let mut agent = baux::tokamak::ppo::CandlePpo::new(&init_weights, &mut rng)?;
    let steps = 20usize;
    let total_rounds = config.rounds;
    let mut best_score = f64::NEG_INFINITY;
    let mut initial_error = 0.0;
    let mut total_accepts = 0usize;
    let mut best_scores = Vec::with_capacity(total_rounds);
    let start_time = Instant::now();

    let mut weights = agent.actor_weights()?;
    for round in 1..=total_rounds {
        let det = round == 1 || round % 8 == 1;
        let eps = agent.sample_noise(steps, det, &mut rng)?;
        let (score, obs, u, rewards) = match worker.as_mut() {
            Some(w) => w.eval_traj(&weights, &eps, steps)?,
            None => baux::tokamak::rollout_traj(&weights, &eps, steps),
        };
        if score > best_score {
            best_score = score;
            total_accepts += 1;
        }
        if round == 1 {
            initial_error = tracking_error(config, score);
        }
        if agent.record_step(obs, u, rewards)? {
            weights = agent.actor_weights()?;
        }
        best_scores.push(best_score);
    }

    let elapsed_seconds = start_time.elapsed().as_secs_f64();
    let final_error = tracking_error(config, best_score);
    let reduction_pct = if initial_error > 0.0 {
        ((initial_error - final_error) / initial_error * 100.0).max(0.0)
    } else {
        0.0
    };
    eprintln!(
        "tokamak.seed rep={}/{} seed={} final_score={:.6} final_error={:.6} reduction={:.2}% accepts={}/{} elapsed_s={:.2}",
        rep + 1,
        config.repetitions.max(1),
        seed,
        best_score,
        final_error,
        reduction_pct,
        total_accepts,
        total_rounds,
        elapsed_seconds
    );
    let p_dim = if is_tokagym(config) {
        baux::tokamak::TOKAGYM_PARAMS
    } else {
        baux::tokamak::TOTAL_PARAMS
    };
    Ok(SeedRun {
        rep,
        seed,
        num_dim: p_dim,
        initial_error,
        final_score: best_score,
        final_error,
        reduction_pct,
        total_accepts,
        elapsed_seconds,
        best_scores,
    })
}

struct SeedRun {
    rep: u32,
    seed: u64,
    num_dim: usize,
    initial_error: f64,
    final_score: f64,
    final_error: f64,
    reduction_pct: f64,
    total_accepts: usize,
    elapsed_seconds: f64,
    best_scores: Vec<f64>,
}

fn mean_sem(values: &[f64]) -> (f64, f64) {
    let n = values.len();
    if n == 0 {
        return (0.0, 0.0);
    }
    let mean = values.iter().sum::<f64>() / (n as f64);
    if n <= 1 {
        return (mean, 0.0);
    }
    let var = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / ((n - 1) as f64);
    (mean, (var / (n as f64)).sqrt())
}

fn run_seed(
    config: &Config,
    rep: u32,
    worker: &mut Option<ToraxWorker>,
) -> Result<SeedRun, String> {
    let seed = baux::tune_seed(&config.experiment, "proposal", rep, 0);
    let mut rng = prng(seed);
    let mut optimizer = make_opt(config, &mut rng)?;
    let num_dim = optimizer.num_dim();
    let arms = config
        .optimizer
        .candidate_generation()
        .map(|c| c.arms as usize)
        .unwrap_or(1);
    let total_rounds = config.rounds;
    let mut best_score = f64::NEG_INFINITY;
    let mut initial_error = 0.0;
    let mut total_accepts = 0usize;
    let mut best_scores = Vec::with_capacity(total_rounds);
    let start_time = Instant::now();

    let domain = exp_domain(config);
    for round in 1..=total_rounds {
        let unit = optimizer.ask(arms, &mut rng).map_err(|e| e.to_string())?;
        let points = domain.from_unit(&unit.view());
        let slice = points
            .as_slice_memory_order()
            .ok_or("point non-contiguous")?;
        let scores = eval_batch(worker, slice, arms)?;
        let score = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        if score > best_score {
            best_score = score;
            total_accepts += 1;
        }
        if round == 1 {
            initial_error = tracking_error(config, score);
        }
        let y = Array2::from_shape_fn((arms, 1), |(i, _)| scores[i]);
        optimizer
            .tell(&unit.view(), &y.view(), &mut rng)
            .map_err(|e| e.to_string())?;
        best_scores.push(best_score);
    }

    let elapsed_seconds = start_time.elapsed().as_secs_f64();
    let final_error = tracking_error(config, best_score);
    let reduction_pct = if initial_error > 0.0 {
        ((initial_error - final_error) / initial_error * 100.0).max(0.0)
    } else {
        0.0
    };
    eprintln!(
        "tokamak.seed rep={}/{} seed={} final_score={:.6} final_error={:.6} reduction={:.2}% accepts={}/{} elapsed_s={:.2}",
        rep + 1,
        config.repetitions.max(1),
        seed,
        best_score,
        final_error,
        reduction_pct,
        total_accepts,
        total_rounds,
        elapsed_seconds
    );
    Ok(SeedRun {
        rep,
        seed,
        num_dim,
        initial_error,
        final_score: best_score,
        final_error,
        reduction_pct,
        total_accepts,
        elapsed_seconds,
        best_scores,
    })
}

fn aggregate_traces(cfg: &Config, runs: &[SeedRun], total_rounds: usize) -> Vec<Value> {
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

fn spawn_worker(config: &Config, root: Option<&Path>) -> Result<Option<ToraxWorker>, String> {
    let Some(r) = root else {
        return Ok(None);
    };
    let (mod_name, lock, obs, act) = if is_tokagym(config) {
        (
            "benchmarks.tokagym.tokagym_rollout",
            "tokagym",
            baux::tokamak::TOKAGYM_OBS,
            baux::tokamak::TOKAGYM_ACT,
        )
    } else {
        (
            "benchmarks.torax.torax_rollout",
            "torax",
            baux::tokamak::OBS_DIM,
            baux::tokamak::ACTION_DIM,
        )
    };
    ToraxWorker::spawn(r, mod_name, lock, obs, act).map(Some)
}

fn run_opt(config: &Config, root: Option<&Path>) -> Result<Value, String> {
    let reps = config.repetitions.max(1);
    let regions = config
        .optimizer
        .regional_search()
        .map(|r| r.count)
        .unwrap_or(1);
    let mut worker = spawn_worker(config, root)?;
    let is_ppo = matches!(config.optimizer, OptimizerSpec::Ppo);
    eprintln!(
        "[tune] starting {} closed-loop {} | rounds={} regions={} repetitions={}",
        if is_tokagym(config) {
            "TokaGym"
        } else {
            "TORAX 2.0.0"
        },
        if is_ppo {
            "Candle PPO (Metal GPU)"
        } else {
            "multi-trust-region BO"
        },
        config.rounds,
        regions,
        reps
    );
    let runs: Vec<SeedRun> = (0..reps)
        .map(|rep| {
            if is_ppo {
                run_ppo(config, rep, &mut worker)
            } else {
                run_seed(config, rep, &mut worker)
            }
        })
        .collect::<Result<_, _>>()?;
    let num_dim = runs.first().map(|r| r.num_dim).unwrap_or(0);
    let init_errs: Vec<f64> = runs.iter().map(|r| r.initial_error).collect();
    let final_scores: Vec<f64> = runs.iter().map(|r| r.final_score).collect();
    let final_errs: Vec<f64> = runs.iter().map(|r| r.final_error).collect();
    let reductions: Vec<f64> = runs.iter().map(|r| r.reduction_pct).collect();
    let (init_mean, init_sem) = mean_sem(&init_errs);
    let (score_mean, score_sem) = mean_sem(&final_scores);
    let (err_mean, err_sem) = mean_sem(&final_errs);
    let (red_mean, red_sem) = mean_sem(&reductions);
    let elapsed: f64 = runs.iter().map(|r| r.elapsed_seconds).sum();
    let traces = aggregate_traces(config, &runs, config.rounds);
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
    Ok(json!({
        "experiment": exp_name,
        "task": task_name,
        "parameters": num_dim,
        "regions": regions,
        "rounds": config.rounds,
        "repetitions": reps,
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
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tokamak() {
        let toml_str = r#"
version = 4
experiment = "tokamak-tcv"
rounds = 2

[optimizer]
algorithm = "bayesian-search"
candidates = 2
arms = 1
acquisition = "random"

[optimizer.perturbation]
sampler = "raasp"
num-pert = 16053

[optimizer.trust-region]
initial = 0.8
min = 0.0078125
max = 1.6

[optimizer.regional-search]
count = 2
sharing = "ellipsoid-ball"
allocation = "incumbent-ucb1"
controller = "turbo"
success-tolerance = 3
failure-tolerance = 5
"#;
        let config: Config = baux::artifact::toml::from_str(toml_str).expect("parse v4 config");
        let report = run_opt(&config, None).expect("run tokamak multi-region opt");
        assert_eq!(report["experiment"], "tokamak-tcv");
        assert_eq!(report["regions"], 2);
        assert_eq!(report["rounds"], 2);

        let ppo_toml = r#"
version = 4
experiment = "tokamak-tcv"
rounds = 8

[optimizer]
algorithm = "ppo"
"#;
        let ppo_cfg: Config = baux::artifact::toml::from_str(ppo_toml).expect("parse ppo config");
        let ppo_rep = run_opt(&ppo_cfg, None).expect("run tokamak candle ppo");
        assert_eq!(ppo_rep["experiment"], "tokamak-tcv");
        assert_eq!(ppo_rep["rounds"], 8);
    }
}
