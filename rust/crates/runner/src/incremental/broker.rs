//! Deduplicated Singleflight and vectorized ShmSlab micro-batch broker.

use std::collections::BTreeMap;
use std::path::Path;
use std::sync::Mutex;

use baux::artifact::inc::{EvalOutcome, IncStore};
use baux::artifact::study::StudyCell;
use baux::artifact::toml::from_str;

use super::digest::hash_halves;
use crate::experiment::julia::config::Config as JuliaConfig;
use crate::experiment::julia::worker::JuliaWorker;
use crate::experiment::tokamak::worker::{TokamakWorker, WorkerSpec};

pub struct EvalBroker {
    pub jobs: usize,
    pub experiment: String,
}

pub fn default_jobs(experiment: &str) -> usize {
    let p_cores = perf_cores().unwrap_or(4);
    if experiment.starts_with("tokamak-tokagym") {
        (p_cores / 2).clamp(1, 8)
    } else if experiment.starts_with("oceananigans") || experiment.starts_with("tokamak-tcv") {
        (p_cores / 2).clamp(1, 6)
    } else if experiment.starts_with("fuse") {
        (p_cores / 3).clamp(1, 4)
    } else {
        (p_cores / 2).clamp(1, 4)
    }
}

fn perf_cores() -> Option<usize> {
    #[cfg(target_os = "macos")]
    {
        let out = std::process::Command::new("sysctl")
            .args(["-n", "hw.perflevel0.logicalcpu"])
            .output()
            .ok()?;
        if out.status.success() {
            let s = std::str::from_utf8(&out.stdout).ok()?;
            return s.trim().parse::<usize>().ok();
        }
    }
    None
}

fn eval_synthetic(misses: &[(String, Vec<u16>)]) -> Result<Vec<(String, EvalOutcome)>, String> {
    let mut out = Vec::with_capacity(misses.len());
    for (h, x) in misses {
        let mut sum_sq = 0.0f64;
        for &v in x {
            let norm = (v as f64 - 32768.0) / 32768.0;
            sum_sq += norm * norm;
        }
        out.push((
            h.clone(),
            EvalOutcome {
                scores: vec![-sum_sq],
                viol: 0.0,
                cost_us: 10,
            },
        ));
    }
    Ok(out)
}

fn tokamak_spec(is_tokagym: bool) -> (WorkerSpec, Vec<f64>, Vec<f64>) {
    let (domain, spec) = if is_tokagym {
        (
            baux::tokamak::tokagym_domain(),
            WorkerSpec {
                kind: "tokagym",
                lock: "tokagym",
                arms: 16,
                num_dim: baux::tokamak::TOKAGYM_PARAMS,
                obs: baux::tokamak::TOKAGYM_OBS,
                act: baux::tokamak::TOKAGYM_ACT,
                steps: 50,
            },
        )
    } else {
        (
            baux::tokamak::domain_frame(),
            WorkerSpec {
                kind: "torax",
                lock: "torax",
                arms: 16,
                num_dim: baux::tokamak::TOTAL_PARAMS,
                obs: baux::tokamak::OBS_DIM,
                act: baux::tokamak::ACTION_DIM,
                steps: 20,
            },
        )
    };
    let n = domain.num_dim();
    let low: Vec<f64> = (0..n).map(|i| domain.bounds()[[i, 0]]).collect();
    let span: Vec<f64> = (0..n)
        .map(|i| domain.bounds()[[i, 1]] - domain.bounds()[[i, 0]])
        .collect();
    (spec, low, span)
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
fn tokamak_worker(
    root: &Path,
    worker_id: usize,
    spec: WorkerSpec,
    low: &[f64],
    span: &[f64],
    chunks: &[&[(String, Vec<u16>)]],
    out_list: &Mutex<Vec<(String, EvalOutcome)>>,
) -> Result<(), String> {
    let mut worker = TokamakWorker::connect_slot(root, spec, Some(worker_id))?;
    for chunk in chunks {
        let rows: Vec<&[u16]> = chunk.iter().map(|(_, x)| x.as_slice()).collect();
        let (scores, cost_us) = worker.eval_halves(&rows, low, span)?;
        let mut local = Vec::with_capacity(chunk.len());
        for ((h, _), score) in chunk.iter().zip(scores) {
            local.push((
                h.clone(),
                EvalOutcome {
                    scores: vec![score],
                    viol: 0.0,
                    cost_us,
                },
            ));
        }
        if let Ok(mut g) = out_list.lock() {
            g.extend(local);
        }
    }
    Ok(())
}

fn julia_worker(
    root: &Path,
    worker_id: usize,
    cfg: &JuliaConfig,
    items: &[&(String, Vec<u16>)],
    out_list: &Mutex<Vec<(String, EvalOutcome)>>,
) -> Result<(), String> {
    let mut worker = JuliaWorker::connect_slot(root, cfg, Some(worker_id))?;
    let mut local = Vec::with_capacity(items.len());
    for (h, x) in items {
        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        {
            let (scores, viol, dt_us) = worker.eval_half(x.as_ptr())?;
            local.push((
                h.clone(),
                EvalOutcome {
                    scores,
                    viol,
                    cost_us: dt_us,
                },
            ));
        }
        #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
        {
            let _ = (worker, x);
            return Err("eval_half requires macOS aarch64".into());
        }
    }
    if let Ok(mut g) = out_list.lock() {
        g.extend(local);
    }
    Ok(())
}

fn partition_items<'a, T>(items: &'a [T], worker_id: usize, num_workers: usize) -> Vec<&'a T> {
    let mut out = Vec::new();
    for (i, item) in items.iter().enumerate() {
        if i % num_workers == worker_id {
            out.push(item);
        }
    }
    out
}

impl EvalBroker {
    pub fn new(experiment: &str, jobs_override: Option<usize>) -> Self {
        let jobs = jobs_override.unwrap_or_else(|| default_jobs(experiment));
        Self {
            jobs,
            experiment: experiment.to_string(),
        }
    }

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    fn eval_tokamak(
        &self,
        root: &Path,
        _cell: &StudyCell,
        misses: &[(String, Vec<u16>)],
    ) -> Result<Vec<(String, EvalOutcome)>, String> {
        let is_tokagym = self.experiment == "tokamak-tokagym";
        let (spec, low, span) = tokamak_spec(is_tokagym);
        let chunks: Vec<&[(String, Vec<u16>)]> = misses.chunks(16).collect();
        let num_workers = self.jobs.min(chunks.len()).max(1);
        let out_list = Mutex::new(Vec::with_capacity(misses.len()));

        std::thread::scope(|s| -> Result<(), String> {
            let mut handles = Vec::new();
            for worker_id in 0..num_workers {
                let worker_chunks = partition_items(&chunks, worker_id, num_workers);
                let sp = spec.clone();
                let lw = &low;
                let sp_dim = &span;
                let out_ref = &out_list;
                handles.push(s.spawn(move || {
                    let owned_chunks: Vec<&[(String, Vec<u16>)]> =
                        worker_chunks.into_iter().copied().collect();
                    tokamak_worker(root, worker_id, sp, lw, sp_dim, &owned_chunks, out_ref)
                }));
            }
            for h in handles {
                h.join()
                    .map_err(|_| "worker thread panicked".to_string())??;
            }
            Ok(())
        })?;
        let res = out_list.into_inner().unwrap_or_default();
        Ok(res)
    }

    #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
    fn eval_tokamak(
        &self,
        _root: &Path,
        _cell: &StudyCell,
        misses: &[(String, Vec<u16>)],
    ) -> Result<Vec<(String, EvalOutcome)>, String> {
        eval_synthetic(misses)
    }

    fn eval_julia(
        &self,
        root: &Path,
        cell: &StudyCell,
        misses: &[(String, Vec<u16>)],
    ) -> Result<Vec<(String, EvalOutcome)>, String> {
        let cfg: JuliaConfig = from_str(&cell.canonical_toml).map_err(|e| e.to_string())?;
        let num_workers = self.jobs.min(misses.len()).max(1);
        let out_list = Mutex::new(Vec::with_capacity(misses.len()));

        std::thread::scope(|s| -> Result<(), String> {
            let mut handles = Vec::new();
            for worker_id in 0..num_workers {
                let items_for_worker = partition_items(misses, worker_id, num_workers);
                let c_ref = &cfg;
                let out_ref = &out_list;
                handles.push(s.spawn(move || {
                    julia_worker(root, worker_id, c_ref, &items_for_worker, out_ref)
                }));
            }
            for h in handles {
                h.join()
                    .map_err(|_| "julia worker panicked".to_string())??;
            }
            Ok(())
        })?;
        let res = out_list.into_inner().unwrap_or_default();
        Ok(res)
    }

    pub fn evaluate_batch(
        &self,
        root: &Path,
        store: &IncStore,
        env_digest: &str,
        cell: &StudyCell,
        proposals: &[(usize, Vec<u16>)],
    ) -> Result<Vec<(usize, EvalOutcome)>, String> {
        let mut hash_map: BTreeMap<String, (Vec<u16>, Vec<usize>)> = BTreeMap::new();
        for &(idx, ref x) in proposals {
            let h = hash_halves(x);
            hash_map
                .entry(h)
                .or_insert_with(|| (x.clone(), Vec::new()))
                .1
                .push(idx);
        }
        let unique_hashes: Vec<String> = hash_map.keys().cloned().collect();
        let cached = store.get_evals(env_digest, &unique_hashes)?;
        let mut misses: Vec<(String, Vec<u16>)> = Vec::new();
        let mut outcomes: BTreeMap<String, EvalOutcome> = BTreeMap::new();

        for (h, maybe_hit) in unique_hashes.iter().zip(cached) {
            if let Some(mut outcome) = maybe_hit {
                outcome.cost_us = 0;
                outcomes.insert(h.clone(), outcome);
            } else if let Some((x, _)) = hash_map.get(h) {
                misses.push((h.clone(), x.clone()));
            }
        }

        if !misses.is_empty() {
            let new_entries = if env_digest.starts_with("synthetic")
                || std::env::var("BAUX_INCREMENTAL_SYNTHETIC").as_deref() == Ok("1")
            {
                eval_synthetic(&misses)?
            } else if self.experiment.starts_with("tokamak") {
                self.eval_tokamak(root, cell, &misses)?
            } else {
                self.eval_julia(root, cell, &misses)?
            };
            store.put_evals(env_digest, &new_entries)?;
            for (h, outcome) in new_entries {
                outcomes.insert(h, outcome);
            }
        }

        let mut results = Vec::with_capacity(proposals.len());
        for (h, (_, indices)) in &hash_map {
            let outcome = outcomes
                .get(h)
                .ok_or_else(|| format!("missing outcome for {h}"))?;
            for &f_idx in indices {
                results.push((f_idx, outcome.clone()));
            }
        }
        results.sort_by_key(|r| r.0);
        Ok(results)
    }
}
