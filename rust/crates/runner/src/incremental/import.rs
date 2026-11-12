//! Legacy campaign importer for disk checkpoints and configs.

use std::fs;
use std::path::{Path, PathBuf};

use baux::artifact::cbor::from_slice;
use baux::artifact::inc::{IncStore, StepRecord, TrajHead, traj_key};
use baux::artifact::study::StudyCell;
use baux::artifact::toml::from_str;

use super::digest::{env_digest, opt_digest};
use crate::experiment::tokamak::ckpt::{LiveCkpt, SeedRecord};

pub fn score_loss(exp: &str, score: f64) -> f64 {
    if exp == "tokamak-tcv" {
        (100.0 - score).max(0.0)
    } else {
        (-score).max(0.0)
    }
}

fn live_steps(live: &LiveCkpt) -> Vec<StepRecord> {
    let n = live.completed_round.min(live.eval_scores.len());
    let mut out = Vec::with_capacity(n);
    let mut prev_best = f64::NEG_INFINITY;
    for r in 0..n {
        let eval_scores = live.eval_scores[r].clone();
        let eval_viol = live.eval_viols.get(r).copied().unwrap_or(0.0);
        let best_score = live.best_scores.get(r).copied().unwrap_or(live.best_score);
        let best_viol = live.best_viols.get(r).copied().unwrap_or(0.0);
        let accepted = best_score > prev_best;
        prev_best = best_score;
        out.push(StepRecord {
            eval_scores,
            eval_viol,
            best_score,
            best_viol,
            accepted,
            cost_us: 0,
        });
    }
    out
}

fn done_steps(done: &SeedRecord) -> Vec<StepRecord> {
    let mut out = Vec::with_capacity(done.best_scores.len());
    let mut prev_best = f64::NEG_INFINITY;
    for &sc in &done.best_scores {
        let accepted = sc > prev_best;
        prev_best = sc;
        out.push(StepRecord {
            eval_scores: vec![sc],
            eval_viol: 0.0,
            best_score: sc,
            best_viol: 0.0,
            accepted,
            cost_us: 0,
        });
    }
    out
}

fn import_cbor(
    store: &IncStore,
    t_key: &str,
    head_meta: (&str, &str, u32, &str),
    bytes: &[u8],
) -> Result<bool, String> {
    let (env_d, opt_d, rep, exp) = head_meta;
    if let Ok(done) = from_slice::<SeedRecord>(bytes) {
        let steps = done_steps(&done);
        let head = TrajHead {
            env_digest: env_d.to_string(),
            opt_digest: opt_d.to_string(),
            rep,
            seed: done.seed,
            num_dim: done.num_dim,
            completed_round: steps.len() as u32,
            has_eval_trace: false,
            initial_error: done.initial_error,
            best_score: done.final_score,
            total_accepts: done.total_accepts as u32,
            elapsed_s: done.elapsed_seconds,
            worker_s: 0.0,
        };
        store.append_steps(t_key, &head, 1, &steps)?;
        return Ok(true);
    }
    if let Ok(live) = from_slice::<LiveCkpt>(bytes) {
        let steps = live_steps(&live);
        let seed = baux::tune_seed(exp, "proposal", rep, 0);
        let head = TrajHead {
            env_digest: env_d.to_string(),
            opt_digest: opt_d.to_string(),
            rep,
            seed,
            num_dim: 0,
            completed_round: steps.len() as u32,
            has_eval_trace: !steps.is_empty(),
            initial_error: live.initial_error,
            best_score: live.best_score,
            total_accepts: live.total_accepts as u32,
            elapsed_s: live.elapsed_seconds,
            worker_s: live.worker_seconds,
        };
        store.append_steps(t_key, &head, 1, &steps)?;
        return Ok(true);
    }
    Ok(false)
}

fn find_ckpts(camp_dir: &Path, exp: &str, cid: &str) -> Option<PathBuf> {
    let p1 = camp_dir.join("checkpoints").join(exp).join(cid);
    if p1.is_dir() {
        return Some(p1);
    }
    let p2 = camp_dir.join("checkpoints").join(cid);
    if p2.is_dir() {
        return Some(p2);
    }
    None
}

fn import_ckpts(
    store: &IncStore,
    cell: &StudyCell,
    env_d: &str,
    opt_d: &str,
    ckpt_dir: &Path,
) -> Result<usize, String> {
    let mut imported = 0;
    for rep in 0..cell.repetitions {
        let t_key = traj_key(env_d, opt_d, rep);
        if let Ok(Some(existing)) = store.get_head(&t_key)
            && existing.completed_round >= cell.rounds
        {
            continue;
        }
        let done_p = ckpt_dir.join(format!("seed_{rep}_done.cbor"));
        let live_p = ckpt_dir.join(format!("seed_{rep}_live.cbor"));
        let bytes = if done_p.exists() {
            fs::read(&done_p).ok()
        } else if live_p.exists() {
            fs::read(&live_p).ok()
        } else {
            None
        };
        if let Some(b) = bytes {
            let meta = (env_d, opt_d, rep, cell.experiment.as_str());
            if import_cbor(store, &t_key, meta, &b)? {
                imported += 1;
            }
        }
    }
    Ok(imported)
}

fn load_cells(dir: &Path) -> Vec<(StudyCell, String, String)> {
    let mut out = Vec::new();
    let Some(entries) = fs::read_dir(dir).ok() else {
        return out;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some("toml") {
            continue;
        }
        let Some(stem) = path.file_stem().and_then(|s| s.to_str()) else {
            continue;
        };
        let (exp, cid) = match stem.split_once("__") {
            Some((e, c)) => (e.to_string(), c.to_string()),
            None => ("oceananigans".to_string(), stem.to_string()),
        };
        let Ok(raw) = fs::read_to_string(&path) else {
            continue;
        };
        let Ok(val) = from_str::<baux::artifact::json::Value>(&raw) else {
            continue;
        };
        let rounds = val.get("rounds").and_then(|v| v.as_u64()).unwrap_or(1000) as u32;
        let repetitions = val
            .get("repetitions")
            .and_then(|v| v.as_u64())
            .unwrap_or(10) as u32;
        let cell = StudyCell {
            env_id: exp.clone(),
            arm_id: cid.clone(),
            cell_id: format!("{exp}__{cid}"),
            baseline: cid.contains("turbo_zero"),
            factors: Vec::new(),
            experiment: exp.clone(),
            rounds,
            repetitions,
            canonical_toml: raw,
            value: val,
        };
        out.push((cell, exp, cid));
    }
    out
}

pub fn import_campaign(root: &Path, store: &IncStore, camp_dir: &Path) -> Result<usize, String> {
    let cfg_dir = camp_dir.join("configs");
    let specs_dir = camp_dir.join("specs");
    let items = if cfg_dir.is_dir() {
        load_cells(&cfg_dir)
    } else if specs_dir.is_dir() {
        load_cells(&specs_dir)
    } else {
        return Err(format!("no configs/ or specs/ in {}", camp_dir.display()));
    };

    let mut total_imported = 0;
    for (cell, exp, cid) in items {
        let env_d = env_digest(root, &cell)?;
        let opt_d = opt_digest(root, &cell)?;
        if let Some(ckpt_dir) = find_ckpts(camp_dir, &exp, &cid) {
            let n = import_ckpts(store, &cell, &env_d, &opt_d, &ckpt_dir)?;
            total_imported += n;
        }
    }
    println!(
        "Imported {} trajectory seeds from {}",
        total_imported,
        camp_dir.display()
    );
    Ok(total_imported)
}
