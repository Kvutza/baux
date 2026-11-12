//! Atomic mid-seed and per-seed checkpointing for crash-resilient long runs.

use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

use baux::BayesOptimizer;
use deser::{Deserialize, Serialize};
use ndarray::Array2;

use super::label::round_rng;

const INLINE_FREEZE: usize = 64 * 1024 * 1024;

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct SeedRecord {
    pub rep: u32,
    pub seed: u64,
    pub num_dim: usize,
    pub initial_error: f64,
    pub final_score: f64,
    pub final_error: f64,
    pub reduction_pct: f64,
    pub total_accepts: usize,
    pub elapsed_seconds: f64,
    pub best_scores: Vec<f64>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct LiveCkpt {
    pub completed_round: usize,
    pub best_score: f64,
    pub initial_error: f64,
    pub total_accepts: usize,
    pub elapsed_seconds: f64,
    pub best_scores: Vec<f64>,
    #[deser(default)]
    pub eval_scores: Vec<Vec<f64>>,
    #[deser(default)]
    pub eval_viols: Vec<f64>,
    #[deser(default)]
    pub best_viols: Vec<f64>,
    #[deser(default)]
    pub worker_seconds: f64,
}

impl LiveCkpt {
    pub fn empty(rounds: usize) -> Self {
        Self {
            completed_round: 0,
            best_score: f64::NEG_INFINITY,
            initial_error: 0.0,
            total_accepts: 0,
            elapsed_seconds: 0.0,
            best_scores: Vec::with_capacity(rounds),
            eval_scores: Vec::with_capacity(rounds),
            eval_viols: Vec::new(),
            best_viols: Vec::new(),
            worker_seconds: 0.0,
        }
    }
}

fn atomic_write(path: &Path, bytes: &[u8]) -> Result<(), String> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    }
    let tmp = path.with_extension(format!("tmp.{}", std::process::id()));
    let mut file = fs::File::create(&tmp).map_err(|e| e.to_string())?;
    file.write_all(bytes).map_err(|e| e.to_string())?;
    file.sync_all().map_err(|e| e.to_string())?;
    drop(file);
    fs::rename(&tmp, path).map_err(|e| e.to_string())
}

pub fn ckpt_dir(output: &Path) -> PathBuf {
    if let Ok(dir) = std::env::var("BAUX_TUNE_CKPT_DIR")
        && !dir.trim().is_empty()
    {
        return PathBuf::from(dir);
    }
    output.join("checkpoints")
}

pub fn ckpt_interval() -> usize {
    std::env::var("BAUX_CKPT_EVERY")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .filter(|&v| v > 0)
        .unwrap_or(100)
}

pub fn rep_indices(repetitions: u32) -> Result<Vec<u32>, String> {
    let reps = repetitions.max(1);
    if let Ok(raw) = std::env::var("BAUX_TUNE_REP")
        && !raw.trim().is_empty()
    {
        let rep: u32 = raw
            .trim()
            .parse()
            .map_err(|_| format!("invalid BAUX_TUNE_REP={raw:?}"))?;
        if rep >= reps {
            return Err(format!("BAUX_TUNE_REP={rep} out of 0..{reps}"));
        }
        return Ok(vec![rep]);
    }
    Ok((0..reps).collect())
}

pub fn load_done(dir: &Path, rep: u32) -> Option<SeedRecord> {
    let path = dir.join(format!("seed_{rep}_done.cbor"));
    let bytes = fs::read(path).ok()?;
    baux::artifact::cbor::from_slice(&bytes).ok()
}

pub fn save_done(dir: &Path, rec: &SeedRecord) -> Result<(), String> {
    let bytes = baux::artifact::cbor::to_vec(rec).map_err(|e| e.to_string())?;
    atomic_write(&dir.join(format!("seed_{}_done.cbor", rec.rep)), &bytes)?;
    let _ = fs::remove_file(dir.join(format!("seed_{}_live.cbor", rec.rep)));
    let _ = fs::remove_file(dir.join(format!("seed_{}_opt.bin", rec.rep)));
    Ok(())
}

pub fn save_meta(dir: &Path, rep: u32, state: &LiveCkpt) -> Result<(), String> {
    let meta_bytes = baux::artifact::cbor::to_vec(state).map_err(|e| e.to_string())?;
    atomic_write(&dir.join(format!("seed_{rep}_live.cbor")), &meta_bytes)
}

pub fn load_meta(dir: &Path, rep: u32) -> Result<Option<LiveCkpt>, String> {
    let meta_path = dir.join(format!("seed_{rep}_live.cbor"));
    if !meta_path.exists() {
        return Ok(None);
    }
    let meta_bytes = fs::read(&meta_path).map_err(|e| e.to_string())?;
    let state: LiveCkpt =
        baux::artifact::cbor::from_slice(&meta_bytes).map_err(|e| e.to_string())?;
    if state.eval_scores.len() == state.completed_round {
        return Ok(Some(state));
    }
    Ok(None)
}

pub fn save_live(
    dir: &Path,
    rep: u32,
    opt: &BayesOptimizer,
    state: &LiveCkpt,
) -> Result<(), String> {
    let opt_path = dir.join(format!("seed_{rep}_opt.bin"));
    let est_bytes = opt
        .num_dim()
        .saturating_mul(state.completed_round)
        .saturating_mul(8);
    if est_bytes <= INLINE_FREEZE
        && let Ok(opt_bytes) = opt.freeze()
        && opt_bytes.len() <= INLINE_FREEZE
    {
        atomic_write(&opt_path, &opt_bytes)?;
    } else {
        let _ = fs::remove_file(&opt_path);
    }
    save_meta(dir, rep, state)
}

fn replay_optimizer(
    mut opt: BayesOptimizer,
    seed: u64,
    eval_scores: &[Vec<f64>],
) -> Result<BayesOptimizer, String> {
    for (idx, scores) in eval_scores.iter().enumerate() {
        let arms = scores.len().max(1);
        let mut step_rng = round_rng(seed, idx + 1);
        let unit = opt.ask(arms, &mut step_rng).map_err(|e| e.to_string())?;
        let y = Array2::from_shape_fn((arms, 1), |(i, _)| scores[i]);
        opt.tell(&unit.view(), &y.view(), &mut step_rng)
            .map_err(|e| e.to_string())?;
    }
    Ok(opt)
}

pub fn load_live(
    dir: &Path,
    rep: u32,
    seed: u64,
    template: BayesOptimizer,
) -> Result<Option<(BayesOptimizer, LiveCkpt)>, String> {
    let meta_path = dir.join(format!("seed_{rep}_live.cbor"));
    if !meta_path.exists() {
        return Ok(None);
    }
    let meta_bytes = fs::read(&meta_path).map_err(|e| e.to_string())?;
    let state: LiveCkpt =
        baux::artifact::cbor::from_slice(&meta_bytes).map_err(|e| e.to_string())?;
    let opt_path = dir.join(format!("seed_{rep}_opt.bin"));
    if opt_path.exists()
        && let Ok(opt_bytes) = fs::read(&opt_path)
        && let Ok(restored) = template.thaw(&opt_bytes)
    {
        return Ok(Some((restored, state)));
    }
    if state.eval_scores.len() == state.completed_round {
        let replayed = replay_optimizer(template, seed, &state.eval_scores)?;
        return Ok(Some((replayed, state)));
    }
    Ok(None)
}
