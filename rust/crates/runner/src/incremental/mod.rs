//! Incremental execution engine for study sweeps, micro-batching, and resume
//! memoization.

use std::fs;
use std::path::{Path, PathBuf};

use baux::artifact::inc::{IncStore, default_dir};
use baux::artifact::study::{CompileOpts, StudyPrune, compile_file, tune_schema};

pub mod broker;
pub mod digest;
pub mod engine;
pub mod import;
pub mod observer;
pub mod plot_web;
#[cfg(test)]
pub mod tests;

#[derive(Clone, Debug, Default)]
pub struct TuneRequest {
    pub config: PathBuf,
    pub rounds: Option<u32>,
    pub repetitions: Option<u32>,
    pub jobs: Option<usize>,
    pub cli_sets: Vec<String>,
    pub prune: Option<String>,
    pub plan_only: bool,
    pub plot: bool,
    pub import_campaign: Option<PathBuf>,
}

fn use_incremental(req: &TuneRequest, is_multi: bool, has_metal_backend: bool) -> bool {
    is_multi
        || !req.cli_sets.is_empty()
        || req.plan_only
        || req.repetitions.is_some()
        || req.jobs.is_some()
        || req.prune.is_some()
        || req.plot
        || has_metal_backend
}

fn is_supported(exp: &str) -> bool {
    matches!(
        exp,
        "tokamak-tokagym" | "tokamak-tcv" | "oceananigans" | "fuse"
    )
}

pub fn write_schema(root: &Path, out: Option<&Path>) -> Result<PathBuf, String> {
    let schema_json = tune_schema();
    let target = out
        .map(PathBuf::from)
        .unwrap_or_else(|| root.join("schemas/tune-spec.schema.json"));
    if let Some(parent) = target.parent() {
        fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    }
    fs::write(&target, schema_json).map_err(|e| e.to_string())?;
    Ok(target)
}

pub fn try_execute(root: &Path, req: &TuneRequest) -> Result<bool, String> {
    let store_dir = default_dir(root);
    let store = IncStore::open(&store_dir)?;

    if let Some(ref camp) = req.import_campaign {
        import::import_campaign(root, &store, camp)?;
        if req.config.as_os_str().is_empty() || !req.config.exists() {
            return Ok(true);
        }
    }

    if req.config.as_os_str().is_empty() || !req.config.exists() {
        return Ok(false);
    }

    let prune = match req.prune.as_deref() {
        Some(s) => Some(StudyPrune::parse_opt(s)?),
        None => None,
    };
    let opts = CompileOpts {
        cli_sets: &req.cli_sets,
        rounds: req.rounds,
        repetitions: req.repetitions,
        prune,
    };
    let Ok(study) = compile_file(&req.config, &opts) else {
        return Ok(false);
    };

    let all_supported = study.cells.iter().all(|c| is_supported(&c.experiment));
    if !all_supported {
        return Ok(false);
    }

    let has_metal = study
        .cells
        .iter()
        .any(|c| c.value.get("backend").and_then(|v| v.as_str()) == Some("resident-metal"));
    if !use_incremental(req, study.is_multi, has_metal) {
        return Ok(false);
    }

    if req.plan_only {
        engine::plan_execution(root, &store, &study)?;
        return Ok(true);
    }

    engine::run_study(root, &store, &study, req.jobs, req.plot)?;
    Ok(true)
}
