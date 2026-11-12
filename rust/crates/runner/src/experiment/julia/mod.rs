//! Merkle-locked `Oceananigans.jl` and `FUSE.jl` physics experiment runner for
//! `baux tune`, backed by zero-copy page-aligned `ShmSlab` (`mmap`) IPC.

pub mod config;
pub mod driver;
pub mod label;
pub mod metal;
pub mod metal_step;
pub mod mode;
pub mod space;
pub mod worker;

#[cfg(test)]
mod tests;

use baux::artifact::json::Value;
use baux::artifact::runs;
use std::fs;
use std::path::Path;

use self::config::{Config, is_ocean};
use self::driver::run_opt;
use self::label::{model_label, opt_label};
use self::mode::resolve_mode;

fn write_artifacts(output: &Path, cbor_name: &str, report: &Value) -> Result<(), String> {
    let bytes = baux::artifact::cbor::to_vec(report).map_err(|e| e.to_string())?;
    fs::write(output.join(cbor_name), &bytes).map_err(|e| e.to_string())?;
    let summary = baux::artifact::json::to_string(report).map_err(|e| e.to_string())?;
    fs::write(output.join("summary.json"), summary).map_err(|e| e.to_string())
}

pub(crate) fn execute(root: &Path, path: &Path, rounds: Option<u32>) -> Result<bool, String> {
    let text = fs::read_to_string(path).map_err(|e| e.to_string())?;
    let val: Value = baux::artifact::toml::from_str(&text).map_err(|e| e.to_string())?;
    let exp = val["experiment"].as_str().unwrap_or_default();
    if exp != "oceananigans" && exp != "fuse" {
        return Ok(false);
    }

    let mut cfg: Config = baux::artifact::toml::from_str(&text).map_err(|e| e.to_string())?;
    if cfg.version != 4 {
        return Err(format!(
            "{exp} experiment requires version 4 tune-spec schema"
        ));
    }
    if let Some(r) = rounds {
        cfg.rounds = r as usize;
    }
    if cfg.rounds == 0 {
        return Err("rounds must be positive".into());
    }
    let mode = resolve_mode(&cfg)?;

    let commit = runs::jj(root, &["log", "-r", "@", "--no-graph", "-T", "commit_id"])?;
    let exe = std::env::current_exe().map_err(|e| e.to_string())?;
    let nonce = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_err(|e| e.to_string())?
        .as_nanos();
    let subdir = if is_ocean(&cfg) {
        "oceananigans"
    } else {
        "fuse"
    };
    let output = root
        .join(".cache/baux/runs")
        .join(subdir)
        .join(nonce.to_string());
    fs::create_dir_all(&output).map_err(|e| e.to_string())?;
    let resolved = baux::artifact::toml::pretty_string(&cfg).map_err(|e| e.to_string())?;
    fs::write(output.join("config.toml"), &resolved).map_err(|e| e.to_string())?;

    let mut record = runs::start_worker(root, &output, &resolved, commit.trim(), &exe)?;
    record.model = model_label(&cfg);
    record.optimizer = opt_label(&cfg, mode);
    record.protocol = runs::digest(resolved.as_bytes());

    let cbor_name = format!("{subdir}.cbor");
    let result = run_opt(&cfg, mode, root, &output).and_then(|report| {
        write_artifacts(&output, &cbor_name, &report)?;
        println!(
            "{} ({}) experiment completed. Evidence: {}",
            if is_ocean(&cfg) {
                "Oceananigans.jl"
            } else {
                "FUSE.jl"
            },
            opt_label(&cfg, mode),
            output.display()
        );
        Ok(())
    });
    runs::finish(root, &mut record, &result)?;
    result?;
    Ok(true)
}
