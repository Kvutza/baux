//! Semantic content-addressed digests for incremental memoization.

use std::collections::BTreeMap;
use std::path::Path;
use std::sync::{Mutex, OnceLock};

use baux::artifact::blake3::hash_hex;
use baux::artifact::merkle::compute_tree;
use baux::artifact::runs::file_digest;
use baux::artifact::study::StudyCell;
use baux::artifact::toml::{from_value, pretty_string};
use baux::config::OptimizerSpec;

static CODE_DIGEST: OnceLock<String> = OnceLock::new();
static TREE_CACHE: OnceLock<Mutex<BTreeMap<String, String>>> = OnceLock::new();

pub fn hash_halves(x: &[u16]) -> String {
    let mut bytes = Vec::with_capacity(x.len() * 2);
    for &val in x {
        bytes.extend_from_slice(&val.to_le_bytes());
    }
    hash_hex(&bytes)
}

fn cached_tree(path: &Path) -> Result<String, String> {
    let key = path.to_string_lossy().to_string();
    let cache = TREE_CACHE.get_or_init(|| Mutex::new(BTreeMap::new()));
    if let Ok(guard) = cache.lock()
        && let Some(hit) = guard.get(&key)
    {
        return Ok(hit.clone());
    }
    let tree = compute_tree(path)?;
    let val = tree.root_blake3;
    if let Ok(mut guard) = cache.lock() {
        guard.insert(key, val.clone());
    }
    Ok(val)
}

fn code_digest(root: &Path) -> Result<String, String> {
    if let Ok(tag) = std::env::var("BAUX_OPT_CODE_TAG")
        && !tag.is_empty()
    {
        return Ok(tag);
    }
    if cfg!(test) {
        return Ok("test-v1".to_string());
    }
    if let Some(cached) = CODE_DIGEST.get() {
        return Ok(cached.clone());
    }
    let res_dir = root.join("rust/crates/baux/src/resident_metal");
    let tree_d = cached_tree(&res_dir)?;
    let spec_path = root.join("rust/crates/baux/src/config/spec/optimizer.rs");
    let file_d = file_digest(&spec_path)?;
    let combined = format!("{tree_d}:{file_d}");
    let digest = hash_hex(combined.as_bytes());
    let _ = CODE_DIGEST.set(digest.clone());
    Ok(digest)
}

fn format_subtable(cell: &StudyCell, key: &str) -> String {
    cell.value
        .get(key)
        .and_then(|v| pretty_string(v).ok())
        .map(|s| format!("[{key}]\n{s}\n"))
        .unwrap_or_default()
}

pub fn opt_digest(root: &Path, cell: &StudyCell) -> Result<String, String> {
    let opt_val = cell
        .value
        .get("optimizer")
        .cloned()
        .unwrap_or_else(baux::artifact::json::Value::null);
    let opt_spec: OptimizerSpec =
        from_value(opt_val).map_err(|e| format!("parse OptimizerSpec: {e}"))?;
    let opt_str = pretty_string(&opt_spec).map_err(|e| format!("serialize OptimizerSpec: {e}"))?;

    let mut buf = format!("optimizer:\n{opt_str}\n");
    let backend = cell
        .value
        .get("backend")
        .and_then(|v| v.as_str())
        .unwrap_or("default");
    buf.push_str(&format!("backend = {backend}\n"));
    buf.push_str(&format_subtable(cell, "morbo"));
    buf.push_str(&format_subtable(cell, "manifold"));
    let code_d = code_digest(root)?;
    buf.push_str(&format!("code_digest = {code_d}\n"));
    Ok(hash_hex(buf.as_bytes()))
}

fn tokamak_digest(root: &Path, cell: &StudyCell) -> Result<String, String> {
    let evals_dir = if cell.experiment == "tokamak-tokagym" {
        root.join("evals/tokagym")
    } else {
        root.join("evals/torax")
    };
    let sim_tree = cached_tree(&evals_dir)?;
    let d_py = file_digest(&root.join("evals/python/tokamak_daemon.py"))?;
    let d_shm = file_digest(&root.join("evals/python/shm_ipc.py"))?;
    let lock_path = root.join("research/upstream/tokagym.lock.toml");
    let d_lock = if lock_path.exists() {
        file_digest(&lock_path)?
    } else {
        String::new()
    };
    let sim_table = if cell.value.get("tokagym").is_some() {
        format_subtable(cell, "tokagym")
    } else {
        format_subtable(cell, "torax")
    };
    let payload = format!(
        "{}:{}:{}:{}:{}:{}",
        cell.experiment, sim_tree, d_py, d_shm, d_lock, sim_table
    );
    Ok(hash_hex(payload.as_bytes()))
}

fn julia_digest(root: &Path, cell: &StudyCell) -> Result<String, String> {
    let proj = if cell.experiment == "oceananigans" {
        root.join("evals/oceananigans")
    } else {
        root.join("evals/fuse")
    };
    let sim_tree = cached_tree(&proj)?;
    let sim_table = if cell.experiment == "oceananigans" {
        format_subtable(cell, "ocean")
    } else {
        format_subtable(cell, "fuse")
    };
    let payload = format!("{}:{}:{}", cell.experiment, sim_tree, sim_table);
    Ok(hash_hex(payload.as_bytes()))
}

pub fn env_digest(root: &Path, cell: &StudyCell) -> Result<String, String> {
    if std::env::var("BAUX_INCREMENTAL_SYNTHETIC").as_deref() == Ok("1") {
        return Ok(format!("synthetic:{}", cell.experiment));
    }
    if cell.experiment.starts_with("tokamak") {
        tokamak_digest(root, cell)
    } else {
        julia_digest(root, cell)
    }
}
