//! Record native model/runtime integration with the ordinary tune provenance.
use baux::artifact::{json::Value, runs};
use baux_tooling::build::BuildSystem;
use std::{fs, path::Path, process::Command};

pub(super) fn execute(
    root: &Path,
    path: &Path,
    rounds: Option<u32>,
    engine: BuildSystem,
) -> Result<bool, String> {
    let text = fs::read_to_string(path).map_err(|e| e.to_string())?;
    let mut config: Value = baux::artifact::toml::from_str(&text).map_err(|e| e.to_string())?;
    if !matches!(
        config["experiment"].as_str(),
        Some("resident-program" | "resident-forward")
    ) {
        return Ok(false);
    }
    if let Some(rounds) = rounds
        && config["experiment"].as_str() == Some("resident-program")
    {
        config["rounds"] = rounds.into();
    }
    let resolved = baux::artifact::toml::pretty_string(&config).map_err(|e| e.to_string())?;
    // Snapshot without moving this workspace or any bookmark.
    let revision = runs::jj(root, &["log", "-r", "@", "--no-graph", "-T", "commit_id"])?;
    let revision = revision.trim();
    let worker = engine.build_artifact(root, "//rust/crates/fbt:baux-tune-worker")?;
    let numerical = baux_tooling::numerical::build(root)?;
    let numerical_root = root.join(".cache/baux/numerical");
    let compiler =
        fs::read_to_string(root.join("research/lean/lean-toolchain")).map_err(|e| e.to_string())?;
    let receipt = baux::artifact::numerical::prepare(
        &numerical_root,
        &numerical,
        &baux::artifact::numerical::Provenance {
            revision: revision.into(),
            compiler: compiler.trim().into(),
        },
    )?;
    if !runs::jj(
        root,
        &[
            "diff",
            "--from",
            revision,
            "--to",
            "@",
            "--summary",
            "--no-pager",
        ],
    )?
    .trim()
    .is_empty()
    {
        return Err(
            "source changed while building; rerun tune to capture the current snapshot".into(),
        );
    }
    let nonce = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_err(|e| e.to_string())?
        .as_nanos();
    let output = root
        .join(".cache/baux/runs/program")
        .join(nonce.to_string());
    fs::create_dir_all(&output).map_err(|e| e.to_string())?;
    let config_path = output.join("config.toml");
    fs::write(&config_path, &resolved).map_err(|e| e.to_string())?;
    let mut record = runs::start_worker(root, &output, &resolved, revision, &worker.path)?;
    record.model = config["model"].as_str().unwrap_or("unknown").into();
    record.optimizer = config["experiment"]
        .as_str()
        .unwrap_or("resident-program")
        .into();
    record.protocol = runs::digest(resolved.as_bytes());
    let mut command = Command::new(&worker.path);
    command
        .current_dir(root)
        .arg("--program")
        .arg(config_path)
        .arg(&output)
        .env("BAUX_SOURCE_REVISION", revision)
        .env("BAUX_NUMERICAL_ROOT", numerical_root)
        .env("BAUX_NUMERICAL_RECEIPT", receipt.id);
    let result = crate::execute(command);
    runs::finish(root, &mut record, &result)?;
    result?;
    println!("Resident program evidence: {}", output.display());
    Ok(true)
}
