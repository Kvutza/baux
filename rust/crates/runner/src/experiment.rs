use sha2::{Digest, Sha256};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;

use baux_tooling::terminal::{self, Tag};

use baux_tooling::build::BuildSystem;

mod pretrain;

pub(crate) fn run(
    root: &Path,
    path: &Path,
    prepare: bool,
    rounds: Option<u32>,
    engine: BuildSystem,
) -> Result<(), String> {
    let prepare_started = Instant::now();
    terminal::stderr(
        Tag::Orchestration,
        "prepare · resolve configuration and corpus",
    );
    let resolved = resolve(root, path, rounds, engine)?;
    terminal::stderr(
        Tag::Orchestration,
        &format!(
            "prepare · ready · {:.1}s",
            prepare_started.elapsed().as_secs_f64()
        ),
    );
    if prepare {
        terminal::stderr(Tag::Orchestration, "pretraining inputs are ready");
        return Ok(());
    }

    let revision = seal_source(root)?;
    terminal::stderr(
        Tag::Orchestration,
        &format!("snapshot · {}", &revision[..12]),
    );
    let build_started = Instant::now();
    terminal::stderr(Tag::Orchestration, "build · experiment runner and worker");
    let worker = engine.build_artifact(root, "//rust/crates/baux:baux-tune-worker")?;
    let runner = engine.build_artifact(root, "//rust/crates/baux:baux-tune")?;
    let numerical = baux_tooling::numerical::build(root)?;
    let numerical_root = root.join(".cache/baux/numerical");
    let compiler = fs::read_to_string(root.join("research/lean/lean-toolchain"))
        .map_err(|error| format!("read Lean toolchain: {error}"))?;
    let numerical = baux::artifact::numerical::prepare(
        &numerical_root,
        &numerical,
        &baux::artifact::numerical::Provenance {
            revision: revision.clone(),
            compiler: compiler.trim().into(),
        },
    )?;
    let changed = baux::artifact::runs::jj(
        root,
        &[
            "diff",
            "--from",
            &revision,
            "--to",
            "@",
            "--summary",
            "--no-pager",
        ],
    )?;
    if !changed.trim().is_empty() {
        return Err(
            "source changed while building; rerun tune to capture the current snapshot".into(),
        );
    }
    terminal::stderr(
        Tag::Orchestration,
        &format!(
            "build · ready · {:.1}s",
            build_started.elapsed().as_secs_f64()
        ),
    );
    terminal::stderr(Tag::Orchestration, "launch · pretraining");
    let mut command = Command::new(runner.path);
    command.current_dir(root).arg(resolved);
    command.env("BAUX_WORKER_EXECUTABLE", worker.path);
    command.env("BAUX_SOURCE_REVISION", revision);
    command.env("BAUX_NUMERICAL_ROOT", numerical_root);
    command.env("BAUX_NUMERICAL_RECEIPT", numerical.id);
    crate::execute(command)
}

pub(crate) fn resolve(
    root: &Path,
    path: &Path,
    rounds: Option<u32>,
    engine: BuildSystem,
) -> Result<PathBuf, String> {
    let source = path.canonicalize().map_err(|error| error.to_string())?;
    let parent = source.parent().ok_or("config has no parent directory")?;
    let text = fs::read_to_string(&source).map_err(|error| error.to_string())?;

    if let Ok(control_cfg) = baux::artifact::toml::from_str::<baux::artifact::json::Value>(&text) {
        if control_cfg
            .get("experiment")
            .and_then(|e| e.get("env"))
            .is_some()
        {
            return Ok(source);
        }
    }

    let mut spec = baux::config::ExperimentSpec::parse(&text)?;
    if spec.experiment != baux::ExperimentKind::Pretrain {
        return Err("tune accepts version-4 pretraining experiments".into());
    }
    validate_authored(&spec)?;
    if let Some(generation) = &mut spec.generation {
        generation.resolve(parent)?;
    }
    pretrain::prepare(root, &spec, engine)?;
    pretrain::resolve(root, &mut spec)?;
    if let Some(rounds) = rounds {
        spec.execution.rounds = rounds;
    } else if spec.execution.rounds < 128 {
        return Err("pretraining experiments require at least 128 rounds".into());
    }
    resolved_config(root, &mut spec)
}

fn validate_authored(spec: &baux::config::ExperimentSpec) -> Result<(), String> {
    if spec.output.is_some() || spec.data.train.is_some() || spec.data.validation.is_some() {
        return Err(
            "authored pretraining configs select data.recipe; output, data.train, and data.validation exist only in frozen run records"
                .into(),
        );
    }
    Ok(())
}

fn seal_source(root: &Path) -> Result<String, String> {
    let revision = Command::new("jj")
        .current_dir(root)
        .args(["log", "-r", "@", "--no-graph", "-T", "commit_id"])
        .output()
        .map_err(|error| format!("read JJ revision: {error}"))?;
    if !revision.status.success() {
        return Err("read JJ revision: command failed".into());
    }
    let revision = String::from_utf8(revision.stdout)
        .map_err(|error| format!("read JJ revision: {error}"))?
        .trim()
        .to_owned();
    if !(40..=64).contains(&revision.len())
        || !revision.bytes().all(|byte| byte.is_ascii_hexdigit())
    {
        return Err("read JJ revision: invalid commit ID".into());
    }
    let status = Command::new("jj")
        .current_dir(root)
        .arg("new")
        .status()
        .map_err(|error| format!("seal JJ revision: {error}"))?;
    if !status.success() {
        return Err("seal JJ revision: command failed".into());
    }
    Ok(revision)
}

fn resolved_config(
    root: &Path,
    spec: &mut baux::config::ExperimentSpec,
) -> Result<PathBuf, String> {
    let mut hash = Sha256::new();
    hash.update(b"baux-experiment-v2\0");
    hash.update(spec.to_toml()?.as_bytes());
    let id = format!("{:x}", hash.finalize())[..20].to_owned();
    spec.output = Some(root.join(".cache/baux/runs/pretrain").join(&id));
    let path = root
        .join(".cache/baux/experiments/pretrain")
        .join(format!("{id}.toml"));
    fs::create_dir_all(path.parent().ok_or("invalid resolved config path")?)
        .map_err(|error| error.to_string())?;
    let temporary = path.with_extension(format!("{}.tmp", std::process::id()));
    fs::write(&temporary, spec.to_toml()?).map_err(|error| error.to_string())?;
    fs::rename(temporary, &path).map_err(|error| error.to_string())?;
    Ok(path)
}

#[cfg(test)]
mod tests {
    use super::validate_authored;

    const BASE: &str = "version=4\n\
        experiment='pretrain'\n\
        model='fbt-pisa1-mhc4-v1'\n\
        [data]\n\
        recipe='stack-v3-rust-go-v1'\n\
        [objective]\n\
        function='cross-entropy'\n\
        sampling='natural'\n\
        [optimizer]\n\
        algorithm='muon'\n";

    #[test]
    fn authored_paths() {
        let clean = baux::config::ExperimentSpec::parse(BASE).unwrap();
        assert!(validate_authored(&clean).is_ok());
        for field in [
            "output='run'\n",
            "train='train.bauxptn'\n",
            "validation='validation.bauxptn'\n",
        ] {
            let text = if field.starts_with("output") {
                BASE.replacen(
                    "experiment='pretrain'\n",
                    &format!("experiment='pretrain'\n{field}"),
                    1,
                )
            } else {
                BASE.replacen(
                    "recipe='stack-v3-rust-go-v1'\n",
                    &format!("recipe='stack-v3-rust-go-v1'\n{field}"),
                    1,
                )
            };
            let spec = baux::config::ExperimentSpec::parse(&text).unwrap();
            assert!(validate_authored(&spec).is_err());
        }
    }
}
