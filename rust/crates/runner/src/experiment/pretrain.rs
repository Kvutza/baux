//! Prepare and verify cached corpora selected by the experiment's data recipe.

use baux::artifact::json::{Value, json};
use baux::config::ExperimentSpec;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use baux_tooling::build::BuildSystem;
use baux_tooling::terminal::{self, Tag};

mod encoding;
mod mixture;
mod schema;

pub(super) use schema::{
    canonical, exact_fields, file_hash, max_repetitions, mix_parts, mix_seed, recipe, recipe_key,
    string, validate_recipe, verify,
};

pub(super) fn prepare(
    root: &Path,
    spec: &ExperimentSpec,
    engine: BuildSystem,
) -> Result<(), String> {
    let recipe_name = spec
        .recipe
        .as_deref()
        .ok_or("pretraining requires recipe")?;
    let document = recipe(root, recipe_name)?;
    encoding::check_model(spec, &document)?;
    if string(&document, "format")? == "baux.pretraining-mixture.v4" {
        mixture::check(root, &document, spec)?;
    }
    prepare_named(root, recipe_name, engine)
}

fn prepare_named(root: &Path, recipe_name: &str, engine: BuildSystem) -> Result<(), String> {
    let recipe = recipe(root, recipe_name)?;
    let id = recipe_key(root, &recipe)?;
    let parent = root.join(".cache/baux/corpora");
    let destination = parent.join(&id);
    if destination.join("manifest.json").is_file() {
        return Ok(());
    }
    if string(&recipe, "format")? == "baux.pretraining-mixture.v4" {
        return mixture::prepare(root, &recipe, &id, engine);
    }

    let tokenizer = if encoding::is_bytes(&recipe) {
        None
    } else {
        Some(tokenizer(root, &recipe)?)
    };
    fs::create_dir_all(&parent).map_err(|error| error.to_string())?;
    let stage = parent.join(format!(".{id}.{}.preparing", std::process::id()));
    fs::create_dir(&stage).map_err(|error| error.to_string())?;
    let executable = corpus_path(root, engine)?;
    let job = match string(&recipe, "serialization")? {
        "repository_file_v1" => {
            let source = &recipe["source"];
            let base = format!(
                "https://huggingface.co/datasets/{}/resolve/{}",
                string(source, "id")?,
                string(source, "revision")?,
            );
            let paths = source["paths"]
                .as_seq()
                .ok_or("Stack source paths missing")?;
            json!({"kind":"stack","root":base,"paths":paths,"tokenizer":tokenizer,
                "output":stage,"recipe_id":id,"recipe":recipe})
        }
        "role_tool_channel_v4" => {
            let paths = trace_sources(root, &recipe)?;
            json!({"kind":"traces","paths":paths,"tokenizer":tokenizer,
                "output":stage,"recipe_id":id,"recipe":recipe})
        }
        "text_document_v1" => {
            let source = &recipe["source"];
            let base = format!(
                "https://huggingface.co/datasets/{}/resolve/{}",
                string(source, "id")?,
                string(source, "revision")?,
            );
            let paths = source["paths"]
                .as_seq()
                .ok_or("text source paths missing")?;
            json!({"kind":"text","root":base,"paths":paths,"tokenizer":tokenizer,
                "output":stage,"recipe_id":id,"recipe":recipe})
        }
        serialization => {
            return Err(format!(
                "pretraining recipe {recipe_name} uses unsupported serialization {serialization}"
            ));
        }
    };

    run_worker(&executable, &job, &stage, "corpus preparation")?;
    if !stage.join("manifest.json").is_file() {
        return Err("corpus worker did not write a manifest".into());
    }
    fs::rename(stage, destination).map_err(|error| error.to_string())
}

pub(super) fn corpus_path(root: &Path, engine: BuildSystem) -> Result<PathBuf, String> {
    if let Some(path) = std::env::var_os("BAUX_CORPUS_EXECUTABLE") {
        let path = PathBuf::from(path);
        if path.is_file() {
            return Ok(path);
        }
        return Err(format!(
            "BAUX_CORPUS_EXECUTABLE is not a file: {}",
            path.display()
        ));
    }
    Ok(engine
        .build_artifact(root, "//rust/crates/corpus:corpus")?
        .path)
}

fn run_worker(executable: &Path, job: &Value, stage: &Path, label: &str) -> Result<(), String> {
    let mut child = Command::new(executable)
        .stdin(Stdio::piped())
        .spawn()
        .map_err(|error| error.to_string())?;
    child
        .stdin
        .take()
        .ok_or("corpus worker stdin unavailable")?
        .write_all(
            baux::artifact::json::to_string(job)
                .map_err(|error| error.to_string())?
                .as_bytes(),
        )
        .map_err(|error| error.to_string())?;
    if !child.wait().map_err(|error| error.to_string())?.success() {
        return Err(format!(
            "{label} failed; diagnostics retained in {}",
            stage.display()
        ));
    }
    Ok(())
}

pub(super) fn resolve(root: &Path, spec: &mut ExperimentSpec) -> Result<(), String> {
    let recipe_name = spec
        .recipe
        .as_deref()
        .ok_or("pretraining requires recipe")?;
    let recipe = recipe(root, recipe_name)?;
    encoding::check_model(spec, &recipe)?;
    let id = recipe_key(root, &recipe)?;
    let directory = root.join(".cache/baux/corpora").join(&id);
    let manifest_path = directory.join("manifest.json");
    let manifest: Value = baux::artifact::json::from_reader(
        fs::File::open(&manifest_path)
            .map_err(|error| format!("corpus {id} is not cached: {error}"))?,
    )
    .map_err(|error| format!("invalid corpus manifest: {error}"))?;
    let format = string(&recipe, "format")?;
    if manifest["format"].kind().as_str() != Some(format)
        || manifest["recipe_id"].kind().as_str() != Some(&id)
        || canonical(
            manifest
                .get("recipe")
                .ok_or("corpus manifest has no recipe")?,
        )? != canonical(&recipe)?
    {
        return Err(format!("cached corpus {id} does not match its recipe"));
    }
    if format == "baux.pretraining-mixture.v4" {
        return mixture::resolve(spec, &recipe, &manifest, &directory);
    }

    let mut paths = Vec::new();
    let mut split_tokens = Vec::new();
    for split in ["train", "validation", "test"] {
        let record = &manifest["splits"][split];
        let expected = recipe["materialization"]["tokens"][split]
            .kind()
            .as_u64()
            .ok_or("invalid split size in corpus recipe")?;
        if record["tokens"].kind().as_u64() != Some(expected) {
            return Err(format!("cached corpus {id} has the wrong {split} size"));
        }
        if let Some(window_tokens) = spec.window_tokens {
            let required = u64::from(window_tokens) + 1;
            if record["tokens"].kind().as_u64().unwrap_or(0) < required {
                return Err(format!(
                    "cached corpus {id} {split} split needs at least {required} tokens for a \
                     {window_tokens}-token objective window"
                ));
            }
        }
        split_tokens.push(record["tokens"].kind().as_u64().unwrap_or(0));
        let path = directory.join(format!("{split}.bauxptn"));
        verify(&path, record["sha256"].kind().as_str())?;
        encoding::check_header(&path, &recipe)?;
        paths.push(path);
    }
    encoding::verify(&directory, &recipe, &manifest)?;
    spec.train = Some(paths.remove(0));
    spec.validation = Some(paths.remove(0));
    report_budget(spec, &recipe, &split_tokens)
}

fn report_budget(
    spec: &ExperimentSpec,
    recipe: &Value,
    split_tokens: &[u64],
) -> Result<(), String> {
    if let Some(window_tokens) = spec.window_tokens {
        let train_tokens = split_tokens[0];
        let validation_tokens = split_tokens[1];
        let objective_tokens = u64::from(window_tokens)
            .checked_mul(u64::from(spec.rounds))
            .ok_or("pretraining token budget overflow")?;
        let corpus_passes = objective_tokens as f64 / train_tokens as f64;
        let context = recipe["context"]
            .kind()
            .as_u64()
            .ok_or("invalid context in corpus recipe")?;
        let stride = u64::from(window_tokens).min(context);
        let train_windows = aligned_windows(train_tokens, u64::from(window_tokens), stride);
        let validation_windows =
            aligned_windows(validation_tokens, u64::from(window_tokens), stride);
        terminal::stderr(
            Tag::Orchestration,
            &format!(
                "data · {} · train={} tokens · objective={} tokens · \
                 corpus-pass-equivalent={corpus_passes:.3}x · windows={train_windows} \
                 train/{validation_windows} validation",
                string(recipe, "preset")?,
                train_tokens,
                objective_tokens,
            ),
        );
    }
    Ok(())
}

fn aligned_windows(tokens: u64, window_tokens: u64, alignment: u64) -> u64 {
    tokens
        .checked_sub(window_tokens.saturating_add(1))
        .map_or(0, |last| last / alignment + 1)
}

fn tokenizer(root: &Path, recipe: &Value) -> Result<PathBuf, String> {
    let metadata = &recipe["tokenizer"];
    let revision = string(metadata, "revision")?;
    let expected = string(metadata, "sha256")?;
    let directory = root.join(".cache/baux/sources/gemma4").join(revision);
    fs::create_dir_all(&directory).map_err(|error| error.to_string())?;
    let target = directory.join("tokenizer.json");
    if !target.is_file() {
        let partial = directory.join(format!("tokenizer.{}.part", std::process::id()));
        let url = format!(
            "https://huggingface.co/{}/resolve/{revision}/tokenizer.json",
            string(metadata, "id")?
        );
        let status = Command::new("curl")
            .args(["--fail", "--location", "--retry", "2", "--output"])
            .arg(&partial)
            .arg(url)
            .status()
            .map_err(|error| error.to_string())?;
        if !status.success() {
            return Err("tokenizer download failed".into());
        }
        verify(&partial, Some(expected))?;
        fs::rename(partial, &target).map_err(|error| error.to_string())?;
    }
    verify(&target, Some(expected))?;
    Ok(target)
}

fn trace_sources(root: &Path, recipe: &Value) -> Result<Vec<PathBuf>, String> {
    let directory = root.join(".cache/baux/sources/open-swe");
    fs::create_dir_all(&directory).map_err(|error| error.to_string())?;
    let source = &recipe["source"];
    let mut local = Vec::new();
    for value in source["paths"].as_seq().ok_or("trace paths missing")? {
        let path = value.kind().as_str().ok_or("invalid trace path")?;
        let target = directory.join(
            Path::new(path)
                .file_name()
                .ok_or("invalid trace filename")?,
        );
        if !target.is_file() {
            let partial = target.with_extension("part");
            let url = format!(
                "https://huggingface.co/datasets/{}/resolve/{}/{path}",
                string(source, "id")?,
                string(source, "revision")?
            );
            let status = Command::new("curl")
                .args(["--fail", "--location", "--retry", "2", "--output"])
                .arg(&partial)
                .arg(url)
                .status()
                .map_err(|error| error.to_string())?;
            if !status.success() {
                return Err("trace download failed".into());
            }
            fs::rename(partial, &target).map_err(|error| error.to_string())?;
        }
        local.push(target);
    }
    Ok(local)
}

#[cfg(test)]
mod tests {
    use super::{aligned_windows, mixture, validate_recipe};
    use baux::artifact::json::json;

    #[test]
    fn window_count() {
        assert_eq!(aligned_windows(8_388_608, 1_048_576, 4096), 1792);
        assert_eq!(aligned_windows(1_056_768, 1_048_576, 4096), 2);
        assert_eq!(aligned_windows(1_048_576, 1_048_576, 4096), 0);
    }

    #[test]
    fn closed_recipe() {
        let recipe = json!({"preset":"stack_v3_rust_go_v1","unknown":true});
        assert!(validate_recipe("stack-v3-rust-go-v1", &recipe).is_err());
    }

    #[test]
    fn closed_mixture() {
        let source = |name: &str, recipe: &str| json!({"name":name,"recipe":recipe});
        let part = |name: &str, recipe: &str| json!({"name":name,"recipe":recipe,"weight":1});
        let recipe = json!({
            "preset":"code_trace_v1",
            "format":"baux.pretraining-mixture.v4",
            "training":{
                "construction":"unit-interleave",
                "seed":7,
                "max_repetitions":8,
                "sources":[source("code", "stack-v3-rust-go-v1"), source("traces", "agent-v1-swe-traces-1m")],
            },
            "evaluation":{
                "aggregation":"weighted-mean",
                "validation":[part("code", "stack-v3-rust-go-v1"), part("traces", "agent-v1-swe-traces-1m")],
                "test":[part("code", "stack-v3-rust-go-v1"), part("traces", "agent-v1-swe-traces-1m")],
            },
        });
        assert!(validate_recipe("code-trace-v1", &recipe).is_ok());
        assert_eq!(
            mixture::summary(&recipe, "validation").unwrap(),
            "code=1/2,traces=1/2"
        );
        let invalid = json!({
            "preset":"code_trace_v1",
            "format":"baux.pretraining-mixture.v4",
            "training":{"construction":"unit-interleave","seed":7,"max_repetitions":8,"sources":[]},
            "evaluation":{"aggregation":"weighted-mean","validation":[],"test":[]},
            "path":"mutable",
        });
        assert!(validate_recipe("code-trace-v1", &invalid).is_err());
    }
}
