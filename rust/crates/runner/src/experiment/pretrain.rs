//! Prepare and verify cached corpora selected by the experiment's data recipe.

use baux::artifact::json::{Value, json};
use baux::config::ExperimentSpec;
use sha2::{Digest, Sha256};
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use baux_tooling::build::BuildSystem;
use baux_tooling::terminal::{self, Tag};

mod mixture;

pub(super) fn prepare(
    root: &Path,
    spec: &ExperimentSpec,
    engine: BuildSystem,
) -> Result<(), String> {
    let recipe_name = spec
        .data
        .recipe
        .as_deref()
        .ok_or("pretraining requires data.recipe")?;
    let document = recipe(root, recipe_name)?;
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

    let tokenizer = tokenizer(root, &recipe)?;
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
        .data
        .recipe
        .as_deref()
        .ok_or("pretraining requires data.recipe")?;
    let recipe = recipe(root, recipe_name)?;
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
        if let Some(window_tokens) = spec.objective.window_tokens {
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
        paths.push(path);
    }
    verify(
        &directory.join("tokenizer.json"),
        manifest["tokenizer"]["sha256"].kind().as_str(),
    )?;
    spec.data.train = Some(paths.remove(0));
    spec.data.validation = Some(paths.remove(0));
    report_budget(spec, &recipe, &split_tokens)
}

fn report_budget(
    spec: &ExperimentSpec,
    recipe: &Value,
    split_tokens: &[u64],
) -> Result<(), String> {
    if let Some(window_tokens) = spec.objective.window_tokens {
        let train_tokens = split_tokens[0];
        let validation_tokens = split_tokens[1];
        let objective_tokens = u64::from(window_tokens)
            .checked_mul(u64::from(spec.execution.rounds))
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

fn recipe(root: &Path, name: &str) -> Result<Value, String> {
    if !baux::config::valid_recipe(name) {
        return Err("pretraining recipe name must be lowercase kebab-case".into());
    }
    let path = root.join("data/recipes").join(format!("{name}.json"));
    let value = baux::artifact::json::from_reader(
        fs::File::open(&path)
            .map_err(|error| format!("pretraining recipe {}: {error}", path.display()))?,
    )
    .map_err(|error| format!("pretraining recipe {}: {error}", path.display()))?;
    validate_recipe(name, &value)?;
    Ok(value)
}

fn validate_recipe(name: &str, recipe: &Value) -> Result<(), String> {
    if string(recipe, "format")? == "baux.pretraining-mixture.v4" {
        return validate_mix(name, recipe);
    }
    let serialization = string(recipe, "serialization")?;
    let (top, source) = match serialization {
        "repository_file_v1" => (
            &[
                "preset",
                "format",
                "source",
                "split_policy",
                "deduplication",
                "filter",
                "serialization",
                "boundaries",
                "tokenizer",
                "context",
                "materialization",
                "seed",
            ][..],
            &["id", "revision", "paths"][..],
        ),
        "role_tool_channel_v4" => (
            &[
                "preset",
                "format",
                "source",
                "split_policy",
                "deduplication",
                "serialization",
                "boundaries",
                "tokenizer",
                "context",
                "materialization",
                "seed",
            ][..],
            &["id", "revision", "sha256", "paths"][..],
        ),
        "text_document_v1" => (
            &[
                "preset",
                "format",
                "source",
                "split_policy",
                "deduplication",
                "serialization",
                "boundaries",
                "tokenizer",
                "context",
                "materialization",
                "seed",
            ][..],
            &["id", "revision", "paths", "text_field", "id_field"][..],
        ),
        value => return Err(format!("unsupported recipe serialization {value}")),
    };
    exact_fields(recipe, top, "recipe")?;
    exact_fields(&recipe["source"], source, "recipe source")?;
    exact_fields(
        &recipe["tokenizer"],
        &[
            "kind",
            "vocabulary",
            "package",
            "id",
            "revision",
            "sha256",
            "markers",
            "add_special_tokens",
        ],
        "recipe tokenizer",
    )?;
    exact_fields(
        &recipe["materialization"],
        &["tokens"],
        "recipe materialization",
    )?;
    exact_fields(
        &recipe["materialization"]["tokens"],
        &["train", "validation", "test"],
        "materialized tokens",
    )?;
    let tokenizer = &recipe["tokenizer"];
    if string(recipe, "format")? != "baux.pretraining.v2"
        || recipe["context"].kind().as_u64() != Some(4096)
        || tokenizer["kind"].kind().as_str() != Some("gemma4_bpe")
        || tokenizer["vocabulary"].kind().as_u64() != Some(262_144)
        || tokenizer["add_special_tokens"].as_bool() != Some(false)
        || recipe["source"]["paths"]
            .as_seq()
            .is_none_or(|paths| paths.is_empty())
        || ["train", "validation", "test"].iter().any(|split| {
            recipe["materialization"]["tokens"][*split]
                .kind()
                .as_u64()
                .is_none_or(|tokens| {
                    tokens < 1_048_576
                        || tokens % 8_192 != 0
                        || tokens > u64::from(u32::MAX) * 4_096
                })
        })
    {
        return Err(format!("invalid {name} recipe contract"));
    }
    validate_content(name, recipe, serialization)?;
    if string(recipe, "preset")?.replace('_', "-") != name {
        return Err(format!("invalid {name} recipe contract"));
    }
    Ok(())
}

fn validate_content(name: &str, recipe: &Value, serialization: &str) -> Result<(), String> {
    let tokenizer = &recipe["tokenizer"];
    if serialization == "repository_file_v1" {
        exact_fields(
            &recipe["filter"],
            &[
                "languages",
                "license_type",
                "exclude_forks",
                "exclude_vendor",
                "exclude_lockfiles",
                "max_file_characters",
                "max_repository_characters",
            ],
            "recipe filter",
        )?;
        let filter = &recipe["filter"];
        if string(recipe, "split_policy")?
            != "repository_sha256_mod20_train_0_15_validation_16_17_test_18_19"
            || string(recipe, "deduplication")? != "stack_v3_cross_language_minhash_then_content_id"
            || string(recipe, "boundaries")? != "gemma_bos_eos_per_repository_v1"
            || tokenizer["markers"].kind().as_str()
                != Some("unused0..14_to_baux_markers_ids6..20_v1")
            || !valid_values(&filter["languages"], 1, 32)
            || filter["license_type"].kind().as_str() != Some("permissive")
            || filter["exclude_forks"].as_bool() != Some(true)
            || filter["exclude_vendor"].as_bool() != Some(true)
            || !valid_values(&filter["exclude_lockfiles"], 1, 32)
            || filter["max_file_characters"].kind().as_u64() != Some(262_144)
            || filter["max_repository_characters"].kind().as_u64() != Some(2_097_152)
        {
            return Err(format!("invalid {name} repository recipe contract"));
        }
    } else if serialization == "role_tool_channel_v4" {
        if string(recipe, "split_policy")?
            != "repository_sha256_mod20_train_0_17_validation_18_test_19"
            || string(recipe, "deduplication")? != "instance_id"
            || string(recipe, "boundaries")? != "gemma_bos_eos_per_episode_v1"
            || tokenizer["markers"].kind().as_str()
                != Some("unused0..14_to_baux_markers_ids6..20_v1")
            || recipe["source"]["paths"]
                .as_seq()
                .is_none_or(|paths| paths.len() != 1)
        {
            return Err(format!("invalid {name} trace recipe contract"));
        }
    } else if string(recipe, "split_policy")?
        != "document_sha256_mod20_train_0_17_validation_18_test_19"
        || string(recipe, "deduplication")? != "document_id"
        || string(recipe, "boundaries")? != "gemma_bos_eos_per_document_v1"
        || tokenizer["markers"].kind().as_str() != Some("unused0..14_to_baux_markers_ids6..20_v1")
        || !valid_field(string(&recipe["source"], "text_field")?)
        || !valid_field(string(&recipe["source"], "id_field")?)
    {
        return Err(format!("invalid {name} text recipe contract"));
    }
    Ok(())
}

fn validate_mix(name: &str, recipe: &Value) -> Result<(), String> {
    exact_fields(
        recipe,
        &["preset", "format", "training", "evaluation"],
        "mixture recipe",
    )?;
    exact_fields(
        &recipe["training"],
        &["construction", "seed", "max_repetitions", "sources"],
        "mixture training policy",
    )?;
    exact_fields(
        &recipe["evaluation"],
        &["aggregation", "validation", "test"],
        "mixture evaluation policy",
    )?;
    if string(recipe, "preset")?.replace('_', "-") != name
        || string(recipe, "format")? != "baux.pretraining-mixture.v4"
        || string(&recipe["training"], "construction")? != "unit-interleave"
        || mix_seed(recipe).is_err()
        || max_repetitions(recipe).is_err()
        || string(&recipe["evaluation"], "aggregation")? != "weighted-mean"
    {
        return Err(format!("invalid {name} mixture recipe contract"));
    }
    for split in ["train", "validation", "test"] {
        let parts = mix_parts(recipe, split)?;
        if !(2..=64).contains(&parts.len()) {
            return Err(format!("mixture {split} requires 2..=64 parts"));
        }
        let mut names = std::collections::HashSet::new();
        for part in parts {
            if split == "train" {
                exact_fields(part, &["name", "recipe"], "mixture training source")?;
            } else {
                exact_fields(
                    part,
                    &["name", "recipe", "weight"],
                    "mixture evaluation part",
                )?;
            }
            let part_name = string(part, "name")?;
            let component = string(part, "recipe")?;
            if !baux::config::valid_recipe(part_name)
                || !baux::config::valid_recipe(component)
                || !names.insert(part_name)
                || (split != "train"
                    && part["weight"]
                        .kind()
                        .as_u64()
                        .is_none_or(|weight| weight == 0 || weight > u64::from(u32::MAX)))
            {
                return Err(format!("invalid {split} mixture part"));
            }
        }
    }
    Ok(())
}

fn mix_parts<'a>(recipe: &'a Value, split: &str) -> Result<Vec<&'a Value>, String> {
    let parts = if split == "train" {
        &recipe["training"]["sources"]
    } else {
        &recipe["evaluation"][split]
    };
    parts
        .as_seq()
        .map(|parts| parts.iter().collect())
        .ok_or_else(|| format!("mixture recipe has no {split} parts"))
}

fn mix_seed(recipe: &Value) -> Result<u64, String> {
    recipe["training"]["seed"]
        .kind()
        .as_u64()
        .ok_or_else(|| "mixture training seed missing".into())
}

fn max_repetitions(recipe: &Value) -> Result<u64, String> {
    recipe["training"]["max_repetitions"]
        .kind()
        .as_u64()
        .filter(|value| *value > 0)
        .ok_or_else(|| "mixture training max_repetitions must be positive".into())
}

fn valid_values(value: &Value, minimum: usize, maximum: usize) -> bool {
    let Some(values) = value.as_seq() else {
        return false;
    };
    if !(minimum..=maximum).contains(&values.len()) {
        return false;
    }
    let mut seen = std::collections::HashSet::with_capacity(values.len());
    values.iter().all(|value| {
        value
            .as_str()
            .is_some_and(|text| !text.is_empty() && text.len() <= 64 && seen.insert(text))
    })
}

fn valid_field(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || byte == b'_')
}

fn exact_fields(value: &Value, expected: &[&str], label: &str) -> Result<(), String> {
    let fields = value
        .as_map()
        .ok_or_else(|| format!("{label} must be an object"))?;
    let names = fields
        .keys()
        .map(|key| {
            key.kind()
                .as_str()
                .ok_or_else(|| format!("{label} contains a non-string field"))
        })
        .collect::<Result<Vec<_>, _>>()?;
    if names.len() != expected.len()
        || names
            .iter()
            .any(|name| !expected.iter().any(|expected| expected == name))
    {
        return Err(format!(
            "{label} fields must be exactly {}",
            expected.join(", ")
        ));
    }
    Ok(())
}

fn recipe_id(recipe: &Value) -> Result<String, String> {
    let mut hash = Sha256::new();
    hash.update(b"baux-corpus-v1\0");
    hash.update(canonical(recipe)?.as_bytes());
    Ok(format!("{:x}", hash.finalize())[..20].to_owned())
}

fn recipe_key(root: &Path, document: &Value) -> Result<String, String> {
    if string(document, "format")? != "baux.pretraining-mixture.v4" {
        return recipe_id(document);
    }
    let mut hash = Sha256::new();
    hash.update(b"baux-mixture-v4-unit-interleave-v1\0");
    hash.update(canonical(document)?.as_bytes());
    for split in ["train", "validation", "test"] {
        for part in mix_parts(document, split)? {
            let component = recipe(root, string(part, "recipe")?)?;
            if string(&component, "format")? != "baux.pretraining.v2" {
                return Err("pretraining mixtures may only contain corpus recipes".into());
            }
            hash.update(canonical(&component)?.as_bytes());
        }
    }
    Ok(format!("{:x}", hash.finalize())[..20].to_owned())
}

fn canonical(value: &Value) -> Result<String, String> {
    if let Some(map) = value.as_map() {
        let mut entries = map.iter().collect::<Vec<_>>();
        entries.sort_by(|left, right| left.0.kind().as_str().cmp(&right.0.kind().as_str()));
        let fields = entries
            .into_iter()
            .map(|(key, value)| {
                Ok(format!(
                    "{}:{}",
                    baux::artifact::json::to_string(key).map_err(|error| error.to_string())?,
                    canonical(value)?
                ))
            })
            .collect::<Result<Vec<_>, String>>()?;
        Ok(format!("{{{}}}", fields.join(",")))
    } else if let Some(sequence) = value.as_seq() {
        Ok(format!(
            "[{}]",
            sequence
                .iter()
                .map(canonical)
                .collect::<Result<Vec<_>, _>>()?
                .join(",")
        ))
    } else {
        baux::artifact::json::to_string(value).map_err(|error| error.to_string())
    }
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

fn verify(path: &Path, expected: Option<&str>) -> Result<(), String> {
    let expected = expected.ok_or_else(|| format!("missing digest for {}", path.display()))?;
    if file_hash(path)? == expected {
        Ok(())
    } else {
        Err(format!("checksum mismatch: {}", path.display()))
    }
}

fn file_hash(path: &Path) -> Result<String, String> {
    baux::artifact::runs::file_digest(path)
}

fn string<'a>(value: &'a Value, key: &str) -> Result<&'a str, String> {
    value[key]
        .kind()
        .as_str()
        .ok_or_else(|| format!("corpus recipe is missing {key}"))
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
