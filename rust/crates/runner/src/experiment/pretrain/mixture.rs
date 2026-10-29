//! Construct mixtures from verified component corpora and resolve their
//! artifacts.
//!
//! Training interleaves complete source units; held-out splits retain weighted
//! source records. Tokenizer and boundary identities must agree across all
//! splits.

use super::{
    canonical, corpus_path, file_hash, max_repetitions, mix_parts, mix_seed, prepare_named, recipe,
    recipe_key, run_worker, string, verify,
};
use baux::artifact::json::{Value, json};
use baux::config::ExperimentSpec;
use baux_tooling::build::BuildSystem;
use baux_tooling::terminal::{self, Tag};
use std::fs;
use std::path::Path;

#[derive(Default)]
struct Identity {
    tokenizer: Option<String>,
    boundaries: Option<(u64, u64)>,
}

impl Identity {
    fn observe<'a>(&mut self, manifest: &'a Value, directory: &Path) -> Result<&'a str, String> {
        let tokenizer = string(&manifest["tokenizer"], "sha256")?;
        verify(&directory.join("tokenizer.json"), Some(tokenizer))?;
        if self
            .tokenizer
            .as_deref()
            .is_some_and(|expected| expected != tokenizer)
        {
            return Err("pretraining mixture components use different tokenizers".into());
        }
        self.tokenizer = Some(tokenizer.to_owned());
        let bos = manifest["tokenizer"]["special_tokens"]["<bos>"]
            .kind()
            .as_u64()
            .ok_or("component corpus has no BOS token")?;
        let eos = manifest["tokenizer"]["special_tokens"]["<eos>"]
            .kind()
            .as_u64()
            .ok_or("component corpus has no EOS token")?;
        if self
            .boundaries
            .is_some_and(|expected| expected != (bos, eos))
        {
            return Err("pretraining mixture components use different boundaries".into());
        }
        self.boundaries = Some((bos, eos));
        Ok(tokenizer)
    }
}

pub(super) fn prepare(
    root: &Path,
    document: &Value,
    id: &str,
    engine: BuildSystem,
) -> Result<(), String> {
    let parent = root.join(".cache/baux/corpora");
    fs::create_dir_all(&parent).map_err(|error| error.to_string())?;
    let stage = parent.join(format!(".{id}.{}.preparing", std::process::id()));
    fs::create_dir(&stage).map_err(|error| error.to_string())?;
    let mut identity = Identity::default();
    let mut split_records = baux::artifact::json::Map::new();
    for split in ["train", "validation", "test"] {
        let (parts, minimum) = parts(root, document, split, engine, &mut identity)?;
        let record = if split == "train" {
            let boundaries = identity
                .boundaries
                .ok_or("mixture has no token boundaries")?;
            training(root, document, &stage, &parts, boundaries, engine)?
        } else {
            evaluation(document, &stage, split, parts, minimum)?
        };
        split_records.insert(split, record);
    }
    let manifest = json!({
        "format":"baux.pretraining-mixture.v4",
        "recipe_id":id,
        "recipe":document,
        "tokenizer":{"sha256":identity.tokenizer.ok_or("mixture has no tokenizer")?},
        "splits":Value::from(split_records),
    });
    fs::write(
        stage.join("manifest.json"),
        baux::artifact::json::pretty_string(&manifest).map_err(|error| error.to_string())?,
    )
    .map_err(|error| error.to_string())?;
    fs::rename(stage, parent.join(id)).map_err(|error| error.to_string())
}

fn parts(
    root: &Path,
    document: &Value,
    split: &str,
    engine: BuildSystem,
    identity: &mut Identity,
) -> Result<(Vec<Value>, u64), String> {
    let mut records = Vec::new();
    let mut minimum = u64::MAX;
    for part in mix_parts(document, split)? {
        let (record, tokens) = component(root, part, split, engine, identity)?;
        minimum = minimum.min(tokens);
        records.push(record);
    }
    Ok((records, minimum))
}

/// Validate cached component identity and checksums before constructing a split
/// record.
fn component(
    root: &Path,
    part: &Value,
    split: &str,
    engine: BuildSystem,
    identity: &mut Identity,
) -> Result<(Value, u64), String> {
    let name = string(part, "recipe")?;
    let document = recipe(root, name)?;
    if string(&document, "format")? != "baux.pretraining.v2" {
        return Err("pretraining mixtures may only contain corpus recipes".into());
    }
    prepare_named(root, name, engine)?;
    let id = recipe_key(root, &document)?;
    let directory = root.join(".cache/baux/corpora").join(&id);
    let manifest: Value = baux::artifact::json::from_reader(
        fs::File::open(directory.join("manifest.json")).map_err(|error| error.to_string())?,
    )
    .map_err(|error| error.to_string())?;
    if manifest["format"].kind().as_str() != Some("baux.pretraining.v2")
        || manifest["recipe_id"].kind().as_str() != Some(id.as_str())
        || canonical(
            manifest
                .get("recipe")
                .ok_or("component corpus manifest has no recipe")?,
        )? != canonical(&document)?
    {
        return Err(format!(
            "cached component corpus {id} does not match recipe {name}"
        ));
    }
    let source = &manifest["splits"][split];
    let tokens = source["tokens"]
        .kind()
        .as_u64()
        .ok_or("component corpus split has no token count")?;
    verify(
        &directory.join(format!("{split}.bauxptn")),
        source["sha256"].kind().as_str(),
    )?;
    let tokenizer = identity.observe(&manifest, &directory)?;
    let mut record = json!({
        "name":string(part, "name")?,
        "tokens":tokens,
        "path":format!("../{id}/{split}.bauxptn"),
        "sha256":string(source, "sha256")?,
        "tokenizer-sha256":tokenizer,
    });
    if split == "train" {
        let marker = match string(&document, "serialization")? {
            "repository_file_v1" => Some("<|repository|>"),
            "role_tool_channel_v4" => Some("<|episode|>"),
            "text_document_v1" => None,
            _ => return Err("mixture component has an unsupported serialization".into()),
        };
        if let Some(marker) = marker {
            record["marker"] = manifest["tokenizer"]["special_tokens"][marker]
                .kind()
                .as_u64()
                .ok_or("component corpus has no structural marker")?
                .into();
        }
    } else {
        record["weight"] = Value::from(
            part["weight"]
                .kind()
                .as_u64()
                .ok_or("mixture weight missing")?,
        );
    }
    Ok((record, tokens))
}

/// Run the corpus interleave worker and verify its output before recording its
/// identity.
fn training(
    root: &Path,
    document: &Value,
    stage: &Path,
    parts: &[Value],
    (bos, eos): (u64, u64),
    engine: BuildSystem,
) -> Result<Value, String> {
    let artifact = stage.join("train.bauxptn");
    let report = stage.join("train.json");
    let job_parts = parts
        .iter()
        .map(|part| {
            let path = string(part, "path")?;
            Ok(json!({
                "name":string(part, "name")?,
                "path":stage.join(path),
                "marker":part.get("marker"),
            }))
        })
        .collect::<Result<Vec<_>, String>>()?;
    let executable = corpus_path(root, engine)?;
    let job = json!({
        "kind":"mix",
        "output":artifact,
        "report":report,
        "bos":bos,
        "eos":eos,
        "seed":mix_seed(document)?,
        "parts":job_parts,
    });
    run_worker(&executable, &job, stage, "corpus interleave")?;
    let report_value: Value = baux::artifact::json::from_reader(
        fs::File::open(&report).map_err(|error| error.to_string())?,
    )
    .map_err(|error| error.to_string())?;
    verify(&artifact, report_value["sha256"].kind().as_str())?;
    Ok(json!({
        "artifact":"train.bauxptn",
        "sha256":string(&report_value, "sha256")?,
        "tokens":report_value["tokens"].kind().as_u64().ok_or("interleave report has no token count")?,
        "construction":string(&report_value, "construction")?,
        "sources":report_value["sources"].clone(),
    }))
}

fn evaluation(
    document: &Value,
    stage: &Path,
    split: &str,
    parts: Vec<Value>,
    minimum: u64,
) -> Result<Value, String> {
    let artifact = stage.join(format!("{split}.json"));
    fs::write(
        &artifact,
        baux::artifact::json::pretty_string(&json!({
            "format":"baux.pretraining-evaluation.v1",
            "aggregation":string(&document["evaluation"], "aggregation")?,
            "parts":parts,
        }))
        .map_err(|error| error.to_string())?,
    )
    .map_err(|error| error.to_string())?;
    Ok(json!({
        "artifact":format!("{split}.json"),
        "sha256":file_hash(&artifact)?,
        "minimum-tokens":minimum,
    }))
}

pub(super) fn resolve(
    spec: &mut ExperimentSpec,
    recipe: &Value,
    manifest: &Value,
    directory: &Path,
) -> Result<(), String> {
    let mut paths = Vec::new();
    for split in ["train", "validation", "test"] {
        let record = &manifest["splits"][split];
        if let Some(window_tokens) = spec.objective.window_tokens {
            let required = u64::from(window_tokens) + 1;
            let available = if split == "train" {
                record["tokens"].kind().as_u64().unwrap_or(0)
            } else {
                record["minimum-tokens"].kind().as_u64().unwrap_or(0)
            };
            if available < required {
                return Err(format!(
                    "mixture {split} needs at least {required} tokens for a {window_tokens}-token \
                     objective window"
                ));
            }
        }
        let artifact = string(record, "artifact")?;
        let expected = if split == "train" {
            "train.bauxptn".to_owned()
        } else {
            format!("{split}.json")
        };
        if artifact != expected {
            return Err(format!("mixture has an invalid {split} artifact"));
        }
        let path = directory.join(artifact);
        verify(&path, record["sha256"].kind().as_str())?;
        paths.push(path);
    }
    spec.data.train = Some(paths.remove(0));
    spec.data.validation = Some(paths.remove(0));
    let train_record = &manifest["splits"]["train"];
    let train_tokens = train_record["tokens"]
        .kind()
        .as_u64()
        .ok_or("mixture train token count missing")?;
    let train = source_summary(train_record)?;
    let validation = summary(recipe, "validation")?;
    let window_tokens = u64::from(
        spec.objective
            .window_tokens
            .ok_or("pretraining mixture requires objective.window-tokens")?,
    );
    let evaluations = u64::from(spec.execution.rounds)
        .checked_add(1)
        .ok_or("pretraining evaluation count overflow")?;
    let objective_tokens = window_tokens
        .checked_mul(evaluations)
        .ok_or("pretraining token budget overflow")?;
    let repetitions = objective_tokens as f64 / train_tokens as f64;
    if objective_tokens
        > train_tokens
            .checked_mul(max_repetitions(recipe)?)
            .ok_or("mixture repetition capacity overflow")?
    {
        return Err(format!(
            "mixture exceeds training.max_repetitions={}: {repetitions:.3} corpus-pass-equivalents",
            max_repetitions(recipe)?
        ));
    }
    terminal::stderr(
        Tag::Orchestration,
        &format!(
            "data · {} · train=unit-interleave{{{train}}} · \
             validation=weighted-mean{{{validation}}} · evaluations={evaluations} · \
             objective={objective_tokens} tokens · corpus-pass-equivalent={repetitions:.3}x",
            string(recipe, "preset")?,
        ),
    );
    Ok(())
}

pub(super) fn check(root: &Path, document: &Value, spec: &ExperimentSpec) -> Result<(), String> {
    let window_tokens = u64::from(
        spec.objective
            .window_tokens
            .ok_or("pretraining mixture requires objective.window-tokens")?,
    );
    let tokens = mix_parts(document, "train")?.into_iter().try_fold(
        0u64,
        |total, part| -> Result<u64, String> {
            let component = recipe(root, string(part, "recipe")?)?;
            let tokens = component["materialization"]["tokens"]["train"]
                .kind()
                .as_u64()
                .filter(|tokens| *tokens > 0)
                .ok_or_else(|| {
                    "mixture component has no materialized training token count".to_owned()
                })?;
            total
                .checked_add(tokens)
                .ok_or_else(|| "mixture token count overflow".to_owned())
        },
    )?;
    let evaluations = u64::from(spec.execution.rounds)
        .checked_add(1)
        .ok_or("pretraining evaluation count overflow")?;
    let objective = window_tokens
        .checked_mul(evaluations)
        .ok_or("pretraining token budget overflow")?;
    let capacity = tokens
        .checked_mul(max_repetitions(document)?)
        .ok_or("mixture repetition capacity overflow")?;
    if objective > capacity {
        return Err(format!(
            "mixture exceeds training.max_repetitions={}: {objective} objective tokens over \
             {tokens} corpus tokens",
            max_repetitions(document)?
        ));
    }
    Ok(())
}

fn source_summary(record: &Value) -> Result<String, String> {
    record["sources"]
        .as_seq()
        .ok_or("mixture train sources missing")?
        .iter()
        .map(|source| {
            Ok(format!(
                "{}={} tokens",
                string(source, "name")?,
                source["tokens"]
                    .kind()
                    .as_u64()
                    .ok_or("mixture source token count missing")?
            ))
        })
        .collect::<Result<Vec<_>, String>>()
        .map(|parts| parts.join(","))
}

pub(super) fn summary(recipe: &Value, split: &str) -> Result<String, String> {
    let parts = mix_parts(recipe, split)?;
    let total = parts.iter().try_fold(0u64, |total, part| {
        total
            .checked_add(
                part["weight"]
                    .kind()
                    .as_u64()
                    .ok_or("mixture weight missing")?,
            )
            .ok_or("pretraining mixture weight overflow")
    })?;
    parts
        .into_iter()
        .map(|part| {
            Ok(format!(
                "{}={}/{}",
                string(part, "name")?,
                part["weight"]
                    .kind()
                    .as_u64()
                    .ok_or("mixture weight missing")?,
                total,
            ))
        })
        .collect::<Result<Vec<_>, String>>()
        .map(|parts| parts.join(","))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scratch() -> std::path::PathBuf {
        let stamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path =
            std::env::temp_dir().join(format!("baux-mixture-{}-{stamp}", std::process::id()));
        fs::create_dir_all(&path).unwrap();
        path
    }

    #[test]
    fn tokenizer_identity() {
        let directory = scratch();
        let path = directory.join("tokenizer.json");
        fs::write(&path, b"tokenizer").unwrap();
        let hash = file_hash(&path).unwrap();
        let manifest = |hash: &str, eos: u64| {
            json!({
                "tokenizer":{"sha256":hash,"special_tokens":{"<bos>":2,"<eos>":eos}}
            })
        };
        let mut identity = Identity::default();
        let matching = manifest(&hash, 3);
        assert_eq!(identity.observe(&matching, &directory).unwrap(), hash);
        assert_eq!(identity.observe(&matching, &directory).unwrap(), hash);
        let different = manifest(&hash, 4);
        assert!(
            identity
                .observe(&different, &directory)
                .unwrap_err()
                .contains("boundaries")
        );
        fs::write(&path, b"other tokenizer").unwrap();
        let different = manifest(&file_hash(&path).unwrap(), 3);
        assert!(
            identity
                .observe(&different, &directory)
                .unwrap_err()
                .contains("different tokenizers")
        );
        assert!(
            identity
                .observe(&matching, &directory)
                .unwrap_err()
                .contains("checksum mismatch")
        );
        fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn heldout_record() {
        let directory = scratch();
        let document = json!({"evaluation":{"aggregation":"weighted-mean"}});
        let parts = vec![json!({"name":"code","weight":1,"tokens":128})];
        let record = evaluation(&document, &directory, "validation", parts.clone(), 128).unwrap();
        let artifact = directory.join("validation.json");
        verify(&artifact, record["sha256"].kind().as_str()).unwrap();
        assert_eq!(record["artifact"].kind().as_str(), Some("validation.json"));
        assert_eq!(record["minimum-tokens"].kind().as_u64(), Some(128));
        let stored: Value =
            baux::artifact::json::from_reader(fs::File::open(artifact).unwrap()).unwrap();
        assert_eq!(
            stored["format"].kind().as_str(),
            Some("baux.pretraining-evaluation.v1")
        );
        assert_eq!(stored["aggregation"].kind().as_str(), Some("weighted-mean"));
        assert_eq!(stored["parts"], Value::from(parts));
        fs::remove_dir_all(directory).unwrap();
    }
}
