use super::encoding;
use baux::artifact::json::Value;
use sha2::{Digest, Sha256};
use std::fs;
use std::path::Path;

pub(crate) fn recipe(root: &Path, name: &str) -> Result<Value, String> {
    if !baux::config::valid_recipe(name) {
        return Err("pretraining recipe name must be lowercase kebab-case".into());
    }
    let path = root.join("data/recipes").join(format!("{name}.toml"));
    let value = baux::artifact::toml::from_reader(
        fs::File::open(&path)
            .map_err(|error| format!("pretraining recipe {}: {error}", path.display()))?,
    )
    .map_err(|error| format!("pretraining recipe {}: {error}", path.display()))?;
    validate_recipe(name, &value)?;
    Ok(value)
}

pub(crate) fn validate_recipe(name: &str, recipe: &Value) -> Result<(), String> {
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
    encoding::validate(recipe)?;
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
    if string(recipe, "format")? != "baux.pretraining.v2"
        || recipe["context"].kind().as_u64() != Some(4096)
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
    if encoding::is_bytes(recipe) && serialization != "text_document_v1" {
        return Err("byte encoding requires text_document_v1 serialization".into());
    }
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
        || !encoding::text_boundaries(recipe)
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

pub(crate) fn mix_parts<'a>(recipe: &'a Value, split: &str) -> Result<Vec<&'a Value>, String> {
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

pub(crate) fn mix_seed(recipe: &Value) -> Result<u64, String> {
    recipe["training"]["seed"]
        .kind()
        .as_u64()
        .ok_or_else(|| "mixture training seed missing".into())
}

pub(crate) fn max_repetitions(recipe: &Value) -> Result<u64, String> {
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

pub(crate) fn exact_fields(value: &Value, expected: &[&str], label: &str) -> Result<(), String> {
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

pub(crate) fn recipe_key(root: &Path, document: &Value) -> Result<String, String> {
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

pub(crate) fn canonical(value: &Value) -> Result<String, String> {
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

pub(crate) fn verify(path: &Path, expected: Option<&str>) -> Result<(), String> {
    let expected = expected.ok_or_else(|| format!("missing digest for {}", path.display()))?;
    if file_hash(path)? == expected {
        Ok(())
    } else {
        Err(format!("checksum mismatch: {}", path.display()))
    }
}

pub(crate) fn file_hash(path: &Path) -> Result<String, String> {
    baux::artifact::runs::file_digest(path)
}

pub(crate) fn string<'a>(value: &'a Value, key: &str) -> Result<&'a str, String> {
    value[key]
        .kind()
        .as_str()
        .ok_or_else(|| format!("corpus recipe is missing {key}"))
}
