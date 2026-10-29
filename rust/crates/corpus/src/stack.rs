//! Repository-preserving Rust and Go streams from the filtered Stack v3 corpus.
use super::{BUCKETS, Document, Job as SelectionJob, Result, Selection, reader};
use baux::artifact::json::{Value, json};
use deser::Deserialize;
use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::PathBuf;
use tokenizers::Tokenizer;

const CONTEXT: u32 = 4096;
const VOCABULARY: u32 = 262_144;
const CHAR_RESERVE: u64 = 4;
const SOURCE_MARKERS: [&str; 6] = [
    "<|repository|>",
    "<|file|>",
    "<|content|>",
    "<|endfile|>",
    "<|endrepository|>",
    "<|pad|>",
];

#[derive(Deserialize)]
struct Job {
    root: String,
    paths: Vec<String>,
    tokenizer: PathBuf,
    output: PathBuf,
    recipe_id: String,
    recipe: Value,
}

pub(super) async fn prepare(value: Value) -> Result<()> {
    let job: Job = baux::artifact::json::from_str(&baux::artifact::json::to_string(&value)?)?;
    fs::create_dir_all(&job.output)?;
    if job.output.join("manifest.json").exists() {
        return Err("refusing to overwrite an existing corpus manifest".into());
    }
    let tokenizer = tokenizer(&job)?;
    let selection = reader::collect(selection_job(&job)?).await?;
    pack(&job, selection, &tokenizer)
}

fn selection_job(job: &Job) -> Result<SelectionJob> {
    let tokens = job.recipe["materialization"]["tokens"]
        .as_map()
        .ok_or("Stack recipe requires split token budgets")?;
    let mut target_characters = BTreeMap::new();
    for (key, value) in tokens {
        let split = key.as_str().ok_or("invalid Stack split name")?;
        let count = value.as_u64().ok_or("invalid Stack token budget")?;
        let characters = count
            .checked_mul(CHAR_RESERVE)
            .and_then(|count| usize::try_from(count).ok())
            .ok_or("Stack character target overflow")?;
        target_characters.insert(split.to_owned(), characters);
    }
    let filter = &job.recipe["filter"];
    let languages = filter["languages"]
        .as_seq()
        .ok_or("Stack recipe requires languages")?
        .iter()
        .map(|value| {
            value
                .as_str()
                .filter(|language| !language.is_empty())
                .map(str::to_owned)
                .ok_or_else(|| "Stack recipe contains an invalid language".into())
        })
        .collect::<Result<Vec<_>>>()?;
    let excluded_names = filter["exclude_lockfiles"]
        .as_seq()
        .ok_or("Stack recipe requires excluded lockfiles")?
        .iter()
        .map(|value| {
            value
                .as_str()
                .filter(|name| !name.is_empty())
                .map(str::to_lowercase)
                .ok_or_else(|| "Stack recipe contains an invalid excluded filename".into())
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(SelectionJob {
        root: job.root.clone(),
        paths: job.paths.clone(),
        targets: BTreeMap::new(),
        target_characters,
        languages,
        excluded_names,
        holdout_partitions: 2,
        max_file: usize::try_from(
            filter["max_file_characters"]
                .as_u64()
                .ok_or("Stack recipe requires max_file_characters")?,
        )?,
        max_repository: usize::try_from(
            filter["max_repository_characters"]
                .as_u64()
                .ok_or("Stack recipe requires max_repository_characters")?,
        )?,
    })
}

fn tokenizer(job: &Job) -> Result<Tokenizer> {
    super::tokenizer::from_recipe(&job.tokenizer, &job.recipe)
}

fn pack(job: &Job, selection: Selection, tokenizer: &Tokenizer) -> Result<()> {
    let tokenizer_path = job.output.join("tokenizer.json");
    tokenizer.save(&tokenizer_path, true)?;
    let mut records = BTreeMap::new();
    for split in ["train", "validation", "test"] {
        records.insert(split, pack_split(job, &selection, split, tokenizer)?);
    }
    let mut special: BTreeMap<_, _> = super::tokenizer::MARKERS
        .iter()
        .map(|marker| (*marker, tokenizer.token_to_id(marker)))
        .collect();
    special.insert("<bos>", tokenizer.token_to_id("<bos>"));
    special.insert("<eos>", tokenizer.token_to_id("<eos>"));
    let manifest = json!({
        "format":"baux.pretraining.v2",
        "recipe_id":job.recipe_id,
        "recipe":job.recipe,
        "source":job.recipe.get("source"),
        "split_policy":job.recipe.get("split_policy"),
        "filter":job.recipe.get("filter"),
        "serialization":"repository_file_v1",
        "tokenizer":{
            "sha256":super::tokenizer::digest(&tokenizer_path)?,
            "vocabulary":VOCABULARY,
            "base":job.recipe.get("tokenizer"),
            "special_tokens":special,
        },
        "splits":records,
    });
    fs::write(
        job.output.join("manifest.json"),
        baux::artifact::json::pretty_string(&manifest)?,
    )?;
    Ok(())
}

fn pack_split(
    job: &Job,
    selection: &Selection,
    split: &str,
    tokenizer: &Tokenizer,
) -> Result<Value> {
    let (bos, eos) = super::tokenizer::boundaries(tokenizer)?;
    let quota = job.recipe["materialization"]["tokens"][split]
        .as_u64()
        .ok_or("Stack recipe has no split token budget")?;
    let sequences = quota / u64::from(CONTEXT);
    let quota = usize::try_from(quota)?;
    let mut documents = selection.documents[split]
        .values()
        .flatten()
        .collect::<Vec<_>>();
    documents.sort_by(|left, right| {
        (&left.repository, &left.path).cmp(&(&right.repository, &right.path))
    });
    let mut tokens = Vec::with_capacity(quota + CONTEXT as usize);
    let mut repositories = BTreeSet::new();
    let mut files = 0usize;
    let mut previous: Option<&str> = None;
    for document in documents {
        if previous != Some(document.repository.as_str()) {
            if previous.is_some() {
                encode(tokenizer, "<|endrepository|>\n", &mut tokens)?;
                tokens.push(eos);
            }
            tokens.push(bos);
            let header = format!(
                "<|repository|>\n{}\n{}\n",
                document.repository, document.commit
            );
            encode(tokenizer, &header, &mut tokens)?;
            repositories.insert(document.repository.clone());
            previous = Some(&document.repository);
        }
        if SOURCE_MARKERS
            .iter()
            .any(|marker| document.text.contains(marker))
        {
            return Err("Stack source contains a reserved serialization marker".into());
        }
        let header = format!("<|file|>\n{}\n<|content|>", document.path);
        encode(tokenizer, &header, &mut tokens)?;
        encode(tokenizer, &document.text, &mut tokens)?;
        encode(tokenizer, "<|endfile|>\n", &mut tokens)?;
        files += 1;
        if tokens.len() >= quota {
            break;
        }
    }
    if previous.is_some() {
        encode(tokenizer, "<|endrepository|>\n", &mut tokens)?;
        tokens.push(eos);
    }
    if tokens.len() < quota {
        return Err(format!(
            "{split} produced {} tokens; {quota} required; add Stack shards",
            tokens.len()
        )
        .into());
    }
    let discarded = tokens.len() - quota;
    tokens.truncate(quota);
    let path = job.output.join(format!("{split}.bauxptn"));
    write_stream(&path, sequences, &tokens)?;
    let buckets = BUCKETS
        .iter()
        .map(|bucket| {
            (
                *bucket,
                json!({
                    "documents":selection.documents[split][*bucket].len(),
                    "characters":selection.characters[split][*bucket],
                }),
            )
        })
        .collect::<BTreeMap<_, _>>();
    eprintln!(
        "corpus | {split}: {quota} tokens from {files} files in {} repositories",
        repositories.len()
    );
    Ok(json!({
        "sequences":sequences,
        "tokens":quota,
        "sha256":super::tokenizer::digest(&path)?,
        "files":files,
        "repositories":repositories.len(),
        "buckets":buckets,
        "discarded_tail_tokens":discarded,
    }))
}

fn write_stream(path: &std::path::Path, sequences: u64, tokens: &[u32]) -> Result<()> {
    let mut output = BufWriter::new(File::create(path)?);
    output.write_all(b"BAUXPTN1")?;
    for value in [VOCABULARY, CONTEXT, sequences as u32, 0] {
        output.write_all(&value.to_le_bytes())?;
    }
    for &token in tokens {
        if token >= VOCABULARY {
            return Err("Stack tokenizer emitted an out-of-range token".into());
        }
        output.write_all(&token.to_le_bytes())?;
    }
    output.flush()?;
    Ok(())
}

fn encode(tokenizer: &Tokenizer, text: &str, output: &mut Vec<u32>) -> Result<()> {
    let encoded = tokenizer.encode(text, false)?;
    output.extend_from_slice(encoded.get_ids());
    Ok(())
}
