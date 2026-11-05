//! Repository-disjoint pretraining streams from recorded software-agent traces.
use super::reader::{Result, holdout, split_with};
use baux::artifact::json::{Value, json};
use deser::Deserialize;
use parquet::file::reader::{FileReader, SerializedFileReader};
use std::collections::{BTreeMap, HashSet};
use std::fs::{self, File};
use std::io::Write;
use std::path::{Path, PathBuf};
use tokenizers::Tokenizer;

const SOURCE_MARKERS: [&str; 12] = [
    "<|episode|>",
    "<|repository|>",
    "<|turn|>",
    "<|endturn|>",
    "<|reasoning|>",
    "<|content|>",
    "<|toolcalls|>",
    "<|system|>",
    "<|user|>",
    "<|assistant|>",
    "<|tool|>",
    "<|pad|>",
];
const SPLITS: [&str; 3] = ["train", "validation", "test"];

#[derive(Deserialize)]
struct Job {
    paths: Vec<PathBuf>,
    tokenizer: PathBuf,
    output: PathBuf,
    recipe_id: String,
    recipe: Value,
}
#[derive(Deserialize)]
struct Trace {
    instance_id: String,
    repo: String,
    messages: Vec<Message>,
    tools: Option<Value>,
}
#[derive(Deserialize)]
struct Message {
    role: String,
    content: Option<String>,
    reasoning_content: Option<String>,
    tool_calls: Option<Value>,
    name: Option<String>,
    tool_call_id: Option<String>,
}

type Documents = [Vec<(String, String)>; 3];

pub(super) fn prepare(value: Value) -> Result<()> {
    let job: Job = baux::artifact::json::from_str(&baux::artifact::json::to_string(&value)?)?;
    fs::create_dir_all(&job.output)?;
    if job.output.join("manifest.json").exists() {
        return Err("refusing to overwrite an existing corpus manifest".into());
    }
    let (documents, sources) = collect(&job)?;
    let tokenizer = tokenizer(&job)?;
    pack(&job, &documents, &sources, &tokenizer)
}

fn collect(job: &Job) -> Result<(Documents, Vec<Value>)> {
    if job.paths.len() != 1 {
        return Err("trace recipe requires exactly one checksum-pinned source shard".into());
    }
    let mut documents: Documents = std::array::from_fn(|_| Vec::new());
    let parts = holdout(job.recipe["split_policy"].as_str())?;
    let mut seen = HashSet::new();
    let mut sources = Vec::new();
    for path in &job.paths {
        let checksum = digest(path)?;
        if job.paths.len() == 1 {
            let expected = job
                .recipe
                .get("source")
                .and_then(|v| v.get("sha256"))
                .and_then(|v| v.kind().as_str())
                .ok_or("pinned trace source requires sha256")?;
            if checksum != expected {
                return Err("trace source checksum mismatch".into());
            }
        }
        eprintln!("corpus | reading {}", path.display());
        let reader = SerializedFileReader::new(File::open(path)?)?;
        for row in reader.get_row_iter(None)? {
            let serialized = row?.to_json_value().to_string();
            if SOURCE_MARKERS
                .iter()
                .any(|marker| serialized.contains(marker))
            {
                return Err("source trace contains a reserved serialization marker".into());
            }
            let trace: Trace = baux::artifact::json::from_str(&serialized)?;
            if trace.repo.is_empty() || trace.messages.is_empty() || trace.instance_id.is_empty() {
                return Err("source trace has no repository, instance, or messages".into());
            }
            if !seen.insert((trace.repo.clone(), trace.instance_id.clone())) {
                continue;
            }
            let index = split_with(&trace.repo, parts);
            documents[index].push((trace.repo.clone(), render(&trace)?));
        }
        sources.push(json!({"file":path,"sha256":checksum}));
    }
    for split in &mut documents {
        split.sort_by(|left, right| left.0.cmp(&right.0));
    }
    if documents.iter().any(Vec::is_empty) {
        return Err("source shards do not contain all three repository-disjoint splits".into());
    }
    Ok((documents, sources))
}

fn tokenizer(job: &Job) -> Result<Tokenizer> {
    super::tokenizer::from_recipe(&job.tokenizer, &job.recipe)
}

fn pack(job: &Job, documents: &Documents, sources: &[Value], tokenizer: &Tokenizer) -> Result<()> {
    let tokenizer_path = job.output.join("tokenizer.json");
    tokenizer.save(&tokenizer_path, true)?;
    let mut records = BTreeMap::new();
    for (index, name) in SPLITS.iter().enumerate() {
        records.insert(*name, pack_split(job, &documents[index], name, tokenizer)?);
    }
    let mut special: BTreeMap<_, _> = super::tokenizer::MARKERS
        .iter()
        .map(|s| (*s, tokenizer.token_to_id(s)))
        .collect();
    special.insert("<bos>", tokenizer.token_to_id("<bos>"));
    special.insert("<eos>", tokenizer.token_to_id("<eos>"));
    let manifest = json!({"format":"baux.pretraining.v2","recipe_id":job.recipe_id,"recipe":job.recipe,
        "source":job.recipe.get("source"),"split_policy":job.recipe.get("split_policy"),"source_files":sources,
        "tokenizer":{"sha256":digest(&tokenizer_path)?,"vocabulary":262144,
            "base":job.recipe.get("tokenizer"),"special_tokens":special},"splits":records});
    fs::write(
        job.output.join("manifest.json"),
        baux::artifact::json::pretty_string(&manifest)?,
    )?;
    Ok(())
}

fn pack_split(
    job: &Job,
    documents: &[(String, String)],
    name: &str,
    tokenizer: &Tokenizer,
) -> Result<Value> {
    let quota = job
        .recipe
        .get("materialization")
        .and_then(|value| value.get("tokens"))
        .and_then(|v| v.get(name))
        .and_then(|v| v.kind().as_u64())
        .ok_or("recipe has no token budget")?;
    if quota < 1_048_576 || quota % 8_192 != 0 {
        return Err(
            "trace split token budgets must be multiples of 8192 and at least 1048576".into(),
        );
    }
    let sequences = quota / 4096;
    if sequences > u32::MAX as u64 {
        return Err("trace split token budget exceeds the corpus format".into());
    }
    let quota = usize::try_from(quota)?;
    let text_path = job.output.join(format!("{name}.txt"));
    let (mut tokens, episodes, repository_names) =
        encode_episodes(documents, &text_path, tokenizer, quota)?;
    if tokens.len() < quota {
        return Err(format!(
            "{name} has {} tokens; {quota} required; add source shards, do not pad",
            tokens.len()
        )
        .into());
    }
    let tail = tokens.len() - quota;
    tokens.truncate(quota);
    if tokens.iter().any(|&token| token >= 262_144) {
        return Err("encoded token exceeds the model vocabulary".into());
    }
    let path = job.output.join(format!("{name}.bauxptn"));
    let mut output = std::io::BufWriter::new(File::create(&path)?);
    output.write_all(b"BAUXPTN1")?;
    for value in [262144, 4096, sequences as u32, 0] {
        output.write_all(&value.to_le_bytes())?;
    }
    if cfg!(target_endian = "little") {
        let bytes =
            unsafe { std::slice::from_raw_parts(tokens.as_ptr().cast::<u8>(), tokens.len() * 4) };
        output.write_all(bytes)?;
    } else {
        for token in tokens {
            output.write_all(&token.to_le_bytes())?;
        }
    }
    output.flush()?;
    let record = json!({"sequences":sequences,"tokens":quota,"sha256":digest(&path)?,"episodes":episodes,"repositories":repository_names,"discarded_tail_tokens":tail});
    eprintln!("corpus | {name}: {quota} tokens from {episodes} episodes");
    Ok(record)
}

fn encode_episodes(
    documents: &[(String, String)],
    text_path: &Path,
    tokenizer: &Tokenizer,
    quota: usize,
) -> Result<(Vec<u32>, usize, Vec<String>)> {
    let (bos, eos) = super::tokenizer::boundaries(tokenizer)?;
    let mut tokens = Vec::with_capacity(quota);
    let mut episodes = 0usize;
    let mut used_repositories = HashSet::new();
    let mut previous_repository: Option<&str> = None;
    let mut text = File::create(text_path)?;
    for chunk in documents.chunks(32) {
        let batch = serialize_chunk(chunk, &mut previous_repository);
        let chunk_slices: Vec<&str> = batch.iter().map(String::as_str).collect();
        let encoded = tokenizer.encode_batch(chunk_slices, false)?;
        for ((repository, _), (serialized, ids)) in chunk.iter().zip(batch.iter().zip(&encoded)) {
            if tokenizer.decode(ids.get_ids(), false)? != *serialized {
                return Err("tokenizer round-trip changed trace text".into());
            }
            used_repositories.insert(repository.clone());
            text.write_all(serialized.as_bytes())?;
            tokens.push(bos);
            tokens.extend_from_slice(ids.get_ids());
            tokens.push(eos);
            episodes += 1;
            if tokens.len() >= quota {
                break;
            }
        }
        if tokens.len() >= quota {
            break;
        }
    }
    let mut repository_names: Vec<_> = used_repositories.into_iter().collect();
    repository_names.sort();
    Ok((tokens, episodes, repository_names))
}

fn serialize_chunk<'a>(
    chunk: &'a [(String, String)],
    previous: &mut Option<&'a str>,
) -> Vec<String> {
    chunk
        .iter()
        .map(|(repository, document)| {
            let first = *previous != Some(repository.as_str());
            *previous = Some(repository.as_str());
            if first {
                document.replacen("<|episode|>\n", "<|episode|><|repository|>\n", 1)
            } else {
                document.clone()
            }
        })
        .collect()
}

fn render(trace: &Trace) -> Result<String> {
    let mut text = String::from("<|episode|>\n");
    if let Some(tools) = &trace.tools {
        text.push_str("tools=");
        text.push_str(&baux::artifact::json::to_string(tools)?);
        text.push(char::from(10));
    }
    for message in &trace.messages {
        if !matches!(
            message.role.as_str(),
            "system" | "user" | "assistant" | "tool"
        ) {
            return Err(format!("unsupported source role {:?}", message.role).into());
        }
        text.push_str("<|turn|>");
        text.push_str(match message.role.as_str() {
            "system" => "<|system|>",
            "user" => "<|user|>",
            "assistant" => "<|assistant|>",
            "tool" => "<|tool|>",
            _ => unreachable!("role was validated above"),
        });
        if let Some(name) = &message.name {
            text.push_str(&format!(" name={}", baux::artifact::json::to_string(name)?));
        }
        if let Some(id) = &message.tool_call_id {
            text.push_str(&format!(" call={}", baux::artifact::json::to_string(id)?));
        }
        text.push('\n');
        if let Some(reasoning) = &message.reasoning_content {
            text.push_str("<|reasoning|>");
            text.push_str(reasoning);
        }
        if let Some(content) = &message.content {
            text.push_str("<|content|>");
            text.push_str(content);
        }
        if let Some(calls) = &message.tool_calls {
            text.push_str("<|toolcalls|>");
            text.push_str(&baux::artifact::json::to_string(calls)?);
        }
        text.push_str("<|endturn|>\n");
    }
    Ok(text)
}

fn digest(path: &Path) -> Result<String> {
    super::tokenizer::digest(path)
}
