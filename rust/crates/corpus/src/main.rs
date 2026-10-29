use std::collections::{BTreeMap, HashSet};
use std::io::{self, BufReader, BufWriter, Write};
use std::sync::Arc;
use std::sync::atomic::{AtomicU16, Ordering};
use std::time::{Duration, Instant};

use arrow_array::{Array, BooleanArray, ListArray, RecordBatch, StringArray, StructArray};
use deser::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

type Result<T> = std::result::Result<T, Box<dyn std::error::Error + Send + Sync>>;

mod fineweb;
mod mix;
mod reader;
mod stack;
mod text;
mod tokenizer;
mod traces;
use reader::collect;
type Counts = BTreeMap<String, BTreeMap<String, usize>>;
const BUCKETS: [&str; 4] = ["implementation", "tests", "documentation", "configuration"];
const SPLITS: [&str; 3] = ["train", "validation", "test"];

#[derive(Deserialize)]
pub(crate) struct Job {
    root: String,
    paths: Vec<String>,
    #[deser(default)]
    targets: Counts,
    #[deser(default)]
    target_characters: BTreeMap<String, usize>,
    #[deser(default)]
    languages: Vec<String>,
    #[deser(default)]
    excluded_names: Vec<String>,
    #[deser(default = one())]
    holdout_partitions: u64,
    max_file: usize,
    max_repository: usize,
}

#[derive(Serialize)]
pub(crate) struct Document {
    pub(crate) repository: String,
    pub(crate) commit: String,
    pub(crate) content_id: String,
    pub(crate) path: String,
    pub(crate) bucket: String,
    pub(crate) text: String,
}

#[derive(Serialize)]
pub(crate) struct Selection {
    pub(crate) documents: BTreeMap<String, BTreeMap<String, Vec<Document>>>,
    pub(crate) characters: Counts,
    pub(crate) total_characters: BTreeMap<String, usize>,
    #[deser(skip)]
    seen: HashSet<String>,
}

fn split(repository: &str) -> usize {
    split_with(repository, 1)
}

fn one() -> u64 {
    1
}

fn split_with(repository: &str, holdout: u64) -> usize {
    let mut hash = Sha256::new();
    hash.update(repository);
    let partition = u64::from_le_bytes(hash.finalize()[..8].try_into().unwrap()) % 20;
    if partition >= 20 - holdout {
        2
    } else if partition >= 20 - 2 * holdout {
        1
    } else {
        0
    }
}

fn classify(
    path: &str,
    language: &str,
    languages: &[String],
    excluded_names: &[String],
) -> Option<&'static str> {
    let path = format!("/{}", path.to_lowercase().trim_start_matches('/'));
    let name = path.rsplit('/').next().unwrap_or("");
    if excluded_names.iter().any(|excluded| excluded == name) {
        return None;
    }
    if languages.iter().any(|allowed| allowed == language) {
        return Some(
            if path.contains("/test/")
                || path.contains("/tests/")
                || path.contains("/benches/")
                || name.ends_with("_test.go")
            {
                "tests"
            } else {
                "implementation"
            },
        );
    }
    if matches!(language, "Markdown" | "reStructuredText")
        || ["readme", "contributing", "architecture"]
            .iter()
            .any(|s| name.starts_with(s))
    {
        return Some("documentation");
    }
    (matches!(
        language,
        "Dockerfile" | "INI" | "JSON" | "Makefile" | "Protocol Buffer" | "Shell" | "TOML" | "YAML"
    ) || matches!(name, "go.mod" | "build.rs"))
    .then_some("configuration")
}

fn typed<T: 'static>(array: &dyn Array) -> Result<&T> {
    array
        .as_any()
        .downcast_ref()
        .ok_or_else(|| format!("unexpected Arrow type: {:?}", array.data_type()).into())
}

fn field<'a, T: 'static>(array: &'a StructArray, name: &str) -> Result<&'a T> {
    typed(
        array
            .column_by_name(name)
            .ok_or_else(|| format!("missing field {name}"))?
            .as_ref(),
    )
}

fn column<'a, T: 'static>(batch: &'a RecordBatch, name: &str) -> Result<&'a T> {
    typed(
        batch
            .column_by_name(name)
            .ok_or_else(|| format!("missing column {name}"))?
            .as_ref(),
    )
}

fn eligible(batch: &RecordBatch, active: u16, holdout: u64) -> Result<BooleanArray> {
    let repos: &StringArray = column(batch, "repo_path")?;
    let commits: &StringArray = column(batch, "commit_id")?;
    let metadata: &StructArray = column(batch, "github_metadata")?;
    let forks: &BooleanArray = field(metadata, "is_fork")?;
    Ok((0..batch.num_rows())
        .map(|row| {
            Some(
                !(metadata.is_valid(row) && forks.is_valid(row) && forks.value(row))
                    && repos.is_valid(row)
                    && commits.is_valid(row)
                    && active & (15 << (4 * split_with(repos.value(row), holdout))) != 0,
            )
        })
        .collect())
}

fn eligible_files(
    batch: &RecordBatch,
    active: u16,
    holdout: u64,
    allowed_languages: &[String],
    excluded_names: &[String],
) -> Result<BooleanArray> {
    let repos: &StringArray = column(batch, "repo_path")?;
    let lists: &ListArray = column(batch, "files")?;
    let files: &StructArray = typed(lists.values().as_ref())?;
    let paths: &StringArray = field(files, "file_path")?;
    let file_languages: &StringArray = field(files, "language")?;
    let licenses: &StringArray = field(files, "license_type")?;
    let vendors: &BooleanArray = field(files, "is_vendor")?;
    let offsets = lists.value_offsets();
    Ok((0..batch.num_rows())
        .map(|row| {
            let needed = active >> (4 * split_with(repos.value(row), holdout));
            let range = offsets[row] as usize..offsets[row + 1] as usize;
            Some(
                lists.is_valid(row)
                    && has_primary(
                        range.clone(),
                        files,
                        file_languages,
                        licenses,
                        vendors,
                        allowed_languages,
                    )
                    && has_requested(
                        range,
                        files,
                        paths,
                        file_languages,
                        licenses,
                        vendors,
                        needed,
                        allowed_languages,
                        excluded_names,
                    ),
            )
        })
        .collect())
}

fn admissible(
    index: usize,
    files: &StructArray,
    languages: &StringArray,
    licenses: &StringArray,
    vendors: &BooleanArray,
) -> bool {
    files.is_valid(index)
        && !(vendors.is_valid(index) && vendors.value(index))
        && languages.is_valid(index)
        && licenses.is_valid(index)
        && licenses.value(index) == "permissive"
}

fn has_primary(
    range: std::ops::Range<usize>,
    files: &StructArray,
    languages: &StringArray,
    licenses: &StringArray,
    vendors: &BooleanArray,
    allowed: &[String],
) -> bool {
    for index in range {
        if admissible(index, files, languages, licenses, vendors)
            && allowed
                .iter()
                .any(|language| language == languages.value(index))
        {
            return true;
        }
    }
    false
}

#[allow(clippy::too_many_arguments)]
fn has_requested(
    range: std::ops::Range<usize>,
    files: &StructArray,
    paths: &StringArray,
    languages: &StringArray,
    licenses: &StringArray,
    vendors: &BooleanArray,
    needed: u16,
    allowed: &[String],
    excluded_names: &[String],
) -> bool {
    for index in range {
        if admissible(index, files, languages, licenses, vendors)
            && paths.is_valid(index)
            && requested_bucket(
                paths.value(index),
                languages.value(index),
                needed,
                allowed,
                excluded_names,
            )
        {
            return true;
        }
    }
    false
}

fn requested_bucket(
    path: &str,
    language: &str,
    needed: u16,
    languages: &[String],
    excluded_names: &[String],
) -> bool {
    let Some(bucket) = classify(path, language, languages, excluded_names) else {
        return false;
    };
    let bit = BUCKETS.iter().position(|name| *name == bucket).unwrap();
    needed & (1 << bit) != 0
}

impl Selection {
    fn new(job: &Job) -> Self {
        let splits = if job.target_characters.is_empty() {
            job.targets.keys().cloned().collect::<Vec<_>>()
        } else {
            job.target_characters.keys().cloned().collect::<Vec<_>>()
        };
        Self {
            documents: splits
                .iter()
                .map(|s| {
                    (
                        s.clone(),
                        BUCKETS
                            .into_iter()
                            .map(|b| (b.into(), Vec::new()))
                            .collect(),
                    )
                })
                .collect(),
            characters: splits
                .iter()
                .map(|s| {
                    (
                        s.clone(),
                        BUCKETS.into_iter().map(|b| (b.into(), 0)).collect(),
                    )
                })
                .collect(),
            total_characters: splits.into_iter().map(|split| (split, 0)).collect(),
            seen: HashSet::new(),
        }
    }

    fn active(&self, job: &Job) -> u16 {
        let mut mask = 0;
        if !job.target_characters.is_empty() {
            for (index, split) in SPLITS.iter().enumerate() {
                if job.target_characters.get(*split).is_some_and(|target| {
                    self.total_characters.get(*split).copied().unwrap_or(0) < *target
                }) {
                    mask |= 15 << (4 * index);
                }
            }
            return mask;
        }
        for (index, split) in SPLITS.iter().enumerate() {
            let Some(buckets) = job.targets.get(*split) else {
                continue;
            };
            for (bucket, name) in BUCKETS.iter().enumerate() {
                if self.characters[*split][*name] < buckets[*name] {
                    mask |= 1 << (4 * index + bucket);
                }
            }
        }
        mask
    }

    fn merge(&mut self, shard: Selection, job: &Job) {
        for split in SPLITS {
            for bucket in BUCKETS {
                for document in shard.documents[split][bucket].iter() {
                    let characters = document.text.chars().count();
                    let full = if job.target_characters.is_empty() {
                        self.characters[split][bucket] >= job.targets[split][bucket]
                    } else {
                        self.total_characters[split] >= job.target_characters[split]
                    };
                    if full || !self.seen.insert(document.content_id.clone()) {
                        continue;
                    }
                    self.characters
                        .get_mut(split)
                        .unwrap()
                        .entry(bucket.into())
                        .and_modify(|count| *count += characters);
                    *self.total_characters.get_mut(split).unwrap() += characters;
                    self.documents
                        .get_mut(split)
                        .unwrap()
                        .get_mut(bucket)
                        .unwrap()
                        .push(Document {
                            repository: document.repository.clone(),
                            commit: document.commit.clone(),
                            content_id: document.content_id.clone(),
                            path: document.path.clone(),
                            bucket: document.bucket.clone(),
                            text: document.text.clone(),
                        });
                }
            }
        }
    }

    fn accept(&mut self, batch: &RecordBatch, job: &Job) -> Result<()> {
        let repos: &StringArray = column(batch, "repo_path")?;
        let commits: &StringArray = column(batch, "commit_id")?;
        let lists: &ListArray = column(batch, "files")?;
        let files: &StructArray = typed(lists.values().as_ref())?;
        let paths: &StringArray = field(files, "file_path")?;
        let languages: &StringArray = field(files, "language")?;
        let licenses: &StringArray = field(files, "license_type")?;
        let vendors: &BooleanArray = field(files, "is_vendor")?;
        let ids: &StringArray = field(files, "content_id")?;
        let texts: &StringArray = field(files, "content")?;
        for row in 0..batch.num_rows() {
            let repository = repos.value(row);
            let commit = commits.value(row);
            let index = split_with(repository, job.holdout_partitions);
            let split = SPLITS[index];
            if self.active(job) & (15 << (4 * index)) == 0 || lists.is_null(row) {
                continue;
            }
            let mut used = 0;
            let offsets = lists.value_offsets();
            let range = offsets[row] as usize..offsets[row + 1] as usize;
            if !has_primary(
                range.clone(),
                files,
                languages,
                licenses,
                vendors,
                &job.languages,
            ) {
                continue;
            }
            for index in offsets[row] as usize..offsets[row + 1] as usize {
                if used >= job.max_repository {
                    break;
                }
                if !admissible(index, files, languages, licenses, vendors)
                    || paths.is_null(index)
                    || ids.is_null(index)
                    || texts.is_null(index)
                {
                    continue;
                }
                let Some(bucket) = classify(
                    paths.value(index),
                    languages.value(index),
                    &job.languages,
                    &job.excluded_names,
                ) else {
                    continue;
                };
                if (!job.target_characters.is_empty()
                    && self.total_characters[split] >= job.target_characters[split])
                    || (job.target_characters.is_empty()
                        && self.characters[split][bucket] >= job.targets[split][bucket])
                    || self.seen.contains(ids.value(index))
                {
                    continue;
                }
                let text = texts.value(index);
                // Match Python len(str), not UTF-8 byte length.
                let count = text.chars().count();
                if !(64..=job.max_file).contains(&count)
                    || count > job.max_repository - used
                    || text.contains('\0')
                {
                    continue;
                }
                self.seen.insert(ids.value(index).into());
                self.documents
                    .get_mut(split)
                    .unwrap()
                    .get_mut(bucket)
                    .unwrap()
                    .push(Document {
                        repository: repository.into(),
                        commit: commit.into(),
                        content_id: ids.value(index).into(),
                        path: paths.value(index).into(),
                        bucket: bucket.into(),
                        text: text.into(),
                    });
                *self
                    .characters
                    .get_mut(split)
                    .unwrap()
                    .get_mut(bucket)
                    .unwrap() += count;
                *self.total_characters.get_mut(split).unwrap() += count;
                used += count;
            }
            if self.active(job) == 0 {
                break;
            }
        }
        Ok(())
    }

    fn report(&self, job: &Job, started: Instant) {
        let mut remaining = Vec::new();
        if job.target_characters.is_empty() {
            for (split, buckets) in &job.targets {
                for (bucket, target) in buckets {
                    let left = target.saturating_sub(self.characters[split][bucket]);
                    if left > 0 {
                        remaining.push(format!("{split}/{bucket}={left}"));
                    }
                }
            }
        } else {
            for (split, target) in &job.target_characters {
                let left = target.saturating_sub(self.total_characters[split]);
                if left > 0 {
                    remaining.push(format!("{split}={left}"));
                }
            }
        }
        eprintln!(
            "collect | kept {} documents | {:.1}s | characters remaining: {}",
            self.seen.len(),
            started.elapsed().as_secs_f64(),
            if remaining.is_empty() {
                "none".into()
            } else {
                remaining.join(", ")
            }
        );
    }
}

#[tokio::main(worker_threads = 4)]
async fn main() -> Result<()> {
    let job: baux::artifact::json::Value =
        baux::artifact::json::from_reader(BufReader::new(io::stdin().lock()))?;
    if job.get("kind").and_then(|v| v.kind().as_str()) == Some("traces") {
        return traces::prepare(job);
    }
    if job.get("kind").and_then(|v| v.kind().as_str()) == Some("stack") {
        return stack::prepare(job).await;
    }
    if job.get("kind").and_then(|v| v.kind().as_str()) == Some("text") {
        return text::prepare(job).await;
    }
    if job.get("kind").and_then(|v| v.kind().as_str()) == Some("fineweb") {
        return fineweb::prepare(job).await;
    }
    if job.get("kind").and_then(|v| v.kind().as_str()) == Some("mix") {
        return mix::prepare(job);
    }
    let job = baux::artifact::json::from_str(&baux::artifact::json::to_string(&job)?)?;
    let selected = collect(job).await?;
    let mut output = BufWriter::new(io::stdout().lock());
    baux::artifact::json::to_writer(&mut output, &selected)?;
    output.flush()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classification() {
        let languages = vec!["Rust".into(), "Go".into()];
        assert_eq!(
            classify("tests/model.rs", "Rust", &languages, &[]),
            Some("tests")
        );
        assert_eq!(
            classify("/src/model.rs", "Rust", &languages, &[]),
            Some("implementation")
        );
        assert_eq!(
            classify("Readme.custom", "Other", &languages, &[]),
            Some("documentation")
        );
        assert_eq!(
            classify("Cargo.toml", "TOML", &languages, &[]),
            Some("configuration")
        );
        assert_eq!(classify("main.py", "Python", &languages, &[]), None);
        assert_eq!(
            classify("Cargo.lock", "Rust", &languages, &["cargo.lock".into()]),
            None
        );
    }

    #[test]
    fn completion() {
        let targets = BTreeMap::from([(
            "train".into(),
            BUCKETS.into_iter().map(|b| (b.into(), 80)).collect(),
        )]);
        let job = Job {
            root: String::new(),
            paths: Vec::new(),
            targets: targets.clone(),
            target_characters: BTreeMap::new(),
            languages: vec!["Rust".into(), "Go".into()],
            excluded_names: vec!["cargo.lock".into(), "go.sum".into()],
            holdout_partitions: 1,
            max_file: 1024,
            max_repository: 4096,
        };
        let mut selected = Selection::new(&job);
        assert_eq!(selected.active(&job), 15);
        selected.characters = targets.clone();
        assert_eq!(selected.active(&job), 0);
    }
}
