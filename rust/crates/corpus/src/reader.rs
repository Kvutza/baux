use std::collections::{BTreeMap, HashSet};

use arrow_array::{Array, BooleanArray, ListArray, RecordBatch, StringArray, StructArray};
use deser::{Deserialize, Serialize};
use futures::{StreamExt, TryStreamExt};
use opendal::{Operator, layers, services};
use parquet::arrow::arrow_reader::{ArrowPredicateFn, RowFilter};
use parquet::arrow::{ParquetRecordBatchStreamBuilder, ProjectionMask};
use parquet_opendal::AsyncReader;
use sha2::Digest;

pub(crate) type Result<T> = std::result::Result<T, Box<dyn std::error::Error + Send + Sync>>;
pub(crate) type Counts = BTreeMap<String, BTreeMap<String, usize>>;
pub(crate) const BUCKETS: [&str; 4] = ["implementation", "tests", "documentation", "configuration"];
pub(crate) const SPLITS: [&str; 3] = ["train", "validation", "test"];

#[derive(Deserialize)]
pub(crate) struct Job {
    pub(crate) root: String,
    pub(crate) paths: Vec<String>,
    #[deser(default)]
    pub(crate) targets: Counts,
    #[deser(default)]
    pub(crate) target_characters: BTreeMap<String, usize>,
    #[deser(default)]
    pub(crate) languages: Vec<String>,
    #[deser(default)]
    pub(crate) excluded_names: Vec<String>,
    #[deser(default = one())]
    pub(crate) holdout_partitions: u64,
    pub(crate) max_file: usize,
    pub(crate) max_repository: usize,
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
    pub(crate) seen: HashSet<String>,
}

pub(crate) fn split(repository: &str) -> usize {
    split_with(repository, 1)
}

fn one() -> u64 {
    1
}

pub(crate) fn split_with(repository: &str, holdout: u64) -> usize {
    let mut hash = sha2::Sha256::new();
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

pub(crate) fn classify(
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
    pub(crate) fn new(job: &Job) -> Self {
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

    pub(crate) fn active(&self, job: &Job) -> u16 {
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

    fn report(&self, job: &Job, started: std::time::Instant) {
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

pub(super) async fn collect(job: Job) -> Result<Selection> {
    if (job.targets.is_empty() == job.target_characters.is_empty())
        || (!job.targets.is_empty()
            && job.targets.iter().any(|(s, b)| {
                !SPLITS.contains(&s.as_str())
                    || b.len() != BUCKETS.len()
                    || BUCKETS.iter().any(|name| !b.contains_key(*name))
            }))
        || (!job.target_characters.is_empty()
            && job
                .target_characters
                .iter()
                .any(|(split, target)| !SPLITS.contains(&split.as_str()) || *target == 0))
        || job.languages.is_empty()
    {
        return Err("invalid corpus language or split targets".into());
    }
    let operator = if job.root.starts_with("https://") || job.root.starts_with("http://") {
        Operator::new(services::Http::default().endpoint(&job.root))?.finish()
    } else {
        Operator::new(services::Fs::default().root(&job.root))?.finish()
    }
    .layer(layers::TimeoutLayer::new().with_timeout(std::time::Duration::from_secs(60)))
    .layer(layers::RetryLayer::new().with_max_times(3));
    let mut selected = Selection::new(&job);
    let started = std::time::Instant::now();
    let reads = futures::stream::iter(job.paths.iter().enumerate())
        .map(|(index, path)| {
            let operator = &operator;
            let job = &job;
            async move {
                eprintln!(
                    "read | shard {}/{} | {}",
                    index + 1,
                    job.paths.len(),
                    path.rsplit('/').next().unwrap_or(path)
                );
                read_shard(operator, path, job).await
            }
        })
        .buffered(8);
    futures::pin_mut!(reads);
    while let Some(shard) = reads.next().await {
        selected.merge(shard?, &job);
        selected.report(&job, started);
        if selected.active(&job) == 0 {
            break;
        }
    }
    selected.report(&job, started);
    if selected.active(&job) != 0 {
        return Err("source exhausted before corpus quotas were met".into());
    }
    Ok(selected)
}

async fn read_shard(operator: &Operator, path: &str, job: &Job) -> Result<Selection> {
    use std::sync::atomic::Ordering;
    let mut selected = Selection::new(job);
    let active = std::sync::Arc::new(std::sync::atomic::AtomicU16::new(selected.active(job)));
    let length = operator.stat(path).await?.content_length();
    let reader = operator
        .reader_with(path)
        .gap(512 * 1024)
        .chunk(8 * 1024 * 1024)
        .concurrent(4)
        .await?;
    let builder = ParquetRecordBatchStreamBuilder::new(AsyncReader::new(reader, length)).await?;
    let schema = builder.parquet_schema();
    let projection = projection(schema);
    let filter_mask = ProjectionMask::leaves(
        schema,
        schema.columns().iter().enumerate().filter_map(|(i, c)| {
            (matches!(c.path().parts()[0].as_str(), "repo_path" | "commit_id")
                || (c.path().parts()[0] == "github_metadata" && c.name() == "is_fork"))
                .then_some(i)
        }),
    );
    let mask = active.clone();
    let file_mask = ProjectionMask::leaves(
        schema,
        schema.columns().iter().enumerate().filter_map(|(i, c)| {
            (matches!(c.path().parts()[0].as_str(), "repo_path" | "commit_id")
                || (c.path().parts()[0] == "files"
                    && ["file_path", "language", "license_type", "is_vendor"].contains(&c.name())))
            .then_some(i)
        }),
    );
    let holdout = job.holdout_partitions;
    let predicate = ArrowPredicateFn::new(filter_mask, move |batch| {
        Ok(eligible(&batch, mask.load(Ordering::Relaxed), holdout)
            .map_err(|e| parquet::errors::ParquetError::General(e.to_string()))?)
    });
    let mask = active.clone();
    let holdout = job.holdout_partitions;
    let languages = job.languages.clone();
    let excluded_names = job.excluded_names.clone();
    let files = ArrowPredicateFn::new(file_mask, move |batch| {
        Ok(eligible_files(
            &batch,
            mask.load(Ordering::Relaxed),
            holdout,
            &languages,
            &excluded_names,
        )
        .map_err(|e| parquet::errors::ParquetError::General(e.to_string()))?)
    });
    let mut stream = builder
        .with_batch_size(64)
        .with_projection(projection)
        .with_row_filter(RowFilter::new(vec![Box::new(predicate), Box::new(files)]))
        .build()?;
    while let Some(batch) = stream.try_next().await? {
        selected.accept(&batch, job)?;
        active.store(selected.active(job), Ordering::Relaxed);
        if active.load(Ordering::Relaxed) == 0 {
            break;
        }
    }
    Ok(selected)
}

fn projection(schema: &parquet::schema::types::SchemaDescriptor) -> ProjectionMask {
    ProjectionMask::leaves(
        schema,
        schema.columns().iter().enumerate().filter_map(|(i, c)| {
            let parts = c.path().parts();
            let keep = match parts[0].as_str() {
                "repo_path" | "commit_id" => true,
                "github_metadata" => c.name() == "is_fork",
                "files" => [
                    "file_path",
                    "language",
                    "license_type",
                    "is_vendor",
                    "content_id",
                    "content",
                ]
                .contains(&c.name()),
                _ => false,
            };
            keep.then_some(i)
        }),
    )
}
