//! Document-preserving text streams for base pretraining corpora.
use super::encoding::Encoding;
use super::reader::{Result, SPLITS, split};
use arrow_array::{Array, LargeStringArray, RecordBatch, StringArray};
use baux::artifact::json::{Value, json};
use deser::Deserialize;
use futures::TryStreamExt;
use opendal::Operator;
use parquet::arrow;
use parquet_opendal::AsyncReader;
use std::collections::HashSet;
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

const CONTEXT: u32 = 4096;

#[derive(Deserialize)]
struct Job {
    root: String,
    paths: Vec<String>,
    tokenizer: Option<PathBuf>,
    output: PathBuf,
    recipe_id: String,
    recipe: Value,
}

enum TextColumn<'a> {
    Utf8(&'a StringArray),
    Large(&'a LargeStringArray),
}

impl TextColumn<'_> {
    fn value(&self, row: usize) -> Option<&str> {
        match self {
            Self::Utf8(values) => values.is_valid(row).then(|| values.value(row)),
            Self::Large(values) => values.is_valid(row).then(|| values.value(row)),
        }
    }
}

struct SplitWriter {
    name: &'static str,
    path: PathBuf,
    output: BufWriter<File>,
    quota: u64,
    tokens: u64,
    documents: u64,
    discarded: u64,
    vocabulary: u32,
}

impl SplitWriter {
    fn new(root: &Path, name: &'static str, quota: u64, vocabulary: u32) -> Result<Self> {
        if quota == 0 || quota % (2 * u64::from(CONTEXT)) != 0 {
            return Err("text quota must contain a positive even number of contexts".into());
        }
        let path = root.join(format!("{name}.bauxptn"));
        let mut output = BufWriter::new(File::create(&path)?);
        output.write_all(b"BAUXPTN1")?;
        let sequences = quota / u64::from(CONTEXT);
        for value in [vocabulary, CONTEXT, u32::try_from(sequences)?, 0] {
            output.write_all(&value.to_le_bytes())?;
        }
        Ok(Self {
            name,
            path,
            output,
            quota,
            tokens: 0,
            documents: 0,
            discarded: 0,
            vocabulary,
        })
    }

    fn full(&self) -> bool {
        self.tokens == self.quota
    }

    fn write_document(&mut self, bos: u32, ids: &[u32], eos: u32) -> Result<()> {
        if self.full() {
            return Ok(());
        }
        let available = usize::try_from(self.quota - self.tokens)?;
        let total = ids.len().checked_add(2).ok_or("document token overflow")?;
        let count = available.min(total);
        for token in std::iter::once(bos)
            .chain(ids.iter().copied())
            .chain(std::iter::once(eos))
            .take(count)
        {
            if token >= self.vocabulary {
                return Err("text tokenizer emitted an out-of-range token".into());
            }
            if self.vocabulary <= 65_536 {
                self.output
                    .write_all(&u16::try_from(token)?.to_le_bytes())?;
            } else {
                self.output.write_all(&token.to_le_bytes())?;
            }
        }
        self.tokens += u64::try_from(count)?;
        self.discarded += u64::try_from(total - count)?;
        self.documents += 1;
        Ok(())
    }

    fn finish(mut self) -> Result<(&'static str, Value)> {
        if !self.full() {
            return Err(format!(
                "{} produced {} tokens; {} required; add source shards",
                self.name, self.tokens, self.quota
            )
            .into());
        }
        self.output.flush()?;
        let record = json!({
            "sequences":self.quota / u64::from(CONTEXT),
            "tokens":self.tokens,
            "sha256":super::tokenizer::digest(&self.path)?,
            "documents":self.documents,
            "discarded_tail_tokens":self.discarded,
        });
        eprintln!(
            "corpus | {}: {} tokens from {} documents",
            self.name, self.tokens, self.documents
        );
        Ok((self.name, record))
    }
}

pub(super) async fn prepare(value: Value) -> Result<()> {
    let job: Job = baux::artifact::json::from_str(&baux::artifact::json::to_string(&value)?)?;
    fs::create_dir_all(&job.output)?;
    if job.output.join("manifest.json").exists() {
        return Err("refusing to overwrite an existing corpus manifest".into());
    }
    let tokenizer = Encoding::new(job.tokenizer.as_deref(), &job.recipe)?;
    let identity = tokenizer.persist(&job.output, &job.recipe)?;
    let records = collect(&job, &tokenizer).await?;
    let manifest = json!({
        "format":"baux.pretraining.v2",
        "recipe_id":job.recipe_id,
        "recipe":job.recipe,
        "source":job.recipe.get("source"),
        "split_policy":job.recipe.get("split_policy"),
        "serialization":"text_document_v1",
        "tokenizer":identity,
        "splits":records,
    });
    fs::write(
        job.output.join("manifest.json"),
        baux::artifact::json::pretty_string(&manifest)?,
    )?;
    Ok(())
}

fn build_operator(root: &str) -> Result<Operator> {
    let op = if root.starts_with("https://") || root.starts_with("http://") {
        Operator::new(opendal::services::Http::default().endpoint(root))?.finish()
    } else {
        Operator::new(opendal::services::Fs::default().root(root))?.finish()
    };
    Ok(op
        .layer(
            opendal::layers::TimeoutLayer::new().with_timeout(std::time::Duration::from_secs(60)),
        )
        .layer(opendal::layers::RetryLayer::new().with_max_times(3)))
}

async fn collect(job: &Job, tokenizer: &Encoding) -> Result<Value> {
    let source = &job.recipe["source"];
    let text_field = source["text_field"]
        .as_str()
        .ok_or("text recipe has no text_field")?;
    let id_field = source["id_field"]
        .as_str()
        .ok_or("text recipe has no id_field")?;
    let quotas = &job.recipe["materialization"]["tokens"];
    let mut writers = Vec::with_capacity(SPLITS.len());
    for name in SPLITS {
        let quota = quotas[name]
            .as_u64()
            .ok_or("text recipe has no split token budget")?;
        writers.push(SplitWriter::new(
            &job.output,
            name,
            quota,
            tokenizer.vocabulary(),
        )?);
    }
    let operator = build_operator(&job.root)?;
    let (bos, eos) = tokenizer.boundaries()?;
    let mut seen = HashSet::new();
    for (number, path) in job.paths.iter().enumerate() {
        if writers.iter().all(SplitWriter::full) {
            break;
        }
        eprintln!(
            "corpus | text shard {}/{} | {}",
            number + 1,
            job.paths.len(),
            path.rsplit('/').next().unwrap_or(path)
        );
        let length = operator.stat(path).await?.content_length();
        let reader = operator
            .reader_with(path)
            .gap(512 * 1024)
            .chunk(8 * 1024 * 1024)
            .concurrent(4)
            .await?;
        let builder =
            arrow::ParquetRecordBatchStreamBuilder::new(AsyncReader::new(reader, length)).await?;
        let schema = builder.parquet_schema();
        let projection = arrow::ProjectionMask::leaves(
            schema,
            schema.columns().iter().enumerate().filter_map(|(index, field)| {
                matches!(field.path().parts()[0].as_str(), value if value == text_field || value == id_field)
                    .then_some(index)
            }),
        );
        let mut stream = builder
            .with_batch_size(64)
            .with_projection(projection)
            .build()?;
        while let Some(batch) = stream.try_next().await? {
            consume(
                &batch,
                text_field,
                id_field,
                tokenizer,
                bos,
                eos,
                &mut seen,
                &mut writers,
            )?;
            if writers.iter().all(SplitWriter::full) {
                break;
            }
        }
    }
    let missing = writers
        .iter()
        .filter(|writer| !writer.full())
        .map(|writer| format!("{}={}", writer.name, writer.quota - writer.tokens))
        .collect::<Vec<_>>();
    if !missing.is_empty() {
        return Err(format!(
            "text sources exhausted with tokens remaining: {}",
            missing.join(", ")
        )
        .into());
    }
    let mut records = baux::artifact::json::Map::new();
    for writer in writers {
        let (name, record) = writer.finish()?;
        records.insert(name, record);
    }
    Ok(Value::from(records))
}

#[allow(clippy::too_many_arguments)]
fn consume(
    batch: &RecordBatch,
    text_field: &str,
    id_field: &str,
    tokenizer: &Encoding,
    bos: u32,
    eos: u32,
    seen: &mut HashSet<String>,
    writers: &mut [SplitWriter],
) -> Result<()> {
    let texts = text_column(batch, text_field)?;
    let ids = text_column(batch, id_field)?;
    for row in 0..batch.num_rows() {
        let Some(id) = ids.value(row) else {
            continue;
        };
        let Some(text) = texts.value(row) else {
            continue;
        };
        if text.is_empty() || !seen.insert(id.to_owned()) || !tokenizer.accepts(text) {
            continue;
        }
        let writer = &mut writers[split(id)];
        if writer.full() {
            continue;
        }
        let encoding = tokenizer.encode(text)?;
        writer.write_document(bos, &encoding, eos)?;
    }
    Ok(())
}

fn text_column<'a>(batch: &'a RecordBatch, name: &str) -> Result<TextColumn<'a>> {
    let column = batch
        .column_by_name(name)
        .ok_or_else(|| format!("text source has no {name} column"))?;
    if let Some(values) = column.as_any().downcast_ref::<StringArray>() {
        return Ok(TextColumn::Utf8(values));
    }
    if let Some(values) = column.as_any().downcast_ref::<LargeStringArray>() {
        return Ok(TextColumn::Large(values));
    }
    Err(format!("text source {name} column is not UTF-8").into())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    #[test]
    fn bytes() -> Result<()> {
        let root = std::env::temp_dir().join(format!("baux-text-bytes-{}", std::process::id()));
        fs::create_dir_all(&root)?;
        let recipe =
            json!({"tokenizer":{"kind":"utf8_bytes","vocabulary":256,"bos":254,"eos":255}});
        let encoding = Encoding::new(None, &recipe)?;
        let metadata = encoding.persist(&root, &recipe)?;
        assert!(!root.join("tokenizer.json").exists());
        assert_eq!(metadata["special_tokens"]["<eos>"].as_u64(), Some(255));
        let mut writers: Vec<_> = SPLITS
            .iter()
            .map(|&name| SplitWriter::new(&root, name, 8192, encoding.vocabulary()))
            .collect::<Result<_>>()?;
        // Each split gets a distinct document. Literal marker text remains
        // ordinary bytes.
        let ids: Vec<_> = (0..3)
            .map(|target| {
                (0..1000)
                    .map(|id| format!("document-{id}"))
                    .find(|id| split(id) == target)
                    .expect("split has an ID")
            })
            .collect();
        let text = "é 🦀 <bos> <|repo|>\n".repeat(600);
        let batch = RecordBatch::try_from_iter([
            (
                "text",
                Arc::new(StringArray::from(vec![text.as_str(); 3])) as Arc<dyn Array>,
            ),
            (
                "id",
                Arc::new(StringArray::from(
                    ids.iter().map(String::as_str).collect::<Vec<_>>(),
                )) as Arc<dyn Array>,
            ),
        ])?;
        consume(
            &batch,
            "text",
            "id",
            &encoding,
            254,
            255,
            &mut HashSet::new(),
            &mut writers,
        )?;
        for writer in writers {
            let path = writer.path.clone();
            let (_, record) = writer.finish()?;
            assert_eq!(record["tokens"].as_u64(), Some(8192));
            assert_eq!(fs::metadata(&path)?.len(), 24 + 8192 * 2);
            let stored = baux::artifact::pretrain::load_tokens(&path, 8192)?;
            let expected: Vec<u32> = std::iter::once(254)
                .chain(text.as_bytes().iter().map(|&byte| u32::from(byte)))
                .take(8192)
                .collect();
            assert_eq!(stored, expected);
        }
        fs::remove_dir_all(root)?;
        Ok(())
    }
}
