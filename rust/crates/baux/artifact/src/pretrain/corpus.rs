//! Immutable training-corpus and held-out mixture scheduling.

use super::{CausalBatch, PretrainDataset, has_magic};

use deser::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashSet;
use std::io::Read;
use std::path::{Path, PathBuf};

const MIX_FORMAT: &str = "baux.pretraining-evaluation.v1";

#[derive(Debug)]
pub struct CorpusSet {
    parts: Vec<CorpusPart>,
    weight: u64,
    evaluation: bool,
}

#[derive(Debug)]
struct CorpusPart {
    name: String,
    weight: u32,
    data: PretrainDataset,
}

#[derive(Debug, Clone)]
pub struct CorpusBatch {
    pub source: String,
    pub weight: u32,
    pub batch: CausalBatch,
}

#[derive(Deserialize, Serialize)]
#[deser(deny_unknown_fields, rename_all = "kebab-case")]
struct MixSpec {
    format: String,
    aggregation: MixAggregation,
    parts: Vec<MixPart>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[deser(rename_all = "kebab-case")]
enum MixAggregation {
    WeightedMean,
}

#[derive(Deserialize, Serialize)]
#[deser(deny_unknown_fields, rename_all = "kebab-case")]
struct MixPart {
    name: String,
    weight: u32,
    tokens: u64,
    path: PathBuf,
    sha256: String,
    tokenizer_sha256: String,
}

/// Relocation plan for one corpus or held-out mixture. Only paths change;
/// source names, weights, counts, hashes and tokenizer identities are retained.
/// Every destination is generated here, never taken from an input pathname.
pub struct CorpusBundle {
    pub files: Vec<(PathBuf, PathBuf)>,
    pub manifest: Option<Vec<u8>>,
}

impl CorpusBundle {
    pub fn new(path: &Path) -> Result<Self, String> {
        let mut file = std::fs::File::open(path).map_err(|error| error.to_string())?;
        let mut magic = [0; 8];
        let count = file.read(&mut magic).map_err(|error| error.to_string())?;
        let mut bundle = Self {
            files: Vec::new(),
            manifest: None,
        };
        if count == 8 && has_magic(&magic) {
            bundle.add(path, Path::new("data.bauxptn"))?;
        } else {
            let mut spec: MixSpec = baux::artifact::json::from_reader(
                std::fs::File::open(path).map_err(|error| error.to_string())?,
            )
            .map_err(|error| error.to_string())?;
            if spec.format != MIX_FORMAT || !(2..=64).contains(&spec.parts.len()) {
                return Err("invalid evaluation mixture for relocation".into());
            }
            let parent = path.parent().ok_or("mixture has no parent")?;
            for (index, part) in spec.parts.iter_mut().enumerate() {
                let source = parent.join(&part.path);
                verify(&source, &part.sha256)?;
                verify(
                    &source
                        .parent()
                        .ok_or("corpus has no parent")?
                        .join("tokenizer.json"),
                    &part.tokenizer_sha256,
                )?;
                let destination = PathBuf::from(format!("part-{index}/data.bauxptn"));
                bundle.add(&source, &destination)?;
                part.path = destination;
            }
            bundle.manifest =
                Some(baux::artifact::json::to_vec(&spec).map_err(|error| error.to_string())?);
        }
        Ok(bundle)
    }

    fn add(&mut self, source: &Path, destination: &Path) -> Result<(), String> {
        self.files.push((source.to_owned(), destination.to_owned()));
        let parent = source.parent().ok_or("corpus has no parent")?;
        let target = destination.parent().ok_or("destination has no parent")?;
        // These are the exact companion names consumed by PretrainDataset.
        for name in ["manifest.json", "tokenizer.json"] {
            let path = parent.join(name);
            if path.is_file() {
                self.files.push((path, target.join(name)));
            }
        }
        Ok(())
    }
}

fn file_magic(path: &Path) -> Result<bool, String> {
    let mut file = std::fs::File::open(path)
        .map_err(|error| format!("pretraining source {}: {error}", path.display()))?;
    let mut magic = [0u8; 8];
    let count = file
        .read(&mut magic)
        .map_err(|error| format!("pretraining source {}: {error}", path.display()))?;
    Ok(count == 8 && has_magic(&magic))
}

impl CorpusSet {
    pub fn load_eval(path: &Path) -> Result<Self, String> {
        if file_magic(path)? {
            return Ok(Self {
                parts: vec![CorpusPart {
                    name: "corpus".into(),
                    weight: 1,
                    data: PretrainDataset::load(path)?,
                }],
                weight: 1,
                evaluation: true,
            });
        }
        Self::load(path)
    }

    pub fn load(path: &Path) -> Result<Self, String> {
        if file_magic(path)? {
            return Ok(Self {
                parts: vec![CorpusPart {
                    name: "corpus".into(),
                    weight: 1,
                    data: PretrainDataset::load(path)?,
                }],
                weight: 1,
                evaluation: false,
            });
        }
        let bytes = std::fs::read(path)
            .map_err(|error| format!("pretraining source {}: {error}", path.display()))?;
        let spec: MixSpec = baux::artifact::json::from_reader(bytes.as_slice())
            .map_err(|error| format!("pretraining mixture {}: {error}", path.display()))?;
        if spec.format != MIX_FORMAT || !(2..=64).contains(&spec.parts.len()) {
            return Err(format!(
                "pretraining mixture requires format={MIX_FORMAT} and 2..=64 parts"
            ));
        }
        if spec.aggregation != MixAggregation::WeightedMean {
            return Err("pretraining evaluation requires weighted-mean aggregation".into());
        }
        let parent = path
            .parent()
            .ok_or("pretraining mixture has no parent directory")?;
        let mut names = HashSet::with_capacity(spec.parts.len());
        let mut tokenizers = HashSet::with_capacity(spec.parts.len());
        let mut weight = 0u64;
        let mut parts = Vec::with_capacity(spec.parts.len());
        for part in spec.parts {
            if part.name.is_empty()
                || part.name.len() > 64
                || !part
                    .name
                    .bytes()
                    .all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit() || byte == b'-')
                || !names.insert(part.name.clone())
                || part.weight == 0
                || part.tokens == 0
                || part.sha256.len() != 64
                || part.tokenizer_sha256.len() != 64
            {
                return Err("pretraining mixture has an invalid name, weight, or digest".into());
            }
            let source = if part.path.is_absolute() {
                part.path
            } else {
                parent.join(part.path)
            };
            verify(&source, &part.sha256)?;
            let tokenizer = source
                .parent()
                .ok_or("mixture corpus has no parent directory")?
                .join("tokenizer.json");
            verify(&tokenizer, &part.tokenizer_sha256)?;
            tokenizers.insert(part.tokenizer_sha256);
            weight = weight
                .checked_add(u64::from(part.weight))
                .ok_or("pretraining mixture weight overflow")?;
            let data = PretrainDataset::load(&source)?;
            if data.tokens() as u64 != part.tokens {
                return Err("pretraining mixture token count differs from its artifact".into());
            }
            parts.push(CorpusPart {
                name: part.name,
                weight: part.weight,
                data,
            });
        }
        if tokenizers.len() != 1 {
            return Err("pretraining mixture parts must use the same tokenizer artifact".into());
        }
        Ok(Self {
            parts,
            weight,
            evaluation: true,
        })
    }

    /// Sample a deterministic causal batch from the primary training partition.
    ///
    /// # Invariants and Guarantees
    /// - **Disallow on evaluation mixtures**: Fails if `self.evaluation` is
    ///   true.
    /// - **Patch alignment**: Window offset is aligned to 64-token FBT patch
    ///   boundaries.
    /// - **Seed reproducibility**: Deterministic sampling via `(seed, round)`
    ///   passed to the partition's `causal_batch`.
    /// - **Lifetime / Ownership contract**: Returns an owned [`CorpusBatch`]
    ///   containing an owned [`CausalBatch`].
    pub fn causal_batch(
        &self,
        tokens: usize,
        seed: u64,
        round: u32,
    ) -> Result<CorpusBatch, String> {
        if self.evaluation {
            return Err("an evaluation mixture cannot supply a training batch".into());
        }
        let part = &self.parts[0];
        Ok(CorpusBatch {
            source: part.name.clone(),
            weight: part.weight,
            batch: part.data.causal_batch(tokens, seed, round)?,
        })
    }

    /// Retrieve weighted evaluation batches across all partitions in the
    /// held-out mixture.
    ///
    /// # Invariants and Guarantees
    /// - **Require evaluation mixture**: Fails if `!self.evaluation`.
    /// - **Round invariance**: Evaluates each partition at fixed round index 0,
    ///   guaranteeing identical evaluation windows across experimental rounds
    ///   for a given `seed`.
    /// - **Patch alignment**: Every batch offset and token count aligns with
    ///   64-token FBT patch boundaries.
    /// - **Seed reproducibility**: Each partition's window is deterministically
    ///   drawn using `(seed, 0)`.
    /// - **Lifetime / Ownership contract**: Returns an owned
    ///   `Vec<CorpusBatch>`.
    pub fn causal_batches(&self, tokens: usize, seed: u64) -> Result<Vec<CorpusBatch>, String> {
        if !self.evaluation {
            return Err("a training corpus cannot supply weighted evaluation batches".into());
        }
        self.parts
            .iter()
            .map(|part| {
                Ok(CorpusBatch {
                    source: part.name.clone(),
                    weight: part.weight,
                    batch: part.data.causal_batch(tokens, seed, 0)?,
                })
            })
            .collect()
    }

    pub const fn total_weight(&self) -> u64 {
        self.weight
    }
}

fn verify(path: &Path, expected: &str) -> Result<(), String> {
    let mut file = std::fs::File::open(path)
        .map_err(|error| format!("mixture artifact {}: {error}", path.display()))?;
    let mut hash = Sha256::new();
    let mut buffer = [0u8; 65_536];
    loop {
        let size = file.read(&mut buffer).map_err(|error| error.to_string())?;
        if size == 0 {
            break;
        }
        hash.update(&buffer[..size]);
    }
    if format!("{:x}", hash.finalize()) != expected {
        return Err(format!(
            "mixture artifact checksum mismatch: {}",
            path.display()
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::super::{CONTEXT, MAGIC};
    use super::*;

    fn corpus(path: &Path, token: u16) {
        let sequences = 258u32;
        let mut bytes = Vec::new();
        bytes.extend_from_slice(MAGIC);
        for value in [8192u32, CONTEXT, sequences, 0] {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        for _ in 0..u64::from(sequences) * u64::from(CONTEXT) {
            bytes.extend_from_slice(&token.to_le_bytes());
        }
        std::fs::write(path, bytes).unwrap();
    }

    fn digest(path: &Path) -> String {
        let bytes = std::fs::read(path).unwrap();
        format!("{:x}", Sha256::digest(bytes))
    }

    #[test]
    fn evaluation_sources() {
        let directory = tempfile::tempdir().unwrap();
        let mut parts = Vec::new();
        for (name, weight, token) in [("code", 3, 101), ("traces", 1, 202)] {
            let root = directory.path().join(name);
            std::fs::create_dir(&root).unwrap();
            let path = root.join("train.bauxptn");
            let tokenizer = root.join("tokenizer.json");
            corpus(&path, token);
            std::fs::write(&tokenizer, b"shared-tokenizer").unwrap();
            parts.push(baux::artifact::json::json!({
                "name":name,"weight":weight,"path":path,
                "tokens":u64::from(258u32) * u64::from(CONTEXT),
                "sha256":digest(&path),"tokenizer-sha256":digest(&tokenizer),
            }));
        }
        let manifest = directory.path().join("validation.json");
        std::fs::write(
            &manifest,
            baux::artifact::json::pretty_string(&baux::artifact::json::json!({
                "format":MIX_FORMAT,"aggregation":"weighted-mean","parts":parts,
            }))
            .unwrap(),
        )
        .unwrap();
        let source = CorpusSet::load(&manifest).unwrap();
        assert_eq!(source.total_weight(), 4);
        assert_eq!(source.causal_batches(4096, 11).unwrap().len(), 2);
        assert!(source.causal_batch(4096, 11, 9).is_err());
        let bundle = CorpusBundle::new(&manifest).unwrap();
        let remote = tempfile::tempdir().unwrap();
        for (source, relative) in &bundle.files {
            let destination = remote.path().join(relative);
            std::fs::create_dir_all(destination.parent().unwrap()).unwrap();
            std::fs::copy(source, destination).unwrap();
        }
        let relocated = remote.path().join("data.bauxptn");
        std::fs::write(&relocated, bundle.manifest.unwrap()).unwrap();
        let actual = CorpusSet::load_eval(&relocated).unwrap();
        assert_eq!(actual.total_weight(), source.total_weight());
        for (left, right) in source
            .causal_batches(4096, 11)
            .unwrap()
            .iter()
            .zip(actual.causal_batches(4096, 11).unwrap())
        {
            assert_eq!(left.source, right.source);
            assert_eq!(left.weight, right.weight);
            assert_eq!(left.batch, right.batch);
        }
        // The relocated manifest still enforces the original hashes.
        std::fs::write(remote.path().join("part-0/tokenizer.json"), b"changed").unwrap();
        assert!(CorpusSet::load_eval(&relocated).is_err());
    }

    #[test]
    fn corpus_bundle() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("source.bauxptn");
        corpus(&path, 101);
        std::fs::write(directory.path().join("manifest.json"), b"{}").unwrap();
        let bundle = CorpusBundle::new(&path).unwrap();
        assert!(bundle.manifest.is_none());
        assert_eq!(bundle.files.len(), 2);
        assert_eq!(bundle.files[0], (path, PathBuf::from("data.bauxptn")));
        assert_eq!(bundle.files[1].1, PathBuf::from("manifest.json"));
    }

    #[test]
    fn tokenizer_mismatch() {
        let directory = tempfile::tempdir().unwrap();
        let mut parts = Vec::new();
        for (name, token) in [("left", 101), ("right", 202)] {
            let root = directory.path().join(name);
            std::fs::create_dir(&root).unwrap();
            let path = root.join("train.bauxptn");
            let tokenizer = root.join("tokenizer.json");
            corpus(&path, token);
            std::fs::write(&tokenizer, name).unwrap();
            parts.push(baux::artifact::json::json!({
                "name":name,"weight":1,"path":path,
                "tokens":u64::from(258u32) * u64::from(CONTEXT),
                "sha256":digest(&path),"tokenizer-sha256":digest(&tokenizer),
            }));
        }
        let manifest = directory.path().join("validation.json");
        std::fs::write(
            &manifest,
            baux::artifact::json::pretty_string(&baux::artifact::json::json!({
                "format":MIX_FORMAT,"aggregation":"weighted-mean","parts":parts,
            }))
            .unwrap(),
        )
        .unwrap();
        assert!(CorpusSet::load(&manifest).is_err());
    }
}
