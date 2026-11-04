//! Immutable packed token streams for long-context pretraining experiments.

use std::fs::File;
use std::path::{Path, PathBuf};

#[path = "pretrain/corpus.rs"]
mod corpus;
#[path = "pretrain/index.rs"]
mod index;
use index::{MAX_TOKENS, SpanIndex, permute_index};

const MAGIC: &[u8; 8] = b"BAUXPTN1";
const LEGACY_MAGIC: &[u8; 8] = b"BLCKPTN1";
const HEADER_BYTES: usize = 24;
pub const VOCAB: u32 = 8192;
pub const CONTEXT: u32 = 4096;

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

/// Relocation plan for one corpus or held-out mixture. Only paths change;
/// source names, weights, counts, hashes and tokenizer identities are retained.
/// Every destination is generated here, never taken from an input pathname.
pub struct CorpusBundle {
    pub files: Vec<(PathBuf, PathBuf)>,
    pub manifest: Option<Vec<u8>>,
}

#[cfg(unix)]
unsafe extern "C" {
    fn mmap(
        addr: *mut std::ffi::c_void,
        len: usize,
        prot: std::ffi::c_int,
        flags: std::ffi::c_int,
        fd: std::ffi::c_int,
        offset: i64,
    ) -> *mut std::ffi::c_void;
    fn munmap(addr: *mut std::ffi::c_void, len: usize) -> std::ffi::c_int;
}

#[derive(Debug)]
enum TokenStore {
    Owned(Vec<u32>),
    #[cfg(unix)]
    Mapped {
        ptr: *mut u8,
        byte_len: usize,
        elements: usize,
    },
}

#[cfg(unix)]
unsafe impl Send for TokenStore {}
#[cfg(unix)]
unsafe impl Sync for TokenStore {}

impl Drop for TokenStore {
    fn drop(&mut self) {
        #[cfg(unix)]
        if let Self::Mapped { ptr, byte_len, .. } = *self {
            unsafe {
                let _ = munmap(ptr.cast(), byte_len);
            }
        }
    }
}

impl std::ops::Deref for TokenStore {
    type Target = [u32];

    fn deref(&self) -> &[u32] {
        match self {
            Self::Owned(vec) => vec.as_slice(),
            #[cfg(unix)]
            Self::Mapped { ptr, elements, .. } => unsafe {
                std::slice::from_raw_parts(ptr.add(HEADER_BYTES).cast::<u32>(), *elements)
            },
        }
    }
}

#[derive(Debug)]
pub struct PretrainDataset {
    vocabulary: u32,
    sequences: u32,
    tokens: TokenStore,
    index: Option<SpanIndex>,
}

/// One fixed corpus condition for causal next-token likelihood.
///
/// The model receives `tokens`; `labels` are the immediately following corpus
/// tokens. Candidate parameters never alter either sequence.
///
/// # Alignment and Invariants
/// - **Patch alignment**: Sequence tokens and corpus origins are aligned to
///   64-token FBT patch boundaries (since `stride` divides or equals `CONTEXT =
///   4096 = 64 * 64`, and offsets are multiples of `stride`).
/// - **1-shifted target invariant**: `tokens.len() == labels.len() ==
///   loss_mask.len()`, where `labels[i] == source[offset + i + 1]` and
///   `tokens[i] == source[offset + i]`.
/// - **Loss mask**: Every token is unmasked (`loss_mask[i] == 1`).
/// - **Corpus bounds**: `offset + tokens.len() + 1 <= dataset.tokens.len()`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CausalBatch {
    pub tokens: Vec<u32>,
    pub labels: Vec<u32>,
    pub loss_mask: Vec<u8>,
    pub offset: usize,
}

fn read_tokens(
    path: &Path,
    file: &mut File,
    vocab: u32,
    elements: usize,
) -> Result<TokenStore, String> {
    use std::io::Read;
    if vocab > 65536 && cfg!(target_endian = "little") {
        #[cfg(unix)]
        if elements > 0 {
            use std::os::fd::AsRawFd;
            let byte_len = HEADER_BYTES + elements * 4;
            let ptr = unsafe { mmap(std::ptr::null_mut(), byte_len, 1, 2, file.as_raw_fd(), 0) };
            if ptr != usize::MAX as *mut std::ffi::c_void && !ptr.is_null() {
                return Ok(TokenStore::Mapped {
                    ptr: ptr.cast::<u8>(),
                    byte_len,
                    elements,
                });
            }
        }
        let mut tokens = vec![0u32; elements];
        let dest = unsafe {
            std::slice::from_raw_parts_mut(tokens.as_mut_ptr().cast::<u8>(), elements * 4)
        };
        file.read_exact(dest)
            .map_err(|error| format!("pretraining dataset {}: {error}", path.display()))?;
        return Ok(TokenStore::Owned(tokens));
    }
    let mut body = vec![0u8; elements * 2];
    file.read_exact(&mut body)
        .map_err(|error| format!("pretraining dataset {}: {error}", path.display()))?;
    Ok(TokenStore::Owned(
        body.chunks_exact(2)
            .map(|val| u32::from(u16::from_le_bytes([val[0], val[1]])))
            .collect(),
    ))
}

fn manifest_digest(path: &Path) -> Option<String> {
    let stem = path.file_stem()?.to_str()?;
    let manifest = path.parent()?.join("manifest.json");
    let doc: baux::artifact::json::Value =
        baux::artifact::json::from_reader(File::open(manifest).ok()?).ok()?;
    doc["splits"][stem]["sha256"].as_str().map(str::to_owned)
}

fn check_tokens(
    path: &Path,
    meta: &std::fs::Metadata,
    vocab: u32,
    tokens: &[u32],
) -> Result<Option<String>, String> {
    let mtime = meta
        .modified()
        .ok()
        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let key = (tokens.len() >= 1_048_576 && mtime > 0).then(|| {
        if let Some(sha256) = manifest_digest(path) {
            baux::artifact::runs::digest(
                format!("v1:sha256:{sha256}:{}:{vocab}", meta.len()).as_bytes(),
            )
        } else {
            baux::artifact::runs::digest(
                format!("v1:{}:{}:{mtime}:{vocab}", path.display(), meta.len()).as_bytes(),
            )
        }
    });
    let cache_dir = Path::new(".cache/baux/digests");
    let marker = key
        .as_ref()
        .map(|hash| cache_dir.join(format!("{hash}.tokens_ok")));
    if marker.as_ref().is_some_and(|file| file.is_file()) {
        return Ok(key);
    }
    if let Some((index, token)) = tokens
        .iter()
        .copied()
        .enumerate()
        .find(|(_, token)| *token >= vocab)
    {
        return Err(format!(
            "pretraining token {token} at offset {index} exceeds the vocabulary"
        ));
    }
    if let Some(marker) = marker {
        if std::fs::create_dir_all(cache_dir).is_ok() {
            let _ = std::fs::write(&marker, b"ok");
        }
    }
    Ok(key)
}

impl PretrainDataset {
    pub fn load(path: &Path) -> Result<Self, String> {
        let (mut dataset, key) = Self::open_raw(path)?;
        dataset.index = SpanIndex::load(path, &dataset.tokens, key.as_deref())?;
        Ok(dataset)
    }

    pub fn load_raw(path: &Path) -> Result<Self, String> {
        Ok(Self::open_raw(path)?.0)
    }

    fn open_raw(path: &Path) -> Result<(Self, Option<String>), String> {
        use std::io::Read;
        let mut file = File::open(path)
            .map_err(|error| format!("pretraining dataset {}: {error}", path.display()))?;
        let meta = file
            .metadata()
            .map_err(|error| format!("pretraining dataset {}: {error}", path.display()))?;
        let file_len = usize::try_from(meta.len())
            .map_err(|_| "pretraining dataset size does not fit this machine")?;
        let mut bytes = [0u8; HEADER_BYTES];
        if file_len < HEADER_BYTES || file.read_exact(&mut bytes).is_err() || !has_magic(&bytes) {
            return Err("pretraining dataset has an invalid BAUXPTN1 header".into());
        }
        let field = |offset| {
            u32::from_le_bytes(
                bytes[offset..offset + 4]
                    .try_into()
                    .expect("four-byte field"),
            )
        };
        let vocab = field(8);
        let context = field(12);
        let sequences = field(16);
        let reserved = field(20);
        if vocab == 0 || context != CONTEXT || sequences < 2 || sequences % 2 != 0 {
            return Err(format!(
                "pretraining dataset requires nonzero vocab, context={CONTEXT}, and a positive \
                 even sequence count; found vocab={vocab}, context={context}, \
                 sequences={sequences}"
            ));
        }
        if reserved != 0 {
            return Err("pretraining dataset reserved header field must be zero".into());
        }
        let elements = usize::try_from(u64::from(sequences) * u64::from(context))
            .map_err(|_| "pretraining dataset size does not fit this machine")?;
        let stride = if vocab <= 65536 { 2 } else { 4 };
        let expected = HEADER_BYTES
            .checked_add(
                elements
                    .checked_mul(stride)
                    .ok_or("pretraining token byte overflow")?,
            )
            .ok_or("pretraining dataset byte size overflow")?;
        if file_len != expected {
            return Err(format!(
                "pretraining dataset has {file_len} bytes, expected {expected}"
            ));
        }
        let tokens = read_tokens(path, &mut file, vocab, elements)?;
        let key = check_tokens(path, &meta, vocab, &tokens)?;
        Ok((
            Self {
                vocabulary: vocab,
                sequences,
                tokens,
                index: None,
            },
            key,
        ))
    }

    pub fn batches(&self) -> u32 {
        self.sequences / 2
    }

    pub fn sequences(&self) -> u32 {
        self.sequences
    }

    pub fn tokens(&self) -> usize {
        self.tokens.len()
    }

    pub fn vocabulary(&self) -> u32 {
        self.vocabulary
    }

    pub fn prefix(&self, count: usize) -> Result<&[u32], String> {
        self.range(0, count)
    }

    pub fn range(&self, start: usize, count: usize) -> Result<&[u32], String> {
        let end = start
            .checked_add(count)
            .ok_or("pretraining token range overflow")?;
        self.tokens.get(start..end).ok_or_else(|| {
            format!(
                "context workload needs token positions {start}..{end}, dataset contains {}",
                self.tokens.len()
            )
        })
    }

    /// Select a contiguous, non-wrapping corpus sequence and construct its
    /// shifted next-token targets. Origins remain aligned to the corpus'
    /// immutable 4096-token records (and 64-token FBT patch boundaries),
    /// but no runtime packing or repetition is performed.
    pub fn causal_batch(
        &self,
        sequence_tokens: usize,
        seed: u64,
        round: u32,
    ) -> Result<CausalBatch, String> {
        if sequence_tokens == 0 || sequence_tokens > MAX_TOKENS as usize {
            return Err(format!(
                "causal sequence length must be in 1..={MAX_TOKENS}"
            ));
        }
        let required = sequence_tokens
            .checked_add(1)
            .ok_or("causal sequence length overflow")?;
        if required > self.tokens.len() {
            return Err(format!(
                "causal sequence needs {required} contiguous corpus tokens, dataset contains {}",
                self.tokens.len()
            ));
        }
        let stride = sequence_tokens.min(CONTEXT as usize);
        let origins = ((self.tokens.len() - required) / stride + 1) as u64;
        let offset = permute_index(origins, seed, round) * stride;
        let source = &self.tokens[offset..offset + required];
        Ok(CausalBatch {
            tokens: source[..sequence_tokens].to_vec(),
            labels: source[1..].to_vec(),
            loss_mask: vec![1; sequence_tokens],
            offset,
        })
    }

    /// Return one contiguous corpus window on the circular token stream.
    /// The requested window may cross the physical end once, but it may not
    /// repeat the corpus. This lets long-context rounds cover every possible
    /// sequence-aligned origin without manufacturing repeated-token proxies.
    pub fn circular(&self, start: usize, count: usize) -> Result<Vec<u32>, String> {
        if count > self.tokens.len() {
            return Err(format!(
                "context workload needs {count} source tokens, dataset contains {}",
                self.tokens.len()
            ));
        }
        if self.tokens.is_empty() {
            return Err("pretraining dataset is empty".into());
        }
        let start = start % self.tokens.len();
        let first = count.min(self.tokens.len() - start);
        let mut window = Vec::with_capacity(count);
        window.extend_from_slice(&self.tokens[start..start + first]);
        window.extend_from_slice(&self.tokens[..count - first]);
        Ok(window)
    }

    pub fn sequence(&self, index: u32) -> Result<&[u32], String> {
        if index >= self.sequences {
            return Err(format!(
                "pretraining sequence {index} exceeds the {} available sequences",
                self.sequences
            ));
        }
        let start = index as usize * CONTEXT as usize;
        Ok(&self.tokens[start..start + CONTEXT as usize])
    }

    /// Sample a natural corpus continuation within the complete context budget.
    ///
    /// Trace corpora yield complete assistant actions. Source corpora begin at
    /// a file-content boundary and continue over the packed repository stream.
    pub fn natural_task(
        &self,
        prompt_cap: usize,
        output_cap: usize,
        seed: u64,
        round: u32,
    ) -> Result<Option<(Vec<u32>, Vec<u32>, usize)>, String> {
        if let Some(index) = &self.index {
            return index.task(&self.tokens, prompt_cap, output_cap, seed, round);
        }
        Ok(None)
    }

    pub fn window(
        &self,
        prompt_cap: usize,
        output_cap: usize,
        seed: u64,
        round: u32,
    ) -> Result<(Vec<u32>, Vec<u32>, usize), String> {
        if let Some(task) = self.natural_task(prompt_cap, output_cap, seed, round)? {
            return Ok(task);
        }
        let total = prompt_cap
            .checked_add(output_cap)
            .ok_or("pretraining corpus window overflow")?;
        if total > self.tokens() {
            return Err(format!(
                "one pretraining condition needs {total} source tokens, corpus contains {}",
                self.tokens()
            ));
        }
        let stride = total.next_multiple_of(64).min(CONTEXT as usize).max(64);
        let origins = ((self.tokens() - total) / stride + 1) as u64;
        let offset = permute_index(origins, seed ^ 0x7769_6e64_6f77_2d78, round) * stride;
        let source = &self.tokens[offset..offset + total];
        Ok((
            source[..prompt_cap].to_vec(),
            source[prompt_cap..].to_vec(),
            offset,
        ))
    }

    pub fn batch(&self, index: u32) -> Result<&[u32], String> {
        if index >= self.batches() {
            return Err(format!(
                "pretraining batch {index} exceeds the {} available batches",
                self.batches()
            ));
        }
        let elements = 2 * CONTEXT as usize;
        let start = index as usize * elements;
        Ok(&self.tokens[start..start + elements])
    }
}

fn has_magic(bytes: &[u8]) -> bool {
    bytes.starts_with(MAGIC) || bytes.starts_with(LEGACY_MAGIC)
}

/// Load one contiguous prefix from an BAUXPTN1 corpus for an accelerator run.
pub fn load_tokens(path: &Path, count: usize) -> Result<Vec<u32>, String> {
    Ok(PretrainDataset::load(path)?.prefix(count)?.to_vec())
}

#[cfg(test)]
mod tests {
    use super::index::{Span, SpanIndex};
    use super::{CONTEXT, HEADER_BYTES, MAGIC, PretrainDataset, VOCAB};

    fn document(sequences: u32) -> Vec<u8> {
        let elements = sequences as usize * CONTEXT as usize;
        let mut bytes = Vec::with_capacity(HEADER_BYTES + elements * 2);
        bytes.extend_from_slice(MAGIC);
        bytes.extend_from_slice(&VOCAB.to_le_bytes());
        bytes.extend_from_slice(&CONTEXT.to_le_bytes());
        bytes.extend_from_slice(&sequences.to_le_bytes());
        bytes.extend_from_slice(&0u32.to_le_bytes());
        for index in 0..elements {
            bytes.extend_from_slice(&((index as u16) % VOCAB as u16).to_le_bytes());
        }
        bytes
    }

    #[test]
    fn fixed_batches() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("pretrain.bin");
        std::fs::write(&path, document(4)).unwrap();
        let data = PretrainDataset::load(&path).unwrap();
        assert_eq!(data.batches(), 2);
        assert_eq!(data.sequences(), 4);
        assert_eq!(data.sequence(3).unwrap().len(), CONTEXT as usize);
        assert!(data.sequence(4).is_err());
        assert_eq!(data.batch(1).unwrap().len(), 2 * CONTEXT as usize);
        assert!(data.batch(2).is_err());
    }

    #[test]
    fn causal_batches() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("pretrain.bin");
        std::fs::write(&path, document(4)).unwrap();
        let data = PretrainDataset::load(&path).unwrap();
        let batch = data.causal_batch(4096, 17, 3).unwrap();
        assert_eq!(batch.tokens.len(), 4096);
        assert_eq!(batch.labels.len(), 4096);
        assert_eq!(batch.loss_mask, vec![1; 4096]);
        assert_eq!(batch.offset % CONTEXT as usize, 0);
        // Verify alignment to 64-token FBT patch boundary
        assert_eq!(batch.offset % 64, 0);
        assert_eq!(batch.tokens.len() % 64, 0);
        assert_eq!(&batch.tokens[1..], &batch.labels[..4095]);
        assert_eq!(batch.labels[4095], data.tokens[batch.offset + 4096]);
        assert_eq!(batch, data.causal_batch(4096, 17, 3).unwrap());

        // Verify sub-context lengths also align to 64-token FBT patch boundary
        let small_batch = data.causal_batch(128, 42, 1).unwrap();
        assert_eq!(small_batch.offset % 64, 0);
        assert_eq!(small_batch.tokens.len(), 128);
    }

    #[test]
    fn epoch_permutation() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("pretrain.bin");
        std::fs::write(&path, document(4)).unwrap();
        let data = PretrainDataset::load_raw(&path).unwrap();
        let mut offsets = (0..3)
            .map(|round| data.causal_batch(4096, 17, round).unwrap().offset)
            .collect::<Vec<_>>();
        offsets.sort_unstable();
        assert_eq!(offsets, vec![0, 4096, 8192]);
    }

    #[test]
    fn nonwrapping_batches() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("pretrain.bin");
        std::fs::write(&path, document(2)).unwrap();
        let data = PretrainDataset::load(&path).unwrap();
        assert!(data.causal_batch(data.tokens() + 1, 0, 0).is_err());
    }

    #[test]
    fn invalid_tokens() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("pretrain.bin");
        let mut bytes = document(2);
        bytes[HEADER_BYTES..HEADER_BYTES + 2].copy_from_slice(&(VOCAB as u16).to_le_bytes());
        std::fs::write(&path, bytes).unwrap();
        assert!(
            PretrainDataset::load(&path)
                .unwrap_err()
                .contains("exceeds the vocabulary")
        );
    }

    #[test]
    fn wide_batches() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("wide.bin");
        let vocab: u32 = 256_000;
        let sequences: u32 = 4;
        let elements = sequences as usize * CONTEXT as usize;
        let mut bytes = Vec::with_capacity(HEADER_BYTES + elements * 4);
        bytes.extend_from_slice(MAGIC);
        bytes.extend_from_slice(&vocab.to_le_bytes());
        bytes.extend_from_slice(&CONTEXT.to_le_bytes());
        bytes.extend_from_slice(&sequences.to_le_bytes());
        bytes.extend_from_slice(&0u32.to_le_bytes());
        for index in 0..elements {
            bytes.extend_from_slice(&((index as u32) % vocab).to_le_bytes());
        }
        std::fs::write(&path, bytes).unwrap();
        let data = PretrainDataset::load(&path).unwrap();
        assert_eq!(data.batches(), 2);
        assert_eq!(data.sequences(), 4);
        assert_eq!(data.sequence(3).unwrap().len(), CONTEXT as usize);
        assert_eq!(data.batch(1).unwrap().len(), 2 * CONTEXT as usize);
    }

    #[test]
    fn code_caps() {
        let tokens = (0..1024).collect::<Vec<_>>();
        let code = SpanIndex {
            episodes: Vec::new(),
            origins: Vec::new(),
            spans: vec![Span {
                group: 0,
                start: 128,
                end: 256,
            }],
            pad: 0,
            bounded: false,
        };
        let (prompt, output, _) = code.task(&tokens, 512, 256, 17, 0).unwrap().unwrap();
        assert_eq!(prompt.len(), 128);
        assert_eq!(output.len(), 256);
    }
}
