//! Exact GPU compilation inputs and available outputs, never proof
//! certificates.
//!
//! Call after successful compilation/loading, outside the kernel launch loop.
//! Metal source or PTX capture does not imply possession of executable device
//! code.

use baux::artifact::{cbor, runs::digest};
use deser::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};

#[path = "gpu/inspect.rs"]
pub mod inspect;

const MAX_TEXT: usize = 16 * 1024;
const MAX_LIST: usize = 256;
const MAX_META: u64 = 1024 * 1024;
const MAX_BLOB: u64 = 256 * 1024 * 1024;
static NEXT: AtomicU64 = AtomicU64::new(0);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[deser(rename_all = "kebab-case")]
pub enum DeviceKind {
    Metal,
    Cuda,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[deser(rename_all = "kebab-case")]
/// Successfully observed compilation/link/loading stage, not a kernel
/// execution.
pub enum Stage {
    Compile,
    Link,
    Load,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[deser(rename_all = "kebab-case")]
pub enum Kind {
    MetalSource,
    Ptx,
    NvvmIr,
    LtoIr,
    Cubin,
    /// `native-image` is accepted only when reading legacy evidence records.
    #[deser(alias = "native-image")]
    DeviceImage,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[deser(deny_unknown_fields)]
pub struct Constant {
    pub index: u64,
    pub kind: String,
    pub bytes: Vec<u8>,
}

/// Ordered options/constants preserve the exact caller-supplied recipe.
/// `compiler` and `target` must describe what is known; use an explicit
/// unreported marker when the platform does not expose an identity.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[deser(deny_unknown_fields)]
pub struct Recipe {
    /// `backend` is accepted only when reading legacy evidence records.
    #[deser(alias = "backend")]
    pub device_kind: DeviceKind,
    pub device: String,
    pub target: String,
    pub compiler: String,
    pub stage: Stage,
    pub entry: String,
    pub input_kind: Kind,
    pub output_kind: Option<Kind>,
    pub options: Vec<String>,
    pub constants: Vec<Constant>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[deser(deny_unknown_fields)]
pub struct Blob {
    pub sha256: String,
    pub bytes: u64,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[deser(deny_unknown_fields)]
pub struct Record {
    pub schema: u32,
    pub recipe: Recipe,
    pub input: Blob,
    pub output: Option<Blob>,
}

/// Content identities only; this type carries no correctness assertion.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Receipt {
    pub id: String,
    pub input: String,
    pub output: Option<String>,
}

fn text(value: &str) -> Result<(), String> {
    if value.is_empty() || value.len() > MAX_TEXT {
        return Err("GPU artifact metadata text is empty or too large".into());
    }
    Ok(())
}

impl Recipe {
    fn validate(&self) -> Result<(), String> {
        for value in [&self.device, &self.target, &self.compiler, &self.entry] {
            text(value)?;
        }
        if self.options.len() > MAX_LIST || self.constants.len() > MAX_LIST {
            return Err("GPU artifact metadata has too many options or constants".into());
        }
        for option in &self.options {
            text(option)?;
        }
        for constant in &self.constants {
            text(&constant.kind)?;
            if constant.bytes.is_empty() || constant.bytes.len() > 4096 {
                return Err("GPU function constant is empty or too large".into());
            }
        }
        for kind in [Some(self.input_kind), self.output_kind]
            .into_iter()
            .flatten()
        {
            let valid = match self.device_kind {
                DeviceKind::Metal => matches!(kind, Kind::MetalSource | Kind::DeviceImage),
                DeviceKind::Cuda => !matches!(kind, Kind::MetalSource),
            };
            if !valid {
                return Err("GPU artifact kind does not belong to its device kind".into());
            }
        }
        Ok(())
    }
}

fn blob(bytes: &[u8]) -> Result<Blob, String> {
    if bytes.is_empty() || bytes.len() as u64 > MAX_BLOB {
        return Err("GPU code artifact is empty or exceeds 256 MiB".into());
    }
    Ok(Blob {
        sha256: digest(bytes),
        bytes: bytes.len() as u64,
    })
}

fn valid_id(id: &str) -> bool {
    id.len() == 64
        && id
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn directory(path: &Path) -> Result<(), String> {
    fs::create_dir_all(path).map_err(|error| format!("create {}: {error}", path.display()))?;
    let metadata = fs::symlink_metadata(path)
        .map_err(|error| format!("inspect {}: {error}", path.display()))?;
    if !metadata.is_dir() {
        return Err(format!(
            "GPU artifact directory must be a real directory: {}",
            path.display()
        ));
    }
    Ok(())
}

fn verify(path: &Path, expected: &Blob) -> Result<(), String> {
    if !valid_id(&expected.sha256) || expected.bytes == 0 || expected.bytes > MAX_BLOB {
        return Err("GPU artifact has an invalid content identity".into());
    }
    let metadata = fs::symlink_metadata(path)
        .map_err(|error| format!("inspect {}: {error}", path.display()))?;
    if !metadata.is_file() || metadata.len() != expected.bytes {
        return Err(format!(
            "GPU artifact was altered or is not a regular file: {}",
            path.display()
        ));
    }
    let mut file = File::open(path).map_err(|error| format!("open {}: {error}", path.display()))?;
    let mut hash = Sha256::new();
    let mut buffer = [0u8; 65536];
    let mut total = 0u64;
    loop {
        let read = file.read(&mut buffer).map_err(|error| error.to_string())?;
        if read == 0 {
            break;
        }
        total += read as u64;
        if total > expected.bytes {
            return Err(format!(
                "GPU artifact grew while reading: {}",
                path.display()
            ));
        }
        hash.update(&buffer[..read]);
    }
    if total != expected.bytes || format!("{:x}", hash.finalize()) != expected.sha256 {
        return Err(format!(
            "GPU artifact content digest mismatch: {}",
            path.display()
        ));
    }
    Ok(())
}

/// Publish with a hard link so a concurrent writer can never replace evidence.
/// Filesystems without same-directory hard links return an error.
fn publish(path: &Path, bytes: &[u8], expected: &Blob) -> Result<(), String> {
    match fs::symlink_metadata(path) {
        Ok(_) => return verify(path, expected),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => return Err(format!("inspect {}: {error}", path.display())),
    }
    let parent = path.parent().ok_or("GPU artifact path has no parent")?;
    let temporary = parent.join(format!(
        ".capture-{}-{}.tmp",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    let mut file = OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(&temporary)
        .map_err(|error| format!("create {}: {error}", temporary.display()))?;
    let result = (|| {
        file.write_all(bytes)
            .map_err(|error| format!("write {}: {error}", temporary.display()))?;
        match fs::hard_link(&temporary, path) {
            Ok(()) => Ok(()),
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
                verify(path, expected)
            }
            Err(error) => Err(format!("publish {}: {error}", path.display())),
        }
    })();
    drop(file);
    let cleanup = fs::remove_file(&temporary);
    result?;
    cleanup.map_err(|error| format!("remove {}: {error}", temporary.display()))?;
    Ok(())
}

/// Capture code bytes only. `root` is an evidence directory, not a repository.
/// Executable device output is recorded only when the caller actually supplies
/// its bytes.
pub fn record(
    root: &Path,
    recipe: &Recipe,
    input: &[u8],
    output: Option<&[u8]>,
) -> Result<Receipt, String> {
    recipe.validate()?;
    if recipe.output_kind.is_some() != output.is_some() {
        return Err("GPU output kind and captured output bytes disagree".into());
    }
    let input_blob = blob(input)?;
    let output_blob = output.map(blob).transpose()?;
    let record = Record {
        schema: 1,
        recipe: recipe.clone(),
        input: input_blob,
        output: output_blob,
    };
    let encoded = cbor::to_vec(&record).map_err(|error| error.to_string())?;
    if encoded.len() as u64 > MAX_META {
        return Err("GPU artifact metadata exceeds 1 MiB".into());
    }
    let identity = blob(&encoded)?;
    directory(root)?;
    let blobs = root.join("blobs");
    let records = root.join("records");
    directory(&blobs)?;
    directory(&records)?;
    publish(
        &blobs.join(format!("{}.bin", record.input.sha256)),
        input,
        &record.input,
    )?;
    if let (Some(bytes), Some(blob)) = (output, record.output.as_ref()) {
        publish(&blobs.join(format!("{}.bin", blob.sha256)), bytes, blob)?;
    }
    publish(
        &records.join(format!("{}.cbor", identity.sha256)),
        &encoded,
        &identity,
    )?;
    Ok(Receipt {
        id: identity.sha256,
        input: record.input.sha256,
        output: record.output.map(|blob| blob.sha256),
    })
}

/// Disabled capture does not validate metadata or touch the filesystem.
pub fn record_env(
    recipe: &Recipe,
    input: &[u8],
    output: Option<&[u8]>,
) -> Result<Option<Receipt>, String> {
    let Some(root) = std::env::var_os("BAUX_GPU_EVIDENCE") else {
        return Ok(None);
    };
    if root.is_empty() {
        return Err("BAUX_GPU_EVIDENCE must name a directory".into());
    }
    record(Path::new(&root), recipe, input, output).map(Some)
}

/// Read a record and verify its content identity and every referenced raw blob.
pub fn load(root: &Path, id: &str) -> Result<Record, String> {
    let record = read_record(root, id)?;
    verify_blobs(root, &record, &mut BTreeMap::new())?;
    Ok(record)
}

fn read_record(root: &Path, id: &str) -> Result<Record, String> {
    if !valid_id(id) {
        return Err("invalid GPU artifact record identity".into());
    }
    let path = root.join("records").join(format!("{id}.cbor"));
    let metadata = fs::symlink_metadata(&path)
        .map_err(|error| format!("inspect {}: {error}", path.display()))?;
    if !metadata.is_file() || metadata.len() > MAX_META {
        return Err("GPU artifact record is not a bounded regular file".into());
    }
    let mut bytes = Vec::new();
    File::open(&path)
        .map_err(|error| error.to_string())?
        .take(MAX_META + 1)
        .read_to_end(&mut bytes)
        .map_err(|error| error.to_string())?;
    if bytes.len() as u64 > MAX_META || digest(&bytes) != id {
        return Err("GPU artifact record digest mismatch".into());
    }
    let record: Record = cbor::from_slice(&bytes).map_err(|error| error.to_string())?;
    if record.schema != 1 || record.recipe.output_kind.is_some() != record.output.is_some() {
        return Err("GPU artifact record schema or output is invalid".into());
    }
    record.recipe.validate()?;
    Ok(record)
}

fn verify_blobs(
    root: &Path,
    record: &Record,
    checked: &mut BTreeMap<String, u64>,
) -> Result<(), String> {
    for blob in [Some(&record.input), record.output.as_ref()]
        .into_iter()
        .flatten()
    {
        if !valid_id(&blob.sha256) {
            return Err("invalid GPU blob identity".into());
        }
        if let Some(bytes) = checked.get(&blob.sha256) {
            if *bytes != blob.bytes {
                return Err("GPU records disagree about a shared blob size".into());
            }
            continue;
        }
        verify(
            &root.join("blobs").join(format!("{}.bin", blob.sha256)),
            blob,
        )?;
        checked.insert(blob.sha256.clone(), blob.bytes);
    }
    Ok(())
}

/// List immutable compilation records, verifying all referenced bytes.
/// Missing evidence returns an empty list; a malformed published record fails.
/// Shared blobs are streamed once per listing. This is not an atomic snapshot
/// against concurrent external tampering; callers must retain immutable
/// storage.
pub fn records(root: &Path) -> Result<Vec<(Receipt, Record)>, String> {
    let directory = root.join("records");
    match fs::symlink_metadata(&directory) {
        Ok(metadata) if metadata.is_dir() => {}
        Ok(_) => return Err("GPU records path is not a real directory".into()),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(error) => return Err(format!("inspect {}: {error}", directory.display())),
    }
    let mut ids = Vec::new();
    for entry in fs::read_dir(&directory).map_err(|error| error.to_string())? {
        let entry = entry.map_err(|error| error.to_string())?;
        let name = entry.file_name();
        let name = name.to_str().ok_or("GPU record filename is not UTF-8")?;
        if name.starts_with(".capture-") && name.ends_with(".tmp") {
            continue;
        }
        let id = name
            .strip_suffix(".cbor")
            .filter(|id| valid_id(id))
            .ok_or("GPU record directory contains an invalid filename")?;
        ids.push(id.to_owned());
        if ids.len() > 65_536 {
            return Err("GPU evidence has too many records".into());
        }
    }
    selected(root, &ids)
}

/// Verify only named records, ignoring later unrelated compilation captures.
/// Shared blobs are verified once per call; duplicate IDs are rejected.
pub fn selected(root: &Path, ids: &[String]) -> Result<Vec<(Receipt, Record)>, String> {
    if ids.len() > 65_536 || ids.iter().any(|id| !valid_id(id)) {
        return Err("GPU selection has invalid or too many record identities".into());
    }
    let mut ids = ids.to_vec();
    ids.sort();
    if ids.windows(2).any(|pair| pair[0] == pair[1]) {
        return Err("GPU selection contains duplicate record identities".into());
    }
    let mut checked = BTreeMap::new();
    ids.into_iter()
        .map(|id| {
            let record = read_record(root, &id)?;
            verify_blobs(root, &record, &mut checked)?;
            let receipt = Receipt {
                id,
                input: record.input.sha256.clone(),
                output: record.output.as_ref().map(|blob| blob.sha256.clone()),
            };
            Ok((receipt, record))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn recipe() -> Recipe {
        Recipe {
            device_kind: DeviceKind::Cuda,
            device: "test GPU".into(),
            target: "sm_test".into(),
            compiler: "test compiler".into(),
            stage: Stage::Compile,
            entry: "kernel".into(),
            input_kind: Kind::Ptx,
            output_kind: Some(Kind::Cubin),
            options: vec!["-O2".into(), "--ftz=false".into()],
            constants: vec![Constant {
                index: 0,
                kind: "bool".into(),
                bytes: vec![1],
            }],
        }
    }

    #[test]
    fn roundtrip() {
        let root = tempfile::tempdir().unwrap();
        let recipe = recipe();
        let receipt = record(root.path(), &recipe, b"ptx", Some(b"device image")).unwrap();
        let actual = load(root.path(), &receipt.id).unwrap();
        assert_eq!(actual.recipe, recipe);
        assert_eq!(actual.output.unwrap().sha256, receipt.output.unwrap());
        assert_eq!(
            fs::read(
                root.path()
                    .join("blobs")
                    .join(format!("{}.bin", receipt.input))
            )
            .unwrap(),
            b"ptx"
        );
        assert_eq!(records(root.path()).unwrap().len(), 1);
    }

    #[test]
    fn vocabulary_migration() {
        let canonical = baux::artifact::json::to_string(&recipe()).unwrap();
        assert!(canonical.contains("\"device_kind\""));
        assert!(!canonical.contains("\"backend\""));
        let legacy: Recipe =
            baux::artifact::json::from_str(&canonical.replace("\"device_kind\"", "\"backend\""))
                .unwrap();
        assert_eq!(legacy.device_kind, DeviceKind::Cuda);

        assert_eq!(
            baux::artifact::json::from_str::<Kind>("\"native-image\"").unwrap(),
            Kind::DeviceImage
        );
        assert_eq!(
            baux::artifact::json::to_string(&Kind::DeviceImage).unwrap(),
            "\"device-image\""
        );
    }

    #[test]
    fn identities() {
        let root = tempfile::tempdir().unwrap();
        let mut recipe = recipe();
        let first = record(root.path(), &recipe, b"ptx", Some(b"device image")).unwrap();
        let second = record(
            root.path(),
            &recipe,
            b"ptx",
            Some(b"different device image"),
        )
        .unwrap();
        assert_ne!(first.id, second.id);
        assert_eq!(first.input, second.input);
        recipe.options.reverse();
        assert_ne!(
            first.id,
            record(root.path(), &recipe, b"ptx", Some(b"device image"))
                .unwrap()
                .id
        );
        recipe.output_kind = None;
        let source = record(root.path(), &recipe, b"ptx", None).unwrap();
        assert!(load(root.path(), &source.id).unwrap().output.is_none());
        assert!(record(root.path(), &recipe, b"ptx", Some(b"device image")).is_err());
    }

    #[test]
    fn tampering() {
        let root = tempfile::tempdir().unwrap();
        let recipe = recipe();
        let receipt = record(root.path(), &recipe, b"ptx", Some(b"device image")).unwrap();
        let blob = root
            .path()
            .join("blobs")
            .join(format!("{}.bin", receipt.input));
        fs::write(&blob, b"bad").unwrap();
        assert!(load(root.path(), &receipt.id).is_err());
        assert!(record(root.path(), &recipe, b"ptx", Some(b"device image")).is_err());
        assert_eq!(fs::read(blob).unwrap(), b"bad");
    }

    #[test]
    fn concurrent() {
        let root = tempfile::tempdir().unwrap();
        let receipts = std::thread::scope(|scope| {
            let threads = (0..8)
                .map(|_| {
                    scope.spawn(|| {
                        record(root.path(), &recipe(), b"ptx", Some(b"device image")).unwrap()
                    })
                })
                .collect::<Vec<_>>();
            threads
                .into_iter()
                .map(|thread| thread.join().unwrap())
                .collect::<Vec<_>>()
        });
        assert!(receipts.iter().all(|receipt| receipt == &receipts[0]));
        load(root.path(), &receipts[0].id).unwrap();
        assert_eq!(
            fs::read_dir(root.path().join("records")).unwrap().count(),
            1
        );
        assert_eq!(fs::read_dir(root.path().join("blobs")).unwrap().count(), 2);
    }

    #[test]
    fn bounded() {
        let root = tempfile::tempdir().unwrap();
        let mut recipe = recipe();
        recipe.options = vec!["x".into(); MAX_LIST + 1];
        assert!(record(root.path(), &recipe, b"ptx", Some(b"device image")).is_err());
        assert!(!root.path().join("records").exists());
        assert!(load(root.path(), "../escape").is_err());
    }

    #[test]
    fn record_tampering() {
        let root = tempfile::tempdir().unwrap();
        let recipe = recipe();
        let receipt = record(root.path(), &recipe, b"ptx", Some(b"device image")).unwrap();
        let path = root
            .path()
            .join("records")
            .join(format!("{}.cbor", receipt.id));
        fs::write(&path, b"bad record").unwrap();
        assert!(load(root.path(), &receipt.id).is_err());
        assert!(record(root.path(), &recipe, b"ptx", Some(b"device image")).is_err());
        assert_eq!(fs::read(path).unwrap(), b"bad record");
        assert!(records(root.path()).is_err());
    }

    #[test]
    fn shared_blobs() {
        let root = tempfile::tempdir().unwrap();
        let mut recipe = recipe();
        let first = record(root.path(), &recipe, b"ptx", Some(b"device image")).unwrap();
        recipe.entry = "another".into();
        record(root.path(), &recipe, b"ptx", Some(b"device image")).unwrap();
        assert_eq!(records(root.path()).unwrap().len(), 2);
        let path = root
            .path()
            .join("blobs")
            .join(format!("{}.bin", first.input));
        fs::write(path, b"bad").unwrap();
        assert!(records(root.path()).is_err());
    }

    #[test]
    fn selection_scope() {
        let root = tempfile::tempdir().unwrap();
        let mut recipe = recipe();
        let first = record(root.path(), &recipe, b"ptx", Some(b"device image")).unwrap();
        recipe.entry = "another".into();
        let second = record(root.path(), &recipe, b"other", Some(b"other device image")).unwrap();
        fs::write(
            root.path()
                .join("blobs")
                .join(format!("{}.bin", second.input)),
            b"bad",
        )
        .unwrap();
        assert_eq!(selected(root.path(), &[first.id.clone()]).unwrap().len(), 1);
        assert!(selected(root.path(), &[first.id.clone(), first.id]).is_err());
        assert!(selected(root.path(), &["../escape".into()]).is_err());
        assert!(records(root.path()).is_err());
    }

    #[cfg(unix)]
    #[test]
    fn symlinks() {
        let root = tempfile::tempdir().unwrap();
        let recipe = recipe();
        let receipt = record(root.path(), &recipe, b"ptx", Some(b"device image")).unwrap();
        let path = root
            .path()
            .join("blobs")
            .join(format!("{}.bin", receipt.input));
        fs::remove_file(&path).unwrap();
        let outside = root.path().join("outside");
        fs::write(&outside, b"ptx").unwrap();
        std::os::unix::fs::symlink(&outside, path).unwrap();
        assert!(record(root.path(), &recipe, b"ptx", Some(b"device image")).is_err());
        assert!(load(root.path(), &receipt.id).is_err());
        assert_eq!(fs::read(outside).unwrap(), b"ptx");
    }

    #[test]
    fn env_disabled() {
        if std::env::var_os("BAUX_GPU_CHILD").is_some() {
            let mut invalid = recipe();
            invalid.device.clear();
            assert!(record_env(&invalid, &[], None).unwrap().is_none());
            return;
        }
        let status = std::process::Command::new(std::env::current_exe().unwrap())
            .args(["--exact", "gpu::tests::env_disabled"])
            .env_remove("BAUX_GPU_EVIDENCE")
            .env("BAUX_GPU_CHILD", "1")
            .status()
            .unwrap();
        assert!(status.success());
    }
}
