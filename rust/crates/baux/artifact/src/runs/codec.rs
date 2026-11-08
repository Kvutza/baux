//! New records use CBOR; previously saved JSON remains a read format.

use baux::artifact::json::{self, Value};
use deser::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fs::{self, File};
use std::io::Write;
use std::path::{Path, PathBuf};

/// A versioned execution record; IDs are internal, time and task are
/// searchable.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Record {
    pub schema: u32,
    pub id: String,
    pub started: i64,
    pub task: String,
    pub task_store: String,
    pub workspace: String,
    pub output: String,
    pub commit: String,
    pub change: String,
    pub runner_sha256: String,
    pub worker_sha256: String,
    pub config_sha256: String,
    pub data_sha256: String,
    pub model: String,
    pub optimizer: String,
    pub recipe: String,
    /// Same resolved workload, evaluation and seeds; optimizer is excluded.
    pub protocol: String,
    pub status: String,
    pub ended: Option<i64>,
    pub report_status: String,
    pub report_error: Option<String>,
    pub error: Option<String>,
    pub measurements: Value,
}

pub fn field<'a>(value: &'a Value, key: &str) -> &'a str {
    value.get(key).and_then(|v| v.as_str()).unwrap_or("")
}

pub fn digest(bytes: &[u8]) -> String {
    baux::artifact::blake3::hash_hex(bytes)
}

/// Locate the common repository owner without depending on Git or a Beads ID.
pub fn directory(root: &Path) -> Result<PathBuf, String> {
    let pointer = root.join(".jj/repo");
    let repo = if pointer.is_file() {
        root.join(".jj").join(
            fs::read_to_string(&pointer)
                .map_err(|e| e.to_string())?
                .trim(),
        )
    } else {
        pointer
    };
    let repo = repo
        .canonicalize()
        .map_err(|e| format!("locate shared JJ repository: {e}"))?;
    let owner = repo
        .parent()
        .and_then(Path::parent)
        .ok_or("invalid JJ repository path")?;
    Ok(owner.join(".cache/baux/evidence"))
}

pub(super) fn validate(record: &Record) -> Result<(), String> {
    if record.schema != 1
        || record.id.is_empty()
        || !record
            .id
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'-')
    {
        return Err("unsupported or invalid run manifest".into());
    }
    Ok(())
}

pub(crate) fn atomic(path: &Path, bytes: &[u8]) -> Result<(), String> {
    fs::create_dir_all(path.parent().ok_or("missing parent directory")?)
        .map_err(|e| e.to_string())?;
    let temporary = path.with_extension(format!("{}.tmp", std::process::id()));
    let mut file = File::create(&temporary).map_err(|e| e.to_string())?;
    file.write_all(bytes).map_err(|e| e.to_string())?;
    fs::rename(temporary, path).map_err(|e| e.to_string())
}

pub fn jj(root: &Path, args: &[&str]) -> Result<String, String> {
    let output = std::process::Command::new("jj")
        .current_dir(root)
        .args(args)
        .output()
        .map_err(|e| e.to_string())?;
    if !output.status.success() {
        return Err(format!(
            "JJ failed: {}",
            String::from_utf8_lossy(&output.stderr)
        ));
    }
    String::from_utf8(output.stdout).map_err(|e| e.to_string())
}

pub(super) fn encode(record: &Record) -> Result<Vec<u8>, String> {
    validate(record)?;
    baux::artifact::cbor::to_vec(record).map_err(|e| e.to_string())
}

pub(super) fn decode(bytes: &[u8]) -> Result<Record, String> {
    // A run is an object/map in both formats. JSON object starts cannot collide
    // with CBOR map headers. Never retry corrupt CBOR as another format.
    let first = bytes.iter().find(|byte| !byte.is_ascii_whitespace());
    let record = if first == Some(&b'{') {
        json::from_slice(bytes).map_err(|e| format!("JSON run record: {e}"))?
    } else {
        baux::artifact::cbor::from_slice(bytes).map_err(|e| format!("CBOR run record: {e}"))?
    };
    validate(&record)?;
    Ok(record)
}

/// Prefer the CBOR revision of a run when both formats exist. Do not use file
/// timestamps or enumeration order, and do not fall back to stale JSON on
/// error.
pub(super) fn manifests(directory: &Path) -> Result<Vec<Record>, String> {
    if !directory.exists() {
        return Ok(Vec::new());
    }
    let mut paths = BTreeMap::new();
    for entry in fs::read_dir(directory).map_err(|e| e.to_string())? {
        let path = entry.map_err(|e| e.to_string())?.path();
        let extension = path.extension().and_then(|s| s.to_str());
        if !matches!(extension, Some("json" | "cbor")) {
            continue;
        }
        let id = path
            .file_stem()
            .and_then(|s| s.to_str())
            .ok_or("invalid manifest filename")?
            .to_owned();
        if extension == Some("cbor") || !paths.contains_key(&id) {
            paths.insert(id, path);
        }
    }
    paths
        .into_iter()
        .map(|(id, path)| {
            let bytes = fs::read(&path).map_err(|e| format!("{}: {e}", path.display()))?;
            let record = decode(&bytes).map_err(|e| format!("{}: {e}", path.display()))?;
            if record.id != id {
                return Err(format!(
                    "{}: manifest ID does not match its filename",
                    path.display()
                ));
            }
            Ok(record)
        })
        .collect()
}

/// Calendar dates are local; RFC3339 inputs specify their own offset.
pub fn since(input: &str) -> Result<i64, String> {
    let now = jiff::Zoned::now();
    let date = match input {
        "today" => Some(now.date()),
        "yesterday" => Some(
            now.date()
                .checked_sub(jiff::Span::new().days(1))
                .map_err(|e| e.to_string())?,
        ),
        _ => input.parse::<jiff::civil::Date>().ok(),
    };
    let timestamp = if let Some(date) = date {
        date.at(0, 0, 0, 0)
            .to_zoned(now.time_zone().clone())
            .map_err(|e| e.to_string())?
            .timestamp()
    } else {
        input
            .parse::<jiff::Timestamp>()
            .map_err(|_| "use today, yesterday, YYYY-MM-DD or an RFC3339 timestamp".to_string())?
    };
    Ok(timestamp.as_millisecond())
}

pub fn local_time(millis: i64) -> Result<String, String> {
    Ok(jiff::Timestamp::from_millisecond(millis)
        .map_err(|e| e.to_string())?
        .to_zoned(jiff::tz::TimeZone::system())
        .strftime("%Y-%m-%d %H:%M:%S %Z")
        .to_string())
}
