//! New records use CBOR; previously saved JSON remains a read format.

use super::*;
use std::collections::BTreeMap;

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
