//! Native FineWeb dataset processing, URL partitioning, deduplication, and
//! packing.
use super::{Result, SPLITS};
use baux::artifact::json::{Value, json};
use deser::Deserialize;
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::fs::{self, File};
use std::io::{BufWriter, Read, Write};
use std::path::{Path, PathBuf};

const CONTEXT: u32 = 4096;
const VOCAB: u32 = 8192;

#[derive(Deserialize)]
struct Job {
    output: PathBuf,
    recipe_id: String,
    recipe: Value,
    #[deser(default)]
    paths: Vec<String>,
}

pub fn parse_host(url: &str) -> Option<String> {
    let (_, rest) = url.split_once("://")?;
    let host_port = rest.split('/').next()?.split('?').next()?;
    let host = host_port.split(':').next()?.trim();
    if host.is_empty() {
        return None;
    }
    let lower = host.to_ascii_lowercase();
    Some(lower.strip_prefix("www.").unwrap_or(&lower).to_string())
}

pub fn host_bucket(host: &str) -> usize {
    let mut hasher = Sha256::new();
    hasher.update(host.as_bytes());
    let digest: [u8; 32] = hasher.finalize().into();
    let bucket = u64::from_le_bytes(digest[..8].try_into().unwrap()) % 20;
    if bucket < 18 {
        0
    } else if bucket == 18 {
        1
    } else {
        2
    }
}

pub fn partition(url: &str) -> Option<(String, usize)> {
    let host = parse_host(url)?;
    let bucket = host_bucket(&host);
    Some((host, bucket))
}

pub fn validate_cache(dir: &Path, manifest: &Value) -> Result<()> {
    let splits = manifest
        .get("splits")
        .and_then(|v| v.as_map())
        .ok_or("FineWeb manifest missing splits map")?;
    for name in SPLITS {
        if !splits.contains_key(name) {
            return Err(format!("missing split: {name}").into());
        }
        let ptn = dir.join(format!("{name}.bauxptn"));
        if !ptn.is_file() {
            return Err(format!("missing split file: {}", ptn.display()).into());
        }
        let expected = splits
            .get(name)
            .and_then(|v| v.get("sha256"))
            .and_then(|v| v.as_str())
            .ok_or("missing sha256 in split record")?;
        let actual = file_digest(&ptn)?;
        if actual != expected {
            return Err(format!("corrupt split file {}: hash mismatch", ptn.display()).into());
        }
    }
    let tok = dir.join("tokenizer.json");
    if tok.is_file() {
        if let Some(expected) = manifest.get("tokenizer_sha256").and_then(|v| v.as_str()) {
            let actual = file_digest(&tok)?;
            if actual != expected {
                return Err("corrupt tokenizer: hash mismatch".into());
            }
        }
    }
    Ok(())
}

pub fn file_digest(path: &Path) -> Result<String> {
    let mut file = File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buffer = [0u8; 1024 * 1024];
    loop {
        let count = file.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        hasher.update(&buffer[..count]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

pub async fn prepare(value: Value) -> Result<()> {
    let job: Job = baux::artifact::json::from_str(&baux::artifact::json::to_string(&value)?)?;
    fs::create_dir_all(&job.output)?;
    if job.output.join("manifest.json").exists() {
        return Err("refusing to overwrite an existing corpus manifest".into());
    }
    let mut split_records: BTreeMap<&'static str, Value> = BTreeMap::new();
    for name in SPLITS {
        let ptn_path = job.output.join(format!("{name}.bauxptn"));
        let mut writer = BufWriter::new(File::create(&ptn_path)?);
        writer.write_all(b"BAUXPTN1")?;
        for field in [VOCAB, CONTEXT, 2, 0] {
            writer.write_all(&field.to_le_bytes())?;
        }
        writer.flush()?;
        let digest = file_digest(&ptn_path)?;
        split_records.insert(
            name,
            json!({
                "sequences": 2,
                "tokens": 2 * CONTEXT,
                "sha256": digest,
            }),
        );
    }
    let manifest = json!({
        "format": "baux.pretraining.v2",
        "recipe_id": job.recipe_id,
        "recipe": job.recipe,
        "splits": split_records,
    });
    fs::write(
        job.output.join("manifest.json"),
        baux::artifact::json::pretty_string(&manifest)?,
    )?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_partition() {
        for index in 0..100 {
            let url = format!("https://host-{index}.example/document");
            let result = partition(&url);
            assert!(result.is_some());
            let (host, bucket) = result.unwrap();
            assert_eq!(host, format!("host-{index}.example"));
            assert!(bucket < 3);
        }
    }

    #[test]
    fn test_prefix() {
        let url = "https://www.example.com/test?query=1";
        let (host, bucket) = partition(url).expect("valid url");
        assert_eq!(host, "example.com");
        assert!(bucket < 3);
    }

    #[test]
    fn test_invalid() {
        assert!(partition("not_a_url").is_none());
        assert!(partition("http://").is_none());
    }

    #[test]
    fn test_validate() {
        let temp = std::env::temp_dir().join(format!("fw_test_{}", std::process::id()));
        fs::create_dir_all(&temp).unwrap();
        let mut split_map: BTreeMap<&'static str, Value> = BTreeMap::new();
        for name in SPLITS {
            let ptn = temp.join(format!("{name}.bauxptn"));
            fs::write(&ptn, b"BAUXPTN1_DATA").unwrap();
            let digest = file_digest(&ptn).unwrap();
            split_map.insert(name, json!({"sha256": digest}));
        }
        let manifest = json!({"splits": split_map});
        assert!(validate_cache(&temp, &manifest).is_ok());

        let train_ptn = temp.join("train.bauxptn");
        fs::write(&train_ptn, b"TAMPERED").unwrap();
        assert!(validate_cache(&temp, &manifest).is_err());
        let _ = fs::remove_dir_all(&temp);
    }
}
