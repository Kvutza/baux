//! Saved experiment evidence and its rebuildable, shared run catalogue.
//!
//! Manifests are the durable record. Redb indexes those manifests; neither
//! the database nor a Beads note is the owner of training measurements.

use baux::artifact::json::{self, Value, json};
use deser::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

#[path = "runs/backfill.rs"]
mod backfill;
#[path = "runs/catalog.rs"]
mod catalog;
#[path = "runs/codec.rs"]
mod codec;
#[path = "runs/dashboard.rs"]
pub mod dashboard;
#[path = "runs/plot.rs"]
mod plot;
pub use catalog::{Query, history, local_time, reindex, since};
pub use dashboard::{serve, write_html};
pub use plot::compare;

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
    format!("{:x}", Sha256::digest(bytes))
}

pub fn file_digest(path: &Path) -> Result<String, String> {
    let meta = fs::metadata(path).map_err(|e| format!("stat {}: {e}", path.display()))?;
    let mtime = meta
        .modified()
        .ok()
        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let cache_key = digest(format!("v1:{}:{}:{mtime}", path.display(), meta.len()).as_bytes());
    let cache_dir = Path::new(".cache/baux/digests");
    let cache_path = cache_dir.join(format!("{cache_key}.sha256"));
    if mtime > 0
        && let Ok(cached) = fs::read_to_string(&cache_path)
        && cached.len() == 64
    {
        return Ok(cached);
    }
    let mut file = File::open(path).map_err(|e| format!("hash {}: {e}", path.display()))?;
    let mut hash = Sha256::new();
    let mut buffer = [0u8; 65536];
    loop {
        let size = file.read(&mut buffer).map_err(|e| e.to_string())?;
        if size == 0 {
            break;
        }
        hash.update(&buffer[..size]);
    }
    let result = format!("{:x}", hash.finalize());
    if mtime > 0 && fs::create_dir_all(cache_dir).is_ok() {
        let _ = fs::write(&cache_path, &result);
    }
    Ok(result)
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

/// Construct provenance before launching a worker. No work is attributed by
/// guess.
pub fn start(root: &Path, output: &Path, config: &str, commit: &str) -> Result<Record, String> {
    let config_value: Value = baux::artifact::toml::from_str(config).map_err(|e| e.to_string())?;
    let binding_path = root.join(".cache/baux/task.json");
    let binding = match fs::read_to_string(&binding_path) {
        Ok(text) => {
            let value: Value = json::from_str(&text).map_err(|e| format!("active task: {e}"))?;
            if value["version"].as_u64() != Some(1) || Path::new(field(&value, "workspace")) != root
            {
                return Err(
                    "active task belongs to another workspace or has an unsupported version".into(),
                );
            }
            value
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Value::null(),
        Err(e) => return Err(e.to_string()),
    };
    let change = jj(
        root,
        &["log", "-r", commit, "--no-graph", "-T", "change_id"],
    )?;
    let worker = std::env::var_os("BAUX_WORKER_EXECUTABLE").ok_or("missing worker executable")?;
    let started = jiff::Timestamp::now().as_millisecond();
    let output = output.canonicalize().map_err(|e| e.to_string())?;
    let record = Record {
        schema: 1,
        id: digest(output.to_string_lossy().as_bytes()),
        started,
        task: field(&binding, "id").into(),
        task_store: field(&binding, "store").into(),
        workspace: root.to_string_lossy().into_owned(),
        output: output.to_string_lossy().into_owned(),
        commit: commit.into(),
        change: change.trim().into(),
        runner_sha256: file_digest(&std::env::current_exe().map_err(|e| e.to_string())?)?,
        worker_sha256: file_digest(Path::new(&worker))?,
        config_sha256: digest(config.as_bytes()),
        model: field(&config_value, "model").into(),
        optimizer: field(&config_value["optimizer"], "algorithm").into(),
        recipe: field(&config_value["data"], "recipe").into(),
        data_sha256: data_digest(&config_value)?,
        protocol: protocol(&config_value)?,
        status: "running".into(),
        ended: None,
        report_status: "pending".into(),
        report_error: None,
        error: None,
        measurements: Value::null(),
    };
    publish(root, &record)?;
    Ok(record)
}

fn data_digest(config: &Value) -> Result<String, String> {
    let mut hashes = Vec::new();
    for name in ["train", "validation", "tokenizer"] {
        let path = field(&config["data"], name);
        if !path.is_empty() {
            hashes.push((name, file_digest(Path::new(path))?));
        }
    }
    if hashes.is_empty() {
        return Ok(String::new());
    }
    Ok(digest(
        json::to_string(&hashes)
            .map_err(|e| e.to_string())?
            .as_bytes(),
    ))
}

fn protocol(config: &Value) -> Result<String, String> {
    // Conservative equivalence: include resolved paths, all seeds and workload
    // settings. Do not infer matching data merely from a human-readable name.
    let value = json!({"model": config["model"], "data": config["data"],
        "objective": config["objective"], "evaluation": config["evaluation"],
        "execution": config["execution"], "seeds": config["seeds"], "generation": config["generation"]});
    Ok(digest(
        json::to_string(&value)
            .map_err(|e| e.to_string())?
            .as_bytes(),
    ))
}

/// Training outcome and report outcome are independent. Plot errors are saved.
pub fn finish(root: &Path, record: &mut Record, result: &Result<(), String>) -> Result<(), String> {
    record.status = if result.is_ok() {
        "completed"
    } else {
        "failed"
    }
    .into();
    record.error = result.as_ref().err().cloned();
    record.ended = Some(jiff::Timestamp::now().as_millisecond());
    match plot::render(Path::new(&record.output)) {
        Ok((count, measurements)) => {
            record.report_status = if count == 0 {
                "no_measurements"
            } else {
                "complete"
            }
            .into();
            record.measurements = measurements;
        }
        Err(error) => {
            record.report_status = "failed".into();
            record.report_error = Some(error);
        }
    }
    // The child has exited before this function is called. Freeze even a failed
    // run's successfully captured subset, without replacing the process result
    // if capture finalization fails.
    let sealed = seal_gpu(record);
    publish(root, record)?;
    sealed
}

/// Association with a finished run is distinct from live directory contents.
pub struct GpuEvidence {
    pub sealed: bool,
    pub captures: Vec<(baux::artifact::gpu::Receipt, baux::artifact::gpu::Record)>,
}

#[derive(Serialize, Deserialize)]
#[deser(deny_unknown_fields)]
struct GpuSet {
    schema: u32,
    run: String,
    ids: Vec<String>,
}

#[derive(Serialize, Deserialize)]
#[deser(deny_unknown_fields)]
struct GpuSeal {
    sha256: String,
    payload: GpuSet,
}

const SEAL_LIMIT: u64 = 8 * 1024 * 1024;
static SEAL_NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

fn read_seal(record: &Record) -> Result<Option<GpuSet>, String> {
    let path = Path::new(&record.output).join("gpu-seal.cbor");
    match fs::symlink_metadata(&path) {
        Ok(metadata) if metadata.is_file() && metadata.len() <= SEAL_LIMIT => {}
        Ok(_) => return Err("GPU run seal is not a bounded regular file".into()),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(format!("inspect {}: {error}", path.display())),
    }
    let mut bytes = Vec::new();
    File::open(&path)
        .map_err(|error| error.to_string())?
        .take(SEAL_LIMIT + 1)
        .read_to_end(&mut bytes)
        .map_err(|error| error.to_string())?;
    if bytes.len() as u64 > SEAL_LIMIT {
        return Err("GPU run seal is too large".into());
    }
    let seal: GpuSeal = baux::artifact::cbor::from_slice(&bytes)
        .map_err(|error| format!("GPU run seal: {error}"))?;
    if seal.payload.schema != 1
        || seal.payload.run != record.id
        || seal.payload.ids.len() > 65_536
        || seal.payload.ids.iter().any(|id| {
            id.len() != 64
                || !id
                    .bytes()
                    .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
        })
        || seal.payload.ids.windows(2).any(|pair| pair[0] >= pair[1])
    {
        return Err("GPU run seal has invalid run or record identities".into());
    }
    let payload = baux::artifact::cbor::to_vec(&seal.payload).map_err(|error| error.to_string())?;
    if digest(&payload) != seal.sha256 {
        return Err("GPU run seal digest mismatch".into());
    }
    Ok(Some(seal.payload))
}

/// Return only the receipt set frozen at finish. Legacy/running runs expose
/// explicitly unsealed directory observations. A seal proves neither producer
/// authenticity nor execution; its checksum detects accidental modification.
pub fn gpu_records(record: &Record) -> Result<GpuEvidence, String> {
    let root = Path::new(&record.output).join("gpu");
    if record.status != "running" {
        if let Some(seal) = read_seal(record)? {
            return Ok(GpuEvidence {
                sealed: true,
                captures: baux::artifact::gpu::selected(&root, &seal.ids)?,
            });
        }
    }
    Ok(GpuEvidence {
        sealed: false,
        captures: baux::artifact::gpu::records(&root)?,
    })
}

fn seal_gpu(record: &Record) -> Result<(), String> {
    if !matches!(record.status.as_str(), "completed" | "failed") {
        return Err("GPU capture can only be sealed after the run finishes".into());
    }
    if read_seal(record)?.is_some() {
        gpu_records(record)?;
        return Ok(());
    }
    let root = Path::new(&record.output);
    let payload = GpuSet {
        schema: 1,
        run: record.id.clone(),
        ids: baux::artifact::gpu::records(&root.join("gpu"))?
            .into_iter()
            .map(|(receipt, _)| receipt.id)
            .collect(),
    };
    let payload_bytes =
        baux::artifact::cbor::to_vec(&payload).map_err(|error| error.to_string())?;
    let bytes = baux::artifact::cbor::to_vec(&GpuSeal {
        sha256: digest(&payload_bytes),
        payload,
    })
    .map_err(|error| error.to_string())?;
    if bytes.len() as u64 > SEAL_LIMIT {
        return Err("GPU run seal is too large".into());
    }
    let path = root.join("gpu-seal.cbor");
    let temporary = root.join(format!(
        ".gpu-seal-{}-{}.tmp",
        std::process::id(),
        SEAL_NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
    ));
    let mut file = fs::OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(&temporary)
        .map_err(|error| error.to_string())?;
    let published = (|| {
        file.write_all(&bytes)
            .and_then(|()| file.sync_all())
            .map_err(|error| error.to_string())?;
        match fs::hard_link(&temporary, &path) {
            Ok(()) => Ok(()),
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
                if read_seal(record)?.is_none() {
                    return Err("GPU run seal disappeared".into());
                }
                gpu_records(record).map(|_| ())
            }
            Err(error) => Err(format!("publish {}: {error}", path.display())),
        }
    })();
    drop(file);
    let cleanup = fs::remove_file(&temporary);
    published?;
    cleanup.map_err(|error| error.to_string())
}

/// Explicit provenance edges, usable by agents or a graph renderer. A graph
/// relates evidence; it does not assert that a code change caused improvement.
pub fn graph(records: &[Record]) -> Value {
    let mut nodes = std::collections::BTreeMap::new();
    let mut edges = Vec::new();
    for record in records {
        let run = format!("run:{}", record.id);
        let source = format!("jj:{}", record.commit);
        let config = format!("config:{}", record.config_sha256);
        let artifacts = format!("artifacts:{}", record.id);
        nodes.insert(
            run.clone(),
            json!({"kind": "run", "status": record.status, "started": record.started}),
        );
        nodes.insert(
            source.clone(),
            json!({"kind": "source", "commit": record.commit, "change": record.change}),
        );
        nodes.insert(
            config.clone(),
            json!({"kind": "configuration", "sha256": record.config_sha256}),
        );
        nodes.insert(
            artifacts.clone(),
            json!({"kind": "artifacts", "path": record.output}),
        );
        for (from, to, relation) in [
            (&source, &run, "source_of"),
            (&config, &run, "input_to"),
            (&run, &artifacts, "produced"),
        ] {
            edges.push(json!({"from": from, "to": to, "relation": relation}));
        }
        if !record.task.is_empty() {
            let task = format!("beads:{}:{}", record.task_store, record.task);
            nodes.insert(
                task.clone(),
                json!({"kind": "task", "id": record.task, "store": record.task_store}),
            );
            edges.push(json!({"from": task, "to": run, "relation": "investigated_by"}));
        }
        match gpu_records(record) {
            Ok(evidence) => {
                let association = if evidence.sealed {
                    "sealed-at-run-finish"
                } else {
                    "unsealed-directory-observation"
                };
                for (receipt, capture) in evidence.captures {
                    let compilation = format!("gpu:{}", receipt.id);
                    nodes.insert(compilation.clone(), json!({
                        "kind": "gpu-compilation", "status": "captured",
                        "meaning": "compilation/loading evidence, not an execution trace or correctness proof",
                        "record": capture,
                    }));
                    edges.push(json!({"from": run, "to": compilation, "relation": "captured", "association": association}));
                }
            }
            Err(error) => {
                let capture = format!("gpu-error:{}", record.id);
                nodes.insert(
                    capture.clone(),
                    json!({
                        "kind": "gpu-capture-error", "status": "invalid", "error": error,
                    }),
                );
                edges.push(json!({"from": run, "to": capture, "relation": "capture_invalid"}));
            }
        }
    }
    json!({"nodes": nodes, "edges": edges})
}

pub fn publish(root: &Path, record: &Record) -> Result<(), String> {
    validate(record)?;
    let directory = directory(root)?;
    let bytes = codec::encode(record)?;
    atomic(&Path::new(&record.output).join("run.cbor"), &bytes)?;
    atomic(
        &Path::new(&record.output).join("provenance.cbor"),
        &baux::artifact::cbor::to_vec(&graph(std::slice::from_ref(record)))
            .map_err(|e| e.to_string())?,
    )?;
    atomic(
        &directory
            .join("manifests")
            .join(format!("{}.cbor", record.id)),
        &bytes,
    )?;
    catalog::upsert(&directory, record)
}

fn validate(record: &Record) -> Result<(), String> {
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

/// Bounded task evidence, with source differences rather than inferred
/// causality.
pub fn brief(root: &Path, task: &str, current: &str) -> Result<Value, String> {
    let candidates = history(
        root,
        &Query {
            task: Some(task.into()),
            limit: 100,
            ..Query::default()
        },
    )?;
    let records = select(candidates);
    let baseline = baseline(root, records.first())?;
    let provenance = graph(&records);
    let mut evidence = Vec::new();
    for record in records {
        let diff = jj(
            root,
            &[
                "diff",
                "--from",
                &record.commit,
                "--to",
                current,
                "--summary",
                "--no-pager",
            ],
        );
        let (changed, error, truncated) = match diff {
            Ok(text) => (
                Some(text.lines().take(30).collect::<Vec<_>>().join("\n")),
                None,
                text.lines().count() > 30,
            ),
            Err(error) => (None, Some(error), false),
        };
        evidence.push(json!({"run": record, "source_diff": changed, "source_diff_error": error,
            "source_diff_truncated": truncated,
            "selection_reason": "same Beads task; newest first", "evidence_kind": "execution_record",
            "claim": "Recorded outcome, not a proof or attribution of improvement"}));
    }
    Ok(
        json!({"runs": evidence, "baseline_candidate": baseline, "graph": provenance, "limit": 6, "scope": "active_task",
        "selection": "Newest four task runs, plus latest failure and latest completion within the newest 100 task runs",
        "missing_evidence": "Task completion and causal improvement require explicit checks; recorded runs do not establish either."}),
    )
}

fn select(candidates: Vec<Record>) -> Vec<Record> {
    let mut chosen: Vec<Record> = candidates.iter().take(4).cloned().collect();
    for status in ["failed", "completed"] {
        if let Some(row) = candidates.iter().find(|r| r.status == status)
            && !chosen.iter().any(|r| r.id == row.id)
        {
            chosen.push(row.clone());
        }
    }
    chosen
}

fn baseline(root: &Path, anchor: Option<&Record>) -> Result<Value, String> {
    let Some(anchor) = anchor else {
        return Ok(Value::null());
    };
    if anchor.data_sha256.is_empty() {
        return Ok(Value::null());
    }
    let rows = history(
        root,
        &Query {
            protocol: Some(anchor.protocol.clone()),
            optimizer: Some("muon".into()),
            limit: 100,
            ..Query::default()
        },
    )?;
    let candidate = rows.into_iter().find(|r| {
        r.id != anchor.id
            && r.status == "completed"
            && r.data_sha256 == anchor.data_sha256
            && r.started <= anchor.started
    });
    Ok(candidate.map(|r| json!({"run": r,
        "selection_reason": "matching resolved workload, evaluation, seeds and data hashes; completed Muon run",
        "limitation": "Hardware equivalence and statistical significance are not established"})).unwrap_or_else(Value::null))
}

const OPTIMIZER_PATHS: &[&str] = &[
    "examples/tuning",
    "experiments",
    "rust/crates/optimizer",
    "rust/crates/ennx/src/bf16_metal",
    "rust/crates/ennx/src/bf16_search.metal",
    "rust/crates/ennx/src/bf16_search.rs",
    "rust/crates/ennx/src/fbt_generation",
    "rust/crates/ennx/src/optimizer",
    "rust/crates/ennx/src/posterior",
    "rust/crates/baux/src/resident_cuda.rs",
    "rust/crates/baux/src/resident_metal",
    "rust/crates/baux/src/resident_search.metal",
    "rust/crates/baux/src/fbt_generation",
    "rust/crates/baux/src/fbt_moe",
    "rust/crates/baux/src/muon",
    "rust/crates/baux/src/search",
    "rust/crates/baux/src/trust_region.rs",
    "rust/crates/baux/semantics/src/optimizer",
    "rust/crates/semantics/src/optimizer",
];

/// Compute JJ source and optimizer-specific diffs across a historical run
/// slice.
pub fn source_diffs(root: &Path, records: &[Record], include_patch: bool) -> Value {
    let mut out = Vec::new();
    for (idx, record) in records.iter().enumerate() {
        let prev = records[idx + 1..]
            .iter()
            .find(|older| older.commit != record.commit);
        let to_prev_summary = prev.and_then(|older| {
            jj(
                root,
                &[
                    "diff",
                    "--from",
                    &older.commit,
                    "--to",
                    &record.commit,
                    "--summary",
                    "--no-pager",
                ],
            )
            .ok()
        });
        let optimizer_files: Vec<String> = to_prev_summary
            .as_deref()
            .unwrap_or("")
            .lines()
            .filter(|line| OPTIMIZER_PATHS.iter().any(|p| line.contains(p)))
            .take(20)
            .map(str::to_owned)
            .collect();
        let optimizer_patch = if include_patch {
            prev.and_then(|older| {
                let mut args = vec![
                    "diff",
                    "--from",
                    &older.commit,
                    "--to",
                    &record.commit,
                    "--git",
                    "--no-pager",
                ];
                args.extend_from_slice(OPTIMIZER_PATHS);
                jj(root, &args)
                    .ok()
                    .map(|s| s.lines().take(120).collect::<Vec<_>>().join("\n"))
            })
        } else {
            None
        };
        out.push(json!({
            "id": record.id,
            "commit": record.commit,
            "change": record.change,
            "compared_from_commit": prev.map(|p| p.commit.as_str()),
            "optimizer_files_changed": optimizer_files,
            "optimizer_patch": optimizer_patch,
        }));
    }
    json!(out)
}

#[cfg(test)]
#[path = "runs/tests.rs"]
mod tests;
