use super::*;

#[path = "tests/encoding.rs"]
mod encoding;

fn root() -> tempfile::TempDir {
    let root = tempfile::tempdir().unwrap();
    fs::create_dir_all(root.path().join(".jj/repo")).unwrap();
    root
}

fn record(root: &Path, id: &str, started: i64, task: &str) -> Record {
    let output = root.join(id);
    fs::create_dir_all(&output).unwrap();
    Record {
        schema: 1,
        id: id.into(),
        started,
        task: task.into(),
        task_store: root.display().to_string(),
        workspace: root.display().to_string(),
        output: output.display().to_string(),
        commit: "a".repeat(40),
        change: "change".into(),
        runner_sha256: "runner".into(),
        worker_sha256: "worker".into(),
        config_sha256: "config".into(),
        data_sha256: "data".into(),
        model: "model".into(),
        optimizer: "muon".into(),
        recipe: "recipe".into(),
        protocol: "protocol".into(),
        status: "running".into(),
        ended: None,
        report_status: "pending".into(),
        report_error: None,
        error: None,
        measurements: Value::null(),
    }
}

#[test]
fn history_recovery() {
    let root = root();
    for (id, time, task) in [
        ("a", 100, "task"),
        ("b", 200, "task.child"),
        ("c", 300, "task"),
    ] {
        publish(root.path(), &record(root.path(), id, time, task)).unwrap();
    }
    let rows = history(
        root.path(),
        &Query {
            since: Some(100),
            until: Some(300),
            limit: 10,
            ..Query::default()
        },
    )
    .unwrap();
    assert_eq!(
        rows.iter().map(|r| r.id.as_str()).collect::<Vec<_>>(),
        ["b", "a"]
    );
    let rows = history(
        root.path(),
        &Query {
            task: Some("task".into()),
            limit: 1,
            ..Query::default()
        },
    )
    .unwrap();
    assert_eq!(rows[0].id, "c");
    let dir = directory(root.path()).unwrap();
    fs::write(dir.join("catalog.db"), b"corrupted derived index").unwrap();
    assert!(
        history(
            root.path(),
            &Query {
                limit: 10,
                ..Query::default()
            }
        )
        .is_err()
    );
    assert_eq!(reindex(root.path()).unwrap(), 3);
    assert_eq!(
        history(
            root.path(),
            &Query {
                limit: 10,
                ..Query::default()
            }
        )
        .unwrap()
        .len(),
        3
    );
}

#[test]
fn shared_catalogue() {
    let root = root();
    let workspace = tempfile::tempdir().unwrap();
    fs::create_dir_all(workspace.path().join(".jj")).unwrap();
    fs::write(
        workspace.path().join(".jj/repo"),
        root.path().join(".jj/repo").to_string_lossy().as_bytes(),
    )
    .unwrap();
    publish(root.path(), &record(root.path(), "a", 100, "task")).unwrap();
    assert_eq!(
        directory(root.path()).unwrap(),
        directory(workspace.path()).unwrap()
    );
    assert_eq!(
        history(
            workspace.path(),
            &Query {
                limit: 10,
                ..Query::default()
            }
        )
        .unwrap()
        .len(),
        1
    );
}

#[test]
fn writer_process() {
    let Ok(path) = std::env::var("BAUX_TEST_CATALOG_ROOT") else {
        return;
    };
    let id = std::env::var("BAUX_TEST_CATALOG_ID").unwrap();
    let root = Path::new(&path);
    publish(root, &record(root, &id, 100, "task")).unwrap();
}

#[test]
fn concurrent_writers() {
    let root = root();
    let mut children = Vec::new();
    for id in ["a", "b", "c"] {
        children.push(
            std::process::Command::new(std::env::current_exe().unwrap())
                .args(["--exact", "artifact::runs::tests::writer_process"])
                .env("BAUX_TEST_CATALOG_ROOT", root.path())
                .env("BAUX_TEST_CATALOG_ID", id)
                .stdout(std::process::Stdio::null())
                .spawn()
                .unwrap(),
        );
    }
    for mut child in children {
        assert!(child.wait().unwrap().success());
    }
    assert_eq!(
        history(
            root.path(),
            &Query {
                limit: 10,
                ..Query::default()
            }
        )
        .unwrap()
        .len(),
        3
    );
}

#[test]
fn training_failure() {
    let root = root();
    let mut row = record(root.path(), "failed", 100, "task");
    fs::write(
        Path::new(&row.output).join("validation.jsonl"),
        "{\"round\":0,\"reward\":-3.0}\n{\"round\":1,\"reward\":-2.0}\n",
    )
    .unwrap();
    finish(root.path(), &mut row, &Err("worker exited 7".into())).unwrap();
    assert_eq!(row.status, "failed");
    assert_eq!(row.report_status, "complete");
    assert_eq!(row.measurements.as_seq().unwrap().len(), 1);
    let svg =
        fs::read_to_string(Path::new(&row.output).join("validation.jsonl.reward.svg")).unwrap();
    assert!(svg.contains("<svg"));
    assert_eq!(
        history(
            root.path(),
            &Query {
                limit: 1,
                ..Query::default()
            }
        )
        .unwrap()[0]
            .error
            .as_deref(),
        Some("worker exited 7")
    );
}

#[test]
fn plotting_failure() {
    let root = root();
    let mut row = record(root.path(), "complete", 100, "task");
    fs::write(
        Path::new(&row.output).join("validation.jsonl"),
        "{truncated",
    )
    .unwrap();
    finish(root.path(), &mut row, &Ok(())).unwrap();
    assert_eq!(row.status, "completed");
    assert_eq!(row.report_status, "failed");
    assert!(row.report_error.is_some());
}

#[test]
fn provenance() {
    let root = root();
    let row = record(root.path(), "a", 100, "task");
    let graph = graph(&[row]);
    assert_eq!(graph["edges"].as_seq().unwrap().len(), 4);
    let a = json!({"model": "model", "data": {"train": "a"}, "optimizer": {"algorithm": "muon"}});
    let b = json!({"model": "model", "data": {"train": "a"}, "optimizer": {"algorithm": "enn"}});
    assert_eq!(protocol(&a).unwrap(), protocol(&b).unwrap());
    let c = json!({"model": "model", "data": {"train": "b"}});
    assert_ne!(protocol(&a).unwrap(), protocol(&c).unwrap());
    assert!(since("nonsense").is_err());
    assert_eq!(since("2026-10-23T00:00:00Z").unwrap(), 1792713600000);
}

#[test]
fn gpu_provenance() {
    use baux::artifact::gpu::{DeviceKind, Kind, Recipe, Stage};
    let root = root();
    let row = record(root.path(), "gpu", 100, "task");
    let output = Path::new(&row.output).join("gpu");
    let recipe = Recipe {
        device_kind: DeviceKind::Metal,
        device: "test device".into(),
        target: "test target".into(),
        compiler: "test compiler".into(),
        stage: Stage::Compile,
        entry: "entry".into(),
        input_kind: Kind::MetalSource,
        output_kind: None,
        options: vec![],
        constants: vec![],
    };
    let receipt = baux::artifact::gpu::record(&output, &recipe, b"source", None).unwrap();
    let captured = graph(std::slice::from_ref(&row));
    let id = format!("gpu:{}", receipt.id);
    assert_eq!(
        captured["nodes"][id.as_str()]["status"].as_str(),
        Some("captured")
    );
    assert!(captured["edges"].as_seq().unwrap().iter().any(|edge| {
        edge["from"].as_str() == Some("run:gpu") && edge["to"].as_str() == Some(id.as_str())
    }));
    fs::write(
        output.join("blobs").join(format!("{}.bin", receipt.input)),
        b"broken",
    )
    .unwrap();
    let broken = graph(&[row]);
    assert_eq!(
        broken["nodes"]["gpu-error:gpu"]["status"].as_str(),
        Some("invalid")
    );
    assert!(broken["nodes"].get(id.as_str()).is_none());
}

fn capture_gpu(row: &Record, entry: &str) -> baux::artifact::gpu::Receipt {
    use baux::artifact::gpu::{DeviceKind, Kind, Recipe, Stage};
    let recipe = Recipe {
        device_kind: DeviceKind::Metal,
        device: "device".into(),
        target: "target".into(),
        compiler: "compiler".into(),
        stage: Stage::Compile,
        entry: entry.into(),
        input_kind: Kind::MetalSource,
        output_kind: None,
        options: vec![],
        constants: vec![],
    };
    baux::artifact::gpu::record(
        &Path::new(&row.output).join("gpu"),
        &recipe,
        entry.as_bytes(),
        None,
    )
    .unwrap()
}

#[test]
fn gpu_sealed() {
    let root = root();
    let mut row = record(root.path(), "sealed", 100, "task");
    publish(root.path(), &row).unwrap();
    assert!(!Path::new(&row.output).join("gpu-seal.cbor").exists());
    let first = capture_gpu(&row, "first");
    assert!(!gpu_records(&row).unwrap().sealed);
    finish(root.path(), &mut row, &Err("worker failed".into())).unwrap();
    let sealed = gpu_records(&row).unwrap();
    assert!(sealed.sealed);
    assert_eq!(sealed.captures.len(), 1);
    assert_eq!(sealed.captures[0].0.id, first.id);
    let second = capture_gpu(&row, "after-finish");
    assert_eq!(gpu_records(&row).unwrap().captures.len(), 1);
    // Even a corrupt unrelated later capture cannot invalidate this run's set.
    fs::write(
        Path::new(&row.output)
            .join("gpu/blobs")
            .join(format!("{}.bin", second.input)),
        b"bad",
    )
    .unwrap();
    assert_eq!(gpu_records(&row).unwrap().captures.len(), 1);
    let provenance = graph(std::slice::from_ref(&row));
    assert!(
        provenance["nodes"]
            .get(format!("gpu:{}", second.id).as_str())
            .is_none()
    );
    assert!(
        provenance["edges"]
            .as_seq()
            .unwrap()
            .iter()
            .any(|edge| edge["association"].as_str() == Some("sealed-at-run-finish"))
    );
    fs::write(
        Path::new(&row.output)
            .join("gpu/blobs")
            .join(format!("{}.bin", first.input)),
        b"bad",
    )
    .unwrap();
    assert!(gpu_records(&row).is_err());
    assert_eq!(
        graph(&[row])["nodes"]["gpu-error:sealed"]["status"].as_str(),
        Some("invalid")
    );
}

#[test]
fn seal_damage() {
    let root = root();
    let mut row = record(root.path(), "damaged", 100, "task");
    capture_gpu(&row, "first");
    finish(root.path(), &mut row, &Err("worker failed".into())).unwrap();
    let path = Path::new(&row.output).join("gpu-seal.cbor");
    let mut seal: GpuSeal = baux::artifact::cbor::from_slice(&fs::read(&path).unwrap()).unwrap();
    seal.payload.ids.clear();
    let damaged = baux::artifact::cbor::to_vec(&seal).unwrap();
    fs::write(&path, &damaged).unwrap();
    assert!(gpu_records(&row).err().unwrap().contains("digest mismatch"));
    assert!(finish(root.path(), &mut row, &Err("worker failed".into())).is_err());
    assert_eq!(row.status, "failed");
    assert_eq!(row.error.as_deref(), Some("worker failed"));
    let saved = history(
        root.path(),
        &Query {
            limit: 1,
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(saved[0].status, "failed");
    assert_eq!(saved[0].error.as_deref(), Some("worker failed"));
    assert_eq!(fs::read(path).unwrap(), damaged);
    assert!(
        graph(&[row])["nodes"]["gpu-error:damaged"]["error"]
            .as_str()
            .unwrap()
            .contains("digest mismatch")
    );
}

#[test]
fn gpu_legacy() {
    let root = root();
    let mut row = record(root.path(), "legacy", 100, "task");
    row.status = "completed".into();
    capture_gpu(&row, "first");
    assert!(!gpu_records(&row).unwrap().sealed);
    let provenance = graph(&[row]);
    assert!(
        provenance["edges"]
            .as_seq()
            .unwrap()
            .iter()
            .any(|edge| edge["association"].as_str() == Some("unsealed-directory-observation"))
    );
}

#[test]
fn curation() {
    let root = root();
    let mut rows = Vec::new();
    for index in 0..8 {
        let mut row = record(root.path(), &format!("r{index}"), 1000 - index, "task");
        row.status = if index == 7 { "failed" } else { "completed" }.into();
        rows.push(row);
    }
    let chosen = select(rows);
    assert!(chosen.len() <= 6);
    assert!(chosen.iter().any(|r| r.status == "failed"));
    let anchor = &chosen[0];
    let mut candidate = record(root.path(), "baseline", 900, "other-task");
    candidate.status = "completed".into();
    candidate.data_sha256 = "different-data".into();
    publish(root.path(), &candidate).unwrap();
    assert!(baseline(root.path(), Some(anchor)).unwrap().is_null());
    candidate.data_sha256 = anchor.data_sha256.clone();
    publish(root.path(), &candidate).unwrap();
    assert_eq!(
        field(&baseline(root.path(), Some(anchor)).unwrap()["run"], "id"),
        "baseline"
    );
}

#[test]
fn pretrain_comparison() {
    let root = root();
    let mut row1 = record(root.path(), "bo1", 200, "task");
    row1.optimizer = "operator-bo".into();
    row1.recipe = "stack@4096".into();
    fs::write(
        Path::new(&row1.output).join("optimization.jsonl"),
        "{\"round\":1,\"incumbent_nll\":12.48,\"candidate_nll\":12.48,\"round_wall_ms\":290.0}\n\
         {\"round\":2,\"incumbent_nll\":12.46,\"candidate_nll\":12.46,\"round_wall_ms\":285.0}\n",
    )
    .unwrap();
    fs::write(
        Path::new(&row1.output).join("validation.json"),
        "{\"measurements\":[{\"round\":0,\"nll\":12.48},{\"round\":2,\"nll\":12.46}]}",
    )
    .unwrap();
    finish(root.path(), &mut row1, &Ok(())).unwrap();
    assert_eq!(row1.report_status, "complete");
    assert_eq!(row1.measurements.as_seq().unwrap().len(), 4);
    assert!(
        Path::new(&row1.output)
            .join("optimization.jsonl.incumbent_nll.svg")
            .exists()
    );
    assert!(
        Path::new(&row1.output)
            .join("validation.json.nll.svg")
            .exists()
    );

    let mut row2 = record(root.path(), "muon1", 100, "task");
    row2.optimizer = "muon".into();
    row2.recipe = "stack@4096".into();
    fs::write(
        Path::new(&row2.output).join("optimization.jsonl"),
        "{\"round\":1,\"nll_after\":12.10,\"round_wall_ms\":13000.0}\n{\"round\":2,\"nll_after\":\
         11.83,\"round_wall_ms\":13100.0}\n",
    )
    .unwrap();
    finish(root.path(), &mut row2, &Ok(())).unwrap();

    let cmp = compare(root.path(), &[row1.clone(), row2.clone()]).unwrap();
    let seq = cmp.as_seq().unwrap();
    assert_eq!(seq.len(), 5);
    let diffs = source_diffs(root.path(), &[row1.clone(), row2.clone()], true);
    let html_path = write_html(root.path(), &[row1, row2], &cmp, &diffs).unwrap();
    assert!(html_path.exists());
    let html = fs::read_to_string(&html_path).unwrap();
    assert!(html.contains("BAUX Runs"));
    assert!(html.contains("WB_RAW_CHARTS"));
    assert!(html.contains("echarts.min.js"));
    assert!(
        directory(root.path())
            .unwrap()
            .join("comparisons/comparison.nll.svg")
            .exists()
    );
    assert!(
        directory(root.path())
            .unwrap()
            .join("comparisons/comparison.nll_by_time.svg")
            .exists()
    );
}

#[test]
fn binary_updates() {
    let root = root();
    let mut row = record(root.path(), "upd1", 200, "task");
    let mut bytes = vec![0u8; 24 + 160];
    bytes[0..8].copy_from_slice(b"BAUXUPD1");
    bytes[8..12].copy_from_slice(&1u32.to_le_bytes());
    bytes[12..16].copy_from_slice(&10u32.to_le_bytes());
    bytes[16..20].copy_from_slice(&5u32.to_le_bytes());
    bytes[20..24].copy_from_slice(&2u32.to_le_bytes());
    for (idx, fam) in [0u8, 6u8].into_iter().enumerate() {
        let base = 24 + idx * 80;
        bytes[base + 4..base + 8].copy_from_slice(&4u32.to_le_bytes());
        bytes[base + 8] = fam;
        bytes[base + 20..base + 24].copy_from_slice(&2.0f32.to_le_bytes());
        bytes[base + 64..base + 72].copy_from_slice(&0.5f64.to_le_bytes());
        bytes[base + 72..base + 80].copy_from_slice(&0.4f64.to_le_bytes());
    }
    fs::write(Path::new(&row.output).join("tensor-updates.bin"), bytes).unwrap();
    finish(root.path(), &mut row, &Ok(())).unwrap();
    assert_eq!(row.report_status, "complete");
    assert_eq!(row.measurements.as_seq().unwrap().len(), 2);
    assert!(
        Path::new(&row.output)
            .join("tensor-updates.bin.rms.svg")
            .exists()
    );
    assert!(
        Path::new(&row.output)
            .join("tensor-updates.bin.coherence.svg")
            .exists()
    );
}
