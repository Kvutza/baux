use super::*;

#[test]
fn stored_formats() {
    let root = root();
    let row = record(root.path(), "new", 100, "task");
    publish(root.path(), &row).unwrap();
    let output = Path::new(&row.output);
    let bytes = fs::read(output.join("run.cbor")).unwrap();
    let decoded: Record = baux::artifact::cbor::from_slice(&bytes).unwrap();
    assert_eq!(json::to_vec(&decoded).unwrap(), json::to_vec(&row).unwrap());
    assert!(!output.join("run.json").exists());
    let saved: Value =
        baux::artifact::cbor::from_slice(&fs::read(output.join("provenance.cbor")).unwrap())
            .unwrap();
    assert_eq!(saved, graph(&[row]));
}

#[test]
fn legacy_catalogue() {
    const RUNS: redb::TableDefinition<&str, &[u8]> = redb::TableDefinition::new("runs-v1");
    let root = root();
    let mut row = record(root.path(), "old", 100, "old-task");
    publish(root.path(), &row).unwrap();
    let path = directory(root.path()).unwrap().join("catalog.db");
    {
        let db = redb::Database::open(&path).unwrap();
        let tx = db.begin_write().unwrap();
        tx.open_table(RUNS)
            .unwrap()
            .insert("old", json::to_vec(&row).unwrap().as_slice())
            .unwrap();
        tx.commit().unwrap();
    }
    assert_eq!(
        history(root.path(), &Query::default()).unwrap()[0].task,
        "old-task"
    );
    row.task = "new-task".into();
    row.started = 200;
    publish(root.path(), &row).unwrap();
    assert!(
        history(
            root.path(),
            &Query {
                task: Some("old-task".into()),
                ..Query::default()
            }
        )
        .unwrap()
        .is_empty()
    );
    let rows = history(
        root.path(),
        &Query {
            task: Some("new-task".into()),
            ..Query::default()
        },
    )
    .unwrap();
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0].started, 200);
    let db = redb::Database::open(path).unwrap();
    let tx = db.begin_read().unwrap();
    let table = tx.open_table(RUNS).unwrap();
    let raw = table.get("old").unwrap().unwrap();
    assert_eq!(
        baux::artifact::cbor::from_slice::<Record>(raw.value())
            .unwrap()
            .id,
        "old"
    );
}

#[test]
fn mixed_recovery() {
    let root = root();
    let mut row = record(root.path(), "same", 100, "task");
    publish(root.path(), &row).unwrap();
    let manifests = directory(root.path()).unwrap().join("manifests");
    fs::write(manifests.join("same.json"), json::to_vec(&row).unwrap()).unwrap();
    let legacy = record(root.path(), "legacy", 50, "task");
    fs::write(
        manifests.join("legacy.json"),
        json::to_vec(&legacy).unwrap(),
    )
    .unwrap();
    row.status = "completed".into();
    publish(root.path(), &row).unwrap();
    assert_eq!(reindex(root.path()).unwrap(), 2);
    let rows = history(
        root.path(),
        &Query {
            limit: 10,
            ..Query::default()
        },
    )
    .unwrap();
    assert_eq!(rows[0].id, "same");
    assert_eq!(rows[0].status, "completed");
    assert_eq!(rows[1].id, "legacy");
    fs::write(manifests.join("same.cbor"), b"broken").unwrap();
    assert!(reindex(root.path()).unwrap_err().contains("same.cbor"));
    // Failed rebuild leaves the previous catalogue intact, not stale JSON.
    assert_eq!(
        history(root.path(), &Query::default()).unwrap()[0].status,
        "completed"
    );
}

#[test]
fn invalid_records() {
    let root = root();
    let mut row = record(root.path(), "valid", 100, "task");
    let bytes = codec::encode(&row).unwrap();
    assert!(codec::decode(&bytes[..bytes.len() - 1]).is_err());
    let mut trailing = bytes.clone();
    trailing.push(0);
    assert!(codec::decode(&trailing).is_err());
    row.schema = 2;
    assert!(codec::decode(&baux::artifact::cbor::to_vec(&row).unwrap()).is_err());
    let manifests = root.path().join("manifests");
    fs::create_dir(&manifests).unwrap();
    fs::write(manifests.join("wrong.cbor"), bytes).unwrap();
    assert!(
        codec::manifests(&manifests)
            .unwrap_err()
            .contains("filename")
    );
}

#[test]
fn numeric_fidelity() {
    #[derive(Serialize, Deserialize)]
    struct Numbers {
        integer: u64,
        signed: i64,
        floats: Vec<f64>,
        absent: Option<u64>,
        zero: Option<u64>,
    }
    let numbers = Numbers {
        integer: u64::MAX,
        signed: i64::MIN,
        floats: vec![
            -0.0,
            f64::MIN_POSITIVE,
            f64::MAX,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::NAN,
        ],
        absent: None,
        zero: Some(0),
    };
    let bytes = baux::artifact::cbor::to_vec(&numbers).unwrap();
    let decoded: Numbers = baux::artifact::cbor::from_slice(&bytes).unwrap();
    assert_eq!(decoded.integer, u64::MAX);
    assert_eq!(decoded.signed, i64::MIN);
    for (actual, expected) in decoded.floats.iter().zip(&numbers.floats).take(5) {
        assert_eq!(actual.to_bits(), expected.to_bits());
    }
    assert!(decoded.floats[5].is_nan());
    assert_eq!(decoded.absent, None);
    assert_eq!(decoded.zero, Some(0));
    let value = json!({"null": null, "zero": 0});
    let decoded: Value =
        baux::artifact::cbor::from_slice(&baux::artifact::cbor::to_vec(&value).unwrap()).unwrap();
    assert!(decoded.get("missing").is_none());
    assert!(decoded["null"].is_null());
    assert_eq!(decoded["zero"].as_u64(), Some(0));
}

#[test]
fn map_order() {
    let a: Value = json::from_slice(br#"{"long":1,"a":2,"nested":{"z":3,"b":4}}"#).unwrap();
    let b: Value = json::from_slice(br#"{"nested":{"b":4,"z":3},"a":2,"long":1}"#).unwrap();
    assert_eq!(
        baux::artifact::cbor::to_vec(&a).unwrap(),
        baux::artifact::cbor::to_vec(&b).unwrap()
    );
}

#[test]
#[ignore = "manual codec measurement; no timing assertions"]
fn codec_cost() {
    use std::{hint::black_box, time::Instant};
    let root = root();
    let mut row = record(
        root.path(),
        "representative-run",
        1792713600000,
        "baux-2ox.2",
    );
    row.measurements = json!([{"source":"validation.jsonl","x":"round","y":"reward","points":1000,"first":-3.1,"last":-1.2}]);
    let cbor = codec::encode(&row).unwrap();
    let json = json::to_vec(&row).unwrap();
    println!(
        "fixture bytes: CBOR={}, compact JSON={}, pretty JSON={}",
        cbor.len(),
        json.len(),
        json::pretty_vec(&row).unwrap().len()
    );
    let count = 10_000;
    let start = Instant::now();
    for _ in 0..count {
        black_box(codec::encode(black_box(&row)).unwrap());
    }
    println!(
        "CBOR encode ns/record: {}",
        start.elapsed().as_nanos() / count
    );
    let start = Instant::now();
    for _ in 0..count {
        black_box(json::to_vec(black_box(&row)).unwrap());
    }
    println!(
        "JSON encode ns/record: {}",
        start.elapsed().as_nanos() / count
    );
    let start = Instant::now();
    for _ in 0..count {
        black_box(codec::decode(black_box(&cbor)).unwrap());
    }
    println!(
        "CBOR decode ns/record: {}",
        start.elapsed().as_nanos() / count
    );
    let start = Instant::now();
    for _ in 0..count {
        black_box(codec::decode(black_box(&json)).unwrap());
    }
    println!(
        "JSON decode ns/record: {}",
        start.elapsed().as_nanos() / count
    );
}
