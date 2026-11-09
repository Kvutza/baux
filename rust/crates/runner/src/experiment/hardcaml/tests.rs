use baux::config::OptimizerSpec;

use super::*;

fn config() -> Config {
    let spec: OptimizerSpec = baux::artifact::toml::from_str(
        "algorithm = \"bayesian-search\"\ncandidates = 16\nacquisition = \"thompson\"\n",
    )
    .expect("optimizer fixture");
    Config {
        version: 1,
        experiment: "hardcaml-nested-scaling".into(),
        dimensions: vec![8, 64],
        matrix: None,
        flow: None,
        seeds: vec![1],
        budgets: vec![4],
        methods: vec!["fixed".into(), "random".into(), "baux".into()],
        bits: vec![8, 16],
        lanes: vec![1, 4, 8],
        pipelines: vec![false, true],
        target_seeds: vec![1, 2, 1],
        inner_evaluations: 4,
        cycle_budget: 256,
        cache_capacity: 2,
        workers: 1,
        inner: spec.clone(),
        outer: spec,
    }
}

#[test]
fn design_budget() {
    let cfg = config();
    cfg.validate().expect("valid config");
    for design in cfg.designs() {
        assert_eq!(cfg.decode(&cfg.encode(design)), design);
    }
    assert_eq!(cfg.decode(&[1.0; 3]), *cfg.designs().last().expect("grid"));
    assert_eq!(
        Design {
            bits: 8,
            lanes: 3,
            pipeline: true,
            banks: 0,
            queue: 0,
        }
        .cycles(8),
        7
    );
    let mut invalid = cfg.clone();
    invalid.cycle_budget = 1;
    assert!(invalid.validate().is_err());
    invalid = cfg.clone();
    invalid.workers = 0;
    assert!(invalid.validate().is_err());
    invalid = cfg.clone();
    invalid.bits.push(invalid.bits[0]);
    assert!(invalid.validate().is_err());
    invalid = cfg;
    invalid.target_seeds.clear();
    assert!(invalid.validate().is_err());
}

#[test]
fn matrix_grid() {
    let mut cfg = config();
    cfg.experiment = "hardcaml-matvec".into();
    cfg.dimensions.clear();
    cfg.matrix = Some(matvec::Config {
        shapes: vec![
            matvec::Shape { rows: 2, cols: 8 },
            matvec::Shape { rows: 4, cols: 4 },
        ],
        banks: vec![1, 4],
        queues: vec![1, 8],
        samples: 4,
        ready_periods: vec![1, 7, 3],
        ready_cycles: vec![1],
        initial_stalls: vec![0],
        load_gaps: vec![0],
    });
    cfg.cycle_budget = 10000;
    cfg.validate().expect("valid matrix config");
    let mut insufficient = cfg.clone();
    insufficient.cycle_budget = 1;
    assert!(insufficient.validate().is_err());
    for design in cfg.designs() {
        assert_eq!(cfg.decode(&cfg.encode(design)), design);
    }
    assert_eq!(cfg.designs().len(), 48);
    assert_eq!(jobs(&cfg).len(), 6);
    assert_ne!(
        jobs(&cfg)[0].shape.map(|s| s.rows),
        jobs(&cfg)[3].shape.map(|s| s.rows)
    );
    let matrix = cfg.matrix.as_ref().expect("matrix");
    let design = cfg.decode(&[0.5; 5]);
    let train = matrix.request(
        matrix.shapes[0],
        &json!(design),
        1,
        &[0.0; 16],
        &json!({"window":1,"phase":"inner"}),
        3,
    );
    let test = matrix.request(
        matrix.shapes[0],
        &json!(design),
        1,
        &[0.0; 16],
        &json!({"window":1,"phase":"prequential"}),
        3,
    );
    assert_eq!(train["split"].as_str(), Some("train"));
    assert_eq!(test["split"].as_str(), Some("test"));
    assert_eq!(train["ready_period"].as_u64(), Some(7));
    assert!(
        matvec::check_cycles(
            &json!({"cycle_breakdown":{"reset":1,"weight_load":2,"input_load":3,"compute":4}}),
            10
        )
        .is_ok()
    );
    assert!(
        matvec::check_cycles(
            &json!({"cycle_breakdown":{"reset":1,"weight_load":2,"input_load":3,"compute":4}}),
            9
        )
        .is_err()
    );
}

#[test]
fn baseline_budget() {
    let fixed = Job {
        dimension: 8,
        shape: None,
        model: None,
        seed: 1,
        budget: 16,
        method: "fixed".into(),
    };
    let baux = Job {
        method: "baux".into(),
        ..fixed.clone()
    };
    assert_eq!(fixed.candidates(0), baux.candidates(0));
    assert!(fixed.uses_baux(0));
    assert!(!fixed.uses_baux(1));
    assert_eq!(fixed.candidates(1), 1);
    assert_eq!(baux.candidates(1), 16);
}
