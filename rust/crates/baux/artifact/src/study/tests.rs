use std::path::Path;

use super::*;
use crate::artifact::toml::from_str;

#[test]
fn atomic_passthrough() {
    let opts = CompileOpts::default();
    let path = Path::new("experiments/tokagym-shared.toml");
    let study = compile_file(path, &opts).expect("must compile single atomic config");

    assert!(!study.is_multi);
    assert!(study.rungs.is_empty());
    assert_eq!(study.prune, StudyPrune::None);
    assert_eq!(study.cells.len(), 1);

    let cell = &study.cells[0];
    assert_eq!(cell.env_id, "tokamak-tokagym");
    assert_eq!(cell.arm_id, "default");
    assert_eq!(cell.cell_id, "default");
    assert!(cell.baseline);
    assert_eq!(cell.experiment, "tokamak-tokagym");
    assert_eq!(cell.rounds, 10000);
    assert_eq!(cell.repetitions, 10);
    assert!(cell.factors.is_empty());
}

#[test]
fn variant_cleanup() {
    let text = r#"
extends = "tokagym-shared.toml"
backend = "resident-metal"

[optimizer.surrogate]
metric = "conformal"
fit = "fixed"
"#;
    let opts = CompileOpts::default();
    let study = compile_str(text, Path::new("experiments"), &opts)
        .expect("must compile extends with resident metal");

    assert_eq!(study.cells.len(), 1);
    let cell = &study.cells[0];
    let map = cell.value.as_map().expect("cell value must be a map");
    let opt = map
        .get("optimizer")
        .and_then(|v| v.as_map())
        .expect("optimizer table");

    assert!(!opt.contains_key("perturbation"));
    let sur = opt
        .get("surrogate")
        .and_then(|v| v.as_map())
        .expect("surrogate table");
    assert!(!sur.contains_key("fit-candidates"));
    assert!(!sur.contains_key("fit-samples"));
    assert_eq!(sur.get("local-neighbors").and_then(|v| v.as_u64()), Some(4));
}

#[test]
fn suite_sixteen() {
    let opts = CompileOpts::default();
    let path = Path::new("experiments/suite-b.toml");
    let study = compile_file(path, &opts).expect("must compile suite-b.toml");

    assert!(study.is_multi);
    assert_eq!(study.rungs, vec![250, 1000, 3000, 10000]);
    assert_eq!(study.prune, StudyPrune::None);
    assert_eq!(study.cells.len(), 16);

    let toka_cells: Vec<_> = study
        .cells
        .iter()
        .filter(|c| c.env_id == "tokamak-tokagym")
        .collect();
    let ocean_cells: Vec<_> = study
        .cells
        .iter()
        .filter(|c| c.env_id == "oceananigans:kelvin-helmholtz")
        .collect();
    assert_eq!(toka_cells.len(), 8);
    assert_eq!(ocean_cells.len(), 8);

    for cell in &study.cells {
        assert!(
            cell.cell_id
                .starts_with(&format!("{}__{}", cell.env_id, cell.arm_id))
        );
        assert_eq!(cell.rounds, 10000);
        assert_eq!(cell.repetitions, 10);
    }

    assert_eq!(toka_cells.iter().filter(|c| c.baseline).count(), 4);
    assert_eq!(ocean_cells.iter().filter(|c| c.baseline).count(), 4);
}

#[test]
fn suite_tokagym() {
    let opts = CompileOpts::default();
    let path = Path::new("experiments/suite-tokagym.toml");
    let study = compile_file(path, &opts).expect("must compile suite-tokagym.toml");

    assert!(study.is_multi);
    assert_eq!(study.cells.len(), 8);

    for cell in &study.cells {
        assert_eq!(cell.env_id, "tokamak-tokagym");
        assert!(!cell.cell_id.contains("__"));
    }
    assert_eq!(study.cells[0].cell_id, "turbo-zero_origin-only");
    assert!(study.cells[0].baseline);
    assert!(!study.cells[4].baseline);

    let cache_path = Path::new(
        ".cache/baux/campaigns/suite_b_tokagym_metal_10k/configs/tokamak-tokagym__turbo_zero_r10_origin_only.toml",
    );
    if cache_path.exists() {
        let cache_text = std::fs::read_to_string(cache_path).unwrap();
        let cache_val: deser_value::Value = from_str(&cache_text).unwrap();
        assert_eq!(study.cells[0].value, cache_val);
    }
}

#[test]
fn suite_ocean() {
    let opts = CompileOpts::default();
    let path = Path::new("experiments/suite-oceananigans.toml");
    let study = compile_file(path, &opts).expect("must compile suite-oceananigans.toml");

    assert!(study.is_multi);
    assert_eq!(study.cells.len(), 8);

    for cell in &study.cells {
        assert_eq!(cell.env_id, "oceananigans:kelvin-helmholtz");
        assert!(!cell.cell_id.contains("__"));
    }
    assert_eq!(study.cells[0].cell_id, "turbo-zero_origin-only");
    assert!(study.cells[0].baseline);

    let cache_path = Path::new(
        ".cache/baux/campaigns/suite_b_ocean_metal_10k/configs/oceananigans__turbo_zero_r10_origin_only.toml",
    );
    if cache_path.exists() {
        let cache_text = std::fs::read_to_string(cache_path).unwrap();
        let cache_val: deser_value::Value = from_str(&cache_text).unwrap();
        assert_eq!(study.cells[0].value, cache_val);
    }
}

#[test]
fn comma_sweep() {
    let cli_sets = vec!["optimizer.regional-search.count=5,10".to_string()];
    let opts = CompileOpts {
        cli_sets: &cli_sets,
        rounds: Some(500),
        repetitions: Some(3),
        prune: Some(StudyPrune::PairedLcb),
    };
    let path = Path::new("experiments/tokagym-shared.toml");
    let study = compile_file(path, &opts).expect("must compile with cli sets");

    assert!(study.is_multi);
    assert_eq!(study.cells.len(), 2);
    assert_eq!(study.prune, StudyPrune::PairedLcb);

    assert_eq!(study.cells[0].rounds, 500);
    assert_eq!(study.cells[0].repetitions, 3);
    assert_eq!(
        study.cells[0].factors,
        vec![(
            "optimizer.regional-search.count".to_string(),
            "5".to_string()
        )]
    );

    assert_eq!(study.cells[1].rounds, 500);
    assert_eq!(study.cells[1].repetitions, 3);
    assert_eq!(
        study.cells[1].factors,
        vec![(
            "optimizer.regional-search.count".to_string(),
            "10".to_string()
        )]
    );
}

#[test]
fn cycle_detection() {
    let tmp = std::env::temp_dir();
    let a_path = tmp.join("baux_test_cycle_a.toml");
    let b_path = tmp.join("baux_test_cycle_b.toml");
    let _ = std::fs::write(
        &a_path,
        "extends = \"baux_test_cycle_b.toml\"\nversion = 4\nexperiment = \"test\"\n",
    );
    let _ = std::fs::write(
        &b_path,
        "extends = \"baux_test_cycle_a.toml\"\nversion = 4\nexperiment = \"test\"\n",
    );

    let opts = CompileOpts::default();
    let result = compile_file(&a_path, &opts);

    let _ = std::fs::remove_file(&a_path);
    let _ = std::fs::remove_file(&b_path);

    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(err.contains("inheritance cycle detected"), "got err: {err}");
}

#[test]
fn schema_exact() {
    let generated = tune_schema();
    let on_disk = std::fs::read_to_string("schemas/tune-spec.schema.json")
        .expect("must read schemas/tune-spec.schema.json");
    assert_eq!(generated, on_disk);

    let parsed_gen: deser_value::Value =
        crate::artifact::json::from_str(&generated).expect("generated must be valid JSON");
    let parsed_disk: deser_value::Value =
        crate::artifact::json::from_str(&on_disk).expect("on_disk must be valid JSON");
    assert_eq!(parsed_gen, parsed_disk);
}
