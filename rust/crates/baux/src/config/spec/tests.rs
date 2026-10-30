use super::*;

const PRETRAIN_MUON: &str = "version=4\n\
    experiment='pretrain'\n\
    model='fbt-pisa1-mhc4-v1'\n\
    [data]\n\
    recipe='stack-v3-rust-go-v1'\n\
    [optimizer]\n\
    algorithm='muon'\n\
    [objective]\n\
    function='cross-entropy'\n\
    sampling='natural'\n";

const GAUSSIAN: &str = "version = 4\n\
    experiment = \"pretrain\"\n\
    model = \"fbt-pisa1-mhc4-v1\"\n\
    [data]\n\
    recipe = \"stack-v3-rust-go-v1\"\n\
    [objective]\n\
    function = \"cross-entropy\"\n\
    sampling = \"natural\"\n\
    window-tokens = 4096\n\
    [execution]\n\
    rounds = 128\n\
    round-target-ms = 200\n\
    [evaluation]\n\
    interval = 128\n\
    [optimizer]\n\
    algorithm = \"operator-bo\"\n\
    [optimizer.candidate-generation]\n\
    generator = \"independent\"\n\
    arms = 1\n\
    candidates = 1\n\
    [optimizer.candidate-generation.perturbation]\n\
    distribution = \"gaussian\"\n\
    [optimizer.surrogate]\n\
    estimator = \"knn\"\n\
    scaling = \"global\"\n\
    neighbors = 32\n\
    neighbor-rule = \"fixed\"\n\
    epistemic-scale = 0.05\n\
    aleatoric-scale = 0.05\n\
    fit = \"leave-one-out\"\n\
    fit-candidates = 4\n\
    fit-samples = 4\n\
    [optimizer.acceptance]\n\
    criterion = \"positive-improvement\"\n\
    [optimizer.radius-control]\n\
    policy = \"turbo\"\n\
    initial = 0.001\n\
    min = 0.0001\n\
    max = 0.016\n\
    [diagnostics]\n\
    stage-samples = 3\n";

#[cfg(feature = "buck2-test-data")]
const OPERATOR_BO: &str = include_str!("../../../specs/mhc4-1m.toml");
#[cfg(not(feature = "buck2-test-data"))]
const OPERATOR_BO: &str = include_str!("../../../../../../experiments/mhc4-1m.toml");

fn resident(acquisition: &str, candidates: u32, arms: u32, fit: &str) -> String {
    format!(
        "version=4\n\
            experiment='end-to-end'\n\
            [optimizer]\n\
            algorithm='turbo-enn'\n\
            [optimizer.candidate-generation]\n\
            generator='independent'\n\
            candidates={candidates}\n\
            arms={arms}\n\
            [optimizer.surrogate]\n\
            estimator='enn'\n\
            {fit}\n\
            [optimizer.acquisition]\n\
            {acquisition}\n\
            [optimizer.trust-region]\n\
            policy='turbo'\n"
    )
}

#[test]
fn roundtrip() {
    macro_rules! examples {
        ($($name:literal),+ $(,)?) => {
            [$({
                #[cfg(feature = "buck2-test-data")]
                let text = include_str!(concat!("../../../specs/", $name));
                #[cfg(not(feature = "buck2-test-data"))]
                let text = include_str!(concat!("../../../../../../experiments/", $name));
                ($name, text)
            }),+]
        };
    }
    for (name, text) in examples!("mhc4-1m.toml", "mhc4-operator-bo.toml") {
        let spec = ExperimentSpec::parse(text).unwrap_or_else(|error| panic!("{name}: {error}"));
        let resolved = spec.to_toml().unwrap();
        let restored = ExperimentSpec::parse(&resolved).unwrap();
        assert_eq!(resolved, restored.to_toml().unwrap(), "{name}");
    }
}

#[test]
fn fixed_data() {
    let spec = ExperimentSpec::parse(GAUSSIAN).unwrap();
    assert_eq!(spec.objective.window_tokens, Some(4_096));
    assert!(spec.generation.is_none());
    let restored = ExperimentSpec::parse(&spec.to_toml().unwrap()).unwrap();
    assert_eq!(restored.objective.window_tokens, Some(4_096));
}

#[test]
fn closed() {
    for text in [
        format!("{PRETRAIN_MUON}neighbor=10"),
        format!("{PRETRAIN_MUON}[run]\nround=10"),
        PRETRAIN_MUON.replace("algorithm='muon'", "algorithm='unknown'"),
        PRETRAIN_MUON.replace("algorithm='muon'", "algorithm='muon-v1'"),
        format!("{PRETRAIN_MUON}[run]\nselection='enn'"),
        PRETRAIN_MUON.replace("stack-v3-rust-go-v1", "../fineweb"),
        PRETRAIN_MUON.replace(
            "algorithm='muon'",
            "algorithm='muon'\n[optimizer.acquisition]\nfunction='thompson'",
        ),
        GAUSSIAN.replace(
            "generator = \"independent\"",
            "generator = \"independent\"\ncandidate = 4",
        ),
        GAUSSIAN.replace("estimator = \"knn\"", "estimator = \"knn\"\nneigbors = 10"),
        GAUSSIAN.replace("fit = \"leave-one-out\"", "procedure = \"leave-one-out\""),
        GAUSSIAN.replace("fit = \"leave-one-out\"\n", ""),
        OPERATOR_BO.replace("estimator = \"enn\"", "estimator = \"sub-nn\""),
        format!("{GAUSSIAN}[optimizer.acquisition]\nfunction = \"upper-confidence-bound\"\n"),
        OPERATOR_BO.replace(
            "function = \"upper-confidence-bound\"",
            "function = \"thompson\"\nbeta = 1.0",
        ),
        GAUSSIAN.replace(
            "policy = \"turbo\"",
            "policy = \"turbo\"\nunknown-field = 0.9",
        ),
    ] {
        assert!(parse_tune(&text).is_err(), "accepted:\n{text}");
    }
}

#[test]
fn recipe_names() {
    let text = PRETRAIN_MUON.replace("stack-v3-rust-go-v1", "code-mixture-v1");
    assert!(ExperimentSpec::parse(&text).is_ok());
    let slowrun = PRETRAIN_MUON.replace("stack-v3-rust-go-v1", "fineweb-slowrun-100m-v1");
    let slowrun_cfg = parse_tune(&slowrun).unwrap();
    let stack_cfg = parse_tune(PRETRAIN_MUON).unwrap();
    assert_ne!(slowrun_cfg.model_seed(), stack_cfg.model_seed());
    for name in ["123", "Code-Mixture", "code--mixture", "code-mixture-"] {
        assert!(
            ExperimentSpec::parse(&PRETRAIN_MUON.replace("stack-v3-rust-go-v1", name)).is_err()
        );
    }
}

#[test]
fn guards() {
    for text in [
        PRETRAIN_MUON.replace(
            "algorithm='muon'",
            "algorithm='muon'\n[execution]\nrounds=0",
        ),
        GAUSSIAN.replace("candidates = 1", "candidates = 0"),
        GAUSSIAN.replace("candidates = 1", "candidates = 4"),
        GAUSSIAN.replace("arms = 1", "arms = 2"),
        resident("function='upper-confidence-bound'", 4, 3, "fit='fixed'"),
        GAUSSIAN.replace("neighbors = 32", "neighbors = 0"),
        GAUSSIAN.replace("epistemic-scale = 0.05", "epistemic-scale = nan"),
        GAUSSIAN.replace("min = 0.0001", "min = 0.2"),
        GAUSSIAN.replace("policy = \"turbo\"", "policy = \"reliability\""),
        GAUSSIAN.replace("min = 0.0001", "min = nan"),
        GAUSSIAN.replace("max = 0.016", "max = 0.00001"),
        GAUSSIAN.replace("initial = 0.001", "initial = -1.0"),
    ] {
        assert!(
            parse_tune(&text).is_err(),
            "accepted invalid document:\n{text}"
        );
    }
}

#[test]
fn distinct_roles() {
    assert!(parse_tune(&GAUSSIAN.replace("knn", "enn")).is_err());
    assert!(parse_tune(&GAUSSIAN.replace("operator-bo", "turbo-enn")).is_err());
}

#[test]
fn random_acquisition() {
    let text = OPERATOR_BO.replace("upper-confidence-bound", "random");
    let config = parse_tune(&text).unwrap();
    assert_eq!(config.selection, Some(PretrainSelection::Random));
    let spec = ExperimentSpec::from_runtime(&config).unwrap();
    let OptimizerSpec::TurboEnn { acquisition, .. } = spec.optimizer else {
        panic!("expected turbo-enn");
    };
    assert!(matches!(acquisition, AcquisitionSpec::Random));
}

#[test]
fn turbo_radius() {
    let config = parse_tune(GAUSSIAN).unwrap();
    assert_eq!(config.reliability_controller(), None);
    let spec = ExperimentSpec::from_runtime(&config).unwrap();
    let roundtrip = parse_tune(&spec.to_toml().unwrap()).unwrap();
    assert_eq!(roundtrip.reliability_controller(), None);
    assert!(
        parse_tune(&GAUSSIAN.replace("policy = \"turbo\"", "policy = \"reliability\"")).is_err()
    );
}

#[test]
fn hyperbolic_scaling() {
    let text = GAUSSIAN.replace(
        "scaling = \"global\"",
        "scaling = \"hyperbolic\"\nlocal-neighbors = 8",
    );
    let config = parse_tune(&text).unwrap();
    assert_eq!(config.distance_scaling, Some(DistanceScaling::Hyperbolic));
    assert_eq!(config.local_scale_neighbors, Some(8));
    let spec = ExperimentSpec::from_runtime(&config).unwrap();
    let roundtrip = parse_tune(&spec.to_toml().unwrap()).unwrap();
    assert_eq!(
        roundtrip.distance_scaling,
        Some(DistanceScaling::Hyperbolic)
    );
    assert_eq!(roundtrip.local_scale_neighbors, Some(8));
}

#[test]
fn bounded_likelihood() {
    let text = GAUSSIAN.replace(
        "function = \"cross-entropy\"",
        "function = \"bounded-log-likelihood\"",
    );
    let config = parse_tune(&text).unwrap();
    assert_eq!(
        config.objective_function(),
        ObjectiveFunction::BoundedLogLikelihood
    );
    let spec = ExperimentSpec::from_runtime(&config).unwrap();
    let roundtrip = parse_tune(&spec.to_toml().unwrap()).unwrap();
    assert_eq!(
        roundtrip.objective_function(),
        ObjectiveFunction::BoundedLogLikelihood
    );
    let bll = ObjectiveFunction::BoundedLogLikelihood;
    assert_eq!(bll.token_loss(0.0), 0.0);
    let small = bll.token_loss(1e-4);
    assert!((small - 1e-4).abs() < 1e-7);
    let large = bll.token_loss(50.0);
    assert!((large - std::f32::consts::LN_2).abs() < 1e-6);
}

#[test]
fn paths() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("experiment.toml");
    let text = resident("function='thompson'", 4, 1, "fit='fixed'").replace(
        "experiment='end-to-end'",
        "experiment='end-to-end'\noutput='results'",
    );
    std::fs::write(&path, &text).unwrap();
    let authored = std::fs::read_to_string(&path).unwrap();
    let (config, text) = load_tune(&path).unwrap();
    let restored = parse_tune(&text).unwrap();
    assert_eq!(
        restored.output(),
        directory.path().canonicalize().unwrap().join("results")
    );
    assert_eq!(restored.length(), config.length());
    assert_eq!(restored.acquisition_seed(), config.acquisition_seed());
    assert!(matches!(
        restored.acquisition,
        Some(AcquisitionConfig::Thompson)
    ));
    assert!(text.contains("version = 4"));
    assert!(text.contains("[optimizer.candidate-generation]"));
    assert_eq!(std::fs::read_to_string(path).unwrap(), authored);
}

#[test]
fn models() {
    assert_eq!(
        ModelArchitecture::ALL.map(ModelArchitecture::id),
        [
            "fbt-pisa1-legacy-v1",
            "fbt-pisa1-residual1-v1",
            "fbt-pisa1-projected-boundary-v1",
            "fbt-pisa1-hc4-v1",
            "fbt-pisa1-mhc4-v1",
            "fbt-pisa1-looped-mhc4-v1",
            "fbt-pisa1-diffusion-mhc4-v1",
            "fbt-pisa1-hnet-mhc4-v1",
        ]
    );
}

#[test]
fn budgets() {
    let config = parse_tune(&resident(
        "function='upper-confidence-bound'",
        4,
        2,
        "fit='leave-one-out'\nfit-candidates=64\nfit-samples=32",
    ))
    .unwrap();
    let surrogate = config.resident_enn(17).unwrap();
    assert_eq!(surrogate.pool.count(), 4);
    assert_eq!(surrogate.pool.arms(), 2);
    assert_eq!(surrogate.num_candidates, 64);
    assert_eq!(surrogate.num_samples, 32);
    assert!(surrogate.fit_params);
    assert!(!surrogate.adapt_neighbors);
}

#[test]
fn search_conflict() {
    let ambiguous = resident(
        "function='upper-confidence-bound'",
        4,
        1,
        "fit='leave-one-out-neighbor-search-v1'\nfit-candidates=4\nfit-samples=4",
    );
    assert!(parse_tune(&ambiguous).is_err());
}

#[test]
fn neighbor_controls() {
    let fixed = resident("function='upper-confidence-bound'", 4, 1, "fit='fixed'");
    let fixed = parse_tune(&fixed).unwrap().resident_enn(17).unwrap();
    assert!(!fixed.fit_params);
    assert!(!fixed.adapt_neighbors);

    let fitted = resident(
        "function='upper-confidence-bound'",
        4,
        1,
        "fit='leave-one-out'\nfit-candidates=4\nfit-samples=4",
    );
    let fitted = parse_tune(&fitted).unwrap().resident_enn(17).unwrap();
    assert!(fitted.fit_params);
    assert!(!fitted.adapt_neighbors);

    let adaptive = resident("function='upper-confidence-bound'", 4, 1, "fit='fixed'").replace(
        "estimator='enn'",
        "estimator='enn'\nneighbor-rule='risk-prefix'",
    );
    let adaptive = parse_tune(&adaptive).unwrap().resident_enn(17).unwrap();
    assert!(!adaptive.fit_params);
    assert!(adaptive.adapt_neighbors);

    let fitted_adaptive = resident(
        "function='upper-confidence-bound'",
        4,
        1,
        "fit='leave-one-out'\nfit-candidates=4\nfit-samples=4",
    )
    .replace(
        "estimator='enn'",
        "estimator='enn'\nneighbor-rule='risk-prefix'",
    );
    let fitted_adaptive = parse_tune(&fitted_adaptive)
        .unwrap()
        .resident_enn(17)
        .unwrap();
    assert!(fitted_adaptive.fit_params);
    assert!(fitted_adaptive.adapt_neighbors);
}

#[test]
fn vectors() {
    let generation = "\n\
        [generation]\n\
        purpose='systems-probe'\n\
        max-tokens=16\n\
        temperature=0.0\n\
        [[generation.tasks]]\n\
        prompt=[1]\n\
        [generation.reward]\n\
        kind='command-objectives'\n\
        program='score'\n\
        args=[]\n\
        timeout-ms=10\n";
    let base = resident("function='pareto'\nscales=[1.0,2.0]", 4, 1, "fit='fixed'").replace(
        "experiment='end-to-end'",
        "experiment='generation'\nmodel='fbt-pisa1-legacy-v1'",
    );
    let config = parse_tune(&(base.clone() + generation)).unwrap();
    assert_eq!(
        config.objective_acquisition.unwrap().mode,
        ResidentObjectiveMode::Pareto
    );
    assert!(
        parse_tune(
            &(base.replace(
                "scales=[1.0,2.0]",
                "scales=[1.0,2.0]\npreferences=[1.0,1.0]"
            ) + generation)
        )
        .is_err()
    );
}

#[test]
fn causal_recipe() {
    let base = resident("function='thompson'", 1, 1, "fit='fixed'").replace(
        "experiment='end-to-end'",
        "experiment='pretrain'\n\
            model='fbt-pisa1-mhc4-v1'\n\
            [data]\n\
            recipe='stack-v3-rust-go-v1'\n\
            [objective]\n\
            function='cross-entropy'\n\
            sampling='natural'\n\
            window-tokens=1048576",
    );
    let config = parse_tune(&base).unwrap();
    assert_eq!(config.window_tokens, Some(1_048_576));
    assert!(config.generation.is_none());
    assert!(config.objective_acquisition.is_none());
    assert!(parse_tune(&base.replace("recipe=", "mixture=")).is_err());
    assert!(parse_tune(&(base.clone() + "\n[objective]\nreference='moving-incumbent'\n")).is_err());
    assert!(
        parse_tune(
            &(base
                + "\n\
                    [generation]\n\
                    purpose='pretrain'\n\
                    max-tokens=16\n\
                    temperature=1.0\n\
                    [generation.reward]\n\
                    kind='causal-objectives'\n")
        )
        .is_err()
    );
}

#[test]
fn natural_objective() {
    let spec = ExperimentSpec::parse(PRETRAIN_MUON).unwrap();
    let authored = spec.to_toml().unwrap();
    assert!(authored.contains("function = \"cross-entropy\""));
    assert!(authored.contains("sampling = \"natural\""));
    assert!(!authored.contains("[generation]"));
    assert!(!authored.contains("max-tokens"));
    assert!(!authored.contains("corpus-prompt-tokens"));
}

#[test]
fn window_budget() {
    assert!(ExperimentSpec::parse(&format!("{PRETRAIN_MUON}window-tokens=4096\n")).is_err());
    assert!(ExperimentSpec::parse(&GAUSSIAN.replace("window-tokens = 4096\n", "")).is_err());
    assert!(
        ExperimentSpec::parse(&GAUSSIAN.replace("sampling = \"natural\"", "sampling = \"window\""))
            .is_err()
    );
}
