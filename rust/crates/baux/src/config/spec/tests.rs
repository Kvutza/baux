use super::*;

const PRETRAIN_MUON: &str = "version=4\n\
    experiment='pretrain'\n\
    model='fbt-pisa1-mhc4-v1'\n\
    recipe='stack-v3-rust-go-v1'\n\
    objective='cross-entropy'\n\
    [optimizer]\n\
    algorithm='muon'\n";

const GAUSSIAN: &str = "version = 4\n\
    experiment = \"pretrain\"\n\
    model = \"fbt-pisa1-mhc4-v1\"\n\
    recipe = \"stack-v3-rust-go-v1\"\n\
    objective = \"cross-entropy\"\n\
    window-tokens = 4096\n\
    rounds = 128\n\
    round-target-ms = 200\n\
    validation-interval = 128\n\
    stage-samples = 3\n\
    [optimizer]\n\
    algorithm = \"operator-bo\"\n\
    proposal = \"independent\"\n\
    distribution = \"gaussian\"\n\
    [optimizer.surrogate]\n\
    scaling = \"global\"\n\
    neighbors = 32\n\
    epistemic-scale = 0.05\n\
    aleatoric-scale = 0.05\n\
    fit = \"leave-one-out\"\n\
    fit-candidates = 4\n\
    fit-samples = 4\n\
    [optimizer.trust-region]\n\
    initial = 0.001\n\
    min = 0.0001\n\
    max = 0.016\n";

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
            proposal='independent'\n\
            candidates={candidates}\n\
            arms={arms}\n\
            {acquisition}\n\
            [optimizer.surrogate]\n\
            {fit}\n"
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
    for (name, text) in examples!(
        "baux.toml",
        "muon.toml",
        "mhc4-slowrun-1000r.toml",
        "mhc4-slowrun-100m.toml",
        "hnet-slowrun-1000r.toml",
        "mhc4-1m.toml",
        "mhc4-native.toml"
    ) {
        let spec = ExperimentSpec::parse(text).unwrap_or_else(|error| panic!("{name}: {error}"));
        let resolved = spec.to_toml().unwrap();
        let restored = ExperimentSpec::parse(&resolved).unwrap();
        assert_eq!(resolved, restored.to_toml().unwrap(), "{name}");
    }
}

#[test]
fn hnet_long() {
    let text = GAUSSIAN
        .replace("fbt-pisa1-mhc4-v1", "fbt-pisa1-hnet-mhc4-v1")
        .replace("stack-v3-rust-go-v1", "fineweb-bytes-100m-v1");
    assert!(ExperimentSpec::parse(&text).is_ok());
    assert!(
        ExperimentSpec::parse(&text.replace("window-tokens = 4096", "window-tokens = 1048576"))
            .is_ok()
    );
    assert!(
        ExperimentSpec::parse(&text.replace("window-tokens = 4096", "window-tokens = 2097152"))
            .is_err()
    );
    assert!(ExperimentSpec::parse(&text.replace("cross-entropy", "simpo")).is_err());
}

#[test]
fn fixed_data() {
    let spec = ExperimentSpec::parse(GAUSSIAN).unwrap();
    assert_eq!(spec.window_tokens, Some(4_096));
    assert!(spec.generation.is_none());
    let restored = ExperimentSpec::parse(&spec.to_toml().unwrap()).unwrap();
    assert_eq!(restored.window_tokens, Some(4_096));
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
            "algorithm='muon'\nacquisition='thompson'",
        ),
        GAUSSIAN.replace(
            "proposal = \"independent\"",
            "proposal = \"independent\"\ncandidate = 4",
        ),
        GAUSSIAN.replace(
            "proposal = \"independent\"",
            "proposal = \"independent\"\ncandidates = 4",
        ),
        GAUSSIAN.replace(
            "proposal = \"independent\"",
            "proposal = \"independent\"\narms = 2",
        ),
        GAUSSIAN.replace("neighbors = 32", "neighbors = 32\nneigbors = 10"),
        GAUSSIAN.replace("fit = \"leave-one-out\"", "procedure = \"leave-one-out\""),
        GAUSSIAN.replace("fit = \"leave-one-out\"\n", ""),
        OPERATOR_BO.replace("neighbors = 10", "estimator = \"sub-nn\""),
        GAUSSIAN.replace(
            "proposal = \"independent\"",
            "proposal = \"independent\"\nacquisition = \"upper-confidence-bound\"",
        ),
        OPERATOR_BO.replace(
            "acquisition = \"upper-confidence-bound\"",
            "acquisition = \"thompson\"\nbeta = 1.0",
        ),
        GAUSSIAN.replace("initial = 0.001", "initial = 0.001\nunknown-field = 0.9"),
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
            "objective='cross-entropy'",
            "objective='cross-entropy'\nrounds=0",
        ),
        resident("acquisition='upper-confidence-bound'", 0, 1, "fit='fixed'"),
        resident("acquisition='upper-confidence-bound'", 4, 3, "fit='fixed'"),
        GAUSSIAN.replace("neighbors = 32", "neighbors = 0"),
        GAUSSIAN.replace("epistemic-scale = 0.05", "epistemic-scale = nan"),
        GAUSSIAN.replace("min = 0.0001", "min = 0.2"),
        GAUSSIAN.replace(
            "initial = 0.001",
            "policy = \"reliability\"\ninitial = 0.001",
        ),
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
    assert!(parse_tune(&GAUSSIAN.replace("operator-bo", "turbo-enn")).is_err());
    assert!(parse_tune(&OPERATOR_BO.replace("turbo-enn", "operator-bo")).is_err());
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
        parse_tune(&GAUSSIAN.replace(
            "initial = 0.001",
            "policy = \"reliability\"\ninitial = 0.001"
        ))
        .is_err()
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
        "objective = \"cross-entropy\"",
        "objective = \"bounded-log-likelihood\"",
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
    let text = resident("acquisition='thompson'", 4, 1, "fit='fixed'").replace(
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
    assert!(text.contains("[optimizer]"));
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
        "acquisition='upper-confidence-bound'",
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
        "acquisition='upper-confidence-bound'",
        4,
        1,
        "fit='leave-one-out-neighbor-search-v1'\nfit-candidates=4\nfit-samples=4",
    );
    assert!(parse_tune(&ambiguous).is_err());
}

#[test]
fn neighbor_controls() {
    let fixed = resident("acquisition='upper-confidence-bound'", 4, 1, "fit='fixed'");
    let fixed = parse_tune(&fixed).unwrap().resident_enn(17).unwrap();
    assert!(!fixed.fit_params);
    assert!(!fixed.adapt_neighbors);

    let fitted = resident(
        "acquisition='upper-confidence-bound'",
        4,
        1,
        "fit='leave-one-out'\nfit-candidates=4\nfit-samples=4",
    );
    let fitted = parse_tune(&fitted).unwrap().resident_enn(17).unwrap();
    assert!(fitted.fit_params);
    assert!(!fitted.adapt_neighbors);

    let adaptive = resident(
        "acquisition='upper-confidence-bound'",
        4,
        1,
        "neighbor-rule='risk-prefix'\nfit='fixed'",
    );
    let adaptive = parse_tune(&adaptive).unwrap().resident_enn(17).unwrap();
    assert!(!adaptive.fit_params);
    assert!(adaptive.adapt_neighbors);

    let fitted_adaptive = resident(
        "acquisition='upper-confidence-bound'",
        4,
        1,
        "neighbor-rule='risk-prefix'\nfit='leave-one-out'\nfit-candidates=4\nfit-samples=4",
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
    let base = resident(
        "acquisition='pareto'\nscales=[1.0,2.0]",
        4,
        1,
        "fit='fixed'",
    )
    .replace(
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
    let base = resident("acquisition='thompson'", 1, 1, "").replace(
        "experiment='end-to-end'",
        "experiment='pretrain'\n\
            model='fbt-pisa1-mhc4-v1'\n\
            recipe='stack-v3-rust-go-v1'\n\
            objective='cross-entropy'\n\
            window-tokens=1048576",
    );
    let config = parse_tune(&base).unwrap();
    assert_eq!(config.window_tokens, Some(1_048_576));
    assert!(config.generation.is_none());
    assert!(config.objective_acquisition.is_none());
    assert!(parse_tune(&base.replace("recipe=", "mixture=")).is_err());
    assert!(
        parse_tune(&base.replace(
            "window-tokens=1048576",
            "window-tokens=1048576\nreference='moving-incumbent'"
        ))
        .is_err()
    );
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
    assert!(authored.contains("objective = \"cross-entropy\""));
    assert!(!authored.contains("[generation]"));
    assert!(!authored.contains("max-tokens"));
    assert!(!authored.contains("corpus-prompt-tokens"));
}

#[test]
fn window_budget() {
    let with_window = PRETRAIN_MUON.replace(
        "objective='cross-entropy'\n",
        "objective='cross-entropy'\nwindow-tokens=4096\n",
    );
    let spec = ExperimentSpec::parse(&with_window).unwrap();
    let config = spec.compile().unwrap();
    assert_eq!(config.window_tokens, Some(4096));
    assert!(config.generation.is_none());
    assert_eq!(config.optimizer(), TrainingOptimizer::MuonV1);
    for tokens in [128, 2048, 8192] {
        let invalid = PRETRAIN_MUON.replace(
            "objective='cross-entropy'\n",
            &format!("objective='cross-entropy'\nwindow-tokens={tokens}\n"),
        );
        assert!(ExperimentSpec::parse(&invalid).is_err());
    }
    assert!(ExperimentSpec::parse(&GAUSSIAN.replace("window-tokens = 4096\n", "")).is_err());
    assert!(
        ExperimentSpec::parse(&GAUSSIAN.replace(
            "objective = \"cross-entropy\"",
            "objective = \"cross-entropy\"\nsampling = \"window\""
        ))
        .is_err()
    );
}
