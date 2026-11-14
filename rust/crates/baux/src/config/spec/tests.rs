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
    distribution = \"gaussian\"\n\
    [optimizer.surrogate]\n\
    metric = \"euclidean\"\n\
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
            algorithm='bayesian-search'\n\
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
            "distribution = \"gaussian\"",
            "distribution = \"gaussian\"\ncandidate = 4",
        ),
        GAUSSIAN.replace(
            "distribution = \"gaussian\"",
            "distribution = \"gaussian\"\ncandidates = 4",
        ),
        GAUSSIAN.replace(
            "distribution = \"gaussian\"",
            "distribution = \"gaussian\"\narms = 2",
        ),
        GAUSSIAN.replace("neighbors = 32", "neighbors = 32\nneigbors = 10"),
        GAUSSIAN.replace("fit = \"leave-one-out\"", "procedure = \"leave-one-out\""),
        GAUSSIAN.replace("fit = \"leave-one-out\"\n", ""),
        OPERATOR_BO.replace("neighbors = 10", "estimator = \"sub-nn\""),
        GAUSSIAN.replace(
            "distribution = \"gaussian\"",
            "distribution = \"gaussian\"\nacquisition = \"upper-confidence-bound\"",
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
    assert!(parse_tune(&GAUSSIAN.replace("operator-bo", "bayesian-search")).is_err());
    assert!(parse_tune(&OPERATOR_BO.replace("bayesian-search", "single-state-bo")).is_err());
    assert!(parse_tune(&OPERATOR_BO.replace("bayesian-search", "operator-bo")).is_err());
}

#[test]
fn single_bo() {
    let spec_single =
        ExperimentSpec::parse(&GAUSSIAN.replace("operator-bo", "single-state-bo")).unwrap();
    let spec_legacy = ExperimentSpec::parse(GAUSSIAN).unwrap();
    assert!(matches!(
        spec_single.optimizer,
        OptimizerSpec::SingleStateBo { .. }
    ));
    assert!(matches!(
        spec_legacy.optimizer,
        OptimizerSpec::SingleStateBo { .. }
    ));

    let toml_single = spec_single.to_toml().unwrap();
    assert!(
        toml_single.contains("algorithm = \"single-state-bo\"")
            || toml_single.contains("algorithm='single-state-bo'")
    );

    let toml_legacy = spec_legacy.to_toml().unwrap();
    assert!(
        toml_legacy.contains("algorithm = \"single-state-bo\"")
            || toml_legacy.contains("algorithm='single-state-bo'")
    );
}

#[test]
fn random_acquisition() {
    let text = OPERATOR_BO.replace("upper-confidence-bound", "random");
    let config = parse_tune(&text).unwrap();
    assert_eq!(config.selection, Some(PretrainSelection::Random));
    let spec = ExperimentSpec::from_runtime(&config).unwrap();
    let OptimizerSpec::BayesianSearch { acquisition, .. } = spec.optimizer else {
        panic!("expected Bayesian search");
    };
    assert!(matches!(acquisition, AcquisitionSpec::Random));
}

#[test]
fn regional_recipe() {
    use ndarray::array;
    use rand::{SeedableRng, rngs::StdRng};

    let text = format!(
        "{}[optimizer.regional-search]\ncount=3\nsharing='nearest-center'\nallocation='least-trials'\nfailure-tolerance=4\n",
        resident("acquisition='thompson'", 3, 3, "fit='fixed'")
    );
    let spec = ExperimentSpec::parse(&text).unwrap();
    let regional = spec.optimizer.regional_search().unwrap();
    assert_eq!(regional.count, 3);
    assert_eq!(regional.sharing, ObservationSharing::NearestCenter);
    assert_eq!(regional.allocation, RegionAllocation::LeastTrials);
    assert_eq!(regional.failure_tolerance, 4);

    let resolved = spec.to_toml().unwrap();
    let restored = ExperimentSpec::parse(&resolved).unwrap();
    assert_eq!(resolved, restored.to_toml().unwrap());
    assert_eq!(
        restored.compile().unwrap().regional_search.unwrap().count,
        3
    );

    let mut rng = StdRng::seed_from_u64(41);
    let optimizer = crate::optimizer_factory::spec_optimizer(
        array![[0.0, 1.0], [0.0, 1.0]],
        &spec.optimizer,
        2,
        &mut rng,
    )
    .unwrap();
    let crate::optimizer::TrustRegionState::Multi { state, .. } = optimizer.trust_region() else {
        panic!("regional recipe did not construct regional host search");
    };
    assert_eq!(state.num_regions(), 3);
}

#[test]
fn turbo_radius() {
    let config = parse_tune(GAUSSIAN).unwrap();
    let spec = ExperimentSpec::from_runtime(&config).unwrap();
    let _roundtrip = parse_tune(&spec.to_toml().unwrap()).unwrap();
    assert!(
        parse_tune(&GAUSSIAN.replace(
            "initial = 0.001",
            "policy = \"reliability\"\ninitial = 0.001"
        ))
        .is_err()
    );
}

#[test]
fn hyperbolic_metric() {
    let text = GAUSSIAN.replace(
        "metric = \"euclidean\"",
        "metric = \"hyperbolic\"\nlocal-neighbors = 8",
    );
    let config = parse_tune(&text).unwrap();
    assert_eq!(config.distance_metric, Some(DistanceMetric::Hyperbolic));
    assert_eq!(config.local_scale_neighbors, Some(8));
    let spec = ExperimentSpec::from_runtime(&config).unwrap();
    let roundtrip = parse_tune(&spec.to_toml().unwrap()).unwrap();
    assert_eq!(roundtrip.distance_metric, Some(DistanceMetric::Hyperbolic));
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
            "fbt-pisa-legacy-v1",
            "pisa-residual1-v1",
            "fbt-pisa-projected-boundary-v1",
            "pisa-hc4-v1",
            "pisa-mhc4-v1",
            "pisa-looped-mhc4-v1",
            "pisa-diffusion-mhc4-v1",
            "pisa-hnet-mhc4-v1",
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

#[test]
fn perturbation_spec() {
    let text = format!(
        "{GAUSSIAN}\
        [optimizer.perturbation]\n\
        sampler = \"raasp\"\n\
        num-pert = 12\n\
        raasp-fast = true\n\
        coord-system = \"spherical\"\n"
    );
    let config = parse_tune(&text).unwrap();
    assert_eq!(config.candidate_rv, Some(CandidateSampler::RAASP));
    assert_eq!(config.num_pert, Some(12));
    assert_eq!(config.raasp_fast, Some(true));
    assert_eq!(config.coord_system, Some(crate::CoordSystem::Spherical));

    let spec = ExperimentSpec::from_runtime(&config).unwrap();
    let opt_cfg = OptimizerConfig::from_spec(&spec.optimizer).unwrap();
    assert_eq!(
        opt_cfg.candidates.coord_system,
        crate::CoordSystem::Spherical
    );
    assert_eq!(opt_cfg.candidates.num_pert, 12);
    assert!(opt_cfg.candidates.raasp_fast);

    let roundtrip = parse_tune(&spec.to_toml().unwrap()).unwrap();
    assert_eq!(roundtrip.coord_system, Some(crate::CoordSystem::Spherical));
    assert_eq!(roundtrip.num_pert, Some(12));
    assert_eq!(roundtrip.raasp_fast, Some(true));

    let invalid = text.replace("num-pert = 12", "num-pert = 0");
    assert!(parse_tune(&invalid).is_err());
}

#[test]
fn recipe_fidelity() {
    use crate::semantics::neighbor::DistanceMetric as GeomMetric;
    use crate::surrogate::Surrogate;
    let recipe = SurrogateRecipe {
        neighbors: 12,
        neighbor_rule: NeighborRule::RiskPrefix,
        epistemic_scale: 2.5,
        aleatoric_scale: 0.05,
        y_scale: 3.0,
        metric: DistanceMetric::Hyperbolic,
        local_neighbors: Some(6),
        fit: SurrogateFit::Fixed,
        fit_candidates: None,
        fit_samples: None,
    };
    let SurrogateConfig::ENN(enn) = SurrogateConfig::from_recipe(&recipe).unwrap() else {
        panic!("expected ENN surrogate config");
    };
    assert_eq!(enn.k, 12);
    assert!(enn.adapt_neighbors);
    assert!((enn.epistemic_scale - 2.5).abs() < 1.0e-12);
    assert!((enn.aleatoric_scale - 0.05).abs() < 1.0e-12);
    assert_eq!(enn.y_scale, Some(3.0));
    assert_eq!(enn.geometry, GeomMetric::Hyperbolic { local_k: 6 });

    let mut sur = crate::surrogate::ENNSurrogate::new(enn);
    let x = ndarray::Array2::from_shape_fn((8, 3), |(i, j)| (i as f64 * 0.2) + (j as f64 * 0.1));
    let y = ndarray::Array2::from_shape_fn((8, 1), |(i, _)| (i as f64).sin());
    let mut rng = <rand::rngs::StdRng as rand::SeedableRng>::seed_from_u64(7);
    sur.fit(&x.view(), &y.view(), None, &mut rng).unwrap();
    let pred = sur.predict(&x.view()).unwrap();
    assert_eq!(pred.mu.shape(), &[8, 1]);
    assert!(pred.se.iter().all(|v| v.is_finite() && *v > 0.0));
    let params = sur.params().expect("seeded params");
    assert!((params.epistemic_scale - 2.5).abs() < 1.0e-12);
    assert!((params.aleatoric_scale - 0.05).abs() < 1.0e-12);

    let bad_euclid = SurrogateRecipe {
        metric: DistanceMetric::Euclidean,
        local_neighbors: Some(4),
        ..SurrogateRecipe::default()
    };
    assert!(SurrogateConfig::from_recipe(&bad_euclid).is_err());
}
