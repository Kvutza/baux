use super::*;

fn parse(fields: &str) -> RunConfig {
    baux::artifact::toml::from_str(&format!("[objective_acquisition]\n{fields}")).unwrap()
}

#[test]
fn explicit_vectors() {
    let pareto = parse("mode='pareto'\nscales=[1.0,2.0]");
    assert!(
        pareto
            .objective_acquisition
            .as_ref()
            .unwrap()
            .validate()
            .is_ok()
    );
    assert!(
        pareto
            .validate_objectives()
            .unwrap_err()
            .contains("scalar rewards only")
    );
    assert!(
        parse("mode='pareto'\nscales=[1.0,2.0]\nalpha=0.0")
            .objective_acquisition
            .unwrap()
            .validate()
            .is_err()
    );
    assert!(
        parse("mode='morbo'\nscales=[1.0,2.0]")
            .objective_acquisition
            .unwrap()
            .validate()
            .is_err()
    );
}

#[test]
fn morbo_preferences() {
    let config = parse(
        "mode='morbo'\n\
            scales=[1.0,2.0]\n\
            preferences=[2.0,1.0]\n\
            alpha=0.05\n\
            seed-domain='coding-objectives'\n\
            rescalarize='on-restart'\n\
            clip=true",
    );
    assert!(
        config
            .objective_acquisition
            .as_ref()
            .unwrap()
            .validate()
            .is_ok()
    );
    let first = config.resident_objectives(0).unwrap().unwrap();
    let next = config.resident_objectives(1).unwrap().unwrap();
    let ObjectiveAcquisition::Morbo { seed: a, .. } = first.acquisition else {
        panic!()
    };
    let ObjectiveAcquisition::Morbo { seed: b, .. } = next.acquisition else {
        panic!()
    };
    assert_eq!(a, config.derived_seed(0, "coding-objectives", 0));
    assert_ne!(a, b);
}

#[test]
fn morbo_geometry() {
    let mut config = parse(
        "mode='morbo'\n\
            scales=[1.0,2.0]\n\
            preferences=[2.0,1.0]\n\
            regions=2\n\
            alpha=0.05\n\
            seed-domain='coding-objectives'\n\
            rescalarize='on-restart'\n\
            clip=true",
    );
    config.history_geometry = Some(HistoryGeometry::Realized);
    assert!(
        config
            .validate_objectives()
            .unwrap_err()
            .contains("latent history geometry")
    );
    config.history_geometry = Some(HistoryGeometry::Latent);
    assert!(
        config
            .validate_objectives()
            .unwrap_err()
            .contains("scalar rewards only")
    );
}
