//! Factory functions for creating optimizers with preset configs.

use ndarray::Array2;
use rand::RngCore;

use crate::config::{
    InitStrategy, OptimizerSpec, RunConfig, SurrogateConfig, lhd_only, turbo_enn, turbo_zero,
};
use crate::error::ENNError;
use crate::optimizer::BayesOptimizer;
use crate::strategy::Strategy;

fn validate_k(k: i32) -> Result<(), ENNError> {
    if k < 1 {
        return Err(ENNError::InvalidParameter(format!(
            "k (number of neighbors) must be > 0, got {k}"
        )));
    }
    Ok(())
}

/// Create an optimizer directly from a closed [`OptimizerSpec`].
pub fn spec_optimizer(
    bounds: Array2<f64>,
    spec: &OptimizerSpec,
    num_init: usize,
    rng: &mut dyn RngCore,
) -> Result<BayesOptimizer, ENNError> {
    BayesOptimizer::from_spec(bounds, spec, num_init, rng)
}

/// Create an optimizer for TuRBO-ENN.
pub fn enn_optimizer(
    bounds: Array2<f64>,
    k: i32,
    num_init: usize,
    rng: &mut dyn RngCore,
) -> Result<BayesOptimizer, ENNError> {
    enn_configured(bounds, k, num_init, rng, None)
}

/// Create an experimental multi-trust-region optimizer for TuRBO-ENN.
pub fn enn_tr(
    bounds: Array2<f64>,
    k: i32,
    num_init: usize,
    num_regions: usize,
    rng: &mut dyn RngCore,
) -> Result<BayesOptimizer, ENNError> {
    region_configured(bounds, k, num_init, num_regions, rng, None)
}

/// Create TuRBO-ENN from an optional resolved runtime configuration.
pub fn enn_configured(
    bounds: Array2<f64>,
    k: i32,
    num_init: usize,
    rng: &mut dyn RngCore,
    settings: Option<&RunConfig>,
) -> Result<BayesOptimizer, ENNError> {
    validate_k(k)?;
    let mut config = turbo_enn();
    if let SurrogateConfig::ENN(enn_cfg) = &mut config.surrogate {
        enn_cfg.k = k;
    }
    if let Some(settings) = settings {
        config = settings.apply_to(config);
    }
    let strategy = Strategy::hybrid(InitStrategy::LHD, num_init);
    BayesOptimizer::new_strategy(bounds, config, strategy, rng)
}

/// Create experimental multi-trust-region TuRBO-ENN from resolved settings.
pub fn region_configured(
    bounds: Array2<f64>,
    k: i32,
    num_init: usize,
    num_regions: usize,
    rng: &mut dyn RngCore,
    settings: Option<&RunConfig>,
) -> Result<BayesOptimizer, ENNError> {
    validate_k(k)?;
    let mut config = turbo_enn();
    if let SurrogateConfig::ENN(enn_cfg) = &mut config.surrogate {
        enn_cfg.k = k;
    }
    if let Some(settings) = settings {
        config = settings.apply_to(config);
    }
    let strategy = Strategy::experimental(bounds.nrows(), num_regions, num_init, rng)?;
    BayesOptimizer::new_strategy(bounds, config, strategy, rng)
}

/// Create an optimizer for TuRBO-ZERO.
pub fn create_optimizer(
    bounds: Array2<f64>,
    num_init: usize,
    rng: &mut dyn RngCore,
) -> Result<BayesOptimizer, ENNError> {
    create_configured(bounds, num_init, rng, None)
}

/// Create TuRBO-ZERO from an optional resolved runtime configuration.
pub fn create_configured(
    bounds: Array2<f64>,
    num_init: usize,
    rng: &mut dyn RngCore,
    settings: Option<&RunConfig>,
) -> Result<BayesOptimizer, ENNError> {
    let mut config = turbo_zero();
    if let Some(settings) = settings {
        config = settings.apply_to(config);
    }
    let strategy = Strategy::hybrid(InitStrategy::LHD, num_init);
    BayesOptimizer::new_strategy(bounds, config, strategy, rng)
}

/// Create an optimizer for LHD-only.
pub fn create_lhd(
    bounds: Array2<f64>,
    num_init: usize,
    rng: &mut dyn RngCore,
) -> Result<BayesOptimizer, ENNError> {
    lhd_configured(bounds, num_init, rng, None)
}

/// Create LHD-only from an optional resolved runtime configuration.
pub fn lhd_configured(
    bounds: Array2<f64>,
    num_init: usize,
    rng: &mut dyn RngCore,
    settings: Option<&RunConfig>,
) -> Result<BayesOptimizer, ENNError> {
    let mut config = lhd_only();
    if let Some(settings) = settings {
        config = settings.apply_to(config);
    }
    let strategy = Strategy::init(InitStrategy::LHD, num_init);
    BayesOptimizer::new_strategy(bounds, config, strategy, rng)
}

#[cfg(test)]
mod tests {
    use super::{enn_configured, spec_optimizer};
    use crate::config::{
        AcquisitionSpec, CandidateGenerationSpec, DistanceScaling, HistoryGeometry, OptimizerSpec,
        RunConfig, SurrogateEstimator, SurrogateFit, SurrogateRecipe, TrustRegionBounds,
        TrustRegionSpec,
    };
    use crate::procedural_pool::ProposalMethod;
    use ndarray::{Array2, array};
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    #[derive(Debug, PartialEq)]
    struct Trajectory {
        candidates: Vec<u64>,
        incumbents: Vec<u64>,
        observations: usize,
        restarts: usize,
    }

    fn objective(x: &[f64], round: usize) -> f64 {
        let phase = round as f64 * 0.013;
        -x.iter()
            .enumerate()
            .map(|(axis, value)| {
                let target = 0.2 + 0.15 * axis as f64 + phase;
                (value - target).powi(2)
            })
            .sum::<f64>()
    }

    fn trajectory(seed: u64) -> Trajectory {
        let bounds = array![[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]];
        let overrides = RunConfig {
            noise_aware: Some(true),
            num_candidates_factor: Some(1.0),
            min_candidates: Some(32),
            max_candidates: Some(32),
            ..Default::default()
        };
        let mut rng = StdRng::seed_from_u64(seed);
        let mut optimizer = enn_configured(bounds, 3, 4, &mut rng, Some(&overrides)).unwrap();
        let mut candidates = Vec::new();
        let mut incumbents = Vec::new();
        let mut expected_observations = 0;

        for round in 0..16 {
            let arms = 1 + (round + seed as usize) % 4;
            let generation = optimizer.restart_generation();
            let x = optimizer.ask(arms, &mut rng).unwrap();
            if optimizer.restart_generation() != generation {
                expected_observations = 0;
            }
            assert_eq!(x.shape(), &[arms, 3], "seed={seed} round={round}");
            assert!(
                x.iter()
                    .all(|value| value.is_finite() && (0.0..=1.0).contains(value)),
                "candidate escaped its declared domain: seed={seed} round={round}"
            );
            candidates.extend(x.iter().map(|value| value.to_bits()));

            let y = Array2::from_shape_fn((arms, 1), |(row, _)| {
                objective(x.row(row).as_slice().unwrap(), round)
            });
            optimizer.tell(&x.view(), &y.view(), &mut rng).unwrap();
            expected_observations += arms;

            assert_eq!(optimizer.obs_count(), expected_observations);
            assert_eq!(optimizer.x_obs().unwrap().nrows(), expected_observations);
            assert_eq!(optimizer.y_obs().unwrap().nrows(), expected_observations);
            assert_eq!(
                optimizer.incumbent.observation_count(),
                expected_observations,
                "posterior tracker diverged: seed={seed} round={round}"
            );
            assert!(optimizer.tr_length().is_finite() && optimizer.tr_length() > 0.0);
            let incumbent = optimizer
                .y_scalar()
                .expect("tell must establish an incumbent");
            assert!(incumbent.iter().all(|value| value.is_finite()));
            incumbents.extend(incumbent.iter().map(|value| value.to_bits()));
        }

        Trajectory {
            candidates,
            incumbents,
            observations: optimizer.obs_count(),
            restarts: optimizer.restart_generation(),
        }
    }

    #[test]
    fn trajectory_replay() {
        for seed in [0, 1, 17, 0x5eed] {
            assert_eq!(trajectory(seed), trajectory(seed), "replay seed={seed}");
        }
    }

    #[test]
    fn rejects_badk() {
        let bounds = array![[0.0, 1.0], [0.0, 1.0]];
        for k in [0, -3] {
            let mut rng = StdRng::seed_from_u64(0);
            let error = enn_configured(bounds.clone(), k, 3, &mut rng, None)
                .err()
                .expect("k < 1 must be rejected at construction");
            assert_eq!(
                error.to_string(),
                format!("Invalid parameter: k (number of neighbors) must be > 0, got {k}")
            );
        }
    }

    #[test]
    fn tracker_restarts() {
        let bounds = array![[0.0, 1.0], [0.0, 1.0]];
        let overrides = RunConfig {
            noise_aware: Some(true),
            ..Default::default()
        };
        let mut rng = StdRng::seed_from_u64(57);
        let mut optimizer = enn_configured(bounds, 3, 4, &mut rng, Some(&overrides)).unwrap();

        for round in 0..64 {
            let x = array![[0.1 + 0.01 * round as f64, 0.2]];
            let y = array![[-(round as f64)]];
            optimizer.tell(&x.view(), &y.view(), &mut rng).unwrap();
            assert_eq!(
                optimizer.incumbent.observation_count(),
                optimizer.obs_count(),
                "tracker diverged after round {round}"
            );
        }
        assert_eq!(optimizer.restart_generation(), 0);
        let restart_sample = optimizer.ask(1, &mut rng).unwrap();
        assert_eq!(restart_sample.nrows(), 1);
        assert_eq!(optimizer.restart_generation(), 1);
        assert_eq!(optimizer.obs_count(), 0);
        assert_eq!(optimizer.incumbent.observation_count(), 0);
    }

    #[test]
    fn spec_builds() {
        let bounds = array![[-1.0, 1.0], [-2.0, 2.0]];
        let spec = OptimizerSpec::TurboEnn {
            candidate_generation: CandidateGenerationSpec {
                generator: ProposalMethod::Additive,
                candidates: 16,
                arms: 2,
                perturbation: None,
            },
            surrogate: SurrogateRecipe {
                estimator: SurrogateEstimator::Enn,
                neighbors: 3,
                neighbor_rule: Default::default(),
                epistemic_scale: 0.7,
                aleatoric_scale: 0.05,
                y_scale: 1.0,
                geometry: HistoryGeometry::Realized,
                scaling: DistanceScaling::Global,
                local_neighbors: None,
                fit: SurrogateFit::LeaveOneOut {
                    fit_candidates: 8,
                    fit_samples: 4,
                },
            },
            acquisition: AcquisitionSpec::Ucb { beta: 1.5 },
            trust_region: TrustRegionSpec::Turbo {
                bounds: TrustRegionBounds::default(),
            },
        };
        let mut rng = StdRng::seed_from_u64(99);
        let mut opt = spec_optimizer(bounds, &spec, 2, &mut rng).unwrap();
        let x0 = opt.ask_natural(2, &mut rng).unwrap();
        assert_eq!(x0.shape(), &[2, 2]);
        opt.tell_natural(&x0.view(), &array![[1.0], [2.0]].view(), None, &mut rng)
            .unwrap();
        let x1 = opt.ask_natural(2, &mut rng).unwrap();
        assert_eq!(x1.shape(), &[2, 2]);
        assert!(spec_optimizer(array![[0.0, 1.0]], &OptimizerSpec::Muon, 2, &mut rng).is_err());
    }
}
