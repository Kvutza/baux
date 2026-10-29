//! Regression tests for Morbo acquisition / incumbent RNG contracts.

use super::{select_thompson, select_ucb};
use crate::config::{AcquisitionConfig, InitStrategy, OptimizerConfig, SurrogateConfig, turbo_enn};
use crate::error::ENNError;
use crate::morbo::{MorboTRSettings, Rescalarize};
use crate::optimizer::BayesOptimizer;
use crate::strategy::Strategy;
use crate::surrogate::{ENNSurrogateConfig, Surrogate, SurrogatePrediction};
use crate::trust_region::{TRLengthConfig, TrustRegionConfig};
use crate::util::argmax_tie;
use ndarray::{Array1, Array2, Array3, ArrayView2, array};
use rand::RngCore;
use rand::SeedableRng;
use rand::rngs::StdRng;

struct TieSurrogate {
    sample_value: f64,
    mu: f64,
    se: f64,
}

struct ArmDraws;

impl Surrogate for ArmDraws {
    fn fit(
        &mut self,
        _x: &ArrayView2<f64>,
        _y: &ArrayView2<f64>,
        _yvar: Option<&ArrayView2<f64>>,
        _rng: &mut dyn RngCore,
    ) -> Result<(), ENNError> {
        Ok(())
    }

    fn predict(&self, x: &ArrayView2<f64>) -> Result<SurrogatePrediction, ENNError> {
        Ok(SurrogatePrediction {
            mu: Array2::zeros((x.nrows(), 1)),
            se: Array2::zeros((x.nrows(), 1)),
        })
    }

    fn sample(
        &self,
        x: &ArrayView2<f64>,
        num_samples: usize,
        _rng: &mut dyn RngCore,
    ) -> Result<Array3<f64>, ENNError> {
        assert_eq!(x.nrows(), 3);
        assert_eq!(num_samples, 2);
        Ok(
            Array3::from_shape_vec((2, 3, 1), vec![3.0, 1.0, 2.0, 0.0, 4.0, 1.0])
                .expect("fixed posterior shape"),
        )
    }

    fn lengthscales(&self) -> Option<Array1<f64>> {
        None
    }
}

impl Surrogate for TieSurrogate {
    fn fit(
        &mut self,
        _x: &ArrayView2<f64>,
        _y: &ArrayView2<f64>,
        _yvar: Option<&ArrayView2<f64>>,
        _rng: &mut dyn RngCore,
    ) -> Result<(), ENNError> {
        Ok(())
    }

    fn predict(&self, x: &ArrayView2<f64>) -> Result<SurrogatePrediction, ENNError> {
        let n = x.nrows();
        let m = 2usize;
        Ok(SurrogatePrediction {
            mu: Array2::from_elem((n, m), self.mu),
            se: Array2::from_elem((n, m), self.se),
        })
    }

    fn sample(
        &self,
        x: &ArrayView2<f64>,
        num_samples: usize,
        _rng: &mut dyn RngCore,
    ) -> Result<Array3<f64>, ENNError> {
        let n = x.nrows();
        let m = 2usize;
        Ok(Array3::from_elem((num_samples, n, m), self.sample_value))
    }

    fn lengthscales(&self) -> Option<Array1<f64>> {
        None
    }
}

fn morbo_ready(seed: u64) -> BayesOptimizer {
    let bounds = array![[0.0, 1.0], [0.0, 1.0]];
    let mut rng = StdRng::seed_from_u64(seed);
    let mut cfg = turbo_enn();
    cfg.trust_region = TrustRegionConfig::Morbo(MorboTRSettings {
        num_metrics: 2,
        alpha: 0.05,
        length: TRLengthConfig::default(),
        rescalarize: Rescalarize::OnRestart,
        noise_aware: false,
    });
    let mut opt = BayesOptimizer::new_strategy(bounds, cfg, Strategy::turbo(), &mut rng).unwrap();
    let y_fit = array![[1.0, 2.0], [2.0, 1.0], [1.5, 1.5]];
    let y_inc = array![2.0, 1.0];
    let morbo = opt.trust_mut().morbo_mut().expect("morbo tr");
    morbo.set_arms(2);
    morbo.update(&y_fit.view(), &y_inc.view()).unwrap();
    opt
}

#[test]
fn thompson_draws() {
    let mut rng = StdRng::seed_from_u64(9);
    let optimizer =
        BayesOptimizer::new_strategy(array![[0.0, 1.0]], turbo_enn(), Strategy::turbo(), &mut rng)
            .unwrap();
    let candidates = array![[0.1], [0.2], [0.3]];
    let selected = select_thompson(&optimizer, &ArmDraws, &candidates.view(), 2, &mut rng).unwrap();
    assert_eq!(selected, array![[0.1], [0.2]]);
}

#[test]
fn morbo_rng2() {
    let opt = morbo_ready(10);
    let x_cand = array![[0.11, 0.21], [0.12, 0.22], [0.13, 0.23], [0.14, 0.24],];
    let sur = TieSurrogate {
        sample_value: 1.0,
        mu: 0.0,
        se: 0.0,
    };
    let mut picks = Vec::new();
    for seed in 0..40u64 {
        let mut rng = StdRng::seed_from_u64(seed);
        let out = select_thompson(&opt, &sur, &x_cand.view(), 2, &mut rng).unwrap();
        picks.push(format!("{:.8},{:.8}", out[[0, 0]], out[[0, 1]]));
    }
    let unique: std::collections::HashSet<String> = picks.into_iter().collect();
    assert!(
        unique.len() > 1,
        "Morbo Thompson should break ties with rng (match Python argmax_tie)"
    );
}

#[test]
fn morbo_rng3() {
    let opt = morbo_ready(11);
    let x_cand = array![
        [0.11, 0.21],
        [0.12, 0.22],
        [0.13, 0.23],
        [0.14, 0.24],
        [0.15, 0.25],
        [0.16, 0.26],
    ];
    let sur = TieSurrogate {
        sample_value: 0.0,
        mu: 1.0,
        se: 0.0,
    };
    let mut picks = Vec::new();
    for seed in 0..40u64 {
        let mut rng = StdRng::seed_from_u64(seed);
        let out = select_ucb(&opt, &sur, &x_cand.view(), 2, 1.0, &mut rng).unwrap();
        picks.push(format!("{:.8},{:.8}", out[[0, 0]], out[[0, 1]]));
    }
    let unique: std::collections::HashSet<String> = picks.into_iter().collect();
    assert!(
        unique.len() > 1,
        "Morbo UCB should shuffle equal scores with rng (match Python permutation+argpartition)"
    );
}

#[test]
fn morbo_tell() {
    let bounds = array![[-1.0, 1.0], [-1.0, 1.0]];
    let mut rng = StdRng::seed_from_u64(14);
    let mut cfg = turbo_enn();
    cfg.acquisition = AcquisitionConfig::Pareto;
    cfg.candidates.min_candidates = 32;
    cfg.candidates.num_candidates_factor = 1.0;
    cfg.candidates.num_candidates_per_arm = Some(32);
    cfg.trust_region = TrustRegionConfig::Morbo(MorboTRSettings {
        num_metrics: 2,
        alpha: 0.05,
        length: TRLengthConfig::default(),
        rescalarize: Rescalarize::OnRestart,
        noise_aware: false,
    });
    let mut opt = BayesOptimizer::new_strategy(
        bounds,
        cfg,
        Strategy::hybrid(InitStrategy::LHD, 8),
        &mut rng,
    )
    .unwrap();
    let y_fit = array![
        [1.0, 0.5],
        [0.8, 0.9],
        [0.2, 0.3],
        [0.4, 0.7],
        [0.6, 0.2],
        [0.3, 0.8],
        [0.7, 0.4],
        [0.5, 0.6],
    ];
    for i in 0..4 {
        let x = opt.ask(2, &mut rng).unwrap();
        let y = y_fit.slice(ndarray::s![i * 2..i * 2 + 2, ..]);
        opt.tell(&x.view(), &y, &mut rng).unwrap();
    }
    let x_arms = opt.ask(2, &mut rng).unwrap();
    assert_eq!(x_arms.nrows(), 2);
    assert!(opt.trust_region().is_morbo());
}

#[test]
fn morbo_turbo() {
    let bounds = array![[0.0, 1.0], [0.0, 1.0]];
    let mut rng = StdRng::seed_from_u64(302);
    let mut cfg = turbo_enn();
    cfg.candidates.num_candidates_factor = 1.0;
    cfg.candidates.num_candidates_per_arm = Some(32);
    cfg.trust_region = TrustRegionConfig::Morbo(MorboTRSettings {
        num_metrics: 2,
        alpha: 0.05,
        length: TRLengthConfig::default(),
        rescalarize: Rescalarize::OnRestart,
        noise_aware: false,
    });
    let mut opt = BayesOptimizer::new_strategy(
        bounds,
        cfg,
        Strategy::hybrid(InitStrategy::LHD, 8),
        &mut rng,
    )
    .unwrap();
    let y_fit = array![
        [1.0, 0.5],
        [0.8, 0.9],
        [0.2, 0.3],
        [0.4, 0.7],
        [0.6, 0.2],
        [0.3, 0.8],
        [0.7, 0.4],
        [0.5, 0.6],
    ];
    for i in 0..4 {
        let x = opt.ask(2, &mut rng).unwrap();
        let y = y_fit.slice(ndarray::s![i * 2..i * 2 + 2, ..]);
        opt.tell(&x.view(), &y, &mut rng).unwrap();
    }
    let w0 = opt
        .trust_region()
        .morbo()
        .expect("morbo tr")
        .weights()
        .to_owned();
    let _ = opt.ask(2, &mut rng).unwrap();
    let w1 = opt
        .trust_region()
        .morbo()
        .expect("morbo tr")
        .weights()
        .to_owned();
    assert_eq!(w0, w1);
    opt.trust_mut().restart(Some(&mut rng));
    let w2 = opt
        .trust_region()
        .morbo()
        .expect("morbo tr")
        .weights()
        .to_owned();
    assert_ne!(w1, w2);
}

#[test]
fn morbo_turbo2() {
    let bounds = array![[0.0, 1.0], [0.0, 1.0]];
    let mut rng = StdRng::seed_from_u64(301);
    let mut cfg = turbo_enn();
    cfg.candidates.num_candidates_factor = 1.0;
    cfg.candidates.num_candidates_per_arm = Some(32);
    cfg.trust_region = TrustRegionConfig::Morbo(MorboTRSettings {
        num_metrics: 2,
        alpha: 0.05,
        length: TRLengthConfig::default(),
        rescalarize: Rescalarize::OnPropose,
        noise_aware: false,
    });
    let mut opt = BayesOptimizer::new_strategy(
        bounds,
        cfg,
        Strategy::hybrid(InitStrategy::LHD, 8),
        &mut rng,
    )
    .unwrap();
    let y_fit = array![
        [1.0, 0.5],
        [0.8, 0.9],
        [0.2, 0.3],
        [0.4, 0.7],
        [0.6, 0.2],
        [0.3, 0.8],
        [0.7, 0.4],
        [0.5, 0.6],
    ];
    for i in 0..4 {
        let x = opt.ask(2, &mut rng).unwrap();
        let y = y_fit.slice(ndarray::s![i * 2..i * 2 + 2, ..]);
        opt.tell(&x.view(), &y, &mut rng).unwrap();
    }
    let w0 = opt
        .trust_region()
        .morbo()
        .expect("morbo tr")
        .weights()
        .to_owned();
    let _ = opt.ask(2, &mut rng).unwrap();
    let w1 = opt
        .trust_region()
        .morbo()
        .expect("morbo tr")
        .weights()
        .to_owned();
    assert_ne!(w0, w1);
}

#[test]
fn surrogate_predicts() {
    let sur = TieSurrogate {
        sample_value: 0.0,
        mu: 0.0,
        se: 1.0,
    };
    let x = array![[0.0, 0.0]];
    let pred = sur.predict(&x.view()).unwrap();
    assert_eq!(pred.mu.nrows(), 1);
}

fn morbo_obs(seed: u64) -> BayesOptimizer {
    let bounds = array![[0.0, 1.0], [0.0, 1.0]];
    let mut rng = StdRng::seed_from_u64(seed);
    let mut cfg = turbo_enn();
    cfg.trust_region = TrustRegionConfig::Morbo(MorboTRSettings {
        num_metrics: 2,
        alpha: 0.05,
        length: TRLengthConfig::default(),
        rescalarize: Rescalarize::OnRestart,
        noise_aware: false,
    });
    let mut opt = BayesOptimizer::new_strategy(bounds, cfg, Strategy::turbo(), &mut rng).unwrap();
    let y_fit = array![[1.0, 2.0], [2.0, 1.0], [1.5, 1.5]];
    let y_inc = array![1.0, 2.0];
    let morbo = opt.trust_mut().morbo_mut().expect("morbo tr");
    morbo.set_arms(2);
    morbo.update(&y_fit.view(), &y_inc.view()).unwrap();
    let x = array![[0.2, 0.3], [0.4, 0.5], [0.6, 0.7]];
    let y = array![[1.0, 2.0], [1.0, 2.0], [1.0, 2.0]];
    let delta = opt.add_observations(&x.view(), &y.view()).unwrap();
    let surrogate = opt.surrogate_mut().expect("enn surrogate");
    surrogate
        .fit_append(&delta.x_view(), &delta.y_view(), None, &mut rng)
        .unwrap();
    opt
}

#[test]
fn morbo_rng() {
    let mut indices = Vec::new();
    for seed in 0..40u64 {
        let mut opt = morbo_obs(seed);
        let mut rng = StdRng::seed_from_u64(seed + 1000);
        opt.update_incumbent(&mut rng).unwrap();
        let x_inc = opt.unit_incumbent().expect("incumbent x");
        indices.push(format!("{:.8},{:.8}", x_inc[0], x_inc[1]));
    }
    let unique: std::collections::HashSet<String> = indices.into_iter().collect();
    assert!(
        unique.len() > 1,
        "Morbo incumbent should break Chebyshev ties with rng"
    );
}

fn argmax_scores(scores: &ndarray::Array1<f64>) -> usize {
    argmax_tie(
        scores.as_slice().unwrap_or(&[]),
        &mut StdRng::seed_from_u64(0),
    )
}

#[test]
fn morbo_selection() {
    let bounds = array![[0.0, 1.0], [0.0, 1.0]];
    let mut rng = StdRng::seed_from_u64(9001);
    let mut cfg = OptimizerConfig {
        surrogate: SurrogateConfig::ENN(ENNSurrogateConfig {
            k: 3,
            num_samples: 6,
            num_candidates: 4,
            ..Default::default()
        }),
        trust_region: TrustRegionConfig::Morbo(MorboTRSettings {
            num_metrics: 2,
            alpha: 0.05,
            length: TRLengthConfig::default(),
            rescalarize: Rescalarize::OnRestart,
            noise_aware: true,
        }),
        ..OptimizerConfig::default()
    };
    cfg.noise_aware = true;

    let mut opt = BayesOptimizer::new_strategy(bounds, cfg, Strategy::turbo(), &mut rng).unwrap();

    let x = array![[0.02, 0.02], [0.98, 0.02], [0.02, 0.98], [0.60, 0.60],];
    let y = array![[2.0, 2.0], [50.0, 1.0], [1.0, 50.0], [3.0, 3.0],];
    opt.tell(&x.view(), &y.view(), &mut rng).unwrap();

    let sur = opt.surrogate().expect("enn surrogate");
    let y_all = opt.y_obs().expect("y observations");
    let x_all = opt.x_obs().expect("x observations");
    let n = y_all.nrows();
    assert_eq!(n, 4);

    let pred = sur.predict(&x_all.view()).expect("predict");
    let mu = pred.mu;

    let scores_on_y = opt
        .trust_region()
        .morbo_scalarize(&y_all.view(), true)
        .expect("scalarize y");
    let scores_on_mu = opt
        .trust_region()
        .morbo_scalarize(&mu.view(), true)
        .expect("scalarize mu");

    let y_winner = argmax_scores(&scores_on_y);
    let mu_winner = argmax_scores(&scores_on_mu);

    assert_ne!(
        y_winner, mu_winner,
        "test setup must make mu and raw-y Morbo rankings diverge (y_winner={y_winner}, \
         mu_winner={mu_winner})"
    );

    let y_inc = opt.y_scalar().expect("incumbent y").to_owned();
    let expected_mu = mu.row(mu_winner).to_owned();
    let raw_y_at_mu_winner = y_all.row(mu_winner).to_owned();

    assert!(
        y_inc
            .iter()
            .zip(expected_mu.iter())
            .all(|(a, b)| (a - b).abs() < 1e-6),
        "noise_aware Morbo must store posterior mu row {:?} used to select incumbent, got {:?}",
        expected_mu.as_slice().unwrap(),
        y_inc.as_slice().unwrap()
    );
    assert!(
        y_inc
            .iter()
            .zip(raw_y_at_mu_winner.iter())
            .any(|(a, b)| (a - b).abs() > 1e-6),
        "stored incumbent must not be raw observed y {:?} when mu differs {:?}",
        raw_y_at_mu_winner.as_slice().unwrap(),
        expected_mu.as_slice().unwrap()
    );
}
