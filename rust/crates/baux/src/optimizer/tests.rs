use super::BayesOptimizer;
use crate::config::turbo_zero;
use crate::error::ENNError;
use crate::optimizer::Strategy;
use crate::surrogate::{Surrogate, SurrogatePrediction};
use ndarray::{Array1, Array2, Array3, ArrayView2, array};
use rand::RngCore;
use rand::SeedableRng;
use rand::rngs::StdRng;

struct CoordinateSurrogate;

impl Surrogate for CoordinateSurrogate {
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
            mu: x.to_owned(),
            se: Array2::zeros(x.raw_dim()),
        })
    }

    fn sample(
        &self,
        x: &ArrayView2<f64>,
        num_samples: usize,
        _rng: &mut dyn RngCore,
    ) -> Result<Array3<f64>, ENNError> {
        Ok(Array3::zeros((num_samples, x.nrows(), x.ncols())))
    }

    fn lengthscales(&self) -> Option<Array1<f64>> {
        None
    }
}

#[test]
fn invalid_observation() {
    let bounds = array![[0.0, 1.0], [0.0, 1.0]];
    let mut rng = StdRng::seed_from_u64(42);
    let mut optimizer = BayesOptimizer::new(bounds, turbo_zero(), &mut rng).unwrap();
    let x = array![[0.5, 0.5], [0.3, 0.7], [0.1, 0.9]];
    let y = array![[1.0], [2.0]];

    let error = optimizer
        .add_observations(&x.view(), &y.view())
        .expect_err("mismatched rows must fail");
    assert!(matches!(error, ENNError::InvalidShape { .. }));
    assert_eq!(optimizer.obs_count(), 0);
}

#[test]
fn posterior_natural() {
    let bounds = array![[-2.0, 2.0], [10.0, 20.0]];
    let mut rng = StdRng::seed_from_u64(42);
    let optimizer = BayesOptimizer::new_surrogate(
        bounds,
        turbo_zero(),
        Strategy::turbo(),
        Box::new(CoordinateSurrogate),
        &mut rng,
    )
    .unwrap();

    assert!(optimizer.has_surrogate());
    let mu = optimizer.posterior_mu(&array![[0.0, 15.0]].view()).unwrap();
    assert_eq!(mu, array![[0.5, 0.5]]);
}

#[test]
fn posterior_errors() {
    let bounds = array![[0.0, 1.0], [0.0, 1.0]];
    let mut rng = StdRng::seed_from_u64(42);
    let optimizer = BayesOptimizer::new(bounds, turbo_zero(), &mut rng).unwrap();

    assert!(!optimizer.has_surrogate());
    assert!(matches!(
        optimizer.posterior_mu(&array![[0.5]].view()),
        Err(ENNError::InvalidShape { .. })
    ));
    assert_eq!(
        optimizer.posterior_mu(&array![[0.5, 0.5]].view()),
        Err(ENNError::InvalidParameter("No surrogate".to_string()))
    );
}
