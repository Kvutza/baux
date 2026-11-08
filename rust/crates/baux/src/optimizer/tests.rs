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

#[test]
fn fork_transparency() {
    let bounds = array![[-2.0, 2.0], [-2.0, 2.0]];
    let mut rng = StdRng::seed_from_u64(42);
    let mut branch_a = BayesOptimizer::new(bounds, turbo_zero(), &mut rng).unwrap();
    let x0 = branch_a.ask_natural(4, &mut rng).unwrap();
    let y0 = array![[1.0], [2.0], [1.5], [0.5]];
    branch_a
        .tell_natural(&x0.view(), &y0.view(), None, &mut rng)
        .unwrap();

    let mut branch_b = branch_a.fork().unwrap();
    assert_eq!(branch_a.digest(), branch_b.digest());
    assert_eq!(branch_a.digest_hex().len(), 64);

    let key = crate::hash::Key::new(777);
    let mut rng_a = key.prng();
    let mut rng_b = key.prng();
    let prop_a = branch_a.ask_ticket(2, &mut rng_a).unwrap();
    let prop_b = branch_b.ask_ticket(2, &mut rng_b).unwrap();
    assert_eq!(prop_a.as_array(), prop_b.as_array());
    assert_eq!(branch_a.digest(), branch_b.digest());
    assert_eq!(prop_a.origin_digest(), Some(branch_a.digest()));
    let _ = branch_a.cancel_ticket(prop_a, &mut rng_a);
    let _ = branch_b.cancel_ticket(prop_b, &mut rng_b);
}

#[test]
fn ticket_rejection() {
    let bounds = array![[0.0, 1.0], [0.0, 1.0]];
    let mut rng = StdRng::seed_from_u64(99);
    let mut branch_a = BayesOptimizer::new(bounds, turbo_zero(), &mut rng).unwrap();
    let mut branch_b = branch_a.fork().unwrap();

    let stale = branch_a
        .ask_ticket(2, &mut crate::hash::Key::new(1).prng())
        .unwrap();
    let fresh = branch_a
        .ask_ticket(2, &mut crate::hash::Key::new(2).prng())
        .unwrap();
    let y = array![[1.0], [2.0]];
    branch_a
        .tell_ticket(fresh, &y.view(), None, &mut rng)
        .unwrap();

    let err_stale = branch_a
        .tell_ticket(stale, &y.view(), None, &mut rng)
        .expect_err("stale ticket must be rejected");
    assert!(err_stale.to_string().contains("digest mismatch"));

    let ticket_b = branch_b
        .ask_ticket(2, &mut crate::hash::Key::new(3).prng())
        .unwrap();
    let err_cross = branch_a
        .tell_ticket(ticket_b, &y.view(), None, &mut rng)
        .expect_err("cross-branch ticket must be rejected");
    assert!(err_cross.to_string().contains("digest mismatch"));
}

#[test]
fn freeze_checkpoint() {
    let bounds = array![[-1.0, 1.0], [0.0, 2.0]];
    let mut rng = StdRng::seed_from_u64(17);
    let mut opt = BayesOptimizer::new(bounds, turbo_zero(), &mut rng).unwrap();
    let x = opt.ask_natural(3, &mut rng).unwrap();
    let y = array![[0.2], [0.8], [0.5]];
    opt.tell_natural(&x.view(), &y.view(), None, &mut rng)
        .unwrap();

    let mut frozen = opt.freeze().unwrap();
    let thawed = opt.thaw(&frozen).unwrap();
    assert_eq!(thawed.obs_count(), opt.obs_count());
    assert_eq!(thawed.digest(), opt.digest());

    frozen[16] ^= 0xFF;
    assert!(opt.thaw(&frozen).is_err());
}
