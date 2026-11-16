use super::*;
use crate::semantics::optimizer::surrogate::SurrogateSpec;

fn estimator(rewards: [f32; 2]) -> NeighborValueEstimator {
    let mut estimator = NeighborValueEstimator::with_spec(
        rewards[0],
        0.0,
        17,
        SurrogateSpec {
            neighbors: 2,
            epistemic: 1.0,
            aleatoric: 1.0,
            ..Default::default()
        },
    )
    .unwrap()
    .with_capacity(2)
    .unwrap();
    estimator.append(&[4.0], rewards[1], 0.0, true).unwrap();
    estimator
}

#[test]
fn precision_weights() {
    let estimator = estimator([0.0, 0.0]);
    let mixture = estimator.mixture(&[0.0, 4.0], 0.0).unwrap();
    assert!((mixture.probabilities()[0] - 5.0 / 6.0).abs() < 1.0e-9);
    assert!((mixture.probabilities()[1] - 1.0 / 6.0).abs() < 1.0e-9);
}

#[test]
fn reward_shift() {
    let base = estimator([-2.0, -1.0]).mixture(&[1.0, 1.0], 100.0).unwrap();
    let shifted = estimator([98.0, 99.0]).mixture(&[1.0, 1.0], 100.0).unwrap();
    assert!(base.probabilities()[1] > 0.999);
    for (left, right) in base.probabilities().iter().zip(shifted.probabilities()) {
        assert!((left - right).abs() < 1.0e-20);
    }
    let extreme = estimator([-2.0, -1.0])
        .mixture(&[1.0, 1.0], 1.0e300)
        .unwrap();
    assert_eq!(extreme.probabilities(), &[0.0, 1.0]);
}

#[test]
fn archive_limit() {
    let mut estimator = estimator([-2.0, -1.0]);
    assert_eq!(estimator.next_slot().unwrap(), 0);
    assert_eq!(estimator.append(&[9.0, 1.0], -3.0, 0.0, true).unwrap(), 0);
    assert_eq!(estimator.len(), 2);
    assert_eq!(estimator.base_slot(), 0);
    assert_eq!(estimator.next_slot().unwrap(), 1);
    assert_eq!(estimator.identities(), &[3, 2]);
    assert_eq!(estimator.distance_row(0).unwrap(), &[0.0, 1.0]);
    assert!(estimator.mixture(&[0.0, 1.0], 1.0).is_ok());
    assert!(estimator.with_capacity(1).is_err());
}

#[test]
fn categorical_draw() {
    let mixture = estimator([0.0, 0.0]).mixture(&[0.0, 4.0], 0.0).unwrap();
    let mut first = 0;
    for seed in 0..10_000 {
        let anchor = mixture.sample(seed);
        assert_eq!(anchor, mixture.sample(seed));
        assert_eq!(anchor.probability, mixture.probabilities()[anchor.slot]);
        first += usize::from(anchor.slot == 0);
    }
    assert!(
        (8_100..8_550).contains(&first),
        "first-anchor count: {first}"
    );
    assert!(
        estimator([0.0, 0.0])
            .mixture(&[f64::NAN, 0.0], 1.0)
            .is_err()
    );
    assert!(estimator([0.0, 0.0]).mixture(&[0.0], 1.0).is_err());
    assert!(estimator([0.0, 0.0]).mixture(&[0.0, 0.0], -1.0).is_err());
}

#[test]
fn fitted_kernel() {
    let mut estimator = NeighborValueEstimator::with_spec(
        0.0,
        0.01,
        73,
        SurrogateSpec {
            neighbors: 3,
            epistemic: 100.0,
            aleatoric: 100.0,
            fit: true,
            fit_candidates: 8,
            fit_samples: 3,
            ..Default::default()
        },
    )
    .unwrap()
    .with_capacity(3)
    .unwrap();
    estimator.append(&[1.0], 1.0, 0.01, true).unwrap();
    estimator.append(&[4.0, 1.0], -1.0, 0.01, true).unwrap();
    let fit = estimator.fit_state();
    assert_ne!((fit.epistemic, fit.aleatoric), (100.0, 100.0));
    let mixture = estimator.mixture(&[0.25, 0.25, 2.25], 0.0).unwrap();
    let noise = fit.aleatoric + f64::from(0.01_f32) / (fit.scale * fit.scale).max(1.0e-12);
    let ratio = (1.0e-9 + fit.epistemic * 2.25 + noise) / (1.0e-9 + fit.epistemic * 0.25 + noise);
    assert!((mixture.probabilities()[0] / mixture.probabilities()[2] - ratio).abs() < 1.0e-12);
}

#[test]
fn adaptive_support() {
    let mut estimator = estimator([0.0, 1.0]);
    estimator.adapt_neighbors = true;
    let distances = [0.01, 10.0];
    let count = estimator.acquire(&distances, 0.0, None).unwrap().neighbors;
    let mixture = estimator.mixture(&distances, 1.0).unwrap();
    assert_eq!(
        mixture
            .probabilities()
            .iter()
            .filter(|&&weight| weight > 0.0)
            .count(),
        count
    );
}

#[test]
fn shared_anchor() {
    let mixture = Mixture {
        probabilities: vec![0.5, 0.5],
        means: vec![1.0, 1.0],
    };
    let archive = [&[-5.0, -10.0][..], &[5.0, 10.0][..]];
    let mutation = Mutation::new(1.0, 0.0, 0.0).unwrap();
    for seed in 0..128 {
        let (anchor, candidate) = mixture.draw(&[0.0, 0.0], &archive, mutation, seed).unwrap();
        assert_eq!(candidate, archive[anchor.slot]);
        assert_eq!(candidate[1], 2.0 * candidate[0]);
    }
}

#[test]
fn coordinate_noise() {
    let mixture = Mixture {
        probabilities: vec![1.0],
        means: vec![0.0],
    };
    let current = vec![0.0; 1025];
    let archive = [current.as_slice()];
    let mutation = Mutation::new(0.25, 0.5, 0.01).unwrap();
    let (_, candidate) = mixture.draw(&current, &archive, mutation, 73).unwrap();
    assert_eq!(candidate.len(), current.len());
    assert!(candidate.iter().all(|&value| value != 0.0));
    assert!(candidate.windows(2).all(|pair| pair[0] != pair[1]));
    assert_eq!(
        candidate,
        mixture.draw(&current, &archive, mutation, 73).unwrap().1
    );
    assert_ne!(
        candidate,
        mixture.draw(&current, &archive, mutation, 74).unwrap().1
    );
    assert!(current.iter().all(|&value| value == 0.0));
    assert!(mixture.draw(&current, &[], mutation, 73).is_err());
    assert!(Mutation::new(f32::NAN, 0.5, 0.01).is_err());
    assert!(Mutation::new(0.25, -0.5, 0.01).is_err());
}
