use ndarray::{Array2, array, s};
use rand::{SeedableRng, rngs::StdRng};

use baux::optimizer::{
    BayesOptimizer, MultiTrustRegionConfig, MultiTrustRegionState, SharingPolicy, Strategy,
    TrustRegionState,
};
use baux::trust_region::TRLengthConfig;

fn optimizer(bounds: Array2<f64>, rng: &mut StdRng) -> BayesOptimizer {
    let strategy = Strategy::experimental(bounds.nrows(), 2, 0, rng).unwrap();
    BayesOptimizer::new_strategy(bounds, baux::turbo_zero(), strategy, rng).unwrap()
}

#[path = "provenance/overlap.rs"]
mod overlap;

#[test]
fn cancelled_evaluations() {
    let mut rng = StdRng::seed_from_u64(53);
    let opt = optimizer(array![[0.0, 1.0], [0.0, 1.0]], &mut rng);
    let (x, opt) = opt.ask(3, &mut rng).unwrap();
    let ids = opt.pending_ids().unwrap();
    let opt = opt
        .complete_ids(&ids[2..], &array![[1.0]].view(), None, &mut rng)
        .unwrap();
    assert!(opt.clone().cancel_ids(&[ids[0], ids[0]], &mut rng).is_err());
    assert_eq!(opt.pending_ids().unwrap(), ids[..2]);
    let opt = opt.cancel_ids(&ids[..2], &mut rng).unwrap();
    assert!(opt.pending_ids().is_none());
    assert_eq!(opt.obs_count(), 1);
    if let TrustRegionState::Multi { state, .. } = opt.trust_region() {
        assert_eq!(state.trial_counts().iter().sum::<u64>(), 1);
    }
    assert!(opt.ask(1, &mut rng).is_ok());
}

#[test]
fn cancelled_warmup() {
    let mut rng = StdRng::seed_from_u64(59);
    let bounds = array![[0.0, 1.0], [0.0, 1.0]];
    let strategy = Strategy::experimental(bounds.nrows(), 2, 1, &mut rng).unwrap();
    let opt = BayesOptimizer::new_strategy(bounds, baux::turbo_zero(), strategy, &mut rng).unwrap();
    let (x, opt) = opt.ask(1, &mut rng).unwrap();
    let first = opt.pending_ids().unwrap();
    let opt = opt.cancel_ids(&first, &mut rng).unwrap();
    assert_eq!(opt.obs_count(), 0);
    let (x, opt) = opt.ask(1, &mut rng).unwrap();
    let second = opt.pending_ids().unwrap();
    assert_ne!(first, second);
    let opt = opt
        .complete_ids(&second, &array![[1.0]].view(), None, &mut rng)
        .unwrap();
    assert!(opt.ask(1, &mut rng).is_ok());
}

#[test]
fn natural_roundtrip() {
    let mut rng = StdRng::seed_from_u64(19);
    let opt = optimizer(array![[-2.7, 3.1], [100.1, 101.3]], &mut rng);
    let (x, opt) = opt.ask_natural(2, &mut rng).unwrap();
    let opt = opt
        .tell_natural(&x.view(), &array![[1.0], [2.0]].view(), None, &mut rng)
        .unwrap();
    assert_eq!(opt.obs_count(), 2);
}

#[test]
fn natural_overlap() {
    let mut rng = StdRng::seed_from_u64(79);
    let opt = optimizer(array![[-2.7, 3.1], [100.1, 101.3]], &mut rng);
    let (x, opt) = opt.ask_natural(2, &mut rng).unwrap();
    let first_ids = opt.pending_ids().unwrap();
    let opt = opt
        .complete_ids(&first_ids[..1], &array![[1.0]].view(), None, &mut rng)
        .unwrap();
    let (_x, opt) = opt.ask_natural(1, &mut rng).unwrap();
    let opt = opt
        .complete_ids(&first_ids[1..], &array![[2.0]].view(), None, &mut rng)
        .unwrap();
    let observed = opt.x_obs().unwrap();
    for dimension in 0..2 {
        assert!((observed[[1, dimension]] - x[[1, dimension]]).abs() < 1e-12);
    }
}

#[test]
fn modeled_completion() {
    let mut rng = StdRng::seed_from_u64(19);
    let opt = baux::optimizer_factory::region_configured(
        array![[0.0, 1.0], [0.0, 1.0]],
        1,
        0,
        2,
        &mut rng,
        None,
    )
    .unwrap();
    let opt = opt
        .import_variance(
            &array![[0.2, 0.2], [0.8, 0.8]].view(),
            &array![[1.0], [2.0]].view(),
            None,
            &mut rng,
        )
        .unwrap();
    let (x, opt) = opt.ask_natural(2, &mut rng).unwrap();
    let y = array![[3.0], [4.0]];
    assert!(
        opt.clone()
            .tell_natural(&x.slice(s![..;-1, ..]), &y.view(), None, &mut rng)
            .is_err()
    );
    assert_eq!(opt.obs_count(), 2);
    let opt = opt
        .tell_natural(&x.view(), &y.view(), None, &mut rng)
        .unwrap();
    assert_eq!(opt.obs_count(), 4);
    assert!(opt.ask_natural(1, &mut rng).is_ok());
}

#[test]
fn imported_observations() {
    let mut rng = StdRng::seed_from_u64(7);
    let opt = optimizer(array![[0.0, 1.0], [0.0, 1.0]], &mut rng);
    let opt = opt
        .import_variance(
            &array![[0.5, 0.5]].view(),
            &array![[1.0]].view(),
            None,
            &mut rng,
        )
        .unwrap();
    assert_eq!(opt.obs_count(), 1);
    if let TrustRegionState::Multi { state, pending } = opt.trust_region() {
        assert_eq!(state.trial_counts(), vec![0, 0]);
        assert_eq!(state.fail_counts, vec![0, 0]);
        assert_eq!(state.succ_counts, vec![0, 0]);
        assert!(pending.is_none());
    } else {
        panic!("expected multi-region state");
    }
    assert!(opt.ask(2, &mut rng).is_ok());
}

#[test]
fn explicit_import() {
    let mut rng = StdRng::seed_from_u64(7);
    let opt = optimizer(array![[0.0, 1.0], [0.0, 1.0]], &mut rng);
    let (x, opt) = opt.ask(2, &mut rng).unwrap();
    let opt = opt
        .import_variance(
            &array![[0.5, 0.5]].view(),
            &array![[1.0]].view(),
            None,
            &mut rng,
        )
        .unwrap();
    assert_eq!(opt.obs_count(), 1);
    let opt = opt
        .tell(&x.view(), &array![[2.0], [3.0]].view(), None, &mut rng)
        .unwrap();
    assert_eq!(opt.obs_count(), 3);
    let duplicate = opt
        .clone()
        .tell(&x.view(), &array![[2.0], [3.0]].view(), None, &mut rng);
    assert!(
        duplicate
            .unwrap_err()
            .to_string()
            .contains("explicit import")
    );
    assert_eq!(opt.obs_count(), 3);
}

#[test]
fn invalid_observations() {
    let mut rng = StdRng::seed_from_u64(7);
    let opt = optimizer(array![[0.0, 1.0], [0.0, 1.0]], &mut rng);
    let (x, opt) = opt.ask(2, &mut rng).unwrap();
    let y = array![[1.0], [2.0]];
    assert!(
        opt.clone()
            .import_variance(
                &array![[0.5, 0.5]].view(),
                &array![[1.0, f64::NAN]].view(),
                None,
                &mut rng,
            )
            .unwrap_err()
            .to_string()
            .contains("finite")
    );
    assert_eq!(opt.obs_count(), 0);
    for variance in [
        array![[0.1]],
        array![[-0.1], [0.1]],
        array![[f64::NAN], [0.1]],
    ] {
        assert!(
            opt.clone()
                .tell_variance(&x.view(), &y.view(), Some(&variance.view()), &mut rng)
                .is_err()
        );
        assert_eq!(opt.obs_count(), 0);
    }
    assert!(
        opt.clone()
            .tell(&x.view(), &array![[f64::NAN], [2.0]].view(), None, &mut rng)
            .is_err()
    );
    let opt = opt.tell(&x.view(), &y.view(), None, &mut rng).unwrap();
    assert_eq!(opt.obs_count(), 2);
}

#[test]
fn stale_generation() {
    let mut rng = StdRng::seed_from_u64(7);
    let opt = optimizer(array![[0.0, 1.0], [0.0, 1.0]], &mut rng);
    let (x, mut opt) = opt.ask(2, &mut rng).unwrap();
    if let TrustRegionState::Multi { state, .. } = opt.trust_mut() {
        state.restart_region(0, &array![0.5, 0.5].view()).unwrap();
    }
    let error = opt
        .clone()
        .tell(&x.view(), &array![[1.0], [2.0]].view(), None, &mut rng)
        .unwrap_err();
    assert!(error.to_string().contains("stale region"));
    assert_eq!(opt.obs_count(), 0);
    assert!(matches!(
        opt.trust_region(),
        TrustRegionState::Multi {
            pending: Some(_),
            ..
        }
    ));
}

#[test]
fn abandon_restart() {
    let mut rng = StdRng::seed_from_u64(89);
    let opt = optimizer(array![[0.0, 1.0], [0.0, 1.0]], &mut rng);
    let (x, opt) = opt.ask(3, &mut rng).unwrap();
    let ids = opt.pending_ids().unwrap();
    let opt = opt
        .complete_ids(&ids[..1], &array![[1.0]].view(), None, &mut rng)
        .unwrap();
    let old_generation = match opt.trust_region() {
        TrustRegionState::Multi { state, .. } => state.generation(0).unwrap(),
        _ => panic!("expected multi-region state"),
    };
    let (abandoned, opt) = opt
        .restart_region(0, &array![0.4, 0.4].view(), &mut rng)
        .unwrap();
    assert_eq!(abandoned, ids[1..2]);
    assert_eq!(opt.pending_ids().unwrap(), ids[2..]);
    assert!(
        opt.clone()
            .complete_ids(&abandoned, &array![[2.0]].view(), None, &mut rng)
            .is_err()
    );
    if let TrustRegionState::Multi { state, .. } = opt.trust_region() {
        assert_eq!(state.generation(0), Some(old_generation + 1));
        assert_eq!(state.trial_counts()[0], 0);
    }
    let opt = opt
        .import_variance(
            &x.slice(s![1..2, ..]),
            &array![[2.0]].view(),
            None,
            &mut rng,
        )
        .unwrap();
    if let TrustRegionState::Multi { state, .. } = opt.trust_region() {
        assert_eq!(state.trial_counts()[0], 0);
    }
    let opt = opt
        .complete_ids(&ids[2..], &array![[3.0]].view(), None, &mut rng)
        .unwrap();
    assert!(opt.pending_ids().is_none());
}
