use super::*;

#[test]
fn automatic_restart() {
    use baux::optimizer::{InitStrategyState, StrategyState};

    let mut rng = StdRng::seed_from_u64(97);
    let mut config = MultiTrustRegionConfig::new(1, TRLengthConfig::new(0.8, 0.5, 1.6));
    config.fail_tolerance = 1;
    let state = MultiTrustRegionState::new(1, config, None, &mut rng).unwrap();
    let strategy = Strategy::Experimental(StrategyState {
        init: InitStrategyState::new(baux::InitStrategy::LHD, 0),
        multi_tr: state,
        in_init: false,
        pending_proposal: None,
    });
    let mut opt =
        BayesOptimizer::new_strategy(array![[0.0, 1.0]], baux::turbo_enn(), strategy, &mut rng)
            .unwrap();
    for reward in [10.0, 0.0] {
        opt.ask(1, &mut rng).unwrap();
        let ids = opt.pending_ids().unwrap();
        opt.complete_ids(&ids, &array![[reward]].view(), None, &mut rng)
            .unwrap();
    }
    let TrustRegionState::Multi { state, .. } = opt.trust_region() else {
        panic!("expected multi-region state");
    };
    assert_eq!(state.generation(0), Some(1));
    assert!(state.active_mask[0]);
    assert_eq!(state.trial_counts()[0], 0);
    assert!(state.centers[[0, 0]].is_finite());
    assert_eq!(opt.model_counts(), Some(vec![0]));
    assert!(opt.ask(1, &mut rng).is_ok());
}

#[test]
fn automatic_retention() {
    use baux::optimizer::{InitStrategyState, RestartMemoryPolicy, StrategyState};

    let mut rng = StdRng::seed_from_u64(101);
    let mut config = MultiTrustRegionConfig::new(1, TRLengthConfig::new(0.8, 0.5, 1.6));
    config.fail_tolerance = 1;
    config.restart_memory_policy = RestartMemoryPolicy::RetainRelevant;
    let state = MultiTrustRegionState::new(1, config, None, &mut rng).unwrap();
    let strategy = Strategy::Experimental(StrategyState {
        init: InitStrategyState::new(baux::InitStrategy::LHD, 0),
        multi_tr: state,
        in_init: false,
        pending_proposal: None,
    });
    let mut opt =
        BayesOptimizer::new_strategy(array![[0.0, 1.0]], baux::turbo_enn(), strategy, &mut rng)
            .unwrap();
    for reward in [10.0, 0.0] {
        opt.ask(1, &mut rng).unwrap();
        let ids = opt.pending_ids().unwrap();
        opt.complete_ids(&ids, &array![[reward]].view(), None, &mut rng)
            .unwrap();
    }
    assert_eq!(opt.model_counts(), Some(vec![2]));
    assert!(
        matches!(opt.trust_region(), TrustRegionState::Multi { state, .. } if state.generation(0) == Some(1))
    );
}

#[test]
fn restartfailure() {
    use baux::config::{OptimizerConfig, SurrogateConfig};
    use baux::error::ENNError;
    use baux::optimizer::{AllocationPolicy, RegionalModelFactory, StrategyState};
    use baux::surrogate::{BoxedSurrogate, ENNSurrogate, ENNSurrogateConfig};

    let mut rng = StdRng::seed_from_u64(103);
    let mut config = MultiTrustRegionConfig::new(2, TRLengthConfig::new(0.8, 0.5, 1.6));
    config.allocation_policy = AllocationPolicy::LeastTrials;
    config.fail_tolerance = 1;
    let strategy =
        Strategy::Experimental(StrategyState::configured(1, config, 0, &mut rng).unwrap());
    let model_config = ENNSurrogateConfig::default();
    let local_config = model_config.clone();
    let factory: RegionalModelFactory = Box::new(move |_, generation| {
        if generation > 0 {
            return Err(ENNError::InvalidParameter("restart factory failed".into()));
        }
        Ok(Box::new(ENNSurrogate::new(local_config.clone())) as BoxedSurrogate)
    });
    let mut opt = BayesOptimizer::new_regional(
        array![[0.0, 1.0]],
        OptimizerConfig {
            surrogate: SurrogateConfig::None,
            ..Default::default()
        },
        strategy,
        Box::new(ENNSurrogate::new(model_config)),
        factory,
        &mut rng,
    )
    .unwrap();
    opt.ask(2, &mut rng).unwrap();
    let first = opt.pending_ids().unwrap();
    opt.complete_ids(&first[..1], &array![[10.0]].view(), None, &mut rng)
        .unwrap();
    opt.ask(1, &mut rng).unwrap();
    let second = *opt.pending_ids().unwrap().last().unwrap();
    opt.complete_ids(&[second], &array![[0.0]].view(), None, &mut rng)
        .unwrap();
    assert_eq!(opt.pending_ids(), Some(first[1..].to_vec()));
    assert!(
        opt.ask(1, &mut rng)
            .unwrap_err()
            .to_string()
            .contains("restart factory failed")
    );
    assert!(opt.ingestion_failure().is_some());
    assert!(
        opt.ask(1, &mut rng)
            .unwrap_err()
            .to_string()
            .contains("ingestion failed")
    );
}

#[test]
fn pending_overwrite() {
    let mut rng = StdRng::seed_from_u64(7);
    let mut opt = optimizer(array![[0.0, 1.0], [0.0, 1.0]], &mut rng);
    let proposal = opt.ask(2, &mut rng).unwrap();
    assert_eq!(proposal.shape(), &[2, 2]);
    let error = opt.ask(1, &mut rng).unwrap_err().to_string();
    assert!(error.contains("all regional slots have outstanding evaluations"));
    opt.tell(&proposal.view(), &array![[1.0], [2.0]].view(), &mut rng)
        .unwrap();
    assert!(opt.ask(1, &mut rng).is_ok());
}

#[test]
fn overlapping_asks() {
    let mut rng = StdRng::seed_from_u64(71);
    let mut opt = optimizer(array![[0.0, 1.0], [0.0, 1.0]], &mut rng);
    opt.ask(2, &mut rng).unwrap();
    let first = opt.pending_ids().unwrap();
    opt.complete_ids(&first[..1], &array![[1.0]].view(), None, &mut rng)
        .unwrap();
    assert_eq!(opt.pending_ids().unwrap(), first[1..]);
    opt.ask(1, &mut rng).unwrap();
    let pending = opt.pending_ids().unwrap();
    assert_eq!(pending.len(), 2);
    let ordered_error = opt
        .tell(&array![[0.5, 0.5]].view(), &array![[8.0]].view(), &mut rng)
        .unwrap_err();
    assert!(
        ordered_error
            .to_string()
            .contains("completion by evaluation ID")
    );
    assert_eq!(pending[0], first[1]);
    assert_ne!(pending[1], first[0]);
    assert_ne!(pending[1], first[1]);
    opt.complete_ids(&pending[1..], &array![[2.0]].view(), None, &mut rng)
        .unwrap();
    if let TrustRegionState::Multi { state, .. } = opt.trust_region() {
        assert_eq!(state.trial_counts(), vec![2, 0]);
    } else {
        panic!("expected multi-region state");
    }
    opt.ask(1, &mut rng).unwrap();
    let pending = opt.pending_ids().unwrap();
    assert_eq!(pending.len(), 2);
    opt.complete_ids(&pending[1..], &array![[4.0]].view(), None, &mut rng)
        .unwrap();
    opt.complete_ids(&pending[..1], &array![[3.0]].view(), None, &mut rng)
        .unwrap();
    assert!(opt.pending_ids().is_none());
    if let TrustRegionState::Multi { state, .. } = opt.trust_region() {
        assert_eq!(state.trial_counts(), vec![3, 1]);
    } else {
        panic!("expected multi-region state");
    }
}

#[test]
fn arrival_order() {
    let mut rng = StdRng::seed_from_u64(83);
    let centers = array![[0.5], [0.5]];
    let mut config = MultiTrustRegionConfig::new(2, TRLengthConfig::default());
    config.sharing_policy = SharingPolicy::BoxContain;
    let mut first_high =
        MultiTrustRegionState::new(1, config.clone(), Some(&centers.view()), &mut rng).unwrap();
    let mut first_low =
        MultiTrustRegionState::new(1, config, Some(&centers.view()), &mut rng).unwrap();
    let x = array![[0.5]];
    first_high
        .tell(&x.view(), &array![10.0].view(), Some(&[0]))
        .unwrap();
    first_high
        .tell(&x.view(), &array![5.0].view(), Some(&[1]))
        .unwrap();
    first_low
        .tell(&x.view(), &array![5.0].view(), Some(&[1]))
        .unwrap();
    first_low
        .tell(&x.view(), &array![10.0].view(), Some(&[0]))
        .unwrap();
    assert_eq!(first_high.trial_counts(), vec![1, 1]);
    assert_eq!(first_low.trial_counts(), vec![1, 1]);
    assert_eq!(first_high.incumbents(), first_low.incumbents());
    assert_eq!(first_high.succ_counts, vec![1, 0]);
    assert_eq!(first_low.succ_counts, vec![1, 1]);
}

#[test]
fn ordered_completion() {
    let mut rng = StdRng::seed_from_u64(7);
    let mut opt = optimizer(array![[0.0, 1.0], [0.0, 1.0]], &mut rng);
    let x = opt.ask(2, &mut rng).unwrap();
    let y = array![[1.0], [2.0]];
    let reversed = x.slice(s![..;-1, ..]);
    let error = opt
        .tell_variance(&reversed, &y.view(), Some(&y.view()), &mut rng)
        .unwrap_err();
    assert!(error.to_string().contains("original order"));
    assert_eq!(opt.obs_count(), 0);
    let mut unrelated = x.clone();
    unrelated[[0, 0]] += 0.01;
    assert!(opt.tell(&unrelated.view(), &y.view(), &mut rng).is_err());
    assert!(
        opt.tell(&x.slice(s![..1, ..]), &y.slice(s![..1, ..]), &mut rng)
            .is_err()
    );
    opt.tell(&x.view(), &y.view(), &mut rng).unwrap();
    assert_eq!(opt.obs_count(), 2);
    if let TrustRegionState::Multi { state, pending } = opt.trust_region() {
        assert!(pending.is_none());
        assert_eq!(state.trial_counts().iter().sum::<u64>(), 2);
    } else {
        panic!("expected multi-region state");
    }
}

#[test]
fn partial_completion() {
    let mut rng = StdRng::seed_from_u64(29);
    let mut opt = optimizer(array![[0.0, 1.0], [0.0, 1.0]], &mut rng);
    opt.ask(3, &mut rng).unwrap();
    let ids = opt.pending_ids().unwrap();
    assert_eq!(ids.len(), 3);

    opt.complete_ids(&ids[2..], &array![[3.0]].view(), None, &mut rng)
        .unwrap();
    assert_eq!(opt.obs_count(), 1);
    assert_eq!(opt.pending_ids().unwrap(), ids[..2]);
    assert!(
        opt.complete_ids(&ids[2..], &array![[3.0]].view(), None, &mut rng)
            .unwrap_err()
            .to_string()
            .contains("duplicate")
    );
    opt.ask(1, &mut rng).unwrap();
    let extra = *opt.pending_ids().unwrap().last().unwrap();
    opt.cancel_ids(&[extra], &mut rng).unwrap();
    if let TrustRegionState::Multi { state, .. } = opt.trust_region() {
        // The second region has finished its own issued group. Its controller
        // need not wait for the first region's outstanding rows.
        assert_eq!(state.trial_counts().iter().sum::<u64>(), 1);
    }

    opt.complete_ids(&ids[..2], &array![[1.0], [2.0]].view(), None, &mut rng)
        .unwrap();
    assert_eq!(opt.obs_count(), 3);
    assert!(opt.pending_ids().is_none());
    if let TrustRegionState::Multi { state, .. } = opt.trust_region() {
        assert_eq!(state.trial_counts().iter().sum::<u64>(), 3);
    }
    assert!(opt.ask(1, &mut rng).is_ok());
}

#[test]
fn regional_settlement() {
    let mut rng = StdRng::seed_from_u64(67);
    let mut opt = optimizer(array![[0.0, 1.0], [0.0, 1.0]], &mut rng);
    opt.ask(2, &mut rng).unwrap();
    let ids = opt.pending_ids().unwrap();
    opt.complete_ids(&ids[..1], &array![[1.0]].view(), None, &mut rng)
        .unwrap();
    assert_eq!(opt.pending_ids().unwrap(), ids[1..]);
    if let TrustRegionState::Multi { state, .. } = opt.trust_region() {
        assert_eq!(state.trial_counts(), vec![1, 0]);
    } else {
        panic!("expected multi-region state");
    }
    opt.cancel_ids(&ids[1..], &mut rng).unwrap();
    if let TrustRegionState::Multi { state, pending } = opt.trust_region() {
        assert!(pending.is_none());
        assert_eq!(state.trial_counts(), vec![1, 0]);
    } else {
        panic!("expected multi-region state");
    }
}

#[test]
fn warmup_completion() {
    let mut rng = StdRng::seed_from_u64(41);
    let bounds = array![[0.0, 1.0], [0.0, 1.0]];
    let strategy = Strategy::experimental(bounds.nrows(), 2, 2, &mut rng).unwrap();
    let mut opt =
        BayesOptimizer::new_strategy(bounds, baux::turbo_zero(), strategy, &mut rng).unwrap();
    opt.ask(2, &mut rng).unwrap();
    let ids = opt.pending_ids().unwrap();
    assert_eq!(ids.len(), 2);
    assert!(opt.ask(1, &mut rng).is_err());
    opt.import_variance(
        &array![[0.5, 0.5]].view(),
        &array![[4.0]].view(),
        None,
        &mut rng,
    )
    .unwrap();
    assert_eq!(opt.pending_ids().unwrap(), ids);
    opt.complete_ids(&ids[1..], &array![[2.0]].view(), None, &mut rng)
        .unwrap();
    assert_eq!(opt.pending_ids().unwrap(), ids[..1]);
    opt.complete_ids(&ids[..1], &array![[1.0]].view(), None, &mut rng)
        .unwrap();
    assert!(opt.pending_ids().is_none());
    assert_eq!(opt.obs_count(), 3);
    assert!(opt.ask(1, &mut rng).is_ok());
}
