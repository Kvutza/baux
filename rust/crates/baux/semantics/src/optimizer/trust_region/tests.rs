use super::*;

#[test]
fn paired_streams() {
    for left in 0..4 {
        for right in left + 1..4 {
            assert_ne!(candidate_seed(73, left), candidate_seed(73, right));
        }
    }
}

#[test]
fn pool_parity() {
    let legacy = PoolShape::new(1, 4).unwrap();
    assert_eq!(legacy, PoolShape::legacy());
    for left in 0..4 {
        for right in left + 1..4 {
            assert_ne!(legacy.seed(73, left), legacy.seed(73, right));
        }
    }

    let paired = PoolShape::new(2, 4).unwrap();
    for left in 0..4 {
        for right in left + 1..4 {
            assert_ne!(paired.seed(73, left), paired.seed(73, right));
        }
    }
}

#[test]
fn arm_distinct() {
    let shape = PoolShape::new(4, 4).unwrap();
    let seeds = std::array::from_fn::<_, 4, _>(|candidate| shape.seed(73, candidate));
    for left in 0..4 {
        for right in left + 1..4 {
            assert_ne!(seeds[left], seeds[right]);
        }
    }
}

#[test]
fn neighbor_query() {
    use baux::semantics::optimizer::{ChainedObservation, ParamFamily, ParamTree, WorldId};
    let estimator = NeighborValueEstimator::new(1.25, 0.04, 73)
        .unwrap()
        .with_law(UncertaintyLaw::InversePrecision);
    let prediction = estimator.predict(&[0.0_f64]).unwrap();
    assert_eq!(prediction.mean.to_bits(), 1.25_f32.to_bits());
    assert!(prediction.standard_error.is_finite());
    let bv_pred = estimator
        .predict_law(&[0.0_f64], UncertaintyLaw::BiasVariance)
        .unwrap();
    assert!(bv_pred.standard_error.is_finite());
    let geom_pred = estimator
        .predict_law(&[0.0_f64], UncertaintyLaw::Geometric)
        .unwrap();
    assert!(geom_pred.standard_error.is_finite());
    let hyp_pred = estimator.predict_hyperbolic(&[0.0_f64], 1).unwrap();
    assert_eq!(hyp_pred.mean.to_bits(), prediction.mean.to_bits());
    assert!(estimator.predict(&[]).is_err());

    let counts = ParamTree::<usize>::mhc4_counts(1);
    assert_eq!(ParamTree::<usize>::total_parameters(1), 265_623_320);
    assert_eq!(
        ParamTree::<usize>::total_parameters(1) as u64,
        baux::semantics::optimizer::param_tree::total_count()
    );
    assert_eq!(
        *counts.get(ParamFamily::RoutedGateUp) as u64 + *counts.get(ParamFamily::RoutedDown) as u64,
        baux::semantics::optimizer::param_tree::routed_count()
    );
    assert_eq!(ParamTree::<usize>::total_parameters(64), 265_656_600);
    let shares = ParamTree::<usize>::mhc4_shares(1);
    assert!(*shares.get(ParamFamily::RoutedGateUp) > 0.3);
    let doubled = counts.zip_with(&counts, |&a, &b| a + b);
    assert_eq!(
        *doubled.get(ParamFamily::TokenEmbed),
        counts.get(ParamFamily::TokenEmbed) * 2
    );

    let w1 = WorldId::FixedCorpus {
        seed: 7,
        tokens: 1024,
    };
    let w2 = WorldId::RotatingCondition {
        step: 1,
        window_id: 2,
        tokens: 1024,
    };
    let paired = w1.pair(3.0, w1, 2.5).unwrap();
    let (anchored, anchored_val, anchored_var) = paired.anchor(0, 1.0, 0.05, 0.1);
    assert!((anchored_val - 1.5).abs() < 1.0e-6);
    assert!((anchored_var - 0.15).abs() < 1.0e-6);
    assert!(!w1.pair(3.0, w2, 2.5).unwrap_err().message.is_empty());
    let rot_paired = w2.pair(3.0, w2, 2.5).unwrap().with_gain(Some(0.5));
    let (rot_obs, rot_val, rot_var) = rot_paired.anchor(0, 1.0, 0.05, 0.1);
    assert!((rot_val - 1.5).abs() < 1.0e-6);
    assert!((rot_var - 0.15).abs() < 1.0e-6);
    assert!(matches!(rot_obs, ChainedObservation::DriftedChain { .. }));
    let drifted = ChainedObservation::DriftedChain {
        from_step: 0,
        to_step: 16,
        step_delta: 0.1,
        drift_variance_floor: 0.01,
    };
    assert!(drifted.noise_variance(0.1) > anchored.noise_variance(0.1));
}

#[test]
fn acceptance_margin() {
    assert!(!accept_candidate(0.0, 0.0));
    assert!(!accept_candidate(0.2, 0.2));
    assert!(accept_candidate(0.200_001, 0.2));
    assert!(!accept_candidate(-0.1, 0.2));
}

#[test]
fn bits_match() {
    let edge = [
        f64::NEG_INFINITY,
        -f64::MAX,
        -1.0,
        -f64::MIN_POSITIVE,
        -f64::from_bits(1),
        -0.0,
        0.0,
        f64::from_bits(1),
        f64::MIN_POSITIVE,
        1.0,
        f64::MAX,
        f64::INFINITY,
        f64::NAN,
        f64::from_bits(0xfff8_0000_0000_0001),
    ];
    for improvement in edge {
        for margin in edge {
            assert_eq!(
                accept_bits(improvement.to_bits(), margin.to_bits()),
                improvement > margin,
                "improvement={improvement:?} margin={margin:?}",
            );
        }
    }

    let mut left = 0x243f_6a88_85a3_08d3;
    let mut right = 0x1319_8a2e_0370_7344;
    for _ in 0..100_000 {
        left = splitmix64(left);
        right = splitmix64(right);
        let improvement = f64::from_bits(left);
        let margin = f64::from_bits(right);
        assert_eq!(accept_bits(left, right), improvement > margin);
    }
}

#[test]
fn inclusive_incumbent() {
    let spec = baux::semantics::SurrogateSpec {
        neighbors: 4,
        epistemic: 0.5,
        aleatoric: 0.05,
        scale: 0.05,
        fit: false,
        adaptive: false,
        ..Default::default()
    };
    let mut est = NeighborValueEstimator::with_spec(-15.0, 0.0028, 7, spec)
        .unwrap()
        .with_metric(baux::semantics::neighbor::DistanceMetric::Conformal { local_k: 2 });
    // Append two older worse points and one accepted better incumbent at tiny
    // raw distances (1.6e-5).
    est.append(&[1.6e-5_f64], -15.2, 0.0028, false).unwrap();
    est.append(&[3.2e-5_f64, 1.6e-5_f64], -15.3, 0.0028, false)
        .unwrap();
    est.append(&[4.8e-5_f64, 3.2e-5_f64, 1.6e-5_f64], -14.6, 0.0028, true)
        .unwrap();

    // Both incumbent_prediction and incumbent_baseline use inclusive in-sample
    // query (excluded = None), including the accepted -14.6 incumbent at d^2 =
    // 0.
    let inc_pred = est.incumbent_prediction().unwrap();
    let inc_base = est.incumbent_baseline().unwrap();
    assert_eq!(inc_pred.mean.to_bits(), inc_base.mean.to_bits());
    assert!(
        inc_base.mean > -15.05,
        "inclusive baseline {} must reflect the -14.6 incumbent at d^2 = 0",
        inc_base.mean
    );

    // Self-tuning normalizes 1.6e-5 distances so a candidate near the good
    // incumbent (slot 3) scores strictly higher than a candidate near the
    // bad point (slot 2).
    let near_good = est
        .acquire(&[6.4e-5_f64, 4.8e-5_f64, 3.2e-5_f64, 0.4e-5_f64], 0.0, None)
        .unwrap();
    let near_bad = est
        .acquire(&[3.2e-5_f64, 1.6e-5_f64, 0.4e-5_f64, 3.2e-5_f64], 0.0, None)
        .unwrap();
    assert!(
        near_good.mean > near_bad.mean,
        "self-tuning must distinguish near_good ({}) from near_bad ({}) at 1e-5 scale",
        near_good.mean,
        near_bad.mean
    );

    // Verify DistanceMetric::Hyperbolic and UncertaintyLaw::BiasVariance also
    // take effect in acquire().
    let hyp_bv = NeighborValueEstimator::with_spec(-15.0, 0.0028, 7, spec)
        .unwrap()
        .with_metric(DistanceMetric::Hyperbolic { local_k: 2 })
        .with_law(UncertaintyLaw::BiasVariance);
    let acq_hyp = hyp_bv.acquire(&[1.6e-5_f64], 1.0, None).unwrap();
    assert!(acq_hyp.standard_error.is_finite() && acq_hyp.epistemic_error.is_finite());

    // Verify Euclidean LOO fitting with tiny distances (1.6e-5) normalizes
    // epistemic scale by characteristic distance and selects adaptive k.
    let mut adaptive_est =
        NeighborValueEstimator::configured(-15.0, 0.0028, 7, true, true).unwrap();
    for step in 1..32 {
        let row: Vec<f64> = (0..step)
            .map(|prev| ((step - prev) as f64).powi(2) * 1.6e-5)
            .collect();
        adaptive_est
            .append(&row, -15.0 + step as f32 * 0.02, 0.0028, step % 4 == 0)
            .unwrap();
    }
    let state = adaptive_est.fit_state();
    assert!((1..=31).contains(&state.neighbors));
    assert!(state.epistemic > 1.0);
    assert!(state.aleatoric <= 1.0);
}

#[test]
fn turbo_selection() {
    let mut tr = TrustRegionController::with_turbo(0.004, 0.001, 0.008, 4).unwrap();
    let spec = baux::semantics::SurrogateSpec {
        neighbors: 2,
        ..Default::default()
    };
    let mut sur = NeighborValueEstimator::with_spec(-15.0, 0.0028, 7, spec).unwrap();
    let rule = baux::semantics::optimizer::surrogate::AcquisitionRule::Ucb { beta: 0.0 };
    let warm_sel = sur.select_pool(&[vec![1.6e-5_f64]], 0, rule, None).unwrap();
    assert!(warm_sel.initializing);
    // Warmup always uses threshold 0.0 (upstream enn parity) without advancing
    // radius.
    let same_var_dec = tr
        .evaluate_selection(-14.99, 0.0028, 0.004, &warm_sel)
        .unwrap();
    assert_eq!(same_var_dec.threshold, 0.0);
    assert!(same_var_dec.accepted);
    let warm_dec = tr
        .evaluate_selection(-14.88, 0.0025, 0.004, &warm_sel)
        .unwrap();
    assert_eq!(warm_dec.threshold, 0.0);
    assert!(warm_dec.accepted);
    assert_eq!(warm_dec.action, RadiusAction::Hold);

    // Advance surrogate past warmup with a rejected point strictly below the
    // incumbent so the convex-combination KNN prediction satisfies
    // candidate.mean < incumbent.mean.
    sur.append(&[1.6e-5_f64], -15.2, 0.0028, false).unwrap();
    let post_sel = sur
        .select_pool(&[vec![3.2e-5_f64, 1.6e-5_f64]], 1, rule, None)
        .unwrap();
    assert!(!post_sel.initializing);
    assert!(post_sel.candidate.mean < post_sel.incumbent.mean);

    // Post-warmup TuRBO uses positive-improvement (threshold == 0.0) and
    // expands after 3 accepts.
    for idx in 0..3 {
        let dec = tr
            .evaluate_selection(-14.99, 0.0025, 0.004, &post_sel)
            .unwrap();
        assert_eq!(dec.threshold, 0.0);
        assert!(dec.accepted);
        if idx < 2 {
            assert_eq!(dec.action, RadiusAction::Hold);
            assert!((dec.next_radius - 0.004).abs() < 1e-7);
        } else {
            assert_eq!(dec.action, RadiusAction::Expand);
            assert!((dec.next_radius - 0.008).abs() < 1e-7);
        }
    }

    // 4 consecutive rejections shrink the radius by 0.5x.
    for idx in 0..4 {
        let dec = tr
            .evaluate_selection(-15.01, 0.0025, 0.008, &post_sel)
            .unwrap();
        assert!(!dec.accepted);
        if idx < 3 {
            assert_eq!(dec.action, RadiusAction::Hold);
            assert!((dec.next_radius - 0.008).abs() < 1e-7);
        } else {
            assert_eq!(dec.action, RadiusAction::Shrink);
            assert!((dec.next_radius - 0.004).abs() < 1e-7);
        }
    }
}
