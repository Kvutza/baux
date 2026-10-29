use super::*;
use ndarray::array;

#[test]
fn test_001() {
    let config = TRLengthConfig::default();
    let tr = TurboTrustRegion::new(5, config);

    assert_eq!(tr.length(), 0.8);
    assert!(!tr.needs_restart());
}

#[test]
fn test_turboupdatesuccess() {
    let config = TRLengthConfig::default();
    let mut tr = TurboTrustRegion::new(5, config);
    tr.set_arms(1);

    // First update - establish best value (y_all = [1.0])
    tr.update(&array![1.0].view(), 1).unwrap();
    assert_eq!(tr.length(), 0.8);

    // Three consecutive improvements should expand (y_all grows each time)
    tr.update(&array![1.0, 2.0].view(), 2).unwrap();
    tr.update(&array![1.0, 2.0, 3.0].view(), 3).unwrap();
    tr.update(&array![1.0, 2.0, 3.0, 4.0].view(), 4).unwrap();

    assert!(tr.length() > 0.8); // Should have expanded
}

#[test]
fn test_turboupdatefailure() {
    let config = TRLengthConfig::default();
    let mut tr = TurboTrustRegion::new(5, config);
    tr.set_arms(1);

    // Establish best value
    tr.update(&array![1.0].view(), 1).unwrap();

    // Multiple failures should contract (y_all grows: [1, 0.5], [1, 0.5, 0.5],
    // ...)
    let failure_tol = tr.failure_tolerance.unwrap();
    let mut y_all = vec![1.0];
    for _ in 0..failure_tol {
        y_all.push(0.5);
        let y_arr = ndarray::Array1::from_vec(y_all.clone());
        tr.update(&y_arr.view(), y_all.len()).unwrap();
    }

    assert!(tr.length() < 0.8); // Should have contracted
}

#[test]
fn effective_tolerance() {
    let config = TRLengthConfig::default();
    let mut tr = TurboTrustRegion::new(1_000, config);

    tr.set_arms(4);
    assert_eq!(tr.failure_tolerance, Some(250));

    tr.set_dim(16.0);
    assert_eq!(tr.failure_tolerance, Some(4));

    tr.set_arms(8);
    assert_eq!(tr.failure_tolerance, Some(2));
}

#[test]
fn negative_rewards() {
    let mut tr = TurboTrustRegion::new(1_065_494_016, TRLengthConfig::new(0.01, 0.0001, 0.1));
    tr.set_arms(1);
    assert_eq!(tr.failure_tolerance(), 1_065_494_016);
    tr.set_tolerance(4).unwrap();
    tr.set_dim(1_065_494_016.0);
    tr.set_arms(8);
    assert_eq!(tr.failure_tolerance(), 4);
    assert!(tr.set_tolerance(0).is_err());
    assert!(tr.set_tolerance(i32::MAX as usize + 1).is_err());
    assert_eq!(tr.failure_tolerance(), 4);

    let mut values = vec![-12.0];
    tr.update(&ArrayView1::from(&values), values.len()).unwrap();
    for value in [-11.0, -10.0, -9.0] {
        values.push(value);
        tr.update(&ArrayView1::from(&values), values.len()).unwrap();
    }
    assert_eq!(tr.length(), 0.02);
    for failures in 1..=4 {
        values.push(-13.0);
        tr.update(&ArrayView1::from(&values), values.len()).unwrap();
        assert_eq!(tr.length(), if failures < 4 { 0.02 } else { 0.01 });
    }
    tr.restart();
    assert_eq!(tr.failure_tolerance(), 4);
    assert_eq!(tr.length(), 0.01);
}

#[test]
fn inconclusive_evidence() {
    let mut tr = TurboTrustRegion::new(1_065_494_016, TRLengthConfig::new(0.01, 0.0001, 0.1));
    tr.set_tolerance(4).unwrap();

    tr.update_outcome(
        &array![-12.0].view(),
        1,
        -12.0,
        TrustRegionOutcome::Inconclusive,
    )
    .unwrap();
    for (observation, value) in [(2, -11.0), (3, -10.0)] {
        tr.update_outcome(
            &array![value].view(),
            observation,
            value,
            TrustRegionOutcome::Success,
        )
        .unwrap();
    }
    assert_eq!(tr.success_counter, 2);
    assert_eq!(tr.length(), 0.01);

    tr.update_outcome(
        &array![-10.0].view(),
        4,
        -10.0,
        TrustRegionOutcome::Inconclusive,
    )
    .unwrap();
    assert_eq!(tr.success_counter, 2);
    assert_eq!(tr.failure_counter, 0);
    assert_eq!(tr.length(), 0.01);

    tr.update_outcome(&array![-9.0].view(), 5, -9.0, TrustRegionOutcome::Success)
        .unwrap();
    assert_eq!(tr.length(), 0.02);

    for (observation, value) in [(6, -13.0), (8, -13.0), (10, -13.0), (12, -13.0)] {
        tr.update_outcome(
            &array![value].view(),
            observation,
            -9.0,
            TrustRegionOutcome::Failure,
        )
        .unwrap();
        if observation < 12 {
            tr.update_outcome(
                &array![-9.0].view(),
                observation + 1,
                -9.0,
                TrustRegionOutcome::Inconclusive,
            )
            .unwrap();
        }
    }
    assert_eq!(tr.length(), 0.01);
    assert_eq!(tr.failure_counter, 0);
}

#[test]
fn test_turborestart() {
    let config = TRLengthConfig::default();
    let mut tr = TurboTrustRegion::new(5, config);
    tr.set_arms(1);

    // Contract length (y_all grows: [1], [1,0.5], [1,0.5,0.5], ...)
    tr.update(&array![1.0].view(), 1).unwrap();
    let mut y_all = vec![1.0, 0.5];
    for _ in 0..10 {
        let y_arr = ndarray::Array1::from_vec(y_all.clone());
        tr.update(&y_arr.view(), y_all.len()).unwrap();
        y_all.push(0.5);
    }

    assert!(tr.length() < 0.8);

    // Restart
    tr.restart();
    assert_eq!(tr.length(), 0.8);
    assert!(!tr.needs_restart());
}

#[test]
fn test_turbobounds() {
    let config = TRLengthConfig::default();
    let tr = TurboTrustRegion::new(5, config);

    let center = array![0.5, 0.5, 0.5, 0.5, 0.5];
    let (lb, ub) = tr.compute_bounds(&center.view(), None);

    assert!(lb.iter().all(|&v| (0.0..=1.0).contains(&v)));
    assert!(ub.iter().all(|&v| (0.0..=1.0).contains(&v)));

    for i in 0..5 {
        assert!(lb[i] < ub[i]);
    }
}

#[test]
fn test_notrustregion() {
    let tr = NoTrustRegion::new(5);

    assert_eq!(tr.length(), 1.0);
    assert!(!tr.needs_restart());

    let center = array![0.5, 0.5, 0.5, 0.5, 0.5];
    let (lb, ub) = tr.compute_bounds(&center.view(), None);

    assert_eq!(lb, array![0.0, 0.0, 0.0, 0.0, 0.0]);
    assert_eq!(ub, array![1.0, 1.0, 1.0, 1.0, 1.0]);
}

#[test]
fn test_008() {
    let e1 = TrustRegionError::InvalidParameter("bad param".to_string());
    let e2 = TrustRegionError::InvalidState("bad state".to_string());
    assert!(e1.to_string().contains("Invalid parameter"));
    assert!(e2.to_string().contains("Invalid state"));
}

#[test]
fn update_history() {
    let config = TRLengthConfig::default();
    let mut full = TurboTrustRegion::new(2, config);
    let mut batch = TurboTrustRegion::new(2, config);
    full.set_arms(1);
    batch.set_arms(1);

    let batches: [&[f64]; 4] = [&[1.0], &[0.5, 0.4], &[2.0], &[1.9, 1.8, 2.1]];
    let incumbents = [1.0, 1.0, 2.0, 2.1];
    let mut y_all: Vec<f64> = Vec::new();
    for (b, &inc) in batches.iter().zip(incumbents.iter()) {
        y_all.extend_from_slice(b);
        let y_arr = ndarray::Array1::from_vec(y_all.clone());
        full.update_history(&y_arr.view(), y_all.len(), inc)
            .unwrap();
        let y_new = ndarray::Array1::from_vec(b.to_vec());
        batch.update_batch(&y_new.view(), y_all.len(), inc).unwrap();
        assert_eq!(full.length(), batch.length());
        assert_eq!(full.prev_obs, batch.prev_obs);
        assert_eq!(full.success_counter, batch.success_counter);
        assert_eq!(full.failure_counter, batch.failure_counter);
    }
}

#[test]
fn seeded_scale() {
    let mut tr = TurboTrustRegion::new(2, TRLengthConfig::default());
    tr.set_arms(1);
    tr.seed_history(&array![-20.0, -5.0].view());
    tr.set_watermark(2);
    tr.update_batch(&array![-4.0].view(), 3, -4.0).unwrap();
    let length_before = tr.length();
    tr.update_batch(&array![-3.999].view(), 4, -3.999).unwrap();
    assert_eq!(tr.length(), length_before);
    assert_eq!(tr.failure_counter, 1);
}

#[test]
fn absolute_scale() {
    let mut tr = TurboTrustRegion::new(2, TRLengthConfig::default());
    tr.set_arms(1);
    tr.seed_history(&array![-1000.0, 10.0].view());
    tr.set_watermark(2);
    tr.update_batch(&array![10.02].view(), 3, 10.02).unwrap();
    assert_eq!(
        tr.success_counter, 1,
        "0.02 exceeds 0.001 * |10| but not 0.001 * the historical range"
    );
}

#[test]
fn restart_scale() {
    let mut tr = TurboTrustRegion::new(2, TRLengthConfig::default());
    tr.set_arms(1);
    tr.update_batch(&array![0.0, 100.0].view(), 2, 100.0)
        .unwrap();
    tr.restart();
    assert_eq!(tr.prev_obs(), 2);
    assert_eq!(tr.hist_ymin, 0.0);
    assert_eq!(tr.hist_ymax, 100.0);
}

#[test]
fn update_full() {
    // After restart, prev_obs is kept so the next tell stays O(batch).
    let config = TRLengthConfig::default();
    let mut full = TurboTrustRegion::new(2, config);
    let mut mixed = TurboTrustRegion::new(2, config);
    full.set_arms(1);
    mixed.set_arms(1);

    let pre: [&[f64]; 2] = [&[1.0, 0.5], &[0.8]];
    let mut y_all: Vec<f64> = Vec::new();
    for (b, &inc) in pre.iter().zip([1.0_f64, 1.0].iter()) {
        y_all.extend_from_slice(b);
        let y_arr = ndarray::Array1::from_vec(y_all.clone());
        full.update_history(&y_arr.view(), y_all.len(), inc)
            .unwrap();
        mixed
            .update_history(&y_arr.view(), y_all.len(), inc)
            .unwrap();
    }

    let watermark = mixed.prev_obs();
    assert!(watermark > 0);
    full.restart();
    mixed.restart();
    assert_eq!(mixed.prev_obs(), watermark);
    assert_eq!(full.prev_obs(), watermark);

    let post: [&[f64]; 3] = [&[2.0], &[1.5, 2.2], &[2.3]];
    let post_inc = [2.0_f64, 2.2, 2.3];
    for (b, &inc) in post.iter().zip(post_inc.iter()) {
        y_all.extend_from_slice(b);
        let y_arr = ndarray::Array1::from_vec(y_all.clone());
        full.update_history(&y_arr.view(), y_all.len(), inc)
            .unwrap();
        let y_new = ndarray::Array1::from_vec(b.to_vec());
        mixed.update_batch(&y_new.view(), y_all.len(), inc).unwrap();
        assert_eq!(full.length(), mixed.length());
        assert_eq!(full.prev_obs, mixed.prev_obs);
        assert_eq!(full.success_counter, mixed.success_counter);
        assert_eq!(full.failure_counter, mixed.failure_counter);
        assert_eq!(full.hist_ymin, mixed.hist_ymin);
        assert_eq!(full.hist_ymax, mixed.hist_ymax);
    }
}

#[test]
fn length_bounds() {
    let valid = TRLengthConfig::try_new(0.8, 0.01, 1.6).unwrap();
    assert_eq!(valid.clamp_radius(0.5), 0.5);
    assert_eq!(valid.clamp_radius(0.001), 0.01);
    assert_eq!(valid.clamp_radius(2.0), 1.6);

    for (init, min, max) in [
        (0.8, 1.6, 0.4),
        (0.05, 0.1, 1.0),
        (2.0, 0.1, 1.0),
        (0.8, 0.0, 1.6),
        (0.8, -0.1, 1.6),
        (f64::NAN, 0.1, 1.0),
        (0.5, f64::NAN, 1.0),
        (0.5, 0.1, f64::NAN),
        (0.5, 0.1, f64::INFINITY),
    ] {
        assert!(TRLengthConfig::try_new(init, min, max).is_err());
    }

    let inverted = TRLengthConfig::new(0.8, 1.6, 0.1);
    assert_eq!(inverted.clamp_radius(0.5), 0.5);
    assert_eq!(inverted.clamp_radius(0.01), 0.1);
    assert_eq!(inverted.clamp_radius(2.0), 1.6);
    assert!(
        TRLengthConfig::new(0.8, f64::NAN, 1.6)
            .clamp_radius(0.5)
            .is_nan()
    );
}
