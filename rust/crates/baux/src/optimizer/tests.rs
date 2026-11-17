use super::BayesOptimizer;
use crate::config::turbo_zero;
use crate::error::ENNError;
use crate::optimizer::Strategy;
use crate::surrogate::{BoxedSurrogate, Surrogate, SurrogatePrediction};
use ndarray::{Array1, Array2, Array3, ArrayView2, array};
use rand::RngCore;
use rand::SeedableRng;
use rand::rngs::StdRng;

struct CoordinateSurrogate;

impl Surrogate for CoordinateSurrogate {
    fn with_fit(
        self: Box<Self>,
        _x: &ArrayView2<f64>,
        _y: &ArrayView2<f64>,
        _yvar: Option<&ArrayView2<f64>>,
        _rng: &mut dyn RngCore,
    ) -> Result<BoxedSurrogate, ENNError> {
        Ok(self)
    }

    fn clone_box(&self) -> Result<BoxedSurrogate, ENNError> {
        Ok(Box::new(CoordinateSurrogate))
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
    let (next_a, x0) = branch_a.ask_natural(4, &mut rng).unwrap();
    branch_a = next_a;
    let y0 = array![[1.0], [2.0], [1.5], [0.5]];
    branch_a = branch_a
        .tell_natural(&x0.view(), &y0.view(), None, &mut rng)
        .unwrap();

    let mut branch_b = branch_a.fork().unwrap();
    assert_eq!(branch_a.digest(), branch_b.digest());
    assert_eq!(branch_a.digest_hex().len(), 64);

    let key = crate::hash::Key::new(777);
    let mut rng_a = key.prng();
    let mut rng_b = key.prng();
    let (prop_a, next_a) = branch_a.ask_ticket(2, &mut rng_a).unwrap();
    branch_a = next_a;
    let (prop_b, next_b) = branch_b.ask_ticket(2, &mut rng_b).unwrap();
    branch_b = next_b;
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
    let branch_b = branch_a.fork().unwrap();

    let (stale, next_a) = branch_a
        .ask_ticket(2, &mut crate::hash::Key::new(1).prng())
        .unwrap();
    branch_a = next_a;
    let (fresh, next_a) = branch_a
        .ask_ticket(2, &mut crate::hash::Key::new(2).prng())
        .unwrap();
    branch_a = next_a;
    let y = array![[1.0], [2.0]];
    branch_a = branch_a
        .tell_ticket(fresh, &y.view(), None, &mut rng)
        .unwrap();

    let err_stale = branch_a
        .fork()
        .unwrap()
        .tell_ticket(stale, &y.view(), None, &mut rng)
        .expect_err("stale ticket must be rejected");
    assert!(err_stale.to_string().contains("digest mismatch"));

    let (ticket_b, next_b) = branch_b
        .ask_ticket(2, &mut crate::hash::Key::new(3).prng())
        .unwrap();
    let _ = next_b;
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
    let (next, x) = opt.ask_natural(3, &mut rng).unwrap();
    opt = next;
    let y = array![[0.2], [0.8], [0.5]];
    opt = opt
        .tell_natural(&x.view(), &y.view(), None, &mut rng)
        .unwrap();

    let mut frozen = opt.freeze().unwrap();
    let thawed = opt.thaw(&frozen).unwrap();
    assert_eq!(thawed.obs_count(), opt.obs_count());
    assert_eq!(thawed.digest(), opt.digest());

    frozen[16] ^= 0xFF;
    assert!(opt.thaw(&frozen).is_err());
}

fn bits_equal(a: &Array2<f64>, b: &Array2<f64>) -> bool {
    a.shape() == b.shape()
        && a.iter()
            .zip(b.iter())
            .all(|(u, v)| u.to_bits() == v.to_bits())
}

fn seal_test(mut payload: Vec<u8>) -> Vec<u8> {
    let tag = crate::artifact::blake3::hash(&payload);
    payload.extend_from_slice(&tag);
    payload
}

#[test]
fn freeze_turbo() {
    use crate::candidates::CandidateSampler;
    use crate::config::{SurrogateConfig, turbo_enn};

    let bounds = Array2::from_shape_fn((40, 2), |(i, j)| {
        if j == 0 {
            -3.0
        } else {
            3.0 + 0.05 * (i as f64)
        }
    });
    let mut cfg = turbo_enn();
    cfg.candidates.candidate_rv = CandidateSampler::Sobol;
    cfg.noise_aware = true;
    if let SurrogateConfig::ENN(ref mut enn) = cfg.surrogate {
        enn.k = 5;
        enn.scale_x = true;
    }

    let mut init_rng = StdRng::seed_from_u64(101);
    let template =
        BayesOptimizer::new(bounds.clone(), cfg.clone(), &mut StdRng::seed_from_u64(1)).unwrap();
    let mut live = BayesOptimizer::new_strategy(
        bounds,
        cfg,
        Strategy::hybrid(crate::config::InitStrategy::LHD, 8),
        &mut init_rng,
    )
    .unwrap();

    // 10 rounds * 8 arms = 80 observations (d = 40, arms = 8 =>
    // failure_tolerance = 5): round 0 sets high warmup rewards; rounds 1..=9
    // strictly deteriorate so TuRBO contracts tr_length once (0.8 -> 0.4).
    for step in 0..10u64 {
        let mut step_rng = StdRng::seed_from_u64(1000 + step);
        let (next_live, x) = live.ask_natural(8, &mut step_rng).unwrap();
        live = next_live;
        let mut y = Array2::zeros((8, 1));
        let mut yvar = Array2::zeros((8, 1));
        for r in 0..8 {
            let norm: f64 = x.row(r).iter().map(|v| v * v).sum();
            y[[r, 0]] = if step == 0 {
                100.0 - 1e-3 * norm
            } else {
                -50.0 * (step as f64) - 1e-3 * norm
            };
            yvar[[r, 0]] = 0.01 + 0.005 * (r as f64);
        }
        live = live
            .tell_natural(&x.view(), &y.view(), Some(&yvar.view()), &mut step_rng)
            .unwrap();
    }

    assert_eq!(live.obs_count(), 80);
    assert!((live.tr_length() - 0.4).abs() < 1e-12);

    let frozen = live.freeze().unwrap();
    let mut restored = template.thaw(&frozen).unwrap();
    assert_eq!(restored.obs_count(), 80);
    assert_eq!(restored.tr_length().to_bits(), live.tr_length().to_bits());
    assert_eq!(restored.digest(), live.digest());
    assert_eq!(restored.freeze().unwrap(), frozen);

    let xq = Array2::from_shape_fn((2, 40), |(r, c)| {
        0.25 * ((r + 1) as f64) - 0.02 * (c as f64)
    });
    let mu_live = live.posterior_mu(&xq.view()).unwrap();
    let mu_rest = restored.posterior_mu(&xq.view()).unwrap();
    assert!(bits_equal(&mu_live, &mu_rest));

    for step in 20..24u64 {
        let mut rng_a = StdRng::seed_from_u64(step);
        let mut rng_b = StdRng::seed_from_u64(step);
        let (next_live, xa) = live.ask_natural(8, &mut rng_a).unwrap();
        live = next_live;
        let (next_rest, xb) = restored.ask_natural(8, &mut rng_b).unwrap();
        restored = next_rest;
        assert!(bits_equal(&xa, &xb));

        let mut y = Array2::zeros((8, 1));
        let mut yv = Array2::from_elem((8, 1), 0.02);
        for r in 0..8 {
            y[[r, 0]] = -500.0 - xa.row(r).iter().map(|v| (v - 0.3).powi(2)).sum::<f64>();
            yv[[r, 0]] += 0.001 * (r as f64);
        }
        live = live
            .tell_natural(&xa.view(), &y.view(), Some(&yv.view()), &mut rng_a)
            .unwrap();
        restored = restored
            .tell_natural(&xb.view(), &y.view(), Some(&yv.view()), &mut rng_b)
            .unwrap();
        if step == 21 {
            assert!((restored.tr_length() - 0.2).abs() < 1e-12);
        }
        assert_eq!(restored.tr_length().to_bits(), live.tr_length().to_bits());
        assert_eq!(restored.digest(), live.digest());
    }
}

#[test]
fn freeze_morbo() {
    use crate::config::turbo_enn;
    use crate::morbo::{MorboTRSettings, Rescalarize};
    use crate::trust_region::{TRLengthConfig, TrustRegionConfig};

    let bounds = array![[-2.0, 2.0], [-2.0, 2.0], [-2.0, 2.0]];
    let mut cfg = turbo_enn();
    cfg.trust_region = TrustRegionConfig::Morbo(MorboTRSettings {
        num_metrics: 2,
        alpha: 0.05,
        length: TRLengthConfig::default(),
        rescalarize: Rescalarize::OnPropose,
        noise_aware: false,
    });

    let template =
        BayesOptimizer::new(bounds.clone(), cfg.clone(), &mut StdRng::seed_from_u64(7)).unwrap();
    let mut live = BayesOptimizer::new_strategy(
        bounds,
        cfg,
        Strategy::hybrid(crate::config::InitStrategy::LHD, 6),
        &mut StdRng::seed_from_u64(77),
    )
    .unwrap();

    for step in 0..4u64 {
        let mut step_rng = StdRng::seed_from_u64(500 + step);
        let (next_live, x) = live.ask_natural(3, &mut step_rng).unwrap();
        live = next_live;
        let mut y = Array2::zeros((3, 2));
        for r in 0..3 {
            y[[r, 0]] = -x.row(r).iter().map(|v| (v - 0.5).powi(2)).sum::<f64>();
            y[[r, 1]] = -x.row(r).iter().map(|v| (v + 0.5).powi(2)).sum::<f64>();
        }
        live = live
            .tell_natural(&x.view(), &y.view(), None, &mut step_rng)
            .unwrap();
    }

    let frozen = live.freeze().unwrap();
    let mut restored = template.thaw(&frozen).unwrap();
    assert_eq!(restored.digest(), live.digest());
    assert_eq!(restored.freeze().unwrap(), frozen);

    for step in 10..14u64 {
        let mut rng_a = StdRng::seed_from_u64(step);
        let mut rng_b = StdRng::seed_from_u64(step);
        let (next_live, xa) = live.ask_natural(3, &mut rng_a).unwrap();
        live = next_live;
        let (next_rest, xb) = restored.ask_natural(3, &mut rng_b).unwrap();
        restored = next_rest;
        assert!(bits_equal(&xa, &xb));
        let mut y = Array2::zeros((3, 2));
        for r in 0..3 {
            y[[r, 0]] = -xa.row(r).iter().map(|v| v * v).sum::<f64>();
            y[[r, 1]] = -xa.row(r).iter().map(|v| (v - 1.0).powi(2)).sum::<f64>();
        }
        live = live
            .tell_natural(&xa.view(), &y.view(), None, &mut rng_a)
            .unwrap();
        restored = restored
            .tell_natural(&xb.view(), &y.view(), None, &mut rng_b)
            .unwrap();
        assert_eq!(restored.digest(), live.digest());
    }
}

#[test]
fn freeze_portfolio() {
    use crate::config::turbo_enn;
    use crate::optimizer::{
        MultiTrustRegionConfig, RestartMemoryPolicy, SharingPolicy, StrategyState,
    };
    use crate::trust_region::TRLengthConfig;

    let bounds = array![[-2.0, 2.0], [-2.0, 2.0], [-2.0, 2.0]];
    let cfg = turbo_enn();
    let mut mcfg = MultiTrustRegionConfig::new(4, TRLengthConfig::default());
    mcfg.sharing_policy = SharingPolicy::EllipsoidBall;
    mcfg.restart_memory_policy = RestartMemoryPolicy::RetainRelevant;

    let t_state =
        StrategyState::configured(3, mcfg.clone(), 6, &mut StdRng::seed_from_u64(3)).unwrap();
    let template = BayesOptimizer::new_strategy(
        bounds.clone(),
        cfg.clone(),
        Strategy::Experimental(t_state),
        &mut StdRng::seed_from_u64(3),
    )
    .unwrap();

    let mut l_rng = StdRng::seed_from_u64(33);
    let l_state = StrategyState::configured(3, mcfg, 6, &mut l_rng).unwrap();
    let mut live =
        BayesOptimizer::new_strategy(bounds, cfg, Strategy::Experimental(l_state), &mut l_rng)
            .unwrap();

    // Warm-up
    let (next_live, x_warm) = live
        .ask_natural(6, &mut StdRng::seed_from_u64(100))
        .unwrap();
    live = next_live;
    let warm_ids = live.pending_ids().unwrap();
    let mut y_warm = Array2::zeros((6, 1));
    for r in 0..6 {
        y_warm[[r, 0]] = -x_warm.row(r).iter().map(|v| v * v).sum::<f64>();
    }
    live = live
        .complete_ids(
            &warm_ids,
            &y_warm.view(),
            None,
            &mut StdRng::seed_from_u64(101),
        )
        .unwrap();

    // Overlapping regional asks with partial completion and cancellation
    let (next_live, x_first) = live
        .ask_natural(2, &mut StdRng::seed_from_u64(102))
        .unwrap();
    live = next_live;
    let ids_first = live.pending_ids().unwrap();
    let (next_live, x_second) = live
        .ask_natural(2, &mut StdRng::seed_from_u64(103))
        .unwrap();
    live = next_live;
    let all_ids = live.pending_ids().unwrap();
    let ids_second = all_ids[2..].to_vec();

    let y_one = array![[-x_second.row(0).iter().map(|v| v * v).sum::<f64>()]];
    live = live
        .complete_ids(
            &[ids_second[0]],
            &y_one.view(),
            None,
            &mut StdRng::seed_from_u64(104),
        )
        .unwrap();
    live = live
        .cancel_ids(&[ids_second[1]], &mut StdRng::seed_from_u64(105))
        .unwrap();

    // Freeze while `ids_first` are STILL pending!
    assert_eq!(live.pending_ids(), Some(ids_first.clone()));
    let frozen = live.freeze().unwrap();
    let mut restored = template.thaw(&frozen).unwrap();

    assert_eq!(restored.pending_ids(), live.pending_ids());
    assert_eq!(restored.digest(), live.digest());
    assert_eq!(restored.freeze().unwrap(), frozen);

    // Complete remaining `ids_first` on both live and restored, then continue
    // rollouts
    let mut y_rem = Array2::zeros((2, 1));
    for r in 0..2 {
        y_rem[[r, 0]] = -x_first
            .row(r)
            .iter()
            .map(|v| (v - 0.25).powi(2))
            .sum::<f64>();
    }
    live = live
        .complete_ids(
            &ids_first,
            &y_rem.view(),
            None,
            &mut StdRng::seed_from_u64(106),
        )
        .unwrap();
    restored = restored
        .complete_ids(
            &ids_first,
            &y_rem.view(),
            None,
            &mut StdRng::seed_from_u64(106),
        )
        .unwrap();
    assert_eq!(restored.pending_ids(), None);
    assert_eq!(restored.digest(), live.digest());

    for step in 200..203u64 {
        let mut rng_a = StdRng::seed_from_u64(step);
        let mut rng_b = StdRng::seed_from_u64(step);
        let (next_live, xa) = live.ask_natural(2, &mut rng_a).unwrap();
        live = next_live;
        let (next_rest, xb) = restored.ask_natural(2, &mut rng_b).unwrap();
        restored = next_rest;
        assert!(bits_equal(&xa, &xb));
        assert_eq!(restored.digest(), live.digest());

        let ids_a = live.pending_ids().unwrap();
        let ids_b = restored.pending_ids().unwrap();
        let mut y = Array2::zeros((2, 1));
        for r in 0..2 {
            y[[r, 0]] = -xa.row(r).iter().map(|v| v * v).sum::<f64>();
        }
        live = live
            .complete_ids(&ids_a, &y.view(), None, &mut rng_a)
            .unwrap();
        restored = restored
            .complete_ids(&ids_b, &y.view(), None, &mut rng_b)
            .unwrap();
        assert_eq!(restored.digest(), live.digest());
    }
}

#[test]
fn thaw_adversarial() {
    let bounds = array![[-1.0, 1.0], [0.0, 2.0]];
    let mut rng = StdRng::seed_from_u64(19);
    let mut opt = BayesOptimizer::new(bounds, turbo_zero(), &mut rng).unwrap();
    let (next, x) = opt.ask_natural(2, &mut rng).unwrap();
    opt = next;
    opt = opt
        .tell_natural(&x.view(), &array![[1.0], [2.0]].view(), None, &mut rng)
        .unwrap();
    let valid = opt.freeze().unwrap();

    // 1. Empty and truncated inputs
    assert!(opt.thaw(&[]).is_err());
    assert!(opt.thaw(&[0u8; 47]).is_err());
    assert!(opt.thaw(&valid[..valid.len() - 1]).is_err());

    // 2. Legacy BAUXFRZ1 magic with valid BLAKE3 tag
    let mut legacy = b"BAUXFRZ1".to_vec();
    legacy.extend_from_slice(&[0u8; 32]);
    assert!(opt.thaw(&seal_test(legacy)).is_err());

    // 3. Unsupported wire version with valid BLAKE3 tag
    let mut bad_ver = valid[..valid.len() - 32].to_vec();
    bad_ver[8..16].copy_from_slice(&99u64.to_le_bytes());
    assert!(opt.thaw(&seal_test(bad_ver)).is_err());

    // 4. Dimension and bounds mismatch
    let other_dim = BayesOptimizer::new(
        array![[-1.0, 1.0]],
        turbo_zero(),
        &mut StdRng::seed_from_u64(1),
    )
    .unwrap();
    assert!(other_dim.thaw(&valid).is_err());
    let other_bounds = BayesOptimizer::new(
        array![[-1.0, 1.0], [0.0, 3.0]],
        turbo_zero(),
        &mut StdRng::seed_from_u64(1),
    )
    .unwrap();
    assert!(other_bounds.thaw(&valid).is_err());

    // 5. Invalid boolean tag (clamped = 2) with valid BLAKE3 tag
    let mut bad_bool = valid[..valid.len() - 32].to_vec();
    bad_bool[24] = 2;
    assert!(opt.thaw(&seal_test(bad_bool)).is_err());

    // 6. Crafted huge row count (u64::MAX and MAX_OBS) on tiny payload: must
    //    fail cleanly without OOM
    let mut huge_len = valid[..valid.len() - 32].to_vec();
    // Overwrite empirical observation count `n` with MAX_OBS (2_000_000) while
    // keeping payload tiny
    for pos in (40..huge_len.len().saturating_sub(16)).step_by(1) {
        if u64::from_le_bytes(huge_len[pos..pos + 8].try_into().unwrap()) == 2
            && u64::from_le_bytes(huge_len[pos + 8..pos + 16].try_into().unwrap()) == 1
        {
            huge_len[pos..pos + 8].copy_from_slice(&(2_000_000u64).to_le_bytes());
            break;
        }
    }
    assert!(opt.thaw(&seal_test(huge_len)).is_err());

    // 7. Trailing garbage byte with recomputed valid BLAKE3 tag
    let mut trailing = valid[..valid.len() - 32].to_vec();
    trailing.push(0xAA);
    assert!(opt.thaw(&seal_test(trailing)).is_err());
}

#[test]
fn fork_isolation() {
    use crate::config::turbo_enn;
    use crate::optimizer::{MultiTrustRegionConfig, StrategyState};
    use crate::trust_region::TRLengthConfig;

    let bounds = array![[-2.0, 2.0], [-2.0, 2.0]];
    let cfg = turbo_enn();
    let mcfg = MultiTrustRegionConfig::new(2, TRLengthConfig::default());
    let mut rng = StdRng::seed_from_u64(42);

    let state = StrategyState::configured(2, mcfg, 4, &mut rng).unwrap();
    let opt =
        BayesOptimizer::new_strategy(bounds, cfg, Strategy::Experimental(state), &mut rng).unwrap();

    // Fork root before any asks
    let fork_a = opt.fork().unwrap();
    let fork_b = opt.fork().unwrap();

    // Both forks should have identical digests initially
    assert_eq!(fork_a.digest(), fork_b.digest());

    // Propose from fork_a with a deterministic RNG
    let mut rng_a = StdRng::seed_from_u64(100);
    let (prop_a1, fork_a) = fork_a.ask_ticket(2, &mut rng_a).unwrap();

    // Propose from fork_b with the exact same deterministic RNG
    let mut rng_b = StdRng::seed_from_u64(100);
    let (prop_b1, fork_b) = fork_b.ask_ticket(2, &mut rng_b).unwrap();

    // Both should produce identical proposal coordinates and deterministic
    // evaluation IDs
    assert_eq!(prop_a1.as_array(), prop_b1.as_array());
    assert_eq!(prop_a1.ids(), prop_b1.ids());

    // Tell divergent evaluations to fork_a vs fork_b
    let y_a = array![[10.0], [20.0]];
    let y_b = array![[-10.0], [-20.0]];
    let fork_a = fork_a
        .tell_ticket(prop_a1, &y_a.view(), None, &mut rng_a)
        .unwrap();
    let fork_b = fork_b
        .tell_ticket(prop_b1, &y_b.view(), None, &mut rng_b)
        .unwrap();

    // Divergent branches must not interfere: digests must differ!
    assert_ne!(fork_a.digest(), fork_b.digest());
    assert_eq!(fork_a.obs_count(), 2);
    assert_eq!(fork_b.obs_count(), 2);

    // Root optimizer was never mutated (remains 0 observations)
    assert_eq!(opt.obs_count(), 0);

    // Sub-forking from fork_a produces another isolated branch
    let fork_a_sub = fork_a.fork().unwrap();
    assert_eq!(fork_a.digest(), fork_a_sub.digest());
    assert!(fork_a.shares_storage(&fork_a_sub));
}

#[test]
fn fbip_lifecycle() {
    let bounds = array![[0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0]];
    let mut rng = StdRng::seed_from_u64(314);

    // 1. Empirical (`turbo_zero`) O(1) structural fork & FBIP in-place append
    let mut emp = BayesOptimizer::new_strategy(
        bounds.clone(),
        crate::turbo_zero(),
        Strategy::hybrid(crate::config::InitStrategy::LHD, 2),
        &mut rng,
    )
    .unwrap();
    assert_eq!(emp.warmup.mode, crate::optimizer::Lifecycle::Warmup);
    let (next_emp, x0) = emp.ask(2, &mut rng).unwrap();
    emp = next_emp
        .tell(&x0.view(), &array![[1.0], [2.0]].view(), None, &mut rng)
        .unwrap();
    assert_eq!(emp.warmup.mode, crate::optimizer::Lifecycle::Active);
    assert!(emp.ingestion_failure().is_none());

    let emp_fork = emp.fork().unwrap();
    assert!(emp.shares_storage(&emp_fork));

    // 2. Built-in ENN (`turbo_enn`) O(1) structural fork & FBIP slab sharing
    let mut enn_opt = BayesOptimizer::new_strategy(
        bounds,
        crate::turbo_enn(),
        Strategy::hybrid(crate::config::InitStrategy::LHD, 3),
        &mut rng,
    )
    .unwrap();
    let (next_enn, x_init) = enn_opt.ask(3, &mut rng).unwrap();
    enn_opt = next_enn
        .tell(
            &x_init.view(),
            &array![[0.5], [1.5], [2.5]].view(),
            None,
            &mut rng,
        )
        .unwrap();
    let enn_fork = enn_opt.fork().unwrap();
    assert!(enn_opt.shares_storage(&enn_fork));
    assert_eq!(enn_opt.digest(), enn_fork.digest());
}
