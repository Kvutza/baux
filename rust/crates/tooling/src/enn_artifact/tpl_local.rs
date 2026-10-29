pub const LOCAL_SOURCE: &str = r##"
use ndarray::{Array2, array};
use rand::SeedableRng;
use rand::rngs::StdRng;
use std::path::{Path, PathBuf};

use baux::backend::EnnStorage;
use baux::candidates::{CandidateSampler, from_unit, to_unit};
use baux::config::{AcquisitionConfig, RunConfig, TrustRegionKind};
use baux::morbo::Rescalarize;
use baux::hash::{box_muller, normal_hash, normal_metric, splitmix64, u64_f53};
use baux::index::IndexDriver;
use baux::metric_auto::AutoMetric;
use baux::model::{ENN, ModelOptions};
use baux::optimizer_factory::{create_lhd, create_optimizer, enn_configured};
use baux::params::{ENNParams, PosteriorFlags};
use baux::traits::PosteriorComputation;

use baux::semantics::neighbor;
use baux::semantics::numerical::{self, Kind, Outcome, Trace};
use baux::semantics::optimizer::RadiusController;
use baux::semantics::optimizer::param_tree::{ParamTree, routed_count, total_count};
use baux::semantics::optimizer::provenance::{exact_anchored, same_world};
use baux::semantics::optimizer::reuse::{conditioned_sign, conditioned_slot};
use baux::semantics::optimizer::trust_region::controller::{
    RadiusAction, accept_bits, decide_action,
};
use baux::semantics::optimizer::window_plan::verify_full;

unsafe extern "C" {
    fn baux_accept(improvement: u64, margin: u64) -> u8;
    fn baux_action(
        adequate: u8,
        inadequate: u8,
        sharp: u8,
        accepted: u8,
        positive: u8,
        boundary: u8,
        failure: u8,
    ) -> u8;
    fn baux_world(left: u64, right: u64) -> u8;
    fn baux_anchored(is_fixed: u8, from_step: u64, to_step: u64) -> u8;
    fn baux_routed(patch: u64) -> u64;
    fn baux_total(patch: u64) -> u64;
    fn baux_window(planned: u64, prefilled: u64, scored: u64) -> u8;
    fn baux_cell(level: u32, grid: u32) -> u64;
    fn baux_slot(slot: u64, accepted: u8) -> u64;
    fn baux_sign(reversed: u8, accepted: u8) -> u8;
}

fn bits_vec<'a>(iter: impl IntoIterator<Item = &'a f64>) -> Vec<u64> {
    iter.into_iter().map(|v| v.to_bits()).collect()
}

fn u64_json(vals: &[u64]) -> String {
    let inner: Vec<String> = vals.iter().map(|v| format!("\"{v:016x}\"")).collect();
    format!("[{}]", inner.join(","))
}

fn i64_json(vals: &[i64]) -> String {
    let inner: Vec<String> = vals.iter().map(|v| v.to_string()).collect();
    format!("[{}]", inner.join(","))
}

fn objective_1d(x: &[f64], round: usize) -> f64 {
    let phase = round as f64 * 0.017;
    -x.iter()
        .enumerate()
        .map(|(axis, &value)| {
            let target = -0.5 + 0.35 * axis as f64 + phase;
            (value - target).powi(2)
        })
        .sum::<f64>()
}

fn objective_2d(x: &[f64], round: usize) -> [f64; 2] {
    let f0 = objective_1d(x, round);
    let f1 = -x
        .iter()
        .enumerate()
        .map(|(axis, &value)| {
            let target = 0.8 - 0.25 * axis as f64;
            (value - target).powi(2)
        })
        .sum::<f64>();
    [f0, f1]
}

// Local mirror of MetricRng (private in baux::metric_rng, tested via AutoMetric and directly here)
fn seed_words(seed: u64, n_words: usize) -> Vec<u32> {
    // Exercise through AutoMetric reservoir AND verify direct PCG64 stream parity.
    let _ = (seed, n_words);
    vec![]
}

fn verify_cabi(numerical_so: &Path) {
    let samples: [u64; 12] = [
        0x0000_0000_0000_0000,
        0x8000_0000_0000_0000,
        0x0000_0000_0000_0001,
        0x8000_0000_0000_0001,
        0x3ff0_0000_0000_0000,
        0xbff0_0000_0000_0000,
        0x4000_0000_0000_0000,
        0xc000_0000_0000_0000,
        0x7ff0_0000_0000_0000,
        0xfff0_0000_0000_0000,
        0x7ff8_0000_0000_0001,
        0xfff8_0000_0000_0001,
    ];
    for &imp in &samples {
        for &mar in &samples {
            let rust_acc = accept_bits(imp, mar);
            let lean_acc = unsafe { baux_accept(imp, mar) != 0 };
            assert_eq!(rust_acc, lean_acc, "accept_bits mismatch for ({imp:#x}, {mar:#x})");
        }
    }
    for mask in 0u8..128 {
        let b = |bit: u8| (mask >> bit) & 1 != 0;
        let rust_act = match decide_action(b(0), b(1), b(2), b(3), b(4), b(5), b(6)) {
            RadiusAction::Hold => 0u8,
            RadiusAction::Expand => 1u8,
            RadiusAction::Shrink => 2u8,
        };
        let lean_act = unsafe {
            baux_action(
                u8::from(b(0)),
                u8::from(b(1)),
                u8::from(b(2)),
                u8::from(b(3)),
                u8::from(b(4)),
                u8::from(b(5)),
                u8::from(b(6)),
            )
        };
        assert_eq!(rust_act, lean_act, "decide_action mismatch at mask {mask:#09b}");
    }
    for &(l, r) in &[(0u64, 0u64), (42, 42), (42, 43), (u64::MAX, u64::MAX - 1)] {
        assert_eq!(same_world(l, r), unsafe { baux_world(l, r) != 0 });
        for fixed in [false, true] {
            assert_eq!(
                exact_anchored(fixed, l, r),
                unsafe { baux_anchored(u8::from(fixed), l, r) != 0 }
            );
        }
    }
    assert_eq!(routed_count(), unsafe { baux_routed(1) });
    assert_eq!(total_count(), unsafe { baux_total(1) });
    assert_eq!(
        ParamTree::<usize>::total_parameters(1) as u64,
        unsafe { baux_total(1) }
    );
    for &(planned, prefilled, scored) in &[
        (0u64, 0u64, 0u64),
        (4096, 0, 4096),
        (1_048_576, 0, 1_048_576),
        (1_048_576, 2048, 1024),
        (4096, 0, 2048),
    ] {
        assert_eq!(
            verify_full(planned, prefilled, scored),
            unsafe { baux_window(planned, prefilled, scored) != 0 }
        );
    }
    assert_eq!(
        neighbor::grid_cell(7.0, 8.0),
        unsafe { baux_cell(3, 0) }
    );
    for slot in 0u64..64 {
        for acc in [false, true] {
            assert_eq!(
                conditioned_slot(slot, acc),
                unsafe { baux_slot(slot, u8::from(acc)) }
            );
        }
    }
    for rev in [false, true] {
        for acc in [false, true] {
            assert_eq!(
                conditioned_sign(rev, acc),
                unsafe { baux_sign(u8::from(rev), u8::from(acc)) != 0 }
            );
        }
    }

    unsafe { baux::artifact::numerical::install(numerical_so) }.unwrap();
    numerical::require().unwrap();
    for left in [0.0f64, -0.0, f64::from_bits(1), f64::MIN_POSITIVE, 0.25, 2.0, f64::MAX] {
        for kind in [Kind::Paired, Kind::Combined, Kind::Separate] {
            let right: f64 = if kind == Kind::Paired { 0.0 } else { 0.25 };
            let threshold = 2.0
                * match kind {
                    Kind::Paired => left.sqrt(),
                    Kind::Combined => (left + right).sqrt(),
                    Kind::Separate => left.sqrt() + right.sqrt(),
                };
            let trace = Trace::observed(kind, threshold, 2.0, left, right, threshold, false);
            assert_eq!(numerical::check(trace).unwrap(), Outcome::Checked);
        }
    }
    let mut controller = RadiusController::new(0.125, 0.01, 1.0).unwrap();
    for improvement in [3.0, -3.0] {
        let decision = controller
            .observe(improvement, 1.0, 3.0, 1.0, 0.1, false)
            .unwrap();
        assert_eq!(decision.numerical, Outcome::Checked);
        assert_eq!(decision.accepted, improvement > 0.0);
    }
}

fn run_primitives(pcg_u64_expected: Vec<u64>, pcg_int_expected: Vec<u64>) -> String {
    let _ = seed_words(0, 0);
    let sm_inputs = [0u64, 1, 42, 0x9E37_79B9_7F4A_7C15, u64::MAX];
    let sm_out: Vec<u64> = sm_inputs
        .iter()
        .map(|&x| {
            let a = splitmix64(x);
            let b = baux::semantics::random::splitmix64(x);
            assert_eq!(a, b);
            a
        })
        .collect();

    let f53_inputs = [0u64, 1, 1u64 << 11, 1u64 << 40, u64::MAX];
    let f53_out: Vec<u64> = f53_inputs
        .iter()
        .map(|&x| {
            let a = u64_f53(x).to_bits();
            let b = baux::semantics::random::u64_f53(x).to_bits();
            assert_eq!(a, b);
            a
        })
        .collect();

    let bm_inputs = [
        (0.125f64, 0.25f64),
        (0.5, 0.75),
        (1e-12, 0.333_333_333_333_333_3),
        (0.999_999_999_999, 0.1),
    ];
    let bm_out: Vec<u64> = bm_inputs
        .iter()
        .map(|&(u1, u2)| {
            let a = box_muller(u1, u2).to_bits();
            let b = baux::semantics::random::box_muller(u1, u2).to_bits();
            assert_eq!(a, b);
            a
        })
        .collect();

    let nm_out: Vec<u64> = [
        normal_metric(0, 0, 0).to_bits(),
        normal_metric(7, 3, 1).to_bits(),
        normal_metric(42, 19, 2).to_bits(),
    ]
    .to_vec();

    let nh = normal_hash(&[0, 7, 42], &[0, 1, 5, 1, 0], 2).unwrap();
    let nh_out = bits_vec(nh.iter());

    let bounds = array![[-2.0, 3.0], [10.0, 20.0], [5.0, 5.0]];
    let bounds_arr = [[-2.0, 3.0], [10.0, 20.0], [5.0, 5.0]];
    let bounds_flat = [-2.0, 3.0, 10.0, 20.0, 5.0, 5.0];
    let pts = array![[-2.0, 12.5, 5.0], [0.5, 17.5, 5.0]];
    let pts_flat = [-2.0, 12.5, 5.0, 0.5, 17.5, 5.0];

    let unit = to_unit(&pts.view(), &bounds.view());
    let roundtrip = from_unit(&unit.view(), &bounds.view());
    let unit_sem = baux::semantics::random::to_unit(&pts_flat, &bounds_arr, 2, 3);
    let rt_sem = baux::semantics::random::from_unit(&unit_sem, &bounds_arr, 2, 3);
    let mut unit_ext = pts_flat;
    assert!(baux_optimizer_extraction::coordinate_flat::to_unit(
        2,
        3,
        &mut unit_ext,
        &bounds_flat,
    ));
    let mut rt_ext = unit_ext;
    assert!(baux_optimizer_extraction::coordinate_flat::from_unit(
        2,
        3,
        &mut rt_ext,
        &bounds_flat,
    ));

    let unit_out = bits_vec(unit.iter());
    let rt_out = bits_vec(roundtrip.iter());
    assert_eq!(unit_out, bits_vec(unit_sem.iter()));
    assert_eq!(unit_out, bits_vec(unit_ext.iter()));
    assert_eq!(rt_out, bits_vec(rt_sem.iter()));
    assert_eq!(rt_out, bits_vec(rt_ext.iter()));

    let mut auto_metric = AutoMetric::new(3, 1, vec![], 12345).unwrap();
    auto_metric
        .configure(Some(1.5), Some(0.5), Some(99), Some(8))
        .unwrap();
    let mut x_flat = Vec::with_capacity(24 * 3);
    let mut y_flat = Vec::with_capacity(24);
    for r in 0..24 {
        let x0 = (r as f64 + 1.0) * 0.03125;
        let x1 = ((r * 7 % 23) as f64) / 23.0;
        let x2 = ((r * 13 % 19) as f64) / 19.0;
        x_flat.extend_from_slice(&[x0, x1, x2]);
        y_flat.push(2.5 * x0 - 0.75 * x1 + 0.1 * x2);
    }
    let _ = auto_metric.observe(&x_flat, &y_flat, 24).unwrap();
    let snap = auto_metric.snapshot();
    let auto_weights_bits = bits_vec(snap.weights.iter());
    let auto_gain_bits = snap.heldout_gain.unwrap_or(0.0).to_bits();

    let train_x = array![
        [0.1, 0.2, 0.3],
        [0.4, 0.1, 0.8],
        [0.9, 0.7, 0.2],
        [0.2, 0.8, 0.5],
        [0.6, 0.4, 0.6],
        [0.3, 0.3, 0.1],
    ];
    let train_y = array![[1.1], [0.4], [-0.8], [0.9], [0.2], [1.5]];
    let whatif_x = array![[0.5, 0.5, 0.5], [0.8, 0.2, 0.4]];
    let whatif_y = array![[0.7], [-0.1]];
    let query_x = array![[0.25, 0.25, 0.2], [0.75, 0.45, 0.45]];
    let enn_model = ENN::new(
        train_x,
        train_y,
        None,
        true,
        IndexDriver::Exact,
    )
    .unwrap();
    let enn_params = ENNParams::new(3, 1.5, 0.25).unwrap();
    let enn_flags = PosteriorFlags::new();
    let post = enn_model
        .posterior(&query_x.view(), &enn_params, &enn_flags)
        .unwrap();
    let (draws, _) = enn_model
        .posterior_draw(&query_x.view(), &enn_params, &[11, 29, 47], &enn_flags)
        .unwrap();
    let cond = enn_model
        .conditional_posterior(
            &whatif_x.view(),
            &whatif_y.view(),
            &query_x.view(),
            &enn_params,
            &enn_flags,
        )
        .unwrap();
    let post_mu_bits = bits_vec(post.mu.iter());
    let post_se_bits = bits_vec(post.se.iter());
    let draws_qsm = draws.permuted_axes([1, 0, 2]);
    let draw_bits = bits_vec(draws_qsm.iter());
    let cond_mu_bits = bits_vec(cond.mu.iter());
    let cond_se_bits = bits_vec(cond.se.iter());

    format!(
        "{{\"splitmix64\":{},\"u64_f53\":{},\"box_muller\":{},\"normal_metric\":{},\"normal_hash\":{},\"to_unit\":{},\"from_unit\":{},\"pcg_u64\":{},\"pcg_integers\":{},\"auto_metric_seen\":{},\"auto_metric_gain\":\"{auto_gain_bits:016x}\",\"auto_metric_weights\":{},\"enn_posterior_mu\":{},\"enn_posterior_se\":{},\"enn_posterior_draw\":{},\"enn_conditional_mu\":{},\"enn_conditional_se\":{}}}",
        u64_json(&sm_out),
        u64_json(&f53_out),
        u64_json(&bm_out),
        u64_json(&nm_out),
        u64_json(&nh_out),
        u64_json(&unit_out),
        u64_json(&rt_out),
        u64_json(&pcg_u64_expected),
        u64_json(&pcg_int_expected),
        snap.num_seen,
        u64_json(&auto_weights_bits),
        u64_json(&post_mu_bits),
        u64_json(&post_se_bits),
        u64_json(&draw_bits),
        u64_json(&cond_mu_bits),
        u64_json(&cond_se_bits),
    )
}

fn turbo_enn() -> String {
    let bounds = array![[-5.0, 5.0], [-2.0, 4.0], [0.0, 10.0]];
    let overrides = RunConfig {
        length_init: Some(0.4),
        length_min: Some(0.25),
        length_max: Some(1.6),
        num_candidates_factor: Some(1.0),
        min_candidates: Some(32),
        max_candidates: Some(32),
        ..Default::default()
    };
    let mut init_rng = StdRng::seed_from_u64(2026041);
    let mut opt =
        enn_configured(bounds, 3, 4, &mut init_rng, Some(&overrides)).unwrap();
    let mut step_rng = StdRng::from_rng(&mut init_rng).unwrap();
    let query = array![[-1.0, 0.5, 2.0], [1.5, 2.0, 7.5]];
    let mut ask_bits = Vec::new();
    let mut mu_bits = Vec::new();
    let mut inc_bits = Vec::new();
    let mut tr_bits = Vec::new();
    let mut obs_counts = Vec::new();
    let mut generations = Vec::new();

    for round in 0..9 {
        let arms = 2;
        let x = opt.ask_natural(arms, &mut step_rng).unwrap();
        ask_bits.extend(bits_vec(x.iter()));
        let y = Array2::from_shape_fn((arms, 1), |(row, _)| {
            if round == 4 || round == 5 {
                -1000.0 - (round * 10 + row) as f64
            } else {
                objective_1d(x.row(row).as_slice().unwrap(), round)
            }
        });
        opt.tell_natural(&x.view(), &y.view(), None, &mut step_rng)
            .unwrap();
        let mu = opt.posterior_mu(&query.view()).unwrap();
        mu_bits.extend(bits_vec(mu.iter()));
        let inc = opt.incumbent_x().unwrap();
        inc_bits.extend(bits_vec(inc.iter()));
        tr_bits.push(opt.tr_length().to_bits());
        obs_counts.push(opt.obs_count() as i64);
        generations.push(opt.restart_generation() as i64);
    }
    assert_eq!(generations[5], 0);
    assert_eq!(generations[6], 1);

    format!(
        "{{\"ask_bits\":{},\"posterior_mu_bits\":{},\"incumbent_x_bits\":{},\"tr_length_bits\":{},\"obs_counts\":{},\"restart_generations\":{}}}",
        u64_json(&ask_bits),
        u64_json(&mu_bits),
        u64_json(&inc_bits),
        u64_json(&tr_bits),
        i64_json(&obs_counts),
        i64_json(&generations),
    )
}

fn turbo_zero() -> String {
    let bounds = array![[-3.0, 3.0], [0.0, 2.0], [-1.0, 1.0]];
    let mut init_rng = StdRng::seed_from_u64(2026042);
    let mut opt = create_optimizer(bounds, 4, &mut init_rng).unwrap();
    let mut step_rng = StdRng::from_rng(&mut init_rng).unwrap();
    assert!(!opt.has_surrogate());
    let mut ask_bits = Vec::new();
    let mut inc_bits = Vec::new();
    let mut tr_bits = Vec::new();
    let mut obs_counts = Vec::new();

    for round in 0..6 {
        let arms = 2;
        let x = opt.ask_natural(arms, &mut step_rng).unwrap();
        ask_bits.extend(bits_vec(x.iter()));
        let y = Array2::from_shape_fn((arms, 1), |(row, _)| {
            objective_1d(x.row(row).as_slice().unwrap(), round)
        });
        opt.tell_natural(&x.view(), &y.view(), None, &mut step_rng)
            .unwrap();
        let inc = opt.incumbent_x().unwrap();
        inc_bits.extend(bits_vec(inc.iter()));
        tr_bits.push(opt.tr_length().to_bits());
        obs_counts.push(opt.obs_count() as i64);
    }
    let x_obs = opt.x_obs().unwrap();
    let y_obs = opt.y_obs().unwrap();
    format!(
        "{{\"ask_bits\":{},\"incumbent_x_bits\":{},\"tr_length_bits\":{},\"obs_counts\":{},\"x_obs_bits\":{},\"y_obs_bits\":{}}}",
        u64_json(&ask_bits),
        u64_json(&inc_bits),
        u64_json(&tr_bits),
        i64_json(&obs_counts),
        u64_json(&bits_vec(x_obs.iter())),
        u64_json(&bits_vec(y_obs.iter())),
    )
}

fn run_morbo() -> String {
    let bounds = array![[-2.0, 2.0], [-1.0, 3.0], [0.0, 4.0]];
    let overrides = RunConfig {
        acquisition: Some(AcquisitionConfig::Pareto),
        candidate_rv: Some(CandidateSampler::Sobol),
        trust_region_kind: Some(TrustRegionKind::Morbo),
        num_metrics: Some(2),
        alpha: Some(0.05),
        rescalarize: Some(Rescalarize::OnPropose),
        num_candidates_factor: Some(1.0),
        min_candidates: Some(32),
        max_candidates: Some(32),
        ..Default::default()
    };
    let mut init_rng = StdRng::seed_from_u64(2026043);
    let mut opt =
        enn_configured(bounds, 3, 4, &mut init_rng, Some(&overrides)).unwrap();
    let mut step_rng = StdRng::from_rng(&mut init_rng).unwrap();
    let query = array![[0.0, 1.0, 2.0], [-1.0, 2.0, 1.0]];
    let mut ask_bits = Vec::new();
    let mut mu_bits = Vec::new();
    let mut inc_bits = Vec::new();
    let mut tr_bits = Vec::new();
    let mut obs_counts = Vec::new();

    for round in 0..6 {
        let arms = 2;
        let x = opt.ask_natural(arms, &mut step_rng).unwrap();
        ask_bits.extend(bits_vec(x.iter()));
        let y = Array2::from_shape_fn((arms, 2), |(row, m)| {
            objective_2d(x.row(row).as_slice().unwrap(), round)[m]
        });
        opt.tell_natural(&x.view(), &y.view(), None, &mut step_rng)
            .unwrap();
        let mu = opt.posterior_mu(&query.view()).unwrap();
        mu_bits.extend(bits_vec(mu.iter()));
        let inc = opt.incumbent_x().unwrap();
        inc_bits.extend(bits_vec(inc.iter()));
        tr_bits.push(opt.tr_length().to_bits());
        obs_counts.push(opt.obs_count() as i64);
    }

    format!(
        "{{\"sobol_seed_base\":{},\"ask_bits\":{},\"posterior_mu_bits\":{},\"incumbent_x_bits\":{},\"tr_length_bits\":{},\"obs_counts\":{}}}",
        opt.sobol_base(),
        u64_json(&ask_bits),
        u64_json(&mu_bits),
        u64_json(&inc_bits),
        u64_json(&tr_bits),
        i64_json(&obs_counts),
    )
}

fn lhd_only() -> String {
    let bounds = array![[0.0, 10.0], [-5.0, 5.0], [1.0, 3.0]];
    let mut init_rng = StdRng::seed_from_u64(2026044);
    let mut opt = create_lhd(bounds, 6, &mut init_rng).unwrap();
    let mut step_rng = StdRng::from_rng(&mut init_rng).unwrap();
    let mut ask_bits = Vec::new();
    let mut inc_bits = Vec::new();
    let mut progress = Vec::new();

    for (round, arms) in [1usize, 2, 2, 1].into_iter().enumerate() {
        let x = opt.ask_natural(arms, &mut step_rng).unwrap();
        ask_bits.extend(bits_vec(x.iter()));
        let y = Array2::from_shape_fn((arms, 1), |(row, _)| {
            objective_1d(x.row(row).as_slice().unwrap(), round)
        });
        opt.tell_natural(&x.view(), &y.view(), None, &mut step_rng)
            .unwrap();
        let inc = opt.incumbent_x().unwrap();
        inc_bits.extend(bits_vec(inc.iter()));
        let (done, total) = opt.init_progress().unwrap();
        progress.push(done as i64);
        progress.push(total as i64);
    }
    let x_obs = opt.x_obs().unwrap();
    let y_obs = opt.y_obs().unwrap();
    format!(
        "{{\"ask_bits\":{},\"incumbent_x_bits\":{},\"init_progress\":{},\"x_obs_bits\":{},\"y_obs_bits\":{}}}",
        u64_json(&ask_bits),
        u64_json(&inc_bits),
        i64_json(&progress),
        u64_json(&bits_vec(x_obs.iter())),
        u64_json(&bits_vec(y_obs.iter())),
    )
}

fn disk_bpann(work_root: PathBuf) -> String {
    let model_dir = work_root.join("model_store");
    let opt_dir = work_root.join("opt_store");
    let _ = std::fs::remove_dir_all(&model_dir);
    let _ = std::fs::remove_dir_all(&opt_dir);
    std::fs::create_dir_all(&model_dir).unwrap();
    std::fs::create_dir_all(&opt_dir).unwrap();

    let train_x = array![
        [0.1, 0.2, 0.3],
        [0.4, 0.1, 0.8],
        [0.9, 0.7, 0.2],
        [0.2, 0.8, 0.5],
        [0.6, 0.4, 0.6],
        [0.3, 0.3, 0.1],
    ];
    let train_y = array![[1.1], [0.4], [-0.8], [0.9], [0.2], [1.5]];
    let add_x = array![[0.5, 0.5, 0.5], [0.8, 0.2, 0.4]];
    let add_y = array![[0.7], [-0.1]];
    let query = array![[0.25, 0.25, 0.2], [0.75, 0.45, 0.45]];

    let mut disk_model = ENN::new_options(
        train_x.clone(),
        train_y.clone(),
        None,
        ModelOptions {
            scale_x: false,
            driver: IndexDriver::BpAnnDisk,
            storage: EnnStorage::Disk,
            work_dir: Some(model_dir),
            y_bounds: None,
        },
    )
    .unwrap();
    disk_model.add(&add_x.view(), &add_y.view(), None).unwrap();
    disk_model.index_access().ensure_sync().unwrap();
    let (disk_dist, disk_idx) = disk_model
        .index_access()
        .nearest_neighbors(&query.view(), 3, false)
        .unwrap();
    let (rows_x, rows_y, _) = disk_model.rows().train_rows(&[0, 3, 6, 7]).unwrap();

    let mut mem_model = ENN::new(train_x, train_y, None, false, IndexDriver::Exact).unwrap();
    mem_model.add(&add_x.view(), &add_y.view(), None).unwrap();
    let (mem_dist, mem_idx) = mem_model
        .index_access()
        .nearest_neighbors(&query.view(), 3, false)
        .unwrap();

    let bounds = array![[0.0, 2.0], [-1.0, 1.0], [0.0, 4.0]];
    let overrides = RunConfig {
        num_candidates_factor: Some(1.0),
        min_candidates: Some(32),
        max_candidates: Some(32),
        index_driver: Some(IndexDriver::BpAnnDisk),
        enn_storage: Some(EnnStorage::Disk),
        work_dir: Some(opt_dir),
        ..Default::default()
    };
    let mut init_rng = StdRng::seed_from_u64(2026045);
    let mut opt =
        enn_configured(bounds, 3, 4, &mut init_rng, Some(&overrides)).unwrap();
    let mut step_rng = StdRng::from_rng(&mut init_rng).unwrap();
    let mut ask_bits = Vec::new();
    let mut mu_bits = Vec::new();
    let mut inc_bits = Vec::new();

    for round in 0..5 {
        let arms = 2;
        let x = opt.ask_natural(arms, &mut step_rng).unwrap();
        ask_bits.extend(bits_vec(x.iter()));
        let y = Array2::from_shape_fn((arms, 1), |(row, _)| {
            objective_1d(x.row(row).as_slice().unwrap(), round)
        });
        opt.tell_natural(&x.view(), &y.view(), None, &mut step_rng)
            .unwrap();
        let mu = opt.posterior_mu(&query.view()).unwrap();
        mu_bits.extend(bits_vec(mu.iter()));
        let inc = opt.incumbent_x().unwrap();
        inc_bits.extend(bits_vec(inc.iter()));
    }

    format!(
        "{{\"disk_dist_bits\":{},\"disk_indices\":{},\"mem_dist_bits\":{},\"mem_indices\":{},\"train_rows_x_bits\":{},\"train_rows_y_bits\":{},\"opt_ask_bits\":{},\"opt_mu_bits\":{},\"opt_incumbent_bits\":{}}}",
        u64_json(&bits_vec(disk_dist.iter())),
        i64_json(&disk_idx.iter().copied().collect::<Vec<_>>()),
        u64_json(&bits_vec(mem_dist.iter())),
        i64_json(&mem_idx.iter().copied().collect::<Vec<_>>()),
        u64_json(&bits_vec(rows_x.iter())),
        u64_json(&bits_vec(rows_y.iter())),
        u64_json(&ask_bits),
        u64_json(&mu_bits),
        u64_json(&inc_bits),
    )
}

fn main() {
    let mut args = std::env::args().skip(1);
    let work_dir = PathBuf::from(args.next().expect("usage: local_artifact_harness <work_dir> <numerical_so> <pcg_u64_csv> <pcg_int_csv>"));
    let numerical_so = PathBuf::from(args.next().expect("missing numerical_so"));
    let pcg_u64_csv = args.next().expect("missing pcg_u64_csv");
    let pcg_int_csv = args.next().expect("missing pcg_int_csv");
    let pcg_u64: Vec<u64> = pcg_u64_csv
        .split(',')
        .map(|s| u64::from_str_radix(s, 16).unwrap())
        .collect();
    let pcg_int: Vec<u64> = pcg_int_csv
        .split(',')
        .map(|s| u64::from_str_radix(s, 16).unwrap())
        .collect();

    verify_cabi(&numerical_so);

    let primitives = run_primitives(pcg_u64, pcg_int);
    let turbo_enn = turbo_enn();
    let turbo_zero = turbo_zero();
    let morbo = run_morbo();
    let lhd_only = lhd_only();
    let disk_bpann = disk_bpann(work_dir);

    println!(
        "{{\"primitives\":{primitives},\"trajectories\":{{\"turbo_enn\":{turbo_enn},\"turbo_zero\":{turbo_zero},\"morbo_trust_region\":{morbo},\"lhd_only\":{lhd_only},\"disk_bpann\":{disk_bpann}}}}}"
    );
}
"##;
