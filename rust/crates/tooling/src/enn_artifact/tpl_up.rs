pub const UPSTREAM_SOURCE: &str = r##"
use ndarray::{Array2, array};
use rand::SeedableRng;
use rand::rngs::StdRng;
use std::path::PathBuf;

use ennbo::candidates::{CandidateRV, from_unit, to_unit};
use ennbo::config::{AcquisitionConfig, ConfigOverrides};
use ennbo::enn_overrides::EnnOverrides;
use ennbo::hash::{
    box_muller, normal_for_seed_index_metric, normal_hash_batch_multi_seed_fast, splitmix64,
    u64_to_f53,
};
use ennbo::index::IndexDriver;
use ennbo::layout::EnnLayout;
use ennbo::metric_auto::AutoMetric;
use ennbo::model::EpistemicNearestNeighbors;
use ennbo::morbo_override::MorboOverride;
use ennbo::morbo_trust_region::Rescalarize;
use ennbo::numpy_pcg::NumpyPcg64;
use ennbo::optimizer_factory::{
    create_optimizer_enn_with_overrides, create_optimizer_lhd, create_optimizer_zero,
};
use ennbo::params::{ENNParams, PosteriorFlags};
use ennbo::trust_region_config::TrustRegionKind;
use ennbo::EnnStorage;

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

fn run_primitives() -> String {
    let sm_inputs = [0u64, 1, 42, 0x9E37_79B9_7F4A_7C15, u64::MAX];
    let sm_out: Vec<u64> = sm_inputs.iter().map(|&x| splitmix64(x)).collect();

    let f53_inputs = [0u64, 1, 1u64 << 11, 1u64 << 40, u64::MAX];
    let f53_out: Vec<u64> = f53_inputs
        .iter()
        .map(|&x| u64_to_f53(x).to_bits())
        .collect();

    let bm_inputs = [
        (0.125f64, 0.25f64),
        (0.5, 0.75),
        (1e-12, 0.333_333_333_333_333_3),
        (0.999_999_999_999, 0.1),
    ];
    let bm_out: Vec<u64> = bm_inputs
        .iter()
        .map(|&(u1, u2)| box_muller(u1, u2).to_bits())
        .collect();

    let nm_out: Vec<u64> = [
        normal_for_seed_index_metric(0, 0, 0).to_bits(),
        normal_for_seed_index_metric(7, 3, 1).to_bits(),
        normal_for_seed_index_metric(42, 19, 2).to_bits(),
    ]
    .to_vec();

    let nh = normal_hash_batch_multi_seed_fast(&[0, 7, 42], &[0, 1, 5, 1, 0], 2).unwrap();
    let nh_out = bits_vec(nh.iter());

    let bounds = array![[-2.0, 3.0], [10.0, 20.0], [5.0, 5.0]];
    let pts = array![[-2.0, 12.5, 5.0], [0.5, 17.5, 5.0]];
    let unit = to_unit(&pts.view(), &bounds.view());
    let roundtrip = from_unit(&unit.view(), &bounds.view());
    let unit_out = bits_vec(unit.iter());
    let rt_out = bits_vec(roundtrip.iter());

    let mut pcg = NumpyPcg64::from_seed(2026041);
    let pcg_u64: Vec<u64> = (0..6).map(|_| pcg.next_u64()).collect();
    let pcg_int: Vec<u64> = [1u64, 7, 16, 1000, (1u64 << 32) + 99]
        .iter()
        .map(|&hi| pcg.integers_high(hi))
        .collect();

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
    let enn_model = EpistemicNearestNeighbors::new(
        train_x,
        train_y,
        None,
        true,
        IndexDriver::Flat,
    )
    .unwrap();
    let enn_params = ENNParams::new(3, 1.5, 0.25).unwrap();
    let enn_flags = PosteriorFlags::new();
    let post = enn_model
        .posterior(&query_x.view(), &enn_params, &enn_flags)
        .unwrap();
    let (draws, _) = enn_model
        .posterior_function_draw(&query_x.view(), &enn_params, &[11, 29, 47], &enn_flags)
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
    let draw_bits = bits_vec(draws.iter());
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
        u64_json(&pcg_u64),
        u64_json(&pcg_int),
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
    let overrides = ConfigOverrides {
        length_init: Some(0.4),
        length_min: Some(0.25),
        length_max: Some(1.6),
        min_candidates: Some(32),
        max_candidates: Some(32),
        num_candidates_per_dim: Some(1),
        ..Default::default()
    };
    let mut opt =
        create_optimizer_enn_with_overrides(bounds, Some(3), Some(4), 2026041, Some(&overrides))
            .unwrap();
    let query = array![[-1.0, 0.5, 2.0], [1.5, 2.0, 7.5]];
    let mut ask_bits = Vec::new();
    let mut mu_bits = Vec::new();
    let mut inc_bits = Vec::new();
    let mut tr_bits = Vec::new();
    let mut obs_counts = Vec::new();
    let mut generations = Vec::new();

    for round in 0..9 {
        let arms = 2;
        let x = opt.ask(arms).unwrap();
        ask_bits.extend(bits_vec(x.iter()));
        let y = Array2::from_shape_fn((arms, 1), |(row, _)| {
            if round == 4 || round == 5 {
                // Force two consecutive non-improving rounds to collapse trust region (0.4 -> 0.2 < 0.25).
                -1000.0 - (round * 10 + row) as f64
            } else {
                objective_1d(x.row(row).as_slice().unwrap(), round)
            }
        });
        opt.tell(&x.view(), &y.view(), None).unwrap();
        let mu = opt.posterior_mu(&query.view()).unwrap();
        mu_bits.extend(bits_vec(mu.iter()));
        let inc = opt.incumbent_x().unwrap();
        inc_bits.extend(bits_vec(inc.iter()));
        tr_bits.push(opt.tr_length().to_bits());
        obs_counts.push(opt.obs_count() as i64);
        generations.push(opt.restart_generation() as i64);
    }
    assert_eq!(generations[5], 0, "before restart ask, generation must be 0");
    assert_eq!(generations[6], 1, "round 6 ask must trigger begin_local_restart");

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
    let mut opt = create_optimizer_zero(bounds, Some(4), 2026042).unwrap();
    assert!(!opt.has_surrogate());
    let mut ask_bits = Vec::new();
    let mut inc_bits = Vec::new();
    let mut tr_bits = Vec::new();
    let mut obs_counts = Vec::new();

    for round in 0..6 {
        let arms = 2;
        let x = opt.ask(arms).unwrap();
        ask_bits.extend(bits_vec(x.iter()));
        let y = Array2::from_shape_fn((arms, 1), |(row, _)| {
            objective_1d(x.row(row).as_slice().unwrap(), round)
        });
        opt.tell(&x.view(), &y.view(), None).unwrap();
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
    let overrides = ConfigOverrides {
        acquisition: Some(AcquisitionConfig::Pareto),
        candidate_rv: Some(CandidateRV::Sobol),
        trust_region_kind: Some(TrustRegionKind::Morbo),
        morbo: Some(MorboOverride {
            num_metrics: 2,
            alpha: 0.05,
            rescalarize: Rescalarize::OnPropose,
        }),
        min_candidates: Some(32),
        max_candidates: Some(32),
        num_candidates_per_dim: Some(1),
        ..Default::default()
    };
    let mut opt =
        create_optimizer_enn_with_overrides(bounds, Some(3), Some(4), 2026043, Some(&overrides))
            .unwrap();
    let query = array![[0.0, 1.0, 2.0], [-1.0, 2.0, 1.0]];
    let mut ask_bits = Vec::new();
    let mut mu_bits = Vec::new();
    let mut inc_bits = Vec::new();
    let mut tr_bits = Vec::new();
    let mut obs_counts = Vec::new();

    for round in 0..6 {
        let arms = 2;
        let x = opt.ask(arms).unwrap();
        ask_bits.extend(bits_vec(x.iter()));
        let y = Array2::from_shape_fn((arms, 2), |(row, m)| {
            objective_2d(x.row(row).as_slice().unwrap(), round)[m]
        });
        opt.tell(&x.view(), &y.view(), None).unwrap();
        let mu = opt.posterior_mu(&query.view()).unwrap();
        mu_bits.extend(bits_vec(mu.iter()));
        let inc = opt.incumbent_x().unwrap();
        inc_bits.extend(bits_vec(inc.iter()));
        tr_bits.push(opt.tr_length().to_bits());
        obs_counts.push(opt.obs_count() as i64);
    }

    format!(
        "{{\"sobol_seed_base\":{},\"ask_bits\":{},\"posterior_mu_bits\":{},\"incumbent_x_bits\":{},\"tr_length_bits\":{},\"obs_counts\":{}}}",
        opt.sobol_seed_base(),
        u64_json(&ask_bits),
        u64_json(&mu_bits),
        u64_json(&inc_bits),
        u64_json(&tr_bits),
        i64_json(&obs_counts),
    )
}

fn lhd_only() -> String {
    let bounds = array![[0.0, 10.0], [-5.0, 5.0], [1.0, 3.0]];
    let mut opt = create_optimizer_lhd(bounds, Some(6), 2026044).unwrap();
    let mut ask_bits = Vec::new();
    let mut inc_bits = Vec::new();
    let mut progress = Vec::new();

    for (round, arms) in [1usize, 2, 2, 1].into_iter().enumerate() {
        let x = opt.ask(arms).unwrap();
        ask_bits.extend(bits_vec(x.iter()));
        let y = Array2::from_shape_fn((arms, 1), |(row, _)| {
            objective_1d(x.row(row).as_slice().unwrap(), round)
        });
        opt.tell(&x.view(), &y.view(), None).unwrap();
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

    let mut disk_model = EpistemicNearestNeighbors::new_with_storage(
        train_x.clone(),
        train_y.clone(),
        None,
        EnnLayout::disk(model_dir, false),
        None,
    )
    .unwrap();
    disk_model.add(&add_x.view(), &add_y.view(), None).unwrap();
    disk_model.index_access().ensure_sync().unwrap();
    let (disk_dist, disk_idx) = disk_model
        .index_access()
        .neighbor_distances_and_indices(&query.view(), 3, false)
        .unwrap();
    let (rows_x, rows_y, _) = disk_model.rows().train_rows_at(&[0, 3, 6, 7]).unwrap();

    let mut mem_model = EpistemicNearestNeighbors::new(
        train_x,
        train_y,
        None,
        false,
        IndexDriver::Flat,
    )
    .unwrap();
    mem_model.add(&add_x.view(), &add_y.view(), None).unwrap();
    let (mem_dist, mem_idx) = mem_model
        .index_access()
        .neighbor_distances_and_indices(&query.view(), 3, false)
        .unwrap();

    let bounds = array![[0.0, 2.0], [-1.0, 1.0], [0.0, 4.0]];
    let overrides = ConfigOverrides {
        min_candidates: Some(32),
        max_candidates: Some(32),
        num_candidates_per_dim: Some(1),
        enn: Some(EnnOverrides {
            index_driver: Some(IndexDriver::BpAnnDisk),
            enn_storage: Some(EnnStorage::Disk),
            work_dir: Some(opt_dir),
            ..Default::default()
        }),
        ..Default::default()
    };
    let mut opt =
        create_optimizer_enn_with_overrides(bounds, Some(3), Some(4), 2026045, Some(&overrides))
            .unwrap();
    let mut ask_bits = Vec::new();
    let mut mu_bits = Vec::new();
    let mut inc_bits = Vec::new();

    for round in 0..5 {
        let arms = 2;
        let x = opt.ask(arms).unwrap();
        ask_bits.extend(bits_vec(x.iter()));
        let y = Array2::from_shape_fn((arms, 1), |(row, _)| {
            objective_1d(x.row(row).as_slice().unwrap(), round)
        });
        opt.tell(&x.view(), &y.view(), None).unwrap();
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
    let work_dir = PathBuf::from(args.next().expect("usage: upstream_artifact_harness <work_dir>"));
    let _ = StdRng::seed_from_u64(0);
    let primitives = run_primitives();
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
