use crate::experiment::julia::config::{Config, ExecMode, is_ocean, obj_count};
use crate::experiment::julia::driver::{opt_cfg, run_opt};
use crate::experiment::julia::metal::{lhd_batch, objective_policy, res_enn};
use crate::experiment::julia::metal_step::{MetalFiber, MetalJobSpec, typed_warmup, unit_domain};
use crate::experiment::julia::mode::resolve_mode;
use baux::config::OptimizerSpec;
use baux::{AffineKey, AnyBackend, DomainFrame, OptBackend, UnifiedEngine};
use ndarray::Array2;
use std::fs;
use std::path::PathBuf;

#[test]
fn parse_configs() {
    let oc_toml = r#"
version = 4
experiment = "oceananigans"
rounds = 2

[ocean]
task = "kelvin-helmholtz"
nx = 512
ny = 1
nz = 512
actuators = 4
steps = 10
disk = true

[optimizer]
algorithm = "bayesian-search"
candidates = 2
arms = 1
acquisition = "random"

[optimizer.perturbation]
sampler = "raasp"
num-pert = 4096

[optimizer.trust-region]
initial = 0.8
min = 0.0078125
max = 1.6
"#;
    let cfg: Config = baux::artifact::toml::from_str(oc_toml).expect("parse ocean v4");
    assert!(is_ocean(&cfg));
    assert_eq!(cfg.ocean.nx * cfg.ocean.nz * cfg.ocean.actuators, 1_048_576);
    assert!(cfg.ocean.disk);

    let fu_toml = r#"
version = 4
experiment = "fuse"
rounds = 2

[fuse]
case = "ARC"
workflow = "whole-facility"
objective = "min_capital_cost"
penalty = 50.0

[optimizer]
algorithm = "bayesian-search"
candidates = 4
arms = 2
acquisition = "upper-confidence-bound"
"#;
    let fcfg: Config = baux::artifact::toml::from_str(fu_toml).expect("parse fuse v4");
    assert!(!is_ocean(&fcfg));
    assert_eq!(fcfg.fuse.case, "ARC");
    assert_eq!(fcfg.fuse.workflow, "whole-facility");
}

#[test]
fn disk_opt() {
    let toml_str = r#"
algorithm = "bayesian-search"
candidates = 4
arms = 2
acquisition = "random"

[perturbation]
sampler = "raasp"
num-pert = 128
raasp-fast = true

[trust-region]
initial = 0.8
min = 0.0078125
max = 1.6

[regional-search]
count = 2
sharing = "ellipsoid-ball"
allocation = "incumbent-ucb1"
controller = "turbo"
success-tolerance = 2
failure-tolerance = 3
"#;
    let spec: OptimizerSpec =
        baux::artifact::toml::from_str(toml_str).expect("parse optimizer spec");
    let cfg_opt = baux::OptimizerConfig::from_spec(&spec).expect("cfg_opt");
    let dim = 2048;
    let bounds = Array2::from_shape_fn((dim, 2), |(_, j)| if j == 0 { -1.0 } else { 1.0 });
    let domain = DomainFrame::new(bounds).expect("domain");
    let dir = std::env::temp_dir().join(format!("baux_disk_opt_test_{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    let job = MetalJobSpec::with_backend(
        0,
        42,
        domain,
        1,
        3,
        &spec,
        cfg_opt,
        None,
        AnyBackend::disk(dir.clone()),
    )
    .expect("disk job spec");
    let mut fiber = MetalFiber::new(job);
    assert_eq!(fiber.device_name(), "disk");
    for r in 0..3 {
        let u = fiber.prepare_unit().expect("prepare unit");
        assert_eq!(u.nrows(), 2);
        let scs = vec![-(r as f64), -((r + 1) as f64)];
        let _ = fiber
            .commit_unit(&scs, &[0.0, 0.0], 1, &|s| (-s).max(0.0))
            .expect("commit unit");
    }
    let _ = fs::remove_dir_all(&dir);
}

#[test]
#[ignore = "spawns heavy external Julia (Oceananigans.jl / FUSE.jl) workers; run via -- --ignored"]
fn live_julia() {
    let mut cwd = std::env::current_dir().expect("cwd");
    if !cwd
        .join("research/upstream/oceananigans.lock.toml")
        .exists()
    {
        if let Ok(ws) = std::env::var("BAUX_WORKSPACE") {
            cwd = PathBuf::from(ws);
        }
    }
    if !cwd
        .join("research/upstream/oceananigans.lock.toml")
        .exists()
        || !cwd.join(".cache/baux/upstream/oceananigans").exists()
        || !cwd.join(".cache/baux/upstream/fuse").exists()
    {
        return;
    }
    baux::artifact::merkle::verify_lock(&cwd, "oceananigans", None)
        .expect("verify oceananigans merkle lock");
    baux::artifact::merkle::verify_lock(&cwd, "fuse", None).expect("verify fuse merkle lock");

    let oc_toml = r#"
version = 4
experiment = "oceananigans"
backend = "host-disk"
rounds = 2

[ocean]
task = "kelvin-helmholtz"
nx = 512
ny = 1
nz = 512
actuators = 4
steps = 2
dt = 0.02
disk = true

[optimizer]
algorithm = "bayesian-search"
candidates = 2
arms = 1
acquisition = "random"

[optimizer.perturbation]
sampler = "raasp"
num-pert = 4096
raasp-fast = true

[optimizer.trust-region]
initial = 0.8
min = 0.0078125
max = 1.6

[optimizer.regional-search]
count = 2
sharing = "ellipsoid-ball"
allocation = "incumbent-ucb1"
controller = "turbo"
success-tolerance = 2
failure-tolerance = 3
"#;
    let oc_cfg: Config = baux::artifact::toml::from_str(oc_toml).expect("parse ocean 1m");
    let oc_mode = resolve_mode(&oc_cfg).expect("resolve ocean 1m mode");
    let tmp_oc = std::env::temp_dir().join(format!("baux_oc1m_{}", std::process::id()));
    let _ = fs::remove_dir_all(&tmp_oc);
    fs::create_dir_all(&tmp_oc).expect("mkdir tmp_oc");
    let oc_rep = run_opt(&oc_cfg, oc_mode, &cwd, &tmp_oc).expect("run live oceananigans 1M");
    let _ = fs::remove_dir_all(&tmp_oc);
    assert_eq!(oc_rep["parameters"], 1_048_576);
    assert_eq!(oc_rep["disk_backed"], true);
    assert_eq!(oc_rep["rounds"], 2);
    assert!(oc_rep["final_score_mean"].as_f64().unwrap().is_finite());

    let fu_toml = r#"
version = 4
experiment = "fuse"
rounds = 2

[fuse]
case = "ARC"
workflow = "equilibrium"
objective = "max_q95"
penalty = 25.0

[optimizer]
algorithm = "bayesian-search"
candidates = 4
arms = 2
acquisition = "upper-confidence-bound"

[optimizer.perturbation]
sampler = "raasp"
num-pert = 8

[optimizer.trust-region]
initial = 0.8
min = 0.0078125
max = 1.6

[optimizer.regional-search]
count = 2
sharing = "ellipsoid-ball"
allocation = "incumbent-ucb1"
controller = "turbo"
success-tolerance = 2
failure-tolerance = 3
"#;
    let fu_cfg: Config = baux::artifact::toml::from_str(fu_toml).expect("parse fuse live");
    let fu_mode = resolve_mode(&fu_cfg).expect("resolve fuse mode");
    let tmp_fu = std::env::temp_dir().join(format!("baux_fuse_{}", std::process::id()));
    let _ = fs::remove_dir_all(&tmp_fu);
    fs::create_dir_all(&tmp_fu).expect("mkdir tmp_fu");
    let fu_rep = run_opt(&fu_cfg, fu_mode, &cwd, &tmp_fu).expect("run live fuse ARC");
    let _ = fs::remove_dir_all(&tmp_fu);
    assert_eq!(fu_rep["experiment"], "fuse");
    assert_eq!(fu_rep["rounds"], 2);
    assert!(fu_rep["parameters"].as_u64().unwrap() >= 8);
    assert!(fu_rep["final_score_mean"].as_f64().unwrap().is_finite());
}

#[test]
fn parse_modes() {
    let metal_toml = r#"
version = 4
experiment = "oceananigans"
backend = "resident-metal"
rounds = 2

[ocean]
task = "wind-mixing"
nx = 4
ny = 1
nz = 4
actuators = 1
steps = 1
objectives = ["energy", "enstrophy"]

[morbo]
regions = 2
alpha = 0.05
scales = [1.0, 1.0]
preferences = [0.6, 0.4]
rescalarize = "on-restart"

[optimizer]
algorithm = "single-state-bo"
distribution = "rademacher"

[optimizer.surrogate]
neighbors = 8
neighbor-rule = "risk-prefix"
metric = "hyperbolic"
local-neighbors = 6
epistemic-scale = 1.25
aleatoric-scale = 0.1
y-scale = 2.0
fit = "fixed"
"#;
    let mcfg: Config = baux::artifact::toml::from_str(metal_toml).expect("parse metal mode");
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    assert_eq!(
        resolve_mode(&mcfg).expect("metal ok"),
        ExecMode::ResidentMetal
    );
    assert_eq!(obj_count(&mcfg), 2);
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    {
        let ocfg = opt_cfg(&mcfg, 2).expect("opt cfg");
        let res = res_enn(&mcfg, &ocfg, 7).expect("build res enn");
        assert!(res.adapt_neighbors);
        assert!(!res.fit_params);
        assert_eq!(res.local_scale_neighbors, 6);
        assert_eq!(res.distance_metric, baux::config::Dissimilarity::Hyperbolic);
    }

    let muon_toml = r#"
version = 4
experiment = "oceananigans"
rounds = 2

[optimizer]
algorithm = "muon"
"#;
    let ucfg: Config = baux::artifact::toml::from_str(muon_toml).expect("parse muon");
    let err = resolve_mode(&ucfg).expect_err("muon must fail explicitly");
    assert!(err.contains("muon") || err.contains("adamw"), "{err}");

    let host_toml = r#"
version = 4
experiment = "fuse"
backend = "host-ram"
rounds = 2

[fuse]
case = "ARC"
workflow = "equilibrium"
objectives = ["max_q95", "min_capital_cost"]

[manifold]
scale-x = true
metric-learning = true
stiefel-rank = 4
calibrate-alpha = 0.9
noise-aware = true

[morbo]
regions = 2
alpha = 0.05
rescalarize = "on-restart"

[optimizer]
algorithm = "bayesian-search"
candidates = 4
arms = 2
acquisition = "thompson"

[optimizer.surrogate]
neighbors = 8
metric = "wasserstein"
local-neighbors = 6
"#;
    let hcfg: Config = baux::artifact::toml::from_str(host_toml).expect("parse host manifold");
    assert_eq!(resolve_mode(&hcfg).expect("host ok"), ExecMode::HostRam);
    let ocfg = opt_cfg(&hcfg, 2).expect("build opt cfg");
    assert!(ocfg.noise_aware);
    if let baux::optimizer_config::SurrogateConfig::ENN(enn) = ocfg.surrogate {
        assert!(enn.scale_x);
        assert_eq!(enn.stiefel_rank, Some(4));
        assert_eq!(enn.calibrate_alpha, Some(0.9));
    } else {
        panic!("expected ENN surrogate");
    }
}

#[test]
#[ignore = "spawns external Julia workers across ResidentMetal/HostRam/MetalPpo modes; run via -- --ignored"]
fn live_modes() {
    let mut cwd = std::env::current_dir().expect("cwd");
    if !cwd
        .join("research/upstream/oceananigans.lock.toml")
        .exists()
    {
        if let Ok(ws) = std::env::var("BAUX_WORKSPACE") {
            cwd = PathBuf::from(ws);
        }
    }
    if !cwd
        .join("research/upstream/oceananigans.lock.toml")
        .exists()
        || !cwd.join(".cache/baux/upstream/oceananigans").exists()
        || !cwd.join(".cache/baux/upstream/fuse").exists()
    {
        return;
    }
    let tmp = std::env::temp_dir().join(format!("baux_live_modes_{}", std::process::id()));
    let _ = fs::remove_dir_all(&tmp);
    fs::create_dir_all(&tmp).expect("mkdir tmp");

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    {
        let rm_toml = r#"
version = 4
experiment = "oceananigans"
backend = "resident-metal"
rounds = 2

[ocean]
task = "wind-mixing"
nx = 4
ny = 1
nz = 4
actuators = 1
steps = 1
objectives = ["energy", "enstrophy"]

[morbo]
regions = 2
alpha = 0.05
scales = [1.0, 1.0]
preferences = [0.6, 0.4]
rescalarize = "on-restart"

[optimizer]
algorithm = "single-state-bo"
distribution = "rademacher"

[optimizer.surrogate]
neighbors = 4
neighbor-rule = "risk-prefix"
metric = "hyperbolic"
local-neighbors = 4
fit = "fixed"
"#;
        let rm_cfg: Config = baux::artifact::toml::from_str(rm_toml).expect("parse rm");
        let rm_mode = resolve_mode(&rm_cfg).expect("rm mode");
        let rm_rep = run_opt(&rm_cfg, rm_mode, &cwd, &tmp).expect("run resident metal");
        assert_eq!(rm_rep["backend_mode"], "resident-metal-morbo");
        assert_eq!(rm_rep["objectives"], 2);
        assert!(rm_rep["final_score_mean"].as_f64().unwrap().is_finite());
    }

    let hr_toml = r#"
version = 4
experiment = "oceananigans"
backend = "host-ram"
rounds = 2

[ocean]
task = "wind-mixing"
nx = 4
ny = 1
nz = 4
actuators = 1
steps = 1
objectives = ["energy", "enstrophy"]

[manifold]
scale-x = true
metric-learning = true
stiefel-rank = 4
noise-aware = true

[morbo]
regions = 2
alpha = 0.05

[optimizer]
algorithm = "bayesian-search"
candidates = 4
arms = 2
acquisition = "thompson"

[optimizer.surrogate]
neighbors = 4
metric = "wasserstein"
local-neighbors = 4
"#;
    let hr_cfg: Config = baux::artifact::toml::from_str(hr_toml).expect("parse hr");
    let hr_mode = resolve_mode(&hr_cfg).expect("hr mode");
    let hr_rep = run_opt(&hr_cfg, hr_mode, &cwd, &tmp).expect("run host ram morbo");
    assert_eq!(hr_rep["backend_mode"], "host-ram-morbo");
    assert_eq!(hr_rep["objectives"], 2);
    assert!(hr_rep["final_score_mean"].as_f64().unwrap().is_finite());

    let ppo_toml = r#"
version = 4
experiment = "oceananigans"
rounds = 3

[ocean]
task = "wind-mixing"
nx = 4
ny = 1
nz = 4
actuators = 1
steps = 1

[optimizer]
algorithm = "ppo"
"#;
    let ppo_cfg: Config = baux::artifact::toml::from_str(ppo_toml).expect("parse ppo");
    let ppo_mode = resolve_mode(&ppo_cfg).expect("ppo mode");
    let ppo_rep = run_opt(&ppo_cfg, ppo_mode, &cwd, &tmp).expect("run ppo");
    assert_eq!(ppo_rep["backend_mode"], "candle-ppo-metal");
    assert!(ppo_rep["final_score_mean"].as_f64().unwrap().is_finite());

    let _ = fs::remove_dir_all(&tmp);
}

fn quad_score(row: &[u16], shift: f64) -> f64 {
    let mut sum = 0.0f64;
    for &w in row {
        let u = f64::from(baux::unified_mem::decode_half(w)).clamp(0.0, 1.0);
        let d = u - shift;
        sum -= d * d;
    }
    sum
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[test]
fn metal_parity() {
    use baux::artifact::inc::EvalOutcome;
    use baux::resident_metal::engine::make_enn;

    let rich_toml = r#"
version = 4
experiment = "oceananigans"
rounds = 12

[ocean]
task = "wind-mixing"
nx = 4
ny = 2
nz = 4
actuators = 1
steps = 1
disk = false
objectives = ["energy", "enstrophy"]

[manifold]
scale-x = true
metric-learning = true
stiefel-rank = 2
calibrate-alpha = 0.25
noise-aware = true

[morbo]
regions = 12
alpha = 0.05
rescalarize = "on-propose"

[optimizer]
algorithm = "bayesian-search"
candidates = 128
arms = 4
acquisition = "upper-confidence-bound"

[optimizer.perturbation]
sampler = "sobol"
num-pert = 8

[optimizer.surrogate]
neighbors = 4
metric = "conformal"
local-neighbors = 4
fit = "leave-one-out"
fit-candidates = 4
fit-samples = 2

[optimizer.regional-search]
count = 4
sharing = "nearest-center"
allocation = "joint-acquisition"
controller = "turbo"
restart-memory = "retain-relevant"
restart-center = "maximin"
restart-candidates = 8
success-tolerance = 2
failure-tolerance = 2
"#;
    let rcfg: Config = baux::artifact::toml::from_str(rich_toml).expect("parse rich julia");
    assert_eq!(
        resolve_mode(&rcfg).expect("auto metal"),
        ExecMode::ResidentMetal
    );
    let ocfg = opt_cfg(&rcfg, 2).expect("rich opt cfg");
    assert_eq!(
        ocfg.acquisition,
        baux::optimizer_config::AcquisitionConfig::UCB { beta: 1.0 }
    );
    let enn_cfg = res_enn(&rcfg, &ocfg, 17).expect("rich res_enn");
    assert_eq!(enn_cfg.pool.arms(), 4);
    assert_eq!(enn_cfg.pool.count(), 128);

    let mut auto_opt = ocfg.clone();
    auto_opt.candidates = baux::optimizer_config::CandidateConfig::default();
    let enn_d20 = make_enn(&auto_opt, 20, 4, false, 17).expect("d20 pool");
    assert_eq!(enn_d20.pool.arms(), 4);
    assert_eq!(enn_d20.pool.count(), 2000);
    let enn_cap = make_enn(&auto_opt, 100, 4, false, 17).expect("cap pool");
    assert_eq!(enn_cap.pool.count(), 5000);

    let strata = lhd_batch(99, 8, 6);
    assert_eq!(strata.len(), 8);
    for col in 0..6 {
        let mut seen = [false; 8];
        for row in &strata {
            let u = f64::from(baux::unified_mem::decode_half(row[col])).clamp(0.0, 0.9999);
            seen[(u * 8.0) as usize] = true;
        }
        assert!(
            seen.into_iter().all(|v| v),
            "LHD column {col} missed stratum"
        );
    }

    let bad_single: Config = baux::artifact::toml::from_str(
        "version = 4\nexperiment = \"oceananigans\"\nbackend = \"resident-metal\"\nrounds = 2\n[optimizer]\nalgorithm = \"single-state-bo\"\n[optimizer.perturbation]\nsampler = \"uniform\"\nnum-pert = 4\n",
    )
    .expect("bad single");
    assert!(resolve_mode(&bad_single).is_err());

    let bad_mf: Config = baux::artifact::toml::from_str(
        "version = 4\nexperiment = \"oceananigans\"\nbackend = \"resident-metal\"\nrounds = 2\n[manifold]\nstiefel-rank = 0\n[optimizer]\nalgorithm = \"bayesian-search\"\nacquisition = \"upper-confidence-bound\"\n",
    )
    .expect("bad manifold");
    assert!(resolve_mode(&bad_mf).is_err());

    let single_toml = r#"
version = 4
experiment = "oceananigans"
backend = "resident-metal"
rounds = 10

[manifold]
scale-x = true
metric-learning = true
calibrate-alpha = 0.2

[optimizer]
algorithm = "bayesian-search"
candidates = 64
arms = 4
acquisition = "upper-confidence-bound"

[optimizer.perturbation]
sampler = "raasp"
num-pert = 6

[optimizer.surrogate]
neighbors = 4

[optimizer.trust-region]
initial = 0.4
min = 0.2
max = 0.8

[optimizer.regional-search]
count = 2
sharing = "box-contain"
allocation = "joint-acquisition"
controller = "turbo"
restart-memory = "retain-relevant"
restart-center = "maximin"
restart-candidates = 8
success-tolerance = 2
failure-tolerance = 1
"#;
    let scfg: Config = baux::artifact::toml::from_str(single_toml).expect("parse single obj");
    let socfg = opt_cfg(&scfg, 1).expect("single opt cfg");
    let dom = unit_domain(16).expect("16d unit domain");
    let s_spec = MetalJobSpec::with_backend(
        0,
        4242,
        dom.clone(),
        1,
        10,
        &scfg.optimizer,
        socfg.clone(),
        None,
        AnyBackend::Metal,
    )
    .expect("metal job spec");
    let mut m_fiber = MetalFiber::new(s_spec);
    assert_eq!(m_fiber.device_name(), "metal");
    let mut m_best = f64::NEG_INFINITY;
    for _ in 0..10 {
        let rows = m_fiber.prepare_batch().expect("prepare metal batch");
        assert!(!rows.is_empty() && rows.len() <= 4);
        let scs: Vec<f64> = rows.iter().map(|r| quad_score(r, 0.35)).collect();
        let vls = vec![0.0; rows.len()];
        let rec = m_fiber
            .commit_batch(&scs, &vls, 10, &|s| (-s).max(0.0))
            .expect("commit batch");
        assert!(rec.best_score >= m_best);
        m_best = rec.best_score;
    }
    assert!(m_best.is_finite() && m_best > -4.0);

    let h_spec = MetalJobSpec::with_backend(
        0,
        4242,
        dom,
        1,
        10,
        &scfg.optimizer,
        socfg,
        None,
        AnyBackend::Cpu,
    )
    .expect("cpu job spec");
    let mut h_fiber = MetalFiber::new(h_spec);
    assert_eq!(h_fiber.device_name(), "cpu");
    let mut h_best = f64::NEG_INFINITY;
    for _ in 0..10 {
        let u = h_fiber.prepare_unit().expect("host prepare unit 4");
        assert_eq!(u.nrows(), 4);
        let scs: Vec<f64> = (0..4)
            .map(|i| (0..16).map(|j| -(u[[i, j]] - 0.35).powi(2)).sum::<f64>())
            .collect();
        let rec = h_fiber
            .commit_unit(&scs, &[0.0; 4], 10, &|s| (-s).max(0.0))
            .expect("host commit unit 4");
        assert!(rec.best_score >= h_best);
        h_best = rec.best_score;
    }
    assert!(h_best.is_finite() && h_best > -4.0);

    let pol = objective_policy(&rcfg, 2, 777);
    let m_spec = MetalJobSpec::from_parts(0, 777, 16, 2, 8, &rcfg.optimizer, ocfg, pol)
        .expect("morbo job spec");
    let mut live_f = MetalFiber::new(m_spec.clone());
    let mut prefix = Vec::new();
    for _ in 0..6 {
        let row = live_f.prepare_proposal().expect("morbo prepare");
        let out = EvalOutcome {
            scores: vec![quad_score(&row, 0.25), quad_score(&row, 0.75)],
            viol: 0.0,
            cost_us: 5,
        };
        let step = live_f
            .commit_eval(&out, &|s| (-s).max(0.0))
            .expect("morbo commit");
        prefix.push(step);
    }
    let next_live = live_f.prepare_proposal().expect("next live prop");
    let mut rep_f = MetalFiber::new(m_spec);
    rep_f
        .replay_prefix(&prefix, &|s| (-s).max(0.0))
        .expect("morbo replay");
    let next_rep = rep_f.prepare_proposal().expect("next rep prop");
    assert_eq!(rep_f.best_score, live_f.best_score);
    assert_eq!(next_rep, next_live);

    // 1. AcquisitionConfig::Pareto -> AcquisitionKind::Pareto
    let par_spec: OptimizerSpec = baux::artifact::toml::from_str(
        "algorithm = \"bayesian-search\"\ncandidates = 32\narms = 4\nacquisition = \"pareto\"\nscales = [1.0, 1.0]\n",
    )
    .expect("pareto spec");
    let par_opt = baux::optimizer_config::OptimizerConfig::from_spec(&par_spec).expect("par opt");
    let par_enn = make_enn(&par_opt, 16, 4, false, 19).expect("pareto enn");
    assert_eq!(
        par_enn.ask.acquisition.kind,
        baux::weights::AcquisitionKind::Pareto
    );

    // 2. 4-arm (n_reg = 1) prepare_batch + commit_batch on n_obj = 2 MORBO
    //    (full neighbors=6 LHD warmup + arm 2 winning over arm 0 promotes base)
    let morbo_batch = r#"
version = 4
experiment = "oceananigans"
backend = "resident-metal"
rounds = 6

[morbo]
regions = 4
alpha = 0.05
rescalarize = "on-propose"

[optimizer]
algorithm = "bayesian-search"
candidates = 64
arms = 4
acquisition = "upper-confidence-bound"

[optimizer.perturbation]
sampler = "raasp"
num-pert = 6

[optimizer.surrogate]
neighbors = 6

[optimizer.regional-search]
count = 1
sharing = "origin-only"
allocation = "least-trials"
controller = "turbo"
restart-memory = "retain-relevant"
restart-center = "maximin"
restart-candidates = 4
success-tolerance = 2
failure-tolerance = 1
"#;
    let mb_cfg: Config = baux::artifact::toml::from_str(morbo_batch).expect("parse morbo batch");
    let mb_opt = opt_cfg(&mb_cfg, 2).expect("morbo batch opt");
    let mb_pol = objective_policy(&mb_cfg, 2, 313);
    let mb_spec = MetalJobSpec::from_parts(0, 313, 16, 2, 6, &mb_cfg.optimizer, mb_opt, mb_pol)
        .expect("morbo batch spec");
    let mut mb_fiber = MetalFiber::new(mb_spec);
    let w1 = mb_fiber.prepare_batch().expect("warm 1..4");
    assert_eq!(w1.len(), 4);
    let _ = mb_fiber
        .commit_batch(&[-2.0; 8], &[0.0; 4], 1, &|s| (-s).max(0.0))
        .expect("commit warm 1..4");
    assert_eq!(
        mb_fiber
            .engine
            .as_ref()
            .and_then(UnifiedEngine::init_progress),
        Some((4, 6)),
        "must remain in warmup until k=6 observations"
    );
    let w2 = mb_fiber.prepare_batch().expect("warm 5..6");
    assert_eq!(w2.len(), 2);
    let _ = mb_fiber
        .commit_batch(&[-2.0; 4], &[0.0; 2], 1, &|s| (-s).max(0.0))
        .expect("commit warm 5..6");
    assert_eq!(
        mb_fiber
            .engine
            .as_ref()
            .and_then(UnifiedEngine::init_progress),
        None
    );

    let act_rows = mb_fiber.prepare_batch().expect("active 4-arm morbo");
    assert_eq!(act_rows.len(), 4);
    let win_scores = [-5.0, -5.0, -4.0, -4.0, 5.0, 5.0, -6.0, -6.0];
    let _ = mb_fiber
        .commit_batch(&win_scores, &[0.0; 4], 1, &|s| (-s).max(0.0))
        .expect("commit winning arm 2");
    assert_eq!(mb_fiber.best_row, act_rows[2]);
    assert_eq!(
        mb_fiber
            .engine
            .as_ref()
            .and_then(UnifiedEngine::best_halves)
            .as_deref(),
        Some(act_rows[2].as_slice())
    );
    let next_act = mb_fiber.prepare_batch().expect("next active 4-arm batch");
    assert_eq!(
        next_act.len(),
        4,
        "winning arm 2 must count as Success, not Failure"
    );

    // 3. Tolerance (failure-tolerance = 1) & FixedRadius preservation across
    //    restart
    let mut tol_cfg = scfg.clone();
    if let OptimizerSpec::BayesianSearch {
        ref mut regional_search,
        ..
    } = tol_cfg.optimizer
    {
        let rs = regional_search.as_mut().unwrap();
        rs.count = 1;
        rs.failure_tolerance = 1;
    }
    let tol_opt = opt_cfg(&tol_cfg, 1).expect("tol opt");
    let tol_spec =
        MetalJobSpec::from_parts(0, 515, 16, 1, 8, &tol_cfg.optimizer, tol_opt, None).unwrap();
    let mut tol_f = MetalFiber::new(tol_spec);
    let _ = tol_f.prepare_batch().unwrap();
    let _ = tol_f
        .commit_batch(&[0.0; 4], &[0.0; 4], 1, &|s| (-s).max(0.0))
        .unwrap();
    let _ = tol_f.prepare_batch().unwrap();
    let _ = tol_f
        .commit_batch(&[-10.0; 4], &[0.0; 4], 1, &|s| (-s).max(0.0))
        .unwrap();
    let rep_rows = tol_f.prepare_batch().unwrap();
    assert_eq!(rep_rows.len(), 1, "retirement #1 must queue 1 replace row");
    let _ = tol_f
        .commit_batch(&[0.0], &[0.0], 1, &|s| (-s).max(0.0))
        .unwrap();
    let mut retired_again = false;
    for step in 1..=3 {
        let rows = tol_f.prepare_batch().unwrap();
        if rows.len() == 1 {
            retired_again = true;
            break;
        }
        let bad = -10.0 * (step as f64);
        let _ = tol_f
            .commit_batch(&[bad; 4], &[0.0; 4], 1, &|s| (-s).max(0.0))
            .unwrap();
    }
    assert!(
        retired_again,
        "restarted region must preserve failure_tolerance = 1"
    );

    let mut fix_cfg = tol_cfg.clone();
    if let OptimizerSpec::BayesianSearch {
        ref mut regional_search,
        ..
    } = fix_cfg.optimizer
    {
        let rs = regional_search.as_mut().unwrap();
        rs.controller = baux::config::RegionController::FixedRadius;
    }
    let fix_opt = opt_cfg(&fix_cfg, 1).expect("fix opt");
    let fix_spec =
        MetalJobSpec::from_parts(0, 616, 16, 1, 8, &fix_cfg.optimizer, fix_opt, None).unwrap();
    let mut fix_f = MetalFiber::new(fix_spec);
    let _ = fix_f.prepare_batch().unwrap();
    let _ = fix_f
        .commit_batch(&[0.0; 4], &[0.0; 4], 1, &|s| (-s).max(0.0))
        .unwrap();
    for step in 1..=4 {
        let rows = fix_f.prepare_batch().unwrap();
        assert_eq!(rows.len(), 4, "FixedRadius must not retire");
        let bad = -10.0 * (step as f64);
        let _ = fix_f
            .commit_batch(&[bad; 4], &[0.0; 4], 1, &|s| (-s).max(0.0))
            .unwrap();
        let tr = fix_f.engine.as_ref().unwrap().tr_length();
        assert!((tr - 0.4).abs() < 1e-6, "FixedRadius drifted: {tr}");
    }
}

fn drive_julia<B: OptBackend>(
    cfg: &Config,
    n_obj: usize,
    backend: B,
    seed: u64,
) -> (&'static str, usize, f64, [u8; 32]) {
    let bounds = Array2::from_shape_fn((12, 2), |(_, j)| if j == 0 { -2.0 } else { 2.0 });
    let domain = DomainFrame::new(bounds).expect("12d domain");
    let cfg_opt = opt_cfg(cfg, n_obj).expect("opt cfg");
    let pol = objective_policy(cfg, n_obj, seed);
    let (warmup, key) = typed_warmup(
        domain,
        &cfg.optimizer,
        cfg_opt,
        pol,
        backend,
        AffineKey::new(seed),
    )
    .expect("typed warmup");
    let dev = warmup.device_name();
    let (x0, warmup, key) = warmup.design_step(4, key).expect("design step");
    let y0 = Array2::from_shape_fn((x0.nrows(), n_obj), |(i, m)| {
        let shift = if m == 0 { 0.4 } else { -0.4 };
        -(0..12).map(|j| (x0[[i, j]] - shift).powi(2)).sum::<f64>()
    });
    let (ready, key) = warmup
        .activate_ready(&x0.view(), &y0.view(), key)
        .expect("activate ready");
    let (ready_end, key, traces) = baux::scan_step(ready, key, 3, 2, |_, x| {
        Ok(Array2::from_shape_fn((x.nrows(), n_obj), |(i, m)| {
            let shift = if m == 0 { 0.4 } else { -0.4 };
            -(0..12).map(|j| (x[[i, j]] - shift).powi(2)).sum::<f64>()
        }))
    })
    .expect("scan step");
    let _ = key.into_key();
    assert_eq!(traces.len(), 3);
    (
        dev,
        ready_end.num_obs(),
        ready_end.best_y().expect("best_y"),
        ready_end.digest(),
    )
}

#[test]
fn unified_backends() {
    let morbo_toml = r#"
version = 4
experiment = "oceananigans"
rounds = 4

[manifold]
scale-x = true
metric-learning = true
noise-aware = true

[morbo]
regions = 4
alpha = 0.05
rescalarize = "on-propose"

[optimizer]
algorithm = "bayesian-search"
candidates = 64
arms = 2
acquisition = "upper-confidence-bound"

[optimizer.perturbation]
sampler = "raasp"
num-pert = 6

[optimizer.surrogate]
neighbors = 4
metric = "hyperbolic"
local-neighbors = 4

[optimizer.regional-search]
count = 2
sharing = "box-contain"
allocation = "incumbent-ucb1"
controller = "turbo"
success-tolerance = 2
failure-tolerance = 3
"#;
    let cfg: Config = baux::artifact::toml::from_str(morbo_toml).expect("parse unified morbo");
    let (d_cpu, n_cpu, b_cpu, dig_cpu) = drive_julia(&cfg, 2, baux::CpuBackend, 909);
    let (d_any, n_any, b_any, dig_any) = drive_julia(&cfg, 2, AnyBackend::Cpu, 909);
    assert_eq!(d_cpu, "cpu");
    assert_eq!(d_any, "cpu");
    assert_eq!(n_cpu, 10);
    assert_eq!(n_any, 10);
    assert_eq!(dig_cpu, dig_any);
    assert_eq!(b_cpu, b_any);
    assert!(b_cpu.is_finite());

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    {
        let (d_met, n_met, b_met, dig_met) = drive_julia(&cfg, 2, baux::MetalBackend, 909);
        let (d_am, n_am, b_am, dig_am) = drive_julia(&cfg, 2, AnyBackend::Metal, 909);
        assert_eq!(d_met, "metal");
        assert_eq!(d_am, "metal");
        assert_eq!(n_met, 10);
        assert_eq!(n_am, 10);
        assert_eq!(dig_met, dig_am);
        assert_eq!(b_met, b_am);
        assert!(b_met.is_finite());

        // Also verify SingleStateBo zero-copy FP16 MetalFiber path via
        // UnifiedEngine
        let single_toml = r#"
version = 4
experiment = "oceananigans"
backend = "resident-metal"
rounds = 4

[optimizer]
algorithm = "single-state-bo"
distribution = "rademacher"

[optimizer.surrogate]
neighbors = 4
neighbor-rule = "risk-prefix"
metric = "hyperbolic"
local-neighbors = 4
fit = "fixed"
"#;
        let scfg: Config = baux::artifact::toml::from_str(single_toml).expect("parse single");
        let socfg = opt_cfg(&scfg, 1).expect("single opt");
        let sspec =
            MetalJobSpec::from_parts(0, 314, 12, 1, 4, &scfg.optimizer, socfg, None).expect("spec");
        let mut sfiber = MetalFiber::new(sspec);
        assert_eq!(sfiber.device_name(), "metal");
        for _ in 0..4 {
            let rows = sfiber.prepare_batch().expect("single prepare");
            assert_eq!(rows.len(), 1);
            let sc = quad_score(&rows[0], 0.5);
            let _ = sfiber
                .commit_batch(&[sc], &[0.0], 5, &|s| (-s).max(0.0))
                .expect("single commit");
        }
        assert!(sfiber.best_score.is_finite());
    }
}
