use crate::experiment::julia::config::{Config, ExecMode, is_ocean, obj_count};
use crate::experiment::julia::driver::{opt_cfg, run_opt};
use crate::experiment::julia::metal::res_enn;
use crate::experiment::julia::mode::resolve_mode;
use baux::config::OptimizerSpec;
use baux::{BayesOptimizer, DomainFrame, prng};
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
    let dim = 2048;
    let mut bounds = Array2::zeros((dim, 2));
    for j in 0..dim {
        bounds[[j, 0]] = -1.0;
        bounds[[j, 1]] = 1.0;
    }
    let domain = DomainFrame::new(bounds).expect("domain");
    let dir = std::env::temp_dir().join(format!("baux_disk_opt_test_{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    let mut rng = prng(42);
    let mut opt =
        BayesOptimizer::from_disk(domain, &spec, 0, dir.clone(), &mut rng).expect("from_disk");
    for r in 0..3 {
        let u = opt.ask(2, &mut rng).expect("ask");
        let y = Array2::from_shape_fn((2, 1), |(i, _)| -((r + i) as f64));
        opt.tell(&u.view(), &y.view(), &mut rng).expect("tell");
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

    // 1. Live 1,048,576-parameter (512 x 1 x 512 x 4) zero-copy ShmSlab +
    //    DiskBpann Oceananigans.jl NonhydrostaticModel rollout!
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

    // 2. Live FUSE.jl v1.2.1 ARC equilibrium + IMAS ObjectiveFunction /
    //    ConstraintFunction rollout over ShmSlab!
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
algorithm = "operator-bo"
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
        assert_eq!(
            res.distance_metric,
            baux::config::DistanceMetric::Hyperbolic
        );
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

    // 1. Live ResidentMetal OperatorBo (Hyperbolic + RiskPrefix + Rademacher +
    //    Morbo)
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
algorithm = "operator-bo"
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

    // 2. Live HostRam multi-objective + Wasserstein + Manifold
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

    // 3. Live Metal FlatPpo on Oceananigans.jl
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
