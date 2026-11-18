use super::*;

#[test]
fn test_tokamak() {
    let toml_str = r#"
version = 4
experiment = "tokamak-tcv"
rounds = 2

[optimizer]
algorithm = "bayesian-search"
candidates = 2
arms = 1
acquisition = "random"

[optimizer.perturbation]
sampler = "raasp"
num-pert = 16053

[optimizer.trust-region]
initial = 0.8
min = 0.0078125
max = 1.6

[optimizer.regional-search]
count = 2
sharing = "ellipsoid-ball"
allocation = "incumbent-ucb1"
controller = "turbo"
success-tolerance = 3
failure-tolerance = 5
"#;
    let config: Config = baux::artifact::toml::from_str(toml_str).expect("parse v4 config");
    let report = run_opt(&config, None, None).expect("run tokamak multi-region opt");
    let cbor_bytes = baux::artifact::cbor::to_vec(&report).expect("encode tokamak cbor");
    let decoded: Value =
        baux::artifact::cbor::from_slice(&cbor_bytes).expect("decode tokamak cbor");
    assert_eq!(decoded["experiment"], "tokamak-tcv");
    assert_eq!(decoded["optimizer_label"], "turbo-zero-r2-ellipsoid-ball");
    assert_eq!(decoded["regions"], 2);
    assert_eq!(decoded["rounds"], 2);

    let ppo_toml = r#"
version = 4
experiment = "tokamak-tcv"
rounds = 8

[optimizer]
algorithm = "ppo"
"#;
    let ppo_cfg: Config = baux::artifact::toml::from_str(ppo_toml).expect("parse ppo config");
    let ppo_rep = run_opt(&ppo_cfg, None, None).expect("run tokamak candle ppo");
    assert_eq!(ppo_rep["experiment"], "tokamak-tcv");
    assert_eq!(ppo_rep["optimizer_label"], "candle-ppo-metal");
    assert_eq!(ppo_rep["rounds"], 8);
}

#[test]
fn test_resume() {
    let zero_r10 = r#"
version = 4
experiment = "tokamak-tcv"
rounds = 4

[optimizer]
algorithm = "bayesian-search"
candidates = 4
arms = 1
acquisition = "random"

[optimizer.perturbation]
sampler = "raasp"
num-pert = 16053
raasp-fast = true

[optimizer.trust-region]
initial = 0.8
min = 0.00048828125
max = 1.6

[optimizer.regional-search]
count = 10
sharing = "origin-only"
allocation = "incumbent-ucb1"
controller = "turbo"
success-tolerance = 2
failure-tolerance = 3
"#;
    let enn_r1 = r#"
version = 4
experiment = "tokamak-tcv"
rounds = 4

[optimizer]
algorithm = "bayesian-search"
candidates = 4
arms = 1
acquisition = "upper-confidence-bound"

[optimizer.surrogate]
metric = "hyperbolic"
local-neighbors = 4
neighbors = 10
fit = "leave-one-out"
fit-candidates = 6
fit-samples = 3

[optimizer.perturbation]
sampler = "raasp"
num-pert = 16053
raasp-fast = true

[optimizer.trust-region]
initial = 0.8
min = 0.00048828125
max = 1.6

[optimizer.regional-search]
count = 1
sharing = "origin-only"
allocation = "incumbent-ucb1"
controller = "turbo"
success-tolerance = 2
failure-tolerance = 3
"#;
    let cfg_z: Config = baux::artifact::toml::from_str(zero_r10).expect("parse zero r10");
    let cfg_e: Config = baux::artifact::toml::from_str(enn_r1).expect("parse enn r1");
    assert_eq!(opt_label(&cfg_z.optimizer), "turbo-zero-r10-origin-only");
    assert_eq!(opt_label(&cfg_e.optimizer), "turbo-enn-r1-hyperbolic");

    let full = run_seed(&cfg_e, 0, None, &mut None).expect("full 4-round seed");
    let z_seed = run_seed(&cfg_z, 0, None, &mut None).expect("zero r10 seed");
    assert!(
        (full.initial_error - z_seed.initial_error).abs() < 1e-9,
        "paired initial error mismatch: {} vs {}",
        full.initial_error,
        z_seed.initial_error
    );

    let dir = std::env::temp_dir().join(format!("baux_ckpt_test_{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    let mut half_cfg = cfg_e.clone();
    half_cfg.rounds = 2;
    let seed = baux::tune_seed(&half_cfg.experiment, "proposal", 0, 0);
    let mut opt = make_opt(&half_cfg, Some(&dir), 0, seed).expect("make opt");
    let dom = exp_domain(&half_cfg);
    let mut st = LiveCkpt::empty(2);
    st.elapsed_seconds = 0.1;
    for round in 1..=2 {
        let mut sr = round_rng(seed, round);
        let (new_opt, u) = opt.ask(1, &mut sr).expect("ask");
        opt = new_opt;
        let pts = dom.from_unit(&u.view());
        let sl = pts.as_slice_memory_order().expect("slice");
        let sc = eval_batch(&mut None, sl, 1).expect("eval")[0];
        if sc > st.best_score {
            st.best_score = sc;
            st.total_accepts += 1;
        }
        if round == 1 {
            st.initial_error = tracking_error(&half_cfg, sc);
        }
        let y = Array2::from_elem((1, 1), sc);
        opt = opt.tell(&u.view(), &y.view(), None, &mut sr).expect("tell");
        st.best_scores.push(st.best_score);
        st.eval_scores.push(vec![sc]);
        st.completed_round = round;
    }
    save_live(&dir, 0, &opt, &st).expect("save live ckpt");
    let tpl = make_opt(&cfg_e, Some(&dir), 0, seed).expect("make template");
    let (thawed, thawed_st) = load_live(&dir, 0, seed, tpl)
        .expect("load live")
        .expect("live ckpt present");
    assert_eq!(thawed_st.completed_round, 2);
    assert_eq!(thawed.digest(), opt.digest());

    // Also verify zero-simulator-call deterministic replay when seed_0_opt.bin
    // is absent (long-run >64MB observation regime).
    let _ = fs::remove_file(dir.join("seed_0_opt.bin"));
    let resumed = run_seed(&cfg_e, 0, Some(&dir), &mut None).expect("resumed 4-round seed");
    assert_eq!(resumed.best_scores, full.best_scores);
    assert!((resumed.final_score - full.final_score).abs() < 1e-12);

    let cached = run_seed(&cfg_e, 0, Some(&dir), &mut None).expect("cached done seed");
    assert_eq!(cached.best_scores, full.best_scores);
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn test_recovery() {
    let dir = std::env::temp_dir().join(format!("baux_chaos_worker_{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).expect("mkdir");
    let state_file = dir.join("counter.txt");
    let script = dir.join("chaos_daemon.py");
    let py_code = format!(
        r#"import mmap, os, socket, struct, sys, time
STATE = {state_path:?}
sock_path = sys.argv[sys.argv.index("--sock") + 1]
if os.path.exists(sock_path):
    os.remove(sock_path)
srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
srv.bind(sock_path)
srv.listen(2)
while True:
    conn, _ = srv.accept()
    line = conn.recv(4096).decode().strip()
    if line == "PING":
        conn.sendall(b"PONG\n")
        conn.close()
        continue
    parts = line.split("\t")
    fd = os.open(parts[1], os.O_RDWR)
    mm = mmap.mmap(fd, os.path.getsize(parts[1]), access=mmap.ACCESS_WRITE)
    os.close(fd)
    hdr = struct.unpack_from("<10Q", mm, 0)
    score_off = hdr[6]
    conn.sendall(b"OK\t8\t4096\t1\n")
    last_seq = 0
    while True:
        seq = struct.unpack_from("<Q", mm, 17 * 8)[0]
        if seq <= last_seq:
            time.sleep(0.001)
            continue
        cmd = struct.unpack_from("<Q", mm, 16 * 8)[0]
        if cmd in (2, 3):
            struct.pack_into("<Q", mm, 33 * 8, 0)
            struct.pack_into("<Q", mm, 32 * 8, seq)
            conn.close()
            mm.close()
            sys.exit(0)
        n = int(open(STATE).read()) + 1 if os.path.exists(STATE) else 1
        open(STATE, "w").write(str(n))
        if n == 2:
            os._exit(1)
        if n in (4, 5):
            time.sleep(2.0)
        struct.pack_into("<d", mm, score_off, -10.0 - n)
        struct.pack_into("<Q", mm, 33 * 8, 0)
        struct.pack_into("<Q", mm, 34 * 8, 50)
        struct.pack_into("<Q", mm, 35 * 8, 1)
        struct.pack_into("<Q", mm, 32 * 8, seq)
        last_seq = seq
"#,
        state_path = state_file.display().to_string()
    );
    fs::write(&script, py_code).expect("write chaos script");

    // SAFETY: test-local env overrides for chaos timing & mock script.
    unsafe {
        std::env::set_var("BAUX_DAEMON_SCRIPT", &script);
        std::env::set_var("BAUX_EVAL_TIMEOUT_MS", "250");
        std::env::set_var("BAUX_DAEMON_MAX_EVALS", "3");
    }
    let spec = WorkerSpec {
        kind: "tokagym",
        lock: "",
        arms: 1,
        num_dim: 8,
        obs: 4,
        act: 2,
        steps: 2,
    };
    let mut worker = TokamakWorker::spawn(&dir, spec).expect("spawn chaos worker");
    let w = vec![0.1f64; 8];
    let s1 = worker.eval(&w, 1).expect("eval 1 normal")[0];
    let s2 = worker.eval(&w, 1).expect("eval 2 recovers from crash")[0];
    let s3 = worker
        .eval(&w, 1)
        .expect("eval 3 recovers & triggers penalty on double hang")[0];
    let s4 = worker
        .eval(&w, 1)
        .expect("eval 4 normal after penalty respawn")[0];
    drop(worker);
    unsafe {
        std::env::remove_var("BAUX_DAEMON_SCRIPT");
        std::env::remove_var("BAUX_EVAL_TIMEOUT_MS");
        std::env::remove_var("BAUX_DAEMON_MAX_EVALS");
    }
    assert!(s1 > -100.0 && s2 > -100.0 && s4 > -100.0);
    assert_eq!(s3, -500.0);
    let _ = fs::remove_dir_all(&dir);
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[test]
fn test_metal() {
    let zero_box = r#"
version = 4
experiment = "tokamak-tcv"
backend = "resident-metal"
rounds = 24

[optimizer]
algorithm = "bayesian-search"
distribution = "gaussian"
candidates = 4
arms = 1
acquisition = "random"

[optimizer.trust-region]
initial = 0.8
min = 0.25
max = 1.6

[optimizer.regional-search]
count = 10
sharing = "box-contain"
allocation = "incumbent-ucb1"
controller = "turbo"
success-tolerance = 3
failure-tolerance = 3
"#;
    let enn_conf = r#"
version = 4
experiment = "tokamak-tcv"
backend = "resident-metal"
rounds = 24

[optimizer]
algorithm = "bayesian-search"
distribution = "gaussian"
candidates = 4
arms = 1
acquisition = "upper-confidence-bound"

[optimizer.surrogate]
metric = "conformal"
local-neighbors = 4
neighbors = 6
fit = "leave-one-out"
fit-candidates = 6
fit-samples = 3

[optimizer.trust-region]
initial = 0.8
min = 0.25
max = 1.6

[optimizer.regional-search]
count = 1
sharing = "origin-only"
allocation = "incumbent-ucb1"
controller = "turbo"
success-tolerance = 3
failure-tolerance = 3
"#;
    let cfg_z: Config = baux::artifact::toml::from_str(zero_box).expect("parse metal zero");
    let cfg_e: Config = baux::artifact::toml::from_str(enn_conf).expect("parse metal enn");
    let dom = exp_domain(&cfg_e);
    let rec_z = metal::metal_seed(&cfg_z, &dom, 0, None, &mut None).expect("run metal zero");
    let rec_e = metal::metal_seed(&cfg_e, &dom, 0, None, &mut None).expect("run metal enn");
    assert_eq!(rec_z.best_scores.len(), 24);
    assert_eq!(rec_e.best_scores.len(), 24);
    assert!(
        (rec_z.initial_error - rec_e.initial_error).abs() == 0.0,
        "paired metal init drift: {} vs {}",
        rec_z.initial_error,
        rec_e.initial_error
    );

    let dir = std::env::temp_dir().join(format!("baux_metal_ckpt_{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    unsafe { std::env::set_var("BAUX_CKPT_EVERY", "8") };
    let blocker = dir.join("seed_0_done.cbor").join("simulate_crash");
    fs::create_dir_all(&blocker).expect("create crash blocker");
    let mut stop_cfg = cfg_e.clone();
    stop_cfg.rounds = 17;
    assert!(metal::metal_seed(&stop_cfg, &dom, 0, Some(&dir), &mut None).is_err());
    let live = ckpt::load_meta(&dir, 0)
        .expect("read live meta")
        .expect("live meta present at round 16");
    assert_eq!(live.completed_round, 16);
    fs::remove_dir_all(dir.join("seed_0_done.cbor")).expect("clear blocker");

    let resumed = metal::metal_seed(&cfg_e, &dom, 0, Some(&dir), &mut None).expect("resume 16->24");
    assert_eq!(resumed.best_scores, rec_e.best_scores);
    assert_eq!(resumed.final_score, rec_e.final_score);

    let cached = metal::metal_seed(&cfg_e, &dom, 0, Some(&dir), &mut None).expect("cached done");
    unsafe { std::env::remove_var("BAUX_CKPT_EVERY") };
    assert_eq!(cached.best_scores, rec_e.best_scores);
    let _ = fs::remove_dir_all(&dir);
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[test]
fn test_parity() {
    let bad_single = r#"
algorithm = "single-state-bo"
[perturbation]
sampler = "raasp"
num-pert = 8
"#;
    let bad_spec: OptimizerSpec = baux::artifact::toml::from_str(bad_single).expect("parse single");
    assert!(metal::check_metal(&bad_spec).is_err());

    let bad_div = r#"
algorithm = "bayesian-search"
candidates = 33
arms = 4
acquisition = "upper-confidence-bound"
"#;
    let div_spec: OptimizerSpec = baux::artifact::toml::from_str(bad_div).expect("parse bad div");
    assert!(metal::check_metal(&div_spec).is_err());

    let raasp_toml = r#"
version = 4
experiment = "tokamak-tcv"
backend = "resident-metal"
rounds = 6

[optimizer]
algorithm = "bayesian-search"
candidates = 32
arms = 2
acquisition = "upper-confidence-bound"

[optimizer.perturbation]
sampler = "raasp"
num-pert = 64
raasp-fast = true

[optimizer.surrogate]
metric = "hyperbolic"
local-neighbors = 4
neighbors = 4
fit = "leave-one-out"
fit-candidates = 4
fit-samples = 2

[optimizer.trust-region]
initial = 0.8
min = 0.125
max = 1.6
"#;
    let cfg_r: Config = baux::artifact::toml::from_str(raasp_toml).expect("parse raasp 32");
    metal::check_metal(&cfg_r.optimizer).expect("check metal raasp 32");
    let opt_r = baux::optimizer_config::OptimizerConfig::from_spec(&cfg_r.optimizer)
        .expect("opt cfg raasp");
    assert_eq!(
        opt_r.acquisition,
        baux::optimizer_config::AcquisitionConfig::UCB { beta: 1.0 }
    );
    let dom = exp_domain(&cfg_r);
    let mut host_r = cfg_r.clone();
    host_r.backend = Backend::HostRam;
    let rec_hr = run_seed(&host_r, 0, None, &mut None).expect("host raasp 32");
    let rec_mr = metal::metal_seed(&cfg_r, &dom, 0, None, &mut None).expect("metal raasp 32");
    assert_eq!(rec_hr.best_scores.len(), 6);
    assert_eq!(rec_mr.best_scores.len(), 6);
    assert!(rec_hr.final_score.is_finite() && rec_mr.final_score.is_finite());
    assert!(rec_mr.best_scores.windows(2).all(|w| w[1] >= w[0]));

    let unif_toml = r#"
version = 4
experiment = "tokamak-tcv"
backend = "resident-metal"
rounds = 6

[optimizer]
algorithm = "bayesian-search"
candidates = 128
arms = 4
acquisition = "upper-confidence-bound"

[optimizer.perturbation]
sampler = "uniform"
num-pert = 64

[optimizer.surrogate]
neighbors = 4
fit = "fixed"

[optimizer.trust-region]
initial = 0.4
min = 0.2
max = 0.8

[optimizer.regional-search]
count = 3
sharing = "ellipsoid-ball"
allocation = "joint-acquisition"
controller = "turbo"
restart-memory = "retain-relevant"
restart-center = "maximin"
restart-candidates = 8
success-tolerance = 2
failure-tolerance = 1
"#;
    let cfg_u: Config = baux::artifact::toml::from_str(unif_toml).expect("parse uniform 128");
    metal::check_metal(&cfg_u.optimizer).expect("check metal uniform 128");
    let mut host_u = cfg_u.clone();
    host_u.backend = Backend::HostRam;
    let rec_hu = run_seed(&host_u, 0, None, &mut None).expect("host uniform 128");
    let rec_mu = metal::metal_seed(&cfg_u, &dom, 0, None, &mut None).expect("metal uniform 128");
    assert_eq!(rec_hu.best_scores.len(), 6);
    assert_eq!(rec_mu.best_scores.len(), 6);
    assert!(rec_hu.final_score.is_finite() && rec_mu.final_score.is_finite());
    assert!(rec_mu.best_scores.windows(2).all(|w| w[1] >= w[0]));

    let dir = std::env::temp_dir().join(format!("baux_parity_ckpt_{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    unsafe { std::env::set_var("BAUX_CKPT_EVERY", "2") };
    let blocker = dir.join("seed_0_done.cbor").join("crash");
    fs::create_dir_all(&blocker).expect("mkdir crash blocker");
    let mut half_u = cfg_u.clone();
    half_u.rounds = 5;
    assert!(metal::metal_seed(&half_u, &dom, 0, Some(&dir), &mut None).is_err());
    fs::remove_dir_all(dir.join("seed_0_done.cbor")).expect("remove crash blocker");
    let resumed = metal::metal_seed(&cfg_u, &dom, 0, Some(&dir), &mut None).expect("resume unif");
    unsafe { std::env::remove_var("BAUX_CKPT_EVERY") };
    let _ = fs::remove_dir_all(&dir);
    assert_eq!(resumed.best_scores, rec_mu.best_scores);
    assert_eq!(resumed.final_score, rec_mu.final_score);

    let sobol_toml = unif_toml.replace("sampler = \"uniform\"", "sampler = \"sobol\"");
    let cfg_s: Config = baux::artifact::toml::from_str(&sobol_toml).expect("parse sobol 128");
    metal::check_metal(&cfg_s.optimizer).expect("check metal sobol 128");
    let mut host_s = cfg_s.clone();
    host_s.backend = Backend::HostRam;
    let err_hs = run_seed(&host_s, 0, None, &mut None).expect_err("host sobol >21201");
    let err_ms =
        metal::metal_seed(&cfg_s, &dom, 0, None, &mut None).expect_err("metal sobol >21201");
    assert!(err_hs.contains("21201") && err_ms.contains("21201"));

    let small_bounds = Array2::from_shape_fn((32, 2), |(_, j)| if j == 0 { -1.0 } else { 1.0 });
    let mut sb_opt = UnifiedEngine::from_spec(
        small_bounds,
        &cfg_s.optimizer,
        0,
        AnyBackend::Cpu,
        baux::Key::new(91),
    )
    .expect("host 32d");
    for r in 1..=6 {
        let mut sr = round_rng(91, r);
        let (next_opt, u) = sb_opt.ask(4, &mut sr).expect("sobol ask 4");
        sb_opt = next_opt;
        let y = Array2::from_shape_fn((4, 1), |(i, _)| {
            (0..32).map(|j| -(u[[i, j]] - 0.5).powi(2)).sum::<f64>()
        });
        sb_opt = sb_opt
            .tell(&u.view(), &y.view(), None, &mut sr)
            .expect("sobol tell 4");
    }
    let opt_s = baux::optimizer_config::OptimizerConfig::from_spec(&cfg_s.optimizer)
        .expect("opt cfg sobol");
    let spec_s = crate::experiment::julia::metal_step::MetalJobSpec::from_parts(
        0,
        91,
        32,
        1,
        6,
        &cfg_s.optimizer,
        opt_s,
        None,
    )
    .expect("32d sobol metal spec");
    let (st_s, _) = crate::experiment::julia::metal::metal_job(
        &spec_s,
        None,
        |sc| (-sc).max(0.0),
        |rows| {
            let scs: Vec<f64> = rows
                .iter()
                .map(|row| {
                    row.iter()
                        .map(|&w| {
                            let u = f64::from(baux::shm_slab::decode_half(w)).clamp(0.0, 1.0);
                            -(u - 0.5).powi(2)
                        })
                        .sum()
                })
                .collect();
            let n = scs.len();
            Ok((scs, vec![0.0; n], 0))
        },
    )
    .expect("32d sobol metal job");
    assert_eq!(st_s.best_scores.len(), 6);
    assert!(st_s.best_score.is_finite());
}

fn drive_typed<B: OptBackend>(
    cfg: &Config,
    backend: B,
    seed: u64,
) -> (&'static str, Vec<baux::StepTrace>, f64, [u8; 32]) {
    let key = AffineKey::new(seed);
    let (warmup, key) = typed_warmup(cfg, backend, key).expect("typed warmup");
    let dev = warmup.device_name();
    let (x0, warmup, key) = warmup.design_step(2, key).expect("design step");
    let s0 = x0.as_slice_memory_order().expect("contiguous design");
    let v0 = eval_batch(&mut None, s0, x0.nrows()).expect("eval warmup");
    let y0 = Array2::from_shape_fn((v0.len(), 1), |(i, _)| v0[i]);
    let (ready, key) = warmup
        .activate_ready(&x0.view(), &y0.view(), key)
        .expect("activate ready");
    let forked = ready.clone();
    assert!(ready.shares_storage(&forked));
    let fork_digest = forked.digest();

    let (ready_end, key, traces) = baux::scan_step(ready, key, 3, 2, |_, x| {
        let sl = x.as_slice_memory_order().expect("contiguous batch");
        let scs = eval_batch(&mut None, sl, x.nrows()).expect("eval active");
        Ok(Array2::from_shape_fn((scs.len(), 1), |(i, _)| scs[i]))
    })
    .expect("scan step");
    let _ = key.into_key();
    assert_eq!(forked.digest(), fork_digest);
    let best = ready_end.best_y().expect("incumbent y");
    (dev, traces, best, ready_end.digest())
}

#[test]
fn unified_backends() {
    let toml_str = r#"
version = 4
experiment = "tokamak-tcv"
rounds = 4

[optimizer]
algorithm = "bayesian-search"
candidates = 32
arms = 2
acquisition = "upper-confidence-bound"

[optimizer.perturbation]
sampler = "raasp"
num-pert = 32
raasp-fast = true

[optimizer.surrogate]
metric = "hyperbolic"
local-neighbors = 2
neighbors = 2
fit = "fixed"

[optimizer.trust-region]
initial = 0.8
min = 0.125
max = 1.6

[optimizer.regional-search]
count = 2
sharing = "box-contain"
allocation = "incumbent-ucb1"
controller = "turbo"
success-tolerance = 2
failure-tolerance = 3
"#;
    let mut cfg: Config = baux::artifact::toml::from_str(toml_str).expect("parse tokagym unified");
    let (dev_c, tr_c, best_c, dig_c) = drive_typed(&cfg, baux::CpuBackend, 404);
    let (dev_a, tr_a, best_a, dig_a) = drive_typed(&cfg, AnyBackend::Cpu, 404);
    assert_eq!(dev_c, "cpu");
    assert_eq!(dev_a, "cpu");
    assert_eq!(tr_c.len(), 3);
    assert_eq!(dig_c, dig_a);
    assert_eq!(best_c, best_a);
    assert!(best_c.is_finite());

    cfg.backend = Backend::HostRam;
    let host_rep = run_opt(&cfg, None, None).expect("run_opt host");
    assert_eq!(host_rep["rounds"], 4);
    assert_eq!(host_rep["regions"], 2);

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    {
        let (dev_m, tr_m, best_m, dig_m) = drive_typed(&cfg, baux::MetalBackend, 404);
        let (dev_am, tr_am, best_am, dig_am) = drive_typed(&cfg, AnyBackend::Metal, 404);
        assert_eq!(dev_m, "metal");
        assert_eq!(dev_am, "metal");
        assert_eq!(tr_m.len(), 3);
        assert_eq!(dig_m, dig_am);
        assert_eq!(best_m, best_am);
        assert!(best_m.is_finite());
        assert!(tr_am.windows(2).all(|w| w[1].best_y >= w[0].best_y));

        cfg.backend = Backend::ResidentMetal;
        let metal_rep = run_opt(&cfg, None, None).expect("run_opt metal");
        assert_eq!(metal_rep["rounds"], 4);
        assert_eq!(metal_rep["regions"], 2);
        assert!(metal_rep["final_score_mean"].as_f64().unwrap().is_finite());
    }
}
