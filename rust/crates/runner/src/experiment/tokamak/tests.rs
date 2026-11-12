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
    let mut init_rng = prng(seed);
    let mut opt = make_opt(&half_cfg, Some(&dir), 0, &mut init_rng).expect("make opt");
    let dom = exp_domain(&half_cfg);
    let mut st = LiveCkpt::empty(2);
    st.elapsed_seconds = 0.1;
    for round in 1..=2 {
        let mut sr = round_rng(seed, round);
        let u = opt.ask(1, &mut sr).expect("ask");
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
        opt.tell(&u.view(), &y.view(), &mut sr).expect("tell");
        st.best_scores.push(st.best_score);
        st.eval_scores.push(vec![sc]);
        st.completed_round = round;
    }
    save_live(&dir, 0, &opt, &st).expect("save live ckpt");

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
