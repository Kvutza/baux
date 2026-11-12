//! Merkle-locked `Oceananigans.jl` and `FUSE.jl` physics experiment runner for
//! `baux tune`, backed by zero-copy page-aligned `ShmSlab` (`mmap`) IPC and
//! the full BAUX optimizer surface (resident Apple Metal GPU, host in-memory
//! and disk-backed `BayesOptimizer`, MORBO/Pareto, and Candle Metal PPO).

use baux::artifact::{
    json::{Value, json},
    runs,
};
use baux::config::{AcquisitionSpec, OptimizerSpec};
use baux::optimizer::DomainFrame;
use baux::unified_mem::ShmSlab;
use baux::{BayesOptimizer, prng};
use deser::{Deserialize, Serialize};
use ndarray::Array2;
use std::{
    fs,
    io::Write,
    path::{Path, PathBuf},
    process::{Command, Stdio},
    time::Instant,
};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Deserialize, Serialize)]
#[deser(rename_all = "kebab-case")]
enum Backend {
    #[default]
    Auto,
    ResidentMetal,
    HostRam,
    HostDisk,
}

#[derive(Clone, Deserialize, Serialize)]
#[deser(deny_unknown_fields, rename_all = "kebab-case")]
struct MorboCfg {
    #[deser(default = def_mreg())]
    regions: usize,
    #[deser(default = def_alpha())]
    alpha: f64,
    #[deser(default)]
    preferences: Vec<f32>,
    #[deser(default)]
    scales: Vec<f32>,
    #[deser(default = def_rescal())]
    rescalarize: baux::Rescalarize,
}

impl Default for MorboCfg {
    fn default() -> Self {
        Self {
            regions: def_mreg(),
            alpha: def_alpha(),
            preferences: Vec::new(),
            scales: Vec::new(),
            rescalarize: def_rescal(),
        }
    }
}

#[derive(Clone, Default, Deserialize, Serialize)]
#[deser(default, deny_unknown_fields, rename_all = "kebab-case")]
struct ManifoldCfg {
    scale_x: bool,
    metric_learning: bool,
    stiefel_rank: Option<usize>,
    calibrate_alpha: Option<f64>,
    noise_aware: bool,
}

#[derive(Clone, Deserialize, Serialize)]
#[deser(deny_unknown_fields, rename_all = "kebab-case")]
struct OceanCfg {
    #[deser(default = def_task())]
    task: String,
    #[deser(default = def_grid())]
    nx: usize,
    #[deser(default = def_ny())]
    ny: usize,
    #[deser(default = def_grid())]
    nz: usize,
    #[deser(default = def_act())]
    actuators: usize,
    #[deser(default = def_steps())]
    steps: usize,
    #[deser(default = def_dt())]
    dt: f64,
    #[deser(default = def_bound())]
    bound: f64,
    #[deser(default)]
    disk: bool,
    #[deser(default)]
    objectives: Vec<String>,
}

impl Default for OceanCfg {
    fn default() -> Self {
        Self {
            task: def_task(),
            nx: def_grid(),
            ny: def_ny(),
            nz: def_grid(),
            actuators: def_act(),
            steps: def_steps(),
            dt: def_dt(),
            bound: def_bound(),
            disk: false,
            objectives: Vec::new(),
        }
    }
}

#[derive(Clone, Deserialize, Serialize)]
#[deser(deny_unknown_fields, rename_all = "kebab-case")]
struct FuseCfg {
    #[deser(default = def_case())]
    case: String,
    #[deser(default = def_workflow())]
    workflow: String,
    #[deser(default = def_obj())]
    objective: String,
    #[deser(default)]
    objectives: Vec<String>,
    #[deser(default)]
    constraints: Vec<String>,
    #[deser(default = def_penalty())]
    penalty: f64,
}

impl Default for FuseCfg {
    fn default() -> Self {
        Self {
            case: def_case(),
            workflow: def_workflow(),
            objective: def_obj(),
            objectives: Vec::new(),
            constraints: Vec::new(),
            penalty: def_penalty(),
        }
    }
}

#[derive(Clone, Deserialize, Serialize)]
#[deser(deny_unknown_fields, rename_all = "kebab-case")]
struct Config {
    version: u32,
    experiment: String,
    rounds: usize,
    #[deser(default = def_reps())]
    repetitions: u32,
    #[deser(default)]
    backend: Backend,
    #[deser(default)]
    ocean: OceanCfg,
    #[deser(default)]
    fuse: FuseCfg,
    #[deser(default)]
    morbo: Option<MorboCfg>,
    #[deser(default)]
    manifold: Option<ManifoldCfg>,
    optimizer: OptimizerSpec,
}

const fn def_mreg() -> usize {
    4
}
const fn def_alpha() -> f64 {
    0.05
}
const fn def_rescal() -> baux::Rescalarize {
    baux::Rescalarize::OnRestart
}

fn def_task() -> String {
    "kelvin-helmholtz".into()
}
const fn def_grid() -> usize {
    64
}
const fn def_ny() -> usize {
    1
}
const fn def_act() -> usize {
    2
}
const fn def_steps() -> usize {
    20
}
const fn def_dt() -> f64 {
    0.04
}
const fn def_bound() -> f64 {
    0.25
}
fn def_case() -> String {
    "ARC".into()
}
fn def_workflow() -> String {
    "stationary-plasma".into()
}
fn def_obj() -> String {
    "max_fusion".into()
}
const fn def_penalty() -> f64 {
    25.0
}
const fn def_reps() -> u32 {
    1
}

fn is_ocean(cfg: &Config) -> bool {
    cfg.experiment == "oceananigans"
}

fn obj_count(cfg: &Config) -> usize {
    let explicit = if is_ocean(cfg) {
        cfg.ocean.objectives.len()
    } else {
        cfg.fuse.objectives.len()
    };
    if explicit > 0 {
        return explicit;
    }
    if let Some(m) = &cfg.morbo {
        if !m.scales.is_empty() {
            return m.scales.len();
        }
        if !m.preferences.is_empty() {
            return m.preferences.len();
        }
    }
    if let OptimizerSpec::BayesianSearch {
        acquisition: AcquisitionSpec::Pareto { scales },
        ..
    } = &cfg.optimizer
    {
        if !scales.is_empty() {
            return scales.len();
        }
    }
    1
}

fn fuse_objs(cfg: &Config) -> String {
    if cfg.fuse.objectives.is_empty() {
        cfg.fuse.objective.clone()
    } else {
        cfg.fuse.objectives.join(",")
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ExecMode {
    ResidentMetal,
    HostRam,
    HostDisk,
    MetalPpo,
}

fn metal_ok(cfg: &Config) -> Result<(), String> {
    if !cfg!(all(target_os = "macos", target_arch = "aarch64")) {
        return Err("resident-metal backend requires macOS with Metal feature".into());
    }
    if cfg.manifold.is_some() {
        return Err("manifold host options are not supported on resident-metal".into());
    }
    if let Some(ref m) = cfg.morbo {
        let n_obj = obj_count(cfg);
        if m.rescalarize != baux::Rescalarize::OnRestart {
            return Err("resident-metal MORBO requires morbo.rescalarize = 'on-restart'".into());
        }
        if !(2..=8).contains(&m.regions) {
            return Err("resident-metal MORBO requires morbo.regions in 2..=8".into());
        }
        if !m.alpha.is_finite() || m.alpha < 0.0 {
            return Err("morbo.alpha must be finite and nonnegative".into());
        }
        if !m.preferences.is_empty()
            && (m.preferences.len() != n_obj
                || m.preferences.iter().any(|w| !w.is_finite() || *w <= 0.0))
        {
            return Err(
                "morbo.preferences must supply one positive finite weight per objective".into(),
            );
        }
        if !m.scales.is_empty()
            && (m.scales.len() != n_obj || m.scales.iter().any(|s| !s.is_finite() || *s <= 0.0))
        {
            return Err("morbo.scales must supply one positive finite scale per objective".into());
        }
    }
    match &cfg.optimizer {
        OptimizerSpec::OperatorBo {
            perturbation: Some(_),
            ..
        } => Err("resident-metal does not accept host [optimizer.perturbation] overrides".into()),
        OptimizerSpec::OperatorBo { .. } => Ok(()),
        OptimizerSpec::BayesianSearch {
            candidates,
            arms,
            perturbation,
            ..
        } => {
            if perturbation.is_some() {
                return Err(
                    "resident-metal does not accept host [optimizer.perturbation] overrides".into(),
                );
            }
            if *arms == 0
                || *candidates == 0
                || *candidates > 4
                || !candidates.is_multiple_of(*arms)
            {
                return Err(
                    "resident-metal requires 1..=4 total candidates divisible by arms".into(),
                );
            }
            Ok(())
        }
        OptimizerSpec::Ppo | OptimizerSpec::Muon | OptimizerSpec::Adamw => {
            Err("resident-metal ENN backend requires operator-bo or bayesian-search".into())
        }
    }
}

fn resolve_mode(cfg: &Config) -> Result<ExecMode, String> {
    match &cfg.optimizer {
        OptimizerSpec::Muon => {
            return Err(
                "optimizer = 'muon' requires analytical parameter gradients, which black-box Oceananigans.jl / FUSE.jl workers do not expose".into(),
            );
        }
        OptimizerSpec::Adamw => {
            return Err(
                "optimizer = 'adamw' requires analytical parameter gradients, which black-box Oceananigans.jl / FUSE.jl workers do not expose".into(),
            );
        }
        OptimizerSpec::Ppo => {
            return match cfg.backend {
                Backend::Auto | Backend::ResidentMetal => Ok(ExecMode::MetalPpo),
                Backend::HostRam | Backend::HostDisk => {
                    Err("optimizer = 'ppo' runs on Candle/Metal GPU; use backend = 'auto'".into())
                }
            };
        }
        _ => {}
    }

    if is_ocean(cfg) && cfg.ocean.disk && cfg.backend == Backend::Auto {
        return Ok(ExecMode::HostDisk);
    }

    match cfg.backend {
        Backend::ResidentMetal => {
            metal_ok(cfg)?;
            Ok(ExecMode::ResidentMetal)
        }
        Backend::HostRam => Ok(ExecMode::HostRam),
        Backend::HostDisk => Ok(ExecMode::HostDisk),
        Backend::Auto => {
            let prefer_metal = match &cfg.optimizer {
                OptimizerSpec::OperatorBo {
                    perturbation: None, ..
                } => true,
                OptimizerSpec::BayesianSearch {
                    distribution: Some(_),
                    perturbation: None,
                    ..
                } => true,
                _ => false,
            };
            if prefer_metal && metal_ok(cfg).is_ok() {
                Ok(ExecMode::ResidentMetal)
            } else {
                Ok(ExecMode::HostRam)
            }
        }
    }
}

fn opt_label(cfg: &Config, mode: ExecMode) -> String {
    let n_obj = obj_count(cfg);
    match mode {
        ExecMode::MetalPpo => "candle-ppo-metal".into(),
        ExecMode::ResidentMetal => {
            if cfg.morbo.is_some() && n_obj >= 2 {
                "resident-metal-morbo".into()
            } else if n_obj >= 2 {
                "resident-metal-pareto".into()
            } else if matches!(cfg.optimizer, OptimizerSpec::OperatorBo { .. }) {
                "resident-metal-operator-bo".into()
            } else {
                "resident-metal-bo".into()
            }
        }
        ExecMode::HostRam => {
            if cfg.morbo.is_some() && n_obj >= 2 {
                "host-ram-morbo".into()
            } else {
                "host-ram-bo".into()
            }
        }
        ExecMode::HostDisk => {
            if cfg.morbo.is_some() && n_obj >= 2 {
                "host-disk-bpann-morbo".into()
            } else {
                "host-disk-bpann-bo".into()
            }
        }
    }
}

pub(super) fn execute(root: &Path, path: &Path, rounds: Option<u32>) -> Result<bool, String> {
    let text = fs::read_to_string(path).map_err(|e| e.to_string())?;
    let val: Value = baux::artifact::toml::from_str(&text).map_err(|e| e.to_string())?;
    let exp = val["experiment"].as_str().unwrap_or_default();
    if exp != "oceananigans" && exp != "fuse" {
        return Ok(false);
    }

    let mut cfg: Config = baux::artifact::toml::from_str(&text).map_err(|e| e.to_string())?;
    if cfg.version != 4 {
        return Err(format!(
            "{exp} experiment requires version 4 tune-spec schema"
        ));
    }
    if let Some(r) = rounds {
        cfg.rounds = r as usize;
    }
    if cfg.rounds == 0 {
        return Err("rounds must be positive".into());
    }
    let mode = resolve_mode(&cfg)?;

    let commit = runs::jj(root, &["log", "-r", "@", "--no-graph", "-T", "commit_id"])?;
    let exe = std::env::current_exe().map_err(|e| e.to_string())?;
    let nonce = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_err(|e| e.to_string())?
        .as_nanos();
    let subdir = if is_ocean(&cfg) {
        "oceananigans"
    } else {
        "fuse"
    };
    let output = root
        .join(".cache/baux/runs")
        .join(subdir)
        .join(nonce.to_string());
    fs::create_dir_all(&output).map_err(|e| e.to_string())?;
    let resolved = baux::artifact::toml::pretty_string(&cfg).map_err(|e| e.to_string())?;
    fs::write(output.join("config.toml"), &resolved).map_err(|e| e.to_string())?;

    let mut record = runs::start_worker(root, &output, &resolved, commit.trim(), &exe)?;
    record.model = model_label(&cfg);
    record.optimizer = opt_label(&cfg, mode);
    record.protocol = runs::digest(resolved.as_bytes());

    let cbor_name = format!("{subdir}.cbor");
    let result = run_opt(&cfg, mode, root, &output).and_then(|report| {
        let bytes = baux::artifact::cbor::to_vec(&report).map_err(|e| e.to_string())?;
        fs::write(output.join(&cbor_name), &bytes).map_err(|e| e.to_string())?;
        println!(
            "{} ({}) experiment completed. Evidence: {}",
            if is_ocean(&cfg) {
                "Oceananigans.jl"
            } else {
                "FUSE.jl"
            },
            opt_label(&cfg, mode),
            output.display()
        );
        Ok(())
    });
    runs::finish(root, &mut record, &result)?;
    result?;
    Ok(true)
}

fn model_label(cfg: &Config) -> String {
    if is_ocean(cfg) {
        let oc = &cfg.ocean;
        format!(
            "oceananigans-0.117.2-{}-{}x{}x{}x{}",
            oc.task, oc.nx, oc.ny, oc.nz, oc.actuators
        )
    } else {
        let fc = &cfg.fuse;
        format!(
            "fuse-1.2.1-{}-{}-{}",
            fc.case.to_ascii_lowercase(),
            fc.workflow,
            fc.objective
        )
    }
}

fn julia_bin() -> String {
    if let Ok(bin) = std::env::var("JULIA_BIN")
        && !bin.trim().is_empty()
    {
        return bin;
    }
    if let Ok(home) = std::env::var("HOME") {
        let up = PathBuf::from(home).join(".juliaup/bin/julia");
        if up.exists() {
            return up.to_string_lossy().into_owned();
        }
    }
    "julia".into()
}

fn ensure_upstream(root: &Path, lock_name: &str) -> Result<Value, String> {
    if let Ok(v) = baux::artifact::merkle::verify_lock(root, lock_name, None) {
        return Ok(v);
    }
    let lock_path = root
        .join("research/upstream")
        .join(format!("{lock_name}.lock.toml"));
    let text =
        fs::read_to_string(&lock_path).map_err(|e| format!("read {}: {e}", lock_path.display()))?;
    let lock: Value = baux::artifact::toml::from_str(&text).map_err(|e| e.to_string())?;
    let rel = lock["merkle"]["cache_relpath"]
        .as_str()
        .ok_or("missing merkle.cache_relpath")?;
    let target = root.join(rel);
    if !target.exists() {
        let url = lock["repository"]["url"]
            .as_str()
            .ok_or("missing repository.url")?;
        let rev = lock["repository"]["revision"]
            .as_str()
            .ok_or("missing repository.revision")?;
        if let Some(p) = target.parent() {
            fs::create_dir_all(p).map_err(|e| e.to_string())?;
        }
        let init = Command::new("git")
            .arg("init")
            .arg(&target)
            .status()
            .map_err(|e| e.to_string())?;
        if !init.success() {
            return Err(format!("git init failed for {}", target.display()));
        }
        let fetch = Command::new("git")
            .arg("-C")
            .arg(&target)
            .args(["fetch", "--depth", "1", url, rev])
            .status()
            .map_err(|e| e.to_string())?;
        if !fetch.success() {
            return Err(format!("git fetch {rev} failed from {url}"));
        }
        let co = Command::new("git")
            .arg("-C")
            .arg(&target)
            .args(["checkout", "--detach", "FETCH_HEAD"])
            .status()
            .map_err(|e| e.to_string())?;
        if !co.success() {
            return Err(format!("git checkout {rev} failed"));
        }
    }
    baux::artifact::merkle::verify_lock(root, lock_name, None)
}

fn resolve_space(
    root: &Path,
    cfg: &Config,
) -> Result<(DomainFrame, Vec<f64>, Vec<f64>, usize), String> {
    if is_ocean(cfg) {
        let oc = &cfg.ocean;
        if oc.nx == 0 || oc.nz == 0 || oc.steps == 0 || oc.bound <= 0.0 {
            return Err("oceananigans config requires positive grid, steps, and bound".into());
        }
        let (dim, state_len) = if oc.task == "kelvin-helmholtz" {
            let ny_eff = oc.ny.max(1);
            if oc.actuators == 0 {
                return Err("oceananigans actuators must be >= 1".into());
            }
            (oc.nx * ny_eff * oc.nz * oc.actuators, 2 * oc.nx * oc.nz)
        } else if oc.task == "wind-mixing" {
            let ny_eff = oc.ny.max(1);
            (2 * oc.nx * ny_eff, 2 * oc.nx * oc.nz)
        } else {
            return Err(format!("unknown oceananigans task '{}'", oc.task));
        };
        let mut bounds = Array2::zeros((dim, 2));
        let lo = if oc.task == "wind-mixing" {
            -2.5
        } else {
            -oc.bound
        };
        let hi = if oc.task == "wind-mixing" {
            2.5
        } else {
            oc.bound
        };
        for j in 0..dim {
            bounds[[j, 0]] = lo;
            bounds[[j, 1]] = hi;
        }
        let low = vec![lo; dim];
        let span = vec![hi - lo; dim];
        let domain = DomainFrame::new(bounds).map_err(|e| e.to_string())?;
        Ok((domain, low, span, state_len))
    } else {
        let sock_path = ensure_daemon(root, false)?;
        let key = fuse_key(cfg);
        let reply = daemon_request(&sock_path, &format!("BOUNDS\t{key}\n"))?;
        let rest = reply
            .trim()
            .strip_prefix("BOUNDS\t")
            .ok_or_else(|| format!("unexpected FUSE daemon BOUNDS reply: {reply}"))?;
        let (lo_csv, hi_csv) = rest
            .split_once('\t')
            .ok_or_else(|| format!("malformed FUSE daemon BOUNDS reply: {reply}"))?;
        let low: Vec<f64> = lo_csv
            .split(',')
            .filter(|s| !s.is_empty())
            .map(|s| {
                s.trim()
                    .parse::<f64>()
                    .map_err(|e| format!("parse lo: {e}"))
            })
            .collect::<Result<_, _>>()?;
        let high: Vec<f64> = hi_csv
            .split(',')
            .filter(|s| !s.is_empty())
            .map(|s| {
                s.trim()
                    .parse::<f64>()
                    .map_err(|e| format!("parse hi: {e}"))
            })
            .collect::<Result<_, _>>()?;
        if low.is_empty() || low.len() != high.len() {
            return Err(format!(
                "invalid FUSE daemon bounds lengths: {} vs {}",
                low.len(),
                high.len()
            ));
        }
        let dim = low.len();
        let mut bounds = Array2::zeros((dim, 2));
        let mut span = Vec::with_capacity(dim);
        for j in 0..dim {
            let (lo, hi) = (low[j], high[j]);
            if !(hi > lo) {
                return Err(format!("invalid FUSE interval [{lo}, {hi}] at dim {j}"));
            }
            bounds[[j, 0]] = lo;
            bounds[[j, 1]] = hi;
            span.push(hi - lo);
        }
        let domain = DomainFrame::new(bounds).map_err(|e| e.to_string())?;
        Ok((domain, low, span, 256))
    }
}

fn fuse_key(cfg: &Config) -> String {
    let fc = &cfg.fuse;
    format!(
        "{}|{}|{}|{}|{}",
        fc.case,
        fc.workflow,
        fuse_objs(cfg),
        fc.penalty,
        fc.constraints.join(",")
    )
}

fn ocean_key(cfg: &Config) -> String {
    let oc = &cfg.ocean;
    format!(
        "{}|{}|{}|{}|{}|{}|{}|{}",
        oc.task,
        oc.nx,
        oc.ny,
        oc.nz,
        oc.actuators,
        oc.steps,
        oc.dt,
        obj_count(cfg)
    )
}

fn daemon_sock(root: &Path, ocean: bool) -> PathBuf {
    let tag = &baux::artifact::blake3::hash_hex(root.as_os_str().as_encoded_bytes())[..10];
    let kind = if ocean { "ocean" } else { "fuse" };
    std::env::temp_dir().join(format!("baux_jl_{tag}_{kind}.sock"))
}

fn daemon_request(sock_path: &Path, msg: &str) -> Result<String, String> {
    use std::io::BufRead;
    use std::os::unix::net::UnixStream;
    let mut stream = UnixStream::connect(sock_path)
        .map_err(|e| format!("connect {}: {e}", sock_path.display()))?;
    stream
        .write_all(msg.as_bytes())
        .map_err(|e| e.to_string())?;
    stream.flush().map_err(|e| e.to_string())?;
    let mut reader = std::io::BufReader::new(stream);
    let mut line = String::new();
    reader.read_line(&mut line).map_err(|e| e.to_string())?;
    Ok(line)
}

fn ensure_daemon(root: &Path, ocean: bool) -> Result<PathBuf, String> {
    let sock_path = daemon_sock(root, ocean);
    if let Ok(pong) = daemon_request(&sock_path, "PING\n") {
        if pong.trim() == "PONG" {
            return Ok(sock_path);
        }
    }
    let proj = if ocean {
        root.join("evals/oceananigans")
    } else {
        root.join("evals/fuse")
    };
    let log_dir = root.join(".cache/baux/shm");
    fs::create_dir_all(&log_dir).map_err(|e| e.to_string())?;
    let log_path = log_dir.join(if ocean {
        "julia_ocean_daemon.log"
    } else {
        "julia_fuse_daemon.log"
    });
    let log_file = fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&log_path)
        .map_err(|e| format!("open daemon log {}: {e}", log_path.display()))?;

    let mut cmd = Command::new(julia_bin());
    cmd.args(["--startup-file=no", "--threads=auto"]);
    if !ocean {
        cmd.arg("-O1");
    }
    cmd.arg(format!("--project={}", proj.display()))
        .arg(proj.join("worker.jl"))
        .arg("--daemon")
        .arg(&sock_path)
        .current_dir(root)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::from(log_file));
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        cmd.process_group(0);
    }
    let mut child = cmd
        .spawn()
        .map_err(|e| format!("spawn julia daemon {}: {e}", proj.display()))?;

    let deadline = Instant::now() + std::time::Duration::from_secs(240);
    while Instant::now() < deadline {
        if let Ok(Some(status)) = child.try_wait() {
            let log_tail = fs::read_to_string(&log_path).unwrap_or_default();
            return Err(format!(
                "julia daemon exited prematurely with {status}: {log_tail}"
            ));
        }
        if let Ok(pong) = daemon_request(&sock_path, "PING\n") {
            if pong.trim() == "PONG" {
                return Ok(sock_path);
            }
        }
        std::thread::sleep(std::time::Duration::from_millis(80));
    }
    Err(format!(
        "timed out waiting for julia daemon socket {}",
        sock_path.display()
    ))
}

struct JuliaWorker {
    _sock: std::os::unix::net::UnixStream,
    slab: ShmSlab,
    low: Vec<f64>,
    span: Vec<f64>,
    num_obj: usize,
}

impl JuliaWorker {
    fn spawn(
        root: &Path,
        cfg: &Config,
        max_batch: usize,
        low: Vec<f64>,
        span: Vec<f64>,
        state_len: usize,
    ) -> Result<Self, String> {
        use std::io::BufRead;
        use std::os::unix::net::UnixStream;

        let num_dim = low.len();
        let num_obj = obj_count(cfg);
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_err(|e| e.to_string())?
            .as_nanos();
        let shm_path = root.join(".cache/baux/shm").join(format!(
            "{}_{}_{}.mmap",
            cfg.experiment,
            std::process::id(),
            nonce
        ));
        let slab = ShmSlab::create_multi(shm_path.clone(), max_batch, num_dim, num_obj, state_len)?;

        let ocean = is_ocean(cfg);
        let sock_path = ensure_daemon(root, ocean)?;
        let cfg_key = if ocean { ocean_key(cfg) } else { fuse_key(cfg) };
        let mut sock = UnixStream::connect(&sock_path)
            .map_err(|e| format!("connect julia daemon {}: {e}", sock_path.display()))?;
        let open_cmd = format!("OPEN\t{}\t{cfg_key}\n", shm_path.display());
        sock.write_all(open_cmd.as_bytes())
            .map_err(|e| e.to_string())?;
        sock.flush().map_err(|e| e.to_string())?;
        let mut line = String::new();
        {
            let mut reader = std::io::BufReader::new(&sock);
            reader
                .read_line(&mut line)
                .map_err(|e| format!("julia daemon OPEN handshake failed: {e}"))?;
        }
        if !line.starts_with("OK\t") {
            return Err(format!("julia daemon rejected OPEN: {}", line.trim()));
        }

        Ok(Self {
            _sock: sock,
            slab,
            low,
            span,
            num_obj,
        })
    }

    fn call_doorbell(&mut self, cmd_id: u32, count: usize) -> Result<u64, String> {
        let shm_cmd = match cmd_id {
            1 => baux::unified_mem::CMD_EVAL,
            2 => baux::unified_mem::CMD_SNAP,
            other => return Err(format!("unsupported julia doorbell cmd {other}")),
        };
        let (dt_us, _threads, _rt_ns) =
            self.slab
                .doorbell_wait(shm_cmd, count, std::time::Duration::from_secs(600), || {
                    Ok(())
                })?;
        Ok(dt_us)
    }

    fn eval_unit(
        &mut self,
        unit: &ndarray::ArrayView2<f64>,
    ) -> Result<(Vec<f64>, Vec<f64>, u64), String> {
        let count = self.slab.write_unit(unit, &self.low, &self.span)?;
        let dt_us = self.call_doorbell(1, count)?;
        let scores = self.slab.read_scores(count).to_vec();
        let viols = self.slab.read_viols(count).to_vec();
        Ok((scores, viols, dt_us))
    }

    fn eval_snap(&mut self, unit_row: &[f64]) -> Result<(f64, f64, Vec<f32>), String> {
        let arr = Array2::from_shape_vec((1, unit_row.len()), unit_row.to_vec())
            .map_err(|e| e.to_string())?;
        self.slab.write_unit(&arr.view(), &self.low, &self.span)?;
        let _ = self.call_doorbell(2, 1)?;
        let sc = self.slab.read_scores(1)[0];
        let vl = self.slab.read_viols(1)[0];
        let st = self.slab.read_state().to_vec();
        Ok((sc, vl, st))
    }

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    fn eval_half(&mut self, half_ptr: *const u16) -> Result<(Vec<f64>, f64, u64), String> {
        unsafe { self.slab.write_half(0, half_ptr, &self.low, &self.span)? };
        let dt_us = self.call_doorbell(1, 1)?;
        let scores = self.slab.read_scores(1).to_vec();
        let viol = self.slab.read_viols(1)[0];
        Ok((scores, viol, dt_us))
    }

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    fn half_snap(&mut self, half_ptr: *const u16) -> Result<Vec<f32>, String> {
        unsafe { self.slab.write_half(0, half_ptr, &self.low, &self.span)? };
        let _ = self.call_doorbell(2, 1)?;
        Ok(self.slab.read_state().to_vec())
    }
}

impl Drop for JuliaWorker {
    fn drop(&mut self) {
        let _ = self.slab.doorbell_wait(
            baux::unified_mem::CMD_DETACH,
            0,
            std::time::Duration::from_secs(2),
            || Ok(()),
        );
    }
}

struct SeedRun {
    rep: u32,
    seed: u64,
    num_dim: usize,
    init_err: f64,
    final_score: f64,
    final_err: f64,
    final_viol: f64,
    reduction_pct: f64,
    accepts: usize,
    elapsed_s: f64,
    worker_s: f64,
    best_scores: Vec<f64>,
    best_viols: Vec<f64>,
    state_snap: Vec<f32>,
}

fn loss_metric(cfg: &Config, score: f64) -> f64 {
    if is_ocean(cfg) {
        (-score).max(0.0)
    } else {
        // Shift into strictly positive loss so maximize objectives (where
        // `score > 0`, e.g. `max_fusion` MW or `max_power_electric_net` MW)
        // stay monotonic in `tracking_error` SVG plots.
        (2500.0 - score).max(0.0)
    }
}

fn mean_sem(vals: &[f64]) -> (f64, f64) {
    let n = vals.len();
    if n == 0 {
        return (0.0, 0.0);
    }
    let m = vals.iter().sum::<f64>() / (n as f64);
    if n <= 1 {
        return (m, 0.0);
    }
    let var = vals.iter().map(|v| (v - m).powi(2)).sum::<f64>() / ((n - 1) as f64);
    (m, (var / (n as f64)).sqrt())
}

fn opt_cfg(cfg: &Config, n_obj: usize) -> Result<baux::optimizer_config::OptimizerConfig, String> {
    let mut cfg_opt = baux::optimizer_config::OptimizerConfig::from_spec(&cfg.optimizer)
        .map_err(|e| e.to_string())?;
    let noise = cfg.manifold.as_ref().is_some_and(|m| m.noise_aware);
    if noise {
        cfg_opt.noise_aware = true;
    }
    if let Some(ref mf) = cfg.manifold {
        if let baux::optimizer_config::SurrogateConfig::ENN(ref mut enn) = cfg_opt.surrogate {
            if mf.scale_x {
                enn.scale_x = true;
            }
            if mf.metric_learning {
                enn.metric_learning = baux::layout::MetricLearning::Auto;
            }
            if let Some(rank) = mf.stiefel_rank {
                enn.stiefel_rank = Some(rank);
            }
            if let Some(alpha) = mf.calibrate_alpha {
                enn.calibrate_alpha = Some(alpha);
            }
        }
    }
    if let Some(ref m) = cfg.morbo {
        if n_obj >= 2 {
            let len = match cfg_opt.trust_region.clone() {
                baux::trust_region::TrustRegionConfig::Turbo(l) => l,
                baux::trust_region::TrustRegionConfig::Morbo(ms) => ms.length,
            };
            cfg_opt.trust_region =
                baux::trust_region::TrustRegionConfig::Morbo(baux::trust_region::MorboTRSettings {
                    num_metrics: n_obj,
                    alpha: m.alpha,
                    length: len,
                    rescalarize: m.rescalarize,
                    noise_aware: noise || cfg_opt.noise_aware,
                });
        }
    } else if n_obj >= 2 {
        let len = match cfg_opt.trust_region.clone() {
            baux::trust_region::TrustRegionConfig::Turbo(l) => l,
            baux::trust_region::TrustRegionConfig::Morbo(ms) => ms.length,
        };
        cfg_opt.trust_region =
            baux::trust_region::TrustRegionConfig::Morbo(baux::trust_region::MorboTRSettings {
                num_metrics: n_obj,
                alpha: 0.05,
                length: len,
                rescalarize: baux::Rescalarize::OnRestart,
                noise_aware: noise || cfg_opt.noise_aware,
            });
    } else if noise {
        if let baux::trust_region::TrustRegionConfig::Morbo(ref mut ms) = cfg_opt.trust_region {
            ms.noise_aware = true;
        }
    }
    Ok(cfg_opt)
}

fn scalar_score(scores: &[f64], row: usize, n_obj: usize) -> f64 {
    let base = row * n_obj;
    scores[base..base + n_obj].iter().sum()
}

fn run_host(
    cfg: &Config,
    mode: ExecMode,
    domain: &DomainFrame,
    out_dir: &Path,
    rep: u32,
    worker: &mut JuliaWorker,
) -> Result<SeedRun, String> {
    let seed = baux::tune_seed(&cfg.experiment, "proposal", rep, 0);
    let mut rng = prng(seed);
    let n_obj = worker.num_obj;
    let cfg_opt = opt_cfg(cfg, n_obj)?;
    let mut opt = if mode == ExecMode::HostDisk {
        let work = out_dir.join(format!("bpann_rep{rep}"));
        BayesOptimizer::with_disk(domain.clone(), cfg_opt, &cfg.optimizer, 0, work, &mut rng)
            .map_err(|e| e.to_string())?
    } else {
        BayesOptimizer::with_domain(domain.clone(), cfg_opt, &cfg.optimizer, 0, &mut rng)
            .map_err(|e| e.to_string())?
    };

    let num_dim = opt.num_dim();
    let arms = cfg
        .optimizer
        .candidate_generation()
        .map(|c| c.arms as usize)
        .unwrap_or(1)
        .max(1);

    let mut best_score = f64::NEG_INFINITY;
    let mut best_viol = f64::INFINITY;
    let mut best_unit = vec![0.5f64; num_dim];
    let mut init_err = 0.0;
    let mut accepts = 0usize;
    let mut worker_us = 0u64;
    let mut best_scores = Vec::with_capacity(cfg.rounds);
    let mut best_viols = Vec::with_capacity(cfg.rounds);
    let t0 = Instant::now();

    for round in 1..=cfg.rounds {
        let unit = opt.ask(arms, &mut rng).map_err(|e| e.to_string())?;
        let (scores, viols, dt_us) = worker.eval_unit(&unit.view())?;
        worker_us += dt_us;

        for (i, &vl) in viols.iter().enumerate() {
            let sc = scalar_score(&scores, i, n_obj);
            if sc > best_score {
                best_score = sc;
                best_viol = vl;
                accepts += 1;
                if let Some(sl) = unit.as_slice() {
                    best_unit.copy_from_slice(&sl[i * num_dim..(i + 1) * num_dim]);
                } else {
                    for j in 0..num_dim {
                        best_unit[j] = unit[[i, j]];
                    }
                }
            }
        }
        if round == 1 {
            init_err = loss_metric(cfg, best_score);
        }
        let y = Array2::from_shape_fn((arms, n_obj), |(i, m)| scores[i * n_obj + m]);
        opt.tell(&unit.view(), &y.view(), &mut rng)
            .map_err(|e| e.to_string())?;
        best_scores.push(best_score);
        best_viols.push(best_viol);
    }

    let (_, _, state_snap) = worker.eval_snap(&best_unit)?;
    let elapsed_s = t0.elapsed().as_secs_f64();
    let worker_s = (worker_us as f64) * 1.0e-6;
    let final_err = loss_metric(cfg, best_score);
    let reduction_pct = if init_err > 0.0 {
        ((init_err - final_err) / init_err * 100.0).max(0.0)
    } else {
        0.0
    };
    Ok(SeedRun {
        rep,
        seed,
        num_dim,
        init_err,
        final_score: best_score,
        final_err,
        final_viol: best_viol,
        reduction_pct,
        accepts,
        elapsed_s,
        worker_s,
        best_scores,
        best_viols,
        state_snap,
    })
}

fn run_ppo(
    cfg: &Config,
    num_dim: usize,
    rep: u32,
    worker: &mut JuliaWorker,
) -> Result<SeedRun, String> {
    let seed = baux::tune_seed(&cfg.experiment, "ppo", rep, 0);
    let mut rng = prng(seed);
    let n_obj = worker.num_obj;
    let batch_cap = (cfg.rounds / 2).clamp(2, 16);
    let mut ppo = baux::tokamak::ppo::FlatPpo::new(num_dim, batch_cap)?;
    if !ppo.is_metal() {
        return Err("FlatPpo requires Apple Metal GPU device".into());
    }

    let mut row = Array2::<f64>::zeros((1, num_dim));
    let mut best_score = f64::NEG_INFINITY;
    let mut best_viol = f64::INFINITY;
    let mut best_unit = vec![0.5f64; num_dim];
    let mut init_err = 0.0;
    let mut accepts = 0usize;
    let mut worker_us = 0u64;
    let mut best_scores = Vec::with_capacity(cfg.rounds);
    let mut best_viols = Vec::with_capacity(cfg.rounds);
    let t0 = Instant::now();

    for round in 1..=cfg.rounds {
        let slice = row
            .as_slice_mut()
            .ok_or("ppo row buffer must be contiguous")?;
        let raw_u = ppo.sample_unit(false, &mut rng, slice)?;
        let (scores, viols, dt_us) = worker.eval_unit(&row.view())?;
        worker_us += dt_us;
        let sc = scalar_score(&scores, 0, n_obj);
        let vl = viols[0];
        if sc > best_score {
            best_score = sc;
            best_viol = vl;
            accepts += 1;
            best_unit.copy_from_slice(row.as_slice().unwrap());
        }
        if round == 1 {
            init_err = loss_metric(cfg, best_score);
        }
        let _ = ppo.observe(&raw_u, sc)?;
        best_scores.push(best_score);
        best_viols.push(best_viol);
    }

    let (_, _, state_snap) = worker.eval_snap(&best_unit)?;
    let elapsed_s = t0.elapsed().as_secs_f64();
    let worker_s = (worker_us as f64) * 1.0e-6;
    let final_err = loss_metric(cfg, best_score);
    let reduction_pct = if init_err > 0.0 {
        ((init_err - final_err) / init_err * 100.0).max(0.0)
    } else {
        0.0
    };
    Ok(SeedRun {
        rep,
        seed,
        num_dim,
        init_err,
        final_score: best_score,
        final_err,
        final_viol: best_viol,
        reduction_pct,
        accepts,
        elapsed_s,
        worker_s,
        best_scores,
        best_viols,
        state_snap,
    })
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
fn res_enn(
    cfg: &Config,
    cfg_opt: &baux::optimizer_config::OptimizerConfig,
    seed: u64,
) -> Result<baux::config::ResidentEnnConfig, String> {
    let (arms, cands) = cfg
        .optimizer
        .candidate_generation()
        .map(|cg| (cg.arms, cg.candidates))
        .unwrap_or((1, 4));
    let pool = baux::procedural_pool::ProceduralPoolConfig {
        arms,
        candidates_per_arm: cands,
    }
    .resolve_resident()?;

    let (kind, beta) = match cfg_opt.acquisition {
        baux::optimizer_config::AcquisitionConfig::Random => {
            (baux::weights::AcquisitionKind::Random, 0.0)
        }
        baux::optimizer_config::AcquisitionConfig::UCB { beta } => {
            (baux::weights::AcquisitionKind::Ucb, beta as f32)
        }
        baux::optimizer_config::AcquisitionConfig::Thompson
        | baux::optimizer_config::AcquisitionConfig::Pareto => {
            (baux::weights::AcquisitionKind::Thompson, 0.0)
        }
    };

    let mut ask = baux::trials::RoundConfig::default();
    let tr_init = match &cfg_opt.trust_region {
        baux::trust_region::TrustRegionConfig::Turbo(l) => l.length_init,
        baux::trust_region::TrustRegionConfig::Morbo(m) => m.length.length_init,
    };
    ask.proposal.radius = tr_init as f32;
    ask.acquisition = baux::trials::AcquisitionParams { kind, beta, seed };

    let baux::optimizer_config::SurrogateConfig::ENN(ref enn) = cfg_opt.surrogate else {
        return Err("resident Metal backend requires an ENN surrogate".into());
    };
    ask.surrogate.neighbors = usize::try_from(enn.k.max(1)).unwrap_or(10);
    ask.surrogate.epistemic_scale = enn.epistemic_scale as f32;
    ask.surrogate.aleatoric_scale = enn.aleatoric_scale as f32;
    ask.surrogate.output_scale = enn.y_scale.unwrap_or(1.0) as f32;

    let (distance_metric, local_scale_neighbors) = match enn.geometry {
        baux::semantics::neighbor::DistanceMetric::Euclidean => {
            (baux::config::DistanceMetric::Euclidean, 10)
        }
        baux::semantics::neighbor::DistanceMetric::Conformal { local_k } => {
            (baux::config::DistanceMetric::Conformal, local_k)
        }
        baux::semantics::neighbor::DistanceMetric::Hyperbolic { local_k } => {
            (baux::config::DistanceMetric::Hyperbolic, local_k)
        }
        baux::semantics::neighbor::DistanceMetric::Wasserstein { local_k } => {
            (baux::config::DistanceMetric::Wasserstein, local_k)
        }
    };

    let fit_params = enn.num_candidates > 0 && enn.num_samples > 0;
    Ok(baux::config::ResidentEnnConfig {
        pool,
        ask,
        num_candidates: enn.num_candidates.max(1),
        num_samples: enn.num_samples.max(1),
        fit_params,
        adapt_neighbors: enn.adapt_neighbors,
        distance_metric,
        local_scale_neighbors,
    })
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
fn encode_half(val: f32) -> u16 {
    let bits = val.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xff) as i32 - 127 + 15;
    let mant = (bits >> 13) & 0x03ff;
    if exp <= 0 {
        sign
    } else if exp >= 31 {
        sign | 0x7c00
    } else {
        sign | ((exp as u16) << 10) | (mant as u16)
    }
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
fn obj_obs(
    scores: &[f64],
    n_obj: usize,
) -> Result<baux::objective_observation::ObjectiveObservation, String> {
    let items: Vec<baux::objective_observation::ObjectiveEstimate> = scores[..n_obj]
        .iter()
        .map(|&v| baux::objective_observation::ObjectiveEstimate {
            mean: v as f32,
            variance: 0.0,
        })
        .collect();
    let mean = items.iter().map(|e| e.mean).sum::<f32>() / (items.len() as f32);
    let ctrl = baux::objective_observation::ObjectiveEstimate {
        mean,
        variance: 0.0,
    };
    baux::objective_observation::ObjectiveObservation::new(&items, ctrl)
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
fn run_metal(
    cfg: &Config,
    num_dim: usize,
    rep: u32,
    worker: &mut JuliaWorker,
) -> Result<SeedRun, String> {
    use baux::config::{
        ObjectiveAcquisition, ObjectivePolicy, ObservationSharing, RegionAllocation,
        RegionController, RegionMemory, RegionalSearchSpec, RestartCenter,
    };
    use baux::proposal_kernel::ProposalKernel;
    use baux::resident_metal::{ParamBlock, RegionalSearch, SearchState};

    let seed = baux::tune_seed(&cfg.experiment, "resident-metal", rep, 0);
    let n_obj = worker.num_obj;
    let cfg_opt = opt_cfg(cfg, n_obj)?;
    let enn_res = res_enn(cfg, &cfg_opt, seed)?;

    let tr_cfg = match cfg_opt.trust_region {
        baux::trust_region::TrustRegionConfig::Turbo(l) => l,
        baux::trust_region::TrustRegionConfig::Morbo(m) => m.length,
    };
    let kernel = match &cfg.optimizer {
        OptimizerSpec::OperatorBo { distribution, .. }
        | OptimizerSpec::BayesianSearch {
            distribution: Some(distribution),
            ..
        } => *distribution,
        _ => ProposalKernel::Gaussian,
    };

    let reg_spec = cfg
        .optimizer
        .regional_search()
        .cloned()
        .unwrap_or_else(|| RegionalSearchSpec {
            count: 1,
            sharing: ObservationSharing::OriginOnly,
            allocation: RegionAllocation::LeastTrials,
            controller: RegionController::Turbo,
            acquisitions: Vec::new(),
            restart_memory: RegionMemory::Fresh,
            restart_candidates: 1,
            restart_center: RestartCenter::First,
            success_tolerance: 3,
            failure_tolerance: 3,
        });
    let num_regions = reg_spec.count.max(1);

    let obj_policy = if n_obj >= 2 {
        let scales = match cfg.morbo {
            Some(ref m) if m.scales.len() == n_obj => m.scales.clone(),
            _ => vec![1.0f32; n_obj],
        };
        let acq = if let Some(ref m) = cfg.morbo {
            let weights = if m.preferences.len() == n_obj {
                m.preferences.clone()
            } else {
                vec![1.0f32; n_obj]
            };
            ObjectiveAcquisition::Morbo {
                weights,
                regions: m.regions,
                alpha: m.alpha as f32,
                seed,
                rescalarize: m.rescalarize,
                clip: true,
            }
        } else {
            ObjectiveAcquisition::Pareto
        };
        Some(ObjectivePolicy {
            acquisition: acq,
            scales,
        })
    } else {
        None
    };

    let mut states = Vec::with_capacity(num_regions);
    let cap = (cfg.rounds + 16).min(512).max(16);
    for slot in 0..num_regions {
        let base_val = 0.5f32 + ((slot as f32) * 0.03f32).min(0.25f32);
        let base = vec![encode_half(base_val); num_dim];
        let r_seed = seed.wrapping_add((slot as u64 + 1).wrapping_mul(0x9e37_79b9_7f4a_7c15));
        let scale = 1.0f32 / ((num_dim.max(1) as f32).sqrt());
        let block = ParamBlock::new(r_seed, 0, num_dim, 1.0, scale)?;
        let mut st = SearchState::new_implicit(&base, vec![block], cap, tr_cfg.clone(), kernel)?;
        st.configure_enn(enn_res)?;
        let init_buf = st.base_buffer();
        let (init_scores, _, dt_us) = worker.eval_half(init_buf.contents() as *const u16)?;
        let _ = dt_us;
        if n_obj >= 2 {
            st.observe_objectives(obj_obs(&init_scores, n_obj)?)?;
        } else {
            st.observe_initial(init_scores[0] as f32, 0.0)?;
        }
        if let Some(ref policy) = obj_policy {
            st.configure_objectives(Some(policy.clone()))?;
        }
        states.push(st);
    }

    let mut search = RegionalSearch::from_spec(states, &reg_spec)?;
    let mut best_score = f64::NEG_INFINITY;
    let mut best_viol = f64::INFINITY;
    let mut best_slot = 0usize;
    let mut init_err = 0.0;
    let mut accepts = 0usize;
    let mut worker_us = 0u64;
    let mut best_scores = Vec::with_capacity(cfg.rounds);
    let mut best_viols = Vec::with_capacity(cfg.rounds);
    let t0 = Instant::now();

    for round in 1..=cfg.rounds {
        let r_seed = seed.wrapping_add(round as u64);
        let mut r_ask = enn_res.ask;
        r_ask.acquisition.seed = r_seed;
        let prop = search.ask(r_seed, r_ask)?;
        let slot = prop.ticket().region.slot;
        let buf = search.buffer(&prop)?;
        let (scores, vl, dt_us) = worker.eval_half(buf.contents() as *const u16)?;
        if n_obj >= 2 {
            let _ = search.tell_objectives(&prop, obj_obs(&scores, n_obj)?)?;
        } else {
            let _ = search.tell(&prop, scores[0] as f32, 0.0)?;
        }
        worker_us += dt_us;
        let sc = scalar_score(&scores, 0, n_obj);
        if sc > best_score {
            best_score = sc;
            best_viol = vl;
            best_slot = slot;
            accepts += 1;
        }
        if round == 1 {
            init_err = loss_metric(cfg, best_score);
        }
        best_scores.push(best_score);
        best_viols.push(best_viol);
    }

    let inc = search.incumbent(best_slot)?;
    let state_snap = worker.half_snap(inc.contents() as *const u16)?;
    let elapsed_s = t0.elapsed().as_secs_f64();
    let worker_s = (worker_us as f64) * 1.0e-6;
    let final_err = loss_metric(cfg, best_score);
    let reduction_pct = if init_err > 0.0 {
        ((init_err - final_err) / init_err * 100.0).max(0.0)
    } else {
        0.0
    };
    Ok(SeedRun {
        rep,
        seed,
        num_dim,
        init_err,
        final_score: best_score,
        final_err,
        final_viol: best_viol,
        reduction_pct,
        accepts,
        elapsed_s,
        worker_s,
        best_scores,
        best_viols,
        state_snap,
    })
}

#[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
fn run_metal(
    _cfg: &Config,
    _num_dim: usize,
    _rep: u32,
    _worker: &mut JuliaWorker,
) -> Result<SeedRun, String> {
    Err("resident Metal backend requires macOS + metal feature".into())
}

fn run_opt(cfg: &Config, mode: ExecMode, root: &Path, out_dir: &Path) -> Result<Value, String> {
    let lock_name = if is_ocean(cfg) {
        "oceananigans"
    } else {
        "fuse"
    };
    let lock_val = ensure_upstream(root, lock_name)?;
    let merkle_root = lock_val["merkle"]["merkle_root_blake3"]
        .as_str()
        .unwrap_or("")
        .to_owned();
    let upstream_rev = lock_val["repository"]["revision"]
        .as_str()
        .unwrap_or("")
        .to_owned();

    let (domain, low, span, state_len) = resolve_space(root, cfg)?;
    let num_dim = domain.num_dim();
    let arms = if mode == ExecMode::HostRam || mode == ExecMode::HostDisk {
        cfg.optimizer
            .candidate_generation()
            .map(|c| c.arms as usize)
            .unwrap_or(1)
            .max(1)
    } else {
        1
    };
    let reps = cfg.repetitions.max(1);
    let regions = cfg
        .optimizer
        .regional_search()
        .map(|r| r.count)
        .unwrap_or(1);

    let mut worker = JuliaWorker::spawn(root, cfg, arms, low, span, state_len)?;
    let label = opt_label(cfg, mode);
    eprintln!(
        "[tune] starting {} ({}) zero-copy mmap rollout | dim={} obj={} rounds={} arms={} regions={} reps={} merkle={}",
        model_label(cfg),
        label,
        num_dim,
        worker.num_obj,
        cfg.rounds,
        arms,
        regions,
        reps,
        &merkle_root[..12.min(merkle_root.len())],
    );

    let runs: Vec<SeedRun> = (0..reps)
        .map(|rep| {
            let sr = match mode {
                ExecMode::HostRam | ExecMode::HostDisk => {
                    run_host(cfg, mode, &domain, out_dir, rep, &mut worker)
                }
                ExecMode::ResidentMetal => run_metal(cfg, num_dim, rep, &mut worker),
                ExecMode::MetalPpo => run_ppo(cfg, num_dim, rep, &mut worker),
            }?;
            eprintln!(
                "{}.seed mode={} rep={}/{} seed={} dim={} best_score={:.6} viol={:.6} accepts={}/{} worker_s={:.2} total_s={:.2}",
                cfg.experiment,
                label,
                sr.rep + 1,
                reps,
                sr.seed,
                sr.num_dim,
                sr.final_score,
                sr.final_viol,
                sr.accepts,
                cfg.rounds,
                sr.worker_s,
                sr.elapsed_s,
            );
            Ok::<_, String>(sr)
        })
        .collect::<Result<_, _>>()?;

    let traces: Vec<Value> = (0..cfg.rounds)
        .map(|idx| {
            let scores: Vec<f64> = runs.iter().map(|r| r.best_scores[idx]).collect();
            let errs: Vec<f64> = scores.iter().map(|&s| loss_metric(cfg, s)).collect();
            let viols: Vec<f64> = runs.iter().map(|r| r.best_viols[idx]).collect();
            let (ms, ss) = mean_sem(&scores);
            let (me, se) = mean_sem(&errs);
            let (mv, sv) = mean_sem(&viols);
            json!({
                "round": idx + 1,
                "best_score": ms,
                "mean_best_score": ms,
                "sem_best_score": ss,
                "tracking_error": me,
                "mean_tracking_error": me,
                "sem_tracking_error": se,
                "mean_violation": mv,
                "sem_violation": sv,
            })
        })
        .collect();

    let init_errs: Vec<f64> = runs.iter().map(|r| r.init_err).collect();
    let final_scores: Vec<f64> = runs.iter().map(|r| r.final_score).collect();
    let final_errs: Vec<f64> = runs.iter().map(|r| r.final_err).collect();
    let final_viols: Vec<f64> = runs.iter().map(|r| r.final_viol).collect();
    let reds: Vec<f64> = runs.iter().map(|r| r.reduction_pct).collect();
    let (init_m, init_s) = mean_sem(&init_errs);
    let (sc_m, sc_s) = mean_sem(&final_scores);
    let (err_m, err_s) = mean_sem(&final_errs);
    let (vl_m, vl_s) = mean_sem(&final_viols);
    let (red_m, red_s) = mean_sem(&reds);
    let total_s: f64 = runs.iter().map(|r| r.elapsed_s).sum();
    let worker_s: f64 = runs.iter().map(|r| r.worker_s).sum();

    let seed_runs: Vec<Value> = runs
        .iter()
        .map(|r| {
            let snap_head: Vec<f32> = r.state_snap.iter().copied().take(64).collect();
            json!({
                "rep": r.rep,
                "seed": r.seed,
                "initial_tracking_error": r.init_err,
                "final_score": r.final_score,
                "final_tracking_error": r.final_err,
                "final_violation": r.final_viol,
                "tracking_error_reduction_pct": r.reduction_pct,
                "total_accepts": r.accepts,
                "elapsed_seconds": r.elapsed_s,
                "worker_seconds": r.worker_s,
                "best_scores": r.best_scores,
                "best_violations": r.best_viols,
                "state_preview": snap_head,
            })
        })
        .collect();

    Ok(json!({
        "experiment": cfg.experiment,
        "model": model_label(cfg),
        "backend_mode": label,
        "upstream_revision": upstream_rev,
        "upstream_merkle_blake3": merkle_root,
        "parameters": num_dim,
        "objectives": worker.num_obj,
        "regions": regions,
        "rounds": cfg.rounds,
        "repetitions": reps,
        "disk_backed": mode == ExecMode::HostDisk,
        "initial_tracking_error": init_m,
        "initial_tracking_error_sem": init_s,
        "final_score_mean": sc_m,
        "final_score_sem": sc_s,
        "final_tracking_error": err_m,
        "final_tracking_error_sem": err_s,
        "final_violation_mean": vl_m,
        "final_violation_sem": vl_s,
        "tracking_error_reduction_pct": red_m,
        "tracking_error_reduction_pct_sem": red_s,
        "elapsed_seconds": total_s,
        "worker_seconds": worker_s,
        "traces": traces,
        "seed_runs": seed_runs,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

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

        // 1. Live ResidentMetal OperatorBo (Hyperbolic + RiskPrefix +
        //    Rademacher + Morbo)
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
}
