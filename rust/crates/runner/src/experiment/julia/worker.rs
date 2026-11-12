//! IPC and daemon worker client for Julia physics evaluators over ShmSlab.

use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::Instant;

use baux::artifact::blake3::hash_hex;
use baux::unified_mem::ShmSlab;
use ndarray::Array2;

use crate::experiment::julia::config::{Config, fuse_key, is_ocean, obj_count, ocean_key};

fn daemon_slot() -> Option<String> {
    std::env::var("BAUX_DAEMON_SLOT")
        .ok()
        .filter(|s| !s.is_empty())
}

fn isolated_daemon() -> bool {
    std::env::var("BAUX_ISOLATE_DAEMON").ok().as_deref() == Some("1")
}

pub fn daemon_sock(root: &Path, ocean: bool, slot: Option<usize>) -> PathBuf {
    let tag = &hash_hex(root.as_os_str().as_encoded_bytes())[..10];
    let kind = if ocean { "ocean" } else { "fuse" };
    if let Some(s) = slot {
        return std::env::temp_dir().join(format!("baux_jl_{tag}_{kind}_slot{s}.sock"));
    }
    if let Some(slot) = daemon_slot() {
        return std::env::temp_dir().join(format!("baux_jl_{tag}_{kind}_slot{slot}.sock"));
    }
    if isolated_daemon() {
        let pid = std::process::id();
        return std::env::temp_dir().join(format!("baux_jl_{tag}_{kind}_{pid}.sock"));
    }
    std::env::temp_dir().join(format!("baux_jl_{tag}_{kind}.sock"))
}

pub fn daemon_request(sock_path: &Path, msg: &str) -> Result<String, String> {
    use std::io::BufRead;
    use std::os::unix::net::UnixStream;
    let mut stream = UnixStream::connect(sock_path)
        .map_err(|e| format!("connect {}: {e}", sock_path.display()))?;
    let secs = if msg.starts_with("PING") { 3 } else { 300 };
    let _ = stream.set_read_timeout(Some(std::time::Duration::from_secs(secs)));
    stream
        .write_all(msg.as_bytes())
        .map_err(|e| e.to_string())?;
    stream.flush().map_err(|e| e.to_string())?;
    let mut reader = std::io::BufReader::new(stream);
    let mut line = String::new();
    reader.read_line(&mut line).map_err(|e| e.to_string())?;
    Ok(line)
}

fn spawn_daemon(
    root: &Path,
    proj: &Path,
    sock_path: &Path,
    log_file: fs::File,
    ocean: bool,
    single_thread: bool,
) -> Result<std::process::Child, String> {
    let mut cmd = Command::new(baux::util::julia_bin());
    let threads = if single_thread || isolated_daemon() || daemon_slot().is_some() {
        "--threads=1"
    } else {
        "--threads=auto"
    };
    cmd.args(["--startup-file=no", threads])
        .env("OPENBLAS_NUM_THREADS", "1");
    if !ocean {
        cmd.arg("-O1");
    }
    cmd.arg(format!("--project={}", proj.display()))
        .arg(proj.join("worker.jl"))
        .arg("--daemon")
        .arg(sock_path)
        .current_dir(root)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::from(log_file));
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        cmd.process_group(0);
    }
    cmd.spawn()
        .map_err(|e| format!("spawn julia daemon {}: {e}", proj.display()))
}

fn kill_stale(pid_path: &Path) {
    if let Ok(raw) = fs::read_to_string(pid_path)
        && let Ok(pid) = raw.trim().parse::<u32>()
    {
        let _ = Command::new("kill")
            .args(["-9", &pid.to_string()])
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status();
    }
    let _ = fs::remove_file(pid_path);
}

pub fn ensure_slot(root: &Path, ocean: bool, slot: Option<usize>) -> Result<PathBuf, String> {
    let sock_path = daemon_sock(root, ocean, slot);
    if let Ok(pong) = daemon_request(&sock_path, "PING\n")
        && pong.trim() == "PONG"
    {
        return Ok(sock_path);
    }
    let pid_path = sock_path.with_extension("pid");
    kill_stale(&pid_path);
    let _ = fs::remove_file(&sock_path);
    let proj = if ocean {
        root.join("evals/oceananigans")
    } else {
        root.join("evals/fuse")
    };
    let log_dir = root.join(".cache/baux/shm");
    fs::create_dir_all(&log_dir).map_err(|e| e.to_string())?;
    let slot_tag = slot.map_or_else(String::new, |s| format!("_slot{s}"));
    let log_name = if ocean {
        format!("julia_ocean_daemon{slot_tag}.log")
    } else {
        format!("julia_fuse_daemon{slot_tag}.log")
    };
    let log_path = log_dir.join(log_name);
    let log_file = fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&log_path)
        .map_err(|e| format!("open daemon log {}: {e}", log_path.display()))?;

    let single = slot.is_some();
    let mut child = spawn_daemon(root, &proj, &sock_path, log_file, ocean, single)?;
    let _ = fs::write(&pid_path, format!("{}\n", child.id()));
    let deadline = Instant::now() + std::time::Duration::from_secs(300);
    while Instant::now() < deadline {
        if let Ok(Some(status)) = child.try_wait() {
            let log_tail = fs::read_to_string(&log_path).unwrap_or_default();
            return Err(format!(
                "julia daemon exited prematurely with {status}: {log_tail}"
            ));
        }
        if let Ok(pong) = daemon_request(&sock_path, "PING\n")
            && pong.trim() == "PONG"
        {
            return Ok(sock_path);
        }
        std::thread::sleep(std::time::Duration::from_millis(80));
    }
    let _ = child.kill();
    let _ = child.wait();
    let _ = fs::remove_file(&pid_path);
    Err(format!(
        "timed out waiting for julia daemon socket {}",
        sock_path.display()
    ))
}

pub fn ensure_daemon(root: &Path, ocean: bool) -> Result<PathBuf, String> {
    ensure_slot(root, ocean, None)
}

pub struct JuliaWorker {
    pub _sock: std::os::unix::net::UnixStream,
    pub slab: ShmSlab,
    pub low: Vec<f64>,
    pub span: Vec<f64>,
    pub num_obj: usize,
}

impl JuliaWorker {
    pub fn spawn_slot(
        root: &Path,
        cfg: &Config,
        max_batch: usize,
        low: Vec<f64>,
        span: Vec<f64>,
        state_len: usize,
        slot: Option<usize>,
    ) -> Result<Self, String> {
        use std::io::BufRead;
        use std::os::unix::net::UnixStream;

        let num_dim = low.len();
        let num_obj = obj_count(cfg);
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_err(|e| e.to_string())?
            .as_nanos();
        let slot_id = slot.map_or(0, |s| s + 1);
        let shm_path = root.join(".cache/baux/shm").join(format!(
            "{}_{}_{}_{}.mmap",
            cfg.experiment,
            std::process::id(),
            slot_id,
            nonce
        ));
        let slab = ShmSlab::create_multi(shm_path.clone(), max_batch, num_dim, num_obj, state_len)?;

        let ocean = is_ocean(cfg);
        let sock_path = ensure_slot(root, ocean, slot)?;
        let cfg_key = if ocean { ocean_key(cfg) } else { fuse_key(cfg) };
        let mut sock = UnixStream::connect(&sock_path)
            .map_err(|e| format!("connect julia daemon {}: {e}", sock_path.display()))?;
        let _ = sock.set_read_timeout(Some(std::time::Duration::from_secs(240)));
        let open_cmd = format!("OPEN\t{}\t{cfg_key}\n", shm_path.display());
        sock.write_all(open_cmd.as_bytes())
            .map_err(|e| e.to_string())?;
        sock.flush().map_err(|e| e.to_string())?;
        let mut line = String::new();
        {
            let mut reader = std::io::BufReader::new(&sock);
            reader
                .read_line(&mut line)
                .map_err(|e| format!("handshake failed: {e}"))?;
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

    pub fn spawn(
        root: &Path,
        cfg: &Config,
        max_batch: usize,
        low: Vec<f64>,
        span: Vec<f64>,
        state_len: usize,
    ) -> Result<Self, String> {
        Self::spawn_slot(root, cfg, max_batch, low, span, state_len, None)
    }

    pub fn connect_slot(root: &Path, cfg: &Config, slot: Option<usize>) -> Result<Self, String> {
        let (_domain, low, span, state_len) =
            crate::experiment::julia::space::resolve_space(root, cfg)?;
        let max_batch = cfg
            .optimizer
            .candidate_generation()
            .map_or(1, |cg| cg.arms as usize)
            .max(1);
        Self::spawn_slot(root, cfg, max_batch, low, span, state_len, slot)
    }

    pub fn connect(root: &Path, cfg: &Config) -> Result<Self, String> {
        Self::connect_slot(root, cfg, None)
    }

    pub fn call_doorbell(&mut self, cmd_id: u32, count: usize) -> Result<u64, String> {
        let shm_cmd = match cmd_id {
            1 => baux::unified_mem::CMD_EVAL,
            2 => baux::unified_mem::CMD_SNAP,
            other => return Err(format!("unsupported julia doorbell cmd {other}")),
        };
        let timeout = std::env::var("BAUX_EVAL_TIMEOUT_MS")
            .ok()
            .and_then(|s| s.parse::<u64>().ok())
            .filter(|&v| v > 0)
            .map_or(
                std::time::Duration::from_secs(120),
                std::time::Duration::from_millis,
            );
        let (dt_us, _threads, _rt_ns) = self
            .slab
            .doorbell_wait(shm_cmd, count, timeout, || Ok(()))?;
        Ok(dt_us)
    }

    pub fn eval_unit(
        &mut self,
        unit: &ndarray::ArrayView2<f64>,
    ) -> Result<(Vec<f64>, Vec<f64>, u64), String> {
        let count = self.slab.write_unit(unit, &self.low, &self.span)?;
        let dt_us = self.call_doorbell(1, count)?;
        let scores = self
            .slab
            .read_scores(count)
            .iter()
            .map(|&s| {
                if s.is_finite() && s > -1.0e8 {
                    s
                } else {
                    -500.0
                }
            })
            .collect();
        let viols = self.slab.read_viols(count).to_vec();
        Ok((scores, viols, dt_us))
    }

    pub fn eval_snap(&mut self, unit_row: &[f64]) -> Result<(f64, f64, Vec<f32>), String> {
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
    pub fn eval_half(&mut self, half_ptr: *const u16) -> Result<(Vec<f64>, f64, u64), String> {
        unsafe { self.slab.write_half(0, half_ptr, &self.low, &self.span)? };
        let dt_us = self.call_doorbell(1, 1)?;
        let scores = self
            .slab
            .read_scores(1)
            .iter()
            .map(|&s| {
                if s.is_finite() && s > -1.0e8 {
                    s
                } else {
                    -500.0
                }
            })
            .collect();
        let viol = self.slab.read_viols(1)[0];
        Ok((scores, viol, dt_us))
    }

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    pub fn half_snap(&mut self, half_ptr: *const u16) -> Result<Vec<f32>, String> {
        unsafe { self.slab.write_half(0, half_ptr, &self.low, &self.span)? };
        let _ = self.call_doorbell(2, 1)?;
        Ok(self.slab.read_state().to_vec())
    }
}

impl Drop for JuliaWorker {
    fn drop(&mut self) {
        let cmd = if isolated_daemon() && daemon_slot().is_none() {
            baux::unified_mem::CMD_SHUTDOWN
        } else {
            baux::unified_mem::CMD_DETACH
        };
        let _ = self
            .slab
            .doorbell_wait(cmd, 0, std::time::Duration::from_secs(2), || Ok(()));
    }
}
