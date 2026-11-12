//! Self-healing, isolated shared-memory daemon worker for TORAX and TokaGym.

use std::fs;
use std::io::{BufRead, Write};
use std::os::unix::net::UnixStream;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use baux::artifact::blake3::hash_hex;
use baux::unified_mem::ShmSlab;

#[path = "worker_batch.rs"]
pub mod batch;

static WORKER_SEQ: AtomicU64 = AtomicU64::new(1);

#[derive(Clone)]
pub struct WorkerSpec {
    pub kind: &'static str,
    pub lock: &'static str,
    pub arms: usize,
    pub num_dim: usize,
    pub obs: usize,
    pub act: usize,
    pub steps: usize,
}

fn eval_timeout(kind: &str) -> Duration {
    if let Ok(ms) = std::env::var("BAUX_EVAL_TIMEOUT_MS")
        && let Ok(v) = ms.parse::<u64>()
        && v > 0
    {
        return Duration::from_millis(v);
    }
    if kind == "tokagym" {
        Duration::from_secs(30)
    } else {
        Duration::from_secs(60)
    }
}

fn max_evals() -> usize {
    std::env::var("BAUX_DAEMON_MAX_EVALS")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .filter(|&v| v > 0)
        .unwrap_or(500)
}

fn penalty_for(kind: &str) -> f64 {
    if kind == "tokagym" { -500.0 } else { -400.0 }
}

fn active_slot() -> Option<String> {
    if std::env::var("BAUX_DAEMON_SCRIPT")
        .ok()
        .is_some_and(|s| !s.is_empty())
    {
        return None;
    }
    std::env::var("BAUX_DAEMON_SLOT")
        .ok()
        .filter(|s| !s.is_empty())
}

fn daemon_ping(sock_path: &Path) -> bool {
    let Ok(mut stream) = UnixStream::connect(sock_path) else {
        return false;
    };
    let _ = stream.set_read_timeout(Some(Duration::from_secs(2)));
    if stream.write_all(b"PING\n").is_err() || stream.flush().is_err() {
        return false;
    }
    let mut line = String::new();
    let mut reader = std::io::BufReader::new(stream);
    reader.read_line(&mut line).is_ok() && line.trim() == "PONG"
}

fn spawn_daemon(
    root: &Path,
    kind: &str,
    sock_path: &Path,
    log_file: fs::File,
) -> Result<Child, String> {
    let py_override = std::env::var("BAUX_DAEMON_SCRIPT").ok();
    let python_exe = root.join(".pixi/envs/baux-py314/bin/python");
    let mut cmd = if python_exe.exists() {
        Command::new(&python_exe)
    } else {
        let mut c = Command::new("uv");
        c.args(["run", "python3"]);
        c
    };
    if let Some(script) = py_override.filter(|s| !s.is_empty()) {
        cmd.arg(script)
            .args(["--kind", kind, "--sock"])
            .arg(sock_path);
    } else {
        cmd.args([
            "-m",
            "evals.python.tokamak_daemon",
            "--kind",
            kind,
            "--sock",
        ])
        .arg(sock_path);
    }
    let jax_cache = root.join(".cache/baux/jax_cache");
    let _ = fs::create_dir_all(&jax_cache);
    cmd.current_dir(root)
        .env("JAX_PLATFORMS", "cpu")
        .env("JAX_COMPILATION_CACHE_DIR", &jax_cache)
        .env("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
        .env("OMP_NUM_THREADS", "1")
        .env("OPENBLAS_NUM_THREADS", "1")
        .env("MKL_NUM_THREADS", "1")
        .env("VECLIB_MAXIMUM_THREADS", "1")
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::from(log_file));
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        cmd.process_group(0);
    }
    cmd.spawn().map_err(|e| format!("spawn {kind} daemon: {e}"))
}

fn kill_group(child: &mut Child) {
    let pid = child.id();
    let _ = Command::new("kill")
        .args(["-9", &format!("-{pid}")])
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status();
    let _ = child.kill();
    let _ = child.wait();
}

fn pkill_sock(sock_path: &Path) {
    if let Ok(mut stream) = UnixStream::connect(sock_path) {
        let _ = stream.set_write_timeout(Some(Duration::from_millis(200)));
        let _ = stream.write_all(b"SHUTDOWN\n");
        let _ = stream.flush();
    }
    let _ = Command::new("pkill")
        .args(["-9", "-f", &sock_path.to_string_lossy()])
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status();
    let _ = fs::remove_file(sock_path);
}

fn wait_ready(child: &mut Child, sock_path: &Path, kind: &str) -> Result<(), String> {
    let deadline = Instant::now() + Duration::from_secs(120);
    while !daemon_ping(sock_path) {
        if let Ok(Some(st)) = child.try_wait() {
            let _ = fs::remove_file(sock_path);
            return Err(format!("{kind} daemon exited early with {st}"));
        }
        if Instant::now() >= deadline {
            kill_group(child);
            let _ = fs::remove_file(sock_path);
            return Err(format!("timed out starting {kind} daemon"));
        }
        std::thread::sleep(Duration::from_millis(50));
    }
    Ok(())
}

fn ensure_daemon(
    root: &Path,
    kind: &str,
    sock_path: &Path,
    slot_mode: bool,
) -> Result<Option<Child>, String> {
    if slot_mode && daemon_ping(sock_path) {
        return Ok(None);
    }
    let log_dir = root.join(".cache/baux/shm");
    fs::create_dir_all(&log_dir).map_err(|e| e.to_string())?;
    let log_path = log_dir.join(format!("python_{kind}_{}.log", std::process::id()));
    let log_file = fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&log_path)
        .map_err(|e| format!("open daemon log: {e}"))?;
    let mut child = spawn_daemon(root, kind, sock_path, log_file)?;
    wait_ready(&mut child, sock_path, kind)?;
    if slot_mode { Ok(None) } else { Ok(Some(child)) }
}

fn handshake_open(sock_path: &Path, shm_path: &Path, kind: &str) -> Result<UnixStream, String> {
    let mut sock =
        UnixStream::connect(sock_path).map_err(|e| format!("connect {kind} daemon: {e}"))?;
    let _ = sock.set_read_timeout(Some(Duration::from_secs(240)));
    let open_cmd = format!("OPEN\t{}\t{kind}\n", shm_path.display());
    sock.write_all(open_cmd.as_bytes())
        .map_err(|e| e.to_string())?;
    sock.flush().map_err(|e| e.to_string())?;
    let mut line = String::new();
    std::io::BufReader::new(&sock)
        .read_line(&mut line)
        .map_err(|e| format!("handshake failed: {e}"))?;
    if !line.starts_with("OK\t") {
        return Err(format!("{kind} rejected OPEN: {}", line.trim()));
    }
    Ok(sock)
}

pub(crate) struct LiveSession {
    pub(crate) child: Option<Child>,
    pub(crate) _sock: UnixStream,
    pub(crate) slab: ShmSlab,
    pub(crate) sock_path: PathBuf,
    pub(crate) shm_path: PathBuf,
    pub(crate) slot_mode: bool,
}

impl LiveSession {
    pub(crate) fn open_slot(
        root: &Path,
        spec: &WorkerSpec,
        slot_override: Option<usize>,
    ) -> Result<Self, String> {
        let seq = WORKER_SEQ.fetch_add(1, Ordering::Relaxed);
        let pid = std::process::id();
        let tag = &hash_hex(root.as_os_str().as_encoded_bytes())[..10];
        let slot = slot_override.map(|s| s.to_string()).or_else(active_slot);
        let slot_mode = slot.is_some();
        let sock_path = match slot {
            Some(s) => {
                std::env::temp_dir().join(format!("baux_py_{tag}_{}_slot{s}.sock", spec.kind))
            }
            None => {
                std::env::temp_dir().join(format!("baux_py_{tag}_{}_{pid}_{seq}.sock", spec.kind))
            }
        };
        let mut child = ensure_daemon(root, spec.kind, &sock_path, slot_mode)?;
        let shm_dir = root.join(".cache/baux/shm");
        let shm_path = shm_dir.join(format!("tokamak_{}_{pid}_{seq}.mmap", spec.kind));
        let slab = match ShmSlab::create_multi(shm_path.clone(), spec.arms, spec.num_dim, 1, 4096) {
            Ok(s) => s,
            Err(e) => {
                if let Some(ref mut c) = child {
                    kill_group(c);
                }
                return Err(e);
            }
        };
        let sock = match handshake_open(&sock_path, &shm_path, spec.kind) {
            Ok(s) => s,
            Err(e) => {
                if let Some(ref mut c) = child {
                    kill_group(c);
                } else {
                    pkill_sock(&sock_path);
                }
                let _ = fs::remove_file(&shm_path);
                return Err(e);
            }
        };
        Ok(Self {
            child,
            _sock: sock,
            slab,
            sock_path,
            shm_path,
            slot_mode,
        })
    }

    fn close(&mut self, hard_kill: bool) {
        let cmd = if self.slot_mode && !hard_kill {
            baux::unified_mem::CMD_DETACH
        } else {
            baux::unified_mem::CMD_SHUTDOWN
        };
        let _ = self
            .slab
            .doorbell_wait(cmd, 0, Duration::from_millis(500), || Ok(()));
        if let Some(ref mut c) = self.child {
            kill_group(c);
            let _ = fs::remove_file(&self.sock_path);
        } else if hard_kill {
            pkill_sock(&self.sock_path);
        }
        let _ = fs::remove_file(&self.shm_path);
    }
}

pub struct TokamakWorker {
    pub(crate) root: PathBuf,
    pub(crate) spec: WorkerSpec,
    pub(crate) session: Option<LiveSession>,
    pub(crate) evals_done: usize,
    pub(crate) max_evals: usize,
    pub(crate) timeout: Duration,
    pub(crate) penalty: f64,
    pub(crate) slot: Option<usize>,
}

impl TokamakWorker {
    pub fn connect_slot(
        root: &Path,
        spec: WorkerSpec,
        slot: Option<usize>,
    ) -> Result<Self, String> {
        if !spec.lock.is_empty() {
            baux::artifact::merkle::verify_lock(root, spec.lock, None)
                .map_err(|e| format!("{} lock: {e}", spec.lock))?;
        }
        let timeout = eval_timeout(spec.kind);
        let penalty = penalty_for(spec.kind);
        let session = LiveSession::open_slot(root, &spec, slot)?;
        Ok(Self {
            root: root.to_path_buf(),
            spec,
            session: Some(session),
            evals_done: 0,
            max_evals: max_evals(),
            timeout,
            penalty,
            slot,
        })
    }

    pub fn spawn(root: &Path, spec: WorkerSpec) -> Result<Self, String> {
        Self::connect_slot(root, spec, None)
    }

    pub(crate) fn respawn(&mut self) -> Result<(), String> {
        if let Some(mut old) = self.session.take() {
            old.close(true);
        }
        self.session = Some(LiveSession::open_slot(&self.root, &self.spec, self.slot)?);
        self.evals_done = 0;
        Ok(())
    }

    fn eval_once(&mut self, batch_weights: &[f64], count: usize) -> Result<Vec<f64>, String> {
        let sess = self.session.as_mut().ok_or("no active daemon session")?;
        if let Some(ref mut c) = sess.child
            && let Ok(Some(st)) = c.try_wait()
        {
            return Err(format!("daemon exited with {st}"));
        }
        sess.slab.write_f64(batch_weights, count)?;
        let timeout = self.timeout;
        let _ = sess
            .slab
            .doorbell_wait(baux::unified_mem::CMD_EVAL, count, timeout, || Ok(()))?;
        let pen = self.penalty;
        let out = sess
            .slab
            .read_scores(count)
            .iter()
            .map(|&s| if s.is_finite() && s > -1.0e8 { s } else { pen })
            .collect();
        Ok(out)
    }

    pub fn eval(&mut self, batch_weights: &[f64], count: usize) -> Result<Vec<f64>, String> {
        if self.evals_done >= self.max_evals || self.session.is_none() {
            self.respawn()?;
        }
        match self.eval_once(batch_weights, count) {
            Ok(scores) => {
                self.evals_done += 1;
                Ok(scores)
            }
            Err(err) => {
                eprintln!("[tokamak.worker] eval error ({err}); respawning & retrying once");
                self.respawn()?;
                match self.eval_once(batch_weights, count) {
                    Ok(scores) => {
                        self.evals_done += 1;
                        Ok(scores)
                    }
                    Err(err2) => {
                        eprintln!("[tokamak.worker] retry failed ({err2}); assigning penalty");
                        self.respawn()?;
                        Ok(vec![self.penalty; count])
                    }
                }
            }
        }
    }

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    fn half_once(
        &mut self,
        half_ptr: *const u16,
        low: &[f64],
        span: &[f64],
    ) -> Result<(f64, u64), String> {
        let sess = self.session.as_mut().ok_or("no active daemon session")?;
        if let Some(ref mut c) = sess.child
            && let Ok(Some(st)) = c.try_wait()
        {
            return Err(format!("daemon exited with {st}"));
        }
        unsafe { sess.slab.write_half(0, half_ptr, low, span)? };
        let timeout = self.timeout;
        let (dt_us, _, _) =
            sess.slab
                .doorbell_wait(baux::unified_mem::CMD_EVAL, 1, timeout, || Ok(()))?;
        let raw = sess.slab.read_scores(1)[0];
        let sc = if raw.is_finite() && raw > -1.0e8 {
            raw
        } else {
            self.penalty
        };
        Ok((sc, dt_us))
    }

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    pub fn eval_half(
        &mut self,
        half_ptr: *const u16,
        low: &[f64],
        span: &[f64],
    ) -> Result<(f64, u64), String> {
        if self.evals_done >= self.max_evals || self.session.is_none() {
            self.respawn()?;
        }
        if let Ok(res) = self.half_once(half_ptr, low, span) {
            self.evals_done += 1;
            return Ok(res);
        }
        self.respawn()?;
        if let Ok(res) = self.half_once(half_ptr, low, span) {
            self.evals_done += 1;
            return Ok(res);
        }
        self.respawn()?;
        Ok((self.penalty, 0))
    }

    pub fn eval_halves(
        &mut self,
        rows: &[&[u16]],
        low: &[f64],
        span: &[f64],
    ) -> Result<(Vec<f64>, u64), String> {
        batch::eval_halves(self, rows, low, span)
    }

    pub fn eval_traj(
        &mut self,
        weights: &[f64],
        eps: &[f32],
        steps: usize,
    ) -> Result<(f64, Vec<f32>, Vec<f32>, Vec<f32>), String> {
        if self.evals_done >= self.max_evals || self.session.is_none() {
            self.respawn()?;
        }
        let sess = self.session.as_mut().ok_or("no active daemon session")?;
        sess.slab.write_f64(weights, 1)?;
        sess.slab.write_state(eps)?;
        let _ = sess
            .slab
            .doorbell_wait(baux::unified_mem::CMD_TRAJ, 1, self.timeout, || Ok(()))?;
        self.evals_done += 1;
        let raw = sess.slab.read_scores(1)[0];
        let score = if raw.is_finite() && raw > -1.0e8 {
            raw
        } else {
            self.penalty
        };
        let state = sess.slab.read_state();
        let obs_len = steps * self.spec.obs;
        let u_len = steps * self.spec.act;
        let obs = state[..obs_len].to_vec();
        let u = state[obs_len..obs_len + u_len].to_vec();
        let r = state[obs_len + u_len..obs_len + u_len + steps].to_vec();
        Ok((score, obs, u, r))
    }
}

impl Drop for TokamakWorker {
    fn drop(&mut self) {
        if let Some(mut sess) = self.session.take() {
            sess.close(false);
        }
    }
}
