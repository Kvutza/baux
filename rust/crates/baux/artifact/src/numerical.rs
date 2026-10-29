//! Host loading for the checked scalar arithmetic predicate.
//!
//! One dedicated thread owns initialization, Lean allocations and finalizers.
//! Callers send only seven scalar words. The library is retained for process
//! lifetime: unloading it while Lean globals exist would invalidate callbacks.

use baux::artifact::{cbor, runs::digest};
use baux::semantics::numerical::{self, Trace};
use deser::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock, mpsc};

struct Request {
    trace: Trace,
    reply: mpsc::SyncSender<Result<(), String>>,
}

struct Host {
    path: PathBuf,
    requests: mpsc::SyncSender<Request>,
}

static HOST: OnceLock<Result<Host, String>> = OnceLock::new();
static START: Mutex<()> = Mutex::new(());
static INSTALLED: OnceLock<Result<(), String>> = OnceLock::new();

const MAX_IMAGE: u64 = 512 * 1024 * 1024;
const MAX_RECORD: u64 = 16 * 1024;
static NEXT: AtomicU64 = AtomicU64::new(0);

/// Caller-recorded source/build association, not a proof of compiler
/// correctness.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[deser(deny_unknown_fields)]
pub struct Provenance {
    pub revision: String,
    pub compiler: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[deser(deny_unknown_fields)]
pub struct Record {
    pub schema: u32,
    pub abi: String,
    pub provenance: Provenance,
    pub sha256: String,
    pub bytes: u64,
}

/// Exact selected artifact. The receipt does not assert that its ABI is
/// trustworthy.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Receipt {
    pub id: String,
    pub library: PathBuf,
    pub record: Record,
}

fn identity(id: &str) -> bool {
    id.len() == 64
        && id
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

fn validate(record: &Record) -> Result<(), String> {
    if record.schema != 1
        || record.abi != "baux-numerical-7-v1"
        || !identity(&record.sha256)
        || record.bytes == 0
        || record.bytes > MAX_IMAGE
        || record.provenance.revision.is_empty()
        || record.provenance.revision.len() > 256
        || record.provenance.compiler.is_empty()
        || record.provenance.compiler.len() > 4096
    {
        return Err("invalid numerical artifact provenance or content identity".into());
    }
    Ok(())
}

fn directory(path: &Path) -> Result<(), String> {
    fs::create_dir_all(path).map_err(|e| format!("create {}: {e}", path.display()))?;
    if !fs::symlink_metadata(path)
        .map_err(|e| e.to_string())?
        .is_dir()
    {
        return Err(format!(
            "numerical artifact directory is not a real directory: {}",
            path.display()
        ));
    }
    Ok(())
}

fn verify(path: &Path, sha256: &str, bytes: u64) -> Result<(), String> {
    let meta =
        fs::symlink_metadata(path).map_err(|e| format!("inspect {}: {e}", path.display()))?;
    if !meta.is_file() || meta.len() != bytes || bytes == 0 || bytes > MAX_IMAGE {
        return Err(format!(
            "numerical artifact content changed: {}",
            path.display()
        ));
    }
    if bytes >= 1_048_576 {
        if baux::artifact::runs::file_digest(path)? != sha256 {
            return Err(format!(
                "numerical artifact digest changed: {}",
                path.display()
            ));
        }
        return Ok(());
    }
    let mut input = File::open(path).map_err(|e| e.to_string())?;
    let mut hash = Sha256::new();
    let mut buffer = [0u8; 65536];
    let mut total = 0u64;
    loop {
        let size = input.read(&mut buffer).map_err(|e| e.to_string())?;
        if size == 0 {
            break;
        }
        total += size as u64;
        if total > bytes {
            return Err("numerical artifact grew while verifying".into());
        }
        hash.update(&buffer[..size]);
    }
    if total != bytes || format!("{:x}", hash.finalize()) != sha256 {
        return Err(format!(
            "numerical artifact digest changed: {}",
            path.display()
        ));
    }
    Ok(())
}

fn temporary(root: &Path) -> Result<(PathBuf, File), String> {
    let path = root.join(format!(
        ".numerical-{}-{}.tmp",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    let file = OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(&path)
        .map_err(|e| format!("create {}: {e}", path.display()))?;
    Ok((path, file))
}

fn commit(temp: &Path, target: &Path, sha256: &str, bytes: u64) -> Result<(), String> {
    match fs::hard_link(temp, target) {
        Ok(()) => verify(target, sha256, bytes),
        Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => verify(target, sha256, bytes),
        Err(e) => Err(format!("publish {}: {e}", target.display())),
    }
}

fn immutable(path: &Path, bytes: &[u8]) -> Result<(), String> {
    let parent = path.parent().ok_or("numerical artifact has no parent")?;
    let (temp, mut file) = temporary(parent)?;
    let result = (|| {
        file.write_all(bytes).map_err(|e| e.to_string())?;
        commit(&temp, path, &digest(bytes), bytes.len() as u64)
    })();
    drop(file);
    let cleanup = fs::remove_file(&temp);
    result?;
    cleanup.map_err(|e| e.to_string())
}

fn encoded(record: &Record) -> Result<Vec<u8>, String> {
    validate(record)?;
    let bytes = cbor::to_vec(record).map_err(|e| e.to_string())?;
    if bytes.len() as u64 > MAX_RECORD {
        return Err("numerical artifact record is too large".into());
    }
    Ok(bytes)
}

fn fast_prepare(
    root: &Path,
    blobs: &Path,
    records: &Path,
    library: &Path,
    provenance: &Provenance,
    len: u64,
) -> Option<Receipt> {
    if len < 1_048_576 {
        return None;
    }
    let sha256 = baux::artifact::runs::file_digest(library).ok()?;
    let target = blobs.join(format!("{sha256}.so"));
    verify(&target, &sha256, len).ok()?;
    let record = Record {
        schema: 1,
        abi: "baux-numerical-7-v1".into(),
        provenance: provenance.clone(),
        sha256,
        bytes: len,
    };
    let bytes = encoded(&record).ok()?;
    let id = digest(&bytes);
    immutable(&records.join(format!("{id}.cbor")), &bytes).ok()?;
    load(root, &id).ok()
}

/// Copy the exact native image into immutable, content-addressed storage.
/// The caller must ensure the supplied source revision actually produced this
/// build. Capture occurs once at build/launch, never at each numerical
/// decision.
pub fn prepare(root: &Path, library: &Path, provenance: &Provenance) -> Result<Receipt, String> {
    let meta =
        fs::symlink_metadata(library).map_err(|e| format!("inspect {}: {e}", library.display()))?;
    if !meta.is_file() || meta.len() == 0 || meta.len() > MAX_IMAGE {
        return Err("numerical library must be a bounded regular file".into());
    }
    // Validate caller metadata before performing any publication.
    validate(&Record {
        schema: 1,
        abi: "baux-numerical-7-v1".into(),
        provenance: provenance.clone(),
        sha256: "0".repeat(64),
        bytes: meta.len(),
    })?;
    directory(root)?;
    let blobs = root.join("blobs");
    let records = root.join("records");
    directory(&blobs)?;
    directory(&records)?;
    if let Some(receipt) = fast_prepare(root, &blobs, &records, library, provenance, meta.len()) {
        return Ok(receipt);
    }
    let (temp, mut output) = temporary(&blobs)?;
    let copied: Result<Record, String> = (|| {
        let mut input = File::open(library).map_err(|e| e.to_string())?;
        let mut hash = Sha256::new();
        let mut buffer = [0u8; 65536];
        let mut total = 0u64;
        loop {
            let size = input.read(&mut buffer).map_err(|e| e.to_string())?;
            if size == 0 {
                break;
            }
            total += size as u64;
            if total > MAX_IMAGE {
                return Err("numerical image grew beyond limit".into());
            }
            hash.update(&buffer[..size]);
            output
                .write_all(&buffer[..size])
                .map_err(|e| e.to_string())?;
        }
        let sha256 = format!("{:x}", hash.finalize());
        if total != meta.len() {
            return Err("numerical build image changed while capturing".into());
        }
        verify(library, &sha256, total)?;
        output.sync_all().map_err(|e| e.to_string())?;
        let target = blobs.join(format!("{sha256}.so"));
        commit(&temp, &target, &sha256, total)?;
        Ok(Record {
            schema: 1,
            abi: "baux-numerical-7-v1".into(),
            provenance: provenance.clone(),
            sha256,
            bytes: total,
        })
    })();
    drop(output);
    let cleanup = fs::remove_file(&temp);
    let record = copied?;
    cleanup.map_err(|e| e.to_string())?;
    let bytes = encoded(&record)?;
    let id = digest(&bytes);
    immutable(&records.join(format!("{id}.cbor")), &bytes)?;
    load(root, &id)
}

/// Validate the selected immutable receipt and the native image it names.
pub fn load(root: &Path, id: &str) -> Result<Receipt, String> {
    if !identity(id) {
        return Err("invalid numerical receipt identity".into());
    }
    let path = root.join("records").join(format!("{id}.cbor"));
    let meta = fs::symlink_metadata(&path).map_err(|e| e.to_string())?;
    if !meta.is_file() || meta.len() > MAX_RECORD {
        return Err("numerical receipt is not a bounded regular file".into());
    }
    let mut bytes = Vec::new();
    File::open(&path)
        .map_err(|e| e.to_string())?
        .take(MAX_RECORD + 1)
        .read_to_end(&mut bytes)
        .map_err(|e| e.to_string())?;
    if bytes.len() as u64 > MAX_RECORD || digest(&bytes) != id {
        return Err("numerical receipt content changed".into());
    }
    let record: Record = cbor::from_slice(&bytes).map_err(|e| e.to_string())?;
    validate(&record)?;
    let library = root.join("blobs").join(format!("{}.so", record.sha256));
    verify(&library, &record.sha256, record.bytes)?;
    let library = library.canonicalize().map_err(|e| e.to_string())?;
    Ok(Receipt {
        id: id.into(),
        library,
        record,
    })
}

/// Preserve the selected image identity in a run directory before loading it.
/// This establishes a local provenance record, not authenticity or execution
/// proof.
pub fn publish(run_dir: &Path, receipt: &Receipt) -> Result<(), String> {
    let bytes = encoded(&receipt.record)?;
    if digest(&bytes) != receipt.id {
        return Err("numerical receipt identity mismatch".into());
    }
    verify(
        &receipt.library,
        &receipt.record.sha256,
        receipt.record.bytes,
    )?;
    directory(run_dir)?;
    immutable(&run_dir.join("numerical.cbor"), &bytes)
}

/// Load the library selected by the run launcher. Missing configuration is an
/// error.
///
/// # Safety
/// The environment must name the trusted build of the exact documented Lean/C
/// ABI. Its exported functions must implement the numerical predicate and
/// runtime lifecycle. Content-addressed provenance detects drift; it does not
/// establish these properties.
pub unsafe fn install_env() -> Result<(), String> {
    let path = std::env::var_os("BAUX_NUMERICAL_LIBRARY")
        .filter(|value| !value.is_empty())
        .ok_or("BAUX_NUMERICAL_LIBRARY is required for numerical checking")?;
    // SAFETY: the caller guarantees the library ABI and implementation.
    unsafe { install(Path::new(&path)) }
}

/// Install exactly one trusted checker library for this worker process.
///
/// # Safety
/// The library must export the exact numerical.c signatures, implement the
/// proved Lean predicate, and support the documented
/// initialization/finalization lifecycle. Its transitive native dependencies
/// must satisfy the same runtime assumptions.
pub unsafe fn install(path: &Path) -> Result<(), String> {
    let path = path
        .canonicalize()
        .map_err(|error| format!("numerical library {}: {error}", path.display()))?;
    let _guard = START.lock().map_err(|_| "numerical loader lock poisoned")?;
    if let Some(state) = HOST.get() {
        return match state {
            Ok(host) if host.path == path => INSTALLED
                .get()
                .cloned()
                .unwrap_or_else(|| Err("numerical verifier installation did not finish".into())),
            Ok(_) => Err("a different numerical library is already installed".into()),
            Err(error) => Err(error.clone()),
        };
    }
    let state = start(path);
    let result = state.as_ref().map(|_| ()).map_err(Clone::clone);
    HOST.set(state).map_err(|_| "numerical loader raced")?;
    result?;
    // SAFETY: dispatch only succeeds after the pinned Lean predicate returns
    // true. The host retains its library and all calls stay on its owner
    // thread.
    let installed = unsafe { numerical::install(dispatch) };
    let _ = INSTALLED.set(installed.clone());
    installed
}

fn start(path: PathBuf) -> Result<Host, String> {
    let (requests, inbox) = mpsc::sync_channel::<Request>(64);
    let async_init = std::env::var_os("BAUX_NATIVE_TEST").is_none()
        && fs::metadata(&path).is_ok_and(|m| m.len() >= 1_048_576);
    let (ready, initialized) = mpsc::sync_channel(1);
    let library = path.clone();
    std::thread::Builder::new()
        .name("lean-numerical".into())
        .stack_size(16 * 1024 * 1024)
        .spawn(move || {
            let first = if async_init {
                let Ok(req) = inbox.recv() else {
                    return;
                };
                Some(req)
            } else {
                None
            };
            let loaded = native::load(&library);
            let native = match loaded {
                Ok(native) => native,
                Err(error) => {
                    let _ = ready.send(Err(error.clone()));
                    if let Some(req) = first {
                        let _ = req.reply.send(Err(error.clone()));
                    }
                    while let Ok(request) = inbox.recv() {
                        let _ = request.reply.send(Err(error.clone()));
                    }
                    return;
                }
            };
            let _ = ready.send(Ok(()));
            if let Some(req) = first {
                let _ = req.reply.send(native.check(req.trace));
            }
            while let Ok(request) = inbox.recv() {
                let result = native.check(request.trace);
                let _ = request.reply.send(result);
            }
            // Finalize this thread only; global Lean objects/library remain
            // alive.
            native.finish();
        })
        .map_err(|error| format!("start numerical checker: {error}"))?;
    if !async_init {
        initialized
            .recv()
            .map_err(|_| "numerical checker exited during initialization")??;
    }
    Ok(Host { path, requests })
}

fn dispatch(trace: Trace) -> Result<(), String> {
    let host = HOST
        .get()
        .ok_or("numerical checker is not initialized")?
        .as_ref()
        .map_err(Clone::clone)?;
    let (reply, received) = mpsc::sync_channel(1);
    host.requests
        .send(Request { trace, reply })
        .map_err(|_| "numerical checker thread stopped")?;
    received
        .recv()
        .map_err(|_| "numerical checker returned no result")?
}

#[cfg(unix)]
mod native {
    use super::*;
    use std::ffi;
    use std::os::unix::ffi::OsStrExt;

    type Init = unsafe extern "C" fn() -> u8;
    type Finish = unsafe extern "C" fn();
    type Check = unsafe extern "C" fn(u8, u64, u64, u64, u64, u64, u8) -> u8;

    #[cfg_attr(target_os = "linux", link(name = "dl"))]
    unsafe extern "C" {
        fn dlopen(path: *const ffi::c_char, flags: ffi::c_int) -> *mut ffi::c_void;
        fn dlsym(handle: *mut ffi::c_void, symbol: *const ffi::c_char) -> *mut ffi::c_void;
        fn dlerror() -> *const ffi::c_char;
    }

    pub(super) struct Native {
        check: Check,
        finish: Finish,
    }

    fn error() -> String {
        // SAFETY: dlerror's NUL-terminated pointer is copied before another
        // loader call.
        let pointer = unsafe { dlerror() };
        if pointer.is_null() {
            "unknown loader error".into()
        } else {
            unsafe { ffi::CStr::from_ptr(pointer) }
                .to_string_lossy()
                .into_owned()
        }
    }

    pub(super) fn load(path: &Path) -> Result<Native, String> {
        let path = ffi::CString::new(path.as_os_str().as_bytes())
            .map_err(|_| "numerical library path contains NUL")?;
        // SAFETY: path is a valid C string. RTLD_NOW resolves all references
        // now. The handle is intentionally retained, including after a
        // failed initializer.
        let handle = unsafe { dlopen(path.as_ptr(), 2) };
        if handle.is_null() {
            return Err(format!("load numerical library: {}", error()));
        }
        let symbol = |name: &ffi::CStr| -> Result<*mut ffi::c_void, String> {
            // SAFETY: handle remains loaded and the symbol name is
            // NUL-terminated.
            let pointer = unsafe { dlsym(handle, name.as_ptr()) };
            if pointer.is_null() {
                Err(format!(
                    "numerical symbol {}: {}",
                    name.to_string_lossy(),
                    error()
                ))
            } else {
                Ok(pointer)
            }
        };
        // SAFETY: these exact names and signatures are the numerical.c/Lean
        // export ABI.
        let init: Init = unsafe { std::mem::transmute(symbol(c"baux_numerical_init")?) };
        let finish: Finish = unsafe { std::mem::transmute(symbol(c"baux_numerical_fini")?) };
        let check: Check = unsafe { std::mem::transmute(symbol(c"baux_numerical")?) };
        if unsafe { init() } != 1 {
            return Err("Lean numerical module initialization failed".into());
        }
        Ok(Native { check, finish })
    }

    impl Native {
        pub(super) fn check(&self, trace: Trace) -> Result<(), String> {
            // SAFETY: native callback runs on its initialized owner thread; all
            // arguments are scalar ABI values, and the library remains loaded.
            let result = unsafe {
                (self.check)(
                    trace.kind as u8,
                    trace.improvement,
                    trace.multiplier,
                    trace.left,
                    trace.right,
                    trace.threshold,
                    u8::from(trace.accepted),
                )
            };
            match result {
                1 => Ok(()),
                0 => Err(format!("Lean numerical check rejected {trace:?}")),
                _ => Err("Lean numerical checker returned an invalid Boolean".into()),
            }
        }

        pub(super) fn finish(self) {
            // SAFETY: finalization occurs on the thread that initialized Lean.
            unsafe { (self.finish)() };
        }
    }
}

#[cfg(not(unix))]
mod native {
    use super::*;
    pub(super) struct Native;
    pub(super) fn load(_: &Path) -> Result<Native, String> {
        Err("numerical library loading is unsupported on this host".into())
    }
    impl Native {
        pub(super) fn check(&self, _: Trace) -> Result<(), String> {
            Err("numerical checker unavailable".into())
        }
        pub(super) fn finish(self) {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use baux::semantics::numerical::{Kind, Outcome};

    fn provenance() -> Provenance {
        Provenance {
            revision: "test-sealed-jj-commit".into(),
            compiler: "leanprover/lean4:v4.35.0".into(),
        }
    }

    #[test]
    fn image_identity() {
        let temp = tempfile::tempdir().unwrap();
        let library = temp.path().join("build.so");
        let store = temp.path().join("images");
        fs::write(&library, b"test native image").unwrap();
        let first = prepare(&store, &library, &provenance()).unwrap();
        assert_eq!(load(&store, &first.id).unwrap(), first);
        assert_eq!(prepare(&store, &library, &provenance()).unwrap(), first);
        let mut changed = provenance();
        changed.revision = "different sealed revision".into();
        let second = prepare(&store, &library, &changed).unwrap();
        assert_ne!(first.id, second.id);
        assert_eq!(first.library, second.library);
        let run = temp.path().join("run");
        publish(&run, &first).unwrap();
        publish(&run, &first).unwrap();
        assert!(publish(&run, &second).is_err());
        let recorded: Record =
            cbor::from_slice(&fs::read(run.join("numerical.cbor")).unwrap()).unwrap();
        assert_eq!(recorded, first.record);
        fs::write(&library, b"new native build image").unwrap();
        let third = prepare(&store, &library, &provenance()).unwrap();
        assert_ne!(third.id, first.id);
        assert_ne!(third.library, first.library);
        assert_eq!(load(&store, &first.id).unwrap(), first);
    }

    #[test]
    fn image_tamper() {
        let temp = tempfile::tempdir().unwrap();
        let library = temp.path().join("build.so");
        fs::write(&library, b"native").unwrap();
        let store = temp.path().join("images");
        let receipt = prepare(&store, &library, &provenance()).unwrap();
        assert!(load(&store, "../../outside").is_err());
        fs::write(&receipt.library, b"broken").unwrap();
        assert!(load(&store, &receipt.id).is_err());
        assert!(publish(&temp.path().join("run"), &receipt).is_err());
        assert!(prepare(&store, &library, &provenance()).is_err());
        fs::write(&receipt.library, b"native").unwrap();
        let record = store.join("records").join(format!("{}.cbor", receipt.id));
        let mut bytes = fs::read(&record).unwrap();
        bytes[0] ^= 1;
        fs::write(&record, &bytes).unwrap();
        assert!(load(&store, &receipt.id).is_err());
        assert!(prepare(&store, &library, &provenance()).is_err());
        assert_eq!(fs::read(&record).unwrap(), bytes);
    }

    #[test]
    fn image_parallel() {
        let temp = tempfile::tempdir().unwrap();
        let library = temp.path().join("build.so");
        fs::write(&library, b"native parallel").unwrap();
        let store = temp.path().join("images");
        let receipts = std::thread::scope(|scope| {
            let workers: Vec<_> = (0..8)
                .map(|_| scope.spawn(|| prepare(&store, &library, &provenance()).unwrap()))
                .collect();
            workers
                .into_iter()
                .map(|w| w.join().unwrap())
                .collect::<Vec<_>>()
        });
        assert!(receipts.iter().all(|r| r == &receipts[0]));
        assert_eq!(fs::read_dir(store.join("blobs")).unwrap().count(), 1);
        assert_eq!(fs::read_dir(store.join("records")).unwrap().count(), 1);
    }

    #[cfg(unix)]
    #[test]
    fn image_links() {
        let temp = tempfile::tempdir().unwrap();
        let library = temp.path().join("build.so");
        let link = temp.path().join("alias.so");
        fs::write(&library, b"native").unwrap();
        std::os::unix::fs::symlink(&library, &link).unwrap();
        let store = temp.path().join("images");
        assert!(prepare(&store, &link, &provenance()).is_err());
        let receipt = prepare(&store, &library, &provenance()).unwrap();
        fs::remove_file(&receipt.library).unwrap();
        std::os::unix::fs::symlink(&library, &receipt.library).unwrap();
        assert!(load(&store, &receipt.id).is_err());
    }

    #[test]
    #[ignore = "requires the built Lean checker in BAUX_TEST_NUMERICAL"]
    fn native_evidence() {
        if std::env::var_os("BAUX_NATIVE_TEST").is_none() {
            let status = std::process::Command::new(std::env::current_exe().unwrap())
                .args([
                    "--exact",
                    "numerical::tests::native_evidence",
                    "--ignored",
                    "--nocapture",
                ])
                .env("BAUX_NATIVE_TEST", "1")
                .status()
                .unwrap();
            assert!(status.success());
            return;
        }
        assert!(numerical::require().is_err());
        let library = std::env::var_os("BAUX_TEST_NUMERICAL").expect("explicit checker path");
        let snapshot = tempfile::tempdir().unwrap();
        let receipt = prepare(snapshot.path(), Path::new(&library), &provenance()).unwrap();
        publish(&snapshot.path().join("run"), &receipt).unwrap();
        // SAFETY: this explicit integration test receives the pinned, built
        // Lean bridge. Loading the captured copy also checks relocation
        // and native dependency resolution.
        unsafe { install(&receipt.library) }.unwrap();
        numerical::require().unwrap();
        let start = std::time::Instant::now();
        let mut checks = 0;
        for left in [
            0.0f64,
            -0.0,
            f64::from_bits(1),
            f64::MIN_POSITIVE,
            0.25,
            2.0,
            f64::MAX,
        ] {
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
                let mut changed = trace;
                changed.accepted = true;
                assert!(numerical::check(changed).is_err());
                changed = trace;
                changed.threshold ^= 1;
                assert!(numerical::check(changed).is_err());
                checks += 3;
            }
        }
        for invalid in [f64::NAN, f64::INFINITY, -1.0] {
            let trace = Trace::observed(Kind::Paired, 1.0, 2.0, invalid, 0.0, 0.0, true);
            assert!(numerical::check(trace).is_err());
            checks += 1;
        }
        let threads: Vec<_> = (0..8)
            .map(|_| {
                std::thread::spawn(|| {
                    for _ in 0..16 {
                        let trace = Trace::observed(Kind::Paired, 3.0, 2.0, 1.0, 0.0, 2.0, true);
                        assert_eq!(numerical::check(trace).unwrap(), Outcome::Checked);
                    }
                })
            })
            .collect();
        for thread in threads {
            thread.join().unwrap();
        }
        checks += 8 * 16;
        let mut controller =
            baux::semantics::optimizer::RadiusController::new(0.125, 0.01, 1.0).unwrap();
        for improvement in [3.0, -3.0] {
            let decision = controller
                .observe(improvement, 1.0, 3.0, 1.0, 0.1, false)
                .unwrap();
            assert_eq!(decision.numerical, Outcome::Checked);
            assert_eq!(decision.accepted, improvement > 0.0);
            checks += 1;
        }
        assert!(numerical::unchecked("vector override").is_err());
        eprintln!(
            "Lean numerical checker: {checks} valid/tampered calls in {:?}",
            start.elapsed()
        );
    }
}
