//! Lazy runtime dynamic-library loader (`dlopen` / `dlsym`) for optional
//! native compiler and device-runtime plugins (IREE, TVM, OpenXLA PJRT).
//!
//! Core `baux` carries zero link-time `LC_LOAD_DYLIB` / `DT_NEEDED` edges to
//! optional compiler toolchains. Plugins are opened with `RTLD_NOW |
//! RTLD_GLOBAL` on first use and retained for the lifetime of the process.

use super::Result;
use std::ffi::{CStr, CString, c_char, c_void};
use std::path::{Path, PathBuf};

#[cfg(unix)]
#[repr(C)]
struct DlInfo {
    fname: *const c_char,
    fbase: *mut c_void,
    sname: *const c_char,
    saddr: *mut c_void,
}

#[cfg(unix)]
unsafe extern "C" {
    fn dlopen(path: *const c_char, flags: i32) -> *mut c_void;
    fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
    fn dlerror() -> *const c_char;
    fn dladdr(addr: *const c_void, info: *mut DlInfo) -> i32;
}

#[cfg(target_os = "macos")]
const RTLD_FLAGS: i32 = 2 | 8; // RTLD_NOW | RTLD_GLOBAL
#[cfg(all(unix, not(target_os = "macos")))]
const RTLD_FLAGS: i32 = 2 | 0x100; // RTLD_NOW | RTLD_GLOBAL

/// Process-lifetime handle to a dynamically loaded plugin library.
#[derive(Clone, Copy)]
pub(super) struct Library(*mut c_void);

unsafe impl Send for Library {}
unsafe impl Sync for Library {}

#[cfg(unix)]
fn last_dlerror(fallback: &str) -> String {
    let err = unsafe { dlerror() };
    if err.is_null() {
        fallback.to_owned()
    } else {
        unsafe { CStr::from_ptr(err) }
            .to_string_lossy()
            .into_owned()
    }
}

#[cfg(unix)]
fn self_directory() -> Option<PathBuf> {
    let mut info = DlInfo {
        fname: std::ptr::null(),
        fbase: std::ptr::null_mut(),
        sname: std::ptr::null(),
        saddr: std::ptr::null_mut(),
    };
    let ok = unsafe { dladdr(self_directory as *const () as *const c_void, &raw mut info) };
    if ok == 0 || info.fname.is_null() {
        return None;
    }
    let path = PathBuf::from(
        unsafe { CStr::from_ptr(info.fname) }
            .to_string_lossy()
            .as_ref(),
    );
    path.parent().map(Path::to_path_buf)
}

fn site_roots(self_dir: Option<&Path>) -> Vec<PathBuf> {
    let mut roots = Vec::new();
    let mut push = |candidate: PathBuf| {
        if candidate.is_dir() && !roots.contains(&candidate) {
            roots.push(candidate);
        }
    };
    if let Some(dir) = self_dir
        && let Some(parent) = dir.parent()
    {
        push(parent.to_path_buf());
    }
    for var in ["BAUX_SITE_PACKAGES", "PYTHONPATH"] {
        if let Some(raw) = std::env::var_os(var) {
            for entry in std::env::split_paths(&raw) {
                push(entry);
            }
        }
    }
    for env_var in ["VIRTUAL_ENV", "CONDA_PREFIX"] {
        if let Some(prefix) = std::env::var_os(env_var) {
            let lib = PathBuf::from(prefix).join("lib");
            if let Ok(entries) = std::fs::read_dir(&lib) {
                for entry in entries.flatten() {
                    let name = entry.file_name();
                    if name.to_string_lossy().starts_with("python3.") {
                        push(entry.path().join("site-packages"));
                    }
                }
            }
        }
    }
    if let Ok(exe) = std::env::current_exe()
        && let Some(bin_dir) = exe.parent()
        && let Some(prefix) = bin_dir.parent()
    {
        let lib = prefix.join("lib");
        if let Ok(entries) = std::fs::read_dir(&lib) {
            for entry in entries.flatten() {
                if entry.file_name().to_string_lossy().starts_with("python3.") {
                    push(entry.path().join("site-packages"));
                }
            }
        }
    }
    roots
}

/// Locate an existing plugin file on disk without opening it (used e.g. to pass
/// library search directories to isolated child processes like `hlo_import`).
pub(super) fn find_file(
    env_var: &str,
    sonames: &[&str],
    python_rel_dirs: &[&str],
) -> Option<PathBuf> {
    if let Some(explicit) = std::env::var_os(env_var)
        && !explicit.is_empty()
    {
        let path = PathBuf::from(explicit);
        if path.is_file() {
            return Some(path);
        }
    }
    #[cfg(unix)]
    let self_dir = self_directory();
    #[cfg(not(unix))]
    let self_dir: Option<PathBuf> = None;

    if let Some(dir) = self_dir.as_deref() {
        for base in [dir.join(".dylibs"), dir.to_path_buf()] {
            for name in sonames {
                let candidate = base.join(name);
                if candidate.is_file() {
                    return Some(candidate);
                }
            }
        }
    }
    for root in site_roots(self_dir.as_deref()) {
        for rel in python_rel_dirs {
            let base = if rel.is_empty() {
                root.clone()
            } else {
                root.join(rel)
            };
            for name in sonames {
                let candidate = base.join(name);
                if candidate.is_file() {
                    return Some(candidate);
                }
            }
        }
    }
    None
}

#[cfg(unix)]
fn try_dlopen(spec: &str) -> std::result::Result<Library, String> {
    let c_spec = CString::new(spec).map_err(|e| e.to_string())?;
    let handle = unsafe { dlopen(c_spec.as_ptr(), RTLD_FLAGS) };
    if handle.is_null() {
        Err(last_dlerror(spec))
    } else {
        Ok(Library(handle))
    }
}

/// Open a native plugin shared library lazily at runtime.
pub(super) fn open_plugin(
    env_var: &str,
    sonames: &[&str],
    python_rel_dirs: &[&str],
    label: &str,
) -> Result<Library> {
    #[cfg(not(unix))]
    {
        let _ = (env_var, sonames, python_rel_dirs);
        return Err(format!("{label} native runtime requires a Unix platform"));
    }

    #[cfg(unix)]
    {
        if let Some(explicit) = std::env::var_os(env_var)
            && !explicit.is_empty()
        {
            let spec = explicit.to_string_lossy();
            return try_dlopen(&spec)
                .map_err(|err| format!("failed to load {label} from {env_var}={spec}: {err}"));
        }

        let mut last_error = None;
        if let Some(found) = find_file(env_var, sonames, python_rel_dirs) {
            match try_dlopen(&found.to_string_lossy()) {
                Ok(lib) => return Ok(lib),
                Err(err) => last_error = Some(err),
            }
        }

        for name in sonames {
            #[cfg(target_os = "macos")]
            if let Ok(lib) = try_dlopen(&format!("@rpath/{name}")) {
                return Ok(lib);
            }
            match try_dlopen(name) {
                Ok(lib) => return Ok(lib),
                Err(err) => {
                    if last_error.is_none() {
                        last_error = Some(err);
                    }
                }
            }
        }

        let detail = last_error.unwrap_or_else(|| "not found".into());
        Err(format!(
            "{label} native library not available ({detail}); install optional compiler extras with `pip install \"baux[compiler]\"` or set {env_var}"
        ))
    }
}

impl Library {
    /// Resolve a typed `extern "C"` function pointer from an opened library.
    pub(super) unsafe fn sym<F: Copy>(&self, name: &CStr) -> Result<F> {
        #[cfg(not(unix))]
        {
            let _ = name;
            return Err("native symbol resolution requires Unix".into());
        }
        #[cfg(unix)]
        {
            assert_eq!(
                std::mem::size_of::<F>(),
                std::mem::size_of::<*mut c_void>(),
                "function pointer width mismatch"
            );
            let ptr = unsafe { dlsym(self.0, name.as_ptr()) };
            if ptr.is_null() {
                return Err(format!(
                    "missing symbol {}: {}",
                    name.to_string_lossy(),
                    last_dlerror("dlsym returned null")
                ));
            }
            Ok(unsafe { std::mem::transmute_copy::<*mut c_void, F>(&ptr) })
        }
    }
}
