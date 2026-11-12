//! Private C ABI; all access is serialized by the owning Session mutex.
use crate::portable::dl::open_plugin;
use std::ffi::{CStr, c_char, c_int, c_void};
use std::sync::OnceLock;
mod api;
pub use api::Api;

pub struct IreeBridge {
    pub api: Api,
    pub open: unsafe extern "C" fn(*const c_char, *mut *mut c_void, *mut c_char, usize) -> c_int,
    pub import: unsafe extern "C" fn(
        *mut c_void,
        *mut c_void,
        usize,
        usize,
        *const usize,
        usize,
        u32,
        *mut *mut c_void,
        *mut c_char,
        usize,
    ) -> c_int,
    pub metal: unsafe extern "C" fn(
        *mut c_void,
        *mut c_void,
        *mut *mut c_void,
        *mut u64,
        *mut c_char,
        usize,
    ) -> c_int,
    pub load: unsafe extern "C" fn(
        *const u8,
        usize,
        *const c_char,
        *const c_char,
        *mut *mut c_void,
        *mut c_char,
        usize,
    ) -> c_int,
}

pub struct PjrtBridge {
    pub api: Api,
    pub compile: unsafe extern "C" fn(
        *const u8,
        usize,
        *const c_char,
        *const u8,
        usize,
        usize,
        *mut *mut c_void,
        *mut c_char,
        usize,
    ) -> c_int,
}

pub struct TvmBridge {
    pub api: Api,
    pub load: unsafe extern "C" fn(
        *const c_char,
        *const c_char,
        *const c_char,
        *mut *mut c_void,
        *mut c_char,
        usize,
    ) -> c_int,
}

pub fn iree() -> Result<&'static IreeBridge, String> {
    static BRIDGE: OnceLock<Result<IreeBridge, String>> = OnceLock::new();
    BRIDGE
        .get_or_init(|| {
            let lib = open_plugin(
                "BAUX_IREE_BRIDGE",
                &["libbaux_iree.dylib", "libbaux_iree.so"],
                &["baux/.dylibs"],
                "IREE runtime bridge",
            )?;
            Ok(IreeBridge {
                api: unsafe { Api::resolve(&lib, "bauxiree", true)? },
                open: unsafe { lib.sym(c"bauxiree_open")? },
                import: unsafe { lib.sym(c"bauxiree_import")? },
                metal: unsafe { lib.sym(c"bauxiree_metal")? },
                load: unsafe { lib.sym(c"bauxiree_load")? },
            })
        })
        .as_ref()
        .map_err(Clone::clone)
}

pub fn pjrt() -> Result<&'static PjrtBridge, String> {
    static BRIDGE: OnceLock<Result<PjrtBridge, String>> = OnceLock::new();
    BRIDGE
        .get_or_init(|| {
            // Load the OpenXLA PJRT CPU plugin into global symbol scope before
            // initializing the thin C++ bridge.
            let _plugin = open_plugin(
                "BAUX_PJRT_LIBRARY",
                &["libpjrt_cpu.dylib", "pjrt_c_api_cpu_plugin.so"],
                &["cyxla_pjrt_cpu", "jaxlib", "baux/.dylibs"],
                "OpenXLA PJRT CPU plugin",
            )?;
            let lib = open_plugin(
                "BAUX_PJRT_BRIDGE",
                &["libbaux_pjrt.dylib", "libbaux_pjrt.so"],
                &["baux/.dylibs"],
                "PJRT runtime bridge",
            )?;
            Ok(PjrtBridge {
                api: unsafe { Api::resolve(&lib, "bauxpjrt", false)? },
                compile: unsafe { lib.sym(c"bauxpjrt_compile")? },
            })
        })
        .as_ref()
        .map_err(Clone::clone)
}

pub fn tvm() -> Result<&'static TvmBridge, String> {
    static BRIDGE: OnceLock<Result<TvmBridge, String>> = OnceLock::new();
    BRIDGE
        .get_or_init(|| {
            crate::portable::tvm::ensure_runtime()?;
            let lib = open_plugin(
                "BAUX_TVM_BRIDGE",
                &["libbaux_tvm.dylib", "libbaux_tvm.so"],
                &["baux/.dylibs"],
                "TVM runtime bridge",
            )?;
            Ok(TvmBridge {
                api: unsafe { Api::resolve(&lib, "bauxtvm", false)? },
                load: unsafe { lib.sym(c"bauxtvm_load")? },
            })
        })
        .as_ref()
        .map_err(Clone::clone)
}

pub fn status(code: c_int, error: &[c_char; 2048]) -> Result<(), String> {
    if code == 0 {
        return Ok(());
    }
    // The bridge always terminates this initialized buffer, even on truncation.
    let message = unsafe { CStr::from_ptr(error.as_ptr()) }.to_string_lossy();
    Err(format!("native runtime status {code}: {message}"))
}

pub struct Context(pub *mut c_void, pub &'static Api);
// Native sessions are thread-compatible. Session's mutex serializes every use,
// including buffer readback and destruction. No reference escapes that lock.
unsafe impl Send for Context {}
impl Drop for Context {
    fn drop(&mut self) {
        // This is the sole owning context handle; every buffer retains Session.
        unsafe { (self.1.close)(self.0) };
    }
}
