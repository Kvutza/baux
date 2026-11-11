//! Explicitly load TVM's pinned device plugin. It has no referenced C symbols,
//! so the linker is allowed to remove it from ordinary dylib dependencies.
use crate::portable::Result;
use std::ffi::{CStr, c_char, c_void};
use std::sync::OnceLock;

unsafe extern "C" {
    fn dlopen(path: *const c_char, flags: i32) -> *mut c_void;
    fn dlerror() -> *const c_char;
}

pub(super) fn metal() -> Result<()> {
    static PLUGIN: OnceLock<Result<usize>> = OnceLock::new();
    PLUGIN
        .get_or_init(|| {
            // NOW | GLOBAL. Bundles and Buck executables provide an rpath to
            // the pinned artifact. Registry entries require its
            // code for process life.
            let handle = unsafe { dlopen(c"@rpath/libtvm_runtime_metal.dylib".as_ptr(), 2 | 8) };
            if handle.is_null() {
                let error = unsafe { dlerror() };
                return Err(if error.is_null() {
                    "TVM Metal plugin could not load".into()
                } else {
                    unsafe { CStr::from_ptr(error) }
                        .to_string_lossy()
                        .into_owned()
                });
            }
            Ok(handle as usize)
        })
        .as_ref()
        .map(|_| ())
        .map_err(Clone::clone)
}
