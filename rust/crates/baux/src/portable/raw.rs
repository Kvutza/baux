//! Private C ABI; all access is serialized by the owning Session mutex.
use std::ffi::{CStr, c_char, c_int, c_void};
mod api;
pub use api::{Api, IREE, TVM};

unsafe extern "C" {
    pub fn bauxiree_open(
        driver: *const c_char,
        output: *mut *mut c_void,
        error: *mut c_char,
        capacity: usize,
    ) -> c_int;
    pub fn bauxiree_import(
        context: *mut c_void,
        handle: *mut c_void,
        offset: usize,
        length: usize,
        shape: *const usize,
        rank: usize,
        dtype: u32,
        output: *mut *mut c_void,
        error: *mut c_char,
        capacity: usize,
    ) -> c_int;
    pub fn bauxiree_metal(
        context: *mut c_void,
        view: *mut c_void,
        handle: *mut *mut c_void,
        offset: *mut u64,
        error: *mut c_char,
        capacity: usize,
    ) -> c_int;
    pub fn bauxiree_load(
        data: *const u8,
        length: usize,
        driver: *const c_char,
        entry: *const c_char,
        output: *mut *mut c_void,
        error: *mut c_char,
        capacity: usize,
    ) -> c_int;
    pub fn bauxtvm_load(
        path: *const c_char,
        entry: *const c_char,
        device: *const c_char,
        output: *mut *mut c_void,
        error: *mut c_char,
        capacity: usize,
    ) -> c_int;
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
