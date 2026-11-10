//! Private C ABI; all access is serialized by the owning Session mutex.
use std::ffi::{CStr, c_char, c_int, c_void};

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
    pub fn bauxiree_close(context: *mut c_void);
    pub fn bauxiree_put(
        context: *mut c_void,
        shape: *const usize,
        rank: usize,
        dtype: u32,
        data: *const u8,
        length: usize,
        output: *mut *mut c_void,
        error: *mut c_char,
        capacity: usize,
    ) -> c_int;
    pub fn bauxiree_read(
        context: *mut c_void,
        view: *mut c_void,
        output: *mut u8,
        length: usize,
        error: *mut c_char,
        capacity: usize,
    ) -> c_int;
    pub fn bauxiree_retain(view: *mut c_void);
    pub fn bauxiree_release(view: *mut c_void);
    pub fn bauxiree_call(
        context: *mut c_void,
        inputs: *const *mut c_void,
        count: usize,
        outputs: *mut *mut c_void,
        capacity: usize,
        error: *mut c_char,
        error_capacity: usize,
    ) -> c_int;
    pub fn bauxiree_matches(
        view: *mut c_void,
        shape: *const usize,
        rank: usize,
        dtype: u32,
    ) -> c_int;
}

pub fn status(code: c_int, error: &[c_char; 2048]) -> Result<(), String> {
    if code == 0 {
        return Ok(());
    }
    // The bridge always terminates this initialized buffer, even on truncation.
    let message = unsafe { CStr::from_ptr(error.as_ptr()) }.to_string_lossy();
    Err(format!("IREE status {code}: {message}"))
}

pub struct Context(pub *mut c_void);
// IREE sessions are thread-compatible. Session's mutex serializes every use,
// including buffer readback and destruction. No reference escapes that lock.
unsafe impl Send for Context {}
impl Drop for Context {
    fn drop(&mut self) {
        // This is the sole owning context handle; every buffer retains Session.
        unsafe { bauxiree_close(self.0) };
    }
}
