//! C ABI for language bindings. See portable.h for pointer/ownership contracts.
use super::{Buffer, DTYPES, Executable, Pending, Result, Tensor};
use std::cell::RefCell;
use std::ffi::{CStr, CString, c_char, c_int};

thread_local! { static ERROR: RefCell<CString> = RefCell::new(CString::default()); }

#[repr(C)]
pub struct Spec {
    shape: *const usize,
    rank: usize,
    dtype: usize,
}

fn report<T>(f: impl FnOnce() -> Result<T>) -> Option<T> {
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(f)) {
        Ok(Ok(value)) => Some(value),
        result => {
            let message = match result {
                Ok(Err(e)) => e,
                _ => "native API panicked".into(),
            };
            ERROR.with(|error| {
                *error.borrow_mut() = CString::new(message.replace('\0', " ")).expect("NUL removed")
            });
            None
        }
    }
}

unsafe fn slice<'a, T>(pointer: *const T, length: usize) -> Result<&'a [T]> {
    if length == 0 {
        return Ok(&[]);
    }
    if pointer.is_null() {
        return Err("null array pointer".into());
    }
    // ABI caller guarantees alignment and a live range of length elements.
    Ok(unsafe { std::slice::from_raw_parts(pointer, length) })
}
unsafe fn string<'a>(pointer: *const c_char) -> Result<&'a str> {
    if pointer.is_null() {
        return Err("null string pointer".into());
    }
    unsafe { CStr::from_ptr(pointer) }
        .to_str()
        .map_err(|e| e.to_string())
}
unsafe fn specs(pointer: *const Spec, length: usize) -> Result<Vec<Tensor>> {
    unsafe { slice(pointer, length) }?
        .iter()
        .map(|spec| {
            let dtype = DTYPES.get(spec.dtype).ok_or("invalid dtype code")?.0;
            Tensor::new(unsafe { slice(spec.shape, spec.rank) }?.to_vec(), dtype)
        })
        .collect()
}
unsafe fn buffers(pointer: *const *const Buffer, length: usize) -> Result<Vec<Buffer>> {
    unsafe { slice(pointer, length) }?
        .iter()
        .map(|&p| {
            unsafe { p.as_ref() }
                .cloned()
                .ok_or_else(|| "null buffer handle".into())
        })
        .collect()
}

#[unsafe(no_mangle)]
pub extern "C" fn bauxprogram_error() -> *const c_char {
    ERROR.with(|error| error.borrow().as_ptr())
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn bauxprogram_xla(
    data: *const u8,
    length: usize,
    device: *const c_char,
    inputs: *const Spec,
    input_count: usize,
    outputs: *const Spec,
    output_count: usize,
) -> *mut Executable {
    report(|| unsafe {
        Executable::compile_xla(
            std::str::from_utf8(slice(data, length)?).map_err(|e| e.to_string())?,
            string(device)?,
            specs(inputs, input_count)?,
            specs(outputs, output_count)?,
        )
    })
    .map(|value| Box::into_raw(Box::new(value)))
    .unwrap_or(std::ptr::null_mut())
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn bauxprogram_load(
    data: *const u8,
    length: usize,
    entry: *const c_char,
    device: *const c_char,
    inputs: *const Spec,
    input_count: usize,
    outputs: *const Spec,
    output_count: usize,
) -> *mut Executable {
    report(|| unsafe {
        Executable::load(
            slice(data, length)?,
            string(entry)?,
            string(device)?,
            specs(inputs, input_count)?,
            specs(outputs, output_count)?,
        )
    })
    .map(|value| Box::into_raw(Box::new(value)))
    .unwrap_or(std::ptr::null_mut())
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn bauxprogram_tvm(
    library: *const c_char,
    entry: *const c_char,
    device: *const c_char,
    inputs: *const Spec,
    input_count: usize,
    outputs: *const Spec,
    output_count: usize,
) -> *mut Executable {
    report(|| unsafe {
        Executable::load_tvm(
            string(library)?,
            string(entry)?,
            string(device)?,
            specs(inputs, input_count)?,
            specs(outputs, output_count)?,
        )
    })
    .map(|value| Box::into_raw(Box::new(value)))
    .unwrap_or(std::ptr::null_mut())
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn bauxprogram_drop(pointer: *mut Executable) {
    if !pointer.is_null() {
        drop(unsafe { Box::from_raw(pointer) });
    }
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn bauxbuffer_drop(pointer: *mut Buffer) {
    if !pointer.is_null() {
        drop(unsafe { Box::from_raw(pointer) });
    }
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn bauxpending_drop(pointer: *mut Pending) {
    if !pointer.is_null() {
        drop(unsafe { Box::from_raw(pointer) });
    }
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn bauxbatch_drop(pointer: *mut Vec<Buffer>) {
    if !pointer.is_null() {
        drop(unsafe { Box::from_raw(pointer) });
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn bauxprogram_put(
    program: *const Executable,
    spec: *const Spec,
    data: *const u8,
    length: usize,
) -> *mut Buffer {
    report(|| {
        let program = unsafe { program.as_ref() }.ok_or("null program handle")?;
        let spec = unsafe { specs(spec, 1) }?.remove(0);
        program.put(spec, unsafe { slice(data, length) }?)
    })
    .map(|value| Box::into_raw(Box::new(value)))
    .unwrap_or(std::ptr::null_mut())
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn bauxbuffer_read(
    buffer: *const Buffer,
    output: *mut u8,
    length: usize,
) -> c_int {
    report(|| {
        let buffer = unsafe { buffer.as_ref() }.ok_or("null buffer handle")?;
        if length != 0 && output.is_null() {
            return Err("invalid readback destination".into());
        }
        // The ABI caller supplies a writable, exclusive range for this call.
        let destination = if length == 0 {
            &mut []
        } else {
            unsafe { std::slice::from_raw_parts_mut(output, length) }
        };
        buffer.read_into(destination)
    })
    .is_some() as c_int
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn bauxprogram_run(
    program: *const Executable,
    inputs: *const *const Buffer,
    length: usize,
) -> *mut Vec<Buffer> {
    report(|| {
        unsafe { program.as_ref() }
            .ok_or("null program handle")?
            .run(&unsafe { buffers(inputs, length) }?)
    })
    .map(|value| Box::into_raw(Box::new(value)))
    .unwrap_or(std::ptr::null_mut())
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn bauxprogram_submit(
    program: *const Executable,
    inputs: *const *const Buffer,
    length: usize,
) -> *mut Pending {
    report(|| {
        unsafe { program.as_ref() }
            .ok_or("null program handle")?
            .submit(unsafe { buffers(inputs, length) }?)
    })
    .map(|value| Box::into_raw(Box::new(value)))
    .unwrap_or(std::ptr::null_mut())
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn bauxpending_ready(pending: *const Pending) -> c_int {
    unsafe { pending.as_ref() }.is_some_and(Pending::is_ready) as c_int
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn bauxpending_wait(pending: *mut Pending) -> *mut Vec<Buffer> {
    report(|| {
        if pending.is_null() {
            return Err("completion was already consumed".into());
        }
        // wait consumes the handle on success and failure alike.
        unsafe { Box::from_raw(pending) }.wait()
    })
    .map(|value| Box::into_raw(Box::new(value)))
    .unwrap_or(std::ptr::null_mut())
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn bauxbatch_count(batch: *const Vec<Buffer>) -> usize {
    unsafe { batch.as_ref() }.map_or(0, Vec::len)
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn bauxbatch_take(batch: *const Vec<Buffer>, index: usize) -> *mut Buffer {
    report(|| {
        unsafe { batch.as_ref() }
            .and_then(|batch| batch.get(index))
            .cloned()
            .ok_or_else(|| "invalid result index".into())
    })
    .map(|value| Box::into_raw(Box::new(value)))
    .unwrap_or(std::ptr::null_mut())
}
