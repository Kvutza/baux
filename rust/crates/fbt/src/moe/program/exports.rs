//! Stable C entry point for the initialized fixed-shape model.
use super::Resident;
use baux::portable::Pending;
use std::{
    cell::RefCell,
    ffi::{CString, c_char},
    sync::Mutex,
};

thread_local! { static ERROR: RefCell<CString> = RefCell::new(CString::default()); }

pub struct Model(Mutex<Resident>);

fn report<T>(call: impl FnOnce() -> Result<T, String>) -> Option<T> {
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(call)) {
        Ok(Ok(value)) => Some(value),
        result => {
            let message = match result {
                Ok(Err(error)) => error,
                _ => "resident model API panicked".into(),
            };
            ERROR.with(|error| {
                *error.borrow_mut() =
                    CString::new(message.replace('\0', " ")).expect("NUL removed");
            });
            None
        }
    }
}

unsafe fn slice<'a, T>(pointer: *const T, length: usize) -> Result<&'a [T], String> {
    if length == 0 {
        return Ok(&[]);
    }
    if pointer.is_null() {
        return Err("null model input pointer".into());
    }
    Ok(unsafe { std::slice::from_raw_parts(pointer, length) })
}

#[unsafe(no_mangle)]
pub extern "C" fn bauxmodel_error() -> *const c_char {
    ERROR.with(|error| error.borrow().as_ptr())
}

#[unsafe(no_mangle)]
pub extern "C" fn bauxmodel_create(seed: u64) -> *mut Model {
    bauxmodel_createon(seed, 0)
}

#[unsafe(no_mangle)]
pub extern "C" fn bauxmodel_createon(seed: u64, backend: u32) -> *mut Model {
    report(|| {
        let backend = match backend {
            0 => super::Backend::Native,
            1 => super::Backend::Tvm,
            _ => return Err("unknown resident model backend".into()),
        };
        Resident::with_backend(
            crate::config::ModelArchitecture::FbtPisa1Mhc4V1,
            crate::config::ModelInitialization::FbtPisa1Mhc4,
            seed,
            crate::trust_region::TRLengthConfig::default(),
            crate::ProposalKernel::Gaussian,
            backend,
        )
        .map(|resident| Model(Mutex::new(resident)))
    })
    .map(|model| Box::into_raw(Box::new(model)))
    .unwrap_or(std::ptr::null_mut())
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn bauxmodel_drop(model: *mut Model) {
    if !model.is_null() {
        drop(unsafe { Box::from_raw(model) });
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn bauxmodel_parameters(model: *const Model) -> usize {
    unsafe { model.as_ref() }
        .and_then(|model| model.0.lock().ok().map(|resident| resident.parameters))
        .unwrap_or(0)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn bauxmodel_rows(model: *const Model) -> usize {
    unsafe { model.as_ref() }
        .and_then(|model| model.0.lock().ok().map(|resident| resident.rows()))
        .unwrap_or(0)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn bauxmodel_submit(
    model: *const Model,
    tokens: *const u32,
    labels: *const u32,
    mask: *const u8,
    length: usize,
) -> *mut Pending {
    report(|| {
        let model = unsafe { model.as_ref() }.ok_or("null model handle")?;
        let resident = model
            .0
            .lock()
            .map_err(|_| "resident model mutex poisoned")?;
        let batch = resident.batch(
            unsafe { slice(tokens, length) }?,
            unsafe { slice(labels, length) }?,
            unsafe { slice(mask, length) }?,
        )?;
        let weights = resident.search.program_base(&resident.forward)?;
        resident.submit(weights, &batch)
    })
    .map(|pending| Box::into_raw(Box::new(pending)))
    .unwrap_or(std::ptr::null_mut())
}
