//! Owned Rust entry points into Apache TVM's versioned C ABI.
//!
//! The ABI is supplied by `apache-tvm-ffi`, not by a Python extension. See
//! <https://tvm.apache.org/ffi/reference/c_api.html>. Object and function
//! handles below own one TVM reference and release it exactly once.

use std::ffi::{CString, c_char, c_void};
use std::ptr::NonNull;
mod callback;
mod comparison;
mod compiler;
mod contraction;
mod device;
mod element;
mod gather;
mod graph;
mod lower;
#[cfg(test)]
#[path = "tvm/tests.rs"]
mod model_tests;
mod network;
mod session;
mod tensor;
mod transform;
pub use lower::TvmExecutable;
pub use tensor::TvmTensor;

impl super::Executable {
    /// Load an already compiled TVM artifact through the shared portable ABI.
    pub fn load_tvm(
        library: &str,
        entry: &str,
        device: &str,
        inputs: Vec<super::Tensor>,
        outputs: Vec<super::Tensor>,
    ) -> super::Result<Self> {
        for spec in inputs.iter().chain(&outputs) {
            super::layout(spec)?;
        }
        let library = CString::new(library).map_err(|error| error.to_string())?;
        let entry = CString::new(entry).map_err(|error| error.to_string())?;
        let device = CString::new(device).map_err(|error| error.to_string())?;
        let bridge = super::raw::tvm()?;
        let mut pointer = std::ptr::null_mut();
        let mut error = [0; 2048];
        let code = unsafe {
            (bridge.load)(
                library.as_ptr(),
                entry.as_ptr(),
                device.as_ptr(),
                &raw mut pointer,
                error.as_mut_ptr(),
                error.len(),
            )
        };
        super::raw::status(code, &error)?;
        Ok(Self {
            owner: std::sync::Arc::new(super::Session(std::sync::Mutex::new(super::raw::Context(
                pointer,
                &bridge.api,
            )))),
            inputs: inputs.into(),
            outputs: outputs.into(),
            queue: super::pending::Queue::new()?,
            kernel: None,
        })
    }
}

const TYPE_NONE: i32 = 0;
const TYPE_INTEGER: i32 = 1;
const TYPE_BOOLEAN: i32 = 2;
const RAW_STRING: i32 = 8;
const TYPE_OBJECT: i32 = 64;
const TYPE_FUNCTION: i32 = 68;
const TYPE_MODULE: i32 = 73;

#[repr(C)]
struct RawVersion {
    major: u32,
    minor: u32,
    patch: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct RawBytes {
    data: *const c_char,
    size: usize,
}

#[repr(C)]
struct RawObject {
    references: u64,
    type_index: i32,
    padding: u32,
    deleter: usize,
}

#[repr(C)]
struct RawErrorCell {
    kind: RawBytes,
    message: RawBytes,
    backtrace: RawBytes,
    update_backtrace: usize,
    cause: *mut c_void,
    context: *mut c_void,
}

#[repr(C)]
struct RawError {
    object: RawObject,
    cell: RawErrorCell,
}

#[repr(C)]
#[derive(Clone, Copy)]
union RawValue {
    integer: i64,
    float: f64,
    pointer: *mut c_void,
    string: *const c_char,
    dtype: tensor::Dtype,
    device: tensor::Device,
}

#[repr(C)]
struct RawAny {
    type_index: i32,
    padding: u32,
    value: RawValue,
}

impl RawAny {
    fn none() -> Self {
        Self {
            type_index: 0,
            padding: 0,
            value: RawValue { integer: 0 },
        }
    }
    fn boolean(value: bool) -> Self {
        Self {
            type_index: 2,
            padding: 0,
            value: RawValue {
                integer: i64::from(value),
            },
        }
    }
    fn string(value: &str) -> super::Result<Self> {
        let f = ffi()?;
        let bytes = RawBytes {
            data: value.as_ptr().cast(),
            size: value.len(),
        };
        let mut result = Self::none();
        status(unsafe { (f.string)(&bytes, &mut result) }, "create string")?;
        Ok(result)
    }
    fn integer(value: i64) -> Self {
        Self {
            type_index: TYPE_INTEGER,
            padding: 0,
            value: RawValue { integer: value },
        }
    }
    fn float(value: f64) -> Self {
        Self {
            type_index: 3,
            padding: 0,
            value: RawValue { float: value },
        }
    }

    fn take(&mut self) -> *mut c_void {
        let pointer = unsafe { self.value.pointer };
        self.type_index = TYPE_NONE;
        pointer
    }
}

impl Drop for RawAny {
    fn drop(&mut self) {
        if self.type_index >= TYPE_OBJECT {
            // Owned results carry exactly one reference, including unexpected
            // result types and error paths after a successful native call.
            let pointer = unsafe { self.value.pointer };
            if !pointer.is_null()
                && let Ok(f) = ffi()
            {
                let _ = unsafe { (f.decrement)(pointer) };
            }
        }
    }
}

impl Clone for RawAny {
    fn clone(&self) -> Self {
        if self.type_index >= TYPE_OBJECT {
            // This object is live for the duration of the increment.
            let f = ffi().expect("TVM FFI already initialized for live RawAny");
            let code = unsafe { (f.increment)(self.value.pointer) };
            assert_eq!(code, 0, "TVM reference increment failed");
        }
        Self {
            type_index: self.type_index,
            padding: self.padding,
            value: self.value,
        }
    }
}

pub(super) struct TvmFfi {
    pub string: unsafe extern "C" fn(*const RawBytes, *mut RawAny) -> i32,
    pub increment: unsafe extern "C" fn(*mut c_void) -> i32,
    pub version: unsafe extern "C" fn(*mut RawVersion),
    pub global: unsafe extern "C" fn(*const RawBytes, *mut *mut c_void) -> i32,
    pub call: unsafe extern "C" fn(*mut c_void, *mut RawAny, i32, *mut RawAny) -> i32,
    pub decrement: unsafe extern "C" fn(*mut c_void) -> i32,
    pub error: unsafe extern "C" fn(*mut *mut c_void),
    pub export_tensor: unsafe extern "C" fn(*mut c_void, *mut *mut c_void) -> i32,
    pub import_tensor: unsafe extern "C" fn(*mut c_void, i32, i32, *mut *mut c_void) -> i32,
    pub create_function: unsafe extern "C" fn(
        *mut c_void,
        unsafe extern "C" fn(*mut c_void, *const RawAny, i32, *mut RawAny) -> i32,
        unsafe extern "C" fn(*mut c_void),
        *mut *mut c_void,
    ) -> i32,
    pub register_global: unsafe extern "C" fn(*const RawBytes, *mut c_void, i32) -> i32,
    pub raise_error: unsafe extern "C" fn(*const c_char, *const c_char),
}

pub(super) fn ffi() -> super::Result<&'static TvmFfi> {
    static FFI: std::sync::OnceLock<super::Result<TvmFfi>> = std::sync::OnceLock::new();
    FFI.get_or_init(|| {
        let lib = crate::portable::dl::open_plugin(
            "BAUX_TVM_FFI_LIB",
            &["libtvm_ffi.dylib", "libtvm_ffi.so"],
            &["tvm_ffi/lib", "baux/.dylibs"],
            "Apache TVM FFI",
        )?;
        let _runtime = crate::portable::dl::open_plugin(
            "BAUX_TVM_RUNTIME_LIB",
            &["libtvm_runtime.dylib", "libtvm_runtime.so"],
            &["tvm/lib", "baux/.dylibs"],
            "Apache TVM runtime",
        )?;
        Ok(TvmFfi {
            string: unsafe { lib.sym(c"TVMFFIStringFromByteArray")? },
            increment: unsafe { lib.sym(c"TVMFFIObjectIncRef")? },
            version: unsafe { lib.sym(c"TVMFFIGetVersion")? },
            global: unsafe { lib.sym(c"TVMFFIFunctionGetGlobal")? },
            call: unsafe { lib.sym(c"TVMFFIFunctionCall")? },
            decrement: unsafe { lib.sym(c"TVMFFIObjectDecRef")? },
            error: unsafe { lib.sym(c"TVMFFIErrorMoveFromRaised")? },
            export_tensor: unsafe { lib.sym(c"TVMFFITensorToDLPackVersioned")? },
            import_tensor: unsafe { lib.sym(c"TVMFFITensorFromDLPackVersioned")? },
            create_function: unsafe { lib.sym(c"TVMFFIFunctionCreate")? },
            register_global: unsafe { lib.sym(c"TVMFFIFunctionSetGlobal")? },
            raise_error: unsafe { lib.sym(c"TVMFFIErrorSetRaisedFromCStr")? },
        })
    })
    .as_ref()
    .map_err(Clone::clone)
}

pub(crate) fn ensure_runtime() -> super::Result<()> {
    ffi().map(|_| ())
}

pub(super) fn ensure_compiler() -> super::Result<()> {
    ensure_runtime()?;
    static COMPILER: std::sync::OnceLock<super::Result<crate::portable::dl::Library>> =
        std::sync::OnceLock::new();
    COMPILER
        .get_or_init(|| {
            crate::portable::dl::open_plugin(
                "BAUX_TVM_COMPILER_LIB",
                &["libtvm_compiler.dylib", "libtvm_compiler.so"],
                &["tvm/lib", "baux/.dylibs"],
                "Apache TVM compiler",
            )
        })
        .as_ref()
        .map(|_| ())
        .map_err(Clone::clone)
}

fn status(code: i32, operation: &str) -> super::Result<()> {
    if code == 0 {
        Ok(())
    } else {
        let f = ffi()?;
        let mut pointer = std::ptr::null_mut();
        // A nonzero safe-call status stores one owned Error in thread-local
        // state. Moving it clears that state and transfers the reference here.
        unsafe { (f.error)(&raw mut pointer) };
        let Some(pointer) = NonNull::new(pointer) else {
            return Err(format!("TVM {operation} failed with status {code}"));
        };
        // The pinned ABI lays the public Error cell directly after its public
        // object header. Its byte arrays remain valid until the final decref.
        let error = unsafe { pointer.cast::<RawError>().as_ref() };
        let bytes = error.cell.message;
        let message = if bytes.data.is_null() {
            format!("status {code}")
        } else {
            let bytes = unsafe { std::slice::from_raw_parts(bytes.data.cast(), bytes.size) };
            String::from_utf8_lossy(bytes).into_owned()
        };
        let _ = unsafe { (f.decrement)(pointer.as_ptr()) };
        Err(format!("TVM {operation} failed: {message}"))
    }
}

/// Version of the native TVM FFI library linked into this process.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TvmFfiVersion {
    /// ABI major version.
    pub major: u32,
    /// ABI minor version.
    pub minor: u32,
    /// ABI patch version.
    pub patch: u32,
}

impl TvmFfiVersion {
    /// Read the version through TVM's stable C entry point.
    pub fn linked() -> super::Result<Self> {
        let f = ffi()?;
        let mut version = RawVersion {
            major: 0,
            minor: 0,
            patch: 0,
        };
        // `version` is valid writable storage for the duration of the call.
        unsafe { (f.version)(&raw mut version) };
        Ok(Self {
            major: version.major,
            minor: version.minor,
            patch: version.patch,
        })
    }
}

/// One owned reference to a function in TVM's process-wide registry.
pub struct TvmFunction<'module>(
    NonNull<c_void>,
    std::marker::PhantomData<&'module TvmModule>,
);

impl TvmFunction<'_> {
    /// Look up a registered function without importing Python.
    pub fn global(name: &str) -> super::Result<Option<Self>> {
        let f = ffi()?;
        let name = RawBytes {
            data: name.as_ptr().cast(),
            size: name.len(),
        };
        let mut output = std::ptr::null_mut();
        // TVM borrows `name` only for this call and initializes `output` with
        // one owned reference when the function exists.
        let code = unsafe { (f.global)(&raw const name, &raw mut output) };
        status(code, "global function lookup")?;
        Ok(NonNull::new(output).map(|pointer| Self(pointer, std::marker::PhantomData)))
    }

    fn call(&self, arguments: &mut [RawAny]) -> super::Result<RawAny> {
        let f = ffi()?;
        let count = i32::try_from(arguments.len()).map_err(|error| error.to_string())?;
        let mut result = RawAny {
            type_index: TYPE_NONE,
            padding: 0,
            value: RawValue { integer: 0 },
        };
        // The function and arguments outlive the call. TVM initializes
        // `result` as an owned Any because it enters as the required empty
        // value.
        let code = unsafe {
            (f.call)(
                self.0.as_ptr(),
                arguments.as_mut_ptr(),
                count,
                &raw mut result,
            )
        };
        status(code, "function call")?;
        Ok(result)
    }

    fn call_string(&self, value: &str) -> super::Result<RawAny> {
        let value = CString::new(value).map_err(|error| error.to_string())?;
        let argument = RawAny {
            type_index: RAW_STRING,
            padding: 0,
            value: RawValue {
                string: value.as_ptr(),
            },
        };
        self.call(&mut [argument])
    }

    /// Invoke an integer function through TVM's native packed-call ABI.
    pub fn call_i64(&self, value: i64) -> super::Result<i64> {
        let argument = RawAny {
            type_index: TYPE_INTEGER,
            padding: 0,
            value: RawValue { integer: value },
        };
        let result = self.call(&mut [argument])?;
        if result.type_index != TYPE_INTEGER {
            return Err(format!(
                "TVM integer call returned type {}",
                result.type_index
            ));
        }
        Ok(unsafe { result.value.integer })
    }
}

impl Drop for TvmFunction<'_> {
    fn drop(&mut self) {
        // This handle was returned with one owned TVM reference.
        if let Ok(f) = ffi() {
            let _ = unsafe { (f.decrement)(self.0.as_ptr()) };
        }
    }
}

/// Native TVM dynamic module with Rust-owned lifetime.
pub struct TvmModule(NonNull<c_void>);

impl TvmModule {
    /// Load a compiled TVM artifact through `ffi.ModuleLoadFromFile`.
    pub fn load(path: &str) -> super::Result<Self> {
        let loader = TvmFunction::global("ffi.ModuleLoadFromFile")?
            .ok_or("TVM module loader is not registered")?;
        let mut result = loader.call_string(path)?;
        if result.type_index != TYPE_MODULE {
            return Err(format!(
                "TVM module loader returned type {}",
                result.type_index
            ));
        }
        let pointer = result.take();
        NonNull::new(pointer)
            .map(Self)
            .ok_or_else(|| "TVM module loader returned a null module".into())
    }

    /// Retrieve a callable exported by this module.
    pub fn function(&self, name: &str) -> super::Result<Option<TvmFunction<'_>>> {
        let lookup = TvmFunction::global("ffi.ModuleGetFunction")?
            .ok_or("TVM module function lookup is not registered")?;
        let name = CString::new(name).map_err(|error| error.to_string())?;
        let module = RawAny {
            type_index: TYPE_MODULE,
            padding: 0,
            value: RawValue {
                pointer: self.0.as_ptr(),
            },
        };
        // The argument owns its own reference; self keeps the original.
        let f = ffi()?;
        status(unsafe { (f.increment)(self.0.as_ptr()) }, "retain module")?;
        let name = RawAny {
            type_index: RAW_STRING,
            padding: 0,
            value: RawValue {
                string: name.as_ptr(),
            },
        };
        let imports = RawAny {
            type_index: TYPE_BOOLEAN,
            padding: 0,
            value: RawValue { integer: 0 },
        };
        let mut result = lookup.call(&mut [module, name, imports])?;
        if result.type_index == TYPE_NONE {
            return Ok(None);
        }
        if result.type_index != TYPE_FUNCTION {
            return Err(format!(
                "TVM module function lookup returned type {}",
                result.type_index
            ));
        }
        let pointer = result.take();
        // The callable cannot outlive the library containing its machine code.
        Ok(NonNull::new(pointer).map(|pointer| TvmFunction(pointer, std::marker::PhantomData)))
    }
}

impl Drop for TvmModule {
    fn drop(&mut self) {
        // The result Any transferred one owned module reference into `Self`.
        if let Ok(f) = ffi() {
            let _ = unsafe { (f.decrement)(self.0.as_ptr()) };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{TvmFfiVersion, TvmFunction};

    #[test]
    fn loads_ffi() {
        let version = match TvmFfiVersion::linked() {
            Ok(v) => v,
            Err(err) => {
                assert!(
                    std::env::var_os("BAUX_MLIR_IMPORTER").is_none(),
                    "TVM FFI must load in compiler-configured test environments: {err}"
                );
                return;
            }
        };
        assert_eq!(version.major, 0);
        assert_eq!(version.minor, 1);
        assert!(version.patch >= 15);
        assert!(
            TvmFunction::global("ffi.ModuleLoadFromFile")
                .unwrap()
                .is_some()
        );
    }
}
