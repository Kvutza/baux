//! Owned native packed callbacks. The VM and registry retain the callback;
//! removal drops the registry reference only after its executable is dropped.
//! ABI: tvm/ffi/c_api.h, TVMFFIFunctionCreate and TVMFFISafeCallType.
use super::{RawAny, RawBytes, TvmTensor};
use crate::portable::{Result, Tensor};
use std::ffi::{CString, c_void};
use std::sync::atomic::{AtomicU64, Ordering};

type Kernel = dyn Fn(&[TvmTensor]) -> Result<Vec<TvmTensor>> + Send + Sync;
struct Callback {
    inputs: Vec<Tensor>,
    outputs: Vec<Tensor>,
    kernel: Box<Kernel>,
}
pub(super) struct Registration {
    pub name: String,
}

unsafe extern "C" fn destroy(context: *mut c_void) {
    drop(unsafe { Box::from_raw(context.cast::<Callback>()) });
}

unsafe extern "C" fn invoke(
    context: *mut c_void,
    args: *const RawAny,
    count: i32,
    output: *mut RawAny,
) -> i32 {
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| -> Result<RawAny> {
        let callback = unsafe { &*context.cast::<Callback>() };
        if count < 0 || count as usize != callback.inputs.len() {
            return Err("native TVM callback input count differs".into());
        }
        let args = if count == 0 {
            &[][..]
        } else {
            unsafe { std::slice::from_raw_parts(args, count as usize) }
        };
        // RawAny::clone retains each borrowed TVM object before Rust owns it.
        let inputs = args
            .iter()
            .map(|value| TvmTensor::from_value(value.clone()))
            .collect::<Result<Vec<_>>>()?;
        if inputs
            .iter()
            .zip(&callback.inputs)
            .any(|(value, spec)| value.spec() != spec)
        {
            return Err("native TVM callback input signature differs".into());
        }
        let outputs = (callback.kernel)(&inputs)?;
        if outputs.len() != callback.outputs.len()
            || outputs
                .iter()
                .zip(&callback.outputs)
                .any(|(value, spec)| value.spec() != spec)
        {
            return Err("native TVM callback output signature differs".into());
        }
        super::compiler::array(&outputs.iter().map(|v| v.value.clone()).collect::<Vec<_>>())
    }));
    match result {
        Ok(Ok(value)) => {
            unsafe { output.write(value) };
            0
        }
        failure => {
            let message = match failure {
                Ok(Err(message)) => message,
                _ => "native TVM callback panicked".into(),
            };
            let message = CString::new(message.replace('\0', " ")).expect("NUL removed");
            if let Ok(f) = super::ffi() {
                unsafe { (f.raise_error)(c"RuntimeError".as_ptr(), message.as_ptr()) };
            }
            -1
        }
    }
}

impl Registration {
    pub fn new(
        inputs: Vec<Tensor>,
        outputs: Vec<Tensor>,
        kernel: impl Fn(&[TvmTensor]) -> Result<Vec<TvmTensor>> + Send + Sync + 'static,
    ) -> Result<Self> {
        let f = super::ffi()?;
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let name = format!("baux.native.{}", NEXT.fetch_add(1, Ordering::Relaxed));
        let context = Box::into_raw(Box::new(Callback {
            inputs,
            outputs,
            kernel: Box::new(kernel),
        }))
        .cast();
        let mut pointer = std::ptr::null_mut();
        let code = unsafe { (f.create_function)(context, invoke, destroy, &raw mut pointer) };
        if code != 0 {
            unsafe { destroy(context) };
            super::status(code, "create native callback")?;
        }
        let function = RawAny {
            type_index: super::TYPE_FUNCTION,
            padding: 0,
            value: super::RawValue { pointer },
        };
        let bytes = RawBytes {
            data: name.as_ptr().cast(),
            size: name.len(),
        };
        super::status(
            unsafe { (f.register_global)(&bytes, function.value.pointer, 0) },
            "register native callback",
        )?;
        Ok(Self { name })
    }
}

impl Drop for Registration {
    fn drop(&mut self) {
        if let Ok(name) = RawAny::string(&self.name) {
            let _ = super::compiler::call("ffi.FunctionRemoveGlobal", &mut [name]);
        }
    }
}
