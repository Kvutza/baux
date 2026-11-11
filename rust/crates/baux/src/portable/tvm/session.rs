//! Compiled TVM programs behind the shared resident-buffer ABI.
//! Buffers own TVM tensors directly; invocation performs no host staging.
use super::{TvmExecutable, TvmTensor};
use crate::portable::{DTYPES, Executable, Result, Session, Tensor, pending, raw};
use std::ffi::{c_char, c_int, c_void};
use std::sync::{Arc, Mutex};

impl Executable {
    /// Compile verified StableHLO to CPU TIR/Relax with resident TVM buffers.
    ///
    /// # Parameters
    /// - `source`: Static StableHLO module text; parameters remain explicit.
    /// - `entry`: Exported tensor function to invoke.
    /// - `importer`: Bundled `hlo_import` executable. MLIR runs in a child
    ///   process to isolate its LLVM from TVM's compiler.
    /// - `inputs`, `outputs`: Ordered shapes and dtypes, checked against the
    ///   imported function before returning an executable.
    ///
    /// # Returns
    /// An owned CPU session using the shared `put`/`run`/`submit` buffer API.
    /// Compilation blocks and uploads no user inputs. Buffers and pending
    /// calls retain the session; invocation does not stage tensors through
    /// host copies. Output buffers are immutable to callers.
    ///
    /// # Errors
    /// Returns an error for importer failures, unsupported operations or
    /// tensor types, signature mismatches, and native compiler/VM failures.
    /// No alternate compiler or device is selected on failure.
    ///
    /// See [`TvmExecutable::compile`] for the lower-level TVM tensor API.
    pub fn compile_tvm(
        source: &str,
        entry: &str,
        importer: &std::path::Path,
        inputs: Vec<Tensor>,
        outputs: Vec<Tensor>,
    ) -> Result<Self> {
        let program = TvmExecutable::compile(source, entry, importer)?;
        if program.inputs() != inputs || program.outputs() != outputs {
            return Err("TVM source differs from declared program signature".into());
        }
        let queue = pending::Queue::new()?;
        Ok(Self {
            owner: Arc::new(Session(Mutex::new(raw::Context(
                Box::into_raw(Box::new(program)).cast(),
                &API,
            )))),
            inputs: inputs.into(),
            outputs: outputs.into(),
            queue,
            kernel: None,
        })
    }
}

fn guard(error: *mut c_char, capacity: usize, f: impl FnOnce() -> Result<()>) -> c_int {
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(f));
    let message = match result {
        Ok(Ok(())) => return 0,
        Ok(Err(message)) => message,
        Err(_) => "native TVM session panicked".into(),
    };
    if !error.is_null() && capacity > 0 {
        let count = message.len().min(capacity - 1);
        // The private ABI supplies capacity writable bytes, including the NUL.
        unsafe {
            std::ptr::copy_nonoverlapping(message.as_ptr(), error.cast(), count);
            *error.add(count) = 0;
        }
    }
    1
}

unsafe fn slice<'a, T>(pointer: *const T, count: usize) -> Result<&'a [T]> {
    if count == 0 {
        return Ok(&[]);
    }
    if pointer.is_null() {
        return Err("null TVM session range".into());
    }
    // Private ABI caller guarantees alignment and count live elements.
    Ok(unsafe { std::slice::from_raw_parts(pointer, count) })
}

unsafe extern "C" fn close(pointer: *mut c_void) {
    // Context uniquely owns the Box; its buffers retain Context until released.
    unsafe {
        drop(Box::from_raw(pointer.cast::<TvmExecutable>()));
    }
}
unsafe extern "C" fn retain(pointer: *mut c_void) {
    // Every non-null handle originates from Arc::into_raw below.
    unsafe {
        Arc::increment_strong_count(pointer.cast::<TvmTensor>());
    }
}
unsafe extern "C" fn release(pointer: *mut c_void) {
    // Releases exactly one owning reference, serialized by Session.
    unsafe {
        Arc::decrement_strong_count(pointer.cast::<TvmTensor>());
    }
}

unsafe extern "C" fn put(
    _context: *mut c_void,
    shape: *const usize,
    rank: usize,
    dtype: u32,
    data: *const u8,
    length: usize,
    output: *mut *mut c_void,
    error: *mut c_char,
    capacity: usize,
) -> c_int {
    guard(error, capacity, || {
        let dtype = DTYPES.get(dtype as usize).ok_or("invalid TVM dtype")?.0;
        // Source ranges live for this synchronous upload; output is writable.
        let spec = Tensor::new(unsafe { slice(shape, rank)? }.to_vec(), dtype)?;
        let tensor = TvmTensor::put(spec, unsafe { slice(data, length)? })?;
        unsafe {
            *output = Arc::into_raw(Arc::new(tensor)).cast_mut().cast();
        }
        Ok(())
    })
}

unsafe extern "C" fn read(
    _context: *mut c_void,
    pointer: *mut c_void,
    output: *mut u8,
    length: usize,
    error: *mut c_char,
    capacity: usize,
) -> c_int {
    guard(error, capacity, || {
        // Buffer retains a live tensor and caller supplies length writable
        // bytes.
        let tensor = unsafe { &*pointer.cast::<TvmTensor>() };
        let bytes = if length == 0 {
            &mut []
        } else {
            unsafe { std::slice::from_raw_parts_mut(output, length) }
        };
        tensor.read_into(bytes)
    })
}

unsafe extern "C" fn call(
    context: *mut c_void,
    inputs: *const *mut c_void,
    count: usize,
    outputs: *mut *mut c_void,
    output_count: usize,
    error: *mut c_char,
    capacity: usize,
) -> c_int {
    guard(error, capacity, || {
        // Session owns the program and all input handles through completion.
        let program = unsafe { &*context.cast::<TvmExecutable>() };
        if output_count != program.outputs().len() {
            return Err("TVM output count differs from compiled signature".into());
        }
        let inputs = unsafe { slice(inputs, count)? }
            .iter()
            .map(|&pointer| unsafe { &*pointer.cast::<TvmTensor>() })
            .collect::<Vec<_>>();
        let results = program.run(&inputs)?;
        // No output is published before the complete native call succeeds.
        for (i, tensor) in results.into_iter().enumerate() {
            unsafe {
                *outputs.add(i) = Arc::into_raw(Arc::new(tensor)).cast_mut().cast();
            }
        }
        Ok(())
    })
}

unsafe extern "C" fn matches(
    pointer: *mut c_void,
    shape: *const usize,
    rank: usize,
    dtype: u32,
) -> c_int {
    // Called only with a retained result and a checked metadata range.
    let tensor = unsafe { &*pointer.cast::<TvmTensor>() };
    let Ok(shape) = (unsafe { slice(shape, rank) }) else {
        return 0;
    };
    i32::from(
        tensor.spec().shape == shape
            && DTYPES
                .get(dtype as usize)
                .is_some_and(|(name, _)| *name == tensor.spec().dtype),
    )
}

static API: raw::Api = raw::Api {
    metal: false,
    close,
    retain,
    release,
    put,
    read,
    call,
    matches,
};
