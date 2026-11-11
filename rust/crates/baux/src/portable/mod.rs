//! Native compiled execution shared by Rust, Python and OxCaml.
//!
//! IREE owns device code and allocations. This module owns session lifetime,
//! typed buffer contracts and a serial asynchronous invocation queue.
//! Reference: <https://iree.dev/reference/bindings/c-api/#runtime-api>.
mod exports;
mod interop;
mod pending;
mod raw;
mod tvm;

pub use crate::semantics::compute::Tensor;
pub use pending::Pending;
use std::ffi::{CString, c_void};
use std::sync::{Arc, Mutex};
pub use tvm::{TvmExecutable, TvmFfiVersion, TvmFunction, TvmModule, TvmTensor};

pub type Result<T> = std::result::Result<T, String>;
/// A native device kernel must finish its writes before returning resident
/// outputs.
type NativeKernel = dyn Fn(&Executable, &[Buffer]) -> Result<Vec<Buffer>> + Send + Sync;

const DTYPES: [(&str, usize); 9] = [
    ("bool", 1),
    ("int32", 4),
    ("int64", 8),
    ("uint32", 4),
    ("uint64", 8),
    ("float16", 2),
    ("bfloat16", 2),
    ("float32", 4),
    ("float64", 8),
];

fn layout(spec: &Tensor) -> Result<(u32, usize)> {
    Tensor::new(spec.shape.clone(), &spec.dtype)?;
    let (kind, (_, width)) = DTYPES
        .iter()
        .enumerate()
        .find(|(_, (name, _))| *name == spec.dtype)
        .ok_or("unsupported dtype")?;
    let bytes = spec
        .shape
        .iter()
        .try_fold(*width, |size, n| size.checked_mul(*n))
        .ok_or("tensor byte length overflows")?;
    Ok((kind as u32, bytes))
}

struct Session(Mutex<raw::Context>);
struct Storage {
    owner: Arc<Session>,
    pointer: *mut c_void,
    spec: Tensor,
    leases: Vec<Arc<dyn std::any::Any + Send + Sync>>,
}
// Buffer handles are only accessed under their Session mutex. Metadata is
// immutable, and Arc keeps both the buffer and session live during every use.
unsafe impl Send for Storage {}
unsafe impl Sync for Storage {}
impl Drop for Storage {
    fn drop(&mut self) {
        let _guard = self.owner.0.lock().unwrap_or_else(|e| e.into_inner());
        // Exactly one Storage owns this retained IREE reference.
        unsafe { raw::bauxiree_release(self.pointer) };
    }
}

/// Immutable resident tensor; clones share the allocation and retain its owner.
#[derive(Clone)]
pub struct Buffer(Arc<Storage>);
impl Buffer {
    pub fn spec(&self) -> &Tensor {
        &self.0.spec
    }

    pub fn read(&self) -> Result<Vec<u8>> {
        let (_, length) = layout(self.spec())?;
        let mut bytes = vec![0; length];
        self.read_into(&mut bytes)?;
        Ok(bytes)
    }

    /// Copy into caller-owned storage without an intermediate host allocation.
    pub fn read_into(&self, bytes: &mut [u8]) -> Result<()> {
        let (_, length) = layout(self.spec())?;
        if bytes.len() != length {
            return Err("readback byte length differs from tensor signature".into());
        }
        let context = self
            .0
            .owner
            .0
            .lock()
            .map_err(|_| "session mutex poisoned")?;
        let mut error = [0; 2048];
        // The buffer retains this session. The destination is writable for
        // length.
        let code = unsafe {
            raw::bauxiree_read(
                context.0,
                self.0.pointer,
                bytes.as_mut_ptr(),
                length,
                error.as_mut_ptr(),
                error.len(),
            )
        };
        raw::status(code, &error)
    }
}

/// A loaded VMFB function. State and RNG are ordinary tensor inputs/outputs.
#[derive(Clone)]
pub struct Executable {
    owner: Arc<Session>,
    inputs: Arc<[Tensor]>,
    outputs: Arc<[Tensor]>,
    queue: pending::Queue,
    kernel: Option<Arc<NativeKernel>>,
}
impl Executable {
    pub fn load(
        bytecode: &[u8],
        entry: &str,
        device: &str,
        inputs: Vec<Tensor>,
        outputs: Vec<Tensor>,
    ) -> Result<Self> {
        for spec in inputs.iter().chain(&outputs) {
            layout(spec)?;
        }
        let driver = match device {
            "cpu" => "local-task",
            "metal" => "metal",
            _ => return Err(format!("unsupported native device {device:?}")),
        };
        let driver = CString::new(driver).map_err(|e| e.to_string())?;
        let entry = CString::new(entry).map_err(|e| e.to_string())?;
        let mut context = std::ptr::null_mut();
        let mut error = [0; 2048];
        // C copies bytecode; no borrowed input pointer survives this call.
        let code = unsafe {
            raw::bauxiree_load(
                bytecode.as_ptr(),
                bytecode.len(),
                driver.as_ptr(),
                entry.as_ptr(),
                &mut context,
                error.as_mut_ptr(),
                error.len(),
            )
        };
        raw::status(code, &error)?;
        let owner = Arc::new(Session(Mutex::new(raw::Context(context))));
        Ok(Self {
            owner,
            inputs: inputs.into(),
            outputs: outputs.into(),
            queue: pending::Queue::new()?,
            kernel: None,
        })
    }

    pub fn put(&self, spec: Tensor, bytes: &[u8]) -> Result<Buffer> {
        let (dtype, length) = layout(&spec)?;
        if bytes.len() != length {
            return Err("upload byte length differs from tensor signature".into());
        }
        let shape: Vec<usize> = spec.shape.clone();
        let context = self.owner.0.lock().map_err(|_| "session mutex poisoned")?;
        let mut pointer = std::ptr::null_mut();
        let mut error = [0; 2048];
        // C copies the supplied bytes into a device-owned allocation.
        let code = unsafe {
            raw::bauxiree_put(
                context.0,
                shape.as_ptr(),
                shape.len(),
                dtype,
                bytes.as_ptr(),
                bytes.len(),
                &mut pointer,
                error.as_mut_ptr(),
                error.len(),
            )
        };
        raw::status(code, &error)?;
        drop(context);
        Ok(Buffer(Arc::new(Storage {
            owner: self.owner.clone(),
            pointer,
            spec,
            leases: Vec::new(),
        })))
    }

    fn validate(&self, inputs: &[Buffer]) -> Result<()> {
        if inputs.len() != self.inputs.len() {
            return Err("input count differs from program signature".into());
        }
        for (input, spec) in inputs.iter().zip(self.inputs.iter()) {
            if !Arc::ptr_eq(&self.owner, &input.0.owner) {
                return Err("buffer belongs to another session".into());
            }
            if input.spec() != spec {
                return Err("input tensor differs from program signature".into());
            }
        }
        Ok(())
    }

    pub fn run(&self, inputs: &[Buffer]) -> Result<Vec<Buffer>> {
        self.validate(inputs)?;
        if let Some(kernel) = &self.kernel {
            let outputs = kernel(self, inputs)?;
            self.check_outputs(&outputs)?;
            return Ok(outputs);
        }
        let pointers: Vec<_> = inputs.iter().map(|v| v.0.pointer).collect();
        let mut outputs = vec![std::ptr::null_mut(); self.outputs.len()];
        let context = self.owner.0.lock().map_err(|_| "session mutex poisoned")?;
        let mut error = [0; 2048];
        // Inputs remain retained and the bridge returns owned result
        // references.
        let code = unsafe {
            raw::bauxiree_call(
                context.0,
                pointers.as_ptr(),
                pointers.len(),
                outputs.as_mut_ptr(),
                outputs.len(),
                error.as_mut_ptr(),
                error.len(),
            )
        };
        raw::status(code, &error)?;
        let valid = outputs
            .iter()
            .zip(self.outputs.iter())
            .all(|(&pointer, spec)| {
                let shape: Vec<usize> = spec.shape.clone();
                let dtype = layout(spec).expect("validated signature").0;
                // Output references are live and Session is locked.
                unsafe { raw::bauxiree_matches(pointer, shape.as_ptr(), shape.len(), dtype) != 0 }
            });
        if !valid {
            for pointer in outputs {
                unsafe { raw::bauxiree_release(pointer) };
            }
            return Err("compiled output differs from program signature".into());
        }
        drop(context);
        let mut outputs: Vec<Buffer> = outputs
            .into_iter()
            .zip(self.outputs.iter())
            .map(|(pointer, spec)| {
                Buffer(Arc::new(Storage {
                    owner: self.owner.clone(),
                    pointer,
                    spec: spec.clone(),
                    leases: Vec::new(),
                }))
            })
            .collect();
        self.inherit(inputs, &mut outputs);
        Ok(outputs)
    }

    /// Submit without blocking the caller. Completion contains resident
    /// tensors.
    pub fn submit(&self, inputs: Vec<Buffer>) -> Result<Pending> {
        self.validate(&inputs)?;
        let executable = self.clone();
        self.queue.submit(move || executable.run(&inputs))
    }
}
