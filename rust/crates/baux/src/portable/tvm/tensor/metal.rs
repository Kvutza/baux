//! DLPack retains the producer's allocation and mutation lease until TVM drops
//! its final tensor alias. No host tensor copy occurs at this seam.
use super::*;
use std::any::Any;
use std::sync::Arc;

struct External {
    managed: Managed,
    shape: Vec<i64>,
    _lease: Arc<dyn Any + Send + Sync>,
}

fn import_tensor(
    source: *mut Managed,
    alignment: i32,
    contiguous: i32,
    output: *mut *mut c_void,
) -> Result<i32> {
    let f = super::super::ffi()?;
    Ok(unsafe { (f.import_tensor)(source.cast(), alignment, contiguous, output) })
}

unsafe extern "C" fn release(tensor: *mut Managed) {
    let context = unsafe { (*tensor).context.cast::<External>() };
    drop(unsafe { Box::from_raw(context) });
}

impl TvmTensor {
    /// Borrow a synchronized Metal allocation through an owned DLPack lease.
    ///
    /// # Safety
    /// The handle must be an MTLBuffer on Metal device zero with at least
    /// offset + tensor byte length bytes. The lease must retain that buffer
    /// and prevent all writes until TVM releases every alias. Producer GPU
    /// writes must have finished before import.
    pub unsafe fn import_metal(
        spec: Tensor,
        handle: *mut c_void,
        offset: u64,
        lease: Arc<dyn Any + Send + Sync>,
    ) -> Result<Self> {
        layout(&spec)?;
        if handle.is_null() {
            return Err("null Metal allocation".into());
        }
        let (code, bits) = match spec.dtype.as_str() {
            "bool" => (6, 8),
            "int32" => (0, 32),
            "int64" => (0, 64),
            "uint32" => (1, 32),
            "uint64" => (1, 64),
            "float16" => (2, 16),
            "bfloat16" => (4, 16),
            "float32" => (2, 32),
            "float64" => (2, 64),
            _ => return Err("unsupported Metal tensor dtype".into()),
        };
        let shape = spec
            .shape
            .iter()
            .map(|&n| i64::try_from(n).map_err(|e| e.to_string()))
            .collect::<Result<Vec<_>>>()?;
        let mut owner = Box::new(External {
            managed: Managed {
                major: 1,
                minor: 0,
                context: std::ptr::null_mut(),
                deleter: Some(release),
                flags: 0,
                tensor: DlTensor {
                    data: handle,
                    device: Device { kind: 8, index: 0 },
                    dimensions: i32::try_from(shape.len()).map_err(|e| e.to_string())?,
                    dtype: Dtype {
                        code,
                        bits,
                        lanes: 1,
                    },
                    shape: std::ptr::null(),
                    strides: std::ptr::null(),
                    offset,
                },
            },
            shape,
            _lease: lease,
        });
        owner.managed.tensor.shape = owner.shape.as_ptr();
        let pointer = &raw mut *owner;
        owner.managed.context = pointer.cast();
        let mut output = std::ptr::null_mut();
        let code = import_tensor(&raw mut owner.managed, 0, 1, &raw mut output)?;
        super::super::status(code, "import Metal tensor")?;
        // Successful import transfers the managed-tensor deleter to TVM.
        let _ = Box::into_raw(owner);
        Self::from_value(RawAny {
            type_index: 70,
            padding: 0,
            value: RawValue { pointer: output },
        })
    }

    /// Borrow the MTLBuffer handle and byte offset without a transfer.
    ///
    /// # Safety
    /// Do not release or mutate the handle, or let GPU work using it outlive
    /// this tensor. Producer completion is the caller's responsibility.
    pub unsafe fn metal(&self) -> Result<(*mut c_void, u64)> {
        let mut pointer = std::ptr::null_mut();
        super::super::status(
            export_tensor(unsafe { self.value.value.pointer }, &raw mut pointer)?,
            "borrow Metal tensor",
        )?;
        let lease = Lease(NonNull::new(pointer).ok_or("null DLPack lease")?);
        let tensor = &unsafe { lease.0.as_ref() }.tensor;
        if tensor.device.kind != 8 || tensor.device.index != 0 {
            return Err("tensor is not on Metal device zero".into());
        }
        Ok((tensor.data, tensor.offset))
    }
}
