//! Native TVM tensor allocation and synchronous transfers through its registry.
//! Reference: Apache TVM 0.27 runtime/tensor.cc, TVMTensorAllocWithScope.
use super::{RawAny, RawValue, TvmFunction};
use crate::portable::{Result, Tensor, layout};
use std::ffi::c_void;
use std::ptr::NonNull;
mod metal;

#[repr(C)]
#[derive(Clone, Copy)]
pub(super) struct Dtype {
    code: u8,
    bits: u8,
    lanes: u16,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub(super) struct Device {
    kind: i32,
    index: i32,
}

#[repr(C)]
struct DlTensor {
    data: *mut c_void,
    device: Device,
    dimensions: i32,
    dtype: Dtype,
    shape: *const i64,
    strides: *const i64,
    offset: u64,
}

#[repr(C)]
struct Managed {
    major: u32,
    minor: u32,
    context: *mut c_void,
    deleter: Option<unsafe extern "C" fn(*mut Managed)>,
    flags: u64,
    tensor: DlTensor,
}

struct Lease(NonNull<Managed>);
impl Drop for Lease {
    fn drop(&mut self) {
        // TVM supplied this owned DLPack lease and its matching deleter.
        if let Some(deleter) = unsafe { self.0.as_ref().deleter } {
            unsafe { deleter(self.0.as_ptr()) };
        }
    }
}

unsafe extern "C" {
    #[link_name = "TVMFFITensorToDLPackVersioned"]
    fn export_tensor(tensor: *mut c_void, output: *mut *mut Managed) -> i32;
}

fn invoke(name: &str, args: &mut [RawAny]) -> Result<RawAny> {
    TvmFunction::global(name)?
        .ok_or_else(|| format!("missing TVM function {name}"))?
        .call(args)
}

/// An owned native TVM allocation. Transfers finish before returning.
#[derive(Clone)]
pub struct TvmTensor {
    pub(super) value: RawAny,
    pub(super) device: i32,
    spec: Tensor,
}

// TVM uses atomic refcounts. Public tensor methods only read; uploads finish
// before construction and outputs are published after native completion.
unsafe impl Send for TvmTensor {}
unsafe impl Sync for TvmTensor {}

impl TvmTensor {
    pub(super) fn from_value(value: RawAny) -> Result<Self> {
        if value.type_index != 70 {
            return Err("TVM result is not a tensor".into());
        }
        let mut output = std::ptr::null_mut();
        // value owns a live Tensor; the native API creates an independent
        // lease.
        super::status(
            unsafe { export_tensor(value.value.pointer, &raw mut output) },
            "export tensor",
        )?;
        let lease = Lease(NonNull::new(output).ok_or("TVM returned a null DLPack lease")?);
        let managed = unsafe { lease.0.as_ref() };
        if managed.major != 1 {
            return Err("unsupported DLPack major version".into());
        }
        let tensor = &managed.tensor;
        if ![1, 8].contains(&tensor.device.kind) || tensor.device.index != 0 {
            return Err("TVM tensor device must be CPU or Metal device zero".into());
        }
        let rank = usize::try_from(tensor.dimensions).map_err(|e| e.to_string())?;
        let dimensions = if rank == 0 {
            &[][..]
        } else {
            if tensor.shape.is_null() {
                return Err("TVM returned null tensor dimensions".into());
            }
            // DLPack guarantees rank valid dimension entries during the lease.
            unsafe { std::slice::from_raw_parts(tensor.shape, rank) }
        };
        let shape = dimensions
            .iter()
            .map(|&n| usize::try_from(n).map_err(|e| e.to_string()))
            .collect::<Result<Vec<_>>>()?;
        let dtype = match (tensor.dtype.code, tensor.dtype.bits, tensor.dtype.lanes) {
            (6, 8, 1) => "bool",
            (0, 32, 1) => "int32",
            (0, 64, 1) => "int64",
            (1, 32, 1) => "uint32",
            (1, 64, 1) => "uint64",
            (2, 16, 1) => "float16",
            (4, 16, 1) => "bfloat16",
            (2, 32, 1) => "float32",
            (2, 64, 1) => "float64",
            _ => return Err("unsupported TVM result dtype".into()),
        };
        let spec = Tensor::new(shape, dtype)?;
        layout(&spec)?;
        Ok(Self {
            value,
            spec,
            device: tensor.device.kind,
        })
    }
    /// Allocate and upload a CPU tensor, validating byte size before FFI.
    pub fn put(spec: Tensor, bytes: &[u8]) -> Result<Self> {
        let (_, length) = layout(&spec)?;
        if bytes.len() != length {
            return Err("TVM upload byte length differs from signature".into());
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
            _ => return Err("unsupported TVM tensor dtype".into()),
        };
        let mut dimensions = spec
            .shape
            .iter()
            .map(|&size| {
                i64::try_from(size)
                    .map(RawAny::integer)
                    .map_err(|e| e.to_string())
            })
            .collect::<Result<Vec<_>>>()?;
        let shape = invoke("ffi.Shape", &mut dimensions)?;
        let dtype = RawAny {
            type_index: 5,
            padding: 0,
            value: RawValue {
                dtype: Dtype {
                    code,
                    bits,
                    lanes: 1,
                },
            },
        };
        let device = RawAny {
            type_index: 6,
            padding: 0,
            value: RawValue {
                device: Device { kind: 1, index: 0 },
            },
        };
        let scope = RawAny {
            type_index: 0,
            padding: 0,
            value: RawValue { integer: 0 },
        };
        let value = invoke(
            "runtime.TVMTensorAllocWithScope",
            &mut [shape, dtype, device, scope],
        )?;
        if value.type_index != 70 {
            return Err("TVM allocator returned a non-tensor".into());
        }
        let tensor = Self {
            value,
            spec,
            device: 1,
        };
        tensor.transfer(
            "runtime.TVMTensorCopyFromBytes",
            bytes.as_ptr().cast_mut(),
            length,
        )?;
        Ok(tensor)
    }

    fn transfer(&self, name: &str, pointer: *mut u8, length: usize) -> Result<()> {
        let pointer = RawAny {
            type_index: 4,
            padding: 0,
            value: RawValue {
                pointer: pointer.cast(),
            },
        };
        let size = RawAny::integer(i64::try_from(length).map_err(|e| e.to_string())?);
        invoke(name, &mut [self.value.clone(), pointer, size])?;
        Ok(())
    }

    /// Copy the allocation to owned host bytes.
    pub fn read(&self) -> Result<Vec<u8>> {
        let (_, length) = layout(&self.spec)?;
        let mut bytes = vec![0; length];
        self.transfer("runtime.TVMTensorCopyToBytes", bytes.as_mut_ptr(), length)?;
        Ok(bytes)
    }

    /// Tensor signature fixed at allocation.
    pub fn spec(&self) -> &Tensor {
        &self.spec
    }
}

impl TvmFunction<'_> {
    /// Execute a native packed function on resident CPU tensors.
    /// Returned tensors retain their allocations independently of the inputs.
    pub fn run(&self, inputs: &[&TvmTensor]) -> Result<Vec<TvmTensor>> {
        let mut arguments = inputs
            .iter()
            .map(|tensor| tensor.value.clone())
            .collect::<Vec<_>>();
        let result = self.call(&mut arguments)?;
        if result.type_index == 70 {
            return Ok(vec![TvmTensor::from_value(result)?]);
        }
        if result.type_index != 71 {
            return Err("TVM function must return a tensor or tensor array".into());
        }
        let size = invoke("ffi.ArraySize", &mut [result.clone()])?;
        if size.type_index != 1 {
            return Err("TVM array size is not an integer".into());
        }
        let length = usize::try_from(unsafe { size.value.integer }).map_err(|e| e.to_string())?;
        let mut outputs = Vec::new();
        outputs.try_reserve(length).map_err(|e| e.to_string())?;
        for index in 0..length {
            let value = invoke(
                "ffi.ArrayGetItem",
                &mut [result.clone(), RawAny::integer(index as i64)],
            )?;
            outputs.push(TvmTensor::from_value(value)?);
        }
        Ok(outputs)
    }
}
