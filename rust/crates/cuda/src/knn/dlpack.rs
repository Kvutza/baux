//! Zero-copy DLPack 2D device matrix descriptors (`kDLCUDA`) for `cuVS` FFI.

use std::ffi::c_void;
use std::mem::{offset_of, size_of};
use std::ptr;

use super::CudaResult;

#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
#[allow(non_camel_case_types, dead_code)]
pub enum DLDeviceType {
    kDLCPU = 1,
    kDLCUDA = 2,
    kDLCUDAHost = 3,
    kDLOpenCL = 4,
    kDLVulkan = 7,
    kDLMetal = 8,
    kDLVPI = 9,
    kDLROCM = 10,
    kDLROCMHost = 11,
    kDLExtDev = 12,
    kDLCUDAManaged = 13,
    kDLOneAPI = 14,
    kDLWebGPU = 15,
    kDLHexagon = 16,
}

#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct DLDevice {
    pub device_type: DLDeviceType,
    pub device_id: i32,
}

#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
#[allow(non_camel_case_types, dead_code)]
pub enum DLDataTypeCode {
    kDLInt = 0,
    kDLUInt = 1,
    kDLFloat = 2,
    kDLOpaqueHandle = 3,
    kDLBfloat = 4,
    kDLComplex = 5,
    kDLBool = 6,
}

#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct DLDataType {
    pub code: u8,
    pub bits: u8,
    pub lanes: u16,
}

impl DLDataType {
    pub const F32: Self = Self {
        code: DLDataTypeCode::kDLFloat as u8,
        bits: 32,
        lanes: 1,
    };
    pub const U32: Self = Self {
        code: DLDataTypeCode::kDLUInt as u8,
        bits: 32,
        lanes: 1,
    };
    pub const I64: Self = Self {
        code: DLDataTypeCode::kDLInt as u8,
        bits: 64,
        lanes: 1,
    };
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct DLTensor {
    pub data: *mut c_void,
    pub device: DLDevice,
    pub ndim: i32,
    pub dtype: DLDataType,
    pub shape: *mut i64,
    pub strides: *mut i64,
    pub byte_offset: u64,
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct DLManagedTensor {
    pub dl_tensor: DLTensor,
    pub manager_ctx: *mut c_void,
    pub deleter: Option<unsafe extern "C" fn(self_: *mut DLManagedTensor)>,
}

const _: () = {
    assert!(size_of::<DLDevice>() == 8);
    assert!(size_of::<DLDataType>() == 4);
    assert!(size_of::<DLTensor>() == 48);
    assert!(offset_of!(DLTensor, shape) == 24);
    assert!(offset_of!(DLTensor, byte_offset) == 40);
    assert!(size_of::<DLManagedTensor>() == 64);
};

/// Stack-backed 2D DLPack tensor descriptor for zero-copy FFI calls.
pub struct DlMatrix2D {
    shape: [i64; 2],
    data: *mut c_void,
    device: DLDevice,
    dtype: DLDataType,
}

impl DlMatrix2D {
    pub fn cuda(device_ptr: u64, rows: usize, cols: usize, dtype: DLDataType) -> CudaResult<Self> {
        let rows_i64 = i64::try_from(rows).map_err(|_| "cuVS matrix row count overflows i64")?;
        let cols_i64 = i64::try_from(cols).map_err(|_| "cuVS matrix col count overflows i64")?;
        Ok(Self {
            shape: [rows_i64, cols_i64],
            data: device_ptr as usize as *mut c_void,
            device: DLDevice {
                device_type: DLDeviceType::kDLCUDA,
                device_id: 0,
            },
            dtype,
        })
    }

    pub fn managed(&mut self) -> DLManagedTensor {
        DLManagedTensor {
            dl_tensor: DLTensor {
                data: self.data,
                device: self.device,
                ndim: 2,
                dtype: self.dtype,
                shape: self.shape.as_mut_ptr(),
                strides: ptr::null_mut(),
                byte_offset: 0,
            },
            manager_ctx: ptr::null_mut(),
            deleter: None,
        }
    }
}
