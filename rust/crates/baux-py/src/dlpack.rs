use std::ffi::{c_char, c_void};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use pyo3::exceptions::{PyBufferError, PyTypeError, PyValueError};
use pyo3::ffi;
use pyo3::prelude::*;
use pyo3::types::PyDict;

type PyObject = Py<PyAny>;

const CUDA: i32 = 2;
const FLOAT: u8 = 2;
const BF16: u8 = 4;
const READ_ONLY: u64 = 1;
const LEGACY: &[u8] = b"dltensor\0";
const LEGACY_USED: &[u8] = b"used_dltensor\0";
const VERSIONED: &[u8] = b"dltensor_versioned\0";
const VERSIONED_USED: &[u8] = b"used_dltensor_versioned\0";

#[repr(C)]
#[derive(Clone, Copy)]
struct Version {
    major: u32,
    minor: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct Device {
    kind: i32,
    index: i32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct DataType {
    code: u8,
    bits: u8,
    lanes: u16,
}

#[repr(C)]
struct Tensor {
    data: *mut c_void,
    device: Device,
    ndim: i32,
    dtype: DataType,
    shape: *mut i64,
    strides: *mut i64,
    byte_offset: u64,
}

#[repr(C)]
struct Managed {
    tensor: Tensor,
    context: *mut c_void,
    deleter: Option<unsafe extern "C" fn(*mut Managed)>,
}

#[repr(C)]
struct ManagedV1 {
    version: Version,
    context: *mut c_void,
    deleter: Option<unsafe extern "C" fn(*mut ManagedV1)>,
    flags: u64,
    tensor: Tensor,
}

const _: [(); 48] = [(); std::mem::size_of::<Tensor>()];
const _: [(); 64] = [(); std::mem::size_of::<Managed>()];
const _: [(); 80] = [(); std::mem::size_of::<ManagedV1>()];

pub(crate) struct Input {
    capsule: PyObject,
    legacy: *mut Managed,
    versioned: *mut ManagedV1,
    pub(crate) pointer: u64,
    pub(crate) len: usize,
    pub(crate) offset: usize,
    pub(crate) shape: Vec<usize>,
}

fn open_capsule<'py>(
    source: &Bound<'py, PyAny>,
    kind_expected: i32,
) -> PyResult<Bound<'py, PyAny>> {
    let stream = (kind_expected == CUDA).then_some(1_i64);
    let kwargs = PyDict::new(source.py());
    kwargs.set_item("stream", stream)?;
    kwargs.set_item("max_version", (1_u32, 3_u32))?;
    match source.call_method("__dlpack__", (), Some(&kwargs)) {
        Ok(capsule) => Ok(capsule),
        Err(error) if error.is_instance_of::<PyTypeError>(source.py()) => {
            let kwargs = PyDict::new(source.py());
            kwargs.set_item("stream", stream)?;
            source.call_method("__dlpack__", (), Some(&kwargs))
        }
        Err(error) => Err(error),
    }
}

impl Input {
    pub(crate) fn new(source: &Bound<'_, PyAny>) -> PyResult<Self> {
        Self::read(source, BF16, 16, "BF16", CUDA)
    }

    pub(crate) fn f32(source: &Bound<'_, PyAny>) -> PyResult<Self> {
        Self::read(source, FLOAT, 32, "FP32", CUDA)
    }

    pub(crate) fn posterior(source: &Bound<'_, PyAny>, device: i32) -> PyResult<Self> {
        Self::read(source, FLOAT, 32, "FP32", device)
    }

    fn read(
        source: &Bound<'_, PyAny>,
        code: u8,
        bits: u8,
        name: &str,
        kind_expected: i32,
    ) -> PyResult<Self> {
        let device: (i32, i32) = source.call_method0("__dlpack_device__")?.extract()?;
        if device != (kind_expected, 0) || ![CUDA, 8].contains(&kind_expected) {
            return Err(PyValueError::new_err(format!(
                "{name} input requires DLPack device ({kind_expected}, 0)"
            )));
        }
        let capsule = open_capsule(source, kind_expected)?;
        let (legacy, versioned, tensor) = unsafe {
            if ffi::PyCapsule_IsValid(capsule.as_ptr(), LEGACY.as_ptr().cast::<c_char>()) != 0 {
                let managed =
                    ffi::PyCapsule_GetPointer(capsule.as_ptr(), LEGACY.as_ptr().cast::<c_char>())
                        .cast::<Managed>();
                (managed, std::ptr::null_mut(), &(*managed).tensor)
            } else if ffi::PyCapsule_IsValid(capsule.as_ptr(), VERSIONED.as_ptr().cast::<c_char>())
                != 0
            {
                let managed = ffi::PyCapsule_GetPointer(
                    capsule.as_ptr(),
                    VERSIONED.as_ptr().cast::<c_char>(),
                )
                .cast::<ManagedV1>();
                if (*managed).version.major != 1 || (*managed).version.minor > 3 {
                    return Err(PyBufferError::new_err("unsupported DLPack major version"));
                }
                (std::ptr::null_mut(), managed, &(*managed).tensor)
            } else {
                return Err(PyBufferError::new_err("invalid DLPack capsule"));
            }
        };
        validate_tensor(tensor, code, bits, name, kind_expected)?;
        let dimensions = usize::try_from(tensor.ndim)
            .map_err(|_| PyValueError::new_err("invalid DLPack rank"))?;
        let shape = unsafe { std::slice::from_raw_parts(tensor.shape, dimensions) };
        if !is_contiguous(tensor, shape) {
            return Err(PyBufferError::new_err(format!(
                "{name} input requires a contiguous DLPack tensor"
            )));
        }
        let len = shape.iter().try_fold(1usize, |size, &dimension| {
            let dimension = usize::try_from(dimension)
                .map_err(|_| PyValueError::new_err("invalid DLPack shape"))?;
            size.checked_mul(dimension)
                .ok_or_else(|| PyValueError::new_err("DLPack shape overflow"))
        })?;
        if len == 0 {
            return Err(PyValueError::new_err(format!(
                "{name} input requires a non-empty tensor"
            )));
        }
        let pointer = (tensor.data as u64)
            .checked_add(if kind_expected == CUDA {
                tensor.byte_offset
            } else {
                0
            })
            .ok_or_else(|| PyValueError::new_err("DLPack pointer overflow"))?;
        Ok(Self {
            capsule: capsule.unbind(),
            legacy,
            versioned,
            pointer,
            len,
            offset: if kind_expected == CUDA {
                0
            } else {
                tensor.byte_offset as usize
            },
            shape: shape.iter().map(|&v| v as usize).collect(),
        })
    }
}

impl Drop for Input {
    fn drop(&mut self) {
        Python::attach(|py| unsafe {
            let capsule = self.capsule.bind(py);
            let result = if !self.legacy.is_null() {
                let result =
                    ffi::PyCapsule_SetName(capsule.as_ptr(), LEGACY_USED.as_ptr().cast::<c_char>());
                if result == 0 {
                    if let Some(deleter) = (*self.legacy).deleter {
                        deleter(self.legacy);
                    }
                }
                result
            } else {
                let result = ffi::PyCapsule_SetName(
                    capsule.as_ptr(),
                    VERSIONED_USED.as_ptr().cast::<c_char>(),
                );
                if result == 0 {
                    if let Some(deleter) = (*self.versioned).deleter {
                        deleter(self.versioned);
                    }
                }
                result
            };
            if result != 0 {
                ffi::PyErr_Clear();
            }
        });
    }
}

type Lease = (Option<Arc<AtomicBool>>, Option<Arc<AtomicUsize>>);

struct LegacyOwner {
    managed: Managed,
    shape: [i64; 2],
    _owner: PyObject,
    lease: Lease,
}

struct VersionOwner {
    managed: ManagedV1,
    shape: [i64; 2],
    strides: [i64; 2],
    _owner: PyObject,
    lease: Lease,
}

#[derive(Clone, Copy)]
struct ExportShape {
    values: [i64; 2],
    strides: [i64; 2],
    rank: i32,
    device: i32,
    bits: u8,
    code: u8,
}

impl ExportShape {
    fn vector(len: usize) -> PyResult<Self> {
        let len =
            i64::try_from(len).map_err(|_| PyValueError::new_err("BF16 length exceeds i64"))?;
        Ok(Self {
            values: [len, 1],
            strides: [1, 1],
            rank: 1,
            device: CUDA,
            bits: 16,
            code: BF16,
        })
    }

    fn matrix(rows: usize, columns: usize) -> PyResult<Self> {
        let rows =
            i64::try_from(rows).map_err(|_| PyValueError::new_err("BF16 row count exceeds i64"))?;
        let columns = i64::try_from(columns)
            .map_err(|_| PyValueError::new_err("BF16 column count exceeds i64"))?;
        Ok(Self {
            values: [rows, columns],
            strides: [columns, 1],
            rank: 2,
            device: CUDA,
            bits: 16,
            code: BF16,
        })
    }
}

fn release_lease(lease: &Lease) {
    if let Some(flag) = &lease.0 {
        flag.store(false, Ordering::Release);
    }
    if let Some(count) = &lease.1 {
        count.fetch_sub(1, Ordering::AcqRel);
    }
}

unsafe extern "C" fn release_legacy(managed: *mut Managed) {
    if managed.is_null() {
        return;
    }
    let context = unsafe { (*managed).context.cast::<LegacyOwner>() };
    if !context.is_null() {
        let owner = unsafe { Box::from_raw(context) };
        release_lease(&owner.lease);
        // Py<T> queues decrefs off-GIL; CUDA cleanup must not wait for the GIL.
        drop(owner);
    }
}

unsafe extern "C" fn release_versioned(managed: *mut ManagedV1) {
    if managed.is_null() {
        return;
    }
    let context = unsafe { (*managed).context.cast::<VersionOwner>() };
    if !context.is_null() {
        let owner = unsafe { Box::from_raw(context) };
        release_lease(&owner.lease);
        drop(owner);
    }
}

unsafe extern "C" fn drop_legacy(capsule: *mut ffi::PyObject) {
    if unsafe { ffi::PyCapsule_IsValid(capsule, LEGACY.as_ptr().cast::<c_char>()) } == 0 {
        return;
    }
    let managed = unsafe {
        ffi::PyCapsule_GetPointer(capsule, LEGACY.as_ptr().cast::<c_char>()).cast::<Managed>()
    };
    if !managed.is_null() {
        if let Some(deleter) = unsafe { (*managed).deleter } {
            unsafe { deleter(managed) };
        }
    }
}

unsafe extern "C" fn drop_versioned(capsule: *mut ffi::PyObject) {
    if unsafe { ffi::PyCapsule_IsValid(capsule, VERSIONED.as_ptr().cast::<c_char>()) } == 0 {
        return;
    }
    let managed = unsafe {
        ffi::PyCapsule_GetPointer(capsule, VERSIONED.as_ptr().cast::<c_char>()).cast::<ManagedV1>()
    };
    if !managed.is_null() {
        if let Some(deleter) = unsafe { (*managed).deleter } {
            unsafe { deleter(managed) };
        }
    }
}

pub(crate) fn export(
    py: Python<'_>,
    owner: PyObject,
    lease: Arc<AtomicBool>,
    pointer: u64,
    len: usize,
    max_version: Option<(u32, u32)>,
) -> PyResult<PyObject> {
    export_with(
        py,
        owner,
        (Some(lease), None),
        pointer,
        ExportShape::vector(len)?,
        max_version,
    )
}

pub(crate) fn export_count(
    py: Python<'_>,
    owner: PyObject,
    lease: Arc<AtomicUsize>,
    pointer: u64,
    len: usize,
    max_version: Option<(u32, u32)>,
) -> PyResult<PyObject> {
    export_with(
        py,
        owner,
        (None, Some(lease)),
        pointer,
        ExportShape::vector(len)?,
        max_version,
    )
}

pub(crate) fn export_batch(
    py: Python<'_>,
    owner: PyObject,
    lease: Arc<AtomicUsize>,
    pointer: u64,
    rows: usize,
    columns: usize,
    max_version: Option<(u32, u32)>,
) -> PyResult<PyObject> {
    export_with(
        py,
        owner,
        (None, Some(lease)),
        pointer,
        ExportShape::matrix(rows, columns)?,
        max_version,
    )
}

pub(crate) fn export_array(
    py: Python<'_>,
    owner: PyObject,
    pointer: u64,
    rows: usize,
    columns: usize,
    device: i32,
    bits: u8,
    max_version: Option<(u32, u32)>,
) -> PyResult<PyObject> {
    let mut shape = ExportShape::matrix(rows, columns)?;
    shape.device = device;
    shape.bits = bits;
    shape.code = FLOAT;
    export_with(py, owner, (None, None), pointer, shape, max_version)
}

fn export_with(
    py: Python<'_>,
    owner: PyObject,
    lease: Lease,
    pointer: u64,
    shape: ExportShape,
    max_version: Option<(u32, u32)>,
) -> PyResult<PyObject> {
    if let Some((major, minor)) = max_version.filter(|version| version.0 >= 1) {
        let version = Version {
            major: 1,
            minor: if major == 1 { minor.min(3) } else { 3 },
        };
        export_versioned(py, owner, lease, pointer, shape, version)
    } else {
        export_legacy(py, owner, lease, pointer, shape)
    }
}

fn export_legacy(
    py: Python<'_>,
    owner: PyObject,
    lease: Lease,
    pointer: u64,
    shape: ExportShape,
) -> PyResult<PyObject> {
    let mut state = Box::new(LegacyOwner {
        managed: Managed {
            tensor: tensor(
                pointer,
                shape.rank,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                shape,
            ),
            context: std::ptr::null_mut(),
            deleter: Some(release_legacy),
        },
        shape: shape.values,
        _owner: owner,
        lease,
    });
    state.managed.tensor.shape = state.shape.as_mut_ptr();
    state.managed.context = (&mut *state as *mut LegacyOwner).cast();
    let managed = &mut state.managed as *mut Managed;
    let _state = Box::into_raw(state);
    make_capsule(py, managed.cast(), LEGACY, drop_legacy, || unsafe {
        release_legacy(managed)
    })
}

fn export_versioned(
    py: Python<'_>,
    owner: PyObject,
    lease: Lease,
    pointer: u64,
    shape: ExportShape,
    version: Version,
) -> PyResult<PyObject> {
    let mut state = Box::new(VersionOwner {
        managed: ManagedV1 {
            version,
            context: std::ptr::null_mut(),
            deleter: Some(release_versioned),
            flags: READ_ONLY,
            tensor: tensor(
                pointer,
                shape.rank,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                shape,
            ),
        },
        shape: shape.values,
        strides: shape.strides,
        _owner: owner,
        lease,
    });
    state.managed.tensor.shape = state.shape.as_mut_ptr();
    state.managed.tensor.strides = state.strides.as_mut_ptr();
    state.managed.context = (&mut *state as *mut VersionOwner).cast();
    let managed = &mut state.managed as *mut ManagedV1;
    let _state = Box::into_raw(state);
    make_capsule(py, managed.cast(), VERSIONED, drop_versioned, || unsafe {
        release_versioned(managed)
    })
}

fn tensor(
    pointer: u64,
    rank: i32,
    shape: *mut i64,
    strides: *mut i64,
    format: ExportShape,
) -> Tensor {
    Tensor {
        data: pointer as *mut c_void,
        device: Device {
            kind: format.device,
            index: 0,
        },
        ndim: rank,
        dtype: DataType {
            code: format.code,
            bits: format.bits,
            lanes: 1,
        },
        shape,
        strides,
        byte_offset: 0,
    }
}

fn make_capsule<F>(
    py: Python<'_>,
    pointer: *mut c_void,
    name: &'static [u8],
    destructor: ffi::PyCapsule_Destructor,
    failure: F,
) -> PyResult<PyObject>
where
    F: FnOnce(),
{
    let capsule =
        unsafe { ffi::PyCapsule_New(pointer, name.as_ptr().cast::<c_char>(), Some(destructor)) };
    if capsule.is_null() {
        failure();
        Err(PyErr::fetch(py))
    } else {
        Ok(unsafe { Bound::from_owned_ptr(py, capsule).unbind() })
    }
}

fn validate_tensor(tensor: &Tensor, code: u8, bits: u8, name: &str, device: i32) -> PyResult<()> {
    if tensor.device.kind != device || tensor.device.index != 0 {
        return Err(PyBufferError::new_err(format!(
            "{name} input requires a CUDA device-0 DLPack tensor"
        )));
    }
    if tensor.dtype.code != code || tensor.dtype.bits != bits || tensor.dtype.lanes != 1 {
        return Err(PyBufferError::new_err(format!(
            "{name} input requires {name} DLPack data"
        )));
    }
    if tensor.ndim <= 0 || tensor.shape.is_null() || tensor.data.is_null() {
        return Err(PyValueError::new_err(format!(
            "{name} input requires a non-empty tensor"
        )));
    }
    Ok(())
}

fn is_contiguous(tensor: &Tensor, shape: &[i64]) -> bool {
    if tensor.strides.is_null() {
        return true;
    }
    let strides = unsafe { std::slice::from_raw_parts(tensor.strides, shape.len()) };
    let mut expected = 1_i64;
    for (&dimension, &stride) in shape.iter().zip(strides).rev() {
        if dimension > 1 && stride != expected {
            return false;
        }
        let Some(next) = expected.checked_mul(dimension.max(1)) else {
            return false;
        };
        expected = next;
    }
    true
}
