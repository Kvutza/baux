//! Core C-ABI data types for `libcuvs_c`.

use std::ffi::c_void;
use std::mem::size_of;

use super::dlpack::DLDataType;

#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
#[allow(non_camel_case_types)]
pub enum cuvsError_t {
    CUVS_ERROR = 0,
    CUVS_SUCCESS = 1,
}

#[allow(non_camel_case_types)]
pub type cuvsResources_t = usize;

#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
#[allow(non_camel_case_types, dead_code)]
pub enum cuvsDatasetLayout_t {
    CUVS_DATASET_LAYOUT_STANDARD = 0,
    CUVS_DATASET_LAYOUT_PADDED = 1,
}

#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
#[allow(non_camel_case_types, dead_code)]
pub enum cuvsDatasetMemType_t {
    CUVS_DATASET_MEM_TYPE_HOST = 0,
    CUVS_DATASET_MEM_TYPE_DEVICE = 1,
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
#[allow(non_camel_case_types)]
pub struct cuvsDataset {
    pub addr: usize,
    pub destroy_addr: Option<unsafe extern "C" fn(arg1: *mut c_void)>,
    pub dtype: DLDataType,
    pub mem_type: cuvsDatasetMemType_t,
    pub layout: cuvsDatasetLayout_t,
    pub is_owning: bool,
}

#[allow(non_camel_case_types)]
pub type cuvsDataset_t = *mut cuvsDataset;

#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
#[allow(non_camel_case_types, dead_code)]
pub enum cuvsDistanceType {
    L2Expanded = 0,
    L2SqrtExpanded = 1,
    CosineExpanded = 2,
    L1 = 3,
    L2Unexpanded = 4,
    L2SqrtUnexpanded = 5,
    InnerProduct = 6,
    Linf = 7,
    Canberra = 8,
    LpUnexpanded = 9,
}

#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
#[allow(non_camel_case_types, dead_code)]
pub enum cuvsFilterType {
    NO_FILTER = 0,
    BITSET = 1,
    BITMAP = 2,
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
#[allow(non_camel_case_types)]
pub struct cuvsFilter {
    pub addr: usize,
    pub type_: cuvsFilterType,
}

impl cuvsFilter {
    pub const NONE: Self = Self {
        addr: 0,
        type_: cuvsFilterType::NO_FILTER,
    };
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
#[allow(non_camel_case_types)]
pub struct cuvsBruteForceIndex {
    pub addr: usize,
    pub dtype: DLDataType,
}

#[allow(non_camel_case_types)]
pub type cuvsBruteForceIndex_t = *mut cuvsBruteForceIndex;

#[repr(C)]
#[derive(Debug, Copy, Clone)]
#[allow(non_camel_case_types)]
pub struct cuvsCagraIndex {
    pub addr: usize,
    pub dtype: DLDataType,
}

#[allow(non_camel_case_types)]
pub type cuvsCagraIndex_t = *mut cuvsCagraIndex;

const _: () = {
    assert!(size_of::<cuvsDataset>() == 32);
    assert!(size_of::<cuvsFilter>() == 16);
    assert!(size_of::<cuvsBruteForceIndex>() == 16);
    assert!(size_of::<cuvsCagraIndex>() == 16);
};
