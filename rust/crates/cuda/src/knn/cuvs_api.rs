//! Dynamic loader and CAGRA parameter types for `libcuvs_c`.

use std::ffi::{CStr, CString, c_char, c_int, c_void};
use std::mem::size_of;
use std::path::PathBuf;
use std::ptr;
use std::sync::OnceLock;

use super::CudaResult;
use super::cuvs_ffi::{
    cuvsBruteForceIndex_t, cuvsCagraIndex_t, cuvsDataset_t, cuvsDatasetMemType_t, cuvsDistanceType,
    cuvsError_t, cuvsFilter, cuvsResources_t,
};
use super::dlpack::DLManagedTensor;

#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
#[allow(non_camel_case_types, dead_code)]
pub enum cuvsCagraGraphBuildAlgo {
    AUTO_SELECT = 0,
    IVF_PQ = 1,
    NN_DESCENT = 2,
    ITERATIVE_CAGRA_SEARCH = 3,
    ACE = 4,
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
#[allow(non_camel_case_types)]
pub struct cuvsCagraIndexParams {
    pub metric: cuvsDistanceType,
    pub intermediate_graph_degree: usize,
    pub graph_degree: usize,
    pub build_algo: cuvsCagraGraphBuildAlgo,
    pub nn_descent_niter: usize,
    pub graph_build_params: *mut c_void,
}

#[allow(non_camel_case_types)]
pub type cuvsCagraIndexParams_t = *mut cuvsCagraIndexParams;

#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
#[allow(non_camel_case_types, dead_code)]
pub enum cuvsCagraSearchAlgo {
    SINGLE_CTA = 0,
    MULTI_CTA = 1,
    MULTI_KERNEL = 2,
    AUTO = 100,
}

#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
#[allow(non_camel_case_types, dead_code)]
pub enum cuvsCagraHashMode {
    HASH = 0,
    SMALL = 1,
    AUTO_HASH = 100,
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
#[allow(non_camel_case_types)]
pub struct cuvsCagraSearchParams {
    pub max_queries: usize,
    pub itopk_size: usize,
    pub max_iterations: usize,
    pub algo: cuvsCagraSearchAlgo,
    pub team_size: usize,
    pub search_width: usize,
    pub min_iterations: usize,
    pub thread_block_size: usize,
    pub hashmap_mode: cuvsCagraHashMode,
    pub hashmap_min_bitlen: usize,
    pub hashmap_max_fill_rate: f32,
    pub num_random_samplings: u32,
    pub rand_xor_mask: u64,
    pub persistent: bool,
    pub persistent_lifetime: f32,
    pub persistent_device_usage: f32,
}

#[allow(non_camel_case_types)]
pub type cuvsCagraSearchParams_t = *mut cuvsCagraSearchParams;

const _: () = {
    assert!(size_of::<cuvsCagraIndexParams>() == 48);
    assert!(size_of::<cuvsCagraSearchParams>() == 112);
};

const RTLD_LAZY: c_int = 1;
const RTLD_LOCAL: c_int = 4;

#[link(name = "dl")]
unsafe extern "C" {
    fn dlopen(filename: *const c_char, flags: c_int) -> *mut c_void;
    fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
    fn dlerror() -> *const c_char;
}

#[allow(non_snake_case)]
pub(super) struct CuvsApi {
    pub cuvsGetLastErrorText: unsafe extern "C" fn() -> *const c_char,
    pub cuvsResourcesCreate: unsafe extern "C" fn(*mut cuvsResources_t) -> cuvsError_t,
    pub cuvsResourcesDestroy: unsafe extern "C" fn(cuvsResources_t) -> cuvsError_t,
    pub cuvsStreamSet: unsafe extern "C" fn(cuvsResources_t, *mut c_void) -> cuvsError_t,
    pub cuvsDatasetMakePadded: unsafe extern "C" fn(
        cuvsResources_t,
        *mut DLManagedTensor,
        cuvsDatasetMemType_t,
        *mut cuvsDataset_t,
    ) -> cuvsError_t,
    pub cuvsDatasetMakePaddedView: unsafe extern "C" fn(
        cuvsResources_t,
        *mut DLManagedTensor,
        *mut cuvsDataset_t,
    ) -> cuvsError_t,
    pub cuvsDatasetDestroy: unsafe extern "C" fn(cuvsDataset_t) -> cuvsError_t,
    pub cuvsBruteForceIndexCreate: unsafe extern "C" fn(*mut cuvsBruteForceIndex_t) -> cuvsError_t,
    pub cuvsBruteForceIndexDestroy: unsafe extern "C" fn(cuvsBruteForceIndex_t) -> cuvsError_t,
    pub cuvsBruteForceBuild: unsafe extern "C" fn(
        cuvsResources_t,
        *mut DLManagedTensor,
        cuvsDistanceType,
        f32,
        cuvsBruteForceIndex_t,
    ) -> cuvsError_t,
    pub cuvsBruteForceSearch: unsafe extern "C" fn(
        cuvsResources_t,
        cuvsBruteForceIndex_t,
        *mut DLManagedTensor,
        *mut DLManagedTensor,
        *mut DLManagedTensor,
        cuvsFilter,
    ) -> cuvsError_t,
    pub cuvsCagraIndexParamsCreate:
        unsafe extern "C" fn(*mut cuvsCagraIndexParams_t) -> cuvsError_t,
    pub cuvsCagraIndexParamsDestroy: unsafe extern "C" fn(cuvsCagraIndexParams_t) -> cuvsError_t,
    pub cuvsCagraSearchParamsCreate:
        unsafe extern "C" fn(*mut cuvsCagraSearchParams_t) -> cuvsError_t,
    pub cuvsCagraSearchParamsDestroy: unsafe extern "C" fn(cuvsCagraSearchParams_t) -> cuvsError_t,
    pub cuvsCagraIndexCreate: unsafe extern "C" fn(*mut cuvsCagraIndex_t) -> cuvsError_t,
    pub cuvsCagraIndexDestroy: unsafe extern "C" fn(cuvsCagraIndex_t) -> cuvsError_t,
    pub cuvsCagraBuild: unsafe extern "C" fn(
        cuvsResources_t,
        cuvsCagraIndexParams_t,
        cuvsDataset_t,
        cuvsCagraIndex_t,
    ) -> cuvsError_t,
    pub cuvsCagraSearch: unsafe extern "C" fn(
        cuvsResources_t,
        cuvsCagraSearchParams_t,
        cuvsCagraIndex_t,
        *mut DLManagedTensor,
        *mut DLManagedTensor,
        *mut DLManagedTensor,
        cuvsFilter,
    ) -> cuvsError_t,
}

unsafe fn load_sym<T>(handle: *mut c_void, name: &'static CStr) -> CudaResult<T> {
    let ptr = unsafe { dlsym(handle, name.as_ptr()) };
    if ptr.is_null() {
        let err = unsafe { dlerror() };
        let detail = if err.is_null() {
            "symbol not found".to_string()
        } else {
            unsafe { CStr::from_ptr(err) }
                .to_string_lossy()
                .into_owned()
        };
        return Err(format!(
            "cuVS library is missing symbol {}: {detail}",
            name.to_string_lossy()
        ));
    }
    Ok(unsafe { std::mem::transmute_copy(&ptr) })
}

fn candidate_library_paths() -> Vec<String> {
    let mut candidates = Vec::new();
    if let Some(custom) = std::env::var_os("CUVS_LIB_PATH") {
        let path = PathBuf::from(&custom);
        if path.is_dir() {
            candidates.push(path.join("libcuvs_c.so").to_string_lossy().into_owned());
            candidates.push(path.join("libcuvs_c.dylib").to_string_lossy().into_owned());
        } else {
            candidates.push(path.to_string_lossy().into_owned());
        }
    }
    if let Some(conda) = std::env::var_os("CONDA_PREFIX") {
        let lib = PathBuf::from(conda).join("lib");
        candidates.push(lib.join("libcuvs_c.so").to_string_lossy().into_owned());
    }
    candidates.push("libcuvs_c.so".to_string());
    candidates.push("libcuvs_c.so.26.06".to_string());
    candidates.push("libcuvs_c.so.26.04".to_string());
    candidates.push("libcuvs_c.so.26.02".to_string());
    candidates.push("libcuvs_c.dylib".to_string());
    candidates
}

fn preload_sibling_cuvs(candidate: &str) {
    let path = PathBuf::from(candidate);
    let Some(parent) = path.parent() else {
        return;
    };
    if parent.as_os_str().is_empty() {
        return;
    }
    let sibling = parent.join("libcuvs.so");
    if sibling.is_file()
        && let Ok(c_str) = CString::new(sibling.to_string_lossy().into_owned())
    {
        unsafe {
            let _ = dlopen(c_str.as_ptr(), RTLD_LAZY | RTLD_LOCAL);
        }
    }
}

pub(super) fn is_available() -> bool {
    cuvs_api().is_ok()
}

pub(super) fn cuvs_api() -> CudaResult<&'static CuvsApi> {
    static API: OnceLock<Result<CuvsApi, String>> = OnceLock::new();
    API.get_or_init(|| {
        let mut handle = ptr::null_mut();
        for candidate in candidate_library_paths() {
            preload_sibling_cuvs(&candidate);
            let Ok(c_str) = CString::new(candidate) else {
                continue;
            };
            let opened = unsafe { dlopen(c_str.as_ptr(), RTLD_LAZY | RTLD_LOCAL) };
            if !opened.is_null() {
                handle = opened;
                break;
            }
        }
        if handle.is_null() {
            return Err(
                "failed to load libcuvs_c.so (install cuVS or set CUVS_LIB_PATH)".to_string(),
            );
        }
        unsafe {
            Ok(CuvsApi {
                cuvsGetLastErrorText: load_sym(handle, c"cuvsGetLastErrorText")?,
                cuvsResourcesCreate: load_sym(handle, c"cuvsResourcesCreate")?,
                cuvsResourcesDestroy: load_sym(handle, c"cuvsResourcesDestroy")?,
                cuvsStreamSet: load_sym(handle, c"cuvsStreamSet")?,
                cuvsDatasetMakePadded: load_sym(handle, c"cuvsDatasetMakePadded")?,
                cuvsDatasetMakePaddedView: load_sym(handle, c"cuvsDatasetMakePaddedView")?,
                cuvsDatasetDestroy: load_sym(handle, c"cuvsDatasetDestroy")?,
                cuvsBruteForceIndexCreate: load_sym(handle, c"cuvsBruteForceIndexCreate")?,
                cuvsBruteForceIndexDestroy: load_sym(handle, c"cuvsBruteForceIndexDestroy")?,
                cuvsBruteForceBuild: load_sym(handle, c"cuvsBruteForceBuild")?,
                cuvsBruteForceSearch: load_sym(handle, c"cuvsBruteForceSearch")?,
                cuvsCagraIndexParamsCreate: load_sym(handle, c"cuvsCagraIndexParamsCreate")?,
                cuvsCagraIndexParamsDestroy: load_sym(handle, c"cuvsCagraIndexParamsDestroy")?,
                cuvsCagraSearchParamsCreate: load_sym(handle, c"cuvsCagraSearchParamsCreate")?,
                cuvsCagraSearchParamsDestroy: load_sym(handle, c"cuvsCagraSearchParamsDestroy")?,
                cuvsCagraIndexCreate: load_sym(handle, c"cuvsCagraIndexCreate")?,
                cuvsCagraIndexDestroy: load_sym(handle, c"cuvsCagraIndexDestroy")?,
                cuvsCagraBuild: load_sym(handle, c"cuvsCagraBuild")?,
                cuvsCagraSearch: load_sym(handle, c"cuvsCagraSearch")?,
            })
        }
    })
    .as_ref()
    .map_err(Clone::clone)
}

pub(super) fn check_cuvs(api: &CuvsApi, status: cuvsError_t, op: &str) -> CudaResult<()> {
    if status == cuvsError_t::CUVS_SUCCESS {
        return Ok(());
    }
    let detail = unsafe {
        let ptr = (api.cuvsGetLastErrorText)();
        if ptr.is_null() {
            "unknown cuVS error".to_string()
        } else {
            CStr::from_ptr(ptr).to_string_lossy().into_owned()
        }
    };
    Err(format!("cuVS {op} failed: {detail}"))
}
