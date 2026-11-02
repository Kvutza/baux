//! NVIDIA cuVS C-API integration (`libcuvs_c`) for exact Brute-Force and CAGRA
//! graph k-NN.
//!
//! Uses zero-copy DLPack device tensor views (`DLManagedTensor` with `kDLCUDA`)
//! bound to `CudaIndex`'s existing `CudaStream` and `DeviceBuffer` allocations.

use std::ffi::c_void;
use std::mem::{MaybeUninit, size_of};
use std::ptr;

use cuda_core::{CudaStream, DeviceBuffer, IntoResult};

use super::cuvs_api::{CuvsApi, check_cuvs, cuvs_api};
use super::cuvs_ffi::{
    cuvsBruteForceIndex_t, cuvsCagraIndex_t, cuvsDataset_t, cuvsDistanceType, cuvsFilter,
    cuvsResources_t,
};
use super::dlpack::{DLDataType, DlMatrix2D};
use super::{CudaIndex, CudaResult, cuda_error, next_capacity};

pub const CUVS_MAX_K: usize = baux_cuda_device::POST_MAX_K;
const CAGRA_MIN_ROWS: usize = 128;

pub fn is_available() -> bool {
    super::cuvs_api::is_available()
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum CudaKnnMode {
    #[default]
    Kernel,
    CuvsBruteForce,
    CuvsCagra,
}

impl CudaKnnMode {
    pub const fn max_neighbors(self) -> usize {
        match self {
            Self::Kernel => super::KNN_K,
            Self::CuvsBruteForce | Self::CuvsCagra => CUVS_MAX_K,
        }
    }
}

impl CudaIndex {
    pub fn with_cuvs_brute_force(dims: usize, rows: &[f32]) -> CudaResult<Self> {
        Self::with_mode(dims, rows, CudaKnnMode::CuvsBruteForce)
    }

    pub fn with_cuvs_cagra(dims: usize, rows: &[f32]) -> CudaResult<Self> {
        Self::with_mode(dims, rows, CudaKnnMode::CuvsCagra)
    }

    pub const fn mode(&self) -> CudaKnnMode {
        self.mode
    }

    pub const fn max_neighbors(&self) -> usize {
        self.mode.max_neighbors()
    }

    pub(super) fn invalidate_cuvs(&mut self) {
        if let Some(cuvs) = self.cuvs.as_mut() {
            cuvs.invalidate();
        }
    }
}

struct CagraBuilt {
    dataset: cuvsDataset_t,
    index: cuvsCagraIndex_t,
}

enum ActiveCuvsIndex {
    None,
    BruteForce(cuvsBruteForceIndex_t),
    Cagra(CagraBuilt),
}

pub(super) struct CuvsState {
    api: &'static CuvsApi,
    res: cuvsResources_t,
    active: ActiveCuvsIndex,
    dirty: bool,
    i64_capacity: usize,
    indices_i64: DeviceBuffer<i64>,
}

// SAFETY: CuvsState is accessed exclusively through `&mut CudaIndex` and
// synchronized on `CudaIndex`'s stream.
unsafe impl Send for CuvsState {}
unsafe impl Sync for CuvsState {}

impl CuvsState {
    pub(super) fn new(stream: &CudaStream) -> CudaResult<Self> {
        let api = cuvs_api()?;
        let mut res: cuvsResources_t = 0;
        check_cuvs(
            api,
            unsafe { (api.cuvsResourcesCreate)(&mut res) },
            "cuvsResourcesCreate",
        )?;
        if let Err(err) = check_cuvs(
            api,
            unsafe { (api.cuvsStreamSet)(res, stream.cu_stream() as *mut c_void) },
            "cuvsStreamSet",
        ) {
            let _ = unsafe { (api.cuvsResourcesDestroy)(res) };
            return Err(err);
        }
        let indices_i64 = DeviceBuffer::zeroed(stream, 1).map_err(cuda_error)?;
        Ok(Self {
            api,
            res,
            active: ActiveCuvsIndex::None,
            dirty: true,
            i64_capacity: 1,
            indices_i64,
        })
    }

    pub(super) fn invalidate(&mut self) {
        self.destroy_active();
        self.dirty = true;
    }

    fn destroy_active(&mut self) {
        match std::mem::replace(&mut self.active, ActiveCuvsIndex::None) {
            ActiveCuvsIndex::None => {}
            ActiveCuvsIndex::BruteForce(index) => {
                let _ = unsafe { (self.api.cuvsBruteForceIndexDestroy)(index) };
            }
            ActiveCuvsIndex::Cagra(built) => {
                let _ = unsafe { (self.api.cuvsCagraIndexDestroy)(built.index) };
                let _ = unsafe { (self.api.cuvsDatasetDestroy)(built.dataset) };
            }
        }
    }

    fn build_cagra(&mut self, dataset_view: &mut DlMatrix2D, dims: usize) -> CudaResult<()> {
        let mut tensor = dataset_view.managed();
        let mut dataset_handle: cuvsDataset_t = ptr::null_mut();
        let make_status = unsafe {
            if dims.is_multiple_of(4) {
                (self.api.cuvsDatasetMakePaddedView)(self.res, &mut tensor, &mut dataset_handle)
            } else {
                (self.api.cuvsDatasetMakePadded)(
                    self.res,
                    &mut tensor,
                    super::cuvs_ffi::cuvsDatasetMemType_t::CUVS_DATASET_MEM_TYPE_DEVICE,
                    &mut dataset_handle,
                )
            }
        };
        check_cuvs(self.api, make_status, "cuvsDatasetMakePadded")?;

        let mut params: super::cuvs_api::cuvsCagraIndexParams_t = ptr::null_mut();
        if let Err(err) = check_cuvs(
            self.api,
            unsafe { (self.api.cuvsCagraIndexParamsCreate)(&mut params) },
            "cuvsCagraIndexParamsCreate",
        ) {
            let _ = unsafe { (self.api.cuvsDatasetDestroy)(dataset_handle) };
            return Err(err);
        }
        unsafe {
            (*params).metric = cuvsDistanceType::L2Expanded;
        }

        let mut index: cuvsCagraIndex_t = ptr::null_mut();
        if let Err(err) = check_cuvs(
            self.api,
            unsafe { (self.api.cuvsCagraIndexCreate)(&mut index) },
            "cuvsCagraIndexCreate",
        ) {
            let _ = unsafe { (self.api.cuvsCagraIndexParamsDestroy)(params) };
            let _ = unsafe { (self.api.cuvsDatasetDestroy)(dataset_handle) };
            return Err(err);
        }

        let build_res = check_cuvs(
            self.api,
            unsafe { (self.api.cuvsCagraBuild)(self.res, params, dataset_handle, index) },
            "cuvsCagraBuild",
        );
        let _ = unsafe { (self.api.cuvsCagraIndexParamsDestroy)(params) };
        if let Err(err) = build_res {
            let _ = unsafe { (self.api.cuvsCagraIndexDestroy)(index) };
            let _ = unsafe { (self.api.cuvsDatasetDestroy)(dataset_handle) };
            return Err(err);
        }
        self.active = ActiveCuvsIndex::Cagra(CagraBuilt {
            dataset: dataset_handle,
            index,
        });
        Ok(())
    }

    fn build_brute_force(&mut self, dataset_view: &mut DlMatrix2D) -> CudaResult<()> {
        let mut index = MaybeUninit::<cuvsBruteForceIndex_t>::uninit();
        check_cuvs(
            self.api,
            unsafe { (self.api.cuvsBruteForceIndexCreate)(index.as_mut_ptr()) },
            "cuvsBruteForceIndexCreate",
        )?;
        let index = unsafe { index.assume_init() };
        let mut tensor = dataset_view.managed();
        let build_res = check_cuvs(
            self.api,
            unsafe {
                (self.api.cuvsBruteForceBuild)(
                    self.res,
                    &mut tensor,
                    cuvsDistanceType::L2Expanded,
                    2.0,
                    index,
                )
            },
            "cuvsBruteForceBuild",
        );
        if let Err(err) = build_res {
            let _ = unsafe { (self.api.cuvsBruteForceIndexDestroy)(index) };
            return Err(err);
        }
        self.active = ActiveCuvsIndex::BruteForce(index);
        Ok(())
    }

    fn ensure_built(
        &mut self,
        mode: CudaKnnMode,
        rows: &DeviceBuffer<f32>,
        row_count: usize,
        dims: usize,
    ) -> CudaResult<()> {
        if !self.dirty && !matches!(self.active, ActiveCuvsIndex::None) {
            return Ok(());
        }
        self.destroy_active();
        let mut dataset_view =
            DlMatrix2D::cuda(rows.cu_deviceptr(), row_count, dims, DLDataType::F32)?;
        if mode == CudaKnnMode::CuvsCagra && row_count >= CAGRA_MIN_ROWS {
            self.build_cagra(&mut dataset_view, dims)?;
        } else {
            self.build_brute_force(&mut dataset_view)?;
        }
        self.dirty = false;
        Ok(())
    }

    fn search_cagra(
        &self,
        index: cuvsCagraIndex_t,
        queries_view: &mut DlMatrix2D,
        distances_view: &mut DlMatrix2D,
        indices_out: &mut DeviceBuffer<u32>,
        query_count: usize,
        neighbors: usize,
    ) -> CudaResult<&'static str> {
        let mut indices_view = DlMatrix2D::cuda(
            indices_out.cu_deviceptr(),
            query_count,
            neighbors,
            DLDataType::U32,
        )?;
        let mut search_params: super::cuvs_api::cuvsCagraSearchParams_t = ptr::null_mut();
        check_cuvs(
            self.api,
            unsafe { (self.api.cuvsCagraSearchParamsCreate)(&mut search_params) },
            "cuvsCagraSearchParamsCreate",
        )?;
        unsafe {
            (*search_params).itopk_size =
                neighbors.max(64).checked_next_power_of_two().unwrap_or(512);
        }
        let mut q_tensor = queries_view.managed();
        let mut n_tensor = indices_view.managed();
        let mut d_tensor = distances_view.managed();
        let status = unsafe {
            (self.api.cuvsCagraSearch)(
                self.res,
                search_params,
                index,
                &mut q_tensor,
                &mut n_tensor,
                &mut d_tensor,
                cuvsFilter::NONE,
            )
        };
        let _ = unsafe { (self.api.cuvsCagraSearchParamsDestroy)(search_params) };
        check_cuvs(self.api, status, "cuvsCagraSearch")?;
        Ok("cuvs_cagra")
    }

    #[allow(clippy::too_many_arguments)]
    fn search_brute_force(
        &self,
        index: cuvsBruteForceIndex_t,
        stream: &CudaStream,
        queries_view: &mut DlMatrix2D,
        distances_view: &mut DlMatrix2D,
        indices_out: &mut DeviceBuffer<u32>,
        query_count: usize,
        neighbors: usize,
        total_neighbors: usize,
    ) -> CudaResult<&'static str> {
        let mut indices_i64_view = DlMatrix2D::cuda(
            self.indices_i64.cu_deviceptr(),
            query_count,
            neighbors,
            DLDataType::I64,
        )?;
        let mut q_tensor = queries_view.managed();
        let mut n_tensor = indices_i64_view.managed();
        let mut d_tensor = distances_view.managed();
        let status = unsafe {
            (self.api.cuvsBruteForceSearch)(
                self.res,
                index,
                &mut q_tensor,
                &mut n_tensor,
                &mut d_tensor,
                cuvsFilter::NONE,
            )
        };
        check_cuvs(self.api, status, "cuvsBruteForceSearch")?;
        narrow_i64_to_u32_async(&self.indices_i64, indices_out, total_neighbors, stream)?;
        Ok("cuvs_brute_force")
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn search_into(
        &mut self,
        mode: CudaKnnMode,
        stream: &CudaStream,
        rows: &DeviceBuffer<f32>,
        row_count: usize,
        dims: usize,
        queries: &DeviceBuffer<f32>,
        query_count: usize,
        neighbors: usize,
        distances_out: &mut DeviceBuffer<f32>,
        indices_out: &mut DeviceBuffer<u32>,
    ) -> CudaResult<&'static str> {
        let total_neighbors = query_count
            .checked_mul(neighbors)
            .ok_or("cuVS output neighbor count overflow")?;
        if total_neighbors > self.i64_capacity {
            let cap = next_capacity(total_neighbors)?;
            self.indices_i64 = DeviceBuffer::zeroed(stream, cap).map_err(cuda_error)?;
            self.i64_capacity = cap;
        }
        self.ensure_built(mode, rows, row_count, dims)?;

        let mut queries_view =
            DlMatrix2D::cuda(queries.cu_deviceptr(), query_count, dims, DLDataType::F32)?;
        let mut distances_view = DlMatrix2D::cuda(
            distances_out.cu_deviceptr(),
            query_count,
            neighbors,
            DLDataType::F32,
        )?;

        match &self.active {
            ActiveCuvsIndex::Cagra(built) => self.search_cagra(
                built.index,
                &mut queries_view,
                &mut distances_view,
                indices_out,
                query_count,
                neighbors,
            ),
            ActiveCuvsIndex::BruteForce(index) => self.search_brute_force(
                *index,
                stream,
                &mut queries_view,
                &mut distances_view,
                indices_out,
                query_count,
                neighbors,
                total_neighbors,
            ),
            ActiveCuvsIndex::None => Err("cuVS index was not initialized".to_string()),
        }
    }
}

impl Drop for CuvsState {
    fn drop(&mut self) {
        self.destroy_active();
        if self.res != 0 {
            let _ = unsafe { (self.api.cuvsResourcesDestroy)(self.res) };
        }
    }
}

fn narrow_i64_to_u32_async(
    src_i64: &DeviceBuffer<i64>,
    dst_u32: &mut DeviceBuffer<u32>,
    count: usize,
    stream: &CudaStream,
) -> CudaResult<()> {
    if count == 0 {
        return Ok(());
    }
    let copy_desc = cuda_core::sys::CUDA_MEMCPY2D_st {
        srcXInBytes: 0,
        srcY: 0,
        srcMemoryType: cuda_core::sys::CUmemorytype_enum_CU_MEMORYTYPE_DEVICE,
        srcHost: ptr::null(),
        srcDevice: src_i64.cu_deviceptr(),
        srcArray: ptr::null_mut(),
        srcPitch: size_of::<i64>(),
        dstXInBytes: 0,
        dstY: 0,
        dstMemoryType: cuda_core::sys::CUmemorytype_enum_CU_MEMORYTYPE_DEVICE,
        dstHost: ptr::null_mut(),
        dstDevice: dst_u32.cu_deviceptr(),
        dstArray: ptr::null_mut(),
        dstPitch: size_of::<u32>(),
        WidthInBytes: size_of::<u32>(),
        Height: count,
    };
    unsafe {
        cuda_core::sys::cuMemcpy2DAsync_v2(&copy_desc, stream.cu_stream())
            .result()
            .map_err(cuda_error)
    }
}

#[cfg(test)]
mod tests {
    use super::super::dlpack::DLDeviceType;
    use super::*;

    #[test]
    fn dlpack_matrix_view_layout() {
        let mut mat = DlMatrix2D::cuda(0x1234_5678, 32, 16, DLDataType::F32).unwrap();
        let managed = mat.managed();
        let tensor = &managed.dl_tensor;
        assert_eq!(tensor.data as usize, 0x1234_5678);
        assert_eq!(tensor.device.device_type, DLDeviceType::kDLCUDA);
        assert_eq!(tensor.device.device_id, 0);
        assert_eq!(tensor.ndim, 2);
        assert_eq!(tensor.dtype, DLDataType::F32);
        assert!(tensor.strides.is_null());
        assert_eq!(tensor.byte_offset, 0);
        let shape = unsafe { std::slice::from_raw_parts(tensor.shape, 2) };
        assert_eq!(shape, &[32, 16]);
    }
}
