use std::mem::size_of;
use std::sync::{Arc, OnceLock};

pub mod cuvs;
mod cuvs_api;
mod cuvs_ffi;
mod dlpack;
mod knn_post;
mod resident;
pub use cuvs::{CUVS_MAX_K, CudaKnnMode};
use knn_post::PosteriorScratch;
pub use knn_post::{
    BatchOutput, BatchSpec, DrawOutput, PosteriorOutput, PosteriorSpec, WeightedOutput,
    WeightedSpec,
};
pub use resident::{DeviceDraw, DevicePosterior};

use super::capture::{
    CudaResult, Runtime, copy_prefix, cuda_error, read_prefix, timing_event, to_u32,
};
use baux_cuda_device::{KNN_K, KNN_THREADS, KNN_WARPS, KnnParams, MergeParams, THREADS};
use cuda_core::{CudaEvent, CudaStream, DeviceBuffer, DeviceCopy, LaunchConfig1D};
use cuda_host::kernel_family::{KernelFamily, KernelProblem, KernelSelector, KernelVariant};

const WARP_DIM: usize = 32;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ScanKind {
    Rows,
    Warps,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct ScanMeta {
    tile_rows: u32,
}

#[derive(Clone, Copy, Debug)]
pub(super) struct ScanProblem {
    pub(super) rows: usize,
    pub(super) dims: usize,
    pub(super) queries: usize,
    pub(super) neighbors: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct ScanReject(&'static str);

impl std::fmt::Display for ScanReject {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(self.0)
    }
}

impl std::error::Error for ScanReject {}

type ScanVariant = KernelVariant<ScanKind, ScanKind, ScanMeta>;
type ScanFamily = KernelFamily<ScanKind, ScanKind, ScanMeta, 2>;

impl KernelProblem<ScanVariant> for ScanProblem {
    type Rejection = ScanReject;

    fn validate(&self, variant: &ScanVariant) -> Result<(), Self::Rejection> {
        if self.rows == 0 || self.dims == 0 || self.queries == 0 {
            return Err(ScanReject("CUDA index shape must be non-zero"));
        }
        if self.neighbors == 0 || self.neighbors > KNN_K || self.neighbors > self.rows {
            return Err(ScanReject("CUDA index neighbor count is unsupported"));
        }
        if variant.metadata().tile_rows == 0 {
            return Err(ScanReject("CUDA index tile cannot be empty"));
        }
        Ok(())
    }
}

struct ScanSelector;

impl KernelSelector<ScanProblem, ScanVariant, ScanKind> for ScanSelector {
    type Error = std::convert::Infallible;

    fn select(
        &mut self,
        _family: cuda_host::kernel_family::KernelFamilyId,
        problem: &ScanProblem,
        _eligible: &[&ScanVariant],
    ) -> Result<ScanKind, Self::Error> {
        Ok(if problem.dims >= WARP_DIM {
            ScanKind::Warps
        } else {
            ScanKind::Rows
        })
    }
}

fn scan_family() -> CudaResult<&'static ScanFamily> {
    static FAMILY: OnceLock<Result<ScanFamily, String>> = OnceLock::new();
    FAMILY
        .get_or_init(|| {
            KernelFamily::try_new(
                "baux/knn-scan",
                1,
                [
                    KernelVariant::new(
                        ScanKind::Rows,
                        ScanKind::Rows,
                        ScanMeta {
                            tile_rows: KNN_THREADS,
                        },
                    ),
                    KernelVariant::new(
                        ScanKind::Warps,
                        ScanKind::Warps,
                        ScanMeta {
                            tile_rows: KNN_WARPS,
                        },
                    ),
                ],
            )
            .map_err(|error| error.to_string())
        })
        .as_ref()
        .map_err(Clone::clone)
}

#[derive(Clone, Copy, Debug, Default)]
pub struct KnnProfile {
    pub rows: usize,
    pub queries: usize,
    pub dims: usize,
    pub neighbors: usize,
    pub scan_ms: f32,
    pub merge_ms: f32,
    pub posterior_ms: f32,
    pub total_ms: f32,
    pub kind: &'static str,
}

pub(super) struct SearchStage {
    pub(super) plan_name: &'static str,
    pub(super) input_a: bool,
    pub(super) queries: usize,
    pub(super) neighbors: usize,
    pub(super) start: CudaEvent,
    pub(super) scan_end: CudaEvent,
    pub(super) merge_end: CudaEvent,
}

pub(super) struct KnnScratch {
    pub(super) query_capacity: usize,
    pub(super) partial_capacity: usize,
    pub(super) queries: DeviceBuffer<f32>,
    pub(super) distances_a: DeviceBuffer<f32>,
    pub(super) distances_b: DeviceBuffer<f32>,
    pub(super) indices_a: DeviceBuffer<u32>,
    pub(super) indices_b: DeviceBuffer<u32>,
}

impl KnnScratch {
    fn new(stream: &CudaStream) -> CudaResult<Self> {
        Ok(Self {
            query_capacity: 1,
            partial_capacity: 1,
            queries: DeviceBuffer::zeroed(stream, 1).map_err(cuda_error)?,
            distances_a: DeviceBuffer::zeroed(stream, 1).map_err(cuda_error)?,
            distances_b: DeviceBuffer::zeroed(stream, 1).map_err(cuda_error)?,
            indices_a: DeviceBuffer::zeroed(stream, 1).map_err(cuda_error)?,
            indices_b: DeviceBuffer::zeroed(stream, 1).map_err(cuda_error)?,
        })
    }

    fn ensure(&mut self, stream: &CudaStream, queries: usize, partials: usize) -> CudaResult<()> {
        let query_capacity = next_capacity(queries)?;
        let partial_capacity = next_capacity(partials)?;
        if query_capacity > self.query_capacity {
            self.queries = DeviceBuffer::zeroed(stream, query_capacity).map_err(cuda_error)?;
            self.query_capacity = query_capacity;
        }
        if partial_capacity > self.partial_capacity {
            self.distances_a =
                DeviceBuffer::zeroed(stream, partial_capacity).map_err(cuda_error)?;
            self.distances_b =
                DeviceBuffer::zeroed(stream, partial_capacity).map_err(cuda_error)?;
            self.indices_a = DeviceBuffer::zeroed(stream, partial_capacity).map_err(cuda_error)?;
            self.indices_b = DeviceBuffer::zeroed(stream, partial_capacity).map_err(cuda_error)?;
            self.partial_capacity = partial_capacity;
        }
        Ok(())
    }
}

pub struct CudaIndex {
    pub(super) runtime: Arc<Runtime>,
    pub(super) rows: DeviceBuffer<f32>,
    pub(super) row_capacity: usize,
    pub(super) row_count: usize,
    pub(super) dims: usize,
    pub(super) mode: CudaKnnMode,
    pub(super) cuvs: Option<cuvs::CuvsState>,
    pub(super) scratch: KnnScratch,
    pub(super) posterior: PosteriorScratch,
    pub(super) profile: Option<KnnProfile>,
}

impl CudaIndex {
    pub fn new(dims: usize, rows: &[f32]) -> CudaResult<Self> {
        if cuvs::is_available()
            && let Ok(index) = Self::with_mode(dims, rows, CudaKnnMode::CuvsBruteForce)
        {
            return Ok(index);
        }
        Self::with_mode(dims, rows, CudaKnnMode::Kernel)
    }

    pub fn with_mode(dims: usize, rows: &[f32], mode: CudaKnnMode) -> CudaResult<Self> {
        if dims == 0 || rows.len() % dims != 0 {
            return Err("CUDA index rows do not match the dimension".to_string());
        }
        if rows.iter().any(|value| !value.is_finite()) {
            return Err("CUDA index rows must be finite".to_string());
        }
        let runtime = Arc::new(Runtime::new()?);
        let cuvs = if mode != CudaKnnMode::Kernel {
            Some(cuvs::CuvsState::new(&runtime.stream)?)
        } else {
            None
        };
        let scratch = KnnScratch::new(&runtime.stream)?;
        let posterior = PosteriorScratch::new(&runtime.stream)?;
        let row_capacity = next_capacity(rows.len())?;
        let row_buffer = DeviceBuffer::zeroed(&runtime.stream, row_capacity).map_err(cuda_error)?;
        copy_prefix(&row_buffer, rows, &runtime.stream)?;
        runtime.stream.synchronize().map_err(cuda_error)?;
        Ok(Self {
            runtime,
            rows: row_buffer,
            row_capacity,
            row_count: rows.len() / dims,
            dims,
            mode,
            cuvs,
            scratch,
            posterior,
            profile: None,
        })
    }

    pub fn len(&self) -> usize {
        self.row_count
    }

    pub fn is_empty(&self) -> bool {
        self.row_count == 0
    }

    pub fn memory_bytes(&self) -> usize {
        self.row_count
            .saturating_mul(self.dims)
            .saturating_mul(std::mem::size_of::<f32>())
    }

    pub fn profile(&self) -> Option<KnnProfile> {
        self.profile
    }

    pub fn rebuild(&mut self, rows: &[f32]) -> CudaResult<()> {
        self.check_rows(rows)?;
        let capacity = next_capacity(rows.len())?;
        if capacity != self.row_capacity {
            self.rows = DeviceBuffer::zeroed(&self.runtime.stream, capacity).map_err(cuda_error)?;
            self.row_capacity = capacity;
        }
        copy_prefix(&self.rows, rows, &self.runtime.stream)?;
        self.runtime.stream.synchronize().map_err(cuda_error)?;
        self.row_count = rows.len() / self.dims;
        self.invalidate_cuvs();
        Ok(())
    }

    pub fn add(&mut self, rows: &[f32]) -> CudaResult<()> {
        self.check_rows(rows)?;
        if rows.is_empty() {
            return Ok(());
        }
        let used = self
            .row_count
            .checked_mul(self.dims)
            .ok_or("CUDA index row size overflow")?;
        let required = used
            .checked_add(rows.len())
            .ok_or("CUDA index row size overflow")?;
        if required > self.row_capacity {
            let capacity = next_capacity(required)?;
            let next = DeviceBuffer::zeroed(&self.runtime.stream, capacity).map_err(cuda_error)?;
            if used > 0 {
                let used_bytes = used
                    .checked_mul(std::mem::size_of::<f32>())
                    .ok_or("CUDA index copy size overflow")?;
                unsafe {
                    cuda_core::simt::memory::memcpy_dtod_async(
                        next.cu_deviceptr(),
                        self.rows.cu_deviceptr(),
                        used_bytes,
                        self.runtime.stream.cu_stream(),
                    )
                    .map_err(cuda_error)?;
                }
            }
            self.rows = next;
            self.row_capacity = capacity;
        }
        copy_at(&self.rows, used, rows, &self.runtime.stream)?;
        self.runtime.stream.synchronize().map_err(cuda_error)?;
        self.row_count = self
            .row_count
            .checked_add(rows.len() / self.dims)
            .ok_or("CUDA index row count overflow")?;
        self.invalidate_cuvs();
        Ok(())
    }

    pub fn search(
        &mut self,
        queries: &[f32],
        query_count: usize,
        neighbors: usize,
    ) -> CudaResult<(Vec<f32>, Vec<u32>)> {
        let stage = self.stage_search(queries, query_count, neighbors)?;
        let output = self.read_neighbors(&stage)?;
        self.set_profile(&stage, &stage.merge_end)?;
        Ok(output)
    }

    pub(super) fn stage_search(
        &mut self,
        queries: &[f32],
        query_count: usize,
        neighbors: usize,
    ) -> CudaResult<SearchStage> {
        let problem = self.check_search(queries, query_count, neighbors)?;
        self.stage_input(problem, Some(queries), 0)
    }

    pub(super) fn stage_input(
        &mut self,
        problem: ScanProblem,
        host: Option<&[f32]>,
        pointer: u64,
    ) -> CudaResult<SearchStage> {
        let query_count = problem.queries;
        let neighbors = problem.neighbors;
        let query_len = query_count
            .checked_mul(self.dims)
            .ok_or("query size overflow")?;
        if self.mode != CudaKnnMode::Kernel {
            let partials = query_count
                .checked_mul(neighbors)
                .ok_or("CUDA index scratch size overflow")?;
            self.scratch
                .ensure(&self.runtime.stream, query_len, partials)?;
            if let Some(values) = host {
                copy_prefix(&self.scratch.queries, values, &self.runtime.stream)?;
            } else {
                unsafe {
                    cuda_core::simt::memory::memcpy_dtod_async(
                        self.scratch.queries.cu_deviceptr(),
                        pointer,
                        query_len
                            .checked_mul(std::mem::size_of::<f32>())
                            .ok_or("query byte size overflow")?,
                        self.runtime.stream.cu_stream(),
                    )
                    .map_err(cuda_error)?;
                }
            }
            let start = timing_event(&self.runtime.stream)?;
            let cuvs = self
                .cuvs
                .as_mut()
                .ok_or("cuVS backend was not initialized")?;
            let plan_name = cuvs.search_into(
                self.mode,
                &self.runtime.stream,
                &self.rows,
                self.row_count,
                self.dims,
                &self.scratch.queries,
                query_count,
                neighbors,
                &mut self.scratch.distances_a,
                &mut self.scratch.indices_a,
            )?;
            let scan_end = timing_event(&self.runtime.stream)?;
            let merge_end = timing_event(&self.runtime.stream)?;
            self.runtime.context.check_err().map_err(cuda_error)?;
            return Ok(SearchStage {
                plan_name,
                input_a: true,
                queries: query_count,
                neighbors,
                start,
                scan_end,
                merge_end,
            });
        }

        let family = scan_family()?;
        let mut selector = ScanSelector;
        let mut cache = cuda_host::kernel_family::NoKernelSelectionCache;
        let selected = family
            .select(
                &problem,
                cuda_host::kernel_family::SelectionMode::Auto,
                &mut selector,
                &mut cache,
            )
            .map_err(|error| error.to_string())?;
        let kind = *selected.variant().entry();
        let tile_rows = selected.variant().metadata().tile_rows as usize;
        let lists = self.row_count.div_ceil(tile_rows);
        let partials = query_count
            .checked_mul(lists)
            .and_then(|value| value.checked_mul(neighbors))
            .ok_or("CUDA index scratch size overflow")?;
        self.scratch
            .ensure(&self.runtime.stream, query_len, partials)?;
        if let Some(values) = host {
            copy_prefix(&self.scratch.queries, values, &self.runtime.stream)?;
        } else {
            // SAFETY: device_posterior validates length and retains the DLPack
            // input until this stream has completed its device-to-device copy.
            unsafe {
                cuda_core::simt::memory::memcpy_dtod_async(
                    self.scratch.queries.cu_deviceptr(),
                    pointer,
                    query_len
                        .checked_mul(size_of::<f32>())
                        .ok_or("query byte size overflow")?,
                    self.runtime.stream.cu_stream(),
                )
                .map_err(cuda_error)?;
            }
        }

        let params = KnnParams {
            rows: to_u32(self.row_count, "index row count")?,
            dims: to_u32(self.dims, "index dimension")?,
            queries: to_u32(query_count, "index query count")?,
            lists: to_u32(lists, "index list count")?,
            neighbors: to_u32(neighbors, "index neighbor count")?,
            pad0: 0,
            pad1: 0,
            pad2: 0,
        };
        let total_lists = params
            .queries
            .checked_mul(params.lists)
            .ok_or("CUDA index grid size overflow")?;
        let start = timing_event(&self.runtime.stream)?;
        self.launch_scan(kind, total_lists, params)?;
        let scan_end = timing_event(&self.runtime.stream)?;
        let input_a = self.launch_merge(query_count, lists, neighbors)?;
        let merge_end = timing_event(&self.runtime.stream)?;
        self.runtime.context.check_err().map_err(cuda_error)?;
        Ok(SearchStage {
            plan_name: match kind {
                ScanKind::Rows => "cuda_rows",
                ScanKind::Warps => "cuda_warps",
            },
            input_a,
            queries: query_count,
            neighbors,
            start,
            scan_end,
            merge_end,
        })
    }

    fn read_neighbors(&self, stage: &SearchStage) -> CudaResult<(Vec<f32>, Vec<u32>)> {
        let output_count = stage
            .queries
            .checked_mul(stage.neighbors)
            .ok_or("CUDA index output size overflow")?;
        if stage.input_a {
            Ok((
                read_prefix(
                    &self.scratch.distances_a,
                    &self.runtime.stream,
                    output_count,
                )?,
                read_prefix(&self.scratch.indices_a, &self.runtime.stream, output_count)?,
            ))
        } else {
            Ok((
                read_prefix(
                    &self.scratch.distances_b,
                    &self.runtime.stream,
                    output_count,
                )?,
                read_prefix(&self.scratch.indices_b, &self.runtime.stream, output_count)?,
            ))
        }
    }

    pub(super) fn set_profile(&mut self, stage: &SearchStage, end: &CudaEvent) -> CudaResult<()> {
        self.profile = Some(KnnProfile {
            rows: self.row_count,
            queries: stage.queries,
            dims: self.dims,
            neighbors: stage.neighbors,
            scan_ms: stage
                .start
                .elapsed_ms(&stage.scan_end)
                .map_err(cuda_error)?,
            merge_ms: stage
                .scan_end
                .elapsed_ms(&stage.merge_end)
                .map_err(cuda_error)?,
            posterior_ms: stage.merge_end.elapsed_ms(end).map_err(cuda_error)?,
            total_ms: stage.start.elapsed_ms(end).map_err(cuda_error)?,
            kind: stage.plan_name,
        });
        Ok(())
    }

    fn launch_scan(&mut self, kind: ScanKind, lists: u32, params: KnnParams) -> CudaResult<()> {
        match kind {
            ScanKind::Rows => {
                let launch = self
                    .runtime
                    .module
                    .prepare_scan_rows(LaunchConfig1D::new(lists, THREADS, 0))
                    .map_err(cuda_error)?;
                self.runtime
                    .module
                    .scan_rows(
                        &self.runtime.stream,
                        &launch,
                        &self.rows,
                        &self.scratch.queries,
                        &mut self.scratch.distances_a,
                        &mut self.scratch.indices_a,
                        params,
                    )
                    .map_err(cuda_error)
            }
            ScanKind::Warps => {
                let launch = self
                    .runtime
                    .module
                    .prepare_scan_warps(LaunchConfig1D::new(lists, THREADS, 0))
                    .map_err(cuda_error)?;
                self.runtime
                    .module
                    .scan_warps(
                        &self.runtime.stream,
                        &launch,
                        &self.rows,
                        &self.scratch.queries,
                        &mut self.scratch.distances_a,
                        &mut self.scratch.indices_a,
                        params,
                    )
                    .map_err(cuda_error)
            }
        }
    }

    fn launch_merge(
        &mut self,
        queries: usize,
        mut input_lists: usize,
        neighbors: usize,
    ) -> CudaResult<bool> {
        let mut input_a = true;
        while input_lists > 1 {
            let output_lists = input_lists.div_ceil(2);
            let params = MergeParams {
                queries: to_u32(queries, "merge query count")?,
                input_lists: to_u32(input_lists, "merge input list count")?,
                output_lists: to_u32(output_lists, "merge output list count")?,
                neighbors: to_u32(neighbors, "merge neighbor count")?,
            };
            let blocks = params
                .queries
                .checked_mul(params.output_lists)
                .ok_or("CUDA merge grid size overflow")?;
            let launch = self
                .runtime
                .module
                .prepare_merge_topk(LaunchConfig1D::new(blocks, THREADS, 0))
                .map_err(cuda_error)?;
            if input_a {
                self.runtime
                    .module
                    .merge_topk(
                        &self.runtime.stream,
                        &launch,
                        &self.scratch.distances_a,
                        &self.scratch.indices_a,
                        &mut self.scratch.distances_b,
                        &mut self.scratch.indices_b,
                        params,
                    )
                    .map_err(cuda_error)?;
            } else {
                self.runtime
                    .module
                    .merge_topk(
                        &self.runtime.stream,
                        &launch,
                        &self.scratch.distances_b,
                        &self.scratch.indices_b,
                        &mut self.scratch.distances_a,
                        &mut self.scratch.indices_a,
                        params,
                    )
                    .map_err(cuda_error)?;
            }
            input_a = !input_a;
            input_lists = output_lists;
        }
        Ok(input_a)
    }

    fn check_rows(&self, rows: &[f32]) -> CudaResult<()> {
        if rows.len() % self.dims != 0 {
            Err("CUDA index rows do not match the dimension".to_string())
        } else if rows.iter().any(|value| !value.is_finite()) {
            Err("CUDA index rows must be finite".to_string())
        } else {
            Ok(())
        }
    }

    fn check_search(
        &self,
        queries: &[f32],
        query_count: usize,
        neighbors: usize,
    ) -> CudaResult<ScanProblem> {
        if self.is_empty() || query_count == 0 {
            return Err("CUDA index search requires rows and queries".to_string());
        }
        let expected = query_count
            .checked_mul(self.dims)
            .ok_or("CUDA index query size overflow")?;
        if queries.len() != expected || queries.iter().any(|value| !value.is_finite()) {
            return Err("CUDA index queries do not match the finite input shape".to_string());
        }
        let problem = ScanProblem {
            rows: self.row_count,
            dims: self.dims,
            queries: query_count,
            neighbors,
        };
        if self.mode == CudaKnnMode::Kernel {
            problem
                .validate(
                    scan_family()?
                        .variants()
                        .first()
                        .expect("CUDA scan family is non-empty"),
                )
                .map_err(|error| error.to_string())?;
        } else if neighbors == 0 || neighbors > self.max_neighbors() || neighbors > self.row_count {
            return Err("CUDA index neighbor count is unsupported".to_string());
        }
        Ok(problem)
    }
}

fn copy_at<T: DeviceCopy>(
    buffer: &DeviceBuffer<T>,
    offset: usize,
    values: &[T],
    stream: &CudaStream,
) -> CudaResult<()> {
    let end = offset
        .checked_add(values.len())
        .ok_or("CUDA copy range overflow")?;
    if end > buffer.len() {
        return Err("CUDA copy exceeds device allocation".to_string());
    }
    if values.is_empty() {
        return Ok(());
    }
    let byte_offset = offset
        .checked_mul(std::mem::size_of::<T>())
        .ok_or("CUDA copy byte offset overflow")?;
    unsafe {
        cuda_core::simt::memory::memcpy_htod_async(
            buffer.cu_deviceptr() + byte_offset as u64,
            values.as_ptr(),
            std::mem::size_of_val(values),
            stream.cu_stream(),
        )
        .map_err(cuda_error)
    }
}

pub(super) fn next_capacity(value: usize) -> CudaResult<usize> {
    value
        .max(1)
        .checked_next_power_of_two()
        .ok_or("CUDA index capacity overflow".to_string())
}
