extern crate self as baux;

#[path = "../../../rust/crates/baux/artifact/src/lib.rs"]
pub mod artifact;
pub use baux_cuda_device::semantics;

use std::ffi::{CStr, OsStr, c_char, c_int, c_void};
use std::mem::{MaybeUninit, size_of, size_of_val};
use std::os::unix::ffi::OsStrExt;
use std::path::PathBuf;
use std::sync::{Arc, OnceLock};

use baux_cuda_device::trials;
pub use baux_cuda_device::{
    BatchValue, Bf16Change, Bf16Leaf, Bf16Score, CenterStep, DenseLeaf, DenseLinearParams,
    DenseTerm, DenseTile, FbtShape, FeedbackShape, KNN_K, Leaf, MAX_DEPTH, MAX_HISTORY,
    MatmulShape, MoeShape, ObjectiveShape, PisaShape, RouteShape, RoutedTile, ScoreStats,
    SearchState, Seed, Selection, SparseEdit, THREADS, TellParams, TellSummary, Tile, fbt_model,
};
use cuda_core::{
    CudaContext, CudaEvent, CudaStream, DeviceBuffer, DeviceCopy, IntoResult, LaunchConfig1D,
    simt::LaunchConfig,
};
use cuda_host::embedded::{ArtifactPayloadKind, EmbeddedModuleError, OwnedArtifactBundle};

pub type CudaResult<T> = Result<T, String>;

pub fn target_name(context: &CudaContext) -> CudaResult<&'static str> {
    let (major, minor) = context.compute_capability().map_err(cuda_error)?;
    ptx_synth::TargetArch::from_compute(major, minor)
        .map(|target| target.ptx_target())
        .ok_or_else(|| format!("unsupported compute capability {major}.{minor}"))
}

pub fn device_target() -> CudaResult<&'static str> {
    let context = CudaContext::new(0).map_err(cuda_error)?;
    target_name(&context)
}

pub use baux::semantics::optimizer::calibration;
mod capture;
pub mod gemm;
mod graph;
mod mhc4_probe;
pub mod muon;
pub mod neighbors;
mod precision;

/// Verify column-major checkpoint packing and token-major device lookup with
/// values that expose token/column transposition errors.
pub fn embed_parity() -> CudaResult<()> {
    const ROWS: usize = 3;
    const WIDTH: usize = 17;
    const VOCAB: usize = 19;
    let context = CudaContext::new(0).map_err(cuda_error)?;
    let stream = context.default_stream();
    let module = crate::capture::load_model(&context)?;
    let checkpoint = (0..WIDTH * VOCAB)
        .map(|index| index as u16)
        .collect::<Vec<_>>();
    let packed = prefill::pack_embedding(&checkpoint, WIDTH, VOCAB);
    let tokens = [3_u32, 7, 18];
    let weights = DeviceBuffer::from_host(&stream, &packed).map_err(cuda_error)?;
    let tokens_device = DeviceBuffer::from_host(&stream, &tokens).map_err(cuda_error)?;
    let mut output = DeviceBuffer::zeroed(&stream, ROWS * WIDTH).map_err(cuda_error)?;
    let shape =
        FbtShape::new(ROWS as u32, WIDTH as u32, VOCAB as u32, 1.0e-5).map_err(str::to_string)?;
    let launch = module
        .prepare_embed_packed(LaunchConfig1D::new(ROWS as u32, 256, 0))
        .map_err(cuda_error)?;
    module
        .embed_packed(
            &stream,
            &launch,
            &weights,
            &tokens_device,
            &mut output,
            shape,
        )
        .map_err(cuda_error)?;
    context.check_err().map_err(cuda_error)?;
    let actual = read_prefix(&output, &stream, ROWS * WIDTH)?;
    let mut expected = Vec::with_capacity(ROWS * WIDTH);
    for token in tokens {
        for column in 0..WIDTH {
            expected.push(checkpoint[column * VOCAB + token as usize]);
        }
    }
    if actual != expected {
        return Err("CUDA packed embedding parity mismatch".into());
    }
    Ok(())
}

/// Exercise the CUDA feedback primitives on-device with exact FP16 fixtures.
pub fn feedback_parity() -> CudaResult<()> {
    let context = CudaContext::new(0).map_err(cuda_error)?;
    let stream = context.default_stream();
    let module = crate::capture::load_model(&context)?;
    let mut shape = FeedbackShape::fbt(4, 2).map_err(str::to_string)?;
    shape.token_unit_rms = 0;
    shape.fused_unit_rms = 0;
    let elements = (shape.rows * shape.width) as usize;
    let history = vec![0x3c00_u16; elements];
    let tokens = vec![0x4000_u16; elements];
    let value = vec![0x4000_u16; elements];
    let logit = vec![0_u16; elements];
    let history = DeviceBuffer::from_host(&stream, &history).map_err(cuda_error)?;
    let tokens = DeviceBuffer::from_host(&stream, &tokens).map_err(cuda_error)?;
    let value = DeviceBuffer::from_host(&stream, &value).map_err(cuda_error)?;
    let logit = DeviceBuffer::from_host(&stream, &logit).map_err(cuda_error)?;
    let mut previous = DeviceBuffer::zeroed(&stream, elements).map_err(cuda_error)?;
    let mut normalized = DeviceBuffer::zeroed(&stream, elements).map_err(cuda_error)?;
    let mut fused = DeviceBuffer::zeroed(&stream, shape.rows as usize).map_err(cuda_error)?;
    let mut output = DeviceBuffer::zeroed(&stream, elements).map_err(cuda_error)?;
    let shift_launch = module
        .prepare_feedback_shift(LaunchConfig1D::new(shape.rows, 256, 0))
        .map_err(cuda_error)?;
    let norm_launch = module
        .prepare_feedback_norm(LaunchConfig1D::new(shape.rows, 256, 0))
        .map_err(cuda_error)?;
    let combine_launch = module
        .prepare_feedback_combine(LaunchConfig1D::new(shape.rows, 256, 0))
        .map_err(cuda_error)?;
    // Generated launch contracts validate the shapes and disjoint outputs.
    module
        .feedback_shift(
            &stream,
            &shift_launch,
            &history,
            &mut previous,
            &mut fused,
            shape,
        )
        .map_err(cuda_error)?;
    shape.token_unit_rms = 1;
    module
        .feedback_norm(&stream, &norm_launch, &history, &mut normalized, shape)
        .map_err(cuda_error)?;
    shape.token_unit_rms = 0;
    module
        .feedback_combine(
            &stream,
            &combine_launch,
            &value,
            &logit,
            &tokens,
            &fused,
            &mut output,
            shape,
        )
        .map_err(cuda_error)?;
    context.check_err().map_err(cuda_error)?;
    let previous = read_prefix(&previous, &stream, elements)?;
    let normalized = read_prefix(&normalized, &stream, elements)?;
    let fused = read_prefix(&fused, &stream, shape.rows as usize)?;
    let output = read_prefix(&output, &stream, elements)?;
    if fused != [0, 1, 0, 1]
        || previous[..shape.width as usize].iter().any(|&v| v != 0)
        || previous[shape.width as usize..2 * shape.width as usize]
            .iter()
            .any(|&v| v != 0x3c00)
        || normalized.iter().any(|&v| v != 0x3c00)
        || output[..shape.width as usize].iter().any(|&v| v != 0x4000)
        || output[shape.width as usize..2 * shape.width as usize]
            .iter()
            .any(|&v| v != 0x3c00)
    {
        return Err("CUDA feedback parity mismatch".into());
    }
    Ok(())
}

#[cfg(feature = "native-flame")]
pub mod flame;

mod bf16_search;
pub use bf16_search::{Bf16SearchEngine, ProposalDescription, TellOutput};
pub mod context;
mod dense_engine;
pub use dense_engine::{Bf16Engine, DenseLinearEngine, dense_apply};
mod knn;
mod model;
mod prefill;
mod trial_engine;
pub use knn::{
    BatchOutput, BatchSpec, CudaIndex, DeviceDraw, DevicePosterior, DrawOutput, KnnProfile,
    PosteriorOutput, PosteriorSpec, WeightedOutput, WeightedSpec,
};
pub use model::run::initialized_check;
pub use model::run::{
    bench as generation_bench, check as model_check, generate as model_generate, mhc4_bench,
    mhc4_bo, run as model_run,
};
pub use model::weights::proposal_check;
pub use model::{FbtModel, GenerationOutput, ModelOutput, WaveProfile};
pub use prefill::{FbtPrefill, FbtPrefillProfile};
pub use trial_engine::TrialEngine;

static TRIAL_BUNDLE: OnceLock<Result<OwnedArtifactBundle, String>> = OnceLock::new();
#[repr(C)]
struct DlInfo {
    filename: *const c_char,
    _base: *mut c_void,
    _symbol: *const c_char,
    _symbol_address: *mut c_void,
}

#[link(name = "dl")]
unsafe extern "C" {
    fn dladdr(address: *const c_void, info: *mut DlInfo) -> c_int;
}

#[inline(never)]
fn containing_binary() -> CudaResult<PathBuf> {
    let mut info = MaybeUninit::<DlInfo>::zeroed();
    // SAFETY: `info` is writable and the function address remains valid while
    // its executable or shared library is loaded.
    if unsafe {
        dladdr(
            containing_binary as *const () as *const c_void,
            info.as_mut_ptr(),
        )
    } == 0
    {
        return Err("dladdr could not locate the Baux CUDA host library".to_string());
    }
    // SAFETY: a successful dladdr call initialized `info`.
    let info = unsafe { info.assume_init() };
    if info.filename.is_null() {
        return Err("dladdr returned no Baux CUDA host library path".to_string());
    }
    // SAFETY: dladdr owns a NUL-terminated filename for the loaded object.
    let bytes = unsafe { CStr::from_ptr(info.filename) }.to_bytes();
    Ok(PathBuf::from(OsStr::from_bytes(bytes)))
}

fn trial_bundle() -> CudaResult<&'static OwnedArtifactBundle> {
    TRIAL_BUNDLE
        .get_or_init(|| {
            let path = containing_binary()?;
            cuda_host::embedded::artifact_bundles_from_binary_path(&path)
                .map_err(|error| {
                    format!(
                        "failed reading CUDA artifacts from {}: {error}",
                        path.display()
                    )
                })?
                .into_iter()
                .find(|bundle| bundle.name == baux_cuda_device::MODULE_NAME)
                .ok_or_else(|| {
                    format!(
                        "embedded CUDA module {:?} was not found in {}",
                        baux_cuda_device::MODULE_NAME,
                        path.display()
                    )
                })
        })
        .as_ref()
        .map_err(Clone::clone)
}

fn load_module(context: &Arc<CudaContext>) -> CudaResult<trials::LoadedModule> {
    // SAFETY: this crate launches only the embedded module generated from the
    // matching kernel crate and uses checked prepared launches below.
    match unsafe { trials::load(context) } {
        Ok(module) => return Ok(module),
        Err(EmbeddedModuleError::ModuleNotFound { .. } | EmbeddedModuleError::NoModules) => {}
        Err(error) => return Err(cuda_error(error)),
    }

    let bundle = trial_bundle()?;
    let image = if let Some(cubin) = bundle.payload(ArtifactPayloadKind::Cubin) {
        cubin.to_vec()
    } else if let Some(ptx) = bundle.payload(ArtifactPayloadKind::Ptx) {
        ptx.to_vec()
    } else if let Some(nvvm_ir) = bundle.payload(ArtifactPayloadKind::NvvmIr) {
        cuda_host::ltoir::build_cubin_from_nvvm_ir_with_compile_options(
            nvvm_ir,
            &bundle.name,
            &bundle.target,
            bundle.compile_options,
        )
        .map_err(cuda_error)?
    } else if let Some(ltoir) = bundle.payload(ArtifactPayloadKind::Ltoir) {
        cuda_host::ltoir::link_ltoir_to_cubin_with_compile_options(
            ltoir,
            &bundle.name,
            &bundle.target,
            bundle.compile_options,
        )
        .map_err(cuda_error)?
    } else {
        return Err(format!(
            "embedded CUDA module {:?} has no supported payload",
            bundle.name
        ));
    };
    let module = context.load_module_from_image(&image).map_err(cuda_error)?;
    // SAFETY: the fallback image is extracted from the same embedded kernel
    // bundle and therefore has the generated module's exact ABI.
    unsafe { trials::from_module(module) }.map_err(cuda_error)
}

struct Runtime {
    context: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    module: trials::LoadedModule,
}

impl Runtime {
    fn new() -> CudaResult<Self> {
        let context = CudaContext::new(0).map_err(cuda_error)?;
        let stream = context.default_stream();
        let module = load_module(&context)?;
        Ok(Self {
            context,
            stream,
            module,
        })
    }
}

#[derive(Debug, Clone, Copy)]
pub struct RoundConfig {
    pub neighbors: usize,
    pub acquisition: u32,
    pub epistemic_scale: f32,
    pub aleatoric_scale: f32,
    pub y_scale: f32,
    pub beta: f32,
    pub adaptive_neighbors: bool,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct ProposalProfile {
    pub score_ms: f32,
    pub pick_ms: f32,
    pub materialize_ms: f32,
    pub total_ms: f32,
}

struct ProposalEvents {
    score_start: CudaEvent,
    score_end: CudaEvent,
    pick_end: CudaEvent,
    materialize_end: Option<CudaEvent>,
}

struct Scratch {
    history_capacity: usize,
    candidate_capacity: usize,
    center_capacity: usize,
    edit_capacity: usize,
    history_slots: DeviceBuffer<u32>,
    outcomes: DeviceBuffer<f32>,
    seeds: DeviceBuffer<Seed>,
    draws: DeviceBuffer<f32>,
    scores: DeviceBuffer<f32>,
    selection: DeviceBuffer<Selection>,
    centers: DeviceBuffer<CenterStep>,
    candidate_centers: DeviceBuffer<u32>,
    edits: DeviceBuffer<SparseEdit>,
    base_distances: DeviceBuffer<f32>,
}

impl Scratch {
    fn new(stream: &CudaStream) -> CudaResult<Self> {
        Ok(Self {
            history_capacity: 1,
            candidate_capacity: 1,
            center_capacity: 1,
            edit_capacity: 1,
            history_slots: DeviceBuffer::zeroed(stream, 1).map_err(cuda_error)?,
            outcomes: DeviceBuffer::zeroed(stream, 1).map_err(cuda_error)?,
            seeds: DeviceBuffer::zeroed(stream, 1).map_err(cuda_error)?,
            draws: DeviceBuffer::zeroed(stream, 1).map_err(cuda_error)?,
            scores: DeviceBuffer::zeroed(stream, 1).map_err(cuda_error)?,
            selection: DeviceBuffer::zeroed(stream, 1).map_err(cuda_error)?,
            centers: DeviceBuffer::zeroed(stream, 1).map_err(cuda_error)?,
            candidate_centers: DeviceBuffer::zeroed(stream, 1).map_err(cuda_error)?,
            edits: DeviceBuffer::zeroed(stream, 1).map_err(cuda_error)?,
            base_distances: DeviceBuffer::zeroed(stream, 1).map_err(cuda_error)?,
        })
    }

    fn ensure(&mut self, stream: &CudaStream, history: usize, candidates: usize) -> CudaResult<()> {
        let history_capacity = history
            .checked_next_power_of_two()
            .ok_or("CUDA trial history capacity overflow")?;
        let candidate_capacity = candidates
            .checked_next_power_of_two()
            .ok_or("CUDA trial candidate capacity overflow")?;
        if history_capacity > self.history_capacity {
            self.history_slots =
                DeviceBuffer::zeroed(stream, history_capacity).map_err(cuda_error)?;
            self.outcomes = DeviceBuffer::zeroed(stream, history_capacity).map_err(cuda_error)?;
            self.draws = DeviceBuffer::zeroed(stream, history_capacity).map_err(cuda_error)?;
            self.base_distances =
                DeviceBuffer::zeroed(stream, history_capacity).map_err(cuda_error)?;
            self.history_capacity = history_capacity;
        }
        if candidate_capacity > self.candidate_capacity {
            self.seeds = DeviceBuffer::zeroed(stream, candidate_capacity).map_err(cuda_error)?;
            self.scores = DeviceBuffer::zeroed(stream, candidate_capacity).map_err(cuda_error)?;
            self.selection =
                DeviceBuffer::zeroed(stream, candidate_capacity).map_err(cuda_error)?;
            self.candidate_centers =
                DeviceBuffer::zeroed(stream, candidate_capacity).map_err(cuda_error)?;
            self.candidate_capacity = candidate_capacity;
        }
        Ok(())
    }

    fn ensure_edits(&mut self, stream: &CudaStream, edits: usize) -> CudaResult<()> {
        let capacity = edits
            .checked_next_power_of_two()
            .ok_or("CUDA sparse edit capacity overflow")?;
        if capacity > self.edit_capacity {
            self.edits = DeviceBuffer::zeroed(stream, capacity).map_err(cuda_error)?;
            self.edit_capacity = capacity;
        }
        Ok(())
    }

    fn ensure_centers(&mut self, stream: &CudaStream, centers: usize) -> CudaResult<()> {
        let center_capacity = centers
            .max(1)
            .checked_next_power_of_two()
            .ok_or("CUDA trial center capacity overflow")?;
        if center_capacity > self.center_capacity {
            self.centers = DeviceBuffer::zeroed(stream, center_capacity).map_err(cuda_error)?;
            self.center_capacity = center_capacity;
        }
        Ok(())
    }
}

impl ProposalEvents {
    fn profile(&self) -> CudaResult<ProposalProfile> {
        let end = self.materialize_end.as_ref().unwrap_or(&self.pick_end);
        Ok(ProposalProfile {
            score_ms: self
                .score_start
                .elapsed_ms(&self.score_end)
                .map_err(cuda_error)?,
            pick_ms: self
                .score_end
                .elapsed_ms(&self.pick_end)
                .map_err(cuda_error)?,
            materialize_ms: self
                .materialize_end
                .as_ref()
                .map(|materialize_end| self.pick_end.elapsed_ms(materialize_end))
                .transpose()
                .map_err(cuda_error)?
                .unwrap_or(0.0),
            total_ms: self.score_start.elapsed_ms(end).map_err(cuda_error)?,
        })
    }
}

fn timing_event(stream: &CudaStream) -> CudaResult<CudaEvent> {
    stream
        .record_event(Some(cuda_core::sys::CUevent_flags_enum_CU_EVENT_DEFAULT))
        .map_err(cuda_error)
}

fn set_profile(enabled: bool, profiling: &mut bool, profile: &mut Option<ProposalProfile>) {
    *profiling = enabled;
    if !enabled {
        *profile = None;
    }
}

fn sync_stream(runtime: &Runtime, stream: Option<i64>) -> CudaResult<()> {
    let Some(stream) = stream else {
        return runtime.stream.synchronize().map_err(cuda_error);
    };
    if stream == -1 {
        return Ok(());
    }
    let event = runtime.stream.record_event(None).map_err(cuda_error)?;
    runtime.context.bind_to_thread().map_err(cuda_error)?;
    let stream = stream as usize as cuda_core::sys::CUstream;
    unsafe {
        cuda_core::sys::cuStreamWaitEvent(
            stream,
            event.cu_event(),
            cuda_core::sys::CUevent_wait_flags_enum_CU_EVENT_WAIT_DEFAULT,
        )
        .result()
        .map_err(cuda_error)
    }
}

const DENSE_ELEMENTS: usize = 65_536;

/// Upload a borrowed host slice. Completion precedes return, so callers may
/// immediately mutate or drop the source; raw async copies do not provide
/// that lifetime guarantee, even when pageable transfers often finish early.
fn copy_prefix<T: DeviceCopy>(
    buffer: &DeviceBuffer<T>,
    values: &[T],
    stream: &CudaStream,
) -> CudaResult<()> {
    if values.len() > buffer.len() {
        return Err(format!(
            "CUDA input has {} elements, buffer capacity is {}",
            values.len(),
            buffer.len()
        ));
    }
    if values.is_empty() {
        return Ok(());
    }
    if !std::sync::Arc::ptr_eq(buffer.context(), stream.context()) {
        return Err("CUDA upload buffer and stream have different contexts".into());
    }
    stream.context().bind_to_thread().map_err(cuda_error)?;
    // SAFETY: checked capacity/context, and the borrowed source remains alive
    // through the stream synchronization below, including the enqueue-error
    // path.
    let enqueued = unsafe {
        cuda_core::simt::memory::memcpy_htod_async(
            buffer.cu_deviceptr(),
            values.as_ptr(),
            size_of_val(values),
            stream.cu_stream(),
        )
    };
    let completed = stream.synchronize();
    enqueued.map_err(cuda_error)?;
    completed.map_err(cuda_error)
}

fn read_prefix<T: DeviceCopy>(
    buffer: &DeviceBuffer<T>,
    stream: &CudaStream,
    len: usize,
) -> CudaResult<Vec<T>> {
    if len > buffer.len() {
        return Err(format!(
            "CUDA output requests {len} elements from capacity {}",
            buffer.len()
        ));
    }
    let mut output = Vec::<T>::with_capacity(len);
    if len == 0 {
        return Ok(output);
    }
    unsafe {
        cuda_core::simt::memory::memcpy_dtoh_async(
            output.as_mut_ptr(),
            buffer.cu_deviceptr(),
            len * size_of::<T>(),
            stream.cu_stream(),
        )
        .map_err(cuda_error)?;
    }
    stream.synchronize().map_err(cuda_error)?;
    unsafe {
        output.set_len(len);
    }
    Ok(output)
}

fn to_u32(value: usize, name: &str) -> CudaResult<u32> {
    value
        .try_into()
        .map_err(|_| format!("CUDA {name} exceeds u32"))
}

fn validate_centers(
    centers: &[CenterStep],
    region_centers: &[u32],
    regions: usize,
) -> CudaResult<()> {
    if centers.is_empty() {
        if !region_centers.is_empty() {
            return Err("CUDA root-based search must not provide region centers".to_string());
        }
        return Ok(());
    }
    if region_centers.len() != regions {
        return Err(format!(
            "CUDA expected {regions} region centers, got {}",
            region_centers.len()
        ));
    }
    for (index, center) in centers.iter().enumerate() {
        if center.parent != u32::MAX && center.parent as usize >= index {
            return Err(format!(
                "CUDA center {index} must reference an earlier parent"
            ));
        }
        let mut depth = 1_usize;
        let mut parent = center.parent;
        while parent != u32::MAX {
            depth += 1;
            if depth > MAX_DEPTH {
                return Err(format!("CUDA center chain exceeds depth {MAX_DEPTH}"));
            }
            parent = centers[parent as usize].parent;
        }
    }
    if region_centers
        .iter()
        .any(|&center| center as usize >= centers.len())
    {
        return Err("CUDA region center index is out of bounds".to_string());
    }
    Ok(())
}

fn trial_leaves(row_bytes: usize, leaves: &[Leaf]) -> CudaResult<()> {
    if leaves.is_empty() || row_bytes == 0 || row_bytes > u32::MAX as usize {
        return Err("CUDA trial layout requires non-empty u32-sized rows and leaves".to_string());
    }
    let mut expected_byte_offset = 0usize;
    let mut expected_element_offset = 0u32;
    for (index, leaf) in leaves.iter().enumerate() {
        if leaf.length == 0 || !matches!(leaf.bits, 4 | 8) {
            return Err(format!("CUDA trial leaf {index} has an invalid shape"));
        }
        let encoding_matches = matches!((leaf.bits, leaf.encoding), (4, 0 | 2) | (8, 1 | 3 | 4));
        if !encoding_matches
            || !leaf.scale.is_finite()
            || leaf.scale <= 0.0
            || !leaf.weight.is_finite()
            || leaf.weight <= 0.0
        {
            return Err(format!(
                "CUDA trial leaf {index} has an invalid encoding or scale"
            ));
        }
        if leaf.byte_offset as usize != expected_byte_offset
            || leaf.element_offset != expected_element_offset
        {
            return Err(format!("CUDA trial leaf {index} is not contiguous"));
        }
        let bytes = if leaf.bits == 4 {
            leaf.length.div_ceil(2)
        } else {
            leaf.length
        };
        expected_byte_offset = expected_byte_offset
            .checked_add(bytes as usize)
            .ok_or("CUDA trial row byte count overflow")?;
        expected_element_offset = expected_element_offset
            .checked_add(leaf.length)
            .ok_or("CUDA trial element count overflow")?;
        let max_code = (1u32 << leaf.bits) - 1;
        if leaf.whole > max_code {
            return Err(format!(
                "CUDA trial leaf {index} perturbation exceeds its encoding"
            ));
        }
    }
    if expected_byte_offset != row_bytes {
        return Err(format!(
            "CUDA trial leaves cover {expected_byte_offset} bytes, expected {row_bytes}"
        ));
    }
    Ok(())
}

fn trial_layout(row_bytes: usize, leaves: &[Leaf], tiles: &[Tile]) -> CudaResult<()> {
    trial_leaves(row_bytes, leaves)?;
    if tiles.is_empty() {
        return Err("CUDA trial layout requires at least one tile".to_string());
    }
    let mut covered = vec![0u32; leaves.len()];
    for (index, tile) in tiles.iter().enumerate() {
        let leaf_index = tile.leaf as usize;
        let Some(leaf) = leaves.get(leaf_index) else {
            return Err(format!(
                "CUDA trial tile {index} references an invalid leaf"
            ));
        };
        if tile.length == 0 || tile.start != covered[leaf_index] {
            return Err(format!(
                "CUDA trial tile {index} is empty, overlapping, or out of order"
            ));
        }
        let end = tile
            .start
            .checked_add(tile.length)
            .ok_or("CUDA trial tile extent overflow")?;
        if end > leaf.length || (leaf.bits == 4 && tile.start & 1 != 0) {
            return Err(format!(
                "CUDA trial tile {index} exceeds or misaligns its leaf"
            ));
        }
        covered[leaf_index] = end;
    }
    if covered
        .iter()
        .zip(leaves)
        .any(|(&length, leaf)| length != leaf.length)
    {
        return Err("CUDA trial tiles do not cover every leaf exactly once".to_string());
    }
    Ok(())
}

fn cuda_error(error: impl std::fmt::Display) -> String {
    error.to_string()
}

pub use model::diffusion_run::{check as diffusion_check, run as model_diffusion};
