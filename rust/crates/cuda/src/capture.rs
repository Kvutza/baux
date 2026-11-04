//! Optional evidence from the actual model loader path.
//!
//! A cubin record identifies loader input, not relocated device instructions or
//! a correctness proof. PTX still passes through the driver's unobserved JIT.
//! The pinned cuda-oxide IR loader hides its materialized image and routing
//! policy; preserve that loader and record only the IR input in that case.
//! Records precede graph capture/replay: they do not attest graph nodes, launch
//! arguments, buffer contents, execution, or coverage of other CUDA modules.

use baux::artifact::gpu::{self, Backend, Kind, Recipe, Stage};
use baux_cuda_device::{MODULE_NAME, fbt_model, trials};
use cuda_core::{CudaContext, CudaEvent, CudaStream, DeviceBuffer, DeviceCopy};
use cuda_host::embedded::{ArtifactPayloadKind, EmbeddedModuleError, OwnedArtifactBundle};
use std::{
    borrow::Cow,
    ffi::{c_char, c_int, c_void},
    sync::{Arc, OnceLock},
};

pub type CudaResult<T> = Result<T, String>;

pub fn target_name(context: &CudaContext) -> CudaResult<&'static str> {
    let (major, minor) = context.compute_capability().map_err(cuda_error)?;
    match (major, minor) {
        (7, 5) => Ok("sm_75"),
        (8, 0) => Ok("sm_80"),
        (9, 0) => Ok("sm_90a"),
        _ => Err(format!("unsupported compute capability {major}.{minor}")),
    }
}

pub fn device_target() -> CudaResult<&'static str> {
    let context = CudaContext::new(0).map_err(cuda_error)?;
    target_name(&context)
}

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
fn containing_binary() -> CudaResult<std::path::PathBuf> {
    let mut info = std::mem::MaybeUninit::<DlInfo>::zeroed();
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
    let bytes = unsafe { std::ffi::CStr::from_ptr(info.filename) }.to_bytes();
    Ok(std::path::PathBuf::from(
        <std::ffi::OsStr as std::os::unix::ffi::OsStrExt>::from_bytes(bytes),
    ))
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
                .find(|bundle| bundle.name == MODULE_NAME)
                .ok_or_else(|| {
                    format!(
                        "embedded CUDA module {:?} was not found in {}",
                        MODULE_NAME,
                        path.display()
                    )
                })
        })
        .as_ref()
        .map_err(Clone::clone)
}

pub(crate) fn load_module(context: &Arc<CudaContext>) -> CudaResult<trials::LoadedModule> {
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

pub(crate) struct Runtime {
    pub(crate) context: Arc<CudaContext>,
    pub(crate) stream: Arc<CudaStream>,
    pub(crate) module: trials::LoadedModule,
}

impl Runtime {
    pub(crate) fn new() -> CudaResult<Self> {
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

pub(crate) fn cuda_error(error: impl std::fmt::Display) -> String {
    error.to_string()
}

pub(crate) fn timing_event(stream: &CudaStream) -> CudaResult<CudaEvent> {
    stream
        .record_event(Some(cuda_core::sys::CUevent_flags_enum_CU_EVENT_DEFAULT))
        .map_err(cuda_error)
}

/// Upload a borrowed host slice. Completion precedes return, so callers may
/// immediately mutate or drop the source; raw async copies do not provide
/// that lifetime guarantee, even when pageable transfers often finish early.
pub(crate) fn copy_prefix<T: DeviceCopy>(
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
    if !Arc::ptr_eq(buffer.context(), stream.context()) {
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
            std::mem::size_of_val(values),
            stream.cu_stream(),
        )
    };
    let completed = stream.synchronize();
    enqueued.map_err(cuda_error)?;
    completed.map_err(cuda_error)
}

pub(crate) fn read_prefix<T: DeviceCopy>(
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
            len * std::mem::size_of::<T>(),
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

pub(crate) fn to_u32(value: usize, name: &str) -> CudaResult<u32> {
    value
        .try_into()
        .map_err(|_| format!("CUDA {name} exceeds u32"))
}

pub(crate) fn enabled() -> bool {
    std::env::var_os("BAUX_GPU_EVIDENCE").is_some()
}

/// cuda-core 0.3.1 otherwise adds this terminator inside its loader. Supplying
/// it explicitly lets the load record identify the complete driver input.
pub(crate) fn driver_image(bytes: &[u8]) -> Cow<'_, [u8]> {
    if bytes.last() == Some(&0) {
        Cow::Borrowed(bytes)
    } else {
        let mut image = Vec::with_capacity(bytes.len() + 1);
        image.extend_from_slice(bytes);
        image.push(0);
        Cow::Owned(image)
    }
}

/// Match cuda-oxide 6abfaa0's non-generic embedded loader payload priority.
pub(crate) fn load_model(context: &Arc<CudaContext>) -> CudaResult<fbt_model::LoadedModule> {
    if !enabled() {
        // SAFETY: generated bindings load their matching embedded artifact.
        return unsafe { fbt_model::load(context) }.map_err(cuda_error);
    }
    let bundle = cuda_host::embedded::artifact_bundles_from_current_exe()
        .map_err(cuda_error)?
        .into_iter()
        .find(|bundle| bundle.name == MODULE_NAME)
        .ok_or_else(|| format!("embedded CUDA module '{MODULE_NAME}' was not found"))?;
    let mut options = vec![bundle.compile_options.sidecar_text().to_string()];
    if let Some(target) = std::env::var_os("CUDA_OXIDE_TARGET") {
        options.push(format!("CUDA_OXIDE_TARGET={}", target.to_string_lossy()));
    }
    // fbt_model has no generic kernels. Its generated load_named() selects the
    // first MODULE_NAME bundle, Cubin before PTX, exactly as above/below.
    // Keep calls to its original load() for IR and the disabled fast path: this
    // also retains the generated artifact-anchor references at link time.
    for (payload, kind) in [
        (ArtifactPayloadKind::Cubin, Kind::Cubin),
        (ArtifactPayloadKind::Ptx, Kind::Ptx),
    ] {
        if let Some(image) = bundle.payload(payload) {
            let image = driver_image(image);
            let module = context.load_module_from_image(&image).map_err(cuda_error)?;
            // SAFETY: this is the original matching bundle and unchanged image
            // selection used by generated load(); no symbol-only substitution.
            let module = unsafe { fbt_model::from_module(module) }.map_err(cuda_error)?;
            let recipe = Recipe {
                backend: Backend::Cuda,
                device: context.device_name().map_err(cuda_error)?,
                target: bundle.target.clone(),
                compiler: "embedded producer unreported; cuda-oxide 6abfaa0; CUDA driver version \
                           unreported"
                    .into(),
                stage: Stage::Load,
                entry: bundle.name.clone(),
                input_kind: kind,
                output_kind: None,
                options,
                constants: Vec::new(),
            };
            gpu::record_env(&recipe, &image, None)?;
            return Ok(module);
        }
    }
    // Never reproduce the private architecture/forward-compatibility policy.
    // Successful load entails successful IR materialization, but does not give
    // access to that output. No Load record or invented cubin is emitted here.
    // SAFETY: the generated loader retains its matching artifact and ABI.
    let module = unsafe { fbt_model::load(context) }.map_err(cuda_error)?;
    for (payload, kind, stage) in [
        (ArtifactPayloadKind::NvvmIr, Kind::NvvmIr, Stage::Compile),
        (ArtifactPayloadKind::Ltoir, Kind::LtoIr, Stage::Link),
    ] {
        if let Some(ir) = bundle.payload(payload) {
            let recipe = Recipe {
                backend: Backend::Cuda,
                device: context.device_name().map_err(cuda_error)?,
                target: bundle.target.clone(),
                compiler: "cuda-oxide 6abfaa0; IR toolchain version unreported; materialized \
                           loader input unobserved"
                    .into(),
                stage,
                entry: bundle.name.clone(),
                input_kind: kind,
                output_kind: None,
                options,
                constants: Vec::new(),
            };
            gpu::record_env(&recipe, ir, None)?;
            return Ok(module);
        }
    }
    Err("CUDA model loaded without a recognized evidence payload".into())
}
