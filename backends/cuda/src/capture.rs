//! Optional evidence from the actual model and generated GEMM loader paths.
//!
//! A cubin record identifies loader input, not relocated device instructions or
//! a correctness proof. PTX still passes through the driver's unobserved JIT.
//! The pinned cuda-oxide IR loader hides its materialized image and routing
//! policy; preserve that loader and record only the IR input in that case.
//! Records precede graph capture/replay: they do not attest graph nodes, launch
//! arguments, buffer contents, execution, or coverage of other CUDA modules.

use crate::{CudaResult, cuda_error};
use baux::artifact::gpu::{self, Backend, Kind, Recipe, Stage};
use baux_cuda_device::{MODULE_NAME, fbt_model};
use cuda_core::CudaContext;
use cuda_host::embedded::{ArtifactPayloadKind, artifact_bundles_from_current_exe};
use std::{borrow::Cow, sync::Arc};

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
    let bundle = artifact_bundles_from_current_exe()
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

/// Record the PTX compilation and the identical cubin passed to the driver.
pub(crate) fn assembled(
    context: &CudaContext,
    entry: String,
    target: String,
    ptx: &[u8],
    cubin: &[u8],
    options: Vec<String>,
) -> CudaResult<()> {
    let mut recipe = Recipe {
        backend: Backend::Cuda,
        device: context.device_name().map_err(cuda_error)?,
        target,
        compiler: "ptxas resolved through PATH; executable identity/version unreported".into(),
        stage: Stage::Compile,
        entry,
        input_kind: Kind::Ptx,
        output_kind: Some(Kind::Cubin),
        options,
        constants: Vec::new(),
    };
    gpu::record_env(&recipe, ptx, Some(cubin))?;
    recipe.compiler = "CUDA driver version unreported".into();
    recipe.stage = Stage::Load;
    recipe.input_kind = Kind::Cubin;
    recipe.output_kind = None;
    recipe.options.clear();
    gpu::record_env(&recipe, &driver_image(cubin), None)?;
    Ok(())
}
