//! Immutable ENN posterior snapshots whose arrays remain on the selected
//! device.
#[cfg(any(
    all(target_os = "macos", feature = "metal"),
    all(target_os = "linux", target_arch = "x86_64", feature = "cuda"),
    target_arch = "wasm32"
))]
use crate::knn::KnnBackend;
use crate::{ENN, ENNParams, PosteriorFlags};

enum Storage {
    #[cfg(all(target_os = "macos", feature = "metal"))]
    Metal(crate::knn::metal_index::MetalPosterior),
    #[cfg(all(target_os = "linux", target_arch = "x86_64", feature = "cuda"))]
    Cuda(baux_cuda::DevicePosterior),
    #[cfg(target_arch = "wasm32")]
    WebGpu(crate::webgpu::WebGpuPosterior),
    #[allow(dead_code)]
    Unavailable,
}

pub struct Posterior {
    storage: Storage,
    pub rows: usize,
    pub metrics: usize,
}

pub enum PosteriorDraw {
    #[cfg(all(target_os = "macos", feature = "metal"))]
    Metal(metal::Buffer),
    #[cfg(all(target_os = "linux", target_arch = "x86_64", feature = "cuda"))]
    Cuda(baux_cuda::DeviceDraw),
    #[cfg(target_arch = "wasm32")]
    WebGpu(Vec<f32>),
    #[allow(dead_code)]
    Unavailable,
}

pub type DevicePosterior = Posterior;
pub type DeviceDraw = PosteriorDraw;

impl ENN {
    /// Evaluate contiguous FP32 device queries. Training observations are
    /// synced from the ENN; query tensors, posterior arrays, and draws stay
    /// on-device.
    ///
    /// # Safety
    /// pointer is a retained CUDA allocation (device 2), MTLBuffer (device 8),
    /// or WebGPU handle / linear-memory pointer (device 15) covering
    /// rows*num_dim FP32 values at byte_offset, ready for reading.
    pub unsafe fn device_posterior(
        &self,
        pointer: u64,
        byte_offset: usize,
        device: i32,
        rows: usize,
        params: &ENNParams,
        flags: &PosteriorFlags,
    ) -> Result<DevicePosterior, String> {
        #[cfg(not(any(
            all(target_os = "macos", feature = "metal"),
            all(target_os = "linux", target_arch = "x86_64", feature = "cuda"),
            target_arch = "wasm32"
        )))]
        {
            let _ = (pointer, byte_offset, device, rows, params, flags);
            Err("device posterior requires a compiled CUDA, Metal, or WebGPU backend".into())
        }
        #[cfg(any(
            all(target_os = "macos", feature = "metal"),
            all(target_os = "linux", target_arch = "x86_64", feature = "cuda"),
            target_arch = "wasm32"
        ))]
        {
            if self.scale_x || self.bounded_outputs() {
                return Err(
                    "device posterior currently requires scale_x=false and unbounded outputs"
                        .into(),
                );
            }
            if self.num_obs() == 0 || rows == 0 {
                return Err("device posterior requires observations and queries".into());
            }
            let skip = usize::from(flags.exclude_nearest);
            let k = (params.k_neighbors as usize).min(self.num_obs().saturating_sub(skip));
            if k == 0 {
                return Err("no neighbors remain after exclusion".into());
            }
            self.ensure_sync().map_err(|e| e.to_string())?;
            let index = self
                .backend
                .memory_index()
                .ok_or("device posterior requires in-memory observations")?;
            let outcomes = self
                .y_opt()
                .ok_or("device posterior requires in-memory outcomes")?;
            let y = crate::knn::flatten_f32(&outcomes);
            let var = self.backend.memory_yvar();
            let var = var.as_ref().map(crate::knn::flatten_f32);
            let scales = self
                .output_scale()
                .iter()
                .map(|&v| v as f32)
                .collect::<Vec<_>>();
            let storage = match (&*index.inner, device) {
                #[cfg(all(target_os = "macos", feature = "metal"))]
                (KnnBackend::Metal(inner), 8) => Storage::Metal(unsafe {
                    inner
                        .lock()
                        .map_err(|_| "Metal index lock poisoned")?
                        .device_posterior(
                            pointer,
                            byte_offset,
                            rows,
                            &y,
                            var.as_deref(),
                            &scales,
                            k,
                            skip,
                            params.epistemic_scale as f32,
                            params.aleatoric_scale as f32,
                            flags.observation_noise,
                        )?
                }),
                #[cfg(all(target_os = "linux", target_arch = "x86_64", feature = "cuda"))]
                (KnnBackend::Cuda(inner), 2) => {
                    let spec = baux_cuda::WeightedSpec {
                        metrics: self.num_metrics(),
                        input_k: k + skip,
                        used_k: k,
                        skip,
                        epistemic_scale: params.epistemic_scale as f32,
                        aleatoric_scale: params.aleatoric_scale as f32,
                        epsilon: crate::error::EPS_VAR as f32,
                        observation_noise: flags.observation_noise,
                    };
                    Storage::Cuda(unsafe {
                        inner
                            .lock()
                            .map_err(|_| "CUDA index lock poisoned")?
                            .inner
                            .device_posterior(
                                pointer
                                    .checked_add(byte_offset as u64)
                                    .ok_or("query offset overflow")?,
                                rows,
                                &y,
                                var.as_deref(),
                                &scales,
                                spec,
                            )?
                    })
                }
                #[cfg(target_arch = "wasm32")]
                (KnnBackend::WebGpu(inner), crate::webgpu::WEBGPU_DEVICE) => {
                    let query_elems = rows
                        .checked_mul(self.num_dim())
                        .ok_or("WebGPU query size overflow")?;
                    let query_bytes = query_elems
                        .checked_mul(std::mem::size_of::<f32>())
                        .ok_or("WebGPU query byte size overflow")?;
                    if pointer == 0
                        || (byte_offset as u64)
                            .checked_add(query_bytes as u64)
                            .is_none()
                    {
                        return Err("invalid WebGPU query pointer".into());
                    }
                    let queries_f32 = {
                        let mut buf = vec![0.0f32; query_elems];
                        let read_handle_ok = if byte_offset == 0 {
                            u32::try_from(pointer)
                                .ok()
                                .filter(|&h| h != 0)
                                .is_some_and(|h| crate::webgpu::read_buffer(h, &mut buf).is_ok())
                        } else {
                            false
                        };
                        if !read_handle_ok {
                            if !crate::webgpu::is_available() {
                                return Err("WebGPU runtime bridge is unavailable".into());
                            }
                            let addr = pointer
                                .checked_add(byte_offset as u64)
                                .ok_or("WebGPU query offset overflow")?;
                            if addr <= 0x1000 || (addr % std::mem::align_of::<f32>() as u64) != 0 {
                                return Err(
                                    "invalid WebGPU buffer handle or unaligned query pointer"
                                        .into(),
                                );
                            }
                            let ptr = usize::try_from(addr)
                                .map_err(|_| "WebGPU pointer out of range")?
                                as *const f32;
                            let src = unsafe { std::slice::from_raw_parts(ptr, query_elems) };
                            buf.copy_from_slice(src);
                        }
                        buf
                    };
                    let guard = inner.lock().map_err(|_| "WebGPU index lock poisoned")?;
                    let (distances, ids) = guard.raw_topk(&queries_f32, rows, k + skip)?;
                    let spec = crate::webgpu::PosteriorKernelSpec {
                        queries: rows,
                        n: guard.len(),
                        metrics: self.num_metrics(),
                        input_k: k + skip,
                        k,
                        skip,
                        epistemic_scale: params.epistemic_scale as f32,
                        aleatoric_scale: params.aleatoric_scale as f32,
                        epsilon: crate::error::EPS_VAR as f32,
                        observation_noise: flags.observation_noise,
                    };
                    Storage::WebGpu(crate::webgpu::eval_posterior(
                        spec,
                        &distances,
                        &ids,
                        &y,
                        var.as_deref(),
                        &scales,
                        crate::Device::WebGpu,
                    )?)
                }
                _ => {
                    return Err(
                        "device queries must match an explicit CUDA, Metal, or WebGPU ENN index"
                            .into(),
                    );
                }
            };
            Ok(Posterior {
                storage,
                rows,
                metrics: self.num_metrics(),
            })
        }
    }
}

impl Posterior {
    pub fn device(&self) -> i32 {
        match self.storage {
            #[cfg(all(target_os = "macos", feature = "metal"))]
            Storage::Metal(_) => 8,
            #[cfg(all(target_os = "linux", target_arch = "x86_64", feature = "cuda"))]
            Storage::Cuda(_) => 2,
            #[cfg(target_arch = "wasm32")]
            Storage::WebGpu(_) => crate::webgpu::WEBGPU_DEVICE,
            _ => 0,
        }
    }
    pub fn pointer(&self, _error: bool) -> u64 {
        match &self.storage {
            #[cfg(all(target_os = "macos", feature = "metal"))]
            Storage::Metal(value) => {
                use metal::foreign_types::ForeignType;
                if _error {
                    value.errors.as_ptr() as u64
                } else {
                    value.means.as_ptr() as u64
                }
            }
            #[cfg(all(target_os = "linux", target_arch = "x86_64", feature = "cuda"))]
            Storage::Cuda(value) => value.pointer(_error),
            #[cfg(target_arch = "wasm32")]
            Storage::WebGpu(value) => {
                if _error {
                    value.errors.as_ptr() as u64
                } else {
                    value.means.as_ptr() as u64
                }
            }
            _ => 0,
        }
    }
    pub fn draw(&self, _seeds: &[u64]) -> Result<PosteriorDraw, String> {
        match &self.storage {
            #[cfg(all(target_os = "macos", feature = "metal"))]
            Storage::Metal(value) => value.draw(_seeds).map(PosteriorDraw::Metal),
            #[cfg(all(target_os = "linux", target_arch = "x86_64", feature = "cuda"))]
            Storage::Cuda(value) => value.draw(_seeds).map(PosteriorDraw::Cuda),
            #[cfg(target_arch = "wasm32")]
            Storage::WebGpu(value) => value.draw(_seeds).map(PosteriorDraw::WebGpu),
            _ => Err("no device posterior implementation".into()),
        }
    }
}

impl PosteriorDraw {
    pub fn pointer(&self) -> u64 {
        match self {
            #[cfg(all(target_os = "macos", feature = "metal"))]
            Self::Metal(value) => {
                use metal::foreign_types::ForeignType;
                value.as_ptr() as u64
            }
            #[cfg(all(target_os = "linux", target_arch = "x86_64", feature = "cuda"))]
            Self::Cuda(value) => value.pointer(),
            #[cfg(target_arch = "wasm32")]
            Self::WebGpu(value) => value.as_ptr() as u64,
            _ => 0,
        }
    }
    pub fn bits(&self) -> u8 {
        match self {
            #[cfg(all(target_os = "linux", target_arch = "x86_64", feature = "cuda"))]
            Self::Cuda(_) => 64,
            _ => 32,
        }
    }
}
