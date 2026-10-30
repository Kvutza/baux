//! Immutable ENN posterior snapshots whose arrays remain on the selected
//! device.
#[cfg(any(
    all(target_os = "macos", feature = "metal"),
    all(target_os = "linux", target_arch = "x86_64", feature = "cuda")
))]
use crate::knn::NeighborIndexImpl;
use crate::{ENN, ENNParams, PosteriorFlags};

enum Storage {
    #[cfg(all(target_os = "macos", feature = "metal"))]
    Metal(crate::knn::metal_index::MetalPosterior),
    #[cfg(all(target_os = "linux", target_arch = "x86_64", feature = "cuda"))]
    Cuda(baux_cuda::DevicePosterior),
    #[allow(dead_code)]
    Unavailable,
}

pub struct DevicePosterior {
    storage: Storage,
    pub rows: usize,
    pub metrics: usize,
}

pub enum DeviceDraw {
    #[cfg(all(target_os = "macos", feature = "metal"))]
    Metal(metal::Buffer),
    #[cfg(all(target_os = "linux", target_arch = "x86_64", feature = "cuda"))]
    Cuda(baux_cuda::DeviceDraw),
    #[allow(dead_code)]
    Unavailable,
}

impl ENN {
    /// Evaluate contiguous FP32 device queries. Training observations are
    /// synced from the ENN; query tensors, posterior arrays, and draws stay
    /// on-device.
    ///
    /// # Safety
    /// pointer is a retained CUDA allocation (device 2) or MTLBuffer (device 8)
    /// covering rows*num_dim FP32 values at byte_offset, ready for reading.
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
            all(target_os = "linux", target_arch = "x86_64", feature = "cuda")
        )))]
        {
            let _ = (pointer, byte_offset, device, rows, params, flags);
            Err("device posterior requires a compiled CUDA or Metal backend".into())
        }
        #[cfg(any(
            all(target_os = "macos", feature = "metal"),
            all(target_os = "linux", target_arch = "x86_64", feature = "cuda")
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
                .store
                .memory_index()
                .ok_or("device posterior requires in-memory observations")?;
            let outcomes = self
                .y_opt()
                .ok_or("device posterior requires in-memory outcomes")?;
            let y = crate::knn::flatten_f32(&outcomes);
            let var = self.store.memory_yvar();
            let var = var.as_ref().map(crate::knn::flatten_f32);
            let scales = self
                .output_scale()
                .iter()
                .map(|&v| v as f32)
                .collect::<Vec<_>>();
            let storage = match (&index.inner, device) {
                #[cfg(all(target_os = "macos", feature = "metal"))]
                (NeighborIndexImpl::Metal(inner), 8) => Storage::Metal(unsafe {
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
                (NeighborIndexImpl::Cuda(inner), 2) => {
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
                _ => {
                    return Err(
                        "device queries must match an explicit CUDA or Metal ENN index".into(),
                    );
                }
            };
            Ok(DevicePosterior {
                storage,
                rows,
                metrics: self.num_metrics(),
            })
        }
    }
}

impl DevicePosterior {
    pub fn device(&self) -> i32 {
        match self.storage {
            #[cfg(all(target_os = "macos", feature = "metal"))]
            Storage::Metal(_) => 8,
            #[cfg(all(target_os = "linux", target_arch = "x86_64", feature = "cuda"))]
            Storage::Cuda(_) => 2,
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
            _ => 0,
        }
    }
    pub fn draw(&self, _seeds: &[u64]) -> Result<DeviceDraw, String> {
        match &self.storage {
            #[cfg(all(target_os = "macos", feature = "metal"))]
            Storage::Metal(value) => value.draw(_seeds).map(DeviceDraw::Metal),
            #[cfg(all(target_os = "linux", target_arch = "x86_64", feature = "cuda"))]
            Storage::Cuda(value) => value.draw(_seeds).map(DeviceDraw::Cuda),
            _ => Err("no device posterior implementation".into()),
        }
    }
}

impl DeviceDraw {
    pub fn pointer(&self) -> u64 {
        match self {
            #[cfg(all(target_os = "macos", feature = "metal"))]
            Self::Metal(value) => {
                use metal::foreign_types::ForeignType;
                value.as_ptr() as u64
            }
            #[cfg(all(target_os = "linux", target_arch = "x86_64", feature = "cuda"))]
            Self::Cuda(value) => value.pointer(),
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
