//! Index implementations behind [`crate::index::ENNIndex`].

#[cfg(all(target_os = "macos", feature = "metal"))]
mod auto;
mod exact_backend;

#[cfg(any(feature = "usearch", feature = "usearch-native"))]
mod usearch_backend;

#[cfg(all(target_os = "linux", target_arch = "x86_64", feature = "cuda"))]
pub(crate) mod cuda_index;
#[cfg(all(target_os = "macos", feature = "metal"))]
pub(crate) mod metal_index;
#[cfg(all(target_os = "macos", feature = "metal"))]
mod metal_plan;

use ndarray::{Array2, ArrayView2};
#[cfg(all(target_os = "linux", target_arch = "x86_64", feature = "cuda"))]
use ndarray::{Array3, ArrayView1};
use std::sync::Mutex;
use std::time::Duration;

use crate::error::IndexError;

#[cfg(all(target_os = "macos", feature = "metal"))]
pub(crate) use auto::AutoIndex;
#[cfg(any(
    all(target_os = "macos", feature = "metal"),
    all(target_os = "linux", target_arch = "x86_64", feature = "cuda")
))]
pub(crate) use exact_backend::check_rows;
#[cfg(any(feature = "usearch", feature = "usearch-native"))]
pub(crate) use exact_backend::unpack_search;
pub(crate) use exact_backend::{ExactBackend, flatten_f32, pad_k};
pub use exact_backend::{IndexDriver, IndexKind, disk_driver};

pub type IndexPlan = KnnPlan;
pub type IndexProfile = KnnProfile;

/// KNN execution diagram used by the experimental parity surface.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum KnnPlan {
    /// Validate equivalent diagrams and retain the fastest measured plan.
    Measured,
    /// Materialize tile distances before selecting local neighbors.
    Split,
    /// Fuse tile distance and local top-k computation.
    Fused,
    /// Batch fused tile scans and reduce their neighbor lists as a tree.
    Tree,
    /// Use shared query tiles inside the fused tree scan.
    Tiled,
    /// Cooperatively reduce dimensions before tree top-k selection.
    Simd,
    /// Tile the Gram identity across query and row blocks before tree
    /// selection.
    Gram,
    /// Fuse general tile top-k and pairwise-reduce lists through `k=2048`.
    Wide,
}

/// Device-stage timing summary for the latest accelerated KNN search.
#[derive(Clone, Debug)]
pub struct KnnProfile {
    pub rows: usize,
    pub queries: usize,
    pub dims: usize,
    pub k: usize,
    pub plan: &'static str,
    pub gpu: Duration,
    pub scan: Duration,
    pub select: Duration,
    pub reduce: Duration,
}

#[cfg(all(target_os = "linux", target_arch = "x86_64", feature = "cuda"))]
pub(crate) type KnnPosterior = (
    Array2<f64>,
    Array2<f64>,
    Array2<f64>,
    Array2<f64>,
    Array2<i64>,
);

#[cfg(all(target_os = "linux", target_arch = "x86_64", feature = "cuda"))]
pub(crate) type KnnBatch = (Array3<f64>, Array3<f64>, Array3<f64>, Array3<f64>);

#[cfg(all(target_os = "linux", target_arch = "x86_64", feature = "cuda"))]
#[derive(Clone, Copy, Debug)]
pub(crate) struct CudaParam {
    pub used_k: usize,
    pub epistemic_scale: f64,
    pub aleatoric_scale: f64,
}

/// Unstable low-level KNN surface for accelerator parity and performance work.
pub struct KnnIndex(KnnBackend);

impl KnnIndex {
    /// Builds an index using the measured execution plan.
    pub fn new(train: &ArrayView2<f64>, driver: IndexDriver) -> Result<Self, IndexError> {
        Ok(Self(KnnBackend::new(train.ncols(), driver, train)?))
    }

    /// Builds an index with a requested execution plan.
    pub fn with_plan(
        train: &ArrayView2<f64>,
        driver: IndexDriver,
        plan: KnnPlan,
    ) -> Result<Self, IndexError> {
        Ok(Self(KnnBackend::new_plan(
            train.ncols(),
            driver,
            train,
            plan,
        )?))
    }

    /// Returns the number of indexed rows.
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Finds the `k` exact nearest neighbors for each query row.
    pub fn search(
        &self,
        queries: &ArrayView2<f64>,
        k: usize,
    ) -> Result<(Array2<f64>, Array2<i64>), IndexError> {
        self.0.search(queries, k, k)
    }

    /// Returns the execution plan selected by the last search.
    pub fn plan(&self) -> &'static str {
        self.0.plan()
    }

    /// Returns device-stage timings from the latest accelerated search.
    pub fn profile(&self) -> Option<KnnProfile> {
        self.0.profile()
    }

    /// Returns the backend-owned memory footprint in bytes.
    pub fn memory_bytes(&self) -> usize {
        self.0.memory_bytes()
    }
}

#[cfg(any(feature = "usearch", feature = "usearch-native"))]
use usearch_backend::USearchBackend;

#[cfg(all(target_os = "linux", target_arch = "x86_64", feature = "cuda"))]
use cuda_index::CudaIndex;
#[cfg(all(target_os = "macos", feature = "metal"))]
use metal_index::MetalIndex;

/// In-memory exact and accelerator-backed index implementations.
pub(crate) enum KnnBackend {
    Exact(Mutex<ExactBackend>),
    #[cfg(all(target_os = "macos", feature = "metal"))]
    Auto(Mutex<AutoIndex>),
    #[cfg(any(feature = "usearch", feature = "usearch-native"))]
    USearch(Mutex<USearchBackend>),
    #[cfg(all(target_os = "macos", feature = "metal"))]
    Metal(Mutex<MetalIndex>),
    #[cfg(all(target_os = "linux", target_arch = "x86_64", feature = "cuda"))]
    Cuda(Mutex<CudaIndex>),
}

impl KnnBackend {
    pub(crate) fn new(
        num_dim: usize,
        driver: IndexDriver,
        train_scaled: &ArrayView2<f64>,
    ) -> Result<Self, IndexError> {
        Self::new_plan(num_dim, driver, train_scaled, KnnPlan::Measured)
    }

    fn new_plan(
        num_dim: usize,
        driver: IndexDriver,
        train_scaled: &ArrayView2<f64>,
        plan: KnnPlan,
    ) -> Result<Self, IndexError> {
        if plan != KnnPlan::Measured && driver != IndexDriver::Metal {
            return Err(IndexError::InvalidParameter(
                "explicit KNN plans require the Metal driver".to_string(),
            ));
        }
        match driver {
            IndexDriver::Exact => Ok(Self::Exact(Mutex::new(ExactBackend::new(
                num_dim,
                train_scaled,
            )?))),
            IndexDriver::Auto => {
                #[cfg(all(target_os = "macos", feature = "metal"))]
                {
                    Ok(Self::Auto(Mutex::new(AutoIndex::new(
                        num_dim,
                        train_scaled,
                    )?)))
                }
                #[cfg(not(all(target_os = "macos", feature = "metal")))]
                {
                    Ok(Self::Exact(Mutex::new(ExactBackend::new(
                        num_dim,
                        train_scaled,
                    )?)))
                }
            }
            IndexDriver::USearch => {
                #[cfg(any(feature = "usearch", feature = "usearch-native"))]
                {
                    return Ok(Self::USearch(Mutex::new(USearchBackend::new(
                        num_dim,
                        train_scaled,
                    )?)));
                }
                #[cfg(not(any(feature = "usearch", feature = "usearch-native")))]
                {
                    Err(IndexError::InvalidParameter(
                        "USearch index is unavailable; build with the usearch or usearch-native \
                         feature"
                            .to_string(),
                    ))
                }
            }
            IndexDriver::Metal => {
                #[cfg(all(target_os = "macos", feature = "metal"))]
                {
                    return Ok(Self::Metal(Mutex::new(MetalIndex::new_plan(
                        num_dim,
                        train_scaled,
                        plan,
                    )?)));
                }
                #[cfg(not(all(target_os = "macos", feature = "metal")))]
                {
                    Err(IndexError::InvalidParameter(
                        "Metal index is unavailable; build on macOS with the metal feature"
                            .to_string(),
                    ))
                }
            }
            IndexDriver::OpenCl => Err(IndexError::InvalidParameter(
                "OpenCL index is unsupported".to_string(),
            )),
            IndexDriver::Cuda | IndexDriver::Cuvs | IndexDriver::CuvsCagra => {
                #[cfg(all(target_os = "linux", target_arch = "x86_64", feature = "cuda"))]
                {
                    let index = match driver {
                        IndexDriver::Cuda => CudaIndex::new(num_dim, train_scaled)?,
                        IndexDriver::Cuvs => CudaIndex::new_mode(
                            num_dim,
                            train_scaled,
                            baux_cuda::CudaKnnMode::CuvsBruteForce,
                        )?,
                        IndexDriver::CuvsCagra => CudaIndex::new_mode(
                            num_dim,
                            train_scaled,
                            baux_cuda::CudaKnnMode::CuvsCagra,
                        )?,
                        _ => unreachable!(),
                    };
                    return Ok(Self::Cuda(Mutex::new(index)));
                }
                #[cfg(not(all(target_os = "linux", target_arch = "x86_64", feature = "cuda")))]
                {
                    Err(IndexError::InvalidParameter(
                        "CUDA/cuVS index is unavailable; build on Linux x86_64 with the cuda feature"
                            .to_string(),
                    ))
                }
            }
            IndexDriver::BpAnnDisk => Err(IndexError::InvalidParameter(
                "IndexDriver::BpAnnDisk is disk-only; use DiskBpannEnnBackend".to_string(),
            )),
        }
    }

    pub(crate) fn len(&self) -> usize {
        match self {
            Self::Exact(inner) => inner.lock().expect("knn mutex poisoned").len(),
            #[cfg(all(target_os = "macos", feature = "metal"))]
            Self::Auto(inner) => inner.lock().expect("knn mutex poisoned").cpu.len(),
            #[cfg(any(feature = "usearch", feature = "usearch-native"))]
            Self::USearch(inner) => inner.lock().expect("knn mutex poisoned").len(),
            #[cfg(all(target_os = "macos", feature = "metal"))]
            Self::Metal(inner) => inner.lock().expect("knn mutex poisoned").len(),
            #[cfg(all(target_os = "linux", target_arch = "x86_64", feature = "cuda"))]
            Self::Cuda(inner) => inner.lock().expect("knn mutex poisoned").len(),
        }
    }

    fn plan(&self) -> &'static str {
        match self {
            Self::Exact(_) => "exact",
            #[cfg(all(target_os = "macos", feature = "metal"))]
            Self::Auto(_) => "auto",
            #[cfg(any(feature = "usearch", feature = "usearch-native"))]
            Self::USearch(_) => "usearch",
            #[cfg(all(target_os = "macos", feature = "metal"))]
            Self::Metal(inner) => inner.lock().expect("knn mutex poisoned").plan(),
            #[cfg(all(target_os = "linux", target_arch = "x86_64", feature = "cuda"))]
            Self::Cuda(inner) => inner
                .lock()
                .expect("knn mutex poisoned")
                .profile()
                .map_or("cuda", |profile| profile.plan),
        }
    }

    fn profile(&self) -> Option<KnnProfile> {
        match self {
            #[cfg(all(target_os = "macos", feature = "metal"))]
            Self::Auto(inner) => inner
                .lock()
                .expect("knn mutex poisoned")
                .gpu
                .as_ref()
                .and_then(MetalIndex::profile),
            #[cfg(all(target_os = "macos", feature = "metal"))]
            Self::Metal(inner) => inner.lock().expect("knn mutex poisoned").profile(),
            #[cfg(all(target_os = "linux", target_arch = "x86_64", feature = "cuda"))]
            Self::Cuda(inner) => inner.lock().expect("knn mutex poisoned").profile(),
            _ => None,
        }
    }

    pub(crate) fn memory_bytes(&self) -> usize {
        match self {
            Self::Exact(inner) => inner.lock().expect("knn mutex poisoned").memory_bytes(),
            #[cfg(all(target_os = "macos", feature = "metal"))]
            Self::Auto(inner) => {
                let inner = inner.lock().expect("knn mutex poisoned");
                inner.cpu.memory_bytes() + inner.gpu.as_ref().map_or(0, MetalIndex::memory_bytes)
            }
            #[cfg(any(feature = "usearch", feature = "usearch-native"))]
            Self::USearch(inner) => inner.lock().expect("knn mutex poisoned").memory_bytes(),
            #[cfg(all(target_os = "macos", feature = "metal"))]
            Self::Metal(inner) => inner.lock().expect("knn mutex poisoned").memory_bytes(),
            #[cfg(all(target_os = "linux", target_arch = "x86_64", feature = "cuda"))]
            Self::Cuda(inner) => inner.lock().expect("knn mutex poisoned").memory_bytes(),
        }
    }

    pub(crate) fn rebuild(&self, train_scaled: &ArrayView2<f64>) -> Result<(), IndexError> {
        match self {
            Self::Exact(inner) => inner
                .lock()
                .expect("knn mutex poisoned")
                .rebuild(train_scaled),
            #[cfg(all(target_os = "macos", feature = "metal"))]
            Self::Auto(inner) => {
                let mut inner = inner.lock().expect("knn mutex poisoned");
                inner.cpu.rebuild(train_scaled)?;
                inner.gpu = MetalIndex::new(inner.num_dim, train_scaled).ok();
                inner.decisions.clear();
                Ok(())
            }
            #[cfg(any(feature = "usearch", feature = "usearch-native"))]
            Self::USearch(inner) => inner
                .lock()
                .expect("knn mutex poisoned")
                .rebuild(train_scaled),
            #[cfg(all(target_os = "macos", feature = "metal"))]
            Self::Metal(inner) => inner
                .lock()
                .expect("knn mutex poisoned")
                .rebuild(train_scaled),
            #[cfg(all(target_os = "linux", target_arch = "x86_64", feature = "cuda"))]
            Self::Cuda(inner) => inner
                .lock()
                .expect("knn mutex poisoned")
                .rebuild(train_scaled),
        }
    }

    pub(crate) fn add(
        &self,
        rows_scaled: &ArrayView2<f64>,
        start_key: u64,
    ) -> Result<(), IndexError> {
        match self {
            Self::Exact(inner) => inner
                .lock()
                .expect("knn mutex poisoned")
                .add(rows_scaled, start_key),
            #[cfg(all(target_os = "macos", feature = "metal"))]
            Self::Auto(inner) => {
                let mut inner = inner.lock().expect("knn mutex poisoned");
                inner.cpu.add(rows_scaled, start_key)?;
                if let Some(gpu) = inner.gpu.as_mut() {
                    if gpu.add(rows_scaled, start_key).is_err() {
                        inner.gpu = None;
                    }
                }
                inner.decisions.clear();
                Ok(())
            }
            #[cfg(any(feature = "usearch", feature = "usearch-native"))]
            Self::USearch(inner) => inner
                .lock()
                .expect("knn mutex poisoned")
                .add(rows_scaled, start_key),
            #[cfg(all(target_os = "macos", feature = "metal"))]
            Self::Metal(inner) => inner
                .lock()
                .expect("knn mutex poisoned")
                .add(rows_scaled, start_key),
            #[cfg(all(target_os = "linux", target_arch = "x86_64", feature = "cuda"))]
            Self::Cuda(inner) => inner
                .lock()
                .expect("knn mutex poisoned")
                .add(rows_scaled, start_key),
        }
    }

    pub(crate) fn search(
        &self,
        queries_scaled: &ArrayView2<f64>,
        k_eff: usize,
        search_k: usize,
    ) -> Result<(Array2<f64>, Array2<i64>), IndexError> {
        match self {
            Self::Exact(inner) => {
                inner
                    .lock()
                    .expect("knn mutex poisoned")
                    .search(queries_scaled, k_eff, search_k)
            }
            #[cfg(all(target_os = "macos", feature = "metal"))]
            Self::Auto(inner) => {
                inner
                    .lock()
                    .expect("knn mutex poisoned")
                    .search(queries_scaled, k_eff, search_k)
            }
            #[cfg(any(feature = "usearch", feature = "usearch-native"))]
            Self::USearch(inner) => {
                inner
                    .lock()
                    .expect("knn mutex poisoned")
                    .search(queries_scaled, k_eff, search_k)
            }
            #[cfg(all(target_os = "macos", feature = "metal"))]
            Self::Metal(inner) => {
                inner
                    .lock()
                    .expect("knn mutex poisoned")
                    .search(queries_scaled, k_eff, search_k)
            }
            #[cfg(all(target_os = "linux", target_arch = "x86_64", feature = "cuda"))]
            Self::Cuda(inner) => {
                inner
                    .lock()
                    .expect("knn mutex poisoned")
                    .search(queries_scaled, k_eff, search_k)
            }
        }
    }

    #[cfg(all(target_os = "linux", target_arch = "x86_64", feature = "cuda"))]
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn cuda_posterior(
        &self,
        queries: &ArrayView2<f64>,
        outcomes: &ArrayView2<f64>,
        scales: &ArrayView1<f64>,
        input_k: usize,
        used_k: usize,
        skip: usize,
        epistemic_scale: f64,
        aleatoric_scale: f64,
    ) -> Result<KnnPosterior, IndexError> {
        let Self::Cuda(inner) = self else {
            return Err(IndexError::InvalidParameter(
                "CUDA posterior requires IndexDriver::Cuda".to_string(),
            ));
        };
        inner.lock().expect("knn mutex poisoned").posterior(
            queries,
            outcomes,
            scales,
            input_k,
            used_k,
            skip,
            epistemic_scale,
            aleatoric_scale,
        )
    }

    #[cfg(all(target_os = "linux", target_arch = "x86_64", feature = "cuda"))]
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn cuda_weighted(
        &self,
        queries: &ArrayView2<f64>,
        outcomes: &ArrayView2<f64>,
        variances: Option<&ArrayView2<f64>>,
        scales: &ArrayView1<f64>,
        input_k: usize,
        used_k: usize,
        skip: usize,
        epistemic_scale: f64,
        aleatoric_scale: f64,
        observation_noise: bool,
    ) -> Result<crate::draw::DrawInternals, IndexError> {
        let Self::Cuda(inner) = self else {
            return Err(IndexError::InvalidParameter(
                "CUDA weighted posterior requires IndexDriver::Cuda".to_string(),
            ));
        };
        inner.lock().expect("knn mutex poisoned").weighted(
            queries,
            outcomes,
            variances,
            scales,
            input_k,
            used_k,
            skip,
            epistemic_scale,
            aleatoric_scale,
            observation_noise,
        )
    }

    #[cfg(all(target_os = "linux", target_arch = "x86_64", feature = "cuda"))]
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn cuda_batch(
        &self,
        queries: &ArrayView2<f64>,
        outcomes: &ArrayView2<f64>,
        variances: Option<&ArrayView2<f64>>,
        scales: &ArrayView1<f64>,
        input_k: usize,
        skip: usize,
        values: &[CudaParam],
        observation_noise: bool,
    ) -> Result<KnnBatch, IndexError> {
        let Self::Cuda(inner) = self else {
            return Err(IndexError::InvalidParameter(
                "CUDA batch posterior requires IndexDriver::Cuda".to_string(),
            ));
        };
        inner.lock().expect("knn mutex poisoned").batch(
            queries,
            outcomes,
            variances,
            scales,
            input_k,
            skip,
            values,
            observation_noise,
        )
    }

    #[cfg(all(target_os = "linux", target_arch = "x86_64", feature = "cuda"))]
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn cuda_draws(
        &self,
        queries: &ArrayView2<f64>,
        outcomes: &ArrayView2<f64>,
        variances: Option<&ArrayView2<f64>>,
        scales: &ArrayView1<f64>,
        input_k: usize,
        used_k: usize,
        skip: usize,
        epistemic_scale: f64,
        aleatoric_scale: f64,
        observation_noise: bool,
        seeds: &[i64],
    ) -> Result<(Array3<f64>, Vec<Vec<usize>>), IndexError> {
        let Self::Cuda(inner) = self else {
            return Err(IndexError::InvalidParameter(
                "CUDA draws require IndexDriver::Cuda".to_string(),
            ));
        };
        inner.lock().expect("knn mutex poisoned").draws(
            queries,
            outcomes,
            variances,
            scales,
            input_k,
            used_k,
            skip,
            epistemic_scale,
            aleatoric_scale,
            observation_noise,
            seeds,
        )
    }

    #[cfg(all(target_os = "linux", target_arch = "x86_64", feature = "cuda"))]
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn cuda_conditional(
        &self,
        queries: &ArrayView2<f64>,
        extra_rows: &ArrayView2<f64>,
        outcomes: &ArrayView2<f64>,
        variances: Option<&ArrayView2<f64>>,
        scales: &ArrayView1<f64>,
        input_k: usize,
        used_k: usize,
        skip: usize,
        epistemic_scale: f64,
        aleatoric_scale: f64,
        observation_noise: bool,
    ) -> Result<crate::draw::DrawInternals, IndexError> {
        let Self::Cuda(inner) = self else {
            return Err(IndexError::InvalidParameter(
                "CUDA conditional posterior requires IndexDriver::Cuda".to_string(),
            ));
        };
        inner.lock().expect("knn mutex poisoned").conditional(
            queries,
            extra_rows,
            outcomes,
            variances,
            scales,
            input_k,
            used_k,
            skip,
            epistemic_scale,
            aleatoric_scale,
            observation_noise,
        )
    }

    #[cfg(all(target_os = "linux", target_arch = "x86_64", feature = "cuda"))]
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn condition_draws(
        &self,
        queries: &ArrayView2<f64>,
        extra_rows: &ArrayView2<f64>,
        outcomes: &ArrayView2<f64>,
        variances: Option<&ArrayView2<f64>>,
        scales: &ArrayView1<f64>,
        input_k: usize,
        used_k: usize,
        skip: usize,
        epistemic_scale: f64,
        aleatoric_scale: f64,
        observation_noise: bool,
        seeds: &[i64],
    ) -> Result<(Array3<f64>, Vec<Vec<usize>>), IndexError> {
        let Self::Cuda(inner) = self else {
            return Err(IndexError::InvalidParameter(
                "CUDA conditional draws require IndexDriver::Cuda".to_string(),
            ));
        };
        inner.lock().expect("knn mutex poisoned").conditional_draws(
            queries,
            extra_rows,
            outcomes,
            variances,
            scales,
            input_k,
            used_k,
            skip,
            epistemic_scale,
            aleatoric_scale,
            observation_noise,
            seeds,
        )
    }
}

#[cfg(test)]
mod tests;
