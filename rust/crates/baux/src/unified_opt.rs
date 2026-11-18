//! Unified cross-backend optimizer engine (`Cpu`, `Disk`, `Metal`, `Cuda`) and
//! [`OptBackend`] trait markers for [`crate::typestate::TypedOptimizer`].

use std::path::PathBuf;

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use rand::RngCore;

use crate::config::{ObjectivePolicy, OptimizerConfig, OptimizerSpec};
use crate::coordinate::DomainFrame;
use crate::error::ENNError;
use crate::hash::Key;
use crate::optimizer::{BayesOptimizer, EvaluationId, HostCursor, Strategy, Telemetry};
use crate::ticket::ProposalTicket;
use crate::unified_mem::decode_half;

#[cfg(all(any(target_os = "macos", target_os = "ios"), feature = "metal"))]
use crate::resident_metal::MetalEngine;

#[inline]
pub fn encode_half(val: f32) -> u16 {
    let bits = val.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xff) as i32 - 127 + 15;
    let mant = (bits >> 13) & 0x03ff;
    if exp <= 0 {
        sign
    } else if exp >= 31 {
        sign | 0x7c00
    } else {
        sign | ((exp as u16) << 10) | (mant as u16)
    }
}

/// Runtime-selected execution backend for dynamic Python
/// (`baux.init(device=...)`) and CLI runner dispatch.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub enum AnyBackend {
    #[default]
    Cpu,
    Metal,
    Cuda,
    Disk {
        work_dir: PathBuf,
    },
}

impl AnyBackend {
    /// Resolve `"cpu"`, `"metal"`, `"cuda"`, or `"auto"` into an
    /// [`AnyBackend`].
    pub fn parse(device: &str) -> Result<Self, ENNError> {
        match device.trim().to_ascii_lowercase().as_str() {
            "cpu" | "host" | "ram" => Ok(Self::Cpu),
            "metal" => Ok(Self::Metal),
            "cuda" => Ok(Self::Cuda),
            "auto" => Ok(Self::auto()),
            other => Err(ENNError::InvalidParameter(format!(
                "unknown device {other:?}; expected 'cpu', 'metal', 'cuda', or 'auto'"
            ))),
        }
    }

    /// Select the best available compiled accelerator at runtime (`Metal` on
    /// Apple Silicon, `Cuda` on NVIDIA Linux, falling back to `Cpu` when no GPU
    /// runtime is present).
    pub fn auto() -> Self {
        #[cfg(all(any(target_os = "macos", target_os = "ios"), feature = "metal"))]
        {
            if crate::apple_gpu::Runtime::shared().is_ok() {
                return Self::Metal;
            }
        }
        #[cfg(all(target_os = "linux", target_arch = "x86_64", feature = "cuda"))]
        {
            if baux_cuda::device_target().is_ok() {
                return Self::Cuda;
            }
        }
        Self::Cpu
    }

    /// Construct a disk-backed (`DiskBpann`) host backend rooted at `work_dir`.
    pub fn disk(work_dir: impl Into<PathBuf>) -> Self {
        Self::Disk {
            work_dir: work_dir.into(),
        }
    }

    #[inline]
    pub fn name(&self) -> &'static str {
        match self {
            Self::Cpu => "cpu",
            Self::Metal => "metal",
            Self::Cuda => "cuda",
            Self::Disk { .. } => "disk",
        }
    }
}

/// Compile-time or runtime backend policy for
/// [`crate::typestate::TypedOptimizer`].
pub trait OptBackend: Clone + Send + Sync + 'static {
    fn backend_kind(&self) -> AnyBackend;

    fn build_engine(
        &self,
        domain: DomainFrame,
        config: OptimizerConfig,
        strategy: Strategy,
        key: Key,
    ) -> Result<UnifiedEngine, ENNError>;

    fn from_spec(
        &self,
        domain: DomainFrame,
        config: OptimizerConfig,
        spec: &OptimizerSpec,
        policy: Option<ObjectivePolicy>,
        num_init: usize,
        key: Key,
    ) -> Result<UnifiedEngine, ENNError>;
}

/// Host CPU (`BayesOptimizer` + `ImLog`) backend marker.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CpuBackend;

/// Resident Apple Silicon Metal GPU (`SearchState` / `RegionalSearch` /
/// `ResidentMorbo`) backend marker.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MetalBackend;

/// Resident NVIDIA CUDA GPU (`TurboCuda` / `MorboCuda` / `RegionalCuda`)
/// backend marker.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CudaBackend;

impl OptBackend for CpuBackend {
    #[inline]
    fn backend_kind(&self) -> AnyBackend {
        AnyBackend::Cpu
    }

    fn build_engine(
        &self,
        domain: DomainFrame,
        config: OptimizerConfig,
        strategy: Strategy,
        key: Key,
    ) -> Result<UnifiedEngine, ENNError> {
        let mut rng = key.prng();
        let bounds = domain.bounds().clone();
        let opt = BayesOptimizer::new_strategy(bounds, config, strategy, &mut rng)?;
        Ok(UnifiedEngine::Cpu(opt))
    }

    fn from_spec(
        &self,
        domain: DomainFrame,
        config: OptimizerConfig,
        spec: &OptimizerSpec,
        _policy: Option<ObjectivePolicy>,
        num_init: usize,
        key: Key,
    ) -> Result<UnifiedEngine, ENNError> {
        let mut rng = key.prng();
        let opt = BayesOptimizer::with_domain(domain, config, spec, num_init, &mut rng)?;
        Ok(UnifiedEngine::Cpu(opt))
    }
}

impl OptBackend for MetalBackend {
    #[inline]
    fn backend_kind(&self) -> AnyBackend {
        AnyBackend::Metal
    }

    fn build_engine(
        &self,
        domain: DomainFrame,
        config: OptimizerConfig,
        strategy: Strategy,
        key: Key,
    ) -> Result<UnifiedEngine, ENNError> {
        #[cfg(all(any(target_os = "macos", target_os = "ios"), feature = "metal"))]
        {
            let engine = MetalEngine::from_strategy(domain, config, strategy, key)?;
            return Ok(UnifiedEngine::Metal {
                inner: engine,
                ticket_tag: None,
            });
        }
        #[cfg(not(all(any(target_os = "macos", target_os = "ios"), feature = "metal")))]
        {
            let _ = (domain, config, strategy, key);
            Err(ENNError::InvalidParameter(
                "Metal backend is unavailable; build on macOS with the metal feature enabled"
                    .into(),
            ))
        }
    }

    fn from_spec(
        &self,
        domain: DomainFrame,
        config: OptimizerConfig,
        spec: &OptimizerSpec,
        policy: Option<ObjectivePolicy>,
        num_init: usize,
        key: Key,
    ) -> Result<UnifiedEngine, ENNError> {
        #[cfg(all(any(target_os = "macos", target_os = "ios"), feature = "metal"))]
        {
            let engine = MetalEngine::from_spec(domain, config, spec, policy, num_init, key)?;
            return Ok(UnifiedEngine::Metal {
                inner: engine,
                ticket_tag: None,
            });
        }
        #[cfg(not(all(any(target_os = "macos", target_os = "ios"), feature = "metal")))]
        {
            let _ = (domain, config, spec, policy, num_init, key);
            Err(ENNError::InvalidParameter(
                "Metal backend is unavailable; build on macOS with the metal feature enabled"
                    .into(),
            ))
        }
    }
}

impl OptBackend for CudaBackend {
    #[inline]
    fn backend_kind(&self) -> AnyBackend {
        AnyBackend::Cuda
    }

    fn build_engine(
        &self,
        domain: DomainFrame,
        config: OptimizerConfig,
        strategy: Strategy,
        key: Key,
    ) -> Result<UnifiedEngine, ENNError> {
        #[cfg(all(target_os = "linux", target_arch = "x86_64", feature = "cuda"))]
        {
            let engine = cuda_impl::CudaEngine::from_strategy(domain, config, strategy, key)?;
            return Ok(UnifiedEngine::Cuda {
                inner: engine,
                ticket_tag: None,
            });
        }
        #[cfg(not(all(target_os = "linux", target_arch = "x86_64", feature = "cuda")))]
        {
            let _ = (domain, config, strategy, key);
            Err(ENNError::InvalidParameter(
                "CUDA backend is unavailable; build on Linux x86_64 with the cuda feature enabled"
                    .into(),
            ))
        }
    }

    fn from_spec(
        &self,
        domain: DomainFrame,
        config: OptimizerConfig,
        spec: &OptimizerSpec,
        _policy: Option<ObjectivePolicy>,
        num_init: usize,
        key: Key,
    ) -> Result<UnifiedEngine, ENNError> {
        #[cfg(all(target_os = "linux", target_arch = "x86_64", feature = "cuda"))]
        {
            let mut rng = key.prng();
            let strategy = spec.strategy(domain.num_dim(), num_init, &mut rng)?;
            let engine = cuda_impl::CudaEngine::from_strategy(domain, config, strategy, key)?;
            return Ok(UnifiedEngine::Cuda {
                inner: engine,
                ticket_tag: None,
            });
        }
        #[cfg(not(all(target_os = "linux", target_arch = "x86_64", feature = "cuda")))]
        {
            let _ = (domain, config, spec, num_init, key);
            Err(ENNError::InvalidParameter(
                "CUDA backend is unavailable; build on Linux x86_64 with the cuda feature enabled"
                    .into(),
            ))
        }
    }
}

impl OptBackend for AnyBackend {
    #[inline]
    fn backend_kind(&self) -> AnyBackend {
        self.clone()
    }

    fn build_engine(
        &self,
        domain: DomainFrame,
        config: OptimizerConfig,
        strategy: Strategy,
        key: Key,
    ) -> Result<UnifiedEngine, ENNError> {
        match self {
            Self::Cpu => CpuBackend.build_engine(domain, config, strategy, key),
            Self::Metal => MetalBackend.build_engine(domain, config, strategy, key),
            Self::Cuda => CudaBackend.build_engine(domain, config, strategy, key),
            Self::Disk { work_dir } => {
                let mut cfg = config;
                if let crate::config::SurrogateConfig::ENN(ref mut enn) = cfg.surrogate {
                    enn.storage = crate::backend::EnnStorage::Disk;
                    enn.index_driver = crate::index::IndexDriver::BpAnnDisk;
                    enn.work_dir = Some(work_dir.clone());
                }
                CpuBackend.build_engine(domain, cfg, strategy, key)
            }
        }
    }

    fn from_spec(
        &self,
        domain: DomainFrame,
        config: OptimizerConfig,
        spec: &OptimizerSpec,
        policy: Option<ObjectivePolicy>,
        num_init: usize,
        key: Key,
    ) -> Result<UnifiedEngine, ENNError> {
        match self {
            Self::Cpu => CpuBackend.from_spec(domain, config, spec, policy, num_init, key),
            Self::Metal => MetalBackend.from_spec(domain, config, spec, policy, num_init, key),
            Self::Cuda => CudaBackend.from_spec(domain, config, spec, policy, num_init, key),
            Self::Disk { work_dir } => {
                let mut rng = key.prng();
                let opt = BayesOptimizer::with_disk(
                    domain,
                    config,
                    spec,
                    num_init,
                    work_dir.clone(),
                    &mut rng,
                )?;
                Ok(UnifiedEngine::Cpu(opt))
            }
        }
    }
}

#[cfg(all(target_os = "linux", target_arch = "x86_64", feature = "cuda"))]
mod cuda_impl {
    use std::sync::Arc;
    use std::time::Instant;

    use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
    use rand::RngCore;

    use crate::candidates::CandidateSampler;
    use crate::config::{AcquisitionConfig, InitStrategy, OptimizerConfig, SurrogateConfig};
    use crate::coordinate::DomainFrame;
    use crate::error::ENNError;
    use crate::hash::Key;
    use crate::optimizer::multi_tr::{
        AllocationPolicy, RegionController as HostController, RestartCenterSelection,
        RestartMemoryPolicy, SharingPolicy,
    };
    use crate::optimizer::{EvaluationId, HostCursor, Strategy, Telemetry};
    use crate::scratch::ImLog;
    use crate::sense::BetterScore;
    use crate::snapshot_io::{
        MAX_DIM, MAX_METRICS, MAX_OBS, MAX_PENDING, WireReader, bad_state, seal_frame, write_bool,
        write_byte, write_header, write_len, write_mat, write_word,
    };
    use crate::trust_region::TrustRegionConfig;

    const CUDA_TAG: u8 = 0x43;

    #[derive(Clone)]
    enum CudaDriver {
        Single(baux_cuda::knn::turbo::TurboCuda),
        Morbo {
            inner: baux_cuda::knn::turbo::MorboCuda,
            last_reg: Option<usize>,
        },
        Regional(baux_cuda::knn::turbo::RegionalCuda),
    }

    #[derive(Clone)]
    struct CudaPending {
        ids: Vec<EvaluationId>,
        unit: Array2<f64>,
        tickets: Vec<(
            baux_cuda::knn::turbo::RegionTicket,
            Vec<EvaluationId>,
            Vec<Vec<f32>>,
        )>,
    }

    #[derive(Clone)]
    pub struct CudaEngine {
        domain: Arc<DomainFrame>,
        config: Arc<OptimizerConfig>,
        strategy_snap: Strategy,
        driver: CudaDriver,
        identified: bool,
        init_only: bool,
        has_surr: bool,
        warmup_goal: usize,
        num_regions: usize,
        seed: u64,
        step: u64,
        next_id: u64,
        pending: Option<CudaPending>,
        x_log: ImLog,
        y_log: ImLog,
        best_idx: Option<usize>,
        best_val: f64,
        telemetry: Telemetry,
    }

    impl std::fmt::Debug for CudaEngine {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.debug_struct("CudaEngine")
                .field("num_dim", &self.domain.num_dim())
                .field("obs_count", &self.x_log.nrows())
                .field("num_regions", &self.num_regions)
                .finish()
        }
    }

    impl CudaEngine {
        pub fn from_strategy(
            domain: DomainFrame,
            config: OptimizerConfig,
            strategy: Strategy,
            key: Key,
        ) -> Result<Self, ENNError> {
            use baux::semantics::morbo::RescalarizeMode;
            use baux::semantics::neighbor::UncertaintyLaw;
            use baux::semantics::optimizer::portfolio::{
                Allocation, RegionController as PortController, RestartCenterPolicy,
                RestartCenterSelection as PortSelection, RestartMemoryPolicy as PortMem,
                SharingPolicy as PortShare,
            };
            use baux::semantics::optimizer::surrogate::AcquisitionRule;
            use baux::semantics::optimizer::trust_region::{PoolSampler, RoundPlan, WarmupKind};

            let num_dim = domain.num_dim().max(1);
            let seed = key.seed();
            let mut tspec = baux_cuda::TurboSpec::for_dims(seed, num_dim);
            tspec.num_candidates = config.candidates.num_candidates(num_dim, 1).max(1);
            tspec.noise_aware = config.noise_aware;
            tspec.sampler = match config.candidates.candidate_rv {
                CandidateSampler::RAASP => PoolSampler::Raasp {
                    prob: RoundPlan::raasp_prob(num_dim),
                },
                CandidateSampler::Uniform => PoolSampler::Uniform,
                CandidateSampler::Sobol => PoolSampler::Sobol { base_idx: 0 },
            };
            tspec.acquisition = match config.acquisition {
                AcquisitionConfig::UCB { beta } => AcquisitionRule::Ucb { beta: beta as f32 },
                AcquisitionConfig::Thompson => AcquisitionRule::Thompson { seed },
                AcquisitionConfig::Pareto => AcquisitionRule::Pareto { seed },
                AcquisitionConfig::Random => AcquisitionRule::Random { seed },
            };
            let has_surr = match &config.surrogate {
                SurrogateConfig::ENN(enn) => {
                    tspec.mode = baux_cuda::TurboMode::TurboEnn;
                    tspec.neighbors = enn.k.max(1) as usize;
                    tspec.epistemic = enn.epistemic_scale as f32;
                    tspec.aleatoric = enn.aleatoric_scale as f32;
                    tspec.y_scale = enn.y_scale.unwrap_or(1.0) as f32;
                    tspec.metric = enn.geometry;
                    tspec.law = UncertaintyLaw::InversePrecision;
                    true
                }
                SurrogateConfig::None => {
                    tspec.mode = baux_cuda::TurboMode::TurboZero;
                    false
                }
            };
            let (tr_len, morbo_opt) = match &config.trust_region {
                TrustRegionConfig::Turbo(l) => (*l, None),
                TrustRegionConfig::Morbo(m) => (m.length, Some(m.clone())),
            };
            tspec.initial_len = tr_len.length_init;
            tspec.min_len = tr_len.length_min;
            tspec.max_len = tr_len.length_max;

            let (init_only, identified, warmup_goal, num_regions, driver) = match &strategy {
                Strategy::Init(s) => {
                    tspec.mode = baux_cuda::TurboMode::LhdOnly;
                    tspec.warmup_obs = s.num_init;
                    tspec.warmup_kind = match s.strategy_type {
                        InitStrategy::LHD => WarmupKind::Lhd,
                        InitStrategy::Random => WarmupKind::Uniform,
                    };
                    let tc = baux_cuda::TurboCuda::with_gpu(tspec, baux_cuda::CudaKnnMode::Custom)
                        .map_err(ENNError::InvalidParameter)?;
                    (true, false, s.num_init, 1, CudaDriver::Single(tc))
                }
                Strategy::Turbo(_) => {
                    tspec.warmup_obs = 0;
                    if let Some(m) = morbo_opt {
                        let resc = match m.rescalarize {
                            crate::morbo::Rescalarize::OnRestart => RescalarizeMode::OnRestart,
                            crate::morbo::Rescalarize::OnPropose => RescalarizeMode::OnPropose,
                        };
                        let mc = baux_cuda::MorboCuda::with_gpu(
                            tspec,
                            m.num_metrics.max(2),
                            4,
                            resc,
                            m.alpha as f32,
                            baux_cuda::CudaKnnMode::Custom,
                        )
                        .map_err(ENNError::InvalidParameter)?;
                        (
                            false,
                            false,
                            0,
                            1,
                            CudaDriver::Morbo {
                                inner: mc,
                                last_reg: None,
                            },
                        )
                    } else {
                        let tc =
                            baux_cuda::TurboCuda::with_gpu(tspec, baux_cuda::CudaKnnMode::Custom)
                                .map_err(ENNError::InvalidParameter)?;
                        (false, false, 0, 1, CudaDriver::Single(tc))
                    }
                }
                Strategy::Hybrid { init, in_init, .. } => {
                    let goal = if *in_init { init.num_init } else { 0 };
                    tspec.warmup_obs = goal;
                    tspec.warmup_kind = match init.strategy_type {
                        InitStrategy::LHD => WarmupKind::Lhd,
                        InitStrategy::Random => WarmupKind::Uniform,
                    };
                    if let Some(m) = morbo_opt {
                        let resc = match m.rescalarize {
                            crate::morbo::Rescalarize::OnRestart => RescalarizeMode::OnRestart,
                            crate::morbo::Rescalarize::OnPropose => RescalarizeMode::OnPropose,
                        };
                        let mc = baux_cuda::MorboCuda::with_gpu(
                            tspec,
                            m.num_metrics.max(2),
                            4,
                            resc,
                            m.alpha as f32,
                            baux_cuda::CudaKnnMode::Custom,
                        )
                        .map_err(ENNError::InvalidParameter)?;
                        (
                            false,
                            false,
                            goal,
                            1,
                            CudaDriver::Morbo {
                                inner: mc,
                                last_reg: None,
                            },
                        )
                    } else {
                        let tc =
                            baux_cuda::TurboCuda::with_gpu(tspec, baux_cuda::CudaKnnMode::Custom)
                                .map_err(ENNError::InvalidParameter)?;
                        (false, false, goal, 1, CudaDriver::Single(tc))
                    }
                }
                Strategy::Experimental(s) => {
                    let goal = if s.in_init { s.init.num_init } else { 0 };
                    tspec.warmup_obs = goal;
                    let n_reg = s.multi_tr.num_regions().max(1);
                    let rspec = baux_cuda::RegionalSpec {
                        base: tspec,
                        num_regions: n_reg,
                        allocation: match s.multi_tr.config.allocation_policy {
                            AllocationPolicy::IncumbentUcb1 => {
                                Allocation::IncumbentUcb1 { exploration: 1.0 }
                            }
                            AllocationPolicy::LeastTrials => Allocation::LeastTrials,
                            AllocationPolicy::JointAcquisition => Allocation::JointAcquisition,
                        },
                        sharing: match s.multi_tr.config.sharing_policy {
                            SharingPolicy::BoxContain => PortShare::BoxContain,
                            SharingPolicy::EllipsoidBall => PortShare::EllipsoidBall,
                            SharingPolicy::NearestCenter => PortShare::NearestCenter,
                            SharingPolicy::OriginOnly => PortShare::OriginOnly,
                        },
                        memory: match s.multi_tr.config.restart_memory_policy {
                            RestartMemoryPolicy::Fresh => PortMem::Fresh,
                            RestartMemoryPolicy::RetainRelevant => PortMem::RetainRelevant,
                        },
                        center_policy: RestartCenterPolicy {
                            candidates: s.multi_tr.config.restart_center_policy.candidates.max(1),
                            selection: match s.multi_tr.config.restart_center_policy.selection {
                                RestartCenterSelection::First => PortSelection::First,
                                RestartCenterSelection::Maximin => PortSelection::Maximin,
                            },
                        },
                        controller: match s.multi_tr.config.controller {
                            HostController::Turbo => PortController::Turbo,
                            HostController::FixedRadius => PortController::FixedRadius,
                        },
                    };
                    let rc =
                        baux_cuda::RegionalCuda::with_gpu(rspec, baux_cuda::CudaKnnMode::Custom)
                            .map_err(ENNError::InvalidParameter)?;
                    (false, true, goal, n_reg, CudaDriver::Regional(rc))
                }
            };

            Ok(Self {
                domain: Arc::new(domain),
                config: Arc::new(config),
                strategy_snap: strategy,
                driver,
                identified,
                init_only,
                has_surr,
                warmup_goal,
                num_regions,
                seed,
                step: 0,
                next_id: 0,
                pending: None,
                x_log: ImLog::with_cols(num_dim),
                y_log: ImLog::empty(),
                best_idx: None,
                best_val: f64::NEG_INFINITY,
                telemetry: Telemetry::default(),
            })
        }

        pub fn ask(
            mut self,
            num_arms: usize,
            rng: &mut dyn RngCore,
        ) -> Result<(Self, Array2<f64>), ENNError> {
            if num_arms == 0 {
                return Err(ENNError::InvalidParameter(
                    "num_arms must be > 0, got 0".into(),
                ));
            }
            let t0 = Instant::now();
            self.step = self.step.wrapping_add(1);
            let step_seed = rng.next_u64() ^ self.step.wrapping_mul(0x9e37_79b9_7f4a_7c15);
            let d = self.domain.num_dim();
            let mut ids = Vec::with_capacity(num_arms);
            for _ in 0..num_arms {
                ids.push(EvaluationId(self.next_id));
                self.next_id = self.next_id.saturating_add(1);
            }
            let (unit, tickets) = match &mut self.driver {
                CudaDriver::Single(tc) => {
                    let prop = tc
                        .ask_seeded(step_seed, num_arms)
                        .map_err(ENNError::InvalidParameter)?;
                    let u = Array2::from_shape_vec(
                        (num_arms, d),
                        prop.rows.into_iter().map(f64::from).collect(),
                    )
                    .map_err(|e| ENNError::InvalidParameter(e.to_string()))?;
                    (u, Vec::new())
                }
                CudaDriver::Morbo { inner, last_reg } => {
                    let (reg, prop) = inner
                        .ask_seeded(step_seed, num_arms)
                        .map_err(ENNError::InvalidParameter)?;
                    *last_reg = Some(reg);
                    let u = Array2::from_shape_vec(
                        (num_arms, d),
                        prop.rows.into_iter().map(f64::from).collect(),
                    )
                    .map_err(|e| ENNError::InvalidParameter(e.to_string()))?;
                    (u, Vec::new())
                }
                CudaDriver::Regional(rc) => {
                    let batches: Vec<baux_cuda::RegionalBatch> = rc
                        .ask_seeded(step_seed, num_arms)
                        .map_err(ENNError::InvalidParameter)?;
                    let mut flat = Vec::with_capacity(num_arms * d);
                    let mut tks = Vec::with_capacity(batches.len());
                    let mut cursor = 0usize;
                    for b in batches {
                        let k = b.proposal.num_arms;
                        let b_ids = ids[cursor..cursor + k].to_vec();
                        let b_rows: Vec<Vec<f32>> = b
                            .proposal
                            .rows
                            .chunks_exact(d)
                            .map(|c| c.to_vec())
                            .collect();
                        flat.extend(b.proposal.rows.into_iter().map(f64::from));
                        tks.push((b.ticket, b_ids, b_rows));
                        cursor += k;
                    }
                    let u = Array2::from_shape_vec((num_arms, d), flat)
                        .map_err(|e| ENNError::InvalidParameter(e.to_string()))?;
                    (u, tks)
                }
            };
            self.pending = Some(CudaPending {
                ids,
                unit: unit.clone(),
                tickets,
            });
            self.telemetry.dt_gen = t0.elapsed().as_secs_f64();
            self.telemetry.num_candidates = num_arms;
            Ok((self, unit))
        }

        pub fn ask_natural(
            self,
            num_arms: usize,
            rng: &mut dyn RngCore,
        ) -> Result<(Self, Array2<f64>), ENNError> {
            let (next, unit) = self.ask(num_arms, rng)?;
            let natural = next.domain.from_unit(&unit.view());
            Ok((next, natural))
        }

        pub fn tell(
            mut self,
            x: &ArrayView2<f64>,
            y: &ArrayView2<f64>,
            yvar: Option<&ArrayView2<f64>>,
            _rng: &mut dyn RngCore,
        ) -> Result<Self, ENNError> {
            let d = self.domain.num_dim();
            if x.ncols() != d || x.nrows() != y.nrows() || y.ncols() == 0 {
                return Err(ENNError::InvalidShape {
                    expected: vec![x.nrows(), d],
                    got: vec![y.nrows(), x.ncols()],
                });
            }
            let t0 = Instant::now();
            let old_n = self.x_log.nrows();
            self.x_log = self.x_log.push_rows(x)?;
            self.y_log = self.y_log.push_rows(y)?;
            for i in 0..y.nrows() {
                let sc = y[[i, 0]];
                if self.best_idx.is_none() || BetterScore(sc) > BetterScore(self.best_val) {
                    self.best_idx = Some(old_n + i);
                    self.best_val = sc;
                }
            }
            let rows_f: Vec<f32> = x.iter().map(|&v| v as f32).collect();
            let y_f: Vec<f32> = y.iter().map(|&v| v as f32).collect();
            let v_f: Option<Vec<f32>> = yvar.map(|yv| yv.iter().map(|&v| v as f32).collect());
            let pend = self.pending.take();
            match &mut self.driver {
                CudaDriver::Single(tc) => {
                    tc.tell(&rows_f, &y_f, v_f.as_deref())
                        .map_err(ENNError::InvalidParameter)?;
                    if self.has_surr && tc.len() >= 4 {
                        tc.fit_loo();
                    }
                }
                CudaDriver::Morbo { inner, last_reg } => {
                    let reg = last_reg.take().unwrap_or(0);
                    inner
                        .tell(reg, &rows_f, &y_f, v_f.as_deref())
                        .map_err(ENNError::InvalidParameter)?;
                }
                CudaDriver::Regional(rc) => {
                    if let Some(p) = pend {
                        let mut cursor = 0usize;
                        for (tk, b_ids, _) in p.tickets {
                            let k = b_ids.len();
                            let r_slice = &rows_f[cursor * d..(cursor + k) * d];
                            let o_slice = &y_f[cursor..cursor + k];
                            let v_slice = v_f.as_ref().map(|vs| &vs[cursor..cursor + k]);
                            rc.tell(tk, r_slice, o_slice, v_slice)
                                .map_err(ENNError::InvalidParameter)?;
                            cursor += k;
                        }
                    }
                }
            }
            self.telemetry.dt_tell = t0.elapsed().as_secs_f64();
            Ok(self)
        }

        pub fn tell_natural(
            self,
            x: &ArrayView2<f64>,
            y: &ArrayView2<f64>,
            yvar: Option<&ArrayView2<f64>>,
            rng: &mut dyn RngCore,
        ) -> Result<Self, ENNError> {
            let unit = self.domain.to_unit(x);
            self.tell(&unit.view(), y, yvar, rng)
        }

        pub fn complete_ids(
            mut self,
            ids: &[EvaluationId],
            y: &ArrayView2<f64>,
            yvar: Option<&ArrayView2<f64>>,
            rng: &mut dyn RngCore,
        ) -> Result<Self, ENNError> {
            let p = self.pending.take().ok_or_else(|| {
                ENNError::InvalidParameter("no pending CUDA evaluations to complete".into())
            })?;
            if ids.len() == p.ids.len() && ids == p.ids.as_slice() {
                self.pending = Some(p.clone());
                return self.tell(&p.unit.view(), y, yvar, rng);
            }
            let CudaDriver::Regional(ref mut rc) = self.driver else {
                return Err(ENNError::InvalidParameter(
                    "partial ID completion requires multi-region CUDA optimizer".into(),
                ));
            };
            let mut rem_ids = Vec::new();
            let mut rem_tickets = Vec::new();
            for (tk, b_ids, b_rows) in p.tickets {
                if let Some(pos) = ids.iter().position(|&id| b_ids.contains(&id)) {
                    let flat_r: Vec<f32> = b_rows.iter().flatten().copied().collect();
                    let out_y = vec![y[[pos, 0]] as f32; b_ids.len()];
                    let out_v = yvar.map(|yv| vec![yv[[pos, 0]] as f32; b_ids.len()]);
                    rc.tell(tk, &flat_r, &out_y, out_v.as_deref())
                        .map_err(ENNError::InvalidParameter)?;
                    let d = self.domain.num_dim();
                    let u_arr = Array2::from_shape_vec(
                        (b_rows.len(), d),
                        flat_r.into_iter().map(f64::from).collect(),
                    )
                    .map_err(|e| ENNError::InvalidParameter(e.to_string()))?;
                    let y_sub = Array2::from_elem((b_rows.len(), 1), y[[pos, 0]]);
                    let old_n = self.x_log.nrows();
                    self.x_log = self.x_log.push_rows(&u_arr.view())?;
                    self.y_log = self.y_log.push_rows(&y_sub.view())?;
                    if self.best_idx.is_none()
                        || BetterScore(y[[pos, 0]]) > BetterScore(self.best_val)
                    {
                        self.best_idx = Some(old_n);
                        self.best_val = y[[pos, 0]];
                    }
                } else {
                    rem_ids.extend_from_slice(&b_ids);
                    rem_tickets.push((tk, b_ids, b_rows));
                }
            }
            if !rem_ids.is_empty() {
                self.pending = Some(CudaPending {
                    ids: rem_ids,
                    unit: p.unit,
                    tickets: rem_tickets,
                });
            }
            Ok(self)
        }

        pub fn cancel_ids(
            mut self,
            ids: &[EvaluationId],
            _rng: &mut dyn RngCore,
        ) -> Result<Self, ENNError> {
            let Some(p) = self.pending.take() else {
                return Err(ENNError::InvalidParameter(
                    "no pending CUDA evaluations to cancel".into(),
                ));
            };
            if let CudaDriver::Regional(ref mut rc) = self.driver {
                let mut rem_ids = Vec::new();
                let mut rem_tickets = Vec::new();
                for (tk, b_ids, b_rows) in p.tickets {
                    if b_ids.iter().any(|id| ids.contains(id)) {
                        rc.cancel(tk).map_err(ENNError::InvalidParameter)?;
                    } else {
                        rem_ids.extend_from_slice(&b_ids);
                        rem_tickets.push((tk, b_ids, b_rows));
                    }
                }
                if !rem_ids.is_empty() {
                    self.pending = Some(CudaPending {
                        ids: rem_ids,
                        unit: p.unit,
                        tickets: rem_tickets,
                    });
                }
            }
            Ok(self)
        }

        pub fn restart_natural(
            self,
            _slot: usize,
            _center: &ArrayView1<f64>,
            _rng: &mut dyn RngCore,
        ) -> Result<(Self, Vec<EvaluationId>), ENNError> {
            Ok((self, Vec::new()))
        }

        pub fn import_natural(
            self,
            x: &ArrayView2<f64>,
            y: &ArrayView2<f64>,
            yvar: Option<&ArrayView2<f64>>,
            rng: &mut dyn RngCore,
        ) -> Result<Self, ENNError> {
            self.tell_natural(x, y, yvar, rng)
        }

        pub fn pending_ids(&self) -> Option<Vec<EvaluationId>> {
            if !self.identified {
                return None;
            }
            self.pending.as_ref().map(|p| p.ids.clone())
        }

        pub fn all_pending(&self) -> Option<Vec<EvaluationId>> {
            self.pending.as_ref().map(|p| p.ids.clone())
        }

        pub fn posterior_mu(&self, x: &ArrayView2<f64>) -> Result<Array2<f64>, ENNError> {
            let d = self.domain.num_dim();
            if x.ncols() != d {
                return Err(ENNError::InvalidShape {
                    expected: vec![x.nrows(), d],
                    got: vec![x.nrows(), x.ncols()],
                });
            }
            if !self.has_surr {
                return Err(ENNError::InvalidParameter("No surrogate".into()));
            }
            let n = self.x_log.nrows();
            let m = self.y_log.ncols().max(1);
            if n == 0 {
                return Err(ENNError::InvalidParameter("Surrogate not fitted".into()));
            }
            let uq = self.domain.to_unit(x);
            let ty = self.y_log.to_array();
            let mut out = Array2::<f64>::zeros((uq.nrows(), m));
            for q in 0..uq.nrows() {
                let qr = uq.row(q);
                let dists: Vec<f64> = (0..n)
                    .map(|r| {
                        qr.iter()
                            .zip(self.x_log.row_slice(r))
                            .map(|(&a, &b)| (a - b) * (a - b))
                            .sum::<f64>()
                            / (d.max(1) as f64)
                    })
                    .collect();
                for col in 0..m {
                    let outcomes: Vec<f32> = (0..n).map(|r| ty[[r, col]] as f32).collect();
                    let vars = vec![0.0f32; n];
                    let pred = (crate::semantics::neighbor::query::Query {
                        distances: &dists,
                        outcomes: &outcomes,
                        variances: &vars,
                        draws: None,
                        neighbors: 10.min(n).max(1),
                        epistemic: 0.05,
                        aleatoric: 0.05,
                        scale: 1.0,
                        beta: 1.0,
                        excluded: None,
                        incumbent: self.best_idx.unwrap_or(0).min(n - 1),
                        adaptive: false,
                    })
                    .predict()
                    .map_err(|e| ENNError::InvalidParameter(e.to_string()))?;
                    out[[q, col]] = f64::from(pred.mean);
                }
            }
            Ok(out)
        }

        pub fn has_surrogate(&self) -> bool {
            self.has_surr
        }

        pub fn init_progress(&self) -> Option<(usize, usize)> {
            let done = self.x_log.nrows();
            if self.init_only || done < self.warmup_goal {
                Some((done, self.warmup_goal))
            } else {
                None
            }
        }

        pub fn telemetry(&self) -> &Telemetry {
            &self.telemetry
        }

        pub fn obs_count(&self) -> usize {
            self.x_log.nrows()
        }

        pub fn num_regions(&self) -> usize {
            self.num_regions
        }

        pub fn tr_length(&self) -> f64 {
            match &self.driver {
                CudaDriver::Single(tc) => tc.tr_length(),
                CudaDriver::Morbo { inner, .. } => inner.tr_length(),
                CudaDriver::Regional(rc) => rc.tr_length(),
            }
        }

        pub fn x_obs(&self) -> Option<Array2<f64>> {
            (!self.x_log.is_empty()).then(|| self.domain.from_unit(&self.x_log.to_array().view()))
        }

        pub fn y_obs(&self) -> Option<Array2<f64>> {
            (!self.y_log.is_empty()).then(|| self.y_log.to_array())
        }

        pub fn unit_incumbent(&self) -> Option<Array1<f64>> {
            self.best_idx.map(|idx| self.x_log.row_vec(idx))
        }

        pub fn incumbent_x(&self) -> Option<Array1<f64>> {
            self.unit_incumbent().map(|u| self.domain.natural_point(&u))
        }

        pub fn best_y(&self) -> Option<f64> {
            self.best_idx.map(|_| self.best_val)
        }

        pub fn bounds(&self) -> &Array2<f64> {
            self.domain.bounds()
        }

        pub fn num_dim(&self) -> usize {
            self.domain.num_dim()
        }

        pub fn shares_storage(&self, other: &Self) -> bool {
            let logs =
                self.x_log.shares_storage(&other.x_log) && self.y_log.shares_storage(&other.y_log);
            let drv = match (&self.driver, &other.driver) {
                (CudaDriver::Single(a), CudaDriver::Single(b)) => a.shares_storage(b),
                (CudaDriver::Morbo { inner: a, .. }, CudaDriver::Morbo { inner: b, .. }) => {
                    a.shares_storage(b)
                }
                (CudaDriver::Regional(a), CudaDriver::Regional(b)) => a.shares_storage(b),
                _ => false,
            };
            logs && drv
        }

        pub fn digest(&self) -> [u8; 32] {
            let mut buf = Vec::with_capacity(128);
            buf.extend_from_slice(b"CudaEngine");
            buf.extend_from_slice(&self.seed.to_le_bytes());
            buf.extend_from_slice(&self.step.to_le_bytes());
            let inner_d = match &self.driver {
                CudaDriver::Single(tc) => tc.digest(),
                CudaDriver::Morbo { inner, .. } => inner.digest(),
                CudaDriver::Regional(rc) => rc.digest(),
            };
            buf.extend_from_slice(&inner_d);
            crate::artifact::blake3::hash(&buf)
        }

        pub fn freeze_full(&self, host: Option<&HostCursor>) -> Result<Vec<u8>, ENNError> {
            if self.pending.is_some() {
                return Err(ENNError::InvalidParameter(
                    "cannot freeze CudaEngine while evaluations are pending".into(),
                ));
            }
            let mut buf = Vec::with_capacity(256);
            write_header(&mut buf);
            write_byte(&mut buf, CUDA_TAG);
            write_word(&mut buf, self.seed);
            write_word(&mut buf, self.step);
            write_word(&mut buf, self.next_id);
            write_mat(&mut buf, &self.x_log.to_array());
            write_mat(&mut buf, &self.y_log.to_array());
            match host {
                Some(h) => {
                    write_bool(&mut buf, true);
                    write_word(&mut buf, h.seed);
                    write_word(&mut buf, h.epoch);
                    write_bool(&mut buf, h.identified);
                    write_len(&mut buf, h.id_map.len());
                    for &(k, v) in &h.id_map {
                        write_word(&mut buf, k);
                        write_word(&mut buf, v);
                    }
                }
                None => write_bool(&mut buf, false),
            }
            seal_frame(&mut buf);
            Ok(buf)
        }

        pub fn thaw_full(&self, bytes: &[u8]) -> Result<(Self, Option<HostCursor>), ENNError> {
            let mut r = WireReader::open_frame(bytes)?;
            if r.read_byte()? != CUDA_TAG {
                return Err(bad_state(
                    "frozen state backend tag mismatch (expected Cuda)",
                ));
            }
            let seed = r.read_word()?;
            let step = r.read_word()?;
            let next_id = r.read_word()?;
            let x_arr = r.read_mat(MAX_OBS, MAX_DIM, true)?;
            let y_arr = r.read_mat(MAX_OBS, MAX_METRICS, true)?;
            let host = if r.read_bool()? {
                let h_seed = r.read_word()?;
                let h_epoch = r.read_word()?;
                let h_ident = r.read_bool()?;
                let map_len = r.read_len(MAX_PENDING)?;
                let mut id_map = Vec::new();
                id_map
                    .try_reserve_exact(map_len)
                    .map_err(|_| bad_state("frozen id_map allocation failed"))?;
                for _ in 0..map_len {
                    id_map.push((r.read_word()?, r.read_word()?));
                }
                Some(HostCursor {
                    seed: h_seed,
                    epoch: h_epoch,
                    identified: h_ident,
                    id_map,
                })
            } else {
                None
            };
            r.finish()?;
            let mut fresh = Self::from_strategy(
                (*self.domain).clone(),
                (*self.config).clone(),
                self.strategy_snap.clone(),
                Key::new(seed),
            )?;
            if x_arr.nrows() > 0 {
                let mut rng = crate::hash::prng(seed);
                fresh = fresh.tell(&x_arr.view(), &y_arr.view(), None, &mut rng)?;
            }
            fresh.step = step;
            fresh.next_id = next_id;
            Ok((fresh, host))
        }
    }
}

/// Unified cross-backend optimizer engine with $O(1)$ `Arc` sharing across CPU
/// `ImLog`, Metal `Arc<MTLBuffer>`, and CUDA snapshots.
#[derive(Clone, Debug)]
pub enum UnifiedEngine {
    Cpu(BayesOptimizer),
    #[cfg(all(any(target_os = "macos", target_os = "ios"), feature = "metal"))]
    Metal {
        inner: MetalEngine,
        ticket_tag: Option<[u8; 32]>,
    },
    #[cfg(all(target_os = "linux", target_arch = "x86_64", feature = "cuda"))]
    Cuda {
        inner: cuda_impl::CudaEngine,
        ticket_tag: Option<[u8; 32]>,
    },
}

const _: () = {
    fn assert_sync<T: Send + Sync>() {}
    fn check() {
        assert_sync::<UnifiedEngine>();
    }
};

impl UnifiedEngine {
    /// Wrap an existing CPU or Disk [`BayesOptimizer`] in [`UnifiedEngine`].
    #[inline]
    pub fn from_cpu(opt: BayesOptimizer) -> Self {
        Self::Cpu(opt)
    }

    /// Construct a [`UnifiedEngine`] on `backend` from `(bounds, config,
    /// strategy, key)`.
    pub fn new_strategy(
        bounds: Array2<f64>,
        config: OptimizerConfig,
        strategy: Strategy,
        backend: AnyBackend,
        key: Key,
    ) -> Result<Self, ENNError> {
        backend.build_engine(DomainFrame::new(bounds)?, config, strategy, key)
    }

    /// Construct a [`UnifiedEngine`] on `backend` from a model [`DomainFrame`],
    /// customized [`OptimizerConfig`], closed [`OptimizerSpec`], and optional
    /// multi-objective [`ObjectivePolicy`].
    pub fn from_domain(
        domain: DomainFrame,
        config: OptimizerConfig,
        spec: &OptimizerSpec,
        policy: Option<ObjectivePolicy>,
        num_init: usize,
        backend: AnyBackend,
        key: Key,
    ) -> Result<Self, ENNError> {
        backend.from_spec(domain, config, spec, policy, num_init, key)
    }

    /// Construct a [`UnifiedEngine`] directly from `bounds` and a closed
    /// [`OptimizerSpec`].
    pub fn from_spec(
        bounds: Array2<f64>,
        spec: &OptimizerSpec,
        num_init: usize,
        backend: AnyBackend,
        key: Key,
    ) -> Result<Self, ENNError> {
        let config = OptimizerConfig::from_spec(spec)?;
        Self::from_domain(
            DomainFrame::new(bounds)?,
            config,
            spec,
            None,
            num_init,
            backend,
            key,
        )
    }

    /// Validate that `spec` is supported on the resident Apple Silicon Metal
    /// backend.
    pub fn check_metal(spec: &OptimizerSpec) -> Result<(), ENNError> {
        #[cfg(all(any(target_os = "macos", target_os = "ios"), feature = "metal"))]
        {
            return crate::resident_metal::engine::check_spec(spec);
        }
        #[cfg(not(all(any(target_os = "macos", target_os = "ios"), feature = "metal")))]
        {
            let _ = spec;
            Err(ENNError::InvalidParameter(
                "resident-metal backend requires macOS with the metal feature enabled".into(),
            ))
        }
    }

    /// Name of the active execution device (`"cpu"`, `"disk"`, `"metal"`, or
    /// `"cuda"`).
    pub fn device_name(&self) -> &'static str {
        match self {
            Self::Cpu(opt) => {
                if opt.posterior.is_disk() {
                    "disk"
                } else {
                    "cpu"
                }
            }
            #[cfg(all(any(target_os = "macos", target_os = "ios"), feature = "metal"))]
            Self::Metal { .. } => "metal",
            #[cfg(all(target_os = "linux", target_arch = "x86_64", feature = "cuda"))]
            Self::Cuda { .. } => "cuda",
        }
    }

    /// Borrow the underlying [`BayesOptimizer`] when backed by `Cpu` or `Disk`.
    #[inline]
    pub fn as_cpu(&self) -> Option<&BayesOptimizer> {
        match self {
            Self::Cpu(opt) => Some(opt),
            #[allow(unreachable_patterns)]
            _ => None,
        }
    }

    /// Fork an isolated optimizer branch in $O(1)$ via `Arc` sharing.
    pub fn fork(&self) -> Result<Self, ENNError> {
        match self {
            Self::Cpu(opt) => Ok(Self::Cpu(opt.fork()?)),
            #[cfg(all(any(target_os = "macos", target_os = "ios"), feature = "metal"))]
            Self::Metal { inner, ticket_tag } => Ok(Self::Metal {
                inner: inner.fork()?,
                ticket_tag: *ticket_tag,
            }),
            #[cfg(all(target_os = "linux", target_arch = "x86_64", feature = "cuda"))]
            Self::Cuda { inner, ticket_tag } => Ok(Self::Cuda {
                inner: inner.clone(),
                ticket_tag: *ticket_tag,
            }),
        }
    }

    /// True when `self` and `other` share identical `Arc`-backed storage.
    pub fn shares_storage(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Cpu(a), Self::Cpu(b)) => a.shares_storage(b),
            #[cfg(all(any(target_os = "macos", target_os = "ios"), feature = "metal"))]
            (Self::Metal { inner: a, .. }, Self::Metal { inner: b, .. }) => a.shares_storage(b),
            #[cfg(all(target_os = "linux", target_arch = "x86_64", feature = "cuda"))]
            (Self::Cuda { inner: a, .. }, Self::Cuda { inner: b, .. }) => a.shares_storage(b),
            #[allow(unreachable_patterns)]
            _ => false,
        }
    }

    /// Content-addressed BLAKE3 digest of canonical optimizer state.
    pub fn digest(&self) -> [u8; 32] {
        match self {
            Self::Cpu(opt) => opt.digest(),
            #[cfg(all(any(target_os = "macos", target_os = "ios"), feature = "metal"))]
            Self::Metal { inner, ticket_tag } => match ticket_tag {
                None => inner.digest(),
                Some(tag) => {
                    let mut buf = [0u8; 64];
                    buf[..32].copy_from_slice(&inner.digest());
                    buf[32..].copy_from_slice(tag);
                    crate::artifact::blake3::hash(&buf)
                }
            },
            #[cfg(all(target_os = "linux", target_arch = "x86_64", feature = "cuda"))]
            Self::Cuda { inner, ticket_tag } => match ticket_tag {
                None => inner.digest(),
                Some(tag) => {
                    let mut buf = [0u8; 64];
                    buf[..32].copy_from_slice(&inner.digest());
                    buf[32..].copy_from_slice(tag);
                    crate::artifact::blake3::hash(&buf)
                }
            },
        }
    }

    /// Lowercase hexadecimal BLAKE3 digest of canonical optimizer state.
    pub fn digest_hex(&self) -> String {
        self.digest().iter().map(|b| format!("{b:02x}")).collect()
    }

    /// Lowercase hexadecimal BLAKE3 digest folding host cursor state.
    pub fn digest_host(&self, seed: u64, epoch: u64, identified: bool) -> String {
        let mut buf = [0u8; 49];
        buf[..32].copy_from_slice(&self.digest());
        buf[32..40].copy_from_slice(&seed.to_le_bytes());
        buf[40..48].copy_from_slice(&epoch.to_le_bytes());
        buf[48] = u8::from(identified);
        crate::artifact::blake3::hash(&buf)
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect()
    }

    /// Current incumbent point in natural domain coordinates.
    pub fn best_x(&self) -> Option<Array1<f64>> {
        self.incumbent_x()
    }

    /// Current incumbent point in natural domain coordinates.
    pub fn incumbent_x(&self) -> Option<Array1<f64>> {
        match self {
            Self::Cpu(opt) => opt.incumbent_x(),
            #[cfg(all(any(target_os = "macos", target_os = "ios"), feature = "metal"))]
            Self::Metal { inner, .. } => inner.incumbent_x(),
            #[cfg(all(target_os = "linux", target_arch = "x86_64", feature = "cuda"))]
            Self::Cuda { inner, .. } => inner.incumbent_x(),
        }
    }

    /// Current incumbent point in unit `[0, 1]^D` coordinates.
    pub fn unit_incumbent(&self) -> Option<Array1<f64>> {
        match self {
            Self::Cpu(opt) => opt.unit_incumbent().cloned(),
            #[cfg(all(any(target_os = "macos", target_os = "ios"), feature = "metal"))]
            Self::Metal { inner, .. } => inner.unit_incumbent(),
            #[cfg(all(target_os = "linux", target_arch = "x86_64", feature = "cuda"))]
            Self::Cuda { inner, .. } => inner.unit_incumbent(),
        }
    }

    /// Current primary incumbent objective value in canonical maximization
    /// units.
    pub fn best_y(&self) -> Option<f64> {
        match self {
            Self::Cpu(opt) => opt.y_scalar().and_then(|y| y.first().copied()),
            #[cfg(all(any(target_os = "macos", target_os = "ios"), feature = "metal"))]
            Self::Metal { inner, .. } => inner.best_y(),
            #[cfg(all(target_os = "linux", target_arch = "x86_64", feature = "cuda"))]
            Self::Cuda { inner, .. } => inner.best_y(),
        }
    }

    /// Number of retained observations across the optimizer.
    #[inline]
    pub fn num_obs(&self) -> usize {
        self.obs_count()
    }

    /// Number of retained observations across the optimizer.
    pub fn obs_count(&self) -> usize {
        match self {
            Self::Cpu(opt) => opt.obs_count(),
            #[cfg(all(any(target_os = "macos", target_os = "ios"), feature = "metal"))]
            Self::Metal { inner, .. } => inner.obs_count(),
            #[cfg(all(target_os = "linux", target_arch = "x86_64", feature = "cuda"))]
            Self::Cuda { inner, .. } => inner.obs_count(),
        }
    }

    /// Number of configured trust-region slots.
    pub fn num_regions(&self) -> usize {
        match self {
            Self::Cpu(opt) => opt.num_regions(),
            #[cfg(all(any(target_os = "macos", target_os = "ios"), feature = "metal"))]
            Self::Metal { inner, .. } => inner.num_regions(),
            #[cfg(all(target_os = "linux", target_arch = "x86_64", feature = "cuda"))]
            Self::Cuda { inner, .. } => inner.num_regions(),
        }
    }

    /// Current trust-region length.
    pub fn tr_length(&self) -> f64 {
        match self {
            Self::Cpu(opt) => opt.tr_length(),
            #[cfg(all(any(target_os = "macos", target_os = "ios"), feature = "metal"))]
            Self::Metal { inner, .. } => inner.tr_length(),
            #[cfg(all(target_os = "linux", target_arch = "x86_64", feature = "cuda"))]
            Self::Cuda { inner, .. } => inner.tr_length(),
        }
    }

    /// Warm-up `(completed, budget)` progress when in the initialization phase.
    pub fn init_progress(&self) -> Option<(usize, usize)> {
        match self {
            Self::Cpu(opt) => opt.init_progress(),
            #[cfg(all(any(target_os = "macos", target_os = "ios"), feature = "metal"))]
            Self::Metal { inner, .. } => inner.init_progress(),
            #[cfg(all(target_os = "linux", target_arch = "x86_64", feature = "cuda"))]
            Self::Cuda { inner, .. } => inner.init_progress(),
        }
    }

    /// Outstanding multi-region evaluation identities in ask order.
    pub fn pending_ids(&self) -> Option<Vec<EvaluationId>> {
        match self {
            Self::Cpu(opt) => opt.pending_ids(),
            #[cfg(all(any(target_os = "macos", target_os = "ios"), feature = "metal"))]
            Self::Metal { inner, .. } => inner.pending_ids(),
            #[cfg(all(target_os = "linux", target_arch = "x86_64", feature = "cuda"))]
            Self::Cuda { inner, .. } => inner.pending_ids(),
        }
    }

    /// Whether this optimizer fits a surrogate model.
    pub fn has_surrogate(&self) -> bool {
        match self {
            Self::Cpu(opt) => opt.has_surrogate(),
            #[cfg(all(any(target_os = "macos", target_os = "ios"), feature = "metal"))]
            Self::Metal { inner, .. } => inner.has_surrogate(),
            #[cfg(all(target_os = "linux", target_arch = "x86_64", feature = "cuda"))]
            Self::Cuda { inner, .. } => inner.has_surrogate(),
        }
    }

    /// Non-fatal ingestion warning/failure message, if any.
    pub fn ingestion_failure(&self) -> Option<&str> {
        match self {
            Self::Cpu(opt) => opt.ingestion_failure(),
            #[cfg(all(any(target_os = "macos", target_os = "ios"), feature = "metal"))]
            Self::Metal { inner, .. } => inner.ingestion_failure(),
            #[cfg(all(target_os = "linux", target_arch = "x86_64", feature = "cuda"))]
            Self::Cuda { .. } => None,
        }
    }

    /// Timing telemetry for the latest ask/tell round.
    pub fn telemetry(&self) -> &Telemetry {
        match self {
            Self::Cpu(opt) => opt.telemetry(),
            #[cfg(all(any(target_os = "macos", target_os = "ios"), feature = "metal"))]
            Self::Metal { inner, .. } => inner.telemetry(),
            #[cfg(all(target_os = "linux", target_arch = "x86_64", feature = "cuda"))]
            Self::Cuda { inner, .. } => inner.telemetry(),
        }
    }

    /// Observed parameter matrix in natural domain coordinates.
    pub fn x_obs(&self) -> Option<Array2<f64>> {
        match self {
            Self::Cpu(opt) => opt.x_obs(),
            #[cfg(all(any(target_os = "macos", target_os = "ios"), feature = "metal"))]
            Self::Metal { inner, .. } => inner.x_obs(),
            #[cfg(all(target_os = "linux", target_arch = "x86_64", feature = "cuda"))]
            Self::Cuda { inner, .. } => inner.x_obs(),
        }
    }

    /// Observed objective matrix.
    pub fn y_obs(&self) -> Option<Array2<f64>> {
        match self {
            Self::Cpu(opt) => opt.y_obs(),
            #[cfg(all(any(target_os = "macos", target_os = "ios"), feature = "metal"))]
            Self::Metal { inner, .. } => inner.y_obs(),
            #[cfg(all(target_os = "linux", target_arch = "x86_64", feature = "cuda"))]
            Self::Cuda { inner, .. } => inner.y_obs(),
        }
    }

    /// Learned surrogate lengthscales, if any.
    pub fn lengthscales(&self) -> Option<Array1<f64>> {
        match self {
            Self::Cpu(opt) => opt.lengthscales(),
            #[allow(unreachable_patterns)]
            _ => None,
        }
    }

    /// Domain bounds `(D, 2)`.
    pub fn bounds(&self) -> &Array2<f64> {
        match self {
            Self::Cpu(opt) => opt.bounds(),
            #[cfg(all(any(target_os = "macos", target_os = "ios"), feature = "metal"))]
            Self::Metal { inner, .. } => inner.bounds(),
            #[cfg(all(target_os = "linux", target_arch = "x86_64", feature = "cuda"))]
            Self::Cuda { inner, .. } => inner.bounds(),
        }
    }

    /// Parameter space dimensionality `D`.
    pub fn num_dim(&self) -> usize {
        match self {
            Self::Cpu(opt) => opt.num_dim(),
            #[cfg(all(any(target_os = "macos", target_os = "ios"), feature = "metal"))]
            Self::Metal { inner, .. } => inner.num_dim(),
            #[cfg(all(target_os = "linux", target_arch = "x86_64", feature = "cuda"))]
            Self::Cuda { inner, .. } => inner.num_dim(),
        }
    }

    /// Posterior mean at `x` in natural domain coordinates.
    pub fn posterior_mu(&self, x: &ArrayView2<f64>) -> Result<Array2<f64>, ENNError> {
        match self {
            Self::Cpu(opt) => opt.posterior_mu(x),
            #[cfg(all(any(target_os = "macos", target_os = "ios"), feature = "metal"))]
            Self::Metal { inner, .. } => inner.posterior_mu(x),
            #[cfg(all(target_os = "linux", target_arch = "x86_64", feature = "cuda"))]
            Self::Cuda { inner, .. } => inner.posterior_mu(x),
        }
    }

    /// Propose `num_arms` candidates in unit `[0, 1]^D` coordinates.
    pub fn ask(
        self,
        num_arms: usize,
        rng: &mut dyn RngCore,
    ) -> Result<(Self, Array2<f64>), ENNError> {
        match self {
            Self::Cpu(opt) => {
                let (next, u) = opt.ask(num_arms, rng)?;
                Ok((Self::Cpu(next), u))
            }
            #[cfg(all(any(target_os = "macos", target_os = "ios"), feature = "metal"))]
            Self::Metal { inner, .. } => {
                let (next, u) = inner.ask(num_arms, rng)?;
                Ok((
                    Self::Metal {
                        inner: next,
                        ticket_tag: None,
                    },
                    u,
                ))
            }
            #[cfg(all(target_os = "linux", target_arch = "x86_64", feature = "cuda"))]
            Self::Cuda { inner, .. } => {
                let (next, u) = inner.ask(num_arms, rng)?;
                Ok((
                    Self::Cuda {
                        inner: next,
                        ticket_tag: None,
                    },
                    u,
                ))
            }
        }
    }

    /// Ingest observations in unit `[0, 1]^D` coordinates.
    pub fn tell(
        self,
        x: &ArrayView2<f64>,
        y: &ArrayView2<f64>,
        yvar: Option<&ArrayView2<f64>>,
        rng: &mut dyn RngCore,
    ) -> Result<Self, ENNError> {
        match self {
            Self::Cpu(opt) => Ok(Self::Cpu(opt.tell(x, y, yvar, rng)?)),
            #[cfg(all(any(target_os = "macos", target_os = "ios"), feature = "metal"))]
            Self::Metal { inner, .. } => Ok(Self::Metal {
                inner: inner.tell(x, y, yvar, rng)?,
                ticket_tag: None,
            }),
            #[cfg(all(target_os = "linux", target_arch = "x86_64", feature = "cuda"))]
            Self::Cuda { inner, .. } => Ok(Self::Cuda {
                inner: inner.tell(x, y, yvar, rng)?,
                ticket_tag: None,
            }),
        }
    }

    /// Propose `num_arms` candidates in natural domain coordinates.
    pub fn ask_natural(
        self,
        num_arms: usize,
        rng: &mut dyn RngCore,
    ) -> Result<(Self, Array2<f64>), ENNError> {
        match self {
            Self::Cpu(opt) => {
                let (next, nat) = opt.ask_natural(num_arms, rng)?;
                Ok((Self::Cpu(next), nat))
            }
            #[cfg(all(any(target_os = "macos", target_os = "ios"), feature = "metal"))]
            Self::Metal { inner, .. } => {
                let (next, nat) = inner.ask_natural(num_arms, rng)?;
                Ok((
                    Self::Metal {
                        inner: next,
                        ticket_tag: None,
                    },
                    nat,
                ))
            }
            #[cfg(all(target_os = "linux", target_arch = "x86_64", feature = "cuda"))]
            Self::Cuda { inner, .. } => {
                let (next, nat) = inner.ask_natural(num_arms, rng)?;
                Ok((
                    Self::Cuda {
                        inner: next,
                        ticket_tag: None,
                    },
                    nat,
                ))
            }
        }
    }

    /// Ingest observations in natural domain coordinates.
    pub fn tell_natural(
        self,
        x: &ArrayView2<f64>,
        y: &ArrayView2<f64>,
        yvar: Option<&ArrayView2<f64>>,
        rng: &mut dyn RngCore,
    ) -> Result<Self, ENNError> {
        match self {
            Self::Cpu(opt) => Ok(Self::Cpu(opt.tell_natural(x, y, yvar, rng)?)),
            #[cfg(all(any(target_os = "macos", target_os = "ios"), feature = "metal"))]
            Self::Metal { inner, .. } => Ok(Self::Metal {
                inner: inner.tell_natural(x, y, yvar, rng)?,
                ticket_tag: None,
            }),
            #[cfg(all(target_os = "linux", target_arch = "x86_64", feature = "cuda"))]
            Self::Cuda { inner, .. } => Ok(Self::Cuda {
                inner: inner.tell_natural(x, y, yvar, rng)?,
                ticket_tag: None,
            }),
        }
    }

    /// Import externally evaluated natural-coordinate observations without
    /// resolving pending proposals.
    pub fn import_natural(
        self,
        x: &ArrayView2<f64>,
        y: &ArrayView2<f64>,
        yvar: Option<&ArrayView2<f64>>,
        rng: &mut dyn RngCore,
    ) -> Result<Self, ENNError> {
        match self {
            Self::Cpu(opt) => Ok(Self::Cpu(opt.import_natural(x, y, yvar, rng)?)),
            #[cfg(all(any(target_os = "macos", target_os = "ios"), feature = "metal"))]
            Self::Metal { inner, ticket_tag } => Ok(Self::Metal {
                inner: inner.import_natural(x, y, yvar, rng)?,
                ticket_tag,
            }),
            #[cfg(all(target_os = "linux", target_arch = "x86_64", feature = "cuda"))]
            Self::Cuda { inner, ticket_tag } => Ok(Self::Cuda {
                inner: inner.import_natural(x, y, yvar, rng)?,
                ticket_tag,
            }),
        }
    }

    /// Restart a regional slot at `center` in natural coordinates.
    pub fn restart_natural(
        self,
        slot: usize,
        center: &ArrayView1<f64>,
        rng: &mut dyn RngCore,
    ) -> Result<(Self, Vec<EvaluationId>), ENNError> {
        match self {
            Self::Cpu(opt) => {
                let (next, abandoned) = opt.restart_natural(slot, center, rng)?;
                Ok((Self::Cpu(next), abandoned))
            }
            #[cfg(all(any(target_os = "macos", target_os = "ios"), feature = "metal"))]
            Self::Metal { inner, ticket_tag } => {
                let (next, abandoned) = inner.restart_natural(slot, center, rng)?;
                Ok((
                    Self::Metal {
                        inner: next,
                        ticket_tag,
                    },
                    abandoned,
                ))
            }
            #[cfg(all(target_os = "linux", target_arch = "x86_64", feature = "cuda"))]
            Self::Cuda { inner, ticket_tag } => {
                let (next, abandoned) = inner.restart_natural(slot, center, rng)?;
                Ok((
                    Self::Cuda {
                        inner: next,
                        ticket_tag,
                    },
                    abandoned,
                ))
            }
        }
    }

    /// Complete a subset of pending evaluations by identity.
    pub fn complete_ids(
        self,
        ids: &[EvaluationId],
        y: &ArrayView2<f64>,
        yvar: Option<&ArrayView2<f64>>,
        rng: &mut dyn RngCore,
    ) -> Result<Self, ENNError> {
        match self {
            Self::Cpu(opt) => Ok(Self::Cpu(opt.complete_ids(ids, y, yvar, rng)?)),
            #[cfg(all(any(target_os = "macos", target_os = "ios"), feature = "metal"))]
            Self::Metal { inner, .. } => Ok(Self::Metal {
                inner: inner.complete_ids(ids, y, yvar, rng)?,
                ticket_tag: None,
            }),
            #[cfg(all(target_os = "linux", target_arch = "x86_64", feature = "cuda"))]
            Self::Cuda { inner, .. } => Ok(Self::Cuda {
                inner: inner.complete_ids(ids, y, yvar, rng)?,
                ticket_tag: None,
            }),
        }
    }

    /// Retire unevaluated pending rows without controller penalty.
    pub fn cancel_ids(self, ids: &[EvaluationId], rng: &mut dyn RngCore) -> Result<Self, ENNError> {
        match self {
            Self::Cpu(opt) => Ok(Self::Cpu(opt.cancel_ids(ids, rng)?)),
            #[cfg(all(any(target_os = "macos", target_os = "ios"), feature = "metal"))]
            Self::Metal { inner, .. } => Ok(Self::Metal {
                inner: inner.cancel_ids(ids, rng)?,
                ticket_tag: None,
            }),
            #[cfg(all(target_os = "linux", target_arch = "x86_64", feature = "cuda"))]
            Self::Cuda { inner, .. } => Ok(Self::Cuda {
                inner: inner.cancel_ids(ids, rng)?,
                ticket_tag: None,
            }),
        }
    }

    /// Propose `num_arms` candidates as IEEE FP16 `[0, 1]^D` `u16` rows for
    /// zero-copy `eval_half` / `eval_halves` workers.
    pub fn ask_halves(
        self,
        num_arms: usize,
        rng: &mut dyn RngCore,
    ) -> Result<(Self, Vec<Vec<u16>>), ENNError> {
        #[cfg(all(any(target_os = "macos", target_os = "ios"), feature = "metal"))]
        if let Self::Metal { inner, .. } = self {
            let (next, rows) = inner.ask_halves(num_arms, rng)?;
            return Ok((
                Self::Metal {
                    inner: next,
                    ticket_tag: None,
                },
                rows,
            ));
        }
        let (next, unit) = self.ask(num_arms, rng)?;
        let rows = (0..unit.nrows())
            .map(|r| {
                unit.row(r)
                    .iter()
                    .map(|&u| encode_half((u as f32).clamp(0.0, 1.0)))
                    .collect()
            })
            .collect();
        Ok((next, rows))
    }

    /// Commit observations evaluated on IEEE FP16 `[0, 1]^D` `u16` rows.
    pub fn tell_halves(
        self,
        rows: &[Vec<u16>],
        y: &ArrayView2<f64>,
        yvar: Option<&ArrayView2<f64>>,
        rng: &mut dyn RngCore,
    ) -> Result<Self, ENNError> {
        #[cfg(all(any(target_os = "macos", target_os = "ios"), feature = "metal"))]
        if let Self::Metal { inner, .. } = self {
            let next = inner.tell_halves(rows, y, yvar, rng)?;
            return Ok(Self::Metal {
                inner: next,
                ticket_tag: None,
            });
        }
        let d = self.num_dim();
        if rows.len() != y.nrows() || rows.iter().any(|r| r.len() != d) {
            return Err(ENNError::InvalidShape {
                expected: vec![rows.len(), d],
                got: vec![y.nrows(), d],
            });
        }
        let mut unit = Array2::<f64>::zeros((rows.len(), d));
        for (i, r) in rows.iter().enumerate() {
            for (j, &w) in r.iter().enumerate() {
                unit[[i, j]] = f64::from(decode_half(w)).clamp(0.0, 1.0);
            }
        }
        self.tell(&unit.view(), y, yvar, rng)
    }

    /// Return the incumbent row encoded as IEEE FP16 `[0, 1]^D` `u16` words.
    pub fn best_halves(&self) -> Option<Vec<u16>> {
        #[cfg(all(any(target_os = "macos", target_os = "ios"), feature = "metal"))]
        if let Self::Metal { inner, .. } = self {
            return inner.best_halves();
        }
        self.unit_incumbent().map(|u| {
            u.iter()
                .map(|&v| encode_half((v as f32).clamp(0.0, 1.0)))
                .collect()
        })
    }

    /// Issue an affine [`ProposalTicket`] in natural domain coordinates.
    pub fn ask_ticket(
        self,
        num_arms: usize,
        rng: &mut dyn RngCore,
    ) -> Result<(ProposalTicket, Self), ENNError> {
        match self {
            Self::Cpu(opt) => {
                let (ticket, next) = opt.ask_ticket(num_arms, rng)?;
                Ok((ticket, Self::Cpu(next)))
            }
            #[cfg(all(any(target_os = "macos", target_os = "ios"), feature = "metal"))]
            Self::Metal { inner, .. } => {
                let (next_inner, natural) = inner.ask_natural(num_arms, rng)?;
                let n = natural.nrows();
                let ids = next_inner.all_pending().map(|all| {
                    let total = all.len();
                    all[total.saturating_sub(n)..].to_vec()
                });
                let mut raw = Vec::with_capacity(natural.len() * 8);
                for &v in natural.iter() {
                    raw.extend_from_slice(&v.to_le_bytes());
                }
                let tag = crate::artifact::blake3::hash(&raw);
                let next_self = Self::Metal {
                    inner: next_inner,
                    ticket_tag: Some(tag),
                };
                let origin_digest = Some(next_self.digest());
                Ok((
                    ProposalTicket::new_ticket(natural, ids, origin_digest),
                    next_self,
                ))
            }
            #[cfg(all(target_os = "linux", target_arch = "x86_64", feature = "cuda"))]
            Self::Cuda { inner, .. } => {
                let (next_inner, natural) = inner.ask_natural(num_arms, rng)?;
                let n = natural.nrows();
                let ids = next_inner.all_pending().map(|all| {
                    let total = all.len();
                    all[total.saturating_sub(n)..].to_vec()
                });
                let mut raw = Vec::with_capacity(natural.len() * 8);
                for &v in natural.iter() {
                    raw.extend_from_slice(&v.to_le_bytes());
                }
                let tag = crate::artifact::blake3::hash(&raw);
                let next_self = Self::Cuda {
                    inner: next_inner,
                    ticket_tag: Some(tag),
                };
                let origin_digest = Some(next_self.digest());
                Ok((
                    ProposalTicket::new_ticket(natural, ids, origin_digest),
                    next_self,
                ))
            }
        }
    }

    /// Consume an affine [`ProposalTicket`] by value and record its outcomes.
    pub fn tell_ticket(
        self,
        ticket: ProposalTicket,
        y: &ArrayView2<f64>,
        yvar: Option<&ArrayView2<f64>>,
        rng: &mut dyn RngCore,
    ) -> Result<Self, ENNError> {
        match self {
            Self::Cpu(opt) => Ok(Self::Cpu(opt.tell_ticket(ticket, y, yvar, rng)?)),
            #[cfg(all(any(target_os = "macos", target_os = "ios"), feature = "metal"))]
            Self::Metal { .. } => {
                let (x, ids, origin_digest) = ticket.into_parts();
                if let Some(expected) = origin_digest
                    && self.digest() != expected
                {
                    return Err(ENNError::InvalidParameter(
                        "ProposalTicket state digest mismatch: ticket belongs to a different optimizer branch or epoch"
                            .into(),
                    ));
                }
                let Self::Metal { inner, .. } = self else {
                    unreachable!()
                };
                if let Some(eval_ids) = ids {
                    return Ok(Self::Metal {
                        inner: inner.complete_ids(&eval_ids, y, yvar, rng)?,
                        ticket_tag: None,
                    });
                }
                Ok(Self::Metal {
                    inner: inner.tell_natural(&x.as_array().view(), y, yvar, rng)?,
                    ticket_tag: None,
                })
            }
            #[cfg(all(target_os = "linux", target_arch = "x86_64", feature = "cuda"))]
            Self::Cuda { .. } => {
                let (x, ids, origin_digest) = ticket.into_parts();
                if let Some(expected) = origin_digest
                    && self.digest() != expected
                {
                    return Err(ENNError::InvalidParameter(
                        "ProposalTicket state digest mismatch: ticket belongs to a different optimizer branch or epoch"
                            .into(),
                    ));
                }
                let Self::Cuda { inner, .. } = self else {
                    unreachable!()
                };
                if let Some(eval_ids) = ids {
                    return Ok(Self::Cuda {
                        inner: inner.complete_ids(&eval_ids, y, yvar, rng)?,
                        ticket_tag: None,
                    });
                }
                Ok(Self::Cuda {
                    inner: inner.tell_natural(&x.as_array().view(), y, yvar, rng)?,
                    ticket_tag: None,
                })
            }
        }
    }

    /// Consume an affine [`ProposalTicket`] by value and retire its pending
    /// evaluations without controller penalty.
    pub fn cancel_ticket(
        self,
        ticket: ProposalTicket,
        rng: &mut dyn RngCore,
    ) -> Result<Self, ENNError> {
        match self {
            Self::Cpu(opt) => Ok(Self::Cpu(opt.cancel_ticket(ticket, rng)?)),
            #[cfg(all(any(target_os = "macos", target_os = "ios"), feature = "metal"))]
            Self::Metal { .. } => {
                let (_x, ids, origin_digest) = ticket.into_parts();
                if let Some(expected) = origin_digest
                    && self.digest() != expected
                {
                    return Err(ENNError::InvalidParameter(
                        "ProposalTicket state digest mismatch: ticket belongs to a different optimizer branch or epoch"
                            .into(),
                    ));
                }
                let Self::Metal { inner, .. } = self else {
                    unreachable!()
                };
                if let Some(eval_ids) = ids {
                    return Ok(Self::Metal {
                        inner: inner.cancel_ids(&eval_ids, rng)?,
                        ticket_tag: None,
                    });
                }
                Ok(Self::Metal {
                    inner,
                    ticket_tag: None,
                })
            }
            #[cfg(all(target_os = "linux", target_arch = "x86_64", feature = "cuda"))]
            Self::Cuda { .. } => {
                let (_x, ids, origin_digest) = ticket.into_parts();
                if let Some(expected) = origin_digest
                    && self.digest() != expected
                {
                    return Err(ENNError::InvalidParameter(
                        "ProposalTicket state digest mismatch: ticket belongs to a different optimizer branch or epoch"
                            .into(),
                    ));
                }
                let Self::Cuda { inner, .. } = self else {
                    unreachable!()
                };
                if let Some(eval_ids) = ids {
                    return Ok(Self::Cuda {
                        inner: inner.cancel_ids(&eval_ids, rng)?,
                        ticket_tag: None,
                    });
                }
                Ok(Self::Cuda {
                    inner,
                    ticket_tag: None,
                })
            }
        }
    }

    /// Serialize canonical optimizer state with a BLAKE3 integrity tag.
    pub fn freeze(&self) -> Result<Vec<u8>, ENNError> {
        match self {
            Self::Cpu(opt) => opt.freeze(),
            #[cfg(all(any(target_os = "macos", target_os = "ios"), feature = "metal"))]
            Self::Metal { inner, .. } => inner.freeze_full(None),
            #[cfg(all(target_os = "linux", target_arch = "x86_64", feature = "cuda"))]
            Self::Cuda { inner, .. } => inner.freeze_full(None),
        }
    }

    /// Serialize canonical optimizer state plus host cursor metadata.
    pub fn freeze_host(&self, host: &HostCursor) -> Result<Vec<u8>, ENNError> {
        match self {
            Self::Cpu(opt) => opt.freeze_host(host),
            #[cfg(all(any(target_os = "macos", target_os = "ios"), feature = "metal"))]
            Self::Metal { inner, .. } => inner.freeze_full(Some(host)),
            #[cfg(all(target_os = "linux", target_arch = "x86_64", feature = "cuda"))]
            Self::Cuda { inner, .. } => inner.freeze_full(Some(host)),
        }
    }

    /// Restore an equivalent optimizer state from a BLAKE3-verified checkpoint.
    pub fn thaw(&self, bytes: &[u8]) -> Result<Self, ENNError> {
        match self {
            Self::Cpu(opt) => Ok(Self::Cpu(opt.thaw(bytes)?)),
            #[cfg(all(any(target_os = "macos", target_os = "ios"), feature = "metal"))]
            Self::Metal { inner, .. } => {
                let (next, _) = inner.thaw_full(bytes)?;
                Ok(Self::Metal {
                    inner: next,
                    ticket_tag: None,
                })
            }
            #[cfg(all(target_os = "linux", target_arch = "x86_64", feature = "cuda"))]
            Self::Cuda { inner, .. } => {
                let (next, _) = inner.thaw_full(bytes)?;
                Ok(Self::Cuda {
                    inner: next,
                    ticket_tag: None,
                })
            }
        }
    }

    /// Restore optimizer state and optional host cursor metadata.
    pub fn thaw_host(&self, bytes: &[u8]) -> Result<(Self, Option<HostCursor>), ENNError> {
        match self {
            Self::Cpu(opt) => {
                let (next, host) = opt.thaw_host(bytes)?;
                Ok((Self::Cpu(next), host))
            }
            #[cfg(all(any(target_os = "macos", target_os = "ios"), feature = "metal"))]
            Self::Metal { inner, .. } => {
                let (next, host) = inner.thaw_full(bytes)?;
                Ok((
                    Self::Metal {
                        inner: next,
                        ticket_tag: None,
                    },
                    host,
                ))
            }
            #[cfg(all(target_os = "linux", target_arch = "x86_64", feature = "cuda"))]
            Self::Cuda { inner, .. } => {
                let (next, host) = inner.thaw_full(bytes)?;
                Ok((
                    Self::Cuda {
                        inner: next,
                        ticket_tag: None,
                    },
                    host,
                ))
            }
        }
    }
}
