//! Resident Apple Silicon Metal GPU optimizer engine implementing the unified
//! ask/tell/complete/cancel/fork/digest/freeze protocol over `RegionalSearch`,
//! `SearchState`, `ResidentMorbo`, and `DomainFrame`.

use std::sync::Arc;
use std::time::Instant;

use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};
use rand::RngCore;

use super::{
    DeviceRow, ObjectiveAcquisition, ObjectivePolicy, ParamBlock, RegionalProposal, RegionalSearch,
    ResidentManifold, ResidentSampler, SearchState, regional_restarts, upload,
};
use crate::candidates::CandidateSampler;
use crate::config::{
    AcquisitionConfig, AcquisitionSpec, Dissimilarity, InitStrategy, ObservationSharing,
    OptimizerConfig, OptimizerSpec, RegionAllocation, RegionController, RegionMemory,
    RegionalSearchSpec, ResidentEnnConfig, RestartCenter, SurrogateConfig,
};
use crate::coordinate::DomainFrame;
use crate::error::ENNError;
use crate::hash::Key;
use crate::layout::MetricLearning;
use crate::objective_observation::{ObjectiveEstimate, ObjectiveObservation};
use crate::optimizer::multi_tr::{
    AllocationPolicy, MultiTrustRegionConfig, RegionController as HostController,
    RestartCenterSelection, RestartMemoryPolicy, SharingPolicy,
};
use crate::optimizer::{EvaluationId, HostCursor, Strategy, Telemetry};
use crate::procedural_pool::ProceduralPoolConfig;
use crate::proposal_distribution::{CoordSystem, ProposalDistribution};
use crate::scratch::ImLog;
use crate::sense::BetterScore;
use crate::snapshot_io::{
    MAX_DIM, MAX_METRICS, MAX_OBS, MAX_PENDING, WireReader, bad_state, seal_frame, write_bool,
    write_byte, write_header, write_len, write_mat, write_word,
};
use crate::trials::{AcquisitionParams, RoundConfig};
use crate::trust_region::{TRLengthConfig, TrustRegionConfig};
use crate::unified_mem::decode_half;
use crate::weights::AcquisitionKind;

const METAL_TAG: u8 = 0x4d;

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

#[inline]
fn unit_row(half: &[u16]) -> Vec<f64> {
    half.iter()
        .map(|&w| f64::from(decode_half(w)).clamp(0.0, 1.0))
        .collect()
}

#[inline]
fn half_row(unit: &[f64]) -> Vec<u16> {
    unit.iter()
        .map(|&u| encode_half((u as f32).clamp(0.0, 1.0)))
        .collect()
}

/// Generate `count` stratified Latin Hypercube Design (`LHD`) rows in `[0,
/// 1]^D` encoded as IEEE FP16 words via `DelayedPool::Lhd`.
pub fn lhd_rows(seed: u64, count: usize, num_dim: usize) -> Vec<Vec<u16>> {
    let n = count.max(1);
    let d = num_dim.max(1);
    let lo = vec![0.0f32; d];
    let hi = vec![1.0f32; d];
    let pool = crate::semantics::pool::DelayedPool::Lhd {
        key: Key::new(seed),
        lower: &lo,
        upper: &hi,
        num_candidates: n,
    };
    let mut coords = vec![0.0f32; d];
    let mut out = Vec::with_capacity(n);
    for cand in 0..n {
        pool.write_row(cand, &mut coords);
        out.push(
            coords[..num_dim]
                .iter()
                .map(|&u| encode_half(u.clamp(0.0, 1.0)))
                .collect(),
        );
    }
    out
}

fn random_rows(seed: u64, count: usize, num_dim: usize) -> Vec<Vec<u16>> {
    let n = count.max(1);
    let d = num_dim.max(1);
    let lo = vec![0.0f32; d];
    let hi = vec![1.0f32; d];
    let key = Key::new(seed);
    let mut out = Vec::with_capacity(n);
    for cand in 0..n {
        let row: Vec<u16> = (0..num_dim)
            .map(|j| {
                let u = crate::semantics::pool::eval_uniform::<f32>(key, &lo, &hi, cand, j);
                encode_half(u.clamp(0.0, 1.0))
            })
            .collect();
        out.push(row);
    }
    out
}

#[inline]
fn restart_seed(seed: u64, slot: usize, gen_seq: u64) -> u64 {
    crate::semantics::random::splitmix64(
        seed ^ ((slot as u64 + 1).wrapping_mul(0x9e37_79b9_7f4a_7c15))
            ^ gen_seq.wrapping_mul(0xbf58_476d_1ce4_e5b9),
    )
}

#[inline]
fn scalar_score(scores: &[f64]) -> f64 {
    scores.iter().copied().sum()
}

pub fn obj_obs(scores: &[f64], variances: Option<&[f64]>) -> Result<ObjectiveObservation, String> {
    if scores.len() <= 1 {
        let v = variances.and_then(|vs| vs.first().copied()).unwrap_or(0.0);
        return Ok(ObjectiveObservation::scalar_adapter(
            scores.first().copied().unwrap_or(0.0) as f32,
            v.max(0.0) as f32,
        ));
    }
    let items: Vec<ObjectiveEstimate> = scores
        .iter()
        .enumerate()
        .map(|(i, &val)| ObjectiveEstimate {
            mean: val as f32,
            variance: variances
                .and_then(|vs| vs.get(i).copied())
                .unwrap_or(0.0)
                .max(0.0) as f32,
        })
        .collect();
    let mean = items.iter().map(|e| e.mean).sum::<f32>() / (items.len() as f32);
    let var = items.iter().map(|e| e.variance).sum::<f32>() / ((items.len() * items.len()) as f32);
    ObjectiveObservation::new(
        &items,
        ObjectiveEstimate {
            mean,
            variance: var,
        },
    )
}

pub fn check_spec(spec: &OptimizerSpec) -> Result<(), ENNError> {
    match spec {
        OptimizerSpec::SingleStateBo {
            perturbation: Some(_),
            ..
        } => Err(ENNError::InvalidParameter(
            "resident-metal single-state-bo does not accept [optimizer.perturbation]".into(),
        )),
        OptimizerSpec::SingleStateBo {
            regional_search, ..
        } => {
            if let Some(rs) = regional_search {
                rs.validate().map_err(ENNError::InvalidParameter)?;
            }
            Ok(())
        }
        OptimizerSpec::BayesianSearch {
            candidates,
            arms,
            perturbation,
            regional_search,
            ..
        } => {
            if let Some(pert) = perturbation {
                pert.validate().map_err(ENNError::InvalidParameter)?;
            }
            if *arms == 0 || *candidates == 0 || !candidates.is_multiple_of(*arms) {
                return Err(ENNError::InvalidParameter(
                    "bayesian-search requires positive candidates divisible by arms".into(),
                ));
            }
            if let Some(rs) = regional_search {
                rs.validate().map_err(ENNError::InvalidParameter)?;
            }
            Ok(())
        }
        _ => Err(ENNError::InvalidParameter(
            "resident-metal backend requires single-state-bo or bayesian-search".into(),
        )),
    }
}

fn is_single(spec: &OptimizerSpec) -> bool {
    matches!(spec, OptimizerSpec::SingleStateBo { .. })
}

fn explicit_dist(spec: &OptimizerSpec) -> bool {
    matches!(
        spec,
        OptimizerSpec::BayesianSearch {
            distribution: Some(_),
            perturbation: None,
            ..
        }
    )
}

fn resolve_kernel(spec: &OptimizerSpec) -> ProposalDistribution {
    match spec {
        OptimizerSpec::SingleStateBo { distribution, .. }
        | OptimizerSpec::BayesianSearch {
            distribution: Some(distribution),
            ..
        } => *distribution,
        _ => ProposalDistribution::Gaussian,
    }
}

fn default_reg(num_dim: usize, arms: usize) -> RegionalSearchSpec {
    let tol = i32::try_from(crate::semantics::optimizer::turbo_tolerance(
        num_dim.max(1) as f64,
        arms.max(1),
    ))
    .unwrap_or(i32::MAX);
    RegionalSearchSpec {
        count: 1,
        sharing: ObservationSharing::OriginOnly,
        allocation: RegionAllocation::LeastTrials,
        controller: RegionController::Turbo,
        acquisitions: Vec::new(),
        restart_memory: RegionMemory::Fresh,
        restart_candidates: 1,
        restart_center: RestartCenter::First,
        success_tolerance: 3,
        failure_tolerance: tol,
    }
}

fn convert_multi(cfg: &MultiTrustRegionConfig) -> RegionalSearchSpec {
    let sharing = match cfg.sharing_policy {
        SharingPolicy::BoxContain => ObservationSharing::BoxContain,
        SharingPolicy::EllipsoidBall => ObservationSharing::EllipsoidBall,
        SharingPolicy::NearestCenter => ObservationSharing::NearestCenter,
        SharingPolicy::OriginOnly => ObservationSharing::OriginOnly,
    };
    let allocation = match cfg.allocation_policy {
        AllocationPolicy::IncumbentUcb1 => RegionAllocation::IncumbentUcb1,
        AllocationPolicy::LeastTrials => RegionAllocation::LeastTrials,
        AllocationPolicy::JointAcquisition => RegionAllocation::JointAcquisition,
    };
    let controller = match cfg.controller {
        HostController::Turbo => RegionController::Turbo,
        HostController::FixedRadius => RegionController::FixedRadius,
    };
    let restart_memory = match cfg.restart_memory_policy {
        RestartMemoryPolicy::Fresh => RegionMemory::Fresh,
        RestartMemoryPolicy::RetainRelevant => RegionMemory::RetainRelevant,
    };
    let restart_center = match cfg.restart_center_policy.selection {
        RestartCenterSelection::First => RestartCenter::First,
        RestartCenterSelection::Maximin => RestartCenter::Maximin,
    };
    let acquisitions = cfg
        .region_acquisitions
        .iter()
        .map(|opt| {
            opt.map(|acq| match acq {
                AcquisitionConfig::UCB { beta } => AcquisitionSpec::Ucb { beta },
                AcquisitionConfig::Thompson => AcquisitionSpec::Thompson,
                AcquisitionConfig::Random => AcquisitionSpec::Random,
                AcquisitionConfig::Pareto => AcquisitionSpec::Pareto { scales: Vec::new() },
            })
        })
        .collect();
    RegionalSearchSpec {
        count: cfg.num_regions.max(1),
        sharing,
        allocation,
        controller,
        acquisitions,
        restart_memory,
        restart_candidates: cfg.restart_center_policy.candidates.max(1),
        restart_center,
        success_tolerance: cfg.succ_tolerance.max(1),
        failure_tolerance: cfg.fail_tolerance.max(1),
    }
}

pub fn make_enn(
    cfg_opt: &OptimizerConfig,
    num_dim: usize,
    arms: u32,
    single_state: bool,
    seed: u64,
) -> Result<ResidentEnnConfig, ENNError> {
    let arms = arms.max(1);
    let total = if single_state {
        arms as usize
    } else {
        cfg_opt
            .candidates
            .num_candidates(num_dim.max(1), arms as usize)
            .max(arms as usize)
    };
    let slots = total.div_ceil(arms as usize).max(1) as u32;
    let pool = ProceduralPoolConfig {
        arms,
        candidates_per_arm: slots,
    }
    .resolve_resident()
    .map_err(ENNError::InvalidParameter)?;
    let (kind, beta) = match cfg_opt.acquisition {
        AcquisitionConfig::Random => (AcquisitionKind::Random, 0.0),
        AcquisitionConfig::UCB { beta } => (AcquisitionKind::Ucb, beta as f32),
        AcquisitionConfig::Thompson => (AcquisitionKind::Thompson, 1.0),
        AcquisitionConfig::Pareto => (AcquisitionKind::Pareto, 1.0),
    };
    let mut ask = RoundConfig::default();
    let tr_init = match &cfg_opt.trust_region {
        TrustRegionConfig::Turbo(l) => l.length_init,
        TrustRegionConfig::Morbo(m) => m.length.length_init,
    };
    ask.proposal.radius = tr_init as f32;
    ask.acquisition = AcquisitionParams { kind, beta, seed };
    let SurrogateConfig::ENN(ref enn) = cfg_opt.surrogate else {
        ask.surrogate.neighbors = 1;
        return Ok(ResidentEnnConfig {
            pool,
            ask,
            num_candidates: 1,
            num_samples: 1,
            fit_params: false,
            adapt_neighbors: false,
            distance_metric: Dissimilarity::Euclidean,
            local_scale_neighbors: 1,
        });
    };
    if enn.k < 1 {
        return Err(ENNError::InvalidParameter(format!(
            "k (number of neighbors) must be > 0, got {}",
            enn.k
        )));
    }
    ask.surrogate.neighbors = usize::try_from(enn.k.max(1)).unwrap_or(10).min(128);
    ask.surrogate.epistemic_scale = enn.epistemic_scale as f32;
    ask.surrogate.aleatoric_scale = enn.aleatoric_scale as f32;
    ask.surrogate.output_scale = enn.y_scale.unwrap_or(1.0) as f32;
    let (distance_metric, local_scale_neighbors) = match enn.geometry {
        crate::semantics::neighbor::DistanceMetric::Euclidean => (Dissimilarity::Euclidean, 8),
        crate::semantics::neighbor::DistanceMetric::Conformal { local_k } => {
            (Dissimilarity::Conformal, local_k)
        }
        crate::semantics::neighbor::DistanceMetric::Hyperbolic { local_k } => {
            (Dissimilarity::Hyperbolic, local_k)
        }
        crate::semantics::neighbor::DistanceMetric::Wasserstein { local_k } => {
            (Dissimilarity::Wasserstein, local_k)
        }
    };
    Ok(ResidentEnnConfig {
        pool,
        ask,
        num_candidates: enn.num_candidates.max(1),
        num_samples: enn.num_samples.max(1),
        fit_params: enn.num_candidates > 0 && enn.num_samples > 0,
        adapt_neighbors: enn.adapt_neighbors,
        distance_metric,
        local_scale_neighbors,
    })
}

fn make_manifold(cfg_opt: &OptimizerConfig, no_tr: bool) -> Option<ResidentManifold> {
    let coord_system = cfg_opt.candidates.coord_system;
    let noise_aware = cfg_opt.noise_aware;
    let (
        scale_x,
        metric_learning,
        tied_dims,
        unscaled_dims,
        stiefel_rank,
        calibrate_alpha,
        y_bounds,
    ) = match &cfg_opt.surrogate {
        SurrogateConfig::ENN(enn) => {
            let tied = if !enn.tied_dims.is_empty() {
                enn.tied_dims.clone()
            } else {
                cfg_opt.candidates.tied_dims.clone()
            };
            let yb = enn.y_bounds.as_ref().map(|arr| {
                (0..arr.nrows())
                    .map(|r| (arr[[r, 0]], arr[[r, 1]]))
                    .collect()
            });
            (
                enn.scale_x,
                enn.metric_learning,
                tied,
                enn.unscaled_dims.clone(),
                enn.stiefel_rank,
                enn.calibrate_alpha,
                yb,
            )
        }
        SurrogateConfig::None => (
            false,
            MetricLearning::None,
            cfg_opt.candidates.tied_dims.clone(),
            Vec::new(),
            None,
            None,
            None,
        ),
    };
    let active = scale_x
        || metric_learning != MetricLearning::None
        || !tied_dims.is_empty()
        || !unscaled_dims.is_empty()
        || coord_system != CoordSystem::Cartesian
        || stiefel_rank.is_some()
        || calibrate_alpha.is_some()
        || y_bounds.is_some()
        || noise_aware
        || no_tr;
    if !active {
        return None;
    }
    Some(ResidentManifold {
        scale_x,
        metric_learning,
        tied_dims,
        unscaled_dims,
        categories: Vec::new(),
        coord_system,
        stiefel_rank,
        calibrate_alpha,
        y_bounds,
        noise_aware,
        no_trust_region: no_tr,
    })
}

fn ask_retry(
    search: RegionalSearch,
    r_seed: u64,
    mut r_ask: RoundConfig,
) -> Result<(RegionalSearch, RegionalProposal), String> {
    let mut last_err = String::new();
    let current = search;
    for attempt in 0..4u64 {
        let s = r_seed.wrapping_add(attempt.wrapping_mul(0x9e37_79b9_7f4a_7c15));
        r_ask.acquisition.seed = s;
        match current.clone().ask(s, r_ask) {
            Ok((next_s, p)) => return Ok((next_s, p)),
            Err(e) => last_err = e,
        }
    }
    Err(last_err)
}

#[derive(Clone, Debug)]
enum PendingGroup {
    Warmup {
        id: EvaluationId,
        half: Vec<u16>,
    },
    Replace {
        slot: usize,
        gseq: u64,
        id: EvaluationId,
        half: Vec<u16>,
    },
    Active {
        prop: RegionalProposal,
        dev_rows: Vec<DeviceRow>,
        ids: Vec<EvaluationId>,
        halves: Vec<Vec<u16>>,
        resolved: Vec<Option<(Vec<f64>, Option<Vec<f64>>)>>,
        cancelled: Vec<bool>,
    },
}

impl PendingGroup {
    fn pending_ids(&self, out: &mut Vec<EvaluationId>) {
        match self {
            Self::Warmup { id, .. } | Self::Replace { id, .. } => out.push(*id),
            Self::Active {
                ids,
                resolved,
                cancelled,
                ..
            } => {
                for (i, &id) in ids.iter().enumerate() {
                    if resolved[i].is_none() && !cancelled[i] {
                        out.push(id);
                    }
                }
            }
        }
    }

    fn contains_id(&self, target: EvaluationId) -> bool {
        match self {
            Self::Warmup { id, .. } | Self::Replace { id, .. } => *id == target,
            Self::Active {
                ids,
                resolved,
                cancelled,
                ..
            } => ids
                .iter()
                .enumerate()
                .any(|(i, &id)| id == target && resolved[i].is_none() && !cancelled[i]),
        }
    }
}

/// Resident Apple Silicon Metal GPU optimizer engine with $O(1)$ `Arc` cloning.
#[derive(Clone)]
pub struct MetalEngine {
    domain: Arc<DomainFrame>,
    config: Arc<OptimizerConfig>,
    reg_spec: Arc<RegionalSearchSpec>,
    enn_res: ResidentEnnConfig,
    tr_cfg: TRLengthConfig,
    kernel: ProposalDistribution,
    single_state: bool,
    explicit_dist: bool,
    has_surr: bool,
    init_only: bool,
    identified: bool,
    init_kind: InitStrategy,
    warmup_goal: usize,
    base_cands: usize,
    default_arms: usize,
    seed: u64,
    step: u64,
    next_id: u64,
    policy: Option<ObjectivePolicy>,
    search: Option<RegionalSearch>,
    warm_buf: Arc<Vec<(Vec<u16>, Vec<f64>, Option<Vec<f64>>)>>,
    restarts: Arc<Vec<(usize, u64)>>,
    slot_gen: Arc<Vec<u64>>,
    pending: Arc<Vec<PendingGroup>>,
    x_log: ImLog,
    y_log: ImLog,
    yvar_log: ImLog,
    best_idx: Option<usize>,
    best_val: f64,
    best_row: Arc<Vec<u16>>,
    telemetry: Telemetry,
}

impl std::fmt::Debug for MetalEngine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MetalEngine")
            .field("num_dim", &self.domain.num_dim())
            .field("obs_count", &self.x_log.nrows())
            .field("num_regions", &self.reg_spec.count)
            .field("step", &self.step)
            .finish()
    }
}

impl MetalEngine {
    pub fn from_strategy(
        domain: DomainFrame,
        config: OptimizerConfig,
        strategy: Strategy,
        key: Key,
    ) -> Result<Self, ENNError> {
        let _runtime = crate::apple_gpu::Runtime::shared().map_err(ENNError::InvalidParameter)?;
        let num_dim = domain.num_dim();
        if let SurrogateConfig::ENN(ref enn) = config.surrogate {
            enn.validate(num_dim)?;
        }
        crate::metric_weights::validate_tied(&config.candidates.tied_dims, num_dim)?;
        if config.candidates.candidate_rv == CandidateSampler::Sobol
            && (num_dim == 0 || num_dim > 21201)
        {
            return Err(ENNError::InvalidParameter(format!(
                "Sobol dimension must be in 1..=21201, got {num_dim}"
            )));
        }

        let (init_only, identified, init_kind, raw_init, reg_spec) = match strategy {
            Strategy::Init(s) => (
                true,
                false,
                s.strategy_type,
                s.num_init,
                default_reg(num_dim, 1),
            ),
            Strategy::Turbo(_) => (false, false, InitStrategy::LHD, 0, default_reg(num_dim, 1)),
            Strategy::Hybrid { init, in_init, .. } => (
                false,
                false,
                init.strategy_type,
                if in_init { init.num_init } else { 0 },
                default_reg(num_dim, 1),
            ),
            Strategy::Experimental(s) => (
                false,
                true,
                s.init.strategy_type,
                if s.in_init { s.init.num_init } else { 0 },
                convert_multi(&s.multi_tr.config),
            ),
        };
        reg_spec.validate().map_err(ENNError::InvalidParameter)?;
        let seed = key.seed();
        let enn_res = make_enn(&config, num_dim, 1, false, seed)?;
        let tr_cfg = match &config.trust_region {
            TrustRegionConfig::Turbo(l) => *l,
            TrustRegionConfig::Morbo(m) => m.length,
        };
        let has_surr = matches!(config.surrogate, SurrogateConfig::ENN(_));
        let n_reg = reg_spec.count.max(1);
        let warmup_goal = if init_only {
            raw_init
        } else {
            raw_init.max(n_reg)
        };
        let base_cands = config.candidates.num_candidates(num_dim.max(1), 1).max(1);
        let init_row = lhd_rows(seed ^ 0x517c_c1b7_2722_0a95, 1, num_dim)
            .into_iter()
            .next()
            .unwrap_or_default();
        let policy = match &config.trust_region {
            TrustRegionConfig::Morbo(ms) => {
                let n_obj = ms.num_metrics.max(2);
                Some(ObjectivePolicy {
                    acquisition: ObjectiveAcquisition::Morbo {
                        weights: vec![1.0f32; n_obj],
                        regions: if reg_spec.count > 1 {
                            reg_spec.count
                        } else {
                            4
                        },
                        alpha: ms.alpha as f32,
                        seed,
                        rescalarize: ms.rescalarize,
                        clip: true,
                    },
                    scales: vec![1.0f32; n_obj],
                })
            }
            TrustRegionConfig::Turbo(_) => None,
        };

        Ok(Self {
            domain: Arc::new(domain),
            config: Arc::new(config),
            reg_spec: Arc::new(reg_spec),
            enn_res,
            tr_cfg,
            kernel: ProposalDistribution::Gaussian,
            single_state: false,
            explicit_dist: false,
            has_surr,
            init_only,
            identified,
            init_kind,
            warmup_goal,
            base_cands,
            default_arms: 1,
            seed,
            step: 0,
            next_id: 0,
            policy,
            search: None,
            warm_buf: Arc::new(Vec::new()),
            restarts: Arc::new(Vec::new()),
            slot_gen: Arc::new(vec![0; n_reg]),
            pending: Arc::new(Vec::new()),
            x_log: ImLog::with_cols(num_dim),
            y_log: ImLog::empty(),
            yvar_log: ImLog::empty(),
            best_idx: None,
            best_val: f64::NEG_INFINITY,
            best_row: Arc::new(init_row),
            telemetry: Telemetry::default(),
        })
    }

    pub fn from_spec(
        domain: DomainFrame,
        config: OptimizerConfig,
        spec: &OptimizerSpec,
        policy: Option<ObjectivePolicy>,
        num_init: usize,
        key: Key,
    ) -> Result<Self, ENNError> {
        check_spec(spec)?;
        let _runtime = crate::apple_gpu::Runtime::shared().map_err(ENNError::InvalidParameter)?;
        let num_dim = domain.num_dim();
        let single_state = is_single(spec);
        let explicit = explicit_dist(spec);
        if !single_state
            && !explicit
            && config.candidates.candidate_rv == CandidateSampler::Sobol
            && (num_dim == 0 || num_dim > 21201)
        {
            return Err(ENNError::InvalidParameter(format!(
                "Invalid parameter: Sobol dimension must be in 1..=21201, got {num_dim}"
            )));
        }
        let seed = key.seed();
        let arms_u32 = spec.candidate_generation().map_or(1, |cg| cg.arms.max(1));
        let arms = arms_u32 as usize;
        let enn_res = make_enn(&config, num_dim, arms_u32, single_state, seed)?;
        let tr_cfg = match &config.trust_region {
            TrustRegionConfig::Turbo(l) => *l,
            TrustRegionConfig::Morbo(m) => m.length,
        };
        let has_surr = matches!(config.surrogate, SurrogateConfig::ENN(_));
        let reg_spec = match spec.regional_search() {
            Some(rs) => rs.clone(),
            None => default_reg(num_dim, arms),
        };
        reg_spec.validate().map_err(ENNError::InvalidParameter)?;
        let n_reg = reg_spec.count.max(1);
        let base_goal = if has_surr && !single_state && !explicit {
            enn_res.ask.surrogate.neighbors.max(n_reg)
        } else {
            n_reg
        };
        let warmup_goal = num_init.max(base_goal);
        let base_cands = if single_state {
            arms
        } else {
            enn_res.pool.count() as usize
        };
        let resolved_policy = policy.or_else(|| match &config.trust_region {
            TrustRegionConfig::Morbo(ms) => {
                let n_obj = ms.num_metrics.max(2);
                Some(ObjectivePolicy {
                    acquisition: ObjectiveAcquisition::Morbo {
                        weights: vec![1.0f32; n_obj],
                        regions: if reg_spec.count > 1 {
                            reg_spec.count
                        } else {
                            4
                        },
                        alpha: ms.alpha as f32,
                        seed,
                        rescalarize: ms.rescalarize,
                        clip: true,
                    },
                    scales: vec![1.0f32; n_obj],
                })
            }
            TrustRegionConfig::Turbo(_) => None,
        });
        let init_row = lhd_rows(
            crate::semantics::random::splitmix64(seed ^ 0x9e37_79b9_7f4a_7c15),
            1,
            num_dim,
        )
        .into_iter()
        .next()
        .unwrap_or_default();

        Ok(Self {
            domain: Arc::new(domain),
            config: Arc::new(config),
            reg_spec: Arc::new(reg_spec),
            enn_res,
            tr_cfg,
            kernel: resolve_kernel(spec),
            single_state,
            explicit_dist: explicit,
            has_surr,
            init_only: false,
            identified: spec.regional_search().is_some(),
            init_kind: InitStrategy::LHD,
            warmup_goal,
            base_cands,
            default_arms: arms,
            seed,
            step: 0,
            next_id: 0,
            policy: resolved_policy,
            search: None,
            warm_buf: Arc::new(Vec::new()),
            restarts: Arc::new(Vec::new()),
            slot_gen: Arc::new(vec![0; n_reg]),
            pending: Arc::new(Vec::new()),
            x_log: ImLog::with_cols(num_dim),
            y_log: ImLog::empty(),
            yvar_log: ImLog::empty(),
            best_idx: None,
            best_val: f64::NEG_INFINITY,
            best_row: Arc::new(init_row),
            telemetry: Telemetry::default(),
        })
    }

    #[inline]
    pub fn default_arms(&self) -> usize {
        self.default_arms.max(1)
    }

    fn resolve_policy(&self, n_obj: usize) -> Option<ObjectivePolicy> {
        if n_obj < 2 {
            return None;
        }
        if let Some(ref p) = self.policy {
            let mut p = p.clone();
            if p.scales.len() != n_obj {
                p.scales = vec![1.0f32; n_obj];
            }
            if let ObjectiveAcquisition::Morbo {
                ref mut weights, ..
            } = p.acquisition
                && weights.len() != n_obj
            {
                *weights = vec![1.0f32; n_obj];
            }
            return Some(p);
        }
        Some(ObjectivePolicy {
            acquisition: ObjectiveAcquisition::Pareto,
            scales: vec![1.0f32; n_obj],
        })
    }

    fn make_region(
        &self,
        base: &[u16],
        slot: usize,
        gen_seq: u64,
        scores: &[f64],
        variances: Option<&[f64]>,
    ) -> Result<SearchState, ENNError> {
        let num_dim = self.domain.num_dim();
        let r_seed = restart_seed(self.seed, slot, gen_seq);
        let weight = 1.0f32 / (num_dim.max(1) as f32);
        let block =
            ParamBlock::new(r_seed, 0, num_dim, 0.5, weight).map_err(ENNError::InvalidParameter)?;
        let res_cap = if num_dim > 262_144 {
            (64 * 1024 * 1024 / (num_dim.max(1) * 2)).clamp(2, 128)
        } else {
            128
        };
        let mut st = if self.has_surr {
            let s = SearchState::new_implicit(base, vec![block], res_cap, self.tr_cfg, self.kernel)
                .map_err(ENNError::InvalidParameter)?;
            s.configure_enn(self.enn_res)
                .map_err(ENNError::InvalidParameter)?
        } else {
            let s = SearchState::new_fp16(base, vec![block], 0, self.tr_cfg, self.kernel)
                .map_err(ENNError::InvalidParameter)?;
            s.configure_pool(self.enn_res.pool)
                .map_err(ENNError::InvalidParameter)?
        };
        if !self.single_state {
            if self.explicit_dist {
                st = st
                    .configure_sampler(ResidentSampler::from_distribution(self.kernel), None, true)
                    .map_err(ENNError::InvalidParameter)?;
            } else {
                let n_obj = scores.len().max(1);
                let is_morbo = matches!(self.config.trust_region, TrustRegionConfig::Morbo(_))
                    || matches!(
                        self.resolve_policy(n_obj),
                        Some(ObjectivePolicy {
                            acquisition: ObjectiveAcquisition::Morbo { .. },
                            ..
                        })
                    );
                let sampler = match self.config.candidates.candidate_rv {
                    CandidateSampler::RAASP => ResidentSampler::Raasp,
                    CandidateSampler::Uniform => ResidentSampler::Uniform,
                    CandidateSampler::Sobol if is_morbo => ResidentSampler::SobolMasked,
                    CandidateSampler::Sobol => ResidentSampler::Sobol,
                };
                st = st
                    .configure_sampler(sampler, Some(self.config.candidates.num_pert), true)
                    .map_err(ENNError::InvalidParameter)?;
            }
            st = st
                .configure_lhd(self.base_cands.max(1))
                .map_err(ENNError::InvalidParameter)?;
        }
        if let Some(mf) = make_manifold(&self.config, self.init_only) {
            st = st
                .configure_manifold(mf)
                .map_err(ENNError::InvalidParameter)?;
        }
        let n_obj = scores.len().max(1);
        st = if n_obj >= 2 {
            let obs = obj_obs(scores, variances).map_err(ENNError::InvalidParameter)?;
            st.observe_objectives(obs)
                .map_err(ENNError::InvalidParameter)?
        } else {
            let v = variances.and_then(|vs| vs.first().copied()).unwrap_or(0.0);
            st.observe_initial(scores[0] as f32, v.max(0.0) as f32)
                .map_err(ENNError::InvalidParameter)?
        };
        if let Some(pol) = self.resolve_policy(n_obj) {
            st = st
                .configure_objectives(Some(pol))
                .map_err(ENNError::InvalidParameter)?;
        }
        let successes = usize::try_from(self.reg_spec.success_tolerance)
            .ok()
            .filter(|&v| v > 0)
            .ok_or_else(|| {
                ENNError::InvalidParameter("regional success tolerance must be positive".into())
            })?;
        let failures = match usize::try_from(self.reg_spec.failure_tolerance).ok() {
            Some(f) if f > 0 => f,
            _ => SearchState::failure_tolerance(num_dim, self.default_arms.max(1)),
        };
        let fixed = self.reg_spec.controller == RegionController::FixedRadius || self.init_only;
        st.set_tolerance(failures)
            .and_then(|s| s.set_successes(successes))
            .and_then(|s| s.fix_radius(fixed))
            .map_err(ENNError::InvalidParameter)
    }

    fn finish_warmup(&mut self) -> Result<(), ENNError> {
        let n_reg = self.reg_spec.count.max(1);
        if self.warm_buf.len() < n_reg {
            return Ok(());
        }
        let warm = (*self.warm_buf).clone();
        let n_obj = warm.first().map_or(1, |(_, scs, _)| scs.len().max(1));
        if n_reg == 1 {
            let mut best_idx = 0usize;
            let mut best_sc = BetterScore(f64::NEG_INFINITY);
            for (idx, (_, scs, _)) in warm.iter().enumerate() {
                let sc = BetterScore(scalar_score(scs));
                if idx == 0 || sc > best_sc {
                    best_sc = sc;
                    best_idx = idx;
                }
            }
            let st0 = self.make_region(
                &warm[best_idx].0,
                0,
                0,
                &warm[best_idx].1,
                warm[best_idx].2.as_deref(),
            )?;
            let mut s = RegionalSearch::from_spec(vec![st0], &self.reg_spec)
                .map_err(ENNError::InvalidParameter)?;
            if self.has_surr {
                let extras: Vec<_> = warm
                    .iter()
                    .enumerate()
                    .filter(|&(idx, _)| idx != best_idx)
                    .collect();
                let skip = extras.len().saturating_sub(127);
                for (_, (row, scs, vars)) in extras.into_iter().skip(skip) {
                    let buf = upload(row).map_err(ENNError::InvalidParameter)?;
                    s = if n_obj >= 2 {
                        let obs =
                            obj_obs(scs, vars.as_deref()).map_err(ENNError::InvalidParameter)?;
                        s.import_vec(0, &buf, obs)
                            .map_err(ENNError::InvalidParameter)?
                    } else {
                        let v = vars
                            .as_ref()
                            .and_then(|vs| vs.first().copied())
                            .unwrap_or(0.0);
                        s.import_region(0, &buf, scs[0] as f32, v.max(0.0) as f32)
                            .map_err(ENNError::InvalidParameter)?
                    };
                }
            }
            self.warm_buf = Arc::new(Vec::new());
            self.search = Some(s);
            return Ok(());
        }

        let mut states = Vec::with_capacity(n_reg);
        for slot in 0..n_reg {
            let (ref row, ref scs, ref vars) = warm[slot];
            states.push(self.make_region(row, slot, 0, scs, vars.as_deref())?);
        }
        let mut s = RegionalSearch::from_spec(states, &self.reg_spec)
            .map_err(ENNError::InvalidParameter)?;
        if self.has_surr && warm.len() > n_reg {
            let rem = &warm[n_reg..];
            let max_extra = n_reg * 127;
            let skip = rem.len().saturating_sub(max_extra);
            for (offset, (row, scs, vars)) in rem.iter().skip(skip).enumerate() {
                let buf = upload(row).map_err(ENNError::InvalidParameter)?;
                let slot = offset % n_reg;
                s = if n_obj >= 2 {
                    let obs = obj_obs(scs, vars.as_deref()).map_err(ENNError::InvalidParameter)?;
                    s.import_vec(slot, &buf, obs)
                        .map_err(ENNError::InvalidParameter)?
                } else {
                    let v = vars
                        .as_ref()
                        .and_then(|vs| vs.first().copied())
                        .unwrap_or(0.0);
                    s.import_region(slot, &buf, scs[0] as f32, v.max(0.0) as f32)
                        .map_err(ENNError::InvalidParameter)?
                };
            }
        }
        self.warm_buf = Arc::new(Vec::new());
        self.search = Some(s);
        Ok(())
    }

    fn issue_id(&mut self) -> EvaluationId {
        let id = EvaluationId(self.next_id);
        self.next_id = self.next_id.saturating_add(1);
        id
    }

    fn record_obs(
        &mut self,
        half: &[u16],
        unit_override: Option<&[f64]>,
        y_row: &[f64],
        yvar_row: Option<&[f64]>,
    ) -> Result<(), ENNError> {
        let idx = self.x_log.nrows();
        let u_vec = match unit_override {
            Some(u) => u.to_vec(),
            None => unit_row(half),
        };
        self.x_log = self.x_log.clone().push_slice(&u_vec)?;
        self.y_log = self.y_log.clone().push_slice(y_row)?;
        let v_vec: Vec<f64> = match yvar_row {
            Some(vs) => vs.to_vec(),
            None => vec![0.0; y_row.len()],
        };
        self.yvar_log = self.yvar_log.clone().push_slice(&v_vec)?;
        let sc = if y_row.len() > 1 {
            scalar_score(y_row)
        } else {
            y_row.first().copied().unwrap_or(f64::NEG_INFINITY)
        };
        if self.best_idx.is_none() || BetterScore(sc) > BetterScore(self.best_val) {
            self.best_idx = Some(idx);
            self.best_val = sc;
            self.best_row = Arc::new(half.to_vec());
        }
        Ok(())
    }

    /// Propose `num_arms` candidate rows directly as IEEE FP16 `[0, 1]^D` `u16`
    /// vectors on the Apple Silicon Metal GPU.
    pub fn ask_halves(
        mut self,
        num_arms: usize,
        rng: &mut dyn RngCore,
    ) -> Result<(Self, Vec<Vec<u16>>), ENNError> {
        if num_arms == 0 {
            return Err(ENNError::InvalidParameter(
                "num_arms must be > 0, got 0".to_string(),
            ));
        }
        let t0 = Instant::now();
        self.step = self.step.wrapping_add(1);
        let draw_seed = crate::semantics::random::splitmix64(
            rng.next_u64() ^ self.step.wrapping_mul(0x9e37_79b9_7f4a_7c15),
        );
        let num_dim = self.domain.num_dim();

        if self.init_only || self.search.is_none() {
            let rows = match self.init_kind {
                InitStrategy::LHD => lhd_rows(draw_seed, num_arms, num_dim),
                InitStrategy::Random => random_rows(draw_seed, num_arms, num_dim),
            };
            let p_mut = Arc::make_mut(&mut self.pending);
            for r in &rows {
                let id = EvaluationId(self.next_id);
                self.next_id = self.next_id.saturating_add(1);
                p_mut.push(PendingGroup::Warmup {
                    id,
                    half: r.clone(),
                });
            }
            self.telemetry.dt_gen = t0.elapsed().as_secs_f64();
            self.telemetry.num_candidates = num_arms;
            return Ok((self, rows));
        }

        if self
            .pending
            .iter()
            .any(|g| matches!(g, PendingGroup::Replace { .. }))
        {
            return Err(ENNError::InvalidParameter(
                "pending regional replacement must be completed or cancelled before proposing"
                    .into(),
            ));
        }

        let mut flat_rows = Vec::with_capacity(num_arms);
        let remaining = num_arms;

        if !self.restarts.is_empty() {
            let s_ref = self.search.as_ref().expect("search initialized");
            let r_mut = Arc::make_mut(&mut self.restarts);
            let take_r = r_mut.len().min(remaining);
            let drained: Vec<(usize, u64)> = r_mut.drain(..take_r).collect();
            let mut r_groups = Vec::with_capacity(drained.len());
            for (slot, gseq) in drained {
                let c_seed = restart_seed(draw_seed, slot, gseq);
                let row = s_ref
                    .sample_center(slot, c_seed)
                    .map_err(ENNError::InvalidParameter)?;
                let id = EvaluationId(self.next_id);
                self.next_id = self.next_id.saturating_add(1);
                flat_rows.push(row.clone());
                r_groups.push(PendingGroup::Replace {
                    slot,
                    gseq,
                    id,
                    half: row,
                });
            }
            Arc::make_mut(&mut self.pending).extend(r_groups);
            self.telemetry.num_candidates = take_r;
            self.telemetry.dt_gen = t0.elapsed().as_secs_f64();
            return Ok((self, flat_rows));
        }

        if remaining > 0 {
            let mut s = self.search.take().expect("search initialized");
            let idle_count = (0..s.len())
                .filter(|&slot| {
                    s.is_idle(slot)
                        && !self.restarts.iter().any(|&(rs, _)| rs == slot)
                        && !self.pending.iter().any(
                            |g| matches!(g, PendingGroup::Replace { slot: ps, .. } if *ps == slot),
                        )
                })
                .count();
            if idle_count == 0 {
                self.search = Some(s);
                return Err(ENNError::InvalidParameter(
                    "no eligible idle region slots available; complete or cancel pending evaluations first"
                        .into(),
                ));
            }
            let issue_n = idle_count.min(remaining).max(1);
            let mut new_groups = Vec::with_capacity(issue_n);
            let mut total_cands = 0usize;
            for q in 0..issue_n {
                let quota = (remaining / issue_n) + usize::from(q < (remaining % issue_n));
                let slots_per_arm = if self.single_state {
                    1u32
                } else {
                    self.base_cands.max(quota).div_ceil(quota).max(1) as u32
                };
                let pool = ProceduralPoolConfig {
                    arms: quota as u32,
                    candidates_per_arm: slots_per_arm,
                }
                .resolve_resident()
                .map_err(ENNError::InvalidParameter)?;
                s = s.resize_idle(pool).map_err(ENNError::InvalidParameter)?;
                total_cands += pool.count() as usize;

                let q_seed = crate::semantics::random::splitmix64(
                    draw_seed ^ ((q as u64).wrapping_mul(0xbf58_476d_1ce4_e5b9)),
                );
                let (next_s, prop) =
                    ask_retry(s, q_seed, self.enn_res.ask).map_err(ENNError::InvalidParameter)?;
                s = next_s;
                let (dev_rows, host_rows) = if quota <= 1 {
                    let buf = s.buffer(&prop).map_err(ENNError::InvalidParameter)?;
                    let ptr = buf.contents().cast::<u16>();
                    let row = unsafe { std::slice::from_raw_parts(ptr, num_dim) }.to_vec();
                    let drow = prop.proposal().device_row().clone();
                    drop(buf);
                    (vec![drow], vec![row])
                } else {
                    let batch = s
                        .materialize_batch(&prop)
                        .map_err(ENNError::InvalidParameter)?;
                    let take_n = quota.min(batch.len()).max(1);
                    let mut hrs = Vec::with_capacity(take_n);
                    let mut drs = Vec::with_capacity(take_n);
                    for dev in batch.into_iter().take(take_n) {
                        hrs.push(dev.read_u16().map_err(ENNError::InvalidParameter)?);
                        drs.push(dev);
                    }
                    (drs, hrs)
                };
                let mut ids = Vec::with_capacity(host_rows.len());
                for _ in 0..host_rows.len() {
                    let id = EvaluationId(self.next_id);
                    self.next_id = self.next_id.saturating_add(1);
                    ids.push(id);
                }
                let k = host_rows.len();
                flat_rows.extend(host_rows.iter().cloned());
                new_groups.push(PendingGroup::Active {
                    prop,
                    dev_rows,
                    ids,
                    halves: host_rows,
                    resolved: vec![None; k],
                    cancelled: vec![false; k],
                });
            }
            self.search = Some(s);
            Arc::make_mut(&mut self.pending).extend(new_groups);
            self.telemetry.num_candidates = total_cands;
        }
        self.telemetry.dt_gen = t0.elapsed().as_secs_f64();
        Ok((self, flat_rows))
    }

    fn flush_groups(&mut self) -> Result<(), ENNError> {
        let old_groups = std::mem::take(Arc::make_mut(&mut self.pending));
        let mut kept = Vec::new();
        for group in old_groups {
            match group {
                PendingGroup::Warmup { .. } | PendingGroup::Replace { .. } => kept.push(group),
                PendingGroup::Active {
                    prop,
                    dev_rows,
                    ids,
                    halves,
                    resolved,
                    cancelled,
                } => {
                    let all_done = (0..ids.len()).all(|i| resolved[i].is_some() || cancelled[i]);
                    if !all_done {
                        kept.push(PendingGroup::Active {
                            prop,
                            dev_rows,
                            ids,
                            halves,
                            resolved,
                            cancelled,
                        });
                        continue;
                    }
                    let mut s = self.search.take().ok_or_else(|| {
                        ENNError::InvalidParameter("Metal search missing on commit".into())
                    })?;
                    let slot = prop.ticket().region.slot;
                    let mut active_devs = Vec::new();
                    let mut active_obs = Vec::new();
                    let mut active_sel = Vec::new();
                    for i in 0..ids.len() {
                        if let Some((ref y_row, ref yv_row)) = resolved[i] {
                            let obs = obj_obs(y_row, yv_row.as_deref())
                                .map_err(ENNError::InvalidParameter)?;
                            active_devs.push(dev_rows[i].clone());
                            active_obs.push(obs);
                            if let Some(&cand_idx) = prop.proposal().selected_indices.get(i) {
                                active_sel.push(cand_idx);
                            }
                        }
                    }
                    if active_obs.is_empty() {
                        s = s.cancel(&prop).map_err(ENNError::InvalidParameter)?;
                    } else if ids.len() == 1 && active_obs.len() == 1 {
                        let (next_s, _dec) = if active_obs[0].is_vector() {
                            s.tell_objectives(&prop, active_obs[0])
                                .map_err(ENNError::InvalidParameter)?
                        } else {
                            let ctrl = active_obs[0].control();
                            s.tell(&prop, ctrl.mean, ctrl.variance)
                                .map_err(ENNError::InvalidParameter)?
                        };
                        s = next_s;
                        if regional_restarts(&s, slot).map_err(ENNError::InvalidParameter)? > 0 {
                            let sg = Arc::make_mut(&mut self.slot_gen);
                            sg[slot] += 1;
                            let gseq = sg[slot];
                            Arc::make_mut(&mut self.restarts).push((slot, gseq));
                        }
                    } else {
                        let mut adj_prop = prop;
                        if active_sel.len() == active_devs.len() {
                            adj_prop.proposal.selected_indices = active_sel;
                        }
                        let (next_s, _dec) = s
                            .tell_arms(&adj_prop, &active_devs, &active_obs)
                            .map_err(ENNError::InvalidParameter)?;
                        s = next_s;
                        if regional_restarts(&s, slot).map_err(ENNError::InvalidParameter)? > 0 {
                            let sg = Arc::make_mut(&mut self.slot_gen);
                            sg[slot] += 1;
                            let gseq = sg[slot];
                            Arc::make_mut(&mut self.restarts).push((slot, gseq));
                        }
                    }
                    self.search = Some(s);
                }
            }
        }
        self.pending = Arc::new(kept);
        if !self.init_only && self.search.is_none() && self.warm_buf.len() >= self.warmup_goal {
            self.finish_warmup()?;
        }
        Ok(())
    }

    /// Commit observations directly against issued FP16 `u16` rows without
    /// round-trip loss.
    pub fn tell_halves(
        self,
        rows: &[Vec<u16>],
        y: &ArrayView2<f64>,
        yvar: Option<&ArrayView2<f64>>,
        rng: &mut dyn RngCore,
    ) -> Result<Self, ENNError> {
        let d = self.domain.num_dim();
        if rows.len() != y.nrows() || rows.iter().any(|r| r.len() != d) {
            return Err(ENNError::InvalidShape {
                expected: vec![rows.len(), d],
                got: vec![y.nrows(), d],
            });
        }
        let mut p_ids = Vec::new();
        for g in self.pending.iter() {
            g.pending_ids(&mut p_ids);
        }
        if p_ids.len() == rows.len() && !p_ids.is_empty() {
            return self.complete_inner(&p_ids, y, yvar, None);
        }
        let mut unit = Array2::<f64>::zeros((rows.len(), d));
        for (i, r) in rows.iter().enumerate() {
            for (j, &w) in r.iter().enumerate() {
                unit[[i, j]] = f64::from(decode_half(w)).clamp(0.0, 1.0);
            }
        }
        self.tell(&unit.view(), y, yvar, rng)
    }

    pub fn best_halves(&self) -> Option<Vec<u16>> {
        self.best_idx.map(|_| (*self.best_row).clone())
    }

    pub fn ask(
        self,
        num_arms: usize,
        rng: &mut dyn RngCore,
    ) -> Result<(Self, Array2<f64>), ENNError> {
        let d = self.domain.num_dim();
        let (next, halves) = self.ask_halves(num_arms, rng)?;
        let mut unit = Array2::<f64>::zeros((halves.len(), d));
        for (i, r) in halves.iter().enumerate() {
            for (j, &w) in r.iter().enumerate() {
                unit[[i, j]] = f64::from(decode_half(w)).clamp(0.0, 1.0);
            }
        }
        Ok((next, unit))
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

    fn complete_inner(
        mut self,
        ids: &[EvaluationId],
        y: &ArrayView2<f64>,
        yvar: Option<&ArrayView2<f64>>,
        unit_override: Option<&ArrayView2<f64>>,
    ) -> Result<Self, ENNError> {
        if ids.is_empty() || y.nrows() != ids.len() || y.ncols() == 0 {
            return Err(ENNError::InvalidParameter(
                "ID completion requires non-empty matching outcomes".into(),
            ));
        }
        if let Some(yv) = yvar
            && yv.dim() != y.dim()
        {
            return Err(ENNError::InvalidShape {
                expected: vec![y.nrows(), y.ncols()],
                got: vec![yv.nrows(), yv.ncols()],
            });
        }
        if y.iter().any(|v| !v.is_finite())
            || yvar.is_some_and(|yv| yv.iter().any(|v| !v.is_finite() || *v < 0.0))
        {
            return Err(ENNError::InvalidParameter(
                "observations and variances must be finite (and variances >= 0)".into(),
            ));
        }
        let t0 = Instant::now();
        // Verify all requested IDs exist and are distinct
        let mut seen = std::collections::HashSet::with_capacity(ids.len());
        for &id in ids {
            if !seen.insert(id) || !self.pending.iter().any(|g| g.contains_id(id)) {
                return Err(ENNError::InvalidParameter(format!(
                    "evaluation ID {} is not pending in this optimizer state",
                    id.0
                )));
            }
        }

        for (row_idx, &target_id) in ids.iter().enumerate() {
            let y_row: Vec<f64> = y.row(row_idx).to_vec();
            let yv_row: Option<Vec<f64>> = yvar.map(|yv| yv.row(row_idx).to_vec());
            let u_opt: Option<Vec<f64>> = unit_override.map(|u| u.row(row_idx).to_vec());

            let groups = Arc::make_mut(&mut self.pending);
            let pos = groups
                .iter()
                .position(|g| g.contains_id(target_id))
                .expect("verified pending id");
            match &mut groups[pos] {
                PendingGroup::Warmup { .. } => {
                    let PendingGroup::Warmup { half, .. } = groups.remove(pos) else {
                        unreachable!()
                    };
                    self.record_obs(&half, u_opt.as_deref(), &y_row, yv_row.as_deref())?;
                    Arc::make_mut(&mut self.warm_buf).push((half, y_row, yv_row));
                }
                PendingGroup::Replace { .. } => {
                    let PendingGroup::Replace {
                        slot, gseq, half, ..
                    } = groups.remove(pos)
                    else {
                        unreachable!()
                    };
                    self.record_obs(&half, u_opt.as_deref(), &y_row, yv_row.as_deref())?;
                    let st = self.make_region(&half, slot, gseq, &y_row, yv_row.as_deref())?;
                    let s = self.search.take().ok_or_else(|| {
                        ENNError::InvalidParameter("Metal search missing on restart".into())
                    })?;
                    self.search = Some(s.restart(slot, st).map_err(ENNError::InvalidParameter)?);
                }
                PendingGroup::Active {
                    ids: g_ids,
                    halves,
                    resolved,
                    ..
                } => {
                    let arm_pos = g_ids.iter().position(|&id| id == target_id).unwrap();
                    let half = halves[arm_pos].clone();
                    resolved[arm_pos] = Some((y_row.clone(), yv_row.clone()));
                    self.record_obs(&half, u_opt.as_deref(), &y_row, yv_row.as_deref())?;
                }
            }
        }
        self.flush_groups()?;
        self.telemetry.dt_tell = t0.elapsed().as_secs_f64();
        Ok(self)
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
        if x.iter().any(|v| !v.is_finite())
            || y.iter().any(|v| !v.is_finite())
            || yvar.is_some_and(|yv| {
                yv.dim() != y.dim() || yv.iter().any(|v| !v.is_finite() || *v < 0.0)
            })
        {
            return Err(ENNError::InvalidParameter(
                "observations and variances must be finite with matching shapes".into(),
            ));
        }
        let mut p_ids = Vec::new();
        for g in self.pending.iter() {
            g.pending_ids(&mut p_ids);
        }
        if p_ids.len() == x.nrows() && !p_ids.is_empty() {
            return self.complete_inner(&p_ids, y, yvar, Some(x));
        }

        // Direct external/batch submission when no matching pending ask exists
        let t0 = Instant::now();
        if !self.pending.is_empty() {
            // Cancel any stale pending proposals on un-identified ordered tell
            let all_ids = p_ids;
            self = self.cancel_ids(&all_ids, _rng)?;
        }
        for i in 0..x.nrows() {
            let u_row: Vec<f64> = x.row(i).iter().map(|&v| v.clamp(0.0, 1.0)).collect();
            let h_row = half_row(&u_row);
            let y_row: Vec<f64> = y.row(i).to_vec();
            let yv_row: Option<Vec<f64>> = yvar.map(|yv| yv.row(i).to_vec());
            self.record_obs(&h_row, Some(&u_row), &y_row, yv_row.as_deref())?;
            if let Some(s) = self.search.take() {
                if self.has_surr {
                    let buf = upload(&h_row).map_err(ENNError::InvalidParameter)?;
                    let slot = i % s.len().max(1);
                    let next_s = if y_row.len() >= 2 {
                        let obs = obj_obs(&y_row, yv_row.as_deref())
                            .map_err(ENNError::InvalidParameter)?;
                        s.import_vec(slot, &buf, obs)
                            .map_err(ENNError::InvalidParameter)?
                    } else {
                        let v = yv_row
                            .as_ref()
                            .and_then(|vs| vs.first().copied())
                            .unwrap_or(0.0);
                        s.import_region(slot, &buf, y_row[0] as f32, v.max(0.0) as f32)
                            .map_err(ENNError::InvalidParameter)?
                    };
                    self.search = Some(next_s);
                } else {
                    self.search = Some(s);
                }
            } else {
                Arc::make_mut(&mut self.warm_buf).push((h_row, y_row, yv_row));
            }
        }
        if !self.init_only && self.search.is_none() && self.warm_buf.len() >= self.warmup_goal {
            self.finish_warmup()?;
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
        if x.ncols() != self.domain.num_dim() || x.nrows() != y.nrows() {
            return Err(ENNError::InvalidShape {
                expected: vec![x.nrows(), self.domain.num_dim()],
                got: vec![y.nrows(), x.ncols()],
            });
        }
        let unit = self.domain.to_unit(x);
        self.tell(&unit.view(), y, yvar, rng)
    }

    pub fn import_natural(
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
        let unit = self.domain.to_unit(x);
        for i in 0..unit.nrows() {
            let u_row: Vec<f64> = unit.row(i).iter().map(|&v| v.clamp(0.0, 1.0)).collect();
            let h_row = half_row(&u_row);
            let y_row: Vec<f64> = y.row(i).to_vec();
            let yv_row: Option<Vec<f64>> = yvar.map(|yv| yv.row(i).to_vec());
            self.record_obs(&h_row, Some(&u_row), &y_row, yv_row.as_deref())?;
            if let Some(s) = self.search.take() {
                if self.has_surr {
                    let buf = upload(&h_row).map_err(ENNError::InvalidParameter)?;
                    let slot = i % s.len().max(1);
                    let next_s = if y_row.len() >= 2 {
                        let obs = obj_obs(&y_row, yv_row.as_deref())
                            .map_err(ENNError::InvalidParameter)?;
                        s.import_vec(slot, &buf, obs)
                            .map_err(ENNError::InvalidParameter)?
                    } else {
                        let v = yv_row
                            .as_ref()
                            .and_then(|vs| vs.first().copied())
                            .unwrap_or(0.0);
                        s.import_region(slot, &buf, y_row[0] as f32, v.max(0.0) as f32)
                            .map_err(ENNError::InvalidParameter)?
                    };
                    self.search = Some(next_s);
                } else {
                    self.search = Some(s);
                }
            } else {
                Arc::make_mut(&mut self.warm_buf).push((h_row, y_row, yv_row));
            }
        }
        if !self.init_only && self.search.is_none() && self.warm_buf.len() >= self.warmup_goal {
            self.finish_warmup()?;
        }
        Ok(self)
    }

    pub fn complete_ids(
        self,
        ids: &[EvaluationId],
        y: &ArrayView2<f64>,
        yvar: Option<&ArrayView2<f64>>,
        _rng: &mut dyn RngCore,
    ) -> Result<Self, ENNError> {
        self.complete_inner(ids, y, yvar, None)
    }

    pub fn cancel_ids(
        mut self,
        ids: &[EvaluationId],
        _rng: &mut dyn RngCore,
    ) -> Result<Self, ENNError> {
        if ids.is_empty() {
            return Err(ENNError::InvalidParameter(
                "cancel_ids requires at least one evaluation ID".into(),
            ));
        }
        let mut seen = std::collections::HashSet::with_capacity(ids.len());
        for &id in ids {
            if !seen.insert(id) || !self.pending.iter().any(|g| g.contains_id(id)) {
                return Err(ENNError::InvalidParameter(format!(
                    "evaluation ID {} is not pending in this optimizer state",
                    id.0
                )));
            }
        }
        for &target_id in ids {
            let groups = Arc::make_mut(&mut self.pending);
            let pos = groups
                .iter()
                .position(|g| g.contains_id(target_id))
                .expect("verified pending id");
            match &mut groups[pos] {
                PendingGroup::Warmup { .. } => {
                    groups.remove(pos);
                }
                PendingGroup::Replace { .. } => {
                    let PendingGroup::Replace { slot, gseq, .. } = groups.remove(pos) else {
                        unreachable!()
                    };
                    Arc::make_mut(&mut self.restarts).push((slot, gseq));
                }
                PendingGroup::Active {
                    ids: g_ids,
                    cancelled,
                    ..
                } => {
                    let arm_pos = g_ids.iter().position(|&id| id == target_id).unwrap();
                    cancelled[arm_pos] = true;
                }
            }
        }
        self.flush_groups()?;
        Ok(self)
    }

    pub fn restart_natural(
        mut self,
        slot: usize,
        center: &ArrayView1<f64>,
        rng: &mut dyn RngCore,
    ) -> Result<(Self, Vec<EvaluationId>), ENNError> {
        let d = self.domain.num_dim();
        if center.len() != d {
            return Err(ENNError::InvalidShape {
                expected: vec![d],
                got: vec![center.len()],
            });
        }
        if slot >= self.reg_spec.count {
            return Err(ENNError::InvalidParameter(format!(
                "region slot {slot} out of range 0..{}",
                self.reg_spec.count
            )));
        }
        if self.search.is_none() {
            return Err(ENNError::InvalidParameter(
                "regional restart requires completed warm-up".into(),
            ));
        }
        let nat_mat = center.to_owned().insert_axis(Axis(0));
        let unit = self.domain.to_unit(&nat_mat.view());
        if unit
            .iter()
            .any(|&v| !v.is_finite() || !(0.0..=1.0).contains(&v))
        {
            return Err(ENNError::InvalidParameter(
                "regional restart center must lie within domain bounds".into(),
            ));
        }
        let mut abandoned = Vec::new();
        for g in self.pending.iter() {
            match g {
                PendingGroup::Replace { slot: s, id, .. } if *s == slot => abandoned.push(*id),
                PendingGroup::Active {
                    prop,
                    ids,
                    resolved,
                    cancelled,
                    ..
                } if prop.ticket().region.slot == slot => {
                    for (i, &id) in ids.iter().enumerate() {
                        if resolved[i].is_none() && !cancelled[i] {
                            abandoned.push(id);
                        }
                    }
                }
                _ => {}
            }
        }
        if !abandoned.is_empty() {
            self = self.cancel_ids(&abandoned, rng)?;
        }
        Arc::make_mut(&mut self.restarts).retain(|&(rs, _)| rs != slot);
        let sg = Arc::make_mut(&mut self.slot_gen);
        sg[slot] += 1;
        let gseq = sg[slot];
        let u_vec: Vec<f64> = unit.row(0).to_vec();
        let h_row = half_row(&u_vec);
        let n_obj = self.y_log.ncols().max(1);
        let init_sc = if self.best_val.is_finite() {
            vec![self.best_val; n_obj]
        } else {
            vec![0.0; n_obj]
        };
        let st = self.make_region(&h_row, slot, gseq, &init_sc, None)?;
        let s = self.search.take().expect("checked search");
        self.search = Some(s.restart(slot, st).map_err(ENNError::InvalidParameter)?);
        Ok((self, abandoned))
    }

    pub fn pending_ids(&self) -> Option<Vec<EvaluationId>> {
        if !self.identified {
            return None;
        }
        let mut out = Vec::new();
        for g in self.pending.iter() {
            g.pending_ids(&mut out);
        }
        if out.is_empty() { None } else { Some(out) }
    }

    pub fn all_pending(&self) -> Option<Vec<EvaluationId>> {
        let mut out = Vec::new();
        for g in self.pending.iter() {
            g.pending_ids(&mut out);
        }
        (!out.is_empty()).then_some(out)
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
        if x.nrows() == 0 {
            return Ok(Array2::zeros((0, m)));
        }
        let unit_q = self.domain.to_unit(x);
        let (k_fit, ep, al, ys) = self
            .search
            .as_ref()
            .and_then(RegionalSearch::fitted_enn)
            .unwrap_or((
                self.enn_res.ask.surrogate.neighbors.min(n).max(1),
                self.enn_res.ask.surrogate.epistemic_scale,
                self.enn_res.ask.surrogate.aleatoric_scale,
                self.enn_res.ask.surrogate.output_scale,
            ));
        let k_eff = k_fit.min(n).max(1);
        let train_x = self.x_log.to_array();
        let train_y = self.y_log.to_array();
        if let Ok(rt) = crate::apple_gpu::Runtime::shared()
            && let Ok(idx) = crate::knn::metal_index::MetalIndex::new(d, &train_x.view())
        {
            use metal::foreign_types::ForeignType;
            let q_flat: Vec<f32> = unit_q.iter().map(|&v| v as f32).collect();
            let q_buf = rt.buffer_with(&q_flat);
            let y_flat: Vec<f32> = train_y.iter().map(|&v| v as f32).collect();
            let v_flat: Option<Vec<f32>> = (self.config.noise_aware && self.yvar_log.nrows() == n)
                .then(|| {
                    self.yvar_log
                        .to_array()
                        .iter()
                        .map(|&v| v.max(0.0) as f32)
                        .collect()
                });
            let scales = vec![ys.max(1.0e-6); m];
            // SAFETY: `q_buf` is a live `MTLBuffer` on `self.runtime.device`
            // holding `unit_q.nrows() * d` contiguous `f32` coordinates.
            let post_res = unsafe {
                idx.device_posterior(
                    q_buf.as_ptr() as u64,
                    0,
                    unit_q.nrows(),
                    &y_flat,
                    v_flat.as_deref(),
                    &scales,
                    k_eff,
                    0,
                    ep,
                    al,
                    self.config.noise_aware,
                )
            };
            drop(q_buf);
            if let Ok(post) = post_res {
                let count = unit_q.nrows() * m;
                // SAFETY: `device_posterior` waits until the Metal command
                // buffer completes and populates `count` `f32` posterior means.
                let slice = unsafe {
                    std::slice::from_raw_parts(post.means.contents() as *const f32, count)
                };
                let vals: Vec<f64> = slice.iter().map(|&v| f64::from(v)).collect();
                if let Ok(arr) = Array2::from_shape_vec((unit_q.nrows(), m), vals) {
                    return Ok(arr);
                }
            }
        }
        let mut out = Array2::<f64>::zeros((unit_q.nrows(), m));
        for q in 0..unit_q.nrows() {
            let q_row = unit_q.row(q);
            let dists: Vec<f64> = (0..n)
                .map(|r| {
                    let t_row = self.x_log.row_slice(r);
                    q_row
                        .iter()
                        .zip(t_row)
                        .map(|(&a, &b)| (a - b) * (a - b))
                        .sum::<f64>()
                        / (d.max(1) as f64)
                })
                .collect();
            for col in 0..m {
                let outcomes: Vec<f32> = (0..n).map(|r| train_y[[r, col]] as f32).collect();
                let vars: Vec<f32> = (0..n)
                    .map(|r| {
                        if self.config.noise_aware && self.yvar_log.nrows() == n {
                            self.yvar_log.row_slice(r).get(col).copied().unwrap_or(0.0) as f32
                        } else {
                            0.0
                        }
                    })
                    .collect();
                let pred = (crate::semantics::neighbor::query::Query {
                    distances: &dists,
                    outcomes: &outcomes,
                    variances: &vars,
                    draws: None,
                    neighbors: k_eff,
                    epistemic: ep,
                    aleatoric: al,
                    scale: ys.max(1.0e-6),
                    beta: 1.0,
                    excluded: None,
                    incumbent: self.best_idx.unwrap_or(0).min(n - 1),
                    adaptive: self.enn_res.adapt_neighbors,
                })
                .predict()
                .map_err(|e| ENNError::InvalidParameter(e.to_string()))?;
                out[[q, col]] = f64::from(pred.mean);
            }
        }
        Ok(out)
    }

    #[inline]
    pub fn has_surrogate(&self) -> bool {
        self.has_surr
    }

    #[inline]
    pub fn ingestion_failure(&self) -> Option<&str> {
        None
    }

    pub fn init_progress(&self) -> Option<(usize, usize)> {
        let done = self.x_log.nrows();
        if self.init_only || (self.search.is_none() && done < self.warmup_goal) {
            Some((done, self.warmup_goal))
        } else {
            None
        }
    }

    #[inline]
    pub fn telemetry(&self) -> &Telemetry {
        &self.telemetry
    }

    #[inline]
    pub fn obs_count(&self) -> usize {
        self.x_log.nrows()
    }

    #[inline]
    pub fn num_regions(&self) -> usize {
        self.reg_spec.count.max(1)
    }

    pub fn tr_length(&self) -> f64 {
        match &self.search {
            Some(s) => s.tr_length(),
            None => self.tr_cfg.length_init,
        }
    }

    pub fn x_obs(&self) -> Option<Array2<f64>> {
        if self.x_log.is_empty() {
            None
        } else {
            let unit = self.x_log.to_array();
            Some(self.domain.from_unit(&unit.view()))
        }
    }

    pub fn y_obs(&self) -> Option<Array2<f64>> {
        if self.y_log.is_empty() {
            None
        } else {
            Some(self.y_log.to_array())
        }
    }

    pub fn unit_incumbent(&self) -> Option<Array1<f64>> {
        self.best_idx.map(|idx| self.x_log.row_vec(idx))
    }

    pub fn incumbent_x(&self) -> Option<Array1<f64>> {
        let unit = self.unit_incumbent()?;
        Some(self.domain.natural_point(&unit))
    }

    pub fn best_y(&self) -> Option<f64> {
        self.best_idx.map(|_| self.best_val)
    }

    pub fn lengthscales(&self) -> Option<Array1<f64>> {
        None
    }

    #[inline]
    pub fn bounds(&self) -> &Array2<f64> {
        self.domain.bounds()
    }

    #[inline]
    pub fn num_dim(&self) -> usize {
        self.domain.num_dim()
    }

    pub fn fork(&self) -> Result<Self, ENNError> {
        Ok(self.clone())
    }

    pub fn shares_storage(&self, other: &Self) -> bool {
        let logs_eq =
            self.x_log.shares_storage(&other.x_log) && self.y_log.shares_storage(&other.y_log);
        let metal_eq = match (&self.search, &other.search) {
            (Some(a), Some(b)) => a.shares_buffers(b),
            (None, None) => Arc::ptr_eq(&self.warm_buf, &other.warm_buf),
            _ => false,
        };
        logs_eq && metal_eq
    }

    pub fn digest(&self) -> [u8; 32] {
        let mut buf = Vec::with_capacity(512);
        buf.extend_from_slice(b"MetalEngine");
        for &v in self.domain.bounds().iter() {
            let bits = if v == 0.0 { 0u64 } else { v.to_bits() };
            buf.extend_from_slice(&bits.to_le_bytes());
        }
        buf.extend_from_slice(&self.seed.to_le_bytes());
        buf.extend_from_slice(&self.step.to_le_bytes());
        buf.extend_from_slice(&self.next_id.to_le_bytes());
        buf.extend_from_slice(&(self.x_log.nrows() as u64).to_le_bytes());
        for (_, chunk) in self.x_log.iter_chunks() {
            for &v in chunk.iter() {
                let bits = if v == 0.0 { 0u64 } else { v.to_bits() };
                buf.extend_from_slice(&bits.to_le_bytes());
            }
        }
        buf.extend_from_slice(&(self.y_log.nrows() as u64).to_le_bytes());
        buf.extend_from_slice(&(self.y_log.ncols() as u64).to_le_bytes());
        for (_, chunk) in self.y_log.iter_chunks() {
            for &v in chunk.iter() {
                let bits = if v == 0.0 { 0u64 } else { v.to_bits() };
                buf.extend_from_slice(&bits.to_le_bytes());
            }
        }
        match &self.search {
            Some(s) => {
                buf.push(1);
                s.hash_state(&mut buf);
            }
            None => buf.push(0),
        }
        let mut p_ids = Vec::new();
        for g in self.pending.iter() {
            g.pending_ids(&mut p_ids);
        }
        buf.extend_from_slice(&(p_ids.len() as u64).to_le_bytes());
        for id in p_ids {
            buf.extend_from_slice(&id.0.to_le_bytes());
        }
        crate::artifact::blake3::hash(&buf)
    }

    pub fn freeze_full(&self, host: Option<&HostCursor>) -> Result<Vec<u8>, ENNError> {
        if !self.pending.is_empty() {
            return Err(ENNError::InvalidParameter(
                "cannot freeze MetalEngine while evaluations are pending; complete or cancel first"
                    .into(),
            ));
        }
        let mut buf = Vec::with_capacity(256);
        write_header(&mut buf);
        write_byte(&mut buf, METAL_TAG);
        write_word(&mut buf, self.seed);
        write_word(&mut buf, self.step);
        write_word(&mut buf, self.next_id);
        let x_arr = self.x_log.to_array();
        let y_arr = self.y_log.to_array();
        let yv_arr = self.yvar_log.to_array();
        write_mat(&mut buf, &x_arr);
        write_mat(&mut buf, &y_arr);
        write_mat(&mut buf, &yv_arr);
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
        if r.read_byte()? != METAL_TAG {
            return Err(bad_state(
                "frozen state backend tag mismatch (expected Metal)",
            ));
        }
        let seed = r.read_word()?;
        let step = r.read_word()?;
        let next_id = r.read_word()?;
        let x_arr = r.read_mat(MAX_OBS, MAX_DIM, true)?;
        let y_arr = r.read_mat(MAX_OBS, MAX_METRICS, true)?;
        let yv_arr = r.read_mat(MAX_OBS, MAX_METRICS, true)?;
        if x_arr.nrows() > 0 && x_arr.ncols() != self.domain.num_dim() {
            return Err(bad_state("frozen MetalEngine dimension mismatch"));
        }
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

        let mut fresh = Self {
            domain: Arc::clone(&self.domain),
            config: Arc::clone(&self.config),
            reg_spec: Arc::clone(&self.reg_spec),
            enn_res: self.enn_res,
            tr_cfg: self.tr_cfg,
            kernel: self.kernel,
            single_state: self.single_state,
            explicit_dist: self.explicit_dist,
            has_surr: self.has_surr,
            init_only: self.init_only,
            identified: self.identified,
            init_kind: self.init_kind,
            warmup_goal: self.warmup_goal,
            base_cands: self.base_cands,
            default_arms: self.default_arms,
            seed,
            step: 0,
            next_id: 0,
            policy: self.policy.clone(),
            search: None,
            warm_buf: Arc::new(Vec::new()),
            restarts: Arc::new(Vec::new()),
            slot_gen: Arc::new(vec![0; self.reg_spec.count.max(1)]),
            pending: Arc::new(Vec::new()),
            x_log: ImLog::with_cols(self.domain.num_dim()),
            y_log: ImLog::empty(),
            yvar_log: ImLog::empty(),
            best_idx: None,
            best_val: f64::NEG_INFINITY,
            best_row: Arc::clone(&self.best_row),
            telemetry: Telemetry::default(),
        };
        if x_arr.nrows() > 0 {
            let mut rng = crate::hash::prng(seed);
            let yv_opt = (yv_arr.dim() == y_arr.dim()).then_some(yv_arr.view());
            fresh = fresh.tell(&x_arr.view(), &y_arr.view(), yv_opt.as_ref(), &mut rng)?;
        }
        fresh.step = step;
        fresh.next_id = next_id;
        Ok((fresh, host))
    }
}
