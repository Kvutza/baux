//! Defunctionalized `#![no_std]` round execution plan (`RoundPlan`) and GPU ABI
//! packing for arbitrary-`num_candidates` TuRBO-ENN, TuRBO-Zero, MORBO, and
//! portfolio rounds.

use baux::semantics::abi::{
    ACQ_RANDOM, DEFAULT_BETA, MODE_GAUSSIAN, MODE_LHD, MODE_MASKED, MODE_RAASP, MODE_RADEMACHER,
    MODE_SOBOL, MODE_UNIFORM, NO_FORCED, PoolDispatch, SelectHeader,
};
use baux::semantics::key::Key;
use baux::semantics::neighbor::{MetricKind, UncertaintyLaw};
use baux::semantics::optimizer::surrogate::AcquisitionRule;

use super::controller::{NoTrustState, TurboState};

/// Default TuRBO candidate pool size `clamp(100 * num_dim, 10, 5000)` with no
/// fixed 4-candidate ceiling.
#[inline]
pub const fn default_pool(num_dim: usize) -> usize {
    let scaled = 100usize.saturating_mul(num_dim);
    if scaled < 10 {
        10
    } else if scaled > 5000 {
        5000
    } else {
        scaled
    }
}

/// Stateless warmup design family.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum WarmupKind {
    #[default]
    Lhd,
    Uniform,
    Sobol {
        base_idx: u32,
    },
}

/// Stateless candidate proposal family evaluated per `(cand, dim)` via
/// `DelayedPool`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum PoolSampler {
    Raasp { prob: f64 },
    Sobol { base_idx: u32 },
    SobolMasked { base_idx: u32, prob: f64 },
    Uniform,
    Gaussian { sigma: f64 },
    Rademacher { step_size: f64 },
}

/// Defunctionalized surrogate mode (`NoSurrogate` vs. `Enn` posterior).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum SurrogateMode {
    NoSurrogate,
    Enn {
        neighbors: usize,
        epistemic: f32,
        aleatoric: f32,
        scale: f32,
        metric: MetricKind,
        law: UncertaintyLaw,
        noise_aware: bool,
    },
}

/// Lifecycle phase of a proposal round (`Warmup` vs. `Active`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum LifecyclePhase {
    Warmup {
        kind: WarmupKind,
    },
    Active {
        sampler: PoolSampler,
        surrogate: SurrogateMode,
    },
}

/// Defunctionalized trust-region geometry mode for the round.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum RegionPlan {
    Global {
        state: NoTrustState,
    },
    Single {
        state: TurboState,
    },
    Morbo {
        state: TurboState,
        num_metrics: usize,
    },
    Portfolio {
        num_regions: usize,
        num_active: usize,
    },
}

/// Pure `#![no_std]` defunctionalized `Copy` round specification shared across
/// CPU, Metal, and `cuda-oxide` backends.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RoundPlan {
    pub key: Key,
    pub step: u32,
    pub num_dim: usize,
    pub num_candidates: usize,
    pub num_arms: usize,
    pub phase: LifecyclePhase,
    pub acquisition: AcquisitionRule,
    pub region: RegionPlan,
    pub forced_slot: Option<u32>,
}

impl RoundPlan {
    /// Standard TuRBO RAASP coordinate perturbation rate `min(1, 20 / D)`.
    #[inline]
    pub fn raasp_prob(num_dim: usize) -> f64 {
        (20.0 / (num_dim.max(1) as f64)).clamp(0.0, 1.0)
    }

    /// Construct an active single-region TuRBO + ENN round plan.
    pub fn turbo_enn(
        key: Key,
        step: u32,
        num_dim: usize,
        num_candidates: usize,
        num_arms: usize,
        state: TurboState,
        surrogate: SurrogateMode,
        acquisition: AcquisitionRule,
    ) -> Self {
        Self {
            key,
            step,
            num_dim,
            num_candidates: num_candidates.max(1),
            num_arms: num_arms.max(1),
            phase: LifecyclePhase::Active {
                sampler: PoolSampler::Raasp {
                    prob: Self::raasp_prob(num_dim),
                },
                surrogate,
            },
            acquisition,
            region: RegionPlan::Single { state },
            forced_slot: None,
        }
    }

    /// Construct an active `NoSurrogate` (`TuRBO-Zero`) round plan.
    pub fn turbo_zero(
        key: Key,
        step: u32,
        num_dim: usize,
        num_candidates: usize,
        num_arms: usize,
        state: TurboState,
    ) -> Self {
        Self {
            key,
            step,
            num_dim,
            num_candidates: num_candidates.max(1),
            num_arms: num_arms.max(1),
            phase: LifecyclePhase::Active {
                sampler: PoolSampler::Raasp {
                    prob: Self::raasp_prob(num_dim),
                },
                surrogate: SurrogateMode::NoSurrogate,
            },
            acquisition: AcquisitionRule::Random { seed: key.seed() },
            region: RegionPlan::Single { state },
            forced_slot: None,
        }
    }

    /// Construct a `NoTrustRegion` (`[0, 1]^D`) + ENN round plan.
    pub fn global_enn(
        key: Key,
        step: u32,
        num_dim: usize,
        num_candidates: usize,
        num_arms: usize,
        state: NoTrustState,
        surrogate: SurrogateMode,
        acquisition: AcquisitionRule,
    ) -> Self {
        Self {
            key,
            step,
            num_dim,
            num_candidates: num_candidates.max(1),
            num_arms: num_arms.max(1),
            phase: LifecyclePhase::Active {
                sampler: PoolSampler::Sobol { base_idx: step },
                surrogate,
            },
            acquisition,
            region: RegionPlan::Global { state },
            forced_slot: None,
        }
    }

    /// Construct a stateless warmup design round plan.
    pub const fn warmup_round(
        key: Key,
        step: u32,
        num_dim: usize,
        num_candidates: usize,
        num_arms: usize,
        kind: WarmupKind,
    ) -> Self {
        Self {
            key,
            step,
            num_dim,
            num_candidates: if num_candidates == 0 {
                1
            } else {
                num_candidates
            },
            num_arms: if num_arms == 0 { 1 } else { num_arms },
            phase: LifecyclePhase::Warmup { kind },
            acquisition: AcquisitionRule::Random { seed: key.seed() },
            region: RegionPlan::Global {
                state: NoTrustState::new(),
            },
            forced_slot: None,
        }
    }

    #[inline]
    pub const fn with_forced(mut self, slot: Option<u32>) -> Self {
        self.forced_slot = slot;
        self
    }

    /// Shared GPU ABI proposal-mode discriminant (`MODE_*`).
    #[inline]
    pub const fn abi_mode(&self) -> u32 {
        match self.phase {
            LifecyclePhase::Warmup { kind } => match kind {
                WarmupKind::Lhd => MODE_LHD,
                WarmupKind::Uniform => MODE_UNIFORM,
                WarmupKind::Sobol { .. } => MODE_SOBOL,
            },
            LifecyclePhase::Active { sampler, .. } => match sampler {
                PoolSampler::Raasp { .. } => MODE_RAASP,
                PoolSampler::Sobol { .. } => MODE_SOBOL,
                PoolSampler::SobolMasked { .. } => MODE_MASKED,
                PoolSampler::Uniform => MODE_UNIFORM,
                PoolSampler::Gaussian { .. } => MODE_GAUSSIAN,
                PoolSampler::Rademacher { .. } => MODE_RADEMACHER,
            },
        }
    }

    /// Shared GPU ABI acquisition discriminant (`ACQ_*`).
    #[inline]
    pub const fn abi_acq(&self) -> u32 {
        match self.phase {
            LifecyclePhase::Warmup { .. }
            | LifecyclePhase::Active {
                surrogate: SurrogateMode::NoSurrogate,
                ..
            } => ACQ_RANDOM,
            LifecyclePhase::Active {
                surrogate: SurrogateMode::Enn { .. },
                ..
            } => self.acquisition.code(),
        }
    }

    #[inline]
    pub const fn forced_code(&self) -> u32 {
        match self.forced_slot {
            Some(slot) => slot,
            None => NO_FORCED,
        }
    }

    /// Pack a `#[repr(C)]` [`PoolDispatch`] descriptor for tiled GPU candidate
    /// evaluation (`DelayedPool<f32>`).
    pub fn pool_dispatch(&self, tiles: u32, history: u32, base_slot: u32) -> PoolDispatch {
        let (base_idx, prob, radius, scramble, init) = match self.phase {
            LifecyclePhase::Warmup { kind } => match kind {
                WarmupKind::Sobol { base_idx } => (base_idx, 1.0_f32, 1.0_f32, 1_u32, 1_u32),
                WarmupKind::Lhd | WarmupKind::Uniform => (0, 1.0_f32, 1.0_f32, 0_u32, 1_u32),
            },
            LifecyclePhase::Active { sampler, .. } => match sampler {
                PoolSampler::Raasp { prob } => (0, prob as f32, self.region_length() as f32, 0, 0),
                PoolSampler::Sobol { base_idx } => {
                    (base_idx, 1.0_f32, self.region_length() as f32, 1, 0)
                }
                PoolSampler::SobolMasked { base_idx, prob } => {
                    (base_idx, prob as f32, self.region_length() as f32, 1, 0)
                }
                PoolSampler::Uniform => (0, 1.0_f32, self.region_length() as f32, 0, 0),
                PoolSampler::Gaussian { sigma } => (0, 1.0_f32, sigma as f32, 0, 0),
                PoolSampler::Rademacher { step_size } => (0, 1.0_f32, step_size as f32, 0, 0),
            },
        };
        let pool_key = self.key.fold(0x706f_6f6c);
        let num_pert = ((prob as f64) * (self.num_dim as f64)).ceil().max(1.0) as u32;
        PoolDispatch {
            key_hi: pool_key.0,
            key_lo: pool_key.1,
            root_seed: self.key.seed(),
            stream_seed: pool_key.seed(),
            radius,
            prob_pert: prob,
            num_candidates: self.num_candidates as u32,
            num_arms: self.num_arms as u32,
            num_dim: self.num_dim as u32,
            num_pert: num_pert.min(self.num_dim.max(1) as u32),
            base_idx,
            tiles: tiles.max(1),
            history,
            mode: self.abi_mode(),
            base_slot,
            blocks: 0,
            initialize: init,
            scramble,
        }
    }

    /// Pack a `#[repr(C)]` [`SelectHeader`] descriptor for tiled arbitrary-`K`
    /// GPU surrogate scoring and selection.
    pub fn select_header(
        &self,
        num_obs: u32,
        incumbent_slot: u32,
        tiles: u32,
        has_ybounds: bool,
    ) -> SelectHeader {
        let (acq_seed, beta) = match self.acquisition {
            AcquisitionRule::Ucb { beta } => (self.key.fold(0x6163_7130).seed(), beta),
            AcquisitionRule::Thompson { seed }
            | AcquisitionRule::Pareto { seed }
            | AcquisitionRule::Random { seed } => (seed, DEFAULT_BETA),
        };
        let (neighbors, metric_code, law_code, has_yvar, epi, ale, scale) = match self.phase {
            LifecyclePhase::Active {
                surrogate:
                    SurrogateMode::Enn {
                        neighbors,
                        epistemic,
                        aleatoric,
                        scale,
                        metric,
                        law,
                        noise_aware,
                    },
                ..
            } => (
                neighbors as u32,
                metric.code(),
                law.code(),
                u32::from(noise_aware),
                epistemic,
                aleatoric,
                scale,
            ),
            LifecyclePhase::Warmup { .. }
            | LifecyclePhase::Active {
                surrogate: SurrogateMode::NoSurrogate,
                ..
            } => (
                0,
                MetricKind::Euclidean.code(),
                UncertaintyLaw::InversePrecision.code(),
                0,
                0.0,
                0.0,
                1.0,
            ),
        };
        let num_metrics = match self.region {
            RegionPlan::Morbo { num_metrics, .. } => num_metrics.max(1) as u32,
            _ => 1,
        };
        SelectHeader {
            root_seed: self.key.seed(),
            acq_seed,
            epistemic_scale: epi,
            aleatoric_scale: ale,
            y_scale: scale,
            beta,
            radius: self.region_length() as f32,
            axis_base: 1.0,
            axis_weight: 1.0,
            morbo_alpha: 0.05,
            num_candidates: self.num_candidates as u32,
            num_arms: self.num_arms as u32,
            neighbors,
            history: num_obs,
            acquisition: self.abi_acq(),
            tiles: tiles.max(1),
            mode: self.abi_mode(),
            resident_history: num_obs,
            implicit_history: 0,
            forced_candidate: self.forced_code(),
            distance_metric: metric_code,
            scale_neighbors: neighbors.max(1),
            incumbent_index: incumbent_slot,
            candidate_floor: 0,
            adaptive_k: 0,
            uncertainty_law: law_code,
            axis_enabled: 0,
            num_metrics,
            has_yvar,
            has_ybounds: u32::from(has_ybounds),
        }
    }

    #[inline]
    pub const fn region_length(&self) -> f64 {
        match self.region {
            RegionPlan::Global { .. } | RegionPlan::Portfolio { .. } => 1.0,
            RegionPlan::Single { state } | RegionPlan::Morbo { state, .. } => state.length,
        }
    }
}
