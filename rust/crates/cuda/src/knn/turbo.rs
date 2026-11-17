//! Resident CUDA TuRBO-ENN, TuRBO-Zero, MORBO (`OnRestart` + `OnPropose`), and
//! multi-trust-region (`RegionalCuda`) drivers backed by `baux::semantics`.

use baux::semantics::key::Key;
use baux::semantics::morbo::{
    RescalarizeMode, metric_ranges, nds_ranks, sample_simplex, scalarize_slice,
};
use baux::semantics::neighbor::{
    DistanceMetric, MetricKind, UncertaintyLaw, ard_sides, region_bounds, simd_l2,
};
use baux::semantics::optimizer::portfolio::{
    Allocation, RegionController, RegionTicket, RegionalLedger, RestartCenterPolicy,
    RestartCenterSelection, RestartMemoryPolicy, SharingPolicy, joint_select, maximin_idx,
    retain_mask,
};
use baux::semantics::optimizer::surrogate::{AcquisitionRule, BetterScore, loo_slice};
use baux::semantics::optimizer::trust_region::{
    LifecyclePhase, NoTrustState, PoolSampler, RegionPlan, RoundPlan, SurrogateMode, TurboState,
    WarmupKind, default_pool,
};
use baux::semantics::pool::eval_uniform;
use baux::semantics::ybounds::{is_identity, warp_checked, warp_var};

use super::pool::{
    FencedProposal, MorboSpec, PoolModifiers, ProposalHost, eff_weights, local_radii,
    simulate_round, snap_rows,
};
use super::{CudaIndex, CudaKnnMode, CudaScratch};
use crate::capture::CudaResult;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum TurboMode {
    #[default]
    TurboEnn,
    TurboZero,
    LhdOnly,
    SingleBo,
}

#[derive(Clone, Debug)]
pub struct TurboSpec {
    pub seed: u64,
    pub num_dim: usize,
    pub num_candidates: usize,
    pub num_arms: usize,
    pub warmup_obs: usize,
    pub warmup_kind: WarmupKind,
    pub sampler: PoolSampler,
    pub mode: TurboMode,
    pub neighbors: usize,
    pub epistemic: f32,
    pub aleatoric: f32,
    pub y_scale: f32,
    pub metric: DistanceMetric,
    pub law: UncertaintyLaw,
    pub noise_aware: bool,
    pub acquisition: AcquisitionRule,
    pub initial_len: f64,
    pub min_len: f64,
    pub max_len: f64,
    pub failure_tol: Option<usize>,
    pub mods: PoolModifiers,
}

impl TurboSpec {
    pub fn for_dims(seed: u64, num_dim: usize) -> Self {
        let dims = num_dim.max(1);
        Self {
            seed,
            num_dim: dims,
            num_candidates: default_pool(dims),
            num_arms: 1,
            warmup_obs: (2 * dims).clamp(4, 32),
            warmup_kind: WarmupKind::Lhd,
            sampler: PoolSampler::Raasp {
                prob: RoundPlan::raasp_prob(dims),
            },
            mode: TurboMode::TurboEnn,
            neighbors: 16,
            epistemic: 0.05,
            aleatoric: 0.05,
            y_scale: 1.0,
            metric: DistanceMetric::Euclidean,
            law: UncertaintyLaw::InversePrecision,
            noise_aware: false,
            acquisition: AcquisitionRule::default(),
            initial_len: 0.8,
            min_len: 0.5_f64.powi(7),
            max_len: 1.6,
            failure_tol: None,
            mods: PoolModifiers::default(),
        }
    }

    pub fn surrogate_mode(&self) -> SurrogateMode {
        match self.mode {
            TurboMode::TurboZero | TurboMode::LhdOnly => SurrogateMode::NoSurrogate,
            TurboMode::SingleBo => SurrogateMode::Enn {
                neighbors: 1,
                epistemic: self.epistemic,
                aleatoric: self.aleatoric,
                scale: self.y_scale,
                metric: MetricKind::from_metric(self.metric),
                law: self.law,
                noise_aware: self.noise_aware,
            },
            TurboMode::TurboEnn => SurrogateMode::Enn {
                neighbors: self.neighbors.max(1),
                epistemic: self.epistemic,
                aleatoric: self.aleatoric,
                scale: self.y_scale,
                metric: MetricKind::from_metric(self.metric),
                law: self.law,
                noise_aware: self.noise_aware,
            },
        }
    }

    pub fn scale_k(&self) -> usize {
        match self.metric {
            DistanceMetric::Euclidean => self.neighbors.max(1),
            DistanceMetric::Conformal { local_k }
            | DistanceMetric::Hyperbolic { local_k }
            | DistanceMetric::Wasserstein { local_k } => local_k.max(1),
        }
    }
}

fn step_warmup(kind: WarmupKind, step: u32, cands: usize) -> WarmupKind {
    match kind {
        WarmupKind::Sobol { base_idx } => WarmupKind::Sobol {
            base_idx: base_idx.wrapping_add(step.wrapping_mul(cands as u32)),
        },
        other => other,
    }
}

fn step_sampler(sampler: PoolSampler, step: u32, cands: usize) -> PoolSampler {
    let shift = step.wrapping_mul(cands as u32);
    match sampler {
        PoolSampler::Sobol { base_idx } => PoolSampler::Sobol {
            base_idx: base_idx.wrapping_add(shift),
        },
        PoolSampler::SobolMasked { base_idx, prob } => PoolSampler::SobolMasked {
            base_idx: base_idx.wrapping_add(shift),
            prob,
        },
        other => other,
    }
}

fn warp_outcome(raw_y: f32, raw_v: Option<f32>, yb: Option<&[[f32; 2]]>) -> CudaResult<(f32, f32)> {
    let Some(bounds) = yb.filter(|b| !b.is_empty() && !is_identity(b)) else {
        return Ok((raw_y, raw_v.unwrap_or(0.0).max(0.0)));
    };
    let [lo, hi] = bounds[0];
    let z = warp_checked(raw_y, lo, hi)
        .ok_or_else(|| format!("outcome {raw_y} lies outside open y_bounds ({lo}, {hi})"))?;
    let zv = match raw_v {
        Some(v) => warp_var(raw_y, v.max(0.0), lo, hi)
            .ok_or_else(|| "non-finite warped observation variance".to_string())?,
        None => 0.0,
    };
    Ok((z, zv))
}

pub struct TurboCuda {
    pub spec: TurboSpec,
    state: TurboState,
    step: u32,
    rows: Vec<f32>,
    raw_y: Vec<f32>,
    latent_y: Vec<f32>,
    latent_v: Vec<f32>,
    has_var: bool,
    inc_idx: usize,
    gpu: Option<(CudaIndex, CudaScratch)>,
}

impl TurboCuda {
    pub fn new(spec: TurboSpec) -> CudaResult<Self> {
        if spec.num_dim == 0 || spec.num_candidates == 0 || spec.num_arms == 0 {
            return Err("TurboCuda dimensions, candidates, and arms must be non-zero".to_string());
        }
        if spec.num_arms > spec.num_candidates {
            return Err("TurboCuda num_arms cannot exceed num_candidates".to_string());
        }
        let state = TurboState::for_dims(
            spec.num_dim,
            spec.num_arms,
            spec.initial_len,
            spec.min_len,
            spec.max_len,
            spec.failure_tol,
        );
        Ok(Self {
            spec,
            state,
            step: 0,
            rows: Vec::new(),
            raw_y: Vec::new(),
            latent_y: Vec::new(),
            latent_v: Vec::new(),
            has_var: false,
            inc_idx: 0,
            gpu: None,
        })
    }

    pub fn with_gpu(spec: TurboSpec, mode: CudaKnnMode) -> CudaResult<Self> {
        let mut opt = Self::new(spec)?;
        let idx = CudaIndex::with_mode(opt.spec.num_dim, &[], mode)?;
        let scratch = idx.new_scratch()?;
        opt.gpu = Some((idx, scratch));
        Ok(opt)
    }

    pub fn len(&self) -> usize {
        self.raw_y.len()
    }

    pub fn is_empty(&self) -> bool {
        self.raw_y.is_empty()
    }

    pub fn step(&self) -> u32 {
        self.step
    }

    pub fn state(&self) -> TurboState {
        self.state
    }

    pub fn incumbent(&self) -> Option<(usize, &[f32], f32)> {
        if self.raw_y.is_empty() {
            None
        } else {
            let d = self.spec.num_dim;
            Some((
                self.inc_idx,
                &self.rows[self.inc_idx * d..(self.inc_idx + 1) * d],
                self.raw_y[self.inc_idx],
            ))
        }
    }

    pub fn plan(&self) -> RoundPlan {
        let key = Key::new(self.spec.seed).fold(u64::from(self.step));
        let acq = match self.spec.acquisition {
            AcquisitionRule::Ucb { beta } => AcquisitionRule::Ucb { beta },
            AcquisitionRule::Thompson { .. } => AcquisitionRule::Thompson {
                seed: key.fold(0x7468_6f6d).seed(),
            },
            AcquisitionRule::Pareto { .. } => AcquisitionRule::Pareto {
                seed: key.fold(0x7061_7265).seed(),
            },
            AcquisitionRule::Random { .. } => AcquisitionRule::Random {
                seed: key.fold(0x7261_6e64).seed(),
            },
        };
        if self.len() < self.spec.warmup_obs || self.spec.mode == TurboMode::LhdOnly {
            let kind = if self.spec.mode == TurboMode::LhdOnly {
                WarmupKind::Lhd
            } else {
                step_warmup(self.spec.warmup_kind, self.step, self.spec.num_candidates)
            };
            return RoundPlan::warmup_round(
                key,
                self.step,
                self.spec.num_dim,
                self.spec.num_candidates,
                self.spec.num_arms,
                kind,
            );
        }
        let sampler = step_sampler(self.spec.sampler, self.step, self.spec.num_candidates);
        RoundPlan {
            key,
            step: self.step,
            num_dim: self.spec.num_dim,
            num_candidates: self.spec.num_candidates,
            num_arms: self.spec.num_arms,
            phase: LifecyclePhase::Active {
                sampler,
                surrogate: self.spec.surrogate_mode(),
            },
            acquisition: acq,
            region: if self.spec.mode == TurboMode::LhdOnly {
                RegionPlan::Global {
                    state: NoTrustState::new(),
                }
            } else {
                RegionPlan::Single { state: self.state }
            },
            forced_slot: None,
        }
    }

    fn round_mods(&self) -> PoolModifiers {
        let mut m = self.spec.mods.clone();
        m.scale_k = self.spec.scale_k();
        if !self.raw_y.is_empty() {
            m.inc_mean = self.raw_y[self.inc_idx];
        }
        if MetricKind::from_metric(self.spec.metric) != MetricKind::Euclidean
            && m.local_scales.is_none()
            && !self.raw_y.is_empty()
        {
            let w = eff_weights(self.spec.num_dim, Some(&m));
            m.local_scales = Some(local_radii(
                &self.rows,
                self.len(),
                self.spec.num_dim,
                &w,
                m.scale_k,
            ));
        }
        m
    }

    pub fn fit_loo(&mut self) {
        let n = self.len();
        if n < 4 {
            return;
        }
        let d = self.spec.num_dim;
        let w = eff_weights(d, Some(&self.spec.mods));
        let mut dist_mat = vec![0.0_f32; n * n];
        for i in 0..n {
            let ri = &self.rows[i * d..(i + 1) * d];
            for j in (i + 1)..n {
                let rj = &self.rows[j * d..(j + 1) * d];
                let d2 = baux::semantics::neighbor::ard_dist::<f32>(ri, rj, &w);
                dist_mat[i * n + j] = d2;
                dist_mat[j * n + i] = d2;
            }
        }
        let held: Vec<usize> = (0..n).collect();
        let k = self.spec.neighbors.min(n - 1).max(1);
        let mut best_ll = f32::NEG_INFINITY;
        let mut best_pair = (self.spec.epistemic, self.spec.aleatoric);
        for &epi in &[0.01_f32, 0.05, 0.2, 1.0] {
            for &ale in &[0.005_f32, 0.05, 0.2] {
                let ll = loo_slice::<f32>(
                    self.spec.metric,
                    self.spec.law,
                    &dist_mat,
                    &self.latent_y,
                    &self.latent_v,
                    &held,
                    k,
                    epi,
                    ale,
                    self.spec.y_scale,
                    self.spec.noise_aware && self.has_var,
                );
                if ll.is_finite() && ll > best_ll {
                    best_ll = ll;
                    best_pair = (epi, ale);
                }
            }
        }
        self.spec.epistemic = best_pair.0;
        self.spec.aleatoric = best_pair.1;
    }

    pub fn propose_fenced(&mut self) -> CudaResult<FencedProposal> {
        let plan = self.plan();
        let d = self.spec.num_dim;
        let center = if self.raw_y.is_empty() {
            vec![0.5_f32; d]
        } else {
            self.rows[self.inc_idx * d..(self.inc_idx + 1) * d].to_vec()
        };
        let mods = self.round_mods();
        let var_opt = (self.spec.noise_aware && self.has_var).then_some(self.latent_v.as_slice());
        let (idx, scratch) = self
            .gpu
            .as_mut()
            .ok_or_else(|| "TurboCuda::propose_fenced requires GPU index (with_gpu)".to_string())?;
        let out = idx.propose_fenced(
            scratch,
            &plan,
            &center,
            &self.latent_y,
            var_opt,
            1,
            self.inc_idx,
            Some(&mods),
        )?;
        self.step = self.step.wrapping_add(1);
        Ok(out)
    }

    pub fn ask(&mut self) -> CudaResult<ProposalHost> {
        let mods = self.round_mods();
        if self.gpu.is_some() {
            let fenced = self.propose_fenced()?;
            return fenced.materialize(Some(&mods));
        }
        let plan = self.plan();
        let d = self.spec.num_dim;
        let center = if self.raw_y.is_empty() {
            vec![0.5_f32; d]
        } else {
            self.rows[self.inc_idx * d..(self.inc_idx + 1) * d].to_vec()
        };
        let var_opt = (self.spec.noise_aware && self.has_var).then_some(self.latent_v.as_slice());
        let out = simulate_round(
            &self.rows,
            self.len(),
            &plan,
            &center,
            &self.latent_y,
            var_opt,
            1,
            self.inc_idx,
            Some(&mods),
        )?;
        self.step = self.step.wrapping_add(1);
        Ok(out)
    }

    pub fn tell(
        &mut self,
        new_rows: &[f32],
        outcomes: &[f32],
        variances: Option<&[f32]>,
    ) -> CudaResult<bool> {
        let d = self.spec.num_dim;
        if outcomes.is_empty() || new_rows.len() != outcomes.len() * d {
            return Err("TurboCuda::tell shape mismatch".to_string());
        }
        if let Some(v) = variances
            && v.len() != outcomes.len()
        {
            return Err("TurboCuda::tell variances length mismatch".to_string());
        }
        let prev_best = if self.raw_y.is_empty() {
            f32::NEG_INFINITY
        } else {
            self.raw_y[self.inc_idx]
        };
        let yb = self.spec.mods.ybounds.as_deref();
        let mut improved = false;
        for (i, &y) in outcomes.iter().enumerate() {
            let v_in = variances.map(|v| v[i]);
            let (zy, zv) = warp_outcome(y, v_in, yb)?;
            let idx = self.raw_y.len();
            self.rows.extend_from_slice(&new_rows[i * d..(i + 1) * d]);
            self.raw_y.push(y);
            self.latent_y.push(zy);
            self.latent_v.push(zv);
            if variances.is_some() {
                self.has_var = true;
            }
            if idx == 0 || BetterScore(y) > BetterScore(self.raw_y[self.inc_idx]) {
                self.inc_idx = idx;
            }
            if prev_best.is_finite() && y > prev_best {
                improved = true;
            }
        }
        if let Some((gpu_idx, mut scratch)) = self.gpu.take() {
            let next_idx = gpu_idx.add(&mut scratch, new_rows)?;
            self.gpu = Some((next_idx, scratch));
        }
        if self.len() > self.spec.warmup_obs
            && self.spec.mode != TurboMode::LhdOnly
            && prev_best.is_finite()
        {
            let (next_state, _, _) = self.state.next_step(improved);
            self.state = next_state;
        }
        Ok(improved)
    }
}

#[derive(Clone, Debug)]
pub struct MorboRegion {
    pub state: TurboState,
    pub weights: Vec<f32>,
    pub inc_idx: usize,
    pub best_val: f32,
    pub restarts: usize,
}

pub struct MorboCuda {
    pub spec: TurboSpec,
    pub num_metrics: usize,
    pub num_regions: usize,
    pub rescalarize: RescalarizeMode,
    pub alpha: f32,
    regions: Vec<MorboRegion>,
    step: u32,
    rows: Vec<f32>,
    raw_y: Vec<f32>,
    latent_y: Vec<f32>,
    latent_v: Vec<f32>,
    has_var: bool,
    gpu: Option<(CudaIndex, CudaScratch)>,
}

impl MorboCuda {
    pub fn new(
        spec: TurboSpec,
        num_metrics: usize,
        num_regions: usize,
        rescalarize: RescalarizeMode,
        alpha: f32,
    ) -> CudaResult<Self> {
        if spec.num_dim == 0 || num_metrics == 0 || num_regions == 0 {
            return Err("MorboCuda requires non-zero dims, metrics, and regions".to_string());
        }
        let base_state = TurboState::for_dims(
            spec.num_dim,
            spec.num_arms.max(1),
            spec.initial_len,
            spec.min_len,
            spec.max_len,
            spec.failure_tol,
        );
        let root = Key::new(spec.seed).fold(0x6d6f_7262);
        let mut regions = Vec::with_capacity(num_regions);
        for r in 0..num_regions {
            let mut w = vec![0.0_f32; num_metrics];
            sample_simplex::<f32>(root.fold(r as u64), &mut w);
            regions.push(MorboRegion {
                state: base_state,
                weights: w,
                inc_idx: 0,
                best_val: f32::NEG_INFINITY,
                restarts: 0,
            });
        }
        Ok(Self {
            spec,
            num_metrics,
            num_regions,
            rescalarize,
            alpha,
            regions,
            step: 0,
            rows: Vec::new(),
            raw_y: Vec::new(),
            latent_y: Vec::new(),
            latent_v: Vec::new(),
            has_var: false,
            gpu: None,
        })
    }

    pub fn with_gpu(
        spec: TurboSpec,
        num_metrics: usize,
        num_regions: usize,
        rescalarize: RescalarizeMode,
        alpha: f32,
        mode: CudaKnnMode,
    ) -> CudaResult<Self> {
        let mut m = Self::new(spec, num_metrics, num_regions, rescalarize, alpha)?;
        let idx = CudaIndex::with_mode(m.spec.num_dim, &[], mode)?;
        let scratch = idx.new_scratch()?;
        m.gpu = Some((idx, scratch));
        Ok(m)
    }

    pub fn len(&self) -> usize {
        if self.num_metrics == 0 {
            0
        } else {
            self.raw_y.len() / self.num_metrics
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn regions(&self) -> &[MorboRegion] {
        &self.regions
    }

    pub fn pareto_indices(&self) -> Vec<usize> {
        let n = self.len();
        if n == 0 {
            return Vec::new();
        }
        let mut ranks = vec![0_u32; n];
        nds_ranks::<f32>(&self.raw_y, n, self.num_metrics, &mut ranks);
        (0..n).filter(|&i| ranks[i] == 0).collect()
    }

    fn refresh_incumbent(&mut self, reg: usize) {
        let n = self.len();
        if n == 0 {
            return;
        }
        let m = self.num_metrics;
        let mut y_min = vec![0.0_f32; m];
        let mut y_max = vec![1.0_f32; m];
        metric_ranges::<f32>(&self.raw_y, n, m, &mut y_min, &mut y_max);
        let mut scal = vec![0.0_f32; n];
        scalarize_slice::<f32>(
            &self.raw_y,
            n,
            m,
            &y_min,
            &y_max,
            &self.regions[reg].weights,
            self.alpha,
            true,
            &mut scal,
        );
        let mut best_i = 0usize;
        for i in 1..n {
            if BetterScore(scal[i]) > BetterScore(scal[best_i]) {
                best_i = i;
            }
        }
        self.regions[reg].inc_idx = best_i;
        self.regions[reg].best_val = scal[best_i];
    }

    fn prepare_region(&mut self, reg: usize) -> CudaResult<(RoundPlan, Vec<f32>, PoolModifiers)> {
        if reg >= self.num_regions {
            return Err("MorboCuda region out of bounds".to_string());
        }
        let key = Key::new(self.spec.seed)
            .fold(u64::from(self.step))
            .fold(reg as u64);
        if self.rescalarize == RescalarizeMode::OnPropose {
            sample_simplex::<f32>(key.fold(0x7265_7363), &mut self.regions[reg].weights);
        }
        self.refresh_incumbent(reg);

        let n = self.len();
        let d = self.spec.num_dim;
        let m = self.num_metrics;
        let center = if n == 0 {
            vec![0.5_f32; d]
        } else {
            let inc = self.regions[reg].inc_idx;
            self.rows[inc * d..(inc + 1) * d].to_vec()
        };

        let mut y_min = vec![0.0_f32; m];
        let mut y_max = vec![1.0_f32; m];
        if n > 0 {
            metric_ranges::<f32>(&self.raw_y, n, m, &mut y_min, &mut y_max);
        }
        let mut mods = self.spec.mods.clone();
        mods.scale_k = self.spec.scale_k();
        mods.inc_mean = self.regions[reg].best_val;
        mods.morbo = Some(MorboSpec {
            mode: self.rescalarize,
            alpha: self.alpha,
            y_min,
            y_max,
            weights: self.regions[reg].weights.clone(),
        });
        if MetricKind::from_metric(self.spec.metric) != MetricKind::Euclidean
            && mods.local_scales.is_none()
            && n > 0
        {
            let w = eff_weights(d, Some(&mods));
            mods.local_scales = Some(local_radii(&self.rows, n, d, &w, mods.scale_k));
        }

        let acq = match self.spec.acquisition {
            AcquisitionRule::Ucb { beta } => AcquisitionRule::Ucb { beta },
            AcquisitionRule::Thompson { .. } => AcquisitionRule::Thompson {
                seed: key.fold(0x7468_6f6d).seed(),
            },
            AcquisitionRule::Pareto { .. } => AcquisitionRule::Pareto {
                seed: key.fold(0x7061_7265).seed(),
            },
            AcquisitionRule::Random { .. } => AcquisitionRule::Random {
                seed: key.fold(0x7261_6e64).seed(),
            },
        };
        let plan = if n < self.spec.warmup_obs {
            RoundPlan::warmup_round(
                key,
                self.step,
                d,
                self.spec.num_candidates,
                self.spec.num_arms,
                step_warmup(self.spec.warmup_kind, self.step, self.spec.num_candidates),
            )
        } else {
            RoundPlan {
                key,
                step: self.step,
                num_dim: d,
                num_candidates: self.spec.num_candidates,
                num_arms: self.spec.num_arms,
                phase: LifecyclePhase::Active {
                    sampler: step_sampler(self.spec.sampler, self.step, self.spec.num_candidates),
                    surrogate: self.spec.surrogate_mode(),
                },
                acquisition: acq,
                region: RegionPlan::Morbo {
                    state: self.regions[reg].state,
                    num_metrics: m,
                },
                forced_slot: None,
            }
        };
        Ok((plan, center, mods))
    }

    pub fn propose_fenced(&mut self, reg: usize) -> CudaResult<FencedProposal> {
        let (plan, center, mods) = self.prepare_region(reg)?;
        let inc = self.regions[reg].inc_idx;
        let var_opt = (self.spec.noise_aware && self.has_var).then_some(self.latent_v.as_slice());
        let m = self.num_metrics;
        let (idx, scratch) = self
            .gpu
            .as_mut()
            .ok_or_else(|| "MorboCuda::propose_fenced requires GPU index".to_string())?;
        let out = idx.propose_fenced(
            scratch,
            &plan,
            &center,
            &self.latent_y,
            var_opt,
            m,
            inc,
            Some(&mods),
        )?;
        self.step = self.step.wrapping_add(1);
        Ok(out)
    }

    pub fn ask_region(&mut self, reg: usize) -> CudaResult<ProposalHost> {
        let (plan, center, mods) = self.prepare_region(reg)?;
        if self.gpu.is_some() {
            let inc = self.regions[reg].inc_idx;
            let var_opt =
                (self.spec.noise_aware && self.has_var).then_some(self.latent_v.as_slice());
            let m = self.num_metrics;
            let (idx, scratch) = self.gpu.as_mut().expect("checked gpu");
            let fenced = idx.propose_fenced(
                scratch,
                &plan,
                &center,
                &self.latent_y,
                var_opt,
                m,
                inc,
                Some(&mods),
            )?;
            self.step = self.step.wrapping_add(1);
            return fenced.materialize(Some(&mods));
        }
        let var_opt = (self.spec.noise_aware && self.has_var).then_some(self.latent_v.as_slice());
        let out = simulate_round(
            &self.rows,
            self.len(),
            &plan,
            &center,
            &self.latent_y,
            var_opt,
            self.num_metrics,
            self.regions[reg].inc_idx,
            Some(&mods),
        )?;
        self.step = self.step.wrapping_add(1);
        Ok(out)
    }

    pub fn ask(&mut self) -> CudaResult<(usize, ProposalHost)> {
        let reg = (self.step as usize) % self.num_regions;
        let out = self.ask_region(reg)?;
        Ok((reg, out))
    }

    pub fn tell(
        &mut self,
        reg: usize,
        new_rows: &[f32],
        outcomes: &[f32],
        variances: Option<&[f32]>,
    ) -> CudaResult<bool> {
        let d = self.spec.num_dim;
        let m = self.num_metrics;
        if reg >= self.num_regions {
            return Err("MorboCuda::tell region out of bounds".to_string());
        }
        if outcomes.is_empty() || outcomes.len() % m != 0 {
            return Err("MorboCuda::tell outcomes must be a multiple of num_metrics".to_string());
        }
        let count = outcomes.len() / m;
        if new_rows.len() != count * d {
            return Err("MorboCuda::tell rows length mismatch".to_string());
        }
        if let Some(v) = variances
            && v.len() != outcomes.len()
        {
            return Err("MorboCuda::tell variances length mismatch".to_string());
        }

        let prev_n = self.len();
        let yb = self.spec.mods.ybounds.as_deref();
        for i in 0..count {
            self.rows.extend_from_slice(&new_rows[i * d..(i + 1) * d]);
            for col in 0..m {
                let raw = outcomes[i * m + col];
                let raw_v = variances.map(|v| v[i * m + col]);
                let col_yb = yb.and_then(|b| b.get(col..col + 1));
                let (zy, zv) = warp_outcome(raw, raw_v, col_yb)?;
                self.raw_y.push(raw);
                self.latent_y.push(zy);
                self.latent_v.push(zv);
            }
            if variances.is_some() {
                self.has_var = true;
            }
        }
        if let Some((gpu_idx, mut scratch)) = self.gpu.take() {
            let next_idx = gpu_idx.add(&mut scratch, new_rows)?;
            self.gpu = Some((next_idx, scratch));
        }

        let total_n = self.len();
        let mut y_min = vec![0.0_f32; m];
        let mut y_max = vec![1.0_f32; m];
        metric_ranges::<f32>(&self.raw_y, total_n, m, &mut y_min, &mut y_max);
        let mut scal = vec![0.0_f32; total_n];
        scalarize_slice::<f32>(
            &self.raw_y,
            total_n,
            m,
            &y_min,
            &y_max,
            &self.regions[reg].weights,
            self.alpha,
            true,
            &mut scal,
        );
        let old_best = (0..prev_n)
            .map(|i| scal[i])
            .fold(f32::NEG_INFINITY, f32::max);
        let new_best = (prev_n..total_n)
            .map(|i| scal[i])
            .fold(f32::NEG_INFINITY, f32::max);
        let improved = prev_n > 0 && new_best > old_best;

        if total_n > self.spec.warmup_obs && prev_n > 0 {
            let (next_state, _, restarted) = self.regions[reg].state.next_step(improved);
            self.regions[reg].state = next_state;
            if restarted {
                self.regions[reg].restarts += 1;
                if self.rescalarize == RescalarizeMode::OnRestart {
                    let r_key = Key::new(self.spec.seed)
                        .fold(0x7273_7472)
                        .fold(reg as u64)
                        .fold(self.regions[reg].restarts as u64);
                    sample_simplex::<f32>(r_key, &mut self.regions[reg].weights);
                }
            }
        }
        for r in 0..self.num_regions {
            self.refresh_incumbent(r);
        }
        Ok(improved)
    }
}

#[derive(Clone, Debug)]
pub struct RegionalSpec {
    pub base: TurboSpec,
    pub num_regions: usize,
    pub allocation: Allocation,
    pub sharing: SharingPolicy,
    pub memory: RestartMemoryPolicy,
    pub center_policy: RestartCenterPolicy,
    pub controller: RegionController,
}

impl RegionalSpec {
    pub fn for_dims(seed: u64, num_dim: usize, num_regions: usize) -> Self {
        Self {
            base: TurboSpec::for_dims(seed, num_dim),
            num_regions: num_regions.max(1),
            allocation: Allocation::IncumbentUcb1 { exploration: 1.0 },
            sharing: SharingPolicy::BoxContain,
            memory: RestartMemoryPolicy::Fresh,
            center_policy: RestartCenterPolicy {
                candidates: 16,
                selection: RestartCenterSelection::Maximin,
            },
            controller: RegionController::Turbo,
        }
    }
}

struct SlotMemory {
    rows: Vec<f32>,
    raw_y: Vec<f32>,
    latent_y: Vec<f32>,
    latent_v: Vec<f32>,
    center: Vec<f32>,
    state: TurboState,
    inc_idx: usize,
    gpu: Option<(CudaIndex, CudaScratch)>,
}

#[derive(Clone, Debug)]
pub struct RegionalBatch {
    pub slot: usize,
    pub ticket: RegionTicket,
    pub proposal: ProposalHost,
}

pub struct RegionalCuda {
    pub spec: RegionalSpec,
    ledger: RegionalLedger,
    slots: Vec<SlotMemory>,
    all_rows: Vec<f32>,
    all_y: Vec<f32>,
    all_v: Vec<f32>,
    has_var: bool,
    step: u32,
    gpu_mode: Option<CudaKnnMode>,
}

impl RegionalCuda {
    pub fn new(spec: RegionalSpec) -> CudaResult<Self> {
        if spec.base.num_dim == 0 || spec.num_regions == 0 {
            return Err("RegionalCuda requires non-zero dims and regions".to_string());
        }
        let init_best = vec![f64::NEG_INFINITY; spec.num_regions];
        let ledger = RegionalLedger::new(&init_best, spec.allocation).map_err(|e| e.to_string())?;
        let base_state = TurboState::for_dims(
            spec.base.num_dim,
            spec.base.num_arms.max(1),
            spec.base.initial_len,
            spec.base.min_len,
            spec.base.max_len,
            spec.base.failure_tol,
        );
        let root = Key::new(spec.base.seed).fold(0x7265_6769);
        let lo = vec![0.0_f32; spec.base.num_dim];
        let hi = vec![1.0_f32; spec.base.num_dim];
        let mut slots = Vec::with_capacity(spec.num_regions);
        for s in 0..spec.num_regions {
            let center: Vec<f32> = (0..spec.base.num_dim)
                .map(|j| eval_uniform::<f32>(root, &lo, &hi, s, j))
                .collect();
            slots.push(SlotMemory {
                rows: Vec::new(),
                raw_y: Vec::new(),
                latent_y: Vec::new(),
                latent_v: Vec::new(),
                center,
                state: base_state,
                inc_idx: 0,
                gpu: None,
            });
        }
        Ok(Self {
            spec,
            ledger,
            slots,
            all_rows: Vec::new(),
            all_y: Vec::new(),
            all_v: Vec::new(),
            has_var: false,
            step: 0,
            gpu_mode: None,
        })
    }

    pub fn with_gpu(spec: RegionalSpec, mode: CudaKnnMode) -> CudaResult<Self> {
        let mut reg = Self::new(spec)?;
        reg.gpu_mode = Some(mode);
        for slot in &mut reg.slots {
            let idx = CudaIndex::with_mode(reg.spec.base.num_dim, &[], mode)?;
            let scratch = idx.new_scratch()?;
            slot.gpu = Some((idx, scratch));
        }
        Ok(reg)
    }

    pub fn len(&self) -> usize {
        self.all_y.len()
    }

    pub fn is_empty(&self) -> bool {
        self.all_y.is_empty()
    }

    pub fn ledger(&self) -> &RegionalLedger {
        &self.ledger
    }

    pub fn slot_center(&self, slot: usize) -> Option<&[f32]> {
        self.slots.get(slot).map(|s| s.center.as_slice())
    }

    pub fn slot_len(&self, slot: usize) -> Option<usize> {
        self.slots.get(slot).map(|s| s.raw_y.len())
    }

    fn slot_box(&self, slot: usize) -> (Vec<f64>, Vec<f64>) {
        let d = self.spec.base.num_dim;
        let mut sides = vec![1.0_f32; d];
        if self.spec.base.mods.weights.is_some() || self.spec.base.mods.scale_x.is_some() {
            let w = eff_weights(d, Some(&self.spec.base.mods));
            ard_sides::<f32>(&w, &mut sides);
        }
        let mut lo = vec![0.0_f32; d];
        let mut hi = vec![1.0_f32; d];
        region_bounds::<f32>(
            &self.slots[slot].center,
            self.slots[slot].state.length as f32,
            Some(&sides),
            &mut lo,
            &mut hi,
        );
        (
            lo.into_iter().map(f64::from).collect(),
            hi.into_iter().map(f64::from).collect(),
        )
    }

    fn propose_slot(&mut self, slot: usize, arms: usize) -> CudaResult<ProposalHost> {
        let d = self.spec.base.num_dim;
        let n = self.slots[slot].raw_y.len();
        let key = Key::new(self.spec.base.seed)
            .fold(u64::from(self.step))
            .fold(slot as u64);
        let acq = match self.spec.base.acquisition {
            AcquisitionRule::Ucb { beta } => AcquisitionRule::Ucb { beta },
            AcquisitionRule::Thompson { .. } => AcquisitionRule::Thompson {
                seed: key.fold(0x7468_6f6d).seed(),
            },
            AcquisitionRule::Pareto { .. } => AcquisitionRule::Pareto {
                seed: key.fold(0x7061_7265).seed(),
            },
            AcquisitionRule::Random { .. } => AcquisitionRule::Random {
                seed: key.fold(0x7261_6e64).seed(),
            },
        };
        let cands = self.spec.base.num_candidates.max(arms);
        let plan = if n == 0 {
            RoundPlan::warmup_round(
                key,
                self.step,
                d,
                cands,
                arms,
                step_warmup(self.spec.base.warmup_kind, self.step, cands),
            )
        } else {
            RoundPlan {
                key,
                step: self.step,
                num_dim: d,
                num_candidates: cands,
                num_arms: arms,
                phase: LifecyclePhase::Active {
                    sampler: step_sampler(self.spec.base.sampler, self.step, cands),
                    surrogate: self.spec.base.surrogate_mode(),
                },
                acquisition: acq,
                region: RegionPlan::Single {
                    state: self.slots[slot].state,
                },
                forced_slot: None,
            }
        };
        let mut mods = self.spec.base.mods.clone();
        mods.scale_k = self.spec.base.scale_k();
        if n > 0 {
            mods.inc_mean = self.slots[slot].raw_y[self.slots[slot].inc_idx];
        }
        if MetricKind::from_metric(self.spec.base.metric) != MetricKind::Euclidean
            && mods.local_scales.is_none()
            && n > 0
        {
            let w = eff_weights(d, Some(&mods));
            mods.local_scales = Some(local_radii(&self.slots[slot].rows, n, d, &w, mods.scale_k));
        }

        let use_noise = self.spec.base.noise_aware && self.has_var;
        let SlotMemory {
            rows,
            latent_y,
            latent_v,
            center,
            inc_idx,
            gpu,
            ..
        } = &mut self.slots[slot];
        let var_opt = use_noise.then_some(latent_v.as_slice());

        if let Some((idx, scratch)) = gpu.as_mut() {
            return idx.propose_pool(
                scratch,
                &plan,
                center,
                latent_y,
                var_opt,
                1,
                *inc_idx,
                Some(&mods),
            );
        }
        simulate_round(
            rows,
            n,
            &plan,
            center,
            latent_y,
            var_opt,
            1,
            *inc_idx,
            Some(&mods),
        )
    }

    pub fn ask(&mut self, budget: usize) -> CudaResult<Vec<RegionalBatch>> {
        let total_arms = budget.max(1);
        let eligible: Vec<bool> = (0..self.spec.num_regions)
            .map(|s| !self.ledger.is_pending(s).unwrap_or(true))
            .collect();
        if !eligible.iter().any(|&e| e) {
            return Err("no eligible region slots available".to_string());
        }

        let mut out = Vec::new();
        if self.spec.allocation == Allocation::JointAcquisition {
            let mut flat_scores = Vec::new();
            let mut flat_regs = Vec::new();
            let mut slot_props: Vec<Option<ProposalHost>> = vec![None; self.spec.num_regions];
            for (slot, &ok) in eligible.iter().enumerate() {
                if !ok {
                    continue;
                }
                let prop = self.propose_slot(slot, total_arms)?;
                for dec in &prop.decisions {
                    flat_scores.push(dec.score);
                    flat_regs.push(slot);
                }
                slot_props[slot] = Some(prop);
            }
            let mut winners = vec![0usize; total_arms];
            let j_key = Key::new(self.spec.base.seed).fold(u64::from(self.step));
            let picked =
                joint_select::<f32>(&flat_scores, &flat_regs, &eligible, j_key, &mut winners);

            let d = self.spec.base.num_dim;
            let mut per_slot: Vec<Vec<usize>> = vec![Vec::new(); self.spec.num_regions];
            let mut cursor = 0usize;
            for (slot, prop_opt) in slot_props.iter().enumerate() {
                if let Some(prop) = prop_opt {
                    let count = prop.num_arms;
                    for &w in &winners[..picked] {
                        if w >= cursor && w < cursor + count {
                            per_slot[slot].push(w - cursor);
                        }
                    }
                    cursor += count;
                }
            }
            for (slot, local_arms) in per_slot.into_iter().enumerate() {
                if local_arms.is_empty() {
                    continue;
                }
                let prop = slot_props[slot].as_ref().expect("present");
                let mut rows = Vec::with_capacity(local_arms.len() * d);
                let mut decisions = Vec::with_capacity(local_arms.len());
                let mut selected = Vec::with_capacity(local_arms.len());
                for (arm_i, &li) in local_arms.iter().enumerate() {
                    rows.extend_from_slice(&prop.rows[li * d..(li + 1) * d]);
                    let mut dec = prop.decisions[li];
                    dec.arm = arm_i as u32;
                    decisions.push(dec);
                    selected.push(prop.selected[li]);
                }
                snap_rows(&mut rows, local_arms.len(), d, Some(&self.spec.base.mods));
                let ticket = self.ledger.issue(slot).map_err(|e| e.to_string())?;
                out.push(RegionalBatch {
                    slot,
                    ticket,
                    proposal: ProposalHost {
                        rows,
                        decisions,
                        selected,
                        num_arms: local_arms.len(),
                        num_dim: d,
                        kind: prop.kind,
                    },
                });
            }
            self.step = self.step.wrapping_add(1);
            return Ok(out);
        }

        let utility: Vec<f64> = (0..self.spec.num_regions)
            .map(|s| {
                let b = self.ledger.best(s).unwrap_or(f64::NEG_INFINITY);
                if b.is_finite() { b } else { 0.0 }
            })
            .collect();
        let batches = self
            .ledger
            .allocate_with(total_arms, &eligible, &utility)
            .map_err(|e| e.to_string())?;
        for b in batches {
            let prop = self.propose_slot(b.region, b.len)?;
            let ticket = self.ledger.issue(b.region).map_err(|e| e.to_string())?;
            out.push(RegionalBatch {
                slot: b.region,
                ticket,
                proposal: prop,
            });
        }
        self.step = self.step.wrapping_add(1);
        Ok(out)
    }

    fn admit_point(
        &mut self,
        slot: usize,
        row: &[f32],
        y: f32,
        zy: f32,
        zv: f32,
    ) -> CudaResult<()> {
        let sm = &mut self.slots[slot];
        let idx = sm.raw_y.len();
        sm.rows.extend_from_slice(row);
        sm.raw_y.push(y);
        sm.latent_y.push(zy);
        sm.latent_v.push(zv);
        if idx == 0 || BetterScore(y) > BetterScore(sm.raw_y[sm.inc_idx]) {
            sm.inc_idx = idx;
            sm.center.copy_from_slice(row);
        }
        if let Some((gpu_idx, mut scratch)) = sm.gpu.take() {
            let next_idx = gpu_idx.add(&mut scratch, row)?;
            sm.gpu = Some((next_idx, scratch));
        }
        Ok(())
    }

    fn restart_slot(&mut self, slot: usize) -> CudaResult<()> {
        let d = self.spec.base.num_dim;
        let cands = self.spec.center_policy.candidates.max(1);
        let gen_id = self.ledger.generation(slot).unwrap_or(0) + 1;
        let r_key = Key::new(self.spec.base.seed)
            .fold(0x7265_7374)
            .fold(slot as u64)
            .fold(gen_id);
        let lo = vec![0.0_f32; d];
        let hi = vec![1.0_f32; d];
        let mut cand_pts = vec![0.0_f64; cands * d];
        for c in 0..cands {
            for j in 0..d {
                cand_pts[c * d + j] = f64::from(eval_uniform::<f32>(r_key, &lo, &hi, c, j));
            }
        }
        let chosen = match self.spec.center_policy.selection {
            RestartCenterSelection::First => 0,
            RestartCenterSelection::Maximin => {
                let mut centers = vec![0.0_f64; self.spec.num_regions * d];
                let mut active = vec![true; self.spec.num_regions];
                active[slot] = false;
                for r in 0..self.spec.num_regions {
                    for j in 0..d {
                        centers[r * d + j] = f64::from(self.slots[r].center[j]);
                    }
                }
                maximin_idx(
                    &cand_pts,
                    &centers,
                    &active,
                    cands,
                    self.spec.num_regions,
                    d,
                )
            }
        };
        let new_center: Vec<f32> = cand_pts[chosen * d..(chosen + 1) * d]
            .iter()
            .map(|&v| v as f32)
            .collect();

        let base_state = TurboState::for_dims(
            d,
            self.spec.base.num_arms.max(1),
            self.spec.base.initial_len,
            self.spec.base.min_len,
            self.spec.base.max_len,
            self.spec.base.failure_tol,
        );
        self.slots[slot].center = new_center;
        self.slots[slot].state = base_state;
        self.slots[slot].rows.clear();
        self.slots[slot].raw_y.clear();
        self.slots[slot].latent_y.clear();
        self.slots[slot].latent_v.clear();
        self.slots[slot].inc_idx = 0;

        if self.spec.memory == RestartMemoryPolicy::RetainRelevant && !self.all_y.is_empty() {
            let (lb, ub) = self.slot_box(slot);
            let pts_f64: Vec<f64> = self.all_rows.iter().map(|&v| f64::from(v)).collect();
            let total = self.all_y.len();
            let mut mask = vec![false; total];
            retain_mask(&pts_f64, &lb, &ub, self.spec.sharing, total, d, &mut mask);
            let yb = self.spec.base.mods.ybounds.as_deref();
            for (i, &keep) in mask.iter().enumerate() {
                if keep {
                    let y = self.all_y[i];
                    let v = self.has_var.then_some(self.all_v[i]);
                    let (zy, zv) = warp_outcome(y, v, yb)?;
                    let local_i = self.slots[slot].raw_y.len();
                    self.slots[slot]
                        .rows
                        .extend_from_slice(&self.all_rows[i * d..(i + 1) * d]);
                    self.slots[slot].raw_y.push(y);
                    self.slots[slot].latent_y.push(zy);
                    self.slots[slot].latent_v.push(zv);
                    if local_i == 0
                        || BetterScore(y)
                            > BetterScore(self.slots[slot].raw_y[self.slots[slot].inc_idx])
                    {
                        self.slots[slot].inc_idx = local_i;
                        self.slots[slot]
                            .center
                            .copy_from_slice(&self.all_rows[i * d..(i + 1) * d]);
                    }
                }
            }
        }

        if let Some(mode) = self.gpu_mode {
            let idx = CudaIndex::with_mode(d, &self.slots[slot].rows, mode)?;
            let scratch = idx.new_scratch()?;
            self.slots[slot].gpu = Some((idx, scratch));
        }

        let inc = if self.slots[slot].raw_y.is_empty() {
            f64::NEG_INFINITY
        } else {
            f64::from(self.slots[slot].raw_y[self.slots[slot].inc_idx])
        };
        self.ledger.restart(slot, inc).map_err(|e| e.to_string())?;
        Ok(())
    }

    pub fn tell(
        &mut self,
        ticket: RegionTicket,
        new_rows: &[f32],
        outcomes: &[f32],
        variances: Option<&[f32]>,
    ) -> CudaResult<bool> {
        self.ledger.validate(ticket).map_err(|e| e.to_string())?;
        let slot = ticket.region.slot;
        let d = self.spec.base.num_dim;
        if outcomes.is_empty() || new_rows.len() != outcomes.len() * d {
            return Err("RegionalCuda::tell shape mismatch".to_string());
        }
        if let Some(v) = variances
            && v.len() != outcomes.len()
        {
            return Err("RegionalCuda::tell variances mismatch".to_string());
        }

        let prev_best = if self.slots[slot].raw_y.is_empty() {
            f32::NEG_INFINITY
        } else {
            self.slots[slot].raw_y[self.slots[slot].inc_idx]
        };
        let yb = self.spec.base.mods.ybounds.clone();
        let mut rewards = Vec::with_capacity(outcomes.len());
        let mut improved = false;

        for (i, &y) in outcomes.iter().enumerate() {
            let row = &new_rows[i * d..(i + 1) * d];
            let v_in = variances.map(|v| v[i]);
            let (zy, zv) = warp_outcome(y, v_in, yb.as_deref())?;
            self.all_rows.extend_from_slice(row);
            self.all_y.push(y);
            self.all_v.push(zv);
            if variances.is_some() {
                self.has_var = true;
            }
            rewards.push(f64::from(y));
            if prev_best.is_finite() && y > prev_best {
                improved = true;
            }
            self.admit_point(slot, row, y, zy, zv)?;

            if self.spec.sharing != SharingPolicy::OriginOnly {
                let row64: Vec<f64> = row.iter().map(|&v| f64::from(v)).collect();
                let mut pairs = Vec::with_capacity(self.spec.num_regions);
                for other in 0..self.spec.num_regions {
                    let (lb, ub) = self.slot_box(other);
                    let c64: Vec<f64> = self.slots[other]
                        .center
                        .iter()
                        .map(|&v| f64::from(v))
                        .collect();
                    let pair = match self.spec.sharing {
                        SharingPolicy::NearestCenter => (simd_l2(&row64, &c64), f64::INFINITY),
                        SharingPolicy::BoxContain => {
                            let ok = baux::semantics::optimizer::portfolio::contains_box(
                                d, &row64, &lb, &ub,
                            );
                            (if ok { 0.0 } else { 2.0 }, 1.0)
                        }
                        SharingPolicy::EllipsoidBall => {
                            let ok = baux::semantics::optimizer::portfolio::contains_ball(
                                d, &row64, &lb, &ub,
                            );
                            (if ok { 0.0 } else { 2.0 }, 1.0)
                        }
                        SharingPolicy::OriginOnly => (2.0, 1.0),
                    };
                    pairs.push(pair);
                }
                let mut routed = vec![0usize; self.spec.num_regions];
                let count = self
                    .spec
                    .sharing
                    .route_slots(Some(slot), &pairs, &mut routed);
                for &target in &routed[..count] {
                    self.admit_point(target, row, y, zy, zv)?;
                    let _ = self.ledger.adopt_shared(target, f64::from(y));
                }
            }
        }

        let inc = f64::from(self.slots[slot].raw_y[self.slots[slot].inc_idx]);
        self.ledger
            .complete(ticket, &rewards, inc)
            .map_err(|e| e.to_string())?;

        if self.spec.controller == RegionController::Turbo && prev_best.is_finite() {
            let (next_state, _, restarted) = self.slots[slot].state.next_step(improved);
            self.slots[slot].state = next_state;
            if restarted {
                self.restart_slot(slot)?;
            }
        }
        Ok(improved)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bowl_eval(rows: &[f32], dims: usize) -> Vec<f32> {
        rows.chunks_exact(dims)
            .map(|r| {
                let d2: f32 = r.iter().map(|&x| (x - 0.35) * (x - 0.35)).sum();
                1.0 / (1.0 + d2)
            })
            .collect()
    }

    #[test]
    fn turbo_modes() {
        for mode in [
            TurboMode::TurboEnn,
            TurboMode::TurboZero,
            TurboMode::LhdOnly,
            TurboMode::SingleBo,
        ] {
            let mut spec = TurboSpec::for_dims(99, 6);
            spec.num_candidates = 192;
            spec.num_arms = 4;
            spec.warmup_obs = 8;
            spec.mode = mode;
            spec.noise_aware = true;
            spec.mods.ybounds = Some(vec![[0.0, 2.0]]);

            let mut opt = TurboCuda::new(spec).unwrap();
            for round in 0..5 {
                let prop = opt.ask().unwrap();
                assert_eq!(prop.num_arms, 4);
                assert_eq!(prop.rows.len(), 24);
                let ys = bowl_eval(&prop.rows, 6);
                let vs = vec![0.001_f32; 4];
                opt.tell(&prop.rows, &ys, Some(&vs)).unwrap();
                if round == 2 && mode == TurboMode::TurboEnn {
                    opt.fit_loo();
                }
            }
            assert_eq!(opt.len(), 20);
            let (_, _, best_y) = opt.incumbent().unwrap();
            assert!(best_y > 0.0 && best_y < 2.0);
        }
    }

    #[test]
    fn morbo_modes() {
        for resc in [RescalarizeMode::OnRestart, RescalarizeMode::OnPropose] {
            let mut spec = TurboSpec::for_dims(314, 5);
            spec.num_candidates = 128;
            spec.num_arms = 2;
            spec.warmup_obs = 6;
            spec.min_len = 0.45;
            spec.failure_tol = Some(1);

            let mut morbo = MorboCuda::new(spec, 3, 12, resc, 0.05).unwrap();
            assert_eq!(morbo.regions().len(), 12);
            let initial_w0 = morbo.regions()[0].weights.clone();

            for step in 0..8 {
                let (reg, prop) = morbo.ask().unwrap();
                assert!(reg < 12);
                let mut ys = Vec::with_capacity(prop.num_arms * 3);
                for r in prop.rows.chunks_exact(5) {
                    ys.push(1.0 - (r[0] - 0.2).powi(2));
                    ys.push(1.0 - (r[1] - 0.8).powi(2));
                    ys.push(0.5 - (r[2] - 0.5).abs());
                }
                if step >= 4 {
                    ys.fill(-10.0);
                }
                morbo.tell(reg, &prop.rows, &ys, None).unwrap();
            }
            if resc == RescalarizeMode::OnPropose {
                let _ = morbo.ask_region(0).unwrap();
                assert_ne!(morbo.regions()[0].weights, initial_w0);
            }
            assert!(!morbo.pareto_indices().is_empty());
        }
    }

    #[test]
    fn regional_modes() {
        for alloc in [
            Allocation::LeastTrials,
            Allocation::IncumbentUcb1 { exploration: 0.8 },
            Allocation::JointAcquisition,
        ] {
            for sharing in [
                SharingPolicy::BoxContain,
                SharingPolicy::EllipsoidBall,
                SharingPolicy::NearestCenter,
                SharingPolicy::OriginOnly,
            ] {
                let mut rspec = RegionalSpec::for_dims(512, 4, 10);
                rspec.base.num_candidates = 64;
                rspec.base.num_arms = 2;
                rspec.base.min_len = 0.5;
                rspec.base.failure_tol = Some(1);
                rspec.allocation = alloc;
                rspec.sharing = sharing;
                rspec.memory = RestartMemoryPolicy::RetainRelevant;
                rspec.center_policy = RestartCenterPolicy {
                    candidates: 12,
                    selection: RestartCenterSelection::Maximin,
                };

                let mut reg = RegionalCuda::new(rspec).unwrap();
                for step in 0..4 {
                    let batches = reg.ask(3).unwrap();
                    assert!(!batches.is_empty());
                    let total_arms: usize = batches.iter().map(|b| b.proposal.num_arms).sum();
                    assert_eq!(total_arms, 3);
                    for batch in batches {
                        let mut ys = bowl_eval(&batch.proposal.rows, 4);
                        if step >= 2 {
                            ys.fill(-5.0);
                        }
                        reg.tell(batch.ticket, &batch.proposal.rows, &ys, None)
                            .unwrap();
                    }
                }
                assert_eq!(reg.len(), 12);
            }
        }
    }

    #[test]
    fn sobol_steps() {
        for sampler in [
            PoolSampler::Sobol { base_idx: 0 },
            PoolSampler::SobolMasked {
                base_idx: 0,
                prob: 0.75,
            },
        ] {
            let mut spec = TurboSpec::for_dims(77, 4);
            spec.num_candidates = 32;
            spec.num_arms = 2;
            spec.warmup_obs = 0;
            spec.sampler = sampler;
            let mut opt = TurboCuda::new(spec.clone()).unwrap();
            opt.tell(&[0.5_f32; 8], &[1.0, 0.5], None).unwrap();
            let p0 = opt.plan().pool_dispatch(1, 2, 0);
            let a0 = opt.ask().unwrap();
            let p1 = opt.plan().pool_dispatch(1, 2, 0);
            let a1 = opt.ask().unwrap();
            assert_eq!(p0.base_idx, 0);
            assert_eq!(p1.base_idx, 32);
            assert_ne!(a0.rows, a1.rows);

            let mut morbo = MorboCuda::new(spec, 2, 3, RescalarizeMode::OnPropose, 0.05).unwrap();
            morbo
                .tell(0, &[0.5_f32; 8], &[1.0, 0.2, 0.4, 0.9], None)
                .unwrap();
            let (_, r0) = morbo.ask().unwrap();
            let (_, r1) = morbo.ask().unwrap();
            assert_ne!(r0.rows, r1.rows);
        }
    }
}
