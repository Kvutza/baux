//! Stateless perturbation samplers, ARD metric learning, coordinate manifolds,
//! Stiefel subspace contraction, Affine WLS/conformal calibration, and
//! `y_bounds` output warps for resident Apple Silicon Metal search.

use super::*;
use crate::candidates::CandidateSampler;
use crate::layout::MetricLearning;
use crate::metric_auto::AutoMetric;
use crate::proposal_distribution::CoordSystem;
use crate::y_bounds::{inv_scalar, jac_inv, project_mu, warp_scalar};
use baux::semantics::affine::Affine;
use baux::semantics::stiefel::Stiefel;
use std::sync::Arc;

/// Stateless candidate perturbation law evaluated inside the Metal shader.
#[repr(u32)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ResidentSampler {
    /// 256-strip Ziggurat standard normal around incumbent (`SingleStateBo`
    /// default).
    #[default]
    Gaussian = 0,
    /// Rademacher `{-1, +1}` steps around incumbent.
    Rademacher = 1,
    /// Box-bounded uniform draw inside `[tr_lo, tr_hi]`.
    Uniform = 2,
    /// Box-bounded Random Axis-Aligned Subspace Perturbation (`RAASP`) in `[0,
    /// 1]^D`.
    Raasp = 3,
    /// Stateless Gray-code + Owen-scrambled Sobol sequence in `[tr_lo, tr_hi]`.
    Sobol = 4,
    /// Stratified Latin Hypercube Design (`LHD`) warmup over `[0, 1]^D`.
    Lhd = 5,
    /// Center-masked Sobol perturbation (`RAASP` mask + Owen-scrambled Sobol
    /// draw).
    SobolMasked = 6,
}

impl ResidentSampler {
    pub const fn shader(self) -> u32 {
        self as u32
    }

    pub const fn name(self) -> &'static str {
        match self {
            Self::Gaussian => "independent_gaussian_ziggurat256_v1",
            Self::Rademacher => "independent_rademacher",
            Self::Uniform => "box_uniform",
            Self::Raasp => "box_raasp",
            Self::Sobol => "box_sobol_owen",
            Self::Lhd => "box_lhd",
            Self::SobolMasked => "box_sobol_masked",
        }
    }

    pub const fn from_distribution(dist: ProposalDistribution) -> Self {
        match dist {
            ProposalDistribution::Gaussian => Self::Gaussian,
            ProposalDistribution::Rademacher => Self::Rademacher,
        }
    }

    pub const fn from_sampler(sampler: CandidateSampler) -> Self {
        match sampler {
            CandidateSampler::Uniform => Self::Uniform,
            CandidateSampler::RAASP => Self::Raasp,
            CandidateSampler::Sobol => Self::Sobol,
        }
    }
}

/// Optional input/output manifold and surrogate transformation policy for
/// resident Metal search.
#[derive(Clone, Debug)]
pub struct ResidentManifold {
    pub scale_x: bool,
    pub metric_learning: MetricLearning,
    pub tied_dims: Vec<Vec<usize>>,
    pub unscaled_dims: Vec<usize>,
    /// `(dimension, num_categories)` for categorical dimensions embedded in
    /// `[0, 1]`.
    pub categories: Vec<(usize, usize)>,
    pub coord_system: CoordSystem,
    pub stiefel_rank: Option<usize>,
    pub calibrate_alpha: Option<f64>,
    /// Optional per-metric `(lower, upper)` bounds (`(-inf, +inf)` for
    /// unbounded).
    pub y_bounds: Option<Vec<(f64, f64)>>,
    pub noise_aware: bool,
    pub no_trust_region: bool,
}

impl Default for ResidentManifold {
    fn default() -> Self {
        Self {
            scale_x: false,
            metric_learning: MetricLearning::None,
            tied_dims: Vec::new(),
            unscaled_dims: Vec::new(),
            categories: Vec::new(),
            coord_system: CoordSystem::Cartesian,
            stiefel_rank: None,
            calibrate_alpha: None,
            y_bounds: None,
            noise_aware: true,
            no_trust_region: false,
        }
    }
}

#[derive(Clone)]
pub(super) struct ManifoldState {
    pub(super) config: ResidentManifold,
    pub(super) dim_meta: Vec<DimMeta>,
    pub(super) dim_meta_gpu: Arc<Buffer>,
    pub(super) auto_metric: Option<AutoMetric>,
    pub(super) affine: Option<Affine>,
    pub(super) stiefel: Option<Stiefel>,
}

impl std::fmt::Debug for ManifoldState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ManifoldState")
            .field("config", &self.config)
            .field("dims", &self.dim_meta.len())
            .finish()
    }
}

pub(super) fn bounds_mode(bounds: (f64, f64)) -> (u32, f32, f32) {
    match (bounds.0.is_finite(), bounds.1.is_finite()) {
        (false, false) => (0, 0.0, 0.0),
        (true, false) => (1, bounds.0 as f32, 0.0),
        (false, true) => (2, 0.0, bounds.1 as f32),
        (true, true) => (3, bounds.0 as f32, bounds.1 as f32),
    }
}

pub(super) fn warp_observation(y: f32, var: f32, bounds: Option<(f64, f64)>) -> (f32, f32) {
    let Some((a, b)) = bounds else {
        return (y, var);
    };
    if !a.is_finite() && !b.is_finite() {
        return (y, var);
    }
    let projected = project_mu(f64::from(y), a, b);
    let z = warp_scalar(projected, a, b);
    let jac = jac_inv(z, a, b).max(1.0e-12);
    let z_var = f64::from(var) / (jac * jac);
    (z as f32, z_var as f32)
}

#[allow(dead_code)]
pub(super) fn naturalize_prediction(
    mean_z: f32,
    se_z: f32,
    bounds: Option<(f64, f64)>,
) -> (f32, f32) {
    let Some((a, b)) = bounds else {
        return (mean_z, se_z);
    };
    if !a.is_finite() && !b.is_finite() {
        return (mean_z, se_z);
    }
    let z = f64::from(mean_z);
    let mean_y = inv_scalar(z, a, b);
    let se_y = f64::from(se_z) * jac_inv(z, a, b);
    (mean_y as f32, se_y as f32)
}

pub(super) fn coord_code(cs: CoordSystem) -> u32 {
    match cs {
        CoordSystem::Cartesian => 0,
        CoordSystem::Spherical => 1,
        CoordSystem::Polar => 2,
        CoordSystem::OriginSphere => 3,
    }
}

/// Build 32 direction numbers per dimension matching
/// `baux::semantics::pool::joe_kuo`.
pub(super) fn sobol_directions(dimensions: usize) -> Vec<u32> {
    let count = dimensions.max(1);
    let mut dirs = Vec::with_capacity(count * 32);
    for dim_idx in 0..count {
        dirs.extend_from_slice(&baux::semantics::pool::joe_kuo(dim_idx));
    }
    dirs
}

fn decode_coord(bits: u16, fp16: bool) -> f64 {
    if fp16 {
        f64::from(crate::unified_mem::decode_half(bits))
    } else {
        f64::from(f32::from_bits(u32::from(bits) << 16))
    }
}

impl SearchState {
    /// Configure stateless perturbation sampler (`Gaussian`, `Rademacher`,
    /// `Uniform`, `Raasp`, `Sobol`, `SobolMasked`, `Lhd`), optional `num_pert`
    /// (defaulting to `min(20, D)` for `Raasp`/`SobolMasked`), and `[0, 1]^D`
    /// box clamping.
    pub fn configure_sampler(
        mut self,
        sampler: ResidentSampler,
        num_pert: Option<usize>,
        box_bounded: bool,
    ) -> Result<Self, String> {
        self.check_idle()?;
        self.sampler = sampler;
        self.box_bounded = box_bounded
            || matches!(
                sampler,
                ResidentSampler::Uniform
                    | ResidentSampler::Raasp
                    | ResidentSampler::Sobol
                    | ResidentSampler::SobolMasked
                    | ResidentSampler::Lhd
            );
        self.num_pert = num_pert
            .unwrap_or_else(|| {
                if matches!(
                    sampler,
                    ResidentSampler::Raasp | ResidentSampler::SobolMasked
                ) {
                    self.dimensions.min(20).max(1)
                } else {
                    self.dimensions.max(1)
                }
            })
            .clamp(1, self.dimensions.max(1));
        if matches!(
            sampler,
            ResidentSampler::Sobol | ResidentSampler::SobolMasked
        ) && self.sobol_dirs_gpu.is_none()
        {
            let dirs = sobol_directions(self.dimensions);
            self.sobol_dirs_gpu = Some(Arc::new(self.runtime.buffer_with(&dirs)));
        }
        Ok(self)
    }

    /// Configure LHD warmup horizon (`lhd_points >= 1`).
    pub fn configure_lhd(mut self, points: usize) -> Result<Self, String> {
        self.check_idle()?;
        self.lhd_points = points.max(1) as u32;
        Ok(self)
    }

    /// Configure input/output manifold options (`scale_x`,
    /// `MetricLearning::Auto`, `tied_dims`, `unscaled_dims`, `categories`,
    /// `CoordSystem`, `stiefel_rank`, `calibrate_alpha`, `y_bounds`,
    /// `noise_aware`, `no_trust_region`).
    pub fn configure_manifold(mut self, manifold: ResidentManifold) -> Result<Self, String> {
        self.check_idle()?;
        if let Some(alpha) = manifold.calibrate_alpha
            && (!alpha.is_finite() || alpha <= 0.0 || alpha >= 1.0)
        {
            return Err("calibrate_alpha must be in (0, 1)".into());
        }
        if let Some(bounds) = &manifold.y_bounds {
            for &(lo, hi) in bounds {
                if lo.is_nan() || hi.is_nan() || (lo.is_finite() && hi.is_finite() && lo >= hi) {
                    return Err(format!("invalid y_bounds ({lo}, {hi})"));
                }
            }
        }
        let base_weight = self
            .blocks
            .first()
            .map_or(1.0 / (self.dimensions.max(1) as f32), |b| b.weight);
        let mut dim_meta = vec![
            DimMeta {
                weight: base_weight,
                tr_half_scale: 1.0,
                tied_leader: u32::MAX,
                categories: 0,
            };
            self.dimensions
        ];
        for group in &manifold.tied_dims {
            if let Some(&leader) = group.first()
                && leader < self.dimensions
            {
                for &member in group {
                    if member < self.dimensions {
                        dim_meta[member].tied_leader = leader as u32;
                    }
                }
            }
        }
        for &(dim, cats) in &manifold.categories {
            if dim < self.dimensions && cats >= 2 {
                dim_meta[dim].categories = cats as u32;
            }
        }
        let auto_metric = if manifold.metric_learning == MetricLearning::Auto {
            let mut am = AutoMetric::new(
                self.dimensions.max(1),
                1,
                manifold.tied_dims.clone(),
                self.fit_seed ^ 0x4d45_5452_4943_u64,
            )
            .map_err(|e| e.to_string())?;
            if !manifold.unscaled_dims.is_empty() {
                am.set_unscaled(manifold.unscaled_dims.clone())
                    .map_err(|e| e.to_string())?;
            }
            Some(am)
        } else {
            None
        };
        let dim_meta_gpu = Arc::new(self.runtime.buffer_with(&dim_meta));
        self.noise_aware = manifold.noise_aware;
        if manifold.no_trust_region {
            self.adapt_controller = false;
            self.box_bounded = true;
            self.length = 2.0;
        }
        self.manifold = Some(ManifoldState {
            config: manifold,
            dim_meta,
            dim_meta_gpu,
            auto_metric,
            affine: None,
            stiefel: None,
        });
        Ok(self)
    }

    pub(super) fn refresh_manifold(&mut self) -> Result<(), String> {
        let Some(mut ms) = self.manifold.take() else {
            return Ok(());
        };
        let n = self.history;
        let res_n = self.resident_history.min(self.history_rows.len());
        let d = self.dimensions;
        let base_weight = self
            .blocks
            .first()
            .map_or(1.0 / (d.max(1) as f32), |b| b.weight);

        // 1. Update `scale_x` and `AutoMetric` ARD weights when resident rows
        //    exist
        if (ms.config.scale_x || ms.auto_metric.is_some()) && res_n >= 2 && d <= 65_536 {
            let mut x_rows = ndarray::Array2::<f64>::zeros((res_n, d));
            for r in 0..res_n {
                let raw = read::<u16>(&*self.history_rows[r], d);
                for j in 0..d {
                    x_rows[[r, j]] = decode_coord(raw[j], self.independent_fp16);
                }
            }
            let mut scale_inv_sq = vec![1.0f64; d];
            if ms.config.scale_x {
                for j in 0..d {
                    if ms.config.unscaled_dims.contains(&j) {
                        continue;
                    }
                    let mean = (0..res_n).map(|r| x_rows[[r, j]]).sum::<f64>() / (res_n as f64);
                    let var = (0..res_n)
                        .map(|r| (x_rows[[r, j]] - mean).powi(2))
                        .sum::<f64>()
                        / ((res_n - 1).max(1) as f64);
                    let std = var.sqrt().max(1.0e-3);
                    scale_inv_sq[j] = (1.0 / (std * std)).clamp(1.0e-3, 1.0e3);
                }
            }
            let mut ard_weights = vec![1.0f64; d];
            if ms.auto_metric.is_some() && res_n >= 4 {
                if let Ok(mut am) = AutoMetric::new(
                    d.max(1),
                    1,
                    ms.config.tied_dims.clone(),
                    self.fit_seed ^ 0x4d45_5452_4943_u64,
                ) {
                    if !ms.config.unscaled_dims.is_empty() {
                        let _ = am.set_unscaled(ms.config.unscaled_dims.clone());
                    }
                    let mut x_flat = Vec::with_capacity(res_n * d);
                    let mut y_flat = Vec::with_capacity(res_n);
                    for r in 0..res_n {
                        for j in 0..d {
                            x_flat.push(x_rows[[r, j]]);
                        }
                        let logical = if self.implicit_history {
                            self.identities[..n]
                                .iter()
                                .position(|&id| id == self.resident_identities[r])
                                .unwrap_or(0)
                        } else {
                            r.min(n.saturating_sub(1))
                        };
                        y_flat.push(f64::from(self.outcomes[logical]));
                    }
                    let _ = am.observe(&x_flat, &y_flat, res_n);
                    for (j, &w) in am.weights().iter().take(d).enumerate() {
                        ard_weights[j] = w.clamp(1.0e-3, 1.0e3);
                    }
                    ms.auto_metric = Some(am);
                }
            }
            // Volume-preserving trust-region half-scales: l_j ~ 1 /
            // sqrt(ard_weights[j])
            let log_mean_l = ard_weights
                .iter()
                .map(|&w| -0.5 * w.max(1.0e-12).ln())
                .sum::<f64>()
                / (d.max(1) as f64);
            for j in 0..d {
                let raw_l = (-0.5 * ard_weights[j].max(1.0e-12).ln() - log_mean_l)
                    .exp()
                    .clamp(0.1, 10.0);
                ms.dim_meta[j].weight =
                    (f64::from(base_weight) * scale_inv_sq[j] * ard_weights[j]) as f32;
                ms.dim_meta[j].tr_half_scale = raw_l as f32;
            }
            // Enforce tied_dims equality on weights & scales
            for group in &ms.config.tied_dims {
                if let Some(&leader) = group.first()
                    && leader < d
                {
                    let w = ms.dim_meta[leader].weight;
                    let s = ms.dim_meta[leader].tr_half_scale;
                    for &member in group {
                        if member < d {
                            ms.dim_meta[member].weight = w;
                            ms.dim_meta[member].tr_half_scale = s;
                        }
                    }
                }
            }
            ms.dim_meta_gpu = Arc::new(self.runtime.buffer_with(&ms.dim_meta));

            // Keep CPU pairwise_distances consistent when explicit/resident
            // rows cover history
            if !self.implicit_history && res_n == n {
                for r in 0..n {
                    self.pairwise_distances[r * MAX_HISTORY + r] = 0.0;
                    for c in 0..r {
                        let mut dist = 0.0f64;
                        for j in 0..d {
                            let diff = x_rows[[r, j]] - x_rows[[c, j]];
                            dist += diff * diff * f64::from(ms.dim_meta[j].weight);
                        }
                        let df = dist as f32;
                        self.pairwise_distances[r * MAX_HISTORY + c] = df;
                        self.pairwise_distances[c * MAX_HISTORY + r] = df;
                    }
                }
            }
        }

        // 2. Fit Affine WLS + conformal residual calibration
        //    (`calibrate_alpha`)
        if let Some(alpha) = ms.config.calibrate_alpha
            && n >= 4
        {
            let records: Vec<_> = self.objective_history.rows().collect();
            let cols = if records.len() == n && records[0].observation.is_vector() {
                records[0].observation.estimates().len()
            } else {
                1
            };
            let (k_loo, ep, al, ys) = self.fitted_enn.unwrap_or((
                self.initial_observations.clamp(1, n - 1),
                0.05,
                0.05,
                1.0,
            ));
            let mut mu = vec![0.0f64; n * cols];
            let mut se = vec![1.0f64; n * cols];
            let mut y = vec![0.0f64; n * cols];
            for row in 0..n {
                let dists: Vec<f64> = (0..n)
                    .map(|col| f64::from(self.total_distance(row, col)))
                    .collect();
                for col in 0..cols {
                    let (outcomes_col, vars_col, scale_col): (Vec<f32>, Vec<f32>, f32) = if cols
                        == 1
                    {
                        let bounds = ms.config.y_bounds.as_ref().and_then(|b| b.first().copied());
                        let warped: Vec<(f32, f32)> = (0..n)
                            .map(|i| {
                                warp_observation(
                                    self.outcomes[i],
                                    if self.noise_aware {
                                        self.variances[i]
                                    } else {
                                        0.0
                                    },
                                    bounds,
                                )
                            })
                            .collect();
                        (
                            warped.iter().map(|p| p.0).collect(),
                            warped.iter().map(|p| p.1).collect(),
                            ys,
                        )
                    } else {
                        let bounds = ms
                            .config
                            .y_bounds
                            .as_ref()
                            .and_then(|b| b.get(col).copied());
                        let sc = self
                            .objective_selector
                            .as_ref()
                            .and_then(|s| s.scales().get(col).copied())
                            .unwrap_or(1.0);
                        let warped: Vec<(f32, f32)> = records
                            .iter()
                            .map(|rec| {
                                let est = rec.observation.estimates()[col];
                                warp_observation(
                                    est.mean,
                                    if self.noise_aware { est.variance } else { 0.0 },
                                    bounds,
                                )
                            })
                            .collect();
                        (
                            warped.iter().map(|p| p.0).collect(),
                            warped.iter().map(|p| p.1).collect(),
                            sc,
                        )
                    };
                    y[row * cols + col] = f64::from(outcomes_col[row]);
                    if let Ok(pred) = (baux::semantics::neighbor::query::Query {
                        distances: &dists,
                        outcomes: &outcomes_col,
                        variances: &vars_col,
                        draws: None,
                        neighbors: k_loo.min(n - 1).max(1),
                        epistemic: ep,
                        aleatoric: al,
                        scale: scale_col,
                        beta: 1.0,
                        excluded: Some(row),
                        incumbent: 0,
                        adaptive: self.adapt_neighbors,
                    })
                    .predict()
                    {
                        mu[row * cols + col] = f64::from(pred.mean);
                        se[row * cols + col] = f64::from(pred.standard_error).max(1.0e-6);
                    }
                }
            }
            if let Ok(aff) = Affine::fit_wls(&mu, &y, &se, n, cols) {
                // Conformal residual multiplier at `alpha`
                let mut c_conf = aff.c.clone();
                for col in 0..cols {
                    let scores: Vec<f64> = (0..n)
                        .map(|r| {
                            let pred = aff.a[col] + aff.b[col] * mu[r * cols + col];
                            (y[r * cols + col] - pred).abs() / se[r * cols + col].max(1.0e-6)
                        })
                        .collect();
                    if let Ok(q) = baux::semantics::affine::conformal_quantile(&scores, alpha)
                        && q > 1.0e-6
                    {
                        c_conf[col] = q;
                    }
                }
                if let Ok(calibrated) = Affine::new(aff.a, aff.b, c_conf) {
                    ms.affine = Some(calibrated);
                }
            }
        }

        // 3. Fit Stiefel subspace frame (`stiefel_rank`) for multi-objective
        //    observations
        if let Some(rank) = ms.config.stiefel_rank {
            let records: Vec<_> = self.objective_history.rows().collect();
            if records.len() >= 3 && records[0].observation.is_vector() {
                let cols = records[0].observation.estimates().len();
                let r_eff = rank.clamp(1, cols);
                let mut y_flat = Vec::with_capacity(records.len() * cols);
                for rec in &records {
                    for est in rec.observation.estimates() {
                        y_flat.push(f64::from(est.mean));
                    }
                }
                if let Ok(st) = Stiefel::from_cov(&y_flat, records.len(), cols, r_eff) {
                    ms.stiefel = Some(st);
                }
            }
        }

        self.manifold = Some(ms);
        Ok(())
    }
}
