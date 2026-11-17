//! Candidate generation for trust region optimization.

use std::sync::Arc;

use baux::semantics::key::Key;
use baux::semantics::pool::{DelayedPool, cand_key, joe_kuo, sobol_unit};
use deser::{Deserialize, Serialize};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use rand::Rng;
use rand::RngCore;

pub use crate::coordinate::{from_unit, to_unit};
use crate::error::ENNError;
use crate::posterior::PullPool;
pub use crate::proposal_distribution::CoordSystem;

/// Sampling construction used to generate bounded candidates.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[deser(rename_all = "snake_case")]
pub enum CandidateSampler {
    /// Sobol quasi-random sequence.
    Sobol,
    /// Uniform random sampling.
    #[default]
    Uniform,
    /// RAASP (Random Axis-Aligned Subspace Perturbation; Rashidi,
    /// Johnstonbaugh, and Gao, AISTATS 2024, PMLR 238:3502-3510; Regis and
    /// Shoemaker, Eng. Optim. 45(5):529-555, 2013).
    #[deser(rename = "raasp")]
    RAASP,
}

/// Upstream candidate random variable alias.
pub type CandidateRV = CandidateSampler;

/// Generate candidates within trust region bounds.
#[allow(clippy::too_many_arguments)]
pub fn generate_candidates<R: Rng + ?Sized>(
    compute_bounds: impl Fn() -> (Array1<f64>, Array1<f64>),
    x_center: &ArrayView1<f64>,
    lengthscales: Option<&ArrayView1<f64>>,
    num_candidates: usize,
    candidate_rv: CandidateSampler,
    rng: &mut R,
    sobol_engine: Option<&mut SobolEngine>,
    num_pert: usize,
) -> Result<Array2<f64>, ENNError> {
    let (lower_1d, upper_1d) = compute_bounds();

    let num_dim = x_center.len();

    match candidate_rv {
        CandidateSampler::Sobol => {
            if let Some(engine) = sobol_engine {
                let mut candidates = Array2::zeros((num_candidates, num_dim));
                for i in 0..num_candidates {
                    let sample = engine.sample(rng)?;
                    for j in 0..num_dim {
                        // Scale from unit to TR bounds
                        let unit = sample[j];
                        candidates[[i, j]] = lower_1d[j] + unit * (upper_1d[j] - lower_1d[j]);
                    }
                }
                Ok(candidates)
            } else {
                // Fallback to uniform if no Sobol engine
                generate_uniform(&lower_1d, &upper_1d, num_candidates, rng)
            }
        }
        CandidateSampler::Uniform => generate_uniform(&lower_1d, &upper_1d, num_candidates, rng),
        CandidateSampler::RAASP => generate_raasp(
            x_center,
            lengthscales,
            &lower_1d,
            &upper_1d,
            num_candidates,
            rng,
            num_pert,
        ),
    }
}

/// Upstream masked Sobol candidate generation for MORBO.
pub fn sobol_masked<R: Rng + ?Sized>(
    x_center: &ArrayView1<f64>,
    lower: &Array1<f64>,
    upper: &Array1<f64>,
    num_candidates: usize,
    rng: &mut R,
    engine: &mut SobolEngine,
    num_pert: usize,
) -> Result<Array2<f64>, ENNError> {
    let num_dim = x_center.len();
    if num_dim == 0 {
        return Err(ENNError::InvalidParameter(
            "num_dim must be > 0".to_string(),
        ));
    }
    let mut sobol_samples = Vec::with_capacity(num_candidates * num_dim);
    for _ in 0..num_candidates {
        sobol_samples.extend_from_slice(&engine.sample(rng)?);
    }
    let x_vec = x_center.to_vec();
    let lower_vec = lower.to_vec();
    let upper_vec = upper.to_vec();
    let flat = baux::semantics::random::sobol_masked(
        &x_vec,
        &lower_vec,
        &upper_vec,
        num_candidates,
        rng,
        &sobol_samples,
        num_pert,
    )
    .map_err(ENNError::InvalidParameter)?;
    Array2::from_shape_vec((num_candidates, num_dim), flat)
        .map_err(|e| ENNError::InvalidParameter(format!("masked Sobol shape mismatch: {e}")))
}

/// Upstream standalone Sobol sequence generation.
pub fn sobol_sequence(dim: usize, n: usize, seed: u64) -> Result<Array2<f64>, ENNError> {
    let mut engine = SobolEngine::new(dim)?;
    let mut rng = baux::semantics::random::prng(seed);
    let mut out = Array2::zeros((n, dim));
    for i in 0..n {
        let row = engine.sample(&mut rng)?;
        for (j, value) in row.into_iter().enumerate() {
            out[[i, j]] = value;
        }
    }
    Ok(out)
}

/// Strategy dispatch for TuRBO candidate drawing.
#[allow(clippy::too_many_arguments)]
pub fn draw_turbo<R: Rng + ?Sized>(
    is_morbo: bool,
    candidate_rv: CandidateRV,
    raasp_fast: bool,
    x_center: &ArrayView1<f64>,
    _lengthscales: Option<&Array1<f64>>,
    lower_1d: &Array1<f64>,
    upper_1d: &Array1<f64>,
    num_candidates: usize,
    rng: &mut R,
    mut sobol_engine: Option<&mut SobolEngine>,
    num_pert: usize,
) -> Result<Array2<f64>, ENNError> {
    if raasp_fast && candidate_rv == CandidateRV::RAASP {
        return crate::candidates_fast::generate(
            x_center,
            lower_1d,
            upper_1d,
            num_candidates,
            rng,
            num_pert,
        );
    }
    if is_morbo && candidate_rv == CandidateRV::Sobol {
        if let Some(ref mut engine) = sobol_engine {
            return sobol_masked(
                x_center,
                lower_1d,
                upper_1d,
                num_candidates,
                rng,
                engine,
                num_pert,
            );
        }
    }
    // Bounds already incorporate lengthscales via compute_bounds; pass None so
    // RAASP perturbs all dimensions at equal rate (matching upstream TuRBO).
    generate_candidates(
        || (lower_1d.clone(), upper_1d.clone()),
        x_center,
        None,
        num_candidates,
        candidate_rv,
        rng,
        sobol_engine,
        num_pert,
    )
}

/// Generate candidates with explicit coordinate system and tied coordinate
/// groups.
#[allow(clippy::too_many_arguments)]
pub fn perturb_candidates<R: Rng + ?Sized>(
    x_center: &ArrayView1<f64>,
    lower: &Array1<f64>,
    upper: &Array1<f64>,
    num_candidates: usize,
    candidate_rv: CandidateSampler,
    coord_system: CoordSystem,
    tied_dims: &[Vec<usize>],
    is_morbo: bool,
    raasp_fast: bool,
    rng: &mut R,
    mut sobol_engine: Option<&mut SobolEngine>,
    num_pert: usize,
) -> Result<Array2<f64>, ENNError> {
    if coord_system == CoordSystem::Cartesian && tied_dims.is_empty() {
        return draw_turbo(
            is_morbo,
            candidate_rv,
            raasp_fast,
            x_center,
            None,
            lower,
            upper,
            num_candidates,
            rng,
            sobol_engine,
            num_pert,
        );
    }
    let num_dim = x_center.len();
    if num_dim == 0 || num_candidates == 0 {
        return Ok(Array2::zeros((num_candidates, num_dim)));
    }
    let units = crate::candidates_fast::build_units(num_dim, tied_dims)?;
    let mut candidates = Array2::zeros((num_candidates, num_dim));
    let raasp = candidate_rv == CandidateSampler::RAASP;
    let masked_sobol = is_morbo && candidate_rv == CandidateSampler::Sobol;
    for i in 0..num_candidates {
        candidates.row_mut(i).assign(x_center);
        let sobol_row = match (candidate_rv, sobol_engine.as_deref_mut()) {
            (CandidateSampler::Sobol, Some(engine)) => Some(engine.sample(rng)?),
            _ => None,
        };
        let active =
            crate::candidates_fast::select_units(units.len(), raasp, masked_sobol, num_pert, rng)?;
        let mut row = candidates.row_mut(i);
        let mut singletons = Vec::new();
        for u in active {
            if tied_dims.is_empty() || units[u].len() == 1 {
                singletons.extend_from_slice(&units[u]);
            } else {
                crate::candidates_fast::perturb_group(
                    &mut row,
                    x_center,
                    lower,
                    upper,
                    &units[u],
                    coord_system,
                    sobol_row.as_deref(),
                    rng,
                );
            }
        }
        crate::candidates_fast::perturb_group(
            &mut row,
            x_center,
            lower,
            upper,
            &singletons,
            coord_system,
            sobol_row.as_deref(),
            rng,
        );
    }
    Ok(candidates)
}

/// Generate uniformly random candidates.
pub fn generate_uniform<R: Rng + ?Sized>(
    lower: &Array1<f64>,
    upper: &Array1<f64>,
    num_candidates: usize,
    rng: &mut R,
) -> Result<Array2<f64>, ENNError> {
    let num_dim = lower.len();
    let mut candidates = Array2::zeros((num_candidates, num_dim));

    for i in 0..num_candidates {
        for j in 0..num_dim {
            if upper[j] > lower[j] {
                candidates[[i, j]] = lower[j] + rng.r#gen::<f64>() * (upper[j] - lower[j]);
            } else {
                candidates[[i, j]] = lower[j];
            }
        }
    }

    Ok(candidates)
}

#[cfg(test)]
fn raasp_cdf(cdf: &Array1<f64>, r: f64, num_dim: usize) -> usize {
    let mut lo = 0usize;
    let mut hi = num_dim.saturating_sub(1);
    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        if r <= cdf[mid] {
            hi = mid;
        } else {
            lo = mid + 1;
        }
    }
    lo
}

/// Generate RAASP candidates.
///
/// References:
/// - Bahador Rashidi, Kerrick Johnstonbaugh, and Chao Gao, "Cylindrical
///   Thompson Sampling for High-Dimensional Bayesian Optimization", AISTATS
///   2024 (PMLR 238:3502-3510).
/// - Rommel G. Regis and Christine A. Shoemaker, "Combining Radial Basis
///   Function Surrogates and Dynamic Coordinate Search in High-Dimensional
///   Expensive Black-Box Optimization", Engineering Optimization 45(5):529-555
///   (2013).
/// - Eriksson et al. (2019), "Scalable Global Optimization via Local Bayesian
///   Optimization" (TuRBO), NeurIPS 2019.
/// - Knuth (1998), "The Art of Computer Programming, Vol 2", Algorithm 3.4.2P
///   (Fisher-Yates / Durstenfeld sampling without replacement).
/// For uniform coordinate perturbation in high dimensions (e.g. D = 160,531, k
/// = 96,319), partial Fisher-Yates shuffle provides O(k) coordinate selection
/// without replacement.
fn generate_raasp<R: Rng + ?Sized>(
    x_center: &ArrayView1<f64>,
    _lengthscales: Option<&ArrayView1<f64>>,
    lower: &Array1<f64>,
    upper: &Array1<f64>,
    num_candidates: usize,
    rng: &mut R,
    num_pert: usize,
) -> Result<Array2<f64>, ENNError> {
    let num_dim = x_center.len();
    if num_dim == 0 || num_candidates == 0 {
        return Ok(Array2::zeros((num_candidates, num_dim)));
    }
    let num_pert = num_pert.max(1).min(num_dim);

    let mut candidates = Array2::zeros((num_candidates, num_dim));
    let mut indices: Vec<usize> = (0..num_dim).collect();
    let lo = lower.as_slice().expect("lower bounds contiguous");
    let hi = upper.as_slice().expect("upper bounds contiguous");

    for i in 0..num_candidates {
        candidates.row_mut(i).assign(x_center);
        let row = candidates
            .row_mut(i)
            .into_slice()
            .expect("candidate row contiguous");
        if num_pert >= num_dim {
            for d in 0..num_dim {
                if hi[d] > lo[d] {
                    row[d] = lo[d] + rng.r#gen::<f64>() * (hi[d] - lo[d]);
                } else {
                    row[d] = lo[d];
                }
            }
        } else {
            // Partial Fisher-Yates shuffle for O(num_pert) selection
            for p in 0..num_pert {
                let swap_idx = p + (rng.r#gen::<usize>() % (num_dim - p));
                indices.swap(p, swap_idx);
                let dim_to_perturb = indices[p];
                if hi[dim_to_perturb] > lo[dim_to_perturb] {
                    row[dim_to_perturb] = lo[dim_to_perturb]
                        + rng.r#gen::<f64>() * (hi[dim_to_perturb] - lo[dim_to_perturb]);
                } else {
                    row[dim_to_perturb] = lo[dim_to_perturb];
                }
            }
        }
    }

    Ok(candidates)
}

fn shift_key(shift: Option<&[f64]>) -> Key {
    let mut key = Key::new(0x534f_424f_4c53_4846);
    if let Some(vals) = shift {
        for &v in vals {
            key = key.fold(v.to_bits());
        }
    }
    key
}

/// Stateless $O(1)$-cloned Sobol cursor slice for [`DelayedPool`].
#[derive(Clone, Debug)]
pub struct SobolStep {
    pub dirs: Arc<[[u32; 32]]>,
    pub base_idx: u32,
    pub key: Key,
    pub scramble: bool,
}

/// Sobol quasi-random sequence generator backed by `Arc`-shared direction
/// numbers ($O(1)$ `Clone` across forks).
#[derive(Clone)]
pub struct SobolEngine {
    dimension: usize,
    dirs: Arc<[[u32; 32]]>,
    consumed: u64,
    shift: Option<Arc<[f64]>>,
    scramble_key: Key,
}

impl SobolEngine {
    /// Create a new Sobol engine.
    pub fn new(dimension: usize) -> Result<Self, ENNError> {
        if dimension == 0 || dimension > 21201 {
            return Err(ENNError::InvalidParameter(format!(
                "Sobol dimension must be in 1..=21201, got {}",
                dimension
            )));
        }
        let dirs: Arc<[[u32; 32]]> = (0..dimension).map(joe_kuo).collect();

        Ok(Self {
            dimension,
            dirs,
            consumed: 0,
            shift: None,
            scramble_key: shift_key(None),
        })
    }

    pub(crate) fn restore(
        dimension: usize,
        consumed: u64,
        shift: Option<Vec<f64>>,
    ) -> Result<Self, ENNError> {
        if dimension == 0 || dimension > 21201 {
            return Err(ENNError::InvalidParameter(format!(
                "Sobol dimension must be in 1..=21201, got {}",
                dimension
            )));
        }
        let work = consumed
            .checked_mul(dimension as u64)
            .ok_or_else(|| crate::snapshot_io::bad_state("Sobol cursor overflow"))?;
        if work > crate::snapshot_io::MAX_SOBOL {
            return Err(crate::snapshot_io::bad_state(
                "Sobol cursor exceeds safety limit",
            ));
        }
        if consumed > u32::MAX as u64 {
            return Err(crate::snapshot_io::bad_state("Sobol sequence exhausted"));
        }
        if let Some(ref s) = shift {
            if s.len() != dimension
                || s.iter()
                    .any(|&v| !v.is_finite() || !(0.0..=1.0).contains(&v))
            {
                return Err(crate::snapshot_io::bad_state(
                    "invalid Sobol scramble shift in frozen state",
                ));
            }
        }
        let dirs: Arc<[[u32; 32]]> = (0..dimension).map(joe_kuo).collect();
        let scramble_key = shift_key(shift.as_deref());
        Ok(Self {
            dimension,
            dirs,
            consumed,
            shift: shift.map(Arc::from),
            scramble_key,
        })
    }

    /// Advance the Sobol cursor by `count` candidates in $O(1)$ time and return
    /// a stateless [`SobolStep`] for [`DelayedPool`].
    pub fn advance(&mut self, count: usize) -> Result<SobolStep, ENNError> {
        let base_idx = u32::try_from(self.consumed)
            .map_err(|_| ENNError::InvalidParameter("Sobol sequence exhausted".to_string()))?;
        let next = self
            .consumed
            .checked_add(count as u64)
            .ok_or_else(|| ENNError::InvalidParameter("Sobol sequence exhausted".to_string()))?;
        if next > (u32::MAX as u64) + 1 {
            return Err(ENNError::InvalidParameter(
                "Sobol sequence exhausted".to_string(),
            ));
        }
        self.consumed = next;
        Ok(SobolStep {
            dirs: Arc::clone(&self.dirs),
            base_idx,
            key: self.scramble_key,
            scramble: self.shift.is_some(),
        })
    }

    /// Sample the next point from the Sobol sequence.
    ///
    /// If scrambled (via `scramble()`), applies random shift per dimension.
    pub fn sample<R: Rng + ?Sized>(&mut self, _rng: &mut R) -> Result<Vec<f64>, ENNError> {
        let idx = u32::try_from(self.consumed)
            .map_err(|_| ENNError::InvalidParameter("Sobol sequence exhausted".to_string()))?;
        self.consumed = self.consumed.saturating_add(1);
        let mut sample: Vec<f64> = (0..self.dimension)
            .map(|j| sobol_unit(idx, 0, j, Some(&self.dirs[j]), Key::new(0), false))
            .collect();
        if let Some(ref shift) = self.shift {
            baux::semantics::random::apply_shift(&mut sample, shift);
        }
        Ok(sample)
    }

    pub fn dimension(&self) -> usize {
        self.dimension
    }

    pub fn consumed(&self) -> u64 {
        self.consumed
    }

    pub fn shift(&self) -> Option<&[f64]> {
        self.shift.as_deref()
    }

    /// Scramble the sequence using random digital shift per dimension.
    pub fn scramble<R: RngCore + ?Sized>(&mut self, rng: &mut R) {
        let vec = baux::semantics::random::sobol_shift(self.dimension, rng);
        self.scramble_key = shift_key(Some(&vec));
        self.shift = Some(Arc::from(vec));
    }
}

/// Stateless pull-array for non-Cartesian coordinate systems (`Spherical`,
/// `Polar`, `OriginSphere`) and tied coordinate groups (`tied_dims`).
pub struct StructuredPool<'a> {
    key: Key,
    center: &'a [f64],
    lower: &'a [f64],
    upper: &'a [f64],
    units: Vec<Vec<usize>>,
    num_candidates: usize,
    sampler: CandidateSampler,
    coord_system: CoordSystem,
    has_tied: bool,
    is_morbo: bool,
    num_pert: usize,
    sobol: Option<&'a SobolStep>,
}

impl PullPool for StructuredPool<'_> {
    #[inline(always)]
    fn num_candidates(&self) -> usize {
        self.num_candidates
    }

    #[inline(always)]
    fn num_dim(&self) -> usize {
        self.center.len()
    }

    fn write_row(&self, cand: usize, out: &mut [f64]) {
        let dim = self.center.len().min(out.len());
        out[..dim].copy_from_slice(&self.center[..dim]);
        if dim == 0 {
            return;
        }
        let mut rng = baux::semantics::random::prng(cand_key(self.key, cand).seed());
        let raasp = self.sampler == CandidateSampler::RAASP;
        let masked_sobol = self.is_morbo && self.sampler == CandidateSampler::Sobol;
        let active = crate::candidates_fast::select_units(
            self.units.len(),
            raasp,
            masked_sobol,
            self.num_pert,
            &mut rng,
        )
        .unwrap_or_default();
        let sobol_fn = |d: usize| -> Option<f64> {
            match (self.sampler, self.sobol) {
                (CandidateSampler::Sobol, Some(step)) => Some(sobol_unit(
                    step.base_idx,
                    cand as u32,
                    d,
                    step.dirs.get(d),
                    step.key,
                    step.scramble,
                )),
                _ => None,
            }
        };
        let mut singletons = Vec::new();
        for u in active {
            if !self.has_tied || self.units[u].len() == 1 {
                singletons.extend_from_slice(&self.units[u]);
            } else {
                crate::candidates_fast::perturb_slice(
                    &mut out[..dim],
                    self.center,
                    self.lower,
                    self.upper,
                    &self.units[u],
                    self.coord_system,
                    sobol_fn,
                    &mut rng,
                );
            }
        }
        crate::candidates_fast::perturb_slice(
            &mut out[..dim],
            self.center,
            self.lower,
            self.upper,
            &singletons,
            self.coord_system,
            sobol_fn,
            &mut rng,
        );
    }
}

/// Zero-materialization host candidate pool unifying `@spine`'s `#![no_std]`
/// [`DelayedPool`] with structured coordinate-system/tied-group evaluation.
pub enum HostPool<'a> {
    Delayed(DelayedPool<'a>),
    Structured(StructuredPool<'a>),
}

impl PullPool for HostPool<'_> {
    #[inline(always)]
    fn num_candidates(&self) -> usize {
        match self {
            Self::Delayed(p) => PullPool::num_candidates(p),
            Self::Structured(p) => p.num_candidates(),
        }
    }

    #[inline(always)]
    fn num_dim(&self) -> usize {
        match self {
            Self::Delayed(p) => PullPool::num_dim(p),
            Self::Structured(p) => p.num_dim(),
        }
    }

    #[inline]
    fn write_row(&self, cand: usize, out: &mut [f64]) {
        match self {
            Self::Delayed(p) => PullPool::write_row(p, cand, out),
            Self::Structured(p) => p.write_row(cand, out),
        }
    }

    #[inline]
    fn sparse_center(&self) -> Option<&[f64]> {
        match self {
            Self::Delayed(p) => p.sparse_center(),
            Self::Structured(_) => None,
        }
    }

    #[inline]
    fn pack_deltas(
        &self,
        cand: usize,
        out_idx: &mut [usize],
        out_val: &mut [f64],
    ) -> Option<usize> {
        match self {
            Self::Delayed(p) => p.pack_deltas(cand, out_idx, out_val),
            Self::Structured(_) => None,
        }
    }
}

/// Construct a zero-materialization [`HostPool`] without allocating a
/// `(num_candidates, num_dim)` matrix.
#[allow(clippy::too_many_arguments)]
pub fn build_pool<'a>(
    key: Key,
    center: &'a [f64],
    lower: &'a [f64],
    upper: &'a [f64],
    num_candidates: usize,
    sampler: CandidateSampler,
    coord_system: CoordSystem,
    tied_dims: &[Vec<usize>],
    is_morbo: bool,
    num_pert: usize,
    sobol: Option<&'a SobolStep>,
) -> Result<HostPool<'a>, ENNError> {
    if coord_system == CoordSystem::Cartesian && tied_dims.is_empty() {
        let pool = match sampler {
            CandidateSampler::RAASP => DelayedPool::Raasp {
                key,
                center,
                lower,
                upper,
                num_candidates,
                num_pert: num_pert.max(1).min(center.len().max(1)),
            },
            CandidateSampler::Sobol => match sobol {
                Some(step) if is_morbo => DelayedPool::SobolMasked {
                    key: key.fold(step.key.0).fold(step.key.1),
                    center,
                    lower,
                    upper,
                    dirs: &step.dirs,
                    base_idx: step.base_idx,
                    num_candidates,
                    num_pert: num_pert.max(1).min(center.len().max(1)),
                    scramble: step.scramble,
                },
                Some(step) => DelayedPool::Sobol {
                    key: step.key,
                    lower,
                    upper,
                    dirs: &step.dirs,
                    base_idx: step.base_idx,
                    num_candidates,
                    scramble: step.scramble,
                },
                None => DelayedPool::Uniform {
                    key,
                    lower,
                    upper,
                    num_candidates,
                },
            },
            CandidateSampler::Uniform => DelayedPool::Uniform {
                key,
                lower,
                upper,
                num_candidates,
            },
        };
        return Ok(HostPool::Delayed(pool));
    }

    let units = crate::candidates_fast::build_units(center.len(), tied_dims)?;
    Ok(HostPool::Structured(StructuredPool {
        key,
        center,
        lower,
        upper,
        units,
        num_candidates,
        sampler,
        coord_system,
        has_tied: !tied_dims.is_empty(),
        is_morbo,
        num_pert,
        sobol,
    }))
}

/// Generate Latin Hypercube samples.
pub fn generate_lhd<R: Rng + ?Sized>(
    num_samples: usize,
    num_dim: usize,
    bounds: &ArrayView2<f64>,
    rng: &mut R,
) -> Array2<f64> {
    let pair_bounds: Vec<[f64; 2]> = (0..num_dim)
        .map(|j| [bounds[[j, 0]], bounds[[j, 1]]])
        .collect();
    let flat = baux::semantics::random::gen_lhd(num_samples, num_dim, &pair_bounds, rng);
    Array2::from_shape_vec((num_samples, num_dim), flat).expect("LHD shape matches")
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    #[test]
    fn test_tofromunit() {
        let bounds = array![[0.0, 10.0], [-5.0, 5.0]];
        let x = array![[2.5, 0.0], [7.5, 2.5]];

        let unit = to_unit(&x.view(), &bounds.view());
        let back = from_unit(&unit.view(), &bounds.view());

        // Check roundtrip (with tolerance)
        for i in 0..x.nrows() {
            for j in 0..x.ncols() {
                assert!((back[[i, j]] - x[[i, j]]).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn test_generateuniform() {
        let lower = array![0.0, 0.0];
        let upper = array![1.0, 1.0];
        let mut rng = StdRng::seed_from_u64(42);

        let candidates = generate_uniform(&lower, &upper, 10, &mut rng).unwrap();

        assert_eq!(candidates.nrows(), 10);
        assert_eq!(candidates.ncols(), 2);

        // Check bounds
        for i in 0..candidates.nrows() {
            for j in 0..candidates.ncols() {
                assert!(candidates[[i, j]] >= lower[j]);
                assert!(candidates[[i, j]] <= upper[j]);
            }
        }
    }

    #[test]
    fn test_generatelhd() {
        let bounds = array![[0.0, 1.0], [0.0, 1.0]];
        let mut rng = StdRng::seed_from_u64(42);

        let samples = generate_lhd(5, 2, &bounds.view(), &mut rng);

        assert_eq!(samples.nrows(), 5);
        assert_eq!(samples.ncols(), 2);

        // Check bounds
        for i in 0..samples.nrows() {
            for j in 0..samples.ncols() {
                assert!(samples[[i, j]] >= 0.0);
                assert!(samples[[i, j]] <= 1.0);
            }
        }
    }

    #[test]
    fn test_sobolengine() {
        let mut engine = SobolEngine::new(2).unwrap();
        let mut rng = StdRng::seed_from_u64(42);

        let sample1 = engine.sample(&mut rng).unwrap();
        let sample2 = engine.sample(&mut rng).unwrap();

        assert_eq!(sample1.len(), 2);
        assert_eq!(sample2.len(), 2);

        // Sobol samples should be in [0, 1)
        for v in &sample1 {
            assert!(*v >= 0.0 && *v < 1.0);
        }

        // Consecutive samples should be different
        assert_ne!(sample1, sample2);
    }

    #[test]
    fn test_005() {
        let mut rng = StdRng::seed_from_u64(9);
        let x_center = array![0.5, 0.5];
        let lower = array![0.1, 0.2];
        let upper = array![0.9, 0.8];
        let ls = array![1.0, 2.0];
        let compute = || (lower.clone(), upper.clone());

        let uniform = generate_candidates(
            compute,
            &x_center.view(),
            Some(&ls.view()),
            4,
            CandidateSampler::Uniform,
            &mut rng,
            None,
            2,
        )
        .unwrap();
        assert_eq!(uniform.nrows(), 4);

        let compute = || (lower.clone(), upper.clone());
        let mut sobol = SobolEngine::new(2).unwrap();
        let sobol_cand = generate_candidates(
            compute,
            &x_center.view(),
            Some(&ls.view()),
            4,
            CandidateSampler::Sobol,
            &mut rng,
            Some(&mut sobol),
            2,
        )
        .unwrap();
        assert_eq!(sobol_cand.nrows(), 4);

        let compute = || (lower.clone(), upper.clone());
        let sobol_fallback = generate_candidates(
            compute,
            &x_center.view(),
            Some(&ls.view()),
            4,
            CandidateSampler::Sobol,
            &mut rng,
            None,
            2,
        )
        .unwrap();
        assert_eq!(sobol_fallback.nrows(), 4);

        let compute = || (lower.clone(), upper.clone());
        let raasp = generate_candidates(
            compute,
            &x_center.view(),
            Some(&ls.view()),
            4,
            CandidateSampler::RAASP,
            &mut rng,
            None,
            2,
        )
        .unwrap();
        assert_eq!(raasp.nrows(), 4);
    }

    #[test]
    fn test_006() {
        assert!(SobolEngine::new(0).is_err());
        let mut rng = StdRng::seed_from_u64(3);
        let mut engine = SobolEngine::new(3).unwrap();
        let before = engine.sample(&mut rng).unwrap();
        engine.scramble(&mut rng);
        let after = engine.sample(&mut rng).unwrap();
        assert_eq!(before.len(), 3);
        assert_eq!(after.len(), 3);
    }

    #[test]
    fn test_007() {
        let mut rng = StdRng::seed_from_u64(21);
        let x_center = array![0.4, 0.6, 0.8];
        let lower = array![0.0, 0.0, 0.0];
        let upper = array![1.0, 1.0, 1.0];
        let ls_zero = array![0.0, 0.0, 0.0];
        let compute = || (lower.clone(), upper.clone());
        let out = generate_candidates(
            compute,
            &x_center.view(),
            Some(&ls_zero.view()),
            5,
            CandidateSampler::RAASP,
            &mut rng,
            None,
            0,
        )
        .unwrap();
        assert_eq!(out.nrows(), 5);
        assert_eq!(out.ncols(), 3);
        assert!(out.iter().all(|&v| (0.0..=1.0).contains(&v)));
    }

    #[test]
    fn test_008() {
        let mut rng = StdRng::seed_from_u64(5);
        let mut engine = SobolEngine::new(5).unwrap();
        let s = engine.sample(&mut rng).unwrap();
        assert_eq!(s.len(), 5);
        assert!(s.iter().all(|v| *v >= 0.0 && *v < 1.0));
    }

    #[test]
    fn raasp_seed() {
        let x_center = array![0.25, 0.75, 0.5];
        let lower = array![0.0, 0.0, 0.0];
        let upper = array![1.0, 1.0, 1.0];
        let ls = array![1.0, 2.0, 3.0];
        let run = || {
            generate_raasp(
                &x_center.view(),
                Some(&ls.view()),
                &lower,
                &upper,
                3,
                &mut StdRng::seed_from_u64(4242),
                2,
            )
            .unwrap()
        };
        let out = run();
        let out2 = run();
        assert_eq!(out.shape(), &[3, 3]);
        assert!(
            out.iter()
                .zip(out2.iter())
                .all(|(a, b)| (a - b).abs() < 1e-15)
        );
        assert!(out.iter().all(|&v| (0.0..=1.0).contains(&v)));
    }

    #[test]
    fn test_010() {
        let mut rng = StdRng::seed_from_u64(1234);
        let x_center = array![0.2, 0.4];
        let lower = array![0.0, 0.0];
        let upper = array![1.0, 1.0];
        let raasp = generate_raasp(&x_center.view(), None, &lower, &upper, 4, &mut rng, 1).unwrap();
        assert_eq!(raasp.shape(), &[4, 2]);
        let mut engine = SobolEngine::new(4).unwrap();
        let s0 = engine.sample(&mut rng).unwrap();
        let s1 = engine.sample(&mut rng).unwrap();
        assert_eq!(s0.len(), 4);
        assert_eq!(s1.len(), 4);
        assert_ne!(s0, s1);
    }

    #[test]
    fn test_011() {
        // Reference generated from scipy.stats.qmc.Sobol(d=5, scramble=False),
        // taking the first 8 points from random_base2(m=4).
        let expected = [
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.5, 0.5, 0.5, 0.5, 0.5],
            [0.75, 0.25, 0.25, 0.25, 0.75],
            [0.25, 0.75, 0.75, 0.75, 0.25],
            [0.375, 0.375, 0.625, 0.875, 0.375],
            [0.875, 0.875, 0.125, 0.375, 0.875],
            [0.625, 0.125, 0.875, 0.625, 0.625],
            [0.125, 0.625, 0.375, 0.125, 0.125],
        ];

        let mut rng = StdRng::seed_from_u64(1);
        let mut engine = SobolEngine::new(5).unwrap();
        let mut max_abs = 0.0f64;
        let mut sum_abs = 0.0f64;
        let mut count = 0usize;

        for row_expected in expected {
            let row = engine.sample(&mut rng).unwrap();
            for (got, exp) in row.iter().zip(row_expected.iter()) {
                let err = (got - exp).abs();
                if err > max_abs {
                    max_abs = err;
                }
                sum_abs += err;
                count += 1;
            }
        }

        let mean_abs = sum_abs / count as f64;
        println!("sobol_vs_scipy max_abs={max_abs:.3e} mean_abs={mean_abs:.3e}");
        assert!(max_abs <= 1e-12, "max_abs={max_abs}, mean_abs={mean_abs}");
    }

    #[test]
    fn test_dispatch() {
        // Test SobolEngine validation & exhaustion
        assert!(SobolEngine::new(0).is_err());
        assert!(SobolEngine::new(21202).is_err());
        let mut engine = SobolEngine::new(2).unwrap();
        assert_eq!(engine.dimension(), 2);

        // Deterministic masked Sobol trace
        let mut rng = StdRng::seed_from_u64(42);
        let x_center = array![0.5, 0.5];
        let lower = array![0.1, 0.2];
        let upper = array![0.9, 0.8];
        let masked = sobol_masked(
            &x_center.view(),
            &lower,
            &upper,
            3,
            &mut rng,
            &mut engine,
            1,
        )
        .unwrap();
        assert_eq!(masked.shape(), &[3, 2]);

        // Verify post-call RNG consumption
        let post_rng = rng.next_u64();
        println!("post_call_rng={post_rng}");
        assert_ne!(post_rng, 0);

        // Test fallback when num_pert=0 or k=0
        let mut rng2 = StdRng::seed_from_u64(100);
        let mut engine2 = SobolEngine::new(2).unwrap();
        let fallback = sobol_masked(
            &x_center.view(),
            &lower,
            &upper,
            2,
            &mut rng2,
            &mut engine2,
            0,
        )
        .unwrap();
        assert_eq!(fallback.shape(), &[2, 2]);

        // Strategy dispatch test
        let mut rng3 = StdRng::seed_from_u64(7);
        let mut engine3 = SobolEngine::new(2).unwrap();
        let turbo_morbo = draw_turbo(
            true,
            CandidateRV::Sobol,
            false,
            &x_center.view(),
            None,
            &lower,
            &upper,
            3,
            &mut rng3,
            Some(&mut engine3),
            1,
        )
        .unwrap();
        assert_eq!(turbo_morbo.shape(), &[3, 2]);

        let turbo_std = draw_turbo(
            false,
            CandidateRV::Uniform,
            false,
            &x_center.view(),
            None,
            &lower,
            &upper,
            3,
            &mut rng3,
            None,
            1,
        )
        .unwrap();
        assert_eq!(turbo_std.shape(), &[3, 2]);
    }

    #[test]
    fn aniso_sides() {
        let sides = crate::metric_weights::trust_sides(&[1.0, 4.0]);
        assert!((sides[0] - 2.0).abs() < 1e-12);
        assert!((sides[1] - 0.5).abs() < 1e-12);
        let prod: f64 = sides.iter().product();
        assert!((prod - 1.0).abs() < 1e-12);

        let x_center = array![0.5, 0.5, 0.5, 0.5];
        let lower = array![0.45, 0.1, 0.45, 0.1];
        let upper = array![0.55, 0.9, 0.55, 0.9];
        let ls = array![0.25, 2.0, 0.25, 2.0];
        let mut rng = StdRng::seed_from_u64(19);
        let cand = draw_turbo(
            false,
            CandidateRV::RAASP,
            false,
            &x_center.view(),
            Some(&ls),
            &lower,
            &upper,
            400,
            &mut rng,
            None,
            1,
        )
        .unwrap();
        let mut counts = [0usize; 4];
        for row in cand.rows() {
            for j in 0..4 {
                if (row[j] - x_center[j]).abs() > 1e-12 {
                    counts[j] += 1;
                }
            }
        }
        for c in counts {
            assert!(
                c > 60 && c < 140,
                "expected uniform RAASP rate, got {counts:?}"
            );
        }
    }

    #[test]
    fn coord_systems() {
        let x_center = array![0.6, 0.7, 0.4, 0.3];
        let lower = array![0.2, 0.3, 0.0, 0.0];
        let upper = array![1.0, 1.0, 0.8, 0.6];
        let mut rng = StdRng::seed_from_u64(77);

        let sph = perturb_candidates(
            &x_center.view(),
            &lower,
            &upper,
            16,
            CandidateSampler::Uniform,
            CoordSystem::Spherical,
            &[],
            false,
            false,
            &mut rng,
            None,
            4,
        )
        .unwrap();
        for row in sph.rows() {
            let mut norm = 0.0;
            for j in 0..4 {
                let half = 0.5 * (upper[j] - lower[j]);
                let u = (row[j] - x_center[j]) / half;
                norm += u * u;
            }
            assert!((norm.sqrt() - 1.0).abs() < 1e-6);
        }

        let pol = perturb_candidates(
            &x_center.view(),
            &lower,
            &upper,
            16,
            CandidateSampler::Uniform,
            CoordSystem::Polar,
            &[],
            false,
            false,
            &mut rng,
            None,
            4,
        )
        .unwrap();
        for row in pol.rows() {
            let mut norm = 0.0;
            for j in 0..4 {
                let half = 0.5 * (upper[j] - lower[j]);
                let u = (row[j] - x_center[j]) / half;
                norm += u * u;
            }
            assert!(norm.sqrt() <= 1.0 + 1e-6);
        }

        let orig = perturb_candidates(
            &x_center.view(),
            &array![0.0, 0.0, 0.0, 0.0],
            &array![1.0, 1.0, 1.0, 1.0],
            16,
            CandidateSampler::Uniform,
            CoordSystem::OriginSphere,
            &[],
            false,
            false,
            &mut rng,
            None,
            4,
        )
        .unwrap();
        let r0: f64 = x_center
            .iter()
            .map(|v| (v - 0.5).powi(2))
            .sum::<f64>()
            .sqrt();
        for row in orig.rows() {
            let r1: f64 = row.iter().map(|v| (v - 0.5).powi(2)).sum::<f64>().sqrt();
            assert!((r1 - r0).abs() < 1e-6);
        }
    }

    #[test]
    fn tied_units() {
        let x_center = array![0.5, 0.5, 0.5, 0.5];
        let lower = array![0.1, 0.1, 0.1, 0.1];
        let upper = array![0.9, 0.9, 0.9, 0.9];
        let groups = vec![vec![0, 1], vec![2, 3]];
        let mut rng = StdRng::seed_from_u64(99);
        let cand = perturb_candidates(
            &x_center.view(),
            &lower,
            &upper,
            32,
            CandidateSampler::RAASP,
            CoordSystem::Spherical,
            &groups,
            false,
            false,
            &mut rng,
            None,
            1,
        )
        .unwrap();
        for row in cand.rows() {
            let a0 = (row[0] - x_center[0]).abs() > 1e-12;
            let a1 = (row[1] - x_center[1]).abs() > 1e-12;
            let b0 = (row[2] - x_center[2]).abs() > 1e-12;
            let b1 = (row[3] - x_center[3]).abs() > 1e-12;
            assert_eq!(a0, a1);
            assert_eq!(b0, b1);
            assert_ne!(a0, b0);
        }
    }
}
