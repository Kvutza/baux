//! Candidate generation for trust region optimization.

use deser::{Deserialize, Serialize};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use rand::Rng;
use rand::RngCore;
use sobol::Sobol;
use sobol::params::JoeKuoD6;
use std::sync::OnceLock;

pub use crate::coordinate::{from_unit, to_unit};
use crate::error::ENNError;
pub use crate::proposal_kernel::CoordSystem;

fn sobol_params() -> &'static JoeKuoD6 {
    static PARAMS: OnceLock<JoeKuoD6> = OnceLock::new();
    PARAMS.get_or_init(JoeKuoD6::extended)
}

/// Sampling construction used to generate bounded candidates.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[deser(rename_all = "snake_case")]
pub enum CandidateSampler {
    /// Sobol quasi-random sequence.
    Sobol,
    /// Uniform random sampling.
    #[default]
    Uniform,
    /// RAASP (Random Axis-Aligned Subspace ProposalKernel).
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
/// Reference:
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

/// Sobol quasi-random sequence generator.
#[derive(Clone)]
pub struct SobolEngine {
    dimension: usize,
    sequence: Sobol<f64>,
    /// Random shift per dimension for scrambling (None = not scrambled).
    /// When set, each sample is transformed: (x_d + shift_d) % 1.0
    shift: Option<Vec<f64>>,
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
        let sequence = Sobol::<f64>::new(dimension, sobol_params());

        Ok(Self {
            dimension,
            sequence,
            shift: None,
        })
    }

    /// Sample the next point from the Sobol sequence.
    ///
    /// If scrambled (via `scramble()`), applies random shift per dimension.
    pub fn sample<R: Rng + ?Sized>(&mut self, _rng: &mut R) -> Result<Vec<f64>, ENNError> {
        let mut sample = self
            .sequence
            .next()
            .ok_or_else(|| ENNError::InvalidParameter("Sobol sequence exhausted".to_string()))?;
        if sample.len() != self.dimension {
            return Err(ENNError::InvalidParameter(format!(
                "Sobol sample dimension mismatch: expected {}, got {}",
                self.dimension,
                sample.len()
            )));
        }
        if let Some(ref shift) = self.shift {
            baux::semantics::random::apply_shift(&mut sample, shift);
        }
        Ok(sample)
    }

    pub fn dimension(&self) -> usize {
        self.dimension
    }

    /// Scramble the sequence using random digital shift per dimension.
    ///
    /// Closer to Owen-style scrambling than a simple offset skip: each
    /// dimension gets a random shift in [0,1]; samples are (x + shift) mod
    /// 1. Preserves low-discrepancy structure while randomizing.
    pub fn scramble<R: RngCore + ?Sized>(&mut self, rng: &mut R) {
        self.shift = Some(baux::semantics::random::sobol_shift(self.dimension, rng));
    }
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
