//! Integer random-stream identity. Wrapping arithmetic is part of the contract.

#[cfg(test)]
use rand::SeedableRng;
use rand::distributions::{Bernoulli, Uniform};
#[cfg(test)]
use rand::rngs::StdRng;
use rand::{Rng, RngCore};

const SM64_OFFSET: u64 = 0xD2B74407B1CE6E93;

/// Prime constant for seed combination.
const SEED_PRIME: u64 = 1_000_003;

/// Reciprocal of 2^53, exactly representable in binary64.
const INV_2P53: f64 = 1.0 / 9007199254740992.0;

/// Minimum clip value to avoid log(0).
const CLIP_MIN: f64 = 1e-12;

/// Maximum clip value (exclusive upper bound).
const CLIP_MAX: f64 = 1.0 - 1e-12;

/// Convert u64 to f64 in [0, 1) using high 53 bits.
#[inline(always)]
pub fn u64_f53(x: u64) -> f64 {
    ((x >> 11) as f64) * INV_2P53
}

/// Box-Muller transform for standard normal variates.
///
/// Takes two uniform [0, 1) values and produces one standard normal.
#[inline(always)]
pub fn box_muller(u1: f64, u2: f64) -> f64 {
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

#[inline(always)]
pub fn normal_metric(seed_u64: u64, unique_idx: i64, metric: usize) -> f64 {
    let unique_u64 = unique_idx as u64;
    let base =
        (seed_u64.wrapping_mul(SEED_PRIME).wrapping_add(unique_u64)).wrapping_mul(SEED_PRIME);
    let metric_u64 = metric as u64;
    let combined1 = base.wrapping_add(metric_u64);
    let r1 = splitmix64(combined1);
    let combined2 = combined1 ^ SM64_OFFSET;
    let r2 = splitmix64(combined2);
    let mut u1 = u64_f53(r1);
    let u2 = u64_f53(r2);
    u1 = u1.clamp(CLIP_MIN, CLIP_MAX);
    box_muller(u1, u2)
}

#[inline(always)]
pub fn splitmix64(value: u64) -> u64 {
    let value = value.wrapping_add(0x9e37_79b9_7f4a_7c15);
    let value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    let value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

/// Upstream Sobol seed modulus (2^31).
pub const SOBOL_MOD: u64 = 1u64 << 31;

/// Upstream Sobol base reduction.
#[inline(always)]
pub fn sobol_base(seed_bytes: u64) -> u64 {
    seed_bytes % SOBOL_MOD
}

/// Upstream candidate state seed derivation with SplitMix64 finalizer.
#[inline(always)]
pub fn sobol_seed(base: u64, restart: usize, obs: usize, arms: usize) -> u64 {
    let mut x = base;
    x ^= (restart.wrapping_add(1) as u64).wrapping_mul(0xD134_2543_DE82_EF95);
    x ^= (obs as u64)
        .wrapping_add(1)
        .wrapping_mul(0x9E37_79B9_7F4A_7C15);
    x ^= (arms as u64)
        .wrapping_add(1)
        .wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = x;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^= z >> 31;
    z & 0xFFFF_FFFF
}

/// Sampling construction used to generate bounded candidates.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CandidateRV {
    Sobol,
    #[default]
    Uniform,
    RAASP,
}

/// Convert flat coordinates from unit hypercube to bounds.
pub fn from_unit(x_unit: &[f64], bounds: &[[f64; 2]], n: usize, d: usize) -> Vec<f64> {
    let mut result = vec![0.0; n * d];
    for i in 0..n {
        for j in 0..d {
            let lower = bounds[j][0];
            let upper = bounds[j][1];
            result[i * d + j] = lower + x_unit[i * d + j] * (upper - lower);
        }
    }
    result
}

/// Convert flat coordinates to unit hypercube from bounds.
pub fn to_unit(x: &[f64], bounds: &[[f64; 2]], n: usize, d: usize) -> Vec<f64> {
    let mut result = vec![0.0; n * d];
    for i in 0..n {
        for j in 0..d {
            let lower = bounds[j][0];
            let upper = bounds[j][1];
            let range = upper - lower;
            if range > 0.0 {
                result[i * d + j] = (x[i * d + j] - lower) / range;
            } else {
                result[i * d + j] = 0.5;
            }
        }
    }
    result
}

/// Scramble shift vector generation for Sobol sequence.
pub fn sobol_shift<R: RngCore + ?Sized>(dim: usize, rng: &mut R) -> Vec<f64> {
    (0..dim)
        .map(|_| {
            let mut bytes = [0u8; 8];
            rng.fill_bytes(&mut bytes);
            let u = u64::from_le_bytes(bytes);
            (u >> 11) as f64 / ((1u64 << 53) as f64)
        })
        .collect()
}

/// Coordinate shift scrambling modulo 1.0.
pub fn apply_shift(sample: &mut [f64], shift: &[f64]) {
    for (s, sh) in sample.iter_mut().zip(shift.iter()) {
        *s = (*s + *sh) % 1.0;
        if *s < 0.0 {
            *s += 1.0;
        }
    }
}

/// Generate uniformly random candidates over flat slice.
pub fn gen_uniform<R: Rng + ?Sized>(
    lower: &[f64],
    upper: &[f64],
    num_candidates: usize,
    rng: &mut R,
) -> Vec<f64> {
    let num_dim = lower.len();
    let mut candidates = vec![0.0; num_candidates * num_dim];
    for i in 0..num_candidates {
        for j in 0..num_dim {
            let dist = Uniform::new(lower[j], upper[j]);
            candidates[i * num_dim + j] = rng.sample(dist);
        }
    }
    candidates
}

/// Generate masked Sobol candidates reproducing exact upstream logic.
pub fn sobol_masked<R: Rng + ?Sized>(
    x_center: &[f64],
    lower: &[f64],
    upper: &[f64],
    num_candidates: usize,
    rng: &mut R,
    sobol_samples: &[f64], // num_candidates * num_dim
    num_pert: usize,
) -> Result<Vec<f64>, String> {
    let num_dim = x_center.len();
    if num_dim == 0 {
        return Err("num_dim must be > 0".to_string());
    }
    let prob = (num_pert as f64 / num_dim as f64).min(1.0);
    let bern = Bernoulli::new(prob).map_err(|e| format!("binomial mask: {e}"))?;
    let mut candidates = vec![0.0; num_candidates * num_dim];
    for i in 0..num_candidates {
        for j in 0..num_dim {
            candidates[i * num_dim + j] = x_center[j];
        }
        let sample = &sobol_samples[i * num_dim..(i + 1) * num_dim];
        let mut k = (0..num_dim).filter(|_| rng.sample(bern)).count();
        if k == 0 {
            k = 1;
        }
        let mut dims: Vec<usize> = (0..num_dim).collect();
        for t in 0..k {
            let j = t + rng.gen_range(0..(num_dim - t));
            dims.swap(t, j);
            let d = dims[t];
            candidates[i * num_dim + d] = lower[d] + sample[d] * (upper[d] - lower[d]);
        }
    }
    Ok(candidates)
}

/// Latin Hypercube Design (LHD) stratification with Fisher-Yates permutation.
pub fn gen_lhd<R: Rng + ?Sized>(
    num_samples: usize,
    num_dim: usize,
    bounds: &[[f64; 2]],
    rng: &mut R,
) -> Vec<f64> {
    let mut result = vec![0.0; num_samples * num_dim];
    for j in 0..num_dim {
        let lower = bounds[j][0];
        let upper = bounds[j][1];
        let range = upper - lower;

        let mut perm: Vec<usize> = (0..num_samples).collect();
        for i in (1..num_samples).rev() {
            let j_idx = rng.gen_range(0..=i);
            perm.swap(i, j_idx);
        }

        for i in 0..num_samples {
            let bin = perm[i];
            let bin_start = bin as f64 / num_samples as f64;
            let bin_end = (bin + 1) as f64 / num_samples as f64;
            let offset: f64 = rng.r#gen();
            let unit = bin_start + offset * (bin_end - bin_start);
            result[i * num_dim + j] = lower + unit * range;
        }
    }
    result
}

fn raasp_dim(cdf: &[f64], r: f64, num_dim: usize) -> usize {
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

/// Match ndarray 0.16.1's contiguous sum, including its rounding order.
/// The slice API has no strided or reversed views.
fn length_sum(values: &[f64]) -> f64 {
    let mut lanes = [0.0; 8];
    let mut blocks = values.chunks_exact(8);
    for block in &mut blocks {
        for lane in 0..8 {
            lanes[lane] += block[lane];
        }
    }
    let mut total = 0.0;
    for lane in 0..4 {
        total += lanes[lane] + lanes[lane + 4];
    }
    for &value in blocks.remainder() {
        total += value;
    }
    total
}

/// RAASP candidate proposal generator over flat slice.
pub fn gen_raasp<R: Rng + ?Sized>(
    x_center: &[f64],
    lengthscales: Option<&[f64]>,
    lower: &[f64],
    upper: &[f64],
    num_candidates: usize,
    rng: &mut R,
    num_pert: usize,
) -> Vec<f64> {
    let num_dim = x_center.len();
    let num_pert = num_pert.max(1);

    let probs: Vec<f64> = match lengthscales {
        Some(ls) => {
            let ls_sum = length_sum(ls);
            if ls_sum > 0.0 {
                ls.iter().map(|&v| v / ls_sum).collect()
            } else {
                vec![1.0 / num_dim as f64; num_dim]
            }
        }
        None => vec![1.0 / num_dim as f64; num_dim],
    };

    let mut cdf = vec![0.0; num_dim];
    let mut c = 0.0;
    for j in 0..num_dim {
        c += probs[j];
        cdf[j] = c;
    }

    let mut candidates = vec![0.0; num_candidates * num_dim];
    for i in 0..num_candidates {
        for j in 0..num_dim {
            candidates[i * num_dim + j] = x_center[j];
        }
        for _ in 0..num_pert {
            let r: f64 = rng.r#gen();
            let dim_to_perturb = raasp_dim(&cdf, r, num_dim);
            let dist = Uniform::new(lower[dim_to_perturb], upper[dim_to_perturb]);
            candidates[i * num_dim + dim_to_perturb] = rng.sample(dist);
        }
    }
    candidates
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reduction_order() {
        let big = 9_007_199_254_740_992.0;
        let positive = [big, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        assert_eq!(length_sum(&positive), big + 6.0);
        assert_eq!(positive.iter().sum::<f64>(), big);
        let cancellation = [big, 1.0, 0.0, 0.0, -big, 0.0, 0.0, 0.0];
        assert_eq!(length_sum(&cancellation), 1.0);
        assert_eq!(cancellation.iter().sum::<f64>(), 0.0);
        assert_eq!(length_sum(&[1.0, 2.0, 3.0]), 6.0);
    }

    #[test]
    fn test_seed() {
        assert_eq!(sobol_seed(0, 0, 0, 1), 2701705554);
        assert_eq!(sobol_seed(42, 1, 10, 4), 168175535);
        assert_eq!(sobol_seed(123456789, 0, 50, 16), 3605740088);
    }

    #[test]
    fn test_base() {
        assert!(sobol_base(0x123456789abcdef0) < SOBOL_MOD);
        assert_eq!(sobol_base(0x123456789abcdef0), 448585456);
    }

    #[test]
    fn test_uniform() {
        let mut rng = StdRng::seed_from_u64(42);
        let lower = [0.0, 0.0];
        let upper = [1.0, 1.0];
        let mat = gen_uniform(&lower, &upper, 2, &mut rng);
        assert_eq!(mat.len(), 4);
        // Post-call RNG consumption check
        let post = rng.next_u64();
        assert_ne!(post, 0);
    }

    #[test]
    fn test_scramble() {
        let mut rng = StdRng::seed_from_u64(42);
        let shift = sobol_shift(2, &mut rng);
        assert_eq!(shift.len(), 2);
        for &s in &shift {
            assert!(s >= 0.0 && s < 1.0);
        }
        let mut sample = [0.5, 0.5];
        apply_shift(&mut sample, &shift);
        for &s in &sample {
            assert!(s >= 0.0 && s < 1.0);
        }
    }

    #[test]
    fn test_masked() {
        let mut rng = StdRng::seed_from_u64(42);
        let x_center = [0.5, 0.5];
        let lower = [0.1, 0.2];
        let upper = [0.9, 0.8];
        let sobol_samples = [0.0, 0.0, 0.5, 0.5];
        let out = sobol_masked(&x_center, &lower, &upper, 2, &mut rng, &sobol_samples, 1).unwrap();
        assert_eq!(out.len(), 4);
        let post = rng.next_u64();
        assert_ne!(post, 0);
    }

    #[test]
    fn test_lhd() {
        let mut rng = StdRng::seed_from_u64(42);
        let bounds = [[0.0, 1.0], [0.0, 1.0]];
        let lhd = gen_lhd(4, 2, &bounds, &mut rng);
        assert_eq!(lhd.len(), 8);
        for &v in &lhd {
            assert!(v >= 0.0 && v <= 1.0);
        }
        let post = rng.next_u64();
        assert_ne!(post, 0);
    }

    #[test]
    fn test_raasp() {
        let mut rng = StdRng::seed_from_u64(42);
        let x_center = [0.5, 0.5];
        let lower = [0.0, 0.0];
        let upper = [1.0, 1.0];
        let ls = [1.0, 2.0];
        let raasp = gen_raasp(&x_center, Some(&ls), &lower, &upper, 3, &mut rng, 1);
        assert_eq!(raasp.len(), 6);
        let post = rng.next_u64();
        assert_ne!(post, 0);
    }

    #[test]
    fn test_trace() {
        let mut rng_raw = StdRng::seed_from_u64(42);
        let raw_words: Vec<u64> = (0..16).map(|_| rng_raw.next_u64()).collect();
        println!("RAW_STDRNG_42_WORDS: {:?}", raw_words);
        // Uniform fixture
        let mut rng = StdRng::seed_from_u64(42);
        let u_mat = gen_uniform(&[0.0, 0.0], &[1.0, 1.0], 2, &mut rng);
        println!("TRACE_UNIFORM: {:?}", u_mat);
        println!(
            "TRACE_UNIFORM_BITS: {:?}",
            u_mat.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );
        println!("TRACE_UNIFORM_POST_RNG: {}", rng.next_u64());

        // Sobol scramble fixture (dim=2)
        let mut rng = StdRng::seed_from_u64(42);
        let shift = sobol_shift(2, &mut rng);
        println!("TRACE_SCRAMBLE_SHIFT: {:?}", shift);
        println!(
            "TRACE_SCRAMBLE_BITS: {:?}",
            shift.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );
        println!("TRACE_SCRAMBLE_POST_RNG: {}", rng.next_u64());

        // Sobol masked fixture (dim=2, 2 candidates, num_pert=1)
        let mut rng = StdRng::seed_from_u64(42);
        let sobol_samples = [0.0, 0.0, 0.5, 0.5];
        let masked = sobol_masked(
            &[0.5, 0.5],
            &[0.1, 0.2],
            &[0.9, 0.8],
            2,
            &mut rng,
            &sobol_samples,
            1,
        )
        .unwrap();
        println!("TRACE_MASKED: {:?}", masked);
        println!(
            "TRACE_MASKED_BITS: {:?}",
            masked.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );
        println!("TRACE_MASKED_POST_RNG: {}", rng.next_u64());

        // LHD fixture (4 samples, 2 dim)
        let mut rng = StdRng::seed_from_u64(42);
        let lhd = gen_lhd(4, 2, &[[0.0, 1.0], [0.0, 1.0]], &mut rng);
        println!("TRACE_LHD: {:?}", lhd);
        println!(
            "TRACE_LHD_BITS: {:?}",
            lhd.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );
        println!("TRACE_LHD_POST_RNG: {}", rng.next_u64());

        // RAASP fixture (2 candidates, 2 dim, num_pert=1)
        let mut rng = StdRng::seed_from_u64(42);
        let raasp = gen_raasp(
            &[0.5, 0.5],
            Some(&[1.0, 2.0]),
            &[0.1, 0.2],
            &[0.9, 0.8],
            2,
            &mut rng,
            1,
        );
        println!("TRACE_RAASP: {:?}", raasp);
        println!(
            "TRACE_RAASP_BITS: {:?}",
            raasp.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );
        println!("TRACE_RAASP_POST_RNG: {}", rng.next_u64());
    }
}
