//! Index-pure `#![no_std]` pull-array (`DelayedPool`) for deterministic
//! stateless candidate evaluation across CPU SIMD, Apple Metal, and
//! `cuda-oxide`, parameterized over generic [`Real`] scalars (`f32` and `f64`).
//!
//! Zero `Vec`, zero `&mut RngCore`, and zero `&mut SobolEngine`: every
//! coordinate `(c, j)` is a pure function of `(Key, RoundPlan, c, j)`.

use super::key::Key;
use super::random::splitmix64;
use super::real::Real;

/// Domain-separation tag for per-candidate key derivation.
pub const TAG_CAND: u64 = 0x4341_4e44_5f4b_4559;
/// Domain-separation tag for forced RAASP perturbation coordinate.
pub const TAG_FORCED: u64 = 0x464f_5243_4544_5f44;
/// Domain-separation tag for Bernoulli coordinate perturbation mask.
pub const TAG_MASK: u64 = 0x5241_4153_504d_534b;
/// Domain-separation tag for coordinate value draw.
pub const TAG_VALUE: u64 = 0x5241_4153_5056_414c;
/// Domain-separation tag for Sobol Owen scramble seed.
pub const TAG_SOBOL: u64 = 0x534f_424f_4c4f_574e;
/// Domain-separation tag for Latin Hypercube Feistel permutation and jitter.
pub const TAG_LHD: u64 = 0x4c48_4453_5452_4154;
/// Domain-separation tag for isotropic/scaled noise draws.
pub const TAG_NOISE: u64 = 0x4e4f_4953_4544_5257;

/// Joe & Kuo (2010) D6 primitive polynomials `(s, a, m_init)` for dimensions
/// `2..=64` (0-based coordinate indices `1..=63`).
const JOE_KUO: [(u8, u16, [u16; 9]); 63] = [
    (1, 0, [1, 0, 0, 0, 0, 0, 0, 0, 0]),
    (2, 1, [1, 3, 0, 0, 0, 0, 0, 0, 0]),
    (3, 1, [1, 3, 1, 0, 0, 0, 0, 0, 0]),
    (3, 2, [1, 1, 1, 0, 0, 0, 0, 0, 0]),
    (4, 1, [1, 1, 3, 3, 0, 0, 0, 0, 0]),
    (4, 4, [1, 3, 5, 13, 0, 0, 0, 0, 0]),
    (5, 2, [1, 1, 5, 5, 17, 0, 0, 0, 0]),
    (5, 4, [1, 1, 5, 5, 5, 0, 0, 0, 0]),
    (5, 7, [1, 1, 7, 11, 19, 0, 0, 0, 0]),
    (5, 11, [1, 1, 5, 1, 1, 0, 0, 0, 0]),
    (5, 13, [1, 1, 1, 3, 11, 0, 0, 0, 0]),
    (5, 14, [1, 3, 5, 5, 31, 0, 0, 0, 0]),
    (6, 1, [1, 3, 3, 9, 7, 49, 0, 0, 0]),
    (6, 13, [1, 1, 1, 15, 21, 21, 0, 0, 0]),
    (6, 16, [1, 3, 1, 13, 27, 49, 0, 0, 0]),
    (6, 19, [1, 1, 1, 15, 7, 5, 0, 0, 0]),
    (6, 22, [1, 3, 1, 15, 13, 25, 0, 0, 0]),
    (6, 25, [1, 1, 5, 5, 19, 61, 0, 0, 0]),
    (7, 1, [1, 3, 7, 11, 23, 15, 103, 0, 0]),
    (7, 4, [1, 3, 7, 13, 13, 15, 69, 0, 0]),
    (7, 7, [1, 1, 3, 13, 7, 35, 63, 0, 0]),
    (7, 8, [1, 3, 5, 9, 1, 25, 53, 0, 0]),
    (7, 14, [1, 3, 1, 13, 9, 35, 107, 0, 0]),
    (7, 19, [1, 3, 1, 5, 27, 61, 31, 0, 0]),
    (7, 21, [1, 1, 5, 11, 19, 41, 61, 0, 0]),
    (7, 28, [1, 3, 5, 3, 3, 13, 69, 0, 0]),
    (7, 31, [1, 1, 7, 13, 1, 19, 1, 0, 0]),
    (7, 32, [1, 3, 7, 5, 13, 19, 59, 0, 0]),
    (7, 37, [1, 1, 3, 9, 25, 29, 41, 0, 0]),
    (7, 41, [1, 3, 5, 13, 23, 1, 55, 0, 0]),
    (7, 42, [1, 3, 7, 3, 13, 59, 17, 0, 0]),
    (7, 50, [1, 3, 1, 3, 5, 53, 69, 0, 0]),
    (7, 55, [1, 1, 5, 5, 23, 33, 13, 0, 0]),
    (7, 56, [1, 1, 7, 7, 1, 61, 123, 0, 0]),
    (7, 59, [1, 1, 7, 9, 13, 61, 49, 0, 0]),
    (7, 62, [1, 3, 3, 5, 3, 55, 33, 0, 0]),
    (8, 14, [1, 3, 1, 15, 31, 13, 49, 245, 0]),
    (8, 21, [1, 3, 5, 15, 31, 59, 63, 97, 0]),
    (8, 22, [1, 3, 1, 11, 11, 11, 77, 249, 0]),
    (8, 38, [1, 3, 1, 11, 27, 43, 71, 9, 0]),
    (8, 47, [1, 1, 7, 15, 21, 11, 81, 45, 0]),
    (8, 49, [1, 3, 7, 3, 25, 31, 65, 79, 0]),
    (8, 50, [1, 3, 1, 1, 19, 11, 3, 205, 0]),
    (8, 52, [1, 1, 5, 9, 19, 21, 29, 157, 0]),
    (8, 56, [1, 3, 7, 11, 1, 33, 89, 185, 0]),
    (8, 67, [1, 3, 3, 3, 15, 9, 79, 71, 0]),
    (8, 70, [1, 3, 7, 11, 15, 39, 119, 27, 0]),
    (8, 84, [1, 1, 3, 1, 11, 31, 97, 225, 0]),
    (8, 97, [1, 1, 1, 3, 23, 43, 57, 177, 0]),
    (8, 103, [1, 3, 7, 7, 17, 17, 37, 71, 0]),
    (8, 115, [1, 3, 1, 5, 27, 63, 123, 213, 0]),
    (8, 122, [1, 1, 3, 5, 11, 43, 53, 133, 0]),
    (9, 8, [1, 3, 5, 5, 29, 17, 47, 173, 479]),
    (9, 13, [1, 3, 3, 11, 3, 1, 109, 9, 69]),
    (9, 16, [1, 1, 1, 5, 17, 39, 23, 5, 343]),
    (9, 22, [1, 3, 1, 5, 25, 15, 31, 103, 499]),
    (9, 25, [1, 1, 1, 11, 11, 17, 63, 105, 183]),
    (9, 44, [1, 1, 5, 11, 9, 29, 97, 231, 363]),
    (9, 47, [1, 1, 5, 15, 19, 45, 41, 7, 383]),
    (9, 52, [1, 3, 7, 7, 31, 19, 83, 137, 221]),
    (9, 55, [1, 1, 1, 3, 23, 15, 111, 223, 83]),
    (9, 59, [1, 1, 5, 13, 31, 15, 55, 25, 161]),
    (9, 62, [1, 1, 3, 13, 25, 47, 39, 87, 257]),
];

/// Derive a deterministic per-candidate [`Key`] from a round [`Key`].
#[inline(always)]
pub const fn cand_key(key: Key, cand: usize) -> Key {
    key.fold(TAG_CAND).fold(cand as u64)
}

/// Deterministic forced-perturbation dimension in `0..dim` so every RAASP or
/// masked-Sobol candidate perturbs at least one coordinate.
#[inline(always)]
pub const fn forced_dim(key_c: Key, dim: usize) -> usize {
    key_c.fold(TAG_FORCED).bounded_idx(dim)
}

/// Stateless coordinate uniform hash in `[0, 1)` over [`Real`].
#[inline(always)]
pub fn hash_unit<T: Real>(key_c: Key, j: usize, tag: u64) -> T {
    key_c.fold(tag).fold(j as u64).uniform::<T>()
}

#[inline(always)]
fn lerp_box<T: Real>(lo: T, hi: T, u: T) -> T {
    if hi > lo {
        (lo + u * (hi - lo)).clamp(lo, hi)
    } else {
        lo
    }
}

/// Return `true` iff coordinate `(c, j)` is perturbed under stateless RAASP.
#[inline(always)]
pub fn raasp_perturbs<T: Real>(key_c: Key, j: usize, dim: usize, prob: T) -> bool {
    dim > 0
        && (prob >= T::ONE
            || j == forced_dim(key_c, dim)
            || hash_unit::<T>(key_c, j, TAG_MASK) < prob)
}

/// Perturbation probability `clamp(num_pert / dim, 0, 1)` over [`Real`].
#[inline(always)]
pub fn raasp_prob<T: Real>(num_pert: usize, dim: usize) -> T {
    if dim == 0 {
        T::ZERO
    } else {
        (T::from_idx(num_pert.max(1)) / T::from_idx(dim)).min(T::ONE)
    }
}

/// Stateless RAASP pull evaluation at `(cand, j)`.
#[inline(always)]
pub fn eval_raasp<T: Real>(
    key: Key,
    center: &[T],
    lower: &[T],
    upper: &[T],
    num_pert: usize,
    cand: usize,
    j: usize,
) -> T {
    let dim = center.len();
    let key_c = cand_key(key, cand);
    let prob = raasp_prob::<T>(num_pert, dim);
    if raasp_perturbs(key_c, j, dim, prob) {
        lerp_box(lower[j], upper[j], hash_unit::<T>(key_c, j, TAG_VALUE))
    } else {
        center[j]
    }
}

/// Zero-allocation iterator yielding `(dim_idx, new_value)` for perturbed
/// coordinates of a single RAASP candidate.
#[derive(Clone, Copy, Debug)]
pub struct RaaspDeltas<'a, T: Real = f64> {
    key_c: Key,
    lower: &'a [T],
    upper: &'a [T],
    dim: usize,
    forced: usize,
    prob: T,
    cursor: usize,
}

impl<'a, T: Real> RaaspDeltas<'a, T> {
    #[inline(always)]
    pub fn new(key: Key, cand: usize, lower: &'a [T], upper: &'a [T], num_pert: usize) -> Self {
        let dim = lower.len().min(upper.len());
        let key_c = cand_key(key, cand);
        Self {
            key_c,
            lower,
            upper,
            dim,
            forced: forced_dim(key_c, dim),
            prob: raasp_prob::<T>(num_pert, dim),
            cursor: 0,
        }
    }
}

impl<T: Real> Iterator for RaaspDeltas<'_, T> {
    type Item = (usize, T);

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        while self.cursor < self.dim {
            let j = self.cursor;
            self.cursor += 1;
            if self.prob >= T::ONE
                || j == self.forced
                || hash_unit::<T>(self.key_c, j, TAG_MASK) < self.prob
            {
                let val = lerp_box(
                    self.lower[j],
                    self.upper[j],
                    hash_unit::<T>(self.key_c, j, TAG_VALUE),
                );
                return Some((j, val));
            }
        }
        None
    }
}

/// Pack perturbed `(index, value)` pairs for candidate `cand` into caller
/// slices. Returns `Some(k_pert)` when all perturbations fit in `out_idx`, or
/// `None` if `k_pert > out_idx.len()` (caller falls back to dense evaluation).
#[inline]
pub fn raasp_pack<T: Real>(
    key: Key,
    cand: usize,
    lower: &[T],
    upper: &[T],
    num_pert: usize,
    out_idx: &mut [usize],
    out_val: &mut [T],
) -> Option<usize> {
    let cap = out_idx.len().min(out_val.len());
    let mut k = 0usize;
    for (j, val) in RaaspDeltas::new(key, cand, lower, upper, num_pert) {
        if k >= cap {
            return None;
        }
        out_idx[k] = j;
        out_val[k] = val;
        k += 1;
    }
    Some(k)
}

/// Update squared weighted Euclidean distance from `base_dist = ||center -
/// target||_w^2` in `O(k_pert)` operations using packed sparse deltas.
#[inline]
pub fn sparse_dist<T: Real>(
    base_dist: T,
    center: &[T],
    target: &[T],
    weights: Option<&[T]>,
    pert_idx: &[usize],
    pert_val: &[T],
) -> T {
    let mut acc = base_dist;
    let k = pert_idx.len().min(pert_val.len());
    match weights {
        Some(w) => {
            for t in 0..k {
                let j = pert_idx[t];
                let d_new = pert_val[t] - target[j];
                let d_old = center[j] - target[j];
                acc += w[j] * (d_new * d_new - d_old * d_old);
            }
        }
        None => {
            for t in 0..k {
                let j = pert_idx[t];
                let d_new = pert_val[t] - target[j];
                let d_old = center[j] - target[j];
                acc += d_new * d_new - d_old * d_old;
            }
        }
    }
    acc.max(T::ZERO)
}

/// Binary reflected Gray code `n ^ (n >> 1)`.
#[inline(always)]
pub const fn gray_code(index: u32) -> u32 {
    index ^ (index >> 1)
}

/// Dimension-0 Sobol direction vector `v_b = 1 << (31 - b)`.
#[inline(always)]
pub const fn base_dirs() -> [u32; 32] {
    let mut dirs = [0u32; 32];
    let mut b = 0usize;
    while b < 32 {
        dirs[b] = 1u32 << (31 - b);
        b += 1;
    }
    dirs
}

/// Compute 32-bit Sobol direction numbers from primitive polynomial `(s, a,
/// m_init)` matching Joe & Kuo (2010) and `scipy.stats.qmc.Sobol`.
pub const fn sobol_dirs(s: usize, a: u32, m_init: &[u32]) -> [u32; 32] {
    if s == 0 || m_init.is_empty() {
        return base_dirs();
    }
    let degree = if s <= 32 { s } else { 32 };
    let mut dirs = [0u32; 32];
    let mut i = 1usize;
    while i <= degree {
        let m = if i - 1 < m_init.len() {
            m_init[i - 1]
        } else {
            1
        };
        dirs[i - 1] = m << (32 - i);
        i += 1;
    }
    while i <= 32 {
        let mut next = dirs[i - degree - 1] ^ (dirs[i - degree - 1] >> degree);
        let mut k = 1usize;
        while k < degree {
            let coeff = (a >> (degree - k - 1)) & 1;
            if coeff != 0 {
                next ^= dirs[i - k - 1];
            }
            k += 1;
        }
        dirs[i - 1] = next;
        i += 1;
    }
    dirs
}

/// Built-in `#![no_std]` Joe & Kuo D6 direction vector for coordinate
/// `dim_idx`.
pub const fn joe_kuo(dim_idx: usize) -> [u32; 32] {
    if dim_idx == 0 {
        return base_dirs();
    }
    if dim_idx <= JOE_KUO.len() {
        let (s, a, m_arr) = JOE_KUO[dim_idx - 1];
        let m32 = [
            m_arr[0] as u32,
            m_arr[1] as u32,
            m_arr[2] as u32,
            m_arr[3] as u32,
            m_arr[4] as u32,
            m_arr[5] as u32,
            m_arr[6] as u32,
            m_arr[7] as u32,
            m_arr[8] as u32,
        ];
        return sobol_dirs(s as usize, a as u32, &m32);
    }
    let (_, a, base_m) = JOE_KUO[(dim_idx - 1) % JOE_KUO.len()];
    let s = JOE_KUO[(dim_idx - 1) % JOE_KUO.len()].0 as usize;
    let salt = splitmix64(dim_idx as u64 ^ 0x534f_424f_4c44_4952);
    let mut m32 = [1u32; 9];
    let mut k = 0usize;
    while k < s && k < 9 {
        let mask = (1u32 << (k + 1)) - 1;
        let raw = (base_m[k] as u32) ^ (((salt >> (k * 7)) as u32) & mask);
        m32[k] = (raw | 1) & mask;
        k += 1;
    }
    sobol_dirs(s, a as u32, &m32)
}

/// Stateless bitwise XOR product of Gray code `gray` against `[u32; 32]`
/// direction numbers (`O(1)` per `(c, j)`).
#[inline(always)]
pub const fn sobol_raw(gray: u32, dirs: &[u32; 32]) -> u32 {
    let mut g = gray;
    let mut acc = 0u32;
    while g != 0 {
        let b = g.trailing_zeros() as usize;
        acc ^= dirs[b];
        g &= g - 1;
    }
    acc
}

/// Derive a 32-bit per-dimension Owen scramble seed from `(key, dim_idx)`.
#[inline(always)]
pub const fn owen_seed(key: Key, dim_idx: usize) -> u32 {
    let h = key.fold(TAG_SOBOL).fold(dim_idx as u64).seed();
    (h as u32) ^ ((h >> 32) as u32)
}

/// Laine-Karras / Burley (2020) fast stateless hash-based Owen scramble on a
/// 32-bit Sobol integer, preserving all `(0, m, 1)` dyadic stratifications.
#[inline(always)]
pub const fn owen_scramble(val: u32, seed: u32) -> u32 {
    let mut x = val.reverse_bits();
    x = x.wrapping_add(seed);
    x ^= x.wrapping_mul(0x6c50_b47c);
    x ^= x.wrapping_mul(0xb82f_1e52);
    x ^= x.wrapping_mul(0xc7af_e638);
    x ^= x.wrapping_mul(0x8d22_f6e6);
    x.reverse_bits()
}

/// Stateless 32-bit Sobol word at `(base_idx + cand, dim_idx)` using either a
/// flat `dim * 32` direction buffer (`dirs_flat`) or built-in [`joe_kuo`].
#[inline(always)]
pub fn sobol_word(
    base_idx: u32,
    cand: u32,
    dim_idx: usize,
    dirs_flat: &[u32],
    key: Key,
    scramble: bool,
) -> u32 {
    let g = gray_code(base_idx.wrapping_add(cand));
    let raw = if dirs_flat.len() >= (dim_idx + 1) * 32 {
        let offset = dim_idx * 32;
        let mut rem = g;
        let mut acc = 0u32;
        while rem != 0 {
            let b = rem.trailing_zeros() as usize;
            acc ^= dirs_flat[offset + b];
            rem &= rem - 1;
        }
        acc
    } else {
        let d = joe_kuo(dim_idx);
        sobol_raw(g, &d)
    };
    if scramble {
        owen_scramble(raw, owen_seed(key, dim_idx))
    } else {
        raw
    }
}

/// Stateless Sobol coordinate in `[0, 1)` at `(base_idx + cand, dim_idx)`.
#[inline(always)]
pub fn sobol_unit<T: Real>(
    base_idx: u32,
    cand: u32,
    dim_idx: usize,
    dirs: Option<&[u32; 32]>,
    key: Key,
    scramble: bool,
) -> T {
    let g = gray_code(base_idx.wrapping_add(cand));
    let raw = match dirs {
        Some(d) => sobol_raw(g, d),
        None => {
            let d = joe_kuo(dim_idx);
            sobol_raw(g, &d)
        }
    };
    let word = if scramble {
        owen_scramble(raw, owen_seed(key, dim_idx))
    } else {
        raw
    };
    T::sobol_word(word)
}

/// Stateless Sobol pull evaluation scaled to `[lower[j], upper[j]]`.
#[inline(always)]
pub fn eval_sobol<T: Real>(
    key: Key,
    lower: &[T],
    upper: &[T],
    dirs: &[[u32; 32]],
    base_idx: u32,
    scramble: bool,
    cand: usize,
    j: usize,
) -> T {
    let u = sobol_unit::<T>(base_idx, cand as u32, j, dirs.get(j), key, scramble);
    lerp_box(lower[j], upper[j], u)
}

/// Stateless masked Sobol pull evaluation (perturbing coordinates via the RAASP
/// Bernoulli + forced-dimension mask while drawing perturbed values from
/// Sobol).
#[inline(always)]
#[allow(clippy::too_many_arguments)]
pub fn eval_masked<T: Real>(
    key: Key,
    center: &[T],
    lower: &[T],
    upper: &[T],
    dirs: &[[u32; 32]],
    base_idx: u32,
    num_pert: usize,
    scramble: bool,
    cand: usize,
    j: usize,
) -> T {
    let dim = center.len();
    let key_c = cand_key(key, cand);
    if raasp_perturbs(key_c, j, dim, raasp_prob::<T>(num_pert, dim)) {
        eval_sobol(key, lower, upper, dirs, base_idx, scramble, cand, j)
    } else {
        center[j]
    }
}

/// Stateless uniform pull evaluation in `[lower[j], upper[j]]`.
#[inline(always)]
pub fn eval_uniform<T: Real>(key: Key, lower: &[T], upper: &[T], cand: usize, j: usize) -> T {
    let u = hash_unit::<T>(cand_key(key, cand), j, TAG_VALUE);
    lerp_box(lower[j], upper[j], u)
}

/// Stateless Gaussian pull evaluation `center[j] + radius * scale[j] * z`.
#[inline(always)]
pub fn eval_gaussian<T: Real>(
    key: Key,
    center: &[T],
    scale: &[T],
    bounds: Option<(&[T], &[T])>,
    radius: T,
    cand: usize,
    j: usize,
) -> T {
    let z = cand_key(key, cand)
        .fold(TAG_NOISE)
        .fold(j as u64)
        .normal::<T>();
    let val = center[j] + radius * scale[j] * z;
    match bounds {
        Some((lo, hi)) => {
            if hi[j] > lo[j] {
                val.clamp(lo[j], hi[j])
            } else {
                lo[j]
            }
        }
        None => val,
    }
}

/// Stateless Rademacher pull evaluation `center[j] + radius * scale[j] * (±1)`.
#[inline(always)]
pub fn eval_rademacher<T: Real>(
    key: Key,
    center: &[T],
    scale: &[T],
    bounds: Option<(&[T], &[T])>,
    radius: T,
    cand: usize,
    j: usize,
) -> T {
    let sign = cand_key(key, cand)
        .fold(TAG_NOISE)
        .fold(j as u64)
        .rademacher::<T>();
    let val = center[j] + radius * scale[j] * sign;
    match bounds {
        Some((lo, hi)) => {
            if hi[j] > lo[j] {
                val.clamp(lo[j], hi[j])
            } else {
                lo[j]
            }
        }
        None => val,
    }
}

/// Cycle-walking unbalanced Feistel bijection on `0..num_samples`
/// (`#![no_std]`, zero allocation, `O(1)` expected work per coordinate).
#[inline]
pub const fn lhd_bin(key: Key, sample: usize, num_samples: usize, dim_idx: usize) -> usize {
    if num_samples <= 1 || sample >= num_samples {
        return 0;
    }
    let key_d = key.fold(TAG_LHD).fold(dim_idx as u64);
    let bits = (usize::BITS - (num_samples - 1).leading_zeros()) as usize;
    let a = bits / 2;
    let b = bits - a;
    let mask_a = (1usize << a) - 1;
    let mask_b = (1usize << b) - 1;
    let mut x = sample;
    loop {
        let mut u = x & mask_a;
        let mut v = (x >> a) & mask_b;
        u ^= (splitmix64(key_d.0 ^ (v as u64)) as usize) & mask_a;
        v ^= (splitmix64(key_d.1 ^ (u as u64)) as usize) & mask_b;
        u ^= (splitmix64(key_d.0.rotate_left(17) ^ (v as u64)) as usize) & mask_a;
        v ^= (splitmix64(key_d.1.rotate_left(29) ^ (u as u64)) as usize) & mask_b;
        x = u | (v << a);
        if x < num_samples {
            return x;
        }
    }
}

/// Stateless Latin Hypercube Design (`Lhd`) warmup pull evaluation.
#[inline]
pub fn eval_lhd<T: Real>(
    key: Key,
    lower: &[T],
    upper: &[T],
    num_samples: usize,
    sample: usize,
    j: usize,
) -> T {
    if num_samples == 0 {
        return lower[j];
    }
    let bin = lhd_bin(key, sample, num_samples, j);
    let jitter = hash_unit::<T>(cand_key(key, sample), j, TAG_LHD);
    let unit = (T::from_idx(bin) + jitter) / T::from_idx(num_samples);
    lerp_box(lower[j], upper[j], unit)
}

/// Index-pure `#![no_std]` candidate pull-array parameterized over [`Real`].
/// Evaluates any `(c, j)` coordinate on demand without allocating a
/// `(num_candidates, num_dim)` matrix.
#[derive(Clone, Copy, Debug)]
pub enum DelayedPool<'a, T: Real = f64> {
    Raasp {
        key: Key,
        center: &'a [T],
        lower: &'a [T],
        upper: &'a [T],
        num_candidates: usize,
        num_pert: usize,
    },
    Sobol {
        key: Key,
        lower: &'a [T],
        upper: &'a [T],
        dirs: &'a [[u32; 32]],
        base_idx: u32,
        num_candidates: usize,
        scramble: bool,
    },
    SobolMasked {
        key: Key,
        center: &'a [T],
        lower: &'a [T],
        upper: &'a [T],
        dirs: &'a [[u32; 32]],
        base_idx: u32,
        num_candidates: usize,
        num_pert: usize,
        scramble: bool,
    },
    Uniform {
        key: Key,
        lower: &'a [T],
        upper: &'a [T],
        num_candidates: usize,
    },
    Gaussian {
        key: Key,
        center: &'a [T],
        scale: &'a [T],
        bounds: Option<(&'a [T], &'a [T])>,
        radius: T,
        num_candidates: usize,
    },
    Rademacher {
        key: Key,
        center: &'a [T],
        scale: &'a [T],
        bounds: Option<(&'a [T], &'a [T])>,
        radius: T,
        num_candidates: usize,
    },
    Lhd {
        key: Key,
        lower: &'a [T],
        upper: &'a [T],
        num_candidates: usize,
    },
}

impl<'a, T: Real> DelayedPool<'a, T> {
    #[inline(always)]
    pub const fn num_candidates(&self) -> usize {
        match *self {
            Self::Raasp { num_candidates, .. }
            | Self::Sobol { num_candidates, .. }
            | Self::SobolMasked { num_candidates, .. }
            | Self::Uniform { num_candidates, .. }
            | Self::Gaussian { num_candidates, .. }
            | Self::Rademacher { num_candidates, .. }
            | Self::Lhd { num_candidates, .. } => num_candidates,
        }
    }

    #[inline(always)]
    pub const fn num_dim(&self) -> usize {
        match *self {
            Self::Raasp { center, .. }
            | Self::SobolMasked { center, .. }
            | Self::Gaussian { center, .. }
            | Self::Rademacher { center, .. } => center.len(),
            Self::Sobol { lower, .. } | Self::Uniform { lower, .. } | Self::Lhd { lower, .. } => {
                lower.len()
            }
        }
    }

    /// Evaluate coordinate `(cand, j)` deterministically in `O(1)`.
    #[inline(always)]
    pub fn eval(&self, cand: usize, j: usize) -> T {
        match *self {
            Self::Raasp {
                key,
                center,
                lower,
                upper,
                num_pert,
                ..
            } => eval_raasp(key, center, lower, upper, num_pert, cand, j),
            Self::Sobol {
                key,
                lower,
                upper,
                dirs,
                base_idx,
                scramble,
                ..
            } => eval_sobol(key, lower, upper, dirs, base_idx, scramble, cand, j),
            Self::SobolMasked {
                key,
                center,
                lower,
                upper,
                dirs,
                base_idx,
                num_pert,
                scramble,
                ..
            } => eval_masked(
                key, center, lower, upper, dirs, base_idx, num_pert, scramble, cand, j,
            ),
            Self::Uniform {
                key, lower, upper, ..
            } => eval_uniform(key, lower, upper, cand, j),
            Self::Gaussian {
                key,
                center,
                scale,
                bounds,
                radius,
                ..
            } => eval_gaussian(key, center, scale, bounds, radius, cand, j),
            Self::Rademacher {
                key,
                center,
                scale,
                bounds,
                radius,
                ..
            } => eval_rademacher(key, center, scale, bounds, radius, cand, j),
            Self::Lhd {
                key,
                lower,
                upper,
                num_candidates,
            } => eval_lhd(key, lower, upper, num_candidates, cand, j),
        }
    }

    /// Materialize a single candidate row `cand` into `out[..dim]`.
    #[inline]
    pub fn write_row(&self, cand: usize, out: &mut [T]) {
        let d = self.num_dim().min(out.len());
        for (j, slot) in out.iter_mut().enumerate().take(d) {
            *slot = self.eval(cand, j);
        }
    }

    /// Return sparse `(dim_idx, new_value)` iterator when `self` is
    /// `DelayedPool::Raasp`.
    #[inline]
    pub fn deltas(&self, cand: usize) -> Option<RaaspDeltas<'a, T>> {
        match *self {
            Self::Raasp {
                key,
                lower,
                upper,
                num_pert,
                ..
            } => Some(RaaspDeltas::new(key, cand, lower, upper, num_pert)),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pool_samplers() {
        let key = Key::new(77);
        let center = [0.5_f64; 8];
        let lo = [0.2_f64; 8];
        let hi = [0.8_f64; 8];
        let pool = DelayedPool::Raasp {
            key,
            center: &center,
            lower: &lo,
            upper: &hi,
            num_candidates: 4,
            num_pert: 2,
        };
        let mut row = [0.0; 8];
        pool.write_row(0, &mut row);
        assert!(row.iter().all(|&v| (0.2..=0.8).contains(&v)));
        assert!(pool.deltas(0).unwrap().count() >= 1);

        let center32 = [0.5_f32; 8];
        let lo32 = [0.2_f32; 8];
        let hi32 = [0.8_f32; 8];
        let pool32 = DelayedPool::<f32>::Raasp {
            key,
            center: &center32,
            lower: &lo32,
            upper: &hi32,
            num_candidates: 4,
            num_pert: 2,
        };
        assert!((0.2..=0.8).contains(&pool32.eval(0, 0)));
    }
}
