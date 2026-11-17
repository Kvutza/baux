//! Sealed `#![no_std]` floating-point scalar trait (`f32` and `f64`) unifying
//! host, resident Apple Metal, and `cuda-oxide` device math without duplicate
//! precision-suffixed APIs.

use core::cmp::Ordering;
use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};
use core::simd::f64x4;
use core::simd::num::SimdFloat;

mod sealed {
    pub trait Marker {}
    impl Marker for f32 {}
    impl Marker for f64 {}
}

/// Sealed `#![no_std]` IEEE-754 binary floating-point scalar (`f32` or `f64`).
pub trait Real:
    sealed::Marker
    + Copy
    + Default
    + PartialEq
    + PartialOrd
    + core::fmt::Debug
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + Neg<Output = Self>
    + AddAssign
    + SubAssign
    + MulAssign
    + DivAssign
    + 'static
{
    const ZERO: Self;
    const ONE: Self;
    const TWO: Self;
    const HALF: Self;
    const INFINITY: Self;
    const NEG_INFINITY: Self;
    const NAN: Self;
    const BOUND_EPS: Self;
    const NOISE_FLOOR: Self;
    const VAR_FLOOR: Self;
    const PREC_EPS: Self;

    fn from_val(v: f64) -> Self;
    fn to_val(self) -> f64;
    fn from_idx(n: usize) -> Self;
    fn from_word(w: u32) -> Self;
    fn is_nan(self) -> bool;
    fn is_finite(self) -> bool;
    fn abs(self) -> Self;
    fn sqrt(self) -> Self;
    fn ln(self) -> Self;
    fn exp(self) -> Self;
    fn sin(self) -> Self;
    fn cos(self) -> Self;
    fn min(self, other: Self) -> Self;
    fn max(self, other: Self) -> Self;
    fn clamp(self, lo: Self, hi: Self) -> Self;
    fn mul_add(self, a: Self, b: Self) -> Self;
    fn total_cmp(&self, other: &Self) -> Ordering;

    /// Unit-interval draw in `[0, 1)` from a 64-bit word.
    fn unit_word(word: u64) -> Self;
    /// Open unit-interval draw in `(0, 1)` clamped away from `{0, 1}`.
    fn open_word(word: u64) -> Self;
    /// Unit-interval draw in `[0, 1)` from a 32-bit Sobol word.
    fn sobol_word(word: u32) -> Self;
    /// Standard normal `N(0, 1)` Box-Muller draw from two 64-bit words.
    fn normal_words(w0: u64, w1: u64) -> Self;
    /// ARD weighted squared Euclidean distance `sum_j w_j * (x_j - y_j)^2`.
    fn ard_slice(x: &[Self], y: &[Self], weights: &[Self]) -> Self;
}

impl Real for f32 {
    const ZERO: Self = 0.0;
    const ONE: Self = 1.0;
    const TWO: Self = 2.0;
    const HALF: Self = 0.5;
    const INFINITY: Self = f32::INFINITY;
    const NEG_INFINITY: Self = f32::NEG_INFINITY;
    const NAN: Self = f32::NAN;
    const BOUND_EPS: Self = 1.0e-5;
    const NOISE_FLOOR: Self = 1.0e-12;
    const VAR_FLOOR: Self = 1.0e-24;
    const PREC_EPS: Self = 1.0e-9;

    #[inline(always)]
    fn from_val(v: f64) -> Self {
        v as f32
    }

    #[inline(always)]
    fn to_val(self) -> f64 {
        f64::from(self)
    }

    #[inline(always)]
    fn from_idx(n: usize) -> Self {
        n as f32
    }

    #[inline(always)]
    fn from_word(w: u32) -> Self {
        w as f32
    }

    #[inline(always)]
    fn is_nan(self) -> bool {
        f32::is_nan(self)
    }

    #[inline(always)]
    fn is_finite(self) -> bool {
        f32::is_finite(self)
    }

    #[inline(always)]
    fn abs(self) -> Self {
        f32::abs(self)
    }

    #[inline(always)]
    fn sqrt(self) -> Self {
        f32::sqrt(self)
    }

    #[inline(always)]
    fn ln(self) -> Self {
        f32::ln(self)
    }

    #[inline(always)]
    fn exp(self) -> Self {
        f32::exp(self)
    }

    #[inline(always)]
    fn sin(self) -> Self {
        f32::sin(self)
    }

    #[inline(always)]
    fn cos(self) -> Self {
        f32::cos(self)
    }

    #[inline(always)]
    fn min(self, other: Self) -> Self {
        f32::min(self, other)
    }

    #[inline(always)]
    fn max(self, other: Self) -> Self {
        f32::max(self, other)
    }

    #[inline(always)]
    fn clamp(self, lo: Self, hi: Self) -> Self {
        f32::clamp(self, lo, hi)
    }

    #[inline(always)]
    fn mul_add(self, a: Self, b: Self) -> Self {
        f32::mul_add(self, a, b)
    }

    #[inline(always)]
    fn total_cmp(&self, other: &Self) -> Ordering {
        f32::total_cmp(self, other)
    }

    #[inline(always)]
    fn unit_word(word: u64) -> Self {
        ((word >> 40) as u32 as f32) * (1.0 / 16_777_216.0)
    }

    #[inline(always)]
    fn open_word(word: u64) -> Self {
        Self::unit_word(word).clamp(1.0e-7, 1.0 - 1.0e-7)
    }

    #[inline(always)]
    fn sobol_word(word: u32) -> Self {
        ((word >> 8) as f32) * (1.0 / 16_777_216.0)
    }

    #[inline(always)]
    fn normal_words(w0: u64, w1: u64) -> Self {
        let u1 = Self::open_word(w0);
        let u2 = Self::unit_word(w1);
        let r = (-2.0_f32 * u1.ln()).max(0.0).sqrt();
        r * (core::f32::consts::TAU * u2).cos()
    }

    #[inline]
    fn ard_slice(x: &[Self], y: &[Self], weights: &[Self]) -> Self {
        let n = x.len().min(y.len()).min(weights.len());
        let mut sum = 0.0_f32;
        for j in 0..n {
            let d = x[j] - y[j];
            sum = (weights[j] * d).mul_add(d, sum);
        }
        sum.max(0.0)
    }
}

impl Real for f64 {
    const ZERO: Self = 0.0;
    const ONE: Self = 1.0;
    const TWO: Self = 2.0;
    const HALF: Self = 0.5;
    const INFINITY: Self = f64::INFINITY;
    const NEG_INFINITY: Self = f64::NEG_INFINITY;
    const NAN: Self = f64::NAN;
    const BOUND_EPS: Self = 1.0e-6;
    const NOISE_FLOOR: Self = 1.0e-12;
    const VAR_FLOOR: Self = 1.0e-24;
    const PREC_EPS: Self = 1.0e-9;

    #[inline(always)]
    fn from_val(v: f64) -> Self {
        v
    }

    #[inline(always)]
    fn to_val(self) -> f64 {
        self
    }

    #[inline(always)]
    fn from_idx(n: usize) -> Self {
        n as f64
    }

    #[inline(always)]
    fn from_word(w: u32) -> Self {
        f64::from(w)
    }

    #[inline(always)]
    fn is_nan(self) -> bool {
        f64::is_nan(self)
    }

    #[inline(always)]
    fn is_finite(self) -> bool {
        f64::is_finite(self)
    }

    #[inline(always)]
    fn abs(self) -> Self {
        f64::abs(self)
    }

    #[inline(always)]
    fn sqrt(self) -> Self {
        f64::sqrt(self)
    }

    #[inline(always)]
    fn ln(self) -> Self {
        f64::ln(self)
    }

    #[inline(always)]
    fn exp(self) -> Self {
        f64::exp(self)
    }

    #[inline(always)]
    fn sin(self) -> Self {
        f64::sin(self)
    }

    #[inline(always)]
    fn cos(self) -> Self {
        f64::cos(self)
    }

    #[inline(always)]
    fn min(self, other: Self) -> Self {
        f64::min(self, other)
    }

    #[inline(always)]
    fn max(self, other: Self) -> Self {
        f64::max(self, other)
    }

    #[inline(always)]
    fn clamp(self, lo: Self, hi: Self) -> Self {
        f64::clamp(self, lo, hi)
    }

    #[inline(always)]
    fn mul_add(self, a: Self, b: Self) -> Self {
        f64::mul_add(self, a, b)
    }

    #[inline(always)]
    fn total_cmp(&self, other: &Self) -> Ordering {
        f64::total_cmp(self, other)
    }

    #[inline(always)]
    fn unit_word(word: u64) -> Self {
        ((word >> 11) as f64) * (1.0 / 9_007_199_254_740_992.0)
    }

    #[inline(always)]
    fn open_word(word: u64) -> Self {
        Self::unit_word(word).clamp(1.0e-12, 1.0 - 1.0e-12)
    }

    #[inline(always)]
    fn sobol_word(word: u32) -> Self {
        f64::from(word) * (1.0 / 4_294_967_296.0)
    }

    #[inline(always)]
    fn normal_words(w0: u64, w1: u64) -> Self {
        let u1 = Self::open_word(w0);
        let u2 = Self::unit_word(w1);
        let r = (-2.0_f64 * u1.ln()).max(0.0).sqrt();
        r * (core::f64::consts::TAU * u2).cos()
    }

    #[inline]
    fn ard_slice(x: &[Self], y: &[Self], weights: &[Self]) -> Self {
        let n = x.len().min(y.len()).min(weights.len());
        let xc = x[..n].chunks_exact(4);
        let yc = y[..n].chunks_exact(4);
        let wc = weights[..n].chunks_exact(4);
        let (xr, yr, wr) = (xc.remainder(), yc.remainder(), wc.remainder());
        let mut acc = f64x4::splat(0.0);
        for ((a, b), w) in xc.zip(yc).zip(wc) {
            let d = f64x4::from_slice(a) - f64x4::from_slice(b);
            acc += f64x4::from_slice(w) * d * d;
        }
        let tail: f64 = xr
            .iter()
            .zip(yr.iter())
            .zip(wr.iter())
            .map(|((&a, &b), &w)| {
                let d = a - b;
                w * d * d
            })
            .sum();
        (acc.reduce_sum() + tail).max(0.0)
    }
}
