//! Bit-precise IEEE-754 binary32 and binary64 floating-point operations.
//!
//! Provides integer-backed arithmetic, comparisons, bit decompositions, and
//! casts matching Rust and IEEE-754 semantics. These operations bypass the
//! pinned Aeneas float-translation gap while enabling machine-checked proofs
//! in Lean 4 via FloatLib.

#[path = "float/bits.rs"]
mod bits;
pub use bits::{
    EXP_DOUBLE, EXP_SINGLE, FRAC_DOUBLE, FRAC_SINGLE, HALF_DOUBLE, INF_DOUBLE, INF_SINGLE,
    NAN_DOUBLE, NAN_SINGLE, ONE_DOUBLE, ONE_SINGLE, PI_DOUBLE, PRIOR_DOUBLE, SCALE_DOUBLE,
    SIGN_DOUBLE, SIGN_SINGLE, TWO_DOUBLE, ZERO_DOUBLE, ZERO_SINGLE, double_abs, double_bits,
    double_from, double_neg, exp_double, exp_single, frac_double, frac_single, sign_double,
    sign_single, single_abs, single_bits, single_from, single_neg, to_double, to_single,
};

// --- Classifications ---

#[inline(always)]
pub fn single_nan(bits: u32) -> bool {
    bits & EXP_SINGLE == EXP_SINGLE && bits & FRAC_SINGLE != 0
}

#[inline(always)]
pub fn single_inf(bits: u32) -> bool {
    bits & !SIGN_SINGLE == EXP_SINGLE
}

#[inline(always)]
pub fn single_finite(bits: u32) -> bool {
    bits & EXP_SINGLE != EXP_SINGLE
}

#[inline(always)]
pub fn single_zero(bits: u32) -> bool {
    bits & !SIGN_SINGLE == 0
}

#[inline(always)]
pub fn double_nan(bits: u64) -> bool {
    bits & EXP_DOUBLE == EXP_DOUBLE && bits & FRAC_DOUBLE != 0
}

#[inline(always)]
pub fn double_inf(bits: u64) -> bool {
    bits & !SIGN_DOUBLE == EXP_DOUBLE
}

#[inline(always)]
pub fn double_finite(bits: u64) -> bool {
    bits & EXP_DOUBLE != EXP_DOUBLE
}

#[inline(always)]
pub fn double_zero(bits: u64) -> bool {
    bits & !SIGN_DOUBLE == 0
}

// --- Total ordering mapping (IEEE-754-2008 totalOrder) ---

#[inline(always)]
pub fn single_order(bits: u32) -> u32 {
    if bits & SIGN_SINGLE == 0 {
        bits | SIGN_SINGLE
    } else {
        !bits
    }
}

#[inline(always)]
pub fn double_order(bits: u64) -> u64 {
    if bits & SIGN_DOUBLE == 0 {
        bits | SIGN_DOUBLE
    } else {
        !bits
    }
}

// --- Strict comparisons matching IEEE-754 and Rust native floats ---

#[inline(always)]
pub fn single_lt(x: u32, y: u32) -> bool {
    if single_nan(x) || single_nan(y) || (single_zero(x) && single_zero(y)) {
        false
    } else {
        single_order(x) < single_order(y)
    }
}

#[inline(always)]
pub fn single_le(x: u32, y: u32) -> bool {
    if single_nan(x) || single_nan(y) {
        false
    } else if single_zero(x) && single_zero(y) {
        true
    } else {
        single_order(x) <= single_order(y)
    }
}

#[inline(always)]
pub fn single_gt(x: u32, y: u32) -> bool {
    single_lt(y, x)
}

#[inline(always)]
pub fn single_ge(x: u32, y: u32) -> bool {
    single_le(y, x)
}

#[inline(always)]
pub fn single_eq(x: u32, y: u32) -> bool {
    if single_nan(x) || single_nan(y) {
        false
    } else if single_zero(x) && single_zero(y) {
        true
    } else {
        x == y
    }
}

#[inline(always)]
pub fn single_total(x: u32, y: u32) -> i8 {
    let ox = single_order(x);
    let oy = single_order(y);
    if ox < oy {
        -1
    } else if ox > oy {
        1
    } else {
        0
    }
}

#[inline(always)]
pub fn double_lt(x: u64, y: u64) -> bool {
    if double_nan(x) || double_nan(y) || (double_zero(x) && double_zero(y)) {
        false
    } else {
        double_order(x) < double_order(y)
    }
}

#[inline(always)]
pub fn double_le(x: u64, y: u64) -> bool {
    if double_nan(x) || double_nan(y) {
        false
    } else if double_zero(x) && double_zero(y) {
        true
    } else {
        double_order(x) <= double_order(y)
    }
}

#[inline(always)]
pub fn double_gt(x: u64, y: u64) -> bool {
    double_lt(y, x)
}

#[inline(always)]
pub fn double_ge(x: u64, y: u64) -> bool {
    double_le(y, x)
}

#[inline(always)]
pub fn double_eq(x: u64, y: u64) -> bool {
    if double_nan(x) || double_nan(y) {
        false
    } else if double_zero(x) && double_zero(y) {
        true
    } else {
        x == y
    }
}

#[inline(always)]
pub fn double_total(x: u64, y: u64) -> i8 {
    let ox = double_order(x);
    let oy = double_order(y);
    if ox < oy {
        -1
    } else if ox > oy {
        1
    } else {
        0
    }
}

#[cfg(test)]
#[path = "float/tests.rs"]
mod tests;
