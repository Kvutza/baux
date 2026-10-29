use super::{EXP_DOUBLE, EXP_SINGLE, FRAC_DOUBLE, FRAC_SINGLE, SIGN_DOUBLE, SIGN_SINGLE};

#[inline(always)]
pub fn sign_single(bits: u32) -> bool {
    bits & SIGN_SINGLE != 0
}

#[inline(always)]
pub fn exp_single(bits: u32) -> u32 {
    (bits & EXP_SINGLE) >> 23
}

#[inline(always)]
pub fn frac_single(bits: u32) -> u32 {
    bits & FRAC_SINGLE
}

#[inline(always)]
pub fn sign_double(bits: u64) -> bool {
    bits & SIGN_DOUBLE != 0
}

#[inline(always)]
pub fn exp_double(bits: u64) -> u64 {
    (bits & EXP_DOUBLE) >> 52
}

#[inline(always)]
pub fn frac_double(bits: u64) -> u64 {
    bits & FRAC_DOUBLE
}
