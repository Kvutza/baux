pub const SIGN_SINGLE: u32 = 0x8000_0000;
pub const EXP_SINGLE: u32 = 0x7f80_0000;
pub const FRAC_SINGLE: u32 = 0x007f_ffff;

pub const ZERO_SINGLE: u32 = 0x0000_0000;
pub const ONE_SINGLE: u32 = 0x3f80_0000;
pub const INF_SINGLE: u32 = 0x7f80_0000;
pub const NAN_SINGLE: u32 = 0x7fc0_0000;

pub const SIGN_DOUBLE: u64 = 0x8000_0000_0000_0000;
pub const EXP_DOUBLE: u64 = 0x7ff0_0000_0000_0000;
pub const FRAC_DOUBLE: u64 = 0x000f_ffff_ffff_ffff;

pub const ZERO_DOUBLE: u64 = 0x0000_0000_0000_0000;
pub const ONE_DOUBLE: u64 = 0x3ff0_0000_0000_0000;
pub const TWO_DOUBLE: u64 = 0x4000_0000_0000_0000;
pub const HALF_DOUBLE: u64 = 0x3fe0_0000_0000_0000;
pub const SCALE_DOUBLE: u64 = 0x3d71_9799_812d_ea11; // 1.0e-12
pub const PRIOR_DOUBLE: u64 = 0x3ff0_0000_0000_0000; // 1.0
pub const PI_DOUBLE: u64 = 0x4009_21fb_5444_2d18; // 3.141592653589793
pub const INF_DOUBLE: u64 = 0x7ff0_0000_0000_0000;
pub const NAN_DOUBLE: u64 = 0x7ff8_0000_0000_0000;

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

#[inline(always)]
pub fn single_neg(bits: u32) -> u32 {
    bits ^ SIGN_SINGLE
}

#[inline(always)]
pub fn single_abs(bits: u32) -> u32 {
    bits & !SIGN_SINGLE
}

#[inline(always)]
pub fn double_neg(bits: u64) -> u64 {
    bits ^ SIGN_DOUBLE
}

#[inline(always)]
pub fn double_abs(bits: u64) -> u64 {
    bits & !SIGN_DOUBLE
}

/// Bit-exact widening cast from binary32 to binary64.
///
/// Preserves exact numerical values for all normal and subnormal values,
/// zeros, infinities, and NaN payloads without loss.
#[inline(always)]
pub fn to_double(bits: u32) -> u64 {
    let sign = ((bits & SIGN_SINGLE) as u64) << 32;
    let exp = (bits & EXP_SINGLE) >> 23;
    let frac = (bits & FRAC_SINGLE) as u64;
    if exp == 0 {
        if frac == 0 {
            // Signed zero
            sign
        } else {
            // Subnormal: normalize using 5-step unrolled binary search (no
            // loops)
            let mut f = frac;
            let mut shift = 0u64;
            if (f & 0x007f_ff80) == 0 {
                f <<= 16;
                shift += 16;
            }
            if (f & 0x007f_8000) == 0 {
                f <<= 8;
                shift += 8;
            }
            if (f & 0x0078_0000) == 0 {
                f <<= 4;
                shift += 4;
            }
            if (f & 0x0060_0000) == 0 {
                f <<= 2;
                shift += 2;
            }
            if (f & 0x0040_0000) == 0 {
                f <<= 1;
                shift += 1;
            }
            let norm_frac = (f << 1) & (FRAC_SINGLE as u64);
            let new_exp = (896 - shift) << 52;
            sign | new_exp | (norm_frac << 29)
        }
    } else if exp == 0xff {
        // Inf or NaN
        sign | (0x7ff_u64 << 52) | (frac << 29)
    } else {
        // Normal
        let new_exp = ((exp as u64) + 896) << 52;
        sign | new_exp | (frac << 29)
    }
}

/// Nearest-even narrowing cast from binary64 to binary32.
#[inline(always)]
pub fn to_single(bits: u64) -> u32 {
    (f64::from_bits(bits) as f32).to_bits()
}

/// Native IEEE-754 binary32 from raw bits.
#[inline(always)]
pub fn single_from(bits: u32) -> f32 {
    f32::from_bits(bits)
}

/// Raw bits from native IEEE-754 binary32.
#[inline(always)]
pub fn single_bits(val: f32) -> u32 {
    val.to_bits()
}

/// Native IEEE-754 binary64 from raw bits.
#[inline(always)]
pub fn double_from(bits: u64) -> f64 {
    f64::from_bits(bits)
}

/// Raw bits from native IEEE-754 binary64.
#[inline(always)]
pub fn double_bits(val: f64) -> u64 {
    val.to_bits()
}
