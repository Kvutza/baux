//! Publication rule for a completely materialized parameter proposal.
//! Flags summarize every parameter, not the sampled neighbour geometry.

pub const INVALID: u32 = 1;
pub const CHANGED: u32 = 2;

#[inline(always)]
pub const fn fp16_flags(base: u16, candidate: u16) -> u32 {
    (if candidate & 0x7c00 == 0x7c00 {
        INVALID
    } else {
        0
    }) | (if candidate != base { CHANGED } else { 0 })
}

#[inline(always)]
pub const fn fp32_flags(base: f32, candidate: f32) -> u32 {
    (if !candidate.is_finite() { INVALID } else { 0 })
        | (if candidate.to_bits() != base.to_bits() {
            CHANGED
        } else {
            0
        })
}

pub fn validate(flags: u32, allow_unchanged: bool) -> Result<(), &'static str> {
    if flags & !(INVALID | CHANGED) != 0 {
        return Err("proposal validity has unknown flags");
    }
    if flags & INVALID != 0 {
        return Err("selected proposal is nonfinite");
    }
    if flags & CHANGED == 0 && !allow_unchanged {
        return Err("selected proposal did not change");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn validity() {
        for candidate in 0..=u16::MAX {
            let actual = fp16_flags(0x3c00, candidate);
            assert_eq!(actual & INVALID != 0, (candidate >> 10) & 31 == 31);
            assert_eq!(actual & CHANGED != 0, candidate != 0x3c00);
        }
        assert_eq!(fp32_flags(1.0, 1.0), 0);
        assert_eq!(fp32_flags(1.0, 1.0 + 1.0e-5), CHANGED);
        assert_eq!(fp32_flags(1.0, 1.0 + 1.0e-20), 0);
        assert_eq!(fp32_flags(0.0, -0.0), CHANGED);
        assert_eq!(fp32_flags(1.0, f32::INFINITY), INVALID | CHANGED);
        assert_eq!(fp32_flags(1.0, f32::NEG_INFINITY), INVALID | CHANGED);
        assert_eq!(fp32_flags(1.0, f32::NAN), INVALID | CHANGED);
        assert!(validate(0, false).is_err());
        assert!(validate(0, true).is_ok());
        assert!(validate(CHANGED, false).is_ok());
        for flags in [INVALID, INVALID | CHANGED, 4, u32::MAX] {
            assert!(validate(flags, false).is_err());
            assert!(validate(flags, true).is_err());
        }
    }
}
