//! Scalar kernel shared by production coordinate conversion and extraction.

#[inline(always)]
pub const fn from_unit(value: f64, lower: f64, upper: f64) -> f64 {
    lower + value * (upper - lower)
}

#[inline(always)]
pub const fn to_unit(value: f64, lower: f64, upper: f64) -> f64 {
    let range = upper - lower;
    if range > 0.0 {
        (value - lower) / range
    } else {
        0.5
    }
}

#[inline(always)]
pub const fn clamp_unit(value: f64) -> f64 {
    value.clamp(0.0, 1.0)
}
