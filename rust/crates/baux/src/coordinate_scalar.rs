//! Scalar kernel shared by production coordinate conversion and extraction.

pub fn from_unit(value: f64, lower: f64, upper: f64) -> f64 {
    lower + value * (upper - lower)
}

pub fn to_unit(value: f64, lower: f64, upper: f64) -> f64 {
    let range = upper - lower;
    if range > 0.0 {
        (value - lower) / range
    } else {
        0.5
    }
}
