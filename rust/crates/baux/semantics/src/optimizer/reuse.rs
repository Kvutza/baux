//! Conditioned antipodal slot and sign selection helpers verified by Lean.

/// Antipodal partner slot sharing the same procedural stream (`slot / 2`)
/// with opposite sign in `QuadraticFlow` (`0 <-> 1`, `2 <-> 3`) or paired
/// scale.
pub const fn antipodal_slot(slot: u64) -> u64 {
    (slot & 3) ^ 1
}

/// Conditioned two-step slot selection:
/// - When `accepted` is true, continue along `slot & 3`.
/// - When `accepted` is false, flip to the antipodal partner `(slot & 3) ^ 1`.
pub const fn conditioned_slot(slot: u64, accepted: bool) -> u64 {
    if accepted { slot & 3 } else { (slot & 3) ^ 1 }
}

/// Conditioned two-step sign selection:
/// - When `accepted` is true, preserve the current sign `reversed`.
/// - When `accepted` is false, negate `!reversed` to reverse the uphill probe.
pub const fn conditioned_sign(reversed: bool, accepted: bool) -> bool {
    if accepted { reversed } else { !reversed }
}
