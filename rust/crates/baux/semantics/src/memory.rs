//! Checked byte geometry used at the host/device binding boundary.

/// Exact byte count, or rejection before either multiplication can overflow.
pub fn view_bytes(rows: u64, cols: u64, width: u64) -> Option<u64> {
    match rows.checked_mul(cols) {
        Some(elements) => elements.checked_mul(width),
        None => None,
    }
}

/// End of an in-bounds half-open span, including empty spans at the allocation
/// end.
pub fn span_end(capacity: u64, offset: u64, bytes: u64) -> Option<u64> {
    if offset > capacity {
        return None;
    }
    let remaining = capacity - offset;
    if bytes > remaining {
        None
    } else {
        Some(offset + bytes)
    }
}

/// Intersection of two nonempty half-open spans in the same allocation.
pub fn overlaps(a: u64, b: u64, c: u64, d: u64) -> bool {
    a < b && c < d && a < d && c < b
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bounds() {
        assert_eq!(view_bytes(u64::MAX, 2, 0), None);
        assert_eq!(view_bytes(1, u64::MAX, 2), None);
        assert_eq!(view_bytes(0, u64::MAX, 4), Some(0));
        assert_eq!(span_end(u64::MAX, u64::MAX, 0), Some(u64::MAX));
        assert_eq!(span_end(u64::MAX, u64::MAX, 1), None);
        assert_eq!(span_end(16, 17, 0), None);
        assert_eq!(span_end(16, 12, 4), Some(16));
        assert_eq!(span_end(16, 12, 5), None);
    }

    #[test]
    fn intersections() {
        for a in 0..8 {
            for b in a..8 {
                for c in 0..8 {
                    for d in c..8 {
                        let shared = (a..b).any(|byte| (c..d).contains(&byte));
                        assert_eq!(overlaps(a, b, c, d), shared);
                    }
                }
            }
        }
    }
}
