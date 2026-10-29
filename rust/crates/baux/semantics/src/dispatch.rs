//! Host-side geometry for the optimizer's tiled Metal kernels.
//!
//! The extracted length calculation supplies the bounds used by the Lean lane
//! coverage proofs. It does not verify shader compilation or device execution.

pub const TILE_ELEMENTS: u32 = 65_536;

/// Start of the indexed tile, rejecting multiplication overflow and tiles
/// outside the leaf. The final possible tile starts at 65535 * 65536.
pub fn tile_start(leaf: u32, ordinal: u32) -> Option<u32> {
    if ordinal >= TILE_ELEMENTS {
        return None;
    }
    let start = ordinal * TILE_ELEMENTS;
    if start < leaf { Some(start) } else { None }
}

/// Length of the nonempty tile beginning at `start` inside one parameter leaf.
/// Invalid starting offsets have no tile. Subtraction happens only after the
/// range check, including when `leaf` is the largest representable `u32`.
pub fn tile_len(leaf: u32, start: u32) -> Option<u32> {
    if start >= leaf {
        None
    } else {
        let remaining = leaf - start;
        Some(if remaining > TILE_ELEMENTS {
            TILE_ELEMENTS
        } else {
            remaining
        })
    }
}

/// Enumerate the complete finite tile geometry of one leaf.
/// This setup-only allocation makes the production iteration extractable as a
/// whole; callers attach their leaf identity to each returned pair.
pub fn tiles(leaf: u32) -> Option<Vec<(u32, u32)>> {
    let mut tiles = Vec::new();
    let mut ordinal = 0;
    while let Some(start) = tile_start(leaf, ordinal) {
        let length = match tile_len(leaf, start) {
            Some(length) => length,
            None => return None,
        };
        tiles.push((start, length));
        ordinal += 1;
    }
    Some(tiles)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn partition() {
        for leaf in [
            0,
            1,
            255,
            256,
            257,
            511,
            512,
            513,
            65_535,
            65_536,
            65_537,
            131_073,
            u32::MAX,
        ] {
            let mut end = 0;
            let geometry = tiles(leaf).expect("every u32 leaf has a finite tile layout");
            for (ordinal, &(start, length)) in geometry.iter().enumerate() {
                assert_eq!(tile_start(leaf, ordinal as u32), Some(start));
                assert_eq!(start, end);
                assert_eq!(start % 2, 0);
                assert!(length > 0 && length <= TILE_ELEMENTS);
                end = start.checked_add(length).expect("tile end fits u32");
                assert!(end <= leaf);
            }
            assert_eq!(tile_start(leaf, geometry.len() as u32), None);
            assert_eq!(end, leaf);
        }
    }

    #[test]
    fn machine_edges() {
        assert_eq!(tile_start(0, 0), None);
        assert_eq!(tile_start(1, 0), Some(0));
        assert_eq!(tile_start(1, 1), None);
        assert_eq!(tile_start(u32::MAX, 65535), Some(4294901760));
        assert_eq!(tile_start(u32::MAX, 65536), None);
        assert_eq!(tile_start(u32::MAX, u32::MAX), None);
        assert_eq!(tile_len(0, 0), None);
        assert_eq!(tile_len(1, 1), None);
        assert_eq!(tile_len(1, u32::MAX), None);
        assert_eq!(tile_len(u32::MAX, 0), Some(TILE_ELEMENTS));
        assert_eq!(tile_len(u32::MAX, u32::MAX - 1), Some(1));
        assert_eq!(
            tile_len(u32::MAX, u32::MAX - TILE_ELEMENTS),
            Some(TILE_ELEMENTS)
        );
        assert_eq!(tile_len(u32::MAX, u32::MAX), None);
    }
}
