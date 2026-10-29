//! Canonical mHC-4 dimensions and partial contraction work count.
pub const MHC4_PATCH: u64 = 1;
pub const PISA_BYTES: u64 = 260;
pub const MHC4_LAYERS: u64 = 5;
pub const MHC4_WIDTH: u64 = 512;
pub const MHC4_QUERY: u64 = 64;
pub const MHC4_VOCABULARY: u64 = 262_144;
pub const MHC4_STREAMS: u64 = 4;
pub const MHC4_HEADS: u64 = 8;
pub const EXPERT_WIDTH: u64 = 216;
pub const ROUTED_COUNT: u64 = 625;
pub const SHARED_COUNT: u64 = 1;
pub const EXPERT_COUNT: u64 = ROUTED_COUNT + SHARED_COUNT;
pub const ACTIVE_COUNT: u64 = 3;
pub const QKV_WIDTH: u64 = 640;
pub const GATE_WIDTH: u64 = 2 * EXPERT_WIDTH;
pub const MHC_INPUT: u64 = MHC4_STREAMS * MHC4_WIDTH;
pub const MHC_COEFS: u64 = MHC4_STREAMS * MHC4_STREAMS + 2 * MHC4_STREAMS;
pub const MHC4_PARAMS: u64 = MHC4_LAYERS
    * (MHC4_WIDTH * ROUTED_COUNT
        + MHC4_WIDTH * QKV_WIDTH
        + MHC4_WIDTH * MHC4_WIDTH
        + EXPERT_COUNT * MHC4_WIDTH * GATE_WIDTH
        + EXPERT_COUNT * EXPERT_WIDTH * MHC4_WIDTH
        + 2 * MHC4_WIDTH
        + 2 * MHC_INPUT * MHC_COEFS
        + 2 * MHC_COEFS
        + 8)
    + MHC4_WIDTH * MHC4_VOCABULARY
    + MHC4_WIDTH;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Mhc4Shape {
    pub patch: u64,
    pub width: u64,
    pub query: u64,
    pub streams: u64,
    pub heads: u64,
    pub hidden: u64,
    pub experts: u64,
    pub active: u64,
    pub visits: u64,
    pub levels: u64,
    pub radix: u64,
}

impl Mhc4Shape {
    pub const fn coeffs(self) -> u64 {
        let predictor =
            (self.streams * self.width) * (self.streams * self.streams + 2 * self.streams);
        let attention =
            self.width * ((self.heads + 2) * self.query) + (self.heads * self.query) * self.width;
        let experts = self.width * self.experts + (self.active + 1) * 3 * self.width * self.hidden;
        let encoder = self.patch * self.width;
        let decoder =
            self.width * self.query + self.patch * 2 * self.levels * self.radix * self.query;
        encoder + self.visits * (2 * predictor + attention + experts) + decoder
    }
}

pub const MHC4_SHAPE: Mhc4Shape = Mhc4Shape {
    patch: 64,
    width: MHC4_WIDTH,
    query: MHC4_QUERY,
    streams: MHC4_STREAMS,
    heads: MHC4_HEADS,
    hidden: EXPERT_WIDTH,
    // `experts` is the routed population; the shared expert is separate.
    experts: ROUTED_COUNT,
    active: ACTIVE_COUNT,
    visits: 7,
    levels: 6,
    radix: 8,
};

/// Dense contraction coefficients in one complete forward pass.
/// Selector, reduction, normalization, memory, launch, and synchronization
/// work remain outside this deliberately partial count.
pub const MHC4_COEFFS: u64 = MHC4_SHAPE.coeffs();

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mhc4_shape() {
        assert_eq!(MHC4_PARAMS, 1_177_723_160);
        assert_eq!(EXPERT_COUNT, 626);
        assert_eq!(QKV_WIDTH, (MHC4_HEADS + 2) * MHC4_QUERY);
        assert_eq!(MHC_INPUT, MHC4_STREAMS * MHC4_WIDTH);
        assert_eq!(MHC_COEFS, MHC4_STREAMS * MHC4_STREAMS + 2 * MHC4_STREAMS);
    }
}
