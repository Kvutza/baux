//! Deterministic candidate streams used to describe procedural pools.

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PoolShape {
    arms: usize,
    count: usize,
}

impl PoolShape {
    pub fn new(arms: usize, count: usize) -> Result<Self, String> {
        if arms == 0 || count == 0 || count % arms != 0 {
            return Err("candidate pool count must be positive and divisible by arms".into());
        }
        Ok(Self { arms, count })
    }

    pub const fn legacy() -> Self {
        Self { arms: 1, count: 4 }
    }

    pub const fn count(self) -> usize {
        self.count
    }

    pub fn seed(self, root: u64, candidate: usize) -> u64 {
        let slots = self.count / self.arms;
        let arm = candidate / slots;
        let slot = candidate % slots;
        legacy_seed(arm_seed(root, arm), slot)
    }
}

pub use baux::semantics::random::splitmix64;

#[inline]
fn stream_word(low: u32, high: u32, element: u32) -> u32 {
    let mut value = low ^ element.wrapping_mul(0x9e37_79b9);
    value ^= value >> 16;
    value = value.wrapping_mul(0x7feb_352d);
    value ^= high;
    value = value.wrapping_mul(0x846c_a68b);
    value ^ (value >> 15)
}

fn arm_seed(root: u64, arm: usize) -> u64 {
    if arm == 0 {
        root
    } else {
        splitmix64(root ^ 0x7072_6f63_6172_6d31 ^ arm as u64)
    }
}

fn legacy_seed(root: u64, slot: usize) -> u64 {
    let stream = slot as u32 / 2;
    let low = root as u32;
    let high = (root >> 32) as u32;
    u64::from(stream_word(low, high, stream))
        | (u64::from(stream_word(high, low, stream ^ 0x9e37_79b9)) << 32)
}

/// Historical one-arm, four-slot seed mapping retained for recorded runs.
pub fn candidate_seed(root: u64, candidate: usize) -> u64 {
    PoolShape::legacy().seed(root, candidate)
}
