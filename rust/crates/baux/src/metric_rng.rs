//! Ported from yubo-research/enn, commit
//! 506e98c506eeb849cffbf53d9ddf3a3a799c6830. NumPy `Generator(PCG64(seed))` for
//! `integers(0, high)`.

use crate::hash::Key;
use crate::metric_seed::seed_quad;

const MULT_HI: u128 = 2549297995355413924;
const MULT_LO: u128 = 4865540595714422341;
const MULT: u128 = (MULT_HI << 64) | MULT_LO;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MetricRng {
    state: u128,
    inc: u128,
    has_u32: bool,
    u32_buf: u32,
}

impl MetricRng {
    pub fn from_seed(seed: u64) -> Self {
        let words = seed_quad(seed);
        let initstate = (u128::from(words[0]) << 64) | u128::from(words[1]);
        let initseq = (u128::from(words[2]) << 64) | u128::from(words[3]);
        let inc = (initseq << 1) | 1;
        let mut state = 0u128;
        state = state.wrapping_mul(MULT).wrapping_add(inc);
        state = state.wrapping_add(initstate);
        state = state.wrapping_mul(MULT).wrapping_add(inc);
        Self {
            state,
            inc,
            has_u32: false,
            u32_buf: 0,
        }
    }

    #[inline(always)]
    pub fn from_key(key: Key) -> Self {
        Self::from_seed(key.seed())
    }

    pub(crate) const fn snapshot(&self) -> (u128, u128, bool, u32) {
        (self.state, self.inc, self.has_u32, self.u32_buf)
    }

    pub(crate) const fn restore(state: u128, inc: u128, has_buf: bool, buf_val: u32) -> Self {
        Self {
            state,
            inc: inc | 1,
            has_u32: has_buf,
            u32_buf: buf_val,
        }
    }

    /// Pure state transition advancing one 64-bit PCG64 word.
    #[inline(always)]
    pub const fn step_word(mut self) -> (Self, u64) {
        self.state = self.state.wrapping_mul(MULT).wrapping_add(self.inc);
        let high = (self.state >> 64) as u64;
        let low = self.state as u64;
        let word = (high ^ low).rotate_right((high >> 58) as u32);
        (self, word)
    }

    #[inline(always)]
    const fn step_half(mut self) -> (Self, u32) {
        if self.has_u32 {
            self.has_u32 = false;
            return (self, self.u32_buf);
        }
        let (next_self, next) = self.step_word();
        self = next_self;
        self.has_u32 = true;
        self.u32_buf = (next >> 32) as u32;
        (self, next as u32)
    }

    /// Pure state transition for `Generator.integers(0, high)` (`high >= 1`).
    pub fn step_high(self, high: u64) -> (Self, u64) {
        debug_assert!(high >= 1);
        let rng = high - 1;
        if rng == 0 {
            return (self, 0);
        }
        if rng <= u64::from(u32::MAX) {
            if rng == u64::from(u32::MAX) {
                let (next, val) = self.step_half();
                return (next, u64::from(val));
            }
            let (next, val) = self.lemire_low(rng as u32);
            return (next, u64::from(val));
        }
        self.lemire_wide(rng)
    }

    pub fn next_u64(&mut self) -> u64 {
        let (next, word) = self.step_word();
        *self = next;
        word
    }

    /// `Generator.integers(0, high)` for `high >= 1` (exclusive upper bound).
    pub fn integers_high(&mut self, high: u64) -> u64 {
        let (next, val) = self.step_high(high);
        *self = next;
        val
    }

    fn lemire_low(mut self, rng: u32) -> (Self, u32) {
        let rng_excl = u64::from(rng) + 1;
        let (s, u) = self.step_half();
        self = s;
        let mut m = u64::from(u) * rng_excl;
        let mut leftover = m as u32;
        if u64::from(leftover) < rng_excl {
            let threshold = (u32::MAX - rng) % (rng_excl as u32);
            while leftover < threshold {
                let (s_next, u_next) = self.step_half();
                self = s_next;
                m = u64::from(u_next) * rng_excl;
                leftover = m as u32;
            }
        }
        (self, (m >> 32) as u32)
    }

    fn lemire_wide(mut self, rng: u64) -> (Self, u64) {
        let rng_excl = rng + 1;
        let (s, u) = self.step_word();
        self = s;
        let mut m = (u as u128).wrapping_mul(u128::from(rng_excl));
        let mut leftover = m as u64;
        if leftover < rng_excl {
            let threshold = (u64::MAX - rng) % rng_excl;
            while leftover < threshold {
                let (s_next, u_next) = self.step_word();
                self = s_next;
                m = (u_next as u128).wrapping_mul(u128::from(rng_excl));
                leftover = m as u64;
            }
        }
        (self, (m >> 64) as u64)
    }
}
