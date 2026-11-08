//! Objective optimization senses and niche-optimized domain identifiers.

use core::cmp::Ordering;
use core::num::{NonZeroU16, NonZeroU32, NonZeroU64};
use deser::{Deserialize, Serialize};

/// Canonicalized scalar utility where higher values are strictly better.
///
/// Uses IEEE-754 `total_cmp` so `BetterScore` forms a total order (`Ord`) and
/// can be stored directly inside a `BinaryHeap` without `NaN` panics.
#[repr(transparent)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct BetterScore(pub f64);

impl BetterScore {
    pub const NEG_INF: Self = Self(f64::NEG_INFINITY);

    #[inline(always)]
    pub fn raw(self) -> f64 {
        self.0
    }
}

impl Eq for BetterScore {}

impl PartialOrd for BetterScore {
    #[inline(always)]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for BetterScore {
    #[inline(always)]
    fn cmp(&self, other: &Self) -> Ordering {
        self.0.total_cmp(&other.0)
    }
}

/// Algebraic interpretation of scalar objective measurements.
pub trait ObjectiveSense: Copy + Clone + Send + Sync + 'static {
    /// Map a raw measurement `y` into canonical utility (`higher is better`).
    fn score(&self, raw: f64) -> BetterScore;

    /// Recover the natural objective value from a canonical score when
    /// invertible.
    fn invert(&self, score: BetterScore) -> f64;

    /// Optimistic confidence bound in canonical utility space.
    #[inline(always)]
    fn ucb(&self, mu: f64, sigma: f64, beta: f64) -> BetterScore {
        let base = self.score(mu).raw();
        BetterScore(beta.mul_add(sigma, base))
    }

    /// Check whether `candidate` improves upon `incumbent` by more than
    /// `margin`.
    #[inline(always)]
    fn improved(&self, candidate: f64, incumbent: f64, margin: f64) -> bool {
        self.score(candidate).raw() > self.score(incumbent).raw() + margin
    }
}

/// Zero-sized compile-time maximization sense ($\max_x f(x)$).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Maximize;

impl ObjectiveSense for Maximize {
    #[inline(always)]
    fn score(&self, raw: f64) -> BetterScore {
        BetterScore(raw)
    }

    #[inline(always)]
    fn invert(&self, score: BetterScore) -> f64 {
        score.raw()
    }
}

/// Zero-sized compile-time minimization sense ($\min_x f(x)$).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Minimize;

impl ObjectiveSense for Minimize {
    #[inline(always)]
    fn score(&self, raw: f64) -> BetterScore {
        BetterScore(-raw)
    }

    #[inline(always)]
    fn invert(&self, score: BetterScore) -> f64 {
        -score.raw()
    }
}

/// Target-matching sense ($\min_x |f(x) - \tau|$).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Target(pub f64);

impl ObjectiveSense for Target {
    #[inline(always)]
    fn score(&self, raw: f64) -> BetterScore {
        BetterScore(-(raw - self.0).abs())
    }

    #[inline(always)]
    fn invert(&self, score: BetterScore) -> f64 {
        self.0 - score.raw()
    }
}

/// Runtime-configurable optimization sense sharing the [`ObjectiveSense`]
/// algebra.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
#[deser(rename_all = "snake_case")]
pub enum Sense {
    #[default]
    Maximize,
    Minimize,
    Target(f64),
}

impl ObjectiveSense for Sense {
    #[inline(always)]
    fn score(&self, raw: f64) -> BetterScore {
        match *self {
            Self::Maximize => Maximize.score(raw),
            Self::Minimize => Minimize.score(raw),
            Self::Target(tau) => Target(tau).score(raw),
        }
    }

    #[inline(always)]
    fn invert(&self, score: BetterScore) -> f64 {
        match *self {
            Self::Maximize => Maximize.invert(score),
            Self::Minimize => Minimize.invert(score),
            Self::Target(tau) => Target(tau).invert(score),
        }
    }
}

/// Two-stage lexicographic preference over primary sense `A` and secondary `B`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Lexicographic<A, B> {
    pub primary: A,
    pub secondary: B,
}

impl<A: ObjectiveSense, B: ObjectiveSense> Lexicographic<A, B> {
    pub const fn new(primary: A, secondary: B) -> Self {
        Self { primary, secondary }
    }

    /// Compare two `(y0, y1)` pairs with primary tie tolerance `eps`.
    #[inline]
    pub fn compare(&self, a: (f64, f64), b: (f64, f64), eps: f64) -> Ordering {
        let sa = self.primary.score(a.0).raw();
        let sb = self.primary.score(b.0).raw();
        if (sa - sb).abs() > eps {
            sa.total_cmp(&sb)
        } else {
            self.secondary.score(a.1).cmp(&self.secondary.score(b.1))
        }
    }
}

/// Niche-optimized 0-based observation index (`size_of::<Option<ObsId>>() ==
/// 4`).
#[repr(transparent)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ObsId(NonZeroU32);

const _: () = assert!(core::mem::size_of::<Option<ObsId>>() == 4);

impl ObsId {
    #[inline(always)]
    pub fn new(index: usize) -> Self {
        Self::from_index(index).expect("observation index fits in NonZeroU32 niche")
    }

    #[inline(always)]
    pub fn from_index(index: usize) -> Option<Self> {
        let raw = u32::try_from(index).ok()?.checked_add(1)?;
        NonZeroU32::new(raw).map(Self)
    }

    #[inline(always)]
    pub fn index(self) -> usize {
        (self.0.get() - 1) as usize
    }
}

/// Niche-optimized 0-based trust-region slot identifier
/// (`size_of::<Option<RegionId>>() == 2`).
#[repr(transparent)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct RegionId(NonZeroU16);

const _: () = assert!(core::mem::size_of::<Option<RegionId>>() == 2);

impl RegionId {
    #[inline(always)]
    pub fn new(slot: usize) -> Self {
        Self::from_slot(slot).expect("region slot fits in NonZeroU16 niche")
    }

    #[inline(always)]
    pub fn from_slot(slot: usize) -> Option<Self> {
        let raw = u16::try_from(slot).ok()?.checked_add(1)?;
        NonZeroU16::new(raw).map(Self)
    }

    #[inline(always)]
    pub fn slot(self) -> usize {
        (self.0.get() - 1) as usize
    }
}

/// Niche-optimized 1-based evaluation ticket ID (`size_of::<Option<EvalId>>()
/// == 8`).
#[repr(transparent)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct EvalId(NonZeroU64);

const _: () = assert!(core::mem::size_of::<Option<EvalId>>() == 8);

impl EvalId {
    #[inline(always)]
    pub const fn new(raw: u64) -> Option<Self> {
        match NonZeroU64::new(raw) {
            Some(nz) => Some(Self(nz)),
            None => None,
        }
    }

    #[inline(always)]
    pub const fn get(self) -> u64 {
        self.0.get()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sense_algebra() {
        assert!(Maximize.improved(2.0, 1.0, 0.1));
        assert!(!Maximize.improved(0.5, 1.0, 0.1));
        assert!(Minimize.improved(0.5, 1.0, 0.1));
        assert!(!Minimize.improved(2.0, 1.0, 0.1));

        let target = Target(3.0);
        assert!(target.improved(3.05, 3.5, 0.01));
        assert_eq!(Sense::Minimize.invert(Sense::Minimize.score(4.25)), 4.25);
    }

    #[test]
    fn niche_sizes() {
        assert_eq!(core::mem::size_of::<Option<ObsId>>(), 4);
        assert_eq!(core::mem::size_of::<Option<RegionId>>(), 2);
        assert_eq!(core::mem::size_of::<Option<EvalId>>(), 8);
        let id = ObsId::new(0);
        assert_eq!(id.index(), 0);
        let r = RegionId::new(7);
        assert_eq!(r.slot(), 7);
    }

    #[test]
    fn lex_ordering() {
        let lex = Lexicographic::new(Maximize, Minimize);
        assert_eq!(lex.compare((1.0, 2.0), (1.01, 0.5), 0.05), Ordering::Less);
        assert_eq!(lex.compare((2.0, 5.0), (1.0, 0.1), 0.05), Ordering::Greater);
    }
}
