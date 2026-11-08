//! Immutable, splittable, foldable PRNG key for referentially transparent
//! transitions.

use super::random::{Prng, SM64_OFFSET, prng, splitmix64};

/// Immutable, splittable, foldable PRNG key for referentially transparent
/// transitions.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub struct Key(pub u64, pub u64);

impl Key {
    /// Construct a deterministic key from a 64-bit seed.
    #[inline(always)]
    pub fn new(seed: u64) -> Self {
        let hi = splitmix64(seed);
        let lo = splitmix64(hi ^ SM64_OFFSET);
        Self(hi, lo)
    }

    /// Construct a fresh key from OS entropy.
    #[inline]
    pub fn entropy() -> Self {
        Self::new(super::random::auto_seed())
    }

    /// Construct a deterministic key from a named domain tag.
    #[inline]
    pub fn domain(tag: &str) -> Self {
        Self::new(super::random::tune_seed(tag, "baux-key", 0, 0))
    }

    /// Construct a key directly from two 64-bit words.
    #[inline(always)]
    pub const fn from_pair(hi: u64, lo: u64) -> Self {
        Self(hi, lo)
    }

    /// Extract a canonical 64-bit seed from this key.
    #[inline(always)]
    pub fn seed(self) -> u64 {
        splitmix64(self.0 ^ self.1.rotate_left(31))
    }

    /// Deterministically split this key into two independent child keys.
    #[inline(always)]
    pub fn split(self) -> (Self, Self) {
        let left = self.fold(0x6A09_E667_F3BC_C908);
        let right = self.fold(0xBB67_AE85_84CA_A73B);
        (left, right)
    }

    /// Deterministically split this key into `n` independent child keys.
    pub fn split_n(self, n: usize) -> Vec<Self> {
        (0..n).map(|i| self.fold(i as u64 + 1)).collect()
    }

    /// Deterministically fold a 64-bit tag (stage, step, or region index) into
    /// this key.
    #[inline(always)]
    pub fn fold(self, tag: u64) -> Self {
        let hi = splitmix64(self.0 ^ tag.wrapping_mul(0x9E37_79B9_7F4A_7C15));
        let lo = splitmix64(self.1 ^ hi ^ tag.wrapping_mul(0xBF58_476D_1CE4_E5B9));
        Self(hi, lo)
    }

    /// Materialize a fresh deterministic [`Prng`] stream from this key.
    #[inline(always)]
    pub fn prng(self) -> Prng {
        prng(self.seed())
    }
}

/// Affine PRNG key enforcing single-use move semantics at compile time.
#[must_use = "an AffineKey is an affine PRNG token; consume it via split, step, fold, or prng"]
#[diagnostic::on_move(
    message = "`AffineKey` was already consumed and cannot be reused across transitions",
    label = "call `.split()` or `.step()` to derive independent child keys before moving `AffineKey`",
    note = "reusing an unsplit PRNG key introduces statistical correlation across steps or branches"
)]
#[derive(Debug, PartialEq, Eq, Hash, Default)]
pub struct AffineKey(Key);

impl !Copy for AffineKey {}
impl !Clone for AffineKey {}

impl AffineKey {
    /// Construct an affine key from a 64-bit seed.
    #[inline(always)]
    pub fn new(seed: u64) -> Self {
        Self(Key::new(seed))
    }

    /// Construct a fresh affine key from OS entropy.
    #[inline]
    pub fn entropy() -> Self {
        Self(Key::entropy())
    }

    /// Construct a deterministic affine key from a named domain tag.
    #[inline]
    pub fn domain(tag: &str) -> Self {
        Self(Key::domain(tag))
    }

    /// Wrap an existing [`Key`] in an affine [`AffineKey`].
    #[inline(always)]
    pub const fn from_key(key: Key) -> Self {
        Self(key)
    }

    /// Borrow the underlying [`Key`] without consuming the affine token.
    #[inline(always)]
    pub const fn as_key(&self) -> Key {
        self.0
    }

    /// Consume this affine key and return its inner [`Key`].
    #[inline(always)]
    pub const fn into_key(self) -> Key {
        self.0
    }

    /// Extract a canonical 64-bit seed from this key.
    #[inline(always)]
    pub fn seed(&self) -> u64 {
        self.0.seed()
    }

    /// Consume this key and split it into two independent affine child keys.
    #[inline(always)]
    pub fn split(self) -> (Self, Self) {
        let (l, r) = self.0.split();
        (Self(l), Self(r))
    }

    /// Consume this key and advance one transition step, returning `(draw_key,
    /// next_affine_key)`.
    #[inline(always)]
    pub fn step(self) -> (Key, Self) {
        let (draw, next) = self.0.split();
        (draw, Self(next))
    }

    /// Consume this key and split it into `n` independent affine child keys.
    pub fn split_n(self, n: usize) -> Vec<Self> {
        self.0.split_n(n).into_iter().map(Self).collect()
    }

    /// Consume this key and fold a 64-bit tag into a derived affine key.
    #[inline(always)]
    pub fn fold(self, tag: u64) -> Self {
        Self(self.0.fold(tag))
    }

    /// Consume this affine key to materialize a deterministic [`Prng`] stream.
    #[inline(always)]
    pub fn prng(self) -> Prng {
        self.0.prng()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::RngCore;

    #[test]
    fn key_determinism() {
        let k1 = Key::new(42);
        let k2 = Key::new(42);
        let k3 = Key::new(43);
        assert_eq!(k1, k2);
        assert_ne!(k1, k3);
        assert_eq!(Key::from_pair(k1.0, k1.1), k1);
        assert_eq!(k1.seed(), k2.seed());
        assert_ne!(k1.seed(), k3.seed());
        let mut r1 = k1.prng();
        let mut r2 = k2.prng();
        assert_eq!(r1.next_u64(), r2.next_u64());
    }

    #[test]
    fn key_split() {
        let k = Key::new(99);
        let (left, right) = k.split();
        assert_eq!(k.split(), (left, right));
        assert_ne!(left, right);
        assert_ne!(left, k);
        assert!(k.split_n(0).is_empty());
        let kids = k.split_n(4);
        assert_eq!(kids.len(), 4);
        assert_eq!(kids, k.split_n(4));
        assert_eq!(k.fold(1), k.fold(1));
        assert_ne!(k.fold(1), k.fold(2));
        assert_ne!(k.fold(1).fold(2), k.fold(2).fold(1));
    }

    #[test]
    fn affine_key() {
        let base = Key::new(77);
        let ak = AffineKey::new(77);
        assert_eq!(ak.as_key(), base);
        assert_eq!(ak.seed(), base.seed());
        let (l, r) = ak.split();
        assert_ne!(l, r);
        let (draw, next) = AffineKey::from_key(base).step();
        assert_eq!(draw, l.into_key());
        assert_eq!(next, r);
        let kids = next.split_n(3);
        assert_eq!(kids.len(), 3);
        let mut iter = kids.into_iter();
        let k0 = iter.next().unwrap().fold(5);
        let k1 = iter.next().unwrap();
        let k2 = iter.next().unwrap();
        assert_ne!(k0.seed(), k1.seed());
        let mut rng = k0.prng();
        assert_ne!(rng.next_u64(), k1.prng().next_u64());
        let _ = k2.into_key();
        assert_eq!(Key::domain("sphere"), Key::domain("sphere"));
        assert_ne!(Key::domain("sphere"), Key::domain("rosenbrock"));
        assert_eq!(
            AffineKey::domain("sphere").into_key(),
            Key::domain("sphere")
        );
        let _ = Key::entropy();
        let _ = AffineKey::entropy().into_key();
    }
}
