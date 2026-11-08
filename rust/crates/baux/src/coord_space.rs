//! Zero-cost phantom coordinate space tags (`UnitSpace` vs `DomainSpace`) and
//! reversible non-Cartesian manifold embeddings.

use core::marker::PhantomData;

use baux::semantics::categorical::{
    CatSpec, argmax_metric, log_decode, log_encode, log_width, periodic_decode, periodic_encode,
    rademacher, simplex_decode, simplex_encode, snap_code,
};
use ndarray::{Array2, ArrayView1, ArrayView2};

/// Zero-sized coordinate-space tag for the unit hypercube $[0, 1]^d$.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UnitSpace {}

/// Zero-sized coordinate-space tag for natural domain coordinates $[l_j, u_j]$.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DomainSpace {}

/// Zero-cost borrowed 2D coordinate matrix tagged with its coordinate space
/// (`UnitSpace` or `DomainSpace`).
#[repr(transparent)]
#[diagnostic::on_type_error(
    note = "coordinate space mismatch (`{Expected}` vs `{Found}`): use `DomainFrame::encode_unit` / `decode_domain` or an `Embedding` to convert between `DomainSpace` and `UnitSpace`"
)]
#[derive(Clone, Copy, Debug)]
pub struct Coords<'a, Space>(ArrayView2<'a, f64>, PhantomData<fn() -> Space>);

impl<'a, Space> Coords<'a, Space> {
    #[inline(always)]
    pub fn new(view: ArrayView2<'a, f64>) -> Self {
        Self(view, PhantomData)
    }

    #[inline(always)]
    pub fn view(&self) -> &ArrayView2<'a, f64> {
        &self.0
    }
}

/// Zero-cost borrowed 1D coordinate point tagged with its coordinate space.
#[repr(transparent)]
#[derive(Clone, Copy, Debug)]
pub struct CoordPoint<'a, Space>(pub ArrayView1<'a, f64>, PhantomData<fn() -> Space>);

/// Owned 2D coordinate matrix statically tagged with its coordinate space.
#[diagnostic::on_type_error(
    note = "coordinate space mismatch (`{Expected}` vs `{Found}`): use `DomainFrame::encode_unit` / `decode_domain` or an `Embedding` to convert between `DomainSpace` and `UnitSpace`"
)]
#[derive(Clone, Debug, PartialEq)]
pub struct OwnedCoords<Space>(Array2<f64>, PhantomData<fn() -> Space>);

impl<Space> OwnedCoords<Space> {
    #[inline(always)]
    pub fn new(array: Array2<f64>) -> Self {
        Self(array, PhantomData)
    }

    #[inline(always)]
    pub fn borrow(&self) -> Coords<'_, Space> {
        Coords::new(self.0.view())
    }

    #[inline(always)]
    pub fn as_array(&self) -> &Array2<f64> {
        &self.0
    }
}

/// Reversible manifold embedding and retraction mapping a domain value into
/// `[0, 1]^m` (`UnitSpace`).
pub trait Embedding: Send + Sync {
    type Value;
    fn latent_dim(&self) -> usize;
    fn is_tied(&self) -> bool {
        self.latent_dim() > 1
    }
    fn encode(&self, value: &Self::Value, out: &mut [f64]);
    fn decode(&self, latent: &[f64], weights: Option<&[f64]>) -> Self::Value;
}

/// Isometric `S^1` periodic circle embedding mapping `[low, high)` into `[0,
/// 1]^2`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PeriodicEmbed {
    pub low: f64,
    pub high: f64,
}

impl Embedding for PeriodicEmbed {
    type Value = f64;
    fn latent_dim(&self) -> usize {
        2
    }
    fn encode(&self, value: &f64, out: &mut [f64]) {
        let enc = periodic_encode(&[*value], self.low, self.high).expect("periodic_encode");
        out[..2].copy_from_slice(&enc);
    }
    fn decode(&self, latent: &[f64], _weights: Option<&[f64]>) -> f64 {
        periodic_decode(&latent[..2], self.low, self.high).expect("periodic_decode")[0]
    }
}

/// Fisher-Rao Hellinger square-root simplex embedding on `\Delta^{K-1}` into
/// `S^{K-1}_+ \subset [0, 1]^K`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SimplexEmbed<const K: usize>;

impl<const K: usize> Embedding for SimplexEmbed<K> {
    type Value = [f64; K];
    fn latent_dim(&self) -> usize {
        K
    }
    fn is_tied(&self) -> bool {
        true
    }
    fn encode(&self, value: &[f64; K], out: &mut [f64]) {
        let enc = simplex_encode(value, 1, K).expect("simplex_encode");
        out[..K].copy_from_slice(&enc);
    }
    fn decode(&self, latent: &[f64], _weights: Option<&[f64]>) -> [f64; K] {
        simplex_decode(&latent[..K], 1, K)
            .expect("simplex_decode")
            .try_into()
            .expect("simplex length K")
    }
}

/// Multiplicative Riemannian log-scale embedding on `[low, high] \subset (0,
/// \infty)`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LogEmbed {
    pub low: f64,
    pub high: f64,
}

impl Embedding for LogEmbed {
    type Value = f64;
    fn latent_dim(&self) -> usize {
        1
    }
    fn encode(&self, value: &f64, out: &mut [f64]) {
        out[0] = log_encode(&[*value], self.low, self.high).expect("log_encode")[0];
    }
    fn decode(&self, latent: &[f64], _weights: Option<&[f64]>) -> f64 {
        log_decode(&latent[..1], self.low, self.high).expect("log_decode")[0]
    }
}

/// Isometric Rademacher categorical embedding mapping `0..card` into `[0,
/// 1]^bits`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Categorical {
    pub card: usize,
    pub bits: usize,
    pub seed: u64,
}

impl Categorical {
    pub fn new(card: usize, seed: u64) -> Self {
        let bits = log_width(card.max(1), 48).expect("log_width");
        Self { card, bits, seed }
    }
}

impl Embedding for Categorical {
    type Value = usize;
    fn latent_dim(&self) -> usize {
        self.bits
    }
    fn encode(&self, value: &usize, out: &mut [f64]) {
        let spec = CatSpec {
            variable: 0,
            width: self.bits,
            seed: self.seed,
            sep: (self.bits as f64).sqrt(),
        };
        let code = rademacher(&[*value as i64], spec).expect("rademacher");
        for (dst, &c) in out[..self.bits].iter_mut().zip(&code) {
            *dst = 0.5 * (c + 1.0);
        }
    }
    fn decode(&self, latent: &[f64], weights: Option<&[f64]>) -> usize {
        let centered: Vec<f64> = latent[..self.bits].iter().map(|&u| 2.0 * u - 1.0).collect();
        let domain: Vec<i64> = (0..self.card as i64).collect();
        let spec = CatSpec {
            variable: 0,
            width: self.bits,
            seed: self.seed,
            sep: (self.bits as f64).sqrt(),
        };
        if let Some(w) = weights {
            let codes = rademacher(&domain, spec).expect("rademacher");
            argmax_metric(&centered, &codes, &w[..self.bits], 1, self.card, self.bits)
                .expect("argmax_metric")[0] as usize
        } else {
            snap_code(&centered, 1, &domain, spec).expect("snap_code")[0] as usize
        }
    }
}

impl<A: Embedding, B: Embedding> Embedding for (A, B) {
    type Value = (A::Value, B::Value);
    fn latent_dim(&self) -> usize {
        self.0.latent_dim() + self.1.latent_dim()
    }
    fn is_tied(&self) -> bool {
        self.0.is_tied() || self.1.is_tied()
    }
    fn encode(&self, value: &Self::Value, out: &mut [f64]) {
        let split = self.0.latent_dim();
        self.0.encode(&value.0, &mut out[..split]);
        self.1
            .encode(&value.1, &mut out[split..split + self.1.latent_dim()]);
    }
    fn decode(&self, latent: &[f64], weights: Option<&[f64]>) -> Self::Value {
        let split = self.0.latent_dim();
        let end = split + self.1.latent_dim();
        let (wa, wb) = match weights {
            Some(w) => (Some(&w[..split]), Some(&w[split..end])),
            None => (None, None),
        };
        (
            self.0.decode(&latent[..split], wa),
            self.1.decode(&latent[split..end], wb),
        )
    }
}

/// Encode a batch of domain values into an [`OwnedCoords<UnitSpace>`] matrix.
pub fn encode_batch<C: Embedding>(embed: &C, values: &[C::Value]) -> OwnedCoords<UnitSpace> {
    let dim = embed.latent_dim();
    let mut buf = vec![0.0; values.len() * dim];
    for (row, val) in values.iter().enumerate() {
        embed.encode(val, &mut buf[row * dim..(row + 1) * dim]);
    }
    OwnedCoords::new(Array2::from_shape_vec((values.len(), dim), buf).expect("encode_batch"))
}

/// Decode a borrowed [`Coords<UnitSpace>`] matrix back into domain values.
pub fn decode_batch<C: Embedding>(
    embed: &C,
    coords: Coords<'_, UnitSpace>,
    weights: Option<&[f64]>,
) -> Vec<C::Value> {
    let view = coords.view();
    (0..view.nrows())
        .map(|r| {
            let row = view.row(r);
            if let Some(slice) = row.as_slice() {
                embed.decode(slice, weights)
            } else {
                embed.decode(&row.to_vec(), weights)
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::coordinate::DomainFrame;
    use ndarray::array;

    #[test]
    fn phantom_coords() {
        let bounds = array![[-10.0, 10.0], [0.0, 4.0]];
        let frame = DomainFrame::new(bounds).unwrap();
        let raw = array![[0.0, 2.0], [10.0, 4.0]];
        let dom = Coords::<DomainSpace>::new(raw.view());
        let unit: OwnedCoords<UnitSpace> = frame.encode_unit(dom);
        assert_eq!(unit.as_array(), &array![[0.5, 0.5], [1.0, 1.0]]);
        let back: OwnedCoords<DomainSpace> = frame.decode_domain(unit.borrow());
        assert_eq!(back.as_array(), &raw);

        let embed = (
            PeriodicEmbed {
                low: 0.0,
                high: 6.0,
            },
            (
                SimplexEmbed::<3>,
                (
                    LogEmbed {
                        low: 1.0e-4,
                        high: 1.0,
                    },
                    Categorical::new(5, 42),
                ),
            ),
        );
        assert!(embed.is_tied());
        let samples = vec![
            (1.5, ([0.2, 0.3, 0.5], (1.0e-2, 2usize))),
            (4.5, ([0.6, 0.1, 0.3], (1.0e-3, 4usize))),
        ];
        let unit = encode_batch(&embed, &samples);
        assert_eq!(unit.as_array().ncols(), embed.latent_dim());
        assert!(unit.as_array().iter().all(|&u| (0.0..=1.0).contains(&u)));
        let weights = vec![1.0; embed.latent_dim()];
        let decoded = decode_batch(&embed, unit.borrow(), Some(&weights));
        for ((a0, (s0, (l0, c0))), (a1, (s1, (l1, c1)))) in samples.iter().zip(decoded.iter()) {
            assert!((a0 - a1).abs() < 1.0e-12);
            for (p0, p1) in s0.iter().zip(s1.iter()) {
                assert!((p0 - p1).abs() < 1.0e-12);
            }
            assert!((l0 - l1).abs() < 1.0e-12);
            assert_eq!(c0, c1);
        }
    }
}
