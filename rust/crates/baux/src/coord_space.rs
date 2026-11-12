//! Zero-cost phantom coordinate space tags (`UnitSpace` vs `DomainSpace`) and
//! reversible non-Cartesian manifold embeddings.

mod embed;

use core::marker::PhantomData;

use ndarray::{Array2, ArrayView1, ArrayView2};

pub use embed::{Categorical, LogEmbed, PeriodicEmbed, SimplexEmbed};

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

impl crate::coordinate::DomainFrame {
    /// Type-checked conversion from [`DomainSpace`] into [`UnitSpace`].
    pub fn encode_unit(&self, x: Coords<'_, DomainSpace>) -> OwnedCoords<UnitSpace> {
        OwnedCoords::new(self.to_unit(x.view()))
    }

    /// Type-checked conversion from [`UnitSpace`] into [`DomainSpace`].
    pub fn decode_domain(&self, unit: Coords<'_, UnitSpace>) -> OwnedCoords<DomainSpace> {
        OwnedCoords::new(self.from_unit(unit.view()))
    }
}

#[cfg(test)]
mod tests;
