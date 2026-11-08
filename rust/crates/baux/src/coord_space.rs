//! Zero-cost phantom coordinate space tags (`UnitSpace` vs `DomainSpace`).

use core::marker::PhantomData;
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

    #[inline(always)]
    pub fn into_view(self) -> ArrayView2<'a, f64> {
        self.0
    }

    #[inline(always)]
    pub fn nrows(&self) -> usize {
        self.0.nrows()
    }

    #[inline(always)]
    pub fn ncols(&self) -> usize {
        self.0.ncols()
    }
}

/// Zero-cost borrowed 1D coordinate point tagged with its coordinate space.
#[repr(transparent)]
#[derive(Clone, Copy, Debug)]
pub struct CoordPoint<'a, Space>(ArrayView1<'a, f64>, PhantomData<fn() -> Space>);

impl<'a, Space> CoordPoint<'a, Space> {
    #[inline(always)]
    pub fn new(view: ArrayView1<'a, f64>) -> Self {
        Self(view, PhantomData)
    }

    #[inline(always)]
    pub fn view(&self) -> &ArrayView1<'a, f64> {
        &self.0
    }
}

/// Owned 2D coordinate matrix statically tagged with its coordinate space.
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

    #[inline(always)]
    pub fn into_array(self) -> Array2<f64> {
        self.0
    }
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
    }
}
