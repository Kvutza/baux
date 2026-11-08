//! Natural-coordinate and unit-hypercube conversion.

use ndarray::{Array1, Array2, ArrayView2};

pub use crate::coord_space::{CoordPoint, Coords, DomainSpace, OwnedCoords, UnitSpace};
use crate::error::ENNError;

/// Coordinate frame owning `[num_dim, 2]` natural reference intervals,
/// optional boundary clamping, and affine unit transforms.
#[derive(Debug, Clone)]
pub struct DomainFrame {
    bounds: Array2<f64>,
    num_dim: usize,
    clamped: bool,
}

impl DomainFrame {
    pub fn new(bounds: Array2<f64>) -> Result<Self, ENNError> {
        Self::with_clamping(bounds, true)
    }

    pub fn unbounded(bounds: Array2<f64>) -> Result<Self, ENNError> {
        Self::with_clamping(bounds, false)
    }

    fn with_clamping(bounds: Array2<f64>, clamped: bool) -> Result<Self, ENNError> {
        let num_dim = bounds.nrows();
        if bounds.ncols() != 2 {
            return Err(ENNError::InvalidShape {
                expected: vec![num_dim, 2],
                got: vec![num_dim, bounds.ncols()],
            });
        }
        Ok(Self {
            bounds,
            num_dim,
            clamped,
        })
    }

    /// Construct an unconstrained coordinate frame from an MLP's `(fan_in,
    /// fan_out)` layer dimensions using Kaiming uniform weight bounds
    /// `[-sqrt(3/fan_in), +sqrt(3/fan_in)]` and bias bounds
    /// `[-1/sqrt(fan_in), +1/sqrt(fan_in)]`.
    pub fn kaiming_mlp(layers: &[(usize, usize)]) -> Result<Self, ENNError> {
        if layers.is_empty() || layers.iter().any(|&(i, o)| i == 0 || o == 0) {
            return Err(ENNError::InvalidParameter(
                "MLP layers must be non-empty with positive fan_in and fan_out".into(),
            ));
        }
        let total: usize = layers.iter().map(|&(i, o)| i * o + o).sum();
        let mut bounds = Array2::zeros((total, 2));
        let mut offset = 0;
        for &(fan_in, fan_out) in layers {
            let w_bound = (3.0 / (fan_in as f64)).sqrt();
            let b_bound = 1.0 / (fan_in as f64).sqrt();
            let w_len = fan_in * fan_out;
            for idx in offset..offset + w_len {
                bounds[[idx, 0]] = -w_bound;
                bounds[[idx, 1]] = w_bound;
            }
            offset += w_len;
            for idx in offset..offset + fan_out {
                bounds[[idx, 0]] = -b_bound;
                bounds[[idx, 1]] = b_bound;
            }
            offset += fan_out;
        }
        Self::unbounded(bounds)
    }

    pub fn clamped(&self) -> bool {
        self.clamped
    }

    pub fn bounds(&self) -> &Array2<f64> {
        &self.bounds
    }

    pub fn num_dim(&self) -> usize {
        self.num_dim
    }

    pub fn to_unit(&self, x: &ArrayView2<f64>) -> Array2<f64> {
        to_unit(x, &self.bounds.view())
    }

    pub fn from_unit(&self, unit: &ArrayView2<f64>) -> Array2<f64> {
        from_unit(unit, &self.bounds.view())
    }

    /// Type-checked conversion from [`DomainSpace`] into [`UnitSpace`].
    pub fn encode_unit(&self, x: Coords<'_, DomainSpace>) -> OwnedCoords<UnitSpace> {
        OwnedCoords::new(self.to_unit(x.view()))
    }

    /// Type-checked conversion from [`UnitSpace`] into [`DomainSpace`].
    pub fn decode_domain(&self, unit: Coords<'_, UnitSpace>) -> OwnedCoords<DomainSpace> {
        OwnedCoords::new(self.from_unit(unit.view()))
    }

    pub fn natural_point(&self, unit: &Array1<f64>) -> Array1<f64> {
        let row = unit.clone().insert_axis(ndarray::Axis(0));
        let natural = from_unit(&row.view(), &self.bounds.view());
        natural.row(0).to_owned()
    }

    pub fn unit_center(&self) -> Array1<f64> {
        Array1::from_elem(self.num_dim, 0.5)
    }
}

fn row_major(view: &ArrayView2<f64>) -> Vec<f64> {
    if let Some(slice) = view.as_slice() {
        return slice.to_vec();
    }
    let mut values = Vec::with_capacity(view.len());
    for row in 0..view.nrows() {
        for column in 0..view.ncols() {
            values.push(view[[row, column]]);
        }
    }
    values
}

fn flat_bounds(bounds: &ArrayView2<f64>, columns: usize) -> Vec<f64> {
    let mut values = Vec::with_capacity(columns * 2);
    for column in 0..columns {
        values.push(bounds[[column, 0]]);
        values.push(bounds[[column, 1]]);
    }
    values
}

pub fn from_unit(x_unit: &ArrayView2<f64>, bounds: &ArrayView2<f64>) -> Array2<f64> {
    let shape = (x_unit.nrows(), x_unit.ncols());
    if x_unit.is_empty() {
        return Array2::zeros(shape);
    }
    let mut values = row_major(x_unit);
    if let Some(b_slice) = bounds.as_slice() {
        assert!(crate::coordinate_flat::from_unit(
            shape.0,
            shape.1,
            &mut values,
            &b_slice[..shape.1 * 2],
        ));
    } else {
        let fb = flat_bounds(bounds, x_unit.ncols());
        assert!(crate::coordinate_flat::from_unit(
            shape.0,
            shape.1,
            &mut values,
            &fb,
        ));
    }
    Array2::from_shape_vec(shape, values).expect("coordinate transform preserves shape")
}

pub fn to_unit(x: &ArrayView2<f64>, bounds: &ArrayView2<f64>) -> Array2<f64> {
    let shape = (x.nrows(), x.ncols());
    if x.is_empty() {
        return Array2::zeros(shape);
    }
    let mut values = row_major(x);
    if let Some(b_slice) = bounds.as_slice() {
        assert!(crate::coordinate_flat::to_unit(
            shape.0,
            shape.1,
            &mut values,
            &b_slice[..shape.1 * 2],
        ));
    } else {
        let fb = flat_bounds(bounds, x.ncols());
        assert!(crate::coordinate_flat::to_unit(
            shape.0,
            shape.1,
            &mut values,
            &fb,
        ));
    }
    Array2::from_shape_vec(shape, values).expect("coordinate transform preserves shape")
}
