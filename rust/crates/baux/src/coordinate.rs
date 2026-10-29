//! Natural-coordinate and unit-hypercube conversion.

use ndarray::{Array2, ArrayView2};

fn row_major(view: &ArrayView2<f64>) -> Vec<f64> {
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
    let flat_bounds = flat_bounds(bounds, x_unit.ncols());
    assert!(crate::coordinate_flat::from_unit(
        shape.0,
        shape.1,
        &mut values,
        &flat_bounds,
    ));
    Array2::from_shape_vec(shape, values).expect("coordinate transform preserves shape")
}

pub fn to_unit(x: &ArrayView2<f64>, bounds: &ArrayView2<f64>) -> Array2<f64> {
    let shape = (x.nrows(), x.ncols());
    if x.is_empty() {
        return Array2::zeros(shape);
    }
    let mut values = row_major(x);
    let flat_bounds = flat_bounds(bounds, x.ncols());
    assert!(crate::coordinate_flat::to_unit(
        shape.0,
        shape.1,
        &mut values,
        &flat_bounds,
    ));
    Array2::from_shape_vec(shape, values).expect("coordinate transform preserves shape")
}
