//! Extractable row-major coordinate transforms.

fn shape_ok(rows: usize, columns: usize, values: &[f64], bounds: &[f64]) -> bool {
    let Some(cells) = rows.checked_mul(columns) else {
        return false;
    };
    let Some(bound_cells) = columns.checked_mul(2) else {
        return false;
    };
    values.len() == cells && bounds.len() == bound_cells
}

pub fn from_unit(rows: usize, columns: usize, values: &mut [f64], bounds: &[f64]) -> bool {
    if !shape_ok(rows, columns, values, bounds) {
        return false;
    }
    let mut index = 0;
    while index < values.len() {
        let column = index % columns;
        let lower = bounds[column * 2];
        let upper = bounds[column * 2 + 1];
        values[index] = crate::coordinate_scalar::from_unit(values[index], lower, upper);
        index += 1;
    }
    true
}

pub fn to_unit(rows: usize, columns: usize, values: &mut [f64], bounds: &[f64]) -> bool {
    if !shape_ok(rows, columns, values, bounds) {
        return false;
    }
    let mut index = 0;
    while index < values.len() {
        let column = index % columns;
        let lower = bounds[column * 2];
        let upper = bounds[column * 2 + 1];
        values[index] = crate::coordinate_scalar::to_unit(values[index], lower, upper);
        index += 1;
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shapes() {
        let bounds = [0.0, 10.0, -5.0, 5.0];
        let mut values = [2.5, 0.0, 7.5, 2.5];
        assert!(to_unit(2, 2, &mut values, &bounds));
        assert_eq!(values, [0.25, 0.5, 0.75, 0.75]);
        assert!(from_unit(2, 2, &mut values, &bounds));
        assert_eq!(values, [2.5, 0.0, 7.5, 2.5]);

        assert!(!to_unit(3, 2, &mut values, &bounds));
        assert!(!to_unit(2, 2, &mut values, &bounds[..3]));
        assert!(to_unit(7, 0, &mut [], &[]));
    }
}
