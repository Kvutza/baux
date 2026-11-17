//! Dense 64-byte-aligned FBIP persistent row storage for ENN observation
//! columns.

use ndarray::Array2;

pub use crate::scratch::ImLog as RowStorage;

/// Gathered training rows: `x`, `y`, optional `yvar`.
pub(crate) type TrainRows = (Array2<f64>, Array2<f64>, Option<Array2<f64>>);

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn storage_basic() {
        let a = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let storage = RowStorage::from_array(a.clone());
        assert_eq!(storage.nrows(), 3);
        assert_eq!(storage.ncols(), 2);
        assert_eq!(storage.row_slice(0), &[1.0, 2.0]);
        assert_eq!(storage.row_slice(1), &[3.0, 4.0]);
        assert_eq!(storage.row_slice(2), &[5.0, 6.0]);
        assert_eq!(storage.to_array(), a);
        let gathered = storage.gather_rows(&[2, 0]);
        assert_eq!(gathered, array![[5.0, 6.0], [1.0, 2.0]]);
    }

    #[test]
    fn storage_cow() {
        let a = array![[1.0, 2.0]];
        let s1 = RowStorage::from_array(a);
        let s2 = s1.clone();
        assert!(s1.shares_storage(&s2));
        let s3 = s1.push_rows(&array![[3.0, 4.0]].view()).unwrap();
        assert_eq!(s2.nrows(), 1);
        assert_eq!(s2.to_array(), array![[1.0, 2.0]]);
        assert_eq!(s3.nrows(), 2);
        assert_eq!(s3.to_array(), array![[1.0, 2.0], [3.0, 4.0]]);
    }
}
