//! Dense row-major storage for ENN observation columns.

use ndarray::{Array1, Array2, ArrayView2, Axis};
use std::sync::Arc;

use crate::error::ENNError;

/// Gathered training rows: `x`, `y`, optional `yvar`.
pub(crate) type TrainRows = (Array2<f64>, Array2<f64>, Option<Array2<f64>>);

pub(crate) const CHUNK_ROWS: usize = 256;

#[derive(Debug, Clone)]
pub(crate) struct RowStorage {
    slabs: Vec<Arc<[f64]>>,
    tail: Arc<Vec<f64>>,
    nrows: usize,
    ncols: usize,
}

impl RowStorage {
    pub(crate) fn from_array(a: Array2<f64>) -> Self {
        let (nrows, ncols) = a.dim();
        let a = a.as_standard_layout().into_owned();
        let slice = a.as_slice().expect("standard layout slice");
        let mut slabs = Vec::new();
        let mut offset = 0;
        if ncols > 0 {
            let chunk_elems = CHUNK_ROWS * ncols;
            let full_chunks = nrows / CHUNK_ROWS;
            slabs.reserve(full_chunks);
            for _ in 0..full_chunks {
                slabs.push(Arc::from(&slice[offset..offset + chunk_elems]));
                offset += chunk_elems;
            }
        }
        let remainder = &slice[offset..];
        let mut tail_vec =
            Vec::with_capacity(
                remainder
                    .len()
                    .max(if ncols > 0 { CHUNK_ROWS * ncols } else { 0 }),
            );
        tail_vec.extend_from_slice(remainder);
        Self {
            slabs,
            tail: Arc::new(tail_vec),
            nrows,
            ncols,
        }
    }

    pub(crate) fn nrows(&self) -> usize {
        self.nrows
    }

    pub(crate) fn ncols(&self) -> usize {
        self.ncols
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.nrows == 0
    }

    pub(crate) fn row_slice(&self, i: usize) -> &[f64] {
        assert!(i < self.nrows, "row index {i} out of bounds {}", self.nrows);
        if self.ncols == 0 {
            return &[];
        }
        let chunk_idx = i / CHUNK_ROWS;
        let row_in_chunk = i % CHUNK_ROWS;
        let start = row_in_chunk * self.ncols;
        let end = start + self.ncols;
        if chunk_idx < self.slabs.len() {
            &self.slabs[chunk_idx][start..end]
        } else {
            &self.tail[start..end]
        }
    }

    pub(crate) fn row_vec(&self, i: usize) -> Array1<f64> {
        Array1::from(self.row_slice(i).to_vec())
    }

    pub(crate) fn gather_rows(&self, indices: &[usize]) -> Array2<f64> {
        let mut out = Array2::zeros((indices.len(), self.ncols));
        if self.ncols == 0 {
            return out;
        }
        for (new_i, &old_i) in indices.iter().enumerate() {
            let slice = self.row_slice(old_i);
            out.row_mut(new_i).assign(&ndarray::ArrayView1::from(slice));
        }
        out
    }

    pub(crate) fn to_array(&self) -> Array2<f64> {
        let mut out = Array2::<f64>::zeros((self.nrows, self.ncols));
        if self.nrows == 0 || self.ncols == 0 {
            return out;
        }
        if let Some(out_slice) = out.as_slice_mut() {
            let chunk_elems = CHUNK_ROWS * self.ncols;
            let mut offset = 0;
            for slab in &self.slabs {
                out_slice[offset..offset + chunk_elems].copy_from_slice(slab);
                offset += chunk_elems;
            }
            let tail_rows = self.nrows - self.slabs.len() * CHUNK_ROWS;
            let tail_len = tail_rows * self.ncols;
            if tail_len > 0 {
                out_slice[offset..offset + tail_len].copy_from_slice(&self.tail[..tail_len]);
            }
        } else {
            for (i, mut row) in out.axis_iter_mut(Axis(0)).enumerate() {
                row.assign(&ndarray::ArrayView1::from(self.row_slice(i)));
            }
        }
        out
    }

    pub(crate) fn view(&self) -> Option<ndarray::ArrayView2<'_, f64>> {
        if self.slabs.is_empty() {
            ndarray::ArrayView2::from_shape(
                (self.nrows, self.ncols),
                &self.tail[..self.nrows * self.ncols],
            )
            .ok()
        } else {
            None
        }
    }

    pub(crate) fn view_owned(&self) -> std::borrow::Cow<'_, [f64]> {
        if self.slabs.is_empty() {
            std::borrow::Cow::Borrowed(&self.tail[..self.nrows * self.ncols])
        } else {
            std::borrow::Cow::Owned(self.to_array().into_raw_vec_and_offset().0)
        }
    }

    pub(crate) fn push_rows(mut self, extra: &ArrayView2<f64>) -> Result<Self, ENNError> {
        if extra.ncols() != self.ncols {
            return Err(ENNError::InvalidShape {
                expected: vec![self.nrows, self.ncols],
                got: vec![extra.nrows(), extra.ncols()],
            });
        }
        let n_extra = extra.nrows();
        if n_extra == 0 {
            return Ok(self);
        }
        if self.ncols == 0 {
            self.nrows += n_extra;
            return Ok(self);
        }

        let current_tail_rows = self.tail.len() / self.ncols;
        let needed = CHUNK_ROWS - current_tail_rows;

        if n_extra < needed {
            let tail_mut = Arc::make_mut(&mut self.tail);
            for row in extra.axis_iter(Axis(0)) {
                tail_mut.extend(row.iter().copied());
            }
            self.nrows += n_extra;
            return Ok(self);
        }

        let mut extra_row_start = 0;
        if current_tail_rows > 0 {
            let tail_mut = Arc::make_mut(&mut self.tail);
            for row in extra
                .slice_axis(Axis(0), ndarray::Slice::from(..needed))
                .axis_iter(Axis(0))
            {
                tail_mut.extend(row.iter().copied());
            }
            self.slabs.push(Arc::from(&tail_mut[..]));
            tail_mut.clear();
            extra_row_start = needed;
        }

        let chunk_elems = CHUNK_ROWS * self.ncols;
        while extra_row_start + CHUNK_ROWS <= n_extra {
            let end = extra_row_start + CHUNK_ROWS;
            let chunk_view = extra.slice_axis(Axis(0), ndarray::Slice::from(extra_row_start..end));
            let mut chunk_vec = Vec::with_capacity(chunk_elems);
            for row in chunk_view.axis_iter(Axis(0)) {
                chunk_vec.extend(row.iter().copied());
            }
            self.slabs.push(Arc::from(chunk_vec.into_boxed_slice()));
            extra_row_start = end;
        }

        let rem = n_extra - extra_row_start;
        if rem > 0 {
            let tail_mut = Arc::make_mut(&mut self.tail);
            let rem_view =
                extra.slice_axis(Axis(0), ndarray::Slice::from(extra_row_start..n_extra));
            for row in rem_view.axis_iter(Axis(0)) {
                tail_mut.extend(row.iter().copied());
            }
        }

        self.nrows += n_extra;
        Ok(self)
    }
}

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
        let s3 = s1.push_rows(&array![[3.0, 4.0]].view()).unwrap();
        assert_eq!(s2.nrows(), 1);
        assert_eq!(s2.to_array(), array![[1.0, 2.0]]);
        assert_eq!(s3.nrows(), 2);
        assert_eq!(s3.to_array(), array![[1.0, 2.0], [3.0, 4.0]]);
    }
}
