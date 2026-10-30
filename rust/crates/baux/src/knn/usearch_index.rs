#[cfg(all(feature = "usearch", feature = "usearch-c-api"))]
compile_error!("features \"usearch\" and \"usearch-c-api\" are mutually exclusive");

use ndarray::{Array2, ArrayView1, ArrayView2};

use super::pad_k;
use crate::error::IndexError;

fn usearch_error(error: impl std::fmt::Display) -> IndexError {
    IndexError::InvalidParameter(error.to_string())
}

fn squared_l2(left: ArrayView1<'_, f64>, right: &[f64]) -> f64 {
    left.iter()
        .zip(right)
        .map(|(left, right)| {
            let delta = left - right;
            delta * delta
        })
        .sum()
}

fn rerank(
    query: ArrayView1<'_, f64>,
    keys: &[u64],
    rows: &[f64],
    num_dim: usize,
    k: usize,
) -> Result<Vec<(f64, i64)>, IndexError> {
    let mut neighbors = Vec::with_capacity(keys.len());
    for &key in keys {
        let row = usize::try_from(key)
            .ok()
            .and_then(|key| key.checked_mul(num_dim))
            .and_then(|start| start.checked_add(num_dim).map(|end| (start, end)))
            .and_then(|(start, end)| rows.get(start..end))
            .ok_or_else(|| {
                IndexError::InvalidParameter(format!("USearch returned unknown key {key}"))
            })?;
        neighbors.push((squared_l2(query, row), key as i64));
    }
    neighbors.sort_unstable_by(|left, right| {
        left.0
            .total_cmp(&right.0)
            .then_with(|| left.1.cmp(&right.1))
    });
    neighbors.truncate(k);
    Ok(neighbors)
}

#[cfg(feature = "usearch")]
mod binding {
    use super::{IndexError, usearch_error};
    use usearch::{Index, IndexOptions, MetricKind, ScalarKind, new_index};

    pub(super) struct USearchRawIndex {
        index: Index,
    }

    fn options(num_dim: usize) -> IndexOptions {
        IndexOptions {
            dimensions: num_dim,
            metric: MetricKind::L2sq,
            quantization: ScalarKind::F32,
            connectivity: 0,
            expansion_add: 0,
            expansion_search: 0,
            multi: false,
        }
    }

    impl USearchRawIndex {
        pub(super) fn new(num_dim: usize) -> Result<Self, IndexError> {
            let index = new_index(&options(num_dim)).map_err(usearch_error)?;
            Ok(Self { index })
        }

        pub(super) fn reserve(&mut self, capacity: usize) -> Result<(), IndexError> {
            self.index.reserve(capacity).map_err(usearch_error)
        }

        pub(super) fn add(&mut self, key: u64, vector: &[f32]) -> Result<(), IndexError> {
            self.index.add(key, vector).map_err(usearch_error)
        }

        pub(super) fn search(
            &self,
            query: &[f32],
            wanted: usize,
            exact: bool,
        ) -> Result<Vec<u64>, IndexError> {
            let matches = if exact {
                self.index.exact_search(query, wanted)
            } else {
                self.index.search(query, wanted)
            }
            .map_err(usearch_error)?;
            Ok(matches.keys)
        }

        pub(super) fn expansion_search(&self) -> usize {
            self.index.expansion_search()
        }

        pub(super) fn memory_bytes(&self) -> usize {
            self.index.memory_usage()
        }
    }
}

#[cfg(feature = "usearch-c-api")]
mod binding {
    use super::{IndexError, usearch_error};
    use std::ffi::{CStr, c_char, c_void};
    use std::ptr;

    const ERROR_LEN: usize = 512;

    unsafe extern "C" {
        #[link_name = "baux_usearch_new"]
        fn baux_new(num_dim: usize, error: *mut c_char, error_capacity: usize) -> *mut c_void;
        #[link_name = "baux_usearch_destroy"]
        fn baux_destroy(handle: *mut c_void);
        #[link_name = "baux_usearch_reserve"]
        fn baux_reserve(
            handle: *mut c_void,
            capacity: usize,
            error: *mut c_char,
            error_capacity: usize,
        ) -> bool;
        #[link_name = "baux_usearch_add"]
        fn baux_add(
            handle: *mut c_void,
            key: u64,
            vector: *const f32,
            num_dim: usize,
            error: *mut c_char,
            error_capacity: usize,
        ) -> bool;
        #[link_name = "baux_usearch_search"]
        fn baux_search(
            handle: *const c_void,
            query: *const f32,
            num_dim: usize,
            wanted: usize,
            exact: bool,
            out_keys: *mut u64,
            out_capacity: usize,
            out_count: *mut usize,
            error: *mut c_char,
            error_capacity: usize,
        ) -> bool;
        #[link_name = "baux_usearch_expansion_search"]
        fn baux_search2(handle: *const c_void) -> usize;
        #[link_name = "baux_usearch_memory_usage"]
        fn baux_usage(handle: *const c_void) -> usize;
    }

    fn error_buffer(buffer: &[c_char], fallback: &str) -> IndexError {
        let message = unsafe { CStr::from_ptr(buffer.as_ptr()) }
            .to_string_lossy()
            .trim()
            .to_owned();
        if message.is_empty() {
            usearch_error(fallback)
        } else {
            usearch_error(message)
        }
    }

    fn call_bool(ok: bool, error: &[c_char], fallback: &str) -> Result<(), IndexError> {
        if ok {
            Ok(())
        } else {
            Err(error_buffer(error, fallback))
        }
    }

    pub(super) struct USearchRawIndex {
        handle: *mut c_void,
    }

    unsafe impl Send for USearchRawIndex {}

    impl Drop for USearchRawIndex {
        fn drop(&mut self) {
            if !self.handle.is_null() {
                unsafe { baux_destroy(self.handle) };
                self.handle = ptr::null_mut();
            }
        }
    }

    impl USearchRawIndex {
        pub(super) fn new(num_dim: usize) -> Result<Self, IndexError> {
            let mut error = [0 as c_char; ERROR_LEN];
            let handle = unsafe { baux_new(num_dim, error.as_mut_ptr(), error.len()) };
            if handle.is_null() {
                return Err(error_buffer(
                    &error,
                    "failed to construct a USearch C API index",
                ));
            }
            Ok(Self { handle })
        }

        pub(super) fn reserve(&mut self, capacity: usize) -> Result<(), IndexError> {
            let mut error = [0 as c_char; ERROR_LEN];
            let ok =
                unsafe { baux_reserve(self.handle, capacity, error.as_mut_ptr(), error.len()) };
            call_bool(ok, &error, "failed to reserve USearch C API capacity")
        }

        pub(super) fn add(&mut self, key: u64, vector: &[f32]) -> Result<(), IndexError> {
            let mut error = [0 as c_char; ERROR_LEN];
            let ok = unsafe {
                baux_add(
                    self.handle,
                    key,
                    vector.as_ptr(),
                    vector.len(),
                    error.as_mut_ptr(),
                    error.len(),
                )
            };
            call_bool(ok, &error, "failed to add a USearch C API vector")
        }

        pub(super) fn search(
            &self,
            query: &[f32],
            wanted: usize,
            exact: bool,
        ) -> Result<Vec<u64>, IndexError> {
            let mut error = [0 as c_char; ERROR_LEN];
            let mut keys = vec![0u64; wanted];
            let mut count = 0usize;
            let ok = unsafe {
                baux_search(
                    self.handle,
                    query.as_ptr(),
                    query.len(),
                    wanted,
                    exact,
                    keys.as_mut_ptr(),
                    keys.len(),
                    &mut count,
                    error.as_mut_ptr(),
                    error.len(),
                )
            };
            call_bool(ok, &error, "USearch C API search failed")?;
            keys.truncate(count);
            Ok(keys)
        }

        pub(super) fn expansion_search(&self) -> usize {
            unsafe { baux_search2(self.handle) }
        }

        pub(super) fn memory_bytes(&self) -> usize {
            unsafe { baux_usage(self.handle) }
        }
    }
}

#[cfg(feature = "usearch")]
use binding::USearchRawIndex;
#[cfg(feature = "usearch-c-api")]
use binding::USearchRawIndex;

pub(crate) struct USearchIndex {
    index: USearchRawIndex,
    rows: Vec<f64>,
    num_dim: usize,
}

impl USearchIndex {
    pub(crate) fn new(num_dim: usize, train_scaled: &ArrayView2<f64>) -> Result<Self, IndexError> {
        if num_dim == 0 {
            return Err(IndexError::InvalidParameter(
                "USearch requires at least one dimension".to_string(),
            ));
        }
        let index = USearchRawIndex::new(num_dim)?;
        let mut implementation = Self {
            index,
            rows: Vec::new(),
            num_dim,
        };
        implementation.add(train_scaled, 0)?;
        Ok(implementation)
    }

    pub(crate) fn len(&self) -> usize {
        self.rows.len() / self.num_dim
    }

    pub(crate) fn memory_bytes(&self) -> usize {
        self.index
            .memory_bytes()
            .saturating_add(self.rows.len().saturating_mul(std::mem::size_of::<f64>()))
    }

    pub(crate) fn rebuild(&mut self, train_scaled: &ArrayView2<f64>) -> Result<(), IndexError> {
        *self = Self::new(self.num_dim, train_scaled)?;
        Ok(())
    }

    pub(crate) fn add(
        &mut self,
        rows_scaled: &ArrayView2<f64>,
        start_key: u64,
    ) -> Result<(), IndexError> {
        if rows_scaled.ncols() != self.num_dim {
            return Err(IndexError::InvalidShape {
                expected: self.num_dim,
                got: rows_scaled.ncols(),
            });
        }
        if start_key != self.len() as u64 {
            return Err(IndexError::InvalidParameter(format!(
                "USearch keys must be contiguous: expected {}, got {start_key}",
                self.len()
            )));
        }
        if rows_scaled.is_empty() {
            return Ok(());
        }

        self.index
            .reserve(self.len() + rows_scaled.nrows())
            .map_err(usearch_error)?;
        for (offset, row) in rows_scaled.rows().into_iter().enumerate() {
            let vector: Vec<f32> = row.iter().map(|value| *value as f32).collect();
            self.index
                .add(start_key + offset as u64, &vector)
                .map_err(usearch_error)?;
            self.rows.extend(row);
        }
        Ok(())
    }

    fn shortlist_len(&self, k: usize) -> usize {
        k.max(self.index.expansion_search()).min(self.len())
    }

    fn shortlist(&self, query: &[f32], count: usize) -> Result<Vec<u64>, IndexError> {
        let exact = count == self.len();
        self.index.search(query, count, exact)
    }

    pub(crate) fn search(
        &self,
        queries_scaled: &ArrayView2<f64>,
        k_eff: usize,
        search_k: usize,
    ) -> Result<(Array2<f64>, Array2<i64>), IndexError> {
        if queries_scaled.ncols() != self.num_dim {
            return Err(IndexError::InvalidShape {
                expected: self.num_dim,
                got: queries_scaled.ncols(),
            });
        }

        let mut distances = Array2::from_elem((queries_scaled.nrows(), k_eff), f64::INFINITY);
        let mut indices = Array2::zeros((queries_scaled.nrows(), k_eff));
        let shortlist_len = self.shortlist_len(k_eff);
        for (query_index, query) in queries_scaled.rows().into_iter().enumerate() {
            let query_f32: Vec<f32> = query.iter().map(|value| *value as f32).collect();
            let keys = self.shortlist(&query_f32, shortlist_len)?;
            let neighbors = rerank(query, &keys, &self.rows, self.num_dim, k_eff)?;
            if neighbors.len() != k_eff {
                return Err(IndexError::InvalidParameter(format!(
                    "USearch returned {} candidates for requested k={k_eff}",
                    neighbors.len()
                )));
            }
            for (column, (distance, key)) in neighbors.into_iter().enumerate() {
                distances[[query_index, column]] = distance;
                indices[[query_index, column]] = key;
            }
        }
        Ok(pad_k(distances, indices, search_k))
    }
}

#[cfg(test)]
mod tests {
    use ndarray::array;

    use super::*;

    #[test]
    fn search_rows() {
        let train = array![[1.0 + 5.0e-8], [1.0 + 4.0e-8], [3.0]];
        let implementation = USearchIndex::new(1, &train.view()).unwrap();
        let (distances, indices) = implementation.search(&array![[0.0]].view(), 2, 2).unwrap();
        assert_eq!(indices.row(0).to_vec(), vec![1, 0]);
        assert!(distances[[0, 0]] < distances[[0, 1]]);
    }

    #[test]
    fn add_keys() {
        let train = array![[0.0, 0.0], [1.0, 0.0]];
        let mut implementation = USearchIndex::new(2, &train.view()).unwrap();
        implementation.add(&array![[0.0, 1.0]].view(), 2).unwrap();
        assert_eq!(implementation.len(), 3);
        let (_, indices) = implementation
            .search(&array![[0.0, 0.9]].view(), 2, 2)
            .unwrap();
        assert_eq!(indices[[0, 0]], 2);

        implementation.rebuild(&train.view()).unwrap();
        assert_eq!(implementation.len(), 2);
    }

    #[test]
    fn rejects_keys() {
        let mut implementation = USearchIndex::new(2, &array![[0.0, 0.0]].view()).unwrap();
        let error = implementation
            .add(&array![[1.0, 1.0]].view(), 3)
            .unwrap_err();
        assert!(error.to_string().contains("contiguous"));
    }
}
