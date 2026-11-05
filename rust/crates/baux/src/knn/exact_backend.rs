use deser::{Deserialize, Serialize};
use ndarray::{Array2, ArrayView2};

use crate::error::IndexError;

#[derive(Debug, Clone, Copy, PartialEq, Default, Serialize, Deserialize)]
#[deser(rename_all = "snake_case")]
pub enum IndexDriver {
    #[default]
    Exact,
    /// Exact CPU/accelerator race calibrated from real query shapes.
    Auto,
    /// USearch HNSW shortlist with deterministic exact-distance reranking.
    #[deser(rename = "usearch")]
    USearch,
    /// B+ANN disk index (`EnnStorage::Disk` + `work_dir`).
    #[deser(rename = "bp_ann_disk")]
    BpAnnDisk,
    /// Apple Metal backend for native quantized-weight paths.
    Metal,
    /// OpenCL backend for native quantized-weight paths.
    #[deser(rename = "opencl")]
    OpenCl,
    /// NVIDIA CUDA exact index implemented with CUDA-Oxide.
    #[deser(rename = "cuda")]
    Cuda,
    /// NVIDIA cuVS exact brute-force index (`libcuvs_c`).
    #[deser(rename = "cuvs")]
    Cuvs,
    /// NVIDIA cuVS CAGRA graph index (`libcuvs_c`).
    #[deser(rename = "cuvs_cagra")]
    CuvsCagra,
}

impl IndexDriver {
    pub const fn is_cuda(self) -> bool {
        matches!(self, Self::Cuda | Self::Cuvs | Self::CuvsCagra)
    }
}

pub fn disk_driver(driver: IndexDriver) -> bool {
    matches!(driver, IndexDriver::BpAnnDisk)
}

#[allow(dead_code)]
pub(crate) fn check_rows(rows: &ArrayView2<f64>, expected_dim: usize) -> Result<(), IndexError> {
    if rows.ncols() != expected_dim {
        return Err(IndexError::InvalidShape {
            expected: expected_dim,
            got: rows.ncols(),
        });
    }
    Ok(())
}

#[allow(dead_code)]
pub(crate) fn flatten_f32(a: &ArrayView2<f64>) -> Vec<f32> {
    a.iter().map(|v| *v as f32).collect()
}

pub(crate) fn pad_k(
    dist2s: Array2<f64>,
    idx: Array2<i64>,
    search_k: usize,
) -> (Array2<f64>, Array2<i64>) {
    use ndarray::{Axis, concatenate};
    let k_eff = dist2s.ncols();
    if k_eff >= search_k {
        return (dist2s, idx);
    }
    let n_query = dist2s.nrows();
    if k_eff == 0 {
        return (
            Array2::from_elem((n_query, search_k), f64::INFINITY),
            Array2::zeros((n_query, search_k)),
        );
    }
    let pad_w = search_k - k_eff;
    let pad_dist = Array2::from_elem((n_query, pad_w), f64::INFINITY);
    let far = idx.slice(ndarray::s![.., k_eff - 1..k_eff]).to_owned();
    let mut pad_idx = Array2::zeros((n_query, pad_w));
    for j in 0..pad_w {
        pad_idx.column_mut(j).assign(&far.column(0));
    }
    (
        concatenate![Axis(1), dist2s.view(), pad_dist.view()],
        concatenate![Axis(1), idx.view(), pad_idx.view()],
    )
}

#[allow(dead_code)]
pub(crate) fn unpack_search(
    n_query: usize,
    k: usize,
    distances: &[f32],
    labels: &[i64],
) -> (Array2<f64>, Array2<i64>) {
    let mut dist2s = Array2::zeros((n_query, k));
    let mut indices = Array2::zeros((n_query, k));
    for i in 0..n_query {
        for j in 0..k {
            let o = i * k + j;
            dist2s[[i, j]] = f64::from(distances[o]);
            indices[[i, j]] = labels[o];
        }
    }
    (dist2s, indices)
}

pub(crate) struct ExactBackend {
    rows: Array2<f64>,
    keys: Vec<u64>,
    num_dim: usize,
}

impl ExactBackend {
    pub(crate) fn new(num_dim: usize, train_scaled: &ArrayView2<f64>) -> Result<Self, IndexError> {
        if train_scaled.ncols() != num_dim {
            return Err(IndexError::InvalidShape {
                expected: num_dim,
                got: train_scaled.ncols(),
            });
        }
        Ok(Self {
            rows: train_scaled.to_owned(),
            keys: (0..train_scaled.nrows()).map(|i| i as u64).collect(),
            num_dim,
        })
    }

    pub(crate) fn len(&self) -> usize {
        self.rows.nrows()
    }

    pub(crate) fn memory_bytes(&self) -> usize {
        self.rows.len().saturating_mul(std::mem::size_of::<f64>())
            + self.keys.len().saturating_mul(std::mem::size_of::<u64>())
    }

    pub(crate) fn rebuild(&mut self, train_scaled: &ArrayView2<f64>) -> Result<(), IndexError> {
        if train_scaled.ncols() != self.num_dim {
            return Err(IndexError::InvalidShape {
                expected: self.num_dim,
                got: train_scaled.ncols(),
            });
        }
        self.rows = train_scaled.to_owned();
        self.keys = (0..train_scaled.nrows()).map(|i| i as u64).collect();
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
        if rows_scaled.nrows() == 0 {
            return Ok(());
        }
        self.rows = ndarray::concatenate![ndarray::Axis(0), self.rows.view(), rows_scaled.view()];
        self.keys
            .extend((0..rows_scaled.nrows()).map(|i| start_key + i as u64));
        Ok(())
    }

    pub(crate) fn search(
        &mut self,
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
        let mut dist2s = Array2::zeros((queries_scaled.nrows(), k_eff));
        let mut indices = Array2::zeros((queries_scaled.nrows(), k_eff));
        for (qi, query) in queries_scaled.outer_iter().enumerate() {
            let mut pairs = self
                .rows
                .outer_iter()
                .zip(self.keys.iter().copied())
                .map(|(row, key)| {
                    let dist2 = row
                        .iter()
                        .zip(query.iter())
                        .map(|(left, right)| {
                            let delta = left - right;
                            delta * delta
                        })
                        .sum::<f64>();
                    (dist2, key)
                })
                .collect::<Vec<_>>();
            pairs.sort_unstable_by(|left, right| {
                left.0
                    .total_cmp(&right.0)
                    .then_with(|| left.1.cmp(&right.1))
            });
            for (ki, (dist2, key)) in pairs.into_iter().take(k_eff).enumerate() {
                dist2s[[qi, ki]] = dist2;
                indices[[qi, ki]] = key as i64;
            }
        }
        Ok(pad_k(dist2s, indices, search_k))
    }
}
