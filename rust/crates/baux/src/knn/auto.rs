use std::collections::HashMap;
use std::time::Instant;

use ndarray::{Array2, ArrayView2};

use super::exact_backend::ExactBackend;
use super::metal_index::MetalIndex;
use crate::error::IndexError;

pub(crate) struct AutoIndex {
    pub(super) cpu: ExactBackend,
    pub(super) gpu: Option<MetalIndex>,
    pub(super) decisions: HashMap<(u32, usize, usize), bool>,
    pub(super) num_dim: usize,
}

impl AutoIndex {
    pub(super) fn new(num_dim: usize, train: &ArrayView2<f64>) -> Result<Self, IndexError> {
        Ok(Self {
            cpu: ExactBackend::new(num_dim, train)?,
            gpu: MetalIndex::new(num_dim, train).ok(),
            decisions: HashMap::new(),
            num_dim,
        })
    }

    pub(super) fn bucket(&self, queries: usize) -> u32 {
        self.cpu
            .len()
            .saturating_mul(queries)
            .saturating_mul(self.num_dim)
            .max(1)
            .ilog2()
    }

    pub(super) fn search(
        &mut self,
        queries: &ArrayView2<f64>,
        k_eff: usize,
        search_k: usize,
    ) -> Result<(Array2<f64>, Array2<i64>), IndexError> {
        if self.gpu.is_none() || !MetalIndex::supports_neighbors(k_eff) {
            return self.cpu.search(queries, k_eff, search_k);
        }
        let bucket = (self.bucket(queries.nrows()), k_eff, search_k);
        if let Some(&gpu) = self.decisions.get(&bucket) {
            if gpu {
                if let Some(result) = self.accelerated_search(queries, k_eff, search_k) {
                    return Ok(result);
                }
            }
            return self.cpu.search(queries, k_eff, search_k);
        }

        let cpu_start = Instant::now();
        let cpu = self.cpu.search(queries, k_eff, search_k)?;
        let cpu_time = cpu_start.elapsed();
        let gpu_start = Instant::now();
        let Some(gpu) = self.accelerated_search(queries, k_eff, search_k) else {
            return Ok(cpu);
        };
        let gpu_time = gpu_start.elapsed();
        let use_gpu = cpu.1 == gpu.1 && gpu_time < cpu_time;
        self.decisions.insert(bucket, use_gpu);
        Ok(if use_gpu { gpu } else { cpu })
    }

    fn accelerated_search(
        &mut self,
        queries: &ArrayView2<f64>,
        k_eff: usize,
        search_k: usize,
    ) -> Option<(Array2<f64>, Array2<i64>)> {
        match self.gpu.as_mut()?.search(queries, k_eff, search_k) {
            Ok(result) => Some(result),
            Err(_) => {
                // Keep the valid CPU mirror; retry acceleration after a
                // rebuild.
                self.gpu = None;
                self.decisions.clear();
                None
            }
        }
    }
}
