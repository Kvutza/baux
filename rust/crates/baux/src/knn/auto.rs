use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use std::time::Instant;

use ndarray::{Array2, ArrayView2};

use super::exact_backend::ExactBackend;
use super::metal_index::MetalIndex;
use crate::error::IndexError;

#[derive(Clone)]
pub(crate) struct AutoIndex {
    pub(super) cpu: ExactBackend,
    pub(super) gpu: Arc<RwLock<Option<MetalIndex>>>,
    pub(super) decisions: Arc<RwLock<HashMap<(u32, usize, usize), bool>>>,
    pub(super) num_dim: usize,
}

impl AutoIndex {
    pub(super) fn new(num_dim: usize, train: &ArrayView2<f64>) -> Result<Self, IndexError> {
        Ok(Self {
            cpu: ExactBackend::new(num_dim, train)?,
            gpu: Arc::new(RwLock::new(MetalIndex::new(num_dim, train).ok())),
            decisions: Arc::new(RwLock::new(HashMap::new())),
            num_dim,
        })
    }

    pub(super) fn len(&self) -> usize {
        self.cpu.len()
    }

    pub(super) fn is_empty(&self) -> bool {
        self.cpu.is_empty()
    }

    pub(super) fn memory_bytes(&self) -> usize {
        self.cpu.memory_bytes()
            + self
                .gpu
                .read()
                .ok()
                .and_then(|g| g.as_ref().map(MetalIndex::memory_bytes))
                .unwrap_or(0)
    }

    pub(super) fn rebuild(mut self, train: &ArrayView2<f64>) -> Result<Self, IndexError> {
        self.cpu = self.cpu.rebuild(train)?;
        self.gpu = Arc::new(RwLock::new(MetalIndex::new(self.num_dim, train).ok()));
        self.decisions = Arc::new(RwLock::new(HashMap::new()));
        Ok(self)
    }

    pub(super) fn add(
        mut self,
        rows: &ArrayView2<f64>,
        start_key: u64,
    ) -> Result<Self, IndexError> {
        self.cpu = self.cpu.add(rows, start_key)?;
        let old_gpu = self.gpu.read().ok().and_then(|g| g.clone());
        let new_gpu = if let Some(gpu) = old_gpu {
            gpu.add(rows, start_key).ok()
        } else {
            None
        };
        self.gpu = Arc::new(RwLock::new(new_gpu));
        self.decisions = Arc::new(RwLock::new(HashMap::new()));
        Ok(self)
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
        &self,
        queries: &ArrayView2<f64>,
        k_eff: usize,
        search_k: usize,
    ) -> Result<(Array2<f64>, Array2<i64>), IndexError> {
        let has_gpu = self
            .gpu
            .read()
            .ok()
            .and_then(|g| g.as_ref().map(|_| ()))
            .is_some();
        if !has_gpu || !MetalIndex::supports_neighbors(k_eff) {
            return self.cpu.search(queries, k_eff, search_k);
        }
        let bucket = (self.bucket(queries.nrows()), k_eff, search_k);
        let cached = self
            .decisions
            .read()
            .ok()
            .and_then(|guard| guard.get(&bucket).copied());
        if let Some(use_gpu) = cached {
            if use_gpu {
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
        if let Ok(mut guard) = self.decisions.write() {
            guard.insert(bucket, use_gpu);
        }
        Ok(if use_gpu { gpu } else { cpu })
    }

    fn accelerated_search(
        &self,
        queries: &ArrayView2<f64>,
        k_eff: usize,
        search_k: usize,
    ) -> Option<(Array2<f64>, Array2<i64>)> {
        let gpu = self.gpu.read().ok()?.clone()?;
        match gpu.search(queries, k_eff, search_k) {
            Ok(result) => Some(result),
            Err(_) => {
                // Keep the valid CPU mirror; retry acceleration after a
                // rebuild.
                if let Ok(mut g) = self.gpu.write() {
                    *g = None;
                }
                if let Ok(mut d) = self.decisions.write() {
                    d.clear();
                }
                None
            }
        }
    }

    #[inline]
    pub fn lock(&self) -> Result<&Self, std::convert::Infallible> {
        Ok(self)
    }
}
