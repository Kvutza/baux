//! Python wrapper for MultiTrustRegion state machine.

use numpy::{IntoPyArray, PyArrayDyn, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rand::SeedableRng;
use rand::rngs::StdRng;

/// Python wrapper for MultiTrustRegion state machine
#[pyclass(name = "MultiTrustRegion")]
pub struct PyMultiTrustRegion {
    inner: baux::experimental::MultiTrustRegionState,
}

#[pymethods]
impl PyMultiTrustRegion {
    #[new]
    #[pyo3(signature = (num_dim, num_regions=4, sharing_policy="shared", seed=42))]
    pub fn new(
        num_dim: usize,
        num_regions: usize,
        sharing_policy: &str,
        seed: u64,
    ) -> PyResult<Self> {
        use baux::experimental::{MultiTrustRegionConfig, MultiTrustRegionState, SharingPolicy};

        let mut rng = StdRng::seed_from_u64(seed);
        let policy = match sharing_policy {
            "shared" => SharingPolicy::Shared,
            "nearest_center" => SharingPolicy::NearestCenter,
            "independent" => SharingPolicy::Independent,
            _ => {
                return Err(PyValueError::new_err(format!(
                    "Unknown sharing policy: {}",
                    sharing_policy
                )));
            }
        };
        let mut config = MultiTrustRegionConfig::new(num_regions, Default::default());
        config.sharing_policy = policy;

        let state = MultiTrustRegionState::new(num_dim, config, None, &mut rng)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;

        Ok(Self { inner: state })
    }

    pub fn num_regions(&self) -> usize {
        self.inner.num_regions()
    }

    pub fn num_dim(&self) -> usize {
        self.inner.num_dim()
    }

    pub fn active_count(&self) -> usize {
        self.inner.active_count()
    }

    pub fn get_centers<'py>(&self, py: Python<'py>) -> Bound<'py, PyArrayDyn<f64>> {
        self.inner.centers.clone().into_dyn().into_pyarray(py)
    }

    pub fn get_lengths<'py>(&self, py: Python<'py>) -> Bound<'py, PyArrayDyn<f64>> {
        self.inner.lengths.clone().into_dyn().into_pyarray(py)
    }

    pub fn get_incumbents<'py>(&self, py: Python<'py>) -> Bound<'py, PyArrayDyn<f64>> {
        self.inner.incumbents_y.clone().into_dyn().into_pyarray(py)
    }

    pub fn tell(
        &mut self,
        x_batch: PyReadonlyArray2<'_, f64>,
        y_batch: PyReadonlyArray1<'_, f64>,
    ) -> PyResult<()> {
        let x_view = x_batch.as_array();
        let y_view = y_batch.as_array();
        self.inner
            .tell_update(&x_view, &y_view)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Allocate a candidate budget across regions with the default region
    /// utility.
    pub fn allocate(&self, budget: usize) -> PyResult<Vec<(usize, usize, usize)>> {
        let batches = self
            .inner
            .allocate(budget)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(batches
            .into_iter()
            .map(|batch| (batch.region, batch.start, batch.len))
            .collect())
    }

    /// Allocate a candidate budget across regions using an explicit utility
    /// vector.
    pub fn allocate_with<'py>(
        &self,
        budget: usize,
        utility: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<Vec<(usize, usize, usize)>> {
        let batches = self
            .inner
            .allocate_with(budget, &utility.as_array())
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(batches
            .into_iter()
            .map(|batch| (batch.region, batch.start, batch.len))
            .collect())
    }

    /// Select globally best and diverse region candidates.
    pub fn select(
        &self,
        candidates: Vec<(usize, usize, u64, f64)>,
        num_arms: usize,
    ) -> PyResult<Vec<(usize, usize, u64, f64)>> {
        let candidates = candidates
            .into_iter()
            .map(
                |(index, region, seed, score)| baux::experimental::RegionCandidate {
                    index,
                    region,
                    seed,
                    score,
                },
            )
            .collect::<Vec<_>>();
        let selected = self
            .inner
            .select(&candidates, num_arms)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(selected
            .into_iter()
            .map(|candidate| {
                (
                    candidate.index,
                    candidate.region,
                    candidate.seed,
                    candidate.score,
                )
            })
            .collect())
    }

    /// Variance of completed objectives for a region.
    pub fn variance(&self, region: usize) -> Option<f64> {
        self.inner.variance(region)
    }

    /// Restart a region with a new center in unit coordinates.
    pub fn restart_region<'py>(
        &mut self,
        region: usize,
        new_center: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<()> {
        self.inner
            .restart_region(region, &new_center.as_array())
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }
}
