//! Python orchestration of resident regional transactions. Parameter views
//! lease their allocations; completion requires releasing all exported views.
#![allow(clippy::too_many_arguments)]
use crate::py_metal::{PyMetalParamBlock, PyMetalWeights};
use baux::semantics::optimizer::portfolio::Allocation;
use baux::{
    TRLengthConfig,
    experimental::{
        MetalParamBlock, MetalSearchState, RegionalProposal, RegionalSearch, RoundConfig,
    },
    optimizer::multi_tr::SharingPolicy,
};
use numpy::{PyReadonlyArray1, PyReadonlyArray2};
use pyo3::{
    exceptions::{PyBufferError, PyValueError},
    prelude::*,
};
use std::{cell::Cell, rc::Rc};

fn error(message: String) -> PyErr {
    PyValueError::new_err(message)
}

#[pyclass(name = "Portfolio", module = "baux", unsendable)]
pub struct PyRegionalSearch {
    inner: RegionalSearch,
    blocks: Vec<MetalParamBlock>,
    capacity: usize,
    length: TRLengthConfig,
    leases: Rc<Cell<usize>>,
}

impl PyRegionalSearch {
    fn unleased(&self) -> PyResult<()> {
        if self.leases.get() > 0 {
            return Err(PyBufferError::new_err(
                "release regional weight views after evaluation before mutating the optimizer",
            ));
        }
        Ok(())
    }
}

#[pymethods]
impl PyRegionalSearch {
    #[new]
    #[pyo3(signature=(starts,values,blocks,capacity=8,length_init=0.1,length_min=0.001,length_max=0.8,allocation="least_trials",exploration=1.0,sharing="origin_only"))]
    fn new(
        starts: PyReadonlyArray2<u16>,
        values: PyReadonlyArray1<f32>,
        blocks: Vec<PyRef<PyMetalParamBlock>>,
        capacity: Option<usize>,
        length_init: f64,
        length_min: f64,
        length_max: f64,
        allocation: &str,
        exploration: f64,
        sharing: &str,
    ) -> PyResult<Self> {
        let capacity = capacity.unwrap_or(0);
        let starts = starts.as_array();
        let values = values.as_array();
        if starts.nrows() != values.len() {
            return Err(PyValueError::new_err(
                "one evaluated value is required per start",
            ));
        }
        let policy = match allocation {
            "least_trials" => Allocation::LeastTrials,
            "incumbent_ucb1" => Allocation::IncumbentUcb1 { exploration },
            _ => {
                return Err(PyValueError::new_err(
                    "allocation must be least_trials or incumbent_ucb1",
                ));
            }
        };
        let sharing = match sharing {
            "origin_only" => SharingPolicy::OriginOnly,
            "box_contain" => SharingPolicy::BoxContain,
            "ellipsoid_ball" => SharingPolicy::EllipsoidBall,
            "nearest_center" => SharingPolicy::NearestCenter,
            _ => {
                return Err(PyValueError::new_err(
                    "sharing must be origin_only, box_contain, ellipsoid_ball, or nearest_center",
                ));
            }
        };
        let blocks: Vec<_> = blocks.iter().map(|block| block.inner).collect();
        let length = TRLengthConfig::new(length_init, length_min, length_max);
        let states = starts
            .rows()
            .into_iter()
            .zip(values)
            .map(|(row, &value)| {
                MetalSearchState::new(
                    &row.to_vec(),
                    value,
                    0.0,
                    blocks.clone(),
                    capacity,
                    1,
                    length,
                )
            })
            .collect::<Result<Vec<_>, _>>()
            .map_err(error)?;
        Ok(Self {
            inner: RegionalSearch::with_sharing(states, policy, sharing).map_err(error)?,
            blocks,
            capacity,
            length,
            leases: Rc::new(Cell::new(0)),
        })
    }

    #[pyo3(signature=(seed=None,neighbors=1,region=None,acquisition="thompson"))]
    fn ask(
        mut slf: PyRefMut<Self>,
        seed: Option<u64>,
        neighbors: usize,
        region: Option<usize>,
        acquisition: &str,
    ) -> PyResult<PyRegionalProposal> {
        slf.unleased()?;
        let seed = baux::resolve_seed(seed);
        let mut config = RoundConfig::default();
        config.surrogate.neighbors = neighbors;
        config.acquisition.seed = seed;
        config.acquisition.kind = match acquisition {
            "thompson" => baux::experimental::AcquisitionKind::Thompson,
            "ucb" => baux::experimental::AcquisitionKind::Ucb,
            "random" => baux::experimental::AcquisitionKind::Random,
            _ => {
                return Err(PyValueError::new_err(
                    "acquisition must be thompson, ucb, or random",
                ));
            }
        };
        let inner = match region {
            Some(slot) => slf.inner.ask_region(slot, seed, config),
            None => slf.inner.ask(seed, config),
        }
        .map_err(error)?;
        Ok(PyRegionalProposal {
            owner: slf.into(),
            inner,
        })
    }

    #[pyo3(signature=(proposal,value,variance=0.0))]
    fn tell(
        &mut self,
        proposal: PyRef<PyRegionalProposal>,
        value: f32,
        variance: f32,
    ) -> PyResult<bool> {
        self.unleased()?;
        self.inner
            .tell(&proposal.inner, value, variance)
            .map(|decision| decision.accepted)
            .map_err(error)
    }

    fn cancel(&mut self, proposal: PyRef<PyRegionalProposal>) -> PyResult<()> {
        self.unleased()?;
        self.inner.cancel(&proposal.inner).map_err(error)
    }

    #[pyo3(signature=(region,weights,value,variance=0.0))]
    fn import_region(
        &mut self,
        region: usize,
        weights: PyRef<PyMetalWeights>,
        value: f32,
        variance: f32,
    ) -> PyResult<()> {
        self.unleased()?;
        self.inner
            .import_region(region, &weights.buffer, value, variance)
            .map_err(error)
    }

    #[pyo3(signature=(weights,value,variance=0.0))]
    fn import_observations(
        &mut self,
        weights: PyRef<PyMetalWeights>,
        value: f32,
        variance: f32,
    ) -> PyResult<Vec<usize>> {
        self.unleased()?;
        self.inner
            .import_external(&weights.buffer, value, variance)
            .map_err(error)
    }

    fn restart(&mut self, region: usize, base: PyRef<PyMetalWeights>, value: f32) -> PyResult<()> {
        self.unleased()?;
        let values =
            unsafe { std::slice::from_raw_parts(base.buffer.contents().cast::<u16>(), base.len) };
        let state = MetalSearchState::new(
            values,
            value,
            0.0,
            self.blocks.clone(),
            self.capacity,
            1,
            self.length,
        )
        .map_err(error)?;
        self.inner.restart(region, state).map_err(error)
    }

    fn incumbent(&self, region: usize) -> PyResult<PyMetalWeights> {
        let buffer = self.inner.incumbent(region).map_err(error)?;
        let len = self.inner.dimensions(region).map_err(error)?;
        self.leases.set(self.leases.get() + 1);
        Ok(PyMetalWeights {
            buffer,
            len,
            lease: Some(self.leases.clone()),
        })
    }

    fn flush_sharing(&mut self, region: usize) -> PyResult<()> {
        self.unleased()?;
        self.inner.flush_sharing(region).map_err(error)
    }

    #[getter]
    fn sharing_backlog(&self) -> Vec<usize> {
        self.inner.sharing_backlog()
    }

    #[getter]
    fn delivered_imports(&self) -> u64 {
        self.inner.delivered_imports()
    }

    #[getter]
    fn best(&self) -> (usize, f64) {
        self.inner.best()
    }
    #[getter]
    fn trials(&self) -> Vec<u64> {
        self.inner.trials()
    }
    #[getter]
    fn resident_bytes(&self) -> u64 {
        self.inner.memory_bytes()
    }
}

#[pyclass(name = "RegionTrial", module = "baux", unsendable)]
pub struct PyRegionalProposal {
    owner: Py<PyRegionalSearch>,
    inner: RegionalProposal,
}

#[pymethods]
impl PyRegionalProposal {
    #[getter]
    fn region(&self) -> usize {
        self.inner.ticket().region.slot
    }
    #[getter]
    fn generation(&self) -> u64 {
        self.inner.ticket().region.generation
    }
    fn weights(&self, py: Python<'_>) -> PyResult<PyMetalWeights> {
        let owner = self.owner.bind(py).try_borrow()?;
        let buffer = owner.inner.buffer(&self.inner).map_err(error)?;
        let len = owner.inner.dimensions(self.region()).map_err(error)?;
        owner.leases.set(owner.leases.get() + 1);
        Ok(PyMetalWeights {
            buffer,
            len,
            lease: Some(owner.leases.clone()),
        })
    }
    fn var(&self, py: Python<'_>) -> PyResult<PyMetalWeights> {
        self.weights(py)
    }
}
