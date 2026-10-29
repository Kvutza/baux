//! Optimizer Python bindings.

#![allow(clippy::too_many_arguments)]

use std::path::PathBuf;

use numpy::{IntoPyArray, PyArrayDyn, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rand::SeedableRng;
use rand::rngs::StdRng;

fn opt_val<'py, T>(dict: &Bound<'py, pyo3::types::PyDict>, key: &str) -> PyResult<Option<T>>
where
    T: for<'a> FromPyObject<'a, 'py>,
    for<'a> PyErr: From<<T as FromPyObject<'a, 'py>>::Error>,
{
    match dict.get_item(key)? {
        Some(v) => Ok(Some(v.extract()?)),
        None => Ok(None),
    }
}

fn apply_fields(
    dict: &Bound<'_, pyo3::types::PyDict>,
    config: &mut baux::config::RunConfig,
) -> PyResult<()> {
    config.num_candidates_factor = opt_val(dict, "num_candidates_factor")?;
    config.min_candidates = opt_val(dict, "min_candidates")?;
    config.max_candidates = opt_val(dict, "max_candidates")?;
    config.num_candidates_per_arm = opt_val(dict, "num_candidates_per_arm")?;
    config.num_pert = opt_val(dict, "num_pert")?;
    config.raasp_fast = opt_val(dict, "raasp_fast")?;
    config.length_init = opt_val(dict, "length_init")?;
    config.length_min = opt_val(dict, "length_min")?;
    config.length_max = opt_val(dict, "length_max")?;
    config.num_samples = opt_val(dict, "num_samples")?;
    config.num_candidates = opt_val(dict, "num_candidates")?;
    config.noise_aware = opt_val(dict, "noise_aware")?;
    config.scale_x = opt_val(dict, "scale_x")?;
    config.failure_tolerance_dim = opt_val(dict, "failure_tolerance_dim")?;
    config.num_metrics = opt_val(dict, "num_metrics")?;
    config.alpha = opt_val(dict, "alpha")?;
    if let Some(s) = opt_val::<String>(dict, "enn_storage")? {
        config.enn_storage = Some(match s.to_lowercase().as_str() {
            "disk" => baux::EnnStorage::Disk,
            "memory" | "in_memory" | "inmemory" => baux::EnnStorage::InMemory,
            _ => return Err(PyValueError::new_err(format!("Unknown enn_storage: {s}"))),
        });
    }
    config.work_dir = opt_val::<String>(dict, "work_dir")?.map(PathBuf::from);
    if let Some(v) = dict.get_item("y_bounds")? {
        let bounds: numpy::PyReadonlyArray2<f64> = v.extract()?;
        if bounds.shape()[1] != 2 {
            return Err(PyValueError::new_err(
                "y_bounds must have shape (metrics, 2)",
            ));
        }
        config.y_bounds = Some(
            bounds
                .as_array()
                .rows()
                .into_iter()
                .map(|row| [row[0], row[1]])
                .collect(),
        );
    }
    Ok(())
}

pub fn parse_dict(dict: &Bound<'_, pyo3::types::PyDict>) -> PyResult<baux::config::RunConfig> {
    use baux::index::IndexDriver;
    use baux::{AcquisitionConfig, CandidateSampler};

    let mut config = baux::config::RunConfig::default();

    if let Some(s) = opt_val::<String>(dict, "index_driver")? {
        let drivers = [
            ("exact", IndexDriver::Exact),
            ("flat", IndexDriver::Exact),
            ("auto", IndexDriver::Auto),
            ("usearch", IndexDriver::USearch),
            ("hnsw_usearch", IndexDriver::USearch),
            ("bpann_disk", IndexDriver::BpAnnDisk),
            ("metal", IndexDriver::Metal),
            ("opencl", IndexDriver::OpenCl),
            ("ocl", IndexDriver::OpenCl),
            ("cuda", IndexDriver::Cuda),
        ];
        let lower = s.to_lowercase();
        let (_, drv) = drivers
            .into_iter()
            .find(|(k, _)| *k == lower)
            .ok_or_else(|| PyValueError::new_err(format!("Unknown index_driver: {s}")))?;
        config.index_driver = Some(drv);
    }
    if let Some(s) = opt_val::<String>(dict, "acquisition")? {
        let beta = opt_val::<f64>(dict, "acquisition_beta")?.unwrap_or(2.0);
        let acq = match s.as_str() {
            "ucb" => AcquisitionConfig::UCB { beta },
            "thompson" => AcquisitionConfig::Thompson,
            "random" => AcquisitionConfig::Random,
            "pareto" => AcquisitionConfig::Pareto,
            _ => return Err(PyValueError::new_err(format!("Unknown acquisition: {s}"))),
        };
        config.acquisition = Some(acq);
    }
    if let Some(s) = opt_val::<String>(dict, "candidate_rv")? {
        let samplers = [
            ("sobol", CandidateSampler::Sobol),
            ("uniform", CandidateSampler::Uniform),
            ("raasp", CandidateSampler::RAASP),
        ];
        let (_, rv) = samplers
            .into_iter()
            .find(|(k, _)| *k == s)
            .ok_or_else(|| PyValueError::new_err(format!("Unknown candidate_rv: {s}")))?;
        config.candidate_rv = Some(rv);
    }
    if let Some(s) = opt_val::<String>(dict, "trust_region")? {
        let regions = [
            ("turbo", baux::config::TrustRegionKind::Turbo),
            ("morbo", baux::config::TrustRegionKind::Morbo),
        ];
        let (_, kind) = regions
            .into_iter()
            .find(|(k, _)| *k == s)
            .ok_or_else(|| PyValueError::new_err(format!("Invalid trust_region_kind: {s}")))?;
        config.trust_region_kind = Some(kind);
    }
    if let Some(s) = opt_val::<String>(dict, "rescalarize")? {
        let rescalarizers = [
            ("on_restart", baux::morbo::Rescalarize::OnRestart),
            ("on_propose", baux::morbo::Rescalarize::OnPropose),
        ];
        let (_, resc) = rescalarizers
            .into_iter()
            .find(|(k, _)| *k == s)
            .ok_or_else(|| PyValueError::new_err(format!("Invalid rescalarize: {s}")))?;
        config.rescalarize = Some(resc);
    }
    apply_fields(dict, &mut config)?;
    Ok(config)
}

/// Python wrapper for BayesOptimizer
#[pyclass(name = "BayesOptimizer")]
pub struct PyBayesOptimizer {
    inner: baux::BayesOptimizer,
    rng: StdRng,
}

#[pymethods]
impl PyBayesOptimizer {
    /// Ask for candidate points
    #[pyo3(signature = (num_arms, seed=None))]
    fn ask<'py>(
        &mut self,
        py: Python<'py>,
        num_arms: usize,
        seed: Option<u64>,
    ) -> PyResult<Bound<'py, PyArrayDyn<f64>>> {
        let result = match seed {
            Some(seed) => {
                let mut rng = StdRng::seed_from_u64(seed);
                self.inner.ask_natural(num_arms, &mut rng)
            }
            None => self.inner.ask_natural(num_arms, &mut self.rng),
        }
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(result.into_dyn().into_pyarray(py))
    }

    /// Tell observations
    #[pyo3(signature = (x, y, seed=None, y_var=None))]
    fn tell(
        &mut self,
        x: PyReadonlyArray2<f64>,
        y: PyReadonlyArray2<f64>,
        seed: Option<u64>,
        y_var: Option<PyReadonlyArray2<f64>>,
    ) -> PyResult<()> {
        let x_view = x.as_array();
        let y_view = y.as_array();
        let yvar_view = y_var.as_ref().map(PyReadonlyArray2::as_array);
        match seed {
            Some(seed) => {
                let mut rng = StdRng::seed_from_u64(seed);
                self.inner
                    .tell_natural(&x_view, &y_view, yvar_view.as_ref(), &mut rng)
            }
            None => self
                .inner
                .tell_natural(&x_view, &y_view, yvar_view.as_ref(), &mut self.rng),
        }
        .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Posterior mean of the fitted surrogate at `x` in natural units.
    fn posterior_mu<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<f64>,
    ) -> PyResult<Bound<'py, PyArrayDyn<f64>>> {
        let mu = self
            .inner
            .posterior_mu(&x.as_array())
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(mu.into_dyn().into_pyarray(py))
    }

    /// Whether this optimizer fits a surrogate.
    fn has_surrogate(&self) -> bool {
        self.inner.has_surrogate()
    }

    /// Get init progress if in initialization phase
    fn init_progress(&self) -> Option<(usize, usize)> {
        self.inner.init_progress()
    }

    /// Get current telemetry
    fn telemetry(&self) -> PyTelemetry {
        let t = self.inner.telemetry();
        PyTelemetry {
            dt_fit: t.dt_fit,
            dt_gen: t.dt_gen,
            dt_sel: t.dt_sel,
            dt_tell: t.dt_tell,
            num_candidates: t.num_candidates,
        }
    }

    /// Number of retained trust-region observations.
    fn region_count(&self) -> usize {
        self.inner.obs_count()
    }

    /// Current trust-region length.
    fn tr_length(&self) -> f64 {
        self.inner.tr_length()
    }

    /// Get observations x in natural coordinates (if any).
    fn x_obs<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArrayDyn<f64>>> {
        self.inner.x_obs().map(|x| x.into_dyn().into_pyarray(py))
    }

    /// Get observation values y (if any).
    fn y_obs<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArrayDyn<f64>>> {
        self.inner.y_obs().map(|y| y.into_dyn().into_pyarray(py))
    }

    /// Get incumbent x in unit space (if any).
    fn unit_incumbent<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArrayDyn<f64>>> {
        self.inner
            .unit_incumbent()
            .map(|x| x.view().to_owned().into_dyn().into_pyarray(py))
    }

    /// Get incumbent x in natural coordinates (if any).
    fn incumbent_x<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArrayDyn<f64>>> {
        self.inner
            .incumbent_x()
            .map(|x| x.into_dyn().into_pyarray(py))
    }

    /// Get optimizer bounds.
    fn bounds<'py>(&self, py: Python<'py>) -> Bound<'py, PyArrayDyn<f64>> {
        self.inner
            .bounds()
            .view()
            .to_owned()
            .into_dyn()
            .into_pyarray(py)
    }
}

fn build_optimizer(
    bounds: ndarray::Array2<f64>,
    kind: &str,
    k: i32,
    num_init: usize,
    num_regions: usize,
    seed: u64,
    cfg: Option<&Bound<'_, pyo3::types::PyDict>>,
    gp: Option<Py<PyAny>>,
    fit_steps: usize,
) -> PyResult<PyBayesOptimizer> {
    use baux::optimizer_factory::{
        create_configured, enn_configured, lhd_configured, region_configured,
    };

    let num_dim = bounds.nrows();
    let runtime = cfg.map(parse_dict).transpose()?;
    let mut rng = StdRng::seed_from_u64(seed);
    let optimizer = match kind {
        "enn" => enn_configured(bounds, k, num_init, &mut rng, runtime.as_ref()),
        "enn_multi_tr" => {
            region_configured(bounds, k, num_init, num_regions, &mut rng, runtime.as_ref())
        }
        "zero" => create_configured(bounds, num_init, &mut rng, runtime.as_ref()),
        "lhd" => lhd_configured(bounds, num_init, &mut rng, runtime.as_ref()),
        "gp" => {
            let provider = gp.ok_or_else(|| {
                PyValueError::new_err("kind='gp' requires a Python surrogate provider")
            })?;
            let mut config = baux::turbo_zero();
            if let Some(settings) = runtime.as_ref() {
                config = settings.apply_to(config);
            }
            let strategy = baux::Strategy::hybrid(baux::InitStrategy::LHD, num_init);
            let external = Box::new(crate::adapter::PythonSurrogateAdapter::new(
                provider, num_dim, fit_steps,
            ));
            baux::BayesOptimizer::new_surrogate(bounds, config, strategy, external, &mut rng)
        }
        _ => {
            return Err(PyValueError::new_err(format!(
                "unknown optimizer kind {kind:?}; expected 'enn', 'enn_multi_tr', 'zero', 'lhd', \
                 or 'gp'"
            )));
        }
    }
    .map_err(|e| PyValueError::new_err(e.to_string()))?;
    // ENN 0.4.1 forks an owned StdRng after constructor-time draws.  The
    // wrapper owns that stream because the local core accepts RNGs explicitly.
    let owned_rng = StdRng::from_rng(&mut rng).map_err(|e| PyValueError::new_err(e.to_string()))?;

    Ok(PyBayesOptimizer {
        inner: optimizer,
        rng: owned_rng,
    })
}

/// Create a built-in optimizer, optionally with a batched Python surrogate
/// adapter.
#[pyfunction(name = "create_optimizer")]
#[pyo3(signature = (bounds, kind, k=10, num_init=10, num_regions=4, seed=42, cfg=None, gp=None, fit_steps=50))]
pub fn create_py(
    bounds: PyReadonlyArray2<f64>,
    kind: &str,
    k: i32,
    num_init: usize,
    num_regions: usize,
    seed: u64,
    cfg: Option<Bound<'_, pyo3::types::PyDict>>,
    gp: Option<Py<PyAny>>,
    fit_steps: usize,
) -> PyResult<PyBayesOptimizer> {
    let bounds = bounds.as_array().to_owned();
    build_optimizer(
        bounds,
        kind,
        k,
        num_init,
        num_regions,
        seed,
        cfg.as_ref(),
        gp,
        fit_steps,
    )
}

/// Telemetry data structure for Python
#[pyclass(name = "Telemetry", from_py_object)]
#[derive(Clone, Copy)]
pub struct PyTelemetry {
    #[pyo3(get)]
    pub dt_fit: f64,
    #[pyo3(get)]
    pub dt_gen: f64,
    #[pyo3(get)]
    pub dt_sel: f64,
    #[pyo3(get)]
    pub dt_tell: f64,
    #[pyo3(get)]
    pub num_candidates: usize,
}

/// Create TuRBO-ENN optimizer
#[pyfunction(name = "enn_optimizer")]
#[pyo3(signature = (bounds, k=10, num_init=10, seed=42, config=None))]
pub fn enn_py(
    bounds: PyReadonlyArray2<f64>,
    k: i32,
    num_init: usize,
    seed: u64,
    config: Option<Bound<'_, pyo3::types::PyDict>>,
) -> PyResult<PyBayesOptimizer> {
    build_optimizer(
        bounds.as_array().to_owned(),
        "enn",
        k,
        num_init,
        4,
        seed,
        config.as_ref(),
        None,
        50,
    )
}

/// Create experimental multi-trust-region TuRBO-ENN optimizer
#[pyfunction(name = "enn_tr")]
#[pyo3(signature = (bounds, k=10, num_init=10, num_regions=4, seed=42, config=None))]
pub fn bind_regions(
    bounds: PyReadonlyArray2<f64>,
    k: i32,
    num_init: usize,
    num_regions: usize,
    seed: u64,
    config: Option<Bound<'_, pyo3::types::PyDict>>,
) -> PyResult<PyBayesOptimizer> {
    build_optimizer(
        bounds.as_array().to_owned(),
        "enn_multi_tr",
        k,
        num_init,
        num_regions,
        seed,
        config.as_ref(),
        None,
        50,
    )
}

/// Create TuRBO-ZERO optimizer
#[pyfunction(name = "create_zero")]
#[pyo3(signature = (bounds, num_init=10, seed=42, config=None))]
pub fn bind_zero(
    bounds: PyReadonlyArray2<f64>,
    num_init: usize,
    seed: u64,
    config: Option<Bound<'_, pyo3::types::PyDict>>,
) -> PyResult<PyBayesOptimizer> {
    build_optimizer(
        bounds.as_array().to_owned(),
        "zero",
        10,
        num_init,
        4,
        seed,
        config.as_ref(),
        None,
        50,
    )
}

/// Create LHD-only optimizer
#[pyfunction(name = "create_lhd")]
#[pyo3(signature = (bounds, num_init=10, seed=42, config=None))]
pub fn bind_lhd(
    bounds: PyReadonlyArray2<f64>,
    num_init: usize,
    seed: u64,
    config: Option<Bound<'_, pyo3::types::PyDict>>,
) -> PyResult<PyBayesOptimizer> {
    build_optimizer(
        bounds.as_array().to_owned(),
        "lhd",
        10,
        num_init,
        4,
        seed,
        config.as_ref(),
        None,
        50,
    )
}
