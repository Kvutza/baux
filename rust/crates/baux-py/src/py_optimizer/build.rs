//! Construct Python-facing optimizers and expose factory functions.

use numpy::PyReadonlyArray2;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rand::SeedableRng;
use rand::rngs::StdRng;

use super::PyBayesOptimizer;
use super::config::parse_dict;
use super::regional;

fn build_optimizer(
    bounds: ndarray::Array2<f64>,
    kind: &str,
    k: i32,
    num_init: usize,
    num_regions: usize,
    seed: u64,
    cfg: Option<&Bound<'_, pyo3::types::PyDict>>,
    gp: Option<Py<PyAny>>,
    gp_factory: Option<Py<PyAny>>,
    fit_steps: usize,
) -> PyResult<PyBayesOptimizer> {
    use baux::optimizer_factory::{
        create_configured, enn_configured, lhd_configured, region_configured,
    };

    let num_dim = bounds.nrows();
    if gp_factory.is_some()
        && (kind != "gp"
            || !cfg.is_some_and(|dict| dict.contains("region_sharing").unwrap_or(false)))
    {
        return Err(PyValueError::new_err(
            "gp_factory requires kind='gp' and regional sharing configuration",
        ));
    }
    let runtime = cfg.map(parse_dict).transpose()?;
    let mut rng = baux::prng(seed);
    let optimizer = if cfg.is_some_and(|dict| dict.contains("region_sharing").unwrap_or(false)) {
        regional::build(
            bounds,
            kind,
            k,
            num_init,
            num_regions,
            runtime.as_ref(),
            cfg.unwrap(),
            gp,
            gp_factory,
            fit_steps,
            &mut rng,
        )?
    } else {
        match kind {
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
    .map_err(|e| PyValueError::new_err(e.to_string()))?
    };
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
#[pyo3(signature = (bounds, kind, k=10, num_init=10, num_regions=4, seed=None, cfg=None, gp=None, fit_steps=50, gp_factory=None))]
pub fn create_py(
    bounds: PyReadonlyArray2<f64>,
    kind: &str,
    k: i32,
    num_init: usize,
    num_regions: usize,
    seed: Option<u64>,
    cfg: Option<Bound<'_, pyo3::types::PyDict>>,
    gp: Option<Py<PyAny>>,
    fit_steps: usize,
    gp_factory: Option<Py<PyAny>>,
) -> PyResult<PyBayesOptimizer> {
    let bounds = bounds.as_array().to_owned();
    build_optimizer(
        bounds,
        kind,
        k,
        num_init,
        num_regions,
        baux::resolve_seed(seed),
        cfg.as_ref(),
        gp,
        gp_factory,
        fit_steps,
    )
}

/// Create TuRBO-ENN optimizer
#[pyfunction(name = "enn_optimizer")]
#[pyo3(signature = (bounds, k=10, num_init=10, seed=None, config=None))]
pub fn enn_py(
    bounds: PyReadonlyArray2<f64>,
    k: i32,
    num_init: usize,
    seed: Option<u64>,
    config: Option<Bound<'_, pyo3::types::PyDict>>,
) -> PyResult<PyBayesOptimizer> {
    build_optimizer(
        bounds.as_array().to_owned(),
        "enn",
        k,
        num_init,
        4,
        baux::resolve_seed(seed),
        config.as_ref(),
        None,
        None,
        50,
    )
}

/// Create experimental multi-trust-region TuRBO-ENN optimizer
#[pyfunction(name = "enn_tr")]
#[pyo3(signature = (bounds, k=10, num_init=10, num_regions=4, seed=None, config=None))]
pub fn bind_regions(
    bounds: PyReadonlyArray2<f64>,
    k: i32,
    num_init: usize,
    num_regions: usize,
    seed: Option<u64>,
    config: Option<Bound<'_, pyo3::types::PyDict>>,
) -> PyResult<PyBayesOptimizer> {
    build_optimizer(
        bounds.as_array().to_owned(),
        "enn_multi_tr",
        k,
        num_init,
        num_regions,
        baux::resolve_seed(seed),
        config.as_ref(),
        None,
        None,
        50,
    )
}

/// Create TuRBO-ZERO optimizer
#[pyfunction(name = "create_zero")]
#[pyo3(signature = (bounds, num_init=10, seed=None, config=None))]
pub fn bind_zero(
    bounds: PyReadonlyArray2<f64>,
    num_init: usize,
    seed: Option<u64>,
    config: Option<Bound<'_, pyo3::types::PyDict>>,
) -> PyResult<PyBayesOptimizer> {
    build_optimizer(
        bounds.as_array().to_owned(),
        "zero",
        10,
        num_init,
        4,
        baux::resolve_seed(seed),
        config.as_ref(),
        None,
        None,
        50,
    )
}

/// Create LHD-only optimizer
#[pyfunction(name = "create_lhd")]
#[pyo3(signature = (bounds, num_init=10, seed=None, config=None))]
pub fn bind_lhd(
    bounds: PyReadonlyArray2<f64>,
    num_init: usize,
    seed: Option<u64>,
    config: Option<Bound<'_, pyo3::types::PyDict>>,
) -> PyResult<PyBayesOptimizer> {
    build_optimizer(
        bounds.as_array().to_owned(),
        "lhd",
        10,
        num_init,
        4,
        baux::resolve_seed(seed),
        config.as_ref(),
        None,
        None,
        50,
    )
}
