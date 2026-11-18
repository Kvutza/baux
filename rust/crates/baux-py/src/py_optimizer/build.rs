//! Construct Python-facing optimizers and expose factory functions.

use numpy::PyReadonlyArray2;
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use rand::SeedableRng;
use rand::rngs::StdRng;

use super::PyBayesOptimizer;
use super::config::parse_dict;
use super::regional;

fn parse_backend(device: Option<&Bound<'_, PyAny>>) -> PyResult<baux::AnyBackend> {
    let Some(obj) = device else {
        return Ok(baux::AnyBackend::Cpu);
    };
    if obj.is_none() {
        return Ok(baux::AnyBackend::Cpu);
    }
    if let Ok(dev) = obj.extract::<PyRef<'_, crate::py_device::PyDevice>>() {
        return baux::AnyBackend::parse(dev.kind).map_err(|e| PyValueError::new_err(e.to_string()));
    }
    if let Ok(name) = obj.extract::<&str>() {
        let trimmed = name.trim();
        let raw = if trimmed.is_empty() { "auto" } else { trimmed };
        return baux::AnyBackend::parse(raw).map_err(|e| PyValueError::new_err(e.to_string()));
    }
    Err(PyTypeError::new_err("device must be a str or Device"))
}

fn noncpu_spec(
    num_dim: usize,
    kind: &str,
    k: i32,
    num_init: usize,
    num_regions: usize,
    runtime: Option<&baux::config::RunConfig>,
    rng: &mut baux::Prng,
) -> PyResult<(baux::OptimizerConfig, baux::Strategy)> {
    let (mut config, strategy) = match kind {
        "enn" => {
            if k <= 0 {
                return Err(PyValueError::new_err(format!(
                    "k (number of neighbors) must be > 0, got {k}"
                )));
            }
            let mut cfg = baux::turbo_enn();
            if let baux::SurrogateConfig::ENN(enn) = &mut cfg.surrogate {
                enn.k = k;
            }
            (
                cfg,
                baux::Strategy::hybrid(baux::InitStrategy::LHD, num_init),
            )
        }
        "enn_multi_tr" => {
            if k <= 0 {
                return Err(PyValueError::new_err(format!(
                    "k (number of neighbors) must be > 0, got {k}"
                )));
            }
            let mut cfg = baux::turbo_enn();
            if let baux::SurrogateConfig::ENN(enn) = &mut cfg.surrogate {
                enn.k = k;
            }
            let strat = baux::Strategy::experimental(num_dim, num_regions, num_init, rng)
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
            (cfg, strat)
        }
        "zero" => (
            baux::turbo_zero(),
            baux::Strategy::hybrid(baux::InitStrategy::LHD, num_init),
        ),
        "lhd" => (
            baux::lhd_only(),
            baux::Strategy::init(baux::InitStrategy::LHD, num_init),
        ),
        _ => {
            return Err(PyValueError::new_err(format!(
                "unknown optimizer kind {kind:?}; expected 'enn', 'enn_multi_tr', 'zero', 'lhd', \
                 or 'gp'"
            )));
        }
    };
    if let Some(settings) = runtime {
        config = settings.apply_to(config);
    }
    Ok((config, strategy))
}

#[allow(clippy::too_many_arguments)]
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
    backend: baux::AnyBackend,
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
    let has_sharing = cfg.is_some_and(|dict| dict.contains("region_sharing").unwrap_or(false));

    if backend != baux::AnyBackend::Cpu {
        if kind == "gp" || gp.is_some() || gp_factory.is_some() {
            return Err(PyValueError::new_err(format!(
                "Python GP and custom surrogate callbacks require device='cpu', got {:?}",
                backend.name()
            )));
        }
        let (config, strategy) = if has_sharing {
            regional::config_strategy(
                num_dim,
                kind,
                k,
                num_init,
                num_regions,
                runtime.as_ref(),
                cfg.unwrap(),
                &mut rng,
            )?
        } else {
            noncpu_spec(
                num_dim,
                kind,
                k,
                num_init,
                num_regions,
                runtime.as_ref(),
                &mut rng,
            )?
        };
        let engine = baux::UnifiedEngine::new_strategy(
            bounds,
            config,
            strategy,
            backend,
            baux::Key::new(seed),
        )
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let owned_rng =
            StdRng::from_rng(&mut rng).map_err(|e| PyValueError::new_err(e.to_string()))?;
        return Ok(PyBayesOptimizer {
            inner: engine,
            rng: owned_rng,
        });
    }

    let optimizer = if has_sharing {
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
    let owned_rng = StdRng::from_rng(&mut rng).map_err(|e| PyValueError::new_err(e.to_string()))?;

    Ok(PyBayesOptimizer {
        inner: baux::UnifiedEngine::from_cpu(optimizer),
        rng: owned_rng,
    })
}

/// Create a built-in optimizer, optionally with a batched Python surrogate
/// adapter or target execution backend (`"cpu"`, `"metal"`, `"cuda"`,
/// `"auto"`).
#[pyfunction(name = "create_optimizer")]
#[pyo3(signature = (bounds, kind, k=10, num_init=10, num_regions=4, seed=None, cfg=None, gp=None, fit_steps=50, gp_factory=None, device=None))]
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
    device: Option<&Bound<'_, PyAny>>,
) -> PyResult<PyBayesOptimizer> {
    let backend = parse_backend(device)?;
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
        backend,
    )
}

/// Create experimental multi-trust-region TuRBO-ENN optimizer
#[pyfunction(name = "enn_tr")]
#[pyo3(signature = (bounds, k=10, num_init=10, num_regions=4, seed=None, config=None, device=None))]
pub fn bind_regions(
    bounds: PyReadonlyArray2<f64>,
    k: i32,
    num_init: usize,
    num_regions: usize,
    seed: Option<u64>,
    config: Option<Bound<'_, pyo3::types::PyDict>>,
    device: Option<&Bound<'_, PyAny>>,
) -> PyResult<PyBayesOptimizer> {
    let backend = parse_backend(device)?;
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
        backend,
    )
}
