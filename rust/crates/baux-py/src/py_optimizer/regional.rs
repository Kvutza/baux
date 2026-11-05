//! Bind the existing multi-region strategy independently of surrogate choice.
//! In-memory ENN and supplied provider factories create regional models.
//! Modeled regional search without isolated models is rejected at construction.
use super::config::opt_val;
use super::*;
use baux::optimizer::multi_tr::RegionController;
use baux::optimizer::{
    AllocationPolicy, MultiTrustRegionConfig, RestartCenterPolicy, RestartCenterSelection,
    RestartMemoryPolicy, SharingPolicy, StrategyState,
};
use baux::optimizer_config::AcquisitionConfig;

fn call_provider(
    factory: &Py<PyAny>,
    slot: usize,
    generation: u64,
) -> Result<(Py<PyAny>, usize), baux::error::ENNError> {
    Python::attach(|py| -> PyResult<(Py<PyAny>, usize)> {
        let value = factory.bind(py).call1((slot, generation))?;
        let identity = value.as_ptr() as usize;
        Ok((value.unbind(), identity))
    })
    .map_err(|error| {
        baux::error::ENNError::InvalidParameter(format!(
            "regional Python surrogate factory: {error}"
        ))
    })
}

fn model_factory(
    factory: Py<PyAny>,
    global_identity: usize,
    dimension: usize,
    fit_steps: usize,
) -> baux::optimizer::RegionalModelFactory {
    let identities = std::sync::Mutex::new(std::collections::HashMap::<usize, usize>::new());
    Box::new(move |slot, generation| {
        let (provider, identity) = call_provider(&factory, slot, generation)?;
        let mut identities = identities.lock().map_err(|error| {
            baux::error::ENNError::InvalidParameter(format!("regional provider registry: {error}"))
        })?;
        if identity == global_identity
            || identities.get(&slot) == Some(&identity)
            || identities
                .iter()
                .any(|(&other, &current)| other != slot && current == identity)
        {
            return Err(baux::error::ENNError::InvalidParameter(
                "regional surrogate factory reused a provider instance".into(),
            ));
        }
        identities.insert(slot, identity);
        Ok(Box::new(crate::adapter::PythonSurrogateAdapter::new(
            provider, dimension, fit_steps,
        )) as baux::surrogate::BoxedSurrogate)
    })
}

#[allow(clippy::too_many_arguments)]
fn build_gp(
    bounds: ndarray::Array2<f64>,
    config: baux::OptimizerConfig,
    strategy: baux::Strategy,
    gp: Option<Py<PyAny>>,
    gp_factory: Option<Py<PyAny>>,
    fit_steps: usize,
    rng: &mut StdRng,
) -> Result<baux::BayesOptimizer, baux::error::ENNError> {
    let provider = gp.ok_or_else(|| {
        baux::error::ENNError::InvalidParameter("missing surrogate provider".into())
    })?;
    let dimension = bounds.nrows();
    let global_identity = Python::attach(|py| provider.bind(py).as_ptr() as usize);
    let surrogate = Box::new(crate::adapter::PythonSurrogateAdapter::new(
        provider, dimension, fit_steps,
    ));
    if let Some(factory) = gp_factory {
        baux::BayesOptimizer::new_regional(
            bounds,
            config,
            strategy,
            surrogate,
            model_factory(factory, global_identity, dimension, fit_steps),
            rng,
        )
    } else {
        baux::BayesOptimizer::new_surrogate(bounds, config, strategy, surrogate, rng)
    }
}

fn parse_sharing(dict: &Bound<'_, pyo3::types::PyDict>) -> PyResult<SharingPolicy> {
    match opt_val::<String>(dict, "region_sharing")?.as_deref() {
        Some("shared") => Ok(SharingPolicy::Shared),
        Some("nearest_center") => Ok(SharingPolicy::NearestCenter),
        Some("independent") => Ok(SharingPolicy::Independent),
        _ => Err(PyValueError::new_err("unknown regional sharing policy")),
    }
}

fn parse_allocation(dict: &Bound<'_, pyo3::types::PyDict>) -> PyResult<AllocationPolicy> {
    match opt_val::<String>(dict, "region_allocation")?.as_deref() {
        None | Some("utility_ucb") => Ok(AllocationPolicy::UtilityUcb),
        Some("balanced") => Ok(AllocationPolicy::Balanced),
        _ => Err(PyValueError::new_err("unknown regional allocation policy")),
    }
}

fn parse_restart(dict: &Bound<'_, pyo3::types::PyDict>) -> PyResult<RestartMemoryPolicy> {
    match opt_val::<String>(dict, "region_restart_memory")?.as_deref() {
        None | Some("fresh") => Ok(RestartMemoryPolicy::Fresh),
        Some("retain_relevant") => Ok(RestartMemoryPolicy::RetainRelevant),
        _ => Err(PyValueError::new_err(
            "unknown regional restart memory policy",
        )),
    }
}

fn parse_controller(dict: &Bound<'_, pyo3::types::PyDict>) -> PyResult<RegionController> {
    match opt_val::<String>(dict, "region_controller")?.as_deref() {
        None | Some("turbo") => Ok(RegionController::Turbo),
        Some("fixed_radius") => Ok(RegionController::FixedRadius),
        _ => Err(PyValueError::new_err("unknown regional controller")),
    }
}

fn parse_center(dict: &Bound<'_, pyo3::types::PyDict>) -> PyResult<RestartCenterPolicy> {
    let candidates = opt_val::<usize>(dict, "region_restart_candidates")?.unwrap_or(1);
    if candidates == 0 {
        return Err(PyValueError::new_err(
            "regional restart center candidates must be positive",
        ));
    }
    let selection = match opt_val::<String>(dict, "region_restart_center_selection")?.as_deref() {
        None | Some("first") => RestartCenterSelection::First,
        Some("maximin") => RestartCenterSelection::Maximin,
        _ => {
            return Err(PyValueError::new_err(
                "unknown regional restart center selection",
            ));
        }
    };
    Ok(RestartCenterPolicy {
        candidates,
        selection,
    })
}

fn parse_acquisitions(
    dict: &Bound<'_, pyo3::types::PyDict>,
    num_regions: usize,
) -> PyResult<Vec<Option<AcquisitionConfig>>> {
    let names = opt_val::<Vec<Option<String>>>(dict, "region_acquisitions")?;
    let betas = opt_val::<Vec<Option<f64>>>(dict, "region_acquisition_betas")?;
    let Some(names) = names else {
        if betas.is_some() {
            return Err(PyValueError::new_err(
                "regional acquisition betas require regional acquisitions",
            ));
        }
        return Ok(vec![None; num_regions]);
    };
    if names.len() != num_regions {
        return Err(PyValueError::new_err(format!(
            "region_acquisitions has {} entries; expected {num_regions}",
            names.len()
        )));
    }
    let betas = betas.unwrap_or_else(|| vec![None; names.len()]);
    if betas.len() != names.len() {
        return Err(PyValueError::new_err(
            "region_acquisition_betas must match region_acquisitions",
        ));
    }
    names
        .into_iter()
        .zip(betas)
        .map(|(name, beta)| parse_acquisition(name.as_deref(), beta))
        .collect()
}

fn parse_acquisition(name: Option<&str>, beta: Option<f64>) -> PyResult<Option<AcquisitionConfig>> {
    let Some(name) = name else {
        return if beta.is_none() {
            Ok(None)
        } else {
            Err(PyValueError::new_err(
                "regional acquisition beta requires a UCB override",
            ))
        };
    };
    let acquisition = match name {
        "ucb" => {
            let beta = beta.unwrap_or(2.0);
            if !beta.is_finite() || beta <= 0.0 {
                return Err(PyValueError::new_err(
                    "regional UCB beta must be finite and > 0",
                ));
            }
            AcquisitionConfig::UCB { beta }
        }
        "thompson" if beta.is_none() => AcquisitionConfig::Thompson,
        "random" if beta.is_none() => AcquisitionConfig::Random,
        "pareto" if beta.is_none() => AcquisitionConfig::Pareto,
        "thompson" | "random" | "pareto" => {
            return Err(PyValueError::new_err(
                "regional acquisition beta is only valid for UCB",
            ));
        }
        _ => return Err(PyValueError::new_err("unknown regional acquisition")),
    };
    Ok(Some(acquisition))
}

fn build_state(
    num_regions: usize,
    num_init: usize,
    length: baux::TRLengthConfig,
    dict: &Bound<'_, pyo3::types::PyDict>,
    dimension: usize,
    rng: &mut StdRng,
) -> PyResult<baux::Strategy> {
    let mut regions = MultiTrustRegionConfig::new(num_regions, length);
    regions.succ_tolerance = opt_val(dict, "region_successes")?.unwrap_or(3);
    regions.fail_tolerance = opt_val(dict, "region_failures")?.unwrap_or(5);
    regions.sharing_policy = parse_sharing(dict)?;
    regions.allocation_policy = parse_allocation(dict)?;
    regions.restart_memory_policy = parse_restart(dict)?;
    regions.restart_center_policy = parse_center(dict)?;
    regions.controller = parse_controller(dict)?;
    regions.region_acquisitions = parse_acquisitions(dict, num_regions)?;
    let state = StrategyState::configured(dimension, regions, num_init, rng)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(baux::Strategy::Experimental(state))
}

pub(super) fn build(
    bounds: ndarray::Array2<f64>,
    kind: &str,
    k: i32,
    num_init: usize,
    num_regions: usize,
    runtime: Option<&baux::config::RunConfig>,
    dict: &Bound<'_, pyo3::types::PyDict>,
    gp: Option<Py<PyAny>>,
    gp_factory: Option<Py<PyAny>>,
    fit_steps: usize,
    rng: &mut StdRng,
) -> PyResult<baux::BayesOptimizer> {
    let mut config = match kind {
        "enn" | "enn_multi_tr" => baux::turbo_enn(),
        "zero" | "gp" => baux::turbo_zero(),
        _ => return Err(PyValueError::new_err("unsupported regional optimizer kind")),
    };
    if let baux::SurrogateConfig::ENN(enn) = &mut config.surrogate {
        if k <= 0 {
            return Err(PyValueError::new_err("k must be positive"));
        }
        enn.k = k;
    }
    if let Some(runtime) = runtime {
        config = runtime.apply_to(config);
    }
    let length = baux::TRLengthConfig::new(
        opt_val(dict, "length_init")?.unwrap_or(0.8),
        opt_val(dict, "length_min")?.unwrap_or(0.5_f64.powi(7)),
        opt_val(dict, "length_max")?.unwrap_or(1.6),
    );
    let dimension = bounds.nrows();
    let strategy = build_state(num_regions, num_init, length, dict, dimension, rng)?;
    let result = if kind == "gp" {
        build_gp(bounds, config, strategy, gp, gp_factory, fit_steps, rng)
    } else {
        baux::BayesOptimizer::new_strategy(bounds, config, strategy, rng)
    };
    result.map_err(|e| PyValueError::new_err(e.to_string()))
}
