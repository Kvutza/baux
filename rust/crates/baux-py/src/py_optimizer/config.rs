//! Legacy dictionary configuration parsing for Python bindings.

use std::path::PathBuf;

use numpy::{PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

pub(super) fn opt_val<'py, T>(
    dict: &Bound<'py, pyo3::types::PyDict>,
    key: &str,
) -> PyResult<Option<T>>
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
            ("cuvs", IndexDriver::Cuvs),
            ("cuvs_brute_force", IndexDriver::Cuvs),
            ("cuvs_cagra", IndexDriver::CuvsCagra),
            ("cagra", IndexDriver::CuvsCagra),
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
