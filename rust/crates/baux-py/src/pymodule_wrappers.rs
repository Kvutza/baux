use pyo3::prelude::*;

/// Production CUDA diagnostics embedded in the Python wheel.
#[pymodule]
#[pyo3(name = "cuda")]
pub fn pymodule_cuda(m: &Bound<'_, PyModule>) -> PyResult<()> {
    #[cfg(all(target_os = "linux", target_arch = "x86_64", feature = "cuda"))]
    {
        m.add_function(wrap_pyfunction!(crate::py_cuda::target, m)?)?;
        m.add_function(wrap_pyfunction!(crate::py_cuda::check, m)?)?;
        m.add_function(wrap_pyfunction!(crate::py_cuda::trace, m)?)?;
    }
    Ok(())
}

/// Hypervolume calculation module
#[pymodule]
#[pyo3(name = "hypervolume")]
pub fn pymodule_hypervolume(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(crate::py_hypervolume::hypervolume_py, m)?)?;
    Ok(())
}

/// Hash-based RNG module
#[pymodule]
#[pyo3(name = "hash")]
pub fn pymodule_hash(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(crate::py_hash::normal_py, m)?)?;
    Ok(())
}

/// Utility functions module
#[pymodule]
#[pyo3(name = "util")]
pub fn pymodule_util(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(crate::py_util::y_py, m)?)?;
    m.add_function(wrap_pyfunction!(crate::py_util::pareto_py, m)?)?;
    m.add_function(wrap_pyfunction!(crate::py_util::sobol_py, m)?)?;
    m.add_function(wrap_pyfunction!(crate::py_util::bind_sobol, m)?)?;
    m.add_function(wrap_pyfunction!(crate::py_util::arms_py, m)?)?;
    m.add_function(wrap_pyfunction!(crate::py_util::set_py, m)?)?;
    m.add_function(wrap_pyfunction!(crate::py_util::ensure_py, m)?)?;
    Ok(())
}

/// ENN model module
#[pymodule]
#[pyo3(name = "model")]
pub fn pymodule_model(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<crate::py_posterior::PyDevicePosterior>()?;
    m.add("DevicePosterior", m.getattr("Posterior")?)?;
    m.add_class::<crate::py_posterior::PyDeviceArray>()?;
    m.add("DeviceArray", m.getattr("Tensor")?)?;
    m.add_class::<crate::py_model::PyENN>()?;
    m.add("Enn", m.getattr("ENN")?)?;
    m.add_class::<crate::py_fitter::PyENNParams>()?;
    Ok(())
}

/// Parameter fitting module
#[pymodule]
#[pyo3(name = "fit")]
pub fn pymodule_fit(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<crate::py_fitter::PyENNFitter>()?;
    m.add_function(wrap_pyfunction!(crate::py_fit::subsample_py, m)?)?;
    m.add_function(wrap_pyfunction!(crate::py_fit::rows_py, m)?)?;
    Ok(())
}

/// Experimental resident Metal search API.
#[pymodule]
#[pyo3(name = "experimental")]
pub fn pymodule_experimental(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<crate::py_device::PyDevice>()?;
    #[cfg(all(target_os = "macos", feature = "metal"))]
    {
        m.add_class::<crate::py_metal::PyMetalWeights>()?;
        m.add("ParameterBuffer", m.getattr("Var")?)?;
        m.add("ParamBuffer", m.getattr("Var")?)?;
        m.add("MetalWeights", m.getattr("Var")?)?;
        m.add_class::<crate::py_metal::PyMetalParamBlock>()?;
        m.add("ParamBlock", m.getattr("Span")?)?;
        m.add("MetalParamBlock", m.getattr("Span")?)?;
        m.add_class::<crate::py_metal::PyMetalSearchState>()?;
        m.add("SearchState", m.getattr("Region")?)?;
        m.add("MetalSearchState", m.getattr("Region")?)?;
        m.add_class::<crate::py_metal::PyMetalProposals>()?;
        m.add("Proposals", m.getattr("Trial")?)?;
        m.add("MetalProposals", m.getattr("Trial")?)?;
        m.add_class::<crate::py_portfolio::PyRegionalSearch>()?;
        m.add("RegionOptimizer", m.getattr("Portfolio")?)?;
        m.add("MetalRegionOptimizer", m.getattr("Portfolio")?)?;
        m.add_class::<crate::py_portfolio::PyRegionalProposal>()?;
        m.add("RegionalProposal", m.getattr("RegionTrial")?)?;
    }
    Ok(())
}

/// Optimizer module
#[pymodule]
#[pyo3(name = "optimizer")]
pub fn pymodule_optimizer(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<crate::py_device::PyDevice>()?;
    m.add_class::<crate::py_optimizer::PyBayesOptimizer>()?;
    m.add_class::<crate::py_regions::PyMultiTrustRegion>()?;
    m.add_class::<crate::py_optimizer::PyTelemetry>()?;
    #[cfg(all(target_os = "macos", feature = "metal"))]
    {
        m.add_class::<crate::py_metal::PyMetalWeights>()?;
        m.add_class::<crate::py_metal::PyMetalParamBlock>()?;
        m.add_class::<crate::py_metal::PyMetalSearchState>()?;
        m.add_class::<crate::py_metal::PyMetalProposals>()?;
        m.add_class::<crate::py_portfolio::PyRegionalSearch>()?;
        m.add_class::<crate::py_portfolio::PyRegionalProposal>()?;
    }
    m.add_function(wrap_pyfunction!(crate::py_optimizer::create_py, m)?)?;
    m.add_function(wrap_pyfunction!(crate::py_optimizer::enn_py, m)?)?;
    m.add_function(wrap_pyfunction!(crate::py_optimizer::bind_regions, m)?)?;
    m.add_function(wrap_pyfunction!(crate::py_optimizer::bind_zero, m)?)?;
    m.add_function(wrap_pyfunction!(crate::py_optimizer::bind_lhd, m)?)?;
    Ok(())
}
