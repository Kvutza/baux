//! Unified Candle-style execution device selector for BAUX Python bindings.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

#[pyclass(name = "Device", module = "baux", frozen)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PyDevice {
    pub(crate) kind: &'static str,
}

#[pymethods]
impl PyDevice {
    #[new]
    #[pyo3(signature = (name="auto"))]
    fn new(name: &str) -> PyResult<Self> {
        match name.to_ascii_lowercase().as_str() {
            "auto" | "" => Ok(Self::auto()),
            "cpu" => Ok(Self::cpu()),
            "metal" => Ok(Self { kind: "metal" }),
            "cuda" => Ok(Self { kind: "cuda" }),
            "opencl" => Ok(Self { kind: "opencl" }),
            _ => Err(PyValueError::new_err(format!("Unknown device: {name}"))),
        }
    }

    #[staticmethod]
    fn cpu() -> Self {
        Self { kind: "cpu" }
    }

    #[staticmethod]
    fn metal() -> PyResult<Self> {
        if !baux::Backend::Metal.compiled() {
            return Err(PyValueError::new_err(
                "Metal backend was not compiled into this library",
            ));
        }
        Ok(Self { kind: "metal" })
    }

    #[staticmethod]
    fn cuda() -> PyResult<Self> {
        if !baux::Backend::Cuda.compiled() {
            return Err(PyValueError::new_err(
                "CUDA backend was not compiled into this library",
            ));
        }
        Ok(Self { kind: "cuda" })
    }

    #[staticmethod]
    fn auto() -> Self {
        let kind = if baux::Backend::Metal.compiled() {
            "metal"
        } else if baux::Backend::Cuda.compiled() {
            "cuda"
        } else {
            "cpu"
        };
        Self { kind }
    }

    #[getter]
    fn kind(&self) -> &'static str {
        self.kind
    }

    #[getter]
    fn is_cpu(&self) -> bool {
        self.kind == "cpu"
    }

    #[getter]
    fn is_metal(&self) -> bool {
        self.kind == "metal"
    }

    #[getter]
    fn is_cuda(&self) -> bool {
        self.kind == "cuda"
    }

    fn memory_info<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        #[cfg(all(target_os = "macos", feature = "metal"))]
        if self.kind == "metal" {
            return crate::py_metal::PyMetalWeights::device_info(py);
        }
        let _ = py;
        Err(PyValueError::new_err(format!(
            "memory_info is not available for device {:?}",
            self.kind
        )))
    }

    fn __repr__(&self) -> String {
        format!("Device({:?})", self.kind)
    }

    fn __str__(&self) -> &'static str {
        self.kind
    }

    fn __eq__(&self, other: &Bound<'_, PyAny>) -> bool {
        if let Ok(dev) = other.extract::<PyRef<'_, Self>>() {
            return self.kind == dev.kind;
        }
        if let Ok(s) = other.extract::<&str>() {
            return self.kind == s;
        }
        false
    }
}

pub(crate) fn py_flags(
    tie_neighbors: &std::cell::Cell<bool>,
    exclude_nearest: bool,
    observation_noise: bool,
) -> baux::PosteriorFlags {
    baux::PosteriorFlags::new()
        .exclude_nearest(exclude_nearest)
        .observation_noise(observation_noise)
        .tie_neighbors(tie_neighbors.get())
}

pub(crate) fn parse_driver(index_driver: &str) -> PyResult<baux::IndexDriver> {
    match index_driver {
        "Exact" | "exact" | "FLAT" | "flat" | "cpu" | "CPU" => Ok(baux::IndexDriver::Exact),
        "AUTO" | "Auto" | "auto" => Ok(baux::IndexDriver::Auto),
        "USEARCH" | "USearch" | "usearch" | "hnsw_usearch" | "hnsw" | "HNSW" => {
            Ok(baux::IndexDriver::USearch)
        }
        "BPANN_DISK" | "bpann_disk" | "tree" | "TREE" | "bpann" | "BPANN" => {
            Ok(baux::IndexDriver::BpAnnDisk)
        }
        "METAL" | "metal" => Ok(baux::IndexDriver::Metal),
        "OPENCL" | "OpenCL" | "opencl" | "ocl" => Ok(baux::IndexDriver::OpenCl),
        "CUDA" | "Cuda" | "cuda" => Ok(baux::IndexDriver::Cuda),
        "CUVS" | "CuVS" | "cuvs" | "cuvs_brute_force" => Ok(baux::IndexDriver::Cuvs),
        "CUVS_CAGRA" | "cuvs_cagra" | "CAGRA" | "cagra" => Ok(baux::IndexDriver::CuvsCagra),
        _ => Err(PyValueError::new_err(format!(
            "Unknown index_driver: {index_driver}"
        ))),
    }
}

fn device_driver(dev: &Bound<'_, PyAny>) -> PyResult<baux::IndexDriver> {
    if let Ok(d) = dev.extract::<PyRef<'_, PyDevice>>() {
        return parse_driver(d.kind);
    }
    parse_driver(dev.extract::<&str>()?)
}

pub(crate) fn resolve_driver(
    index_driver: &str,
    device: Option<&Bound<'_, PyAny>>,
    index: Option<&str>,
) -> PyResult<baux::IndexDriver> {
    if let Some(idx) = index {
        return match idx.to_ascii_lowercase().as_str() {
            "tree" | "bpann" | "bpann_disk" => Ok(baux::IndexDriver::BpAnnDisk),
            "hnsw" | "usearch" => Ok(baux::IndexDriver::USearch),
            "cagra" | "cuvs_cagra" => Ok(baux::IndexDriver::CuvsCagra),
            "auto" => Ok(baux::IndexDriver::Auto),
            "exact" | "flat" => device.map_or(Ok(baux::IndexDriver::Exact), device_driver),
            _ => parse_driver(idx),
        };
    }
    if index_driver == "Exact" {
        if let Some(dev) = device {
            return device_driver(dev);
        }
    }
    parse_driver(index_driver)
}

pub(crate) fn parse_storage(
    enn_storage: Option<&str>,
    storage: Option<&str>,
    work_dir: Option<&str>,
    driver: baux::IndexDriver,
) -> PyResult<baux::EnnStorage> {
    let choice = storage.or(enn_storage);
    match choice {
        Some("disk" | "Disk" | "DISK") => Ok(baux::EnnStorage::Disk),
        Some("memory" | "in_memory" | "InMemory" | "MEMORY" | "ram" | "RAM") => {
            Ok(baux::EnnStorage::InMemory)
        }
        None if work_dir.is_some()
            || driver == baux::IndexDriver::BpAnnDisk && baux::EnnStorage::work_env().is_some() =>
        {
            Ok(baux::EnnStorage::Disk)
        }
        None => Ok(baux::EnnStorage::InMemory),
        Some(other) => Err(PyValueError::new_err(format!(
            "Unknown enn_storage: {other}"
        ))),
    }
}

type MetricSpec = (
    Option<String>,
    Option<Vec<Vec<usize>>>,
    Option<Vec<usize>>,
    Option<u64>,
);

fn parse_metric(spec: &Bound<'_, PyAny>) -> PyResult<MetricSpec> {
    if let Ok(s) = spec.extract::<String>() {
        return Ok((Some(s), None, None, None));
    }
    let dict = spec
        .cast::<PyDict>()
        .map_err(|_| PyValueError::new_err("Invalid metric specification"))?;
    let ml_str = dict
        .get_item("metric_learning")?
        .map(|v| v.extract())
        .transpose()?;
    let tied = dict
        .get_item("tied_dims")?
        .map(|v| v.extract())
        .transpose()?;
    let unscaled = dict
        .get_item("unscaled_dims")?
        .map(|v| v.extract())
        .transpose()?;
    let seed = dict
        .get_item("metric_seed")?
        .map(|v| v.extract())
        .transpose()?;
    Ok((ml_str, tied, unscaled, seed))
}

pub(crate) fn apply_metric(
    model: &mut baux::ENN,
    metric: Option<&Bound<'_, PyAny>>,
    scale_x: bool,
) -> PyResult<()> {
    let Some(spec) = metric else {
        return Ok(());
    };
    let (ml_str, tied, unscaled, seed) = parse_metric(spec)?;
    let ml = match ml_str.as_deref() {
        None => baux::MetricLearning::None,
        Some(name) => baux::MetricLearning::parse(name)
            .ok_or_else(|| PyValueError::new_err(format!("Unknown metric_learning: {name}")))?,
    };
    if ml == baux::MetricLearning::Auto {
        if scale_x {
            return Err(PyValueError::new_err(
                "metric_learning='auto' requires scale_x=False",
            ));
        }
        model
            .learn_metric(tied.unwrap_or_default(), baux::resolve_seed(seed))
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
    } else if let Some(groups) = tied {
        model
            .set_tied(groups.clone())
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        if scale_x && unscaled.is_none() {
            let mut uniq: Vec<usize> = groups.into_iter().flatten().collect();
            uniq.sort_unstable();
            uniq.dedup();
            model
                .set_unscaled(uniq)
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
        }
    }
    if let Some(dims) = unscaled {
        model
            .set_unscaled(dims)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
    }
    Ok(())
}

pub(crate) fn metric_dict<'py>(
    py: Python<'py>,
    model: &baux::ENN,
) -> PyResult<Option<Bound<'py, PyDict>>> {
    let Some(dm) = model.diagonal_metric() else {
        return Ok(None);
    };
    let dict = PyDict::new(py);
    dict.set_item("num_seen", dm.num_seen)?;
    dict.set_item("heldout_gain", dm.heldout_gain)?;
    dict.set_item("uses_learned_metric", dm.uses_learned)?;
    dict.set_item("weights", dm.weights)?;
    dict.set_item("built", dm.built)?;
    dict.set_item("num_refits", dm.counters.num_refits)?;
    dict.set_item("num_rescales", dm.counters.num_rescales)?;
    dict.set_item("num_rebuilds", dm.counters.num_rebuilds)?;
    Ok(Some(dict))
}
