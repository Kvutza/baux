//! Full FBT/PISA model entry point over the shared resident tensor runtime.
use crate::py_native::PyNativePending;
use baux_fbt::moe::program::{Backend, Resident};
use numpy::PyReadonlyArray1;
use pyo3::{exceptions::PyValueError, prelude::*};
use std::sync::{Arc, Mutex};

fn architecture(value: &str) -> PyResult<baux::config::ModelArchitecture> {
    match value {
        "fbt-pisa1-mhc4-v1" => Ok(baux::config::ModelArchitecture::FbtPisa1Mhc4V1),
        _ => Err(PyValueError::new_err(format!(
            "unsupported resident model {value:?}"
        ))),
    }
}

fn initialization_kind(value: &str) -> PyResult<baux::config::ModelInitialization> {
    match value {
        "fbt-pisa1-mhc4" => Ok(baux::config::ModelInitialization::FbtPisa1Mhc4),
        _ => Err(PyValueError::new_err(format!(
            "unsupported model initialization {value:?}"
        ))),
    }
}

fn contiguous<'a, T: numpy::Element>(
    value: &'a PyReadonlyArray1<'_, T>,
    name: &str,
) -> PyResult<&'a [T]> {
    value.as_slice().map_err(|_| {
        PyValueError::new_err(format!("{name} must be a contiguous one-dimensional array"))
    })
}

#[pyclass(name = "ResidentModel", frozen)]
pub struct PyResidentModel {
    inner: Arc<Mutex<Resident>>,
}

#[pymethods]
impl PyResidentModel {
    #[new]
    #[pyo3(signature = (*, model="fbt-pisa1-mhc4-v1", initialization="fbt-pisa1-mhc4", seed=42, backend="native"))]
    fn new(
        py: Python<'_>,
        model: &str,
        initialization: &str,
        seed: u64,
        backend: &str,
    ) -> PyResult<Self> {
        let model = architecture(model)?;
        let initialization = initialization_kind(initialization)?;
        let backend = match backend {
            "native" => Backend::Native,
            "tvm" => Backend::Tvm,
            _ => {
                return Err(PyValueError::new_err(format!(
                    "unknown resident backend {backend:?}"
                )));
            }
        };
        let inner = py
            .detach(|| {
                Resident::with_backend(
                    model,
                    initialization,
                    seed,
                    baux::trust_region::TRLengthConfig::default(),
                    baux::ProposalKernel::Gaussian,
                    backend,
                )
            })
            .map_err(PyValueError::new_err)?;
        Ok(Self {
            inner: Arc::new(Mutex::new(inner)),
        })
    }

    #[getter]
    fn backend(&self) -> PyResult<&'static str> {
        Ok(
            match self
                .inner
                .lock()
                .map_err(|_| PyValueError::new_err("resident model mutex poisoned"))?
                .backend
            {
                Backend::Native => "native",
                Backend::Tvm => "tvm",
            },
        )
    }

    #[getter]
    fn parameters(&self) -> PyResult<usize> {
        Ok(self
            .inner
            .lock()
            .map_err(|_| PyValueError::new_err("resident model mutex poisoned"))?
            .parameters)
    }

    #[getter]
    fn rows(&self) -> PyResult<usize> {
        Ok(self
            .inner
            .lock()
            .map_err(|_| PyValueError::new_err("resident model mutex poisoned"))?
            .rows())
    }

    fn submit(
        &self,
        py: Python<'_>,
        tokens: PyReadonlyArray1<'_, u32>,
        labels: PyReadonlyArray1<'_, u32>,
        mask: PyReadonlyArray1<'_, u8>,
    ) -> PyResult<PyNativePending> {
        let tokens = contiguous(&tokens, "tokens")?.to_vec();
        let labels = contiguous(&labels, "labels")?.to_vec();
        let mask = contiguous(&mask, "mask")?.to_vec();
        let pending = py
            .detach(|| {
                let resident = self
                    .inner
                    .lock()
                    .map_err(|_| "resident model mutex poisoned")?;
                let batch = resident.batch(&tokens, &labels, &mask)?;
                let weights = resident.search.program_base(&resident.forward)?;
                resident.submit(weights, &batch)
            })
            .map_err(PyValueError::new_err)?;
        Ok(PyNativePending {
            inner: Mutex::new(Some(pending)),
        })
    }
}
