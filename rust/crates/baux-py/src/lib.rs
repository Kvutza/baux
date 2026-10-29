//! Python bindings for ENN core algorithms using PyO3.

#![allow(
    clippy::pedantic,
    clippy::nursery,
    clippy::cargo,
    clippy::useless_conversion
)]

use pyo3::prelude::*;
use pyo3::wrap_pymodule;

pub mod adapter;
#[cfg_attr(
    not(all(target_os = "linux", target_arch = "x86_64", feature = "cuda")),
    allow(dead_code)
)]
mod dlpack;
pub mod link_rpath;
#[cfg(all(target_os = "linux", target_arch = "x86_64", feature = "cuda"))]
pub mod py_bf16;
#[cfg(all(target_os = "linux", target_arch = "x86_64", feature = "cuda"))]
pub mod py_cuda;
pub mod py_experimental;
pub mod py_fit;
pub mod py_fitter;
#[cfg(all(target_os = "linux", target_arch = "x86_64", feature = "native-flame"))]
pub mod py_flame;
#[cfg(any(
    all(target_os = "linux", target_arch = "x86_64", feature = "native-flame"),
    all(target_os = "macos", feature = "metal")
))]
mod py_flameconfig;
pub mod py_hash;
pub mod py_hypervolume;
#[cfg(all(target_os = "macos", feature = "metal"))]
mod py_metal;
pub mod py_model;
pub mod py_optimizer;
pub mod py_parameter;
pub mod py_posterior;
#[cfg(all(target_os = "macos", feature = "metal"))]
pub mod py_qwen;
pub mod py_regions;
pub mod py_util;
pub mod py_weights;

mod pymodule_wrappers;

pub use pymodule_wrappers::{
    pymodule_cuda, pymodule_experimental, pymodule_fit, pymodule_hash, pymodule_hypervolume,
    pymodule_model, pymodule_optimizer, pymodule_search, pymodule_util,
};

/// Main module, packaged by Bazel as `baux.baux_rust`.
#[pymodule]
#[pyo3(name = "baux_rust")]
pub(crate) fn baux_rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(py_util::capabilities, m)?)?;
    m.add_class::<py_regions::PyMultiTrustRegion>()?;
    #[cfg(all(target_os = "linux", target_arch = "x86_64", feature = "cuda"))]
    m.add_wrapped(wrap_pymodule!(pymodule_cuda))?;
    m.add_wrapped(wrap_pymodule!(pymodule_hypervolume))?;
    m.add_wrapped(wrap_pymodule!(pymodule_hash))?;
    m.add_wrapped(wrap_pymodule!(pymodule_util))?;
    m.add_wrapped(wrap_pymodule!(pymodule_model))?;
    m.add_wrapped(wrap_pymodule!(pymodule_fit))?;
    m.add_wrapped(wrap_pymodule!(pymodule_experimental))?;
    m.add_wrapped(wrap_pymodule!(pymodule_optimizer))?;
    m.add_wrapped(wrap_pymodule!(pymodule_search))?;
    Ok(())
}
