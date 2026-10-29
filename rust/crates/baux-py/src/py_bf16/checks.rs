use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use pyo3::exceptions::{PyBufferError, PyValueError};
use pyo3::prelude::*;

type PyObject = Py<PyAny>;

pub(super) fn err(error: String) -> PyErr {
    PyValueError::new_err(error)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Sampler {
    Independent,
    Gaussian,
    Correlated,
}

pub(super) fn check_sampler(
    sampler: &str,
    max_pending: usize,
    failure_tolerance: Option<usize>,
) -> Result<Sampler, String> {
    let sampler = match sampler {
        "independent" => Sampler::Independent,
        "gaussian" => Sampler::Gaussian,
        "correlated" => Sampler::Correlated,
        _ => return Err("sampler must be 'independent', 'gaussian', or 'correlated'".into()),
    };
    if sampler == Sampler::Correlated && (max_pending != 1 || failure_tolerance.is_some()) {
        return Err("correlated sampling requires max_pending=1 and failure_tolerance=None".into());
    }
    if failure_tolerance.is_some_and(|value| value == 0 || u32::try_from(value).is_err()) {
        return Err("failure_tolerance must be a positive u32".into());
    }
    Ok(sampler)
}

pub(super) fn check_handle(exported: bool, generation: u64, current: u64) -> Result<(), String> {
    if exported || generation != current {
        return Err("Stale or already exported incumbent; request a fresh incumbent()".into());
    }
    Ok(())
}

pub(super) fn check_pair(
    value: f64,
    variance: f64,
    incumbent_value: f64,
    incumbent_variance: f64,
) -> Result<[f32; 4], String> {
    let inputs = [value, variance, incumbent_value, incumbent_variance];
    if variance < 0.0
        || incumbent_variance < 0.0
        || inputs
            .iter()
            .any(|value| !value.is_finite() || !(*value as f32).is_finite())
    {
        return Err(
            "Paired rewards must be finite FP32 values and both variances finite and nonnegative"
                .into(),
        );
    }
    Ok(inputs.map(|value| value as f32))
}

pub(super) fn check_scores(
    value: f64,
    variance: f64,
    incumbent_value: f64,
    incumbent_variance: f64,
    improvement: f64,
    improvement_variance: f64,
) -> Result<[f32; 6], String> {
    let [value, variance, incumbent_value, incumbent_variance] =
        check_pair(value, variance, incumbent_value, incumbent_variance)?;
    if improvement_variance < 0.0
        || [improvement, improvement_variance]
            .iter()
            .any(|value| !value.is_finite() || !(*value as f32).is_finite())
    {
        return Err(
            "Paired improvement must be finite FP32 and its variance finite and nonnegative".into(),
        );
    }
    Ok([
        value,
        variance,
        incumbent_value,
        incumbent_variance,
        improvement as f32,
        improvement_variance as f32,
    ])
}

pub(super) fn check_export(
    stream: Option<i64>,
    dl_device: Option<(i32, i32)>,
    copy: Option<bool>,
) -> PyResult<()> {
    if copy == Some(true) {
        return Err(PyBufferError::new_err("BF16 search does not export copies"));
    }
    if dl_device.is_some_and(|device| device != (2, 0)) {
        return Err(PyBufferError::new_err(
            "BF16 search cannot export to another device",
        ));
    }
    if stream.is_some_and(|value| value == 0 || value < -1) {
        return Err(PyValueError::new_err("invalid DLPack CUDA stream"));
    }
    Ok(())
}

pub(super) fn export_batch(
    py: Python<'_>,
    owner: PyObject,
    lease: Arc<AtomicUsize>,
    pointer: u64,
    rows: usize,
    columns: usize,
    max_version: Option<(u32, u32)>,
) -> PyResult<PyObject> {
    lease.fetch_add(1, Ordering::AcqRel);
    let result = crate::dlpack::export_batch(
        py,
        owner,
        Arc::clone(&lease),
        pointer,
        rows,
        columns,
        max_version,
    );
    if result.is_err() {
        // Capsule construction failure already runs its deleter, whereas shape
        // validation failure does not. Release only a lease still held here.
        let _ = lease.compare_exchange(1, 0, Ordering::AcqRel, Ordering::Acquire);
    }
    result
}

#[cfg(test)]
mod tests {
    use super::{Sampler, check_handle, check_pair, check_sampler, check_scores};

    #[test]
    fn relative_scores() {
        let inputs = [-1.0, 0.25, -2.0, 0.5, 0.125, 0.0625];
        let check = |x: [f64; 6]| check_scores(x[0], x[1], x[2], x[3], x[4], x[5]);
        assert_eq!(check(inputs).unwrap(), inputs.map(|value| value as f32));
        for index in 0..6 {
            for invalid in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, 1e100, -1e100] {
                let mut bad = inputs;
                bad[index] = invalid;
                assert!(check(bad).is_err(), "index {index}, value {invalid}");
            }
        }
        for index in [1, 3, 5] {
            for invalid in [-1.0, -1e-100] {
                let mut bad = inputs;
                bad[index] = invalid;
                assert!(check(bad).is_err());
            }
        }
        // Improvement is supplied directly, not reconstructed from rounded
        // absolute scores.
        assert_eq!(
            check_scores(1e8, 0.0, 1e8, 0.0, -1e-6, 0.0).unwrap()[4],
            -1e-6_f32
        );
    }

    #[test]
    fn pair_scores() {
        assert!(check_pair(-1.0, 0.0, -2.0, 0.5).is_ok());
        for variance in [-1e-100, -1.0, f64::NAN, f64::INFINITY, 1e100] {
            assert!(check_pair(0.0, variance, 0.0, 0.0).is_err());
            assert!(check_pair(0.0, 0.0, 0.0, variance).is_err());
        }
        for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, 1e100] {
            assert!(check_pair(value, 0.0, 0.0, 0.0).is_err());
            assert!(check_pair(0.0, 0.0, value, 0.0).is_err());
        }
    }

    #[test]
    fn handle_guards() {
        assert!(check_handle(false, 1, 1).is_ok());
        assert!(check_handle(true, 1, 1).is_err());
        assert!(check_handle(false, 1, 2).is_err());
        assert!(check_handle(true, 1, 2).is_err());
    }

    #[test]
    fn config_guard() {
        assert_eq!(
            check_sampler("independent", 1, None),
            Ok(Sampler::Independent)
        );
        assert_eq!(
            check_sampler("independent", 8, Some(4)),
            Ok(Sampler::Independent)
        );
        assert_eq!(check_sampler("gaussian", 1, None), Ok(Sampler::Gaussian));
        assert_eq!(check_sampler("gaussian", 8, Some(4)), Ok(Sampler::Gaussian));
        assert_eq!(
            check_sampler("correlated", 1, None),
            Ok(Sampler::Correlated)
        );
        assert!(check_sampler("unknown", 1, None).is_err());
        for pending in [0, 2, 32] {
            assert!(check_sampler("correlated", pending, None).is_err());
        }
        for tolerance in [0, 1, 4, usize::MAX] {
            assert!(check_sampler("correlated", 1, Some(tolerance)).is_err());
        }
        for sampler in ["independent", "gaussian"] {
            assert!(check_sampler(sampler, 1, Some(0)).is_err());
            assert!(check_sampler(sampler, 1, Some(u32::MAX as usize)).is_ok());
            if usize::BITS > 32 {
                assert!(check_sampler(sampler, 1, Some(usize::MAX)).is_err());
            }
        }
    }
}
