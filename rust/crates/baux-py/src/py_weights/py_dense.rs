#![allow(clippy::too_many_arguments)]

#[cfg(all(target_os = "linux", target_arch = "x86_64", feature = "cuda"))]
use std::sync::Arc;
#[cfg(all(target_os = "linux", target_arch = "x86_64", feature = "cuda"))]
use std::sync::atomic::{AtomicBool, Ordering};

use baux::experimental::{
    self, AcquisitionKind, ComputeDevice, DenseLeaf, DenseLinear, DenseTerm, DenseView,
    WeightBlock, WeightSelectConfig,
};
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use super::{array1_vec, array2_vec, err};

pub(super) fn mixed_blocks(
    raw: Vec<(usize, usize, u8, f32, f32, f32)>,
) -> PyResult<Vec<WeightBlock>> {
    raw.into_iter()
        .map(
            |(offset, length, bits, quantization_scale, metric_scale, weight)| {
                WeightBlock::new(
                    offset,
                    length,
                    bits,
                    quantization_scale,
                    metric_scale,
                    weight,
                )
                .map_err(err)
            },
        )
        .collect()
}

#[cfg(all(target_os = "linux", target_arch = "x86_64", feature = "cuda"))]
type PyObject = Py<PyAny>;

#[cfg(all(target_os = "linux", target_arch = "x86_64", feature = "cuda"))]
/// CUDA BF16 parameter storage with DLPack exchange for array frameworks.
///
/// An exported array retains the storage owner. Materialization is rejected
/// until that array releases its lease, preventing writes through a live view.
#[pyclass(name = "ParameterBuffer", unsendable)]
pub struct PyParamBuffer {
    inner: experimental::ParamBuffer,
    exported: Arc<AtomicBool>,
}

#[cfg(all(target_os = "linux", target_arch = "x86_64", feature = "cuda"))]
#[pymethods]
impl PyParamBuffer {
    #[new]
    fn new(
        base: &Bound<'_, PyAny>,
        blocks: Vec<PyRef<'_, crate::py_bf16::PyParamBlock>>,
    ) -> PyResult<Self> {
        let input = crate::dlpack::Input::new(base)?;
        let leaves = blocks
            .iter()
            .map(|block| {
                let block = block.inner;
                DenseLeaf::new(block.key, block.offset, block.len, block.scale).map_err(err)
            })
            .collect::<PyResult<Vec<_>>>()?;
        Ok(Self {
            inner: unsafe {
                experimental::ParamBuffer::from_device(input.pointer, input.len, leaves)
            }
            .map_err(err)?,
            exported: Arc::new(AtomicBool::new(false)),
        })
    }

    fn materialize(&mut self, terms: Vec<(u64, f32)>) -> PyResult<()> {
        if self.exported.load(Ordering::Acquire) {
            return Err(PyValueError::new_err(
                "cannot materialize while an exported candidate is alive",
            ));
        }
        self.inner.materialize(&dense_terms(terms)?).map_err(err)
    }

    fn __dlpack_device__(&self) -> (i32, i32) {
        (2, 0)
    }

    #[pyo3(signature=(stream=None,max_version=None,dl_device=None,copy=None))]
    fn __dlpack__(
        slf: PyRef<'_, Self>,
        stream: Option<i64>,
        max_version: Option<(u32, u32)>,
        dl_device: Option<(i32, i32)>,
        copy: Option<bool>,
    ) -> PyResult<PyObject> {
        if copy == Some(true) {
            return Err(pyo3::exceptions::PyBufferError::new_err(
                "ParameterBuffer does not export copies",
            ));
        }
        if dl_device.is_some_and(|device| device != (2, 0)) {
            return Err(pyo3::exceptions::PyBufferError::new_err(
                "ParameterBuffer cannot export to another device",
            ));
        }
        if stream.is_some_and(|value| value == 0 || value < -1) {
            return Err(PyValueError::new_err("invalid DLPack CUDA stream"));
        }
        if slf
            .exported
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .is_err()
        {
            return Err(PyValueError::new_err(
                "the current BF16 candidate is already exported",
            ));
        }
        let py = slf.py();
        let (pointer, _, _) = match slf.inner.device_ptr(stream) {
            Ok(value) => value,
            Err(error) => {
                slf.exported.store(false, Ordering::Release);
                return Err(err(error));
            }
        };
        let len = slf.inner.len();
        let lease = Arc::clone(&slf.exported);
        let owner = slf.into_pyobject(py).unwrap().into_any().unbind();
        match crate::dlpack::export(py, owner, Arc::clone(&lease), pointer, len, max_version) {
            Ok(capsule) => Ok(capsule),
            Err(error) => {
                lease.store(false, Ordering::Release);
                Err(error)
            }
        }
    }

    #[getter]
    fn len(&self) -> usize {
        self.inner.len()
    }
}

fn int4_blocks(raw: Vec<(usize, usize, f32, f32, f32)>) -> PyResult<Vec<WeightBlock>> {
    raw.into_iter()
        .map(
            |(offset, length, quantization_scale, metric_scale, weight)| {
                WeightBlock::new(offset, length, 4, quantization_scale, metric_scale, weight)
                    .map_err(err)
            },
        )
        .collect()
}

fn dense_leaves(raw: Vec<(u64, usize, usize, f32)>) -> PyResult<Vec<DenseLeaf>> {
    raw.into_iter()
        .map(|(key, offset, len, scale)| DenseLeaf::new(key, offset, len, scale).map_err(err))
        .collect()
}

fn dense_terms(raw: Vec<(u64, f32)>) -> PyResult<Vec<DenseTerm>> {
    raw.into_iter()
        .map(|(seed, coefficient)| DenseTerm::new(seed, coefficient).map_err(err))
        .collect()
}

fn dense_view(raw: (u64, u64, f32)) -> PyResult<DenseView> {
    DenseView::new(raw.0, raw.1, raw.2).map_err(err)
}

#[pyfunction(name = "dense_apply")]
#[pyo3(signature=(base,leaves,terms,device="auto"))]
pub fn dense_py<'py>(
    py: Python<'py>,
    base: PyReadonlyArray1<'_, f32>,
    leaves: Vec<(u64, usize, usize, f32)>,
    terms: Vec<(u64, f32)>,
    device: &str,
) -> PyResult<(Bound<'py, PyArray1<f32>>, usize)> {
    let result = experimental::apply_dense(
        &array1_vec(base),
        &dense_leaves(leaves)?,
        &dense_terms(terms)?,
        ComputeDevice::parse(device).map_err(err)?,
    )
    .map_err(err)?;
    Ok((result.values.into_pyarray(py), result.changed))
}

#[pyfunction(name = "dense_dist2")]
pub fn bind_dist(
    leaves: Vec<(u64, usize, usize, f32)>,
    left: Vec<(u64, f32)>,
    right: Vec<(u64, f32)>,
) -> PyResult<f64> {
    experimental::dense_dist2(
        &dense_leaves(leaves)?,
        &dense_terms(left)?,
        &dense_terms(right)?,
    )
    .map_err(err)
}

#[pyfunction(name = "dense_linear")]
#[pyo3(signature=(input,weight,weight_view,terms,bias=None,bias_view=None,device="auto"))]
pub fn bind_linear<'py>(
    py: Python<'py>,
    input: PyReadonlyArray1<'_, f32>,
    weight: PyReadonlyArray2<'_, f32>,
    weight_view: (u64, u64, f32),
    terms: Vec<(u64, f32)>,
    bias: Option<PyReadonlyArray1<'_, f32>>,
    bias_view: Option<(u64, u64, f32)>,
    device: &str,
) -> PyResult<Bound<'py, PyArray1<f32>>> {
    let input = array1_vec(input);
    let weight = array2_vec(&weight);
    let bias = bias.map(array1_vec);
    let bias_view = bias_view.map(dense_view).transpose()?;
    experimental::dense_linear(
        &input,
        &weight,
        bias.as_deref(),
        dense_view(weight_view)?,
        bias_view,
        &dense_terms(terms)?,
        ComputeDevice::parse(device).map_err(err)?,
    )
    .map(|values| values.into_pyarray(py))
    .map_err(err)
}

#[pyclass(name = "DenseLinear", unsendable)]
pub struct PyDenseLinear {
    inner: DenseLinear,
}

#[pymethods]
impl PyDenseLinear {
    #[new]
    #[pyo3(signature=(weight,weight_view,bias=None,bias_view=None,device="auto"))]
    fn new(
        weight: PyReadonlyArray2<'_, f32>,
        weight_view: (u64, u64, f32),
        bias: Option<PyReadonlyArray1<'_, f32>>,
        bias_view: Option<(u64, u64, f32)>,
        device: &str,
    ) -> PyResult<Self> {
        let columns = weight.as_array().ncols();
        let bias = bias.map(array1_vec);
        Ok(Self {
            inner: DenseLinear::new(
                array2_vec(&weight),
                columns,
                bias,
                dense_view(weight_view)?,
                bias_view.map(dense_view).transpose()?,
                ComputeDevice::parse(device).map_err(err)?,
            )
            .map_err(err)?,
        })
    }

    fn eval<'py>(
        &mut self,
        py: Python<'py>,
        input: PyReadonlyArray1<'_, f32>,
        terms: Vec<(u64, f32)>,
    ) -> PyResult<Bound<'py, PyArray1<f32>>> {
        self.inner
            .eval(&array1_vec(input), &dense_terms(terms)?)
            .map(|values| values.into_pyarray(py))
            .map_err(err)
    }

    #[getter]
    fn input_size(&self) -> usize {
        self.inner.input_size()
    }

    #[getter]
    fn output_size(&self) -> usize {
        self.inner.output_size()
    }
}

fn weight_ucb<'py>(
    py: Python<'py>,
    observations: PyReadonlyArray2<'_, u8>,
    outcomes: PyReadonlyArray1<'_, f32>,
    candidates: PyReadonlyArray2<'_, u8>,
    blocks: Vec<WeightBlock>,
    neighbors: usize,
    epistemic_scale: f32,
    aleatoric_scale: f32,
    y_scale: f32,
    beta: f32,
    device: &str,
) -> PyResult<(Bound<'py, PyArray1<u8>>, usize, f32)> {
    let observation_count = observations.as_array().nrows();
    let candidate_count = candidates.as_array().nrows();
    let observation_bytes = array2_vec(&observations);
    let candidate_bytes = array2_vec(&candidates);
    let outcome_vec = array1_vec(outcomes);
    let result = experimental::select_weights(
        &observation_bytes,
        observation_count,
        &outcome_vec,
        &candidate_bytes,
        candidate_count,
        &blocks,
        WeightSelectConfig {
            neighbors,
            epistemic_scale,
            aleatoric_scale,
            y_scale,
            beta,
            acquisition: AcquisitionKind::Ucb,
            seed: 0,
            device: ComputeDevice::parse(device).map_err(err)?,
        },
    )
    .map_err(err)?;
    let row_bytes = candidates.as_array().ncols();
    let start = result.index * row_bytes;
    let selected = candidate_bytes[start..start + row_bytes].to_vec();
    Ok((selected.into_pyarray(py), result.index, result.score))
}

#[pyfunction(name = "weight_int4_select_ucb")]
#[pyo3(signature=(observations,outcomes,candidates,blocks,neighbors,epistemic_scale,aleatoric_scale,y_scale,beta,device="auto"))]
pub fn weight_py<'py>(
    py: Python<'py>,
    observations: PyReadonlyArray2<'_, u8>,
    outcomes: PyReadonlyArray1<'_, f32>,
    candidates: PyReadonlyArray2<'_, u8>,
    blocks: Vec<(usize, usize, f32, f32, f32)>,
    neighbors: usize,
    epistemic_scale: f32,
    aleatoric_scale: f32,
    y_scale: f32,
    beta: f32,
    device: &str,
) -> PyResult<(Bound<'py, PyArray1<u8>>, usize, f32)> {
    weight_ucb(
        py,
        observations,
        outcomes,
        candidates,
        int4_blocks(blocks)?,
        neighbors,
        epistemic_scale,
        aleatoric_scale,
        y_scale,
        beta,
        device,
    )
}

#[pyfunction(name = "weight_select_ucb")]
#[pyo3(signature=(observations,outcomes,candidates,blocks,neighbors,epistemic_scale,aleatoric_scale,y_scale,beta,device="auto"))]
pub fn select_weight<'py>(
    py: Python<'py>,
    observations: PyReadonlyArray2<'_, u8>,
    outcomes: PyReadonlyArray1<'_, f32>,
    candidates: PyReadonlyArray2<'_, u8>,
    blocks: Vec<(usize, usize, u8, f32, f32, f32)>,
    neighbors: usize,
    epistemic_scale: f32,
    aleatoric_scale: f32,
    y_scale: f32,
    beta: f32,
    device: &str,
) -> PyResult<(Bound<'py, PyArray1<u8>>, usize, f32)> {
    weight_ucb(
        py,
        observations,
        outcomes,
        candidates,
        mixed_blocks(blocks)?,
        neighbors,
        epistemic_scale,
        aleatoric_scale,
        y_scale,
        beta,
        device,
    )
}
