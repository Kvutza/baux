#![allow(clippy::too_many_arguments)]

use baux::experimental::{self, AcquisitionKind, BpannHistory, ComputeDevice, WeightSelectConfig};
use baux::search::Trial;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};

use super::py_dense::mixed_blocks;
use super::{array1_vec, err};

#[pyclass(name = "Trial", module = "baux.search", frozen, from_py_object)]
#[derive(Clone)]
pub struct PyTrial {
    pub(super) inner: Trial,
}

#[pymethods]
impl PyTrial {
    #[getter]
    fn index(&self) -> usize {
        self.inner.index
    }

    #[getter]
    fn seed(&self) -> u64 {
        self.inner.seed
    }

    #[getter]
    fn score(&self) -> f32 {
        self.inner.score
    }
}

#[pyclass(name = "BpannHistory", unsendable)]
pub struct PyBpannHistory {
    inner: BpannHistory,
}

#[pymethods]
impl PyBpannHistory {
    #[new]
    fn new(work_dir: String, descriptor_dim: usize) -> PyResult<Self> {
        Ok(Self {
            inner: BpannHistory::new(work_dir.into(), descriptor_dim).map_err(err)?,
        })
    }

    fn append(&mut self, descriptor: PyReadonlyArray1<'_, f64>, value: f32) -> PyResult<u64> {
        self.inner
            .append(&descriptor.as_array(), value)
            .map(|id| id.0)
            .map_err(err)
    }

    fn search(
        &self,
        descriptors: PyReadonlyArray2<'_, f64>,
        neighbors: usize,
    ) -> PyResult<Vec<Vec<u64>>> {
        let rows = self
            .inner
            .search(&descriptors.as_array(), neighbors)
            .map_err(err)?;
        let mut result = Vec::with_capacity(rows.len());
        for row in rows {
            let ids = row.into_iter().map(|id| id.0).collect();
            result.push(ids);
        }
        Ok(result)
    }

    fn shortlist(
        &self,
        descriptors: PyReadonlyArray2<'_, f64>,
        neighbors_per_candidate: usize,
        max_observations: usize,
    ) -> PyResult<Vec<(u64, f32)>> {
        self.inner
            .shortlist(
                &descriptors.as_array(),
                neighbors_per_candidate,
                max_observations,
            )
            .map(|items| {
                items
                    .into_iter()
                    .map(|item| (item.id.0, item.value))
                    .collect()
            })
            .map_err(err)
    }

    fn sync(&mut self) -> PyResult<()> {
        self.inner.sync().map_err(err)
    }

    fn persist(&mut self) -> PyResult<()> {
        self.inner.persist().map_err(err)
    }

    #[getter]
    fn len(&self) -> usize {
        self.inner.len()
    }

    #[getter]
    fn descriptor_dim(&self) -> usize {
        self.inner.descriptor_dim()
    }
}

fn req_arg<'py, T>(
    args: &Bound<'py, PyTuple>,
    kwargs: Option<&Bound<'py, PyDict>>,
    index: usize,
    name: &str,
) -> PyResult<T>
where
    T: for<'a> FromPyObject<'a, 'py>,
    for<'a> PyErr: From<<T as FromPyObject<'a, 'py>>::Error>,
{
    if index < args.len() {
        return Ok(args.get_item(index)?.extract()?);
    }
    if let Some(kw) = kwargs
        && let Some(val) = kw.get_item(name)?
    {
        return Ok(val.extract()?);
    }
    Err(PyTypeError::new_err(format!(
        "missing required argument '{name}'"
    )))
}

#[pyfunction(name = "sparse_union")]
pub fn sparse_py<'py>(
    py: Python<'py>,
    rows: Vec<PyReadonlyArray1<'_, u32>>,
) -> PyResult<Bound<'py, PyArray1<u32>>> {
    let owned: Vec<Vec<u32>> = rows.into_iter().map(array1_vec).collect();
    let refs: Vec<&[u32]> = owned.iter().map(Vec::as_slice).collect();
    Ok(experimental::sparse_union(&refs).into_pyarray(py))
}

#[pyfunction(name = "sparse_xor")]
pub fn bind_xor<'py>(
    py: Python<'py>,
    left_words: PyReadonlyArray1<'_, u32>,
    left_masks: PyReadonlyArray1<'_, u32>,
    right_words: PyReadonlyArray1<'_, u32>,
    right_masks: PyReadonlyArray1<'_, u32>,
) -> PyResult<(Bound<'py, PyArray1<u32>>, Bound<'py, PyArray1<u32>>)> {
    let (words, masks) = experimental::sparse_xor(
        &array1_vec(left_words),
        &array1_vec(left_masks),
        &array1_vec(right_words),
        &array1_vec(right_masks),
    )
    .map_err(err)?;
    Ok((words.into_pyarray(py), masks.into_pyarray(py)))
}

#[pyfunction(name = "sparse_missing")]
pub fn bind_missing<'py>(
    py: Python<'py>,
    cached: PyReadonlyArray1<'_, u32>,
    query: PyReadonlyArray1<'_, u32>,
) -> PyResult<Bound<'py, PyArray1<u32>>> {
    Ok(experimental::missing_words(&array1_vec(cached), &array1_vec(query)).into_pyarray(py))
}

#[pyfunction(name = "sparse_merge")]
pub fn bind_merge<'py>(
    py: Python<'py>,
    words: PyReadonlyArray1<'_, u32>,
    values: PyReadonlyArray1<'_, u32>,
    extra_words: PyReadonlyArray1<'_, u32>,
    extra_values: PyReadonlyArray1<'_, u32>,
) -> PyResult<(Bound<'py, PyArray1<u32>>, Bound<'py, PyArray1<u32>>)> {
    let (words, values) = experimental::merge_values(
        &array1_vec(words),
        &array1_vec(values),
        &array1_vec(extra_words),
        &array1_vec(extra_values),
    )
    .map_err(err)?;
    Ok((words.into_pyarray(py), values.into_pyarray(py)))
}

#[pyfunction(name = "sparse_take")]
pub fn bind_take<'py>(
    py: Python<'py>,
    words: PyReadonlyArray1<'_, u32>,
    values: PyReadonlyArray1<'_, u32>,
    query: PyReadonlyArray1<'_, u32>,
) -> PyResult<Bound<'py, PyArray1<u32>>> {
    Ok(
        experimental::take_words(&array1_vec(words), &array1_vec(values), &array1_vec(query))
            .map_err(err)?
            .into_pyarray(py),
    )
}

#[pyfunction(name = "sparse_apply")]
pub fn bind_apply<'py>(
    py: Python<'py>,
    words: PyReadonlyArray1<'_, u32>,
    values: PyReadonlyArray1<'_, u32>,
    move_words: PyReadonlyArray1<'_, u32>,
    move_masks: PyReadonlyArray1<'_, u32>,
) -> PyResult<Bound<'py, PyArray1<u32>>> {
    Ok(experimental::apply_sparse(
        &array1_vec(words),
        &array1_vec(values),
        &array1_vec(move_words),
        &array1_vec(move_masks),
    )
    .map_err(err)?
    .into_pyarray(py))
}

#[pyfunction(name = "sparse_blocks")]
pub fn bind_blocks(
    words: PyReadonlyArray1<'_, u32>,
    word_ends: PyReadonlyArray1<'_, u32>,
    widths: PyReadonlyArray1<'_, u8>,
) -> PyResult<(Vec<(usize, usize, u8)>, usize)> {
    experimental::blocks_words(
        &array1_vec(words),
        &array1_vec(word_ends),
        &array1_vec(widths),
    )
    .map_err(err)
}

#[pyfunction(name = "sparse_draw")]
#[pyo3(signature=(count,size,dimension,parameter_ends,parameter_starts,word_offsets,widths,seed))]
pub fn bind_draw<'py>(
    py: Python<'py>,
    count: usize,
    size: usize,
    dimension: u64,
    parameter_ends: PyReadonlyArray1<'_, u64>,
    parameter_starts: PyReadonlyArray1<'_, u64>,
    word_offsets: PyReadonlyArray1<'_, u32>,
    widths: PyReadonlyArray1<'_, u8>,
    seed: u64,
) -> PyResult<(Bound<'py, PyList>, Bound<'py, PyList>)> {
    let rows = experimental::draw_sparse(
        count,
        size,
        dimension,
        &array1_vec(parameter_starts),
        &array1_vec(parameter_ends),
        &array1_vec(word_offsets),
        &array1_vec(widths),
        seed,
    )
    .map_err(err)?;
    let word_rows = PyList::empty(py);
    let mask_rows = PyList::empty(py);
    for (words, masks) in rows {
        word_rows.append(words.into_pyarray(py))?;
        mask_rows.append(masks.into_pyarray(py))?;
    }
    Ok((word_rows, mask_rows))
}

struct SparseWeightsInput<'a> {
    base: PyReadonlyArray1<'a, u32>,
    indices: PyReadonlyArray1<'a, u32>,
    observation_words: Vec<PyReadonlyArray1<'a, u32>>,
    observation_masks: Vec<PyReadonlyArray1<'a, u32>>,
    outcomes: PyReadonlyArray1<'a, f32>,
    candidate_words: Vec<PyReadonlyArray1<'a, u32>>,
    candidate_masks: Vec<PyReadonlyArray1<'a, u32>>,
    blocks: Vec<(usize, usize, u8, f32, f32, f32)>,
    acquisition: String,
    seed: u64,
    neighbors: usize,
    epistemic_scale: f32,
    aleatoric_scale: f32,
    y_scale: f32,
    beta: f32,
    device: String,
}

fn sparse_impl(input: SparseWeightsInput<'_>) -> PyResult<(usize, f32)> {
    if input.observation_words.len() != input.observation_masks.len()
        || input.candidate_words.len() != input.candidate_masks.len()
    {
        return Err(PyValueError::new_err(
            "sparse word and mask row counts must match",
        ));
    }
    let base = array1_vec(input.base);
    let indices = array1_vec(input.indices);
    let observation_words: Vec<Vec<u32>> = input
        .observation_words
        .into_iter()
        .map(array1_vec)
        .collect();
    let observation_masks: Vec<Vec<u32>> = input
        .observation_masks
        .into_iter()
        .map(array1_vec)
        .collect();
    let candidate_words: Vec<Vec<u32>> =
        input.candidate_words.into_iter().map(array1_vec).collect();
    let candidate_masks: Vec<Vec<u32>> =
        input.candidate_masks.into_iter().map(array1_vec).collect();
    let observation_bytes = pack_rows(&base, &indices, &observation_words, &observation_masks)?;
    let candidate_bytes = pack_rows(&base, &indices, &candidate_words, &candidate_masks)?;
    let outcome_vec = array1_vec(input.outcomes);
    let result = experimental::select_weights(
        &observation_bytes,
        observation_words.len(),
        &outcome_vec,
        &candidate_bytes,
        candidate_words.len(),
        &mixed_blocks(input.blocks)?,
        WeightSelectConfig {
            neighbors: input.neighbors,
            epistemic_scale: input.epistemic_scale,
            aleatoric_scale: input.aleatoric_scale,
            y_scale: input.y_scale,
            beta: input.beta,
            acquisition: AcquisitionKind::parse(&input.acquisition).map_err(err)?,
            seed: input.seed,
            device: ComputeDevice::parse(&input.device).map_err(err)?,
        },
    )
    .map_err(err)?;
    Ok((result.index, result.score))
}

#[pyfunction(name = "sparse_select")]
#[pyo3(signature=(*args, **kwargs))]
pub fn bind_select(
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<(usize, f32)> {
    let device = req_arg::<String>(args, kwargs, 15, "device").unwrap_or_else(|_| "auto".into());
    sparse_impl(SparseWeightsInput {
        base: req_arg(args, kwargs, 0, "base")?,
        indices: req_arg(args, kwargs, 1, "indices")?,
        observation_words: req_arg(args, kwargs, 2, "observation_words")?,
        observation_masks: req_arg(args, kwargs, 3, "observation_masks")?,
        outcomes: req_arg(args, kwargs, 4, "outcomes")?,
        candidate_words: req_arg(args, kwargs, 5, "candidate_words")?,
        candidate_masks: req_arg(args, kwargs, 6, "candidate_masks")?,
        blocks: req_arg(args, kwargs, 7, "blocks")?,
        acquisition: req_arg(args, kwargs, 8, "acquisition")?,
        seed: req_arg(args, kwargs, 9, "seed")?,
        neighbors: req_arg(args, kwargs, 10, "neighbors")?,
        epistemic_scale: req_arg(args, kwargs, 11, "epistemic_scale")?,
        aleatoric_scale: req_arg(args, kwargs, 12, "aleatoric_scale")?,
        y_scale: req_arg(args, kwargs, 13, "y_scale")?,
        beta: req_arg(args, kwargs, 14, "beta")?,
        device,
    })
}

#[pyfunction(name = "sparse_select_ucb")]
#[pyo3(signature=(*args, **kwargs))]
pub fn select_sparse(
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<(usize, f32)> {
    sparse_impl(SparseWeightsInput {
        base: req_arg(args, kwargs, 0, "base")?,
        indices: req_arg(args, kwargs, 1, "indices")?,
        observation_words: req_arg(args, kwargs, 2, "observation_words")?,
        observation_masks: req_arg(args, kwargs, 3, "observation_masks")?,
        outcomes: req_arg(args, kwargs, 4, "outcomes")?,
        candidate_words: req_arg(args, kwargs, 5, "candidate_words")?,
        candidate_masks: req_arg(args, kwargs, 6, "candidate_masks")?,
        blocks: req_arg(args, kwargs, 7, "blocks")?,
        acquisition: "ucb".into(),
        seed: 0,
        neighbors: req_arg(args, kwargs, 8, "neighbors")?,
        epistemic_scale: req_arg(args, kwargs, 9, "epistemic_scale")?,
        aleatoric_scale: req_arg(args, kwargs, 10, "aleatoric_scale")?,
        y_scale: req_arg(args, kwargs, 11, "y_scale")?,
        beta: req_arg(args, kwargs, 12, "beta")?,
        device: "auto".into(),
    })
}

fn pack_rows(
    base: &[u32],
    indices: &[u32],
    words: &[Vec<u32>],
    masks: &[Vec<u32>],
) -> PyResult<Vec<u8>> {
    if base.len() != indices.len() {
        return Err(PyValueError::new_err(
            "base and sparse index arrays must have the same length",
        ));
    }
    let mut bytes = Vec::with_capacity(words.len() * base.len() * std::mem::size_of::<u32>());
    for (row_words, row_masks) in words.iter().zip(masks) {
        if row_words.len() != row_masks.len() {
            return Err(PyValueError::new_err(
                "sparse row words and masks must have the same length",
            ));
        }
        let mut row = base.to_vec();
        for (&word, &mask) in row_words.iter().zip(row_masks) {
            let position = indices
                .binary_search(&word)
                .map_err(|_| PyValueError::new_err(format!("sparse word {word} is not indexed")))?;
            row[position] ^= mask;
        }
        for value in row {
            bytes.extend_from_slice(&value.to_ne_bytes());
        }
    }
    Ok(bytes)
}
