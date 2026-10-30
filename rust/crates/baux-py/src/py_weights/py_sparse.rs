#![allow(clippy::too_many_arguments)]

use baux::experimental::{self, BpannHistory};
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use pyo3::types::PyList;

use super::{array1_vec, err};

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
