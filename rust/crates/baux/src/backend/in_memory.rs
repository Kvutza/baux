//! In-memory ENN backend (dev / small N).

use ndarray::{Array1, Array2, ArrayView2, Axis};

use crate::error::ENNError;
use crate::index::{ENNIndex, IndexDriver};
use crate::scratch::ImLog;

#[derive(Clone)]
pub struct InMemoryEnnBackend {
    train_x_rows: ImLog,
    train_y_rows: ImLog,
    train_yvar_rows: Option<ImLog>,
    num_dim: usize,
    num_metrics: usize,
    index: ENNIndex,
    index_stale: bool,
    index_synced_obs: usize,
}

impl InMemoryEnnBackend {
    pub fn new(
        train_x: Array2<f64>,
        train_y: Array2<f64>,
        train_yvar: Option<Array2<f64>>,
        scale_x: bool,
        x_scale: Array1<f64>,
        driver: IndexDriver,
    ) -> Result<Self, ENNError> {
        let num_obs = train_x.nrows();
        let num_dim = train_x.ncols();
        let num_metrics = train_y.ncols();

        let train_x_scaled = if scale_x {
            (&train_x / &x_scale.view().insert_axis(Axis(0))).to_owned()
        } else {
            train_x.clone()
        };

        let index = ENNIndex::new(train_x_scaled, num_dim, x_scale, scale_x, driver)
            .map_err(|e| ENNError::InvalidParameter(e.to_string()))?;
        let index_synced_obs = index.len().min(num_obs);

        Ok(Self {
            train_x_rows: ImLog::from_array(train_x),
            train_y_rows: ImLog::from_array(train_y),
            train_yvar_rows: train_yvar.map(ImLog::from_array),
            num_dim,
            num_metrics,
            index,
            index_stale: false,
            index_synced_obs,
        })
    }

    pub fn new_empty(
        num_dim: usize,
        num_metrics: usize,
        driver: IndexDriver,
    ) -> Result<Self, ENNError> {
        let empty_x = Array2::<f64>::zeros((0, num_dim));
        let empty_y = Array2::<f64>::zeros((0, num_metrics));
        Self::new(empty_x, empty_y, None, false, Array1::ones(num_dim), driver)
    }

    pub fn len(&self) -> usize {
        self.train_x_rows.nrows()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn num_dim(&self) -> usize {
        self.num_dim
    }

    pub fn num_metrics(&self) -> usize {
        self.num_metrics
    }

    pub fn driver(&self) -> IndexDriver {
        self.index.driver()
    }

    pub fn index_len(&self) -> usize {
        self.index.len()
    }

    pub fn x_log(&self) -> &ImLog {
        &self.train_x_rows
    }

    pub fn y_log(&self) -> &ImLog {
        &self.train_y_rows
    }

    pub fn yvar_log(&self) -> Option<&ImLog> {
        self.train_yvar_rows.as_ref()
    }

    pub fn slice_x(&self, i: usize) -> Result<&[f64], ENNError> {
        self.train_x_rows.checked_slice(i)
    }

    pub fn slice_y(&self, i: usize) -> Result<&[f64], ENNError> {
        self.train_y_rows.checked_slice(i)
    }

    pub fn slice_yvar(&self, i: usize) -> Result<Option<&[f64]>, ENNError> {
        match &self.train_yvar_rows {
            Some(log) => log.checked_slice(i).map(Some),
            None => {
                if i >= self.len() {
                    return Err(ENNError::InvalidParameter(format!(
                        "slice_yvar index {i} out of range [0, {})",
                        self.len()
                    )));
                }
                Ok(None)
            }
        }
    }

    pub fn x_view(&self) -> Option<ndarray::ArrayView2<'_, f64>> {
        self.train_x_rows.view()
    }

    pub fn y_view(&self) -> Option<ndarray::ArrayView2<'_, f64>> {
        self.train_y_rows.view()
    }

    pub fn yvar_view(&self) -> Option<ndarray::ArrayView2<'_, f64>> {
        self.train_yvar_rows.as_ref().and_then(ImLog::view)
    }

    pub fn memory_x(&self) -> Array2<f64> {
        self.train_x_rows.to_array()
    }

    pub fn memory_y(&self) -> Array2<f64> {
        self.train_y_rows.to_array()
    }

    pub fn memory_yvar(&self) -> Option<Array2<f64>> {
        self.train_yvar_rows.as_ref().map(|r| r.to_array())
    }

    pub fn index(&self) -> &ENNIndex {
        &self.index
    }

    pub fn append_rows(
        mut self,
        x: &ArrayView2<f64>,
        y: &ArrayView2<f64>,
        yvar: Option<&ArrayView2<f64>>,
    ) -> Result<Self, ENNError> {
        if x.nrows() == 0 {
            return Ok(self);
        }
        let prev_rows = self.train_x_rows.nrows();
        self.train_x_rows = self.train_x_rows.push_rows(x)?;
        self.train_y_rows = self.train_y_rows.push_rows(y)?;
        self.train_yvar_rows = match (self.train_yvar_rows, yvar) {
            (Some(yvar_rows), Some(yv)) => Some(yvar_rows.push_rows(yv)?),
            (None, Some(yv)) => {
                let storage = ImLog::from_array(Array2::zeros((prev_rows, self.num_metrics)));
                Some(storage.push_rows(yv)?)
            }
            (Some(yvar_rows), None) => {
                let zeros = Array2::zeros((x.nrows(), self.num_metrics));
                Some(yvar_rows.push_rows(&zeros.view())?)
            }
            (None, None) => None,
        };

        if !self.index_stale && !self.index.scales_x() && self.index_synced_obs == prev_rows {
            self.index = self
                .index
                .add(x)
                .map_err(|e| ENNError::InvalidParameter(e.to_string()))?;
            self.index_synced_obs = self.train_x_rows.nrows();
        }

        Ok(self)
    }

    pub fn mark_stale(mut self) -> Self {
        self.index_stale = true;
        self
    }

    pub fn index_stale(&self) -> bool {
        self.index_stale
    }

    pub fn ensure_sync(mut self, scale_x: bool, x_scale: &Array1<f64>) -> Result<Self, ENNError> {
        let num_obs = self.len();
        let scale_mismatch =
            self.index.scales_x() != scale_x || (scale_x && self.index.scale_vec() != *x_scale);

        if !self.index_stale
            && !scale_mismatch
            && self.index_synced_obs == self.index.len()
            && self.index.len() < num_obs
        {
            let pending_indices: Vec<usize> = (self.index_synced_obs..num_obs).collect();
            let pending = self.train_x_rows.gather_rows(&pending_indices);
            if pending.nrows() > 0 {
                self.index = self
                    .index
                    .add(&pending.view())
                    .map_err(|e| ENNError::InvalidParameter(e.to_string()))?;
            }
            self.index_synced_obs = num_obs;
            return Ok(self);
        }

        if self.index_stale
            || scale_mismatch
            || self.index.len() != num_obs
            || self.index_synced_obs != num_obs
        {
            let x_array = self.train_x_rows.to_array();
            let scaled_x = if scale_x {
                (&x_array / &x_scale.view().insert_axis(Axis(0))).to_owned()
            } else {
                x_array
            };
            self.index = self.index.with_scaling(scale_x);
            self.index = self.index.rebuild_scaled(scaled_x, x_scale.clone())?;
            self.index_stale = false;
            self.index_synced_obs = num_obs;
        }
        Ok(self)
    }

    pub fn train_rows(&self, indices: &[usize]) -> Result<super::TrainRows, ENNError> {
        let n = self.len();
        for &i in indices {
            if i >= n {
                return Err(ENNError::InvalidParameter(format!(
                    "train_rows index {i} out of range [0, {n})"
                )));
            }
        }
        let x = self.train_x_rows.gather_rows(indices);
        let y = self.train_y_rows.gather_rows(indices);
        let yvar = self
            .train_yvar_rows
            .as_ref()
            .map(|r| r.gather_rows(indices));
        Ok((x, y, yvar))
    }

    pub fn row_x(&self, i: usize) -> Result<Array1<f64>, ENNError> {
        if i >= self.len() {
            return Err(ENNError::InvalidParameter(format!(
                "row_x index {i} out of range [0, {})",
                self.len()
            )));
        }
        Ok(self.train_x_rows.row_vec(i))
    }

    pub fn row_y(&self, i: usize) -> Result<Array1<f64>, ENNError> {
        if i >= self.len() {
            return Err(ENNError::InvalidParameter(format!(
                "row_y index {i} out of range [0, {})",
                self.len()
            )));
        }
        Ok(self.train_y_rows.row_vec(i))
    }

    pub fn row_yvar(&self, i: usize) -> Result<Option<Array1<f64>>, ENNError> {
        if i >= self.len() {
            return Err(ENNError::InvalidParameter(format!(
                "row_yvar index {i} out of range [0, {})",
                self.len()
            )));
        }
        Ok(self.train_yvar_rows.as_ref().map(|r| r.row_vec(i)))
    }

    pub fn search(
        &self,
        x: &ArrayView2<f64>,
        search_k: i32,
        exclude_nearest: bool,
    ) -> Result<(Array2<f64>, Array2<i64>), ENNError> {
        self.index
            .search(x, search_k, exclude_nearest)
            .map_err(|e| ENNError::InvalidParameter(e.to_string()))
    }

    pub fn index_bytes(&self) -> Result<usize, ENNError> {
        Ok(self.index.memory_bytes())
    }
}
