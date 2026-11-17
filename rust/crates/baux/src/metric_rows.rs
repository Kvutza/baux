//! Ported from yubo-research/enn, commit
//! 506e98c506eeb849cffbf53d9ddf3a3a799c6830. Algorithm R reservoir for ordinary
//! disk ENN AUTO metric learning.

use std::sync::Arc;

use ndarray::ArrayView2;

use crate::error::ENNError;
use crate::hash::Key;
use crate::metric_rng::MetricRng;

#[derive(Clone)]
pub struct RowReservoir {
    num_dim: usize,
    num_outputs: usize,
    capacity: usize,
    seed: u64,
    rng: MetricRng,
    xs: Arc<Vec<f64>>,
    ys: Arc<Vec<f64>>,
    len: usize,
    num_seen: usize,
}

impl std::fmt::Debug for RowReservoir {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RowReservoir")
            .field("num_dim", &self.num_dim)
            .field("num_outputs", &self.num_outputs)
            .field("capacity", &self.capacity)
            .field("seed", &self.seed)
            .field("len", &self.len)
            .field("num_seen", &self.num_seen)
            .finish()
    }
}

const _: () = {
    fn assert_sync<T: Send + Sync>() {}
    fn check() {
        assert_sync::<RowReservoir>();
    }
};

impl RowReservoir {
    pub fn new(
        capacity: usize,
        num_dim: usize,
        num_outputs: usize,
        seed: u64,
    ) -> Result<Self, ENNError> {
        if capacity < 1 {
            return Err(ENNError::InvalidParameter(format!(
                "reservoir_capacity must be >= 1, got {capacity}"
            )));
        }
        let x_count = capacity
            .checked_mul(num_dim)
            .ok_or_else(|| ENNError::InvalidParameter("reservoir shape overflow".into()))?;
        let y_count = capacity
            .checked_mul(num_outputs)
            .ok_or_else(|| ENNError::InvalidParameter("reservoir shape overflow".into()))?;
        Ok(Self {
            num_dim,
            num_outputs,
            capacity,
            seed,
            rng: MetricRng::from_seed(seed),
            xs: Arc::new(vec![0.0; x_count]),
            ys: Arc::new(vec![0.0; y_count]),
            len: 0,
            num_seen: 0,
        })
    }

    pub fn from_key(
        capacity: usize,
        num_dim: usize,
        num_outputs: usize,
        key: Key,
    ) -> Result<Self, ENNError> {
        Self::new(capacity, num_dim, num_outputs, key.seed())
    }

    pub fn num_dim(&self) -> usize {
        self.num_dim
    }

    pub fn num_outputs(&self) -> usize {
        self.num_outputs
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    pub fn seed(&self) -> u64 {
        self.seed
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn num_seen(&self) -> usize {
        self.num_seen
    }

    pub fn x(&self) -> &[f64] {
        &self.xs[..self.len * self.num_dim]
    }

    pub fn y(&self) -> &[f64] {
        &self.ys[..self.len * self.num_outputs]
    }

    pub(crate) fn freeze_rows(&self, buf: &mut Vec<u8>) {
        use crate::snapshot_io::{
            write_bool, write_len, write_quad, write_slice, write_uint, write_word,
        };
        write_len(buf, self.capacity);
        write_word(buf, self.seed);
        let (state, inc, has_buf, buf_val) = self.rng.snapshot();
        write_quad(buf, state);
        write_quad(buf, inc);
        write_bool(buf, has_buf);
        write_uint(buf, buf_val);
        write_len(buf, self.len);
        write_len(buf, self.num_seen);
        write_slice(buf, self.x());
        write_slice(buf, self.y());
    }

    pub(crate) fn thaw_rows(
        num_dim: usize,
        num_outputs: usize,
        reader: &mut crate::snapshot_io::WireReader<'_>,
    ) -> Result<Self, ENNError> {
        use crate::snapshot_io::{MAX_OBS, bad_state};
        let capacity = reader.read_len(100_000)?;
        if capacity < 1 {
            return Err(bad_state("frozen reservoir capacity must be >= 1"));
        }
        let seed = reader.read_word()?;
        let state = reader.read_quad()?;
        let inc = reader.read_quad()?;
        let has_buf = reader.read_bool()?;
        let buf_val = reader.read_uint()?;
        let len = reader.read_len(capacity)?;
        let num_seen = reader.read_len(MAX_OBS)?;
        if num_seen < len {
            return Err(bad_state("frozen reservoir num_seen < len"));
        }
        let x_cap = capacity
            .checked_mul(num_dim)
            .ok_or_else(|| bad_state("reservoir x capacity overflow"))?;
        let y_cap = capacity
            .checked_mul(num_outputs)
            .ok_or_else(|| bad_state("reservoir y capacity overflow"))?;
        let x_used = len
            .checked_mul(num_dim)
            .ok_or_else(|| bad_state("reservoir x length overflow"))?;
        let y_used = len
            .checked_mul(num_outputs)
            .ok_or_else(|| bad_state("reservoir y length overflow"))?;
        let x_vals = reader.read_vec(x_used, true)?;
        let y_vals = reader.read_vec(y_used, true)?;
        if x_vals.len() != x_used || y_vals.len() != y_used {
            return Err(bad_state("frozen reservoir slice length mismatch"));
        }
        let mut xs = Vec::new();
        xs.try_reserve_exact(x_cap)
            .map_err(|_| bad_state("reservoir x allocation failed"))?;
        xs.extend_from_slice(&x_vals);
        xs.resize(x_cap, 0.0);
        let mut ys = Vec::new();
        ys.try_reserve_exact(y_cap)
            .map_err(|_| bad_state("reservoir y allocation failed"))?;
        ys.extend_from_slice(&y_vals);
        ys.resize(y_cap, 0.0);
        Ok(Self {
            num_dim,
            num_outputs,
            capacity,
            seed,
            rng: MetricRng::restore(state, inc, has_buf, buf_val),
            xs: Arc::new(xs),
            ys: Arc::new(ys),
            len,
            num_seen,
        })
    }

    pub fn configure(&mut self, seed: u64, capacity: usize) -> Result<(), ENNError> {
        let x_count = capacity
            .checked_mul(self.num_dim)
            .ok_or_else(|| ENNError::InvalidParameter("reservoir shape overflow".into()))?;
        let y_count = capacity
            .checked_mul(self.num_outputs)
            .ok_or_else(|| ENNError::InvalidParameter("reservoir shape overflow".into()))?;
        if capacity < 1 {
            return Err(ENNError::InvalidParameter(format!(
                "reservoir_capacity must be >= 1, got {capacity}"
            )));
        }
        if capacity < self.len {
            return Err(ENNError::InvalidParameter(format!(
                "reservoir_capacity {capacity} is below the {} rows already stored",
                self.len
            )));
        }
        if seed != self.seed && self.num_seen > self.capacity {
            return Err(ENNError::InvalidParameter(
                "seed is fixed once the reservoir starts replacing rows".into(),
            ));
        }
        if seed != self.seed {
            self.seed = seed;
            self.rng = MetricRng::from_seed(seed);
        }
        if capacity != self.capacity {
            let mut xs = vec![0.0; x_count];
            let mut ys = vec![0.0; y_count];
            let n = self.len;
            xs[..n * self.num_dim].copy_from_slice(&self.xs[..n * self.num_dim]);
            ys[..n * self.num_outputs].copy_from_slice(&self.ys[..n * self.num_outputs]);
            self.xs = Arc::new(xs);
            self.ys = Arc::new(ys);
            self.capacity = capacity;
        }
        Ok(())
    }

    /// Append every row. `x` is `(n, num_dim)` and `y` is `(n, num_outputs)`.
    pub fn push_rows(&mut self, x: &ArrayView2<f64>, y: &ArrayView2<f64>) -> Result<(), ENNError> {
        if x.nrows() != y.nrows() {
            return Err(ENNError::InvalidParameter(format!(
                "x has {} rows but y has {}",
                x.nrows(),
                y.nrows()
            )));
        }
        if x.ncols() != self.num_dim {
            return Err(ENNError::InvalidShape {
                expected: vec![x.nrows(), self.num_dim],
                got: vec![x.nrows(), x.ncols()],
            });
        }
        if y.ncols() != self.num_outputs {
            return Err(ENNError::InvalidParameter(format!(
                "y has {} columns, expected {}",
                y.ncols(),
                self.num_outputs
            )));
        }
        let mut x_scratch = Vec::new();
        let mut y_scratch = Vec::new();
        for i in 0..x.nrows() {
            let x_row = x.row(i);
            let y_row = y.row(i);
            let xr = if let Some(slice) = x_row.as_slice() {
                slice
            } else {
                x_scratch.clear();
                x_scratch.extend(x_row.iter().copied());
                &x_scratch
            };
            let yr = if let Some(slice) = y_row.as_slice() {
                slice
            } else {
                y_scratch.clear();
                y_scratch.extend(y_row.iter().copied());
                &y_scratch
            };
            self.push_row(xr, yr);
        }
        Ok(())
    }

    /// Algorithm R: fill the next slot until `capacity`, otherwise replace a
    /// uniform index in `0..num_seen` when that index still lies in the buffer.
    pub fn push_row(&mut self, x: &[f64], y: &[f64]) {
        if self.len < self.capacity {
            let i = self.len;
            Arc::make_mut(&mut self.xs)[i * self.num_dim..(i + 1) * self.num_dim]
                .copy_from_slice(x);
            Arc::make_mut(&mut self.ys)[i * self.num_outputs..(i + 1) * self.num_outputs]
                .copy_from_slice(y);
            self.len += 1;
        } else {
            let (next_rng, slot_u64) = self.rng.step_high(self.num_seen as u64 + 1);
            self.rng = next_rng;
            let slot = slot_u64 as usize;
            if slot < self.capacity {
                Arc::make_mut(&mut self.xs)[slot * self.num_dim..(slot + 1) * self.num_dim]
                    .copy_from_slice(x);
                Arc::make_mut(&mut self.ys)[slot * self.num_outputs..(slot + 1) * self.num_outputs]
                    .copy_from_slice(y);
            }
        }
        self.num_seen += 1;
    }

    /// Pure consumed-self state transition to configure seed and capacity.
    pub fn with_config(mut self, seed: u64, capacity: usize) -> Result<Self, ENNError> {
        self.configure(seed, capacity)?;
        Ok(self)
    }

    /// Pure consumed-self state transition to append matrix rows.
    pub fn pushed_rows(
        mut self,
        x: &ArrayView2<f64>,
        y: &ArrayView2<f64>,
    ) -> Result<Self, ENNError> {
        self.push_rows(x, y)?;
        Ok(self)
    }

    /// Pure consumed-self state transition to append a single row.
    pub fn pushed_row(mut self, x: &[f64], y: &[f64]) -> Self {
        self.push_row(x, y);
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn reservoir_cow() {
        let reservoir = RowReservoir::new(4, 2, 1, 42).unwrap();
        let x = [1.0, 2.0];
        let y = [3.0];
        let res1 = reservoir.pushed_row(&x, &y);
        assert_eq!(res1.len(), 1);
        assert_eq!(res1.num_seen(), 1);

        // Verify clone produces independent state
        let res2 = res1.clone();
        let res1 = res1.pushed_row(&[2.0, 3.0], &[5.0]);
        assert_eq!(res1.len(), 2);
        assert_eq!(res2.len(), 1);

        let x_mat = Array2::from_shape_vec((2, 2), vec![10.0, 20.0, 30.0, 40.0]).unwrap();
        let y_mat = Array2::from_shape_vec((2, 1), vec![50.0, 60.0]).unwrap();
        let res2 = res2.pushed_rows(&x_mat.view(), &y_mat.view()).unwrap();
        assert_eq!(res2.len(), 3);
        assert_eq!(res2.num_seen(), 3);

        let res2 = res2.with_config(42, 8).unwrap();
        assert_eq!(res2.capacity(), 8);
    }
}
