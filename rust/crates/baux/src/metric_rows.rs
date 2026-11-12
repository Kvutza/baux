//! Ported from yubo-research/enn, commit
//! 506e98c506eeb849cffbf53d9ddf3a3a799c6830. Algorithm R reservoir for ordinary
//! disk ENN AUTO metric learning.

use ndarray::ArrayView2;

use crate::error::ENNError;
use crate::metric_rng::MetricRng;

pub struct RowReservoir {
    num_dim: usize,
    num_outputs: usize,
    capacity: usize,
    seed: u64,
    rng: MetricRng,
    xs: Vec<f64>,
    ys: Vec<f64>,
    len: usize,
    num_seen: usize,
}

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
            xs: vec![0.0; x_count],
            ys: vec![0.0; y_count],
            len: 0,
            num_seen: 0,
        })
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
            xs,
            ys,
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
            self.xs = xs;
            self.ys = ys;
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
        for i in 0..x.nrows() {
            let xr: Vec<f64> = x.row(i).iter().copied().collect();
            let yr: Vec<f64> = y.row(i).iter().copied().collect();
            self.push_row(&xr, &yr);
        }
        Ok(())
    }

    /// Algorithm R: fill the next slot until `capacity`, otherwise replace a
    /// uniform index in `0..num_seen` when that index still lies in the buffer.
    pub fn push_row(&mut self, x: &[f64], y: &[f64]) {
        if self.len < self.capacity {
            let i = self.len;
            self.xs[i * self.num_dim..(i + 1) * self.num_dim].copy_from_slice(x);
            self.ys[i * self.num_outputs..(i + 1) * self.num_outputs].copy_from_slice(y);
            self.len += 1;
        } else {
            let slot = self.rng.integers_high(self.num_seen as u64 + 1) as usize;
            if slot < self.capacity {
                self.xs[slot * self.num_dim..(slot + 1) * self.num_dim].copy_from_slice(x);
                self.ys[slot * self.num_outputs..(slot + 1) * self.num_outputs].copy_from_slice(y);
            }
        }
        self.num_seen += 1;
    }
}
