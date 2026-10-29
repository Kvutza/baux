use ndarray::{Array1, ArrayView1};

/// Null trust region (no trust region management).
///
/// Always returns full bounds [0, 1]^d and never needs restart.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NoTrustRegion {
    num_dim: usize,
}

impl NoTrustRegion {
    /// Create a new NoTrustRegion.
    pub fn new(num_dim: usize) -> Self {
        Self { num_dim }
    }

    /// Get current length (always 1.0).
    pub fn length(&self) -> f64 {
        1.0
    }

    /// Check if restart is needed (always false).
    pub fn needs_restart(&self) -> bool {
        false
    }

    /// Update (no-op).
    pub fn update(&mut self, _y_new: &ArrayView1<f64>, _num_obs: usize) {}

    /// Compute trust region bounds (always returns [0, 1]^d).
    pub fn compute_bounds(
        &self,
        _x_center: &ArrayView1<f64>,
        _lengthscales: Option<&ArrayView1<f64>>,
    ) -> (Array1<f64>, Array1<f64>) {
        let lb = Array1::zeros(self.num_dim);
        let ub = Array1::ones(self.num_dim);
        (lb, ub)
    }

    /// Restart (no-op).
    pub fn restart(&mut self) {}
}
