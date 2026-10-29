//! Shared rolling, prequential prediction-error calibration. This is empirical
//! coverage on adaptively collected observations, not a conformal guarantee.

const CAPACITY: usize = 128;

#[derive(Clone, Debug)]
pub struct Calibration {
    errors: [f32; CAPACITY],
    count: usize,
    next: usize,
    covered: u64,
    scored: u64,
}

impl Default for Calibration {
    fn default() -> Self {
        Self {
            errors: [0.0; CAPACITY],
            count: 0,
            next: 0,
            covered: 0,
            scored: 0,
        }
    }
}

impl Calibration {
    /// Empirical 95% residual multiplier with the finite-sample rank
    /// correction. Until that rank exists, retain the existing two-sigma
    /// warm-start scale.
    pub fn multiplier(&self) -> f32 {
        let rank = ((self.count + 1) * 95).div_ceil(100);
        if rank == 0 || rank > self.count {
            return 2.0;
        }
        let mut ordered = self.errors;
        ordered[..self.count].sort_unstable_by(f32::total_cmp);
        ordered[rank - 1].max(2.0)
    }

    pub fn observe(&mut self, value: f32, prediction: f32, scale: f32) -> Result<(), String> {
        if !value.is_finite() || !prediction.is_finite() || !scale.is_finite() || scale < 0.0 {
            return Err(
                "Prediction calibration requires finite values and a nonnegative scale".into(),
            );
        }
        let residual =
            (f64::from(value) - f64::from(prediction)).abs() / f64::from(scale.max(1.0e-6));
        if !residual.is_finite() || residual > f64::from(f32::MAX) {
            return Err("Prediction calibration residual overflow".into());
        }
        // Score with past residuals before inserting the newly observed one.
        self.covered += u64::from(residual <= f64::from(self.multiplier()));
        self.scored += 1;
        self.errors[self.next] = residual as f32;
        self.next = (self.next + 1) % CAPACITY;
        self.count = (self.count + 1).min(CAPACITY);
        Ok(())
    }

    pub fn count(&self) -> usize {
        self.count
    }
    pub fn coverage(&self) -> Option<f64> {
        (self.scored != 0).then(|| self.covered as f64 / self.scored as f64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn past_only() {
        let mut calibration = Calibration::default();
        for _ in 0..19 {
            calibration.observe(1.0, 0.0, 1.0).unwrap();
        }
        assert_eq!(calibration.multiplier(), 2.0);
        calibration.observe(9.0, 0.0, 1.0).unwrap();
        assert_eq!(calibration.coverage(), Some(19.0 / 20.0));
        assert_eq!(calibration.multiplier(), 9.0);
        for _ in 0..128 {
            calibration.observe(1.0, 0.0, 1.0).unwrap();
        }
        assert_eq!(calibration.count(), 128);
        assert_eq!(calibration.multiplier(), 2.0);
        assert!(calibration.observe(f32::NAN, 0.0, 1.0).is_err());
    }
}
