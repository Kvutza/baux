use super::*;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[deser(default, deny_unknown_fields, rename_all = "kebab-case")]
/// Dimensionless radius bounds and optional tensor scaling geometry.
/// A radius multiplies block proposal scales; it is not a full-vector norm.
pub struct TrustRegionBounds {
    pub initial: f64,
    pub min: f64,
    pub max: f64,
    pub shape: Option<TrustRegionShape>,
}

impl Default for TrustRegionBounds {
    fn default() -> Self {
        let length = TRLengthConfig::default();
        Self {
            initial: length.length_init,
            min: length.length_min,
            max: length.length_max,
            shape: None,
        }
    }
}

impl TrustRegionBounds {
    /// Check finite ordered bounds `0 < min <= initial <= max` without
    /// mutation. Valid bounds do not establish that a useful step fits
    /// inside the region.
    pub fn validate(&self) -> Result<(), String> {
        if !self.initial.is_finite()
            || !self.min.is_finite()
            || !self.max.is_finite()
            || self.min <= 0.0
            || self.min > self.initial
            || self.initial > self.max
        {
            return Err(
                "trust region bounds must be finite and satisfy 0 < min <= initial <= max".into(),
            );
        }
        Ok(())
    }
}
