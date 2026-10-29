use super::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[deser(rename_all = "kebab-case")]
pub enum AcceptanceCriterion {
    #[deser(rename = "positive-improvement")]
    PositiveImprovement,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[deser(deny_unknown_fields, rename_all = "kebab-case")]
pub struct AcceptanceSpec {
    pub criterion: AcceptanceCriterion,
}

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

#[derive(Debug, Clone, Serialize, Deserialize)]
#[deser(
    tag = "policy",
    rename_all = "kebab-case",
    rename_all_fields = "kebab-case",
    deny_unknown_fields
)]
pub enum RadiusControlSpec {
    #[deser(rename = "turbo")]
    Turbo {
        #[deser(flatten)]
        bounds: TrustRegionBounds,
    },
}

impl RadiusControlSpec {
    /// Borrow radius bounds only.
    pub fn bounds(&self) -> &TrustRegionBounds {
        let Self::Turbo { bounds } = self;
        bounds
    }
}

/// Default count of MORBO regions, distinct from procedural seed arms.
fn default_regions() -> usize {
    4
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[deser(
    tag = "policy",
    rename_all = "kebab-case",
    rename_all_fields = "kebab-case",
    deny_unknown_fields
)]
pub enum TrustRegionSpec {
    #[deser(rename = "turbo")]
    Turbo {
        #[deser(flatten)]
        bounds: TrustRegionBounds,
    },
    #[deser(rename = "morbo")]
    Morbo {
        #[deser(flatten)]
        bounds: TrustRegionBounds,
        #[deser(default = default_regions())]
        regions: usize,
        rescalarize: Rescalarize,
        clip: bool,
    },
}

impl TrustRegionSpec {
    /// Borrow the same radius-bound role across controller alternatives.
    /// This accessor neither initializes a controller nor equates their
    /// behavior.
    pub fn bounds(&self) -> &TrustRegionBounds {
        match self {
            Self::Turbo { bounds } | Self::Morbo { bounds, .. } => bounds,
        }
    }
}
