use deser::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[deser(rename_all = "kebab-case")]
pub enum TrainingOptimizer {
    #[default]
    ResidentEnnV1,
    /// Acquisition-free single-candidate (`K = 1`) ENN Bayesian optimization.
    #[deser(alias = "operator-bo")]
    SingleStateBo,
    MuonV1,
    AdamwV1,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[deser(rename_all = "kebab-case")]
pub enum PretrainSelection {
    #[default]
    Enn,
    Random,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[deser(rename_all = "kebab-case")]
pub enum Dissimilarity {
    #[default]
    #[deser(rename = "euclidean")]
    Euclidean,
    #[deser(rename = "conformal")]
    Conformal,
    #[deser(rename = "hyperbolic")]
    Hyperbolic,
    #[deser(rename = "wasserstein", alias = "scale-transport")]
    Wasserstein,
}

/// Compatibility alias; non-Euclidean variants are symmetric dissimilarities
/// and do not satisfy the triangle inequality.
pub type DistanceMetric = Dissimilarity;

impl Dissimilarity {
    pub const fn name(self) -> &'static str {
        match self {
            Self::Euclidean => "euclidean",
            Self::Conformal => "conformal",
            Self::Hyperbolic => "hyperbolic",
            Self::Wasserstein => "wasserstein",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[deser(rename_all = "kebab-case")]
pub enum TrustRegionShape {
    Scalar,
    TensorFamilyStatic,
}

impl TrustRegionShape {
    pub const fn name(self) -> &'static str {
        match self {
            Self::Scalar => "scalar",
            Self::TensorFamilyStatic => "tensor-family-static",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[deser(rename_all = "kebab-case")]
pub enum ObjectiveReference {
    #[default]
    MovingIncumbent,
    FrozenInitial,
}

impl ObjectiveReference {
    pub const fn name(self) -> &'static str {
        match self {
            Self::MovingIncumbent => "moving-incumbent",
            Self::FrozenInitial => "frozen-initial",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shape_names() {
        assert_eq!(TrustRegionShape::Scalar.name(), "scalar");
        assert_eq!(
            TrustRegionShape::TensorFamilyStatic.name(),
            "tensor-family-static"
        );
    }
}
