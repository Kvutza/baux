use deser::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[deser(rename_all = "kebab-case")]
pub enum TrainingOptimizer {
    #[default]
    ResidentEnnV1,
    OperatorBo,
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
pub enum DistanceScaling {
    #[default]
    Global,
    #[deser(rename = "self-tuning")]
    SelfTuning,
    #[deser(rename = "hyperbolic")]
    Hyperbolic,
}

impl DistanceScaling {
    pub const fn name(self) -> &'static str {
        match self {
            Self::Global => "global",
            Self::SelfTuning => "self-tuning",
            Self::Hyperbolic => "hyperbolic",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[deser(rename_all = "kebab-case")]
pub enum HistoryGeometry {
    Realized,
    #[default]
    Latent,
}

impl HistoryGeometry {
    pub const fn name(self) -> &'static str {
        match self {
            Self::Realized => "realized",
            Self::Latent => "latent",
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
