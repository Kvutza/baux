//! Full-coordinate proposal kernels used by resident accelerator search.
//!
//! The interface is intentionally small: a law names the distribution generated
//! inside the Metal kernel. Controller, radius, tensor scaling, and candidate
//! pairing remain separate decisions.

use deser::{Deserialize, Serialize};

/// Kernel used to construct a full-coordinate proposal.
#[repr(u32)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[deser(rename_all = "kebab-case")]
pub enum ProposalKernel {
    /// Independent standard normal coordinates generated with a 256-strip
    /// Ziggurat.
    #[default]
    Gaussian = 0,
    /// Independent coordinates drawn uniformly from {-1, +1}.
    Rademacher = 1,
}

impl ProposalKernel {
    /// Stable value consumed by the Metal ABI.
    pub const fn shader(self) -> u32 {
        self as u32
    }

    /// Machine-readable distribution name used by run artifacts.
    pub const fn name(self) -> &'static str {
        match self {
            Self::Gaussian => "independent_gaussian_ziggurat256_v1",
            Self::Rademacher => "independent_rademacher",
        }
    }

    /// Coordinate moments before tensor scaling.
    pub const fn moments(self) -> (f32, f32) {
        (0.0, 1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn contract() {
        #[derive(Deserialize)]
        struct Setting {
            perturbation: ProposalKernel,
        }

        assert_eq!(ProposalKernel::default(), ProposalKernel::Gaussian);
        assert_eq!(ProposalKernel::Gaussian.shader(), 0);
        assert_eq!(
            ProposalKernel::Gaussian.name(),
            "independent_gaussian_ziggurat256_v1"
        );
        assert_eq!(ProposalKernel::Rademacher.shader(), 1);
        assert_eq!(ProposalKernel::Rademacher.moments(), (0.0, 1.0));
        let setting: Setting = baux::artifact::toml::from_str("perturbation='rademacher'").unwrap();
        assert_eq!(setting.perturbation, ProposalKernel::Rademacher);
    }
}
