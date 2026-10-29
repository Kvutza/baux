//! Strongly-typed virtual register representations for PTX.

/// Physical target architecture.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TargetArch {
    Sm75,  // Turing (T4)
    Sm80,  // Ampere (A100)
    Sm90a, // Hopper (H100), architecture-specific feature set
}

impl TargetArch {
    /// Select the exact targets supported by the synthesized-kernel bank.
    pub fn from_compute(major: impl TryInto<u32>, minor: impl TryInto<u32>) -> Option<Self> {
        match (major.try_into().ok()?, minor.try_into().ok()?) {
            (7, 5) => Some(Self::Sm75),
            (8, 0) => Some(Self::Sm80),
            (9, 0) => Some(Self::Sm90a),
            _ => None,
        }
    }

    pub fn ptx_target(&self) -> &'static str {
        match self {
            Self::Sm75 => "sm_75",
            Self::Sm80 => "sm_80",
            Self::Sm90a => "sm_90a",
        }
    }

    pub fn ptx_version(&self) -> &'static str {
        match self {
            Self::Sm75 => "6.5",
            Self::Sm80 => "7.0",
            Self::Sm90a => "8.5",
        }
    }

    pub fn shared_bytes(&self) -> usize {
        match self {
            Self::Sm75 => 65_536,
            Self::Sm80 => 167_936,
            Self::Sm90a => 232_448,
        }
    }
}

/// Strongly-typed 32-bit integer register (`%r<idx>`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Reg32(pub u32);

/// Strongly-typed 64-bit integer/pointer register (`%rd<idx>`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Reg64(pub u32);

/// Strongly-typed 32-bit single-precision float register (`%f<idx>`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RegF32(pub u32);

/// Strongly-typed 1-bit predicate register (`%p<idx>`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RegPred(pub u32);
