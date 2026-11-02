use crate::weights::AcquisitionKind;
pub use crate::weights::EncodingType;

#[derive(Debug, Clone, Copy)]
pub struct ProposalParams {
    /// Trust-region radius used to materialize the parameter perturbation.
    pub radius: f32,
}

#[derive(Debug, Clone, Copy)]
pub struct SurrogateParams {
    /// Maximum number of retained observations used by the local posterior.
    pub neighbors: usize,
    pub epistemic_scale: f32,
    pub aleatoric_scale: f32,
    pub output_scale: f32,
}

#[derive(Debug, Clone, Copy)]
pub struct AcquisitionParams {
    pub kind: AcquisitionKind,
    pub beta: f32,
    /// Sampled-function seed; resident observation slots identify shared
    /// Thompson noise.
    pub seed: u64,
}

/// Fully resolved inputs for one proposal-selection operation.
///
/// The authored experiment schema remains independent: it compiles its
/// proposal, surrogate, and acquisition sections into this runtime value.
#[derive(Debug, Clone, Copy)]
pub struct RoundConfig {
    pub proposal: ProposalParams,
    pub surrogate: SurrogateParams,
    pub acquisition: AcquisitionParams,
}

impl Default for RoundConfig {
    fn default() -> Self {
        Self {
            proposal: ProposalParams { radius: 0.8 },
            surrogate: SurrogateParams {
                neighbors: 10,
                epistemic_scale: 0.7,
                aleatoric_scale: 0.05,
                output_scale: 1.0,
            },
            acquisition: AcquisitionParams {
                kind: AcquisitionKind::Ucb,
                beta: 1.0,
                seed: 0,
            },
        }
    }
}

/// Borrowed device buffer containing one trial's encoded parameters.
pub struct DeviceView<'a> {
    row_bytes: usize,
    #[allow(dead_code)]
    kind: ViewKind<'a>,
}

#[allow(dead_code)]
enum ViewKind<'a> {
    #[cfg(all(target_os = "linux", target_arch = "x86_64", feature = "cuda"))]
    Cuda { ptr: u64, stream: usize },
    #[cfg(all(target_os = "macos", feature = "metal"))]
    Metal {
        buffer: ::metal::Buffer,
        offset: usize,
    },
    #[doc(hidden)]
    Lifetime(std::marker::PhantomData<&'a ()>),
}

impl<'a> DeviceView<'a> {
    #[cfg(all(target_os = "macos", feature = "metal"))]
    pub fn metal(buffer: ::metal::Buffer, offset: usize, row_bytes: usize) -> Self {
        Self {
            row_bytes,
            kind: ViewKind::Metal { buffer, offset },
        }
    }

    pub fn row_bytes(&self) -> usize {
        self.row_bytes
    }

    #[cfg(all(target_os = "linux", target_arch = "x86_64", feature = "cuda"))]
    pub fn as_cuda(&self) -> Option<(u64, usize)> {
        match self.kind {
            ViewKind::Cuda { ptr, stream } => Some((ptr, stream)),
            _ => None,
        }
    }

    #[cfg(all(target_os = "macos", feature = "metal"))]
    pub fn as_metal(&self) -> Option<(&::metal::Buffer, usize)> {
        match &self.kind {
            ViewKind::Metal { buffer, offset } => Some((buffer, *offset)),
            _ => None,
        }
    }
}
