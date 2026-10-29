//! Backend capability declarations.
//!
//! This module is the executable map for accelerator parity work. It should be
//! updated when an operation becomes resident, gains an explicit fallback, or
//! is removed from a backend.
//!
//! General ENN dispatch lives in `posterior.rs` and `posterior/light.rs`.
//! CUDA reuses `knn/cuda_index.rs` and the posterior kernels in the CUDA
//! backend: moments, parameter batches, and shared-seed joint draws. The
//! current interface narrows host f64 inputs to f32 and downloads its results.
//! Kernel neighbor limits still apply; empty models and output transformations
//! have host handling. Metal/OpenCL accelerate neighbor search but use the CPU
//! general posterior and draw implementation. Their fused trial selectors are
//! distinct operations and do not establish device-array ENN posterior support.
//! The separate `device_posterior` interface now provides native CUDA/Metal
//! moments and draws with DLPack query/output storage; the host interface above
//! retains its original dispatch and transfer behavior.

use deser::{Deserialize, Serialize};

/// Execution backend tracked by the capability matrix.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[deser(rename_all = "snake_case")]
pub enum Backend {
    Cpu,
    Cuda,
    Metal,
    OpenCl,
}

impl Backend {
    pub const fn name(self) -> &'static str {
        match self {
            Self::Cpu => "cpu",
            Self::Cuda => "cuda",
            Self::Metal => "metal",
            Self::OpenCl => "opencl",
        }
    }

    /// Whether this Rust library contains the backend, independent of Python.
    pub const fn compiled(self) -> bool {
        match self {
            Self::Cpu => true,
            Self::Cuda => cfg!(all(
                target_os = "linux",
                target_arch = "x86_64",
                feature = "cuda"
            )),
            Self::Metal => cfg!(all(target_os = "macos", feature = "metal")),
            Self::OpenCl => cfg!(feature = "opencl"),
        }
    }

    /// Exercise the native neighbor implementation. This can initialize a GPU
    /// runtime and compile kernels, so capability inspection does not call it
    /// unless the caller explicitly requests a runtime probe.
    pub fn probe(self) -> Result<(), String> {
        use crate::index::IndexDriver;
        use crate::knn::KnnIndex;
        if !self.compiled() {
            return Err(format!(
                "{} backend was not compiled into this library",
                self.name()
            ));
        }
        let driver = match self {
            Self::Cpu => IndexDriver::Exact,
            Self::Cuda => IndexDriver::Cuda,
            Self::Metal => IndexDriver::Metal,
            Self::OpenCl => IndexDriver::OpenCl,
        };
        let rows = ndarray::array![[0.0]];
        let index = KnnIndex::new(&rows.view(), driver).map_err(|error| error.to_string())?;
        let (distance, indices) = index
            .search(&rows.view(), 1)
            .map_err(|error| error.to_string())?;
        if distance[[0, 0]] != 0.0 || indices[[0, 0]] != 0 {
            return Err(format!(
                "{} neighbor probe returned an incorrect result",
                self.name()
            ));
        }
        Ok(())
    }
}

/// Library operation tracked for backend parity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[deser(rename_all = "snake_case")]
pub enum Operation {
    ExactDist,
    /// Nearest-neighbor lookup; GPU implementations here perform exact scans.
    NeighborSearch,
    /// General ENN predictive moments, through its current host-array
    /// interface.
    Posterior,
    /// Predictive moments for multiple ENN parameter settings.
    PosteriorBatch,
    /// General ENN function draws with shared randomness across queries.
    PosteriorDraw,
    /// General ENN moments/draws with device-array input and output.
    DevicePosterior,
    Acquisition,
    Candidate,
    WeightSelect,
    TrialSearch,
    ResidentLoop,
}

impl Operation {
    pub const fn name(self) -> &'static str {
        match self {
            Self::ExactDist => "exact_dist",
            Self::NeighborSearch => "neighbor_search",
            Self::Posterior => "posterior",
            Self::PosteriorBatch => "posterior_batch",
            Self::PosteriorDraw => "posterior_draw",
            Self::DevicePosterior => "device_posterior",
            Self::Acquisition => "acquisition",
            Self::Candidate => "candidate",
            Self::WeightSelect => "weight_select",
            Self::TrialSearch => "trial_search",
            Self::ResidentLoop => "resident_loop",
        }
    }

    /// The general ENN interface accepts/returns host arrays even when its
    /// arithmetic executes on CUDA. This is separate from resident trial
    /// search.
    pub const fn host_arrays(self) -> bool {
        matches!(
            self,
            Self::ExactDist
                | Self::NeighborSearch
                | Self::Posterior
                | Self::PosteriorBatch
                | Self::PosteriorDraw
        )
    }
}

/// Capability status for one backend/operation pair.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[deser(rename_all = "snake_case")]
pub enum Support {
    /// A native implementation exists for the selected backend within its
    /// validated constraints. Host preparation, transfers, and degenerate-case
    /// handling may still occur; this is not a trace or residency guarantee.
    Direct,
    /// Some or all operation arithmetic executes on CPU.
    Fallback,
    Missing,
}

impl Support {
    pub const fn name(self) -> &'static str {
        match self {
            Self::Direct => "direct",
            Self::Fallback => "fallback",
            Self::Missing => "missing",
        }
    }
}

/// One row in the backend capability matrix.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Capability {
    pub backend: Backend,
    pub operation: Operation,
    pub support: Support,
}

const BACKENDS: [Backend; 4] = [Backend::Cpu, Backend::Cuda, Backend::Metal, Backend::OpenCl];

const OPS: [Operation; 11] = [
    Operation::ExactDist,
    Operation::NeighborSearch,
    Operation::Posterior,
    Operation::PosteriorBatch,
    Operation::PosteriorDraw,
    Operation::DevicePosterior,
    Operation::Acquisition,
    Operation::Candidate,
    Operation::WeightSelect,
    Operation::TrialSearch,
    Operation::ResidentLoop,
];

/// Returns every backend tracked by the matrix.
pub fn backends() -> &'static [Backend] {
    &BACKENDS
}

/// Returns every operation tracked by the matrix.
pub fn operations() -> &'static [Operation] {
    &OPS
}

/// Returns the declared support for a backend/operation pair.
pub fn support(backend: Backend, operation: Operation) -> Support {
    if !backend.compiled() {
        return Support::Missing;
    }
    match (backend, operation) {
        (Backend::Cuda | Backend::Metal, Operation::DevicePosterior) => Support::Direct,
        (_, Operation::DevicePosterior) => Support::Missing,
        (_, Operation::ResidentLoop) => Support::Fallback,
        (Backend::Cpu, _) => Support::Direct,
        (_, Operation::ExactDist) => Support::Direct,
        (
            Backend::Cuda,
            Operation::NeighborSearch
            | Operation::TrialSearch
            | Operation::Posterior
            | Operation::PosteriorBatch
            | Operation::PosteriorDraw,
        ) => Support::Direct,
        (
            Backend::Metal,
            Operation::NeighborSearch | Operation::TrialSearch | Operation::WeightSelect,
        ) => Support::Direct,
        (
            Backend::OpenCl,
            Operation::NeighborSearch | Operation::TrialSearch | Operation::WeightSelect,
        ) => Support::Direct,
        (
            _,
            Operation::Posterior
            | Operation::PosteriorBatch
            | Operation::PosteriorDraw
            | Operation::Acquisition
            | Operation::Candidate,
        ) => Support::Fallback,
        (_, Operation::WeightSelect) => Support::Missing,
    }
}

/// Returns the complete capability matrix.
pub fn matrix() -> Vec<Capability> {
    let mut rows = Vec::with_capacity(BACKENDS.len() * OPS.len());
    for &backend in &BACKENDS {
        for &operation in &OPS {
            rows.push(Capability {
                backend,
                operation,
                support: support(backend, operation),
            });
        }
    }
    rows
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compiled_support() {
        Backend::Cpu.probe().unwrap();
        for &backend in backends() {
            assert_eq!(
                support(backend, Operation::DevicePosterior),
                if backend.compiled() && matches!(backend, Backend::Cuda | Backend::Metal) {
                    Support::Direct
                } else {
                    Support::Missing
                }
            );
            if !backend.compiled() {
                assert!(backend.probe().is_err());
                assert!(
                    operations()
                        .iter()
                        .all(|&op| support(backend, op) == Support::Missing)
                );
            }
        }
    }

    #[test]
    fn posterior_paths() {
        assert_eq!(
            support(Backend::Cpu, Operation::PosteriorDraw),
            Support::Direct
        );
        if Backend::Cuda.compiled() {
            assert_eq!(
                support(Backend::Cuda, Operation::PosteriorDraw),
                Support::Direct
            );
            assert!(Operation::PosteriorDraw.host_arrays());
        }
        if Backend::Metal.compiled() {
            assert_eq!(
                support(Backend::Metal, Operation::PosteriorDraw),
                Support::Fallback
            );
        }
    }
}
