//! Mathematical and execution semantics shared by every BAUX adapter.
//!
//! No backend owns these types. Schedules describe implementations, not proofs
//! of numerical equivalence. CUDA enables only scalar ABI trait
//! implementations; the default build has no dependencies and does not require
//! a GPU toolchain.

#[path = "abi.rs"]
mod abi;

#[cfg(feature = "cuda-host")]
pub use baux_cuda_device::semantics::{
    context, float, gaussian, mhc4, muon, neighbor, proposal, random, recurrence, shape,
};

#[cfg(not(feature = "cuda-host"))]
#[path = "context.rs"]
pub mod context;
#[cfg(not(feature = "cuda-host"))]
#[path = "float.rs"]
pub mod float;
#[cfg(not(feature = "cuda-host"))]
#[path = "gaussian.rs"]
pub mod gaussian;
#[cfg(not(feature = "cuda-host"))]
#[path = "mhc4.rs"]
pub mod mhc4;
#[cfg(not(feature = "cuda-host"))]
#[path = "muon.rs"]
pub mod muon;
#[cfg(not(feature = "cuda-host"))]
#[path = "neighbor.rs"]
pub mod neighbor;
#[cfg(not(feature = "cuda-host"))]
#[path = "proposal.rs"]
pub mod proposal;
#[cfg(not(feature = "cuda-host"))]
#[path = "random.rs"]
pub mod random;
#[cfg(not(feature = "cuda-host"))]
#[path = "recurrence.rs"]
pub mod recurrence;
#[cfg(not(feature = "cuda-host"))]
#[path = "shape.rs"]
pub mod shape;

#[path = "optimizer/mod.rs"]
pub mod optimizer;
pub use optimizer::*;

#[cfg(not(feature = "cuda-device"))]
#[path = "calibration.rs"]
pub mod calibration;
#[cfg(not(feature = "cuda-device"))]
#[path = "categorical.rs"]
pub mod categorical;
#[cfg(not(feature = "cuda-device"))]
#[path = "dispatch.rs"]
pub mod dispatch;
#[cfg(not(feature = "cuda-device"))]
#[path = "graph.rs"]
pub mod graph;
#[cfg(not(feature = "cuda-device"))]
#[path = "hazard.rs"]
pub mod hazard;
#[cfg(not(feature = "cuda-device"))]
#[path = "hnet.rs"]
pub mod hnet;
#[cfg(not(feature = "cuda-device"))]
#[path = "key.rs"]
pub mod key;
#[cfg(not(feature = "cuda-device"))]
#[path = "memory.rs"]
pub mod memory;
#[cfg(not(feature = "cuda-device"))]
#[path = "model.rs"]
pub mod model;
#[cfg(not(feature = "cuda-device"))]
#[path = "numerical.rs"]
pub mod numerical;
#[cfg(not(feature = "cuda-device"))]
#[path = "schedule.rs"]
pub mod schedule;
#[cfg(not(feature = "cuda-device"))]
#[path = "shrinkage.rs"]
pub mod shrinkage;
#[cfg(not(feature = "cuda-device"))]
#[path = "tensor.rs"]
pub mod tensor;
