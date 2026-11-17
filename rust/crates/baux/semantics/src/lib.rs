//! Mathematical and execution semantics shared by every baux adapter.
//!
//! No backend owns these types. Schedules describe implementations, not proofs
//! of numerical equivalence. Core proposal (`pool`), surrogate (`neighbor`,
//! `optimizer::surrogate`), trust-region (`optimizer::trust_region`), MORBO
//! (`morbo`), warp (`ybounds`), manifold (`affine`, `categorical`, `stiefel`),
//! scalar (`real::Real`), and GPU layout (`abi`) primitives compile cleanly on
//! `cuda-device`, Metal, CPU, and Charon/Aeneas slices.

#[cfg(feature = "cuda-host")]
pub use baux_cuda_device::semantics::{
    abi, affine, categorical, context, float, gaussian, key, mhc4, morbo, muon, neighbor,
    numerical, pool, proposal, random, real, recurrence, shape, stiefel, ybounds,
};

#[cfg(not(feature = "cuda-host"))]
#[path = "abi.rs"]
pub mod abi;
#[cfg(not(feature = "cuda-host"))]
#[path = "affine.rs"]
pub mod affine;
#[cfg(not(feature = "cuda-host"))]
#[path = "categorical.rs"]
pub mod categorical;
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
#[path = "key.rs"]
pub mod key;
#[cfg(not(feature = "cuda-host"))]
#[path = "mhc4.rs"]
pub mod mhc4;
#[cfg(not(feature = "cuda-host"))]
#[path = "morbo.rs"]
pub mod morbo;
#[cfg(not(feature = "cuda-host"))]
#[path = "muon.rs"]
pub mod muon;
#[cfg(not(feature = "cuda-host"))]
#[path = "neighbor.rs"]
pub mod neighbor;
#[cfg(not(feature = "cuda-host"))]
#[path = "numerical.rs"]
pub mod numerical;
#[cfg(not(feature = "cuda-host"))]
#[path = "pool.rs"]
pub mod pool;
#[cfg(not(feature = "cuda-host"))]
#[path = "proposal.rs"]
pub mod proposal;
#[cfg(not(feature = "cuda-host"))]
#[path = "random.rs"]
pub mod random;
#[cfg(not(feature = "cuda-host"))]
#[path = "real.rs"]
pub mod real;
#[cfg(not(feature = "cuda-host"))]
#[path = "recurrence.rs"]
pub mod recurrence;
#[cfg(not(feature = "cuda-host"))]
#[path = "shape.rs"]
pub mod shape;
#[cfg(not(feature = "cuda-host"))]
#[path = "stiefel.rs"]
pub mod stiefel;
#[cfg(not(feature = "cuda-host"))]
#[path = "ybounds.rs"]
pub mod ybounds;

pub use real::Real;

#[path = "optimizer/mod.rs"]
pub mod optimizer;
pub use optimizer::*;

#[cfg(not(feature = "cuda-device"))]
pub mod compute;
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
#[path = "memory.rs"]
pub mod memory;
#[cfg(not(feature = "cuda-device"))]
#[path = "model.rs"]
pub mod model;
#[cfg(not(feature = "cuda-device"))]
#[path = "schedule.rs"]
pub mod schedule;
#[cfg(not(feature = "cuda-device"))]
#[path = "tensor.rs"]
pub mod tensor;
