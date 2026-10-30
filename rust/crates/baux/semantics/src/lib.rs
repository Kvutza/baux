//! Mathematical and execution semantics shared by every BAUX adapter.
//!
//! No execution target owns these types. Schedules describe implementations,
//! not proofs of numerical equivalence. CUDA enables only scalar ABI trait
//! implementations; the default build has no dependencies and does not require
//! a GPU toolchain.

#[path = "abi.rs"]
mod abi;
#[path = "context.rs"]
pub mod context;
#[path = "dispatch.rs"]
pub mod dispatch;
#[path = "float.rs"]
pub mod float;
#[path = "gaussian.rs"]
pub mod gaussian;
#[path = "graph.rs"]
pub mod graph;
#[path = "hazard.rs"]
pub mod hazard;
#[path = "hnet.rs"]
pub mod hnet;
#[path = "memory.rs"]
pub mod memory;
#[path = "mhc4.rs"]
pub mod mhc4;
#[path = "model.rs"]
pub mod model;
#[path = "muon.rs"]
pub mod muon;
#[path = "neighbor.rs"]
pub mod neighbor;
#[path = "numerical.rs"]
pub mod numerical;
#[path = "optimizer/mod.rs"]
pub mod optimizer;
pub use optimizer::*;
#[path = "proposal.rs"]
pub mod proposal;
#[path = "random.rs"]
pub mod random;
#[path = "recurrence.rs"]
pub mod recurrence;
#[path = "schedule.rs"]
pub mod schedule;
#[path = "shape.rs"]
pub mod shape;
#[path = "tensor.rs"]
pub mod tensor;
