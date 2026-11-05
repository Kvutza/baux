//! Semantic execution IR and performance contract helpers.

mod contract;
pub mod diffusion;

pub use baux::semantics::mhc4;
pub use baux::semantics::model::{ModelProgram, ProgramKind};
pub use baux::semantics::recurrence::RecurrentCore;
pub use baux::semantics::schedule::{Backend, BackendSchedule, Compiler, ScheduledKernel};
pub use baux::semantics::shape::{MHC4_COEFFS, MHC4_PARAMS, MHC4_PATCH, PISA_BYTES};
pub use contract::*;
