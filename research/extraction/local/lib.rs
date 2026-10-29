//! Source-bound extraction crate root for the posterior-mean optimizer path.
//!
//! The extraction command copies the production source tree byte-for-byte and
//! installs this root as `src/lib.rs`. These modules are the exact production
//! implementation and interface used by `BayesOptimizer::posterior_mu`.

#![allow(clippy::pedantic, clippy::nursery, clippy::cargo)]

mod coordinate;
pub mod coordinate_flat;
mod coordinate_scalar;
pub mod error;
pub mod optimizer_dispatch;
pub mod optimizer_mu;
mod surrogate_api;
