//! Compile and exercise the standalone resident search without changing library
//! registrations.
#![cfg(all(target_os = "macos", feature = "metal"))]
#![allow(dead_code)]

#[path = "../src/apple_gpu.rs"]
mod apple_gpu;
#[path = "../src/objective_observation.rs"]
mod objective_observation;
#[path = "../src/resident_metal.rs"]
mod resident_metal;
use baux::procedural_pool;

pub use baux::{
    ProposalKernel, Rescalarize, calibration, config, fit, fitter, hash, hypervolume, morbo,
    params, reliability_region, response_observation, tensor_store, trust_region,
};
mod trials {
    pub use baux::experimental::RoundConfig;
}
mod compute {
    pub use baux::experimental::AcquisitionKind;
}
mod weights {
    pub use baux::experimental::AcquisitionKind;
}
