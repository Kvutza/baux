//! Isolated resident Metal feedback tests, independent of other model
//! implementations.
#![cfg(all(target_os = "macos", feature = "metal"))]
#![allow(dead_code, unused_imports)]

#[path = "../src/apple_gpu.rs"]
mod apple_gpu;
#[path = "../src/fbt.rs"]
mod fbt;
#[path = "../src/fbt_attention.rs"]
mod fbt_attention;
#[path = "../src/fbt_metal.rs"]
mod fbt_metal;
#[path = "../src/fbt_model.rs"]
mod fbt_model;
#[path = "../src/fbt_mps.rs"]
mod fbt_mps;

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
