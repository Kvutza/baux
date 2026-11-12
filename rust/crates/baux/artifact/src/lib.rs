//! BAUX's native Deser formats and dynamic values; no Serde compatibility
//! layer.

#[path = "attestation.rs"]
pub mod attestation;
#[path = "blake3.rs"]
pub mod blake3;
#[path = "cbor.rs"]
pub mod cbor;
#[path = "forward.rs"]
pub mod forward;
#[path = "gpu.rs"]
pub mod gpu;
#[path = "inc.rs"]
pub mod inc;
#[path = "json.rs"]
pub mod json;
#[path = "merkle.rs"]
pub mod merkle;
#[path = "numerical.rs"]
pub mod numerical;
#[path = "pretrain.rs"]
pub mod pretrain;
#[path = "runs.rs"]
pub mod runs;
#[path = "study.rs"]
pub mod study;
#[path = "terminal.rs"]
pub mod terminal;
#[path = "toml.rs"]
pub mod toml;
pub use crate::wire_json;
