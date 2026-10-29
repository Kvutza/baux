//! BAUX's native Deser formats and dynamic values; no Serde compatibility
//! layer.

#[path = "cbor.rs"]
pub mod cbor;
#[path = "forward.rs"]
pub mod forward;
#[path = "gpu.rs"]
pub mod gpu;
#[path = "json.rs"]
pub mod json;
#[path = "numerical.rs"]
pub mod numerical;
#[path = "pretrain.rs"]
pub mod pretrain;
#[path = "runs.rs"]
pub mod runs;
#[path = "terminal.rs"]
pub mod terminal;
#[path = "toml.rs"]
pub mod toml;
pub use crate::wire_json;
