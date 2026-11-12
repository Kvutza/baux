//! Explicitly load TVM's pinned device plugin. It has no referenced C symbols,
//! so the linker is allowed to remove it from ordinary dylib dependencies.
use crate::portable::Result;
use crate::portable::dl::{Library, open_plugin};
use std::sync::OnceLock;

pub(super) fn metal() -> Result<()> {
    super::ensure_runtime()?;
    static PLUGIN: OnceLock<Result<Library>> = OnceLock::new();
    PLUGIN
        .get_or_init(|| {
            open_plugin(
                "BAUX_TVM_METAL_LIB",
                &["libtvm_runtime_metal.dylib"],
                &["tvm/lib", "baux/.dylibs"],
                "Apache TVM Metal runtime",
            )
        })
        .as_ref()
        .map(|_| ())
        .map_err(Clone::clone)
}
