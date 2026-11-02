//! Explicit, opt-in shader inputs for full-loop kernel experiments.

use deser::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

/// A trial changes shader implementations, not the model or optimizer workload.
/// An empty trial measures production with the same validation instrumentation.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[deser(deny_unknown_fields)]
pub struct KernelTrial {
    pub pisa: Option<PathBuf>,
    pub moe: Option<PathBuf>,
    pub decode: Option<PathBuf>,
    pub readout: Option<PathBuf>,
    pub perturb: Option<PathBuf>,
    pub mhc: Option<PathBuf>,
}

impl KernelTrial {
    pub fn resolve(&mut self, parent: &Path) -> Result<(), String> {
        for path in [
            &mut self.pisa,
            &mut self.moe,
            &mut self.decode,
            &mut self.readout,
            &mut self.perturb,
            &mut self.mhc,
        ]
        .into_iter()
        .flatten()
        {
            *path = parent
                .join(&*path)
                .canonicalize()
                .map_err(|error| format!("kernel source {}: {error}", path.display()))?;
        }
        Ok(())
    }
}
