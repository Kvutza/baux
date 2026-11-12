//! Composable prototype-overlay and Sum-of-Products study compiler.

use deser::{Deserialize, Serialize};

#[path = "study/compiler.rs"]
mod compiler;
#[path = "study/overlay.rs"]
mod overlay;
#[path = "study/schema.rs"]
pub mod schema;
#[cfg(test)]
#[path = "study/tests.rs"]
mod tests;

/// Early-stopping / rung pruning policy for multi-fidelity studies.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[deser(rename_all = "kebab-case")]
pub enum StudyPrune {
    #[default]
    None,
    PairedLcb,
}

impl StudyPrune {
    /// Parse a prune strategy name from string.
    pub fn parse_opt(s: &str) -> Result<Self, String> {
        match s.trim().to_ascii_lowercase().as_str() {
            "none" => Ok(Self::None),
            "paired-lcb" | "paired_lcb" => Ok(Self::PairedLcb),
            other => Err(format!(
                "unknown study prune strategy: '{other}'; expected 'none' or 'paired-lcb'"
            )),
        }
    }
}

/// Single executable configuration cell lowered from a study matrix.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct StudyCell {
    pub env_id: String,
    pub arm_id: String,
    pub cell_id: String,
    pub baseline: bool,
    pub factors: Vec<(String, String)>,
    pub experiment: String,
    pub rounds: u32,
    pub repetitions: u32,
    pub canonical_toml: String,
    pub value: crate::artifact::json::Value,
}

/// Compiled study specification containing rungs, policy, and lowered cells.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CompiledStudy {
    pub is_multi: bool,
    pub rungs: Vec<u32>,
    pub prune: StudyPrune,
    pub cells: Vec<StudyCell>,
}

/// Options controlling study compilation and CLI overrides.
#[derive(Clone, Debug, Default)]
pub struct CompileOpts<'a> {
    pub cli_sets: &'a [String],
    pub rounds: Option<u32>,
    pub repetitions: Option<u32>,
    pub prune: Option<StudyPrune>,
}

/// Compile a study TOML file on disk into a CompiledStudy.
pub fn compile_file(
    path: &std::path::Path,
    opts: &CompileOpts<'_>,
) -> Result<CompiledStudy, String> {
    compiler::compile_file(path, opts)
}

/// Compile a study TOML string relative to a base directory into a
/// CompiledStudy.
pub fn compile_str(
    text: &str,
    base_dir: &std::path::Path,
    opts: &CompileOpts<'_>,
) -> Result<CompiledStudy, String> {
    compiler::compile_str(text, base_dir, opts)
}

/// Return the complete JSON Schema string for v4 tune specifications.
pub fn tune_schema() -> String {
    schema::tune_schema()
}
