//! Native checker artifact built with the same pinned Lean project as its
//! proof.

use crate::build::BuildSystem;
use std::path::{Path, PathBuf};
use std::process::Command;

/// Exercise the actual C ABI and loader after kernel-checking its proof.
pub fn verify(root: &Path, engine: BuildSystem) -> Result<(), String> {
    let library = build(root)?;
    let assignment = format!("BAUX_TEST_NUMERICAL={}", library.display());
    let mut command = match engine {
        BuildSystem::Buck2 => {
            let mut command = Command::new(root.join("tools/buck2w"));
            command.args([
                "test",
                "//rust/crates/baux:baux-unit",
                "--console",
                "simple",
                "-v",
                "2",
                "--",
                "--env",
                &assignment,
                "--test-arg=--ignored",
                "--test-arg=artifact::numerical::tests::native_evidence",
            ]);
            command
        }
        BuildSystem::Bazel => {
            let mut command = Command::new("bazel");
            command.args([
                "test",
                "//rust/crates/baux:baux_test",
                "--test_output=errors",
                "--nocache_test_results",
                "--test_arg=--ignored",
                "--test_arg=artifact::numerical::tests::native_evidence",
            ]);
            command.arg(format!("--test_env={assignment}"));
            command
        }
    };
    command.current_dir(root);
    crate::execute(command)
}

/// Check the proof and build its executable predicate through Lake's dependency
/// graph.
pub fn build(root: &Path) -> Result<PathBuf, String> {
    let project = root.join("research/lean");
    let path = project.join(".lake/build/bin/libbaux_numerical.so");
    if path.is_file() {
        return path.canonicalize().map_err(|error| error.to_string());
    }
    let mut command = Command::new("lake");
    command
        .current_dir(&project)
        .args(["build", "BauxResearch.Semantics.Numerical", "numerical"]);
    crate::execute(command)?;
    if !path.is_file() {
        return Err(format!(
            "numerical checker was not built: {}",
            path.display()
        ));
    }
    path.canonicalize().map_err(|error| error.to_string())
}
