pub mod box_pair;
pub mod build;
pub mod enn_artifact;
pub mod enn_coverage;
pub mod extract_ext;
pub mod names;
pub mod numerical;
pub mod radar;
pub mod radar_scip;
pub use baux::artifact::terminal;

use std::io::Write;
use std::process::Command;

pub fn execute(mut command: Command) -> Result<(), String> {
    let program = command.get_program().to_string_lossy();
    let args = command
        .get_args()
        .map(|arg| arg.to_string_lossy())
        .collect::<Vec<_>>();
    let dir = command
        .get_current_dir()
        .map(|p| format!(" (in {})", p.display()))
        .unwrap_or_default();
    println!("o|exec{dir}: {program} {}", args.join(" "));
    let _ = std::io::stdout().flush();
    let _ = std::io::stderr().flush();
    let status = command.status().map_err(|error| error.to_string())?;
    let _ = std::io::stdout().flush();
    let _ = std::io::stderr().flush();
    if status.success() {
        Ok(())
    } else {
        Err(format!("command exited with {status}"))
    }
}
