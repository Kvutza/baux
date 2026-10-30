use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

pub struct BuiltArtifact {
    pub path: PathBuf,
    pub log: Vec<u8>,
}

/// Active build system - now passed explicitly, not read from env
#[derive(Clone, Copy)]
pub enum BuildSystem {
    Buck2,
    Bazel,
}

impl BuildSystem {
    pub fn from_env() -> Self {
        match std::env::var("BAUX_BUILD_SYSTEM").as_deref() {
            Ok("bazel") => Self::Bazel,
            _ => Self::Buck2,
        }
    }

    pub const fn name(self) -> &'static str {
        match self {
            Self::Bazel => "bazel",
            Self::Buck2 => "buck2",
        }
    }

    pub fn build_native(self, root: &Path) -> Result<(), String> {
        let mut command = match self {
            Self::Buck2 => {
                let mut cmd = Command::new("./tools/buck2w");
                cmd.args([
                    "build",
                    "//rust/crates/baux:baux",
                    "//rust/crates/runner:baux-runner",
                    "--local-only",
                    "--console",
                    "simple",
                    "-v",
                    "2",
                ]);
                cmd
            }
            Self::Bazel => {
                let mut cmd = Command::new("bazel");
                cmd.args(["build", "//:cpu", "//:gpu", "--config=constrained"]);
                cmd
            }
        };
        command.current_dir(root);
        crate::execute(command)
    }

    pub fn test_all(self, root: &Path) -> Result<(), String> {
        let mut command = match self {
            Self::Buck2 => {
                let mut cmd = Command::new("./tools/buck2w");
                cmd.args([
                    "test",
                    "//build/buck2/tests:all",
                    "--local-only",
                    "--console",
                    "simple",
                    "-v",
                    "2",
                ]);
                cmd
            }
            Self::Bazel => {
                let mut cmd = Command::new("bazel");
                cmd.args([
                    "test",
                    "//:check",
                    "--config=release",
                    "--config=constrained",
                ]);
                cmd
            }
        };
        command.current_dir(root);
        crate::execute(command)
    }

    pub fn run(self, root: &Path, target: &str, args: &[String]) -> Result<(), String> {
        crate::execute(self.run_command(root, target, args))
    }

    pub fn run_output(self, root: &Path, target: &str, args: &[String]) -> Result<String, String> {
        let mut command = self.run_command(root, target, args);
        command.stderr(std::process::Stdio::inherit());
        let output = command
            .output()
            .map_err(|error| format!("start {}: {error}", self.name()))?;
        checked_stdout(self.name(), output)
    }

    pub fn build_artifact(self, root: &Path, target: &str) -> Result<BuiltArtifact, String> {
        match self {
            Self::Buck2 => buck_artifact(root, target),
            Self::Bazel => bazel_artifact(root, target),
        }
    }

    pub fn run_command(self, root: &Path, target: &str, args: &[String]) -> Command {
        let isolation = std::env::var("BUCK_ISOLATION_DIR").unwrap_or_else(|_| "dev".to_string());
        let mut command = match self {
            Self::Buck2 => {
                let mut command = Command::new("./tools/buck2w");
                command.args([
                    "--isolation-dir",
                    &isolation,
                    "run",
                    "--console",
                    "simple",
                    "-v",
                    "2",
                    target,
                    "--",
                ]);
                command
            }
            Self::Bazel => {
                let mut command = Command::new("bazel");
                command.args([
                    "run",
                    "--config=release",
                    "--config=constrained",
                    target,
                    "--",
                ]);
                command
            }
        };
        command.current_dir(root).args(args);
        command
    }
}

fn buck_artifact(root: &Path, target: &str) -> Result<BuiltArtifact, String> {
    let isolation = std::env::var("BUCK_ISOLATION_DIR").unwrap_or_else(|_| "dev".to_string());
    println!(
        "o|exec: ./tools/buck2w --isolation-dir {isolation} build {target} --show-full-output"
    );
    let output = Command::new("./tools/buck2w")
        .current_dir(root)
        .args([
            "--isolation-dir",
            &isolation,
            "build",
            "--console",
            "simple",
            "-v",
            "2",
            target,
            "--show-full-output",
        ])
        .stderr(std::process::Stdio::inherit())
        .output()
        .map_err(|error| format!("start buck2: {error}"))?;
    let log = combined_log(&output);
    if !output.status.success() {
        return Err(format!(
            "Buck2 worker build failed:\n{}",
            String::from_utf8_lossy(&log)
        ));
    }
    let stdout = String::from_utf8(output.stdout)
        .map_err(|error| format!("Buck2 output was not UTF-8: {error}"))?;
    let path = stdout
        .lines()
        .find_map(|line| {
            line.strip_prefix(&format!("root{target} "))
                .or_else(|| line.strip_prefix(&format!("{target} ")))
        })
        .map(str::trim)
        .map(PathBuf::from)
        .ok_or("Buck2 did not return the worker executable path")?;
    Ok(BuiltArtifact {
        path: absolute(root, path),
        log,
    })
}

fn bazel_artifact(root: &Path, target: &str) -> Result<BuiltArtifact, String> {
    let output = Command::new("bazel")
        .current_dir(root)
        .args(["build", "--config=release", "--config=constrained", target])
        .output()
        .map_err(|error| format!("start bazel: {error}"))?;
    let mut log = combined_log(&output);
    if !output.status.success() {
        return Err(format!(
            "Bazel worker build failed:\n{}",
            String::from_utf8_lossy(&log)
        ));
    }
    let query = Command::new("bazel")
        .current_dir(root)
        .args([
            "cquery",
            "--config=release",
            "--config=constrained",
            target,
            "--output=files",
        ])
        .output()
        .map_err(|error| format!("query Bazel artifact: {error}"))?;
    log.extend_from_slice(&combined_log(&query));
    let relative = checked_stdout("bazel cquery", query)?
        .lines()
        .find(|line| !line.trim().is_empty())
        .map(str::trim)
        .map(PathBuf::from)
        .ok_or("Bazel did not return the worker executable path")?;
    let execution_root = checked_stdout(
        "bazel info",
        Command::new("bazel")
            .current_dir(root)
            .args(["info", "execution_root"])
            .output()
            .map_err(|error| format!("query Bazel execution root: {error}"))?,
    )?;
    Ok(BuiltArtifact {
        path: absolute(Path::new(execution_root.trim()), relative),
        log,
    })
}

fn checked_stdout(program: &str, output: Output) -> Result<String, String> {
    if !output.status.success() {
        return Err(format!(
            "{program} exited with {}: {}",
            output.status,
            String::from_utf8_lossy(&output.stderr).trim()
        ));
    }
    String::from_utf8(output.stdout)
        .map_err(|error| format!("{program} output was not UTF-8: {error}"))
}

fn combined_log(output: &Output) -> Vec<u8> {
    let mut log = output.stdout.clone();
    log.extend_from_slice(&output.stderr);
    log
}

fn absolute(root: &Path, path: PathBuf) -> PathBuf {
    if path.is_absolute() {
        path
    } else {
        root.join(path)
    }
}

pub fn build_dist(root: &Path, out: &Path) -> Result<(), String> {
    let mut command = Command::new("bash");
    command
        .current_dir(root)
        .args(["tools/build-wheels", &out.to_string_lossy(), "smoke"]);
    crate::execute(command)
}
