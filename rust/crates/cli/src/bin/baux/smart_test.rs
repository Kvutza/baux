use baux_tooling::build::BuildSystem;
use clap::Args;
use std::path::Path;
use std::process::Command;

#[derive(Args, Clone, Copy, Debug, Default)]
pub struct TestArgs {
    /// Run the complete test suite across all targets.
    #[arg(long)]
    pub all: bool,
    /// Include GPU hardware tests (Metal, OpenCL, CUDA).
    #[arg(long)]
    pub gpu: bool,
}

pub fn run(root: &Path, args: TestArgs, engine: BuildSystem) -> Result<(), String> {
    if args.all {
        return engine.test_all(root);
    }
    match engine {
        BuildSystem::Buck2 => run_buck2(root, args.gpu),
        BuildSystem::Bazel => engine.test_all(root),
    }
}

fn run_buck2(root: &Path, include_gpu: bool) -> Result<(), String> {
    let files = changed_files(root)?;
    if files.is_empty() {
        println!("o|test: no modified files detected; running CPU test suite.");
        return run_suite(root, include_gpu);
    }
    let targets = impacted_targets(root, &files, include_gpu)?;
    if targets.is_empty() {
        println!("o|test: no impacted test targets for modified files.");
        return Ok(());
    }
    BuildSystem::Buck2.test_targets(root, &targets)
}

fn run_suite(root: &Path, include_gpu: bool) -> Result<(), String> {
    if include_gpu {
        return BuildSystem::Buck2.test_all(root);
    }
    let targets = query_cpu(root)?;
    BuildSystem::Buck2.test_targets(root, &targets)
}

fn changed_files(root: &Path) -> Result<Vec<String>, String> {
    if let Ok(output) = Command::new("jj")
        .current_dir(root)
        .args(["diff", "--name-only", "--no-pager"])
        .output()
    {
        if output.status.success() {
            let files: Vec<String> = String::from_utf8_lossy(&output.stdout)
                .lines()
                .map(str::trim)
                .filter(|line| !line.is_empty() && root.join(line).exists())
                .map(str::to_owned)
                .collect();
            if !files.is_empty() {
                return Ok(files);
            }
        }
    }
    let output = Command::new("git")
        .current_dir(root)
        .args(["status", "--porcelain=v1", "-z"])
        .output()
        .map_err(|e| format!("git status: {e}"))?;
    if !output.status.success() {
        return Ok(Vec::new());
    }
    parse_status(&output.stdout)
}

fn parse_status(raw: &[u8]) -> Result<Vec<String>, String> {
    let mut files = Vec::new();
    let entries = raw.split(|&b| b == 0);
    let mut skip_next = false;
    for entry in entries {
        if entry.is_empty() {
            continue;
        }
        if skip_next {
            skip_next = false;
            continue;
        }
        let text = std::str::from_utf8(entry).map_err(|e| e.to_string())?;
        if text.len() > 3 {
            let status = &text[..2];
            let path = &text[3..];
            files.push(path.to_string());
            if status.contains('R') {
                skip_next = true;
            }
        }
    }
    Ok(files)
}

fn impacted_targets(
    root: &Path,
    files: &[String],
    include_gpu: bool,
) -> Result<Vec<String>, String> {
    let mut cmd = Command::new("./admin/tools/buck2w");
    cmd.current_dir(root).args([
        "bxl",
        "--console",
        "simple",
        "//admin/buck2:baux.bxl:impacted_tests",
        "--",
        "--include-gpu",
        if include_gpu { "true" } else { "false" },
        "--files",
    ]);
    cmd.args(files);
    let out = cmd
        .output()
        .map_err(|e| format!("buck2 bxl impacted_tests: {e}"))?;
    if !out.status.success() {
        return Err(format!(
            "buck2 bxl impacted_tests failed with {}:\n{}",
            out.status,
            String::from_utf8_lossy(&out.stderr)
        ));
    }
    let targets = String::from_utf8_lossy(&out.stdout)
        .lines()
        .map(str::trim)
        .filter(|t| t.starts_with("root//"))
        .map(str::to_owned)
        .collect();
    Ok(targets)
}

fn query_cpu(root: &Path) -> Result<Vec<String>, String> {
    let out = Command::new("./admin/tools/buck2w")
        .current_dir(root)
        .args([
            "bxl",
            "--console",
            "simple",
            "//admin/buck2:baux.bxl:cpu_tests",
        ])
        .output()
        .map_err(|e| format!("buck2 bxl cpu_tests: {e}"))?;
    if !out.status.success() {
        return Err(format!(
            "buck2 bxl cpu_tests failed with {}:\n{}",
            out.status,
            String::from_utf8_lossy(&out.stderr)
        ));
    }
    let targets = String::from_utf8_lossy(&out.stdout)
        .lines()
        .map(str::trim)
        .filter(|t| t.starts_with("root//"))
        .map(str::to_owned)
        .collect();
    Ok(targets)
}
