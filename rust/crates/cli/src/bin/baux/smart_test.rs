use baux_tooling::build::BuildSystem;
use clap::Args;
use std::collections::BTreeSet;
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
    baux_tooling::build::BuildSystem::Buck2.test_targets(root, &targets)
}

fn run_suite(root: &Path, include_gpu: bool) -> Result<(), String> {
    if include_gpu {
        return baux_tooling::build::BuildSystem::Buck2.test_all(root);
    }
    let targets = query_cpu(root)?;
    baux_tooling::build::BuildSystem::Buck2.test_targets(root, &targets)
}

fn changed_files(root: &Path) -> Result<Vec<String>, String> {
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

fn is_gpu(target: &str) -> bool {
    let name = target.split(':').last().unwrap_or(target);
    name.contains("metal") || name.contains("opencl") || name.contains("cuda")
}

fn impacted_targets(
    root: &Path,
    files: &[String],
    include_gpu: bool,
) -> Result<Vec<String>, String> {
    let valid_extensions = [".rs", ".metal", ".cpp", ".c", ".h", ".py", "BUCK"];
    let relevant: Vec<&str> = files
        .iter()
        .map(String::as_str)
        .filter(|f| valid_extensions.iter().any(|ext| f.ends_with(ext)))
        .collect();
    if relevant.is_empty() {
        return Ok(Vec::new());
    }

    let mut targets = BTreeSet::new();
    for chunk in relevant.chunks(30) {
        let mut cmd = Command::new("./tools/buck2w");
        cmd.current_dir(root).args([
            "uquery",
            "kind('.*test', rdeps(//build/buck2/tests:all, owner('%s')))",
        ]);
        cmd.args(chunk);
        let out = cmd.output().map_err(|e| format!("buck2 uquery: {e}"))?;
        if out.status.success() {
            for line in String::from_utf8_lossy(&out.stdout).lines() {
                let t = line.trim();
                if t.starts_with("root//") && (include_gpu || !is_gpu(t)) {
                    targets.insert(t.to_string());
                }
            }
        }
    }
    Ok(targets.into_iter().collect())
}

fn query_cpu(root: &Path) -> Result<Vec<String>, String> {
    let mut cmd = Command::new("./tools/buck2w");
    cmd.current_dir(root)
        .args(["uquery", "kind('.*test', deps(//build/buck2/tests:all))"]);
    let out = cmd.output().map_err(|e| format!("buck2 uquery: {e}"))?;
    let mut targets = Vec::new();
    for line in String::from_utf8_lossy(&out.stdout).lines() {
        let t = line.trim();
        if t.starts_with("root//") && !is_gpu(t) {
            targets.push(t.to_string());
        }
    }
    Ok(targets)
}
