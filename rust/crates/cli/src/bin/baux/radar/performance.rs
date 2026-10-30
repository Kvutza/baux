use crate::cli_radar::{PerformanceAction, PerformanceTarget};
use std::path::Path;

const RUNTIME_TARGET: &str = "//rust/crates/runner:baux-runner";

pub(super) fn run(root: &Path, action: PerformanceAction) -> Result<(), String> {
    baux_tooling::terminal::stderr(
        baux_tooling::terminal::Tag::Orchestration,
        "runtime · load performance analyzer",
    );
    let mut args = vec!["radar".to_owned(), "performance".to_owned()];
    match action {
        PerformanceAction::Show {
            config,
            target,
            run,
            json,
        } => {
            args.extend(["show".to_owned(), config.display().to_string()]);
            target_arg(&mut args, target);
            optional_path(&mut args, "--run", run.as_deref());
            if json {
                args.push("--json".to_owned());
            }
        }
        PerformanceAction::Record {
            config,
            target,
            run,
            output,
        } => {
            args.extend(["record".to_owned(), config.display().to_string()]);
            target_arg(&mut args, target);
            path_arg(&mut args, "--run", &run);
            path_arg(&mut args, "--output", &output);
        }
        PerformanceAction::Gate {
            config,
            target,
            run,
            baseline,
            tolerance_percent,
        } => {
            args.extend(["gate".to_owned(), config.display().to_string()]);
            target_arg(&mut args, target);
            optional_path(&mut args, "--run", run.as_deref());
            path_arg(&mut args, "--baseline", &baseline);
            args.extend([
                "--tolerance-percent".to_owned(),
                tolerance_percent.to_string(),
            ]);
        }
    }
    crate::cli_cfg::Config::from_env()
        .build_system
        .run(root, RUNTIME_TARGET, &args)
}

fn target_arg(args: &mut Vec<String>, target: PerformanceTarget) {
    args.extend(["--target".to_owned(), target.name().to_owned()]);
}

fn optional_path(args: &mut Vec<String>, flag: &str, path: Option<&Path>) {
    if let Some(path) = path {
        path_arg(args, flag, path);
    }
}

fn path_arg(args: &mut Vec<String>, flag: &str, path: &Path) {
    args.extend([flag.to_owned(), path.display().to_string()]);
}
