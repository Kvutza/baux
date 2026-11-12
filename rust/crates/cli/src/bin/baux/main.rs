use clap::{Parser, Subcommand};
use std::env;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

mod config;
use config as cli_cfg;
mod check;
mod cuda_track;
mod experiment;
use experiment as cli_exp;
mod pages;
mod radar;
use radar as cli_radar;
#[path = "research/extract.rs"]
mod ext_ops;
mod research;
#[path = "research/site/aeneas.rs"]
mod site_aeneas;
#[path = "research/site/lean.rs"]
mod site_lean;
#[path = "research/site/llbc.rs"]
mod site_llbc;
#[path = "research/site/payload.rs"]
mod site_payload;
#[path = "research/site/registry.rs"]
mod site_registry;
mod smart_test;
mod work;

#[derive(Parser)]
#[command(
    name = "baux",
    bin_name = "./baux",
    version = env!("CARGO_PKG_VERSION"),
    about = "Build and run BAUX experiments",
    arg_required_else_help = true
)]
struct Cli {
    #[command(subcommand)]
    action: Action,
}

#[derive(Subcommand)]
enum Action {
    /// Find, start, or resume a Beads task for this JJ workspace.
    Context(work::Context),
    /// Show task progress or record a checkpoint against the current revision.
    Feedback(work::Feedback),
    /// Build native BAUX targets and Python distribution wheels.
    Build,
    /// Run the repository verification test suite.
    Test(smart_test::TestArgs),
    /// Format and verify JJ-tracked repository files.
    #[command(visible_alias = "gate", alias = "fmt")]
    Check,
    #[command(hide = true)]
    Names,
    /// Run a pretraining experiment or inspect saved experiment evidence.
    Tune(cli_exp::Tune),
    /// Run CUDA-Oxide locally or on a configured NVIDIA runner.
    #[command(hide = true)]
    Cuda {
        #[command(subcommand)]
        command: cuda_track::Action,
    },
    /// Inspect source reachability and performance evidence.
    Radar {
        #[command(subcommand)]
        command: cli_radar::RadarAction,
    },
    /// Work on the checked optimizer theory.
    Research {
        #[command(subcommand)]
        command: research::ResearchAction,
    },
    /// Build, verify, and serve the Python-free static website & Playground.
    Pages {
        #[command(subcommand)]
        command: pages::PagesAction,
    },
}

fn main() -> ExitCode {
    baux_tooling::terminal::finish(run())
}

fn run() -> Result<(), String> {
    let root = env::var_os("BUILD_WORKSPACE_DIRECTORY")
        .map(PathBuf::from)
        .map_or_else(|| env::current_dir().map_err(|error| error.to_string()), Ok)?;
    env::set_current_dir(&root).map_err(|error| error.to_string())?;
    let settings = cli_cfg::Config::from_env();
    match Cli::parse().action {
        Action::Context(args) => work::context(&root, args),
        Action::Feedback(args) => work::feedback(&root, args),
        Action::Build => build(&root, settings.build_system),
        Action::Test(args) => smart_test::run(&root, args, settings.build_system),
        Action::Check => check::run(&root),
        Action::Names => baux_tooling::names::cli(&root, &[], false),
        Action::Tune(args) => cli_exp::tune(&root, args, settings.build_system),
        Action::Cuda { command } => cuda_track::run(&root, command),
        Action::Radar { command } => cli_radar::run(&root, command),
        Action::Research { command } => research::run(&root, command, settings.build_system),
        Action::Pages { command } => pages::run(&root, command),
    }
}

pub(crate) use baux_tooling::execute;

fn build(root: &Path, engine: cli_cfg::BuildSystem) -> Result<(), String> {
    println!("o|build [1/2]: building native BAUX targets...");
    engine.build_native(root)?;
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    {
        engine.run(root, "//ocaml:tool", &["deps".into()])?;
        engine.run(root, "//ocaml:tool", &["build".into()])?;
    }
    println!("o|build [2/2]: building Python distribution wheels into dist/...");
    baux_tooling::build::build_dist(root, Path::new("dist"))
}
