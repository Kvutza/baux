use baux_tooling::build::BuildSystem;
use baux_tooling::radar::RadarAction;
use clap::{Parser, Subcommand};
use std::env;
use std::path::PathBuf;
use std::process::ExitCode;

mod cuda_tune;
mod experiment;
mod performance;

#[derive(Parser)]
#[command(
    name = "baux-runner",
    bin_name = "./baux",
    version = env!("CARGO_PKG_VERSION"),
    disable_help_subcommand = true
)]
struct Cli {
    #[command(subcommand)]
    action: Action,
}

#[derive(Subcommand)]
enum Action {
    Tune {
        config: PathBuf,
        #[arg(long, conflicts_with = "rounds")]
        prepare: bool,
        #[arg(long, conflicts_with = "prepare")]
        rounds: Option<u32>,
    },
    Radar {
        #[command(subcommand)]
        command: RadarAction,
    },
    #[command(hide = true)]
    Resolve {
        config: PathBuf,
        #[arg(long)]
        rounds: Option<u32>,
    },
    #[command(name = "cuda-tune-t4", hide = true)]
    CudaTuneT4 {
        config: PathBuf,
        #[arg(long)]
        out: PathBuf,
    },
    #[command(name = "research-contract", hide = true)]
    ResearchContract {
        config: PathBuf,
        #[arg(long)]
        parameters: u64,
        #[arg(long)]
        context_tokens: u32,
        #[arg(long)]
        candidates: u32,
        #[arg(long)]
        neighbors: i32,
        #[arg(long)]
        target_ms: u32,
        #[arg(long)]
        ceiling_ms: u32,
        #[arg(long)]
        dense_flops: u64,
        #[arg(long)]
        objective: String,
        #[arg(long)]
        sampling: String,
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
    match Cli::parse().action {
        Action::Tune {
            config,
            prepare,
            rounds,
        } => experiment::run(&root, &config, prepare, rounds, BuildSystem::from_env()),
        Action::Radar {
            command: RadarAction::Performance { command },
        } => performance::run(&root, command),
        Action::Radar { .. } => Err("runtime helper received a lightweight radar command".into()),
        Action::Resolve { config, rounds } => {
            let path = experiment::resolve(&root, &config, rounds, BuildSystem::from_env())?;
            println!("{}", path.display());
            Ok(())
        }
        Action::CudaTuneT4 { config, out } => cuda_tune::run(&root, &config, &out),
        Action::ResearchContract {
            config,
            parameters,
            context_tokens,
            candidates,
            neighbors,
            target_ms,
            ceiling_ms,
            dense_flops,
            objective,
            sampling,
        } => research_contract(
            &root,
            &config,
            parameters,
            context_tokens,
            candidates,
            neighbors,
            target_ms,
            ceiling_ms,
            dense_flops,
            &objective,
            &sampling,
        ),
    }
}

#[allow(clippy::too_many_arguments)]
fn research_contract(
    root: &std::path::Path,
    path: &std::path::Path,
    parameters: u64,
    context_tokens: u32,
    candidates: u32,
    neighbors: i32,
    target_ms: u32,
    ceiling_ms: u32,
    dense_flops: u64,
    objective: &str,
    sampling: &str,
) -> Result<(), String> {
    use baux::config::{
        ExperimentSpec, ModelArchitecture, ObjectiveFunction, ObjectiveSampling, OptimizerSpec,
        SurrogateEstimator,
    };

    let text = std::fs::read_to_string(root.join(path)).map_err(|error| error.to_string())?;
    let spec = ExperimentSpec::parse(&text)?;
    let (generation, surrogate) = match &spec.optimizer {
        OptimizerSpec::TurboEnn {
            candidate_generation,
            surrogate,
            ..
        } => (candidate_generation, surrogate),
        _ => return Err("active Lean target requires optimizer.algorithm = turbo-enn".into()),
    };
    let matches = parameters == baux::experimental::MHC4_PARAMS
        && context_tokens == spec.objective.window_tokens.unwrap_or_default()
        && candidates == generation.candidates
        && neighbors == surrogate.neighbors
        && target_ms <= spec.execution.round_target_ms
        && spec.execution.round_target_ms <= ceiling_ms
        && dense_flops
            == baux::experimental::MHC4_COEFFS
                * 2
                * (u64::from(context_tokens) / baux::experimental::MHC4_PATCH)
        && spec.model == Some(ModelArchitecture::FbtPisa1Mhc4V1)
        && spec.objective.function == Some(ObjectiveFunction::CrossEntropy)
        && spec.objective.sampling == Some(ObjectiveSampling::Natural)
        && surrogate.estimator == SurrogateEstimator::Enn
        && objective == "BauxResearch.ObjectiveMode.causalNll"
        && sampling == "BauxResearch.SamplingMode.natural";
    if !matches {
        return Err("active Lean target disagrees with the typed experiment protocol".into());
    }
    Ok(())
}

pub(crate) use baux_tooling::execute;
