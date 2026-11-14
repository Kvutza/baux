use baux_tooling::build::BuildSystem;
use baux_tooling::radar::RadarAction;
use clap::{Parser, Subcommand};
use std::env;
use std::path::PathBuf;
use std::process::ExitCode;

mod cuda_tune;
mod experiment;
mod incremental;
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

#[derive(clap::Args, Clone, Debug, Default)]
pub(crate) struct TuneCliArgs {
    pub config: Option<PathBuf>,
    #[arg(long)]
    pub rounds: Option<u32>,
    #[arg(long)]
    pub repetitions: Option<u32>,
    #[arg(long)]
    pub jobs: Option<usize>,
    #[arg(short = 's', long = "set")]
    pub set: Vec<String>,
    #[arg(long)]
    pub prune: Option<String>,
    #[arg(long)]
    pub plan: bool,
    #[arg(long)]
    pub plot: bool,
    #[arg(long)]
    pub import_campaign: Option<PathBuf>,
}

#[derive(Subcommand)]
enum Action {
    Tune(TuneCliArgs),
    #[command(hide = true)]
    Schema {
        #[arg(long)]
        out: Option<PathBuf>,
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
        Action::Tune(t) => run_tune(&root, t),
        Action::Schema { out } => run_schema(&root, out),
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

fn run_tune(root: &std::path::Path, t: TuneCliArgs) -> Result<(), String> {
    let req = incremental::TuneRequest {
        config: t.config.clone().unwrap_or_default(),
        rounds: t.rounds,
        repetitions: t.repetitions,
        jobs: t.jobs,
        cli_sets: t.set.clone(),
        prune: t.prune.clone(),
        plan_only: t.plan,
        plot: t.plot,
        import_campaign: t.import_campaign.clone(),
    };
    if incremental::try_execute(root, &req)? {
        return Ok(());
    }
    let config = t.config.as_deref().ok_or("missing experiment config")?;
    experiment::run(root, config, t.rounds, BuildSystem::from_env())
}

fn run_schema(root: &std::path::Path, out: Option<PathBuf>) -> Result<(), String> {
    let written = incremental::write_schema(root, out.as_deref())?;
    println!("{}", written.display());
    Ok(())
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
    use baux::config::{ExperimentSpec, ModelArchitecture, ObjectiveFunction, OptimizerSpec};

    let text = std::fs::read_to_string(root.join(path)).map_err(|error| error.to_string())?;
    let spec = ExperimentSpec::parse(&text)?;
    let (spec_candidates, surrogate) = match &spec.optimizer {
        OptimizerSpec::BayesianSearch {
            candidates,
            surrogate,
            ..
        } => (*candidates, surrogate),
        _ => {
            return Err("active Lean target requires optimizer.algorithm = bayesian-search".into());
        }
    };
    let matches = parameters == baux::experimental::MHC4_PARAMS
        && context_tokens == spec.window_tokens.unwrap_or_default()
        && candidates == spec_candidates
        && neighbors == surrogate.neighbors
        && target_ms <= spec.round_target_ms
        && spec.round_target_ms <= ceiling_ms
        && dense_flops
            == baux::experimental::MHC4_COEFFS
                * 2
                * (u64::from(context_tokens) / baux::experimental::MHC4_PATCH)
        && spec.model == Some(ModelArchitecture::PisaMhc4V1)
        && spec.objective == Some(ObjectiveFunction::CrossEntropy)
        && objective == "BauxResearch.ObjectiveMode.causalNll"
        && sampling == "BauxResearch.SamplingMode.natural";
    if !matches {
        return Err("active Lean target disagrees with the typed experiment protocol".into());
    }
    Ok(())
}

pub(crate) use baux_tooling::execute;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cli_parsing() {
        let cli = Cli::try_parse_from([
            "baux-runner",
            "tune",
            "study.toml",
            "--rounds",
            "10",
            "--repetitions",
            "3",
            "--jobs",
            "4",
            "-s",
            "optimizer.metric=l2",
            "--prune",
            "paired-lcb",
            "--plan",
            "--plot",
            "--import-campaign",
            "old_runs",
        ])
        .expect("must parse tune args");
        if let Action::Tune(t) = cli.action {
            assert_eq!(
                t.config.as_deref(),
                Some(std::path::Path::new("study.toml"))
            );
            assert_eq!(t.rounds, Some(10));
            assert_eq!(t.repetitions, Some(3));
            assert_eq!(t.jobs, Some(4));
            assert_eq!(t.set, vec!["optimizer.metric=l2"]);
            assert_eq!(t.prune.as_deref(), Some("paired-lcb"));
            assert!(t.plan);
            assert!(t.plot);
            assert_eq!(
                t.import_campaign.as_deref(),
                Some(std::path::Path::new("old_runs"))
            );
        } else {
            panic!("expected Action::Tune");
        }

        let cli_schema = Cli::try_parse_from(["baux-runner", "schema"]).expect("must parse schema");
        assert!(matches!(cli_schema.action, Action::Schema { out: None }));
    }
}
