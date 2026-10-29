use clap::Subcommand;
use std::path::{Path, PathBuf};
use std::process::Command;

const RUNTIME_TARGET: &str = "//rust/crates/runner:baux-runner";

#[derive(Debug, PartialEq, Eq, Subcommand)]
pub(crate) enum Action {
    /// Bootstrap the pinned CUDA-Oxide, Rust, and LLVM toolchains on an NVIDIA
    /// Linux host.
    Setup {
        #[arg(trailing_var_arg = true, allow_hyphen_values = true)]
        args: Vec<String>,
    },
    /// Build and validate the sm75 CUDA wheel on a Modal T4.
    Wheel {
        /// Local directory for the validated wheel.
        #[arg(long, default_value = "dist")]
        out: PathBuf,
    },
    /// Compile CUDA-Oxide device modules and synthesized PTX kernels for the
    /// detected GPU.
    ///
    /// Selects the exact SM75 (T4), SM80 (A100), or SM90a (H100) build target.
    Build,
    /// Run CUDA primitive fixtures against host references.
    ///
    /// Checks the implemented perturbation, neighbor and tensor operations on
    /// finite fixtures; this is not whole-optimizer or universal numerical
    /// parity.
    Parity,
    /// Run resident CUDA search and acquisition fixtures.
    ///
    /// Checks proposal, scoring and selection outputs. Host synchronization is
    /// part of validation; this does not establish pretraining trajectory
    /// parity.
    Resident,
    /// Measure device latency and memory bandwidth consumption during the
    /// resident prefill slice.
    ///
    /// Isolates causal attention prefill wall time across variable prompt
    /// lengths, evaluating QKV projection throughput and logarithmic PISA
    /// attention pyramid construction latency.
    Prefill,
    /// Check forward-program components and the recurrent model fixture.
    ///
    /// Includes FP16 conversion, hierarchical mHC-4 readout, routing, sampling,
    /// scoring and recurrence checks. It proves no convergence or learning
    /// bound.
    ModelCheck {
        /// Also capture the full initialized mHC-4 parameters and forward
        /// outputs.
        #[arg(long, value_name = "OUTPUT")]
        initialized: Option<String>,
    },
    /// Compare initialized Metal and CUDA traces under the shared contract.
    Compare {
        reference: PathBuf,
        candidate: PathBuf,
    },
    /// Benchmark bounded gather queries over a persistent million-token PISA
    /// cache.
    ///
    /// Allocates a persistent attention index and checks bounded gather results
    /// against the shared host fixture. Reports measurements, not a latency
    /// guarantee.
    Context {
        #[arg(trailing_var_arg = true, allow_hyphen_values = true)]
        args: Vec<String>,
    },
    /// Execute looped mHC model checkpoints on input token sequences and
    /// persist outputs.
    ///
    /// Runs forward inference over tokenized prompt sequences using pinned
    /// 5-layer recurrent weights, persisting logits, hidden state
    /// checkpoints, and generation metadata to disk.
    Model {
        #[arg(required = true, trailing_var_arg = true, allow_hyphen_values = true)]
        args: Vec<String>,
    },
    /// Generate tokens using device-side speculative accepted-prefix
    /// verification and repair.
    ///
    /// Executes parallel speculative drafting followed by immediate device-side
    /// prefix verification, performing single-cycle suffix repair upon
    /// verification divergence without host roundtrips.
    Generate {
        #[arg(required = true, trailing_var_arg = true, allow_hyphen_values = true)]
        args: Vec<String>,
    },
    /// Measure real accepted-prefix generation on a Modal T4.
    ///
    /// Launches a cloud sandbox on Modal, executes causal prefill and
    /// speculative repair passes on the target GPU, and records exact token
    /// progress and wall/device timings.
    GenerateT4 {
        /// Total context sequence capacity in tokens (powers of two up to
        /// 1,048,576).
        #[arg(long, default_value_t = 4_096, help_heading = "Sequence Dimensions")]
        context: usize,
        /// Prompt prefix length in tokens.
        #[arg(long, default_value_t = 128, help_heading = "Sequence Dimensions")]
        prompt: usize,
        /// Number of recurrent layer visits executed per wave.
        #[arg(long, default_value_t = 2, help_heading = "Recurrence & Execution")]
        visits: usize,
        /// Softmax sampling temperature for token generation.
        #[arg(long, default_value = "0.8", help_heading = "Sampling & Randomness")]
        temperature: String,
        /// Pseudo-random number generator seed for reproducible sampling.
        #[arg(long, default_value_t = 17, help_heading = "Sampling & Randomness")]
        seed: u64,
        /// Verifier wave unroll factor.
        #[arg(long, default_value_t = 2, help_heading = "Recurrence & Execution")]
        unroll: usize,
        /// Stream bounded activation chunks while retaining causal KV; zero
        /// uses packed execution.
        #[arg(long, default_value_t = 0, help_heading = "Memory & Streaming")]
        chunk: usize,
        /// Output path where generation timing, token IDs, and repair traces
        /// are serialized.
        #[arg(
            long,
            default_value = "results/cuda/generation-t4",
            help_heading = "Output Options"
        )]
        out: PathBuf,
    },
    /// Execute an mHC-4 experiment configuration on a Modal T4.
    ///
    /// Uses the same causal corpus windows or generation settings, surrogate
    /// settings and controller policy as Metal. Unsupported fields are
    /// rejected.
    TuneT4 {
        /// Version 4 mHC-4 experiment configuration.
        config: PathBuf,
        /// Fresh output directory for the measured artifacts.
        #[arg(long, default_value = "results/cuda/mhc4-1m-t4")]
        out: PathBuf,
    },
    /// Generate tokens using non-autoregressive block diffusion checkpoint
    /// sampling.
    ///
    /// Performs iterative mask denoising across contiguous token blocks without
    /// sequential autoregressive dependencies, generating candidate
    /// proposals in parallel steps.
    Diffusion {
        #[arg(required = true, trailing_var_arg = true, allow_hyphen_values = true)]
        args: Vec<String>,
    },
    /// Verify learned mask inputs, corruption schedules, and denoising sampling
    /// on the CUDA fixture.
    ///
    /// Runs synthetic test fixtures through the block diffusion pipeline to
    /// guarantee correct coordinate masking, noise variance scheduling, and
    /// cross-entropy loss computation.
    DiffusionCheck,
    /// Run the architecture-aware synthesized PTX hardware probe on the CUDA
    /// device.
    PtxProbe {
        #[arg(trailing_var_arg = true, allow_hyphen_values = true)]
        args: Vec<String>,
    },
    /// Run native CUDA actions under compute-sanitizer memcheck.
    ///
    /// Fails on reported device memory-access errors in the executed fixtures.
    /// This command does not run racecheck, initcheck or synccheck.
    Sanitize {
        #[arg(trailing_var_arg = true, allow_hyphen_values = true)]
        args: Vec<String>,
    },
    /// Measure packed Rademacher perturbation bandwidth and memory bus
    /// throughput.
    ///
    /// Profiles memory bus efficiency for bit-packed pseudo-random coordinate
    /// generation and in-place weight perturbation, reporting effective
    /// GB/s against theoretical device peak.
    Bench {
        /// Optional kernel bank selection; `muon` measures optimizer updates.
        #[arg(trailing_var_arg = true, allow_hyphen_values = true)]
        args: Vec<String>,
    },
    /// Measure the real resident BF16 BO curve on a Modal T4.
    ///
    /// Evaluates sequential candidate acquisition, device-side perturbation,
    /// and surrogate updates directly in GPU VRAM across multiple
    /// optimization iterations.
    Modal {
        /// Number of candidate points acquired per optimization batch.
        #[arg(long, default_value_t = 4, help_heading = "Optimization Workload")]
        candidates: usize,
        /// Historical observation capacity retained in the surrogate Gaussian
        /// process.
        #[arg(long, default_value_t = 32, help_heading = "Optimization Workload")]
        history: usize,
        /// Number of sequential Bayesian optimization iterations to execute.
        #[arg(long, default_value_t = 50, help_heading = "Optimization Workload")]
        iterations: usize,
        /// File path where convergence curve and regret records should be
        /// saved.
        #[arg(
            long,
            default_value = "results/cuda/bf16-bo-curve-t4.json",
            help_heading = "Output Options"
        )]
        out: PathBuf,
    },
}

impl Action {
    fn cloud_label(&self) -> Option<&'static str> {
        match self {
            Self::Setup { .. } => Some("setup"),
            Self::Wheel { .. } => Some("wheel"),
            Self::Compare { .. } => Some("initialized comparison"),
            Self::GenerateT4 { .. } => Some("T4 generation"),
            Self::TuneT4 { .. } => Some("T4 experiment"),
            Self::Diffusion { .. } => Some("diffusion"),
            Self::DiffusionCheck => Some("diffusion check"),
            Self::PtxProbe { .. } => Some("PTX probe"),
            Self::Sanitize { .. } => Some("sanitizer"),
            Self::Bench { .. } => Some("benchmark"),
            Self::Modal { .. } => Some("Modal optimization"),
            _ => None,
        }
    }

    fn label(&self) -> &'static str {
        if let Some(name) = self.cloud_label() {
            return name;
        }
        match self {
            Self::Build => "build",
            Self::Parity => "parity",
            Self::Resident => "resident",
            Self::Prefill => "prefill",
            Self::ModelCheck { .. } => "model check",
            Self::Context { .. } => "context",
            Self::Model { .. } => "model",
            Self::Generate { .. } => "generate",
            _ => unreachable!("handled by cloud_label"),
        }
    }

    fn command(&self) -> (&'static str, Vec<String>) {
        match self {
            Action::Context { args } => ("context", args.clone()),
            Action::Generate { args } => ("generate", args.clone()),
            Action::Diffusion { args } => ("diffusion", args.clone()),
            Action::Model { args } => ("model", args.clone()),
            Action::Sanitize { args } => ("sanitize", args.clone()),
            Action::Bench { args } => ("bench", args.clone()),
            Action::PtxProbe { args } => ("ptx-probe", args.clone()),
            Action::ModelCheck { initialized } => (
                "model-check",
                initialized
                    .as_ref()
                    .map(|path| vec!["--initialized".into(), path.clone()])
                    .unwrap_or_default(),
            ),
            Action::Compare { .. } => unreachable!("comparison is local"),
            _ => (self.diagnostic(), Vec::new()),
        }
    }

    fn diagnostic(&self) -> &'static str {
        match self {
            Action::Build => "build",
            Action::Parity => "parity",
            Action::Resident => "resident",
            Action::Prefill => "prefill",
            Action::ModelCheck { .. } => "model-check",
            Action::DiffusionCheck => "diffusion-check",
            _ => unreachable!("action requires arguments or has no native command"),
        }
    }
}

fn run_modal(
    root: &Path,
    candidates: usize,
    history: usize,
    iterations: usize,
    out: &Path,
) -> Result<(), String> {
    if candidates == 0 || history == 0 || iterations == 0 {
        return Err("CUDA Modal BO dimensions must be positive".into());
    }
    if let Some(parent) = out.parent() {
        std::fs::create_dir_all(parent).map_err(|error| error.to_string())?;
    }
    let status = Command::new(root.join("tools/buck2w"))
        .current_dir(root)
        .args(["run", "//rust/crates/modal-runner:baux-modal", "--", "bo"])
        .arg("--candidates")
        .arg(candidates.to_string())
        .arg("--history")
        .arg(history.to_string())
        .arg("--iterations")
        .arg(iterations.to_string())
        .arg("--output")
        .arg(out)
        .status()
        .map_err(|error| error.to_string())?;
    if status.success() {
        Ok(())
    } else {
        Err(format!("Modal CUDA BO gate exited with {status}"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    #[test]
    fn initialized_command() {
        let parsed = crate::Cli::try_parse_from([
            "baux",
            "cuda",
            "model-check",
            "--initialized",
            "forward.json",
        ])
        .unwrap();
        let crate::Action::Cuda { command } = parsed.action else {
            panic!("expected CUDA action");
        };
        assert_eq!(
            command.command(),
            (
                "model-check",
                vec!["--initialized".into(), "forward.json".into()]
            )
        );
        assert_eq!(
            Action::ModelCheck { initialized: None }.command(),
            ("model-check", Vec::new())
        );
        assert!(
            crate::Cli::try_parse_from(["baux", "cuda", "model-check", "--initialized",]).is_err()
        );
    }

    #[test]
    fn compare_command() {
        let parsed =
            crate::Cli::try_parse_from(["baux", "cuda", "compare", "metal.json", "cuda.json"])
                .unwrap();
        let crate::Action::Cuda { command } = parsed.action else {
            panic!("expected CUDA action");
        };
        assert_eq!(
            command,
            Action::Compare {
                reference: "metal.json".into(),
                candidate: "cuda.json".into(),
            }
        );
    }

    #[test]
    fn wheel_directory() {
        let parsed = crate::Cli::try_parse_from(["baux", "cuda", "wheel"]).unwrap();
        let crate::Action::Cuda { command } = parsed.action else {
            panic!("expected CUDA action");
        };
        assert_eq!(command, Action::Wheel { out: "dist".into() });
    }
}

#[allow(clippy::too_many_arguments)]
fn generate_t4(
    root: &Path,
    context: usize,
    prompt: usize,
    visits: usize,
    temperature: &str,
    seed: u64,
    unroll: usize,
    chunk: usize,
    out: &Path,
) -> Result<(), String> {
    let temperature_value = temperature
        .parse::<f32>()
        .map_err(|_| "CUDA generation temperature must be a number")?;
    if context == 0
        || prompt == 0
        || prompt >= context
        || visits == 0
        || !temperature_value.is_finite()
        || temperature_value < 0.0
        || unroll == 0
        || unroll > 32
        || (chunk != 0
            && (!(64..=4096).contains(&chunk) || !chunk.is_power_of_two() || context % chunk != 0))
    {
        return Err("invalid CUDA generation dimensions or sampling parameters".into());
    }
    let status = Command::new(root.join("tools/buck2w"))
        .current_dir(root)
        .args([
            "run",
            "//rust/crates/modal-runner:baux-modal",
            "--",
            "generate",
        ])
        .arg("--context")
        .arg(context.to_string())
        .arg("--prompt")
        .arg(prompt.to_string())
        .arg("--visits")
        .arg(visits.to_string())
        .arg("--temperature")
        .arg(temperature.to_string())
        .arg("--seed")
        .arg(seed.to_string())
        .arg("--unroll")
        .arg(unroll.to_string())
        .arg("--chunk")
        .arg(chunk.to_string())
        .arg("--output")
        .arg(out)
        .status()
        .map_err(|error| error.to_string())?;
    if status.success() {
        Ok(())
    } else {
        Err(format!("Modal CUDA generation exited with {status}"))
    }
}

fn tune_t4(root: &Path, path: &Path, out: &Path) -> Result<(), String> {
    let args = [
        "cuda-tune-t4".to_owned(),
        path.display().to_string(),
        "--out".to_owned(),
        out.display().to_string(),
    ];
    crate::cli_cfg::Config::from_env()
        .build_system
        .run(root, RUNTIME_TARGET, &args)
}

pub(crate) fn run(root: &Path, action: Action) -> Result<(), String> {
    if let Action::Wheel { out } = &action {
        let status = Command::new("modal")
            .current_dir(root)
            .args(["run", "ops/modal_build.py", "--output-directory"])
            .arg(out)
            .status()
            .map_err(|error| error.to_string())?;
        return status
            .success()
            .then_some(())
            .ok_or_else(|| format!("Modal CUDA wheel exited with {status}"));
    }
    let label = action.label();
    baux_tooling::terminal::stderr(
        baux_tooling::terminal::Tag::Kernel,
        &format!("{label} · running"),
    );
    let result = match action {
        Action::Setup { args } => {
            let status = Command::new(root.join("tools/cuda-setup"))
                .args(args)
                .current_dir(root)
                .status()
                .map_err(|error| format!("start CUDA setup: {error}"))?;
            if status.success() {
                Ok(())
            } else {
                Err(format!("CUDA setup exited with {status}"))
            }
        }
        Action::Modal {
            candidates,
            history,
            iterations,
            out,
        } => run_modal(root, candidates, history, iterations, &out),
        Action::GenerateT4 {
            context,
            prompt,
            visits,
            temperature,
            seed,
            unroll,
            chunk,
            out,
        } => generate_t4(
            root,
            context,
            prompt,
            visits,
            &temperature,
            seed,
            unroll,
            chunk,
            &out,
        ),
        Action::TuneT4 { config, out } => tune_t4(root, &config, &out),
        Action::Compare {
            reference,
            candidate,
        } => compare(&reference, &candidate),
        other => {
            if !cfg!(target_os = "linux") {
                return Err(
                    "CUDA execution requires an NVIDIA Linux host; use the same ./baux cuda \
                     command on the target GPU session"
                        .into(),
                );
            }
            let (name, args) = other.command();
            let status = Command::new(root.join("tools/cuda-run"))
                .arg(name)
                .args(args)
                .current_dir(root)
                .status()
                .map_err(|error| format!("start CUDA execution: {error}"))?;
            if status.success() {
                Ok(())
            } else {
                Err(format!("CUDA execution exited with {status}"))
            }
        }
    };
    if result.is_ok() {
        baux_tooling::terminal::stderr(
            baux_tooling::terminal::Tag::Kernel,
            &format!("{label} · complete"),
        );
    }
    result
}

fn compare(reference: &Path, candidate: &Path) -> Result<(), String> {
    let read = |path: &Path| -> Result<baux::artifact::forward::ForwardTrace, String> {
        let input = std::fs::read(path)
            .map_err(|error| format!("read initialized trace {}: {error}", path.display()))?;
        baux::artifact::json::from_slice(&input)
            .map_err(|error| format!("parse initialized trace {}: {error}", path.display()))
    };
    let reference = read(reference)?;
    let candidate = read(candidate)?;
    let diff = reference.compare(&candidate)?;
    println!(
        "FORWARD_PARITY {}",
        baux::artifact::json::to_string(&diff).map_err(|error| error.to_string())?
    );
    if diff.exact() {
        Ok(())
    } else {
        Err("initialized forward outputs differ".into())
    }
}
