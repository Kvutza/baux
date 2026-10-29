use std::ffi::OsString;
use std::path::{Path, PathBuf};
use std::process::Command;

use baux_tooling::build::BuildSystem;

struct Mhc4Job<'a> {
    root: &'a Path,
    dataset: &'a Path,
    validation: Option<&'a Path>,
    prompt: u32,
    interval: Option<u32>,
    model_seed: u64,
    generated: u32,
    temperature: f32,
    proposal_seed: u64,
    acquisition: &'a str,
    acquisition_seed: u64,
    evaluation_seed: u64,
    validation_seed: u64,
    radius: f64,
    radius_min: f64,
    radius_max: f64,
    arms: u32,
    candidates: u32,
    beta: f32,
    rounds: u32,
    stage_samples: u32,
    reliability: baux::semantics::optimizer::ReliabilityPolicy,
    surrogate: baux::SurrogateSpec,
    window: Option<u32>,
    reference_seed: u64,
    trust_shape: baux::config::TrustRegionShape,
    output: &'a Path,
}

impl Mhc4Job<'_> {
    fn modal_args(&self) -> Result<Vec<OsString>, String> {
        let mut args = vec![
            OsString::from("run"),
            OsString::from("//rust/crates/modal-runner:baux-modal"),
            OsString::from("--"),
            OsString::from("mhc4"),
            OsString::from("--model-seed"),
            OsString::from(self.model_seed.to_string()),
            OsString::from("--generated"),
            OsString::from(self.generated.to_string()),
            OsString::from("--prompt"),
            OsString::from(self.prompt.to_string()),
            OsString::from("--visits"),
            OsString::from("2"),
            OsString::from("--temperature"),
            OsString::from(self.temperature.to_string()),
            OsString::from("--seed"),
            OsString::from(self.proposal_seed.to_string()),
            OsString::from("--radius"),
            OsString::from(self.radius.to_string()),
            OsString::from("--radius-min"),
            OsString::from(self.radius_min.to_string()),
            OsString::from("--radius-max"),
            OsString::from(self.radius_max.to_string()),
            OsString::from("--arms"),
            OsString::from(self.arms.to_string()),
            OsString::from("--candidates"),
            OsString::from(self.candidates.to_string()),
            OsString::from("--beta"),
            OsString::from(self.beta.to_string()),
            OsString::from("--rounds"),
            OsString::from(self.rounds.to_string()),
            OsString::from("--stage-samples"),
            OsString::from(self.stage_samples.to_string()),
            OsString::from("--acquisition"),
            OsString::from(self.acquisition),
            OsString::from("--acquisition-seed"),
            OsString::from(self.acquisition_seed.to_string()),
            OsString::from("--evaluation-seed"),
            OsString::from(self.evaluation_seed.to_string()),
            OsString::from("--validation-seed"),
            OsString::from(self.validation_seed.to_string()),
        ];
        self.append_policy(&mut args)?;
        args.push(OsString::from("--output"));
        args.push(self.output.as_os_str().to_owned());
        args.push(OsString::from("--trust-shape"));
        args.push(OsString::from(self.trust_shape.name()));
        if let (Some(_), Some(interval)) = (self.validation, self.interval) {
            args.push(OsString::from("--validation-interval"));
            args.push(OsString::from(interval.to_string()));
        }
        self.append_window(&mut args);
        Ok(args)
    }

    fn local_args(&self) -> Result<Vec<OsString>, String> {
        let mut args = vec![
            OsString::from("mhc4-bo-bench"),
            OsString::from(self.model_seed.to_string()),
            self.dataset.as_os_str().to_owned(),
            OsString::from(self.prompt.to_string()),
            self.output.as_os_str().to_owned(),
            OsString::from(self.generated.to_string()),
            OsString::from("2"),
            OsString::from(self.temperature.to_string()),
            OsString::from(self.proposal_seed.to_string()),
            OsString::from(self.radius.to_string()),
            OsString::from(self.rounds.to_string()),
            OsString::from(self.acquisition),
            OsString::from(self.acquisition_seed.to_string()),
            OsString::from(self.evaluation_seed.to_string()),
            OsString::from(self.arms.to_string()),
            OsString::from(self.candidates.to_string()),
            OsString::from(self.beta.to_string()),
            OsString::from(self.radius_min.to_string()),
            OsString::from(self.radius_max.to_string()),
            OsString::from(self.validation_seed.to_string()),
            OsString::from(self.stage_samples.to_string()),
        ];
        if let (Some(validation), Some(interval)) = (self.validation, self.interval) {
            args.push(validation.as_os_str().to_owned());
            args.push(OsString::from(self.prompt.to_string()));
            args.push(OsString::from(interval.to_string()));
        }
        self.append_policy(&mut args)?;
        args.push(OsString::from("--trust-shape"));
        args.push(OsString::from(self.trust_shape.name()));
        self.append_window(&mut args);
        Ok(args)
    }

    fn append_policy(&self, args: &mut Vec<OsString>) -> Result<(), String> {
        let reliability = baux::artifact::json::to_string(&self.reliability)
            .map_err(|error| error.to_string())?;
        let surrogate =
            baux::artifact::json::to_string(&self.surrogate).map_err(|error| error.to_string())?;
        args.push(OsString::from("--reliability"));
        args.push(OsString::from(reliability));
        args.push(OsString::from("--surrogate"));
        args.push(OsString::from(surrogate));
        Ok(())
    }

    fn append_window(&self, args: &mut Vec<OsString>) {
        if let Some(window) = self.window {
            args.push(OsString::from("--window-tokens"));
            args.push(OsString::from(window.to_string()));
            args.push(OsString::from("--reference-seed"));
            args.push(OsString::from(self.reference_seed.to_string()));
        }
    }
}

fn launch_modal(job: &Mhc4Job<'_>) -> Result<std::process::ExitStatus, String> {
    let mut command = Command::new(job.root.join("tools/buck2w"));
    command
        .current_dir(job.root)
        .env("BAUX_CUDA_DATASET", job.dataset)
        .env_remove("BAUX_CUDA_VALIDATION_DATASET")
        .args(job.modal_args()?);
    if let (Some(validation), Some(_)) = (job.validation, job.interval) {
        command.env("BAUX_CUDA_VALIDATION_DATASET", validation);
    }
    command.status().map_err(|error| error.to_string())
}

fn launch_local(job: &Mhc4Job<'_>) -> Result<std::process::ExitStatus, String> {
    let executable = if let Some(path) = std::env::var_os("BAUX_CUDA_EXECUTABLE") {
        let path = PathBuf::from(path);
        if !path.is_file() {
            return Err(format!(
                "BAUX_CUDA_EXECUTABLE is not a file: {}",
                path.display()
            ));
        }
        path
    } else {
        let build = Command::new(job.root.join("tools/cuda-run"))
            .current_dir(job.root)
            .env_remove("RUSTUP_TOOLCHAIN")
            .arg("build")
            .status()
            .map_err(|error| format!("build local CUDA experiment: {error}"))?;
        if !build.success() {
            return Ok(build);
        }
        job.root.join("target/release/baux-cuda")
    };
    let mut command = Command::new(executable);
    command
        .current_dir(job.root)
        .env("BAUX_GPU_EVIDENCE", job.output.join("gpu"))
        .args(job.local_args()?);
    command
        .status()
        .map_err(|error| format!("launch local CUDA experiment: {error}"))
}

fn resolve_digest(root: &Path) -> Result<String, String> {
    match std::env::var("BAUX_SOURCE_DIGEST") {
        Ok(value) if value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit()) => {
            Ok(value)
        }
        Ok(_) => Err("BAUX_SOURCE_DIGEST must be a SHA-256 digest".into()),
        Err(_) => baux::experimental::perf_digest(root, baux::experimental::Backend::CudaOxide),
    }
}

fn load_config(root: &Path, config_path: &Path) -> Result<baux::config::RunConfig, String> {
    let (source, _) = baux::config::load_tune(config_path)?;
    let resolved;
    let execution_path = if source.dataset().is_none() {
        resolved = crate::experiment::resolve(root, config_path, None, BuildSystem::from_env())?;
        resolved.as_path()
    } else {
        config_path
    };
    let (run, _) = baux::config::load_tune(execution_path)?;
    run.validate_experiment()?;
    if run.model != Some(baux::ModelArchitecture::FbtPisa1Mhc4V1) {
        return Err("CUDA TOML execution currently requires model=fbt-pisa1-mhc4-v1".into());
    }
    Ok(run)
}

pub(crate) fn run(root: &Path, path: &Path, out: &Path) -> Result<(), String> {
    let source_digest = resolve_digest(root)?;
    let config_path = absolute(root, path);
    let run = load_config(root, &config_path)?;
    let (prompt, generated, temperature, window) = objective_shape(&run)?;
    let dataset = run
        .dataset()
        .ok_or("CUDA mHC-4 execution requires data.train")?;
    let validation = run.validation_dataset.as_deref();
    if let Some(validation) = validation {
        if dataset.canonicalize().map_err(|error| error.to_string())?
            == validation
                .canonicalize()
                .map_err(|error| error.to_string())?
        {
            return Err("CUDA mHC-4 training and validation corpora must be separate files".into());
        }
    }
    let interval = validation.map(|_| run.validation_interval.unwrap_or(run.rounds()));
    let policy = run.resident_enn(run.acquisition_seed())?;
    check_spec(&run, &policy)?;
    let output = absolute(root, out);
    if let Some(parent) = output.parent() {
        std::fs::create_dir_all(parent).map_err(|error| error.to_string())?;
    }
    let (acquisition, beta) = match run
        .acquisition
        .unwrap_or(baux::AcquisitionConfig::UCB { beta: 2.0 })
    {
        baux::AcquisitionConfig::UCB { beta } => ("ucb", beta as f32),
        baux::AcquisitionConfig::Thompson => ("thompson", policy.ask.acquisition.beta),
        baux::AcquisitionConfig::Random | baux::AcquisitionConfig::Pareto => {
            return Err("CUDA mHC-4 resident search requires UCB or Thompson acquisition".into());
        }
    };
    let length = run.length();
    let job = Mhc4Job {
        root,
        dataset,
        validation,
        prompt,
        interval,
        model_seed: run.model_seed(),
        generated,
        temperature,
        proposal_seed: run.proposal_seed(),
        acquisition,
        acquisition_seed: run.acquisition_seed(),
        evaluation_seed: run.sample_seeded(0),
        validation_seed: run.validation_seed(),
        radius: length.length_init,
        radius_min: length.length_min,
        radius_max: length.length_max,
        arms: policy.pool.arms(),
        candidates: policy.pool.count(),
        beta,
        rounds: run.rounds(),
        window,
        reference_seed: run.reference_seed(),
        trust_shape: run
            .trust_region_shape
            .unwrap_or(baux::config::TrustRegionShape::TensorFamilyStatic),
        stage_samples: run.scorer_stage_samples.unwrap_or(0),
        surrogate: policy.surrogate_spec()?,
        reliability: run
            .reliability_controller()
            .unwrap_or_default()
            .validate()?,
        output: &output,
    };
    let status = if cfg!(target_os = "linux") {
        launch_local(&job)
    } else {
        launch_modal(&job)
    };
    let status = status?;
    if status.success() {
        std::fs::copy(&config_path, output.join("experiment.toml"))
            .map_err(|error| error.to_string())?;
        std::fs::write(
            output.join("performance-source.sha256"),
            format!("{source_digest}\n"),
        )
        .map_err(|error| error.to_string())?;
        Ok(())
    } else {
        Err(format!("CUDA mHC-4 execution exited with {status}"))
    }
}

/// Resolve objective geometry without fabricating a prompt/output split.
fn objective_shape(run: &baux::config::RunConfig) -> Result<(u32, u32, f32, Option<u32>), String> {
    if let Some(window) = run.window_tokens {
        if run.generation.is_some() {
            return Err("causal corpus windows cannot also specify generation".into());
        }
        baux::semantics::mhc4::CausalPlan::window(window as usize, 16_384)?;
        // The positional prompt cap is unused by the corpus loader. One patch
        // is the explicit causal boundary, not an invented training prompt.
        return Ok((64, window, 0.0, Some(window)));
    }
    let generation = run
        .generation
        .as_ref()
        .ok_or("CUDA mHC-4 requires a causal window or generation settings")?;
    let prompt = generation
        .corpus_prompt_tokens
        .ok_or("CUDA mHC-4 generation requires corpus-prompt-tokens")?;
    let generated = generation.max_tokens;
    if prompt == 0 || prompt % 64 != 0 || generated == 0 || generated % 64 != 0 {
        return Err("mHC-4 generation requires aligned nonempty prompt/output patches".into());
    }
    Ok((prompt, generated, generation.temperature, None))
}

fn check_spec(
    run: &baux::config::RunConfig,
    policy: &baux::ResidentEnnConfig,
) -> Result<(), String> {
    baux::semantics::mhc4::ScaleShape::parse(
        run.trust_region_shape
            .unwrap_or(baux::config::TrustRegionShape::TensorFamilyStatic)
            .name(),
    )?;
    if policy.proposal_method != baux::procedural_pool::ProposalMethod::Additive
        || run.proposal_kernel() != baux::ProposalKernel::Gaussian
        || policy.distance_scaling != baux::DistanceScaling::Global
        || policy.history_geometry != baux::config::HistoryGeometry::Latent
    {
        return Err(
            "CUDA mHC-4 does not implement every resolved optimizer field; refusing a partial \
             TOML projection"
                .into(),
        );
    }
    Ok(())
}

fn absolute(root: &Path, path: &Path) -> PathBuf {
    if path.is_absolute() {
        path.to_owned()
    } else {
        root.join(path)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn surrogate_projection() {
        let spec = baux::config::ExperimentSpec::parse(
            r#"
version = 4
experiment = "pretrain"
model = "fbt-pisa1-mhc4-v1"
[data]
recipe = "stack-v3-rust-go-v1"
[objective]
function = "cross-entropy"
sampling = "natural"
window-tokens = 1048576
[execution]
rounds = 1000
round-target-ms = 1000
[optimizer]
algorithm = "turbo-enn"
[optimizer.candidate-generation]
generator = "independent"
arms = 1
candidates = 1
[optimizer.candidate-generation.perturbation]
distribution = "gaussian"
[optimizer.surrogate]
estimator = "enn"
scaling = "global"
neighbors = 10
neighbor-rule = "fixed"
epistemic-scale = 0.05
aleatoric-scale = 0.05
fit = "leave-one-out"
fit-candidates = 30
fit-samples = 10
[optimizer.acquisition]
function = "thompson"
[optimizer.trust-region]
policy = "turbo"
initial = 0.02
min = 0.01
max = 0.2
"#,
        )
        .unwrap();
        let run = spec.compile().unwrap();
        let policy = run.resident_enn(73).unwrap();
        assert_eq!(
            objective_shape(&run).unwrap(),
            (64, 1_048_576, 0.0, Some(1_048_576))
        );
        check_spec(&run, &policy).unwrap();
        let settings = policy.surrogate_spec().unwrap();
        assert_eq!(policy.pool.count(), 1);
        assert_eq!(settings.neighbors, 10);
        assert_eq!(settings.fit_candidates, 30);
        assert_eq!(settings.fit_samples, 10);
        assert!(settings.fit);
        assert!(!settings.adaptive);
        let json = baux::artifact::json::to_string(&settings).unwrap();
        let decoded: baux::SurrogateSpec = baux::artifact::json::from_str(&json).unwrap();
        assert_eq!(decoded, settings);
    }

    #[test]
    fn job_args() {
        let root = Path::new("/workspace");
        let dataset = Path::new("/workspace/train.bauxptn");
        let validation = Path::new("/workspace/val.bauxptn");
        let output = Path::new("/workspace/out");
        let job = Mhc4Job {
            root,
            dataset,
            validation: Some(validation),
            prompt: 64,
            interval: Some(128),
            model_seed: 11,
            generated: 4096,
            temperature: 0.0,
            proposal_seed: 17,
            acquisition: "thompson",
            acquisition_seed: 19,
            evaluation_seed: 23,
            validation_seed: 29,
            radius: 0.02,
            radius_min: 0.01,
            radius_max: 0.2,
            arms: 1,
            candidates: 4,
            beta: 2.0,
            rounds: 256,
            stage_samples: 3,
            reliability: baux::semantics::optimizer::ReliabilityPolicy::default(),
            surrogate: baux::SurrogateSpec {
                neighbors: 10,
                fit: true,
                ..Default::default()
            },
            window: Some(4096),
            reference_seed: 31,
            trust_shape: baux::config::TrustRegionShape::TensorFamilyStatic,
            output,
        };
        let modal: Vec<String> = job
            .modal_args()
            .unwrap()
            .into_iter()
            .map(|item| item.into_string().unwrap())
            .collect();
        assert_eq!(
            &modal[..4],
            ["run", "//rust/crates/modal-runner:baux-modal", "--", "mhc4"]
        );
        assert!(modal.windows(2).any(|pair| pair == ["--radius", "0.02"]));
        assert!(
            modal
                .windows(2)
                .any(|pair| pair == ["--radius-min", "0.01"])
        );
        assert!(modal.windows(2).any(|pair| pair == ["--radius-max", "0.2"]));
        assert!(
            modal
                .windows(2)
                .any(|pair| pair == ["--trust-shape", "tensor-family-static"])
        );
        assert!(
            modal
                .windows(2)
                .any(|pair| pair == ["--validation-interval", "128"])
        );
        assert!(
            modal
                .windows(2)
                .any(|pair| pair == ["--window-tokens", "4096"])
        );
        assert!(
            modal
                .windows(2)
                .any(|pair| pair == ["--reference-seed", "31"])
        );

        let local: Vec<String> = job
            .local_args()
            .unwrap()
            .into_iter()
            .map(|item| item.into_string().unwrap())
            .collect();
        assert_eq!(local[0], "mhc4-bo-bench");
        assert_eq!(local[1], "11");
        assert_eq!(local[2], "/workspace/train.bauxptn");
        assert_eq!(local[9], "0.02");
        assert_eq!(local[17], "0.01");
        assert_eq!(local[18], "0.2");
        assert_eq!(local[21], "/workspace/val.bauxptn");
        assert_eq!(local[22], "64");
        assert_eq!(local[23], "128");
        assert!(
            local
                .windows(2)
                .any(|pair| pair == ["--trust-shape", "tensor-family-static"])
        );
        assert!(
            local
                .windows(2)
                .any(|pair| pair == ["--window-tokens", "4096"])
        );
        assert!(
            local
                .windows(2)
                .any(|pair| pair == ["--reference-seed", "31"])
        );
    }
}
