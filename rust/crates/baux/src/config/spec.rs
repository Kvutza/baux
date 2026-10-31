//! Version 4 experiment files deserialize into closed optimizer recipes.
use super::*;

use super::{spec_execution::*, spec_optimizer::*};

/// Closed experiment syntax shared by authored inputs and frozen run records.
/// Authored inputs may omit paths and defaults; resolution materializes them in
/// the run record without rewriting the source file.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[deser(deny_unknown_fields, rename_all = "kebab-case")]
pub struct ExperimentSpec {
    pub version: u32,
    pub experiment: ExperimentKind,
    pub model: Option<ModelArchitecture>,
    pub output: Option<PathBuf>,
    pub recipe: Option<DataRecipe>,
    pub train: Option<PathBuf>,
    pub validation: Option<PathBuf>,
    pub objective: Option<ObjectiveFunction>,
    pub window_tokens: Option<u32>,
    #[deser(default = default_rounds())]
    pub rounds: u32,
    #[deser(default = default_reps())]
    pub repetitions: u32,
    #[deser(default = default_target())]
    pub round_target_ms: u32,
    pub validation_interval: Option<u32>,
    #[deser(default)]
    pub trace: bool,
    pub stage_samples: Option<u32>,
    pub optimizer: OptimizerSpec,
    pub generation: Option<GenerationConfig>,
}

impl ExperimentSpec {
    pub fn parse(text: &str) -> Result<Self, String> {
        let spec: Self = baux::artifact::toml::from_str(text)
            .map_err(|error| format!("invalid experiment: {error}"))?;
        if spec.version != 4 {
            return Err("typed experiment version must be 4".into());
        }
        spec.compile()?;
        Ok(spec)
    }

    pub fn to_toml(&self) -> Result<String, String> {
        self.compile()?;
        baux::artifact::toml::pretty_string(self).map_err(|error| error.to_string())
    }

    /// The corpus preparer receives a concrete request validated by Rust. Its
    /// historical wire format is independent of the public experiment syntax.
    pub fn corpus_request(&self, parent: &std::path::Path) -> Result<String, String> {
        #[derive(Serialize)]
        struct Request<'a> {
            version: u32,
            #[deser(flatten)]
            config: &'a RunConfig,
        }
        let mut config = self.compile()?;
        if let Some(generation) = &mut config.generation {
            generation.resolve(parent)?;
        }
        baux::artifact::toml::pretty_string(&Request {
            version: 1,
            config: &config,
        })
        .map_err(|error| error.to_string())
    }
}
