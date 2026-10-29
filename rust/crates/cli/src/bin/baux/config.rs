pub(crate) use baux_tooling::build::BuildSystem;

pub(crate) struct Config {
    pub(crate) build_system: BuildSystem,
}

impl Config {
    pub(crate) fn from_env() -> Self {
        let build_system = BuildSystem::from_env();
        Self { build_system }
    }
}
