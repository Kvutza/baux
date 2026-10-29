use super::*;

impl RunConfig {
    pub fn model_seed(&self) -> u64 {
        self.model_seeded(0)
    }

    pub fn model_seeded(&self, rep: u32) -> u64 {
        self.model_seed
            .unwrap_or_else(|| self.derived_seed(rep, "model", 0))
    }

    pub fn sample_seeded(&self, rep: u32) -> u64 {
        self.generation
            .as_ref()
            .and_then(|config| config.seed)
            .unwrap_or_else(|| self.derived_seed(rep, "sampling", 0))
    }

    pub fn reference_seed(&self) -> u64 {
        self.reference_seed
            .unwrap_or_else(|| self.derived_seed(0, "reference", 0))
    }

    pub fn validation_seed(&self) -> u64 {
        self.derived_seed(0, "generated-validation-task", 0)
    }

    pub fn proposal_seed(&self) -> u64 {
        self.proposal_seeded(0)
    }

    pub fn proposal_seeded(&self, rep: u32) -> u64 {
        self.proposal_seed
            .unwrap_or_else(|| self.derived_seed(rep, "proposal", 0))
    }

    pub fn acquisition_seed(&self) -> u64 {
        self.acquisition_seeded(0)
    }

    pub fn acquisition_seeded(&self, rep: u32) -> u64 {
        self.acquisition_seed
            .unwrap_or_else(|| self.derived_seed(rep, "acquisition", 0))
    }

    pub(crate) fn derived_seed(&self, rep: u32, domain: &str, index: u64) -> u64 {
        let mut state = 0xcbf2_9ce4_8422_2325u64;
        let mut mix = |bytes: &[u8]| {
            state ^= bytes.len() as u64;
            state = crate::hash::splitmix64(state);
            for byte in bytes {
                state ^= u64::from(*byte);
                state = state.wrapping_mul(0x0000_0100_0000_01b3);
            }
        };
        mix(b"baux-tune-seed-v2");
        let recipe = self.data_recipe.as_deref();
        let recipe_code = recipe_code(recipe);
        mix(&[
            experiment_code(self.experiment),
            model_code(self.model),
            recipe_code,
        ]);
        if recipe_code == u8::MAX {
            mix(recipe
                .expect("custom recipe code requires an identifier")
                .as_bytes());
        }
        mix(domain.as_bytes());
        mix(&rep.to_le_bytes());
        mix(&index.to_le_bytes());
        crate::hash::splitmix64(state) & 0x3fff_ffff_ffff_ffff
    }
}

fn experiment_code(experiment: Option<ExperimentKind>) -> u8 {
    match experiment {
        None => 0,
        Some(ExperimentKind::EndToEnd) => 1,
        Some(ExperimentKind::MoeLayer) => 2,
        Some(ExperimentKind::Pretrain) => 3,
        Some(ExperimentKind::Generation) => 4,
    }
}

fn model_code(model: Option<ModelArchitecture>) -> u8 {
    model.map_or(0, |arch| arch as u8 + 1)
}

fn recipe_code(recipe: Option<&str>) -> u8 {
    match recipe {
        None => 0,
        Some("stack-v3-rust-go-v1") => 5,
        Some("agent-v1-swe-traces-1m") => 4,
        Some("fineweb-slowrun-100m-v1") => 6,
        Some(_) => u8::MAX,
    }
}
