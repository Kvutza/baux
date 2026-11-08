//! Explicit paper-based FlowLOB model and simulation configuration.
use baux::artifact::json::{Value, json};
use deser::{Deserialize, Serialize};

#[derive(Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[deser(deny_unknown_fields)]
pub(super) struct Model {
    pub levels: usize,
    pub steps: usize,
    pub width: usize,
    pub depth: usize,
    pub heads: usize,
    pub expansion: usize,
}
impl Model {
    pub fn parameters(self) -> usize {
        let d = self.width;
        4 * d * d
            + (16 + self.levels + self.steps) * d
            + 2
            + self.depth * ((10 + 2 * self.expansion) * d * d + (11 + self.expansion) * d)
    }
    fn valid(self) -> bool {
        (2..=20).contains(&self.levels)
            && self.levels % 2 == 0
            && (1..=32).contains(&self.steps)
            && (4..=1024).contains(&self.width)
            && self.width % 2 == 0
            && (1..=32).contains(&self.depth)
            && self.heads > 0
            && self.width % self.heads == 0
            && (1..=4).contains(&self.expansion)
    }
}
#[derive(Clone, Deserialize, Serialize)]
#[deser(deny_unknown_fields, rename_all = "kebab-case")]
pub(super) struct Config {
    pub models: Vec<Model>,
    pub banks: Vec<usize>,
    pub queues: Vec<usize>,
    pub samples: usize,
    pub backend: String,
    pub ready_period: usize,
    pub initial_seed: u32,
    pub radius: f64,
    pub solver: String,
    pub solver_steps: usize,
    pub guidance: f64,
    #[deser(default)]
    pub rollout_windows: Option<usize>,
    #[deser(default)]
    pub dataset: Option<String>,
    #[deser(default)]
    pub checkpoint: Option<String>,
}
impl Config {
    pub fn validate(&self) -> Result<(), String> {
        if self.models.is_empty()
            || self.models.iter().any(|m| !m.valid())
            || self
                .models
                .iter()
                .enumerate()
                .any(|(i, m)| self.models[..i].contains(m))
            || !(1..=128).contains(&self.samples)
            || !["integer", "rtl"].contains(&self.backend.as_str())
            || !(1..=1000).contains(&self.ready_period)
            || !self.radius.is_finite()
            || !(0.0..=1.0).contains(&self.radius)
            || self.radius == 0.0
            || !["euler", "heun", "rk4"].contains(&self.solver.as_str())
            || !(1..=1000).contains(&self.solver_steps)
            || !self.guidance.is_finite()
            || !(0.0..=10.0).contains(&self.guidance)
        {
            return Err("invalid FlowLOB model, simulator or sampling configuration".into());
        }
        if !(1..=16).contains(&self.rollout_windows.unwrap_or(1)) {
            return Err("rollout-windows must be in 1..16".into());
        }
        for path in [self.dataset.as_ref(), self.checkpoint.as_ref()]
            .into_iter()
            .flatten()
        {
            if !std::path::Path::new(path).is_file() {
                return Err(format!("missing FlowLOB input {path}"));
            }
        }
        Ok(())
    }
    pub fn request(
        &self,
        model: Model,
        design: &Value,
        target: u32,
        weights: &[f64],
        context: &Value,
        seed: u64,
    ) -> Value {
        let window = context["window"].as_u64().unwrap_or(0);
        let split = match context["phase"].as_str() {
            Some("validation") => "validation",
            Some("prequential") => "test",
            _ => "train",
        };
        json!({"version":1,"model":model,"bits":design["bits"],"lanes":design["lanes"],"banks":design["banks"],
            "queue":design["queue"],"pipeline":design["pipeline"],"weights":weights,
            "samples":self.samples,"backend":self.backend,"ready_period":self.ready_period,
            "initial_seed":self.initial_seed,"radius":self.radius,"solver":self.solver,
            "solver_steps":self.solver_steps,"guidance":self.guidance,"rollout_windows":self.rollout_windows.unwrap_or(1),"dataset":self.dataset,"checkpoint":self.checkpoint,
            "window":window,"target_seed":target,"data_seed":seed.wrapping_add(window*1_000_003)%2_147_483_647,"split":split})
    }
}

pub(super) fn check_cycles(observation: &Value, cycles: usize) -> Result<(), String> {
    let kernels = observation["kernel_counters"]["cycles"]
        .as_u64()
        .ok_or("missing kernel cycles")?;
    let weights = observation["weight_load_cycles"]
        .as_u64()
        .ok_or("missing weight load cycles")?;
    let inputs = observation["input_load_cycles"]
        .as_u64()
        .ok_or("missing input load cycles")?;
    if kernels + weights + inputs != cycles as u64 || kernels == 0 {
        return Err("FlowLOB cycle accounting mismatch".into());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn model_schema() {
        let model = Model {
            levels: 20,
            steps: 32,
            width: 8,
            depth: 2,
            heads: 2,
            expansion: 4,
        };
        assert!(model.valid());
        assert_eq!(model.parameters(), 3346);
        assert_eq!(
            Model {
                width: 768,
                depth: 9,
                heads: 12,
                ..model
            }
            .parameters(),
            98066690
        );
        assert!(!Model { heads: 3, ..model }.valid());
        assert!(check_cycles(&json!({"kernel_counters":{"cycles":3},"weight_load_cycles":2,"input_load_cycles":1}),6).is_ok());
        assert!(check_cycles(&json!({"kernel_counters":{"cycles":3},"weight_load_cycles":2,"input_load_cycles":1}),7).is_err());
    }
}

impl Config {
    pub(super) fn snapshot_inputs(&mut self, output: &std::path::Path) -> Result<(), String> {
        let mut records = Vec::new();
        for (name, path) in [
            ("dataset", self.dataset.take()),
            ("checkpoint", self.checkpoint.take()),
        ] {
            if let Some(source) = path {
                let destination = output.join(format!("input-{name}.json"));
                std::fs::copy(&source, &destination).map_err(|e| e.to_string())?;
                let digest = baux::artifact::runs::file_digest(&destination)?;
                records.push(json!({"role":name,"source":source,"snapshot":destination.display().to_string(),"sha256":digest}));
                let stored = Some(destination.display().to_string());
                if name == "dataset" {
                    self.dataset = stored;
                } else {
                    self.checkpoint = stored;
                }
            }
        }
        std::fs::write(
            output.join("inputs.json"),
            baux::artifact::json::pretty_string(&json!(records)).map_err(|e| e.to_string())?,
        )
        .map_err(|e| e.to_string())
    }
}
