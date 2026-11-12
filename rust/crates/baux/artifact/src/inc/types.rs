use deser::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct EvalOutcome {
    pub scores: Vec<f64>,
    pub viol: f64,
    pub cost_us: u64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct StepRecord {
    pub eval_scores: Vec<f64>,
    pub eval_viol: f64,
    pub best_score: f64,
    pub best_viol: f64,
    pub accepted: bool,
    pub cost_us: u64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TrajHead {
    pub env_digest: String,
    pub opt_digest: String,
    pub rep: u32,
    pub seed: u64,
    pub num_dim: usize,
    pub completed_round: u32,
    pub has_eval_trace: bool,
    pub initial_error: f64,
    pub best_score: f64,
    pub total_accepts: u32,
    pub elapsed_s: f64,
    pub worker_s: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TrajSlice {
    pub head: TrajHead,
    pub horizon: u32,
    pub initial_error: f64,
    pub best_scores: Vec<f64>,
    pub best_viols: Vec<f64>,
    pub final_score: f64,
    pub accepts_at_horizon: u32,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct StoreStats {
    pub eval_memos: u64,
    pub trajectories: u64,
    pub step_rows: u64,
}
