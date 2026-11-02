use super::*;

#[repr(C)]
#[derive(Clone, Copy)]
pub struct Tile {
    pub leaf: u32,
    pub start: u32,
    pub length: u32,
    pub pad: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct Params {
    pub seed: u64,
    pub stream_seed: u64,
    pub basis_seed: u64,
    pub radius: f32,
    pub alternate_radius: f32,
    pub candidate: u32,
    pub tiles: u32,
    pub history: u32,
    pub initialize: u32,
    pub mode: u32,
    pub base_slot: u32,
    pub blocks: u32,
    pub program: u32,
    pub trajectory_count: u32,
    pub trajectory_fresh: f32,
    pub trajectory_seeds: [u64; TRAJECTORY_CAPACITY],
    pub trajectory_weights: [f32; TRAJECTORY_CAPACITY],
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct SelectionParams {
    pub root_seed: u64,
    pub basis_seed: u64,
    pub outcomes: [f32; MAX_HISTORY],
    pub variances: [f32; MAX_HISTORY],
    pub draws: [f32; MAX_HISTORY],
    pub base_distances: [f32; MAX_HISTORY],
    pub local_scales: [f32; MAX_HISTORY],
    pub latent_norms: [f32; 4],
    pub axis_history: [f32; MAX_HISTORY],
    pub axis_candidates: [f32; 4],
    pub axis_base: f32,
    pub axis_weight: f32,
    pub axis_enabled: u32,
    pub epistemic_scale: f32,
    pub aleatoric_scale: f32,
    pub y_scale: f32,
    pub beta: f32,
    pub radius: f32,
    pub alternate_radius: f32,
    pub neighbors: u32,
    pub history: u32,
    pub acquisition: u32,
    pub tiles: u32,
    pub mode: u32,
    pub program: u32,
    pub resident_history: u32,
    pub resident_indices: [u32; 2],
    pub implicit_history: u32,
    pub latent_history: u32,
    pub forced_candidate: u32,
    pub distance_metric: u32,
    pub local_scale_neighbors: u32,
    pub incumbent_index: u32,
    pub candidate_floor: u32,
    pub sampled_history: u32,
    pub adaptive_neighbors: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct Decision {
    pub index: u32,
    pub valid: u32,
    pub root_seed: u64,
    pub seed: u64,
    pub basis_seed: u64,
    pub radius: f32,
    pub score: f32,
    pub mode: u32,
    pub program: u32,
    pub predicted_mean: f32,
    pub predicted_standard_error: f32,
    pub incumbent_mean: f32,
    pub incumbent_standard_error: f32,
    pub neighbors: u32,
    pub incumbent_neighbors: u32,
    pub candidate_means: [f32; 4],
    pub candidate_standard_errors: [f32; 4],
    pub candidate_scores: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct Partial {
    pub anchor: f32,
    pub rejected: f32,
    pub squared: f32,
    pub changed: u32,
    pub invalid: u32,
}
