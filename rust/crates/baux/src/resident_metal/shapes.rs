use super::*;

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Tile {
    pub leaf: u32,
    pub start: u32,
    pub length: u32,
    pub pad: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DimMeta {
    pub weight: f32,
    pub tr_half_scale: f32,
    pub tied_leader: u32,
    pub categories: u32,
}

impl Default for DimMeta {
    fn default() -> Self {
        Self {
            weight: 1.0,
            tr_half_scale: 1.0,
            tied_leader: u32::MAX,
            categories: 0,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub struct Params {
    pub seed: u64,
    pub stream_seed: u64,
    pub radius: f32,
    pub candidate: u32,
    pub tiles: u32,
    pub history: u32,
    pub initialize: u32,
    pub mode: u32,
    pub base_slot: u32,
    pub blocks: u32,
    pub pool_count: u32,
    pub pool_slots: u32,
    pub dimensions: u32,
    pub num_pert: u32,
    pub sobol_offset: u32,
    pub lhd_points: u32,
    pub box_bounded: u32,
    pub has_dim_meta: u32,
    pub coord_system: u32,
    pub pad: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SelectionParams {
    pub root_seed: u64,
    pub outcomes: [f32; MAX_HISTORY],
    pub variances: [f32; MAX_HISTORY],
    pub draws: [f32; MAX_HISTORY],
    pub base_distances: [f32; MAX_HISTORY],
    pub local_scales: [f32; MAX_HISTORY],
    pub axis_history: [f32; MAX_HISTORY],
    pub axis_base: f32,
    pub axis_weight: f32,
    pub axis_enabled: u32,
    pub epistemic_scale: f32,
    pub aleatoric_scale: f32,
    pub y_scale: f32,
    pub beta: f32,
    pub radius: f32,
    pub neighbors: u32,
    pub history: u32,
    pub acquisition: u32,
    pub tiles: u32,
    pub mode: u32,
    pub resident_history: u32,
    pub resident_indices: [u32; MAX_HISTORY],
    pub implicit_history: u32,
    pub forced_candidate: u32,
    pub distance_metric: u32,
    pub local_scale_neighbors: u32,
    pub incumbent_index: u32,
    pub candidate_floor: u32,
    pub adaptive_neighbors: u32,
    pub pool_count: u32,
    pub pool_slots: u32,
    pub num_arms: u32,
    pub noise_aware: u32,
    pub y_bounds_mode: u32,
    pub y_lower: f32,
    pub y_upper: f32,
    pub affine_a: f32,
    pub affine_b: f32,
    pub affine_c: f32,
    pub pad: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub struct CandidateScore {
    pub mean: f32,
    pub standard_error: f32,
    pub score: f32,
    pub neighbors: u32,
    pub changed: u32,
    pub invalid: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub struct Decision {
    pub index: u32,
    pub valid: u32,
    pub root_seed: u64,
    pub seed: u64,
    pub radius: f32,
    pub score: f32,
    pub mode: u32,
    pub num_selected: u32,
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
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub struct Partial {
    pub anchor: f32,
    pub rejected: f32,
    pub squared: f32,
    pub changed: u32,
    pub invalid: u32,
}

const _: () = {
    assert!(core::mem::size_of::<Tile>() == 16);
    assert!(core::mem::size_of::<DimMeta>() == 16);
    assert!(core::mem::size_of::<Params>() == 88);
    assert!(core::mem::size_of::<SelectionParams>() == 3720);
    assert!(core::mem::size_of::<SelectionParams>() <= 4096);
    assert!(core::mem::size_of::<CandidateScore>() == 24);
    assert!(core::mem::size_of::<Decision>() == 112);
    assert!(core::mem::size_of::<Partial>() == 20);
};
