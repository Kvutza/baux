use crate::compute::AcquisitionKind;

mod bpann_history;

pub use bpann_history::{BpannHistory, IndexedObservation, ObservationId};

#[derive(Debug, Clone, Copy)]
pub struct ProposalParams {
    pub radius: f32,
}

#[derive(Debug, Clone, Copy)]
pub struct SurrogateParams {
    pub neighbors: usize,
    pub epistemic_scale: f32,
    pub aleatoric_scale: f32,
    pub output_scale: f32,
}

#[derive(Debug, Clone, Copy)]
pub struct AcquisitionParams {
    pub kind: AcquisitionKind,
    pub beta: f32,
    pub seed: u64,
}

/// Fully resolved inputs for one resident proposal-selection operation.
#[derive(Debug, Clone, Copy)]
pub struct RoundConfig {
    pub proposal: ProposalParams,
    pub surrogate: SurrogateParams,
    pub acquisition: AcquisitionParams,
}

impl Default for RoundConfig {
    fn default() -> Self {
        Self {
            proposal: ProposalParams { radius: 0.8 },
            surrogate: SurrogateParams {
                neighbors: 10,
                epistemic_scale: 0.7,
                aleatoric_scale: 0.05,
                output_scale: 1.0,
            },
            acquisition: AcquisitionParams {
                kind: AcquisitionKind::Ucb,
                beta: 1.0,
                seed: 0,
            },
        }
    }
}
