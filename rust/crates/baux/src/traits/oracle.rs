use crate::objective_observation::ObjectiveObservation;
use crate::search::DeviceView;

/// A black-box function evaluated at a resident candidate.
///
/// The objective vector is the optimizer-facing result. Evidence contains
/// domain-specific material, such as generated tokens, timings, or artifacts,
/// and never participates in acquisition unless converted into the objective
/// observation explicitly.
pub trait Oracle {
    type Evidence;

    fn observe(
        &mut self,
        candidate: DeviceView<'_>,
    ) -> Result<(ObjectiveObservation, Self::Evidence), String>;
}
