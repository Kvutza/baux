//! Device view handles for resident trial buffers.

pub use crate::trials::DeviceView;

#[cfg(all(target_os = "macos", feature = "metal"))]
impl crate::resident_metal::SearchState {
    /// Evaluate one selected candidate and publish its observation atomically
    /// through the resident controller.
    ///
    /// Oracle failures leave the proposal pending. A successful oracle call is
    /// followed by exactly one tell and one device-state synchronization.
    pub fn evaluate<O: crate::traits::Oracle>(
        &mut self,
        proposal: &crate::resident_metal::Proposals,
        candidate: DeviceView<'_>,
        initial: bool,
        oracle: &mut O,
    ) -> Result<
        (
            crate::resident_metal::NoisyDecision,
            O::Evidence,
            std::time::Duration,
        ),
        String,
    > {
        let (observation, evidence) = oracle.observe(candidate)?;
        let start = std::time::Instant::now();
        let decision = if initial {
            self.initial_objectives(proposal, observation)?
        } else {
            self.modeled_objectives(proposal, observation)?
        };
        if self.sync()? != vec![decision.accepted] {
            return Err("resident search acceptance mismatch".into());
        }
        Ok((decision, evidence, start.elapsed()))
    }
}
