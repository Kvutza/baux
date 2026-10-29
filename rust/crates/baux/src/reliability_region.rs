//! Public policy types and the trust-region configuration adapter.
//!
//! Numerical evidence, failure counting, and radius transitions are owned by
//! baux::semantics::optimizer::RadiusControl and shared with the CUDA host
//! adapter.

use crate::trust_region::TRLengthConfig;
pub use baux::semantics::optimizer::{
    ReliabilityAction, ReliabilityEvidence, ReliabilityPolicy, ReliabilityTelemetry,
};

/// Adapt the shared controller to the library's trust-region configuration.
#[derive(Debug, Clone)]
pub struct ReliabilityController(baux::semantics::optimizer::RadiusControl);

impl ReliabilityController {
    pub fn new(config: ReliabilityPolicy, bounds: TRLengthConfig) -> Result<Self, String> {
        baux::semantics::optimizer::RadiusControl::new(
            config,
            bounds.length_init,
            bounds.length_min,
            bounds.length_max,
        )
        .map(Self)
    }

    pub fn length(&self) -> f64 {
        self.0.length()
    }

    pub fn telemetry(&self) -> ReliabilityTelemetry {
        self.0.telemetry()
    }

    pub fn update(
        &mut self,
        evidence: ReliabilityEvidence,
    ) -> Result<ReliabilityTelemetry, String> {
        self.0.update(evidence)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn evidence(observed: f64, predicted: f64, accepted: bool) -> ReliabilityEvidence {
        ReliabilityEvidence {
            observed_improvement: observed,
            observation_variance: 0.01,
            predicted_improvement: predicted,
            prediction_variance: 0.01,
            accepted,
            nominal_radius: 0.01,
            response_radius: 0.02,
        }
    }

    fn controller(threshold: f64) -> ReliabilityController {
        ReliabilityController::new(
            ReliabilityPolicy {
                minimum_evidence: 1.0,
                reliability_threshold: threshold,
                ..Default::default()
            },
            TRLengthConfig::new(0.01, 0.0001, 0.1),
        )
        .unwrap()
    }

    #[test]
    fn expands() {
        let mut state = controller(0.2);
        state.update(evidence(1.0, 1.0, true)).unwrap();
        let result = state.update(evidence(1.0, 1.0, true)).unwrap();
        assert_eq!(result.action, ReliabilityAction::Expand);
        assert!(result.length > 0.01);
    }

    #[test]
    fn shrinks() {
        let mut state = controller(0.9);
        let mut result = state.telemetry();
        for _ in 0..8 {
            result = state.update(evidence(-1.0, 1.0, false)).unwrap();
        }
        assert_eq!(result.action, ReliabilityAction::Shrink);
        assert!(result.length < 0.01);
    }

    #[test]
    fn holds() {
        let mut state = controller(0.2);
        let result = state.update(evidence(-1.0, -1.0, false)).unwrap();
        assert_eq!(result.action, ReliabilityAction::Hold);
        assert_eq!(result.length, 0.01);
    }

    #[test]
    fn sharpness() {
        let mut state = controller(0.9);
        let mut diffuse = evidence(0.0, 0.0, false);
        diffuse.prediction_variance = 100.0;
        let mut result = state.telemetry();
        for _ in 0..8 {
            result = state.update(diffuse).unwrap();
        }
        // Calibrated diffuse uncertainty must hold rather than spuriously
        // shrinking to the floor
        assert_eq!(result.action, ReliabilityAction::Hold);
        assert_eq!(result.length, 0.01);

        // Even with positive accepted progress, high epistemic variance gates
        // expansion
        let mut diffuse_accepted = evidence(1.0, 1.0, true);
        diffuse_accepted.prediction_variance = 100.0;
        let expand_attempt = state.update(diffuse_accepted).unwrap();
        assert_eq!(expand_attempt.action, ReliabilityAction::Hold);
        assert_eq!(expand_attempt.length, 0.01);
    }

    #[test]
    fn invalid_policy() {
        let error = ReliabilityPolicy {
            expand_factor: 1.0,
            ..Default::default()
        }
        .validate()
        .unwrap_err();
        assert!(error.contains("expand_factor"));
    }

    #[test]
    fn failure_contract() {
        let mut controller = ReliabilityController::new(
            ReliabilityPolicy {
                failure_tolerance: 4,
                minimum_evidence: 10.0,
                ..Default::default()
            },
            TRLengthConfig::new(0.01, 0.0001, 0.1),
        )
        .unwrap();
        let unaccepted = evidence(-1.0, -1.0, false);
        // Step 1: Hold (failure 1)
        let t1 = controller.update(unaccepted).unwrap();
        assert_eq!(t1.action, ReliabilityAction::Hold);
        assert_eq!(t1.length, 0.01);
        // Step 2: Hold (failure 2)
        let t2 = controller.update(unaccepted).unwrap();
        assert_eq!(t2.action, ReliabilityAction::Hold);
        // Step 3: Hold (failure 3)
        let t3 = controller.update(unaccepted).unwrap();
        assert_eq!(t3.action, ReliabilityAction::Hold);
        // Step 4: Shrink (failure 4 == failure_tolerance)
        let t4 = controller.update(unaccepted).unwrap();
        assert_eq!(t4.action, ReliabilityAction::Shrink);
        assert_eq!(t4.length, 0.005);
    }
}
