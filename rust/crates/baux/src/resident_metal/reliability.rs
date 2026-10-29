use super::*;

pub(super) use baux::semantics::optimizer::response_rms;

impl SearchState {
    pub(super) fn reliability_evidence(
        &self,
        round: &Proposals,
        observed_improvement: f64,
        observation_variance: f64,
        accepted: bool,
        responses: Option<&[f32]>,
    ) -> Result<Option<reliability_region::ReliabilityEvidence>, String> {
        if self.reliability.is_none() {
            return Ok(None);
        }
        let predicted_improvement =
            f64::from(round.predicted_mean) - f64::from(round.incumbent_mean);
        let prediction_variance = f64::from(round.predicted_standard_error).powi(2)
            + f64::from(round.incumbent_standard_error).powi(2);
        let metric_radius = f64::from(
            *round
                .pool_radii
                .get(round.index)
                .ok_or("Selected pool radius is missing")?,
        );
        let response_radius = response_rms(responses, metric_radius);
        Ok(Some(reliability_region::ReliabilityEvidence {
            observed_improvement,
            observation_variance,
            predicted_improvement,
            prediction_variance,
            accepted,
            nominal_radius: f64::from(round.length.abs()),
            response_radius,
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rms_values() {
        // Uniform responses: sqrt(4^2) = 4.0
        let uniform = [4.0f32, -4.0f32, 4.0f32, -4.0f32];
        let radius = response_rms(Some(&uniform), 1.0);
        assert!((radius - 4.0).abs() < 1e-10);

        // Pythagorean triples: sqrt((3^2 + 4^2) / 2) = sqrt(25 / 2) =
        // sqrt(12.5) ≈ 3.5355339
        let pair = [3.0f32, 4.0f32];
        let radius = response_rms(Some(&pair), 1.0);
        let expected = (12.5f64).sqrt();
        assert!((radius - expected).abs() < 1e-10);
    }

    #[test]
    fn rms_fallback() {
        // None fallback
        let radius_none = response_rms(None, 2.5);
        assert!((radius_none - 2.5).abs() < 1e-10);

        // Empty slice fallback
        let empty: [f32; 0] = [];
        let radius_empty = response_rms(Some(&empty), 3.5);
        assert!((radius_empty - 3.5).abs() < 1e-10);
    }

    #[test]
    fn rms_sign() {
        let positive = [1.0f32, 2.0f32, 3.0f32];
        let negative = [-1.0f32, -2.0f32, -3.0f32];
        let radius_pos = response_rms(Some(&positive), 0.0);
        let radius_neg = response_rms(Some(&negative), 0.0);
        assert!((radius_pos - radius_neg).abs() < 1e-10);
    }
}
