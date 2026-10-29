//! Empirical RMS of caller-supplied response differences, not a distribution
//! distance or a proved behavioral bound. Empty input uses the supplied metric.

pub fn response_rms(responses: Option<&[f32]>, fallback_metric_radius: f64) -> f64 {
    responses
        .filter(|values| !values.is_empty())
        .map(|values| {
            let sumsq = values
                .iter()
                .map(|value| f64::from(*value).powi(2))
                .sum::<f64>();
            (sumsq / values.len() as f64).sqrt()
        })
        .unwrap_or(fallback_metric_radius)
}
