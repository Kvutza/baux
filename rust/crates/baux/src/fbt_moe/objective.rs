use super::*;

const OBJECTIVE_BLOCK: usize = 128;
const OBJECTIVE_BLOCKS: usize = ROWS as usize / OBJECTIVE_BLOCK;

pub(super) struct PairedStats {
    pub(super) mean: f32,
    pub(super) variance: f32,
    pub(super) blocks: [f32; OBJECTIVE_BLOCKS],
}

#[derive(Clone, Copy)]
pub(super) struct ObjectiveStats {
    pub(super) reward: f32,
    pub(super) variance: f32,
    pub(super) scored_tokens: usize,
    pub(super) codelength_bits: f64,
    pub(super) bits_per_token: f64,
    pub(super) sequence_nlls: [f32; 2],
    pub(super) block_nlls: [f32; OBJECTIVE_BLOCKS],
}

pub(super) fn mean_variance(samples: &[f32]) -> (f32, f32) {
    let mean = samples.iter().sum::<f32>() / samples.len() as f32;
    let centered = samples
        .iter()
        .map(|sample| (sample - mean).powi(2))
        .sum::<f32>();
    let variance = centered / (samples.len() * (samples.len() - 1)) as f32;
    (mean, variance)
}

pub(super) fn sequence_objective(buffers: &Buffers) -> Result<ObjectiveStats, String> {
    let values = crate::apple_gpu::buffer_slice::<f32>(&buffers.sequence_scores, BATCH as usize)?;
    let scores = [values[0], values[1]];
    if scores.iter().any(|value| !value.is_finite()) {
        return Err(format!(
            "candidate objective produced invalid sequence scores: {scores:?}"
        ));
    }
    let losses = crate::apple_gpu::buffer_slice::<f32>(&buffers.losses, ROWS as usize)?;
    let mask = crate::apple_gpu::buffer_slice::<u8>(&buffers.score_mask, ROWS as usize)?;
    let mut block_nlls = [0.0f32; OBJECTIVE_BLOCKS];
    let mut total_nll = 0.0f64;
    let mut scored_tokens = 0usize;
    for (block, output) in block_nlls.iter_mut().enumerate() {
        let start = block * OBJECTIVE_BLOCK;
        let end = start + OBJECTIVE_BLOCK;
        let mut sum = 0.0f32;
        let mut count = 0usize;
        for index in start..end {
            if mask[index] != 0 {
                let loss = losses[index];
                if !loss.is_finite() {
                    return Err(format!(
                        "candidate objective produced invalid token loss at {index}"
                    ));
                }
                sum += loss;
                count += 1;
                total_nll += f64::from(loss);
                scored_tokens += 1;
            }
        }
        if count == 0 {
            return Err(format!(
                "candidate objective block {block} has no scored tokens"
            ));
        }
        *output = sum / count as f32;
    }
    let codelength_bits = total_nll / std::f64::consts::LN_2;
    let mean = scores.iter().sum::<f32>() / scores.len() as f32;
    let (_, variance) = mean_variance(&block_nlls);
    Ok(ObjectiveStats {
        reward: -mean,
        variance,
        scored_tokens,
        codelength_bits,
        bits_per_token: codelength_bits / scored_tokens as f64,
        sequence_nlls: scores,
        block_nlls,
    })
}

pub(super) fn paired_improvement(
    candidate: &ObjectiveStats,
    incumbent: &ObjectiveStats,
) -> PairedStats {
    let blocks: [f32; OBJECTIVE_BLOCKS] =
        std::array::from_fn(|index| incumbent.block_nlls[index] - candidate.block_nlls[index]);
    let mean = std::array::from_fn::<_, 2, _>(|index| {
        incumbent.sequence_nlls[index] - candidate.sequence_nlls[index]
    })
    .iter()
    .sum::<f32>()
        / 2.0;
    let (_, variance) = mean_variance(&blocks);
    PairedStats {
        mean,
        variance,
        blocks,
    }
}

#[cfg(test)]
mod objective_tests {
    use super::*;

    #[test]
    pub(super) fn variance_contract() {
        let (mean, variance) = mean_variance(&[1.0, 3.0]);
        assert_eq!(mean, 2.0);
        assert_eq!(variance, 1.0);
    }

    #[test]
    pub(super) fn paired_noise() {
        let candidate = ObjectiveStats {
            reward: -1.0,
            variance: 0.0,
            scored_tokens: ROWS as usize,
            codelength_bits: ROWS as f64 / std::f64::consts::LN_2,
            bits_per_token: 1.0 / std::f64::consts::LN_2,
            sequence_nlls: [1.0, 1.0],
            block_nlls: [1.0; OBJECTIVE_BLOCKS],
        };
        let incumbent = ObjectiveStats {
            reward: -3.0,
            variance: 0.0,
            scored_tokens: ROWS as usize,
            codelength_bits: 3.0 * ROWS as f64 / std::f64::consts::LN_2,
            bits_per_token: 3.0 / std::f64::consts::LN_2,
            sequence_nlls: [2.0, 4.0],
            block_nlls: [2.0; OBJECTIVE_BLOCKS],
        };
        let response = paired_improvement(&candidate, &incumbent);
        assert_eq!(response.mean, 2.0);
        assert_eq!(response.variance, 0.0);
        assert_eq!(response.blocks, [1.0; OBJECTIVE_BLOCKS]);
    }
}

pub(super) fn reliability_json(state: crate::ReliabilityTelemetry) -> baux::artifact::json::Value {
    baux::artifact::json::json!({
        "action": format!("{:?}", state.action),
        "standardized_error": state.standardized_error,
        "sharpness_ratio": state.sharpness_ratio,
        "agreement_ratio": state.agreement_ratio,
        "adequacy_mean": state.adequacy_mean,
        "adequacy_lower": state.adequacy_lower,
        "evidence": state.evidence,
        "response_radius": state.response_radius,
        "response_boundary": state.response_boundary,
        "boundary_fraction": state.boundary_fraction,
        "conversion": state.conversion,
    })
}

#[allow(clippy::too_many_arguments)]
pub(super) fn optimization_record(
    round: u32,
    initializing: bool,
    proposal: &Proposals,
    decision: NoisyDecision,
    trust_region: ControllerInfo,
    reliability: Option<crate::ReliabilityTelemetry>,
    wall_seconds: f64,
    optimizer_seconds: f64,
) -> baux::artifact::json::Value {
    baux::artifact::json::json!({
        "schema": "baux.optimization.v1",
        "round": round,
        "phase": if initializing { "initialization" } else { "optimization" },
        "proposal": {
            "seed": proposal.seed,
            "radius": proposal.length,
        },
        "surrogate": {
            "predicted_mean": proposal.predicted_mean,
            "predicted_standard_error": proposal.predicted_standard_error,
            "incumbent_mean": proposal.incumbent_mean,
        },
        "acquisition": { "selected_candidate": proposal.index },
        "acceptance": {
            "observed_improvement": decision.improvement,
            "threshold": decision.threshold,
            "agreement_ratio": decision.agreement_ratio,
            "accepted": decision.accepted,
            "numerical": decision.numerical.label(),
        },
        "trust_region": {
            "outcome": format!("{:?}", decision.trust_outcome),
            "next_length": trust_region.length,
            "reliability": reliability.map(reliability_json),
        },
        "wall_seconds": wall_seconds,
        "optimizer_seconds": optimizer_seconds,
    })
}

pub(super) fn log_reliability(round: u32, state: crate::ReliabilityTelemetry) {
    eprintln!(
        "optimizer.agreement round={round} action={:?} standardized_error={} sharpness_ratio={} \
         agreement_ratio={} adequacy_mean={:.6} adequacy_lower={:.6} evidence={:.6} \
         response_radius={} response_boundary={} boundary_fraction={} conversion={} \
         next_length={:.9}",
        state.action,
        state
            .standardized_error
            .map_or_else(|| "none".into(), |value| format!("{value:.6}")),
        state
            .sharpness_ratio
            .map_or_else(|| "none".into(), |value| format!("{value:.6}")),
        state
            .agreement_ratio
            .map_or_else(|| "none".into(), |value| format!("{value:.6}")),
        state.adequacy_mean,
        state.adequacy_lower,
        state.evidence,
        state
            .response_radius
            .map_or_else(|| "none".into(), |value| format!("{value:.9}")),
        state
            .response_boundary
            .map_or_else(|| "none".into(), |value| format!("{value:.9}")),
        state
            .boundary_fraction
            .map_or_else(|| "none".into(), |value| format!("{value:.6}")),
        state
            .conversion
            .map_or_else(|| "none".into(), |value| format!("{value:.6}")),
        state.length,
    );
}
