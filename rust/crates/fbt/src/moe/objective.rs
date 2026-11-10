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
    let losses = crate::apple_gpu::buffer_slice::<f32>(&buffers.losses, ROWS as usize)?;
    let mask = crate::apple_gpu::buffer_slice::<u8>(&buffers.score_mask, ROWS as usize)?;
    tensor_objective(values, losses, mask)
}

pub(super) fn tensor_objective(
    values: &[f32],
    losses: &[f32],
    mask: &[u8],
) -> Result<ObjectiveStats, String> {
    if values.len() != BATCH as usize
        || losses.len() != ROWS as usize
        || mask.len() != ROWS as usize
    {
        return Err("objective tensor dimensions differ from the scorer signature".into());
    }
    let scores = [values[0], values[1]];
    if scores.iter().any(|value| !value.is_finite()) {
        return Err("nonfinite sequence observation".into());
    }
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

#[allow(clippy::too_many_arguments)]
pub(super) fn optimization_record(
    round: u32,
    initializing: bool,
    proposal: &Proposals,
    decision: NoisyDecision,
    trust_region: ControllerInfo,
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
        },
        "wall_seconds": wall_seconds,
        "optimizer_seconds": optimizer_seconds,
    })
}

pub(super) fn objective_unfused(
    command: &CommandBufferRef,
    pipelines: &Pipelines,
    tensorops: &TensorOpsPipelines,
    pisa1: &Pisa1,
    buffers: &Buffers,
    weights: CandidateRow<'_>,
) -> Result<(), String> {
    let encoder = command.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipelines.embed);
    encoder.set_buffer(0, Some(weights.buffer), weights.readout);
    encoder.set_buffer(1, Some(&buffers.tokens), 0);
    encoder.set_buffer(2, Some(&buffers.input), 0);
    encoder.set_bytes(
        3,
        size_of::<u32>() as u64,
        std::ptr::from_ref(&VOCAB).cast(),
    );
    encoder.dispatch_threads(
        thread_group(u64::from(ROWS) * u64::from(WIDTH)),
        thread_group(pipelines.embed.max_total_threads_per_threadgroup().min(256)),
    );
    encoder.end_encoding();

    let mut input: &BufferRef = &buffers.input;
    for pass in 0..FEEDBACK_PASSES {
        for layer in 0..MODEL_LAYERS {
            let norm_offset = half_bytes(u64::from(layer) * u64::from(WIDTH));
            let qkv_offset = half_bytes(u64::from(layer) * u64::from(WIDTH) * u64::from(QKV_WIDTH));
            let output_offset = half_bytes(u64::from(layer) * u64::from(WIDTH) * u64::from(WIDTH));
            rms_at(
                command,
                pipelines,
                input,
                weights.buffer,
                weights.attention_norm + norm_offset,
                &buffers.normalized,
            );
            tensor_at(
                command,
                &tensorops.qkv,
                &[
                    (&buffers.normalized, 0),
                    (weights.buffer, weights.qkv + qkv_offset),
                    (&buffers.qkv, 0),
                ],
                MTLSize {
                    width: u64::from(QKV_WIDTH / 64),
                    height: u64::from(ROWS / 128),
                    depth: 1,
                },
            )?;
            pisa1.encode_layer(command, &buffers.qkv);
            tensor_at(
                command,
                &tensorops.output_projection,
                &[
                    (pisa1.output(), 0),
                    (weights.buffer, weights.output + output_offset),
                    (&buffers.projected, 0),
                ],
                MTLSize {
                    width: u64::from(WIDTH / 64),
                    height: u64::from(ROWS / 128),
                    depth: 1,
                },
            )?;
            encode_residual(
                command,
                pipelines,
                input,
                &buffers.projected,
                &buffers.attention_state,
            );
            rms_at(
                command,
                pipelines,
                &buffers.attention_state,
                weights.buffer,
                weights.ffn_norm + norm_offset,
                &buffers.normalized,
            );
            candidate_ffn(
                command,
                pipelines,
                &buffers.normalized,
                &buffers.attention_state,
                layer,
                buffers,
                weights,
                None,
            )?;
            input = &buffers.output;
        }
        if pass + 1 < FEEDBACK_PASSES {
            encode_rms(
                command,
                pipelines,
                input,
                &buffers.unit_norm,
                &buffers.normalized,
            );
            tensor_at(
                command,
                &tensorops.output_projection,
                &[
                    (input, 0),
                    (weights.buffer, weights.feedback_state),
                    (&buffers.feedback_state, 0),
                ],
                MTLSize {
                    width: u64::from(WIDTH / 64),
                    height: u64::from(ROWS / 128),
                    depth: 1,
                },
            )?;
            tensor_at(
                command,
                &tensorops.output_projection,
                &[
                    (&buffers.normalized, 0),
                    (weights.buffer, weights.feedback_gate),
                    (&buffers.feedback_gate, 0),
                ],
                MTLSize {
                    width: u64::from(WIDTH / 64),
                    height: u64::from(ROWS / 128),
                    depth: 1,
                },
            )?;
            let shape = model_shape();
            dispatch(
                command,
                &pipelines.feedback_fuse,
                &[
                    &buffers.feedback_state,
                    &buffers.feedback_gate,
                    &buffers.feedback,
                ],
                (&shape as *const MoeShape).cast(),
                std::mem::size_of::<MoeShape>() as u64,
                u64::from(ROWS) * u64::from(WIDTH),
            );
            input = &buffers.feedback;
        }
    }
    rms_at(
        command,
        pipelines,
        input,
        weights.buffer,
        weights.final_norm,
        &buffers.normalized,
    );
    tensor_at(
        command,
        &tensorops.readout,
        &[
            (&buffers.normalized, 0),
            (weights.buffer, weights.readout),
            (buffers.logits()?, 0),
        ],
        MTLSize {
            width: u64::from(VOCAB / 64),
            height: u64::from(ROWS / 128),
            depth: 1,
        },
    )?;
    candidate_losses(command, pipelines, buffers)?;
    Ok(())
}

fn candidate_losses(
    command: &CommandBufferRef,
    pipelines: &Pipelines,
    buffers: &Buffers,
) -> Result<(), String> {
    let encoder = command.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipelines.cross_entropy);
    encoder.set_buffer(0, Some(buffers.logits()?), 0);
    encoder.set_buffer(1, Some(&buffers.labels), 0);
    encoder.set_buffer(2, Some(&buffers.losses), 0);
    encoder.set_bytes(3, size_of::<u32>() as u64, std::ptr::from_ref(&ROWS).cast());
    encoder.set_bytes(
        4,
        size_of::<u32>() as u64,
        std::ptr::from_ref(&VOCAB).cast(),
    );
    encoder.dispatch_thread_groups(
        thread_group(u64::from(ROWS)),
        MTLSize {
            width: 256,
            height: 1,
            depth: 1,
        },
    );
    encoder.end_encoding();
    let encoder = command.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipelines.sequence_loss);
    encoder.set_buffer(0, Some(&buffers.losses), 0);
    encoder.set_buffer(1, Some(&buffers.score_mask), 0);
    encoder.set_buffer(2, Some(&buffers.sequence_scores), 0);
    encoder.dispatch_thread_groups(
        thread_group(u64::from(BATCH)),
        MTLSize {
            width: 256,
            height: 1,
            depth: 1,
        },
    );
    encoder.end_encoding();
    Ok(())
}
