//! Explicit full-parameter initialized-model evidence. This is deliberately
//! separate from the fast component suite and does not perform optimizer steps.
use super::*;
use baux::artifact::forward::{ForwardTrace, WordHash};
use baux::semantics::mhc4::probe::initialized as case;
use metal::objc::rc::autoreleasepool;

/// Complete causal prefill, including vocabulary readout and loss, without
/// optimizer work. Long sequences retain every visit's KV cache across chunks.
pub fn run_latency(
    context: u32,
    samples: usize,
    profiled: bool,
    output: &std::path::Path,
) -> Result<baux::artifact::json::Value, String> {
    autoreleasepool(|| {
        if ![4096, 65_536, 1_048_576].contains(&context) {
            return Err("forward context must be 4096, 65536, or 1048576".into());
        }
        if samples == 0 {
            return Err("forward samples must be positive".into());
        }
        let runtime = Runtime::shared()?;
        let architecture = ResidualArchitecture::Mhc4;
        let weights = CandidateWeights::initialized(
            &runtime,
            case::SEED,
            crate::config::ModelInitialization::FbtPisa1Mhc4,
            architecture,
        )?;
        let packed = runtime.buffer::<u16>(MHC_FULLPARAMS);
        let command = runtime.queue.new_command_buffer();
        let blit = command.new_blit_command_encoder();
        let mut offset = 0;
        for (_, tensor, _) in weights.tensors() {
            blit.copy_from_buffer(&tensor, 0, &packed, offset, tensor.length());
            offset += tensor.length();
        }
        blit.end_encoding();
        complete(command)?;
        let row = weights.row(&packed)?;
        drop(weights);
        let decoder = decode::Decoder::with_visits(
            &runtime,
            context,
            context > ROWS,
            architecture.layer_steps().len(),
        )?;
        let mut scorer =
            block_decode::BlockDecoder::with_architecture(&runtime, context, architecture)?;
        let sequence: Vec<u32> = (0..=context)
            .map(|i| (crate::hash::splitmix64(73 + u64::from(i)) % u64::from(VOCAB)) as u32)
            .collect();
        let batch = crate::pretrain_data::CausalBatch {
            tokens: sequence[..context as usize].to_vec(),
            labels: sequence[1..].to_vec(),
            loss_mask: vec![1; context as usize],
            offset: 0,
        };
        if profiled {
            scorer.profile(&runtime, Some(context.div_ceil(ROWS) * samples as u32))?;
        }
        let mut measurements = Vec::new();
        for sample in 0..samples {
            eprintln!("FORWARD_START context={context} sample={sample}");
            let start = std::time::Instant::now();
            let score = scorer.score_batch(&runtime, &decoder, row, &batch)?;
            let wall_seconds = start.elapsed().as_secs_f64();
            if score.quality.scored_tokens != context as usize
                || !score.quality.mean_nll.is_finite()
                || !score.gpu_seconds.is_finite()
                || score.gpu_seconds <= 0.0
            {
                return Err("full forward failed coverage, finite loss, or timing checks".into());
            }
            let measurement = baux::artifact::json::json!({
                "sample":sample,"round":sample,"gpu_ms":score.gpu_seconds * 1000.0,
                "wall_ms":wall_seconds * 1000.0,"scored_tokens":score.quality.scored_tokens,
                "mean_nll":score.quality.mean_nll,
            });
            println!(
                "FULL_FORWARD context={context} {}",
                baux::artifact::json::to_string(&measurement).map_err(|error| error.to_string())?
            );
            measurements.push(measurement);
            let evidence = baux::artifact::json::json!({
                "backend":"metal","device":runtime.device.name(),"architecture":"mhc4",
                "parameters":MHC_FULLPARAMS,"seed":case::SEED,"context":context,
                "chunk_rows":ROWS,"visits":architecture.layer_steps().len(),
                "scope":"complete-causal-prefill-with-readout-and-loss",
                "input":"deterministic-synthetic-token-ids","profiled":profiled,
                "initialization_in_timing":false,"measurements":measurements,
            });
            std::fs::write(
                output,
                baux::artifact::json::to_vec(&evidence).map_err(|error| error.to_string())?,
            )
            .map_err(|error| error.to_string())?;
        }
        Ok(baux::artifact::json::json!({
            "backend":"metal","device":runtime.device.name(),"architecture":"mhc4",
            "parameters":MHC_FULLPARAMS,"seed":case::SEED,"context":context,
            "chunk_rows":ROWS,"visits":architecture.layer_steps().len(),
            "scope":"complete-causal-prefill-with-readout-and-loss",
            "input":"deterministic-synthetic-token-ids","profiled":profiled,
            "initialization_in_timing":false,"measurements":measurements,
        }))
    })
}

#[test]
#[ignore = "full-model latency; set BAUX_FORWARD_CONTEXT and BAUX_FORWARD_TIMING"]
fn forward_latency() -> Result<(), String> {
    let context = std::env::var("BAUX_FORWARD_CONTEXT")
        .map_err(|error| error.to_string())?
        .parse::<u32>()
        .map_err(|error| error.to_string())?;
    let output = std::env::var_os("BAUX_FORWARD_TIMING")
        .ok_or("set BAUX_FORWARD_TIMING to an output JSON path")?;
    let samples = std::env::var("BAUX_FORWARD_SAMPLES")
        .unwrap_or_else(|_| "3".into())
        .parse::<usize>()
        .map_err(|error| error.to_string())?;
    let profiled = std::env::var_os("BAUX_FORWARD_PROFILE").is_some();
    run_latency(context, samples, profiled, std::path::Path::new(&output)).map(|_| ())
}

#[test]
#[ignore = "full-parameter initialized forward; set BAUX_FORWARD_TRACE to save evidence"]
fn initialized_forward() -> Result<(), String> {
    autoreleasepool(|| {
        let output = std::env::var_os("BAUX_FORWARD_TRACE")
            .ok_or("set BAUX_FORWARD_TRACE to an output JSON path")?;
        let runtime = Runtime::shared()?;
        let architecture = ResidualArchitecture::Mhc4;
        let weights = CandidateWeights::initialized(
            &runtime,
            case::SEED,
            crate::config::ModelInitialization::FbtPisa1Mhc4,
            architecture,
        )?;
        let mut hashes = Vec::new();
        let tensors = weights.tensors();
        if tensors.iter().map(|(name, _, _)| *name).collect::<Vec<_>>()
            != case::FAMILIES[..tensors.len()]
                .iter()
                .map(|family| family.name)
                .collect::<Vec<_>>()
        {
            return Err(
                "Metal initialized parameter inventory differs from the shared case".into(),
            );
        }
        for (name, tensor, _) in &tensors {
            let words = unsafe {
                std::slice::from_raw_parts(
                    tensor.contents().cast::<u16>(),
                    tensor.length() as usize / 2,
                )
            };
            let mut digest = WordHash::new(name);
            digest.update(words)?;
            hashes.push(digest.finish());
        }
        let elements = hashes.iter().map(|tensor| tensor.elements).sum::<u64>();
        if elements != MHC_FULLPARAMS as u64 {
            return Err(format!(
                "Metal initialized parameter count {elements} differs"
            ));
        }
        let packed = runtime.buffer::<u16>(elements as usize);
        let command = runtime.queue.new_command_buffer();
        let blit = command.new_blit_command_encoder();
        let mut offset = 0;
        for (_, tensor, _) in &tensors {
            blit.copy_from_buffer(tensor, 0, &packed, offset, tensor.length());
            offset += tensor.length();
        }
        blit.end_encoding();
        complete(command)?;
        let row = weights.row(&packed)?;
        drop(tensors);
        drop(weights);
        let decoder = decode::Decoder::with_visits(
            &runtime,
            CONTEXT,
            false,
            architecture.layer_steps().len(),
        )?;
        let scorer =
            block_decode::BlockDecoder::with_architecture(&runtime, CONTEXT, architecture)?;
        let sequence = case::sequence();
        let batch = crate::pretrain_data::CausalBatch {
            tokens: sequence[..case::TOKENS].to_vec(),
            labels: sequence[1..].to_vec(),
            loss_mask: vec![1; case::TOKENS],
            offset: 0,
        };
        let score = scorer.score_batch(&runtime, &decoder, row, &batch)?;
        if score.quality.scored_tokens != case::TARGETS {
            return Err("Metal initialized score length differs".into());
        }
        let words = |buffer: &BufferRef| unsafe {
            std::slice::from_raw_parts(buffer.contents().cast::<u16>(), case::WORDS).to_vec()
        };
        let trace = ForwardTrace {
            format: ForwardTrace::FORMAT.into(),
            seed: case::SEED,
            input: batch.tokens,
            labels: batch.labels,
            rows: case::ROWS,
            width: WIDTH as usize,
            weights: hashes,
            encoded: words(&scorer.buffers.input),
            hidden: words(&scorer.buffers.normalized)[..case::TOKENS * WIDTH as usize].to_vec(),
            predictions: unsafe {
                std::slice::from_raw_parts(scorer.proposals.contents().cast::<u32>(), case::TARGETS)
                    .to_vec()
            },
            loss_bits: unsafe {
                std::slice::from_raw_parts(
                    scorer.buffers.losses.contents().cast::<f32>(),
                    case::TARGETS,
                )
            }
            .iter()
            .map(|v| v.to_bits())
            .collect(),
        };
        trace.validate()?;
        std::fs::write(
            &output,
            baux::artifact::json::to_vec(&trace).map_err(|error| error.to_string())?,
        )
        .map_err(|error| error.to_string())?;
        println!(
            "INITIALIZED_FORWARD backend=metal seed={} parameters={elements} rows={} targets={} \
             output={}",
            case::SEED,
            case::ROWS,
            case::TARGETS,
            std::path::Path::new(&output).display()
        );
        Ok(())
    })
}
