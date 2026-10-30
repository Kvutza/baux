//! Explicit full-parameter initialized-model evidence. This is deliberately
//! separate from the fast component suite and does not perform optimizer steps.
use super::*;
use baux::artifact::forward::{ForwardTrace, WordHash};
use baux::semantics::mhc4::probe::initialized as case;
use metal::objc::rc::autoreleasepool;

#[test]
#[ignore = "full 1.18B-parameter initialized forward; set BAUX_FORWARD_TRACE to save evidence"]
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
            "INITIALIZED_FORWARD target=metal seed={} parameters={elements} rows={} targets={} \
             output={}",
            case::SEED,
            case::ROWS,
            case::TARGETS,
            std::path::Path::new(&output).display()
        );
        Ok(())
    })
}
