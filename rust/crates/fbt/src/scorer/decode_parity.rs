use super::*;
use crate::config::FeedbackTransition;
use decode::Decoder;
use metal::objc::rc::autoreleasepool;

#[test]
fn mhc_scorer() -> Result<(), String> {
    autoreleasepool(|| {
        let runtime = Runtime::shared()?;
        let model = CandidateWeights::seeded_for(&runtime, Some(11), ResidualArchitecture::Mhc4);
        let row = runtime.buffer::<u16>(MHC_FULLPARAMS);
        let mut offset = 0usize;
        for (_, tensor, _) in model.tensors() {
            unsafe {
                std::ptr::copy_nonoverlapping(
                    tensor.contents().cast::<u8>(),
                    row.contents().cast::<u8>().add(offset),
                    tensor.length() as usize,
                );
            }
            offset += tensor.length() as usize;
        }
        let layout = Mhc4Layout::from_weights(&model, 0);
        drop(model);
        let weights = layout.view(&row);
        let buffers = Buffers::new(&runtime);
        let pipelines = Pipelines::new(&runtime)?;
        let tensorops = TensorOpsPipelines::new(&runtime)?;
        let pisa = Pisa::new(&runtime)?;
        let command = runtime.queue.new_command_buffer();
        objective_fused(command, &pipelines, &tensorops, &pisa, &buffers, weights)?;
        complete(command)?;
        let scores = unsafe {
            std::slice::from_raw_parts(
                buffers.sequence_scores.contents().cast::<f32>(),
                BATCH as usize,
            )
        };
        assert!(scores.iter().all(|score| score.is_finite()));
        Ok(())
    })
}

#[test]
#[ignore = "decode parity tolerance needs the shared vocabulary correction"]
fn decode_prefix() -> Result<(), String> {
    autoreleasepool(|| {
        let runtime = Runtime::shared()?;
        let decoder = Decoder::new(&runtime)?;
        let weights = CandidateWeights::new(&runtime);
        let row = runtime.buffer::<u16>(FULL_PARAMETERS);
        let mut offset = 0usize;
        for (_, tensor, _) in weights.tensors() {
            unsafe {
                std::ptr::copy_nonoverlapping(
                    tensor.contents().cast::<u8>(),
                    row.contents().cast::<u8>().add(offset),
                    tensor.length() as usize,
                );
            }
            offset += tensor.length() as usize;
        }
        let weights = weights.row(&row)?;
        let buffers = Buffers::new(&runtime);
        let pipelines = Pipelines::new(&runtime)?;
        let tensorops = TensorOpsPipelines::new(&runtime)?;
        let pisa = Pisa::new(&runtime)?;
        let inputs = unsafe {
            std::slice::from_raw_parts(buffers.tokens.contents().cast::<u32>(), ROWS as usize)
        };
        unsafe {
            std::ptr::copy_nonoverlapping(inputs.as_ptr(), decoder.tokens.contents().cast(), 65);
        }
        let command = runtime.queue.new_command_buffer();
        super::super::objective_unfused(command, &pipelines, &tensorops, &pisa, &buffers, weights)?;
        complete(command)?;
        let reference = unsafe {
            std::slice::from_raw_parts(buffers.losses.contents().cast::<f32>(), ROWS as usize)
        };
        let labels = unsafe {
            std::slice::from_raw_parts(buffers.labels.contents().cast::<u32>(), ROWS as usize)
        };
        let reference_logits = unsafe {
            std::slice::from_raw_parts(
                buffers.logits()?.contents().cast::<u16>(),
                (ROWS * VOCAB) as usize,
            )
        };
        let mut maximum_error = 0.0f64;
        let mut top1_matches = 0usize;
        let mut maximum_missed_margin = 0.0f64;
        let command = runtime.queue.new_command_buffer();
        let encoder = command.new_compute_command_encoder();
        decoder.prepare_router(&encoder, weights);
        encoder.end_encoding();
        complete(command)?;
        for position in 0..65 {
            let command = runtime.queue.new_command_buffer();
            let encoder = command.new_compute_command_encoder();
            decoder.token(
                &encoder,
                weights,
                position,
                FeedbackTransition::ProjectedSigmoid,
            );
            decoder.gemv(
                &encoder,
                (&decoder.normalized, 0),
                (weights.buffer, weights.readout),
                (&decoder.logits, 0),
                WIDTH,
                VOCAB,
                0,
            );
            encoder.end_encoding();
            complete(command)?;
            let logits = unsafe {
                std::slice::from_raw_parts(decoder.logits.contents().cast::<u16>(), VOCAB as usize)
            };
            let maximum = logits
                .iter()
                .map(|&x| decode_half(x))
                .fold(f64::NEG_INFINITY, f64::max);
            let total: f64 = logits
                .iter()
                .map(|&x| (decode_half(x) - maximum).exp())
                .sum();
            let loss =
                total.ln() + maximum - decode_half(logits[labels[position as usize] as usize]);
            assert!(loss.is_finite());
            let error = (loss - f64::from(reference[position as usize])).abs();
            maximum_error = maximum_error.max(error);
            let decode_top1 = logits
                .iter()
                .enumerate()
                .max_by(|(_, left), (_, right)| {
                    decode_half(**left).total_cmp(&decode_half(**right))
                })
                .unwrap()
                .0;
            let row = &reference_logits
                [position as usize * VOCAB as usize..(position as usize + 1) * VOCAB as usize];
            let reference_top1 = row
                .iter()
                .enumerate()
                .max_by(|(_, left), (_, right)| {
                    decode_half(**left).total_cmp(&decode_half(**right))
                })
                .unwrap()
                .0;
            top1_matches += usize::from(decode_top1 == reference_top1);
            if decode_top1 != reference_top1 {
                maximum_missed_margin = maximum_missed_margin
                    .max(decode_half(row[reference_top1]) - decode_half(row[decode_top1]));
            }
        }
        assert!(maximum_error < 0.006, "maximum NLL error={maximum_error}");
        assert!(
            top1_matches >= 64 && maximum_missed_margin < 0.006,
            "top1_matches={top1_matches}/65, maximum_missed_margin={maximum_missed_margin}"
        );
        Ok(())
    })
}

/// Copy a seeded model into one resident parameter row.
fn resident_row(runtime: &Runtime, model: &CandidateWeights) -> Buffer {
    let tensors = model.tensors();
    let bytes: u64 = tensors.iter().map(|(_, tensor, _)| tensor.length()).sum();
    let row = runtime.buffer::<u8>(bytes as usize);
    let mut offset = 0usize;
    for (_, tensor, _) in tensors {
        unsafe {
            std::ptr::copy_nonoverlapping(
                tensor.contents().cast::<u8>(),
                row.contents().cast::<u8>().add(offset),
                tensor.length() as usize,
            );
        }
        offset += tensor.length() as usize;
    }
    row
}

fn digest(buffer: &BufferRef, bytes: usize) -> u64 {
    let data = unsafe { std::slice::from_raw_parts(buffer.contents().cast::<u8>(), bytes) };
    data.iter().fold(0xcbf2_9ce4_8422_2325, |hash, byte| {
        (hash ^ u64::from(*byte)).wrapping_mul(0x0100_0000_01b3)
    })
}

/// Bit-exact fingerprints of every multistream architecture through both the
/// single-token decoder and the chunked scorer. Recorded before the residual
/// stream code moved into `nn::hyper`; any change here is a numerical change.
#[test]
#[ignore = "slow full-device golden integration benchmark; run on demand via cargo test -- \
            --ignored or release gate"]
fn hyper_golden() -> Result<(), String> {
    autoreleasepool(|| {
        let runtime = Runtime::shared()?;
        let decoder = Decoder::new(&runtime)?;
        let buffers = Buffers::new(&runtime);
        let pipelines = Pipelines::new(&runtime)?;
        let tensorops = TensorOpsPipelines::new(&runtime)?;
        let pisa = Pisa::new(&runtime)?;
        let mut observed = Vec::new();
        for (architecture, seed) in [
            (ResidualArchitecture::Hc4, 3),
            (ResidualArchitecture::Mhc4, 7),
            (ResidualArchitecture::LoopedMhc4, 11),
            (ResidualArchitecture::DiffusionMhc4, 13),
        ] {
            let model = CandidateWeights::seeded_for(&runtime, Some(seed), architecture);
            let row = resident_row(&runtime, &model);
            let weights = model.row(&row)?;
            unsafe {
                *decoder.tokens.contents().cast::<u32>() = 17;
            }
            let command = runtime.queue.new_command_buffer();
            let encoder = command.new_compute_command_encoder();
            decoder.prepare_router(&encoder, weights);
            decoder.token(&encoder, weights, 0, FeedbackTransition::Identity);
            encoder.end_encoding();
            complete(command)?;
            let decode = digest(&decoder.normalized, WIDTH as usize * 2);
            let command = runtime.queue.new_command_buffer();
            objective_fused(command, &pipelines, &tensorops, &pisa, &buffers, weights)?;
            complete(command)?;
            let score = digest(&buffers.sequence_scores, BATCH as usize * 4);
            observed.push((architecture, decode, score));
        }
        for (architecture, decode, score) in &observed {
            eprintln!("hyper_golden {architecture:?} decode={decode:#018x} score={score:#018x}");
        }
        let expected: [(u64, u64); 4] = HYPER_GOLDEN;
        for ((architecture, decode, score), (want_decode, want_score)) in
            observed.iter().zip(expected)
        {
            assert_eq!(*decode, want_decode, "{architecture:?} decode changed");
            assert_eq!(*score, want_score, "{architecture:?} score changed");
        }
        Ok(())
    })
}

const HYPER_GOLDEN: [(u64, u64); 4] = [
    (0xccdc_adc2_89db_ee85, 0x0c36_d5b2_036f_c39e),
    (0x5eb7_8c78_f253_d7d4, 0x6a11_0b3c_1204_2957),
    (0x22c3_4cad_156e_5c1a, 0x74d6_e528_69f5_4fe7),
    (0x4c14_dbdc_d084_7408, 0x816e_9596_6d55_d1fd),
];
