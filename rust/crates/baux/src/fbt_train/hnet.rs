//! Adjoint at the byte/chunk boundaries of the PISA/mHC hierarchy.
use super::*;
use crate::apple_gpu::{bind_buffers, encode_value};

pub(super) struct Adjoint {
    combined: Buffer,
    probability: Buffer,
    projected: Buffer,
}

impl Adjoint {
    pub(super) fn new(runtime: &Runtime, rows: u32) -> Self {
        let count = (rows * WIDTH) as usize;
        Self {
            combined: runtime.buffer::<f32>(count),
            probability: runtime.buffer::<f32>(count),
            projected: runtime.buffer::<f32>(2 * count),
        }
    }
}

impl Trainer {
    pub(super) fn reverse_hnet(
        &self,
        command: &CommandBufferRef,
        scorer: &block_decode::BlockDecoder,
        rows: u32,
        chunk: bool,
    ) -> Result<(), String> {
        let state = self.hnet.as_ref().ok_or("missing H-Net adjoint")?;
        let saved = scorer.buffers.hnet()?;
        let waves = &saved.waves;
        let shape = baux::semantics::hnet::HNetShape {
            batch_size: 1,
            seq_len: rows,
            d_model: WIDTH,
            max_chunks: rows,
            vocab_size: 256,
            eps: 1e-4,
            target_ratio: 2.0,
            score: 1,
        };
        let encoder = command.new_compute_command_encoder();
        if !chunk {
            encoder.set_compute_pipeline_state(&waves.dechunk_grad);
            bind_buffers(
                &encoder,
                &[
                    (&self.activations.stream[0], 0),
                    (&saved.core, 0),
                    (&saved.chunk_scratch, 0),
                    (&saved.probs, 0),
                    (&saved.plug_idx, 0),
                    (&saved.origins, 0),
                    (&saved.count, 0),
                    (&state.combined, 0),
                    (&state.probability, 0),
                    (&self.activations.normalized, 0),
                ],
            );
            encode_value(&encoder, 10, &shape);
            encoder.dispatch_threads(thread_group(u64::from(WIDTH)), thread_group(256));
            encoder.end_encoding();
            return Ok(());
        }
        encoder.set_compute_pipeline_state(&waves.route_grad);
        bind_buffers(
            &encoder,
            &[
                (&saved.projected, 0),
                (&state.probability, 0),
                (&state.projected, 0),
            ],
        );
        encode_value(&encoder, 3, &shape);
        encoder.dispatch_thread_groups(thread_group(u64::from(rows)), thread_group(32));
        encoder.memory_barrier_with_resources(&[&state.projected]);
        encoder.set_compute_pipeline_state(&waves.input_grad);
        bind_buffers(
            &encoder,
            &[
                (&self.weights, self.row.hnet),
                (&state.projected, 0),
                (&state.combined, 0),
                (&self.activations.stream[0], 0),
                (&saved.probs, 0),
                (&saved.plug_idx, 0),
                (&self.activations.normalized, 0),
            ],
        );
        encode_value(&encoder, 7, &shape);
        encoder.dispatch_threads(thread_group(u64::from(rows * WIDTH)), thread_group(256));
        encoder.set_compute_pipeline_state(&waves.weight_grad);
        bind_buffers(
            &encoder,
            &[
                (&saved.encoder_streams, 0),
                (&state.projected, 0),
                (&state.combined, 0),
                (&self.gradients.flat, grad_offset(self.row.hnet)),
            ],
        );
        encode_value(&encoder, 4, &shape);
        encoder.dispatch_threads(
            thread_group(u64::from(3 * WIDTH * WIDTH + WIDTH)),
            thread_group(256),
        );
        encoder.end_encoding();
        Ok(())
    }

    pub(super) fn reverse_boundary(
        &self,
        command: &CommandBufferRef,
        scorer: &block_decode::BlockDecoder,
        rows: u32,
    ) -> Result<(), String> {
        zero(command, &[&self.activations.stream[0]]);
        let encoder = command.new_compute_command_encoder();
        scorer.pipelines.mhc_grad()?.mean(
            &encoder,
            mhc::Shape {
                rows,
                width: WIDTH,
                architecture: self.row.architecture.kernel_code(),
                reserved: 0,
            },
            mhc::Mean {
                streams: (&scorer.buffers.mhc_streams[0], 0),
                weight: (&self.weights, self.row.final_norm),
                output_grad: (&self.activations.normalized, 0),
                stream_grad: (&self.activations.stream[0], 0),
                weight_grad: (&self.gradients.flat, grad_offset(self.row.final_norm)),
            },
        );
        encoder.end_encoding();
        Ok(())
    }
}

#[cfg(test)]
#[test]
#[ignore = "full PISA/mHC byte hierarchy forward, replay, and backward"]
fn integrated() -> Result<(), String> {
    metal::objc::rc::autoreleasepool(|| {
        let runtime = Runtime::shared()?;
        let rows = CONTEXT;
        let architecture = ResidualArchitecture::HnetMhc4;
        let initial = CandidateWeights::initialized(
            &runtime,
            17,
            crate::config::ModelInitialization::FbtPisa1Mhc4,
            architecture,
        )?;
        assert_eq!(initial.readout.length(), u64::from(WIDTH * 256 * 2));
        let trainer = Trainer::new(&runtime, &initial, rows)?;
        drop(initial);
        let decoder = decode::Decoder::with_visits(
            &runtime,
            CONTEXT,
            false,
            architecture.layer_steps().len(),
        )?;
        let scorer = block_decode::BlockDecoder::with_architecture(&runtime, rows, architecture)?;
        let tokens: Vec<u32> = (0..=rows).map(|i| (i * 37 + 17) % 254).collect();
        let mut batch = crate::pretrain_data::CausalBatch {
            tokens: tokens[..rows as usize].to_vec(),
            labels: tokens[1..].to_vec(),
            loss_mask: vec![1; rows as usize],
            offset: 0,
        };
        let before = scorer.score_batch(&runtime, &decoder, trainer.row(), &batch)?;
        let original =
            crate::apple_gpu::buffer_slice::<f32>(&scorer.buffers.losses, rows as usize)?.to_vec();
        let chunks = crate::apple_gpu::buffer_slice::<u32>(&scorer.buffers.hnet()?.count, 1)?[0];
        for value in &mut batch.tokens[64..] {
            *value = (*value + 101) % 254;
        }
        scorer.score_batch(&runtime, &decoder, trainer.row(), &batch)?;
        let changed = crate::apple_gpu::buffer_slice::<f32>(&scorer.buffers.losses, rows as usize)?;
        for i in 0..64 {
            assert!(
                (original[i] - changed[i]).abs() < 1e-5,
                "future input changed loss {i}: {} -> {}",
                original[i],
                changed[i]
            );
        }
        batch.tokens.copy_from_slice(&tokens[..rows as usize]);
        let (score, seconds) = trainer.backward_corpus(&runtime, &decoder, &scorer, &batch)?;
        assert_eq!(
            before.quality.mean_nll.to_bits(),
            score.quality.mean_nll.to_bits()
        );
        check_stage("H-Net complete backward", trainer.gradients())?;
        let gradients = crate::apple_gpu::buffer_slice::<f32>(
            trainer.gradients(),
            trainer.gradients().length() as usize / 4,
        )?;
        for (name, offset, count) in [
            ("byte readout", trainer.row.readout, WIDTH * 256),
            ("boundary projections", trainer.row.hnet, 2 * WIDTH * WIDTH),
            (
                "residual projection",
                trainer.row.hnet + half_bytes(u64::from(2 * WIDTH * WIDTH)),
                WIDTH * WIDTH + WIDTH,
            ),
            (
                "core MoE",
                trainer.row.gate_up
                    + half_bytes(u64::from(
                        WIDTH * routing::MOE_GATEUP * routing::MOE_EXPERTS,
                    )),
                WIDTH * routing::MOE_GATEUP * routing::MOE_EXPERTS,
            ),
        ] {
            let start = offset as usize / 2;
            assert!(
                gradients[start..start + count as usize]
                    .iter()
                    .any(|&value| value != 0.0),
                "no gradient reached {name}"
            );
        }
        println!(
            "HNET_INTEGRATED rows={rows} chunks={chunks} vocabulary=256 causal=true backward_seconds={seconds} nll={}",
            score.quality.mean_nll
        );
        let hnet_offset = trainer.row.hnet as usize / 2;
        let weight_count = trainer.weights().length() as usize / 2;
        let before_weights =
            crate::apple_gpu::buffer_slice::<u16>(trainer.weights(), weight_count)?[hnet_offset..]
                .to_vec();
        let mut optimizer = Optimizer::new(&trainer, crate::config::TrainingOptimizer::MuonV1)?;
        let update_seconds = optimizer.encode(&runtime, &trainer)?;
        drop(optimizer);
        let after_weights =
            &crate::apple_gpu::buffer_slice::<u16>(trainer.weights(), weight_count)?[hnet_offset..];
        assert_ne!(
            before_weights, after_weights,
            "optimizer did not update the hierarchy"
        );
        let after = scorer.score_batch(&runtime, &decoder, trainer.row(), &batch)?;
        assert!(after.quality.mean_nll.is_finite());
        println!(
            "HNET_UPDATE seconds={update_seconds} before={} after={}",
            score.quality.mean_nll, after.quality.mean_nll
        );
        checkpoint(&trainer)?;
        Ok(())
    })
}

#[cfg(test)]
fn checkpoint(trainer: &Trainer) -> Result<(), String> {
    let path = std::env::temp_dir().join(format!("baux-hnet-{}.safetensors", std::process::id()));
    gen_checkpoint::save_checkpoint(trainer.weights(), &path, trainer.row.architecture)?;
    let checkpoint = crate::tensor_store::SafeTensors::open(&path)?;
    assert_eq!(
        checkpoint.metadata().get("format").map(String::as_str),
        Some("baux.fbt-pisa1-hnet-bytes-rope.v2")
    );
    for (name, offset, count) in [
        ("hnet", trainer.row.hnet, 3 * WIDTH * WIDTH + WIDTH),
        ("embedding_readout", trainer.row.readout, WIDTH * 256),
    ] {
        let (tensor, bytes) = checkpoint
            .tensor(name)
            .ok_or("checkpoint is missing byte parameters")?;
        assert_eq!(tensor.shape.iter().product::<usize>(), count as usize);
        let saved: Vec<_> = bytes
            .chunks_exact(2)
            .map(|bytes| u16::from_le_bytes([bytes[0], bytes[1]]))
            .collect();
        let start = offset as usize / 2;
        let source =
            crate::apple_gpu::buffer_slice::<u16>(trainer.weights(), start + count as usize)?;
        assert_eq!(saved, source[start..]);
    }
    drop(checkpoint);
    std::fs::remove_file(path).map_err(|error| error.to_string())?;
    println!("HNET_CHECKPOINT byte_head=true hierarchy=true format=v2");
    Ok(())
}
