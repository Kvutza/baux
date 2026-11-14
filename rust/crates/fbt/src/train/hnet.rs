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

    #[allow(clippy::too_many_arguments)]
    pub(super) fn inspect_moe(
        &self,
        runtime: &Runtime,
        pipelines: &Pipelines,
        buffers: &Buffers,
        step: architecture::LayerStep,
        rows: u32,
        output_grad: &BufferRef,
        source_grad: &BufferRef,
    ) -> Result<(), String> {
        let shape = mhc::Shape {
            rows,
            width: WIDTH,
            architecture: self.row.architecture.kernel_code(),
            reserved: 0,
        };
        let source = ((step.execution * 2 + 1) % 2) as usize;
        half_stage("replayed mHC streams", &buffers.mhc_streams[source])?;
        half_values(
            "replayed MoE branch",
            &buffers.output,
            rows as usize * WIDTH as usize,
        )?;
        check_stage("replayed mHC coefficients", &buffers.mhc_coefficients)?;
        check_stage("mHC update output adjoint", output_grad)?;
        let command = runtime.queue.new_command_buffer();
        zero(
            command,
            &[
                source_grad,
                &self.activations.branch,
                &self.activations.normalized,
                &self.activations.coefficient,
            ],
        );
        let encoder = command.new_compute_command_encoder();
        pipelines.mhc_grad()?.update(
            &encoder,
            shape,
            mhc::Update {
                streams: (&buffers.mhc_streams[source], 0),
                branch: (&buffers.output, 0),
                coefficients: (&buffers.mhc_coefficients, 0),
                output_grad: (output_grad, 0),
                stream_grad: (source_grad, 0),
                branch_grad: (&self.activations.branch, 0),
                coefficient_grad: (&self.activations.coefficient, 0),
            },
        );
        encoder.end_encoding();
        command.commit();
        complete_committed(command)?;
        check_stage("mHC update branch", &self.activations.branch)?;
        check_stage("mHC update stream", source_grad)?;
        check_stage("mHC update coefficients", &self.activations.coefficient)?;

        let command = runtime.queue.new_command_buffer();
        let (router, gate, down) = ffn_offsets(step.layer);
        pipelines.moe_grad()?.backward(
            command,
            &buffers.fine_grained,
            buffers.moe_grad()?,
            routing_grad::MoeGrad {
                input: (&buffers.normalized, 0),
                weights: &self.weights,
                router_offset: self.row.router + router,
                gate_offset: self.row.gate_up + gate,
                down_offset: self.row.down + down,
                branch_grad: (&self.activations.branch, 0),
                input_grad: (&self.activations.normalized, 0),
                router_grad: (&self.gradients.flat, grad_offset(self.row.router + router)),
                gate_grad: (&self.gradients.flat, grad_offset(self.row.gate_up + gate)),
                down_grad: (&self.gradients.flat, grad_offset(self.row.down + down)),
                rows,
            },
        )?;
        command.commit();
        complete_committed(command)?;
        check_stage("MoE branch reverse", &self.activations.normalized)?;

        let command = runtime.queue.new_command_buffer();
        let encoder = command.new_compute_command_encoder();
        self.reverse_input(
            &encoder,
            pipelines.mhc_grad()?,
            buffers,
            step,
            false,
            rows,
            source_grad,
        );
        encoder.end_encoding();
        command.commit();
        complete_committed(command)?;
        check_stage("mHC MoE input reverse", source_grad)
    }

    pub(super) fn inspect_replay(
        &self,
        runtime: &Runtime,
        verifier: &block_decode::BlockDecoder,
        rows: u32,
    ) -> Result<(), String> {
        check_stage("mHC mean reverse", &self.activations.stream[0])?;
        for step in self.row.architecture.layer_steps().iter().copied() {
            for (attention, target) in [
                (true, scorer::ReplaySite::Attention(step.execution)),
                (false, scorer::ReplaySite::Moe(step.execution)),
            ] {
                let replay = runtime.queue.new_command_buffer();
                scorer_replay::replay_site(
                    replay,
                    &verifier.pipelines,
                    &verifier.tensorops,
                    &verifier.pisa,
                    &verifier.buffers,
                    self.row(),
                    rows,
                    target,
                )?;
                replay.commit();
                complete_committed(replay)?;
                half_stage("replay input stream", &verifier.buffers.mhc_streams[0])?;
                check_stage("replay coefficients", &verifier.buffers.mhc_coefficients)?;
                half_stage("replay normalized", &verifier.buffers.normalized)?;
                let (label, buffer, stream) = if attention {
                    (
                        "replay attention projection",
                        &verifier.buffers.projected,
                        1,
                    )
                } else {
                    ("replay MoE output", &verifier.buffers.output, 0)
                };
                half_values(label, buffer, rows as usize * WIDTH as usize)?;
                half_stage(
                    &format!(
                        "{} replay execution {}",
                        if attention { "attention" } else { "MoE" },
                        step.execution
                    ),
                    &verifier.buffers.mhc_streams[stream],
                )?;
            }
        }
        Ok(())
    }
}

fn half_stage(name: &str, buffer: &BufferRef) -> Result<(), String> {
    half_values(name, buffer, buffer.length() as usize / size_of::<u16>())
}

fn half_values(name: &str, buffer: &BufferRef, elements: usize) -> Result<(), String> {
    if elements * size_of::<u16>() > buffer.length() as usize {
        return Err(format!("{name} diagnostic extent exceeds its buffer"));
    }
    let values = unsafe { std::slice::from_raw_parts(buffer.contents().cast::<u16>(), elements) };
    let invalid = values
        .iter()
        .filter(|value| **value & 0x7c00 == 0x7c00)
        .count();
    if invalid != 0 {
        return Err(format!("{name} contains {invalid} non-finite FP16 values"));
    }
    Ok(())
}

impl Optimizer {
    pub(in crate::moe) fn new(
        trainer: &Trainer,
        optimizer: crate::config::TrainingOptimizer,
    ) -> Result<Self, String> {
        use crate::muon::{MatrixSlice as Matrix, VectorSlice as Vector};
        let row = trainer.row;
        let weights = trainer.weights();
        let mut matrices = vec![
            Matrix::new(row.router, WIDTH, routing::ROUTED_EXPERTS, MODEL_LAYERS)?,
            Matrix::new(row.qkv, WIDTH, QKV_WIDTH, MODEL_LAYERS)?,
            Matrix::new(row.output, WIDTH, WIDTH, MODEL_LAYERS)?,
            Matrix::new(
                row.gate_up,
                WIDTH,
                routing::MOE_GATEUP,
                MODEL_LAYERS * routing::MOE_EXPERTS,
            )?,
            Matrix::new(
                row.down,
                routing::ROUTED_WIDTH,
                WIDTH,
                MODEL_LAYERS * routing::MOE_EXPERTS,
            )?,
            Matrix::new(
                row.mhc_attention_predictor,
                MHC_INPUT,
                MHC_COEFFICIENTS,
                MODEL_LAYERS,
            )?,
            Matrix::new(
                row.mhc_moe_predictor,
                MHC_INPUT,
                MHC_COEFFICIENTS,
                MODEL_LAYERS,
            )?,
        ];
        let mut vectors = vec![
            Vector::new(row.attention_norm, MODEL_LAYERS * WIDTH),
            Vector::new(row.ffn_norm, MODEL_LAYERS * WIDTH),
            Vector::new(row.readout, WIDTH * row.architecture.vocabulary()),
            Vector::new(row.mhc_attention_bias, MODEL_LAYERS * MHC_COEFFICIENTS),
            Vector::new(row.mhc_attention_control, MODEL_LAYERS * 4),
            Vector::new(row.mhc_moe_bias, MODEL_LAYERS * MHC_COEFFICIENTS),
            Vector::new(row.mhc_moe_control, MODEL_LAYERS * 4),
            Vector::new(row.final_norm, WIDTH),
        ];
        if row.architecture.is_hnet() {
            matrices.push(Matrix::new(row.hnet, WIDTH, WIDTH, 3)?);
            vectors.push(Vector::new(
                row.hnet + half_bytes(u64::from(3 * WIDTH * WIDTH)),
                WIDTH,
            ));
        }
        let inner = match optimizer {
            crate::config::TrainingOptimizer::MuonV1 => {
                crate::muon::Optimizer::new(weights, &matrices, &vectors, (WIDTH, QKV_WIDTH, 32))?
            }
            crate::config::TrainingOptimizer::AdamwV1 => {
                crate::muon::Optimizer::adamw(weights, &matrices, &vectors)?
            }
            _ => return Err(format!("Unsupported training optimizer: {:?}", optimizer)),
        };
        Ok(Self { inner })
    }

    pub(in crate::moe) fn muon(trainer: &Trainer) -> Result<Self, String> {
        Self::new(trainer, crate::config::TrainingOptimizer::MuonV1)
    }

    pub(in crate::moe) fn encode(
        &mut self,
        runtime: &Runtime,
        trainer: &Trainer,
    ) -> Result<f64, String> {
        let weights = trainer.weights();
        let gradients = trainer.gradients();
        let command = runtime.queue.new_command_buffer();
        self.inner.encode(command, weights, gradients)?;
        command.commit();
        complete_committed(command)
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
            crate::config::ModelInitialization::PisaMhc4,
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
    save_checkpoint(trainer.weights(), &path, trainer.row.architecture)?;
    let checkpoint = crate::tensor_store::SafeTensors::open(&path)?;
    assert_eq!(
        checkpoint.metadata().get("format").map(String::as_str),
        Some("baux.pisa-hnet-bytes-rope.v2")
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
