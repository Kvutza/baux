//! Exact reverse-mode training for the PISA/mHC/MoE program.

use super::mhc_grad as mhc;
use super::train_grad as dense;
use super::*;

struct Activations {
    stream: [Buffer; 2],
    branch: Buffer,
    normalized: Buffer,
    coefficient: Buffer,
    attention: Buffer,
    qkv: Buffer,
    input: Buffer,
    mask: Buffer,
    half_grad: Buffer,
}

struct Gradients {
    flat: Buffer,
}

pub(super) struct Trainer {
    weights: Buffer,
    row: Mhc4Layout,
    rows: u32,
    gradients: Gradients,
    activations: Activations,
}

pub(super) struct Optimizer {
    inner: crate::muon::Optimizer,
}

fn zero(command: &CommandBufferRef, buffers: &[&BufferRef]) {
    let encoder = command.new_blit_command_encoder();
    for buffer in buffers {
        encoder.fill_buffer(buffer, metal::NSRange::new(0, buffer.length()), 0);
    }
    encoder.end_encoding();
}

fn grad_offset(offset: u64) -> u64 {
    offset * 2
}

fn check_stage(name: &str, buffer: &BufferRef) -> Result<(), String> {
    let values = unsafe {
        std::slice::from_raw_parts(
            buffer.contents().cast::<f32>(),
            buffer.length() as usize / size_of::<f32>(),
        )
    };
    let invalid = values.iter().filter(|value| !value.is_finite()).count();
    if invalid != 0 {
        return Err(format!("{name} produced {invalid} non-finite adjoints"));
    }
    Ok(())
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

impl Gradients {
    fn new(runtime: &Runtime, flat: &BufferRef) -> Self {
        let buffer = runtime.buffer::<f32>((flat.length() / 2) as usize);
        Self { flat: buffer }
    }

    fn clear(&self, command: &CommandBufferRef) {
        zero(command, &[&self.flat]);
    }
}

impl Activations {
    fn new(runtime: &Runtime, rows: u32) -> Self {
        let rows = rows as usize;
        let width = WIDTH as usize;
        Self {
            stream: [
                runtime.buffer::<f32>(rows * MHC_STREAMS as usize * width),
                runtime.buffer::<f32>(rows * MHC_STREAMS as usize * width),
            ],
            branch: runtime.buffer::<f32>(rows * width),
            normalized: runtime.buffer::<f32>(rows * width),
            coefficient: runtime.buffer::<f32>(rows * MHC_COEFFICIENTS as usize),
            attention: runtime.buffer::<f32>(rows * WIDTH as usize),
            qkv: runtime.buffer::<f32>(rows * QKV_WIDTH as usize),
            input: runtime.buffer::<f32>(rows * width),
            mask: runtime.buffer_with(&vec![1u8; crate::context::MAX_TOKENS as usize]),
            half_grad: runtime.buffer::<u16>(rows * QKV_WIDTH as usize),
        }
    }
}

impl Trainer {
    pub(super) fn new(
        runtime: &Runtime,
        initial: &CandidateWeights,
        rows: u32,
    ) -> Result<Self, String> {
        if rows == 0 || rows % 128 != 0 {
            return Err(format!(
                "training requires positive 128-aligned coarse rows; found {rows}"
            ));
        }
        let weights = initial.flatten(runtime)?;
        let row = initial.row(&weights)?.layout;
        Ok(Self {
            gradients: Gradients::new(runtime, &weights),
            activations: Activations::new(runtime, rows),
            weights,
            row,
            rows,
        })
    }

    pub(super) fn row(&self) -> CandidateRow<'_> {
        self.row.view(&self.weights)
    }

    pub(super) fn weights(&self) -> &BufferRef {
        &self.weights
    }

    pub(super) fn buffer(&self) -> &Buffer {
        &self.weights
    }

    pub(super) fn gradients(&self) -> &BufferRef {
        &self.gradients.flat
    }

    fn mhc_offsets(&self, layer: u32, attention: bool) -> (u64, u64, u64, u64) {
        let row = self.row();
        let predictor = if attention {
            row.mhc_attention_predictor
        } else {
            row.mhc_moe_predictor
        };
        let bias = if attention {
            row.mhc_attention_bias
        } else {
            row.mhc_moe_bias
        };
        let control = if attention {
            row.mhc_attention_control
        } else {
            row.mhc_moe_control
        };
        let norm = if attention {
            row.attention_norm
        } else {
            row.ffn_norm
        };
        (
            predictor + half_bytes(u64::from(layer * MHC_INPUT * MHC_COEFFICIENTS)),
            bias + half_bytes(u64::from(layer * MHC_COEFFICIENTS)),
            control + half_bytes(u64::from(layer * 4)),
            norm + half_bytes(u64::from(layer * WIDTH)),
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn reverse_input(
        &self,
        encoder: &ComputeCommandEncoderRef,
        kernels: &mhc::Kernels,
        buffers: &Buffers,
        step: architecture::LayerStep,
        attention: bool,
        rows: u32,
        source_grad: &BufferRef,
    ) {
        let site = step.execution * 2 + u32::from(!attention);
        let source = (site % 2) as usize;
        let (predictor, bias, control, norm) = self.mhc_offsets(step.layer, attention);
        let coefficient_offset = 0;
        let shape = mhc::Shape {
            rows,
            width: WIDTH,
            architecture: self.row.architecture.kernel_code(),
            reserved: 0,
        };
        kernels.mix(
            encoder,
            shape,
            mhc::Mix {
                streams: (&buffers.mhc_streams[source], 0),
                coefficients: (&buffers.mhc_coefficients, coefficient_offset),
                weight: (&self.weights, norm),
                output_grad: (&self.activations.normalized, 0),
                stream_grad: (source_grad, 0),
                coefficient_grad: (&self.activations.coefficient, 0),
                weight_grad: (&self.gradients.flat, grad_offset(norm)),
            },
        );
        kernels.predict(
            encoder,
            shape,
            mhc::Predict {
                streams: (&buffers.mhc_streams[source], 0),
                predictor: (&self.weights, predictor),
                bias: (&self.weights, bias),
                control: (&self.weights, control),
                coefficient_grad: (&self.activations.coefficient, 0),
                stream_grad: (source_grad, 0),
                predictor_grad: (&self.gradients.flat, grad_offset(predictor)),
                bias_grad: (&self.gradients.flat, grad_offset(bias)),
                control_grad: (&self.gradients.flat, grad_offset(control)),
            },
        );
    }

    fn reverse_moe(
        &self,
        command: &CommandBufferRef,
        pipelines: &Pipelines,
        buffers: &Buffers,
        step: architecture::LayerStep,
        rows: u32,
        output_grad: &BufferRef,
        source_grad: &BufferRef,
    ) -> Result<(), String> {
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
        let source = ((step.execution * 2 + 1) % 2) as usize;
        let shape = mhc::Shape {
            rows,
            width: WIDTH,
            architecture: self.row.architecture.kernel_code(),
            reserved: 0,
        };
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
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn inspect_moe(
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

    fn reverse_attention(
        &self,
        command: &CommandBufferRef,
        pipelines: &Pipelines,
        pisa: &Pisa1,
        buffers: &Buffers,
        step: architecture::LayerStep,
        rows: u32,
        output_grad: &BufferRef,
        source_grad: &BufferRef,
    ) -> Result<(), String> {
        zero(
            command,
            &[
                source_grad,
                &self.activations.branch,
                &self.activations.normalized,
                &self.activations.coefficient,
                &self.activations.attention,
                &self.activations.qkv,
            ],
        );
        let source = ((step.execution * 2) % 2) as usize;
        let shape = mhc::Shape {
            rows,
            width: WIDTH,
            architecture: self.row.architecture.kernel_code(),
            reserved: 0,
        };
        let encoder = command.new_compute_command_encoder();
        pipelines.mhc_grad()?.update(
            &encoder,
            shape,
            mhc::Update {
                streams: (&buffers.mhc_streams[source], 0),
                branch: (&buffers.projected, 0),
                coefficients: (&buffers.mhc_coefficients, 0),
                output_grad: (output_grad, 0),
                stream_grad: (source_grad, 0),
                branch_grad: (&self.activations.branch, 0),
                coefficient_grad: (&self.activations.coefficient, 0),
            },
        );
        let (_, output_offset) = scorer::Scorer::attention_offsets(step.layer);
        pipelines.train_grad()?.dense(
            &encoder,
            dense::DenseShape {
                rows,
                input: WIDTH,
                output: WIDTH,
                reserved: 0,
            },
            dense::DenseGrad {
                input: (pisa.output(), 0),
                weights: (&self.weights, self.row.output + output_offset),
                output_grad: (&self.activations.branch, 0),
                input_grad: (&self.activations.attention, 0),
                weight_grad: (
                    &self.gradients.flat,
                    grad_offset(self.row.output + output_offset),
                ),
                half_grad: &self.activations.half_grad,
            },
        );
        pisa.backward(
            &encoder,
            (&buffers.qkv, 0),
            (&buffers.qkv, 0),
            (&self.activations.attention, 0),
            (&self.activations.qkv, 0),
            (&self.activations.qkv, 0),
            0,
            rows,
            QKV_WIDTH,
            QKV_WIDTH,
        )?;
        pipelines.train_grad()?.rope(
            &encoder,
            (&self.activations.qkv, 0),
            &buffers.rope,
            rows,
            0,
            pisa.context(),
        );
        let (qkv_offset, _) = scorer::Scorer::attention_offsets(step.layer);
        pipelines.train_grad()?.dense(
            &encoder,
            dense::DenseShape {
                rows,
                input: WIDTH,
                output: QKV_WIDTH,
                reserved: 0,
            },
            dense::DenseGrad {
                input: (&buffers.normalized, 0),
                weights: (&self.weights, self.row.qkv + qkv_offset),
                output_grad: (&self.activations.qkv, 0),
                input_grad: (&self.activations.normalized, 0),
                weight_grad: (&self.gradients.flat, grad_offset(self.row.qkv + qkv_offset)),
                half_grad: &self.activations.half_grad,
            },
        );
        self.reverse_input(
            &encoder,
            pipelines.mhc_grad()?,
            buffers,
            step,
            true,
            rows,
            source_grad,
        );
        encoder.end_encoding();
        Ok(())
    }

    fn inspect_replay(
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

    fn reverse_embed(
        &self,
        command: &CommandBufferRef,
        verifier: &block_decode::BlockDecoder,
        rows: u32,
    ) -> Result<(), String> {
        let encoder = command.new_compute_command_encoder();
        verifier.pipelines.train_grad()?.replicate(
            &encoder,
            dense::DenseShape {
                rows,
                input: WIDTH,
                output: MHC_STREAMS,
                reserved: 0,
            },
            (&self.activations.stream[0], 0),
            (&self.activations.input, 0),
        );
        verifier.pipelines.train_grad()?.embed(
            &encoder,
            dense::DenseShape {
                rows,
                input: WIDTH,
                output: VOCAB,
                reserved: self.row.architecture.patch_size() as u32,
            },
            dense::EmbedGrad {
                readout: (&self.weights, self.row.readout),
                tokens: (&verifier.buffers.tokens, 0),
                positions: (&self.weights, self.row.index_query),
                output_grad: (&self.activations.input, 0),
                readout_grad: (&self.gradients.flat, grad_offset(self.row.readout)),
                position_grad: (&self.gradients.flat, grad_offset(self.row.index_query)),
                mask_grad: (&self.gradients.flat, grad_offset(self.row.mask_embed)),
            },
        );
        encoder.end_encoding();
        Ok(())
    }

    fn reverse_readout(
        &self,
        command: &CommandBufferRef,
        verifier: &block_decode::BlockDecoder,
        task: &crate::config::GenerationTask,
        tokens: u32,
    ) -> Result<(), String> {
        self.gradients.clear(command);
        let act = &self.activations;
        zero(
            command,
            &[
                &act.stream[0],
                &act.stream[1],
                &act.branch,
                &act.normalized,
                &act.coefficient,
                &act.attention,
                &act.qkv,
                &act.input,
            ],
        );
        let proposal_start =
            u32::try_from(task.prompt.len().saturating_sub(1)).map_err(|e| e.to_string())?;
        let encoder = command.new_compute_command_encoder();
        verifier.pipelines.train_grad()?.readout(
            &encoder,
            dense::ReadoutShape {
                rows: tokens,
                width: WIDTH,
                vocab: VOCAB,
                start: proposal_start,
                scale: 1.0 / tokens as f32,
            },
            dense::ReadoutGrad {
                hidden: (&verifier.buffers.normalized, 0),
                readout: (&self.weights, self.row.readout),
                expected: (&verifier.expected, 0),
                mask: (&act.mask, 0),
                hidden_grad: (&act.normalized, 0),
                readout_grad: (&self.gradients.flat, grad_offset(self.row.readout)),
            },
        );
        encoder.end_encoding();
        Ok(())
    }

    pub(super) fn backward(
        &self,
        runtime: &Runtime,
        verifier: &block_decode::BlockDecoder,
        task: &crate::config::GenerationTask,
        tokens: u32,
    ) -> Result<f64, String> {
        if self.row.architecture != ResidualArchitecture::Mhc4 {
            return Err("Muon training currently requires the native MHC4 architecture".into());
        }
        if tokens == 0 || task.prompt.len() < 64 || task.expected.len() < tokens as usize {
            return Err("invalid mHC4 training trajectory".into());
        }
        let active_rows = u32::try_from(
            (task.prompt.len() + tokens as usize - 1).div_ceil(self.row.architecture.patch_size()),
        )
        .map_err(|error| error.to_string())?;
        let rows = verifier.context;
        if active_rows > rows {
            return Err("training trajectory exceeds the verifier context".into());
        }
        if rows == 0 || rows > self.rows || rows % 128 != 0 {
            return Err(format!(
                "training requires 128-aligned coarse rows within {}; found {rows}",
                self.rows
            ));
        }

        let mut command = runtime.queue.new_command_buffer();
        self.reverse_readout(command, verifier, task, tokens)?;

        let inspect = std::env::var_os("BAUX_UPDATE_ALIGNMENT_STAGES").is_some();
        if inspect {
            command.commit();
            complete_committed(command)?;
            command = runtime.queue.new_command_buffer();
        }

        let shape = mhc::Shape {
            rows,
            width: WIDTH,
            architecture: self.row.architecture.kernel_code(),
            reserved: 0,
        };
        let encoder = command.new_compute_command_encoder();
        verifier.pipelines.mhc_grad()?.mean(
            &encoder,
            shape,
            mhc::Mean {
                streams: (&verifier.buffers.mhc_streams[0], 0),
                weight: (&self.weights, self.row.final_norm),
                output_grad: (&self.activations.normalized, 0),
                stream_grad: (&self.activations.stream[0], 0),
                weight_grad: (&self.gradients.flat, grad_offset(self.row.final_norm)),
            },
        );
        encoder.end_encoding();
        if inspect {
            command.commit();
            complete_committed(command)?;
            self.inspect_replay(runtime, verifier, rows)?;
            command = runtime.queue.new_command_buffer();
        }

        for step in self.row.architecture.layer_steps().iter().rev().copied() {
            scorer_replay::replay_site(
                command,
                &verifier.pipelines,
                &verifier.tensorops,
                &verifier.pisa,
                &verifier.buffers,
                self.row(),
                rows,
                scorer::ReplaySite::Moe(step.execution),
            )?;
            let output = ((step.execution * 2 + 2) % 2) as usize;
            let source = 1 - output;
            if inspect {
                command.commit();
                complete_committed(command)?;
                self.inspect_moe(
                    runtime,
                    &verifier.pipelines,
                    &verifier.buffers,
                    step,
                    rows,
                    &self.activations.stream[output],
                    &self.activations.stream[source],
                )?;
                command = runtime.queue.new_command_buffer();
            } else {
                self.reverse_moe(
                    command,
                    &verifier.pipelines,
                    &verifier.buffers,
                    step,
                    rows,
                    &self.activations.stream[output],
                    &self.activations.stream[source],
                )?;
            }

            scorer_replay::replay_site(
                command,
                &verifier.pipelines,
                &verifier.tensorops,
                &verifier.pisa,
                &verifier.buffers,
                self.row(),
                rows,
                scorer::ReplaySite::Attention(step.execution),
            )?;
            let output = ((step.execution * 2 + 1) % 2) as usize;
            let source = 1 - output;
            self.reverse_attention(
                command,
                &verifier.pipelines,
                &verifier.pisa,
                &verifier.buffers,
                step,
                rows,
                &self.activations.stream[output],
                &self.activations.stream[source],
            )?;
            if inspect {
                command.commit();
                complete_committed(command)?;
                check_stage("attention reverse", &self.activations.stream[source])?;
                command = runtime.queue.new_command_buffer();
            }
        }

        self.reverse_embed(command, verifier, rows)?;
        command.commit();
        let seconds = complete_committed(command)?;
        Ok(seconds)
    }
}

impl Optimizer {
    pub(super) fn new(
        trainer: &Trainer,
        optimizer: crate::config::TrainingOptimizer,
    ) -> Result<Self, String> {
        use crate::muon::{MatrixSlice as Matrix, VectorSlice as Vector};
        let row = trainer.row;
        let weights = trainer.weights();
        let matrices = [
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
        let vectors = [
            Vector::new(row.attention_norm, MODEL_LAYERS * WIDTH),
            Vector::new(row.ffn_norm, MODEL_LAYERS * WIDTH),
            Vector::new(row.readout, WIDTH * VOCAB),
            Vector::new(row.mhc_attention_bias, MODEL_LAYERS * MHC_COEFFICIENTS),
            Vector::new(row.mhc_attention_control, MODEL_LAYERS * 4),
            Vector::new(row.mhc_moe_bias, MODEL_LAYERS * MHC_COEFFICIENTS),
            Vector::new(row.mhc_moe_control, MODEL_LAYERS * 4),
            Vector::new(row.final_norm, WIDTH),
        ];
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

    pub(super) fn muon(trainer: &Trainer) -> Result<Self, String> {
        Self::new(trainer, crate::config::TrainingOptimizer::MuonV1)
    }

    pub(super) fn encode(&mut self, runtime: &Runtime, trainer: &Trainer) -> Result<f64, String> {
        let weights = trainer.weights();
        let gradients = trainer.gradients();
        let command = runtime.queue.new_command_buffer();
        self.inner.encode(command, weights, gradients)?;
        command.commit();
        complete_committed(command)
    }
}
