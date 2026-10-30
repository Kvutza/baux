use super::*;

impl Trainer {
    /// Score and differentiate the same teacher-forced corpus batch. Keeping
    /// the forward here prevents intervening validation or speculative replay
    /// from replacing the activations consumed by the readout/final norm.
    pub(in crate::fbt_moe) fn backward_corpus(
        &self,
        runtime: &Runtime,
        decoder: &decode::Decoder,
        scorer: &block_decode::BlockDecoder,
        batch: &crate::pretrain_data::CausalBatch,
    ) -> Result<(block_decode::CausalScore, f64), String> {
        let rows = u32::try_from(batch.tokens.len()).map_err(|error| error.to_string())?;
        if self.row.architecture != ResidualArchitecture::Mhc4 {
            return Err("corpus gradients require MHC4".into());
        }
        // The existing replay consumes a complete, unchunked activation window.
        // Cached long-context replay needs a separate adjoint qualification.
        if rows == 0
            || rows % 128 != 0
            || rows > self.rows
            || rows != scorer.context
            || rows > ROWS
            || decoder.cache()[0].kv.is_some()
        {
            return Err("corpus gradients require one complete unchunked context".into());
        }
        if batch.labels.len() != rows as usize
            || batch.loss_mask.len() != rows as usize
            || batch.loss_mask.iter().any(|&mask| mask != 1)
            || batch
                .tokens
                .iter()
                .chain(&batch.labels)
                .any(|&token| token >= VOCAB)
        {
            return Err("corpus gradients require valid tokens and fully scored labels".into());
        }
        if self.readout.get().is_none() {
            self.readout
                .set(readout::Adjoint::new(runtime, self.rows)?)
                .map_err(|_| "readout adjoint was already initialized")?;
        }
        let adjoint = self.readout.get().ok_or("missing readout adjoint")?;
        scorer.clear_pair();
        let score = scorer.score_batch(runtime, decoder, self.row(), batch)?;
        let command = runtime.queue.new_command_buffer();
        self.clear(command);
        adjoint.encode(
            command,
            &scorer.waves,
            rows,
            readout::ReadoutGrad {
                hidden: (&scorer.buffers.normalized, 0),
                qkv: (&self.weights, self.row.qkv),
                readout: (&self.weights, self.row.readout),
                targets: (&scorer.buffers.labels, 0),
                hidden_grad: (&self.activations.normalized, 0),
                qkv_grad: (&self.gradients.flat, grad_offset(self.row.qkv)),
                readout_grad: (&self.gradients.flat, grad_offset(self.row.readout)),
            },
        )?;
        let seconds = self.reverse_core(runtime, scorer, rows, command)?;
        Ok((score, seconds))
    }
}

#[cfg(test)]
#[test]
#[ignore = "full 1.18B-parameter corpus forward/replay/backward qualification"]
fn corpus_replay() -> Result<(), String> {
    metal::objc::rc::autoreleasepool(|| {
        let runtime = Runtime::shared()?;
        let architecture = ResidualArchitecture::Mhc4;
        let initial = CandidateWeights::initialized(
            &runtime,
            17,
            crate::config::ModelInitialization::FbtPisa1Mhc4,
            architecture,
        )?;
        let trainer = Trainer::new(&runtime, &initial, CONTEXT)?;
        drop(initial);
        let decoder = decode::Decoder::with_visits(
            &runtime,
            CONTEXT,
            false,
            architecture.layer_steps().len(),
        )?;
        let scorer =
            block_decode::BlockDecoder::with_architecture(&runtime, CONTEXT, architecture)?;
        let sequence: Vec<u32> = (0..=CONTEXT).map(|i| (i * 4099 + 17) % VOCAB).collect();
        let batch = crate::pretrain_data::CausalBatch {
            tokens: sequence[..CONTEXT as usize].to_vec(),
            labels: sequence[1..].to_vec(),
            loss_mask: vec![1; CONTEXT as usize],
            offset: 0,
        };
        let before = scorer.score_batch(&runtime, &decoder, trainer.row(), &batch)?;
        let count = CONTEXT as usize * MHC_STREAMS as usize * WIDTH as usize;
        let streams =
            crate::apple_gpu::buffer_slice::<u16>(&scorer.buffers.mhc_streams[0], count)?.to_vec();
        let last = architecture
            .layer_steps()
            .last()
            .ok_or("empty recurrent schedule")?;
        let command = runtime.queue.new_command_buffer();
        scorer_replay::replay_site(
            command,
            &scorer.pipelines,
            &scorer.tensorops,
            &scorer.pisa,
            &scorer.buffers,
            trainer.row(),
            CONTEXT,
            scorer::ReplaySite::Moe(last.execution),
        )?;
        complete(command)?;
        let replayed =
            crate::apple_gpu::buffer_slice::<u16>(&scorer.buffers.mhc_streams[0], count)?;
        let mismatches = replayed
            .iter()
            .zip(&streams)
            .filter(|(a, b)| a != b)
            .count();
        assert_eq!(mismatches, 0, "forward/replay stream words differ");
        let (score, seconds) = trainer.backward_corpus(&runtime, &decoder, &scorer, &batch)?;
        assert_eq!(
            score.quality.mean_nll.to_bits(),
            before.quality.mean_nll.to_bits()
        );
        assert_eq!(score.quality.scored_tokens, CONTEXT as usize);
        check_stage("corpus backward", trainer.gradients())?;
        let gradients = crate::apple_gpu::buffer_slice::<f32>(
            trainer.gradients(),
            trainer.gradients().length() as usize / 4,
        )?;
        let nonzero = gradients.iter().filter(|&&value| value != 0.0).count();
        assert!(nonzero > 0);
        println!(
            "CORPUS_REPLAY rows={CONTEXT} stream_words={count} exact=true nll={} nonzero_gradients={nonzero} backward_seconds={seconds} full_gradient_parity=false",
            score.quality.mean_nll
        );
        if std::env::var_os("BAUX_CORPUS_DIRECTIONS").is_some() {
            directions(&runtime, &trainer, &decoder, &scorer, &batch)?;
        }
        Ok(())
    })
}

/// Diagnostic finite secants of the rounded, routed production function.
/// These are evidence about the chosen straight-through adjoint, not exact
/// derivatives of a program with quantization and discrete route selection.
#[cfg(test)]
fn directions(
    runtime: &Runtime,
    trainer: &Trainer,
    decoder: &decode::Decoder,
    scorer: &block_decode::BlockDecoder,
    batch: &crate::pretrain_data::CausalBatch,
) -> Result<(), String> {
    scorer.score_batch(runtime, decoder, trainer.row(), batch)?;
    let selection = Selection::read(scorer)?;
    let row = trainer.row;
    let families = [
        ("final_norm", row.final_norm, WIDTH),
        ("readout", row.readout, VOCAB),
        ("query", row.qkv, WIDTH * QKV_WIDTH),
        ("output", row.output, WIDTH * WIDTH),
        ("attention_norm", row.attention_norm, WIDTH),
        ("ffn_norm", row.ffn_norm, WIDTH),
        ("router", row.router, WIDTH * routing::ROUTED_EXPERTS),
        ("shared_gate", row.gate_up, WIDTH * routing::MOE_GATEUP),
        ("shared_down", row.down, routing::ROUTED_WIDTH * WIDTH),
        (
            "attention_predictor",
            row.mhc_attention_predictor,
            MHC_INPUT * MHC_COEFFICIENTS,
        ),
        (
            "moe_predictor",
            row.mhc_moe_predictor,
            MHC_INPUT * MHC_COEFFICIENTS,
        ),
    ];
    for (name, offset, count) in families {
        let start = offset as usize / 2;
        let end = start + count as usize;
        let original =
            crate::apple_gpu::buffer_slice::<u16>(trainer.weights(), end)?[start..].to_vec();
        let gradient = &crate::apple_gpu::buffer_slice::<f32>(trainer.gradients(), end)?[start..];
        let rms = (original
            .iter()
            .map(|&bits| decode_half(bits).powi(2))
            .sum::<f64>()
            / f64::from(count))
        .sqrt();
        for fraction in [0.03125, 0.015625, 0.0078125] {
            let mut losses = [0.0f64; 2];
            let mut predicted = 0.0;
            let mut changed = [0usize; 2];
            let mut attention = [0usize; 2];
            let mut experts = [0usize; 2];
            for (side, sign) in [-1.0, 1.0].into_iter().enumerate() {
                let weights = crate::apple_gpu::slice_mut::<u16>(trainer.weights(), end)?;
                for (index, (&bits, &adjoint)) in original.iter().zip(gradient).enumerate() {
                    let direction = if adjoint == 0.0 {
                        0.0
                    } else {
                        f64::from(adjoint.signum())
                    };
                    let value = resources::half_bits(
                        (decode_half(bits) + sign * fraction * rms * direction) as f32,
                    );
                    weights[start + index] = value;
                    changed[side] += usize::from(value != bits);
                    predicted +=
                        sign * f64::from(adjoint) * (decode_half(value) - decode_half(bits));
                }
                let result = scorer.score_batch(runtime, decoder, trainer.row(), batch);
                crate::apple_gpu::slice_mut::<u16>(trainer.weights(), end)?[start..]
                    .copy_from_slice(&original);
                losses[side] = f64::from(result?.quality.mean_nll);
                (attention[side], experts[side]) = selection.changes(&Selection::read(scorer)?);
            }
            let observed = losses[1] - losses[0];
            println!(
                "CORPUS_DIRECTION family={name} fraction={fraction} changed={changed:?} predicted={predicted:.9e} observed={observed:.9e} ratio={:.9e} last_attention_sets={attention:?} last_expert_sets={experts:?} diagnostic_only=true",
                observed / predicted
            );
        }
    }
    Ok(())
}

/// Only the final execution remains in these scratch buffers. A mismatch
/// demonstrates a changed branch; equality does not certify earlier visits.
#[cfg(test)]
struct Selection {
    attention: Vec<u32>,
    experts: Vec<u32>,
}

#[cfg(test)]
impl Selection {
    fn read(scorer: &block_decode::BlockDecoder) -> Result<Self, String> {
        use crate::apple_gpu::buffer_slice;
        Ok(Self {
            attention: buffer_slice::<u32>(&scorer.pisa.blocks, scorer.context as usize * 8)?
                .to_vec(),
            experts: buffer_slice::<u32>(
                &scorer.buffers.fine_grained.route_experts,
                scorer.context as usize * routing::ROUTED_TOPK as usize,
            )?
            .to_vec(),
        })
    }

    fn changes(&self, other: &Self) -> (usize, usize) {
        let count = |left: &[u32], right: &[u32], width| {
            left.chunks_exact(width)
                .zip(right.chunks_exact(width))
                .filter(|(a, b)| {
                    let mut a = a.to_vec();
                    let mut b = b.to_vec();
                    a.sort_unstable();
                    b.sort_unstable();
                    a != b
                })
                .count()
        };
        (
            count(&self.attention, &other.attention, 8),
            count(&self.experts, &other.experts, routing::ROUTED_TOPK as usize),
        )
    }
}
