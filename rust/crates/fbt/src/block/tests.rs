use super::*;
use crate::config::GenerationReward;
use metal::objc::rc::autoreleasepool;

#[test]
fn context_chunks() -> Result<(), String> {
    autoreleasepool(|| {
        const TEST_CONTEXT: u32 = CONTEXT;
        const TEST_CHUNK: u32 = 2048;
        let runtime = Runtime::shared()?;
        let regular = decode::Decoder::with_cache(&runtime, TEST_CONTEXT, false)?;
        let compact = decode::Decoder::with_cache(&runtime, TEST_CONTEXT, true)?;
        let verifier = BlockDecoder::with_context(&runtime, TEST_CONTEXT)?;
        let model =
            CandidateWeights::seeded_for(&runtime, Some(17), ResidualArchitecture::LoopedMhc4);
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
        let readout = scorer::ProposalReadout {
            output: &verifier.proposals,
            hidden: None,
            seeds: &verifier.seeds,
            temperature: 0.8,
            score_targets: true,
            feedback: crate::config::FeedbackTransition::Identity,
        };
        unsafe {
            verifier.seeds.contents().cast::<u64>().write(17);
        }
        let command = runtime.queue.new_command_buffer();
        scorer::encode_proposals(
            command,
            &verifier.pipelines,
            &verifier.tensorops,
            &verifier.pisa,
            &verifier.buffers,
            weights,
            readout,
            TEST_CONTEXT,
            Some(regular.cache()),
            None,
        )?;
        complete(command)?;
        let expected = unsafe {
            std::slice::from_raw_parts(
                verifier.proposals.contents().cast::<u32>(),
                TEST_CONTEXT as usize,
            )
        }
        .to_vec();
        let expected_losses = unsafe {
            std::slice::from_raw_parts(
                verifier.buffers.losses.contents().cast::<f32>(),
                TEST_CONTEXT as usize,
            )
        }
        .to_vec();
        for start in (0..TEST_CONTEXT).step_by(TEST_CHUNK as usize) {
            let command = runtime.queue.new_command_buffer();
            scorer::context_chunk(
                command,
                &verifier.pipelines,
                &verifier.tensorops,
                &verifier.pisa,
                &verifier.buffers,
                weights,
                readout,
                start,
                TEST_CHUNK,
                compact.cache(),
                false,
                None,
            )?;
            complete(command)?;
        }
        let actual = unsafe {
            std::slice::from_raw_parts(
                verifier.proposals.contents().cast::<u32>(),
                TEST_CONTEXT as usize,
            )
        };
        let losses = unsafe {
            std::slice::from_raw_parts(
                verifier.buffers.losses.contents().cast::<f32>(),
                TEST_CONTEXT as usize,
            )
        };
        assert_eq!(actual, expected);
        assert!(
            losses
                .iter()
                .zip(&expected_losses)
                .all(|(actual, expected)| (actual - expected).abs() <= 0.003)
        );
        for (compact, regular) in compact
            .cache()
            .iter()
            .zip(regular.cache())
            .take(weights.architecture.layer_steps().len())
        {
            let nodes = (2 * TEST_CONTEXT / PISA_BLOCK - 1) * HEAD_DIM;
            let a = unsafe {
                std::slice::from_raw_parts(compact.pyramid.contents().cast::<u16>(), nodes as usize)
            };
            let b = unsafe {
                std::slice::from_raw_parts(regular.pyramid.contents().cast::<u16>(), nodes as usize)
            };
            assert_eq!(a, b);
        }
        Ok(())
    })
}

#[test]
fn target_windows() {
    let mut stats = TargetStats::new(128);
    for position in 0..256 {
        let loss = if position == 200 { 65.0 } else { 1.0 };
        stats.push(position, loss, position != 200);
    }
    let quality = stats.finish().unwrap();
    assert_eq!(quality.window_tokens, 128);
    assert_eq!(quality.maximum_nll, 65.0);
    assert_eq!(quality.worst_window_nll, 1.5);
    assert_eq!(quality.positional_mismatches, 1);
    assert_eq!(quality.first_positional_mismatch, Some(200));
    assert_eq!(quality.mismatch_target_nll, Some(65.0));
    assert_eq!(quality.positional_accuracy, 255.0 / 256.0);
}

#[test]
#[ignore = "repair expectations still use the previous vocabulary mapping"]
fn parallel_repairs() -> Result<(), String> {
    autoreleasepool(|| {
        let runtime = Runtime::shared()?;
        let decoder = decode::Decoder::new(&runtime)?;
        let verifier = BlockDecoder::new(&runtime)?;
        let model = CandidateWeights::new(&runtime);
        let row = runtime.buffer::<u16>(FULL_PARAMETERS);
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
        let config = GenerationConfig {
            purpose: crate::config::GenerationPurpose::SystemsProbe,
            initialization: crate::config::ModelInitialization::Patterned,
            feedback_transition: crate::config::FeedbackTransition::ProjectedSigmoid,
            max_tokens: 8,
            temperature: 0.0,
            temperature_bounds: None,
            temperature_step: None,
            eos_token: None,
            seed: None,
            checkpoint: None,
            qualification_manifest: None,
            save_checkpoint: None,
            save_final_checkpoint: true,
            record_tensor_updates: true,
            signal_gate: None,
            verify: Default::default(),
            draft: None,
            reward: GenerationReward::ExactMatch,
            corpus_prompt: Vec::new(),
            corpus_prompt_tokens: None,
            episode_dataset: None,
            tasks: vec![GenerationTask {
                prompt: vec![1, 2, 3, 4],
                expected: vec![5],
                decoys: Vec::new(),
            }],
        };
        let sampling_seed = rand::random();
        let drafts = config
            .tasks
            .iter()
            .enumerate()
            .map(|(index, task)| {
                decoder.generate(
                    &runtime,
                    weights,
                    task,
                    &config,
                    crate::hash::splitmix64(sampling_seed ^ index as u64),
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        let mut corrupted = drafts.clone();
        corrupted[0].tokens[6] = (corrupted[0].tokens[6] + 1) % VOCAB;
        let verified = verifier.verify(
            &runtime,
            &decoder,
            weights,
            &config.tasks,
            &config,
            sampling_seed,
            &corrupted,
        )?;
        assert_eq!(verified[0].tokens, drafts[0].tokens);
        assert_eq!(verified[0].broad_passes, 2);
        assert_eq!(verified[0].evaluated_positions, 2 * CONTEXT as usize);
        let fixed = verifier.verify(
            &runtime,
            &decoder,
            weights,
            &config.tasks,
            &config,
            sampling_seed,
            &verified,
        )?;
        assert_eq!(
            verified
                .iter()
                .map(|rollout| &rollout.tokens)
                .collect::<Vec<_>>(),
            fixed
                .iter()
                .map(|rollout| &rollout.tokens)
                .collect::<Vec<_>>()
        );
        assert!(
            fixed
                .iter()
                .all(|rollout| rollout.evaluated_positions == CONTEXT as usize)
        );
        Ok(())
    })
}

#[test]
fn generation_fixedpoint() -> Result<(), String> {
    autoreleasepool(|| {
        let runtime = Runtime::shared()?;
        let decoder = decode::Decoder::new(&runtime)?;
        let verifier = BlockDecoder::new(&runtime)?;
        let model = CandidateWeights::new(&runtime);
        let row = runtime.buffer::<u16>(FULL_PARAMETERS);
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
        let config = GenerationConfig {
            purpose: crate::config::GenerationPurpose::SystemsProbe,
            initialization: crate::config::ModelInitialization::Patterned,
            feedback_transition: crate::config::FeedbackTransition::ProjectedSigmoid,
            max_tokens: 8,
            temperature: 0.0,
            temperature_bounds: None,
            temperature_step: None,
            eos_token: None,
            seed: None,
            checkpoint: None,
            qualification_manifest: None,
            save_checkpoint: None,
            save_final_checkpoint: true,
            record_tensor_updates: true,
            signal_gate: None,
            verify: Default::default(),
            draft: None,
            reward: GenerationReward::ExactMatch,
            corpus_prompt: Vec::new(),
            corpus_prompt_tokens: None,
            episode_dataset: None,
            tasks: vec![GenerationTask {
                prompt: vec![1],
                expected: vec![2],
                decoys: Vec::new(),
            }],
        };
        let sampling_seed = rand::random();
        let generated = verifier.generate(
            &runtime,
            &decoder,
            weights,
            &config.tasks,
            &config,
            sampling_seed,
        )?;
        let verified = verifier.verify(
            &runtime,
            &decoder,
            weights,
            &config.tasks,
            &config,
            sampling_seed,
            &generated,
        )?;
        assert_eq!(verified[0].tokens, generated[0].tokens);
        assert_eq!(verified[0].broad_passes, 1);
        assert_eq!(verified[0].evaluated_positions, CONTEXT as usize);
        Ok(())
    })
}

pub(super) fn hnet_batch(bytes: &[u8]) -> crate::pretrain_data::CausalBatch {
    let n = bytes.len() - 1;
    let tokens: Vec<u32> = bytes[..n].iter().map(|&b| u32::from(b)).collect();
    let labels: Vec<u32> = bytes[1..=n].iter().map(|&b| u32::from(b)).collect();
    crate::pretrain_data::CausalBatch {
        tokens,
        labels,
        loss_mask: vec![1; n],
        offset: 0,
    }
}

fn verify_bridge(scorer: &BlockDecoder, n: usize) -> Result<(), String> {
    let hnet = scorer.buffers.hnet()?;
    let d = WIDTH as usize;
    let active = crate::apple_gpu::buffer_slice::<u32>(&hnet.count, 1)?[0] as usize;
    let probs = crate::apple_gpu::buffer_slice::<f32>(&hnet.probs, n)?;
    let origins = crate::apple_gpu::buffer_slice::<u32>(&hnet.origins, active)?;
    let plug = crate::apple_gpu::buffer_slice::<u32>(&hnet.plug_idx, n)?;
    let proj = crate::apple_gpu::buffer_slice::<f32>(&hnet.projected, n * 2 * d)?;
    let core_f16 = crate::apple_gpu::buffer_slice::<u16>(&hnet.core, active * d)?;
    let ema_gpu = crate::apple_gpu::buffer_slice::<f32>(&hnet.chunk_scratch, active * d)?;
    let mut exp_origins = Vec::new();
    for t in 0..n {
        let p_cpu = if t == 0 {
            1.0f32
        } else {
            let (mut dot, mut nq, mut nk) = (0.0f32, 0.0f32, 0.0f32);
            for col in 0..d {
                let q = proj[(t - 1) * 2 * d + col];
                let k = proj[t * 2 * d + d + col];
                dot = q.mul_add(k, dot);
                nq = q.mul_add(q, nq);
                nk = k.mul_add(k, nk);
            }
            let cos = dot / (nq.sqrt().max(1e-12) * nk.sqrt().max(1e-12));
            ((1.0 - cos) * 0.5).clamp(0.0, 1.0)
        };
        assert!((probs[t] - p_cpu).abs() < 1e-5, "route mismatch at {t}");
        if t == 0 || probs[t] > 0.5 {
            exp_origins.push(t as u32);
        }
        assert_eq!(plug[t], (exp_origins.len() - 1) as u32);
    }
    assert_eq!(origins, exp_origins.as_slice());
    for col in 0..d {
        let mut state = 0.0f32;
        for c in 0..active {
            let p = probs[origins[c] as usize].clamp(1e-4, 1.0 - 1e-4);
            let x = crate::context::probe::half(core_f16[c * d + col]);
            state = p.mul_add(x, (1.0 - p) * state);
            assert!(
                (ema_gpu[c * d + col] - state).abs() < 1e-4,
                "EMA mismatch at c={c} col={col}"
            );
        }
    }
    Ok(())
}

pub(super) fn router_mode(
    model: &CandidateWeights,
    orig_hnet: &[u16],
    enc: Option<&[u16]>,
    mode: i32,
) -> Result<(), String> {
    let d = WIDTH as usize;
    let hnet_w = crate::apple_gpu::slice_mut::<u16>(&model.hnet, 3 * d * d + d)?;
    hnet_w.copy_from_slice(orig_hnet);
    match mode {
        // 0: uncalibrated identity router (W_q = I, W_k = I)
        0 => {
            hnet_w[..2 * d * d].fill(0);
            let one = resources::half_bits(1.0);
            for col in 0..d {
                hnet_w[col * d + col] = one;
                hnet_w[d * d + col * d + col] = one;
            }
        }
        // +1: degenerate all-closed router (W_q = 0, W_k = 0 -> cos = 0 -> p_t = 0.5 -> 1 chunk)
        1 => {
            hnet_w[..2 * d * d].fill(0);
        }
        // -1: degenerate all-open router (W_q = u u^T, W_k = -u u^T -> cos = -1 -> p_t = 1)
        -1 => {
            if let Some(enc_slice) = enc {
                let rows = (enc_slice.len() / d).min(256);
                let mut mat = vec![0.0f32; d * rows];
                for t in 0..rows {
                    for col in 0..d {
                        mat[col * rows + t] = crate::context::probe::half(enc_slice[t * d + col]);
                    }
                }
                let reg = 1e-2f32;
                let mut w = vec![0.0f32; rows];
                let mut r = vec![1.0f32; rows];
                let mut p = r.clone();
                let mut rho: f32 = r.iter().map(|&x| x * x).sum();
                let mut ep = vec![0.0f32; d];
                let mut q = vec![0.0f32; rows];
                for _ in 0..24 {
                    if rho < 1e-10 {
                        break;
                    }
                    for col in 0..d {
                        let row = &mat[col * rows..(col + 1) * rows];
                        ep[col] = row.iter().zip(p.iter()).map(|(&a, &b)| a * b).sum();
                    }
                    for t in 0..rows {
                        let mut acc = reg * p[t];
                        for col in 0..d {
                            acc = mat[col * rows + t].mul_add(ep[col], acc);
                        }
                        q[t] = acc;
                    }
                    let pq: f32 = p.iter().zip(q.iter()).map(|(&a, &b)| a * b).sum();
                    if pq <= 1e-12 {
                        break;
                    }
                    let step = rho / pq;
                    for t in 0..rows {
                        w[t] += step * p[t];
                        r[t] -= step * q[t];
                    }
                    let rho_new: f32 = r.iter().map(|&x| x * x).sum();
                    let beta = rho_new / rho;
                    for t in 0..rows {
                        p[t] = r[t] + beta * p[t];
                    }
                    rho = rho_new;
                }
                let mut u = vec![0.0f32; d];
                let mut norm_sq = 0.0f32;
                for col in 0..d {
                    let row = &mat[col * rows..(col + 1) * rows];
                    let val: f32 = row.iter().zip(w.iter()).map(|(&a, &b)| a * b).sum();
                    u[col] = val;
                    norm_sq = val.mul_add(val, norm_sq);
                }
                let inv = norm_sq.sqrt().max(1e-12).recip();
                for val in &mut u {
                    *val *= inv;
                }
                for row in 0..d {
                    for col in 0..d {
                        let rank1 = u[row] * u[col];
                        hnet_w[row * d + col] = resources::half_bits(rank1);
                        hnet_w[d * d + row * d + col] = resources::half_bits(-rank1);
                    }
                }
            } else {
                for row in 0..d {
                    for col in 0..d {
                        let base = crate::context::probe::half(orig_hnet[row * d + col]);
                        let rank1 = base - if row == col { 1.0 } else { 0.0 };
                        hnet_w[row * d + col] = resources::half_bits(rank1);
                        hnet_w[d * d + row * d + col] = resources::half_bits(-rank1);
                    }
                }
            }
        }
        _ => {}
    }
    Ok(())
}

fn check_parity(
    runtime: &Runtime,
    scorer: &BlockDecoder,
    decoder: &decode::Decoder,
    weights: CandidateRow<'_>,
    bytes: &[u8],
) -> Result<usize, String> {
    let n = bytes.len() - 1;
    let batch = hnet_batch(bytes);
    let compacted = scorer.score_batch(runtime, decoder, weights, &batch)?;
    let comp_losses = crate::apple_gpu::buffer_slice::<f32>(&scorer.buffers.losses, n)?.to_vec();
    let comp_preds = crate::apple_gpu::buffer_slice::<u32>(&scorer.proposals, n)?.to_vec();
    let active_chunks =
        crate::apple_gpu::buffer_slice::<u32>(&scorer.buffers.hnet()?.count, 1)?[0] as usize;
    verify_bridge(scorer, n)?;

    let cmd = runtime.queue.new_command_buffer().to_owned();
    let readout = scorer.readout(0.0, crate::config::FeedbackTransition::Identity, true);
    scorer::encode_hidden(
        &cmd,
        &scorer.pipelines,
        &scorer.tensorops,
        &scorer.pisa,
        &scorer.buffers,
        weights,
        readout,
        n as u32,
        Some(decoder.cache()),
        None,
    )?;
    scorer.encode_waves(&cmd, weights, 0, n as u32, 1)?;
    cmd.commit();
    complete_committed(&cmd)?;
    let ref_losses = crate::apple_gpu::buffer_slice::<f32>(&scorer.buffers.losses, n)?;
    let ref_preds = crate::apple_gpu::buffer_slice::<u32>(&scorer.proposals, n)?;
    assert_eq!(comp_preds, ref_preds);
    for i in 0..n {
        assert!(
            (comp_losses[i] - ref_losses[i]).abs() < 1e-4,
            "compacted vs uncompacted loss mismatch at {i} (n={n}): {} vs {}",
            comp_losses[i],
            ref_losses[i]
        );
    }
    assert!(compacted.quality.mean_nll.is_finite());
    Ok(active_chunks)
}

#[test]
fn hnet_parity() -> Result<(), String> {
    autoreleasepool(|| {
        let runtime = Runtime::shared()?;
        let arch = ResidualArchitecture::HnetMhc4;
        let model = CandidateWeights::initialized(
            &runtime,
            17,
            crate::config::ModelInitialization::FbtPisa1Mhc4,
            arch,
        )?;
        let d = WIDTH as usize;
        let orig_hnet = crate::apple_gpu::buffer_slice::<u16>(&model.hnet, 3 * d * d + d)?.to_vec();
        let decoder =
            decode::Decoder::with_visits(&runtime, CONTEXT, false, arch.layer_steps().len())?;
        let scorer = BlockDecoder::with_architecture(&runtime, CONTEXT, arch)?;

        // 1. Multi-byte UTF-8 sequence at 512 bytes (calibrated learned router)
        let utf8 = "Hello 🌍 — Δx = ∑ f(x_i)! Dynamic chunking over UTF-8 bytes.\n"
            .as_bytes()
            .repeat(12);
        let flat = model.flatten(&runtime)?;
        let utf8_chunks = check_parity(
            &runtime,
            &scorer,
            &decoder,
            model.row(&flat)?,
            &utf8[..=512],
        )?;
        assert!(utf8_chunks > 1 && utf8_chunks < 512);

        // 2. Irregular sequence length (n = 300, non-multiple of 128) on
        //    pseudo-random bytes
        let irreg_bytes: Vec<u8> = (0..=300).map(|i| ((i * 61 + 29) % 256) as u8).collect();
        let irreg_chunks =
            check_parity(&runtime, &scorer, &decoder, model.row(&flat)?, &irreg_bytes)?;
        drop(flat);
        assert!(irreg_chunks > 1 && irreg_chunks < 300);

        // 3. All-open boundary regime (every byte opens a chunk: active_chunks
        //    == n)
        let saved_irreg_enc = crate::apple_gpu::buffer_slice::<u16>(
            &scorer.buffers.hnet()?.encoder_streams,
            256 * d,
        )?
        .to_vec();
        router_mode(&model, &orig_hnet, Some(&saved_irreg_enc), -1)?;
        let flat_open = model.flatten(&runtime)?;
        let open_chunks = check_parity(
            &runtime,
            &scorer,
            &decoder,
            model.row(&flat_open)?,
            &irreg_bytes[..=256],
        )?;
        drop(flat_open);
        assert_eq!(open_chunks, 256);

        // 4. All-closed / single-long-chunk regime (no interior boundaries:
        //    active_chunks == 1)
        router_mode(&model, &orig_hnet, None, 1)?;
        let flat_closed = model.flatten(&runtime)?;
        let closed_chunks = check_parity(
            &runtime,
            &scorer,
            &decoder,
            model.row(&flat_closed)?,
            &irreg_bytes[..=256],
        )?;
        assert_eq!(closed_chunks, 1);
        Ok(())
    })
}

#[test]
fn hnet_continuity() -> Result<(), String> {
    autoreleasepool(|| {
        let runtime = Runtime::shared()?;
        let arch = ResidualArchitecture::HnetMhc4;
        let model = CandidateWeights::initialized(
            &runtime,
            17,
            crate::config::ModelInitialization::FbtPisa1Mhc4,
            arch,
        )?;
        let flat = model.flatten(&runtime)?;
        let weights = model.row(&flat)?;
        let decoder =
            decode::Decoder::with_visits(&runtime, CONTEXT, false, arch.layer_steps().len())?;
        let scorer = BlockDecoder::with_architecture(&runtime, CONTEXT, arch)?;

        let n = 512usize;
        let d = WIDTH as usize;
        let stream_stride = 4 * d;
        let bytes: Vec<u8> = (0..=n).map(|i| ((i * 43 + 11) % 256) as u8).collect();
        let batch = hnet_batch(&bytes);
        scorer.score_batch(&runtime, &decoder, weights, &batch)?;
        let hnet = scorer.buffers.hnet()?;
        let saved_enc =
            crate::apple_gpu::buffer_slice::<u16>(&hnet.encoder_streams, n * d)?.to_vec();

        // Populate 4-stream byte activations from saved encoder hidden states
        let load_streams = |start: usize, len: usize| -> Result<(), String> {
            let dst = crate::apple_gpu::slice_mut::<u16>(
                &scorer.buffers.mhc_streams[0],
                len * stream_stride,
            )?;
            for r in 0..len {
                let row_src = &saved_enc[(start + r) * d..(start + r + 1) * d];
                for s in 0..4 {
                    dst[(r * 4 + s) * d..(r * 4 + s + 1) * d].copy_from_slice(row_src);
                }
            }
            Ok(())
        };

        // Single unbroken 512-byte pass through Scorer::hnet_chunk +
        // Scorer::hnet_dechunk
        hnet.reset_sequence()?;
        load_streams(0, n)?;
        let cmd = runtime.queue.new_command_buffer().to_owned();
        scorer::bridge_tile(
            &cmd,
            &scorer.pipelines,
            &scorer.tensorops,
            &scorer.pisa,
            &scorer.buffers,
            weights,
            n as u32,
        )?;
        cmd.commit();
        complete_committed(&cmd)?;
        let full_count = crate::apple_gpu::buffer_slice::<u32>(&hnet.count, 1)?[0] as usize;
        let full_probs = crate::apple_gpu::buffer_slice::<f32>(&hnet.probs, n)?.to_vec();
        let full_origins =
            crate::apple_gpu::buffer_slice::<u32>(&hnet.origins, full_count)?.to_vec();
        let full_ema =
            crate::apple_gpu::buffer_slice::<f32>(&hnet.chunk_scratch, full_count * d)?.to_vec();
        let full_out = crate::apple_gpu::buffer_slice::<u16>(
            &scorer.buffers.mhc_streams[0],
            n * stream_stride,
        )?
        .to_vec();

        // Multi-tile execution across 4 irregular tiles [100, 150, 93, 169]
        // without state reset
        hnet.reset_sequence()?;
        let mut tiled_probs = Vec::with_capacity(n);
        let mut tiled_origins = Vec::new();
        let mut tiled_ema = Vec::new();
        let mut tiled_out = Vec::with_capacity(n * stream_stride);
        let mut cursor = 0usize;
        for tile_len in [100usize, 150, 93, 169] {
            load_streams(cursor, tile_len)?;
            let cmd = runtime.queue.new_command_buffer().to_owned();
            scorer::bridge_tile(
                &cmd,
                &scorer.pipelines,
                &scorer.tensorops,
                &scorer.pisa,
                &scorer.buffers,
                weights,
                tile_len as u32,
            )?;
            cmd.commit();
            complete_committed(&cmd)?;
            let k = crate::apple_gpu::buffer_slice::<u32>(&hnet.count, 1)?[0] as usize;
            tiled_probs.extend_from_slice(crate::apple_gpu::buffer_slice::<f32>(
                &hnet.probs,
                tile_len,
            )?);
            for &o in crate::apple_gpu::buffer_slice::<u32>(&hnet.origins, k)? {
                tiled_origins.push(cursor as u32 + o);
            }
            tiled_ema.extend_from_slice(crate::apple_gpu::buffer_slice::<f32>(
                &hnet.chunk_scratch,
                k * d,
            )?);
            tiled_out.extend_from_slice(crate::apple_gpu::buffer_slice::<u16>(
                &scorer.buffers.mhc_streams[0],
                tile_len * stream_stride,
            )?);
            cursor += tile_len;
        }
        assert_eq!(cursor, n);
        assert_eq!(
            tiled_origins, full_origins,
            "tiled chunk origins must match unbroken window"
        );
        for i in 0..n {
            assert!(
                (tiled_probs[i] - full_probs[i]).abs() < 1e-5,
                "tiled prob mismatch at {i}: {} vs {}",
                tiled_probs[i],
                full_probs[i]
            );
        }
        for i in 0..full_ema.len() {
            assert!(
                (tiled_ema[i] - full_ema[i]).abs() < 1e-4,
                "tiled EMA mismatch at {i}: {} vs {}",
                tiled_ema[i],
                full_ema[i]
            );
        }
        for i in 0..full_out.len() {
            let a = crate::context::probe::half(tiled_out[i]);
            let b = crate::context::probe::half(full_out[i]);
            assert!(
                (a - b).abs() < 1e-3,
                "tiled dechunked stream mismatch at {i}: {a} vs {b}"
            );
        }
        Ok(())
    })
}

#[test]
fn hnet_causality() -> Result<(), String> {
    autoreleasepool(|| {
        let runtime = Runtime::shared()?;
        let arch = ResidualArchitecture::HnetMhc4;
        let model = CandidateWeights::initialized(
            &runtime,
            17,
            crate::config::ModelInitialization::FbtPisa1Mhc4,
            arch,
        )?;
        let flat = model.flatten(&runtime)?;
        let weights = model.row(&flat)?;
        let decoder =
            decode::Decoder::with_visits(&runtime, CONTEXT, false, arch.layer_steps().len())?;
        let scorer = BlockDecoder::with_architecture(&runtime, CONTEXT, arch)?;

        let n = 512usize;
        let split = 256usize;
        let bytes: Vec<u8> = (0..=n).map(|i| ((i * 73 + 19) % 256) as u8).collect();
        let mut batch = hnet_batch(&bytes);
        scorer.score_batch(&runtime, &decoder, weights, &batch)?;
        let hnet = scorer.buffers.hnet()?;
        let probs_a = crate::apple_gpu::buffer_slice::<f32>(&hnet.probs, split)?.to_vec();
        let plug_a = crate::apple_gpu::buffer_slice::<u32>(&hnet.plug_idx, split)?.to_vec();
        let enc_a =
            crate::apple_gpu::buffer_slice::<u16>(&hnet.encoder_streams, split * WIDTH as usize)?
                .to_vec();
        let loss_a = crate::apple_gpu::buffer_slice::<f32>(&scorer.buffers.losses, split)?.to_vec();

        for tok in &mut batch.tokens[split..] {
            *tok = (*tok + 113) % 256;
        }
        scorer.score_batch(&runtime, &decoder, weights, &batch)?;
        let probs_b = crate::apple_gpu::buffer_slice::<f32>(&hnet.probs, split)?;
        let plug_b = crate::apple_gpu::buffer_slice::<u32>(&hnet.plug_idx, split)?;
        let enc_b =
            crate::apple_gpu::buffer_slice::<u16>(&hnet.encoder_streams, split * WIDTH as usize)?;
        let loss_b = crate::apple_gpu::buffer_slice::<f32>(&scorer.buffers.losses, split)?;
        assert_eq!(plug_a, plug_b);
        assert_eq!(enc_a, enc_b);
        for i in 0..split {
            assert!((probs_a[i] - probs_b[i]).abs() < 1e-6, "prob {i} changed");
            assert!((loss_a[i] - loss_b[i]).abs() < 1e-5, "loss {i} changed");
        }
        Ok(())
    })
}

#[test]
fn hnet_interior() -> Result<(), String> {
    autoreleasepool(|| {
        let runtime = Runtime::shared()?;
        let arch = ResidualArchitecture::HnetMhc4;
        let model = CandidateWeights::initialized(
            &runtime,
            17,
            crate::config::ModelInitialization::FbtPisa1Mhc4,
            arch,
        )?;
        let d = WIDTH as usize;
        let flat = model.flatten(&runtime)?;
        let weights = model.row(&flat)?;
        let decoder =
            decode::Decoder::with_visits(&runtime, CONTEXT, false, arch.layer_steps().len())?;
        let scorer = BlockDecoder::with_architecture(&runtime, CONTEXT, arch)?;

        let n = CONTEXT as usize;
        let bytes: Vec<u8> = (0..=n).map(|i| ((i * 29 + 7) % 251) as u8).collect();
        let mut batch = hnet_batch(&bytes);
        scorer.score_batch(&runtime, &decoder, weights, &batch)?;
        let hnet = scorer.buffers.hnet()?;
        let count = crate::apple_gpu::buffer_slice::<u32>(&hnet.count, 1)?[0] as usize;
        let origins = crate::apple_gpu::buffer_slice::<u32>(&hnet.origins, count)?.to_vec();
        let (chunk_idx, interior_t, next_origin) = (0..count.saturating_sub(1))
            .find_map(|c| {
                let start = origins[c] as usize;
                let end = origins[c + 1] as usize;
                (end - start >= 2).then_some((c, start + 1, end))
            })
            .ok_or("expected at least one multi-byte chunk with an interior byte")?;
        let next_chunk_state_a =
            crate::apple_gpu::buffer_slice::<u16>(&hnet.encoder_streams, (next_origin + 1) * d)?
                [next_origin * d..(next_origin + 1) * d]
                .to_vec();
        let losses_a = crate::apple_gpu::buffer_slice::<f32>(&scorer.buffers.losses, n)?.to_vec();

        // Mutate the interior byte of chunk `chunk_idx` (strictly between
        // origins[c] and origins[c+1])
        batch.tokens[interior_t] = (batch.tokens[interior_t] + 97) % 251;
        scorer.score_batch(&runtime, &decoder, weights, &batch)?;
        let next_chunk_state_b =
            &crate::apple_gpu::buffer_slice::<u16>(&hnet.encoder_streams, (next_origin + 1) * d)?
                [next_origin * d..(next_origin + 1) * d];
        let losses_b = crate::apple_gpu::buffer_slice::<f32>(&scorer.buffers.losses, n)?;

        assert_ne!(
            next_chunk_state_a, next_chunk_state_b,
            "interior byte at {interior_t} in chunk {chunk_idx} must influence subsequent chunk state at {next_origin}"
        );
        assert!(
            (next_origin..n).any(|t| (losses_a[t] - losses_b[t]).abs() > 1e-4),
            "interior byte mutation must influence downstream predictions"
        );
        for t in 0..interior_t {
            assert!((losses_a[t] - losses_b[t]).abs() < 1e-5);
        }
        Ok(())
    })
}

#[test]
fn hnet_objective() -> Result<(), String> {
    autoreleasepool(|| {
        let runtime = Runtime::shared()?;
        let arch = ResidualArchitecture::HnetMhc4;
        let model = CandidateWeights::initialized(
            &runtime,
            17,
            crate::config::ModelInitialization::FbtPisa1Mhc4,
            arch,
        )?;
        let decoder =
            decode::Decoder::with_visits(&runtime, CONTEXT, false, arch.layer_steps().len())?;
        let scorer = BlockDecoder::with_architecture(&runtime, CONTEXT, arch)?;
        let n = CONTEXT as usize;
        let bytes: Vec<u8> = (0..=n).map(|i| ((i * 41 + 13) % 256) as u8).collect();
        let batch = hnet_batch(&bytes);

        // 1. Ratio-objective test: compare calibrated (~4x) vs uncalibrated (I)
        //    and degenerate routing
        let d = WIDTH as usize;
        let orig_hnet = crate::apple_gpu::buffer_slice::<u16>(&model.hnet, 3 * d * d + d)?.to_vec();
        let flat_bal = model.flatten(&runtime)?;
        let score_bal = scorer.score_batch(&runtime, &decoder, model.row(&flat_bal)?, &batch)?;
        drop(flat_bal);
        let stats_bal = scorer.buffers.hnet()?.last_stats.get();
        let saved_enc =
            crate::apple_gpu::buffer_slice::<u16>(&scorer.buffers.hnet()?.encoder_streams, n * d)?
                .to_vec();
        eprintln!(
            "HNET_CALIBRATED_STATS active_chunks={} mean_prob={:.4} penalty={:.6}",
            stats_bal.active_chunks, stats_bal.mean_prob, stats_bal.ratio_penalty
        );
        assert!(
            (768..=1344).contains(&stats_bal.active_chunks),
            "calibrated Stage-0 active_chunks {} must sit near ~4x compression (1024 chunks on 4096 bytes)",
            stats_bal.active_chunks
        );

        router_mode(&model, &orig_hnet, None, 0)?;
        let flat_uncal = model.flatten(&runtime)?;
        let _score_uncal =
            scorer.score_batch(&runtime, &decoder, model.row(&flat_uncal)?, &batch)?;
        drop(flat_uncal);
        let stats_uncal = scorer.buffers.hnet()?.last_stats.get();
        assert!(stats_uncal.active_chunks > stats_bal.active_chunks);
        assert!(stats_uncal.ratio_penalty > stats_bal.ratio_penalty);

        router_mode(&model, &orig_hnet, Some(&saved_enc), -1)?;
        let flat_open = model.flatten(&runtime)?;
        let score_open = scorer.score_batch(&runtime, &decoder, model.row(&flat_open)?, &batch)?;
        let stats_open = scorer.buffers.hnet()?.last_stats.get();
        assert!(stats_open.active_chunks > stats_uncal.active_chunks);

        router_mode(&model, &orig_hnet, None, 1)?;
        let flat_closed = model.flatten(&runtime)?;
        let _score_closed =
            scorer.score_batch(&runtime, &decoder, model.row(&flat_closed)?, &batch)?;
        drop(flat_closed);
        let stats_closed = scorer.buffers.hnet()?.last_stats.get();
        assert_eq!(stats_closed.active_chunks, 1);
        assert!(stats_closed.degenerate_all_closed);
        assert!(stats_closed.ratio_penalty > stats_bal.ratio_penalty);

        crate::apple_gpu::slice_mut::<u16>(&model.hnet, 3 * d * d + d)?.copy_from_slice(&orig_hnet);
        assert!(
            stats_open.ratio_penalty > stats_bal.ratio_penalty * 2.0,
            "all-open penalty {} must exceed calibrated penalty {}",
            stats_open.ratio_penalty,
            stats_bal.ratio_penalty
        );
        let raw_open: f32 = crate::apple_gpu::buffer_slice::<f32>(&scorer.buffers.losses, n)?
            .iter()
            .sum::<f32>()
            / (n as f32);
        let _ = scorer.score_batch(&runtime, &decoder, model.row(&flat_open)?, &batch)?;
        drop(flat_open);
        let raw_open2: f32 = crate::apple_gpu::buffer_slice::<f32>(&scorer.buffers.losses, n)?
            .iter()
            .sum::<f32>()
            / (n as f32);
        let _ = raw_open;
        assert!(
            (score_open.quality.mean_nll - (raw_open2 + stats_open.ratio_penalty)).abs() < 1e-5
        );
        assert!((score_open.observation.control().mean + score_open.quality.mean_nll).abs() < 1e-5);

        // 2. Complete operator-BO round (ask -> materialize -> score -> tell ->
        //    commit -> sync)
        let run = crate::config::parse_tune(
            "version = 4\nexperiment = \"pretrain\"\nmodel = \"fbt-pisa1-hnet-mhc4-v1\"\n\
             recipe = \"fineweb-bytes-100m-v1\"\nobjective = \"cross-entropy\"\n\
             window-tokens = 4096\nrounds = 4\nround-target-ms = 250\n\
             [optimizer]\nalgorithm = \"operator-bo\"\nproposal = \"lie\"\n\
             distribution = \"gaussian\"\n\
             [optimizer.surrogate]\nneighbors = 8\nepistemic-scale = 0.1\n\
             aleatoric-scale = 0.05\ny-scale = 0.005\n\
             [optimizer.trust-region]\nshape = \"tensor-family-static\"\n\
             initial = 0.004\nmin = 0.001\nmax = 0.008\n",
        )?;
        let layout = Mhc4Layout::from_weights(&model, 0);
        let (mut search, _updates) = model.search_shaped(
            run.proposal_kernel(),
            run.length(),
            crate::config::TrustRegionShape::TensorFamilyStatic,
        )?;
        drop(model);
        let policy = run.resident_enn(run.acquisition_seed())?;
        search.configure_enn(policy)?;
        search.observe_objectives(score_bal.observation)?;
        let mut surrogate = baux::semantics::optimizer::Surrogate::with_spec(
            -score_bal.quality.mean_nll,
            score_bal.quality.mean_nll_variance,
            run.acquisition_seed(),
            policy.surrogate_spec()?,
        )?;
        let base_before =
            crate::apple_gpu::buffer_slice::<u16>(&search.base_buffer(), 1024)?.to_vec();
        let ask_t0 = Instant::now();
        let geom = search.sample_distances(101)?;
        let sel = surrogate.select_pool(
            &geom,
            0,
            baux::semantics::optimizer::AcquisitionRule::Ucb { beta: 0.0 },
            None,
        )?;
        let (cand_buf, proposal) = search.materialize_selection(101, &sel)?;
        let ask_ms = ask_t0.elapsed().as_secs_f64() * 1000.0;
        let cand_score = scorer.score_batch(&runtime, &decoder, layout.view(&cand_buf), &batch)?;
        let st = scorer.buffers.hnet()?.last_stats.get();
        let tell_t0 = Instant::now();
        let slot = surrogate.append(
            &geom[sel.selected],
            -cand_score.quality.mean_nll,
            cand_score.quality.mean_nll_variance,
            true,
        )?;
        search.commit_host(&proposal, slot, cand_score.observation, true, 0.004)?;
        assert_eq!(search.sync()?, vec![true]);
        let tell_ms = tell_t0.elapsed().as_secs_f64() * 1000.0;
        let base_after =
            crate::apple_gpu::buffer_slice::<u16>(&search.base_buffer(), 1024)?.to_vec();
        assert_ne!(
            base_before, base_after,
            "accepted BO step must update base weights"
        );
        eprintln!(
            "HNET_SLICE_BENCH bytes={} init_chunks={} uncal_chunks={} chunks={} aligned={} ask_ms={:.3} stage0_ms={:.3} stage1_ms={:.3} tell_ms={:.3} nll={:.6} penalty={:.6}",
            st.total_bytes,
            stats_bal.active_chunks,
            stats_uncal.active_chunks,
            st.active_chunks,
            st.aligned_rows,
            ask_ms,
            st.stage0_gpu_seconds * 1000.0,
            st.stage1_gpu_seconds * 1000.0,
            tell_ms,
            cand_score.quality.mean_nll,
            st.ratio_penalty,
        );
        Ok(())
    })
}

#[path = "tests/tests_hnet2.rs"]
mod tests_hnet2;
