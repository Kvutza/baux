use super::*;
use metal::objc::rc::autoreleasepool;

fn rotate_qkv(
    runtime: &Runtime,
    decoder: &Decoder,
    qkv: &BufferRef,
    position: u32,
) -> Result<(), String> {
    let command = runtime.queue.new_command_buffer();
    let encoder = command.new_compute_command_encoder();
    bytes(&encoder, 2, &[1, position, CONTEXT, ROPE_PAIRS]);
    launch(
        &encoder,
        &decoder.kernels.rope,
        &[(qkv, 0), (&decoder.rope, 0)],
        thread_group(u64::from(ROPE_HEADS * ROPE_PAIRS)),
        256,
    );
    encoder.end_encoding();
    complete(command).map(|_| ())
}

#[test]
fn rope_positions() -> Result<(), String> {
    autoreleasepool(|| {
        let runtime = Runtime::shared()?;
        let decoder = Decoder::new(&runtime)?;
        let qkv = runtime.buffer::<u16>(QKV_WIDTH as usize);
        let values = unsafe {
            std::slice::from_raw_parts_mut(qkv.contents().cast::<u16>(), QKV_WIDTH as usize)
        };
        values.fill(0);
        values[0] = 0x3c00;
        values[512] = 0x3c00;
        values[576] = 0x3c00;
        rotate_qkv(&runtime, &decoder, &qkv, 0)?;
        assert_eq!(values[0], 0x3c00);
        assert_eq!(values[32], 0);
        assert_eq!(values[512], 0x3c00);
        assert_eq!(values[544], 0);
        assert_eq!(values[576], 0x3c00);

        values.fill(0);
        values[0] = 0x3c00;
        values[512] = 0x3c00;
        values[576] = 0x3c00;
        rotate_qkv(&runtime, &decoder, &qkv, 1)?;
        assert_ne!(values[0], 0x3c00);
        assert_ne!(values[32], 0);
        assert_eq!(values[0], values[512]);
        assert_eq!(values[32], values[544]);
        assert_eq!(values[576], 0x3c00);
        Ok(())
    })
}

#[test]
fn qkv_epilogue() -> Result<(), String> {
    autoreleasepool(|| {
        let runtime = Runtime::shared()?;
        let source = include_str!("../moe_tensorops.metal");
        let separate =
            runtime.metal4_precise(source, "QKV reference", "fbt_model_tensorops_qkv_wide", &[])?;
        let fused = runtime.metal4_precise(source, "QKV epilogue", "qkv_rope", &[])?;
        let rope = runtime.pipeline(
            include_str!("../moe.metal"),
            "RoPE reference",
            "fbt_moe_rope",
        )?;
        let rows = 256u32;
        let input = patterned(&runtime, rows as usize * WIDTH as usize, 17, 0x3000);
        let weights = patterned(&runtime, WIDTH as usize * QKV_WIDTH as usize, 31, 0x2800);
        let reference = canary(&runtime, rows as usize * QKV_WIDTH as usize);
        let output = canary(&runtime, rows as usize * QKV_WIDTH as usize);
        let table = rope_table(&runtime, 512);
        let groups = MTLSize {
            width: 5,
            height: 2,
            depth: 1,
        };
        for start in [0u32, 128, 384] {
            let command = runtime.queue.new_command_buffer();
            let encoder = command.new_compute_command_encoder();
            tensor_workspace(
                &encoder,
                &separate,
                &[(&input, 0), (&weights, 0), (&reference, 0)],
                groups,
                8,
            )?;
            encoder.memory_barrier_with_resources(&[&reference]);
            bytes(&encoder, 2, &[rows, start, 512, 32]);
            encoder.set_compute_pipeline_state(&rope);
            encoder.set_buffer(0, Some(&reference), 0);
            encoder.set_buffer(1, Some(&table), 0);
            encoder.dispatch_threads(thread_group(u64::from(rows * 9 * 32)), thread_group(256));
            bytes(&encoder, 4, &[start, 512u32]);
            tensor_workspace(
                &encoder,
                &fused,
                &[(&input, 0), (&weights, 0), (&output, 0), (&table, 0)],
                groups,
                8,
            )?;
            encoder.end_encoding();
            complete(command)?;
            let elements = rows as usize * QKV_WIDTH as usize + 64;
            let expected =
                unsafe { std::slice::from_raw_parts(reference.contents().cast::<u16>(), elements) };
            let actual =
                unsafe { std::slice::from_raw_parts(output.contents().cast::<u16>(), elements) };
            assert!(
                expected.iter().zip(actual).all(|(a, b)| a == b),
                "QKV epilogue changed FP16 output at start {start}"
            );
        }
        Ok(())
    })
}

#[test]
fn sampling_edges() -> Result<(), String> {
    autoreleasepool(|| {
        let runtime = Runtime::shared()?;
        let decoder = Decoder::new(&runtime)?;
        let logits = unsafe {
            std::slice::from_raw_parts_mut(decoder.logits.contents().cast::<u16>(), VOCAB as usize)
        };
        logits.fill(0);
        logits[31] = 0x3c00;
        logits[8191] = 0x3c00;
        let tokens = unsafe {
            std::slice::from_raw_parts_mut(
                decoder.tokens.contents().cast::<u32>(),
                CONTEXT as usize + 1,
            )
        };
        let invalid = unsafe { &mut *decoder.invalid.contents().cast::<u32>() };
        for (input, temperature, expected) in
            [(9, 0.0, Some(31)), (31, 0.0, Some(31)), (9, 0.8, None)]
        {
            tokens[1] = input;
            *invalid = 0;
            let command = runtime.queue.new_command_buffer();
            let encoder = command.new_compute_command_encoder();
            bytes(
                &encoder,
                3,
                &Sample {
                    seed: rand::random(),
                    position: 1,
                    eos: 31,
                    first: 1,
                    vocabulary: VOCAB,
                    temperature,
                },
            );
            launch(
                &encoder,
                &decoder.kernels.sample,
                &[
                    (&decoder.logits, 0),
                    (&decoder.tokens, 0),
                    (&decoder.invalid, 0),
                ],
                thread_group(1),
                256,
            );
            encoder.end_encoding();
            complete(command)?;
            assert_eq!(*invalid, 0);
            assert!(tokens[2] < VOCAB);
            if let Some(expected) = expected {
                assert_eq!(tokens[2], expected);
            }
        }
        logits[0] = 0x7e00;
        *invalid = 0;
        let command = runtime.queue.new_command_buffer();
        let encoder = command.new_compute_command_encoder();
        bytes(
            &encoder,
            3,
            &Sample {
                seed: rand::random(),
                position: 1,
                eos: 31,
                first: 1,
                vocabulary: VOCAB,
                temperature: 0.0,
            },
        );
        launch(
            &encoder,
            &decoder.kernels.sample,
            &[
                (&decoder.logits, 0),
                (&decoder.tokens, 0),
                (&decoder.invalid, 0),
            ],
            thread_group(1),
            256,
        );
        encoder.end_encoding();
        complete(command)?;
        assert_eq!(*invalid, 1);
        Ok(())
    })
}

#[test]
fn mhc_rows() -> Result<(), String> {
    autoreleasepool(|| {
        let runtime = Runtime::shared()?;
        let rows = 129;
        let shape = [rows, WIDTH, ResidualArchitecture::Mhc4.kernel_code(), 0];
        let streams = patterned(&runtime, (rows * MHC_INPUT) as usize, 17, 0x3800);
        let branch = patterned(&runtime, (rows * WIDTH) as usize, 19, 0x3000);
        let coefficients = runtime.buffer_with(
            &(0..rows * MHC_COEFFICIENTS)
                .map(|i| ((i * 17 % 257) as f32 - 128.0) / 128.0)
                .collect::<Vec<_>>(),
        );
        let reference = runtime.buffer::<u16>((rows * MHC_INPUT) as usize);
        let output = runtime.buffer::<u16>((rows * MHC_INPUT) as usize);
        let command = runtime.queue.new_command_buffer();
        let encoder = command.new_compute_command_encoder();
        for (name, buffer, elements) in [
            ("fbt_mhc_update", &reference, rows * MHC_INPUT),
            ("fbt_mhc_update_rows", &output, rows * WIDTH / 4),
        ] {
            let kernel = runtime.pipeline(include_str!("../moe.metal"), "mHC parity", name)?;
            encoder.set_compute_pipeline_state(&kernel);
            for (slot, buffer) in [&streams, &branch, &coefficients, buffer]
                .into_iter()
                .enumerate()
            {
                encoder.set_buffer(slot as u64, Some(buffer), 0);
            }
            bytes(&encoder, 4, &shape);
            encoder.dispatch_threads(thread_group(u64::from(elements)), thread_group(256));
        }
        encoder.end_encoding();
        complete(command)?;
        let values = |buffer: &BufferRef| unsafe {
            std::slice::from_raw_parts(buffer.contents().cast::<u16>(), (rows * MHC_INPUT) as usize)
        };
        assert_eq!(values(&reference), values(&output));
        Ok(())
    })
}

#[test]
fn moe_transport() -> Result<(), String> {
    autoreleasepool(|| {
        let runtime = Runtime::shared()?;
        let rows = 129u32;
        let source = include_str!("../moe.metal");
        let combine = runtime.pipeline(
            include_str!("../routing.metal"),
            "combine reference",
            "fbt_moe_combine_rows",
        )?;
        let update = runtime.pipeline(source, "transport reference", "fbt_mhc_update")?;
        let fused = runtime.pipeline(source, "MoE transport", "combine_mhc")?;
        let streams = patterned(&runtime, (rows * MHC_INPUT) as usize, 41, 0x3800);
        let routed = patterned(&runtime, (rows * 3 * WIDTH) as usize, 43, 0x3400);
        let shared = patterned(&runtime, (rows * WIDTH) as usize, 47, 0x3000);
        let weights = runtime.buffer_with(
            &(0..rows * 3)
                .map(|i| half_bits((i % 7 + 1) as f32 / 16.0))
                .collect::<Vec<_>>(),
        );
        let packed = runtime.buffer_with(
            &(0..rows * 3)
                .map(|i| i * 17 % (rows * 3))
                .collect::<Vec<_>>(),
        );
        let coefficients = runtime.buffer_with(
            &(0..rows * MHC_COEFFICIENTS)
                .map(|i| ((i * 17 % 257) as f32 - 128.0) / 128.0)
                .collect::<Vec<_>>(),
        );
        let branch = canary(&runtime, (rows * WIDTH) as usize);
        let fused_branch = canary(&runtime, (rows * WIDTH) as usize);
        let reference = canary(&runtime, (rows * MHC_INPUT) as usize);
        let output = canary(&runtime, (rows * MHC_INPUT) as usize);
        let command = runtime.queue.new_command_buffer();
        let encoder = command.new_compute_command_encoder();
        bytes(&encoder, 5, &[rows, WIDTH, 64, 3, 64, rows.div_ceil(64)]);
        launch(
            &encoder,
            &combine,
            &[
                (&weights, 0),
                (&packed, 0),
                (&routed, 0),
                (&shared, 0),
                (&branch, 0),
            ],
            thread_group(u64::from(rows)),
            128,
        );
        let shape = [rows, WIDTH, ResidualArchitecture::Mhc4.kernel_code(), 0];
        bytes(&encoder, 4, &shape);
        launch(
            &encoder,
            &update,
            &[
                (&streams, 0),
                (&branch, 0),
                (&coefficients, 0),
                (&reference, 0),
            ],
            thread_group(u64::from(rows * MHC_INPUT / 256)),
            256,
        );
        encoder.set_compute_pipeline_state(&fused);
        for (slot, buffer) in [
            &streams,
            &fused_branch,
            &coefficients,
            &output,
            &weights,
            &packed,
            &routed,
            &shared,
        ]
        .into_iter()
        .enumerate()
        {
            encoder.set_buffer(slot as u64, Some(buffer), 0);
        }
        bytes(&encoder, 8, &shape);
        encoder.dispatch_threads(thread_group(u64::from(rows * WIDTH / 4)), thread_group(256));
        encoder.end_encoding();
        complete(command)?;
        for (expected, actual, elements) in [
            (&reference, &output, rows * MHC_INPUT),
            (&branch, &fused_branch, rows * WIDTH),
        ] {
            let expected = unsafe {
                std::slice::from_raw_parts(
                    expected.contents().cast::<u16>(),
                    elements as usize + 64,
                )
            };
            let actual = unsafe {
                std::slice::from_raw_parts(actual.contents().cast::<u16>(), elements as usize + 64)
            };
            assert_eq!(
                expected, actual,
                "fused MoE transport changed the branch or residual streams"
            );
        }
        Ok(())
    })
}

#[test]
fn mhc_transition() -> Result<(), String> {
    autoreleasepool(|| {
        let runtime = Runtime::shared()?;
        let decoder = Decoder::new(&runtime)?;
        let streams = runtime.buffer_with(&vec![0x3800u16; (MHC_STREAMS * WIDTH) as usize]);
        let branch = runtime.buffer_with(&vec![0x3c00u16; WIDTH as usize]);
        let predictor = runtime.buffer_with(&vec![0u16; (MHC_INPUT * MHC_COEFFICIENTS) as usize]);
        let bias = runtime.buffer_with(&vec![0u16; MHC_COEFFICIENTS as usize]);
        let control = runtime.buffer_with(&[0u16; 4]);
        let coefficients = runtime.buffer::<f32>(MHC_COEFFICIENTS as usize);
        let output = runtime.buffer::<u16>((MHC_STREAMS * WIDTH) as usize);
        let shape = [1u32, WIDTH, ResidualArchitecture::Mhc4.kernel_code(), 0];
        let command = runtime.queue.new_command_buffer();
        let encoder = command.new_compute_command_encoder();
        bytes(&encoder, 5, &shape);
        launch(
            &encoder,
            &decoder.kernels.mhc_predict,
            &[
                (&streams, 0),
                (&predictor, 0),
                (&bias, 0),
                (&control, 0),
                (&coefficients, 0),
            ],
            thread_group(1),
            256,
        );
        bytes(&encoder, 4, &shape);
        launch(
            &encoder,
            &decoder.kernels.mhc_update,
            &[
                (&streams, 0),
                (&branch, 0),
                (&coefficients, 0),
                (&output, 0),
            ],
            thread_group(8),
            256,
        );
        encoder.end_encoding();
        complete(command)?;
        let coefficients = unsafe {
            std::slice::from_raw_parts(
                coefficients.contents().cast::<f32>(),
                MHC_COEFFICIENTS as usize,
            )
        };
        for value in &coefficients[..4] {
            assert_eq!(*value, 0.25);
        }
        for row in 0..4 {
            for column in 0..4 {
                assert_eq!(coefficients[4 + row * 4 + column], f32::from(row == column));
            }
            assert_eq!(coefficients[20 + row], 1.0);
        }
        let output = unsafe {
            std::slice::from_raw_parts(
                output.contents().cast::<u16>(),
                (MHC_STREAMS * WIDTH) as usize,
            )
        };
        assert!(output.iter().all(|&value| value == 0x3e00));
        Ok(())
    })
}

#[test]
fn mhc_predictor() -> Result<(), String> {
    autoreleasepool(|| {
        const ROWS: u32 = 64;
        let runtime = Runtime::shared()?;
        let decoder = Decoder::new(&runtime)?;
        let pipelines = Pipelines::new(&runtime)?;
        let streams = runtime.buffer_with(
            &(0..ROWS * MHC_INPUT)
                .map(|index| if index % 2 == 0 { 0x3800u16 } else { 0xb400u16 })
                .collect::<Vec<_>>(),
        );
        let predictor = runtime.buffer_with(
            &(0..MHC_INPUT * MHC_COEFFICIENTS)
                .map(|index| if index % 3 == 0 { 0x1400u16 } else { 0x9400u16 })
                .collect::<Vec<_>>(),
        );
        let bias = runtime.buffer_with(&vec![0u16; MHC_COEFFICIENTS as usize]);
        let control = runtime.buffer_with(&[0x3000u16, 0x3000, 0x3000, 0x3800]);
        let reference = runtime.buffer::<f32>((ROWS * MHC_COEFFICIENTS) as usize);
        let tiled = runtime.buffer::<f32>((ROWS * MHC_COEFFICIENTS) as usize);
        let shape = [ROWS, WIDTH, ResidualArchitecture::Mhc4.kernel_code(), 0];
        let command = runtime.queue.new_command_buffer();
        let encoder = command.new_compute_command_encoder();
        bytes(&encoder, 5, &shape);
        launch(
            &encoder,
            &decoder.kernels.mhc_predict,
            &[
                (&streams, 0),
                (&predictor, 0),
                (&bias, 0),
                (&control, 0),
                (&reference, 0),
            ],
            thread_group(u64::from(ROWS)),
            256,
        );
        bytes(&encoder, 5, &shape);
        launch(
            &encoder,
            &pipelines.mhc_predict_rows,
            &[
                (&streams, 0),
                (&predictor, 0),
                (&bias, 0),
                (&control, 0),
                (&tiled, 0),
            ],
            thread_group(1),
            128,
        );
        encoder.end_encoding();
        complete(command)?;
        let values = |buffer: &Buffer| unsafe {
            std::slice::from_raw_parts(
                buffer.contents().cast::<f32>(),
                (ROWS * MHC_COEFFICIENTS) as usize,
            )
        };
        for (expected, actual) in values(&reference).iter().zip(values(&tiled)) {
            assert!((expected - actual).abs() < 2.0e-3, "{expected} != {actual}");
        }
        Ok(())
    })
}

#[test]
fn mhc_decode() -> Result<(), String> {
    autoreleasepool(|| {
        let runtime = Runtime::shared()?;
        let decoder = Decoder::new(&runtime)?;
        let model = CandidateWeights::seeded_for(&runtime, Some(7), ResidualArchitecture::Mhc4);
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
        assert_eq!(offset, row.length() as usize);
        let layout = Mhc4Layout::from_weights(&model, 0);
        drop(model);
        let weights = layout.view(&row);
        unsafe {
            *decoder.tokens.contents().cast::<u32>() = 17;
        }
        let command = runtime.queue.new_command_buffer();
        let encoder = command.new_compute_command_encoder();
        decoder.prepare_router(&encoder, weights);
        decoder.token(&encoder, weights, 0, FeedbackTransition::Identity);
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
        assert!(logits.iter().all(|&value| decode_half(value).is_finite()));
        Ok(())
    })
}

#[test]
fn decode_gemv() -> Result<(), String> {
    autoreleasepool(|| {
        let runtime = Runtime::shared()?;
        let decoder = Decoder::new(&runtime)?;
        let seed = rand::random::<u32>();
        eprintln!("decode GEMV seed={seed}");
        unsafe {
            std::ptr::copy_nonoverlapping(
                [1u32, 0, 2].as_ptr(),
                decoder.experts.contents().cast(),
                3,
            );
        }
        for (k, n, mode) in [
            (512, 640, 0),
            (512, 512, 0),
            (512, 128, 0),
            (512, 8192, 0),
            (512, 512, 1),
            (256, 512, 2),
        ] {
            let batches = if mode == 0 { 1 } else { 4 };
            let input = patterned(&runtime, k as usize * batches, seed, 0x2400);
            let weight = patterned(
                &runtime,
                (k * n) as usize * batches,
                seed.wrapping_add(1),
                0x2000,
            );
            let output = filled(&runtime, n as usize * batches + 1, 0x7e00);
            let scalar_output = runtime.buffer::<u16>(n as usize * batches);
            let command = runtime.queue.new_command_buffer();
            let encoder = command.new_compute_command_encoder();
            decoder.gemv(
                &encoder,
                (&input, 0),
                (&weight, 0),
                (&output, 0),
                k,
                n,
                mode,
            );
            encoder.end_encoding();
            complete(command)?;
            let command = runtime.queue.new_command_buffer();
            let encoder = command.new_compute_command_encoder();
            bytes(&encoder, 4, &[k, n, mode, 0]);
            launch(
                &encoder,
                &decoder.kernels.gemv,
                &[
                    (&input, 0),
                    (&weight, 0),
                    (&scalar_output, 0),
                    (&decoder.experts, 0),
                ],
                MTLSize {
                    width: u64::from(n.div_ceil(32)),
                    height: batches as u64,
                    depth: 1,
                },
                128,
            );
            encoder.end_encoding();
            complete(command)?;
            let x = unsafe {
                std::slice::from_raw_parts(input.contents().cast::<u16>(), k as usize * batches)
            };
            let w = unsafe {
                std::slice::from_raw_parts(
                    weight.contents().cast::<u16>(),
                    (k * n) as usize * batches,
                )
            };
            let y = unsafe {
                std::slice::from_raw_parts(
                    output.contents().cast::<u16>(),
                    n as usize * batches + 1,
                )
            };
            let scalar = unsafe {
                std::slice::from_raw_parts(
                    scalar_output.contents().cast::<u16>(),
                    n as usize * batches,
                )
            };
            assert_eq!(&y[..n as usize * batches], scalar);
            let mut max_error = 0.0f64;
            for batch in 0..batches {
                let expert = [0, 2, 1, 3][batch];
                for column in 0..n as usize {
                    let expected: f64 = (0..k as usize)
                        .map(|i| {
                            let source = if mode == 2 { batch * k as usize + i } else { i };
                            let matrix = if mode == 0 {
                                0
                            } else {
                                expert * (k * n) as usize
                            };
                            decode_half(x[source])
                                * decode_half(w[matrix + i * n as usize + column])
                        })
                        .sum();
                    let actual = decode_half(y[batch * n as usize + column]);
                    assert!(actual.is_finite());
                    let error = (actual - expected).abs();
                    max_error = max_error.max(error);
                    assert!(
                        error <= 0.000001 + expected.abs() * 0.001,
                        "{k}x{n} mode={mode}: {actual} vs {expected}"
                    );
                }
            }
            assert_eq!(y[n as usize * batches], 0x7e00);
            eprintln!("decode GEMV {k}x{n} mode={mode} max_error={max_error}");
        }
        Ok(())
    })
}

#[test]
fn router_logits() -> Result<(), String> {
    autoreleasepool(|| {
        let runtime = Runtime::shared()?;
        let decoder = Decoder::new(&runtime)?;
        let seed = rand::random::<u32>();
        let source = patterned(
            &runtime,
            (MODEL_LAYERS * WIDTH * routing::ROUTED_EXPERTS) as usize,
            seed,
            0x2000,
        );
        let input = patterned(&runtime, WIDTH as usize, seed.wrapping_add(1), 0x2400);
        let scalar = runtime.buffer::<u16>((MODEL_LAYERS * routing::ROUTED_EXPERTS) as usize);
        let vector = runtime.buffer::<u16>((MODEL_LAYERS * PADDED_EXPERTS) as usize);
        let command = runtime.queue.new_command_buffer();
        let encoder = command.new_compute_command_encoder();
        launch(
            &encoder,
            &decoder.kernels.pad_router,
            &[(&source, 0), (&decoder.router, 0)],
            thread_group(u64::from(
                (MODEL_LAYERS * WIDTH * PADDED_EXPERTS).div_ceil(256),
            )),
            256,
        );
        for layer in 0..MODEL_LAYERS {
            bytes(&encoder, 4, &[WIDTH, routing::ROUTED_EXPERTS, 0, 0]);
            launch(
                &encoder,
                &decoder.kernels.gemv,
                &[
                    (&input, 0),
                    (
                        &source,
                        half_bytes(u64::from(layer * WIDTH * routing::ROUTED_EXPERTS)),
                    ),
                    (
                        &scalar,
                        half_bytes(u64::from(layer * routing::ROUTED_EXPERTS)),
                    ),
                    (&decoder.experts, 0),
                ],
                thread_group(u64::from(routing::ROUTED_EXPERTS.div_ceil(32))),
                128,
            );
            bytes(&encoder, 4, &[WIDTH, PADDED_EXPERTS, 0, 0]);
            launch(
                &encoder,
                &decoder.kernels.gemv_vector,
                &[
                    (&input, 0),
                    (
                        &decoder.router,
                        half_bytes(u64::from(layer * WIDTH * PADDED_EXPERTS)),
                    ),
                    (&vector, half_bytes(u64::from(layer * PADDED_EXPERTS))),
                    (&decoder.experts, 0),
                ],
                thread_group(u64::from(PADDED_EXPERTS.div_ceil(32))),
                128,
            );
        }
        encoder.end_encoding();
        complete(command)?;
        let expected = unsafe {
            std::slice::from_raw_parts(
                scalar.contents().cast::<u16>(),
                (MODEL_LAYERS * routing::ROUTED_EXPERTS) as usize,
            )
        };
        let actual = unsafe {
            std::slice::from_raw_parts(
                vector.contents().cast::<u16>(),
                (MODEL_LAYERS * PADDED_EXPERTS) as usize,
            )
        };
        for layer in 0..MODEL_LAYERS as usize {
            assert_eq!(
                &expected[layer * routing::ROUTED_EXPERTS as usize
                    ..(layer + 1) * routing::ROUTED_EXPERTS as usize],
                &actual[layer * PADDED_EXPERTS as usize
                    ..layer * PADDED_EXPERTS as usize + routing::ROUTED_EXPERTS as usize]
            );
            assert_eq!(
                &actual[layer * PADDED_EXPERTS as usize + routing::ROUTED_EXPERTS as usize
                    ..(layer + 1) * PADDED_EXPERTS as usize],
                &[0; 0]
            );
        }
        Ok(())
    })
}

#[test]
fn pisa_prefix() -> Result<(), String> {
    check_pisa(CONTEXT)?;
    check_pisa(8192)
}

fn check_pisa(context: u32) -> Result<(), String> {
    autoreleasepool(|| {
        let runtime = Runtime::shared()?;
        let decoder = Decoder::with_context(&runtime, context)?;
        let nodes = 2 * context / PISA_BLOCK - 1;
        let seed = rand::random::<u32>();
        eprintln!("decode cache seed={seed}");
        let input = patterned(&runtime, (context * QKV_WIDTH) as usize, seed, 0x3000);
        let pyramid = runtime.buffer::<u16>((nodes * HEAD_DIM) as usize);
        let output = runtime.buffer::<u16>((context * WIDTH) as usize);
        let blocks = runtime.buffer::<u32>((context * PISA_SELECTED) as usize);
        let source = format!(
            "#define PISA_CONTEXT {context}\n#define PISA_QUERY_TILE 1\n#define \
             PISA_SKIP_IDENTITY_RESCALE\n{}",
            include_str!("../pisa1.metal")
        );
        let pipeline = |name| runtime.metal4_pipeline(&source, "prefix oracle", name, &[]);
        let leaf = pipeline("fbt_pisa1_leaf_means")?;
        let upper = pipeline("fbt_pisa1_upper_means")?;
        let attention = pipeline("fbt_pisa1_select_attention_q4")?;
        let command = runtime.queue.new_command_buffer();
        let encoder = command.new_compute_command_encoder();
        launch(
            &encoder,
            &leaf,
            &[(&input, 0), (&pyramid, 0)],
            thread_group(u64::from(context / PISA_BLOCK)),
            64,
        );
        launch(&encoder, &upper, &[(&pyramid, 0)], thread_group(1), 64);
        bytes(&encoder, 4, &[0u32, context]);
        launch(
            &encoder,
            &attention,
            &[(&input, 0), (&pyramid, 0), (&blocks, 0), (&output, 0)],
            thread_group(u64::from(context)),
            32,
        );
        encoder.end_encoding();
        complete(command)?;
        let cache = &decoder.cache[0];
        // Poison stale summaries: incomplete/future subtrees must never be
        // read.
        unsafe {
            std::slice::from_raw_parts_mut(
                cache.pyramid.contents().cast::<u16>(),
                (nodes * HEAD_DIM) as usize,
            )
            .fill(0x7e00);
        }
        for position in 0..context {
            if position % 64 == 63 || [0, 1, 7, 64, 128, 512].contains(&position) {
                let command = runtime.queue.new_command_buffer();
                let encoder = command.new_compute_command_encoder();
                if position % 64 == 63 {
                    bytes(&encoder, 2, &position);
                    launch(
                        &encoder,
                        &decoder.kernels.leaf,
                        &[(&input, 0), (&cache.pyramid, 0)],
                        thread_group(1),
                        64,
                    );
                }
                bytes(&encoder, 4, &position);
                launch(
                    &encoder,
                    &decoder.kernels.attention,
                    &[
                        (&input, 0),
                        (&cache.pyramid, 0),
                        (&decoder.blocks, 0),
                        (&decoder.attention, 0),
                    ],
                    thread_group(1),
                    32,
                );
                encoder.end_encoding();
                complete(command)?;
                for (a, b, width, size) in [
                    (&output, &decoder.attention, WIDTH as usize, 2),
                    (&blocks, &decoder.blocks, PISA_SELECTED as usize, 4),
                ] {
                    let offset = position as usize * width * size;
                    let left = unsafe {
                        std::slice::from_raw_parts(
                            a.contents().cast::<u8>().add(offset),
                            width * size,
                        )
                    };
                    let right = unsafe {
                        std::slice::from_raw_parts(
                            b.contents().cast::<u8>().add(offset),
                            width * size,
                        )
                    };
                    assert_eq!(
                        left, right,
                        "cache mismatch at position {position}, element bytes={size}"
                    );
                }
            }
        }
        Ok(())
    })
}
