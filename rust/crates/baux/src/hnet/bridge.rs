//! Numerical contract for the production hierarchy boundary and its adjoint.
use super::upstream::{fixtures, floats, size};
use super::*;
use baux::artifact::json::Value;

fn words(value: &Value) -> Vec<u16> {
    value
        .as_seq()
        .unwrap()
        .iter()
        .map(|value| size(value) as u16)
        .collect()
}

fn half(word: u16) -> f32 {
    let sign = if word & 0x8000 == 0 { 1.0 } else { -1.0 };
    let exponent = (word >> 10) & 31;
    let mantissa = word & 1023;
    if exponent == 0 {
        sign * f32::from(mantissa) * 2.0f32.powi(-24)
    } else {
        sign * (1.0 + f32::from(mantissa) / 1024.0) * 2.0f32.powi(i32::from(exponent) - 15)
    }
}

fn close(label: &str, actual: &[f32], expected: &[f32], tolerance: f32) {
    assert_eq!(actual.len(), expected.len());
    for (i, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        assert!(
            actual.is_finite() && (actual - expected).abs() <= tolerance * (1.0 + expected.abs()),
            "{label}[{i}]: {actual} != {expected}"
        );
    }
}

fn finish(command: &metal::CommandBufferRef) {
    command.commit();
    command.wait_until_completed();
    assert_eq!(command.status(), MTLCommandBufferStatus::Completed);
}

struct Case {
    shape: HNetShape,
    input: Buffer,
    weights: Buffer,
    projected: Buffer,
    residual: Buffer,
    probabilities: Buffer,
    streams: Buffer,
    saved: Buffer,
    origins: Buffer,
    plug: Buffer,
    count: Buffer,
    core: Buffer,
    ema: Buffer,
    saved_core: Buffer,
}

impl Case {
    fn new(runtime: &Runtime, case: &Value) -> Self {
        let length = size(&case["length"]);
        let width = size(&case["width"]);
        let n = length * width;
        let mut core = words(&case["core_bits"]);
        core.resize(n, 0);
        Self {
            shape: HNetShape {
                batch_size: 1,
                seq_len: length as u32,
                d_model: width as u32,
                max_chunks: length as u32,
                vocab_size: 256,
                eps: EMA_EPS,
                target_ratio: 2.0,
                score: 1,
            },
            input: runtime.buffer_with(&words(&case["hidden_bits"])),
            weights: runtime.buffer_with(&words(&case["weights_bits"])),
            projected: runtime.buffer::<f32>(2 * n),
            residual: runtime.buffer::<f32>(n),
            probabilities: runtime.buffer::<f32>(length),
            streams: runtime.buffer::<u16>(4 * n),
            saved: runtime.buffer::<u16>(n),
            origins: runtime.buffer::<u32>(length),
            plug: runtime.buffer::<u32>(length),
            count: runtime.buffer::<u32>(1),
            core: runtime.buffer_with(&core),
            ema: runtime.buffer::<f32>(n),
            saved_core: runtime.buffer::<u16>(n),
        }
    }

    fn forward(&self, waves: &HNetWaves, expected: &Value) -> Result<(), String> {
        let runtime = &waves.runtime;
        let command = runtime.queue.new_command_buffer();
        let encoder = command.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&waves.mhc_project);
        bind_buffers(
            &encoder,
            &[
                (&self.input, 0),
                (&self.weights, 0),
                (&self.projected, 0),
                (&self.residual, 0),
            ],
        );
        encode_value(&encoder, 4, &self.shape);
        encoder.dispatch_threads(
            thread_group(u64::from(self.shape.seq_len * self.shape.d_model)),
            thread_group(256),
        );
        barrier_buffers(&encoder);
        encoder.set_compute_pipeline_state(&waves.mhc_route);
        bind_buffers(&encoder, &[(&self.projected, 0), (&self.probabilities, 0)]);
        encode_value(&encoder, 2, &self.shape);
        encoder.dispatch_thread_groups(
            thread_group(u64::from(self.shape.seq_len)),
            thread_group(32),
        );
        barrier_buffers(&encoder);
        encoder.set_compute_pipeline_state(&waves.mhc_chunk);
        bind_buffers(
            &encoder,
            &[
                (&self.input, 0),
                (&self.streams, 0),
                (&self.saved, 0),
                (&self.origins, 0),
                (&self.probabilities, 0),
                (&self.plug, 0),
                (&self.count, 0),
            ],
        );
        encode_value(&encoder, 7, &self.shape);
        encoder.dispatch_thread_groups(thread_group(1), thread_group(256));
        encoder.end_encoding();
        finish(command);
        let (length, width) = (self.shape.seq_len as usize, self.shape.d_model as usize);
        close(
            "routing",
            buffer_slice(&self.probabilities, length)?,
            &floats(&expected["probability"]),
            2e-6,
        );
        let count = buffer_slice::<u32>(&self.count, 1)?[0] as usize;
        assert_eq!(count * width, words(&expected["chunks_bits"]).len());
        let streams = buffer_slice::<u16>(&self.streams, 4 * length * width)?;
        let chunks = words(&expected["chunks_bits"]);
        for row in 0..count {
            for stream in 0..4 {
                assert_eq!(
                    &streams[(row * 4 + stream) * width..(row * 4 + stream + 1) * width],
                    &chunks[row * width..(row + 1) * width]
                );
            }
        }
        assert!(streams[4 * count * width..].iter().all(|&x| x == 0));
        let command = runtime.queue.new_command_buffer();
        let encoder = command.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&waves.mhc_dechunk);
        bind_buffers(
            &encoder,
            &[
                (&self.core, 0),
                (&self.streams, 0),
                (&self.ema, 0),
                (&self.origins, 0),
                (&self.probabilities, 0),
                (&self.plug, 0),
                (&self.count, 0),
                (&self.residual, 0),
                (&self.saved_core, 0),
            ],
        );
        encode_value(&encoder, 9, &self.shape);
        encoder.dispatch_threads(thread_group(width as u64), thread_group(256));
        encoder.end_encoding();
        finish(command);
        let streams = buffer_slice::<u16>(&self.streams, 4 * length * width)?;
        let actual: Vec<_> = (0..length)
            .flat_map(|t| {
                streams[t * 4 * width..t * 4 * width + width]
                    .iter()
                    .copied()
                    .map(half)
            })
            .collect();
        close("combined", &actual, &floats(&expected["combined"]), 1e-3);
        self.backward(waves, expected, count)
    }

    fn backward(&self, waves: &HNetWaves, expected: &Value, count: usize) -> Result<(), String> {
        let runtime = &waves.runtime;
        let (length, width) = (self.shape.seq_len as usize, self.shape.d_model as usize);
        let n = length * width;
        let gradient = floats(&expected["gradient"]);
        let streams: Vec<_> = gradient
            .chunks_exact(width)
            .flat_map(|row| (0..4).flat_map(move |_| row.iter().map(|g| g / 4.0)))
            .collect();
        let streams = runtime.buffer_with(&streams);
        let combined = runtime.buffer::<f32>(n);
        let probability = runtime.buffer::<f32>(n);
        let core_grad = runtime.buffer::<f32>(n);
        let projected_grad = runtime.buffer::<f32>(2 * n);
        let input_grad = runtime.buffer::<f32>(n);
        let weight_grad = runtime.buffer_with(&vec![0.0f32; 3 * width * width + width]);
        let command = runtime.queue.new_command_buffer();
        let encoder = command.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&waves.dechunk_grad);
        bind_buffers(
            &encoder,
            &[
                (&streams, 0),
                (&self.saved_core, 0),
                (&self.ema, 0),
                (&self.probabilities, 0),
                (&self.plug, 0),
                (&self.origins, 0),
                (&self.count, 0),
                (&combined, 0),
                (&probability, 0),
                (&core_grad, 0),
            ],
        );
        encode_value(&encoder, 10, &self.shape);
        encoder.dispatch_threads(thread_group(width as u64), thread_group(256));
        encoder.end_encoding();
        finish(command);
        let core_values = buffer_slice::<f32>(&core_grad, n)?;
        close(
            "core gradient",
            &core_values[..count * width],
            &floats(&expected["core_gradient"]),
            1e-5,
        );
        let chunk_grad: Vec<_> = core_values
            .chunks_exact(width)
            .flat_map(|row| (0..4).flat_map(move |_| row.iter().map(|g| g * 0.5 / 4.0)))
            .collect();
        let chunk_grad = runtime.buffer_with(&chunk_grad);
        let command = runtime.queue.new_command_buffer();
        let encoder = command.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&waves.route_grad);
        bind_buffers(
            &encoder,
            &[
                (&self.projected, 0),
                (&probability, 0),
                (&projected_grad, 0),
            ],
        );
        encode_value(&encoder, 3, &self.shape);
        encoder.dispatch_thread_groups(thread_group(length as u64), thread_group(32));
        barrier_buffers(&encoder);
        encoder.set_compute_pipeline_state(&waves.input_grad);
        bind_buffers(
            &encoder,
            &[
                (&self.weights, 0),
                (&projected_grad, 0),
                (&combined, 0),
                (&chunk_grad, 0),
                (&self.probabilities, 0),
                (&self.plug, 0),
                (&input_grad, 0),
            ],
        );
        encode_value(&encoder, 7, &self.shape);
        encoder.dispatch_threads(thread_group(n as u64), thread_group(256));
        encoder.set_compute_pipeline_state(&waves.weight_grad);
        bind_buffers(
            &encoder,
            &[
                (&self.saved, 0),
                (&projected_grad, 0),
                (&combined, 0),
                (&weight_grad, 0),
            ],
        );
        encode_value(&encoder, 4, &self.shape);
        encoder.dispatch_threads(
            thread_group((3 * width * width + width) as u64),
            thread_group(256),
        );
        encoder.end_encoding();
        finish(command);
        close(
            "input gradient",
            buffer_slice(&input_grad, n)?,
            &floats(&expected["input_gradient"]),
            2e-5,
        );
        close(
            "weight gradient",
            buffer_slice(&weight_grad, 3 * width * width + width)?,
            &floats(&expected["weight_gradient"]),
            2e-5,
        );
        Ok(())
    }
}

#[test]
fn hierarchy() -> Result<(), String> {
    metal::objc::rc::autoreleasepool(|| {
        let waves = HNetWaves::new()?;
        for case in fixtures()["bridge"].as_seq().unwrap() {
            Case::new(&waves.runtime, case).forward(&waves, case)?;
            Case::new(&waves.runtime, case).forward_parallel(&waves, case)?;
        }
        Ok(())
    })
}

impl Case {
    fn forward_parallel(&self, waves: &HNetWaves, expected: &Value) -> Result<(), String> {
        let runtime = &waves.runtime;
        let (length, width) = (self.shape.seq_len as usize, self.shape.d_model as usize);
        let q_state = runtime.buffer_with(&vec![0.0f32; width]);
        let zero_seen = runtime.buffer_with(&[0u32]);
        let ema_in = runtime.buffer_with(&vec![0.0f32; width]);
        let ema_out = runtime.buffer_with(&vec![0.0f32; width]);
        let part_a = runtime.buffer::<f32>(EMA_PARTS * width);
        let part_b = runtime.buffer::<f32>(EMA_PARTS * width);

        let command = runtime.queue.new_command_buffer();
        let encoder = command.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&waves.mhc_project);
        bind_buffers(
            &encoder,
            &[
                (&self.input, 0),
                (&self.weights, 0),
                (&self.projected, 0),
                (&self.residual, 0),
            ],
        );
        encode_value(&encoder, 4, &self.shape);
        encoder.dispatch_threads(
            thread_group(u64::from(self.shape.seq_len * self.shape.d_model)),
            thread_group(256),
        );
        barrier_buffers(&encoder);
        encoder.set_compute_pipeline_state(&waves.mhc_route_state);
        bind_buffers(
            &encoder,
            &[
                (&self.projected, 0),
                (&self.probabilities, 0),
                (&q_state, 0),
                (&zero_seen, 0),
            ],
        );
        encode_value(&encoder, 4, &self.shape);
        encoder.dispatch_thread_groups(
            thread_group(u64::from(self.shape.seq_len)),
            thread_group(32),
        );
        barrier_buffers(&encoder);
        encoder.set_compute_pipeline_state(&waves.mhc_chunk_scan);
        bind_buffers(
            &encoder,
            &[
                (&self.origins, 0),
                (&self.probabilities, 0),
                (&self.plug, 0),
                (&self.count, 0),
                (&self.projected, 0),
                (&q_state, 0),
                (&zero_seen, 0),
            ],
        );
        encode_value(&encoder, 7, &self.shape);
        encoder.dispatch_thread_groups(thread_group(1), thread_group(256));
        barrier_buffers(&encoder);
        encoder.set_compute_pipeline_state(&waves.mhc_chunk_pack);
        bind_buffers(
            &encoder,
            &[
                (&self.input, 0),
                (&self.streams, 0),
                (&self.saved, 0),
                (&self.origins, 0),
                (&self.count, 0),
            ],
        );
        encode_value(&encoder, 5, &self.shape);
        encoder.dispatch_threads(
            thread_group(u64::from(self.shape.seq_len * self.shape.d_model)),
            thread_group(256),
        );
        encoder.end_encoding();
        finish(command);

        close(
            "routing_parallel",
            buffer_slice(&self.probabilities, length)?,
            &floats(&expected["probability"]),
            2e-6,
        );
        let count = buffer_slice::<u32>(&self.count, 1)?[0] as usize;
        assert_eq!(count * width, words(&expected["chunks_bits"]).len());
        let streams = buffer_slice::<u16>(&self.streams, 4 * length * width)?;
        let chunks = words(&expected["chunks_bits"]);
        for row in 0..count {
            for stream in 0..4 {
                assert_eq!(
                    &streams[(row * 4 + stream) * width..(row * 4 + stream + 1) * width],
                    &chunks[row * width..(row + 1) * width]
                );
            }
        }
        assert!(streams[4 * count * width..].iter().all(|&x| x == 0));

        let grid_2d = metal::MTLSize {
            width: width as u64,
            height: EMA_PARTS as u64,
            depth: 1,
        };
        let tg_2d = metal::MTLSize {
            width: (width as u64).min(256),
            height: 1,
            depth: 1,
        };
        let command = runtime.queue.new_command_buffer();
        let encoder = command.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&waves.mhc_dechunk_reduce);
        bind_buffers(
            &encoder,
            &[
                (&self.core, 0),
                (&self.ema, 0),
                (&self.origins, 0),
                (&self.probabilities, 0),
                (&self.count, 0),
                (&self.saved_core, 0),
                (&part_a, 0),
                (&part_b, 0),
                (&ema_in, 0),
            ],
        );
        encode_value(&encoder, 9, &self.shape);
        encoder.dispatch_threads(grid_2d, tg_2d);
        barrier_buffers(&encoder);
        encoder.set_compute_pipeline_state(&waves.mhc_dechunk_scan);
        bind_buffers(
            &encoder,
            &[
                (&self.core, 0),
                (&self.ema, 0),
                (&self.origins, 0),
                (&self.probabilities, 0),
                (&self.count, 0),
                (&part_a, 0),
                (&part_b, 0),
                (&ema_in, 0),
                (&ema_out, 0),
            ],
        );
        encode_value(&encoder, 9, &self.shape);
        encoder.dispatch_threads(grid_2d, tg_2d);
        barrier_buffers(&encoder);
        encoder.set_compute_pipeline_state(&waves.mhc_dechunk_plug);
        bind_buffers(
            &encoder,
            &[
                (&self.streams, 0),
                (&self.ema, 0),
                (&self.plug, 0),
                (&self.count, 0),
                (&self.residual, 0),
                (&ema_in, 0),
            ],
        );
        encode_value(&encoder, 6, &self.shape);
        encoder.dispatch_threads(
            thread_group(u64::from(self.shape.seq_len * self.shape.d_model)),
            thread_group(256),
        );
        encoder.end_encoding();
        finish(command);

        let streams = buffer_slice::<u16>(&self.streams, 4 * length * width)?;
        let actual: Vec<_> = (0..length)
            .flat_map(|t| {
                streams[t * 4 * width..t * 4 * width + width]
                    .iter()
                    .copied()
                    .map(half)
            })
            .collect();
        close(
            "combined_parallel",
            &actual,
            &floats(&expected["combined"]),
            1e-3,
        );
        self.backward(waves, expected, count)
    }
}

fn half_bits(val: f32) -> u16 {
    let bits = val.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xff) as i32 - 127 + 15;
    let mant = bits & 0x7f_ffff;
    if exp <= 0 {
        sign
    } else if exp >= 31 {
        sign | 0x7c00
    } else {
        sign | ((exp as u16) << 10) | ((mant >> 13) as u16)
    }
}

#[test]
fn tiled_bridge() -> Result<(), String> {
    metal::objc::rc::autoreleasepool(|| {
        let waves = HNetWaves::new()?;
        let runtime = &waves.runtime;
        let length = 64usize;
        let width = 32usize;
        let n = length * width;

        // Build deterministic FP16 input with repeated runs (so some
        // continuation tiles have zero boundaries and non-boundary
        // starts) and sign transitions.
        let mut input_bits = vec![0u16; n];
        for t in 0..length {
            let sign = match t {
                0..=9 => 1.0f32,
                10..=25 => -1.0f32,
                26..=45 => 1.0f32,
                _ => -0.75f32,
            };
            for d in 0..width {
                let v = sign * (((d % 7) as f32 + 1.0) * 0.125) + ((t % 3) as f32) * 0.01;
                input_bits[t * width + d] = half_bits(v);
            }
        }
        // Identity-like Q and K projections + small residual projection in
        // FP16.
        let mut weight_bits = vec![0u16; 3 * width * width + width];
        for i in 0..width {
            weight_bits[i * width + i] = half_bits(1.0);
            weight_bits[width * width + i * width + i] = half_bits(1.0);
            weight_bits[2 * width * width + i * width + i] = half_bits(0.25);
            weight_bits[3 * width * width + i] = half_bits(0.05);
        }

        // 1. Run single unbroken 64-byte window on GPU.
        let full_shape = HNetShape {
            batch_size: 1,
            seq_len: length as u32,
            d_model: width as u32,
            max_chunks: length as u32,
            vocab_size: 256,
            eps: EMA_EPS,
            target_ratio: 2.0,
            score: 1,
        };
        let buf_input = runtime.buffer_with(&input_bits);
        let buf_weights = runtime.buffer_with(&weight_bits);
        let buf_proj = runtime.buffer::<f32>(2 * n);
        let buf_res = runtime.buffer::<f32>(n);
        let buf_probs = runtime.buffer::<f32>(length);
        let buf_streams = runtime.buffer::<u16>(4 * n);
        let buf_saved = runtime.buffer::<u16>(n);
        let buf_origins = runtime.buffer::<u32>(length);
        let buf_plug = runtime.buffer::<u32>(length);
        let buf_count = runtime.buffer::<u32>(1);

        let command = runtime.queue.new_command_buffer();
        let encoder = command.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&waves.mhc_project);
        bind_buffers(
            &encoder,
            &[
                (&buf_input, 0),
                (&buf_weights, 0),
                (&buf_proj, 0),
                (&buf_res, 0),
            ],
        );
        encode_value(&encoder, 4, &full_shape);
        encoder.dispatch_threads(thread_group(n as u64), thread_group(256));
        barrier_buffers(&encoder);
        encoder.set_compute_pipeline_state(&waves.mhc_route);
        bind_buffers(&encoder, &[(&buf_proj, 0), (&buf_probs, 0)]);
        encode_value(&encoder, 2, &full_shape);
        encoder.dispatch_thread_groups(thread_group(length as u64), thread_group(32));
        barrier_buffers(&encoder);
        encoder.set_compute_pipeline_state(&waves.mhc_chunk);
        bind_buffers(
            &encoder,
            &[
                (&buf_input, 0),
                (&buf_streams, 0),
                (&buf_saved, 0),
                (&buf_origins, 0),
                (&buf_probs, 0),
                (&buf_plug, 0),
                (&buf_count, 0),
            ],
        );
        encode_value(&encoder, 7, &full_shape);
        encoder.dispatch_thread_groups(thread_group(1), thread_group(256));
        encoder.end_encoding();
        finish(command);

        let full_count = buffer_slice::<u32>(&buf_count, 1)?[0] as usize;
        let full_origins = buffer_slice::<u32>(&buf_origins, full_count)?.to_vec();
        let full_probs = buffer_slice::<f32>(&buf_probs, length)?.to_vec();

        // Deterministic core output per active chunk index `c`.
        let mut full_core_bits = vec![0u16; n];
        for c in 0..full_count {
            for d in 0..width {
                let val = ((c + 1) as f32) * 0.15 - ((d % 5) as f32) * 0.07;
                full_core_bits[c * width + d] = half_bits(val);
            }
        }
        let buf_core = runtime.buffer_with(&full_core_bits);
        let buf_ema = runtime.buffer::<f32>(n);
        let buf_saved_core = runtime.buffer::<u16>(n);
        let command = runtime.queue.new_command_buffer();
        let encoder = command.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&waves.mhc_dechunk);
        bind_buffers(
            &encoder,
            &[
                (&buf_core, 0),
                (&buf_streams, 0),
                (&buf_ema, 0),
                (&buf_origins, 0),
                (&buf_probs, 0),
                (&buf_plug, 0),
                (&buf_count, 0),
                (&buf_res, 0),
                (&buf_saved_core, 0),
            ],
        );
        encode_value(&encoder, 9, &full_shape);
        encoder.dispatch_threads(thread_group(width as u64), thread_group(256));
        encoder.end_encoding();
        finish(command);
        let full_streams = buffer_slice::<u16>(&buf_streams, 4 * n)?.to_vec();

        // 2. Run across 8 irregular tiles carrying continuous state (`q_state`
        //    and `ema_state`).
        let tile_sizes = [5usize, 4, 7, 6, 11, 8, 9, 14];
        assert_eq!(tile_sizes.iter().sum::<usize>(), length);
        let q_state = runtime.buffer_with(&vec![0.0f32; width]);
        let zero_seen = runtime.buffer_with(&[0u32]);
        let one_seen = runtime.buffer_with(&[1u32]);
        let mut ema_in = runtime.buffer_with(&vec![0.0f32; width]);
        let mut ema_out = runtime.buffer_with(&vec![0.0f32; width]);
        let part_a = runtime.buffer::<f32>(EMA_PARTS * width);
        let part_b = runtime.buffer::<f32>(EMA_PARTS * width);

        let mut tiled_probs = Vec::with_capacity(length);
        let mut tiled_origins = Vec::new();
        let mut tiled_streams = Vec::with_capacity(4 * n);
        let mut cursor = 0usize;
        let mut chunk_cursor = 0usize;

        for (tile_idx, &t_len) in tile_sizes.iter().enumerate() {
            let t_n = t_len * width;
            let t_shape = HNetShape {
                batch_size: 1,
                seq_len: t_len as u32,
                d_model: width as u32,
                max_chunks: t_len as u32,
                vocab_size: 256,
                eps: EMA_EPS,
                target_ratio: 2.0,
                score: 1,
            };
            let t_input =
                runtime.buffer_with(&input_bits[cursor * width..(cursor + t_len) * width]);
            let t_proj = runtime.buffer::<f32>(2 * t_n);
            let t_res = runtime.buffer::<f32>(t_n);
            let t_probs = runtime.buffer::<f32>(t_len);
            let t_streams = runtime.buffer::<u16>(4 * t_n);
            let t_saved = runtime.buffer::<u16>(t_n);
            let t_origins = runtime.buffer::<u32>(t_len);
            let t_plug = runtime.buffer::<u32>(t_len);
            let t_count = runtime.buffer::<u32>(1);
            let seen_buf = if tile_idx == 0 { &zero_seen } else { &one_seen };

            let command = runtime.queue.new_command_buffer();
            let encoder = command.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&waves.mhc_project);
            bind_buffers(
                &encoder,
                &[(&t_input, 0), (&buf_weights, 0), (&t_proj, 0), (&t_res, 0)],
            );
            encode_value(&encoder, 4, &t_shape);
            encoder.dispatch_threads(thread_group(t_n as u64), thread_group(256));
            barrier_buffers(&encoder);
            encoder.set_compute_pipeline_state(&waves.mhc_route_state);
            bind_buffers(
                &encoder,
                &[(&t_proj, 0), (&t_probs, 0), (&q_state, 0), (seen_buf, 0)],
            );
            encode_value(&encoder, 4, &t_shape);
            encoder.dispatch_thread_groups(thread_group(t_len as u64), thread_group(32));
            barrier_buffers(&encoder);
            encoder.set_compute_pipeline_state(&waves.mhc_chunk_scan);
            bind_buffers(
                &encoder,
                &[
                    (&t_origins, 0),
                    (&t_probs, 0),
                    (&t_plug, 0),
                    (&t_count, 0),
                    (&t_proj, 0),
                    (&q_state, 0),
                    (seen_buf, 0),
                ],
            );
            encode_value(&encoder, 7, &t_shape);
            encoder.dispatch_thread_groups(thread_group(1), thread_group(256));
            barrier_buffers(&encoder);
            encoder.set_compute_pipeline_state(&waves.mhc_chunk_pack);
            bind_buffers(
                &encoder,
                &[
                    (&t_input, 0),
                    (&t_streams, 0),
                    (&t_saved, 0),
                    (&t_origins, 0),
                    (&t_count, 0),
                ],
            );
            encode_value(&encoder, 5, &t_shape);
            encoder.dispatch_threads(thread_group(t_n as u64), thread_group(256));
            encoder.end_encoding();
            finish(command);

            let c_len = buffer_slice::<u32>(&t_count, 1)?[0] as usize;
            let local_origins = buffer_slice::<u32>(&t_origins, c_len)?;
            for &o in local_origins {
                tiled_origins.push(cursor as u32 + o);
            }
            tiled_probs.extend_from_slice(buffer_slice::<f32>(&t_probs, t_len)?);

            let mut t_core_bits = vec![0u16; t_n];
            if c_len > 0 {
                t_core_bits[..c_len * width].copy_from_slice(
                    &full_core_bits[chunk_cursor * width..(chunk_cursor + c_len) * width],
                );
            }
            chunk_cursor += c_len;
            let t_core = runtime.buffer_with(&t_core_bits);
            let t_ema = runtime.buffer::<f32>(t_n);
            let t_saved_core = runtime.buffer::<u16>(t_n);

            let grid_2d = metal::MTLSize {
                width: width as u64,
                height: EMA_PARTS as u64,
                depth: 1,
            };
            let tg_2d = metal::MTLSize {
                width: (width as u64).min(256),
                height: 1,
                depth: 1,
            };
            let command = runtime.queue.new_command_buffer();
            let encoder = command.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&waves.mhc_dechunk_reduce);
            bind_buffers(
                &encoder,
                &[
                    (&t_core, 0),
                    (&t_ema, 0),
                    (&t_origins, 0),
                    (&t_probs, 0),
                    (&t_count, 0),
                    (&t_saved_core, 0),
                    (&part_a, 0),
                    (&part_b, 0),
                    (&ema_in, 0),
                ],
            );
            encode_value(&encoder, 9, &t_shape);
            encoder.dispatch_threads(grid_2d, tg_2d);
            barrier_buffers(&encoder);
            encoder.set_compute_pipeline_state(&waves.mhc_dechunk_scan);
            bind_buffers(
                &encoder,
                &[
                    (&t_core, 0),
                    (&t_ema, 0),
                    (&t_origins, 0),
                    (&t_probs, 0),
                    (&t_count, 0),
                    (&part_a, 0),
                    (&part_b, 0),
                    (&ema_in, 0),
                    (&ema_out, 0),
                ],
            );
            encode_value(&encoder, 9, &t_shape);
            encoder.dispatch_threads(grid_2d, tg_2d);
            barrier_buffers(&encoder);
            encoder.set_compute_pipeline_state(&waves.mhc_dechunk_plug);
            bind_buffers(
                &encoder,
                &[
                    (&t_streams, 0),
                    (&t_ema, 0),
                    (&t_plug, 0),
                    (&t_count, 0),
                    (&t_res, 0),
                    (&ema_in, 0),
                ],
            );
            encode_value(&encoder, 6, &t_shape);
            encoder.dispatch_threads(thread_group(t_n as u64), thread_group(256));
            encoder.end_encoding();
            finish(command);

            tiled_streams.extend_from_slice(buffer_slice::<u16>(&t_streams, 4 * t_n)?);
            std::mem::swap(&mut ema_in, &mut ema_out);
            cursor += t_len;
        }

        assert_eq!(chunk_cursor, full_count);
        assert_eq!(tiled_origins, full_origins);
        assert_eq!(tiled_probs, full_probs);
        assert_eq!(tiled_streams, full_streams);
        Ok(())
    })
}

#[test]
fn parallel_ema() -> Result<(), String> {
    metal::objc::rc::autoreleasepool(|| {
        let waves = HNetWaves::new()?;
        let runtime = &waves.runtime;
        // Choose L = 16,384, D = 64 with ~4,096 active chunks so all 32 EMA
        // partitions (`HNET_EMA_PARTS = 32`, ~128 chunks/partition when
        // `total > 256` or multiple partitions at `L = 32,768`) are active.
        let length = 32_768usize;
        let width = 64usize;
        let n = length * width;

        let mut input_bits = vec![0u16; n];
        let mut probs_vec = vec![0.0f32; length];
        let mut core_bits = vec![0u16; n];
        probs_vec[0] = 1.0;
        for t in 0..length {
            if t > 0 {
                // Every 4th token is a boundary with p in (0.55, 0.90), plus
                // aperiodic boundaries.
                probs_vec[t] = if t % 4 == 0 || t % 19 == 0 {
                    0.55 + ((t % 7) as f32) * 0.05
                } else {
                    0.15 + ((t % 5) as f32) * 0.05
                };
            }
            for d in 0..width {
                let v = (((t + d) % 23) as f32 - 11.0) * 0.05;
                input_bits[t * width + d] = half_bits(v);
                let c_v = (((t * 3 + d * 7) % 29) as f32 - 14.0) * 0.0625;
                core_bits[t * width + d] = half_bits(c_v);
            }
        }
        let residual_vec: Vec<f32> = (0..n)
            .map(|i| (((i % 17) as f32) - 8.0) * 0.03125)
            .collect();

        let shape = HNetShape {
            batch_size: 1,
            seq_len: length as u32,
            d_model: width as u32,
            max_chunks: length as u32,
            vocab_size: 256,
            eps: EMA_EPS,
            target_ratio: 2.0,
            score: 1,
        };

        // 1. Baseline execution (`mhc_chunk` + `mhc_dechunk`).
        let buf_input = runtime.buffer_with(&input_bits);
        let buf_probs = runtime.buffer_with(&probs_vec);
        let buf_core = runtime.buffer_with(&core_bits);
        let buf_res = runtime.buffer_with(&residual_vec);
        let ref_streams = runtime.buffer::<u16>(4 * n);
        let ref_saved = runtime.buffer::<u16>(n);
        let ref_origins = runtime.buffer::<u32>(length);
        let ref_plug = runtime.buffer::<u32>(length);
        let ref_count = runtime.buffer::<u32>(1);
        let ref_ema = runtime.buffer::<f32>(n);
        let ref_saved_core = runtime.buffer::<u16>(n);

        let command = runtime.queue.new_command_buffer();
        let encoder = command.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&waves.mhc_chunk);
        bind_buffers(
            &encoder,
            &[
                (&buf_input, 0),
                (&ref_streams, 0),
                (&ref_saved, 0),
                (&ref_origins, 0),
                (&buf_probs, 0),
                (&ref_plug, 0),
                (&ref_count, 0),
            ],
        );
        encode_value(&encoder, 7, &shape);
        encoder.dispatch_thread_groups(thread_group(1), thread_group(256));
        barrier_buffers(&encoder);
        encoder.set_compute_pipeline_state(&waves.mhc_dechunk);
        bind_buffers(
            &encoder,
            &[
                (&buf_core, 0),
                (&ref_streams, 0),
                (&ref_ema, 0),
                (&ref_origins, 0),
                (&buf_probs, 0),
                (&ref_plug, 0),
                (&ref_count, 0),
                (&buf_res, 0),
                (&ref_saved_core, 0),
            ],
        );
        encode_value(&encoder, 9, &shape);
        encoder.dispatch_threads(thread_group(width as u64), thread_group(256));
        encoder.end_encoding();
        finish(command);

        // 2. Parallel multi-threadgroup execution (`mhc_chunk_scan` +
        //    `mhc_chunk_pack` + `mhc_dechunk_reduce` + `mhc_dechunk_scan` +
        //    `mhc_dechunk_plug`).
        let par_streams = runtime.buffer::<u16>(4 * n);
        let par_saved = runtime.buffer::<u16>(n);
        let par_origins = runtime.buffer::<u32>(length);
        let par_plug = runtime.buffer::<u32>(length);
        let par_count = runtime.buffer::<u32>(1);
        let par_ema = runtime.buffer::<f32>(n);
        let par_saved_core = runtime.buffer::<u16>(n);
        let dummy_proj = runtime.buffer_with(&vec![0.0f32; 2 * width]);
        let q_state = runtime.buffer_with(&vec![0.0f32; width]);
        let zero_seen = runtime.buffer_with(&[0u32]);
        let ema_in = runtime.buffer_with(&vec![0.0f32; width]);
        let ema_out = runtime.buffer_with(&vec![0.0f32; width]);
        let part_a = runtime.buffer::<f32>(EMA_PARTS * width);
        let part_b = runtime.buffer::<f32>(EMA_PARTS * width);

        let grid_2d = metal::MTLSize {
            width: width as u64,
            height: EMA_PARTS as u64,
            depth: 1,
        };
        let tg_2d = metal::MTLSize {
            width: (width as u64).min(256),
            height: 1,
            depth: 1,
        };
        let command = runtime.queue.new_command_buffer();
        let encoder = command.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&waves.mhc_chunk_scan);
        bind_buffers(
            &encoder,
            &[
                (&par_origins, 0),
                (&buf_probs, 0),
                (&par_plug, 0),
                (&par_count, 0),
                (&dummy_proj, 0),
                (&q_state, 0),
                (&zero_seen, 0),
            ],
        );
        encode_value(&encoder, 7, &shape);
        encoder.dispatch_thread_groups(thread_group(1), thread_group(256));
        barrier_buffers(&encoder);
        encoder.set_compute_pipeline_state(&waves.mhc_chunk_pack);
        bind_buffers(
            &encoder,
            &[
                (&buf_input, 0),
                (&par_streams, 0),
                (&par_saved, 0),
                (&par_origins, 0),
                (&par_count, 0),
            ],
        );
        encode_value(&encoder, 5, &shape);
        encoder.dispatch_threads(thread_group(n as u64), thread_group(256));
        barrier_buffers(&encoder);
        encoder.set_compute_pipeline_state(&waves.mhc_dechunk_reduce);
        bind_buffers(
            &encoder,
            &[
                (&buf_core, 0),
                (&par_ema, 0),
                (&par_origins, 0),
                (&buf_probs, 0),
                (&par_count, 0),
                (&par_saved_core, 0),
                (&part_a, 0),
                (&part_b, 0),
                (&ema_in, 0),
            ],
        );
        encode_value(&encoder, 9, &shape);
        encoder.dispatch_threads(grid_2d, tg_2d);
        barrier_buffers(&encoder);
        encoder.set_compute_pipeline_state(&waves.mhc_dechunk_scan);
        bind_buffers(
            &encoder,
            &[
                (&buf_core, 0),
                (&par_ema, 0),
                (&par_origins, 0),
                (&buf_probs, 0),
                (&par_count, 0),
                (&part_a, 0),
                (&part_b, 0),
                (&ema_in, 0),
                (&ema_out, 0),
            ],
        );
        encode_value(&encoder, 9, &shape);
        encoder.dispatch_threads(grid_2d, tg_2d);
        barrier_buffers(&encoder);
        encoder.set_compute_pipeline_state(&waves.mhc_dechunk_plug);
        bind_buffers(
            &encoder,
            &[
                (&par_streams, 0),
                (&par_ema, 0),
                (&par_plug, 0),
                (&par_count, 0),
                (&buf_res, 0),
                (&ema_in, 0),
            ],
        );
        encode_value(&encoder, 6, &shape);
        encoder.dispatch_threads(thread_group(n as u64), thread_group(256));
        encoder.end_encoding();
        finish(command);

        let c_ref = buffer_slice::<u32>(&ref_count, 1)?[0] as usize;
        let c_par = buffer_slice::<u32>(&par_count, 1)?[0] as usize;
        assert_eq!(c_par, c_ref);
        assert!(
            c_par > 8_192,
            "expected >8192 chunks to exercise all 32 partitions, got {c_par}"
        );
        assert_eq!(
            buffer_slice::<u32>(&par_origins, c_par)?,
            buffer_slice::<u32>(&ref_origins, c_ref)?
        );
        assert_eq!(
            buffer_slice::<u32>(&par_plug, length)?,
            buffer_slice::<u32>(&ref_plug, length)?
        );
        assert_eq!(
            buffer_slice::<u16>(&par_saved_core, c_par * width)?,
            buffer_slice::<u16>(&ref_saved_core, c_ref * width)?
        );
        let ema_ref_slice = buffer_slice::<f32>(&ref_ema, c_ref * width)?;
        let ema_par_slice = buffer_slice::<f32>(&par_ema, c_par * width)?;
        let mut max_ema_err = 0.0f32;
        for (&a, &b) in ema_par_slice.iter().zip(ema_ref_slice.iter()) {
            max_ema_err = max_ema_err.max((a - b).abs());
        }
        assert!(
            max_ema_err <= 1e-6,
            "max_ema_err {max_ema_err} exceeded 1e-6 across 32 partitions"
        );
        // 3. Compare against CPU f32 reference for origin construction, plug
        //    indices, and EMA dechunking.
        let mut cpu_origins = Vec::with_capacity(c_ref);
        let mut cpu_plug = vec![0u32; length];
        for (t, &p) in probs_vec.iter().enumerate() {
            if t == 0 || p > 0.5 {
                cpu_origins.push(t as u32);
            }
            cpu_plug[t] = (cpu_origins.len() - 1) as u32;
        }
        assert_eq!(cpu_origins.len(), c_ref);
        assert_eq!(
            buffer_slice::<u32>(&par_origins, c_par)?,
            cpu_origins.as_slice()
        );
        assert_eq!(buffer_slice::<u32>(&par_plug, length)?, cpu_plug.as_slice());

        let mut cpu_ema = vec![0.0f32; c_ref * width];
        for d in 0..width {
            let mut state = 0.0f32;
            for (c, &orig) in cpu_origins.iter().enumerate() {
                let p = probs_vec[orig as usize].clamp(1e-4, 1.0 - 1e-4);
                let x = half(core_bits[c * width + d]);
                state = p.mul_add(x, (1.0 - p) * state);
                cpu_ema[c * width + d] = state;
            }
        }
        let mut max_cpu_ema_err = 0.0f32;
        for (&a, &b) in ema_par_slice.iter().zip(cpu_ema.iter()) {
            max_cpu_ema_err = max_cpu_ema_err.max((a - b).abs());
        }
        assert!(
            max_cpu_ema_err <= 1e-6,
            "max_cpu_ema_err {max_cpu_ema_err} exceeded 1e-6"
        );

        let streams_ref_slice = buffer_slice::<u16>(&ref_streams, 4 * n)?;
        let streams_par_slice = buffer_slice::<u16>(&par_streams, 4 * n)?;
        for (&a, &b) in streams_par_slice.iter().zip(streams_ref_slice.iter()) {
            assert!(
                (half(a) - half(b)).abs() <= 1e-3,
                "stream mismatch: {} vs {}",
                half(a),
                half(b)
            );
        }

        // 4. Adversarial GPU edge cases: zero-boundary tile, boundary at first
        //    byte only, boundary at final byte only, and all-boundary input.
        let edge_len = 64usize;
        let edge_n = edge_len * width;
        let edge_shape = HNetShape {
            batch_size: 1,
            seq_len: edge_len as u32,
            d_model: width as u32,
            max_chunks: edge_len as u32,
            vocab_size: 256,
            eps: EMA_EPS,
            target_ratio: 2.0,
            score: 1,
        };
        let one_seen = runtime.buffer_with(&[1u32]);
        for (mode, seen_val, expected_chunks) in [
            ("zero_boundary", 1u32, 0usize),
            ("first_only", 0u32, 1usize),
            ("last_only", 1u32, 1usize),
            ("all_boundary", 0u32, edge_len),
        ] {
            let mut p_edge = vec![0.10f32; edge_len];
            match mode {
                "zero_boundary" => {}
                "first_only" => p_edge[0] = 0.90,
                "last_only" => p_edge[edge_len - 1] = 0.90,
                "all_boundary" => p_edge.fill(0.85),
                _ => unreachable!(),
            }
            let b_probs = runtime.buffer_with(&p_edge);
            let b_origins = runtime.buffer::<u32>(edge_len);
            let b_plug = runtime.buffer::<u32>(edge_len);
            let b_count = runtime.buffer::<u32>(1);
            let b_ema = runtime.buffer::<f32>(edge_n);
            let b_streams = runtime.buffer::<u16>(4 * edge_n);
            let b_saved = runtime.buffer::<u16>(edge_n);
            let b_ema_in = runtime.buffer_with(&vec![0.25f32; width]);
            let b_ema_out = runtime.buffer_with(&vec![0.0f32; width]);
            let seen_buf = if seen_val == 0 { &zero_seen } else { &one_seen };

            let cmd = runtime.queue.new_command_buffer();
            let enc = cmd.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&waves.mhc_chunk_scan);
            bind_buffers(
                &enc,
                &[
                    (&b_origins, 0),
                    (&b_probs, 0),
                    (&b_plug, 0),
                    (&b_count, 0),
                    (&buf_res, 0),
                    (&q_state, 0),
                    (seen_buf, 0),
                ],
            );
            encode_value(&enc, 7, &edge_shape);
            enc.dispatch_thread_groups(thread_group(1), thread_group(256));
            barrier_buffers(&enc);
            enc.set_compute_pipeline_state(&waves.mhc_dechunk_reduce);
            bind_buffers(
                &enc,
                &[
                    (&buf_core, 0),
                    (&b_ema, 0),
                    (&b_origins, 0),
                    (&b_probs, 0),
                    (&b_count, 0),
                    (&b_saved, 0),
                    (&part_a, 0),
                    (&part_b, 0),
                    (&b_ema_in, 0),
                ],
            );
            encode_value(&enc, 9, &edge_shape);
            enc.dispatch_threads(grid_2d, tg_2d);
            barrier_buffers(&enc);
            enc.set_compute_pipeline_state(&waves.mhc_dechunk_scan);
            bind_buffers(
                &enc,
                &[
                    (&buf_core, 0),
                    (&b_ema, 0),
                    (&b_origins, 0),
                    (&b_probs, 0),
                    (&b_count, 0),
                    (&part_a, 0),
                    (&part_b, 0),
                    (&b_ema_in, 0),
                    (&b_ema_out, 0),
                ],
            );
            encode_value(&enc, 9, &edge_shape);
            enc.dispatch_threads(grid_2d, tg_2d);
            barrier_buffers(&enc);
            enc.set_compute_pipeline_state(&waves.mhc_dechunk_plug);
            bind_buffers(
                &enc,
                &[
                    (&b_streams, 0),
                    (&b_ema, 0),
                    (&b_plug, 0),
                    (&b_count, 0),
                    (&buf_res, 0),
                    (&b_ema_in, 0),
                ],
            );
            encode_value(&enc, 6, &edge_shape);
            enc.dispatch_threads(thread_group(edge_n as u64), thread_group(256));
            enc.end_encoding();
            finish(cmd);

            let got_chunks = buffer_slice::<u32>(&b_count, 1)?[0] as usize;
            assert_eq!(
                got_chunks, expected_chunks,
                "mode {mode} chunk count mismatch"
            );
            let ema_out_slice = buffer_slice::<f32>(&b_ema_out, width)?;
            if expected_chunks == 0 {
                assert_eq!(ema_out_slice, vec![0.25f32; width].as_slice());
            } else {
                assert!(ema_out_slice.iter().all(|v| v.is_finite()));
            }
        }
        Ok(())
    })
}

#[test]
#[ignore = "EXP-1 1M-byte hierarchy compaction and EMA dechunk benchmark on Apple M4"]
fn exp1_benchmark() -> Result<(), String> {
    metal::objc::rc::autoreleasepool(|| {
        let waves = HNetWaves::new()?;
        let runtime = &waves.runtime;

        // Part A: Benchmark 1-stage compaction + EMA dechunk + 4-stream
        // plug-back at L = 65,536, D = 512 (C ~= 16,384; 16 tiles =
        // 1,048,576 bytes).
        let length = 65_536usize;
        let width = 512usize;
        let n = length * width;
        let mut input_bits = vec![0u16; n];
        let mut probs_vec = vec![0.0f32; length];
        let mut core_bits = vec![0u16; n];
        probs_vec[0] = 1.0;
        for t in 0..length {
            if t > 0 {
                probs_vec[t] = if t % 4 == 0 { 0.75 } else { 0.20 };
            }
            for d in 0..width {
                input_bits[t * width + d] = half_bits((((t + d) % 19) as f32 - 9.0) * 0.05);
                core_bits[t * width + d] = half_bits((((t * 3 + d) % 23) as f32 - 11.0) * 0.05);
            }
        }
        let residual_vec = vec![0.125f32; n];
        let buf_input = runtime.buffer_with(&input_bits);
        let buf_probs = runtime.buffer_with(&probs_vec);
        let buf_core = runtime.buffer_with(&core_bits);
        let buf_res = runtime.buffer_with(&residual_vec);
        let buf_streams = runtime.buffer::<u16>(4 * n);
        let buf_saved = runtime.buffer::<u16>(n);
        let buf_origins = runtime.buffer::<u32>(length);
        let buf_plug = runtime.buffer::<u32>(length);
        let buf_count = runtime.buffer::<u32>(1);
        let buf_ema = runtime.buffer::<f32>(n);
        let buf_saved_core = runtime.buffer::<u16>(n);
        let buf_proj_tail = runtime.buffer_with(&vec![0.0f32; 2 * n]);
        let q_state = runtime.buffer_with(&vec![0.0f32; width]);
        let zero_seen = runtime.buffer_with(&[0u32]);
        let ema_in = runtime.buffer_with(&vec![0.0f32; width]);
        let ema_out = runtime.buffer_with(&vec![0.0f32; width]);
        let part_a = runtime.buffer::<f32>(EMA_PARTS * width);
        let part_b = runtime.buffer::<f32>(EMA_PARTS * width);

        let shape_padded = HNetShape {
            batch_size: 1,
            seq_len: length as u32,
            d_model: width as u32,
            max_chunks: length as u32,
            vocab_size: 256,
            eps: EMA_EPS,
            target_ratio: 2.0,
            score: 1,
        };
        let mut shape_compact = shape_padded;
        shape_compact.max_chunks = 0;

        let run_baseline = || {
            let t0 = std::time::Instant::now();
            let command = runtime.queue.new_command_buffer();
            let encoder = command.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&waves.mhc_chunk);
            bind_buffers(
                &encoder,
                &[
                    (&buf_input, 0),
                    (&buf_streams, 0),
                    (&buf_saved, 0),
                    (&buf_origins, 0),
                    (&buf_probs, 0),
                    (&buf_plug, 0),
                    (&buf_count, 0),
                ],
            );
            encode_value(&encoder, 7, &shape_padded);
            encoder.dispatch_thread_groups(thread_group(1), thread_group(256));
            barrier_buffers(&encoder);
            encoder.set_compute_pipeline_state(&waves.mhc_dechunk);
            bind_buffers(
                &encoder,
                &[
                    (&buf_core, 0),
                    (&buf_streams, 0),
                    (&buf_ema, 0),
                    (&buf_origins, 0),
                    (&buf_probs, 0),
                    (&buf_plug, 0),
                    (&buf_count, 0),
                    (&buf_res, 0),
                    (&buf_saved_core, 0),
                ],
            );
            encode_value(&encoder, 9, &shape_padded);
            encoder.dispatch_threads(thread_group(width as u64), thread_group(256));
            encoder.end_encoding();
            finish(command);
            let wall_ms = t0.elapsed().as_secs_f64() * 1000.0;
            let gpu_ms = crate::apple_gpu::gpu_seconds(command).unwrap_or(0.0) * 1000.0;
            (wall_ms, gpu_ms)
        };

        let grid_2d = metal::MTLSize {
            width: width as u64,
            height: EMA_PARTS as u64,
            depth: 1,
        };
        let tg_2d = metal::MTLSize {
            width: 256,
            height: 1,
            depth: 1,
        };

        let run_parallel = |shape: &HNetShape| {
            let t0 = std::time::Instant::now();
            let command = runtime.queue.new_command_buffer();
            let encoder = command.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&waves.mhc_chunk_scan);
            bind_buffers(
                &encoder,
                &[
                    (&buf_origins, 0),
                    (&buf_probs, 0),
                    (&buf_plug, 0),
                    (&buf_count, 0),
                    (&buf_proj_tail, 0),
                    (&q_state, 0),
                    (&zero_seen, 0),
                ],
            );
            encode_value(&encoder, 7, shape);
            encoder.dispatch_thread_groups(thread_group(1), thread_group(256));
            barrier_buffers(&encoder);
            encoder.set_compute_pipeline_state(&waves.mhc_chunk_pack);
            bind_buffers(
                &encoder,
                &[
                    (&buf_input, 0),
                    (&buf_streams, 0),
                    (&buf_saved, 0),
                    (&buf_origins, 0),
                    (&buf_count, 0),
                ],
            );
            encode_value(&encoder, 5, shape);
            encoder.dispatch_threads(thread_group(n as u64), thread_group(256));
            barrier_buffers(&encoder);
            encoder.set_compute_pipeline_state(&waves.mhc_dechunk_reduce);
            bind_buffers(
                &encoder,
                &[
                    (&buf_core, 0),
                    (&buf_ema, 0),
                    (&buf_origins, 0),
                    (&buf_probs, 0),
                    (&buf_count, 0),
                    (&buf_saved_core, 0),
                    (&part_a, 0),
                    (&part_b, 0),
                    (&ema_in, 0),
                ],
            );
            encode_value(&encoder, 9, shape);
            encoder.dispatch_threads(grid_2d, tg_2d);
            barrier_buffers(&encoder);
            encoder.set_compute_pipeline_state(&waves.mhc_dechunk_scan);
            bind_buffers(
                &encoder,
                &[
                    (&buf_core, 0),
                    (&buf_ema, 0),
                    (&buf_origins, 0),
                    (&buf_probs, 0),
                    (&buf_count, 0),
                    (&part_a, 0),
                    (&part_b, 0),
                    (&ema_in, 0),
                    (&ema_out, 0),
                ],
            );
            encode_value(&encoder, 9, shape);
            encoder.dispatch_threads(grid_2d, tg_2d);
            barrier_buffers(&encoder);
            encoder.set_compute_pipeline_state(&waves.mhc_dechunk_plug);
            bind_buffers(
                &encoder,
                &[
                    (&buf_streams, 0),
                    (&buf_ema, 0),
                    (&buf_plug, 0),
                    (&buf_count, 0),
                    (&buf_res, 0),
                    (&ema_in, 0),
                ],
            );
            encode_value(&encoder, 6, shape);
            encoder.dispatch_threads(thread_group(n as u64), thread_group(256));
            encoder.end_encoding();
            finish(command);
            let wall_ms = t0.elapsed().as_secs_f64() * 1000.0;
            let gpu_ms = crate::apple_gpu::gpu_seconds(command).unwrap_or(0.0) * 1000.0;
            (wall_ms, gpu_ms)
        };

        // Warmup
        let _ = run_baseline();
        let _ = run_parallel(&shape_padded);
        let _ = run_parallel(&shape_compact);

        let mut base_wall = Vec::with_capacity(11);
        let mut base_gpu = Vec::with_capacity(11);
        let mut par_padded_wall = Vec::with_capacity(11);
        let mut par_padded_gpu = Vec::with_capacity(11);
        let mut par_compact_wall = Vec::with_capacity(11);
        let mut par_compact_gpu = Vec::with_capacity(11);
        for _ in 0..11 {
            let (bw, bg) = run_baseline();
            base_wall.push(bw);
            base_gpu.push(bg);
            let (pw, pg) = run_parallel(&shape_padded);
            par_padded_wall.push(pw);
            par_padded_gpu.push(pg);
            let (cw, cg) = run_parallel(&shape_compact);
            par_compact_wall.push(cw);
            par_compact_gpu.push(cg);
        }
        base_wall.sort_by(|a, b| a.partial_cmp(b).unwrap());
        base_gpu.sort_by(|a, b| a.partial_cmp(b).unwrap());
        par_padded_wall.sort_by(|a, b| a.partial_cmp(b).unwrap());
        par_padded_gpu.sort_by(|a, b| a.partial_cmp(b).unwrap());
        par_compact_wall.sort_by(|a, b| a.partial_cmp(b).unwrap());
        par_compact_gpu.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let chunks_64k = buffer_slice::<u32>(&buf_count, 1)?[0] as usize;
        println!(
            "EXP1_TILE64K rows={} chunks={} d_model={} base_wall_p50_ms={:.3} base_wall_p95_ms={:.3} base_wall_max_ms={:.3} base_gpu_p50_ms={:.3} par_padded_wall_p50_ms={:.3} par_padded_wall_p95_ms={:.3} par_padded_gpu_p50_ms={:.3} par_compact_wall_p50_ms={:.3} par_compact_wall_p95_ms={:.3} par_compact_wall_max_ms={:.3} par_compact_gpu_p50_ms={:.3}",
            length,
            chunks_64k,
            width,
            base_wall[5],
            base_wall[9],
            base_wall[10],
            base_gpu[5],
            par_padded_wall[5],
            par_padded_wall[9],
            par_padded_gpu[5],
            par_compact_wall[5],
            par_compact_wall[9],
            par_compact_wall[10],
            par_compact_gpu[5],
        );
        println!(
            "EXP1_1M_SCALED_16X64K base_1m_ms={:.3} par_padded_1m_ms={:.3} par_compact_1m_ms={:.3} speedup_compact={:.2}x",
            base_wall[5] * 16.0,
            par_padded_wall[5] * 16.0,
            par_compact_wall[5] * 16.0,
            base_wall[5] / par_compact_wall[5]
        );

        // Part B: Direct 1,048,576-byte parallel SIMD prefix scan
        // (`mhc_chunk_scan`) and continuous 16 x 65,536 =
        // 1,048,576-byte stateful execution over 16 distinct tiles.
        let len_1m = 1_048_576usize;
        let mut probs_1m = vec![0.18f32; len_1m];
        probs_1m[0] = 1.0;
        for t in 1..len_1m {
            // Aperiodic boundary schedule across 1,048,576 bytes (~25%
            // boundaries)
            let h = (t as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ ((t as u64) >> 11);
            if h % 4 == 0 {
                probs_1m[t] = 0.55 + ((h % 40) as f32) * 0.01;
            } else {
                probs_1m[t] = 0.08 + ((h % 35) as f32) * 0.01;
            }
        }
        let buf_probs_1m = runtime.buffer_with(&probs_1m);
        let buf_origins_1m = runtime.buffer::<u32>(len_1m);
        let buf_plug_1m = runtime.buffer::<u32>(len_1m);
        let buf_count_1m = runtime.buffer::<u32>(1);
        let shape_scan_1m = HNetShape {
            batch_size: 1,
            seq_len: len_1m as u32,
            d_model: 0, // skip projected tail copy in standalone 1M scan
            max_chunks: len_1m as u32,
            vocab_size: 256,
            eps: EMA_EPS,
            target_ratio: 2.0,
            score: 1,
        };
        let mut scan_1m_wall = Vec::with_capacity(11);
        let mut scan_1m_gpu = Vec::with_capacity(11);
        for iter in 0..12 {
            let t0 = std::time::Instant::now();
            let command = runtime.queue.new_command_buffer();
            let encoder = command.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&waves.mhc_chunk_scan);
            bind_buffers(
                &encoder,
                &[
                    (&buf_origins_1m, 0),
                    (&buf_probs_1m, 0),
                    (&buf_plug_1m, 0),
                    (&buf_count_1m, 0),
                    (&buf_proj_tail, 0),
                    (&q_state, 0),
                    (&zero_seen, 0),
                ],
            );
            encode_value(&encoder, 7, &shape_scan_1m);
            encoder.dispatch_thread_groups(thread_group(1), thread_group(256));
            encoder.end_encoding();
            finish(command);
            if iter > 0 {
                scan_1m_wall.push(t0.elapsed().as_secs_f64() * 1000.0);
                scan_1m_gpu.push(crate::apple_gpu::gpu_seconds(command).unwrap_or(0.0) * 1000.0);
            }
        }
        scan_1m_wall.sort_by(|a, b| a.partial_cmp(b).unwrap());
        scan_1m_gpu.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let count_1m = buffer_slice::<u32>(&buf_count_1m, 1)?[0] as usize;
        let ref_origins_1m = buffer_slice::<u32>(&buf_origins_1m, count_1m)?.to_vec();
        let ref_plug_1m = buffer_slice::<u32>(&buf_plug_1m, len_1m)?.to_vec();

        // Verify 16-tile (16 x 65,536 = 1,048,576) continuous parity against
        // the unbroken 1,048,576-byte reference scan and CPU FP32 EMA
        // recurrence.
        let one_seen = runtime.buffer_with(&[1u32]);
        let mut shape_tile_scan = shape_compact;
        shape_tile_scan.d_model = 0;
        let mut global_chunk_offset = 0usize;
        let mut cpu_ema_carry = vec![0.0f32; width];
        let mut max_ema_err_1m = 0.0f32;
        slice_mut::<f32>(&ema_in, width)?.fill(0.0);
        slice_mut::<f32>(&ema_out, width)?.fill(0.0);
        for tile_idx in 0..16usize {
            let tile_base = tile_idx * length;
            let prob_byte_offset = (tile_base * std::mem::size_of::<f32>()) as u64;
            let seen_buf = if tile_idx == 0 { &zero_seen } else { &one_seen };
            let (cur_ema_in, cur_ema_out) = if tile_idx % 2 == 0 {
                (&ema_in, &ema_out)
            } else {
                (&ema_out, &ema_in)
            };
            let command = runtime.queue.new_command_buffer();
            let encoder = command.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&waves.mhc_chunk_scan);
            bind_buffers(
                &encoder,
                &[
                    (&buf_origins, 0),
                    (&buf_probs_1m, prob_byte_offset),
                    (&buf_plug, 0),
                    (&buf_count, 0),
                    (&buf_proj_tail, 0),
                    (&q_state, 0),
                    (seen_buf, 0),
                ],
            );
            encode_value(&encoder, 7, &shape_tile_scan);
            encoder.dispatch_thread_groups(thread_group(1), thread_group(256));
            barrier_buffers(&encoder);
            encoder.set_compute_pipeline_state(&waves.mhc_dechunk_reduce);
            bind_buffers(
                &encoder,
                &[
                    (&buf_core, 0),
                    (&buf_ema, 0),
                    (&buf_origins, 0),
                    (&buf_probs_1m, prob_byte_offset),
                    (&buf_count, 0),
                    (&buf_saved_core, 0),
                    (&part_a, 0),
                    (&part_b, 0),
                    (cur_ema_in, 0),
                ],
            );
            encode_value(&encoder, 9, &shape_compact);
            encoder.dispatch_threads(grid_2d, tg_2d);
            barrier_buffers(&encoder);
            encoder.set_compute_pipeline_state(&waves.mhc_dechunk_scan);
            bind_buffers(
                &encoder,
                &[
                    (&buf_core, 0),
                    (&buf_ema, 0),
                    (&buf_origins, 0),
                    (&buf_probs_1m, prob_byte_offset),
                    (&buf_count, 0),
                    (&part_a, 0),
                    (&part_b, 0),
                    (cur_ema_in, 0),
                    (cur_ema_out, 0),
                ],
            );
            encode_value(&encoder, 9, &shape_compact);
            encoder.dispatch_threads(grid_2d, tg_2d);
            encoder.end_encoding();
            finish(command);

            let tile_chunks = buffer_slice::<u32>(&buf_count, 1)?[0] as usize;
            let tile_origins = buffer_slice::<u32>(&buf_origins, tile_chunks)?;
            let tile_plug = buffer_slice::<u32>(&buf_plug, length)?;
            for (c, &orig) in tile_origins.iter().enumerate() {
                assert_eq!(
                    (orig as usize) + tile_base,
                    ref_origins_1m[global_chunk_offset + c] as usize,
                    "tile {tile_idx} chunk {c} origin mismatch"
                );
            }
            for (t, &pl) in tile_plug.iter().enumerate() {
                let expected_global = ref_plug_1m[tile_base + t] as usize;
                if pl == u32::MAX {
                    assert_eq!(
                        expected_global + 1,
                        global_chunk_offset,
                        "tile {tile_idx} pos {t} carry plug mismatch"
                    );
                } else {
                    assert_eq!(
                        (pl as usize) + global_chunk_offset,
                        expected_global,
                        "tile {tile_idx} pos {t} plug mismatch"
                    );
                }
            }
            let gpu_ema = buffer_slice::<f32>(&buf_ema, tile_chunks * width)?;
            for c in 0..tile_chunks {
                let orig = tile_origins[c] as usize;
                let p = probs_1m[tile_base + orig].clamp(EMA_EPS, 1.0 - EMA_EPS);
                for d in 0..width {
                    let h = half(core_bits[c * width + d]);
                    let s = p.mul_add(h, (1.0 - p) * cpu_ema_carry[d]);
                    cpu_ema_carry[d] = s;
                    let err = (gpu_ema[c * width + d] - s).abs();
                    if err > max_ema_err_1m {
                        max_ema_err_1m = err;
                    }
                }
            }
            global_chunk_offset += tile_chunks;
        }
        assert_eq!(global_chunk_offset, count_1m);
        assert!(
            max_ema_err_1m <= 1e-5,
            "1M continuous EMA max_abs_err {max_ema_err_1m} > 1e-5"
        );

        // Execute a continuous 1,048,576-byte (16 x 65,536) stateful hierarchy
        // pass in a single command buffer carrying `q_state` and
        // ping-ponging `(ema_in, ema_out)` across all 16 distinct tiles.
        let mut cont_1m_wall = Vec::with_capacity(7);
        let mut cont_1m_gpu = Vec::with_capacity(7);
        for iter in 0..8 {
            slice_mut::<f32>(&ema_in, width)?.fill(0.0);
            slice_mut::<f32>(&ema_out, width)?.fill(0.0);
            let t0 = std::time::Instant::now();
            let command = runtime.queue.new_command_buffer();
            let encoder = command.new_compute_command_encoder();
            for tile_idx in 0..16usize {
                let prob_byte_offset = ((tile_idx * length) * std::mem::size_of::<f32>()) as u64;
                let seen_buf = if tile_idx == 0 { &zero_seen } else { &one_seen };
                let (cur_ema_in, cur_ema_out) = if tile_idx % 2 == 0 {
                    (&ema_in, &ema_out)
                } else {
                    (&ema_out, &ema_in)
                };
                encoder.set_compute_pipeline_state(&waves.mhc_chunk_scan);
                bind_buffers(
                    &encoder,
                    &[
                        (&buf_origins, 0),
                        (&buf_probs_1m, prob_byte_offset),
                        (&buf_plug, 0),
                        (&buf_count, 0),
                        (&buf_proj_tail, 0),
                        (&q_state, 0),
                        (seen_buf, 0),
                    ],
                );
                encode_value(&encoder, 7, &shape_compact);
                encoder.dispatch_thread_groups(thread_group(1), thread_group(256));
                barrier_buffers(&encoder);
                encoder.set_compute_pipeline_state(&waves.mhc_chunk_pack);
                bind_buffers(
                    &encoder,
                    &[
                        (&buf_input, 0),
                        (&buf_streams, 0),
                        (&buf_saved, 0),
                        (&buf_origins, 0),
                        (&buf_count, 0),
                    ],
                );
                encode_value(&encoder, 5, &shape_compact);
                encoder.dispatch_threads(thread_group(n as u64), thread_group(256));
                barrier_buffers(&encoder);
                encoder.set_compute_pipeline_state(&waves.mhc_dechunk_reduce);
                bind_buffers(
                    &encoder,
                    &[
                        (&buf_core, 0),
                        (&buf_ema, 0),
                        (&buf_origins, 0),
                        (&buf_probs_1m, prob_byte_offset),
                        (&buf_count, 0),
                        (&buf_saved_core, 0),
                        (&part_a, 0),
                        (&part_b, 0),
                        (cur_ema_in, 0),
                    ],
                );
                encode_value(&encoder, 9, &shape_compact);
                encoder.dispatch_threads(grid_2d, tg_2d);
                barrier_buffers(&encoder);
                encoder.set_compute_pipeline_state(&waves.mhc_dechunk_scan);
                bind_buffers(
                    &encoder,
                    &[
                        (&buf_core, 0),
                        (&buf_ema, 0),
                        (&buf_origins, 0),
                        (&buf_probs_1m, prob_byte_offset),
                        (&buf_count, 0),
                        (&part_a, 0),
                        (&part_b, 0),
                        (cur_ema_in, 0),
                        (cur_ema_out, 0),
                    ],
                );
                encode_value(&encoder, 9, &shape_compact);
                encoder.dispatch_threads(grid_2d, tg_2d);
                barrier_buffers(&encoder);
                encoder.set_compute_pipeline_state(&waves.mhc_dechunk_plug);
                bind_buffers(
                    &encoder,
                    &[
                        (&buf_streams, 0),
                        (&buf_ema, 0),
                        (&buf_plug, 0),
                        (&buf_count, 0),
                        (&buf_res, 0),
                        (cur_ema_in, 0),
                    ],
                );
                encode_value(&encoder, 6, &shape_compact);
                encoder.dispatch_threads(thread_group(n as u64), thread_group(256));
                barrier_buffers(&encoder);
            }
            encoder.end_encoding();
            finish(command);
            if iter > 0 {
                cont_1m_wall.push(t0.elapsed().as_secs_f64() * 1000.0);
                cont_1m_gpu.push(crate::apple_gpu::gpu_seconds(command).unwrap_or(0.0) * 1000.0);
            }
        }
        cont_1m_wall.sort_by(|a, b| a.partial_cmp(b).unwrap());
        cont_1m_gpu.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let working_set_bytes = buf_input.length()
            + buf_probs.length()
            + buf_core.length()
            + buf_res.length()
            + buf_streams.length()
            + buf_saved.length()
            + buf_origins.length()
            + buf_plug.length()
            + buf_count.length()
            + buf_ema.length()
            + buf_saved_core.length()
            + q_state.length()
            + ema_in.length()
            + ema_out.length()
            + part_a.length()
            + part_b.length();
        let dev_alloc_bytes = runtime.device.current_allocated_size();
        println!(
            "EXP1_CONTINUOUS_1M total_bytes={} total_chunks={} max_ema_err={:.3e} scan_1m_wall_p50_ms={:.3} scan_1m_wall_p95_ms={:.3} scan_1m_wall_max_ms={:.3} scan_1m_gpu_p50_ms={:.3} cont_1m_wall_p50_ms={:.3} cont_1m_wall_p95_ms={:.3} cont_1m_wall_max_ms={:.3} cont_1m_gpu_p50_ms={:.3} working_set_mib={:.2} dev_alloc_mib={:.2}",
            len_1m,
            count_1m,
            max_ema_err_1m,
            scan_1m_wall[5],
            scan_1m_wall[9],
            scan_1m_wall[10],
            scan_1m_gpu[5],
            cont_1m_wall[3],
            cont_1m_wall[5],
            cont_1m_wall[6],
            cont_1m_gpu[3],
            (working_set_bytes as f64) / (1024.0 * 1024.0),
            (dev_alloc_bytes as f64) / (1024.0 * 1024.0),
        );
        Ok(())
    })
}
