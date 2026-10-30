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
        }
        Ok(())
    })
}
