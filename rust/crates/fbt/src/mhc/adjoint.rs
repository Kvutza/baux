//! Independent double-precision predictor reference and production GPU VJP.
use super::mhc_grad;
use crate::apple_gpu::{Runtime, bind_buffers, buffer_slice, encode_value, thread_group};
use crate::tensor::{WIDTH, complete, half_bits};

struct Case {
    values: [Vec<f64>; 4],
    adjoint: Vec<f64>,
}

fn sigmoid(value: f64) -> f64 {
    1.0 / (1.0 + (-value).exp())
}

impl Case {
    fn new(zero: bool) -> Self {
        let data = |count, salt, divisor| {
            (0..count)
                .map(|index| {
                    let bits = crate::hash::splitmix64(index as u64 + salt);
                    f64::from((bits % 31) as i32 - 15) / divisor
                })
                .collect()
        };
        Self {
            values: [
                data(2048, 3, 32.0),
                data(2048 * 24, 19, 1024.0),
                data(24, 37, 32.0),
                if zero {
                    vec![0.0, 0.0, 0.0, 0.375]
                } else {
                    vec![0.125, -0.0625, 0.03125, 0.375]
                },
            ],
            adjoint: data(24, 71, 16.0),
        }
    }

    fn coefficients(&self) -> [f64; 24] {
        let [streams, predictor, bias, control] = &self.values;
        let norm = (streams.iter().map(|x| x * x).sum::<f64>() / 2048.0 + 1e-5).sqrt();
        let raw: [f64; 24] = std::array::from_fn(|output| {
            let group = if output < 4 {
                0
            } else if output < 20 {
                1
            } else {
                2
            };
            let dot = streams
                .iter()
                .enumerate()
                .map(|(input, x)| x * predictor[input * 24 + output])
                .sum::<f64>();
            bias[output] + control[group] * dot / norm
        });
        let mut result = [0.0; 24];
        let total: f64 = raw[..4].iter().copied().map(sigmoid).sum();
        for i in 0..4 {
            result[i] = sigmoid(raw[i]) / total;
            result[20 + i] = 2.0 * sigmoid(raw[20 + i]);
        }
        let maximum = raw[4..20].iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let mut matrix: [[f64; 4]; 4] = std::array::from_fn(|r| {
            std::array::from_fn(|c| (raw[4 + r * 4 + c] - maximum).max(-80.0).exp())
        });
        for _ in 0..20 {
            for row in &mut matrix {
                let sum: f64 = row.iter().sum();
                for value in row {
                    *value /= sum;
                }
            }
            for col in 0..4 {
                let sum: f64 = matrix.iter().map(|row| row[col]).sum();
                for row in &mut matrix {
                    row[col] /= sum;
                }
            }
        }
        let lambda = control[3].clamp(0.0, 1.0);
        for r in 0..4 {
            for c in 0..4 {
                result[4 + r * 4 + c] = (1.0 - lambda) * f64::from(r == c) + lambda * matrix[r][c];
            }
        }
        result
    }

    fn loss(&self) -> f64 {
        self.coefficients()
            .iter()
            .zip(&self.adjoint)
            .map(|(a, b)| a * b)
            .sum()
    }

    fn check(&mut self, actual: &[Vec<f32>]) {
        for family in 0..4 {
            let count = self.values[family].len();
            let indices: Vec<usize> = if family < 2 {
                (0..32).map(|i| (i * 1543 + 17) % count).collect()
            } else {
                (0..count).collect()
            };
            for step in [1e-3, 3e-4, 1e-4] {
                for &index in &indices {
                    let original = self.values[family][index];
                    self.values[family][index] = original + step;
                    let plus = self.loss();
                    self.values[family][index] = original - step;
                    let minus = self.loss();
                    self.values[family][index] = original;
                    close(
                        f64::from(actual[family][index]) - 0.125,
                        (plus - minus) / (2.0 * step),
                        family,
                        index,
                    );
                }
                let original = self.values[family].clone();
                let direction: Vec<f64> = (0..count)
                    .map(|i| {
                        if crate::hash::splitmix64(i as u64 + 101) & 1 == 0 {
                            1.0
                        } else {
                            -1.0
                        }
                    })
                    .collect();
                for (value, d) in self.values[family].iter_mut().zip(&direction) {
                    *value += step * d;
                }
                let plus = self.loss();
                for ((value, x), d) in self.values[family]
                    .iter_mut()
                    .zip(&original)
                    .zip(&direction)
                {
                    *value = x - step * d;
                }
                let minus = self.loss();
                self.values[family] = original;
                let predicted: f64 = actual[family]
                    .iter()
                    .zip(&direction)
                    .map(|(&g, d)| (f64::from(g) - 0.125) * d)
                    .sum();
                close(predicted, (plus - minus) / (2.0 * step), family, count);
            }
        }
    }
}

fn close(actual: f64, expected: f64, family: usize, index: usize) {
    assert!(
        actual.is_finite() && (actual - expected).abs() <= 3e-5 * (1.0 + expected.abs()),
        "family={family} index={index} actual={actual} expected={expected}"
    );
}

fn qualify(runtime: &Runtime, zero: bool) -> Result<(), String> {
    let mut case = Case::new(zero);
    let inputs: Vec<_> = case
        .values
        .iter()
        .map(|values| {
            runtime.buffer_with(
                &values
                    .iter()
                    .map(|&x| half_bits(x as f32))
                    .collect::<Vec<_>>(),
            )
        })
        .collect();
    let gradients: Vec<_> = case
        .values
        .iter()
        .map(|values| runtime.buffer_with(&vec![0.125f32; values.len()]))
        .collect();
    let adjoint = runtime.buffer_with(&case.adjoint.iter().map(|&x| x as f32).collect::<Vec<_>>());
    let coefficients = runtime.buffer::<f32>(24);
    let forward = runtime.precise(
        include_str!("../moe.metal"),
        "predictor reference",
        "fbt_mhc_predict_rows",
    )?;
    let backward = mhc_grad::Kernels::new(runtime)?;
    let shape = mhc_grad::Shape {
        rows: 1,
        width: WIDTH,
        architecture: 2,
        reserved: 0,
    };
    let command = runtime.queue.new_command_buffer();
    let encoder = command.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&forward);
    bind_buffers(
        &encoder,
        &[
            (&inputs[0], 0),
            (&inputs[1], 0),
            (&inputs[2], 0),
            (&inputs[3], 0),
            (&coefficients, 0),
        ],
    );
    encode_value(&encoder, 5, &shape);
    encoder.dispatch_thread_groups(thread_group(1), thread_group(128));
    backward.predict(
        &encoder,
        shape,
        mhc_grad::Predict {
            streams: (&inputs[0], 0),
            predictor: (&inputs[1], 0),
            bias: (&inputs[2], 0),
            control: (&inputs[3], 0),
            coefficient_grad: (&adjoint, 0),
            stream_grad: (&gradients[0], 0),
            predictor_grad: (&gradients[1], 0),
            bias_grad: (&gradients[2], 0),
            control_grad: (&gradients[3], 0),
        },
    );
    encoder.end_encoding();
    complete(command)?;
    for (index, (&actual, expected)) in buffer_slice::<f32>(&coefficients, 24)?
        .iter()
        .zip(case.coefficients())
        .enumerate()
    {
        close(f64::from(actual), expected, 4, index);
    }
    let actual: Result<Vec<_>, String> = gradients
        .iter()
        .zip(&case.values)
        .map(|(buffer, values)| {
            buffer_slice::<f32>(buffer, values.len()).map(|slice| slice.to_vec())
        })
        .collect();
    case.check(&actual?);
    println!(
        "MHC_ADJOINT zero_controls={zero} coordinates=92 directions=4 step_sizes=3 tolerance=3e-5"
    );
    Ok(())
}

#[test]
fn predictor_adjoint() -> Result<(), String> {
    metal::objc::rc::autoreleasepool(|| {
        let runtime = Runtime::shared()?;
        qualify(&runtime, false)?;
        qualify(&runtime, true)
    })
}
