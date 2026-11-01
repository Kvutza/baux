//! A sparse full-width MoE fixture, retaining the production packing and banks.
//! The reference anchors casts at saved primals and uses smooth f64 SwiGLU.
use super::super::{complete, decode_half, resources::half_bits, routing_grad};
use super::*;
use crate::apple_gpu::{buffer_slice, slice_mut};

const ROWS: usize = 65;
const INPUT: [usize; 2] = [0, 17];
const OUTPUT: [usize; 2] = [0, 511];
const HIDDEN: [usize; 2] = [0, 255];
const ROUTES: [usize; 3] = [0, 7, 63];
const BANKS: [usize; 4] = [0, 1, 8, 64];
const ROUTER: usize = ROWS * 2;
const GATE: usize = ROUTER + 6;
const DOWN: usize = GATE + 32;
const PARAMETERS: usize = DOWN + 16;
const ROUTER_SIZE: usize = WIDTH as usize * ROUTED_EXPERTS as usize;
const DOWN_BASE: usize = ROUTER_SIZE + GATE_PARAMETERS;
const WEIGHTS: usize = DOWN_BASE + DOWN_PARAMETERS;

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}
fn activation(g: f64, u: f64) -> f64 {
    g * sigmoid(g) * u
}
fn normalize(logits: [f64; 3]) -> [f64; 3] {
    let p = logits.map(sigmoid);
    let sum: f64 = p.iter().sum();
    p.map(|x| x / sum)
}

fn location(index: usize) -> (bool, usize) {
    if index < ROUTER {
        return (true, index / 2 * WIDTH as usize + INPUT[index % 2]);
    }
    if index < GATE {
        let index = index - ROUTER;
        return (
            false,
            INPUT[index % 2] * ROUTED_EXPERTS as usize + ROUTES[index / 2],
        );
    }
    if index < DOWN {
        let index = index - GATE;
        let channel = HIDDEN[index % 2] + index % 4 / 2 * ROUTED_WIDTH as usize;
        return (
            false,
            ROUTER_SIZE
                + (BANKS[index / 8] * WIDTH as usize + INPUT[index % 8 / 4]) * MOE_GATEUP as usize
                + channel,
        );
    }
    let index = index - DOWN;
    (
        false,
        DOWN_BASE
            + (BANKS[index / 4] * ROUTED_WIDTH as usize + HIDDEN[index % 4 / 2]) * WIDTH as usize
            + OUTPUT[index % 2],
    )
}

fn fixture() -> Vec<f64> {
    (0..PARAMETERS)
        .map(|i| {
            if i < ROUTER {
                ((i * 3 % 5) + 2) as f64 / 8.0
            } else if i < GATE {
                ((i % 5) + 1) as f64 / 4.0
            } else {
                ((i * 7 % 13) as f64 - 6.0) / 16.0
            }
        })
        .collect()
}

fn dot(x: &[f64], row: usize, weights: usize, stride: usize) -> f64 {
    (0..2)
        .map(|i| x[row * 2 + i] * x[weights + i * stride])
        .sum()
}

struct Frame {
    logits: [f64; 3],
    probability: [f64; 3],
    projection: [[f64; 4]; 4],
    activation: [[f64; 2]; 4],
    output: [[f64; 2]; 4],
}

impl Frame {
    fn read(saved: &FineGrainedMoeBuffers, row: usize, x: &[f64]) -> Result<Self, String> {
        let ids = buffer_slice::<u32>(&saved.route_experts, ROWS * 3)?;
        let packed = buffer_slice::<u32>(&saved.packed_rows, ROWS * 3)?;
        let route_weights = buffer_slice::<u16>(&saved.route_weights, ROWS * 3)?;
        let score = buffer_slice::<u16>(&saved.scores, ROWS * ROUTED_EXPERTS as usize)?;
        let mut result = Self {
            logits: [0.0; 3],
            probability: [0.0; 3],
            projection: [[0.0; 4]; 4],
            activation: [[0.0; 2]; 4],
            output: [[0.0; 2]; 4],
        };
        for expert in 0..4 {
            let (index, a, y) = if expert == 0 {
                (row, &saved.shared_activation, &saved.shared_output)
            } else {
                let slot = ids[row * 3..row * 3 + 3]
                    .iter()
                    .position(|&id| id as usize == ROUTES[expert - 1])
                    .ok_or("fixture selected an unexpected expert")?;
                result.probability[expert - 1] =
                    f64::from(decode_half(route_weights[row * 3 + slot]));
                result.logits[expert - 1] = f64::from(decode_half(
                    score[row * ROUTED_EXPERTS as usize + ROUTES[expert - 1]],
                ));
                (
                    packed[row * 3 + slot] as usize,
                    &saved.routed_activation,
                    &saved.routed_output,
                )
            };
            let activations = buffer_slice::<u16>(a, (index + 1) * ACTIVATION_STRIDE as usize)?;
            let outputs = buffer_slice::<u16>(y, (index + 1) * WIDTH as usize)?;
            for channel in 0..4 {
                result.projection[expert][channel] =
                    f64::from(decode_half(half_bits(
                        dot(x, row, GATE + expert * 8 + channel, 4) as f32,
                    )));
            }
            for h in 0..2 {
                result.activation[expert][h] = f64::from(decode_half(
                    activations[index * ACTIVATION_STRIDE as usize + HIDDEN[h]],
                ));
                result.output[expert][h] =
                    f64::from(decode_half(outputs[index * WIDTH as usize + OUTPUT[h]]));
                let p = result.projection[expert];
                let smooth = activation(p[h], p[2 + h]);
                assert!(
                    (result.activation[expert][h] - smooth).abs() < 0.002 * smooth.abs() + 2e-7
                );
            }
            for c in 0..2 {
                let expected: f64 = (0..2)
                    .map(|h| result.activation[expert][h] * x[DOWN + expert * 4 + h * 2 + c])
                    .sum();
                assert_eq!(
                    half_bits(result.output[expert][c] as f32),
                    half_bits(expected as f32),
                    "down projection row={row} expert={expert} column={c}"
                );
            }
        }
        Ok(result)
    }

    fn evaluate(&self, original: &[f64], x: &[f64], row: usize) -> [f64; 2] {
        let logits = std::array::from_fn(|e| {
            self.logits[e] + dot(x, row, ROUTER + e * 2, 1) - dot(original, row, ROUTER + e * 2, 1)
        });
        let p = normalize(logits);
        let base = normalize(self.logits);
        let probability: [f64; 3] = std::array::from_fn(|e| self.probability[e] + p[e] - base[e]);
        let mut result = [0.0; 2];
        for expert in 0..4 {
            let projection: [f64; 4] = std::array::from_fn(|c| {
                self.projection[expert][c] + dot(x, row, GATE + expert * 8 + c, 4)
                    - dot(original, row, GATE + expert * 8 + c, 4)
            });
            let a: [f64; 2] = std::array::from_fn(|h| {
                self.activation[expert][h] + activation(projection[h], projection[2 + h])
                    - activation(self.projection[expert][h], self.projection[expert][2 + h])
            });
            for c in 0..2 {
                let value = self.output[expert][c]
                    + (0..2)
                        .map(|h| {
                            let index = DOWN + expert * 4 + h * 2 + c;
                            a[h] * x[index] - self.activation[expert][h] * original[index]
                        })
                        .sum::<f64>();
                result[c] += if expert == 0 {
                    value
                } else {
                    probability[expert - 1] * value
                };
            }
        }
        result
    }
}

fn check(
    frames: &[Frame],
    original: &[f64],
    gradients: &[f32],
    input: &[f32],
    upstream: &[[f64; 2]],
) {
    let loss = |x: &[f64]| {
        frames
            .iter()
            .enumerate()
            .map(|(row, frame)| {
                frame
                    .evaluate(original, x, row)
                    .iter()
                    .zip(upstream[row])
                    .map(|(y, g)| y * g)
                    .sum::<f64>()
            })
            .sum::<f64>()
    };
    let mut changed = original.to_vec();
    for step in [1e-4, 3e-5] {
        for i in 0..PARAMETERS {
            changed[i] = original[i] + step;
            let plus = loss(&changed);
            changed[i] = original[i] - step;
            let minus = loss(&changed);
            changed[i] = original[i];
            let expected = (plus - minus) / (2.0 * step);
            let (is_input, index) = location(i);
            let actual = f64::from(if is_input {
                input[index]
            } else {
                gradients[index]
            }) - 0.125;
            assert!(
                (actual - expected).abs() < 2e-4 * (1.0 + expected.abs()),
                "parameter={i} actual={actual} expected={expected}"
            );
        }
    }
    for (is_input, values) in [(false, gradients), (true, input)] {
        let mut support: Vec<_> = (0..PARAMETERS)
            .map(location)
            .filter(|(input, _)| *input == is_input)
            .map(|(_, i)| i)
            .collect();
        support.sort_unstable();
        let mut first = 0;
        for end in support.into_iter().chain(std::iter::once(values.len())) {
            assert!(
                values[first..end].iter().all(|&v| v == 0.125),
                "unexpected gradient in input={is_input} range={first}..{end}"
            );
            first = end + 1;
        }
    }
}

#[test]
fn expert_adjoint() -> Result<(), String> {
    metal::objc::rc::autoreleasepool(|| {
        let runtime = Runtime::shared()?;
        let x = fixture();
        let input = runtime.buffer::<u16>(ROWS * WIDTH as usize);
        let weights = runtime.buffer::<u16>(WEIGHTS);
        let input_values = slice_mut::<u16>(&input, ROWS * WIDTH as usize)?;
        let weight_values = slice_mut::<u16>(&weights, WEIGHTS)?;
        input_values.fill(0);
        weight_values.fill(0);
        for (i, &value) in x.iter().enumerate() {
            let (is_input, index) = location(i);
            if is_input {
                input_values[index] = half_bits(value as f32);
            } else {
                weight_values[index] = half_bits(value as f32);
            }
        }
        let upstream: Vec<[f64; 2]> = (0..ROWS)
            .map(|row| {
                [
                    ((row % 5) as f64 - 2.0) / 8.0,
                    ((row % 7) as f64 - 3.0) / 8.0,
                ]
            })
            .collect();
        let branch_grad = runtime.buffer::<f32>(ROWS * WIDTH as usize);
        let values = slice_mut::<f32>(&branch_grad, ROWS * WIDTH as usize)?;
        values.fill(0.0);
        for row in 0..ROWS {
            for c in 0..2 {
                values[row * WIDTH as usize + OUTPUT[c]] = upstream[row][c] as f32;
            }
        }
        let output = runtime.buffer::<u16>(ROWS * WIDTH as usize);
        let gradients = runtime.buffer::<f32>(WEIGHTS);
        slice_mut::<f32>(&gradients, WEIGHTS)?.fill(0.125);
        let input_grad = runtime.buffer_with(&vec![0.125f32; ROWS * WIDTH as usize]);
        let pipelines = FineGrainedMoePipelines::new(&runtime)?;
        let saved = FineGrainedMoeBuffers::for_rows(&runtime, ROWS as u32);
        let backward = routing_grad::FineGrainedMoeGrad::new(&runtime)?;
        let scratch = routing_grad::MoeGradBuffers::new(&runtime, ROWS as u32);
        let command = runtime.queue.new_command_buffer();
        pipelines.encode_shape(
            command,
            &saved,
            &input,
            0,
            &weights,
            0,
            (ROUTER_SIZE * 2) as u64,
            (DOWN_BASE * 2) as u64,
            &output,
            Top3RouteShape::new(ROWS as u32, WIDTH),
        )?;
        backward.backward(
            command,
            &saved,
            &scratch,
            routing_grad::MoeGrad {
                input: (&input, 0),
                weights: &weights,
                router_offset: 0,
                gate_offset: (ROUTER_SIZE * 2) as u64,
                down_offset: (DOWN_BASE * 2) as u64,
                branch_grad: (&branch_grad, 0),
                input_grad: (&input_grad, 0),
                router_grad: (&gradients, 0),
                gate_grad: (&gradients, (ROUTER_SIZE * 4) as u64),
                down_grad: (&gradients, (DOWN_BASE * 4) as u64),
                rows: ROWS as u32,
            },
        )?;
        complete(command)?;
        let frames = (0..ROWS)
            .map(|row| Frame::read(&saved, row, &x))
            .collect::<Result<Vec<_>, _>>()?;
        check(
            &frames,
            &x,
            buffer_slice::<f32>(&gradients, WEIGHTS)?,
            buffer_slice::<f32>(&input_grad, ROWS * WIDTH as usize)?,
            &upstream,
        );
        println!(
            "EXPERT_ADJOINT rows={ROWS} parameters={PARAMETERS} steps=2 selected_banks={BANKS:?} smooth_swiglu=true"
        );
        Ok(())
    })
}
