use super::*;
use crate::apple_gpu::{buffer_slice, slice_mut};
use metal::objc::rc::autoreleasepool;

const TARGETS: [u32; 2] = [0, VOCAB - 1];
const GUARD: usize = 64;
const SEED: f32 = 0.125;

fn leaf(token: usize) -> f64 {
    (token % 8) as f64 / 64.0 - 3.0 / 64.0
}

fn query(row: usize) -> f64 {
    decode_half(resources::half_bits(
        (row + 1) as f32 * (2.82842712f32 / 32.0),
    ))
}

fn coefficient(row: usize, token: usize, level: usize) -> f64 {
    let leaves = 1usize << ((5 - level) * 3);
    let target = TARGETS[row] as usize;
    if token / (8 * leaves) != target / (8 * leaves) {
        return 0.0;
    }
    let child = (token / leaves) % 8;
    let probability = if level < 5 {
        0.125
    } else {
        let denominator: f64 = (0..8).map(|v| (query(row) * leaf(v)).exp()).sum();
        (query(row) * leaf(child)).exp() / denominator
    };
    (probability - f64::from(child == (target / leaves) % 8)) / TARGETS.len() as f64
}

fn query_grad(row: usize) -> f64 {
    let target = TARGETS[row] as usize;
    (0..8)
        .map(|child| coefficient(row, (target / 8) * 8 + child, 5) * leaf(child))
        .sum()
}

fn close(actual: f32, expected: f64) {
    assert!(
        actual.is_finite() && (f64::from(actual) - expected).abs() < 3e-6,
        "actual={actual} expected={expected}"
    );
}

struct Fixture {
    hidden: Buffer,
    qkv: Buffer,
    readout: Buffer,
    targets: Buffer,
    gradient: Buffer,
    hidden_grad: Buffer,
    core_grad: Buffer,
    qstart: usize,
    wstart: usize,
}

impl Fixture {
    fn new(runtime: &Runtime) -> Result<Self, String> {
        let mut hidden = vec![0u16; TARGETS.len() * WIDTH as usize];
        for row in 0..TARGETS.len() {
            hidden[row * WIDTH as usize] = resources::half_bits((row + 1) as f32);
        }
        let mut qkv = vec![0u16; (WIDTH * QKV_WIDTH) as usize];
        qkv[0] = resources::half_bits(1.0 / 32.0);
        let readout = runtime.buffer::<u16>(mhc4::QUERY * VOCAB as usize);
        let weights = slice_mut::<u16>(&readout, mhc4::QUERY * VOCAB as usize)?;
        weights.fill(0);
        for (token, value) in weights[..VOCAB as usize].iter_mut().enumerate() {
            *value = resources::half_bits(leaf(token) as f32);
        }
        let qstart = GUARD;
        let wstart = qstart + qkv.len() + GUARD;
        let count = wstart + weights.len() + GUARD;
        let gradient = runtime.buffer::<f32>(count);
        slice_mut::<f32>(&gradient, count)?.fill(SEED);
        let hidden_grad = runtime.buffer_with(&vec![SEED; hidden.len()]);
        Ok(Self {
            hidden: runtime.buffer_with(&hidden),
            qkv: runtime.buffer_with(&qkv),
            readout,
            targets: runtime.buffer_with(&TARGETS),
            gradient,
            hidden_grad,
            core_grad: runtime.buffer_with(&vec![SEED; 128 * WIDTH as usize]),
            qstart,
            wstart,
        })
    }

    fn check(&self) -> Result<(), String> {
        let gradient = buffer_slice::<f32>(&self.gradient, self.gradient.length() as usize / 4)?;
        let projection = 2.0
            * f64::from(2.82842712f32)
            * (0..TARGETS.len())
                .map(|row| (row + 1) as f64 * query_grad(row))
                .sum::<f64>();
        close(gradient[self.qstart], f64::from(SEED) + projection);
        assert!(gradient[..self.qstart].iter().all(|&v| v == SEED));
        let core = self.qstart + QKV_WIDTH as usize + 64;
        close(gradient[core], f64::from(SEED) + 32.0);
        assert!(gradient[self.qstart + 1..core].iter().all(|&v| v == SEED));
        assert!(gradient[core + 1..self.wstart].iter().all(|&v| v == SEED));
        for token in 0..VOCAB as usize {
            let mut expected = f64::from(SEED);
            for row in 0..TARGETS.len() {
                if token == TARGETS[row] as usize {
                    expected += (row + 1) as f64 / 4.0;
                }
                for level in 0..6 {
                    let leaves = 1u32 << ((5 - level) * 3);
                    expected += 2.0 * query(row) * coefficient(row, token, level)
                        / f64::from(leaves).sqrt();
                }
            }
            close(gradient[self.wstart + token], expected);
        }
        assert!(
            gradient[self.wstart + VOCAB as usize..]
                .iter()
                .all(|&v| v == SEED)
        );
        let hidden = buffer_slice::<f32>(&self.hidden_grad, TARGETS.len() * WIDTH as usize)?;
        assert!(
            buffer_slice::<f32>(&self.core_grad, 128 * WIDTH as usize)?
                .iter()
                .all(|&v| v == SEED)
        );
        for (index, &value) in hidden.iter().enumerate() {
            let contribution = if index % WIDTH as usize == 0 {
                2.0 * f64::from(2.82842712f32) / 32.0 * query_grad(index / WIDTH as usize)
            } else {
                0.0
            };
            close(value, f64::from(SEED) + contribution);
        }
        Ok(())
    }

    /// Exercise the actual core and embedding kernels between readout passes.
    fn accumulate(&self, runtime: &Runtime, command: &CommandBufferRef) -> Result<(), String> {
        let kernels = train_grad::Kernels::new(runtime)?;
        let mut input = vec![0u16; 128 * WIDTH as usize];
        let mut output = vec![0.0f32; 128 * QKV_WIDTH as usize];
        for row in 0..128 {
            input[row * WIDTH as usize + 1] = resources::half_bits(1.0);
            output[row * QKV_WIDTH as usize + 64] = 0.25;
        }
        let input = runtime.buffer_with(&input);
        let output = runtime.buffer_with(&output);
        let half = runtime.buffer::<u16>(128 * QKV_WIDTH as usize);
        let mut embedding = vec![0.0f32; TARGETS.len() * mhc4::QUERY];
        for row in 0..TARGETS.len() {
            embedding[row * mhc4::QUERY] = (row + 1) as f32 / 4.0;
        }
        let embedding = runtime.buffer_with(&embedding);
        let encoder =
            command.compute_command_encoder_with_dispatch_type(metal::MTLDispatchType::Serial);
        kernels.dense(
            &encoder,
            train_grad::DenseShape {
                rows: 128,
                input: WIDTH,
                output: QKV_WIDTH,
                reserved: 0,
            },
            train_grad::DenseGrad {
                input: (&input, 0),
                weights: (&self.qkv, 0),
                output_grad: (&output, 0),
                input_grad: (&self.core_grad, 0),
                weight_grad: (&self.gradient, (self.qstart * 4) as u64),
                half_grad: &half,
            },
        );
        encoder.memory_barrier_with_resources(&[&self.gradient]);
        kernels.embed(
            &encoder,
            train_grad::DenseShape {
                rows: TARGETS.len() as u32,
                input: mhc4::QUERY as u32,
                output: VOCAB,
                reserved: 1,
            },
            train_grad::EmbedGrad {
                readout: (&self.readout, 0),
                tokens: (&self.targets, 0),
                positions: (&self.qkv, 0),
                output_grad: (&embedding, 0),
                readout_grad: (&self.gradient, (self.wstart * 4) as u64),
                position_grad: (&self.core_grad, 0),
                mask_grad: (&self.core_grad, 0),
            },
        );
        encoder.end_encoding();
        Ok(())
    }
}

#[test]
fn shared_gradients() -> Result<(), String> {
    autoreleasepool(|| {
        let runtime = Runtime::shared()?;
        let fixture = Fixture::new(&runtime)?;
        let waves = Mhc4Waves::new(&runtime)?;
        let adjoint = Adjoint::new(&runtime, TARGETS.len() as u32)?;
        let output = runtime.buffer::<u32>(TARGETS.len());
        let losses = runtime.buffer::<f32>(TARGETS.len());
        let command = runtime.queue.new_command_buffer();
        waves.encode_mhc(
            command,
            (&fixture.hidden, 0),
            (&fixture.qkv, 0),
            (&fixture.readout, 0),
            (&fixture.targets, 0),
            (&output, 0),
            (&losses, 0),
            TARGETS.len() as u32,
            1,
            true,
        )?;
        for pass in 0..2 {
            adjoint.encode(
                command,
                &waves,
                TARGETS.len() as u32,
                ReadoutGrad {
                    hidden: (&fixture.hidden, 0),
                    qkv: (&fixture.qkv, 0),
                    readout: (&fixture.readout, 0),
                    targets: (&fixture.targets, 0),
                    hidden_grad: (&fixture.hidden_grad, 0),
                    qkv_grad: (&fixture.gradient, (fixture.qstart * 4) as u64),
                    readout_grad: (&fixture.gradient, (fixture.wstart * 4) as u64),
                },
            )?;
            if pass == 0 {
                fixture.accumulate(&runtime, command)?;
            }
        }
        complete(command)?;
        fixture.check()
    })
}
