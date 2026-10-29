//! Finite encoding fixture for full, partial, empty and internally masked
//! patches. Readout storage is token-major on Metal and dimension-major on
//! CUDA.
use baux::semantics::context::probe::{bits, half};

pub const ROWS: usize = 4;
pub const WIDTH: usize = 17;
pub const VOCABULARY: usize = 29;
pub const PATCH: usize = 64;

pub struct Fixture {
    pub tokens: Vec<u32>,
    pub positions: Vec<u16>,
    pub mask: Vec<u16>,
    pub readout: Vec<u16>,
    pub expected: Vec<u16>,
}

impl Default for Fixture {
    fn default() -> Self {
        Self::new()
    }
}

impl Fixture {
    pub fn new() -> Self {
        let tokens = (0..ROWS * PATCH)
            .map(|i| {
                let (row, intra) = (i / PATCH, i % PATCH);
                if row == 2 || (row == 1 && intra >= 17) || (row == 3 && intra % 3 == 0) {
                    u32::MAX
                } else {
                    (i % VOCABULARY) as u32
                }
            })
            .collect::<Vec<_>>();
        let positions = (0..PATCH * WIDTH)
            .map(|i| 0x3000 + ((i * 13) % 1024) as u16)
            .collect::<Vec<_>>();
        let mask = (0..WIDTH).map(|i| 0xb400 + i as u16).collect::<Vec<_>>();
        let readout = (0..VOCABULARY * WIDTH)
            .map(|i| 0x2800 + ((i * 17) % 1024) as u16)
            .collect::<Vec<_>>();
        let expected = (0..ROWS * WIDTH)
            .map(|i| {
                let (row, d) = (i / WIDTH, i % WIDTH);
                let (mut sum, mut count) = (0.0_f32, 0_u32);
                for intra in 0..PATCH {
                    let token = tokens[row * PATCH + intra];
                    if token < VOCABULARY as u32 {
                        sum = half(positions[intra * WIDTH + d])
                            .mul_add(half(readout[token as usize * WIDTH + d]), sum);
                        count += 1;
                    }
                }
                if count == 0 {
                    mask[d]
                } else {
                    bits(sum * (PATCH as f32 / count as f32).sqrt())
                }
            })
            .collect();
        Self {
            tokens,
            positions,
            mask,
            readout,
            expected,
        }
    }

    pub fn transposed(&self) -> Vec<u16> {
        (0..WIDTH * VOCABULARY)
            .map(|i| self.readout[(i % VOCABULARY) * WIDTH + i / VOCABULARY])
            .collect()
    }

    pub fn verify(&self, actual: &[u16]) -> Result<(), String> {
        if actual != self.expected {
            let index = actual.iter().zip(&self.expected).position(|(a, b)| a != b);
            return Err(format!(
                "mHC-4 encoder differs from the FP16 fixture at {index:?}"
            ));
        }
        Ok(())
    }
}
