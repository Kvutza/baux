//! Shared numerical fixture for the production hierarchical mHC-4 readout.
//! It checks two patches, including a partial final patch. It is not a model
//! benchmark or a proof of equivalence outside these inputs.
use super::*;
use baux::semantics::context::probe::{bits, half};
#[path = "probe/encode.rs"]
pub mod encode;
#[path = "probe/initialized.rs"]
pub mod initialized;
#[path = "probe/objective.rs"]
pub mod objective;

pub struct Fixture {
    pub hidden: Vec<u16>,
    pub qkv: Vec<u16>,
    pub positions: Vec<u16>,
    pub readout: Vec<u16>,
    pub targets: Vec<u32>,
    pub centroids: Vec<u16>,
    pub seed: u64,
}

impl Default for Fixture {
    fn default() -> Self {
        Self::new()
    }
}

impl Fixture {
    pub fn new() -> Self {
        let mut state = 73_u64;
        let mut values = |count| {
            (0..count)
                .map(|_| {
                    state = state
                        .wrapping_mul(6_364_136_223_846_793_005)
                        .wrapping_add(1);
                    let magnitude = 0x2400 | ((state >> 42) as u16 & 0x07ff);
                    magnitude | ((state >> 48) as u16 & 0x8000)
                })
                .collect::<Vec<_>>()
        };
        let hidden = values(2 * WIDTH);
        let qkv = values(WIDTH * 640);
        let positions = values(PATCH * WIDTH);
        let readout = values(QUERY * VOCABULARY);
        let decoded = readout.iter().map(|&v| half(v)).collect::<Vec<_>>();
        let mut centroids = vec![0; HIERARCHY_NODES * QUERY];
        for (level, &offset) in [0, 8, 72, 584, 4_680].iter().enumerate() {
            let leaves = 1 << ((5 - level) * 3);
            for node in 0..(VOCABULARY / leaves) {
                for dimension in 0..QUERY {
                    let first = dimension * VOCABULARY + node * leaves;
                    let mut sum = 0.0_f32;
                    for &value in &decoded[first..first + leaves] {
                        sum += value;
                    }
                    centroids[(offset + node) * QUERY + dimension] =
                        bits(sum * (leaves as f32).sqrt().recip());
                }
            }
        }
        Self {
            hidden,
            qkv,
            positions,
            readout,
            centroids,
            targets: (0..65)
                .map(|i| (i * 4_099 + 17) % VOCABULARY as u32)
                .collect(),
            seed: 0x1234_5678_9abc_def0,
        }
    }

    fn block_base(&self, block: usize) -> [f32; QUERY] {
        let mut base = [0.0_f32; QUERY];
        for (dimension, base) in base.iter_mut().enumerate() {
            for inner in 0..WIDTH {
                *base = half(self.hidden[block * WIDTH + inner])
                    .mul_add(half(self.qkv[inner * 640 + dimension]), *base);
            }
            *base *= 0.0441941738;
        }
        base
    }

    fn step_prefix(
        &self,
        intra: usize,
        selected: u32,
        target: u32,
        prefix: &mut [[f32; QUERY]; 2],
    ) {
        for (branch, prefix) in prefix.iter_mut().enumerate() {
            let token = if branch == 0 { selected } else { target } as usize;
            for (d, prefix) in prefix.iter_mut().enumerate() {
                let coordinate = half(self.positions[intra * WIDTH + d]) + 1.0986122886681098;
                let e = (-coordinate.abs()).exp();
                let retention = if coordinate >= 0.0 {
                    1.0 / (1.0 + e)
                } else {
                    e / (1.0 + e)
                };
                let scale = (1.0 - retention * retention).max(1.0e-6).sqrt();
                *prefix =
                    retention.mul_add(*prefix, scale * half(self.readout[d * VOCABULARY + token]));
            }
        }
    }

    /// Scalar F32 reference with explicit F16 weight/query boundaries. Serial
    /// dot products are intentionally independent of accelerator reduction
    /// trees.
    pub fn reference(&self, temperature: f32) -> (Vec<u32>, Vec<f32>) {
        let mut output = Vec::with_capacity(self.targets.len());
        let mut losses = Vec::with_capacity(self.targets.len());
        for block in 0..self.targets.len().div_ceil(PATCH) {
            let base = self.block_base(block);
            let mut prefix = [[0.0_f32; QUERY]; 2];
            for intra in 0..PATCH.min(self.targets.len() - block * PATCH) {
                let position = block * PATCH + intra;
                let target = self.targets[position];
                let query = prefix.map(|prefix| {
                    std::array::from_fn::<_, QUERY, _>(|d| {
                        half(bits(
                            base[d] + half(self.positions[intra * WIDTH + d]) + prefix[d],
                        ))
                    })
                });
                let (mut selected, mut loss) = (0_u32, 0.0_f32);
                for level in 0..RADIX_LEVELS {
                    let logits = std::array::from_fn::<_, 2, _>(|branch| {
                        let parent = if branch == 0 {
                            selected
                        } else {
                            target >> ((6 - level) * 3)
                        };
                        std::array::from_fn::<_, RADIX, _>(|child| {
                            let index = parent as usize * RADIX + child;
                            let mut value = 0.0_f32;
                            for d in 0..QUERY {
                                let weight = if level < 5 {
                                    self.centroids
                                        [([0, 8, 72, 584, 4_680][level] + index) * QUERY + d]
                                } else {
                                    self.readout[d * VOCABULARY + index]
                                };
                                value = query[branch][d].mul_add(half(weight), value);
                            }
                            value
                        })
                    });
                    let mut choice = 0;
                    let sample = |child: usize| {
                        let index = selected as usize * RADIX + child;
                        let key = ((level as u64) << 56) ^ index as u64;
                        if temperature > 0.0 {
                            let u = uniform(splitmix(self.seed ^ ((position as u64) << 32) ^ key));
                            logits[0][child] - temperature * (-u.ln()).ln()
                        } else {
                            logits[0][child]
                        }
                    };
                    for child in 1..RADIX {
                        if sample(child) > sample(choice) {
                            choice = child;
                        }
                    }
                    selected = selected * RADIX as u32 + choice as u32;
                    let maximum = logits[1].iter().copied().fold(f32::NEG_INFINITY, f32::max);
                    let mut sum = 0.0_f32;
                    for logit in logits[1] {
                        sum += (logit - maximum).exp();
                    }
                    loss = loss + maximum + sum.ln()
                        - logits[1][((target >> ((5 - level) * 3)) & 7) as usize];
                }
                output.push(selected);
                losses.push(loss);
                self.step_prefix(intra, selected, target, &mut prefix);
            }
        }
        (output, losses)
    }

    pub fn verify(
        &self,
        temperature: f32,
        tokens: &[u32],
        losses: &[f32],
        centroids: &[u16],
    ) -> Result<(), String> {
        if tokens.len() != self.targets.len()
            || losses.len() != tokens.len()
            || centroids.len() != self.centroids.len()
        {
            return Err("mHC-4 readout fixture has the wrong output lengths".into());
        }
        // The fixture permits one FP16 encoding step for centroid rsqrt
        // differences and 64*FP32-epsilon scaled error for losses. Tokens are
        // exact. These are test tolerances, not derived worst-case bounds.
        for (index, (&actual, &expected)) in centroids.iter().zip(&self.centroids).enumerate() {
            if actual != expected && actual.abs_diff(expected) > 1 {
                return Err(format!(
                    "mHC-4 centroid {index}: {actual:04x} != {expected:04x}"
                ));
            }
        }
        let (expected_tokens, expected_losses) = self.reference(temperature);
        for (index, ((&token, &expected), (&loss, &reference))) in tokens
            .iter()
            .zip(&expected_tokens)
            .zip(losses.iter().zip(&expected_losses))
            .enumerate()
        {
            if token != expected
                || !loss.is_finite()
                || (loss - reference).abs() > 64.0 * f32::EPSILON * reference.abs().max(1.0)
            {
                return Err(format!(
                    "mHC-4 position {index} temperature={temperature}: token {token} != \
                     {expected}, loss {loss} != {reference}"
                ));
            }
        }
        Ok(())
    }
}
