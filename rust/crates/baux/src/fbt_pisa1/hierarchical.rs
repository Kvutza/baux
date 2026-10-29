//! Mass-preserving dual-resolution attention: selected blocks at fine
//! resolution, every other causal leaf as one mean K/V token carrying
//! `ln(BLOCK)` extra log-mass, all under a single softmax.

use super::*;

const COARSEWIDTH: u32 = 2 * HEAD_DIM;
const LEAFMASS: f32 = 4.158_883;
const SAMPLES: [u32; 8] = [0, 63, 64, 511, 2048, 4095, 4096, 8191];

impl Pisa1 {
    /// Builds the K/V pyramid and runs the O(log N) hierarchical kernel into
    /// `output`. Selection (`self.blocks`) must already be populated.
    fn encode_hierarchical(
        &self,
        command: &CommandBufferRef,
        qkv: &BufferRef,
        coarse: &BufferRef,
        output: &BufferRef,
    ) {
        let encoder = command.new_compute_command_encoder();
        dispatch_on(
            &encoder,
            &self.pipelines.leaf_kv,
            &[qkv, coarse],
            MTLSize {
                width: u64::from(self.context / BLOCK),
                height: u64::from(BATCH),
                depth: 1,
            },
            u64::from(HEAD_DIM),
        );
        encoder.memory_barrier_with_resources(&[coarse]);
        dispatch_on(
            &encoder,
            &self.pipelines.upper_kv,
            &[coarse],
            thread_group(u64::from(BATCH)),
            u64::from(COARSEWIDTH),
        );
        encoder.memory_barrier_with_resources(&[coarse]);
        dispatch_on(
            &encoder,
            &self.pipelines.hierarchical,
            &[qkv, coarse, &self.blocks, output],
            thread_group(u64::from(ROWS)),
            128,
        );
        encoder.end_encoding();
    }
}

fn dot(qkv: &[u16], query: usize, key: &[u16]) -> f32 {
    (0..HEAD_DIM as usize)
        .map(|dim| decode_half(qkv[query + dim]) * decode_half(key[dim]))
        .sum()
}

fn reference_head(
    qkv: &[u16],
    coarse: &[u16],
    blocks: &[u32],
    row: u32,
    head: u32,
) -> [f32; HEAD_DIM as usize] {
    let (sequence, position) = (row / CONTEXT, row % CONTEXT);
    let current = position / BLOCK;
    let query = (row * QKV_WIDTH + head * HEAD_DIM) as usize;
    let mut logits = Vec::new();
    let mut values: Vec<Vec<f32>> = Vec::new();
    for &leaf in blocks {
        if leaf == u32::MAX || leaf > current {
            continue;
        }
        for local in 0..BLOCK {
            let token = leaf * BLOCK + local;
            if token > position {
                continue;
            }
            let base = ((sequence * CONTEXT + token) * QKV_WIDTH + QUERY_WIDTH) as usize;
            logits.push(dot(qkv, query, &qkv[base..]) * 0.125);
            let v = &qkv[base + HEAD_DIM as usize..base + COARSEWIDTH as usize];
            values.push(v.iter().map(|&x| decode_half(x)).collect());
        }
    }
    let top_level = (LEAVES.ilog2() as i32) - 4;
    for level in (0..=top_level).rev() {
        let shift = level as u32;
        let offset = 2 * LEAVES - (2 * LEAVES >> shift);
        let mass = (level as f32).mul_add(std::f32::consts::LN_2, LEAFMASS);
        let limit = if level == top_level {
            16
        } else {
            SELECTED as usize
        };
        for cand in 0..limit {
            let node = if level == top_level {
                cand as u32
            } else {
                let sel = blocks[cand];
                if sel == u32::MAX {
                    continue;
                }
                (sel >> shift) ^ 1
            };
            if (node << shift) >= current {
                continue;
            }
            if blocks
                .iter()
                .any(|&leaf| leaf != u32::MAX && (leaf >> shift) == node)
            {
                continue;
            }
            if level < top_level
                && blocks[..cand]
                    .iter()
                    .any(|&prev| prev != u32::MAX && ((prev >> shift) ^ 1) == node)
            {
                continue;
            }
            let base = ((sequence * NODES + offset + node) * COARSEWIDTH) as usize;
            logits.push(dot(qkv, query, &coarse[base..]) * 0.125 + mass);
            let v = &coarse[base + HEAD_DIM as usize..base + COARSEWIDTH as usize];
            values.push(v.iter().map(|&x| decode_half(x)).collect());
        }
    }
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let weights: Vec<f32> = logits.iter().map(|&l| (l - max).exp()).collect();
    let total: f32 = weights.iter().sum();
    let mut result = [0.0f32; HEAD_DIM as usize];
    for (weight, value) in weights.iter().zip(&values) {
        for (out, &v) in result.iter_mut().zip(value) {
            *out += weight * v / total;
        }
    }
    result
}

/// Runs the hierarchical kernel on `qkv` and compares sampled rows against a
/// CPU implementation of the same unified-softmax formula.
pub(super) fn validate(runtime: &Runtime, pisa: &Pisa1, qkv: &BufferRef) -> Result<f64, String> {
    let coarse = runtime.buffer::<u16>((BATCH * NODES * COARSEWIDTH) as usize);
    let output = runtime.buffer_with(&vec![0x7e00u16; (ROWS * QUERY_WIDTH) as usize]);
    let command = runtime.queue.new_command_buffer();
    pisa.encode_hierarchical(command, qkv, &coarse, &output);
    complete(command)?;
    let qkv_values = half_values(qkv, (ROWS * QKV_WIDTH) as usize);
    let coarse_values = half_values(&coarse, (BATCH * NODES * COARSEWIDTH) as usize);
    let actual = half_values(&output, (ROWS * QUERY_WIDTH) as usize);
    let blocks = unsafe {
        std::slice::from_raw_parts(
            pisa.blocks.contents().cast::<u32>(),
            (ROWS * SELECTED) as usize,
        )
    };
    let mut maximum = 0.0f64;
    for row in SAMPLES {
        let selected = &blocks[(row * SELECTED) as usize..((row + 1) * SELECTED) as usize];
        for head in 0..QUERY_HEADS {
            let expected = reference_head(qkv_values, coarse_values, selected, row, head);
            for (dim, &want) in expected.iter().enumerate() {
                let got =
                    decode_half(actual[((row * QUERY_HEADS + head) * HEAD_DIM) as usize + dim]);
                if !got.is_finite() {
                    return Err(format!("hierarchical non-finite at row {row} head {head}"));
                }
                maximum = maximum.max(f64::from((want - got).abs()));
            }
        }
    }
    if maximum > 0.002 {
        return Err(format!(
            "PISA-1 hierarchical max sampled absolute error {maximum:.9} exceeds 0.002"
        ));
    }
    eprintln!("pisa1.hierarchical max_abs_error={maximum:.9}");
    Ok(maximum)
}
