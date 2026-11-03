use super::*;

fn expected_blocks(qkv: &[u16], pyramid: &[u16], row: u32) -> [u32; SELECTED as usize] {
    let position = row % CONTEXT;
    let current = position / BLOCK;
    let previous = current.saturating_sub(1);
    let sequence = row / CONTEXT;
    let mut candidates: [u32; 16] = std::array::from_fn(|candidate| candidate as u32);
    let mut chosen = [u32::MAX; SELECTED as usize];
    for level in (0..=2u32).rev() {
        let mut scores = [f32::NEG_INFINITY; 16];
        for (slot, &node) in candidates.iter().enumerate() {
            if node == u32::MAX {
                continue;
            }
            let begin = node << level;
            let end = begin + (1 << level) - 1;
            let forced = [0, previous, current]
                .into_iter()
                .any(|leaf| leaf >= begin && leaf <= end);
            if forced {
                scores[slot] = f32::INFINITY;
            } else if end < current {
                scores[slot] = if level == 0 {
                    leaf_lse_score(qkv, row, sequence, node)
                } else {
                    intermediate_lse_score(qkv, pyramid, row, sequence, node, level)
                };
            }
        }
        for output in &mut chosen {
            let best = scores
                .iter()
                .enumerate()
                .filter(|(_, score)| **score > f32::NEG_INFINITY)
                .max_by(|(left_index, left), (right_index, right)| {
                    left.total_cmp(right)
                        .then_with(|| candidates[*right_index].cmp(&candidates[*left_index]))
                })
                .map(|(index, _)| index);
            if let Some(best) = best {
                *output = candidates[best];
                scores[best] = f32::NEG_INFINITY;
            } else {
                *output = u32::MAX;
            }
        }
        if level > 0 {
            for (slot, &node) in chosen.iter().enumerate() {
                candidates[2 * slot] = if node == u32::MAX { u32::MAX } else { 2 * node };
                candidates[2 * slot + 1] = if node == u32::MAX {
                    u32::MAX
                } else {
                    2 * node + 1
                };
            }
        }
    }
    chosen
}

fn intermediate_lse_score(
    qkv: &[u16],
    pyramid: &[u16],
    row: u32,
    sequence: u32,
    node: u32,
    level: u32,
) -> f32 {
    debug_assert!(level > 0);
    let child_level = level - 1;
    let child_offset = if child_level == 0 {
        0
    } else {
        2 * LEAVES - (2 * LEAVES >> child_level)
    };
    let mut grouped_score = 0.0f32;
    for head in 0..QUERY_HEADS {
        let mut logits = [0.0f32; 2];
        for child in 0..2 {
            let summary =
                ((sequence * NODES + child_offset + 2 * node + child) * HEAD_DIM) as usize;
            let query = (row * QKV_WIDTH + head * HEAD_DIM) as usize;
            logits[child as usize] = (0..HEAD_DIM as usize)
                .map(|dim| decode_half(qkv[query + dim]) * decode_half(pyramid[summary + dim]))
                .sum::<f32>()
                * 0.125;
        }
        let maximum = logits[0].max(logits[1]);
        grouped_score += maximum + ((logits[0] - maximum).exp() + (logits[1] - maximum).exp()).ln();
    }
    grouped_score
}

fn leaf_lse_score(qkv: &[u16], row: u32, sequence: u32, leaf: u32) -> f32 {
    let mut grouped_score = 0.0f32;
    for head in 0..QUERY_HEADS {
        let query = (row * QKV_WIDTH + head * HEAD_DIM) as usize;
        let mut logits = [0.0f32; BLOCK as usize];
        for token in 0..BLOCK {
            let source = sequence * CONTEXT + leaf * BLOCK + token;
            let key = (source * QKV_WIDTH + QUERY_WIDTH) as usize;
            logits[token as usize] = (0..HEAD_DIM as usize)
                .map(|dim| decode_half(qkv[query + dim]) * decode_half(qkv[key + dim]))
                .sum::<f32>()
                * 0.125;
        }
        let maximum = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        grouped_score += maximum
            + logits
                .iter()
                .map(|&logit| (logit - maximum).exp())
                .sum::<f32>()
                .ln();
    }
    grouped_score
}

fn validate_selection(pisa: &Pisa1, qkv: &BufferRef) -> Result<(), String> {
    let qkv = half_values(qkv, (ROWS * QKV_WIDTH) as usize);
    let pyramid = half_values(&pisa.pyramid, (BATCH * NODES * HEAD_DIM) as usize);
    let actual = unsafe {
        std::slice::from_raw_parts(
            pisa.blocks.contents().cast::<u32>(),
            (ROWS * SELECTED) as usize + 64,
        )
    };
    if !actual[(ROWS * SELECTED) as usize..]
        .iter()
        .all(|&value| value == u32::MAX)
    {
        return Err("PISA-1 selection overwrote its canary".into());
    }
    for row in [0, 63, 64, 511, 1024, 4095, 4096, 8191] {
        let expected = expected_blocks(qkv, pyramid, row);
        let start = (row * SELECTED) as usize;
        if actual[start..start + SELECTED as usize] != expected {
            return Err(format!(
                "PISA-1 selection mismatch at row {row}: {:?} != {expected:?}",
                &actual[start..start + SELECTED as usize]
            ));
        }
    }
    for row in 0..ROWS {
        let position = row % CONTEXT;
        let current = position / BLOCK;
        let expected_count = SELECTED.min(current + 1) as usize;
        let selected = &actual[(row * SELECTED) as usize..((row + 1) * SELECTED) as usize];
        let valid = selected
            .iter()
            .copied()
            .filter(|&block| block != u32::MAX)
            .collect::<Vec<_>>();
        if !causal_selection(&valid, current, expected_count) {
            return Err(format!(
                "PISA-1 invalid forced/causal selection at row {row}"
            ));
        }
    }
    let (selections, unique) = selection_reuse(actual);
    eprintln!(
        "pisa1.query_tile tile=4 selections={} unique_blocks={} kv_reuse={:.3}",
        selections,
        unique,
        selections as f64 / unique as f64,
    );
    Ok(())
}

fn selection_reuse(actual: &[u32]) -> (usize, usize) {
    let (sequence_width, tile_width) = ((CONTEXT * SELECTED) as usize, 4 * SELECTED as usize);
    let mut selections = 0usize;
    let mut unique = 0usize;
    for sequence in actual[..(ROWS * SELECTED) as usize].chunks_exact(sequence_width) {
        for tile in sequence.chunks_exact(tile_width) {
            let mut union = [u32::MAX; 4 * SELECTED as usize];
            let mut union_len = 0usize;
            for &block in tile.iter().filter(|&&block| block != u32::MAX) {
                selections += 1;
                if !union[..union_len].contains(&block) {
                    union[union_len] = block;
                    union_len += 1;
                }
            }
            unique += union_len;
        }
    }
    (selections, unique)
}

fn causal_selection(valid: &[u32], current: u32, expected_count: usize) -> bool {
    valid.len() == expected_count
        && valid.iter().all(|&block| block <= current)
        && !(0..valid.len()).any(|left| valid[left + 1..].contains(&valid[left]))
        && valid.contains(&0)
        && valid.contains(&current)
        && (current == 0 || valid.contains(&(current - 1)))
}

fn validate_attention(pisa: &Pisa1, qkv: &BufferRef) -> Result<f64, String> {
    let qkv = half_values(qkv, (ROWS * QKV_WIDTH) as usize);
    let blocks = unsafe {
        std::slice::from_raw_parts(
            pisa.blocks.contents().cast::<u32>(),
            (ROWS * SELECTED) as usize,
        )
    };
    let output = half_values(&pisa.output, (ROWS * QUERY_WIDTH) as usize + 64);
    if !output[(ROWS * QUERY_WIDTH) as usize..]
        .iter()
        .all(|&value| value == 0x7e00)
    {
        return Err("PISA-1 attention overwrote its canary".into());
    }
    let mut maximum = 0.0f64;
    for row in [0, 63, 64, 511, 1024, 4095, 4096, 8191] {
        let sequence = row / CONTEXT;
        let position = row % CONTEXT;
        for head in 0..QUERY_HEADS {
            let mut logits = Vec::with_capacity((SELECTED * BLOCK) as usize);
            let mut tokens = Vec::with_capacity((SELECTED * BLOCK) as usize);
            for &block in &blocks[(row * SELECTED) as usize..((row + 1) * SELECTED) as usize] {
                if block == u32::MAX {
                    continue;
                }
                for local in 0..BLOCK {
                    let token_position = block * BLOCK + local;
                    if token_position > position {
                        continue;
                    }
                    let source = sequence * CONTEXT + token_position;
                    let mut score = 0.0f32;
                    for dim in 0..HEAD_DIM {
                        score +=
                            decode_half(qkv[(row * QKV_WIDTH + head * HEAD_DIM + dim) as usize])
                                * decode_half(
                                    qkv[(source * QKV_WIDTH + QUERY_WIDTH + dim) as usize],
                                );
                    }
                    logits.push(score * 0.125);
                    tokens.push(source);
                }
            }
            let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let weights = logits
                .iter()
                .map(|&logit| (logit - max).exp())
                .collect::<Vec<_>>();
            let total: f32 = weights.iter().sum();
            for dim in 0..HEAD_DIM {
                let expected = weights
                    .iter()
                    .zip(&tokens)
                    .map(|(&weight, &token)| {
                        weight
                            * decode_half(
                                qkv[(token * QKV_WIDTH + QUERY_WIDTH + HEAD_DIM + dim) as usize],
                            )
                    })
                    .sum::<f32>()
                    / total;
                let index = (row * QUERY_WIDTH + head * HEAD_DIM + dim) as usize;
                let actual = decode_half(output[index]);
                if !actual.is_finite() {
                    return Err(format!("PISA-1 produced non-finite output at {index}"));
                }
                maximum = maximum.max(f64::from((expected - actual).abs()));
            }
        }
    }
    if maximum > 0.002 {
        return Err(format!(
            "PISA-1 attention max sampled absolute error {maximum:.9} exceeds 0.002"
        ));
    }
    Ok(maximum)
}

fn validate_q4(pisa: &Pisa1) -> Result<f64, String> {
    let elements = (ROWS * QUERY_WIDTH) as usize;
    let q4 = half_values(&pisa.output, elements);
    let q1 = half_values(&pisa.oracle_output, elements);
    let mut maximum = 0.0f64;
    let mut maximum_index = 0usize;
    for index in 0..elements {
        let error = f64::from((decode_half(q4[index]) - decode_half(q1[index])).abs());
        if !error.is_finite() {
            return Err(format!("PISA-1 parity found a non-finite value at {index}"));
        }
        if error > maximum {
            maximum = error;
            maximum_index = index;
        }
    }
    if maximum > 0.002 {
        return Err(format!(
            "PISA-1 Q4 versus Q1 max absolute error {maximum:.9} at {maximum_index} exceeds 0.002"
        ));
    }
    eprintln!(
        "pisa1.parity candidate=q4 oracle=q1 elements={elements} max_abs_error={maximum:.9} \
         index={maximum_index}"
    );
    Ok(maximum)
}

fn stage_median(
    runtime: &Runtime,
    pisa: &Pisa1,
    qkv: &BufferRef,
    stage: Stage,
) -> Result<f64, String> {
    let mut samples = Vec::with_capacity(3);
    for _ in 0..3 {
        let command = runtime.queue.new_command_buffer();
        pisa.encode(command, qkv, Some(stage));
        samples.push(complete(command)?);
    }
    samples.sort_by(f64::total_cmp);
    Ok(samples[1])
}

fn check_parity(runtime: &Runtime, pisa: &Pisa1, qkv: &BufferRef) -> Result<f64, String> {
    let command = runtime.queue.new_command_buffer();
    pisa.encode(command, qkv, None);
    complete(command)?;
    validate_selection(pisa, qkv)?;
    let selected = unsafe {
        std::slice::from_raw_parts(
            pisa.blocks.contents().cast::<u32>(),
            (ROWS * SELECTED) as usize,
        )
        .to_vec()
    };
    let command = runtime.queue.new_command_buffer();
    pisa.encode_oracle(command, qkv);
    complete(command)?;
    let oracle = unsafe {
        std::slice::from_raw_parts(
            pisa.blocks.contents().cast::<u32>(),
            (ROWS * SELECTED) as usize,
        )
    };
    if selected != oracle {
        return Err("PISA-1 Q4 selection differs from Q1".into());
    }
    let fixed_error = validate_attention(pisa, qkv)?.max(validate_q4(pisa)?);
    let fixed_error = fixed_error.max(super::hierarchical::validate(runtime, pisa, qkv)?);
    Ok(fixed_error.max(super::fallback::validate(runtime, pisa, qkv)?))
}

#[test]
fn gap_budget() -> Result<(), String> {
    metal::objc::rc::autoreleasepool(|| {
        use crate::apple_gpu::buffer_slice;
        let runtime = Runtime::shared()?;
        let pisa = Pisa1::new(&runtime)?;
        let mut values = vec![0u16; (ROWS * QKV_WIDTH) as usize];
        for row in 0..ROWS {
            let base = (row * QKV_WIDTH) as usize;
            values[base] = 0x4800; // Head 0 has q[0] = 8; other heads are zero.
            match row % CONTEXT / BLOCK {
                1 => values[base + QUERY_WIDTH as usize] = 0x4000, // k[0] = 2
                2 => values[base + (QUERY_WIDTH + HEAD_DIM) as usize] = 0x3c00, // v[0] = 1
                _ => {}
            }
        }
        let qkv = runtime.buffer_with(&values);
        // A routing-score gap of 16 must not remove blocks that fit the budget.
        // Compare production selections and outputs with the existing CPU/Q1
        // oracles.
        check_parity(&runtime, &pisa, &qkv)?;
        let output = buffer_slice::<u16>(&pisa.output, (ROWS * QUERY_WIDTH) as usize)?;
        for sequence in 0..BATCH {
            let row = sequence * CONTEXT + 511;
            for head in 0..QUERY_HEADS {
                let expected = if head == 0 {
                    1.0 / (2.0f32.exp() + 7.0)
                } else {
                    0.125
                };
                let actual = decode_half(output[(row * QUERY_WIDTH + head * HEAD_DIM) as usize]);
                assert!(
                    (actual - expected).abs() < 0.0002,
                    "row {row}, head {head}: {actual}"
                );
            }
        }
        Ok(())
    })
}

pub(crate) fn pisa_probe(
    runtime: &Runtime,
    qkv: &BufferRef,
    rounds: u32,
) -> Result<Pisa1Probe, String> {
    if rounds == 0 {
        return Err("PISA-1 probe requires positive rounds".into());
    }
    let pisa = Pisa1::new(runtime)?;
    let tied = runtime.buffer_with(&vec![0u16; (ROWS * QKV_WIDTH) as usize]);
    check_parity(runtime, &pisa, &tied)?;
    let max_abs_error = check_parity(runtime, &pisa, qkv)?;
    let pyramid_gpu_seconds = stage_median(runtime, &pisa, qkv, Stage::Pyramid)?;
    let selection_gpu_seconds = stage_median(runtime, &pisa, qkv, Stage::Selection)?;
    let attention_gpu_seconds = stage_median(runtime, &pisa, qkv, Stage::Attention)?;
    for _ in 0..3 {
        let command = runtime.queue.new_command_buffer();
        pisa.encode(command, qkv, None);
        complete(command)?;
    }
    let mut gpu = Vec::with_capacity(rounds as usize);
    let mut wall = Vec::with_capacity(rounds as usize);
    for _ in 0..rounds {
        let command = runtime.queue.new_command_buffer();
        pisa.encode(command, qkv, None);
        let start = Instant::now();
        gpu.push(complete(command)?);
        wall.push(start.elapsed().as_secs_f64());
    }
    gpu.sort_by(f64::total_cmp);
    wall.sort_by(f64::total_cmp);
    Ok(Pisa1Probe {
        pyramid_gpu_seconds,
        selection_gpu_seconds,
        attention_gpu_seconds,
        layer_gpu_seconds: gpu[gpu.len() / 2],
        layer_wall_seconds: wall[wall.len() / 2],
        max_abs_error,
    })
}
