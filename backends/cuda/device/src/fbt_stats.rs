use cuda_device::{DisjointSlice, SharedArray, ptx_asm, thread};

use super::fbt_math::{sigmoid, splitmix_uniform};
use super::fbt_types::{FBT_THREADS, ShardStats, VOCAB, WaveShape, half, half_bits};
use super::fbt_wave::merge_shards;

#[inline(always)]
unsafe fn project_scan(
    logits: &[f32],
    seeds: &[u64],
    expected: &[u32],
    maxima: *mut f32,
    targets: *mut f32,
    best: *mut f32,
    ids: *mut u32,
    block: u32,
    shard: u32,
    position: u32,
    shape: WaveShape,
) {
    let tid = thread::threadIdx_x();
    let first_token = shard * 1024 + tid;
    let expected_token = expected[position as usize];
    let mut local_maximum = f32::NEG_INFINITY;
    let mut local_target = f32::NEG_INFINITY;
    let mut local_best = f32::NEG_INFINITY;
    let mut local_id = first_token;
    for item in 0..4 {
        let token = first_token + item * 256;
        let logit = logits[block as usize * VOCAB as usize + token as usize];
        local_maximum = local_maximum.max(logit);
        if shape.score != 0 && expected_token == token {
            local_target = logit;
        }
        let mut sample = logit;
        if shape.temperature > 0.0 && shape.temperature != 1.0 {
            let uniform =
                splitmix_uniform(seeds[0] ^ (u64::from(position) << 32) ^ u64::from(token));
            sample -= shape.temperature * (-uniform.ln()).ln();
        }
        if sample > local_best || (sample == local_best && token < local_id) {
            local_best = sample;
            local_id = token;
        }
    }
    unsafe {
        maxima.add(tid as usize).write(local_maximum);
        targets.add(tid as usize).write(local_target);
        best.add(tid as usize).write(local_best);
        ids.add(tid as usize).write(local_id);
    }
    thread::sync_threads();
    let mut stride = 128_u32;
    while stride > 0 {
        if tid < stride {
            let right = tid + stride;
            unsafe {
                maxima.add(tid as usize).write(
                    maxima
                        .add(tid as usize)
                        .read()
                        .max(maxima.add(right as usize).read()),
                );
                let lt = targets.add(tid as usize).read();
                targets
                    .add(tid as usize)
                    .write(lt.max(targets.add(right as usize).read()));
                let lb = best.add(tid as usize).read();
                let rb = best.add(right as usize).read();
                let li = ids.add(tid as usize).read();
                let ri = ids.add(right as usize).read();
                if rb > lb || (rb == lb && ri < li) {
                    best.add(tid as usize).write(rb);
                    ids.add(tid as usize).write(ri);
                }
            }
        }
        thread::sync_threads();
        stride /= 2;
    }
}

#[inline(always)]
unsafe fn project_weights(
    logits: &[f32],
    seeds: &[u64],
    totals: *mut f32,
    ids: *mut u32,
    maximum: f32,
    block: u32,
    shard: u32,
    position: u32,
    weighted_reservoir: bool,
) {
    let tid = thread::threadIdx_x();
    let first_token = shard * 1024 + tid;
    let mut local_sum = 0.0_f32;
    let mut local_id = first_token;
    for item in 0..4 {
        let token = first_token + item * 256;
        let weight = (logits[block as usize * VOCAB as usize + token as usize] - maximum).exp();
        let next_sum = local_sum + weight;
        if weighted_reservoir {
            let uniform = splitmix_uniform(
                seeds[0] ^ (u64::from(position) << 32) ^ u64::from(token) ^ 0x243f6a8885a308d3,
            );
            if uniform * next_sum < weight {
                local_id = token;
            }
        }
        local_sum = next_sum;
    }
    unsafe {
        totals.add(tid as usize).write(local_sum);
        if weighted_reservoir {
            ids.add(tid as usize).write(local_id);
        }
    }
    thread::sync_threads();
    let mut stride = 128;
    while stride > 0 {
        if tid < stride {
            unsafe {
                let left_sum = totals.add(tid as usize).read();
                let right_sum = totals.add((tid + stride) as usize).read();
                let combined = left_sum + right_sum;
                if weighted_reservoir && right_sum > 0.0 {
                    let uniform = splitmix_uniform(
                        seeds[0]
                            ^ (u64::from(position) << 32)
                            ^ (u64::from(stride) << 16)
                            ^ u64::from(tid)
                            ^ 0x13198a2e03707344,
                    );
                    if uniform * combined < right_sum {
                        ids.add(tid as usize)
                            .write(ids.add((tid + stride) as usize).read());
                    }
                }
                totals.add(tid as usize).write(combined);
            }
        }
        thread::sync_threads();
        stride /= 2;
    }
}

#[inline(always)]
pub(crate) fn run_mhc4project(
    logits: &[f32],
    seeds: &[u64],
    expected: &[u32],
    partials: &mut DisjointSlice<ShardStats>,
    shape: WaveShape,
) {
    static mut MAXIMA: SharedArray<f32, 256> = SharedArray::UNINIT;
    static mut TOTALS: SharedArray<f32, 256> = SharedArray::UNINIT;
    static mut TARGETS: SharedArray<f32, 256> = SharedArray::UNINIT;
    static mut BEST: SharedArray<f32, 256> = SharedArray::UNINIT;
    static mut IDS: SharedArray<u32, 256> = SharedArray::UNINIT;
    let group = thread::blockIdx_x();
    let shard = group % shape.shards;
    let block = group / shape.shards;
    let position = block * 64 + shape.intra;
    if block >= shape.blocks || position >= shape.tokens {
        return;
    }
    let maxima = unsafe { SharedArray::as_raw_mut_ptr(&raw mut MAXIMA) };
    let totals = unsafe { SharedArray::as_raw_mut_ptr(&raw mut TOTALS) };
    let targets = unsafe { SharedArray::as_raw_mut_ptr(&raw mut TARGETS) };
    let best = unsafe { SharedArray::as_raw_mut_ptr(&raw mut BEST) };
    let ids = unsafe { SharedArray::as_raw_mut_ptr(&raw mut IDS) };
    unsafe {
        project_scan(
            logits, seeds, expected, maxima, targets, best, ids, block, shard, position, shape,
        );
        project_weights(
            logits,
            seeds,
            totals,
            ids,
            maxima.read(),
            block,
            shard,
            position,
            shape.temperature == 1.0,
        );
        if thread::threadIdx_x() == 0 {
            *partials.get_unchecked_mut(block as usize * shape.shards as usize + shard as usize) =
                ShardStats {
                    maximum: maxima.read(),
                    sum: totals.read(),
                    target: targets.read(),
                    best: best.read(),
                    token: ids.read(),
                    reserved: 0,
                };
        }
    }
}

#[inline(always)]
pub(crate) fn run_mhc4reduce(
    partials: &[ShardStats],
    base: &[f32],
    prefix: &mut DisjointSlice<f32>,
    query: &mut DisjointSlice<u16>,
    positions: &[u16],
    readout: &[u16],
    seeds: &[u64],
    output: &mut DisjointSlice<u32>,
    losses: &mut DisjointSlice<f32>,
    shape: WaveShape,
) {
    static mut CHOSEN: SharedArray<u32, 1> = SharedArray::UNINIT;
    static mut MERGED: SharedArray<ShardStats, 64> = SharedArray::UNINIT;
    let block = thread::blockIdx_x();
    let position = block * 64 + shape.intra;
    if block >= shape.blocks || position >= shape.tokens {
        return;
    }
    let tid = thread::threadIdx_x();
    let chosen = unsafe { SharedArray::as_raw_mut_ptr(&raw mut CHOSEN) };
    let merged = unsafe { SharedArray::as_raw_mut_ptr(&raw mut MERGED) };
    let row = block as usize * shape.shards as usize;
    let mut stats = partials[row + tid as usize];
    let mut shard = tid + 64;
    while shard < shape.shards {
        stats = merge_shards(
            stats,
            partials[row + shard as usize],
            shape.temperature,
            seeds[0] ^ (u64::from(position) << 32) ^ u64::from(shard),
        );
        shard += 64;
    }
    unsafe { merged.add(tid as usize).write(stats) };
    thread::sync_threads();
    let mut stride = 32_u32;
    while stride > 0 {
        if tid < stride {
            unsafe {
                let left = merged.add(tid as usize).read();
                let right = merged.add((tid + stride) as usize).read();
                merged.add(tid as usize).write(merge_shards(
                    left,
                    right,
                    shape.temperature,
                    seeds[0]
                        ^ (u64::from(position) << 32)
                        ^ (u64::from(stride) << 16)
                        ^ u64::from(tid)
                        ^ 0x452821e638d01377,
                ));
            }
        }
        thread::sync_threads();
        stride /= 2;
    }
    if tid == 0 {
        let stats = unsafe { merged.read() };
        unsafe {
            chosen.write(stats.token);
            *output.get_unchecked_mut(position as usize) = stats.token;
            if shape.score != 0 {
                *losses.get_unchecked_mut(position as usize) =
                    stats.maximum + stats.sum.ln() - stats.target;
            }
        }
    }
    thread::sync_threads();
    if shape.intra + 1 < 64 {
        let index = block * 64 + tid;
        let token = unsafe { chosen.read() };
        let previous = unsafe { *prefix.get_unchecked_mut(index as usize) };
        let retention = sigmoid(half(positions[(shape.intra * 512 + tid) as usize]) + 1.098_612_3);
        let input_scale = (1.0 - retention * retention).max(1.0e-6).sqrt();
        let value = retention.mul_add(
            previous,
            input_scale * half(readout[tid as usize * VOCAB as usize + token as usize]),
        );
        unsafe {
            *prefix.get_unchecked_mut(index as usize) = value;
            *query.get_unchecked_mut(index as usize) = half_bits(
                base[index as usize]
                    + half(positions[((shape.intra + 1) * 512 + tid) as usize])
                    + value,
            );
        }
    }
}

#[inline(always)]
unsafe fn stats_mma(
    weight: &[u16],
    query: *const u16,
    weights: *mut u16,
    logits: *mut f32,
    shard: u32,
) {
    let tid = thread::threadIdx_x();
    let warp = tid >> 5;
    let lane = tid & 31;
    let group = lane >> 2;
    let pair = (lane & 3) << 1;
    let local_row0 = (warp >> 1) * 16 + group;
    let local_row1 = local_row0 + 8;
    let fragment_column = (warp & 1) * 8 + group;
    for panel in 0..8 {
        let mut index = tid;
        while index < 1024 {
            let token = shard * 128 + panel * 16 + (index & 15);
            unsafe {
                weights
                    .add(index as usize)
                    .write(weight[(index >> 4) as usize * VOCAB as usize + token as usize]);
            }
            index += FBT_THREADS;
        }
        thread::sync_threads();
        let (mut c0, mut c1, mut c2, mut c3) = (0.0_f32, 0.0_f32, 0.0_f32, 0.0_f32);
        let mut start = 0_u32;
        while start < 64 {
            let a0_off = local_row0 * 64 + start + pair;
            let a1_off = local_row1 * 64 + start + pair;
            let b_off = (start + pair) * 16 + fragment_column;
            let a0 = unsafe {
                u32::from(query.add(a0_off as usize).read())
                    | (u32::from(query.add((a0_off + 1) as usize).read()) << 16)
            };
            let a1 = unsafe {
                u32::from(query.add(a1_off as usize).read())
                    | (u32::from(query.add((a1_off + 1) as usize).read()) << 16)
            };
            let b = unsafe {
                u32::from(weights.add(b_off as usize).read())
                    | (u32::from(weights.add((b_off + 16) as usize).read()) << 16)
            };
            unsafe {
                ptx_asm!(
                    "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5}, {%6}, {%0, %1, %2, %3};",
                    inout("+f") c0,
                    inout("+f") c1,
                    inout("+f") c2,
                    inout("+f") c3,
                    in("r") a0,
                    in("r") a1,
                    in("r") b,
                    options(register_only),
                );
            }
            start += 8;
        }
        let base_col = panel * 16 + (warp & 1) * 8 + pair;
        unsafe {
            logits.add((local_row0 * 128 + base_col) as usize).write(c0);
            logits
                .add((local_row0 * 128 + base_col + 1) as usize)
                .write(c1);
            logits.add((local_row1 * 128 + base_col) as usize).write(c2);
            logits
                .add((local_row1 * 128 + base_col + 1) as usize)
                .write(c3);
        }
        thread::sync_threads();
    }
}

#[inline(always)]
unsafe fn stats_row(
    seeds: &[u64],
    expected: &[u32],
    logits: *const f32,
    partials: &mut DisjointSlice<ShardStats, thread::Runtime2DIndex>,
    block: u32,
    shard: u32,
    tid: u32,
    shape: WaveShape,
) {
    let position = block * 64 + shape.intra;
    let expected_token = expected[position as usize];
    let row_offset = tid * 128;
    let vocabulary_start = shard * 128;
    let mut maximum = f32::NEG_INFINITY;
    let mut target = f32::NEG_INFINITY;
    let mut best = f32::NEG_INFINITY;
    let mut selected = vocabulary_start;
    for column in 0..128 {
        let token = vocabulary_start + column;
        let value = unsafe { logits.add((row_offset + column) as usize).read() };
        maximum = maximum.max(value);
        if shape.score != 0 && token == expected_token {
            target = value;
        }
        if shape.temperature != 1.0 {
            let mut sample = value;
            if shape.temperature > 0.0 {
                let uniform =
                    splitmix_uniform(seeds[0] ^ (u64::from(position) << 32) ^ u64::from(token));
                sample -= shape.temperature * (-uniform.ln()).ln();
            }
            if sample > best || (sample == best && token < selected) {
                best = sample;
                selected = token;
            }
        }
    }
    let weighted_reservoir = shape.temperature == 1.0;
    let mut sum = 0.0_f32;
    for column in 0..128 {
        let token = vocabulary_start + column;
        let value = unsafe { logits.add((row_offset + column) as usize).read() };
        let weight_value = (value - maximum).exp();
        let next_sum = sum + weight_value;
        if weighted_reservoir {
            let uniform = splitmix_uniform(
                seeds[0] ^ (u64::from(position) << 32) ^ u64::from(token) ^ 0x243f6a8885a308d3,
            );
            if uniform * next_sum < weight_value {
                selected = token;
            }
        }
        sum = next_sum;
    }
    unsafe {
        *partials.get_unchecked_mut(block as usize * shape.shards as usize + shard as usize) =
            ShardStats {
                maximum,
                sum,
                target,
                best,
                token: selected,
                reserved: 0,
            };
    }
}

#[inline(always)]
pub(crate) fn run_mhc4stats(
    input: &[u16],
    weight: &[u16],
    seeds: &[u64],
    expected: &[u32],
    partials: &mut DisjointSlice<ShardStats, thread::Runtime2DIndex>,
    shape: WaveShape,
) {
    static mut QUERY: SharedArray<u16, 4096> = SharedArray::UNINIT;
    static mut WEIGHT: SharedArray<u16, 1024> = SharedArray::UNINIT;
    static mut LOGITS: SharedArray<f32, 8192> = SharedArray::UNINIT;

    let tid = thread::threadIdx_x();
    let shard = thread::blockIdx_x();
    let block_start = thread::blockIdx_y() * 64;
    let query = unsafe { SharedArray::as_raw_mut_ptr(&raw mut QUERY) };
    let weights = unsafe { SharedArray::as_raw_mut_ptr(&raw mut WEIGHT) };
    let logits = unsafe { SharedArray::as_raw_mut_ptr(&raw mut LOGITS) };

    let mut index = tid;
    while index < 4096 {
        let block = block_start + (index >> 6);
        let dimension = index & 63;
        unsafe {
            query.add(index as usize).write(if block < shape.blocks {
                input[(block * 64 + dimension) as usize]
            } else {
                0
            });
        }
        index += FBT_THREADS;
    }
    thread::sync_threads();
    unsafe { stats_mma(weight, query, weights, logits, shard) };
    if tid < 64 {
        let block = block_start + tid;
        if block < shape.blocks && block * 64 + shape.intra < shape.tokens {
            unsafe { stats_row(seeds, expected, logits, partials, block, shard, tid, shape) };
        }
    }
}
