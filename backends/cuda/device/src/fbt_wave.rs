use cuda_device::{DisjointSlice, SharedArray, thread};

use super::fbt_math::{sigmoid, splitmix_uniform};
use super::fbt_types::{Mhc4EncodeShape, ShardStats, VOCAB, WaveShape, half, half_bits};

#[inline(always)]
pub(crate) fn merge_shards(
    mut left: ShardStats,
    right: ShardStats,
    temperature: f32,
    seed: u64,
) -> ShardStats {
    let maximum = left.maximum.max(right.maximum);
    let left_sum = left.sum * (left.maximum - maximum).exp();
    let right_sum = right.sum * (right.maximum - maximum).exp();
    let combined = left_sum + right_sum;
    if temperature == 1.0 && right_sum > 0.0 {
        let uniform = splitmix_uniform(seed ^ 0xa4093822299f31d0);
        if uniform * combined < right_sum {
            left.token = right.token;
        }
    }
    left.sum = combined;
    left.maximum = maximum;
    left.target = left.target.max(right.target);
    if temperature != 1.0
        && (right.best > left.best || (right.best == left.best && right.token < left.token))
    {
        left.best = right.best;
        left.token = right.token;
    }
    left
}

#[inline(always)]
pub(crate) fn run_encode(
    readout: &[u16],
    tokens: &[u32],
    positions: &[u16],
    mask: &[u16],
    output: &mut DisjointSlice<u16>,
    shape: Mhc4EncodeShape,
) {
    let index = thread::blockIdx_x() * 256 + thread::threadIdx_x();
    if index >= shape.rows * shape.width {
        return;
    }
    let row = index / shape.width;
    let dimension = index % shape.width;
    let first = row * shape.patch;
    let mut value = 0.0_f32;
    let mut count = 0_u32;
    let mut intra = 0_u32;
    while intra < shape.patch {
        let token = tokens[(first + intra) as usize];
        if token < shape.vocabulary {
            value = half(positions[(intra * shape.width + dimension) as usize]).mul_add(
                half(readout[dimension as usize * shape.vocabulary as usize + token as usize]),
                value,
            );
            count += 1;
        }
        intra += 1;
    }
    unsafe {
        *output.get_unchecked_mut(index as usize) = if count == 0 {
            mask[dimension as usize]
        } else {
            half_bits(value * (shape.patch as f32 / count as f32).sqrt())
        };
    }
}

#[inline(always)]
pub(crate) fn run_centroids(
    readout: &[u16],
    centroids: &mut DisjointSlice<u16>,
    _shape: WaveShape,
) {
    let gid = thread::blockIdx_x() * 256 + thread::threadIdx_x();
    let dimensions = 64_u32;
    let nodes = 37_448_u32;
    if gid >= nodes * dimensions {
        return;
    }
    let node = gid / dimensions;
    let dimension = gid % dimensions;
    let (offset, leaves) = if node < 8 {
        (0, 32_768)
    } else if node < 72 {
        (8, 4_096)
    } else if node < 584 {
        (72, 512)
    } else if node < 4_680 {
        (584, 64)
    } else {
        (4_680, 8)
    };
    let first = (node - offset) * leaves;
    let mut value = 0.0_f32;
    let mut token = 0_u32;
    while token < leaves {
        value += half(readout[(dimension * VOCAB + first + token) as usize]);
        token += 1;
    }
    unsafe {
        *centroids.get_unchecked_mut(gid as usize) =
            half_bits(value * (leaves as f32).sqrt().recip());
    }
}

#[inline(always)]
unsafe fn expand_init(
    hidden: &[u16],
    qkv: &[u16],
    base: *mut f32,
    prefix: *mut f32,
    block: u32,
    shape: WaveShape,
) {
    let tid = thread::threadIdx_x();
    let branch = tid >> 5;
    let mut dimension = tid & 31;
    while dimension < 64 {
        if branch == 0 {
            let mut value = 0.0_f32;
            let mut inner = 0_u32;
            while inner < 512 {
                value = half(hidden[((shape.proposal_start + block) * 512 + inner) as usize])
                    .mul_add(half(qkv[(inner * 640 + dimension) as usize]), value);
                inner += 1;
            }
            unsafe { base.add(dimension as usize).write(value * 0.0441941738) };
        }
        unsafe { prefix.add((branch * 64 + dimension) as usize).write(0.0) };
        dimension += 32;
    }
}

#[inline(always)]
unsafe fn expand_level(
    centroids: &[u16],
    readout: &[u16],
    seeds: &[u64],
    query: *const u16,
    logits: *mut f32,
    sampled: *mut f32,
    selected: *const u32,
    position: u32,
    target: u32,
    level: u32,
    shape: WaveShape,
) {
    let tid = thread::threadIdx_x();
    let branch = tid >> 5;
    let lane = tid & 31;
    if lane >= 8 {
        return;
    }
    let child = lane & 7;
    let parent = if branch != 0 {
        target >> ((6 - level) * 3)
    } else {
        unsafe { selected.read() }
    };
    let index = parent * 8 + child;
    let mut value = 0.0_f32;
    let mut dimension = 0;
    while dimension < 64 {
        let weight = if level < 5 {
            let offset = match level {
                0 => 0,
                1 => 8,
                2 => 72,
                3 => 584,
                _ => 4_680,
            };
            half(centroids[((offset + index) * 64 + dimension) as usize])
        } else {
            half(readout[(dimension * VOCAB + index) as usize])
        };
        value = half(unsafe { query.add((branch * 64 + dimension) as usize).read() })
            .mul_add(weight, value);
        dimension += 1;
    }
    unsafe { logits.add((branch * 8 + child) as usize).write(value) };
    if branch == 0 {
        let mut choice = value;
        if shape.temperature > 0.0 {
            let key = (u64::from(level) << 56) ^ u64::from(index);
            let uniform = splitmix_uniform(seeds[0] ^ (u64::from(position) << 32) ^ key);
            choice -= shape.temperature * (-uniform.ln()).ln();
        }
        unsafe { sampled.add(child as usize).write(choice) };
    }
}

#[inline(always)]
unsafe fn expand_choose(
    logits: *const f32,
    sampled: *const f32,
    selected: *mut u32,
    token_loss: *mut f32,
    target: u32,
    level: u32,
    score_flag: u32,
) {
    if thread::threadIdx_x() != 0 {
        return;
    }
    let mut choice = 0_u32;
    let mut child_index = 1_u32;
    while child_index < 8 {
        unsafe {
            if sampled.add(child_index as usize).read() > sampled.add(choice as usize).read() {
                choice = child_index;
            }
        }
        child_index += 1;
    }
    unsafe { selected.write(selected.read() * 8 + choice) };
    if score_flag != 0 {
        let mut maximum = unsafe { logits.add(8).read() };
        child_index = 1;
        while child_index < 8 {
            maximum = maximum.max(unsafe { logits.add((8 + child_index) as usize).read() });
            child_index += 1;
        }
        let mut sum = 0.0_f32;
        child_index = 0;
        while child_index < 8 {
            sum += (unsafe { logits.add((8 + child_index) as usize).read() } - maximum).exp();
            child_index += 1;
        }
        let target_child = (target >> ((5 - level) * 3)) & 7;
        unsafe {
            token_loss.write(
                token_loss.read() + maximum + sum.ln()
                    - logits.add((8 + target_child) as usize).read(),
            )
        };
    }
}

#[inline(always)]
unsafe fn expand_advance(
    positions: &[u16],
    readout: &[u16],
    prefix: *mut f32,
    selected: *const u32,
    token_loss: *const f32,
    output: &mut DisjointSlice<u32>,
    losses: &mut DisjointSlice<f32>,
    intra: u32,
    position: u32,
    target: u32,
    score_flag: u32,
) {
    let tid = thread::threadIdx_x();
    if tid == 0 {
        unsafe {
            *output.get_unchecked_mut(position as usize) = selected.read();
            if score_flag != 0 {
                *losses.get_unchecked_mut(position as usize) = token_loss.read();
            }
        }
    }
    let branch = tid >> 5;
    let mut dimension = tid & 31;
    while dimension < 64 {
        let token = if branch == 0 {
            unsafe { selected.read() }
        } else {
            target
        };
        let retention = sigmoid(half(positions[(intra * 512 + dimension) as usize]) + 1.0986123);
        let scale = (1.0 - retention * retention).max(1.0e-6).sqrt();
        let slot = (branch * 64 + dimension) as usize;
        unsafe {
            prefix.add(slot).write(retention.mul_add(
                prefix.add(slot).read(),
                scale * half(readout[(dimension * VOCAB + token) as usize]),
            ));
        }
        dimension += 32;
    }
}

#[inline(always)]
unsafe fn expand_query(
    positions: &[u16],
    query: *mut u16,
    base: *const f32,
    prefix: *const f32,
    selected: *mut u32,
    token_loss: *mut f32,
    intra: u32,
) {
    let tid = thread::threadIdx_x();
    let branch = tid >> 5;
    let mut dimension = tid & 31;
    while dimension < 64 {
        unsafe {
            query
                .add((branch * 64 + dimension) as usize)
                .write(half_bits(
                    base.add(dimension as usize).read()
                        + half(positions[(intra * 512 + dimension) as usize])
                        + prefix.add((branch * 64 + dimension) as usize).read(),
                ));
        }
        dimension += 32;
    }
    if tid == 0 {
        unsafe {
            selected.write(0);
            token_loss.write(0.0);
        }
    }
}

#[inline(always)]
pub(crate) fn run_mhc4expand(
    hidden: &[u16],
    qkv: &[u16],
    positions: &[u16],
    readout: &[u16],
    centroids: &[u16],
    seeds: &[u64],
    expected: &[u32],
    output: &mut DisjointSlice<u32>,
    losses: &mut DisjointSlice<f32>,
    shape: WaveShape,
) {
    static mut QUERY: SharedArray<u16, 128> = SharedArray::UNINIT;
    static mut BASE: SharedArray<f32, 64> = SharedArray::UNINIT;
    static mut PREFIX: SharedArray<f32, 128> = SharedArray::UNINIT;
    static mut LOGITS: SharedArray<f32, 16> = SharedArray::UNINIT;
    static mut SAMPLED: SharedArray<f32, 8> = SharedArray::UNINIT;
    static mut SELECTED: SharedArray<u32, 1> = SharedArray::UNINIT;
    static mut TOKEN_LOSS: SharedArray<f32, 1> = SharedArray::UNINIT;

    let block = thread::blockIdx_x();
    let first = block * 64;
    if first >= shape.tokens {
        return;
    }
    let query = unsafe { SharedArray::as_raw_mut_ptr(&raw mut QUERY) };
    let base = unsafe { SharedArray::as_raw_mut_ptr(&raw mut BASE) };
    let prefix = unsafe { SharedArray::as_raw_mut_ptr(&raw mut PREFIX) };
    let logits = unsafe { SharedArray::as_raw_mut_ptr(&raw mut LOGITS) };
    let sampled = unsafe { SharedArray::as_raw_mut_ptr(&raw mut SAMPLED) };
    let selected = unsafe { SharedArray::as_raw_mut_ptr(&raw mut SELECTED) };
    let token_loss = unsafe { SharedArray::as_raw_mut_ptr(&raw mut TOKEN_LOSS) };
    unsafe { expand_init(hidden, qkv, base, prefix, block, shape) };
    thread::sync_threads();

    let mut intra = 0_u32;
    while intra < 64 && first + intra < shape.tokens {
        let position = first + intra;
        unsafe { expand_query(positions, query, base, prefix, selected, token_loss, intra) };
        thread::sync_threads();
        let target = if shape.score != 0 {
            expected[position as usize]
        } else {
            0
        };
        for level in 0..6 {
            unsafe {
                expand_level(
                    centroids, readout, seeds, query, logits, sampled, selected, position, target,
                    level, shape,
                );
                thread::sync_threads();
                expand_choose(
                    logits,
                    sampled,
                    selected,
                    token_loss,
                    target,
                    level,
                    shape.score,
                );
                thread::sync_threads();
            }
        }
        unsafe {
            expand_advance(
                positions,
                readout,
                prefix,
                selected,
                token_loss,
                output,
                losses,
                intra,
                position,
                target,
                shape.score,
            )
        };
        thread::sync_threads();
        intra += 1;
    }
}

#[inline(always)]
pub(crate) fn run_init(
    hidden: &[u16],
    qkv: &[u16],
    positions: &[u16],
    base: &mut DisjointSlice<f32>,
    prefix: &mut DisjointSlice<f32>,
    query: &mut DisjointSlice<u16>,
    shape: WaveShape,
) {
    let index = thread::blockIdx_x() * 256 + thread::threadIdx_x();
    if index >= shape.blocks * 64 {
        return;
    }
    let block = index / 64;
    let dimension = index % 64;
    let mut value = 0.0_f32;
    let mut inner = 0_u32;
    while inner < 512 {
        value = half(hidden[((shape.proposal_start + block) * 512 + inner) as usize])
            .mul_add(half(qkv[(inner * 640 + dimension) as usize]), value);
        inner += 1;
    }
    let value = value * 0.0441941738_f32;
    unsafe {
        *base.get_unchecked_mut(index as usize) = value;
        *prefix.get_unchecked_mut(index as usize) = 0.0;
        *query.get_unchecked_mut(index as usize) =
            half_bits(value + half(positions[dimension as usize]));
    }
}
