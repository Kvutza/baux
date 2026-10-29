use cuda_device::{DisjointSlice, SharedArray, thread, warp};

use super::fbt_math::reduce_sum;
use super::fbt_types::{Mhc4Point, ObjectiveStats, PairShape, RoutedTile, half};

#[inline(always)]
pub(crate) fn run_pack(
    input: &[u16],
    experts: &[u32],
    offsets: &[u32],
    prefixes: &[u32],
    mapping: &mut DisjointSlice<u32>,
    packed: &mut DisjointSlice<u16>,
    assignments: u32,
) {
    static mut SUMS: SharedArray<f32, 256> = SharedArray::UNINIT;
    let assignment = thread::blockIdx_x();
    if assignment >= assignments {
        return;
    }
    let tid = thread::threadIdx_x();
    let expert = experts[assignment as usize];
    let chunk = assignment / 1024;
    let mut index = chunk * 1024 + tid;
    let mut rank = 0.0;
    while index < assignment {
        if experts[index as usize] == expert {
            rank += 1.0;
        }
        index += 256;
    }
    let sums = unsafe { SharedArray::as_raw_mut_ptr(&raw mut SUMS) };
    unsafe { sums.add(tid as usize).write(rank) };
    let target = offsets[expert as usize]
        + prefixes[(expert * assignments.div_ceil(1024) + chunk) as usize]
        + unsafe { reduce_sum(sums) } as u32;
    if tid == 0 {
        unsafe { *mapping.get_unchecked_mut(assignment as usize) = target };
    }
    for item in 0..2 {
        let column = tid + item * 256;
        unsafe {
            *packed.get_unchecked_mut((target * 512 + column) as usize) =
                input[(assignment / 3 * 512 + column) as usize];
        }
    }
}

#[inline(always)]
pub(crate) fn run_layout(
    counts: &[u32],
    offsets: &mut DisjointSlice<u32>,
    tiles: &mut DisjointSlice<RoutedTile>,
    experts: u32,
    tile_capacity: u32,
) {
    let lane = thread::threadIdx_x();
    let mut start = 0;
    let mut tile = 0;
    for expert in 0..experts {
        if lane == 0 {
            unsafe { *offsets.get_unchecked_mut(expert as usize) = start };
        }
        let count = counts[expert as usize];
        let expert_tiles = count.div_ceil(64);
        let mut local = lane;
        while local < expert_tiles {
            let row = local * 64;
            unsafe {
                *tiles.get_unchecked_mut((tile + local) as usize) = RoutedTile {
                    expert,
                    first_row: start + row,
                    valid_rows: (count - row).min(64),
                    reserved: 0,
                };
            }
            local += 32;
        }
        start += count;
        tile += expert_tiles;
    }
    let mut index = tile + lane;
    while index < tile_capacity {
        unsafe {
            *tiles.get_unchecked_mut(index as usize) = RoutedTile {
                expert: 0,
                first_row: 0,
                valid_rows: 0,
                reserved: 0,
            };
        }
        index += 32;
    }
}

#[inline(always)]
pub(crate) fn run_capture(
    base: &[u16],
    points: &[Mhc4Point],
    history: &mut DisjointSlice<u16>,
    point_start: u32,
    point_count: u32,
    total_points: u32,
    row: u32,
) {
    let local = thread::blockIdx_x() * 256 + thread::threadIdx_x();
    if local >= point_count {
        return;
    }
    let index = point_start + local;
    let point = points[index as usize];
    unsafe {
        *history.get_unchecked_mut((row * total_points + index) as usize) =
            base[point.physical as usize];
    }
}

#[inline(always)]
pub(crate) fn run_distances(
    pool: &[u16],
    history: &[u16],
    points: &[Mhc4Point],
    distances: &mut DisjointSlice<f32>,
    total_points: u32,
    history_count: u32,
) {
    static mut SUMS: SharedArray<f32, 256> = SharedArray::UNINIT;
    let pair = thread::blockIdx_x();
    let pairs = 4 * history_count;
    if pair >= pairs {
        return;
    }
    let candidate = pair / history_count;
    let row = pair % history_count;
    let tid = thread::threadIdx_x();
    let mut sum = 0.0_f32;
    let mut index = tid;
    while index < total_points {
        let point = points[index as usize];
        let delta = half(pool[(candidate * total_points + index) as usize])
            - half(history[(row * total_points + index) as usize]);
        sum += delta * delta * point.metric_weight;
        index += 256;
    }
    let sums = unsafe { SharedArray::as_raw_mut_ptr(&raw mut SUMS) };
    unsafe { sums.add(tid as usize).write(sum) };
    thread::sync_threads();
    let mut step = 128;
    while step > 0 {
        if tid < step {
            unsafe {
                sums.add(tid as usize)
                    .write(sums.add(tid as usize).read() + sums.add((tid + step) as usize).read());
            }
        }
        thread::sync_threads();
        step >>= 1;
    }
    if tid == 0 {
        unsafe { *distances.get_unchecked_mut(pair as usize) = sums.read() };
    }
}

#[inline(always)]
pub(crate) fn run_patchnll(losses: &[f32], patches: &mut DisjointSlice<f32>, tokens: u32) {
    static mut SUMS: SharedArray<f32, 64> = SharedArray::UNINIT;
    let patch = thread::blockIdx_x();
    if patch >= tokens.div_ceil(64) {
        return;
    }
    let tid = thread::threadIdx_x();
    let sums = unsafe { SharedArray::as_raw_mut_ptr(&raw mut SUMS) };
    unsafe {
        sums.add(tid as usize).write(if patch * 64 + tid < tokens {
            losses[(patch * 64 + tid) as usize]
        } else {
            0.0
        });
    }
    thread::sync_threads();
    let mut step = 32;
    while step > 0 {
        if tid < step {
            unsafe {
                let value = sums.add(tid as usize).read() + sums.add((tid + step) as usize).read();
                sums.add(tid as usize).write(value);
            }
        }
        thread::sync_threads();
        step >>= 1;
    }
    if tid == 0 {
        unsafe {
            *patches.get_unchecked_mut(patch as usize) =
                sums.read() / (tokens - patch * 64).min(64) as f32
        };
    }
}

#[inline(always)]
fn obj_tokens(
    losses: &[f32],
    generated: &[u32],
    expected: &[u32],
    baseline: &[f32],
    patches: &mut DisjointSlice<f32>,
    shape: PairShape,
) -> ([f32; 8], [u32; 3], u32) {
    let tid = thread::threadIdx_x();
    let mut values = [0.0_f32; 8];
    values[6] = f32::NEG_INFINITY;
    let mut flags = [0_u32, 0, u32::MAX];
    let mut position = tid;
    while position < shape.tokens {
        let loss = losses[position as usize];
        values[0] += loss;
        if shape.paired != 0 {
            values[3] += baseline[position as usize] - loss;
        }
        values[6] = values[6].max(loss);
        if generated[position as usize] == expected[position as usize] {
            flags[0] += 1;
        } else {
            flags[1] += 1;
            flags[2] = flags[2].min(position);
            values[7] += loss;
        }
        position += 256;
    }
    let batch_tokens = shape.batch_tokens.max(1);
    let batches = shape.tokens.div_ceil(batch_tokens);
    let mut batch = tid;
    while batch < batches {
        let begin = batch * batch_tokens;
        let end = (begin + batch_tokens).min(shape.tokens);
        let (mut loss, mut difference) = (0.0_f32, 0.0_f32);
        for pos in begin..end {
            loss += losses[pos as usize];
            if shape.paired != 0 {
                difference += baseline[pos as usize] - losses[pos as usize];
            }
        }
        let inverse = 1.0 / (end - begin) as f32;
        let centered = loss * inverse - 12.0;
        difference *= inverse;
        if shape.paired != 0 {
            unsafe { *patches.get_unchecked_mut(batch as usize) = difference };
        }
        values[1] += centered;
        values[2] += centered * centered;
        if shape.paired != 0 {
            values[4] += difference;
            values[5] += difference * difference;
        }
        batch += 256;
    }
    (values, flags, batches)
}

#[inline(always)]
unsafe fn obj_octiles(losses: &[f32], octiles: *mut f32, shape: PairShape) {
    let tid = thread::threadIdx_x();
    if tid < 8 {
        let batch_tokens = shape.batch_tokens.max(1);
        let begin = tid * batch_tokens / 8;
        let finish = (tid + 1) * batch_tokens / 8;
        let (mut sum, mut count, mut base) = (0.0_f32, 0_u32, 0_u32);
        while base < shape.tokens {
            let end = (base + finish).min(shape.tokens);
            for pos in (base + begin)..end {
                sum += losses[pos as usize];
                count += 1;
            }
            base += batch_tokens;
        }
        unsafe {
            octiles.add(tid as usize).write(if count == 0 {
                f32::NAN
            } else {
                sum / count as f32
            });
        }
    }
}

#[inline(always)]
unsafe fn obj_reduce(
    output: &mut DisjointSlice<ObjectiveStats>,
    floats: *const f32,
    counts: *const u32,
    octiles: *const f32,
    batches: u32,
) {
    let lane = thread::threadIdx_x() & 31;
    let mut values = [0.0_f32; 8];
    for (channel, slot) in values.iter_mut().enumerate() {
        let value = if lane < 8 {
            unsafe { floats.add(channel * 8 + lane as usize).read() }
        } else if channel == 6 {
            f32::NEG_INFINITY
        } else {
            0.0
        };
        *slot = if channel == 6 {
            warp::reduce_max_f32(value)
        } else {
            warp::reduce_sum_f32(value)
        };
    }
    let mut flags = [0_u32; 3];
    for (channel, slot) in flags.iter_mut().enumerate() {
        *slot = if lane < 8 {
            unsafe { counts.add(channel * 8 + lane as usize).read() }
        } else if channel == 2 {
            u32::MAX
        } else {
            0
        };
    }
    let mut offset = 16;
    while offset > 0 {
        flags[0] += warp::shuffle_xor(flags[0], offset);
        flags[1] += warp::shuffle_xor(flags[1], offset);
        flags[2] = flags[2].min(warp::shuffle_xor(flags[2], offset));
        offset >>= 1;
    }
    if lane == 0 {
        unsafe {
            *output.get_unchecked_mut(0) = ObjectiveStats {
                sum: values[0],
                batch_centered_sum: values[1],
                squared_batch_centered_sum: values[2],
                difference_sum: values[3],
                difference_batch_sum: values[4],
                squared_difference_batch_sum: values[5],
                maximum: values[6],
                mismatch_sum: values[7],
                batches,
                matches: flags[0],
                mismatches: flags[1],
                first_mismatch: flags[2],
                patch_nll_octiles: [
                    *octiles.add(0),
                    *octiles.add(1),
                    *octiles.add(2),
                    *octiles.add(3),
                    *octiles.add(4),
                    *octiles.add(5),
                    *octiles.add(6),
                    *octiles.add(7),
                ],
            };
        }
    }
}

#[inline(always)]
pub(crate) fn run_mhc4obj(
    losses: &[f32],
    generated: &[u32],
    expected: &[u32],
    output: &mut DisjointSlice<ObjectiveStats>,
    baseline: &[f32],
    shape: PairShape,
    patches: &mut DisjointSlice<f32>,
) {
    static mut FLOATS: SharedArray<f32, 64> = SharedArray::UNINIT;
    static mut COUNTS: SharedArray<u32, 24> = SharedArray::UNINIT;
    static mut OCTILES: SharedArray<f32, 8> = SharedArray::UNINIT;
    let tid = thread::threadIdx_x();
    let lane = tid & 31;
    let group = tid >> 5;
    let floats = unsafe { SharedArray::as_raw_mut_ptr(&raw mut FLOATS) };
    let counts = unsafe { SharedArray::as_raw_mut_ptr(&raw mut COUNTS) };
    let octiles = unsafe { SharedArray::as_raw_mut_ptr(&raw mut OCTILES) };
    let (values, mut flags, batches) =
        obj_tokens(losses, generated, expected, baseline, patches, shape);
    unsafe { obj_octiles(losses, octiles, shape) };
    for (channel, &val) in values.iter().enumerate() {
        let reduced = if channel == 6 {
            warp::reduce_max_f32(val)
        } else {
            warp::reduce_sum_f32(val)
        };
        if lane == 0 {
            unsafe { floats.add(channel * 8 + group as usize).write(reduced) };
        }
    }
    let mut offset = 16;
    while offset > 0 {
        flags[0] += warp::shuffle_xor(flags[0], offset);
        flags[1] += warp::shuffle_xor(flags[1], offset);
        flags[2] = flags[2].min(warp::shuffle_xor(flags[2], offset));
        offset >>= 1;
    }
    if lane == 0 {
        for (channel, &flag) in flags.iter().enumerate() {
            unsafe { counts.add(channel * 8 + group as usize).write(flag) };
        }
    }
    thread::sync_threads();
    if group == 0 {
        unsafe { obj_reduce(output, floats, counts, octiles, batches) };
    }
}
