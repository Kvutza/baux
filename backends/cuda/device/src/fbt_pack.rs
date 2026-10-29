use cuda_device::{DisjointSlice, SharedArray, thread};

use super::fbt_math::{reduce_max, reduce_sum};
use super::fbt_types::{ObjectiveShape, ScoreStats, VOCAB, half};

#[inline(always)]
pub(crate) fn run_verify(
    predicted: &[u32],
    tokens: &mut DisjointSlice<u32>,
    frontier: &mut DisjointSlice<u32>,
    rows: u32,
) {
    static mut FIRST: SharedArray<u32, 256> = SharedArray::UNINIT;
    let tid = thread::threadIdx_x();
    let first = unsafe { SharedArray::as_raw_mut_ptr(&raw mut FIRST) };
    let start = unsafe { frontier.as_mut_ptr().read() };
    if start > rows {
        return;
    }
    let data = tokens.as_mut_ptr();
    let mut mismatch = rows;
    let mut position = start + tid;
    while position < rows {
        if unsafe { data.add(position as usize).read() } != predicted[(position - 1) as usize] {
            mismatch = mismatch.min(position);
        }
        position += 256;
    }
    unsafe { first.add(tid as usize).write(mismatch) };
    thread::sync_threads();
    let mut stride = 128;
    while stride > 0 {
        if tid < stride {
            unsafe {
                first.add(tid as usize).write(
                    first
                        .add(tid as usize)
                        .read()
                        .min(first.add((tid + stride) as usize).read()),
                );
            }
        }
        thread::sync_threads();
        stride /= 2;
    }
    let mismatch = unsafe { first.read() };
    position = mismatch + tid;
    while position < rows {
        unsafe {
            *tokens.get_unchecked_mut(position as usize) =
                predicted[(position - 1) as usize].min(VOCAB - 1);
        }
        position += 256;
    }
    thread::sync_threads();
    if tid == 0 {
        unsafe {
            *frontier.get_unchecked_mut(0) = if mismatch == rows {
                rows
            } else if predicted[(mismatch - 1) as usize] < VOCAB {
                mismatch + 1
            } else {
                rows + 1
            };
        }
    }
}

#[inline(always)]
unsafe fn sample_max(logits: &[u16], scores: *mut f32, row: u32, temperature: f32) -> f32 {
    if temperature <= 0.0 {
        return 0.0;
    }
    let tid = thread::threadIdx_x();
    let mut maximum = f32::NEG_INFINITY;
    let mut token = tid;
    while token < VOCAB {
        let value = half(logits[row as usize * VOCAB as usize + token as usize]);
        if value.is_finite() {
            maximum = maximum.max(value);
        }
        token += 256;
    }
    unsafe { scores.add(tid as usize).write(maximum) };
    let maximum = unsafe { reduce_max(scores) };
    thread::sync_threads();
    maximum
}

#[inline(always)]
unsafe fn sample_reduce(scores: *mut f32, ids: *mut u32, best: f32, selected: u32, tid: u32) {
    unsafe {
        scores.add(tid as usize).write(best);
        ids.add(tid as usize).write(selected);
    }
    thread::sync_threads();
    let mut stride = 128;
    while stride > 0 {
        if tid < stride {
            let right = tid + stride;
            let a = unsafe { scores.add(tid as usize).read() };
            let b = unsafe { scores.add(right as usize).read() };
            let x = unsafe { ids.add(tid as usize).read() };
            let y = unsafe { ids.add(right as usize).read() };
            if b > a || (b == a && y < x) {
                unsafe {
                    scores.add(tid as usize).write(b);
                    ids.add(tid as usize).write(y);
                }
            }
        }
        thread::sync_threads();
        stride /= 2;
    }
}

#[inline(always)]
pub(crate) fn run_sample(
    logits: &[u16],
    tokens: &mut DisjointSlice<u32>,
    rows: u32,
    temperature: f32,
    seed: u64,
    first_row: u32,
) {
    static mut SCORES: SharedArray<f32, 256> = SharedArray::UNINIT;
    static mut IDS: SharedArray<u32, 256> = SharedArray::UNINIT;
    let row = thread::blockIdx_x();
    if row >= rows {
        return;
    }
    let tid = thread::threadIdx_x();
    let scores = unsafe { SharedArray::as_raw_mut_ptr(&raw mut SCORES) };
    let maximum = unsafe { sample_max(logits, scores, row, temperature) };
    let mut best = f32::NEG_INFINITY;
    let mut selected = u32::MAX;
    let mut token = tid;
    while token < VOCAB {
        let logit = half(logits[row as usize * VOCAB as usize + token as usize]);
        if logit.is_finite() {
            let mut score = logit;
            if temperature > 0.0 {
                let mut hash = seed ^ (u64::from(first_row + row) << 32) ^ u64::from(token);
                hash = (hash ^ (hash >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
                hash = (hash ^ (hash >> 27)).wrapping_mul(0x94d049bb133111eb);
                hash ^= hash >> 31;
                let uniform = ((hash >> 41) as f32 + 0.5) / 8388608.0;
                score = (logit - maximum) / temperature - (-uniform.ln()).ln();
            }
            if score > best || (score == best && token < selected) {
                best = score;
                selected = token;
            }
        }
        token += 256;
    }
    let ids = unsafe { SharedArray::as_raw_mut_ptr(&raw mut IDS) };
    unsafe { sample_reduce(scores, ids, best, selected, tid) };
    if tid == 0 {
        unsafe { *tokens.get_unchecked_mut((first_row + row) as usize) = ids.read() };
    }
}

#[inline(always)]
pub(crate) fn run_expand(
    macros: &[u32],
    readout: &[u16],
    tokens: &mut DisjointSlice<u32>,
    patches: u32,
    first_patch: u32,
) {
    let mut patch = thread::threadIdx_x() + thread::blockIdx_x() * 256;
    let stride = thread::gridDim_x() * 256;
    while patch < patches {
        let macro_token = macros[(first_patch + patch) as usize].min(VOCAB - 1);
        let base = patch * 64;
        unsafe { *tokens.get_unchecked_mut(base as usize) = macro_token };
        let mut previous = macro_token;
        let mut intra = 1_u32;
        while intra < 64 {
            let dimension = (intra * 17) % 512;
            let weight = readout[dimension as usize * VOCAB as usize + previous as usize];
            let mut mixed = (u64::from(previous) << 32)
                ^ (u64::from(macro_token) << 16)
                ^ (u64::from(intra) << 8)
                ^ u64::from(weight);
            mixed = mixed.wrapping_add(0x9e3779b97f4a7c15);
            mixed = (mixed ^ (mixed >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
            mixed = (mixed ^ (mixed >> 27)).wrapping_mul(0x94d049bb133111eb);
            mixed ^= mixed >> 31;
            previous = (mixed % u64::from(VOCAB)) as u32;
            unsafe { *tokens.get_unchecked_mut((base + intra) as usize) = previous };
            intra += 1;
        }
        patch += stride;
    }
}

#[inline(always)]
pub(crate) fn run_score(
    logits: &[u16],
    targets: &[u32],
    losses: &mut DisjointSlice<f32>,
    rows: u32,
    target_offset: u32,
    loss_offset: u32,
) {
    static mut MAXIMA: SharedArray<f32, 256> = SharedArray::UNINIT;
    static mut TOTALS: SharedArray<f32, 256> = SharedArray::UNINIT;
    let row = thread::blockIdx_x();
    if row >= rows {
        return;
    }
    let tid = thread::threadIdx_x();
    let maxima = unsafe { SharedArray::as_raw_mut_ptr(&raw mut MAXIMA) };
    let totals = unsafe { SharedArray::as_raw_mut_ptr(&raw mut TOTALS) };
    let mut local_maximum = f32::NEG_INFINITY;
    let mut local_total = 0.0;
    let mut token = tid;
    while token < VOCAB {
        let value = half(logits[row as usize * VOCAB as usize + token as usize]);
        let maximum = local_maximum.max(value);
        local_total = local_total * (local_maximum - maximum).exp() + (value - maximum).exp();
        local_maximum = maximum;
        token += 256;
    }
    unsafe {
        maxima.add(tid as usize).write(local_maximum);
        totals.add(tid as usize).write(local_total);
    }
    thread::sync_threads();
    let mut stride = 128;
    while stride > 0 {
        if tid < stride {
            unsafe {
                let left_maximum = maxima.add(tid as usize).read();
                let right_maximum = maxima.add((tid + stride) as usize).read();
                let maximum = left_maximum.max(right_maximum);
                let total = totals.add(tid as usize).read() * (left_maximum - maximum).exp()
                    + totals.add((tid + stride) as usize).read() * (right_maximum - maximum).exp();
                maxima.add(tid as usize).write(maximum);
                totals.add(tid as usize).write(total);
            }
        }
        thread::sync_threads();
        stride /= 2;
    }
    if tid == 0 {
        let target = targets[(target_offset + row) as usize].min(VOCAB - 1);
        let target_logit = half(logits[row as usize * VOCAB as usize + target as usize]);
        let maximum = unsafe { maxima.read() };
        let normalizer = unsafe { totals.read() };
        unsafe {
            *losses.get_unchecked_mut((loss_offset + row) as usize) =
                maximum + normalizer.ln() - target_logit;
        }
    }
}

#[inline(always)]
unsafe fn objective_octiles(losses: &[f32], patch_nll: *mut f32, tid: u32, shape: ObjectiveShape) {
    if tid < 8 {
        let patch = shape.patch.max(1);
        let begin = tid * patch / 8;
        let finish = (tid + 1) * patch / 8;
        let mut patch_sum = 0.0;
        let mut patch_count = 0_u32;
        let mut base = 0_u32;
        while base < shape.losses {
            let end = (base + finish).min(shape.losses);
            let mut position = base + begin;
            while position < end {
                patch_sum += losses[position as usize];
                patch_count += 1;
                position += 1;
            }
            base += patch;
        }
        unsafe {
            patch_nll.add(tid as usize).write(if patch_count == 0 {
                f32::NAN
            } else {
                patch_sum / patch_count as f32
            });
        }
    }
}

#[inline(always)]
pub(crate) fn run_objective(
    losses: &[f32],
    generated: &[u32],
    targets: &[u32],
    output: &mut DisjointSlice<ScoreStats>,
    shape: ObjectiveShape,
) {
    static mut SUMS: SharedArray<f32, 256> = SharedArray::UNINIT;
    static mut MAXIMA: SharedArray<f32, 256> = SharedArray::UNINIT;
    static mut MATCHES: SharedArray<f32, 256> = SharedArray::UNINIT;
    static mut PATCH_NLL: SharedArray<f32, 8> = SharedArray::UNINIT;
    let tid = thread::threadIdx_x();
    let sums = unsafe { SharedArray::as_raw_mut_ptr(&raw mut SUMS) };
    let maxima = unsafe { SharedArray::as_raw_mut_ptr(&raw mut MAXIMA) };
    let matches = unsafe { SharedArray::as_raw_mut_ptr(&raw mut MATCHES) };
    let patch_nll = unsafe { SharedArray::as_raw_mut_ptr(&raw mut PATCH_NLL) };
    let mut sum = 0.0;
    let mut maximum = f32::NEG_INFINITY;
    let mut matched = 0.0;
    let mut row = tid;
    while row < shape.losses {
        let loss = losses[row as usize];
        sum += loss;
        maximum = maximum.max(loss);
        row += 256;
    }
    row = tid;
    while row < shape.tokens {
        if generated[row as usize] == targets[row as usize] {
            matched += 1.0;
        }
        row += 256;
    }
    unsafe { objective_octiles(losses, patch_nll, tid, shape) };
    thread::sync_threads();
    unsafe {
        sums.add(tid as usize).write(sum);
        maxima.add(tid as usize).write(maximum);
        matches.add(tid as usize).write(matched);
    }
    let sum = unsafe { reduce_sum(sums) };
    let maximum = unsafe { reduce_max(maxima) };
    let matched = unsafe { reduce_sum(matches) };
    if tid == 0 {
        let loss_count = shape.losses.max(1) as f32;
        let token_count = shape.tokens.max(1) as f32;
        let mean = sum / loss_count;
        unsafe {
            *output.get_unchecked_mut(0) = ScoreStats {
                reward: -mean,
                mean_nll: mean,
                maximum_nll: maximum,
                positional_accuracy: matched / token_count,
                tokens: shape.tokens,
                mismatches: shape.tokens - matched as u32,
                patch_nll_octiles: [
                    *patch_nll.add(0),
                    *patch_nll.add(1),
                    *patch_nll.add(2),
                    *patch_nll.add(3),
                    *patch_nll.add(4),
                    *patch_nll.add(5),
                    *patch_nll.add(6),
                    *patch_nll.add(7),
                ],
            };
        }
    }
}
