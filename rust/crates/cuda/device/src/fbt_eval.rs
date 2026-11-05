use cuda_device::{DisjointSlice, SharedArray, thread};

use super::fbt_math::{reduce_max, reduce_sum};
use super::fbt_types::{Mhc4EncodeShape, ObjectiveShape, ScoreStats};

#[inline(always)]
pub(crate) fn run_encode(
    readout: &[f32],
    tokens: &[u32],
    positions: &[f32],
    mask: &[f32],
    output: &mut DisjointSlice<f32>,
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
            value = positions[(intra * shape.width + dimension) as usize].mul_add(
                readout[dimension as usize * shape.vocabulary as usize + token as usize],
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
            value * (shape.patch as f32 / count as f32).sqrt()
        };
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
