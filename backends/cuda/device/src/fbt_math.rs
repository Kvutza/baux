use super::dense_mix64;
use super::fbt::{FBT_THREADS, FbtShape};
use cuda_device::{DisjointSlice, SharedArray, thread, warp};

pub use baux::semantics::gaussian::{bits as gaussian_bits, mix32 as gaussian_mix32};

/// One deterministic coordinate in stratified neighbour quadrature.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct NeighborPoint {
    pub physical: u64,
    pub key: u64,
    pub element: u64,
    pub metric_weight: f32,
    pub scale: f32,
}

pub type Mhc4Point = NeighborPoint;

// SAFETY: repr(C), with only DeviceCopy scalar fields.
unsafe impl cuda_core::DeviceCopy for NeighborPoint {}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GaussianShape {
    pub elements: u64,
    pub tensor_id: u64,
    pub logical_start: u64,
    pub seed: u64,
    pub block_len: u32,
    pub radius: f32,
    pub transpose_columns: u32,
    pub pad: u32,
}

// SAFETY: GaussianShape is repr(C) and contains only DeviceCopy scalars.
unsafe impl cuda_core::DeviceCopy for GaussianShape {}

/// OR-reduce the two publication flags, with all 256 lanes participating.
#[inline(always)]
pub fn block_flags(flags: u32) -> u32 {
    static mut FLAGS: SharedArray<u32, 8> = SharedArray::UNINIT;
    let shared = unsafe { SharedArray::as_raw_mut_ptr(&raw mut FLAGS) };
    let tid = thread::threadIdx_x();
    let flags = u32::from(warp::any(flags & baux::semantics::proposal::INVALID != 0))
        * baux::semantics::proposal::INVALID
        | u32::from(warp::any(flags & baux::semantics::proposal::CHANGED != 0))
            * baux::semantics::proposal::CHANGED;
    if tid & 31 == 0 {
        unsafe {
            shared.add((tid >> 5) as usize).write(flags);
        }
    }
    thread::sync_threads();
    let mut result = 0;
    if tid == 0 {
        for warp in 0..8 {
            result |= unsafe { shared.add(warp).read() };
        }
    }
    result
}

#[inline(always)]
pub fn half(bits: u16) -> f32 {
    let sign = u32::from(bits & 0x8000) << 16;
    let exponent = u32::from(bits & 0x7c00);
    let mantissa = u32::from(bits & 0x03ff);
    if exponent == 0x7c00 {
        let payload = if mantissa == 0 {
            0
        } else {
            0x0040_0000 | (mantissa << 13)
        };
        return f32::from_bits(sign | 0x7f80_0000 | payload);
    }
    f16::from_bits(bits) as f32
}

#[inline(always)]
pub fn half_bits(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = (bits & 0x8000_0000) >> 16;
    let exponent = bits & 0x7f80_0000;
    let mantissa = bits & 0x007f_ffff;
    if exponent == 0x7f80_0000 {
        let nan = if mantissa == 0 { 0 } else { 0x0200 };
        return (sign | 0x7c00 | nan | (mantissa >> 13)) as u16;
    }
    (value as f16).to_bits()
}

#[inline(always)]
pub fn gaussian_uniform(bits: u32) -> f32 {
    ((((bits >> 8) + 1) as f32) * 0.000_000_059_604_645).min(0.999_999_94)
}

#[inline(always)]
fn gaussian_tail(seed: u64, key: u64, element: u64, attempt: u32, signed: i32) -> f32 {
    let mut tail = 0_u32;
    loop {
        let x = -gaussian_uniform(gaussian_bits(
            seed,
            key,
            element,
            attempt.wrapping_add(tail),
            1,
        ))
        .ln()
            / 3.654_152_9;
        let y = -gaussian_uniform(gaussian_bits(
            seed,
            key,
            element,
            attempt.wrapping_add(tail),
            2,
        ))
        .ln();
        if 2.0 * y >= x * x {
            return if signed < 0 {
                -3.654_152_9 - x
            } else {
                3.654_152_9 + x
            };
        }
        tail = tail.wrapping_add(1);
    }
}

#[inline(always)]
pub fn gaussian_normal(
    seed: u64,
    key: u64,
    element: u64,
    thresholds: &[u32],
    widths: &[f32],
    densities: &[f32],
) -> f32 {
    let mut attempt = 0_u32;
    loop {
        let bits = gaussian_bits(seed, key, element, attempt, 0);
        let signed = bits as i32;
        let layer = (bits & 255) as usize;
        let magnitude = if signed < 0 {
            (!bits).wrapping_add(1)
        } else {
            bits
        };
        let sample = signed as f32 * widths[layer];
        if magnitude < thresholds[layer] {
            return sample;
        }
        if layer == 0 {
            return gaussian_tail(seed, key, element, attempt, signed);
        }
        let uniform = gaussian_uniform(gaussian_bits(seed, key, element, attempt, 1));
        let density = densities[layer] + uniform * (densities[layer - 1] - densities[layer]);
        if density < (-0.5 * sample * sample).exp() {
            return sample;
        }
        attempt = attempt.wrapping_add(1);
    }
}

#[inline(always)]
pub fn gaussian_block_key(tensor_id: u64, start: u64, length: u64) -> u64 {
    dense_mix64(tensor_id ^ dense_mix64(start) ^ dense_mix64(length ^ 0xa076_1d64_78bd_642f))
}

#[inline(always)]
pub fn splitmix_uniform(mut bits: u64) -> f32 {
    bits = bits.wrapping_add(0x9e3779b97f4a7c15);
    bits = (bits ^ (bits >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    bits = (bits ^ (bits >> 27)).wrapping_mul(0x94d049bb133111eb);
    bits ^= bits >> 31;
    ((bits >> 41) as f32 + 0.5) / 8388608.0
}

#[inline(always)]
pub(crate) fn run_activate(projection: &[u16], output: &mut DisjointSlice<u16>, rows: u32) {
    let index = thread::blockIdx_x() * FBT_THREADS + thread::threadIdx_x();
    if index >= rows * 216 {
        return;
    }
    let row = index / 216;
    let column = index % 216;
    let gate = half(projection[(row * 432 + column) as usize]);
    let up = half(projection[(row * 432 + 216 + column) as usize]);
    unsafe {
        *output.get_unchecked_mut((row * 224 + column) as usize) = half_bits(silu(gate) * up);
    }
}

#[inline(always)]
pub fn silu(value: f32) -> f32 {
    let e = (-value.abs()).exp();
    let sigmoid = if value >= 0.0 {
        1.0 / (1.0 + e)
    } else {
        e / (1.0 + e)
    };
    value * sigmoid
}

#[inline(always)]
pub fn sigmoid(value: f32) -> f32 {
    let e = (-value.abs()).exp();
    if value >= 0.0 {
        1.0 / (1.0 + e)
    } else {
        e / (1.0 + e)
    }
}

#[inline(always)]
pub fn contains_leaf(node: u32, level: u32, leaf: u32) -> bool {
    let begin = node << level;
    leaf >= begin && leaf < begin + (1 << level)
}

#[inline(always)]
pub unsafe fn reduce_sum(values: *mut f32) -> f32 {
    let tid = thread::threadIdx_x() + thread::threadIdx_y() * thread::blockDim_x();
    thread::sync_threads();
    let mut stride = FBT_THREADS / 2;
    while stride > 0 {
        if tid < stride {
            unsafe {
                let value =
                    values.add(tid as usize).read() + values.add((tid + stride) as usize).read();
                values.add(tid as usize).write(value);
            }
        }
        thread::sync_threads();
        stride /= 2;
    }
    unsafe { values.read() }
}

#[inline(always)]
pub unsafe fn reduce_max(values: *mut f32) -> f32 {
    let tid = thread::threadIdx_x() + thread::threadIdx_y() * thread::blockDim_x();
    thread::sync_threads();
    let mut stride = FBT_THREADS / 2;
    while stride > 0 {
        if tid < stride {
            unsafe {
                let left = values.add(tid as usize).read();
                let right = values.add((tid + stride) as usize).read();
                values.add(tid as usize).write(left.max(right));
            }
        }
        thread::sync_threads();
        stride /= 2;
    }
    unsafe { values.read() }
}

#[inline(always)]
pub(crate) fn run_embed(
    weights: &[u16],
    tokens: &[u32],
    output: &mut DisjointSlice<u16>,
    shape: FbtShape,
) {
    let row = thread::blockIdx_x();
    if row >= shape.rows {
        return;
    }
    let token = tokens[row as usize];
    if token >= shape.vocab {
        return;
    }
    let output_base = row * shape.width;
    let source_base = token * shape.width;
    let mut column = thread::threadIdx_x();
    while column < shape.width {
        unsafe {
            *output.get_unchecked_mut((output_base + column) as usize) =
                weights[(source_base + column) as usize];
        }
        column += FBT_THREADS;
    }
}

#[inline(always)]
pub(crate) fn run_residual(
    input: &[u16],
    branch: &[u16],
    output: &mut DisjointSlice<u16>,
    elements: u32,
) {
    let mut index = thread::threadIdx_x() + thread::blockIdx_x() * FBT_THREADS;
    let stride = thread::gridDim_x() * FBT_THREADS;
    while index < elements {
        let value = half(input[index as usize]) + half(branch[index as usize]);
        unsafe { *output.get_unchecked_mut(index as usize) = half_bits(value) };
        index += stride;
    }
}

#[inline(always)]
pub(crate) fn run_rms(
    input: &[u16],
    weight: &[u16],
    output: &mut DisjointSlice<u16>,
    shape: FbtShape,
) {
    static mut REDUCTION: SharedArray<f32, 256> = SharedArray::UNINIT;
    let row = thread::blockIdx_x();
    let tid = thread::threadIdx_x();
    if row >= shape.rows {
        return;
    }
    let base = row as usize * shape.width as usize;
    let mut sum = 0.0;
    let mut column = tid;
    while column < shape.width {
        let value = half(input[base + column as usize]);
        sum += value * value;
        column += FBT_THREADS;
    }
    let reduction = unsafe { SharedArray::as_raw_mut_ptr(&raw mut REDUCTION) };
    unsafe { reduction.add(tid as usize).write(sum) };
    let total = unsafe { reduce_sum(reduction) };
    let scale = (total / shape.width as f32 + shape.epsilon())
        .sqrt()
        .recip();
    column = tid;
    while column < shape.width {
        let value = half(input[base + column as usize]) * scale * half(weight[column as usize]);
        unsafe { *output.get_unchecked_mut(base + column as usize) = half_bits(value) };
        column += FBT_THREADS;
    }
}
