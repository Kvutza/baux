use cuda_device::{DisjointSlice, SharedArray, thread, warp};

use super::fbt_math::{reduce_sum, sigmoid};
use super::fbt_types::{FBT_THREADS, MoeShape, RouteShape, half, half_bits};

#[inline(always)]
pub(crate) fn run_rope(
    qkv: &mut DisjointSlice<u16>,
    table: &[f32],
    rows: u32,
    context: u32,
    first: u32,
) {
    let index = thread::blockIdx_x() * 256 + thread::threadIdx_x();
    if index >= rows * 9 * 32 {
        return;
    }
    let vec_idx = index >> 5;
    let pair = index & 31;
    let row = vec_idx / 9;
    let head = vec_idx - row * 9;
    let pos = if context.is_power_of_two() {
        (first + row) & (context - 1)
    } else {
        (first + row) % context
    };
    let phase = ((pos << 5) + pair) as usize * 2;
    let offset = (row * 640 + (head << 6) + pair) as usize;
    let x = half(unsafe { *qkv.get_unchecked_mut(offset) });
    let y = half(unsafe { *qkv.get_unchecked_mut(offset + 32) });
    let cos_val = unsafe { *table.get_unchecked(phase) };
    let sin_val = unsafe { *table.get_unchecked(phase + 1) };
    unsafe {
        *qkv.get_unchecked_mut(offset) = half_bits((-y).mul_add(sin_val, x * cos_val));
        *qkv.get_unchecked_mut(offset + 32) = half_bits(x.mul_add(sin_val, y * cos_val));
    }
}

#[inline(always)]
pub(crate) fn run_replicate(input: &[u16], streams: &mut DisjointSlice<u16>, rows: u32) {
    let index = thread::blockIdx_x() * 256 + thread::threadIdx_x();
    if index >= rows * 512 {
        return;
    }
    let base = (index / 512) * 2048 + index % 512;
    for stream in 0..4 {
        unsafe {
            *streams.get_unchecked_mut((base + stream * 512) as usize) = input[index as usize];
        }
    }
}

#[inline(always)]
fn predict_cat(output: u32) -> usize {
    if output < 4 {
        0
    } else if output < 20 {
        1
    } else {
        2
    }
}

#[inline(always)]
unsafe fn predict_dots(
    streams: &[u16],
    predictor: &[u16],
    bias: &[u16],
    control: &[u16],
    raw: *mut f32,
    scale: *mut f32,
    row: u32,
) {
    let tid = thread::threadIdx_x();
    let lane = tid % 32;
    let group = tid / 32;
    let active = control[0] != 0 || control[1] != 0 || control[2] != 0;
    let mut squared = 0.0_f32;
    let mut dots = [0.0_f32; 3];
    if active {
        let out0 = group * 3;
        let a0 = control[predict_cat(out0)] != 0;
        let a1 = control[predict_cat(out0 + 1)] != 0;
        let a2 = control[predict_cat(out0 + 2)] != 0;
        let mut input = lane;
        while input < 2048 {
            let value = half(streams[(row * 2048 + input) as usize]);
            if group == 0 {
                squared = value.mul_add(value, squared);
            }
            let pred_base = input * 24 + out0;
            if a0 {
                dots[0] = value.mul_add(half(predictor[pred_base as usize]), dots[0]);
            }
            if a1 {
                dots[1] = value.mul_add(half(predictor[(pred_base + 1) as usize]), dots[1]);
            }
            if a2 {
                dots[2] = value.mul_add(half(predictor[(pred_base + 2) as usize]), dots[2]);
            }
            input += 32;
        }
    }
    let squared = warp::reduce_sum_f32(squared);
    if tid == 0 {
        unsafe {
            scale.write(if active {
                (squared / 2048.0 + 1.0e-5).sqrt().recip()
            } else {
                1.0
            });
        }
    }
    thread::sync_threads();
    for (item, &dot) in dots.iter().enumerate() {
        let sum = warp::reduce_sum_f32(dot);
        let output = group * 3 + item as u32;
        if lane == 0 {
            unsafe {
                raw.add(output as usize).write(
                    half(bias[output as usize])
                        + half(control[predict_cat(output)]) * sum * scale.read(),
                );
            }
        }
    }
}

#[inline(always)]
unsafe fn predict_parallel(
    control: &[u16],
    coefficients: &mut DisjointSlice<f32>,
    raw: *const f32,
    row: u32,
) {
    let tid = thread::threadIdx_x();
    if tid >= 32 {
        return;
    }
    let base = row as usize * 24;
    if tid == 0 {
        let mut total = 0.0;
        for index in 0..4 {
            let v = sigmoid(unsafe { raw.add(index).read() });
            unsafe { *coefficients.get_unchecked_mut(base + index) = v };
            total += v;
        }
        for index in 0..4 {
            unsafe {
                *coefficients.get_unchecked_mut(base + index) /= total.max(1.0e-20);
                *coefficients.get_unchecked_mut(base + 20 + index) =
                    2.0 * sigmoid(raw.add(20 + index).read());
            }
        }
    }
    let entry = (tid % 16) as usize;
    let r = entry / 4;
    let c = entry % 4;
    let mut maximum = f32::NEG_INFINITY;
    for index in 0..16 {
        maximum = maximum.max(unsafe { raw.add(4 + index).read() });
    }
    let mut value = (unsafe { raw.add(4 + entry).read() } - maximum)
        .max(-80.0)
        .exp();
    for _ in 0..20 {
        let mut sum = 0.0;
        for column in 0..4 {
            sum += warp::shuffle_f32_sync(u32::MAX, value, (r * 4 + column) as u32);
        }
        value /= sum.max(1.0e-20);
        let mut sum = 0.0;
        for r_idx in 0..4 {
            sum += warp::shuffle_f32_sync(u32::MAX, value, (r_idx * 4 + c) as u32);
        }
        value /= sum.max(1.0e-20);
    }
    if tid < 16 {
        let lambda = half(control[3]).clamp(0.0, 1.0);
        unsafe {
            *coefficients.get_unchecked_mut(base + 4 + entry) =
                (1.0 - lambda) * if r == c { 1.0 } else { 0.0 } + lambda * value;
        }
    }
}

#[inline(always)]
unsafe fn serial_sinkhorn(transport: *mut f32) {
    let mut maximum = f32::NEG_INFINITY;
    for index in 0..16 {
        maximum = maximum.max(unsafe { transport.add(index).read() });
    }
    for index in 0..16 {
        let value = (unsafe { transport.add(index).read() } - maximum)
            .max(-80.0)
            .exp();
        unsafe { transport.add(index).write(value) };
    }
    for _ in 0..20 {
        for r in 0..4 {
            let mut sum = 0.0;
            for c in 0..4 {
                sum += unsafe { transport.add(r * 4 + c).read() };
            }
            let denom = sum.max(1.0e-20);
            for c in 0..4 {
                unsafe {
                    let ptr = transport.add(r * 4 + c);
                    ptr.write(ptr.read() / denom);
                }
            }
        }
        for c in 0..4 {
            let mut sum = 0.0;
            for r in 0..4 {
                sum += unsafe { transport.add(r * 4 + c).read() };
            }
            let denom = sum.max(1.0e-20);
            for r in 0..4 {
                unsafe {
                    let ptr = transport.add(r * 4 + c);
                    ptr.write(ptr.read() / denom);
                }
            }
        }
    }
}

#[inline(always)]
unsafe fn predict_serial(
    control: &[u16],
    coefficients: &mut DisjointSlice<f32>,
    raw: *mut f32,
    row: u32,
) {
    if thread::threadIdx_x() != 0 {
        return;
    }
    let base = row as usize * 24;
    let mut total = 0.0;
    for index in 0..4 {
        let v = sigmoid(unsafe { raw.add(index).read() });
        unsafe { *coefficients.get_unchecked_mut(base + index) = v };
        total += v;
    }
    for index in 0..4 {
        unsafe { *coefficients.get_unchecked_mut(base + index) /= total.max(1.0e-20) };
    }
    let transport = unsafe { raw.add(4) };
    unsafe { serial_sinkhorn(transport) };
    let lambda = half(control[3]).clamp(0.0, 1.0);
    for r in 0..4 {
        for c in 0..4 {
            unsafe {
                *coefficients.get_unchecked_mut(base + 4 + r * 4 + c) = (1.0 - lambda)
                    * if r == c { 1.0 } else { 0.0 }
                    + lambda * transport.add(r * 4 + c).read();
            }
        }
    }
    for index in 0..4 {
        unsafe {
            *coefficients.get_unchecked_mut(base + 20 + index) =
                2.0 * sigmoid(raw.add(20 + index).read());
        }
    }
}

#[inline(always)]
pub(crate) fn run_predict(
    streams: &[u16],
    predictor: &[u16],
    bias: &[u16],
    control: &[u16],
    coefficients: &mut DisjointSlice<f32>,
    rows: u32,
    parallel: u32,
) {
    static mut RAW: SharedArray<f32, 24> = SharedArray::UNINIT;
    static mut SCALE: SharedArray<f32, 1> = SharedArray::UNINIT;
    let row = thread::blockIdx_x();
    if row >= rows {
        return;
    }
    let raw = unsafe { SharedArray::as_raw_mut_ptr(&raw mut RAW) };
    let scale = unsafe { SharedArray::as_raw_mut_ptr(&raw mut SCALE) };
    unsafe {
        predict_dots(streams, predictor, bias, control, raw, scale, row);
        thread::sync_threads();
        if parallel != 0 {
            predict_parallel(control, coefficients, raw, row);
        } else {
            predict_serial(control, coefficients, raw, row);
        }
    }
}

#[inline(always)]
pub(crate) fn run_mix(
    streams: &[u16],
    coefficients: &[f32],
    weight: &[u16],
    output: &mut DisjointSlice<u16>,
    rows: u32,
    mean: u32,
) {
    static mut SUMS: SharedArray<f32, 256> = SharedArray::UNINIT;
    let row = thread::blockIdx_x();
    if row >= rows {
        return;
    }
    let tid = thread::threadIdx_x();
    let coefs = if mean == 1 {
        [0.25_f32; 4]
    } else {
        let c_base = (row * 24) as usize;
        unsafe {
            [
                *coefficients.get_unchecked(c_base),
                *coefficients.get_unchecked(c_base + 1),
                *coefficients.get_unchecked(c_base + 2),
                *coefficients.get_unchecked(c_base + 3),
            ]
        }
    };
    let row_stream_base = row << 11;
    let mut values = [0.0_f32; 2];
    let mut squared = 0.0;
    for item in 0..2 {
        let column = tid + (item << 8);
        let s_base = (row_stream_base + column) as usize;
        let mut value = 0.0_f32;
        for stream in 0..4 {
            value = coefs[stream].mul_add(
                half(unsafe { *streams.get_unchecked(s_base + (stream << 9)) }),
                value,
            );
        }
        values[item as usize] = half(half_bits(value));
        squared = values[item as usize].mul_add(values[item as usize], squared);
    }
    let sums = unsafe { SharedArray::as_raw_mut_ptr(&raw mut SUMS) };
    unsafe { sums.add(tid as usize).write(squared) };
    let scale = (unsafe { reduce_sum(sums) } * (1.0 / 512.0) + 1.0e-5)
        .sqrt()
        .recip();
    let out_base = row << 9;
    for item in 0..2 {
        let column = tid + (item << 8);
        unsafe {
            *output.get_unchecked_mut((out_base + column) as usize) = half_bits(
                values[item as usize] * scale * half(*weight.get_unchecked(column as usize)),
            );
        }
    }
}

#[inline(always)]
pub(crate) fn run_update(
    streams: &mut DisjointSlice<u16>,
    branch: &[u16],
    coefficients: &[f32],
    rows: u32,
) {
    let index = thread::blockIdx_x() * 256 + thread::threadIdx_x();
    if index >= (rows << 9) {
        return;
    }
    let row = index >> 9;
    let column = index & 511;
    let base = ((row << 11) + column) as usize;
    let coef_base = (row * 24) as usize;
    let branch_val = half(unsafe { *branch.get_unchecked(index as usize) });
    let mut source = [0.0_f32; 4];
    for s in 0..4 {
        source[s] = half(unsafe { *streams.get_unchecked_mut(base + (s << 9)) });
    }
    for stream in 0..4 {
        let mut value =
            unsafe { *coefficients.get_unchecked(coef_base + 20 + stream) } * branch_val;
        for s in 0..4 {
            value = unsafe { *coefficients.get_unchecked(coef_base + 4 + (stream << 2) + s) }
                .mul_add(source[s], value);
        }
        unsafe {
            *streams.get_unchecked_mut(base + (stream << 9)) = half_bits(value);
        }
    }
}
