use cuda_device::{DisjointSlice, SharedArray, thread};

use super::fbt_types::{MatmulShape, TILE};

#[inline(always)]
pub(crate) fn run_matmul(
    input: &[f32],
    weight: &[f32],
    output: &mut DisjointSlice<f32, thread::Runtime2DIndex>,
    shape: MatmulShape,
) {
    static mut LEFT: SharedArray<f32, 256> = SharedArray::UNINIT;
    static mut RIGHT: SharedArray<f32, 256> = SharedArray::UNINIT;
    let x = thread::threadIdx_x();
    let y = thread::threadIdx_y();
    let row = thread::blockIdx_y() * TILE + y;
    let column = thread::blockIdx_x() * TILE + x;
    let index = y * TILE + x;
    let left = unsafe { SharedArray::as_raw_mut_ptr(&raw mut LEFT) };
    let right = unsafe { SharedArray::as_raw_mut_ptr(&raw mut RIGHT) };
    let mut value = 0.0;
    let mut start = 0;
    while start < shape.inner {
        let inner_x = start + x;
        let inner_y = start + y;
        let a = if row < shape.rows && inner_x < shape.inner {
            let offset = shape.input_offset
                + u64::from(row) * u64::from(shape.input_stride)
                + u64::from(inner_x);
            input[offset as usize]
        } else {
            0.0
        };
        let b = if inner_y < shape.inner && column < shape.columns {
            let offset = shape.weight_offset
                + u64::from(inner_y) * u64::from(shape.weight_stride)
                + u64::from(column);
            weight[offset as usize]
        } else {
            0.0
        };
        unsafe {
            left.add(index as usize).write(a);
            right.add(index as usize).write(b);
        }
        thread::sync_threads();
        let mut k = 0;
        while k < TILE && start + k < shape.inner {
            unsafe {
                value += left.add((y * TILE + k) as usize).read()
                    * right.add((k * TILE + x) as usize).read();
            }
            k += 1;
        }
        thread::sync_threads();
        start += TILE;
    }
    if row < shape.rows && column < shape.columns {
        let offset = shape.output_offset
            + u64::from(row) * u64::from(shape.output_stride)
            + u64::from(column);
        unsafe { *output.get_unchecked_mut(offset as usize) = value };
    }
}

#[inline(always)]
unsafe fn turing_load<const THREADS: u32>(
    input: &[f32],
    weight: &[f32],
    left: *mut f32,
    right: *mut f32,
    shape: MatmulShape,
    block_row: u32,
    block_column: u32,
    start: u32,
) {
    let tid = thread::threadIdx_x();
    let mut index = tid;
    while index < THREADS * 2 {
        let row = block_row + (index >> 3);
        let inner = start + (index & 7);
        let val = if row < shape.rows && inner < shape.inner {
            let offset = shape.input_offset
                + u64::from(row) * u64::from(shape.input_stride)
                + u64::from(inner);
            input[offset as usize]
        } else {
            0.0
        };
        unsafe { left.add(index as usize).write(val) };
        index += THREADS;
    }
    index = tid;
    while index < 128 {
        let inner = start + (index >> 4);
        let column = block_column + (index & 15);
        let val = if inner < shape.inner && column < shape.columns {
            let offset = shape.weight_offset
                + u64::from(inner) * u64::from(shape.weight_stride)
                + u64::from(column);
            weight[offset as usize]
        } else {
            0.0
        };
        unsafe { right.add(index as usize).write(val) };
        index += THREADS;
    }
}

#[inline(always)]
pub(crate) unsafe fn turing_mma(left: *const f32, right: *const f32, tid: u32, acc: &mut [f32; 4]) {
    let warp = tid >> 5;
    let lane = tid & 31;
    let group = lane >> 2;
    let local_row0 = (warp >> 1) * 16 + group;
    let local_row1 = local_row0 + 8;
    let local_col0 = (warp & 1) * 8 + ((lane & 3) << 1);
    let local_col1 = local_col0 + 1;
    let mut k = 0;
    while k < 8 {
        let a0 = unsafe { left.add((local_row0 * 8 + k) as usize).read() };
        let a1 = unsafe { left.add((local_row1 * 8 + k) as usize).read() };
        let b0 = unsafe { right.add((k * 16 + local_col0) as usize).read() };
        let b1 = unsafe { right.add((k * 16 + local_col1) as usize).read() };
        acc[0] = a0.mul_add(b0, acc[0]);
        acc[1] = a0.mul_add(b1, acc[1]);
        acc[2] = a1.mul_add(b0, acc[2]);
        acc[3] = a1.mul_add(b1, acc[3]);
        k += 1;
    }
}

#[inline(always)]
pub(crate) fn turing<const THREADS: u32>(
    input: &[f32],
    weight: &[f32],
    output: *mut f32,
    shape: MatmulShape,
    block_row: u32,
) {
    static mut LEFT_TILE: SharedArray<f32, 512> = SharedArray::UNINIT;
    static mut RIGHT_TILE: SharedArray<f32, 128> = SharedArray::UNINIT;
    let tid = thread::threadIdx_x();
    let warp = tid >> 5;
    let lane = tid & 31;
    let block_column = thread::blockIdx_x() * 16;
    let row0 = block_row + (warp >> 1) * 16 + (lane >> 2);
    let column0 = block_column + (warp & 1) * 8 + ((lane & 3) << 1);
    let mut acc = [0.0_f32; 4];
    let left = unsafe { SharedArray::as_raw_mut_ptr(&raw mut LEFT_TILE) };
    let right = unsafe { SharedArray::as_raw_mut_ptr(&raw mut RIGHT_TILE) };
    let mut start = 0;
    while start < shape.inner {
        unsafe {
            turing_load::<THREADS>(
                input,
                weight,
                left,
                right,
                shape,
                block_row,
                block_column,
                start,
            );
            thread::sync_threads();
            turing_mma(left, right, tid, &mut acc);
            thread::sync_threads();
        }
        start += 8;
    }
    for (idx, &val) in acc.iter().enumerate() {
        let r = row0 + (idx as u32 / 2) * 8;
        let c = column0 + (idx as u32 % 2);
        if r < shape.rows && c < shape.columns {
            let offset =
                shape.output_offset + u64::from(r) * u64::from(shape.output_stride) + u64::from(c);
            unsafe { output.add(offset as usize).write(val) };
        }
    }
}
