use cuda_device::{DisjointSlice, SharedArray, ptx_asm, thread};

use super::fbt_route::{gate_load, gate_step, gate_write};
use super::fbt_types::{MatmulShape, MoeShape, RoutedTile, TILE, VOCAB, half, half_bits};

#[inline(always)]
pub(crate) fn run_matmul(
    input: &[u16],
    weight: &[u16],
    output: &mut DisjointSlice<u16, thread::Runtime2DIndex>,
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
            half(input[offset as usize])
        } else {
            0.0
        };
        let b = if inner_y < shape.inner && column < shape.columns {
            let offset = shape.weight_offset
                + u64::from(inner_y) * u64::from(shape.weight_stride)
                + u64::from(column);
            half(weight[offset as usize])
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
        unsafe { *output.get_unchecked_mut(offset as usize) = half_bits(value) };
    }
}

#[inline(always)]
unsafe fn turing_load<const THREADS: u32>(
    input: &[u16],
    weight: &[u16],
    left: *mut u16,
    right: *mut u16,
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
            0
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
            0
        };
        unsafe { right.add(index as usize).write(val) };
        index += THREADS;
    }
}

#[inline(always)]
pub(crate) unsafe fn turing_mma(left: *const u16, right: *const u16, tid: u32, acc: &mut [f32; 4]) {
    let warp = tid >> 5;
    let lane = tid & 31;
    let group = lane >> 2;
    let pair = (lane & 3) << 1;
    let local_row0 = (warp >> 1) * 16 + group;
    let local_row1 = local_row0 + 8;
    let local_column = (warp & 1) * 8 + group;
    let a0_offset = local_row0 * 8 + pair;
    let a1_offset = local_row1 * 8 + pair;
    let b_offset = pair * 16 + local_column;
    let a0 = unsafe {
        u32::from(left.add(a0_offset as usize).read())
            | (u32::from(left.add((a0_offset + 1) as usize).read()) << 16)
    };
    let a1 = unsafe {
        u32::from(left.add(a1_offset as usize).read())
            | (u32::from(left.add((a1_offset + 1) as usize).read()) << 16)
    };
    let b = unsafe {
        u32::from(right.add(b_offset as usize).read())
            | (u32::from(right.add((b_offset + 16) as usize).read()) << 16)
    };
    unsafe {
        ptx_asm!(
            "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5}, {%6}, {%0, %1, %2, %3};",
            inout("+f") acc[0],
            inout("+f") acc[1],
            inout("+f") acc[2],
            inout("+f") acc[3],
            in("r") a0,
            in("r") a1,
            in("r") b,
            options(register_only),
        );
    }
}

#[inline(always)]
pub(crate) fn turing<const THREADS: u32>(
    input: &[u16],
    weight: &[u16],
    output: *mut u16,
    shape: MatmulShape,
    block_row: u32,
) {
    static mut LEFT_HALF: SharedArray<u16, 512> = SharedArray::UNINIT;
    static mut RIGHT_HALF: SharedArray<u16, 128> = SharedArray::UNINIT;
    let tid = thread::threadIdx_x();
    let warp = tid >> 5;
    let lane = tid & 31;
    let block_column = thread::blockIdx_x() * 16;
    let row0 = block_row + (warp >> 1) * 16 + (lane >> 2);
    let column0 = block_column + (warp & 1) * 8 + ((lane & 3) << 1);
    let mut acc = [0.0_f32; 4];
    let left = unsafe { SharedArray::as_raw_mut_ptr(&raw mut LEFT_HALF) };
    let right = unsafe { SharedArray::as_raw_mut_ptr(&raw mut RIGHT_HALF) };
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
            unsafe { output.add(offset as usize).write(half_bits(val)) };
        }
    }
}

#[inline(always)]
pub(crate) fn turing_f32<const THREADS: u32>(
    input: &[u16],
    weight: &[u16],
    output: *mut f32,
    rows: u32,
    block_row: u32,
) {
    static mut LEFT_HALF_F32: SharedArray<u16, 512> = SharedArray::UNINIT;
    static mut RIGHT_HALF_F32: SharedArray<u16, 128> = SharedArray::UNINIT;
    let tid = thread::threadIdx_x();
    let warp = tid >> 5;
    let lane = tid & 31;
    let block_column = thread::blockIdx_x() * 16;
    let row0 = block_row + (warp >> 1) * 16 + (lane >> 2);
    let column0 = block_column + (warp & 1) * 8 + ((lane & 3) << 1);
    let mut acc = [0.0_f32; 4];
    let left = unsafe { SharedArray::as_raw_mut_ptr(&raw mut LEFT_HALF_F32) };
    let right = unsafe { SharedArray::as_raw_mut_ptr(&raw mut RIGHT_HALF_F32) };
    let mut start = 0_u32;
    while start < 64 {
        let mut index = tid;
        while index < THREADS * 2 {
            let row = block_row + (index >> 3);
            let inner = start + (index & 7);
            unsafe {
                left.add(index as usize).write(if row < rows {
                    input[(row * 64 + inner) as usize]
                } else {
                    0
                });
            }
            index += THREADS;
        }
        index = tid;
        while index < 128 {
            unsafe {
                right.add(index as usize).write(
                    weight[(start + (index >> 4)) as usize * VOCAB as usize
                        + (block_column + (index & 15)) as usize],
                );
            }
            index += THREADS;
        }
        thread::sync_threads();
        unsafe { turing_mma(left, right, tid, &mut acc) };
        thread::sync_threads();
        start += 8;
    }
    for (idx, &val) in acc.iter().enumerate() {
        let r = row0 + (idx as u32 / 2) * 8;
        let c = column0 + (idx as u32 % 2);
        if r < rows {
            unsafe {
                output
                    .add(r as usize * VOCAB as usize + c as usize)
                    .write(val);
            }
        }
    }
}

#[inline(always)]
pub(crate) fn routed<const THREADS: u32>(
    input: &[u16],
    weights: &[u16],
    routes: &[RoutedTile],
    output: *mut u16,
    mode: u32,
    tile: u32,
    first: u32,
) {
    let route = routes[tile as usize];
    if first >= route.valid_rows {
        return;
    }
    let (columns, inner, input_stride) = if mode == 0 {
        (432, 512, 512)
    } else {
        (512, 216, 224)
    };
    let expert = if route.reserved == 1 {
        0
    } else {
        route.expert + 1
    };
    let shape = MatmulShape {
        rows: route.valid_rows,
        columns,
        inner,
        input_stride,
        weight_stride: columns,
        output_stride: columns,
        input_offset: u64::from(route.first_row) * u64::from(input_stride),
        weight_offset: u64::from(expert) * u64::from(inner) * u64::from(columns),
        output_offset: u64::from(route.first_row) * u64::from(columns),
    };
    turing::<THREADS>(input, weights, output, shape, first);
}

#[inline(always)]
pub(crate) fn run_gate(
    input: &[u16],
    weights: &[u16],
    routes: &[RoutedTile],
    activation: DisjointSlice<u16, thread::Runtime2DIndex>,
    shape: MoeShape,
) {
    static mut INPUT: SharedArray<f32, 256> = SharedArray::UNINIT;
    static mut GATE: SharedArray<f32, 256> = SharedArray::UNINIT;
    static mut UP: SharedArray<f32, 256> = SharedArray::UNINIT;
    let x = thread::threadIdx_x();
    let y = thread::threadIdx_y();
    let flat = y * TILE + x;
    let route_index = thread::blockIdx_y() / shape.row_tiles;
    let row_tile = thread::blockIdx_y() % shape.row_tiles;
    let route = routes[route_index as usize];
    if route.valid_rows == 0 {
        return;
    }
    let local_row = row_tile * TILE + y;
    let row = route.first_row + local_row;
    let column = thread::blockIdx_x() * TILE + x;
    let input_tile = unsafe { SharedArray::as_raw_mut_ptr(&raw mut INPUT) };
    let gate_tile = unsafe { SharedArray::as_raw_mut_ptr(&raw mut GATE) };
    let up_tile = unsafe { SharedArray::as_raw_mut_ptr(&raw mut UP) };
    let expert_stride = u64::from(shape.width) * u64::from(2 * shape.expert_width);
    let expert = u64::from(if route.reserved == 1 {
        0
    } else {
        route.expert + 1
    }) * expert_stride;
    let mut gate_value = 0.0;
    let mut up_value = 0.0;
    let mut start = 0;
    while start < shape.width {
        let (a, g, u) = unsafe {
            gate_load(
                input, weights, row, local_row, column, expert, start, shape, route,
            )
        };
        unsafe {
            input_tile.add(flat as usize).write(a);
            gate_tile.add(flat as usize).write(g);
            up_tile.add(flat as usize).write(u);
        }
        thread::sync_threads();
        let (dg, du) = unsafe { gate_step(input_tile, gate_tile, up_tile, x, y, start, shape) };
        gate_value += dg;
        up_value += du;
        thread::sync_threads();
        start += TILE;
    }
    if local_row < route.valid_rows && column < shape.expert_width {
        unsafe { gate_write(activation, row, column, gate_value, up_value, shape) };
    }
}

#[inline(always)]
pub(crate) fn run_down(
    activation: &[u16],
    weights: &[u16],
    routes: &[RoutedTile],
    output: &mut DisjointSlice<u16, thread::Runtime2DIndex>,
    shape: MoeShape,
) {
    static mut INPUT: SharedArray<f32, 256> = SharedArray::UNINIT;
    static mut WEIGHT: SharedArray<f32, 256> = SharedArray::UNINIT;
    let x = thread::threadIdx_x();
    let y = thread::threadIdx_y();
    let flat = y * TILE + x;
    let route_index = thread::blockIdx_y() / shape.row_tiles;
    let row_tile = thread::blockIdx_y() % shape.row_tiles;
    let route = routes[route_index as usize];
    if route.valid_rows == 0 {
        return;
    }
    let local_row = row_tile * TILE + y;
    let row = route.first_row + local_row;
    let column = thread::blockIdx_x() * TILE + x;
    let input_tile = unsafe { SharedArray::as_raw_mut_ptr(&raw mut INPUT) };
    let weight_tile = unsafe { SharedArray::as_raw_mut_ptr(&raw mut WEIGHT) };
    let expert_stride = u64::from(shape.expert_width) * u64::from(shape.width);
    let expert = u64::from(if route.reserved == 1 {
        0
    } else {
        route.expert + 1
    }) * expert_stride;
    let mut value = 0.0;
    let mut start = 0;
    while start < shape.expert_width {
        let inner_x = start + x;
        let inner_y = start + y;
        let a = if local_row < route.valid_rows && inner_x < shape.expert_width {
            half(
                activation[(u64::from(row) * u64::from(shape.activation_stride)
                    + u64::from(inner_x)) as usize],
            )
        } else {
            0.0
        };
        let b = if inner_y < shape.expert_width && column < shape.width {
            let offset = expert + u64::from(inner_y) * u64::from(shape.width) + u64::from(column);
            half(weights[offset as usize])
        } else {
            0.0
        };
        unsafe {
            input_tile.add(flat as usize).write(a);
            weight_tile.add(flat as usize).write(b);
        }
        thread::sync_threads();
        let mut k = 0;
        while k < TILE && start + k < shape.expert_width {
            unsafe {
                value += input_tile.add((y * TILE + k) as usize).read()
                    * weight_tile.add((k * TILE + x) as usize).read();
            }
            k += 1;
        }
        thread::sync_threads();
        start += TILE;
    }
    if local_row < route.valid_rows && column < shape.width {
        let destination = u64::from(row) * u64::from(shape.width) + u64::from(column);
        unsafe { *output.get_unchecked_mut(destination as usize) = half_bits(value) };
    }
}
