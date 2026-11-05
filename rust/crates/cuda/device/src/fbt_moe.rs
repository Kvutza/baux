use cuda_device::{DisjointSlice, SharedArray, thread};

use super::fbt_gemm::turing;
use super::fbt_math::{reduce_sum, silu};
use super::fbt_types::{MatmulShape, MoeShape, RoutedTile, TILE};

#[inline(always)]
pub(crate) fn run_counts(
    experts: &[u32],
    counts: &mut DisjointSlice<u32>,
    prefixes: &mut DisjointSlice<u32>,
    assignments: u32,
) {
    static mut SUMS: SharedArray<f32, 256> = SharedArray::UNINIT;
    let expert = thread::blockIdx_x();
    let tid = thread::threadIdx_x();
    let sums = unsafe { SharedArray::as_raw_mut_ptr(&raw mut SUMS) };
    let chunks = assignments.div_ceil(1024);
    let mut total = 0;
    for chunk in 0..chunks {
        if tid == 0 {
            unsafe {
                *prefixes.get_unchecked_mut((expert * chunks + chunk) as usize) = total;
            }
        }
        let mut count = 0.0;
        for item in 0..4 {
            let index = chunk * 1024 + tid + item * 256;
            if index < assignments && experts[index as usize] == expert {
                count += 1.0;
            }
        }
        unsafe { sums.add(tid as usize).write(count) };
        total += unsafe { reduce_sum(sums) } as u32;
        thread::sync_threads();
    }
    if tid == 0 {
        unsafe { *counts.get_unchecked_mut(expert as usize) = total };
    }
}

#[inline(always)]
pub(crate) fn run_pack(
    input: &[f32],
    experts: &[u32],
    offsets: &[u32],
    prefixes: &[u32],
    mapping: &mut DisjointSlice<u32>,
    packed: &mut DisjointSlice<f32>,
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
pub(crate) fn routed<const THREADS: u32>(
    input: &[f32],
    weights: &[f32],
    routes: &[RoutedTile],
    output: *mut f32,
    mode: u32,
    tile: u32,
    first: u32,
) {
    let route = routes[tile as usize];
    if first >= route.valid_rows {
        return;
    }
    let (columns, inner, input_stride) = if mode == 0 {
        (512, 512, 512)
    } else {
        (512, 256, 256)
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
unsafe fn gate_load(
    input: &[f32],
    weights: &[f32],
    row: u32,
    local_row: u32,
    column: u32,
    expert: u64,
    start: u32,
    shape: MoeShape,
    route: RoutedTile,
) -> (f32, f32, f32) {
    let inner_x = start + thread::threadIdx_x();
    let inner_y = start + thread::threadIdx_y();
    let a = if local_row < route.valid_rows && inner_x < shape.width {
        unsafe {
            *input.get_unchecked(
                (u64::from(row) * u64::from(shape.width) + u64::from(inner_x)) as usize,
            )
        }
    } else {
        0.0
    };
    let (g, u) = if inner_y < shape.width && column < shape.expert_width {
        let base =
            expert + u64::from(inner_y) * u64::from(2 * shape.expert_width) + u64::from(column);
        unsafe {
            (
                *weights.get_unchecked(base as usize),
                *weights.get_unchecked((base + u64::from(shape.expert_width)) as usize),
            )
        }
    } else {
        (0.0, 0.0)
    };
    (a, g, u)
}

#[inline(always)]
unsafe fn gate_step(
    input_tile: *mut f32,
    gate_tile: *mut f32,
    up_tile: *mut f32,
    x: u32,
    y: u32,
    start: u32,
    shape: MoeShape,
) -> (f32, f32) {
    let mut gate_val = 0.0;
    let mut up_val = 0.0;
    let mut k = 0;
    while k < TILE && start + k < shape.width {
        let a = unsafe { input_tile.add((y * TILE + k) as usize).read() };
        gate_val += a * unsafe { gate_tile.add((k * TILE + x) as usize).read() };
        up_val += a * unsafe { up_tile.add((k * TILE + x) as usize).read() };
        k += 1;
    }
    (gate_val, up_val)
}

#[inline(always)]
unsafe fn gate_write(
    mut activation: DisjointSlice<f32, thread::Runtime2DIndex>,
    row: u32,
    column: u32,
    gate_value: f32,
    up_value: f32,
    shape: MoeShape,
) {
    let value = silu(gate_value) * up_value;
    let destination = u64::from(row) * u64::from(shape.activation_stride) + u64::from(column);
    unsafe { *activation.get_unchecked_mut(destination as usize) = value };
}

#[inline(always)]
pub(crate) fn run_gate(
    input: &[f32],
    weights: &[f32],
    routes: &[RoutedTile],
    activation: DisjointSlice<f32, thread::Runtime2DIndex>,
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
    activation: &[f32],
    weights: &[f32],
    routes: &[RoutedTile],
    output: &mut DisjointSlice<f32, thread::Runtime2DIndex>,
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
            activation[(u64::from(row) * u64::from(shape.activation_stride) + u64::from(inner_x))
                as usize]
        } else {
            0.0
        };
        let b = if inner_y < shape.expert_width && column < shape.width {
            let offset = expert + u64::from(inner_y) * u64::from(shape.width) + u64::from(column);
            weights[offset as usize]
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
        unsafe { *output.get_unchecked_mut(destination as usize) = value };
    }
}
