use cuda_device::{DisjointSlice, SharedArray, thread};

use super::fbt_math::{gaussian_block_key, gaussian_normal, reduce_sum, sigmoid};
use super::fbt_types::{
    FBT_THREADS, FeedbackShape, GaussianShape, Mhc4Point, MoeShape, RouteShape, block_flags, half,
    half_bits,
};

#[inline(always)]
pub(crate) fn run_basis0(
    base: &[u16],
    scales: &[f32],
    candidate: &mut DisjointSlice<u16>,
    elements: u64,
    block_len: u32,
    radius: f32,
) {
    let mut index = u64::from(thread::blockIdx_x()) * 256 + u64::from(thread::threadIdx_x());
    let stride = u64::from(thread::gridDim_x()) * 256;
    while index < elements {
        let block = index / u64::from(block_len);
        let value = half(base[index as usize]) + scales[block as usize] * radius;
        unsafe { *candidate.get_unchecked_mut(index as usize) = half_bits(value) };
        index += stride;
    }
}

#[inline(always)]
pub(crate) fn run_gaussian(
    base: &[u16],
    scales: &[f32],
    thresholds: &[u32],
    widths: &[f32],
    densities: &[f32],
    candidate: &mut DisjointSlice<u16>,
    status: &mut DisjointSlice<u32>,
    shape: GaussianShape,
) {
    let mut index = u64::from(thread::blockIdx_x()) * 256 + u64::from(thread::threadIdx_x());
    let stride = u64::from(thread::gridDim_x()) * 256;
    let mut flags = 0;
    while index < shape.elements {
        let logical = if shape.transpose_columns == 0 {
            index
        } else {
            let rows = shape.elements / u64::from(shape.transpose_columns);
            (index % u64::from(shape.transpose_columns)) * rows
                + index / u64::from(shape.transpose_columns)
        };
        let block = logical / u64::from(shape.block_len);
        let element = logical % u64::from(shape.block_len);
        let key = gaussian_block_key(
            shape.tensor_id,
            shape.logical_start + block * u64::from(shape.block_len),
            u64::from(shape.block_len),
        );
        let direction = gaussian_normal(
            shape.seed ^ 0x8ebc_6af0_9c88_c6e3,
            key,
            element,
            thresholds,
            widths,
            densities,
        );
        let delta = scales[block as usize] * shape.radius * direction;
        let cand_bits = half_bits(half(base[index as usize]) + delta);
        flags |= baux::semantics::proposal::fp16_flags(base[index as usize], cand_bits);
        unsafe { *candidate.get_unchecked_mut(index as usize) = cand_bits };
        index += stride;
    }
    let flags = block_flags(flags);
    if thread::threadIdx_x() == 0 {
        unsafe { *status.get_unchecked_mut(thread::blockIdx_x() as usize) = flags };
    }
}

#[inline(always)]
pub(crate) fn run_propstatus(
    status: &[u32],
    summary: &mut DisjointSlice<u32>,
    blocks: u32,
    tensor: u32,
) {
    let mut index = thread::threadIdx_x();
    let mut flags = 0;
    while index < blocks {
        flags |= status[index as usize];
        index += 256;
    }
    let flags = block_flags(flags);
    if thread::threadIdx_x() == 0 {
        unsafe { *summary.get_unchecked_mut(tensor as usize) = flags };
    }
}

#[inline(always)]
pub(crate) fn run_pool(
    base: &[u16],
    points: &[Mhc4Point],
    seeds: &[u64],
    thresholds: &[u32],
    widths: &[f32],
    densities: &[f32],
    pool: &mut DisjointSlice<u16>,
    point_start: u32,
    point_count: u32,
    total_points: u32,
    radius: f32,
    candidates: u32,
) {
    let local = thread::blockIdx_x() * 256 + thread::threadIdx_x();
    if local >= point_count {
        return;
    }
    let index = point_start + local;
    let point = points[index as usize];
    let value = half(base[point.physical as usize]);
    let mut candidate = 0_u32;
    while candidate < candidates {
        let noise = gaussian_normal(
            seeds[candidate as usize] ^ 0x8ebc_6af0_9c88_c6e3,
            point.key,
            point.element,
            thresholds,
            widths,
            densities,
        );
        unsafe {
            *pool.get_unchecked_mut((candidate * total_points + index) as usize) =
                half_bits(value + point.scale * radius * noise);
        }
        candidate += 1;
    }
}

#[inline(always)]
pub(crate) fn run_convcheck(
    input: &[f32],
    narrowed: &mut DisjointSlice<u16>,
    widened: &mut DisjointSlice<u32>,
    count: u32,
) {
    let i = thread::blockIdx_x() * 256 + thread::threadIdx_x();
    if i < count {
        unsafe {
            *narrowed.get_unchecked_mut(i as usize) = half_bits(input[i as usize]);
            *widened.get_unchecked_mut(i as usize) = half(i as u16).to_bits();
        }
    }
}

#[inline(always)]
pub(crate) fn run_fbshift(
    history: &[u16],
    previous: &mut DisjointSlice<u16>,
    fused: &mut DisjointSlice<u32>,
    shape: FeedbackShape,
) {
    let row = thread::blockIdx_x();
    let tid = thread::threadIdx_x();
    if row >= shape.rows {
        return;
    }
    let active = row % shape.context != 0;
    if tid == 0 {
        unsafe { *fused.get_unchecked_mut(row as usize) = active as u32 };
    }
    let mut column = tid;
    while column < shape.width {
        let destination = (row * shape.width + column) as usize;
        let value = if active {
            history[((row - 1) * shape.width + column) as usize]
        } else {
            half_bits(0.0)
        };
        unsafe { *previous.get_unchecked_mut(destination) = value };
        column += FBT_THREADS;
    }
}

#[inline(always)]
pub(crate) fn run_fbnorm(input: &[u16], output: &mut DisjointSlice<u16>, shape: FeedbackShape) {
    static mut REDUCTION: SharedArray<f32, 256> = SharedArray::UNINIT;
    let row = thread::blockIdx_x();
    let tid = thread::threadIdx_x();
    if row >= shape.rows {
        return;
    }
    let reduction = unsafe { SharedArray::as_raw_mut_ptr(&raw mut REDUCTION) };
    let base = row * shape.width;
    let mut sum = 0.0;
    let mut column = tid;
    while column < shape.width {
        let value = half(input[(base + column) as usize]);
        sum += value * value;
        column += FBT_THREADS;
    }
    unsafe { reduction.add(tid as usize).write(sum) };
    let scale = if shape.token_unit_rms != 0 {
        (unsafe { reduce_sum(reduction) } / shape.width as f32 + shape.token_epsilon())
            .sqrt()
            .recip()
    } else {
        thread::sync_threads();
        1.0
    };
    column = tid;
    while column < shape.width {
        let index = (base + column) as usize;
        unsafe { *output.get_unchecked_mut(index) = half_bits(half(input[index]) * scale) };
        column += FBT_THREADS;
    }
}

#[inline(always)]
pub(crate) fn run_fbcombine(
    value: &[u16],
    logit: &[u16],
    tokens: &[u16],
    fused: &[u32],
    output: &mut DisjointSlice<u16>,
    shape: FeedbackShape,
) {
    static mut REDUCTION: SharedArray<f32, 256> = SharedArray::UNINIT;
    let row = thread::blockIdx_x();
    let tid = thread::threadIdx_x();
    if row >= shape.rows {
        return;
    }
    let base = row * shape.width;
    if fused[row as usize] == 0 {
        let mut column = tid;
        while column < shape.width {
            let index = (base + column) as usize;
            unsafe { *output.get_unchecked_mut(index) = tokens[index] };
            column += FBT_THREADS;
        }
        return;
    }
    let reduction = unsafe { SharedArray::as_raw_mut_ptr(&raw mut REDUCTION) };
    let mut sum = 0.0;
    let mut column = tid;
    while column < shape.width {
        let index = (base + column) as usize;
        let combined = half(value[index]) * sigmoid(half(logit[index]));
        sum += combined * combined;
        column += FBT_THREADS;
    }
    unsafe { reduction.add(tid as usize).write(sum) };
    let scale = if shape.fused_unit_rms != 0 {
        (unsafe { reduce_sum(reduction) } / shape.width as f32 + shape.fused_epsilon())
            .sqrt()
            .recip()
    } else {
        thread::sync_threads();
        1.0
    };
    column = tid;
    while column < shape.width {
        let index = (base + column) as usize;
        let combined = half(value[index]) * sigmoid(half(logit[index]));
        unsafe { *output.get_unchecked_mut(index) = half_bits(combined * scale) };
        column += FBT_THREADS;
    }
}

#[inline(always)]
pub(crate) fn run_combroutes(
    route_weights: &[u16],
    packed_rows: &[u32],
    routed_output: &[u16],
    shared_output: &[u16],
    output: &mut DisjointSlice<u16>,
    route_shape: RouteShape,
    moe_shape: MoeShape,
) {
    let index = thread::blockIdx_x() * FBT_THREADS + thread::threadIdx_x();
    let elements = route_shape.rows * moe_shape.width;
    if index >= elements {
        return;
    }
    let (token, column) = if moe_shape.width == 512 {
        (index >> 9, index & 511)
    } else {
        (index / moe_shape.width, index % moe_shape.width)
    };
    let mut value = half(unsafe { *shared_output.get_unchecked(index as usize) });
    let mut route = 0;
    while route < route_shape.top_k {
        let assignment = (token * route_shape.top_k + route) as usize;
        let packed = unsafe { *packed_rows.get_unchecked(assignment) };
        let source = u64::from(packed) * u64::from(moe_shape.width) + u64::from(column);
        value += half(unsafe { *route_weights.get_unchecked(assignment) })
            * half(unsafe { *routed_output.get_unchecked(source as usize) });
        route += 1;
    }
    unsafe { *output.get_unchecked_mut(index as usize) = half_bits(value) };
}

#[inline(always)]
pub(crate) fn run_combinemhc(
    route_weights: &[u16],
    packed_rows: &[u32],
    routed_output: &[u16],
    shared_output: &[u16],
    coefficients: &[f32],
    streams: &mut DisjointSlice<u16>,
    route_shape: RouteShape,
    moe_shape: MoeShape,
) {
    let index = thread::blockIdx_x() * FBT_THREADS + thread::threadIdx_x();
    let elements = route_shape.rows * moe_shape.width;
    if index >= elements {
        return;
    }
    let (token, column) = if moe_shape.width == 512 {
        (index >> 9, index & 511)
    } else {
        (index / moe_shape.width, index % moe_shape.width)
    };
    let mut combined = half(unsafe { *shared_output.get_unchecked(index as usize) });
    let mut route = 0;
    while route < route_shape.top_k {
        let assignment = (token * route_shape.top_k + route) as usize;
        let packed = unsafe { *packed_rows.get_unchecked(assignment) };
        let source = u64::from(packed) * u64::from(moe_shape.width) + u64::from(column);
        combined += half(unsafe { *route_weights.get_unchecked(assignment) })
            * half(unsafe { *routed_output.get_unchecked(source as usize) });
        route += 1;
    }
    let branch_val = half(half_bits(combined));
    let base = ((token << 11) + column) as usize;
    let coef_base = (token * 24) as usize;
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
