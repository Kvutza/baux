//! Turing tensor-core GEMM dispatch through the compiled CUDA-Oxide device
//! module.
use crate::capture::{CudaResult, cuda_error};
use baux_cuda_device::{MatmulShape, fbt_model};
use cuda_core::{CudaStream, DeviceBuffer, simt::LaunchConfig2D};

pub(crate) fn matmul(
    module: &fbt_model::LoadedModule,
    stream: &CudaStream,
    input: &DeviceBuffer<f32>,
    weights: &DeviceBuffer<f32>,
    output: &mut DeviceBuffer<f32>,
    rows: usize,
    columns: usize,
    inner: usize,
) -> CudaResult<()> {
    matmul_at(
        module, stream, input, weights, output, rows, columns, inner, 0,
    )
}

pub(crate) fn matmul_at(
    module: &fbt_model::LoadedModule,
    stream: &CudaStream,
    input: &DeviceBuffer<f32>,
    weights: &DeviceBuffer<f32>,
    output: &mut DeviceBuffer<f32>,
    rows: usize,
    columns: usize,
    inner: usize,
    first: usize,
) -> CudaResult<()> {
    if first
        .checked_add(rows)
        .and_then(|v| v.checked_mul(inner))
        .is_none_or(|v| v > input.len())
    {
        return Err("matmul input range exceeds allocation".into());
    }
    let shape = MatmulShape {
        rows: rows as u32,
        columns: columns as u32,
        inner: inner as u32,
        input_stride: inner as u32,
        weight_stride: columns as u32,
        output_stride: columns as u32,
        input_offset: (first * inner) as u64,
        weight_offset: 0,
        output_offset: 0,
    };
    let launch = module
        .prepare_matmul_turing(LaunchConfig2D::new(
            (columns.div_ceil(16) as u32, rows.div_ceil(64) as u32),
            (256, 1),
            0,
        ))
        .map_err(cuda_error)?;
    module
        .matmul_turing(
            stream,
            &launch,
            input,
            weights,
            cuda_host::RowWidth::new(output, columns as u32),
            shape,
        )
        .map_err(cuda_error)
}
