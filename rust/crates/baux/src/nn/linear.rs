//! Linear projection layers on typed [`View`] and [`Out`] operands.
//!
//! Unifies:
//! - Single-step / expert-batched vector projections (`gemv`, `gemv_vector`)
//! - Chunked 2D SIMD-matrix / TensorOps tile projections
//!   (`output_projection_wide`)
//!
//! All dispatches go through [`Dispatch`] with automatic memory barriers.

use metal::{ComputeCommandEncoderRef, ComputePipelineState};

use crate::tensor::{Dispatch, Element, F16, Fence, Out, U32, View, Wave};

const VOCAB: u32 = baux::semantics::mhc4::VOCABULARY as u32;

/// Single-token or 4-expert GEMV projection operator (`Rows::Step`).
#[derive(Clone, Copy)]
pub(crate) struct Linear<'a> {
    gemv: &'a ComputePipelineState,
    gemv_vector: &'a ComputePipelineState,
    experts: View<'a, U32>,
}

impl<'a> Linear<'a> {
    #[inline(always)]
    pub(crate) const fn new(
        gemv: &'a ComputePipelineState,
        gemv_vector: &'a ComputePipelineState,
        experts: View<'a, U32>,
    ) -> Self {
        Self {
            gemv,
            gemv_vector,
            experts,
        }
    }

    /// Forward pass for single-token (`1 -> 1`) or 4-expert (`1 -> 4`, `4 ->
    /// 4`) vector multiplications: `output = input * weight`.
    pub(crate) fn forward_vector<E: Element>(
        self,
        encoder: &ComputeCommandEncoderRef,
        input: View<'_, E>,
        weight: View<'_, F16>,
        output: Out<'_, E>,
    ) {
        let k = input.cols();
        let out = output.view();
        let n = out.cols();
        let height = out.rows();
        let mode = match (input.rows(), height) {
            (1, 1) => 0,
            (1, 4) => 1,
            _ => 2,
        };
        let params = [k, n, mode, 0u32];
        let kernel = if n != VOCAB && n.is_multiple_of(4) {
            self.gemv_vector
        } else {
            self.gemv
        };
        Dispatch::new(encoder, kernel, Fence::Bound)
            .read(input)
            .read(weight)
            .write(output)
            .read(self.experts)
            .bytes(&params)
            .groups_2d([n.div_ceil(32), height], 128);
    }
}

/// Chunked 128x128 tile GEMM projection operator (`Rows::Chunk`) using Metal 4
/// TensorOps.
#[derive(Clone, Copy)]
pub(crate) struct TileLinear<'a> {
    pipeline: &'a ComputePipelineState,
    simdgroups: u64,
}

impl<'a> TileLinear<'a> {
    pub(crate) fn new(pipeline: &'a ComputePipelineState, simdgroups: u64) -> Result<Self, String> {
        let simd_width = pipeline.thread_execution_width();
        let threads = simd_width * simdgroups;
        if simd_width != 32 || pipeline.max_total_threads_per_threadgroup() < threads {
            return Err(format!(
                "TensorOps kernel requires {simdgroups} 32-lane SIMD groups; pipeline reports \
                 SIMD width {simd_width} and max threads {}",
                pipeline.max_total_threads_per_threadgroup()
            ));
        }
        Ok(Self {
            pipeline,
            simdgroups,
        })
    }

    /// Dense 128x128 tiled matrix multiplication: `output = input * weight`.
    pub(crate) fn forward(
        self,
        encoder: &ComputeCommandEncoderRef,
        input: View<'_, F16>,
        weight: View<'_, F16>,
        output: Out<'_, F16>,
    ) {
        let rows = input.rows();
        let cols = output.view().cols();
        Dispatch::new(encoder, self.pipeline, Fence::Writes)
            .read(input)
            .read(weight)
            .write(output)
            .groups_2d([cols / 128, rows / 128], self.simdgroups * 32);
    }

    /// Dense 128x128 tiled matrix multiplication staged inside a range-aware
    /// [`Wave`].
    pub(crate) fn forward_wave<'w>(
        self,
        wave: &mut Wave<'w>,
        input: View<'w, F16>,
        weight: View<'w, F16>,
        output: Out<'w, F16>,
    ) where
        'a: 'w,
    {
        let rows = input.rows();
        let cols = output.view().cols();
        Dispatch::new(wave.encoder(), self.pipeline, Fence::None)
            .read(input)
            .read(weight)
            .write(output)
            .in_wave(wave)
            .groups_2d([cols / 128, rows / 128], self.simdgroups * 32);
    }
}
