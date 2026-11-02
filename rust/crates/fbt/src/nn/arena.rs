//! Zero-allocation resident scratch memory arena inspired by `ggml_tallocr`.
//!
//! Pre-allocates resident [`metal::Buffer`] instances on Apple Silicon unified
//! memory for intermediate activation scratch, eliminating repetitive driver
//! buffer allocations and page faults during 1M-token chunk evaluation and
//! single-token decode.
//!
//! Provides static byte offset computation, intermediate sub-slice creation,
//! type-safe dynamic bump allocation ([`alloc_view`], [`alloc_out`]), and
//! O(1) rapid reset between execution steps without driver reallocation.

use std::cell::Cell;

use metal::{Buffer, BufferRef, DeviceRef, MTLResourceOptions};

use crate::apple_gpu::Runtime;
use crate::nn::{EXPERT_WIDTH, QKV_WIDTH, WIDTH};
use crate::tensor::{Element, F16, F32, Out, View};

/// Default byte alignment matching Apple Silicon unified cacheline and SIMD
/// boundaries.
pub(crate) const HARDWARE_ALIGN: u64 = 256;

/// Gated MLP intermediate column extent (`2 * EXPERT_WIDTH`).
pub(crate) const GATE_WIDTH: u32 = baux::semantics::shape::GATE_WIDTH as u32;

/// Active experts per token in sparse MoE (3 routed + 1 shared).
pub(crate) const ACTIVE_EXPERTS: u32 =
    (baux::semantics::shape::ACTIVE_COUNT + baux::semantics::shape::SHARED_COUNT) as u32;

/// Hyper-connection mixing coefficients per row.
pub(crate) const MHC_COEFS: u32 = baux::semantics::shape::MHC_COEFS as u32;

/// Align a byte offset to the nearest multiple of `align`.
#[inline(always)]
pub(crate) const fn align_offset(offset: u64, align: u64) -> u64 {
    (offset + align - 1) & !(align - 1)
}

/// Static byte offsets for intermediate activations within resident arena
/// memory.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct StaticOffsets {
    pub(crate) qkv_row: u64,
    pub(crate) attn_scratch: u64,
    pub(crate) gate: u64,
    pub(crate) activation: u64,
    pub(crate) down: u64,
    pub(crate) coefficients: u64,
    pub(crate) total_bytes: u64,
}

impl StaticOffsets {
    /// Compute static offsets starting at `base` offset with given alignment.
    pub(crate) fn compute(base: u64, rows: u32, align: u64) -> Self {
        let count = rows.max(1) as u64;
        let qkv_bytes = count * (QKV_WIDTH as u64) * F16::BYTES;
        let attn_bytes = count * (WIDTH as u64) * F16::BYTES;
        let moe_rows = count * (ACTIVE_EXPERTS as u64);
        let gate_bytes = moe_rows * (GATE_WIDTH as u64) * F16::BYTES;
        let act_bytes = moe_rows * (EXPERT_WIDTH as u64) * F16::BYTES;
        let down_bytes = moe_rows * (WIDTH as u64) * F16::BYTES;
        let coef_bytes = count * (MHC_COEFS as u64) * F32::BYTES;

        let qkv_offset = align_offset(base, align);
        let attn_offset = align_offset(qkv_offset + qkv_bytes, align);
        let gate_offset = align_offset(attn_offset + attn_bytes, align);
        let act_offset = align_offset(gate_offset + gate_bytes, align);
        let down_offset = align_offset(act_offset + act_bytes, align);
        let coef_offset = align_offset(down_offset + down_bytes, align);
        let total_bytes = align_offset(coef_offset + coef_bytes, align);

        Self {
            qkv_row: qkv_offset,
            attn_scratch: attn_offset,
            gate: gate_offset,
            activation: act_offset,
            down: down_offset,
            coefficients: coef_offset,
            total_bytes,
        }
    }

    /// Static offsets for single-token decode (`Rows::Step`).
    #[inline(always)]
    pub(crate) fn step(align: u64) -> Self {
        Self::compute(0, 1, align)
    }

    /// Static offsets for chunk evaluation (`Rows::Chunk(rows)`).
    #[inline(always)]
    pub(crate) fn chunk(rows: u32, align: u64) -> Self {
        Self::compute(0, rows, align)
    }
}

/// Typed intermediate activation views sliced from resident arena memory.
#[derive(Clone, Copy)]
pub(crate) struct Activations<'a> {
    pub(crate) qkv_row: View<'a, F16>,
    pub(crate) attn_scratch: View<'a, F16>,
    pub(crate) gate: View<'a, F16>,
    pub(crate) activation: View<'a, F16>,
    pub(crate) down: View<'a, F16>,
    pub(crate) coefficients: View<'a, F32>,
}

impl<'a> Activations<'a> {
    /// Promote all activation views into move-only write destinations (`Out`).
    #[inline(always)]
    pub(crate) fn outs(self) -> ActivationOuts<'a> {
        ActivationOuts {
            qkv_row: self.qkv_row.out(),
            attn_scratch: self.attn_scratch.out(),
            gate: self.gate.out(),
            activation: self.activation.out(),
            down: self.down.out(),
            coefficients: self.coefficients.out(),
        }
    }
}

/// Typed intermediate activation linear write handles (`Out`).
#[must_use = "activation write destinations must be consumed by kernel writes"]
pub(crate) struct ActivationOuts<'a> {
    pub(crate) qkv_row: Out<'a, F16>,
    pub(crate) attn_scratch: Out<'a, F16>,
    pub(crate) gate: Out<'a, F16>,
    pub(crate) activation: Out<'a, F16>,
    pub(crate) down: Out<'a, F16>,
    pub(crate) coefficients: Out<'a, F32>,
}

impl<'a> ActivationOuts<'a> {
    /// Convert write destinations back into read views.
    #[inline(always)]
    pub(crate) fn views(self) -> Activations<'a> {
        Activations {
            qkv_row: self.qkv_row.into_view(),
            attn_scratch: self.attn_scratch.into_view(),
            gate: self.gate.into_view(),
            activation: self.activation.into_view(),
            down: self.down.into_view(),
            coefficients: self.coefficients.into_view(),
        }
    }
}

/// Scratch ring memory arena backed by resident Metal buffers.
pub(crate) struct MemoryArena {
    buffers: Vec<Buffer>,
    cursor: Cell<u64>,
    slot: Cell<usize>,
    alignment: u64,
}

impl MemoryArena {
    /// Create arena backed by one resident buffer with capacity `bytes`.
    pub(crate) fn new(runtime: &Runtime, bytes: usize) -> Self {
        Self::with_device(&runtime.device, bytes)
    }

    /// Create arena from an existing pre-allocated resident Metal buffer.
    pub(crate) fn from_buffer(buffer: Buffer) -> Self {
        Self {
            buffers: vec![buffer],
            cursor: Cell::new(0),
            slot: Cell::new(0),
            alignment: HARDWARE_ALIGN,
        }
    }

    /// Create arena from a pre-allocated ring of resident Metal buffers.
    pub(crate) fn from_ring(buffers: Vec<Buffer>) -> Self {
        assert!(!buffers.is_empty(), "scratch ring cannot be empty");
        Self {
            buffers,
            cursor: Cell::new(0),
            slot: Cell::new(0),
            alignment: HARDWARE_ALIGN,
        }
    }

    /// Create arena using a Metal device handle.
    pub(crate) fn with_device(device: &DeviceRef, bytes: usize) -> Self {
        let size = (bytes.max(256)) as u64;
        let raw = device.new_buffer(
            size,
            MTLResourceOptions::StorageModeShared | MTLResourceOptions::HazardTrackingModeTracked,
        );
        Self::from_buffer(raw)
    }

    /// Pre-allocate a multi-slot resident scratch ring.
    pub(crate) fn ring(runtime: &Runtime, bytes: usize, slots: usize) -> Self {
        let count = slots.max(1);
        let size = (bytes.max(256)) as u64;
        let mut buffers = Vec::with_capacity(count);
        for _ in 0..count {
            buffers.push(runtime.device.new_buffer(
                size,
                MTLResourceOptions::StorageModeShared
                    | MTLResourceOptions::HazardTrackingModeTracked,
            ));
        }
        Self::from_ring(buffers)
    }

    /// Pre-allocate resident memory sized for single-token decode activations.
    pub(crate) fn for_step(runtime: &Runtime) -> Self {
        let offsets = StaticOffsets::step(HARDWARE_ALIGN);
        Self::new(runtime, offsets.total_bytes as usize)
    }

    /// Active resident Metal buffer for the current ring slot.
    #[inline(always)]
    pub(crate) fn active_buffer(&self) -> &BufferRef {
        &self.buffers[self.slot.get()]
    }

    /// Total capacity in bytes of the active resident buffer.
    #[inline(always)]
    pub(crate) fn capacity(&self) -> u64 {
        self.active_buffer().length()
    }

    /// Current allocation offset cursor in bytes.
    #[inline(always)]
    pub(crate) fn cursor(&self) -> u64 {
        self.cursor.get()
    }

    /// Current active ring slot index.
    #[inline(always)]
    pub(crate) fn slot(&self) -> usize {
        self.slot.get()
    }

    /// Total number of resident buffers in the scratch ring.
    #[inline(always)]
    pub(crate) fn ring_size(&self) -> usize {
        self.buffers.len()
    }

    /// Rapid O(1) reset between execution steps without driver reallocation.
    #[inline(always)]
    pub(crate) fn reset(&self) {
        self.cursor.set(0);
    }

    /// Advance to the next scratch ring slot and reset the cursor.
    pub(crate) fn advance(&self) {
        let next = (self.slot.get() + 1) % self.buffers.len();
        self.slot.set(next);
        self.cursor.set(0);
    }

    /// Capture current allocation cursor as a rewind checkpoint.
    #[inline(always)]
    pub(crate) fn checkpoint(&self) -> u64 {
        self.cursor.get()
    }

    /// Rewind cursor to a previous checkpoint without freeing memory.
    pub(crate) fn rewind(&self, mark: u64) {
        assert!(mark <= self.cursor.get(), "cannot rewind past cursor");
        self.cursor.set(mark);
    }

    /// Allocate a type-safe read [`View`] for `rows x cols` elements of type
    /// `T`.
    pub(crate) fn alloc_view<'a, T: Element>(&'a self, rows: u32, cols: u32) -> View<'a, T> {
        let elements = (rows as u64) * (cols as u64);
        let bytes = elements * T::BYTES;
        let start = align_offset(self.cursor.get(), self.alignment);
        let end = start.checked_add(bytes).expect("allocation overflow");
        let cap = self.capacity();
        assert!(
            end <= cap,
            "arena capacity {cap} exceeded by allocation of {end} bytes"
        );
        self.cursor.set(end);
        View::slice(self.active_buffer(), start, rows, cols)
    }

    /// Allocate a type-safe move-only write destination [`Out`].
    #[inline(always)]
    pub(crate) fn alloc_out<'a, T: Element>(&'a self, rows: u32, cols: u32) -> Out<'a, T> {
        self.alloc_view::<T>(rows, cols).out()
    }

    /// Fallible view allocation returning an error instead of panicking.
    pub(crate) fn try_view<'a, T: Element>(
        &'a self,
        rows: u32,
        cols: u32,
    ) -> Result<View<'a, T>, String> {
        let elements = (rows as u64) * (cols as u64);
        let bytes = elements * T::BYTES;
        let start = align_offset(self.cursor.get(), self.alignment);
        let end = start.checked_add(bytes).ok_or("allocation overflow")?;
        let cap = self.capacity();
        if end > cap {
            return Err(format!("arena capacity {cap} exceeded: needed {end} bytes"));
        }
        self.cursor.set(end);
        Ok(View::slice(self.active_buffer(), start, rows, cols))
    }

    /// Sub-slices for all intermediate activations for `rows` tokens.
    pub(crate) fn activations<'a>(&'a self, rows: u32) -> Activations<'a> {
        let offsets = StaticOffsets::compute(0, rows, self.alignment);
        let cap = self.capacity();
        assert!(
            offsets.total_bytes <= cap,
            "arena capacity {cap} too small for static activations {} bytes",
            offsets.total_bytes
        );
        let buf = self.active_buffer();
        let moe_rows = rows.saturating_mul(ACTIVE_EXPERTS);
        Activations {
            qkv_row: View::slice(buf, offsets.qkv_row, rows, QKV_WIDTH),
            attn_scratch: View::slice(buf, offsets.attn_scratch, rows, WIDTH),
            gate: View::slice(buf, offsets.gate, moe_rows, GATE_WIDTH),
            activation: View::slice(buf, offsets.activation, moe_rows, EXPERT_WIDTH),
            down: View::slice(buf, offsets.down, moe_rows, WIDTH),
            coefficients: View::slice(buf, offsets.coefficients, rows, MHC_COEFS),
        }
    }

    /// Intermediate activation sub-slices for single-token decode.
    #[inline(always)]
    pub(crate) fn step_activations<'a>(&'a self) -> Activations<'a> {
        self.activations(1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_offsets() {
        let offsets = StaticOffsets::step(HARDWARE_ALIGN);
        assert_eq!(offsets.qkv_row, 0);
        assert!(offsets.attn_scratch > offsets.qkv_row);
        assert!(offsets.gate > offsets.attn_scratch);
        assert!(offsets.activation > offsets.gate);
        assert!(offsets.down > offsets.activation);
        assert!(offsets.coefficients > offsets.down);
        assert!(offsets.total_bytes > offsets.coefficients);

        assert_eq!(offsets.qkv_row % HARDWARE_ALIGN, 0);
        assert_eq!(offsets.attn_scratch % HARDWARE_ALIGN, 0);
        assert_eq!(offsets.gate % HARDWARE_ALIGN, 0);
        assert_eq!(offsets.activation % HARDWARE_ALIGN, 0);
        assert_eq!(offsets.down % HARDWARE_ALIGN, 0);
        assert_eq!(offsets.coefficients % HARDWARE_ALIGN, 0);
        assert_eq!(offsets.total_bytes % HARDWARE_ALIGN, 0);
    }

    #[test]
    fn test_chunk() {
        let step = StaticOffsets::chunk(1, HARDWARE_ALIGN);
        let chunk = StaticOffsets::chunk(64, HARDWARE_ALIGN);
        assert!(chunk.total_bytes > step.total_bytes);
        assert_eq!(chunk.qkv_row % HARDWARE_ALIGN, 0);
        assert_eq!(chunk.attn_scratch % HARDWARE_ALIGN, 0);
        assert_eq!(chunk.gate % HARDWARE_ALIGN, 0);
        assert_eq!(chunk.activation % HARDWARE_ALIGN, 0);
        assert_eq!(chunk.down % HARDWARE_ALIGN, 0);
        assert_eq!(chunk.coefficients % HARDWARE_ALIGN, 0);
        assert_eq!(chunk.total_bytes % HARDWARE_ALIGN, 0);
    }

    #[test]
    fn test_alloc() {
        let runtime = match Runtime::shared() {
            Ok(rt) => rt,
            Err(_) => return,
        };
        let arena = MemoryArena::new(&runtime, 65536);
        assert_eq!(arena.cursor(), 0);
        assert!(arena.capacity() >= 65536);

        let view = arena.alloc_view::<F16>(2, 64);
        assert_eq!(view.rows(), 2);
        assert_eq!(view.cols(), 64);
        assert!(arena.cursor() >= 256);

        let out = arena.alloc_out::<F32>(1, 32);
        assert_eq!(out.view().rows(), 1);
        assert_eq!(out.view().cols(), 32);
        assert!(arena.cursor() >= 384);
    }

    #[test]
    fn test_reset() {
        let runtime = match Runtime::shared() {
            Ok(rt) => rt,
            Err(_) => return,
        };
        let arena = MemoryArena::new(&runtime, 32768);
        let _ = arena.alloc_view::<F16>(4, 128);
        assert!(arena.cursor() > 0);
        arena.reset();
        assert_eq!(arena.cursor(), 0);
        let next = arena.alloc_view::<F16>(1, 64);
        assert_eq!(next.offset(), 0);
    }

    #[test]
    fn test_ring() {
        let runtime = match Runtime::shared() {
            Ok(rt) => rt,
            Err(_) => return,
        };
        let arena = MemoryArena::ring(&runtime, 16384, 3);
        assert_eq!(arena.ring_size(), 3);
        assert_eq!(arena.slot(), 0);

        let _ = arena.alloc_view::<F16>(1, 64);
        assert!(arena.cursor() > 0);

        arena.advance();
        assert_eq!(arena.slot(), 1);
        assert_eq!(arena.cursor(), 0);

        arena.advance();
        assert_eq!(arena.slot(), 2);
        assert_eq!(arena.cursor(), 0);

        arena.advance();
        assert_eq!(arena.slot(), 0);
        assert_eq!(arena.cursor(), 0);
    }

    #[test]
    fn test_activations() {
        let runtime = match Runtime::shared() {
            Ok(rt) => rt,
            Err(_) => return,
        };
        let arena = MemoryArena::for_step(&runtime);
        let acts = arena.step_activations();
        assert_eq!(acts.qkv_row.rows(), 1);
        assert_eq!(acts.qkv_row.cols(), QKV_WIDTH);
        assert_eq!(acts.attn_scratch.rows(), 1);
        assert_eq!(acts.attn_scratch.cols(), WIDTH);
        assert_eq!(acts.gate.rows(), ACTIVE_EXPERTS);
        assert_eq!(acts.gate.cols(), GATE_WIDTH);
        assert_eq!(acts.activation.rows(), ACTIVE_EXPERTS);
        assert_eq!(acts.activation.cols(), EXPERT_WIDTH);
        assert_eq!(acts.down.rows(), ACTIVE_EXPERTS);
        assert_eq!(acts.down.cols(), WIDTH);
        assert_eq!(acts.coefficients.rows(), 1);
        assert_eq!(acts.coefficients.cols(), MHC_COEFS);

        let outs = acts.outs();
        let views = outs.views();
        assert_eq!(views.qkv_row.cols(), QKV_WIDTH);
        assert_eq!(views.down.rows(), ACTIVE_EXPERTS);
    }

    #[test]
    fn test_checkpoint() {
        let runtime = match Runtime::shared() {
            Ok(rt) => rt,
            Err(_) => return,
        };
        let arena = MemoryArena::new(&runtime, 32768);
        let _ = arena.alloc_view::<F16>(1, 32);
        let mark = arena.checkpoint();
        let _ = arena.alloc_view::<F32>(2, 64);
        assert!(arena.cursor() > mark);
        arena.rewind(mark);
        assert_eq!(arena.cursor(), mark);
    }

    #[test]
    fn test_bounds() {
        let runtime = match Runtime::shared() {
            Ok(rt) => rt,
            Err(_) => return,
        };
        let arena = MemoryArena::new(&runtime, 512);
        let res = arena.try_view::<F32>(1000, 1000);
        assert!(res.is_err());
    }
}
