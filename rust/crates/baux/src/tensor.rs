//! Typed, non-owning views over resident Metal buffers, and kernel dispatch.
//!
//! A [`View`] is a window into a buffer: byte offset plus `rows x cols`
//! elements of one type. The element type is part of the Rust type, so an
//! `F32` buffer cannot be bound where a kernel expects `F16`. Shape is runtime
//! data: only the row count varies (chunk tiles vs. single-token steps); column
//! counts are model constants. Views never allocate, read back, or submit.
//!
//! Access modes follow the OxCaml / DeepInfer discipline:
//! - [`View`] is `Copy` and represents a shared read-only view.
//! - [`Out`] is move-only (`!Copy`, `#[must_use]`) and marks a write
//!   destination (`view.out()`), consumed by [`Dispatch::write`]. It does not
//!   prove that another view cannot alias that destination.
//!
//! [`Dispatch`] binds views to consecutive kernel slots and declares, per slot,
//! whether the kernel reads or writes it. That declaration decides which
//! resources receive a memory barrier after the dispatch, replacing
//! hand-written barrier lists that could silently omit a written buffer.

use std::marker::PhantomData;
use std::mem::MaybeUninit;

use metal::{BufferRef, ComputeCommandEncoderRef, ComputePipelineState, MTLSize, ResourceRef};

#[path = "tensor/wave.rs"]
mod wave;
pub(crate) use self::wave::Wave;
#[cfg(test)]
pub(crate) use self::wave::WaveDispatch;

/// Element type stored in a buffer (unboxed physical layout).
pub(crate) trait Element: Copy {
    const BYTES: u64;
}

/// Signed 8-bit integer; quantized gate-up expert weights.
#[derive(Clone, Copy)]
pub(crate) struct I8;
/// IEEE half; all weights and activations.
#[derive(Clone, Copy)]
pub(crate) struct F16;
/// Single precision; coefficients, losses, accumulators.
#[derive(Clone, Copy)]
pub(crate) struct F32;
/// Token ids, routing indices, indirect dispatch grid dimensions.
#[derive(Clone, Copy)]
pub(crate) struct U32;

impl Element for I8 {
    const BYTES: u64 = 1;
}
impl Element for F16 {
    const BYTES: u64 = 2;
}
impl Element for F32 {
    const BYTES: u64 = 4;
}
impl Element for U32 {
    const BYTES: u64 = 4;
}

/// Row-major `rows x cols` window of `T` elements inside one buffer (read
/// mode).
#[derive(Clone, Copy)]
pub(crate) struct View<'a, T: Element> {
    buffer: &'a BufferRef,
    offset: u64,
    rows: u32,
    cols: u32,
    element: PhantomData<T>,
}

/// Move-only write-destination view; allocation uniqueness is not implied.
///
/// Created via [`View::out`] and consumed by [`Dispatch::write`] or layer
/// methods that mutate the underlying device memory.
#[must_use = "an Out view declares a kernel write destination"]
pub(crate) struct Out<'a, T: Element>(View<'a, T>);

impl<'a, T: Element> Out<'a, T> {
    #[inline(always)]
    pub(crate) const fn view(&self) -> View<'a, T> {
        self.0
    }

    #[inline(always)]
    pub(crate) const fn into_view(self) -> View<'a, T> {
        self.0
    }
}

impl<'a, T: Element> View<'a, T> {
    pub(crate) fn new(buffer: &'a BufferRef, offset: u64, rows: u32, cols: u32) -> Self {
        assert!(T::BYTES > 0, "zero-width GPU element");
        assert_eq!(offset % T::BYTES, 0, "unaligned GPU view");
        baux::semantics::memory::view_bytes(1, cols.into(), T::BYTES)
            .expect("GPU row stride overflow");
        let bytes = baux::semantics::memory::view_bytes(rows.into(), cols.into(), T::BYTES)
            .expect("GPU view byte count overflow");
        assert!(
            baux::semantics::memory::span_end(buffer.length(), offset, bytes).is_some(),
            "GPU view exceeds its buffer"
        );
        let view = Self {
            buffer,
            offset,
            rows,
            cols,
            element: PhantomData,
        };
        view
    }

    /// A slice of a buffer starting at `offset` with explicit shape `(rows,
    /// cols)`.
    #[inline(always)]
    pub(crate) fn slice(buffer: &'a BufferRef, offset: u64, rows: u32, cols: u32) -> Self {
        Self::new(buffer, offset, rows, cols)
    }

    /// The whole buffer as rows of `cols` elements (row capacity from length).
    pub(crate) fn matrix(buffer: &'a BufferRef, cols: u32) -> Self {
        let stride = baux::semantics::memory::view_bytes(1, cols.into(), T::BYTES)
            .filter(|stride| *stride > 0)
            .expect("invalid GPU row width");
        let rows = u32::try_from(buffer.length() / stride).expect("GPU row count exceeds u32");
        Self::new(buffer, 0, rows, cols)
    }

    /// The whole buffer as one flat row, for layouts owned by another module.
    #[inline(always)]
    pub(crate) fn whole(buffer: &'a BufferRef) -> Self {
        assert!(T::BYTES > 0, "zero-width GPU element");
        assert_eq!(buffer.length() % T::BYTES, 0, "partial GPU element");
        let cols =
            u32::try_from(buffer.length() / T::BYTES).expect("flat GPU view exceeds u32 elements");
        Self::new(buffer, 0, 1, cols)
    }

    /// Rows `start..start + rows`.
    pub(crate) fn window(self, start: u32, rows: u32) -> Self {
        assert!(
            baux::semantics::memory::span_end(self.rows.into(), start.into(), rows.into())
                .is_some(),
            "window exceeds {} rows",
            self.rows
        );
        let offset = self.offset + u64::from(start) * self.row_bytes();
        Self::new(self.buffer, offset, rows, self.cols)
    }

    /// Row `index`, e.g. one layer of a layer-stacked parameter.
    #[inline(always)]
    pub(crate) fn row(self, index: u32) -> Self {
        self.window(index, 1)
    }

    /// Promote this view into a move-only write destination (`Out`).
    #[inline(always)]
    pub(crate) const fn out(self) -> Out<'a, T> {
        Out(self)
    }

    #[inline(always)]
    pub(crate) const fn rows(self) -> u32 {
        self.rows
    }

    #[inline(always)]
    pub(crate) const fn cols(self) -> u32 {
        self.cols
    }

    #[inline(always)]
    pub(crate) const fn buffer(self) -> &'a BufferRef {
        self.buffer
    }

    #[inline(always)]
    pub(crate) const fn offset(self) -> u64 {
        self.offset
    }

    /// Byte range `[offset, offset + bytes)` covered by this view.
    #[inline(always)]
    pub(crate) const fn span(self) -> Span<'a> {
        Span {
            buffer: self.buffer,
            start: self.offset,
            end: self.offset + self.bytes(),
        }
    }

    #[inline(always)]
    const fn row_bytes(self) -> u64 {
        (self.cols as u64) * T::BYTES
    }

    #[inline(always)]
    const fn bytes(self) -> u64 {
        (self.rows as u64) * self.row_bytes()
    }
}

/// Contiguous byte range `[start, end)` within a single [`BufferRef`].
#[derive(Clone, Copy)]
pub(crate) struct Span<'a> {
    buffer: &'a BufferRef,
    start: u64,
    end: u64,
}

impl<'a> Span<'a> {
    /// True when both spans reside in the same Metal buffer and their byte
    /// ranges intersect.
    #[inline(always)]
    pub(crate) fn overlaps(self, other: Self) -> bool {
        std::ptr::eq(self.buffer, other.buffer)
            && baux::semantics::memory::overlaps(self.start, self.end, other.start, other.end)
    }
}

/// Which resources a dispatch fences once it is encoded.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum Fence {
    /// Do not emit a memory barrier after this dispatch (caller batches
    /// multiple independent dispatches before a shared barrier).
    None,
    /// Every slot the kernel writes, plus explicit [`Dispatch::fence`] entries.
    Writes,
    /// Every bound slot, in slot order (conservative single-token decode).
    Bound,
}

const SLOTS: usize = 12;
const CONSTANT_BYTES: usize = 4096;

/// One kernel launch: pipeline, slots bound in order, then a grid.
///
/// Buffers and bytes take consecutive slot indices starting at zero, matching
/// the `[[buffer(n)]]` order of the kernel signature.
#[must_use = "a dispatch does nothing until a grid method encodes it"]
pub(crate) struct Dispatch<'a> {
    encoder: &'a ComputeCommandEncoderRef,
    pipeline: &'a ComputePipelineState,
    fence: Fence,
    slot: u64,
    bound: [Option<&'a ResourceRef>; SLOTS],
    spans: [Option<Span<'a>>; SLOTS],
    written: [bool; SLOTS],
    extra: Option<Span<'a>>,
    constants: [Option<(usize, usize)>; SLOTS],
    values: [MaybeUninit<u8>; CONSTANT_BYTES],
    used: usize,
}

impl<'a> Dispatch<'a> {
    pub(crate) fn new(
        encoder: &'a ComputeCommandEncoderRef,
        pipeline: &'a ComputePipelineState,
        fence: Fence,
    ) -> Self {
        Self {
            encoder,
            pipeline,
            fence,
            slot: 0,
            bound: [None; SLOTS],
            spans: [None; SLOTS],
            written: [false; SLOTS],
            extra: None,
            constants: [None; SLOTS],
            values: [MaybeUninit::uninit(); CONSTANT_BYTES],
            used: 0,
        }
    }

    fn bind<T: Element>(mut self, view: View<'a, T>, written: bool) -> Self {
        let slot = self.slot as usize;
        assert!(slot < SLOTS, "GPU dispatch has too many argument slots");
        let resource: &ResourceRef = view.buffer;
        self.bound[slot] = Some(resource);
        self.spans[slot] = Some(view.span());
        self.written[slot] = written;
        self.slot += 1;
        self
    }

    /// The kernel only reads this slot.
    #[inline(always)]
    pub(crate) fn read<T: Element>(self, view: View<'a, T>) -> Self {
        self.bind(view, false)
    }

    /// The kernel writes this slot (consuming the linear `Out` capability).
    #[inline(always)]
    pub(crate) fn write<T: Element>(self, out: Out<'a, T>) -> Self {
        self.bind(out.0, true)
    }

    /// Small constant parameters at the next slot.
    #[inline(always)]
    pub(crate) fn bytes<P: Copy>(mut self, value: &P) -> Self {
        assert!(
            self.slot < SLOTS as u64,
            "GPU dispatch has too many argument slots"
        );
        let length = std::mem::size_of::<P>();
        let end = self
            .used
            .checked_add(length)
            .expect("GPU constant size overflow");
        assert!(
            end <= CONSTANT_BYTES,
            "GPU dispatch constants exceed 4096 bytes"
        );
        // Preserve object representation, including potentially uninitialized
        // padding, without interpreting it as initialized Rust u8 values.
        // Metal copies these bytes when the launch is encoded.
        unsafe {
            std::ptr::copy_nonoverlapping(
                std::ptr::from_ref(value).cast::<MaybeUninit<u8>>(),
                self.values.as_mut_ptr().add(self.used),
                length,
            );
        }
        self.constants[self.slot as usize] = Some((self.used, length));
        self.used = end;
        self.slot += 1;
        self
    }

    /// Also fence a read slot because the next dispatch overwrites it.
    #[inline(always)]
    pub(crate) fn fence<T: Element>(mut self, view: View<'a, T>) -> Self {
        self.extra = Some(view.span());
        self
    }

    /// `groups` 1D threadgroups of `threads` threads.
    pub(crate) fn groups(self, groups: u32, threads: u64) {
        self.prepare(threads);
        self.encoder
            .dispatch_thread_groups(grid(u64::from(groups)), grid(threads));
        self.finish();
    }

    /// 2D threadgroups `(width, height)` of `threads` threads.
    pub(crate) fn groups_2d(self, [width, height]: [u32; 2], threads: u64) {
        self.prepare(threads);
        self.encoder.dispatch_thread_groups(
            MTLSize {
                width: u64::from(width),
                height: u64::from(height),
                depth: 1,
            },
            grid(threads),
        );
        self.finish();
    }

    /// `total` threads in groups of at most `limit`, capped by the pipeline.
    pub(crate) fn threads(self, total: u64, limit: u64) {
        let group = self.pipeline.max_total_threads_per_threadgroup().min(limit);
        self.prepare(group);
        self.encoder.dispatch_threads(grid(total), grid(group));
        self.finish();
    }

    /// Bind the complete captured argument list immediately before launch.
    /// Building another Dispatch cannot replace this launch's pipeline or
    /// values.
    fn prepare(&self, threads: u64) {
        assert!(
            threads > 0 && threads <= self.pipeline.max_total_threads_per_threadgroup(),
            "GPU threadgroup size exceeds pipeline capacity"
        );
        self.encoder.set_compute_pipeline_state(self.pipeline);
        for slot in 0..self.slot as usize {
            if let Some(span) = self.spans[slot] {
                self.encoder
                    .set_buffer(slot as u64, Some(span.buffer), span.start);
            } else if let Some((offset, length)) = self.constants[slot] {
                self.encoder.set_bytes(
                    slot as u64,
                    length as u64,
                    self.values[offset..].as_ptr().cast(),
                );
            } else {
                unreachable!("GPU dispatch contains an unbound slot");
            }
        }
    }

    fn finish(self) {
        if self.fence == Fence::None {
            return;
        }
        let all = self.fence == Fence::Bound;
        let extra = self.extra.map(|span| -> &'a ResourceRef { span.buffer });
        let fenced = self
            .bound
            .iter()
            .zip(self.written)
            .filter_map(move |(resource, written)| resource.filter(|_| written || all))
            .chain(extra);
        let Some(first) = fenced.clone().next() else {
            return;
        };
        let mut resources = [first; SLOTS + 1];
        let mut count = 0;
        for resource in fenced {
            if !resources[..count]
                .iter()
                .any(|existing| std::ptr::eq(*existing, resource))
            {
                resources[count] = resource;
                count += 1;
            }
        }
        self.encoder
            .memory_barrier_with_resources(&resources[..count]);
    }
}

const fn grid(width: u64) -> MTLSize {
    MTLSize {
        width,
        height: 1,
        depth: 1,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn view_bounds() {
        let runtime = crate::apple_gpu::Runtime::shared().unwrap();
        let buffer = runtime.buffer::<u32>(4);
        let invalid = [
            (u64::MAX, 1, 1),
            (0, u32::MAX, u32::MAX),
            (12, 1, 2),
            (1, 1, 1),
        ];
        for (offset, rows, cols) in invalid {
            assert!(
                std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    View::<F32>::new(&buffer, offset, rows, cols)
                }))
                .is_err()
            );
        }
        let view = View::<F32>::new(&buffer, 0, 4, 1);
        assert!(
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                view.window(u32::MAX, 1)
            }))
            .is_err()
        );
        assert!(!view.window(2, 0).span().overlaps(view.span()));
        assert_eq!(view.window(4, 0).span().end, buffer.length());
    }

    #[test]
    fn bound_launch() {
        let runtime = crate::apple_gpu::Runtime::shared().unwrap();
        let source = "#include <metal_stdlib>\nusing namespace metal;\nkernel void first(device \
                      uint *out [[buffer(0)]], constant uint &x [[buffer(1)]]) { out[0] = x + 1u; \
                      }\nkernel void second(device uint *out [[buffer(0)]], constant uint &x \
                      [[buffer(1)]]) { out[0] = x + 2u; }";
        let first = runtime.precise(source, "bound", "first").unwrap();
        let second = runtime.precise(source, "bound", "second").unwrap();
        let a = runtime.buffer::<u32>(1);
        let b = runtime.buffer::<u32>(1);
        let command = runtime.queue.new_command_buffer();
        let encoder = command.new_compute_command_encoder();
        let one = Dispatch::new(encoder, &first, Fence::Writes)
            .write(View::<U32>::new(&a, 0, 1, 1).out())
            .bytes(&12u32);
        let two = Dispatch::new(encoder, &second, Fence::Writes)
            .write(View::<U32>::new(&b, 0, 1, 1).out())
            .bytes(&22u32);
        one.groups(1, 1);
        two.groups(1, 1);
        encoder.end_encoding();
        command.commit();
        command.wait_until_completed();
        assert_eq!(command.status(), metal::MTLCommandBufferStatus::Completed);
        // Both shared allocations contain a u32 written by their completed
        // kernel.
        unsafe {
            assert_eq!(*a.contents().cast::<u32>(), 13);
            assert_eq!(*b.contents().cast::<u32>(), 24);
        }
    }

    #[test]
    fn read_barrier() {
        let runtime = match crate::apple_gpu::Runtime::shared() {
            Ok(runtime) => runtime,
            Err(error) if error.contains("no default Metal device found") => return,
            Err(error) => panic!("{error}"),
        };
        let a = runtime.buffer::<u32>(4);
        let b = runtime.buffer::<u32>(4);
        let command = runtime.queue.new_command_buffer();
        let encoder = command.new_compute_command_encoder();
        let mut wave = Wave::new(encoder);
        let span = |buffer| Span {
            buffer,
            start: 0,
            end: 16,
        };
        for (written, expected) in [(None, 1), (Some(&b), 2), (Some(&a), 1)] {
            wave.reads[0] = Some(span(&a));
            wave.read_count = 1;
            if let Some(buffer) = written {
                wave.writes[0] = Some(span(buffer));
                wave.write_count = 1;
            }
            let mut calls = 0;
            wave.flush_with(|resources| {
                calls += 1;
                assert_eq!(resources.len(), expected);
                let resource: &ResourceRef = &a;
                assert!(resources.iter().any(|item| std::ptr::eq(*item, resource)));
            });
            assert_eq!(calls, 1, "read-only waves must also emit a barrier");
            assert_eq!((wave.read_count, wave.write_count), (0, 0));
        }
        drop(wave);
        encoder.end_encoding();
    }
}
