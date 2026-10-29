use metal::{ComputeCommandEncoderRef, ResourceRef};

use super::{Dispatch, Fence, SLOTS, Span};

const WAVE_SPANS: usize = 32;

/// Range-aware hazard tracker (`ggml_mem_ranges` discipline) across a wave of
/// dispatches.
///
/// Tracks active read and write [`Span`]s since the last barrier on `encoder`.
/// Independent dispatches (disjoint buffers, disjoint byte sub-ranges of the
/// same buffer, or shared read-only ranges) execute back-to-back without an
/// intervening barrier. When a subsequent dispatch introduces a RAW, WAR, or
/// WAW overlap with an active span in the wave, or when the [`Wave`] is
/// flushed/dropped, a single deduplicated `memory_barrier_with_resources` is
/// emitted for the accessed buffers.
pub(crate) struct Wave<'a> {
    pub(super) encoder: &'a ComputeCommandEncoderRef,
    pub(super) reads: [Option<Span<'a>>; WAVE_SPANS],
    pub(super) read_count: usize,
    pub(super) writes: [Option<Span<'a>>; WAVE_SPANS],
    pub(super) write_count: usize,
}

impl<'a> Wave<'a> {
    #[inline(always)]
    pub(crate) const fn new(encoder: &'a ComputeCommandEncoderRef) -> Self {
        Self {
            encoder,
            reads: [None; WAVE_SPANS],
            read_count: 0,
            writes: [None; WAVE_SPANS],
            write_count: 0,
        }
    }

    #[inline(always)]
    pub(crate) const fn encoder(&self) -> &'a ComputeCommandEncoderRef {
        self.encoder
    }

    fn conflicts(&self, dispatch: &Dispatch<'a>) -> bool {
        use baux::semantics::hazard::{self, Access};
        let descriptor = |span: Span<'a>, write| Access {
            buffer: std::ptr::from_ref(span.buffer) as usize as u64,
            start: span.start,
            end: span.end,
            write,
        };
        let mut active = [Access::default(); 2 * WAVE_SPANS];
        let mut active_count = 0;
        for (spans, write) in [
            (&self.reads[..self.read_count], false),
            (&self.writes[..self.write_count], true),
        ] {
            for span in spans.iter().flatten() {
                active[active_count] = descriptor(*span, write);
                active_count += 1;
            }
        }
        let mut pending = [Access::default(); SLOTS + 1];
        let mut pending_count = 0;
        let mut add_reads = 0;
        let mut add_writes = 0;
        for slot in 0..dispatch.slot as usize {
            if let Some(span) = dispatch.spans[slot] {
                let write = dispatch.written[slot];
                pending[pending_count] = descriptor(span, write);
                pending_count += 1;
                if write {
                    add_writes += 1;
                } else {
                    add_reads += 1;
                }
            }
        }
        if let Some(span) = dispatch.extra {
            pending[pending_count] = descriptor(span, true);
            pending_count += 1;
            add_writes += 1;
        }
        hazard::capacity(
            self.read_count as u32,
            self.write_count as u32,
            add_reads,
            add_writes,
        ) || hazard::flush(&active[..active_count], &pending[..pending_count])
    }

    fn stage(&mut self, dispatch: &Dispatch<'a>) {
        if self.conflicts(dispatch) {
            self.flush();
        }
        let count = dispatch.slot as usize;
        for i in 0..count {
            let Some(span) = dispatch.spans[i] else {
                continue;
            };
            if dispatch.written[i] {
                self.writes[self.write_count] = Some(span);
                self.write_count += 1;
            } else {
                self.reads[self.read_count] = Some(span);
                self.read_count += 1;
            }
        }
        if let Some(span) = dispatch.extra {
            self.writes[self.write_count] = Some(span);
            self.write_count += 1;
        }
    }

    /// Fence both reads and writes: an earlier read must finish before a later
    /// overlapping write, including when the current wave is entirely
    /// read-only.
    pub(crate) fn flush(&mut self) {
        let encoder = self.encoder;
        self.flush_with(|resources| encoder.memory_barrier_with_resources(resources));
    }

    pub(super) fn flush_with(&mut self, emit: impl FnOnce(&[&ResourceRef])) {
        if self.write_count == 0 && self.read_count == 0 {
            return;
        }
        let mut accessed = self.writes[..self.write_count]
            .iter()
            .chain(self.reads[..self.read_count].iter())
            .flatten()
            .map(|span| -> &'a ResourceRef { span.buffer });
        let Some(first) = accessed.next() else {
            self.read_count = 0;
            self.write_count = 0;
            return;
        };
        let mut resources = [first; 2 * WAVE_SPANS];
        let mut count = 1;
        for resource in accessed {
            if !resources[..count]
                .iter()
                .any(|existing| std::ptr::eq(*existing, resource))
            {
                resources[count] = resource;
                count += 1;
            }
        }
        emit(&resources[..count]);
        self.read_count = 0;
        self.write_count = 0;
    }
}

impl Drop for Wave<'_> {
    fn drop(&mut self) {
        self.flush();
    }
}

/// A fully bound dispatch staged and launched together under the same wave.
/// No further arguments can be appended after the hazard set is chosen.
#[must_use = "a wave dispatch must be launched"]
pub(crate) struct WaveDispatch<'w, 'a> {
    pub(super) dispatch: Dispatch<'a>,
    pub(super) wave: &'w mut Wave<'a>,
}

impl WaveDispatch<'_, '_> {
    pub(crate) fn groups_2d(self, shape: [u32; 2], threads: u64) {
        self.wave.stage(&self.dispatch);
        self.dispatch.groups_2d(shape, threads);
    }
}

impl<'a> Dispatch<'a> {
    /// Stage this dispatch inside `wave`, eliding the pre-dispatch barrier if
    /// its read/write byte ranges do not conflict with active spans in
    /// `wave`.
    #[inline(always)]
    pub(crate) fn in_wave<'w>(mut self, wave: &'w mut Wave<'a>) -> WaveDispatch<'w, 'a> {
        assert!(
            std::ptr::eq(self.encoder, wave.encoder),
            "GPU wave belongs to another encoder"
        );
        self.fence = Fence::None;
        WaveDispatch {
            dispatch: self,
            wave,
        }
    }
}
