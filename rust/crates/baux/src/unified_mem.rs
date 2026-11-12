//! Page-aligned zero-copy unified memory buffer governed by Rust's borrow
//! checker.

use ndarray::{ArrayView2, ArrayViewMut2};

/// 16 KB page-aligned block matching Apple Silicon unified memory page size.
#[repr(C, align(16384))]
#[derive(Clone, Copy)]
pub struct PageBlock(pub [f64; 2048]);

impl Default for PageBlock {
    fn default() -> Self {
        Self([0.0; 2048])
    }
}

/// Page-aligned unified memory tensor whose CPU/GPU access is statically
/// governed by Rust's borrow checker (`&UnifiedBuffer` for shared CPU/GPU read
/// vs `&mut UnifiedBuffer` for exclusive mutation).
#[derive(Clone)]
pub struct UnifiedBuffer {
    pages: Vec<PageBlock>,
    rows: usize,
    cols: usize,
}

impl UnifiedBuffer {
    pub fn zeros(rows: usize, cols: usize) -> Self {
        let total = rows.saturating_mul(cols);
        let num_pages = total.div_ceil(2048).max(1);
        Self {
            pages: vec![PageBlock::default(); num_pages],
            rows,
            cols,
        }
    }

    pub fn from_view(view: &ArrayView2<f64>) -> Self {
        let mut buf = Self::zeros(view.nrows(), view.ncols());
        buf.write_guard().view_mut().assign(view);
        buf
    }

    #[inline(always)]
    pub fn shape(&self) -> (usize, usize) {
        (self.rows, self.cols)
    }

    #[inline(always)]
    pub fn page_aligned(&self) -> bool {
        (self.pages.as_ptr() as usize).is_multiple_of(16384)
    }

    /// Shared read guard usable simultaneously by CPU `ArrayView2` and Metal
    /// `bytesNoCopy` GPU encoders.
    #[inline(always)]
    pub fn read_guard(&self) -> UnifiedRead<'_> {
        let len = self.rows * self.cols;
        // SAFETY: PageBlock is #[repr(C, align(16384))] wrapping [f64; 2048].
        let slice = unsafe {
            core::slice::from_raw_parts(self.pages.as_ptr().cast::<f64>(), self.pages.len() * 2048)
        };
        UnifiedRead {
            slice: &slice[..len],
            byte_len: self.pages.len() * 16384,
            rows: self.rows,
            cols: self.cols,
        }
    }

    /// Exclusive write guard statically preventing concurrent CPU/GPU reads.
    #[inline(always)]
    pub fn write_guard(&mut self) -> UnifiedWrite<'_> {
        let len = self.rows * self.cols;
        // SAFETY: PageBlock is #[repr(C, align(16384))] wrapping [f64; 2048].
        let slice = unsafe {
            core::slice::from_raw_parts_mut(
                self.pages.as_mut_ptr().cast::<f64>(),
                self.pages.len() * 2048,
            )
        };
        UnifiedWrite {
            slice: &mut slice[..len],
            rows: self.rows,
            cols: self.cols,
        }
    }
}

/// Shared immutable borrow of a [`UnifiedBuffer`].
#[derive(Clone, Copy)]
pub struct UnifiedRead<'a> {
    slice: &'a [f64],
    byte_len: usize,
    rows: usize,
    cols: usize,
}

impl<'a> UnifiedRead<'a> {
    #[inline(always)]
    pub fn view(&self) -> ArrayView2<'a, f64> {
        ArrayView2::from_shape((self.rows, self.cols), self.slice)
            .expect("UnifiedBuffer dimensions match slice length")
    }

    #[inline(always)]
    pub fn as_slice(&self) -> &'a [f64] {
        self.slice
    }

    /// Page-aligned pointer and page-multiple byte length suitable for
    /// zero-copy `MTLDevice::newBufferWithBytesNoCopy`.
    #[inline(always)]
    pub fn metal_nocopy(&self) -> (*const f64, usize) {
        (self.slice.as_ptr(), self.byte_len)
    }
}

/// Exclusive mutable borrow of a [`UnifiedBuffer`].
pub struct UnifiedWrite<'a> {
    slice: &'a mut [f64],
    rows: usize,
    cols: usize,
}

impl<'a> UnifiedWrite<'a> {
    #[inline(always)]
    pub fn view_mut(&mut self) -> ArrayViewMut2<'_, f64> {
        ArrayViewMut2::from_shape((self.rows, self.cols), self.slice)
            .expect("UnifiedBuffer dimensions match slice length")
    }

    #[inline(always)]
    pub fn slice_mut(&mut self) -> &mut [f64] {
        self.slice
    }
}

/// Magic header (`"BAUXSHM1"` little-endian) for cross-process shared-memory
/// slabs.
pub const SHM_MAGIC: u64 = 0x4241_5558_5348_4d31;

const PAGE_BYTES: usize = 16384;

#[inline]
const fn align_page(bytes: usize) -> usize {
    bytes.div_ceil(PAGE_BYTES) * PAGE_BYTES
}

#[inline]
fn put_u64(buf: &mut [u8], slot: usize, val: u64) {
    let off = slot * 8;
    buf[off..off + 8].copy_from_slice(&val.to_le_bytes());
}

/// Page-0 word slot indices for lock-free Rust <-> Julia atomic doorbells.
/// Request slots sit on cache line 2 (byte offset 128..152) and worker ACK
/// slots sit on cache line 4 (byte offset 256..288) to avoid false sharing on
/// both 64B (x86_64) and 128B (Apple Silicon) L1 cache lines.
pub const DBELL_CMD: usize = 16;
pub const DBELL_SEQ: usize = 17;
pub const DBELL_CNT: usize = 18;
pub const ACK_SEQ: usize = 32;
pub const ACK_STATUS: usize = 33;
pub const ACK_MICROS: usize = 34;
pub const ACK_THREADS: usize = 35;

pub const CMD_NOP: u64 = 0;
pub const CMD_EVAL: u64 = 1;
pub const CMD_DETACH: u64 = 2;
pub const CMD_SHUTDOWN: u64 = 3;
pub const CMD_SNAP: u64 = 4;

#[inline]
fn store_slot(ptr: *mut u64, val: u64) {
    // SAFETY: `ptr` points to an 8-byte aligned u64 slot inside page 0 of the
    // page-aligned `mmap`, valid for the lifetime of the slab.
    unsafe { core::sync::atomic::AtomicU64::from_ptr(ptr) }
        .store(val, core::sync::atomic::Ordering::Release);
}

#[inline]
fn load_slot(ptr: *const u64) -> u64 {
    // SAFETY: `ptr` points to an 8-byte aligned u64 slot inside page 0.
    unsafe { core::sync::atomic::AtomicU64::from_ptr(ptr.cast_mut()) }
        .load(core::sync::atomic::Ordering::Acquire)
}

#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn neon_half(src: &[u16], dst: &mut [f32], low: &[f64], span: &[f64]) {
    use core::arch::aarch64::*;
    let n = src.len();
    let mut j = 0usize;
    // SAFETY: Caller guarantees `src.len() == dst.len() == low.len() ==
    // span.len() == n`.
    unsafe {
        let mask_7fff = vdupq_n_u32(0x7fff);
        let mask_7c00 = vdupq_n_u32(0x7c00);
        let mask_sign = vdupq_n_u32(0x8000);
        let bias = vdupq_n_u32(0x1c000);
        let fzero = vdupq_n_f32(0.0);
        let fone = vdupq_n_f32(1.0);
        while j + 4 <= n {
            let h4 = vld1_u16(src.as_ptr().add(j));
            let w4 = vmovl_u16(h4);
            let mag = vandq_u32(w4, mask_7fff);
            let exp = vandq_u32(w4, mask_7c00);
            // Inf/NaN (`exp == 0x7c00`) or non-zero subnormal (`exp == 0 && mag
            // != 0`) falls back to scalar `decode_half` on those 4
            // lanes.
            let is_inf_nan = vceqq_u32(exp, mask_7c00);
            let is_subnormal = vbicq_u32(vmvnq_u32(vceqzq_u32(mag)), vmvnq_u32(vceqzq_u32(exp)));
            if vmaxvq_u32(vorrq_u32(is_inf_nan, is_subnormal)) != 0 {
                for k in j..j + 4 {
                    let u = f64::from(decode_half(src[k])).clamp(0.0, 1.0);
                    dst[k] = (low[k] + u * span[k]) as f32;
                }
                j += 4;
                continue;
            }
            let sign = vshlq_n_u32::<16>(vandq_u32(w4, mask_sign));
            let is_zero = vceqzq_u32(mag);
            let normal_bits = vorrq_u32(sign, vshlq_n_u32::<13>(vaddq_u32(mag, bias)));
            let f_bits = vbslq_u32(is_zero, sign, normal_bits);
            let u4 = vminq_f32(vmaxq_f32(vreinterpretq_f32_u32(f_bits), fzero), fone);

            let u_lo = vcvt_f64_f32(vget_low_f32(u4));
            let u_hi = vcvt_f64_f32(vget_high_f32(u4));
            let y_lo = vfmaq_f64(
                vld1q_f64(low.as_ptr().add(j)),
                u_lo,
                vld1q_f64(span.as_ptr().add(j)),
            );
            let y_hi = vfmaq_f64(
                vld1q_f64(low.as_ptr().add(j + 2)),
                u_hi,
                vld1q_f64(span.as_ptr().add(j + 2)),
            );
            vst1q_f32(
                dst.as_mut_ptr().add(j),
                vcombine_f32(vcvt_f32_f64(y_lo), vcvt_f32_f64(y_hi)),
            );
            j += 4;
        }
    }
    while j < n {
        let u = f64::from(decode_half(src[j])).clamp(0.0, 1.0);
        dst[j] = (low[j] + u * span[j]) as f32;
        j += 1;
    }
}

/// Page-aligned memory-mapped IPC slab shared zero-copy with external physics
/// workers (such as Julia `Mmap.mmap` evaluators for `Oceananigans.jl` and
/// `FUSE.jl`). Uses cache-line-isolated atomic doorbells in page 0.
pub struct ShmSlab {
    path: std::path::PathBuf,
    mmap: memmap2::MmapMut,
    max_batch: usize,
    num_dim: usize,
    num_obj: usize,
    state_len: usize,
    cand_off: usize,
    score_off: usize,
    viol_off: usize,
    state_off: usize,
}

#[inline]
fn decode_half(bits: u16) -> f32 {
    let sign = (u32::from(bits & 0x8000)) << 16;
    let exp = u32::from((bits >> 10) & 0x1f);
    let mant = u32::from(bits & 0x03ff);
    if exp == 0 {
        if mant == 0 {
            f32::from_bits(sign)
        } else {
            let val = (mant as f32) * (1.0 / 16_777_216.0);
            if sign != 0 { -val } else { val }
        }
    } else if exp == 0x1f {
        if mant == 0 {
            f32::from_bits(sign | 0x7f80_0000)
        } else {
            f32::NAN
        }
    } else {
        f32::from_bits(sign | ((exp + 112) << 23) | (mant << 13))
    }
}

impl ShmSlab {
    /// Allocate and map a single-objective shared-memory slab on disk/tmpfs.
    pub fn create(
        path: std::path::PathBuf,
        max_batch: usize,
        num_dim: usize,
        state_len: usize,
    ) -> Result<Self, String> {
        Self::create_multi(path, max_batch, num_dim, 1, state_len)
    }

    /// Allocate and map a multi-objective shared-memory slab (`num_obj >= 1`).
    pub fn create_multi(
        path: std::path::PathBuf,
        max_batch: usize,
        num_dim: usize,
        num_obj: usize,
        state_len: usize,
    ) -> Result<Self, String> {
        if max_batch == 0 || num_dim == 0 || num_obj == 0 {
            return Err("ShmSlab requires positive max_batch, num_dim, and num_obj".into());
        }
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(|e| format!("create shm dir: {e}"))?;
        }
        let cand_bytes = max_batch
            .checked_mul(num_dim)
            .and_then(|n| n.checked_mul(4))
            .ok_or("candidate byte size overflow")?;
        let score_bytes = max_batch
            .checked_mul(num_obj)
            .and_then(|n| n.checked_mul(8))
            .ok_or("score byte size overflow")?;
        let cand_off = PAGE_BYTES;
        let score_off = align_page(cand_off + cand_bytes);
        let viol_off = score_off + score_bytes;
        let state_off = align_page(viol_off + max_batch * 8);
        let total = align_page(state_off + state_len * 4);

        let file = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(&path)
            .map_err(|e| format!("open shm {}: {e}", path.display()))?;
        file.set_len(total as u64)
            .map_err(|e| format!("resize shm {}: {e}", path.display()))?;
        // SAFETY: Exclusively created file sized to `total`; coordinated via
        // explicit pipe doorbell.
        let mut mmap =
            unsafe { memmap2::MmapMut::map_mut(&file) }.map_err(|e| format!("mmap: {e}"))?;

        put_u64(&mut mmap, 0, SHM_MAGIC);
        put_u64(&mut mmap, 1, 1);
        put_u64(&mut mmap, 2, max_batch as u64);
        put_u64(&mut mmap, 3, num_dim as u64);
        put_u64(&mut mmap, 4, state_len as u64);
        put_u64(&mut mmap, 5, cand_off as u64);
        put_u64(&mut mmap, 6, score_off as u64);
        put_u64(&mut mmap, 7, viol_off as u64);
        put_u64(&mut mmap, 8, state_off as u64);
        put_u64(&mut mmap, 9, num_obj as u64);

        let mut slab = Self {
            path,
            mmap,
            max_batch,
            num_dim,
            num_obj,
            state_len,
            cand_off,
            score_off,
            viol_off,
            state_off,
        };
        slab.reset_doorbell();
        Ok(slab)
    }

    #[inline]
    fn slot_mut(&mut self, slot: usize) -> *mut u64 {
        debug_assert!(slot * 8 + 8 <= PAGE_BYTES);
        // SAFETY: `slot * 8 + 8 <= PAGE_BYTES` lies within mapped page 0.
        unsafe { self.mmap.as_mut_ptr().add(slot * 8).cast::<u64>() }
    }

    #[inline]
    fn slot_ptr(&self, slot: usize) -> *const u64 {
        debug_assert!(slot * 8 + 8 <= PAGE_BYTES);
        // SAFETY: `slot * 8 + 8 <= PAGE_BYTES` lies within mapped page 0.
        unsafe { self.mmap.as_ptr().add(slot * 8).cast::<u64>() }
    }

    /// Zero all request/response atomic doorbell slots and publish with Release
    /// ordering.
    pub fn reset_doorbell(&mut self) {
        for slot in [
            DBELL_CMD,
            DBELL_SEQ,
            DBELL_CNT,
            ACK_SEQ,
            ACK_STATUS,
            ACK_MICROS,
            ACK_THREADS,
        ] {
            store_slot(self.slot_mut(slot), 0);
        }
    }

    /// Publish a lock-free atomic doorbell request (`cmd`, `count`) and return
    /// the monotonic request sequence number.
    pub fn doorbell_signal(&mut self, cmd: u64, count: usize) -> u64 {
        let next_seq = load_slot(self.slot_ptr(DBELL_SEQ)).wrapping_add(1).max(1);
        store_slot(self.slot_mut(DBELL_CMD), cmd);
        store_slot(self.slot_mut(DBELL_CNT), count as u64);
        store_slot(self.slot_mut(DBELL_SEQ), next_seq);
        next_seq
    }

    /// Read `(ack_seq, status, worker_us, worker_threads)` from worker cache
    /// line.
    #[inline]
    pub fn poll_ack(&self) -> (u64, u64, u64, u64) {
        let ack_seq = load_slot(self.slot_ptr(ACK_SEQ));
        let status = load_slot(self.slot_ptr(ACK_STATUS));
        let us = load_slot(self.slot_ptr(ACK_MICROS));
        let threads = load_slot(self.slot_ptr(ACK_THREADS));
        (ack_seq, status, us, threads)
    }

    /// Signal the worker atomically and wait with adaptive spin -> yield ->
    /// sub-millisecond backoff until ACK matches, invoking `is_alive` during
    /// waits. Returns `(worker_us, worker_threads, round_trip_ns)`.
    pub fn doorbell_wait<F>(
        &mut self,
        cmd: u64,
        count: usize,
        timeout: std::time::Duration,
        mut is_alive: F,
    ) -> Result<(u64, u64, u64), String>
    where
        F: FnMut() -> Result<(), String>,
    {
        let start = std::time::Instant::now();
        let seq = self.doorbell_signal(cmd, count);
        let mut iter = 0u32;
        let mut backoff = std::time::Duration::from_micros(50);
        let max_backoff = std::time::Duration::from_micros(500);
        loop {
            let ack_seq = load_slot(self.slot_ptr(ACK_SEQ));
            if ack_seq >= seq {
                let status = load_slot(self.slot_ptr(ACK_STATUS));
                let us = load_slot(self.slot_ptr(ACK_MICROS));
                let threads = load_slot(self.slot_ptr(ACK_THREADS));
                let rt_ns = start.elapsed().as_nanos().min(u128::from(u64::MAX)) as u64;
                if status != 0 {
                    return Err(format!("shm worker returned non-zero status {status}"));
                }
                return Ok((us, threads, rt_ns));
            }
            if iter < 256 {
                core::hint::spin_loop();
            } else if iter < 1024 {
                std::thread::yield_now();
            } else {
                is_alive()?;
                if start.elapsed() > timeout {
                    return Err(format!(
                        "shm doorbell timeout after {:?} waiting for seq {seq} (last ack {ack_seq})",
                        start.elapsed()
                    ));
                }
                std::thread::sleep(backoff);
                backoff = (backoff * 2).min(max_backoff);
            }
            iter = iter.saturating_add(1);
        }
    }

    #[inline]
    pub fn path(&self) -> &std::path::Path {
        &self.path
    }

    #[inline]
    pub fn max_batch(&self) -> usize {
        self.max_batch
    }

    #[inline]
    pub fn num_dim(&self) -> usize {
        self.num_dim
    }

    #[inline]
    pub fn num_obj(&self) -> usize {
        self.num_obj
    }

    #[inline]
    pub fn state_len(&self) -> usize {
        self.state_len
    }

    /// Scale unit-cube proposals `[0, 1]^D` directly into mapped `f32`
    /// candidate memory (`low[j] + u * span[j]`) without intermediate heap
    /// allocation.
    pub fn write_unit(
        &mut self,
        unit: &ArrayView2<f64>,
        low: &[f64],
        span: &[f64],
    ) -> Result<usize, String> {
        let count = unit.nrows();
        if count == 0 || count > self.max_batch {
            return Err(format!(
                "batch count {count} outside 1..={}",
                self.max_batch
            ));
        }
        if unit.ncols() != self.num_dim || low.len() != self.num_dim || span.len() != self.num_dim {
            return Err("dimension mismatch in ShmSlab::write_unit".into());
        }
        let elems = count * self.num_dim;
        // SAFETY: `cand_off` is 16 KB page-aligned and sized for `max_batch *
        // num_dim` f32s.
        let dst = unsafe {
            core::slice::from_raw_parts_mut(
                self.mmap.as_mut_ptr().add(self.cand_off).cast::<f32>(),
                elems,
            )
        };
        if let Some(u_slice) = unit.as_slice() {
            for b in 0..count {
                let src_row = &u_slice[b * self.num_dim..(b + 1) * self.num_dim];
                let dst_row = &mut dst[b * self.num_dim..(b + 1) * self.num_dim];
                for j in 0..self.num_dim {
                    dst_row[j] = (low[j] + src_row[j] * span[j]) as f32;
                }
            }
        } else {
            for b in 0..count {
                let dst_row = &mut dst[b * self.num_dim..(b + 1) * self.num_dim];
                for j in 0..self.num_dim {
                    dst_row[j] = (low[j] + unit[[b, j]] * span[j]) as f32;
                }
            }
        }
        core::sync::atomic::fence(core::sync::atomic::Ordering::Release);
        Ok(count)
    }

    /// Copy raw `f32` candidate weights directly into the mapped candidate
    /// region.
    pub fn write_f32(&mut self, batch: &[f32], count: usize) -> Result<(), String> {
        let elems = count * self.num_dim;
        if count == 0 || count > self.max_batch || batch.len() != elems {
            return Err("invalid batch shape in ShmSlab::write_f32".into());
        }
        // SAFETY: `cand_off` is 16 KB page-aligned and sized for `max_batch *
        // num_dim` f32s.
        let dst = unsafe {
            core::slice::from_raw_parts_mut(
                self.mmap.as_mut_ptr().add(self.cand_off).cast::<f32>(),
                elems,
            )
        };
        dst.copy_from_slice(batch);
        core::sync::atomic::fence(core::sync::atomic::Ordering::Release);
        Ok(())
    }

    /// Decode an IEEE FP16 (`u16`) unified-memory row in `[0, 1]^D` directly
    /// into mapped `f32` candidate slot `slot` (`low[j] + clamp(u) * span[j]`)
    /// with zero heap allocation.
    ///
    /// # Safety
    /// `src` must point to at least `self.num_dim` readable `u16` elements.
    pub unsafe fn write_half(
        &mut self,
        slot: usize,
        src: *const u16,
        low: &[f64],
        span: &[f64],
    ) -> Result<(), String> {
        if slot >= self.max_batch || low.len() != self.num_dim || span.len() != self.num_dim {
            return Err("invalid slot or bounds in ShmSlab::write_half".into());
        }
        // SAFETY: `cand_off` holds `max_batch * num_dim` f32s; caller
        // guarantees `src` holds `num_dim` u16s.
        let (src_row, dst_row) = unsafe {
            (
                core::slice::from_raw_parts(src, self.num_dim),
                core::slice::from_raw_parts_mut(
                    self.mmap
                        .as_mut_ptr()
                        .add(self.cand_off)
                        .cast::<f32>()
                        .add(slot * self.num_dim),
                    self.num_dim,
                ),
            )
        };
        #[cfg(target_arch = "aarch64")]
        // SAFETY: `src_row` and `dst_row` both have length `self.num_dim`, matching `low` and
        // `span`.
        unsafe {
            neon_half(src_row, dst_row, low, span);
        }
        #[cfg(not(target_arch = "aarch64"))]
        for j in 0..self.num_dim {
            let u = f64::from(decode_half(src_row[j])).clamp(0.0, 1.0);
            dst_row[j] = (low[j] + u * span[j]) as f32;
        }
        core::sync::atomic::fence(core::sync::atomic::Ordering::Release);
        Ok(())
    }

    /// Zero-copy slice of `count` scalar objective scores written by the
    /// worker (column-major `(num_obj, count)` flattened).
    #[inline]
    pub fn read_scores(&self, count: usize) -> &[f64] {
        core::sync::atomic::fence(core::sync::atomic::Ordering::Acquire);
        let n = count.min(self.max_batch) * self.num_obj;
        // SAFETY: `score_off` is 16 KB page-aligned and sized for
        // `max_batch * num_obj` f64s.
        unsafe {
            core::slice::from_raw_parts(self.mmap.as_ptr().add(self.score_off).cast::<f64>(), n)
        }
    }

    /// Zero-copy slice of `count` scalar constraint violations written by the
    /// worker.
    #[inline]
    pub fn read_viols(&self, count: usize) -> &[f64] {
        core::sync::atomic::fence(core::sync::atomic::Ordering::Acquire);
        let n = count.min(self.max_batch);
        // SAFETY: `viol_off` is 8-byte aligned and sized for `max_batch` f64s.
        unsafe {
            core::slice::from_raw_parts(self.mmap.as_ptr().add(self.viol_off).cast::<f64>(), n)
        }
    }

    /// Zero-copy slice of the optional physical state snapshot written by the
    /// worker.
    #[inline]
    pub fn read_state(&self) -> &[f32] {
        core::sync::atomic::fence(core::sync::atomic::Ordering::Acquire);
        // SAFETY: `state_off` is 16 KB page-aligned and sized for `state_len`
        // f32s.
        unsafe {
            core::slice::from_raw_parts(
                self.mmap.as_ptr().add(self.state_off).cast::<f32>(),
                self.state_len,
            )
        }
    }
}

impl Drop for ShmSlab {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.path);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn unified_pages() {
        let src = array![[1.0, 2.0], [3.0, 4.0]];
        let buf = UnifiedBuffer::from_view(&src.view());
        assert!(buf.page_aligned());
        let rd = buf.read_guard();
        assert_eq!(rd.view(), src.view());
        let (ptr, bytes) = rd.metal_nocopy();
        assert_eq!((ptr as usize) % 16384, 0);
        assert_eq!(bytes % 16384, 0);
    }

    #[test]
    fn shm_slab() {
        let path = std::env::temp_dir().join(format!("baux_shm_test_{}.mmap", std::process::id()));
        {
            let mut slab = ShmSlab::create(path.clone(), 4, 3, 8).expect("create slab");
            assert!(path.exists());
            let unit = array![[0.0, 0.5, 1.0], [1.0, 0.25, 0.0]];
            let n = slab
                .write_unit(&unit.view(), &[-2.0, 0.0, 10.0], &[4.0, 8.0, 20.0])
                .expect("write unit");
            assert_eq!(n, 2);
            let cands = unsafe {
                core::slice::from_raw_parts(slab.mmap.as_ptr().add(slab.cand_off).cast::<f32>(), 6)
            };
            assert_eq!(cands, &[-2.0f32, 4.0, 30.0, 2.0, 2.0, 10.0]);
            assert_eq!(slab.read_scores(2), &[0.0, 0.0]);
            assert_eq!(slab.read_viols(2), &[0.0, 0.0]);
            assert_eq!(slab.read_state().len(), 8);

            let half_row: [u16; 3] = [0x0000, 0x3800, 0x3c00];
            unsafe {
                slab.write_half(0, half_row.as_ptr(), &[-2.0, 0.0, 10.0], &[4.0, 8.0, 20.0])
                    .expect("write half");
            }
            let cands0 = unsafe {
                core::slice::from_raw_parts(slab.mmap.as_ptr().add(slab.cand_off).cast::<f32>(), 3)
            };
            assert_eq!(cands0, &[-2.0f32, 4.0, 30.0]);

            // Verify NEON/scalar parity across normal, negative, subnormal, >1,
            // Inf, NaN, odd D=7.
            let mut slab7 = ShmSlab::create(
                std::env::temp_dir().join(format!("baux_shm_neon_{}.mmap", std::process::id())),
                2,
                7,
                0,
            )
            .expect("create slab7");
            let halfs7: [u16; 7] = [0x0000, 0x3400, 0x3800, 0x3e00, 0xbc00, 0x0001, 0x7c00];
            let low7 = [-1.0, 0.0, 2.0, -5.0, 10.0, -3.0, 100.0];
            let span7 = [4.0, 8.0, 6.0, 10.0, 2.0, 7.0, 50.0];
            unsafe {
                slab7
                    .write_half(1, halfs7.as_ptr(), &low7, &span7)
                    .expect("write half 7");
            }
            let got7 = unsafe {
                core::slice::from_raw_parts(
                    slab7.mmap.as_ptr().add(slab7.cand_off).cast::<f32>().add(7),
                    7,
                )
            };
            for j in 0..7 {
                let u = f64::from(decode_half(halfs7[j])).clamp(0.0, 1.0);
                let expected = (low7[j] + u * span7[j]) as f32;
                assert!(
                    (got7[j] - expected).abs() <= 1e-5,
                    "lane {j}: got {} expected {expected}",
                    got7[j]
                );
            }

            // Verify lock-free atomic doorbell round-trip.
            let seq = slab.doorbell_signal(CMD_EVAL, 2);
            assert_eq!(seq, 1);
            assert_eq!(load_slot(slab.slot_ptr(DBELL_CMD)), CMD_EVAL);
            assert_eq!(load_slot(slab.slot_ptr(DBELL_CNT)), 2);
            store_slot(slab.slot_mut(ACK_STATUS), 0);
            store_slot(slab.slot_mut(ACK_MICROS), 17);
            store_slot(slab.slot_mut(ACK_THREADS), 4);
            store_slot(slab.slot_mut(ACK_SEQ), seq);
            assert_eq!(slab.poll_ack(), (1, 0, 17, 4));
        }
        assert!(!path.exists());
    }

    #[test]
    fn shm_doorbell() {
        let path = std::env::temp_dir().join(format!("baux_doorbell_{}.mmap", std::process::id()));
        let mut slab = ShmSlab::create(path.clone(), 4, 2, 0).expect("create slab");
        let slab_raw = slab.mmap.as_ptr() as usize;
        let handle = std::thread::spawn(move || {
            let base = slab_raw as *mut u64;
            loop {
                // SAFETY: `base` points into `slab.mmap` kept alive until
                // `handle.join()`.
                let req_seq = unsafe { load_slot(base.add(DBELL_SEQ)) };
                if req_seq >= 1 {
                    unsafe {
                        store_slot(base.add(ACK_STATUS), 0);
                        store_slot(base.add(ACK_MICROS), 12);
                        store_slot(base.add(ACK_THREADS), 8);
                        store_slot(base.add(ACK_SEQ), req_seq);
                    }
                    break;
                }
                std::thread::yield_now();
            }
        });
        let (us, threads, _rt) = slab
            .doorbell_wait(CMD_EVAL, 3, std::time::Duration::from_secs(2), || Ok(()))
            .expect("cross-thread doorbell ack");
        handle.join().expect("join");
        assert_eq!((us, threads), (12, 8));
    }
}
