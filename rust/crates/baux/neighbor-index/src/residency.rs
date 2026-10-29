//! Page-residency budget tracker and advisory release policy for memory-mapped
//! stores.

use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};

/// General page-residency tracking and budget enforcement contract.
pub trait ResidencyPolicy: Send + Sync {
    fn budget(&self) -> Option<usize>;
    fn set_budget(&self, bytes: Option<usize>);
    fn estimate(&self) -> usize;
    fn touch(&self, bytes: &[u8], offset: usize, len: usize);
    fn release(&self, bytes: &[u8], offset: usize, len: usize);
    fn forget(&self, bytes: &[u8]);
}

/// Lock-free bitmap page-residency tracker with configurable resident-byte
/// budget.
pub struct Residency {
    budget: AtomicUsize,
    tracking: AtomicBool,
    pages: Vec<AtomicU64>,
    resident: AtomicUsize,
    page_bytes: usize,
}

impl Residency {
    pub fn new(mapped_len: usize, budget: Option<usize>) -> Self {
        let page_bytes = os_page();
        let num_pages = mapped_len.div_ceil(page_bytes);
        let words = num_pages.div_ceil(64);
        let pages = (0..words).map(|_| AtomicU64::new(0)).collect();
        Self {
            budget: AtomicUsize::new(budget.unwrap_or(usize::MAX)),
            tracking: AtomicBool::new(budget.is_some()),
            pages,
            resident: AtomicUsize::new(0),
            page_bytes,
        }
    }

    pub fn tracking(&self) -> bool {
        self.tracking.load(Ordering::Relaxed)
    }

    pub fn update_tracking(&self, pinned_bytes: usize) {
        let raw = self.budget.load(Ordering::Relaxed);
        let on = raw != usize::MAX && raw > pinned_bytes;
        if !on && self.tracking.swap(false, Ordering::Relaxed) {
            self.clear_bitmap();
        } else if on {
            self.tracking.store(true, Ordering::Relaxed);
        }
    }

    pub fn copy_released(&self, bytes: &[u8], offset: usize, dst: &mut [u8]) {
        let len = dst.len();
        dst.copy_from_slice(&bytes[offset..offset + len]);
        match self.budget() {
            Some(0) => self.release(bytes, offset, len),
            Some(_) => self.touch(bytes, offset, len),
            None => {}
        }
    }

    fn clear_bitmap(&self) {
        for word in &self.pages {
            word.store(0, Ordering::Relaxed);
        }
        self.resident.store(0, Ordering::Relaxed);
    }

    fn page_span(&self, total_len: usize, offset: usize, len: usize) -> Option<(usize, usize)> {
        if len == 0 || offset >= total_len {
            return None;
        }
        let end = offset.saturating_add(len).min(total_len);
        let first = offset / self.page_bytes;
        let last = (end - 1) / self.page_bytes;
        Some((first, last))
    }
}

impl ResidencyPolicy for Residency {
    fn budget(&self) -> Option<usize> {
        let raw = self.budget.load(Ordering::Relaxed);
        (raw != usize::MAX).then_some(raw)
    }

    fn set_budget(&self, bytes: Option<usize>) {
        self.budget
            .store(bytes.unwrap_or(usize::MAX), Ordering::Relaxed);
        self.tracking.store(bytes.is_some(), Ordering::Relaxed);
        if bytes.is_none() {
            self.clear_bitmap();
        }
    }

    fn estimate(&self) -> usize {
        self.resident.load(Ordering::Relaxed)
    }

    fn touch(&self, bytes: &[u8], offset: usize, len: usize) {
        if !self.tracking() {
            return;
        }
        let Some(limit) = self.budget() else {
            return;
        };
        if limit == 0 {
            self.release(bytes, offset, len);
            return;
        }
        let Some((first, last)) = self.page_span(bytes.len(), offset, len) else {
            return;
        };
        let mut added = 0usize;
        for page in first..=last {
            let word = page / 64;
            let bit = 1u64 << (page % 64);
            if let Some(slot) = self.pages.get(word) {
                if slot.fetch_or(bit, Ordering::Relaxed) & bit == 0 {
                    added += self.page_bytes;
                }
            }
        }
        if added > 0 {
            let now = self.resident.fetch_add(added, Ordering::Relaxed) + added;
            if now > limit {
                self.forget(bytes);
            }
        }
    }

    fn release(&self, bytes: &[u8], offset: usize, len: usize) {
        let Some((first, last)) = self.page_span(bytes.len(), offset, len) else {
            return;
        };
        let start = first * self.page_bytes;
        let end = ((last + 1) * self.page_bytes).min(bytes.len());
        advise_free(bytes, start, end.saturating_sub(start));
        let mut removed = 0usize;
        for page in first..=last {
            let word = page / 64;
            let bit = 1u64 << (page % 64);
            if let Some(slot) = self.pages.get(word) {
                if slot.fetch_and(!bit, Ordering::Relaxed) & bit != 0 {
                    removed += self.page_bytes;
                }
            }
        }
        if removed > 0 {
            let mut cur = self.resident.load(Ordering::Relaxed);
            while let Err(actual) = self.resident.compare_exchange_weak(
                cur,
                cur.saturating_sub(removed),
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                cur = actual;
            }
        }
    }

    fn forget(&self, bytes: &[u8]) {
        advise_free(bytes, 0, bytes.len());
        self.clear_bitmap();
    }
}

#[cfg(unix)]
unsafe extern "C" {
    fn sysconf(name: i32) -> isize;
    fn madvise(addr: *mut core::ffi::c_void, len: usize, advice: i32) -> i32;
}

fn os_page() -> usize {
    static PAGE: OnceLock<usize> = OnceLock::new();
    *PAGE.get_or_init(|| {
        #[cfg(unix)]
        {
            #[cfg(any(target_os = "macos", target_os = "ios"))]
            const SC_PAGESIZE: i32 = 29;
            #[cfg(not(any(target_os = "macos", target_os = "ios")))]
            const SC_PAGESIZE: i32 = 30;
            let n = unsafe { sysconf(SC_PAGESIZE) };
            if n > 0 { n as usize } else { 4096 }
        }
        #[cfg(not(unix))]
        {
            4096
        }
    })
}

#[cfg(unix)]
fn advise_free(bytes: &[u8], offset: usize, len: usize) {
    if len == 0 || offset >= bytes.len() {
        return;
    }
    let ptr = unsafe { bytes.as_ptr().add(offset) } as *mut core::ffi::c_void;
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    const ADVICE_FREE: i32 = 7; // MADV_FREE_REUSABLE
    #[cfg(not(any(target_os = "macos", target_os = "ios")))]
    const ADVICE_FREE: i32 = 4; // MADV_DONTNEED
    unsafe {
        let _ = madvise(ptr, len.min(bytes.len() - offset), ADVICE_FREE);
    }
}

#[cfg(not(unix))]
fn advise_free(_bytes: &[u8], _offset: usize, _len: usize) {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn residency_budget() {
        let buf = vec![7u8; 64 * 1024];
        let res = Residency::new(buf.len(), Some(32 * 1024));
        assert!(res.tracking());
        assert_eq!(res.budget(), Some(32 * 1024));
        res.touch(&buf, 0, 256);
        assert!(res.estimate() >= 4096);
        res.release(&buf, 0, 256);
        assert_eq!(res.estimate(), 0);
        let mut dst = [0u8; 16];
        res.copy_released(&buf, 128, &mut dst);
        assert_eq!(dst, [7u8; 16]);
        res.set_budget(None);
        assert!(!res.tracking());
        assert_eq!(res.estimate(), 0);
    }
}
