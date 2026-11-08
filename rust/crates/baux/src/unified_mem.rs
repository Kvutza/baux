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
}
