//! Mutation leases between resident search and portable program execution.
use super::*;
use crate::portable::{Buffer as TensorBuffer, Executable, Tensor};
use metal::foreign_types::ForeignTypeRef;
use std::sync::atomic::{AtomicUsize, Ordering};

struct Lease(Arc<AtomicUsize>);
impl Drop for Lease {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::AcqRel);
    }
}

impl SearchState {
    pub(super) fn check_unleased(&self) -> Result<(), String> {
        if self.leases.load(Ordering::Acquire) != 0 {
            return Err("resident parameter views are still leased by a program".into());
        }
        Ok(())
    }

    /// Publish the completed selected row to a native program without copying.
    /// Tell, cancellation and subsequent asks reject mutation until all queued
    /// users and derived outputs release their views. Dropping the search
    /// itself remains safe: IREE retains the underlying Metal allocation.
    pub fn program_candidate(
        &self,
        program: &Executable,
        proposal: &Proposals,
    ) -> Result<TensorBuffer, String> {
        self.check_round(proposal)?;
        let row = self.propose_buffer(proposal)?;
        self.lease_row(program, &row)
    }

    /// Read-only view of the current row, before beginning another ask.
    pub fn program_base(&self, program: &Executable) -> Result<TensorBuffer, String> {
        self.check_idle()?;
        self.lease_row(program, &self.base)
    }

    fn lease_row(
        &self,
        program: &Executable,
        row: &metal::BufferRef,
    ) -> Result<TensorBuffer, String> {
        let dtype = if self.independent_fp16 {
            "float16"
        } else {
            "bfloat16"
        };
        let spec = Tensor::new(vec![self.dimensions], dtype)?;
        self.leases.fetch_add(1, Ordering::AcqRel);
        let lease = Arc::new(Lease(self.leases.clone()));
        // All search producers complete before publication; the lease blocks
        // every managed mutation until invocations and derived views release
        // it.
        unsafe { program.import_metal(spec, row.as_ptr().cast(), 0, lease) }
    }
}
