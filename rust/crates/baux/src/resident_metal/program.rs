//! Mutation leases between resident search and portable program execution.
use super::*;
use crate::portable::{Buffer as TensorBuffer, Executable, Tensor};
use metal::foreign_types::ForeignTypeRef;
use std::sync::Arc;

impl SearchState {
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
        self.lease_row(program, &*row)
    }

    /// Read-only view of the current row, before beginning another ask.
    pub fn program_base(&self, program: &Executable) -> Result<TensorBuffer, String> {
        self.check_idle()?;
        self.lease_row(program, &*self.base)
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
        let lease: Arc<dyn std::any::Any + Send + Sync> = Arc::new(());
        // All search producers complete before publication; the lease blocks
        // every managed mutation until invocations and derived views release
        // it.
        unsafe { program.import_metal(spec, row.as_ptr().cast(), 0, lease) }
    }
}
