//! Native TVM artifacts use the same buffers, validation and invocation queue.
use super::*;

impl Executable {
    /// Load a trusted TVM shared library for CPU execution. The runtime retains
    /// the loaded artifact for every executable, buffer and queued
    /// invocation.
    pub fn load_tvm(
        library: &str,
        entry: &str,
        device: &str,
        inputs: Vec<Tensor>,
        outputs: Vec<Tensor>,
    ) -> Result<Self> {
        for spec in inputs.iter().chain(&outputs) {
            layout(spec)?;
        }
        let library = CString::new(library).map_err(|e| e.to_string())?;
        let entry = CString::new(entry).map_err(|e| e.to_string())?;
        let device = CString::new(device).map_err(|e| e.to_string())?;
        let mut pointer = std::ptr::null_mut();
        let mut error = [0; 2048];
        // All arguments stay live for the call. On success the bridge returns
        // one owning handle; on failure it cleans up every intermediate.
        let code = unsafe {
            raw::bauxtvm_load(
                library.as_ptr(),
                entry.as_ptr(),
                device.as_ptr(),
                &mut pointer,
                error.as_mut_ptr(),
                error.len(),
            )
        };
        raw::status(code, &error)?;
        Ok(Self {
            owner: Arc::new(Session(Mutex::new(raw::Context(pointer, &raw::TVM)))),
            inputs: inputs.into(),
            outputs: outputs.into(),
            queue: pending::Queue::new()?,
            kernel: None,
        })
    }
}
