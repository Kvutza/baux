//! Native-kernel interoperability using retained IREE/Metal allocations.
use super::*;
use std::any::Any;

impl Executable {
    /// Wrap existing device kernels in the same checked call/queue interface.
    /// State remains explicit in input/output buffers. The callback must finish
    /// its GPU writes before returning and publish immutable result
    /// allocations.
    pub fn native(
        device: &str,
        inputs: Vec<Tensor>,
        outputs: Vec<Tensor>,
        kernel: impl Fn(&Executable, &[Buffer]) -> Result<Vec<Buffer>> + Send + Sync + 'static,
    ) -> Result<Self> {
        for spec in inputs.iter().chain(&outputs) {
            layout(spec)?;
        }
        let driver = match device {
            "cpu" => "local-task",
            "metal" => "metal",
            _ => return Err(format!("unsupported native device {device:?}")),
        };
        let driver = CString::new(driver).map_err(|e| e.to_string())?;
        let mut context = std::ptr::null_mut();
        let mut error = [0; 2048];
        // C owns the returned device context and retains no borrowed strings.
        let code = unsafe {
            raw::bauxiree_open(
                driver.as_ptr(),
                &mut context,
                error.as_mut_ptr(),
                error.len(),
            )
        };
        raw::status(code, &error)?;
        Ok(Self {
            owner: Arc::new(Session(Mutex::new(raw::Context(context, &raw::IREE)))),
            inputs: inputs.into(),
            outputs: outputs.into(),
            queue: pending::Queue::new()?,
            kernel: Some(Arc::new(kernel)),
        })
    }

    /// Import an existing Metal tensor without copying its contents.
    ///
    /// # Safety
    /// `handle` must be a live MTLBuffer on this device. All producer writes
    /// must have completed. `lease` must keep every mutable alias from
    /// writing until the last imported/derived buffer and invocation
    /// releases it. The runtime retains MTLBuffer itself; a retained
    /// pointer alone is not a mutation lease.
    pub unsafe fn import_metal(
        &self,
        spec: Tensor,
        handle: *mut c_void,
        offset: usize,
        lease: Arc<dyn Any + Send + Sync>,
    ) -> Result<Buffer> {
        let (dtype, length) = layout(&spec)?;
        let shape: Vec<usize> = spec.shape.clone();
        let context = self.owner.0.lock().map_err(|_| "session mutex poisoned")?;
        if !context.1.metal {
            return Err("runtime does not support Metal imports".into());
        }
        let mut pointer = std::ptr::null_mut();
        let mut error = [0; 2048];
        // Caller supplies a live, synchronized, leased MTLBuffer. C validates
        // device and bounds and returns one owning buffer-view reference.
        let code = unsafe {
            raw::bauxiree_import(
                context.0,
                handle,
                offset,
                length,
                shape.as_ptr(),
                shape.len(),
                dtype,
                &mut pointer,
                error.as_mut_ptr(),
                error.len(),
            )
        };
        raw::status(code, &error)?;
        drop(context);
        Ok(Buffer(Arc::new(Storage {
            owner: self.owner.clone(),
            pointer,
            spec,
            leases: vec![lease],
        })))
    }

    pub(super) fn check_outputs(&self, outputs: &[Buffer]) -> Result<()> {
        if outputs.len() != self.outputs.len() {
            return Err("native output count differs from signature".into());
        }
        for (value, spec) in outputs.iter().zip(self.outputs.iter()) {
            if !Arc::ptr_eq(&self.owner, &value.0.owner) || value.spec() != spec {
                return Err("native output owner or tensor differs from signature".into());
            }
        }
        Ok(())
    }

    // A compiled/native result may alias any input. Carry external mutation
    // leases through all results, including pass-throughs, without retaining
    // complete chains of input allocations. Deduplicate by Arc identity.
    pub(super) fn inherit(&self, inputs: &[Buffer], outputs: &mut [Buffer]) {
        for output in outputs {
            let mut leases = output.0.leases.clone();
            for lease in inputs.iter().flat_map(|v| &v.0.leases) {
                if !leases.iter().any(|other| Arc::ptr_eq(other, lease)) {
                    leases.push(lease.clone());
                }
            }
            if leases.len() == output.0.leases.len() {
                continue;
            }
            // Each new Storage must own a retained native view reference.
            let context = self.owner.0.lock().unwrap_or_else(|e| e.into_inner());
            unsafe { (context.1.retain)(output.0.pointer) };
            let replacement = Buffer(Arc::new(Storage {
                owner: self.owner.clone(),
                pointer: output.0.pointer,
                spec: output.spec().clone(),
                leases,
            }));
            drop(context);
            *output = replacement;
        }
    }
}

impl Buffer {
    /// Borrow the underlying MTLBuffer and its byte offset. No transfer occurs.
    ///
    /// # Safety
    /// The handle must not outlive this buffer, be mutated, or be consumed by
    /// GPU work that outlives it. Native kernels must retain inputs until all
    /// submitted commands complete. The caller must not release this handle.
    pub unsafe fn metal(&self) -> Result<(*mut c_void, u64)> {
        let context = self
            .0
            .owner
            .0
            .lock()
            .map_err(|_| "session mutex poisoned")?;
        if !context.1.metal {
            return Err("runtime does not expose Metal allocations".into());
        }
        let mut handle = std::ptr::null_mut();
        let mut offset = 0;
        let mut error = [0; 2048];
        let code = unsafe {
            raw::bauxiree_metal(
                context.0,
                self.0.pointer,
                &mut handle,
                &mut offset,
                error.as_mut_ptr(),
                error.len(),
            )
        };
        raw::status(code, &error)?;
        Ok((handle, offset))
    }
}

impl Executable {
    /// Retain a completed tensor from another Metal program on the same device.
    /// External mutation leases follow the imported view. No host copy or
    /// device copy occurs, and the original executable need not remain
    /// alive.
    pub fn adopt(&self, source: &Buffer) -> Result<Buffer> {
        // A published Buffer is immutable and its producer has completed.
        let (handle, offset) = unsafe { source.metal()? };
        let mut result = unsafe {
            self.import_metal(
                source.spec().clone(),
                handle,
                usize::try_from(offset).map_err(|_| "Metal offset overflows usize")?,
                Arc::new(()),
            )?
        };
        Arc::get_mut(&mut result.0)
            .expect("new imported view is unique")
            .leases = source.0.leases.clone();
        Ok(result)
    }
}
